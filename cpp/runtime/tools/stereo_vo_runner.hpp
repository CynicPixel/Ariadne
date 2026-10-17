#pragma once
// ---------------------------------------------------------------------------
// The 1.4 + 1.5 stereo VO run over a StereoSource (EuRoC or a sim recording),
// scored against the source's ground truth. Moved verbatim from
// ariadne_euroc_vo.cpp (2026-10-15) with the dataset access replaced by the
// source interface; the EuRoC MH_01 evidence reproduces bit-identically.
//
// Scoring: estimated poses are of the RECTIFIED left camera; they are mapped
// to the body frame with T_B_Crect = T_B_C0 * [R1^T | 0] (the frame trap),
// associated to ground truth by nearest timestamp within 0.01 s (evo's
// default), and scored with SE(3)-aligned ATE (translation + rotation), RPE
// per second, and the Sim(3) scale as a drift diagnostic. Trajectories are
// written in TUM format for evo (the 1.9 oracle).
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/stereo/sgbm_stereo.hpp"
#include "perception/stereo/stereo_lk_matcher.hpp"
#include "perception/stereo/stereo_rectifier.hpp"
#include "perception/tracking/klt_tracker.hpp"
#include "runtime/agent/keyframe_records.hpp"
#include "runtime/datasets/stereo_source.hpp"
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"
#include "runtime/metrics/trajectory_io.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"
#include "slam/odometry/stereo_vo.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;


namespace ariadne::tools {

namespace fs = std::filesystem;
using namespace ariadne::perception;

namespace detail {
inline double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inline double Mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return v.empty() ? -1.0 : s / v.size();
}

inline std::string UtcNow(const char* fmt) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

inline core::Mat3 ToEigen3(const cv::Mat& m) {
    core::Mat3 R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R(r, c) = m.at<double>(r, c);
    return R;
}

}  // namespace detail
using namespace detail;

/// What a keyframe hook sees (1.6 collaboration, 1.7 spool): the canonical records, built here, once, by the
/// producer: the §4.2 Keyframe (pose of the rectified left camera in this agent's odometry frame, its absolute
/// covariance, SGBM depth, observed landmarks) and the §18.2 KeyframeBundle exported at the graph's current
/// version. Plus what only a feature extractor needs: the rectified gray image and the rectified rig.
struct KeyframeView {
    runtime::Keyframe keyframe;
    runtime::KeyframeBundle bundle;
    cv::Mat left_rect;       // 8-bit gray (the bundle carries the RGB)
    core::SE3 T_B_C;         // body <- rectified left camera
    double baseline_m = 0;
};

/// Every frame (optional hook): the estimated rectified-left camera pose and a
/// lazy rectified-RGB getter (only frames that ask pay for the remap).
struct FrameView {
    uint64_t frame_id = 0;
    int64_t t_ns = 0;
    bool is_keyframe = false;
    core::SE3 T_O_C;
    std::function<cv::Mat()> left_rect_rgb;   // empty Mat if the source has no colour
};

struct StereoVoArgs {
    std::string perception_config, slam_config, out;
    int frames = 0;
    uint64_t seed = 1;
    std::string tool = "ariadne_euroc_vo";   // results.yaml header
    std::function<void(const KeyframeView&)> on_keyframe;   // optional; never changes the run
    std::function<void(int64_t, const core::SE3&)> on_body_pose;   // every frame's T_O_B; optional
    std::function<void(const FrameView&)> on_frame;                 // every frame; optional; never changes the run
};

/// The 1.4 + 1.5 pipeline over any StereoSource: rectify -> LK tracker ->
/// StereoVo, keyframes with LK stereo matches + SGBM, then scoring against the
/// source's ground truth. Shared by ariadne_euroc_vo and ariadne_sim_vo.
inline int RunStereoVo(runtime::StereoSource& source, const StereoVoArgs& args, int argc, char** argv) {
    const PerceptionConfig pcfg = LoadPerception(args.perception_config);
    const slam::SlamConfig scfg = slam::LoadSlam(args.slam_config);

    StereoRectifier rectifier(source.Calibration(), 0.0);   // not const: the right map takes the row refinement
    const RectifiedStereoRig& rr = rectifier.Rig();
    const core::RectifiedStereo rig{{rr.fx, rr.fy, rr.cx, rr.cy}, rr.baseline_m};
    const core::SE3 T_B_Crect = runtime::BodyFromRectifiedLeft(source.T_B_C0(), ToEigen3(rectifier.R1()));
    if (scfg.triangulation.min_depth_m < MinObservableRange(pcfg.stereo, rr)) {
        std::fprintf(stderr, "warning: slam min_depth_m %.3f is below the stereo Z_min %.3f\n",
                     scfg.triangulation.min_depth_m, MinObservableRange(pcfg.stereo, rr));
    }

    const OrbEngine orb(pcfg.features);
    KltTracker tracker(pcfg.tracker, orb);
    const SgbmStereo sgbm(pcfg.stereo, rr);
    slam::StereoVo vo(scfg, rig, args.seed, source.Agent());

    const std::vector<runtime::TimedPose> gt_poses = source.GroundTruth();
    const int n_planned = args.frames > 0 ? std::min<int>(args.frames, static_cast<int>(source.FrameCount()))
                                          : static_cast<int>(source.FrameCount());
    std::printf("%s: %d frames; rig fx=%.2f B=%.4f; gt samples %zu\n", source.SceneName().c_str(), n_planned, rr.fx,
                rr.baseline_m, gt_poses.size());

    std::vector<runtime::TimedPose> est_body;
    // ORB-SLAM's trajectory protocol (System::SaveTrajectoryEuRoC): each frame is stored relative to its reference
    // keyframe and re-composed at the end from that keyframe's FINAL (window-refined) pose. Window backend only, next
    // to the causal per-frame score, which stays the primary number (LOCAL_SLAM_2.3 §10 protocol caveat).
    struct FrameRef { uint64_t kf; core::SE3 T_kf_c; };
    std::vector<FrameRef> frame_ref;
    std::vector<double> t_track, t_vo, t_kf;
    int lost = 0, keyframes = 0, ba_runs = 0;
    int lost_run = 0;            // consecutive lost frames (diagnostic)
    // Lost-frame handling (LOST_FRAMES.md, D56): written to results.yaml only when the config has a `lost:` section.
    long lost_bridged = 0, lost_coasted = 0, lost_rot_h = 0, lost_rot_e = 0, lost_rot_none = 0, bridge_retries = 0;
    long reanchor_probes = 0, reanchor_refused = 0, keyframes_lost = 0;
    uint32_t vo_segments = 0;   // loss onsets so far (StereoVo's segment counter)
    // Per-frame trace (vo_trace.csv): what the frontend saw on every frame, so a lost stretch can be explained.
    std::ostringstream trace;
    trace << "frame,t_ns,status,tracks,correspondences,pnp_inliers,keyframe,stereo_valid,created,source,segment,"
             "probe,bridge_model,bridge_tracks,bridge_inliers_h,bridge_inliers_e,bridge_cheiral_e,bridge_disagreement_deg,"
             "bridge_rx,bridge_ry,bridge_rz,anchor_model,anchor_tracks,anchor_age,anchor_drift_deg\n";
    double last_finite_t = 0.0;
    slam::TriangulationStats tri;
    double ba_cost_ratio_sum = 0.0;
    // 2.3 window backend only (results.yaml keeps the v1 bytes): solve outcomes and the post-solve hygiene.
    long win_not_found = 0, win_obs = 0, win_invalid_entry = 0, win_chi2_failed = 0, win_rejected = 0,
         win_culled = 0, win_exit_edges = 0, win_unobservable = 0, win_ill = 0, win_ill_after = 0,
         win_unobservable_after = 0, win_dof = 0, win_degenerate = 0;
    double win_sq = 0.0;
    long stereo_valid = 0, stereo_total = 0;
    long usac_retries = 0, usac_skipped = 0;
    // Online row refinement (D54): estimated at keyframes until it succeeds, then frozen and disclosed.
    perception::RowCorrection row_fix, row_check;   // row_check: re-estimated AFTER the refinement (acceptance: ~0)
    perception::RowSamples row_samples;              // pooled over keyframes until the fit succeeds
    int row_attempts = 0, row_keyframe = -1;
    std::string row_last_refusal = "never attempted";   // OpenCV 5.0.0 USAC defect (SMOKE §4q); written only when non-zero

    int i = 0;
    const int n = static_cast<int>(source.ForEachFrame(static_cast<size_t>(std::max(args.frames, 0)),
                                                       [&](const runtime::StereoFrame& frame) {
        if (frame.agent != source.Agent() || frame.id != static_cast<uint64_t>(i)) {
            throw std::logic_error("stereo source: frame identity out of sequence at " + std::to_string(i));
        }
        cv::Mat l, r;
        rectifier.Rectify(frame.left, frame.right, l, r);

        auto t0 = std::chrono::steady_clock::now();
        const TrackedFrame tf = tracker.Process(l, static_cast<uint64_t>(i), args.seed);
        t_track.push_back(Ms(t0));
        usac_retries += tf.stats.usac_defect_retries;
        usac_skipped += tf.stats.usac_defect_skipped;

        slam::VoFrame vf;
        vf.frame_id = static_cast<uint64_t>(i);
        vf.t_ns = frame.t_ns;
        vf.track_ids = tf.ids;
        vf.points = tf.points;
        t0 = std::chrono::steady_clock::now();
        const slam::VoTrackResult tr = vo.Track(vf);
        t_vo.push_back(Ms(t0));
        lost += tr.status == slam::VoStatus::kLost;
        lost_run = tr.status == slam::VoStatus::kLost ? lost_run + 1 : 0;
        core::SE3 T_W_C = tr.T_W_C;
        if (!T_W_C.R.allFinite() || !T_W_C.t.allFinite()) {
            // A non-finite pose poisons every later prediction and turns the score into NaN: stop here, loudly.
            throw std::runtime_error("stereo VO: non-finite pose at frame " + std::to_string(i) + " (status " +
                                     std::to_string(static_cast<int>(tr.status)) + ", correspondences " +
                                     std::to_string(tr.correspondences) + ", lost for " + std::to_string(lost_run) +
                                     " frames, last finite |t| " + std::to_string(last_finite_t) + " m, tracks " +
                                     std::to_string(tf.ids.size()) + ", keyframes " + std::to_string(keyframes) + ")");
        }
        last_finite_t = T_W_C.t.norm();

        long trace_valid = -1, trace_created = -1;   // keyframes (and re-anchor probes) only
        bool made_keyframe = false;
        if (tr.status == slam::VoStatus::kLost) {
            lost_bridged += tr.source == slam::PoseSource::kBridged;
            lost_coasted += tr.source == slam::PoseSource::kCoasted;
            if (tr.source == slam::PoseSource::kBridged) {
                lost_rot_h += tr.anchor_bridge.ok && tr.anchor_bridge.model == 'H';
                lost_rot_e += tr.anchor_bridge.ok && tr.anchor_bridge.model == 'E';
                lost_rot_none += !tr.anchor_bridge.ok;
            }
        }
        bridge_retries += tr.bridge.usac_retries + tr.anchor_bridge.usac_retries;
        vo_segments = tr.segment;
        // A lost frame under lost.reanchor = gated is only PROBED: its stereo is computed, and it becomes a keyframe
        // only if StereoVo::ShouldReanchor says the next frame's PnP would then have enough landmarks.
        if (tr.need_keyframe || tr.reanchor_probe) {
            t0 = std::chrono::steady_clock::now();
            // The stereo pair as matched: right exposure matched to the left (EuRoC per-camera auto-exposure),
            // then, until it succeeds, the row-refinement estimate on THIS pair -- applied before the first real
            // match, so a misaligned rig does not start out lost (AUDIT_2.3.md, D54).
            const auto condition = [&](const cv::Mat& right) {
                return pcfg.stereo_match.exposure_match ? MatchExposure(l, right) : right;
            };
            cv::Mat rs = condition(r);
            if (tr.need_keyframe && pcfg.stereo_match.row_refinement && !row_fix.ok) {
                ++row_attempts;
                CollectRowSamples(l, rs, tf.points, pcfg.stereo_match, row_samples);
                const RowCorrection est = FitRowCorrection(row_samples, pcfg.stereo_match);
                if (est.ok) {
                    row_fix = est;
                    row_keyframe = keyframes;
                    rectifier.RefineRightRows(est.a, est.b, est.c);
                    cv::Mat l_again;
                    rectifier.Rectify(frame.left, frame.right, l_again, r);
                    rs = condition(r);
                    std::printf("  stereo row refinement at keyframe %d: dy = %+.3f %+.5f x %+.5f y px (%d matches, "
                                "MAD %.3f px%s)\n", keyframes, est.a, est.b, est.c, est.matches, est.mad_px,
                                est.slopes_fitted ? "" : ", offset only");
                } else {
                    row_last_refusal = est.why;
                }
            } else if (tr.need_keyframe && pcfg.stereo_match.row_refinement && !row_check.ok && keyframes > row_keyframe) {
                // Acceptance, measured once on a later pair: the residual left after the refinement (not applied).
                row_check = EstimateRowCorrection(l, rs, tf.points, pcfg.stereo_match);
            }
            const StereoMatches sm = MatchLeftToRight(l, rs, tf.points, pcfg.stereo_match);
            const StereoDepth depth = sgbm.Compute(l, rs);
            const cv::Mat supported = DenseSupportMask(depth, pcfg.stereo);
            cv::Mat z32, s32;
            depth.Z.convertTo(z32, CV_32F);
            depth.sigma_Z.convertTo(s32, CV_32F);
            slam::VoStereo st;
            st.right = sm.right;
            st.valid = sm.valid;
            for (const cv::Point2f& p : tf.points) {
                const int x = cvRound(p.x), y = cvRound(p.y);
                const bool ok = x >= 0 && y >= 0 && x < rr.width && y < rr.height && supported.at<uchar>(y, x);
                st.sgbm_z.push_back(ok ? z32.at<float>(y, x) : std::numeric_limits<float>::quiet_NaN());
                st.sgbm_sigma.push_back(ok ? s32.at<float>(y, x) : std::numeric_limits<float>::quiet_NaN());
            }
            trace_valid = 0;
            for (uchar v : sm.valid) trace_valid += v;
            reanchor_probes += !tr.need_keyframe;
            made_keyframe = tr.need_keyframe || vo.ShouldReanchor(st);
            reanchor_refused += !made_keyframe;
            if (made_keyframe) {
                stereo_valid += trace_valid;
                stereo_total += static_cast<long>(sm.valid.size());
                const slam::VoKeyframeResult kr = vo.AddKeyframe(st);
                t_kf.push_back(Ms(t0));
                ++keyframes;
                keyframes_lost += tr.status == slam::VoStatus::kLost;
                tri.candidates += kr.triangulation.candidates;
                tri.created += kr.triangulation.created;
                trace_created = kr.triangulation.created;
                tri.no_stereo += kr.triangulation.no_stereo;
                tri.nonpositive_disparity += kr.triangulation.nonpositive_disparity;
                tri.depth_range += kr.triangulation.depth_range;
                tri.reprojection += kr.triangulation.reprojection;
                tri.sgbm_inconsistent += kr.triangulation.sgbm_inconsistent;
                tri.uncertain += kr.triangulation.uncertain;
                if (kr.ba_ran) {
                    ++ba_runs;
                    ba_cost_ratio_sum += kr.ba_final_cost / std::max(kr.ba_initial_cost, 1e-12);
                    win_not_found += !kr.ba_found;
                    win_obs += kr.ba_observations;
                    win_invalid_entry += kr.ba_invalid_at_entry;
                    win_chi2_failed += kr.ba_chi2_failed;
                    win_sq += kr.ba_inlier_sq_px;
                    win_dof += kr.ba_inlier_dof;
                    win_ill_after += kr.ba_ill_conditioned_after;
                    win_unobservable_after += kr.ba_unobservable_after;
                    win_degenerate += kr.ba_marginal_degenerate;
                }
                win_rejected += kr.ba_rejected;
                win_culled += kr.ba_culled;
                win_exit_edges += kr.exit_edge;
                win_unobservable += kr.ba_unobservable;
                win_ill += kr.ba_ill_conditioned;
                T_W_C = vo.graph().nodes().rbegin()->second.T_W_C;
                if (args.on_keyframe) {
                    const slam::PoseNode& node = vo.graph().nodes().at(kr.keyframe_id);
                    if (node.agent != frame.agent || node.T_W_C.t != T_W_C.t) {
                        throw std::logic_error("keyframe hook: graph node does not match the keyframe just added");
                    }
                    KeyframeView kv;
                    runtime::Keyframe& kf = kv.keyframe;
                    kf.id = kr.keyframe_id;
                    kf.frame_id = frame.id;
                    kf.agent = frame.agent;
                    kf.t_ns = frame.t_ns;
                    kf.T_world_cam = node.T_W_C;
                    kf.pose_cov = node.covariance;
                    kf.depth = depth;
                    kf.landmark_ids = vo.KeyframeLandmarks(kr.keyframe_id);
                kf.pose_source = kr.source;
                kf.segment = kr.segment;
                    cv::Mat rgb = l;
                    if (!frame.left_rgb.empty()) rectifier.RectifyLeft(frame.left_rgb, rgb);
                    kv.bundle = runtime::BuildKeyframeBundle(vo.graph(), kr.keyframe_id, rig.cam, rgb, depth, supported);
                    kv.left_rect = l;
                    kv.T_B_C = T_B_Crect;
                    kv.baseline_m = rr.baseline_m;
                    args.on_keyframe(kv);
                }
            }
        }
        const core::Vec3 brot = core::SO3Log(tr.bridge.R_prev_cur);
        trace << i << ',' << vf.t_ns << ',' << static_cast<int>(tr.status) << ',' << tf.ids.size() << ','
              << tr.correspondences << ',' << tr.pnp_inliers << ',' << int(made_keyframe) << ',' << trace_valid << ','
              << trace_created << ',' << slam::ToString(tr.source) << ',' << tr.segment << ',' << int(tr.reanchor_probe)
              << ',' << tr.bridge.model << ',' << tr.bridge.tracks << ',' << tr.bridge.inliers_h << ','
              << tr.bridge.inliers_e << ',' << tr.bridge.cheiral_e << ',' << tr.bridge.disagreement_deg << ',' << brot.x() << ',' << brot.y() << ',' << brot.z() << ','
              << tr.anchor_bridge.model << ',' << tr.anchor_bridge.tracks << ',' << tr.anchor_age << ','
              << tr.anchor_drift_deg << '\n';
        est_body.push_back({vf.t_ns, T_W_C * T_B_Crect.inverse()});
        if (!vo.graph().nodes().empty()) {
            const slam::PoseNode& ref = vo.graph().nodes().rbegin()->second;
            frame_ref.push_back({ref.keyframe, ref.T_W_C.inverse() * T_W_C});
        } else {
            frame_ref.push_back({~uint64_t(0), T_W_C});
        }
        if (args.on_body_pose) args.on_body_pose(vf.t_ns, est_body.back().T);
        if (args.on_frame) {
            FrameView fv;
            fv.frame_id = vf.frame_id;
            fv.t_ns = vf.t_ns;
            fv.is_keyframe = made_keyframe;
            fv.T_O_C = T_W_C;
            fv.left_rect_rgb = [&]() {
                cv::Mat o;
                if (!frame.left_rgb.empty()) rectifier.RectifyLeft(frame.left_rgb, o);
                return o;
            };
            args.on_frame(fv);
        }
        if (i % 500 == 0) std::printf("  frame %d: tracks %zu, status %d, keyframes %d\n", i, tf.ids.size(),
                                      static_cast<int>(tr.status), keyframes);
        ++i;
    }));

    // --- Scoring ---------------------------------------------------------------
    const std::vector<int> assoc = runtime::AssociateNearest(est_body, gt_poses);
    std::vector<core::TrajectorySample> e, g;
    for (size_t i = 0; i < est_body.size(); ++i) {
        if (assoc[i] < 0) continue;
        e.push_back({core::Timestamp{est_body[i].t_ns}, 0, est_body[i].T});
        g.push_back({core::Timestamp{gt_poses[static_cast<size_t>(assoc[i])].t_ns}, 0,
                     gt_poses[static_cast<size_t>(assoc[i])].T});
    }
    const auto ate = runtime::ComputeATE(e, g);
    if (!std::isfinite(ate.rmse)) {
        throw std::runtime_error("stereo VO: non-finite ATE over " + std::to_string(e.size()) + " associated poses");
    }
    const auto ate_rot = runtime::ComputeATERotation(e, g);
    const auto sim3 = runtime::AlignTrajectories(e, g, true);
    // 1 s by timestamp, not 20 samples: a stream with dropped frames (EuRoC V2_03) would otherwise stretch the window.
    const auto rpe = runtime::ComputeRPE(e, g, 1.0, 0.05);
    double gt_length = 0.0;
    for (size_t i = 1; i < g.size(); ++i) gt_length += (g[i].pose.t - g[i - 1].pose.t).norm();
    runtime::ErrorStats ate_kf, ate_kf_rot;
    runtime::RelativeError rpe_kf;
    if (scfg.ba.backend == "window") {
        std::vector<core::TrajectorySample> ek;
        for (size_t i = 0; i < est_body.size(); ++i) {
            if (assoc[i] < 0) continue;
            const FrameRef& fr = frame_ref[i];
            const core::SE3 T = fr.kf == ~uint64_t(0) ? fr.T_kf_c : vo.graph().nodes().at(fr.kf).T_W_C * fr.T_kf_c;
            ek.push_back({core::Timestamp{est_body[i].t_ns}, 0, T * T_B_Crect.inverse()});
        }
        ate_kf = runtime::ComputeATE(ek, g);
        ate_kf_rot = runtime::ComputeATERotation(ek, g);
        rpe_kf = runtime::ComputeRPE(ek, g, 1.0, 0.05);
    }

    const fs::path run_dir = fs::path(args.out) / (source.RunName() + "-" + UtcNow("%Y%m%dT%H%M%SZ"));
    fs::create_directories(run_dir);
    runtime::WriteTum((run_dir / "estimate_body.tum").string(), est_body);
    std::ofstream(run_dir / "vo_trace.csv") << trace.str();
    // Ground truth goes under raw/, which .gitignore excludes: for EuRoC it is
    // the dataset's own data (InC-NC, D35), so evidence commits carry our
    // estimate and derived numbers but never a copy of the dataset.
    fs::create_directories(run_dir / "raw");
    runtime::WriteTum((run_dir / "raw" / "groundtruth_body.tum").string(), gt_poses);

    runtime::RunManifest m;
    m.run_id = run_dir.filename().string();
    m.created_utc = UtcNow("%Y-%m-%dT%H:%M:%SZ");
    m.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : runtime::kUnknownField;
    m.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
    m.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : runtime::kUnknownField;
    m.scene = source.SceneName();
    m.camera_config = source.CameraConfig();
    m.seed = args.seed;
    m.seed_recorded = true;
    m.agent_count = 1;
    m.execution_mode = "replay";
    m.perception_config = pcfg.name;
    {
        std::ostringstream rf;
        if (!pcfg.stereo_match.row_refinement) rf << "off";
        else if (!row_fix.ok) rf << "not_estimated: " << row_last_refusal;
        else rf << "a=" << row_fix.a << " b=" << row_fix.b << " c=" << row_fix.c << " matches=" << row_fix.matches
                << " keyframe=" << row_keyframe;
        m.stereo_row_refinement = rf.str();
    }
    m.slam_config = scfg.name;
    m.lost_handling = scfg.lost.motion + "/" + scfg.lost.reanchor;
    m.degradation_config = "none";
    m.feature_engine = orb.Name();
    m.loop_backend = "none";
    m.imu_mode = "none";
    m.bag_checksum = source.BagChecksum();
    runtime::RecordBuildIdentity(m);
    runtime::RecordOpenCvIdentity(m);
    runtime::RecordDeterminismDigests(m);
    m.extra["opencv_threads"] = std::to_string(cv::getNumThreads());
    m.extra["frames"] = std::to_string(n);
    std::string cmd;
    for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
    m.extra["command"] = cmd;
    std::ofstream(run_dir / "manifest.yaml") << runtime::ToYaml(m);

    std::ofstream res(run_dir / "results.yaml");
    res << "# " << args.tool << " results. Generated; do not hand-edit.\n"
        << "sequence: " << source.RunName() << "\nframes: " << n << "\n"
        << "associated_poses: " << e.size() << "  # nearest GT within 0.01 s (evo default)\n"
        << "gt_path_length_m: " << gt_length << "\n"
        << "ate_se3: {rmse_m: " << ate.rmse << ", mean_m: " << ate.mean << ", max_m: " << ate.max
        << ", rotation_rmse_rad: " << ate_rot.rmse << "}\n"
        << "ate_rmse_percent_of_path: " << (gt_length > 0 ? 100.0 * ate.rmse / gt_length : -1.0) << "\n"
        << "sim3_scale: " << sim3.scale << "  # 1.0 = no scale drift; stereo is metric\n"
        << "rpe_1s: {translation_rmse_m: " << rpe.translation.rmse << ", rotation_rmse_rad: " << rpe.rotation.rmse
        << "}\n"
        << "vo: {lost_frames: " << lost << ", keyframes: " << keyframes << ", ba_runs: " << ba_runs
        << ", ba_mean_cost_ratio: " << (ba_runs ? ba_cost_ratio_sum / ba_runs : -1.0)
        << ", landmarks_alive_at_end: " << vo.landmark_count()
        << ", graph_edges: " << vo.graph().edges().size() << ", graph_version: " << vo.graph().version() << "}\n"
        << "stereo_match_valid_fraction: " << (stereo_total ? double(stereo_valid) / stereo_total : -1.0) << "\n"
        << "triangulation: {candidates: " << tri.candidates << ", created: " << tri.created
        << ", no_stereo: " << tri.no_stereo << ", nonpositive_disparity: " << tri.nonpositive_disparity
        << ", depth_range: " << tri.depth_range << ", reprojection: " << tri.reprojection
        << ", sgbm_inconsistent: " << tri.sgbm_inconsistent << ", uncertain: " << tri.uncertain << "}\n"
        << "timing_ms: {track_mean: " << Mean(t_track) << ", vo_track_mean: " << Mean(t_vo)
        << ", keyframe_mean: " << Mean(t_kf) << ", opencv_threads: " << cv::getNumThreads() << "}\n";
    res << "stereo_row_refinement: {enabled: " << (pcfg.stereo_match.row_refinement ? "true" : "false")
        << ", applied: " << (row_fix.ok ? "true" : "false") << ", a_px: " << row_fix.a << ", b: " << row_fix.b
        << ", c: " << row_fix.c << ", matches: " << row_fix.matches << ", mad_px: " << row_fix.mad_px
        << ", slopes_fitted: " << (row_fix.slopes_fitted ? "true" : "false") << ", keyframe: " << row_keyframe
        << ", attempts: " << row_attempts << ", residual_after: {measured: " << (row_check.ok ? "true" : "false")
        << ", a_px: " << row_check.a << ", b: " << row_check.b << ", c: " << row_check.c
        << ", max_abs_dy_px: "
        << std::max({std::abs(row_check.a), std::abs(row_check.a + row_check.b * (rr.width - 1)),
                     std::abs(row_check.a + row_check.c * (rr.height - 1)),
                     std::abs(row_check.a + row_check.b * (rr.width - 1) + row_check.c * (rr.height - 1))})
        << ", matches: " << row_check.matches << "}}\n"
        << "stereo_exposure_match: " << (pcfg.stereo_match.exposure_match ? "true" : "false") << "\n";
    if (usac_retries > 0 || usac_skipped > 0) {
        res << "tracker_opencv_usac_defect: {redraws: " << usac_retries << ", epipolar_check_skipped_frames: "
            << usac_skipped << "}\n";
    }
    if (scfg.ba.backend == "window") {
        res << "keyframe_refined: {ate_rmse_m: " << ate_kf.rmse << ", ate_rotation_rmse_rad: " << ate_kf_rot.rmse
            << ", rpe_1s_translation_rmse_m: " << rpe_kf.translation.rmse
            << "}  # ORB-SLAM protocol: frames re-composed on final keyframe poses\n";
        res << "ba_window: {not_found: " << win_not_found << ", observations: " << win_obs
            << ", invalid_at_entry: " << win_invalid_entry << ", chi2_failed: " << win_chi2_failed
            << ", chi2_failed_fraction: " << (win_obs ? double(win_chi2_failed) / win_obs : -1.0)
            << ", keyframe_observations_rejected: " << win_rejected
            << ", culled: " << win_culled << ", exit_edges: " << win_exit_edges
            << ", keyframes_held: " << win_unobservable << ", keyframes_held_at_solution: " << win_unobservable_after
            << ", landmarks_ill_conditioned: " << win_ill << ", landmarks_ill_conditioned_at_solution: " << win_ill_after
            << ", marginal_degenerate: " << win_degenerate
            << ", inlier_sigma_px: " << (win_dof > 0 ? std::sqrt(win_sq / double(win_dof)) : -1.0) << "}\n";
    }
    if (scfg.lost.motion != "coast" || scfg.lost.reanchor != "every_frame") {
        res << "lost: {motion: " << scfg.lost.motion << ", reanchor: " << scfg.lost.reanchor
            << ", segments: " << vo_segments << ", bridged_frames: " << lost_bridged << ", coasted_frames: " << lost_coasted
            << ", bridge_rotation: {H: " << lost_rot_h << ", E: " << lost_rot_e << ", none: " << lost_rot_none
            << "}, keyframes_while_lost: " << keyframes_lost << ", reanchor_probes: " << reanchor_probes
            << ", reanchor_refused: " << reanchor_refused << ", bridge_usac_redraws: " << bridge_retries << "}\n";
    }
    res.close();

    const auto missing = runtime::MissingRequiredFields(m);
    std::printf("run written to %s\nmanifest missing required fields: %zu", run_dir.c_str(), missing.size());
    for (const auto& f : missing) std::printf(" [%s]", f.c_str());
    std::printf("\n");
    std::fputs(runtime::ReadTextFile((run_dir / "results.yaml").string()).c_str(), stdout);
    return 0;
}

}  // namespace ariadne::tools
