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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
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

struct StereoVoArgs {
    std::string perception_config, slam_config, out;
    int frames = 0;
    uint64_t seed = 1;
    std::string tool = "ariadne_euroc_vo";   // results.yaml header
};

/// The 1.4 + 1.5 pipeline over any StereoSource: rectify -> LK tracker ->
/// StereoVo, keyframes with LK stereo matches + SGBM, then scoring against the
/// source's ground truth. Shared by ariadne_euroc_vo and ariadne_sim_vo.
inline int RunStereoVo(runtime::StereoSource& source, const StereoVoArgs& args, int argc, char** argv) {
    const PerceptionConfig pcfg = LoadPerception(args.perception_config);
    const slam::SlamConfig scfg = slam::LoadSlam(args.slam_config);

    const StereoRectifier rectifier(source.Calibration(), 0.0);
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
    slam::StereoVo vo(scfg, rig, args.seed);

    const std::vector<runtime::TimedPose> gt_poses = source.GroundTruth();
    const int n_planned = args.frames > 0 ? std::min<int>(args.frames, static_cast<int>(source.FrameCount()))
                                          : static_cast<int>(source.FrameCount());
    std::printf("%s: %d frames; rig fx=%.2f B=%.4f; gt samples %zu\n", source.SceneName().c_str(), n_planned, rr.fx,
                rr.baseline_m, gt_poses.size());

    std::vector<runtime::TimedPose> est_body;
    std::vector<double> t_track, t_vo, t_kf;
    int lost = 0, keyframes = 0, ba_runs = 0;
    slam::TriangulationStats tri;
    double ba_cost_ratio_sum = 0.0;
    long stereo_valid = 0, stereo_total = 0;

    int i = 0;
    const int n = static_cast<int>(source.ForEachFrame(static_cast<size_t>(std::max(args.frames, 0)),
                                                       [&](const runtime::StereoFrame& frame) {
        cv::Mat l, r;
        rectifier.Rectify(frame.left, frame.right, l, r);

        auto t0 = std::chrono::steady_clock::now();
        const TrackedFrame tf = tracker.Process(l, static_cast<uint64_t>(i), args.seed);
        t_track.push_back(Ms(t0));

        slam::VoFrame vf;
        vf.frame_id = static_cast<uint64_t>(i);
        vf.t_ns = frame.t_ns;
        vf.track_ids = tf.ids;
        vf.points = tf.points;
        t0 = std::chrono::steady_clock::now();
        const slam::VoTrackResult tr = vo.Track(vf);
        t_vo.push_back(Ms(t0));
        lost += tr.status == slam::VoStatus::kLost;
        core::SE3 T_W_C = tr.T_W_C;

        if (tr.need_keyframe) {
            t0 = std::chrono::steady_clock::now();
            const StereoMatches sm = MatchLeftToRight(l, r, tf.points, pcfg.stereo_match);
            const StereoDepth depth = sgbm.Compute(l, r);
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
            for (uchar v : sm.valid) stereo_valid += v;
            stereo_total += static_cast<long>(sm.valid.size());
            const slam::VoKeyframeResult kr = vo.AddKeyframe(st);
            t_kf.push_back(Ms(t0));
            ++keyframes;
            tri.candidates += kr.triangulation.candidates;
            tri.created += kr.triangulation.created;
            tri.no_stereo += kr.triangulation.no_stereo;
            tri.nonpositive_disparity += kr.triangulation.nonpositive_disparity;
            tri.depth_range += kr.triangulation.depth_range;
            tri.reprojection += kr.triangulation.reprojection;
            tri.sgbm_inconsistent += kr.triangulation.sgbm_inconsistent;
            tri.uncertain += kr.triangulation.uncertain;
            if (kr.ba_ran) {
                ++ba_runs;
                ba_cost_ratio_sum += kr.ba_final_cost / std::max(kr.ba_initial_cost, 1e-12);
            }
            T_W_C = vo.graph().nodes().rbegin()->second.T_W_C;
        }
        est_body.push_back({vf.t_ns, T_W_C * T_B_Crect.inverse()});
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
    const auto ate_rot = runtime::ComputeATERotation(e, g);
    const auto sim3 = runtime::AlignTrajectories(e, g, true);
    const auto rpe = runtime::ComputeRPE(e, g, 20);   // 20 frames = 1 s at 20 Hz
    double gt_length = 0.0;
    for (size_t i = 1; i < g.size(); ++i) gt_length += (g[i].pose.t - g[i - 1].pose.t).norm();

    const fs::path run_dir = fs::path(args.out) / (source.RunName() + "-" + UtcNow("%Y%m%dT%H%M%SZ"));
    fs::create_directories(run_dir);
    runtime::WriteTum((run_dir / "estimate_body.tum").string(), est_body);
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
    m.slam_config = scfg.name;
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
    res.close();

    const auto missing = runtime::MissingRequiredFields(m);
    std::printf("run written to %s\nmanifest missing required fields: %zu", run_dir.c_str(), missing.size());
    for (const auto& f : missing) std::printf(" [%s]", f.c_str());
    std::printf("\n");
    std::fputs(runtime::ReadTextFile((run_dir / "results.yaml").string()).c_str(), stdout);
    return 0;
}

}  // namespace ariadne::tools
