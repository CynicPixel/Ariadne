#include "slam/odometry/stereo_vo.hpp"

#include "core/random/philox.hpp"
#include "core/uncertainty/covariance.hpp"
#include "slam/bundle_adjustment/reprojection_lm.hpp"
#include "slam/bundle_adjustment/window_ba_levmarq.hpp"

#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <set>

namespace ariadne::slam {

namespace {

// Philox site for PnP's RANSAC sampler. Reserved sites count down from the top
// of the 48-bit domain; the tracker's epipolar check holds kMaxSite - 1.
constexpr uint64_t kSitePnpUsac = core::PhiloxStream::kMaxSite - 2;

// Covariance used when a relative pose has no estimate at all (tracking lost):
// 1 m and 1 rad standard deviations, i.e. "this edge knows almost nothing".
// A zero or tiny covariance here would make the graph trust a guess.
core::Mat6 UninformativeCovariance() { return core::Mat6::Identity(); }

core::SE3 FromRvecTvec(const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    core::SE3 T;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) T.R(r, c) = R.at<double>(r, c);
        T.t(r) = tvec.at<double>(r);
    }
    return T;
}

}  // namespace

StereoVo::StereoVo(const SlamConfig& config, const RectifiedStereo& rig, uint64_t seed, uint32_t agent)
    : cfg_(config), rig_(rig), seed_(seed), agent_(agent) {
    if (!(rig_.baseline > 0.0) || !(rig_.cam.fx > 0.0)) throw std::invalid_argument("StereoVo: bad rig");
}

VoTrackResult StereoVo::Track(const VoFrame& frame) {
    if (frame.track_ids.size() != frame.points.size()) throw std::invalid_argument("StereoVo: ids/points size");
    if (frames_tracked_ > 0 && !(frame.t_ns > prev_t_ns_)) {
        throw std::invalid_argument("StereoVo: frame timestamps must increase");
    }
    prev_frame_ = std::move(last_frame_);
    last_frame_ = frame;
    VoTrackResult res;
    res.segment = segment_;

    if (keyframes_.empty()) {
        // The first frame defines the world: its rectified left camera.
        res.status = VoStatus::kInitializing;
        res.T_W_C = core::SE3::Identity();
        res.need_keyframe = true;
        last_ = res;
        return res;
    }

    // Constant-velocity prediction, used as the fallback when PnP fails. The velocity is the last MEASURED one, a
    // twist per SECOND, frozen while coasting:
    //  - per second, not per frame: a stream that drops frames (EuRoC V2_03: 414 gaps of 100 ms; Phase 5 degradation)
    //    must predict by elapsed time;
    //  - frozen: re-deriving it from the model's own extrapolations (R_n = R_{n-1} R_{n-2}^T R_{n-1}) amplified the
    //    rounding error in orthonormality ~2.4x per frame, and MH_04 went non-finite after ~40 lost frames. Exp of a
    //    twist is a rotation by construction.
    core::SE3 predicted = prev_T_W_C_;
    if (frames_tracked_ >= 2) {
        if (!coasting_) velocity_per_s_ = core::SE3Log(prev2_T_W_C_.inverse() * prev_T_W_C_) /
                                          (1e-9 * double(prev_t_ns_ - prev2_t_ns_));
        predicted = prev_T_W_C_ * core::SE3Exp(velocity_per_s_ * (1e-9 * double(frame.t_ns - prev_t_ns_)));
    }

    // lost.motion = bridge: the rotation since the previous frame from the 2-D tracks, on EVERY frame -- used only when
    // PnP fails, but measured always so its error can be checked against ground truth on tracked frames (vo_trace).
    if (cfg_.lost.motion == "bridge") {
        std::map<uint64_t, cv::Point2f> before;
        for (size_t i = 0; i < prev_frame_.track_ids.size(); ++i) before.emplace(prev_frame_.track_ids[i], prev_frame_.points[i]);
        std::vector<cv::Point2f> p0, p1;
        for (size_t i = 0; i < frame.track_ids.size(); ++i) {
            const auto it = before.find(frame.track_ids[i]);
            if (it == before.end()) continue;
            p0.push_back(it->second);
            p1.push_back(frame.points[i]);
        }
        RotationBridgeParams bp;
        bp.threshold_px = cfg_.lost.bridge_threshold_px;
        bp.min_tracks = cfg_.lost.bridge_min_tracks;
        bp.model_ratio = cfg_.lost.model_ratio;
        res.bridge = EstimateFrameRotation(p0, p1, rig_.cam, bp, seed_, frame.frame_id);
    }

    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    std::vector<uint64_t> ids;
    for (size_t i = 0; i < frame.track_ids.size(); ++i) {
        const auto it = landmarks_.find(frame.track_ids[i]);
        if (it == landmarks_.end()) continue;
        obj.emplace_back(it->second.X_w.x(), it->second.X_w.y(), it->second.X_w.z());
        img.emplace_back(frame.points[i].x, frame.points[i].y);
        ids.push_back(frame.track_ids[i]);
    }
    res.correspondences = static_cast<int>(obj.size());

    bool ok = false;
    if (res.correspondences >= cfg_.pnp.min_correspondences) {
        const cv::Mat K = (cv::Mat_<double>(3, 3) << rig_.cam.fx, 0, rig_.cam.cx, 0, rig_.cam.fy, rig_.cam.cy, 0, 0, 1);
        // USAC overload: seeded from (seed, frame_id), single-threaded, so a
        // replay picks the same inliers (the FM_RANSAC-style path draws from
        // cv::theRNG(), see the tracker's epipolar check).
        cv::UsacParams usac;
        usac.threshold = cfg_.pnp.ransac_threshold_px;
        usac.confidence = cfg_.pnp.confidence;
        usac.maxIterations = cfg_.pnp.max_iterations;
        usac.isParallel = false;
        usac.randomGeneratorState = static_cast<int>(
            core::PhiloxStream(seed_, frame.frame_id, kSitePnpUsac).NextBits32() & 0x7FFFFFFFu);
        cv::Mat rvec, tvec;
        std::vector<int> inliers;
        if (cv::solvePnPRansac(obj, img, K, cv::noArray(), rvec, tvec, inliers, usac) &&
            static_cast<int>(inliers.size()) >= cfg_.pnp.min_inliers) {
            // Ariadne's own refinement (§8.1: Ariadne owns robust motion
            // estimation and state covariance): the shared LM, landmarks fixed.
            BaProblem p;
            p.cam = rig_.cam;
            p.poses = {FromRvecTvec(rvec, tvec)};   // T_C_W
            p.pose_fixed = {0};
            for (int k : inliers) {
                p.landmarks.emplace_back(obj[static_cast<size_t>(k)].x, obj[static_cast<size_t>(k)].y,
                                         obj[static_cast<size_t>(k)].z);
                p.obs.push_back({0, static_cast<int>(p.landmarks.size()) - 1,
                                 core::Vec2(img[static_cast<size_t>(k)].x, img[static_cast<size_t>(k)].y),
                                 core::Vec3::Zero(), cfg_.pnp.sigma_px});
            }
            p.landmark_fixed.assign(p.landmarks.size(), 1);
            BaOptions o;
            o.huber_delta = cfg_.pnp.huber_delta;
            const BaSummary s = SolveReprojection(p, o);
            if (s.pose_covariance.size() == 1) {
                res.T_W_C = p.poses[0].inverse();
                res.covariance_given_map = core::InvertCovariance(p.poses[0], s.pose_covariance[0]);
                res.pnp_inliers = s.inliers;
                ok = true;
            }
        }
    }

    if (ok) {
        res.status = VoStatus::kTracking;
    } else {
        // Never reset the world or drop the frame silently: predict, say so (status, source), and let the keyframe
        // rule below decide whether new landmarks are made here.
        res.status = VoStatus::kLost;
        if (!coasting_) {                 // loss onset: a new map segment; the bridge's velocity decays from here
            ++segment_;
            loss_t0_ns_ = prev_t_ns_;
        }
        res.segment = segment_;
        const bool bridge = cfg_.lost.motion == "bridge";   // before 2 tracked frames the velocity is zero: rotation only
        if (bridge && !coasting_) {   // the anchor starts at the last measured frame
            anchor_ = {prev_T_W_C_, prev_frame_.frame_id, {}};
            for (size_t i = 0; i < prev_frame_.track_ids.size(); ++i)
                anchor_.points.emplace(prev_frame_.track_ids[i], prev_frame_.points[i]);
        }
        res.T_W_C = bridge ? BridgePrediction(frame, res) : predicted;
        res.source = bridge ? PoseSource::kBridged : PoseSource::kCoasted;
        res.covariance_given_map = UninformativeCovariance();
    }

    const int ref = keyframes_.back().tracked_landmarks;
    const bool due = 1e-9 * double(frame.t_ns - last_kf_t_ns_) >= cfg_.keyframe.max_interval_s;
    if (ok || cfg_.lost.reanchor == "every_frame") {
        // every_frame (1.5): a lost frame is always a keyframe -- in a long dark stretch, one per frame, each at a
        // predicted pose (MH_04 v2: 206 keyframes in 10 s with 62 landmarks between them).
        res.need_keyframe = !ok || res.pnp_inliers < cfg_.keyframe.ref_ratio * ref ||
                            res.pnp_inliers < cfg_.keyframe.min_tracked_landmarks || due;
    } else {
        // gated (ORB-SLAM3 makes no keyframe while RECENTLY_LOST): only when it can re-anchor tracking, or at the
        // time cap that also applies while tracking.
        res.need_keyframe = due;
        res.reanchor_probe = !due;
    }

    prev2_T_W_C_ = prev_T_W_C_;
    prev_T_W_C_ = res.T_W_C;
    prev2_t_ns_ = prev_t_ns_;
    prev_t_ns_ = frame.t_ns;
    ++frames_tracked_;
    coasting_ = !ok;
    last_ = res;
    return res;
}

core::SE3 StereoVo::BridgePrediction(const VoFrame& frame, VoTrackResult& res) {
    // Translation: the last measured velocity (clamped: a pre-loss speed above lost.max_speed_m_s is itself corrupt,
    // V2_03 read 4.9 m/s), decaying as exp(-t / tau) from the last measured pose -- in a hover it vanishes instead of
    // extrapolating (MH_04: 10 s of constant velocity put a 9 cm hover 7.6 m away). The displacement over
    // [a, b] seconds since loss is v tau (exp(-a/tau) - exp(-b/tau)). The twist is in the camera frame, as in coasting.
    core::Vec6 v = velocity_per_s_;
    const double speed = v.head<3>().norm();
    if (speed > cfg_.lost.max_speed_m_s) v.head<3>() *= cfg_.lost.max_speed_m_s / speed;
    const double tau = cfg_.lost.velocity_decay_s;
    const double a = 1e-9 * double(prev_t_ns_ - loss_t0_ns_), b = 1e-9 * double(frame.t_ns - loss_t0_ns_);
    const double f = tau * (std::exp(-a / tau) - std::exp(-b / tau));
    core::SE3 T;
    T.t = prev_T_W_C_.t + prev_T_W_C_.R * (v.head<3>() * f);
    // Rotation, measured from the tracks: from the anchor (ORB-SLAM's TrackReferenceKeyFrame idea: one two-view
    // problem whose baseline GROWS through the stretch, instead of chaining frame-to-frame estimates whose noise adds
    // up -- 0.06 deg per frame at 1-2 cm, measured). When fewer than bridge_min_tracks reach back to the anchor, the
    // previous frame becomes the anchor. If no model is plausible, the decaying angular velocity.
    const auto shared = [&](std::vector<cv::Point2f>& p0, std::vector<cv::Point2f>& p1) {
        p0.clear();
        p1.clear();
        for (size_t i = 0; i < frame.track_ids.size(); ++i) {
            const auto it = anchor_.points.find(frame.track_ids[i]);
            if (it == anchor_.points.end()) continue;
            p0.push_back(it->second);
            p1.push_back(frame.points[i]);
        }
    };
    std::vector<cv::Point2f> p0, p1;
    shared(p0, p1);
    if (int(p0.size()) < cfg_.lost.bridge_min_tracks && anchor_.frame_id != prev_frame_.frame_id) {
        anchor_ = {prev_T_W_C_, prev_frame_.frame_id, {}};
        for (size_t i = 0; i < prev_frame_.track_ids.size(); ++i)
            anchor_.points.emplace(prev_frame_.track_ids[i], prev_frame_.points[i]);
        shared(p0, p1);
    }
    RotationBridgeParams bp;
    bp.threshold_px = cfg_.lost.bridge_threshold_px;
    bp.min_tracks = cfg_.lost.bridge_min_tracks;
    bp.model_ratio = cfg_.lost.model_ratio;
    res.anchor_bridge = EstimateFrameRotation(p0, p1, rig_.cam, bp, seed_, frame.frame_id, BridgeStream::kFromAnchor);
    res.anchor_age = int(frame.frame_id - anchor_.frame_id);
    // Diagnostic: the anchor's pose is a snapshot, but window solves may since have refined the previous (keyframe)
    // pose that the translation integrates from. Their rotational disagreement, when the previous frame was bridged
    // from this same anchor, measures how stale the snapshot is (trace: anchor_drift_deg).
    if (last_anchor_.anchor_frame == anchor_.frame_id && last_anchor_.for_frame == prev_frame_.frame_id)
        res.anchor_drift_deg =
            core::SO3Log(prev_T_W_C_.R.transpose() * anchor_.T_W_C.R * last_anchor_.R).norm() * 180.0 / M_PI;
    if (res.anchor_bridge.ok) last_anchor_ = {anchor_.frame_id, frame.frame_id, res.anchor_bridge.R_prev_cur};
    T.R = res.anchor_bridge.ok ? core::Mat3(anchor_.T_W_C.R * res.anchor_bridge.R_prev_cur)
                               : core::Mat3(prev_T_W_C_.R * core::SO3Exp(v.tail<3>() * f));
    return T;
}

std::vector<std::pair<size_t, core::Vec3>> StereoVo::TriangulateNew(const VoFrame& f, const VoStereo& st,
                                                                    TriangulationStats& stats) const {
    const size_t n = f.points.size();
    if (st.right.size() != n || st.valid.size() != n || st.sgbm_z.size() != n || st.sgbm_sigma.size() != n) {
        throw std::invalid_argument("StereoVo: stereo arrays must parallel the tracked points");
    }
    std::vector<std::pair<size_t, core::Vec3>> made;
    const double fx = rig_.cam.fx, B = rig_.baseline;
    std::vector<size_t> cand;
    std::vector<cv::Point2d> pl, pr;
    for (size_t i = 0; i < n; ++i) {
        if (landmarks_.count(f.track_ids[i])) continue;
        ++stats.candidates;
        if (!st.valid[i]) { ++stats.no_stereo; continue; }
        if (!(f.points[i].x - st.right[i].x > 0.0f)) { ++stats.nonpositive_disparity; continue; }
        cand.push_back(i);
        pl.emplace_back(f.points[i].x, f.points[i].y);
        pr.emplace_back(st.right[i].x, st.right[i].y);
    }
    if (cand.empty()) return made;
    // Rectified projection matrices: left K[I|0], right K[I|(-B,0,0)].
    const cv::Mat P1 = (cv::Mat_<double>(3, 4) << fx, 0, rig_.cam.cx, 0, 0, rig_.cam.fy, rig_.cam.cy, 0, 0, 0, 1, 0);
    const cv::Mat P2 = (cv::Mat_<double>(3, 4) << fx, 0, rig_.cam.cx, -fx * B, 0, rig_.cam.fy, rig_.cam.cy, 0, 0, 0, 1, 0);
    cv::Mat X4;
    cv::triangulatePoints(P1, P2, pl, pr, X4);   // linear DLT, 4xN homogeneous
    X4.convertTo(X4, CV_64F);
    for (size_t k = 0; k < cand.size(); ++k) {
        const size_t i = cand[k];
        const double w = X4.at<double>(3, static_cast<int>(k));
        const core::Vec3 Xc(X4.at<double>(0, static_cast<int>(k)) / w, X4.at<double>(1, static_cast<int>(k)) / w,
                            X4.at<double>(2, static_cast<int>(k)) / w);
        // No far cap: ORB-SLAM2 (§III-A) classifies points beyond ~40 baselines as "far" and KEEPS them (accurate
        // rotation, weak translation). Removing our 60-baseline discard took MH_04 from 411 lost frames to 30
        // (AUDIT_2.3.md). The relative-sigma test below is the depth bound: Z <= max_relative_sigma * fB /
        // disparity_sigma_px (~9.6 m on EuRoC). Multi-view triangulation of points beyond it is not done here.
        if (!(Xc.z() >= cfg_.triangulation.min_depth_m)) { ++stats.depth_range; continue; }
        const core::Vec2 el = core::Project(rig_.cam, Xc) - core::Vec2(pl[k].x, pl[k].y);
        const core::Vec2 er = core::Project(rig_.cam, Xc - core::Vec3(B, 0, 0)) - core::Vec2(pr[k].x, pr[k].y);
        if (el.norm() > cfg_.triangulation.max_reprojection_px || er.norm() > cfg_.triangulation.max_reprojection_px) {
            ++stats.reprojection;
            continue;
        }
        // §9.5 on the triangulated depth.
        const double sigma_tri = Xc.z() * Xc.z() / (fx * B) * cfg_.triangulation.disparity_sigma_px;
        if (sigma_tri / Xc.z() > cfg_.triangulation.max_relative_sigma) { ++stats.uncertain; continue; }
        // §9.6: cross-check against dense SGBM where it is trustworthy.
        if (std::isfinite(st.sgbm_z[i])) {
            const double s = std::sqrt(sigma_tri * sigma_tri + double(st.sgbm_sigma[i]) * st.sgbm_sigma[i]);
            if (std::abs(Xc.z() - st.sgbm_z[i]) > cfg_.triangulation.sgbm_gate_sigma * s) {
                ++stats.sgbm_inconsistent;
                continue;
            }
        }
        made.emplace_back(i, Xc);
    }
    return made;
}

bool StereoVo::ShouldReanchor(const VoStereo& st) const {
    TriangulationStats ignored;
    const size_t would_make = TriangulateNew(last_frame_, st, ignored).size();
    return last_.correspondences + int(would_make) >= cfg_.lost.reanchor_min_points;
}

VoKeyframeResult StereoVo::AddKeyframe(const VoStereo& st) {
    const VoFrame& f = last_frame_;
    const size_t n = f.points.size();
    if (st.right.size() != n || st.valid.size() != n || st.sgbm_z.size() != n || st.sgbm_sigma.size() != n) {
        throw std::invalid_argument("StereoVo::AddKeyframe: stereo arrays must parallel the tracked points");
    }
    VoKeyframeResult out;
    KeyframeState kf;
    kf.id = keyframes_.size();
    kf.T_W_C = last_.T_W_C;
    kf.source = last_.source;
    kf.segment = last_.segment;
    out.keyframe_id = kf.id;
    out.source = kf.source;
    out.segment = kf.segment;

    // Landmarks whose track died can never be observed again (no
    // re-association in the placeholder): retire them. In the 2.3 window, a
    // dead-track landmark stays while an in-window keyframe still observes it.
    std::set<uint64_t> alive(f.track_ids.begin(), f.track_ids.end());
    const bool window = cfg_.ba.backend == "window";
    if (window) {
        const size_t keep = size_t(std::max(cfg_.ba.window_size - 1, 0));
        for (size_t k = keyframes_.size() > keep ? keyframes_.size() - keep : 0; k < keyframes_.size(); ++k)
            for (const auto& [id, o] : keyframes_[k].obs) alive.insert(id);
    }
    for (auto it = landmarks_.begin(); it != landmarks_.end();) {
        it = alive.count(it->first) ? std::next(it) : landmarks_.erase(it);
    }

    // --- Triangulate new landmarks (§11.2) ---------------------------------
    const double B = rig_.baseline;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t id = f.track_ids[i];
        if (landmarks_.count(id)) kf.obs[id] = {f.points[i], st.right[i], st.valid[i] != 0};
    }
    for (const auto& [i, Xc] : TriangulateNew(f, st, out.triangulation)) {
        const uint64_t id = f.track_ids[i];
        landmarks_[id] = {kf.T_W_C * Xc, kf.id};
        kf.obs[id] = {f.points[i], st.right[i], true};
        ++out.triangulation.created;
    }
    kf.tracked_landmarks = static_cast<int>(kf.obs.size());
    if (window) return AddKeyframeWindow(std::move(kf), out, f.t_ns);

    // --- Two-frame BA stub: previous keyframe fixed (gauge) -----------------
    core::Mat6 cov_wc = last_.status == VoStatus::kTracking ? last_.covariance_given_map : UninformativeCovariance();
    bool have_relative_cov = last_.status == VoStatus::kTracking;
    if (!keyframes_.empty()) {
        const KeyframeState& prev = keyframes_.back();
        BaProblem p;
        p.cam = rig_.cam;
        p.poses = {prev.T_W_C.inverse(), kf.T_W_C.inverse()};
        p.pose_fixed = {1, 0};
        std::vector<uint64_t> lm_ids;
        for (const auto& [id, o] : kf.obs) {
            const auto po = prev.obs.find(id);
            if (po == prev.obs.end()) continue;
            const int j = static_cast<int>(lm_ids.size());
            lm_ids.push_back(id);
            p.landmarks.push_back(landmarks_.at(id).X_w);
            const core::Vec3 off(B, 0, 0);
            for (int fi = 0; fi < 2; ++fi) {
                const KeyframeState::Obs& ob = fi == 0 ? po->second : o;
                p.obs.push_back({fi, j, core::Vec2(ob.left.x, ob.left.y), core::Vec3::Zero(), cfg_.ba.sigma_px});
                if (ob.has_right) p.obs.push_back({fi, j, core::Vec2(ob.right.x, ob.right.y), off, cfg_.ba.sigma_px});
            }
        }
        p.landmark_fixed.assign(p.landmarks.size(), 0);
        out.ba_landmarks = static_cast<int>(lm_ids.size());
        if (out.ba_landmarks >= cfg_.pnp.min_correspondences) {
            BaOptions o;
            o.huber_delta = cfg_.ba.huber_delta;
            o.max_iterations = cfg_.ba.max_iterations;
            const BaSummary s = SolveReprojection(p, o);
            out.ba_ran = true;
            out.ba_iterations = s.iterations;
            out.ba_initial_cost = s.initial_cost;
            out.ba_final_cost = s.final_cost;
            kf.T_W_C = p.poses[1].inverse();
            for (size_t j = 0; j < lm_ids.size(); ++j) landmarks_[lm_ids[j]].X_w = p.landmarks[j];
            if (s.pose_covariance.size() == 1) {
                // Previous keyframe fixed, landmarks marginalised by the Schur
                // complement: exactly the RELATIVE uncertainty an odometry edge
                // should carry.
                cov_wc = core::InvertCovariance(p.poses[1], s.pose_covariance[0]);
                have_relative_cov = true;
            }
        }
    }

    // --- Pose graph ----------------------------------------------------------
    // The BA/PnP covariance is CONDITIONAL: the previous keyframe (or the map)
    // was held fixed. It belongs on the relative edge. The node carries the
    // ABSOLUTE covariance, composed along the odometry chain from the first
    // keyframe (the gauge, exactly known by definition), so it grows with
    // distance the way the actual error does.
    PoseNode node;
    node.keyframe = kf.id;
    node.agent = agent_;
    node.t_ns = f.t_ns;
    node.T_W_C = kf.T_W_C;
    core::SE3 Z;
    core::Mat6 cov_z = core::Mat6::Zero();
    if (!keyframes_.empty()) {
        const KeyframeState& prev = keyframes_.back();
        Z = prev.T_W_C.inverse() * kf.T_W_C;
        // X_new = X_prev Z  =>  a left perturbation xi of X_new is
        // Adj(X_prev^-1) xi on Z.
        cov_z = core::NearestPSD(
            have_relative_cov ? core::ChangeFrame(prev.T_W_C.inverse(), cov_wc) : UninformativeCovariance(), 1e-12);
        node.covariance = core::ComposeCovariance(prev.T_W_C, graph_.nodes().at(prev.id).covariance, cov_z);
    }
    graph_.AddNode(node);
    if (!keyframes_.empty()) graph_.AddEdge(OdometryEdge(keyframes_.back().id, kf.id, Z, cov_z));

    keyframes_.push_back(std::move(kf));
    last_.T_W_C = keyframes_.back().T_W_C;
    prev_T_W_C_ = last_.T_W_C;   // later predictions start from the BA-refined pose
    last_kf_t_ns_ = f.t_ns;
    return out;
}

// --- 2.3 windowed BA (LOCAL_SLAM_2.3.md) --------------------------------------------------------------------------
VoKeyframeResult StereoVo::AddKeyframeWindow(KeyframeState kf, VoKeyframeResult out, int64_t t_ns) {
    const size_t N = size_t(cfg_.ba.window_size);
    // The keyframe leaving the window (it was window[0] of the previous solve, the gauge): freeze its exit edge to
    // its successor from that solve's marginal of window[1] (design §4: with k fixed, the marginal of k+1 IS the
    // relative uncertainty), transported to Z's left tangent.
    if (keyframes_.size() >= N && have_pending_exit_) {
        const KeyframeState& a = keyframes_[keyframes_.size() - N];
        const KeyframeState& b = keyframes_[keyframes_.size() - N + 1];
        const core::Mat6 cov_z = core::NearestPSD(core::ChangeFrame(a.T_W_C.inverse(), pending_exit_cov_), 1e-12);
        graph_.AddEdge(OdometryEdge(a.id, b.id, a.T_W_C.inverse() * b.T_W_C, cov_z));
        out.exit_edge = true;
    }
    have_pending_exit_ = false;
    keyframes_.push_back(std::move(kf));
    const size_t K = std::min(N, keyframes_.size());
    const size_t first = keyframes_.size() - K;

    PoseNode node;
    node.keyframe = keyframes_.back().id;
    node.agent = agent_;
    node.t_ns = t_ns;
    node.T_W_C = keyframes_.back().T_W_C;
    node.covariance = keyframes_.size() == 1 ? core::Mat6::Zero() : UninformativeCovariance();
    graph_.AddNode(node);

    if (K >= 2) {
        BaWindow w;
        w.cam = {rig_.cam.fx, rig_.cam.fy, rig_.cam.cx, rig_.cam.cy, rig_.baseline};
        w.huber_delta = cfg_.ba.huber_delta;
        for (size_t k = first; k < keyframes_.size(); ++k) {
            w.T_W_C.push_back(keyframes_[k].T_W_C);
            w.fixed.push_back(k == first);
        }
        // Landmarks with >= 2 observations (rays) in the window; every observation of them.
        std::map<uint64_t, int> rays;
        for (size_t k = first; k < keyframes_.size(); ++k)
            for (const auto& [id, o] : keyframes_[k].obs)
                if (landmarks_.count(id)) rays[id] += 1 + int(o.has_right);
        for (const auto& [id, n] : rays) {
            if (n < 2) continue;
            w.X[id] = landmarks_.at(id).X_w;
        }
        for (size_t k = first; k < keyframes_.size(); ++k)
            for (const auto& [id, o] : keyframes_[k].obs) {
                if (!w.X.count(id)) continue;
                WindowObservation b;
                b.kf = int(k - first);
                b.landmark = id;
                b.sigma = cfg_.ba.sigma_px;
                b.z = Vec2(o.left.x, o.left.y);
                w.obs.push_back(b);
                if (o.has_right) {
                    b.right = true;
                    b.z = Vec2(o.right.x, o.right.y);
                    w.obs.push_back(b);
                }
            }
        // Observability guards (design §5, window_ba.hpp): drop landmarks too ill-conditioned to locate (near-parallel
        // rays: hover, lost tracking), then tie every free keyframe to a fixed one through covisibility (>=
        // pnp.min_correspondences shared landmarks, the 1.5 stub's rule); an untied group's oldest keyframe is held.
        const double rel = cfg_.triangulation.max_relative_sigma;
        const std::set<uint64_t> ill = IllConditionedLandmarks(w, rel);
        RemoveLandmarks(w, ill);
        out.ba_ill_conditioned = int(ill.size());
        int held = 0;
        AnchorWindow(w, cfg_.pnp.min_correspondences, held);
        out.ba_unobservable = held;
        out.ba_landmarks = int(w.X.size());
        out.ba_window = int(K);
        out.ba_observations = int(w.obs.size());
        for (const auto& o : w.obs)
            out.ba_invalid_at_entry +=
                !LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor).valid;
        const bool any_free = std::find(w.fixed.begin(), w.fixed.end(), false) != w.fixed.end();
        if (any_free && out.ba_landmarks >= cfg_.pnp.min_correspondences) {
            PgoSettings ls;
            ls.name = cfg_.name + ".ba";
            ls.max_iterations = cfg_.ba.max_iterations;
            ls.rel_energy_delta_tolerance = cfg_.ba.rel_energy_tolerance;
            ls.step_norm_tolerance = cfg_.ba.step_norm_tolerance;
            ls.min_gradient_tolerance = cfg_.ba.min_gradient_tolerance;
            ls.recompute_covariance = false;
            const WindowBaReport rep = SolveWindowBa(w, ls);
            out.ba_ran = true;
            out.ba_found = rep.found;
            out.ba_iterations = rep.iterations;
            out.ba_initial_cost = rep.initial_energy;
            out.ba_final_cost = rep.final_energy;
            // The guards again at the solution (the solve moved poses and points): a landmark that became
            // ill-conditioned keeps its previous map position and is left out of the marginals; anchoring is redone
            // on what remains, so the marginals come from a system that is PD by construction.
            BaWindow wc = w;
            const std::set<uint64_t> ill_after = IllConditionedLandmarks(wc, rel);
            RemoveLandmarks(wc, ill_after);
            out.ba_ill_conditioned_after = int(ill_after.size());
            int held_after = 0;
            const std::vector<int> anchor = AnchorWindow(wc, cfg_.pnp.min_correspondences, held_after);
            out.ba_unobservable_after = held_after;
            std::vector<int> want;
            for (int k : {1, int(K) - 1})
                if (!wc.fixed[size_t(k)] && std::find(want.begin(), want.end(), k) == want.end()) want.push_back(k);
            // A joint degeneracy (V1_03: a common scale of two keyframes' translations) leaves the information
            // singular even after the guards: the honest output is "no information" for this solve -- the marginals
            // become uninformative below -- and it is counted (results.yaml ba_window.marginal_degenerate).
            std::map<int, core::Mat6> marg;
            if (!want.empty() && !TryWindowBaCovariances(wc, want, marg, nullptr, &rep)) ++out.ba_marginal_degenerate;
            for (size_t k = first; k < keyframes_.size(); ++k) {
                keyframes_[k].T_W_C = w.T_W_C[k - first];
                if (k + 1 < keyframes_.size() && k != first)
                    graph_.UpdateNode(keyframes_[k].id, keyframes_[k].T_W_C, graph_.nodes().at(keyframes_[k].id).covariance);
            }
            for (const auto& [id, X] : w.X)
                if (!ill_after.count(id)) landmarks_.at(id).X_w = X;
            // Post-solve rejection (design §5): an observation whose whitened residual^2 > chi^2_2(0.95) is dropped.
            // The inliers also feed the sigma_px calibration (§5): sigma^2 = sum |r|^2 / (m - p), m scalar residuals,
            // p free parameters -- post-fit residuals have lost p degrees of freedom (a median would read low).
            const double c2 = cfg_.ba.chi2_reject;
            for (const auto& o : w.obs) {
                const ObsLinearization l =
                    LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor);
                const double s2 = l.valid ? l.r.squaredNorm() / (o.sigma * o.sigma) : 1e300;
                if (s2 <= c2) {
                    out.ba_inlier_sq_px += l.r.squaredNorm();
                    out.ba_inlier_dof += 2;
                    continue;
                }
                ++out.ba_chi2_failed;
                auto& obs = keyframes_[first + size_t(o.kf)].obs;
                if (obs.erase(o.landmark)) ++out.ba_rejected;
            }
            const BaLayout L(w);
            out.ba_inlier_dof -= L.dim();
            // Covariance: node = its anchor's absolute + the marginal relative to the fixed keyframes (a left
            // perturbation of the anchor moves the poses tied to it by the same left perturbation; correlations
            // ignored, design §4). A keyframe held in this solve gets nothing from it: uninformative.
            const int newest = int(K) - 1;
            const auto mn = marg.find(newest);
            const core::Mat6 cov_new =
                mn == marg.end()
                    ? UninformativeCovariance()
                    : core::NearestPSD(graph_.nodes().at(keyframes_[first + size_t(anchor[size_t(newest)])].id).covariance +
                                           mn->second,
                                       1e-12);
            graph_.UpdateNode(keyframes_.back().id, keyframes_.back().T_W_C, cov_new);
            // The exit edge window[0] -> window[1]: the marginal of window[1] is the RELATIVE uncertainty only when
            // window[0] is what it hangs from; otherwise this solve says nothing about that edge.
            const auto m1 = marg.find(1);
            pending_exit_cov_ = m1 != marg.end() && anchor[1] == 0 ? m1->second : UninformativeCovariance();
            have_pending_exit_ = true;
        }
    }
    // Culling (ORB-SLAM VI-B adapted, design §5). Test 1, "found in > 25% of the frames where predicted visible": a
    // landmark whose LK track is alive in this frame is being found, so it is never culled (culling it cut PnP's
    // correspondences and, on V2_02, doubled the lost frames). Test 2: after cull_after_keyframes keyframes, a
    // landmark must be observed from >= min_keyframe_observations in-window keyframes.
    const uint64_t newest = keyframes_.back().id;
    const std::set<uint64_t> tracked_now(last_frame_.track_ids.begin(), last_frame_.track_ids.end());
    std::map<uint64_t, int> seen;
    for (size_t k = first; k < keyframes_.size(); ++k)
        for (const auto& [id, o] : keyframes_[k].obs) ++seen[id];
    for (auto it = landmarks_.begin(); it != landmarks_.end();) {
        const bool old_enough = newest >= it->second.born_kf + uint64_t(cfg_.ba.cull_after_keyframes);
        if (old_enough && !tracked_now.count(it->first) && seen[it->first] < cfg_.ba.min_keyframe_observations) {
            for (size_t k = first; k < keyframes_.size(); ++k) keyframes_[k].obs.erase(it->first);
            it = landmarks_.erase(it);
            ++out.ba_culled;
        } else {
            ++it;
        }
    }
    last_.T_W_C = keyframes_.back().T_W_C;
    prev_T_W_C_ = last_.T_W_C;
    last_kf_t_ns_ = t_ns;
    return out;
}

}  // namespace ariadne::slam
