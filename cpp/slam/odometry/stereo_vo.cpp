#include "slam/odometry/stereo_vo.hpp"

#include "core/random/philox.hpp"
#include "core/uncertainty/covariance.hpp"
#include "slam/bundle_adjustment/reprojection_lm.hpp"

#include <opencv2/geometry.hpp>

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
    last_frame_ = frame;
    ++frames_since_kf_;
    VoTrackResult res;

    if (keyframes_.empty()) {
        // The first frame defines the world: its rectified left camera.
        res.status = VoStatus::kInitializing;
        res.T_W_C = core::SE3::Identity();
        res.need_keyframe = true;
        last_ = res;
        return res;
    }

    // Constant-velocity prediction, used as the fallback when PnP fails.
    const core::SE3 predicted =
        frames_tracked_ >= 2 ? prev_T_W_C_ * (prev2_T_W_C_.inverse() * prev_T_W_C_) : prev_T_W_C_;

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
        // Never reset the world or drop the frame silently: hold the motion
        // model, say so, and force a keyframe so new landmarks can be made.
        res.status = VoStatus::kLost;
        res.T_W_C = predicted;
        res.covariance_given_map = UninformativeCovariance();
    }

    const int ref = keyframes_.back().tracked_landmarks;
    res.need_keyframe = !ok || res.pnp_inliers < cfg_.keyframe.ref_ratio * ref ||
                        res.pnp_inliers < cfg_.keyframe.min_tracked_landmarks ||
                        frames_since_kf_ >= cfg_.keyframe.max_frames_between;

    prev2_T_W_C_ = prev_T_W_C_;
    prev_T_W_C_ = res.T_W_C;
    ++frames_tracked_;
    last_ = res;
    return res;
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
    out.keyframe_id = kf.id;

    // Landmarks whose track died can never be observed again (no
    // re-association in the placeholder): retire them.
    std::set<uint64_t> alive(f.track_ids.begin(), f.track_ids.end());
    for (auto it = landmarks_.begin(); it != landmarks_.end();) {
        it = alive.count(it->first) ? std::next(it) : landmarks_.erase(it);
    }

    // --- Triangulate new landmarks (§11.2) ---------------------------------
    const double fx = rig_.cam.fx, B = rig_.baseline;
    std::vector<size_t> cand;
    std::vector<cv::Point2d> pl, pr;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t id = f.track_ids[i];
        if (landmarks_.count(id)) {
            kf.obs[id] = {f.points[i], st.right[i], st.valid[i] != 0};
            continue;
        }
        ++out.triangulation.candidates;
        if (!st.valid[i]) { ++out.triangulation.no_stereo; continue; }
        if (!(f.points[i].x - st.right[i].x > 0.0f)) { ++out.triangulation.nonpositive_disparity; continue; }
        cand.push_back(i);
        pl.emplace_back(f.points[i].x, f.points[i].y);
        pr.emplace_back(st.right[i].x, st.right[i].y);
    }
    if (!cand.empty()) {
        // Rectified projection matrices: left K[I|0], right K[I|(-B,0,0)].
        const cv::Mat P1 = (cv::Mat_<double>(3, 4) << fx, 0, rig_.cam.cx, 0, 0, rig_.cam.fy, rig_.cam.cy, 0, 0, 0, 1, 0);
        const cv::Mat P2 = (cv::Mat_<double>(3, 4) << fx, 0, rig_.cam.cx, -fx * B, 0, rig_.cam.fy, rig_.cam.cy, 0, 0, 0, 1, 0);
        cv::Mat X4;
        cv::triangulatePoints(P1, P2, pl, pr, X4);   // linear DLT, 4xN homogeneous
        X4.convertTo(X4, CV_64F);
        const double z_max = cfg_.triangulation.max_depth_baselines * B;
        for (size_t k = 0; k < cand.size(); ++k) {
            const size_t i = cand[k];
            const double w = X4.at<double>(3, static_cast<int>(k));
            const core::Vec3 Xc(X4.at<double>(0, static_cast<int>(k)) / w, X4.at<double>(1, static_cast<int>(k)) / w,
                                X4.at<double>(2, static_cast<int>(k)) / w);
            if (!(Xc.z() >= cfg_.triangulation.min_depth_m && Xc.z() <= z_max)) { ++out.triangulation.depth_range; continue; }
            const core::Vec2 el = core::Project(rig_.cam, Xc) - core::Vec2(pl[k].x, pl[k].y);
            const core::Vec2 er = core::Project(rig_.cam, Xc - core::Vec3(B, 0, 0)) - core::Vec2(pr[k].x, pr[k].y);
            if (el.norm() > cfg_.triangulation.max_reprojection_px || er.norm() > cfg_.triangulation.max_reprojection_px) {
                ++out.triangulation.reprojection;
                continue;
            }
            // §9.5 on the triangulated depth.
            const double sigma_tri = Xc.z() * Xc.z() / (fx * B) * cfg_.triangulation.disparity_sigma_px;
            if (sigma_tri / Xc.z() > cfg_.triangulation.max_relative_sigma) { ++out.triangulation.uncertain; continue; }
            // §9.6: cross-check against dense SGBM where it is trustworthy.
            if (std::isfinite(st.sgbm_z[i])) {
                const double s = std::sqrt(sigma_tri * sigma_tri + double(st.sgbm_sigma[i]) * st.sgbm_sigma[i]);
                if (std::abs(Xc.z() - st.sgbm_z[i]) > cfg_.triangulation.sgbm_gate_sigma * s) {
                    ++out.triangulation.sgbm_inconsistent;
                    continue;
                }
            }
            const uint64_t id = f.track_ids[i];
            landmarks_[id] = {kf.T_W_C * Xc};
            kf.obs[id] = {f.points[i], st.right[i], true};
            ++out.triangulation.created;
        }
    }
    kf.tracked_landmarks = static_cast<int>(kf.obs.size());

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
    frames_since_kf_ = 0;
    return out;
}

}  // namespace ariadne::slam
