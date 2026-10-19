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

StereoVo::StereoVo(const SlamConfig& config, const RectifiedStereo& rig, uint64_t seed, uint32_t agent,
                   const VoImuRig* imu)
    : cfg_(config), rig_(rig), seed_(seed), agent_(agent) {
    if (!(rig_.baseline > 0.0) || !(rig_.cam.fx > 0.0)) throw std::invalid_argument("StereoVo: bad rig");
    if (cfg_.imu.mode == "full") {
        if (!imu) throw std::invalid_argument("StereoVo: imu.mode full needs the IMU rig");
        const core::ImuNoise& n = imu->noise;
        if (!(n.gyro_noise_density > 0.0) || !(n.accel_noise_density > 0.0) || !(n.gyro_random_walk > 0.0) ||
            !(n.accel_random_walk > 0.0)) {
            throw std::invalid_argument("StereoVo: imu.mode full needs positive IMU noise densities");
        }
        full_imu_ = true;
        imu_rig_ = *imu;
        T_C_B_ = imu->T_B_C.inverse();
        // INERTIAL_2.4 §4.1: one inflation per sensor, on the white noise and the random walk alike.
        const double ig = cfg_.imu.full.noise_inflation_gyro, ia = cfg_.imu.full.noise_inflation_accel;
        pre_noise_ = {n.gyro_noise_density * ig, n.accel_noise_density * ia};
        gyro_rw_ = n.gyro_random_walk * ig;
        accel_rw_ = n.accel_random_walk * ia;
    }
}

core::SE3 StereoVo::PropagateFromLastKeyframe(int64_t t_ns) const {
    // Forster eq. 33 from the newest keyframe's solved state: R = R_k ΔR, p = p_k + v_k T + ½ g T² + R_k Δp.
    const KeyframeState& k = keyframes_.back();
    const Preintegrated pre = Preintegrate(imu_since_kf_, k.t_ns, t_ns, k.bg, k.ba, pre_noise_);
    const core::SE3 T_W_B = k.T_W_C * T_C_B_;
    const double T = pre.dt();
    core::SE3 out;
    out.R = T_W_B.R * pre.DeltaR(k.bg);
    out.t = T_W_B.t + k.v * T + 0.5 * g_W_ * T * T + T_W_B.R * pre.DeltaP(k.bg, k.ba);
    out = out * imu_rig_.T_B_C;
    out.normalize();
    return out;
}

VoTrackResult StereoVo::Track(const VoFrame& frame) {
    if (frame.track_ids.size() != frame.points.size()) throw std::invalid_argument("StereoVo: ids/points size");
    if (frames_tracked_ > 0 && !(frame.t_ns > prev_t_ns_)) {
        throw std::invalid_argument("StereoVo: frame timestamps must increase");
    }
    if (full_imu_)   // each timestamp once: consecutive frames' slices share their boundary sample
        for (const auto& m : frame.imu)
            if (imu_since_kf_.empty() || m.t_ns > imu_since_kf_.back().t_ns) imu_since_kf_.push_back(m);
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
        // imu_mode attitude_only: the gyro's rotation (2.4 step 2: median 0.019 deg/pair with the bias removed, ~10x
        // the bridge) with the bridge's decaying-velocity translation; the bridge stays the fallback without IMU.
        const bool gyro = cfg_.imu.mode != "none" && frame.has_rotation_prior;
        const bool bridge = gyro || cfg_.lost.motion == "bridge";   // before 2 tracked frames the velocity is zero
        if (bridge && !coasting_) {   // the anchor starts at the last measured frame
            anchor_ = {prev_T_W_C_, prev_frame_.frame_id, {}};
            for (size_t i = 0; i < prev_frame_.track_ids.size(); ++i)
                anchor_.points.emplace(prev_frame_.track_ids[i], prev_frame_.points[i]);
        }
        res.T_W_C = bridge ? BridgePrediction(frame, res) : predicted;
        if (gyro) res.T_W_C.R = prev_T_W_C_.R * frame.R_prev_cur_prior;
        res.source = gyro ? PoseSource::kGyro : bridge ? PoseSource::kBridged : PoseSource::kCoasted;
        // imu_mode full, initialised, lost_motion propagate (INERTIAL_2.4 §4.4): the whole pose from the IMU.
        if (full_imu_ && inertial_ready_ && cfg_.imu.full.lost_motion == "propagate") {
            res.T_W_C = PropagateFromLastKeyframe(frame.t_ns);
            res.source = PoseSource::kInertial;
        }
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

core::Mat6 StereoVo::NextBiasPrior(const core::Mat6& marginal, double dt) const {
    // The next window[0]'s bias prior (INERTIAL_2.4 §4.2, D59 part 3). The current prior, carried over dt by the bias
    // random walk, is information the next window must keep; the marginal adds the window's data on top of it, and
    // ONLY that addition is deflated by imu.full.bias_prior_inflation (the overlapping windows count the same data
    // again). Deflating the whole marginal instead (part 2) loses the prior's own information on every direction the
    // window does not observe -- b_a in a dark window -- and compounds: V2_01, propagate, 10 s outage, |b_a error|
    // 0.15 -> 2.0 m/s^2. Λ_next = Λ_carried + PSD(Λ_marg - Λ_carried) / κ.
    core::Vec6 q;
    q << core::Vec3::Constant(gyro_rw_ * gyro_rw_ * dt), core::Vec3::Constant(accel_rw_ * accel_rw_ * dt);
    const core::Mat6 L_carried = InformationOf<6>(core::Mat6(prior_cov_ + core::Mat6(q.asDiagonal())));
    const core::Mat6 L_marg = InformationOf<6>(core::NearestPSD(marginal, 1e-12));
    const core::Mat6 L_next =
        L_carried + core::NearestPSD(L_marg - L_carried, 0.0) / cfg_.imu.full.bias_prior_inflation;
    return core::NearestPSD(InformationOf<6>(L_next), 1e-12);
}

void StereoVo::FillInertial(BaWindow& w, size_t first, VoKeyframeResult& out) {
    w.imu.T_C_B = T_C_B_;
    w.imu.g_W = g_W_;
    w.imu.gyro_random_walk = gyro_rw_;
    w.imu.accel_random_walk = accel_rw_;
    for (size_t k = first; k < keyframes_.size(); ++k) {
        KeyframeState& kj = keyframes_[k];
        w.imu.v.push_back(kj.v);
        w.imu.bg.push_back(kj.bg);
        w.imu.ba.push_back(kj.ba);
        if (k == first) continue;
        // Re-integrate an interval whose start keyframe's bias moved past the threshold since it was integrated: the
        // first-order bias correction (Forster eq. 44) is only good near its linearisation point. Between solves only,
        // so the energy the LM driver compares never changes under it.
        const KeyframeState& ki = keyframes_[k - 1];
        const bool stale = kj.has_pre && ((ki.bg - kj.pre_from_prev.bg_lin()).norm() > cfg_.imu.full.repropagate_gyro ||
                                          (ki.ba - kj.pre_from_prev.ba_lin()).norm() > cfg_.imu.full.repropagate_accel);
        if (!kj.has_pre || stale) {
            kj.pre_from_prev = Preintegrate(kj.imu_from_prev, ki.t_ns, kj.t_ns, ki.bg, ki.ba, pre_noise_);
            out.imu_repropagated += kj.has_pre;
            kj.has_pre = true;
        }
        w.imu.pre.push_back(kj.pre_from_prev);
    }
    // window[0]'s bias prior. Valid for keyframe prior_kf_; when window[0] has moved past it (a degenerate marginal,
    // or no solve), it is carried forward by the bias random walk: covariance + density^2 dt per axis.
    const KeyframeState& k0 = keyframes_[first];
    if (prior_kf_ != k0.id) {
        const double dt = 1e-9 * double(k0.t_ns - keyframes_.at(size_t(prior_kf_)).t_ns);
        if (!(dt >= 0.0)) throw std::logic_error("StereoVo: the bias prior is for a keyframe after window[0]");
        core::Vec6 q;
        q << core::Vec3::Constant(gyro_rw_ * gyro_rw_ * dt), core::Vec3::Constant(accel_rw_ * accel_rw_ * dt);
        prior_cov_ += core::Mat6(q.asDiagonal());
        prior_kf_ = k0.id;
    }
    w.imu.has_prior = true;
    w.imu.prior_b = prior_b_;
    w.imu.prior_info = InformationOf<6>(prior_cov_);
    if (cfg_.imu.full.estimate_gravity) {
        w.imu.estimate_gravity = true;
        w.imu.has_g_prior = true;
        w.imu.g_prior = g_W_;
        w.imu.g_prior_info = g_prior_cov_.inverse();
    }
}

void StereoVo::TryInitializeInertial(size_t first, VoKeyframeResult& out) {
    // INERTIAL_2.4 §4.3: the newest init_keyframes window keyframes, every one MEASURED (MH_04 and MH_05 start dark:
    // predicted poses never feed it), with their window-refined poses.
    const size_t N = size_t(cfg_.imu.full.init_keyframes);
    if (keyframes_.size() - first < N) return;
    const size_t s0 = keyframes_.size() - N;
    for (size_t k = s0; k < keyframes_.size(); ++k) {
        if (keyframes_[k].source != PoseSource::kMeasured) return;
        const auto& m = keyframes_[k].imu_from_prev;   // the interval must be covered (the IMU may start late)
        if (k > s0 && (m.empty() || m.front().t_ns > keyframes_[k - 1].t_ns || m.back().t_ns < keyframes_[k].t_ns)) return;
    }
    ++init_attempts_;
    std::vector<core::SE3> T_W_B;
    std::vector<std::vector<core::ImuMeasurement>> imu;
    init_t_ns_.clear();
    for (size_t k = s0; k < keyframes_.size(); ++k) {
        T_W_B.push_back(keyframes_[k].T_W_C * T_C_B_);
        init_t_ns_.push_back(keyframes_[k].t_ns);
        if (k > s0) imu.push_back(keyframes_[k].imu_from_prev);
    }
    InertialInitParams p;
    p.gravity_norm_tolerance = cfg_.imu.full.init_gravity_tolerance;
    p.accel_bias_sigma = cfg_.imu.full.init_accel_bias_sigma;
    p.max_chi2_per_dof = cfg_.imu.full.init_max_chi2_per_dof;
    init_ = InitializeInertial(T_W_B, init_t_ns_, imu, pre_noise_, p);
    if (!init_.ok) {
        out.imu_init_refused = true;
        return;
    }
    inertial_ready_ = true;
    out.imu_initialized = true;
    init_kf0_ = keyframes_[s0].id;
    init_done_t_ns_ = keyframes_.back().t_ns;
    refinements_done_ = 0;
    ApplyInertialEstimate(init_, s0, first);
}

void StereoVo::ApplyInertialEstimate(const InertialInitResult& r, size_t s0, size_t first) {
    g_W_ = r.g_W;
    for (size_t k = first; k < keyframes_.size(); ++k) {
        keyframes_[k].bg = r.bg;
        keyframes_[k].ba = r.ba;
        keyframes_[k].has_pre = false;   // integrated afresh at these biases by the next FillInertial
        if (k >= s0) keyframes_[k].v = r.v[k - s0];
    }
    // Window keyframes older than the estimate's first: back from the next one, v_k = v_k+1 - g T - R_k Δv.
    for (size_t k = std::min(s0, keyframes_.size()); k-- > first;) {
        const KeyframeState& kn = keyframes_[k + 1];
        const Preintegrated pre = Preintegrate(kn.imu_from_prev, keyframes_[k].t_ns, kn.t_ns, r.bg, r.ba, pre_noise_);
        keyframes_[k].v = kn.v - g_W_ * pre.dt() - (keyframes_[k].T_W_C * T_C_B_).R * pre.DeltaV(r.bg, r.ba);
    }
    // The next inertial solve's window[0] bias prior: this estimate's own marginal, inflated like every other.
    prior_b_ << r.bg, r.ba;
    if (inertial_prior_set_) {
        // A refinement: its marginal combined with the prior already held, never looser than it (NextBiasPrior).
        const double dt = 1e-9 * double(keyframes_[first].t_ns - keyframes_.at(size_t(prior_kf_)).t_ns);
        prior_cov_ = NextBiasPrior(r.bias_cov, std::max(dt, 0.0));
    } else {
        prior_cov_ = core::NearestPSD(r.bias_cov * cfg_.imu.full.bias_prior_inflation, 1e-12);
        inertial_prior_set_ = true;
    }
    prior_kf_ = keyframes_[first].id;
    g_prior_cov_ = r.gravity_cov * cfg_.imu.full.bias_prior_inflation;
}

void StereoVo::TryRefineInertial(size_t first, VoKeyframeResult& out) {
    const std::vector<double>& due = cfg_.imu.full.refine_after_s;
    if (refinements_done_ >= due.size()) return;
    const double since = 1e-9 * double(keyframes_.back().t_ns - init_done_t_ns_);
    if (since < due[refinements_done_]) return;
    // A refused refinement is retried at the next keyframes until the following one is due (the last: until twice its
    // time), then given up -- counted, never silently.
    const double give_up = refinements_done_ + 1 < due.size() ? due[refinements_done_ + 1] : 2.0 * due.back();
    const size_t s0 = size_t(init_kf0_);
    InertialRefinement rec;
    rec.t_ns = keyframes_.back().t_ns;
    rec.keyframes = int(keyframes_.size() - s0);
    std::vector<core::SE3> T_W_B;
    std::vector<int64_t> t_ns;
    std::vector<std::vector<core::ImuMeasurement>> imu;
    for (size_t k = s0; k < keyframes_.size(); ++k) {
        const auto& m = keyframes_[k].imu_from_prev;
        if (k > s0 && (m.empty() || m.front().t_ns > keyframes_[k - 1].t_ns || m.back().t_ns < keyframes_[k].t_ns)) {
            rec.why = "interval before keyframe " + std::to_string(k) + " not covered";
            break;
        }
        T_W_B.push_back(keyframes_[k].T_W_C * T_C_B_);
        t_ns.push_back(keyframes_[k].t_ns);
        if (k > s0) imu.push_back(m);
    }
    if (rec.why.empty()) {
        InertialInitParams p;
        p.gravity_norm_tolerance = cfg_.imu.full.init_gravity_tolerance;
        p.accel_bias_sigma = cfg_.imu.full.init_accel_bias_sigma;
        p.max_chi2_per_dof = cfg_.imu.full.init_max_chi2_per_dof;
        const InertialInitResult r = InitializeInertial(T_W_B, t_ns, imu, pre_noise_, p);
        rec.ok = r.ok;
        rec.why = r.why;
        rec.chi2_per_dof = r.chi2_per_dof;
        rec.gravity_norm_unconstrained = r.gravity_norm_unconstrained;
        if (r.ok) {
            rec.gravity_change_deg =
                std::acos(std::clamp(r.g_W.normalized().dot(g_W_.normalized()), -1.0, 1.0)) * 180.0 / M_PI;
            ApplyInertialEstimate(r, s0, first);
        }
    }
    refinements_.push_back(rec);
    out.imu_refined = rec.ok;
    out.imu_refine_refused = !rec.ok;
    if (rec.ok || since >= give_up) ++refinements_done_;
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
    kf.t_ns = f.t_ns;
    if (full_imu_) {
        // The interval from the previous keyframe; the next one starts at the last sample at or before this keyframe.
        if (!keyframes_.empty()) kf.imu_from_prev = imu_since_kf_;
        size_t keep = 0;
        for (size_t i = 0; i < imu_since_kf_.size(); ++i)
            if (imu_since_kf_[i].t_ns <= f.t_ns) keep = i;
        imu_since_kf_.erase(imu_since_kf_.begin(), imu_since_kf_.begin() + std::ptrdiff_t(keep));
    }
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
    if (full_imu_ && inertial_ready_) {
        // The new keyframe's velocity and biases: preintegrated from the previous keyframe's solved state.
        KeyframeState& nk = keyframes_.back();
        const KeyframeState& pk = keyframes_[keyframes_.size() - 2];
        nk.pre_from_prev = Preintegrate(nk.imu_from_prev, pk.t_ns, nk.t_ns, pk.bg, pk.ba, pre_noise_);
        nk.has_pre = true;
        nk.v = pk.v + g_W_ * nk.pre_from_prev.dt() + (pk.T_W_C * T_C_B_).R * nk.pre_from_prev.DeltaV(pk.bg, pk.ba);
        nk.bg = pk.bg;
        nk.ba = pk.ba;
    }

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
        if (full_imu_ && inertial_ready_) FillInertial(w, first, out);
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
        // An inertial window is solved even without landmarks: a dark stretch is exactly where the IMU must act.
        if (any_free && (w.inertial() || out.ba_landmarks >= cfg_.pnp.min_correspondences)) {
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
            // §4.8 gauge: the solve estimated gravity's direction with window[0]'s 6-DoF pose fixed; rotate the whole
            // solution about window[0]'s position so gravity is the reference g_W_ again (a 4-DoF gauge in effect:
            // the accelerometer corrects roll/pitch). Every world-frame state StereoVo holds moves with it, or the next
            // frame's PnP would see the map jump.
            bool rotated = false;
            core::SE3 G;
            if (w.inertial() && w.imu.estimate_gravity) {
                G = GravityAlignment(w.imu.g_W, g_W_, w.T_W_C[0].t);
                out.imu_g_prior_sigma_deg =
                    std::sqrt(Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(g_prior_cov_).eigenvalues()(1)) * 180.0 / M_PI;
                out.imu_tilt_deg = core::SO3Log(G.R).norm() * 180.0 / M_PI;
                rotated = out.imu_tilt_deg > 0.0;
                TransformWindow(w, G);
                for (auto& kf : keyframes_) {
                    kf.T_W_C = G * kf.T_W_C;
                    kf.T_W_C.normalize();
                    kf.v = G.R * kf.v;
                }
                for (auto& [id, lm] : landmarks_) lm.X_w = G * lm.X_w;
                prev_T_W_C_ = G * prev_T_W_C_;
                prev2_T_W_C_ = G * prev2_T_W_C_;
                last_.T_W_C = G * last_.T_W_C;
                anchor_.T_W_C = G * anchor_.T_W_C;
            }
            if (w.inertial()) {
                out.ba_inertial = true;
                out.ba_v0_held = w.imu.hold_v0;
                out.ba_inertial_factors = int(K) - 1;
                for (size_t k = first; k < keyframes_.size(); ++k) {
                    keyframes_[k].v = w.imu.v[k - first];
                    keyframes_[k].bg = w.imu.bg[k - first];
                    keyframes_[k].ba = w.imu.ba[k - first];
                }
                for (size_t k = 0; k + 1 < K; ++k) {
                    const InertialLinearization l =
                        LinearizeInertial(w.imu.pre[k], w.T_W_C[k] * T_C_B_, w.imu.v[k], w.imu.bg[k], w.imu.ba[k],
                                          w.T_W_C[k + 1] * T_C_B_, w.imu.v[k + 1], w.imu.g_W);
                    out.ba_inertial_chi2 += l.r.dot(InformationOf<9>(w.imu.pre[k].Sigma()) * l.r);
                }
            }
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
            std::map<int, core::Mat6> marg, bias_marg;
            // Inertial and the window full: window[1] is the next solve's window[0]; its bias marginal is that
            // solve's prior (INERTIAL_2.4 §4.2), from the same factorization.
            const std::vector<int> bias_want = w.inertial() && K == N ? std::vector<int>{1} : std::vector<int>{};
            const bool want_g = !bias_want.empty() && w.imu.estimate_gravity;   // the next window's gravity prior
            Eigen::Matrix2d g_marg = Eigen::Matrix2d::Zero();
            if ((!want.empty() || !bias_want.empty()) &&
                !TryWindowBaCovariances(wc, want, marg, nullptr, &rep, w.inertial() ? &bias_marg : nullptr, bias_want,
                                        want_g ? &g_marg : nullptr))
                ++out.ba_marginal_degenerate;
            if (!bias_want.empty()) {
                const auto b1 = bias_marg.find(1);
                if (b1 != bias_marg.end()) {
                    prior_b_ << w.imu.bg[1], w.imu.ba[1];
                    prior_cov_ = NextBiasPrior(b1->second, 1e-9 * double(keyframes_[first + 1].t_ns - keyframes_[first].t_ns));
                    prior_kf_ = keyframes_[first + 1].id;
                    out.imu_prior = 1;
                    // Gravity's prior for the next window: this marginal, inflated like the biases' (it double-counts
                    // the shared keyframes). Expressed in the basis at the solved direction, which the rotation back
                    // made g_W_ (bases at nearby directions differ by O(tilt)).
                    if (want_g) g_prior_cov_ = g_marg * cfg_.imu.full.bias_prior_inflation;
                } else {
                    out.imu_prior = 2;   // degenerate: the previous priors are carried forward (biases by the walk)
                }
            }
            for (size_t k = first; k < keyframes_.size(); ++k) {
                keyframes_[k].T_W_C = w.T_W_C[k - first];
                if (k + 1 < keyframes_.size() && (k != first || rotated)) {
                    const core::Mat6& c = graph_.nodes().at(keyframes_[k].id).covariance;
                    graph_.UpdateNode(keyframes_[k].id, keyframes_[k].T_W_C, rotated ? core::ChangeFrame(G, c) : c);
                }
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
            out.ba_inlier_dof -= L.dim() - L.dim_i();   // the reprojection residuals fit poses and points only
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
        } else if (w.inertial() && K == N) {
            out.imu_prior = 2;   // no solve: window[0] moves on, its prior is carried forward
        }
        if (full_imu_ && out.ba_ran) {
            if (!inertial_ready_) TryInitializeInertial(first, out);
            else TryRefineInertial(first, out);
        }
    }
    // The raw samples are needed to re-integrate in-window intervals, and, until the last scheduled refinement, every
    // interval since the initialisation.
    const bool refining = inertial_ready_ && refinements_done_ < cfg_.imu.full.refine_after_s.size();
    for (size_t k = 0; full_imu_ && k <= first && k < keyframes_.size(); ++k) {
        if (refining && keyframes_[k].id > init_kf0_) continue;
        std::vector<core::ImuMeasurement>().swap(keyframes_[k].imu_from_prev);
        keyframes_[k].has_pre = false;
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
