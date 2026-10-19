#pragma once
// ---------------------------------------------------------------------------
// Placeholder stereo-VO parameters (plan 1.5; arch §8.3, §11.2-11.6). Strictly
// loaded from configs/slam/*.yaml and named in the run manifest
// (slam_config), because thresholds that decide which landmarks exist and when
// keyframes are created change every trajectory metric downstream.
// OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"

#include <algorithm>
#include <string>

namespace ariadne::slam {

using core::ConfigError;
using core::StrictConfig;

struct TriangulationParams {
    double min_depth_m = 0.5;            // below the stereo search's Z_min nothing is observable
    double max_reprojection_px = 2.0;    // both views (§11.2 "excessive reprojection residual")
    double sgbm_gate_sigma = 3.0;        // §9.6/§11.2: |Z_tri - Z_sgbm| <= k * sigma
    double disparity_sigma_px = 0.5;     // for sigma_Z of the triangulated depth (§9.5)
    double max_relative_sigma = 0.1;     // sigma_Z / Z (§11.2 "high uncertainty")
};

struct PnpParams {
    double ransac_threshold_px = 2.0;
    double confidence = 0.999;
    int max_iterations = 200;
    int min_correspondences = 15;
    int min_inliers = 12;
    double huber_delta = 2.0;            // motion-only refinement (§11.3)
    double sigma_px = 1.0;               // assumed keypoint noise, scales the covariance
};

struct KeyframeParams {
    // ORB-SLAM3's rule: a keyframe when the frame tracks fewer than this share
    // of the landmarks the reference keyframe had. The richer §8.3 score
    // (translation, rotation, parallax, uncertainty) replaces it in Phase 2.
    double ref_ratio = 0.75;
    int min_tracked_landmarks = 50;
    double max_interval_s = 1.0;   // a keyframe at least this often, in TIME (a frame count breaks under dropped frames)
};

struct BaParams {
    double huber_delta = 2.0;            // §11.3
    int max_iterations = 15;             // §11.6
    double sigma_px = 1.0;
    // 2.3 (LOCAL_SLAM_2.3.md): "two_frame_stub" (1.5, the ablation baseline; the default, so v1 configs are
    // unchanged) or "window" (OpenCV LevMarqBase windowed BA). Every window key is REQUIRED when backend == window.
    std::string backend = "two_frame_stub";
    int window_size = 10;                // DSO N_f = 7 ("more than 7 ... marginal"); catalogue 10-15
    double chi2_reject = 5.991;          // chi^2_2(0.95): drop an observation after the solve
    int cull_after_keyframes = 2;        // ORB-SLAM VI-B, adapted to stereo (design §5)
    int min_keyframe_observations = 2;
    double rel_energy_tolerance = 1e-6;  // LevMarqBase termination (D51)
    double step_norm_tolerance = 1e-8;
    double min_gradient_tolerance = 1e-8;
};

/// Behaviour while tracking is lost (LOST_FRAMES.md, D56). The defaults are the Phase-1 behaviour, so a config
/// without a `lost:` section is unchanged; a `lost:` section must name every key.
struct LostParams {
    // How a lost frame's pose is predicted: "coast" (constant velocity frozen at loss, the 1.5 behaviour) or
    // "bridge" (rotation measured from the 2-D tracks, H/E model selection; translation from a decaying velocity).
    std::string motion = "coast";
    // When a lost frame becomes a keyframe: "every_frame" (the 1.5 behaviour, and v2's) or "gated" (only when it would
    // let the next frame's PnP succeed, or after keyframe.max_interval_s). gated RAISED lost frames on EuRoC (V2_01
    // 20 -> 80 coasting, 120 bridged): kept as an ablation, rejected for v2 (D56).
    std::string reanchor = "every_frame";
    double bridge_threshold_px = 1.5;   // H / E inlier threshold on the rectified image
    int bridge_min_tracks = 20;         // tracks shared with the anchor frame needed to try the bridge
    double model_ratio = 0.45;          // ORB-SLAM (T-RO 2015 §IV-B): H when S_H / (S_H + S_F) > this
    double velocity_decay_s = 0.5;      // tau of the exponential decay of the pre-loss velocity
    double max_speed_m_s = 3.0;         // a pre-loss speed above this is itself corrupt: clamp
    int reanchor_min_points = 30;       // gated: tracked landmarks + new ones >= this (2 x pnp.min_correspondences)
};

/// imu.mode full (D59, INERTIAL_2.4 §4): Forster preintegration in the window BA. Required iff mode is full.
struct ImuFullParams {
    double noise_inflation_gyro = 3.0;     // §4.1, measured on EuRoC Machine Hall; the sim's IMU is exact (1)
    double noise_inflation_accel = 9.0;
    double bias_prior_inflation = 10.0;    // window[0]'s bias prior covariance = the previous marginal x this
    int init_keyframes = 10;               // consecutive TRACKED window keyframes before initialisation (<= window)
    double init_accel_bias_sigma = 0.2;    // m/s^2, the zero-mean b_a prior of the initialisation
    double init_gravity_tolerance = 0.1;   // relative |g| error that refuses an initialisation
    double init_max_chi2_per_dof = 1e9;    // the initialisation's fit worse than this refuses (initialization.hpp)
    std::vector<double> refine_after_s = {5.0, 15.0};   // re-run over every keyframe since init (ORB-SLAM3 §V-B)
    bool estimate_gravity = false;         // §4.8 ablation (negative result): gravity's direction in every window
    double repropagate_gyro = 0.01;        // rad/s: re-integrate an interval when its b_g moved more than this
    double repropagate_accel = 0.1;        // m/s^2, the same for b_a (VINS-Mono's thresholds)
    std::string lost_motion = "decay";     // while lost: "decay" (D56/D58 translation) or "propagate" (IMU, §4.4)
};

/// Inertial layer (arch §10, plan 2.4). "none" (default: every pre-2.4 config unchanged), "attitude_only" (§10.7:
/// gyro rotation while lost, gyro-seeded LK, gravity direction; no velocity/bias states in BA) or "full" (D59:
/// velocity and biases in the window BA; attitude_only behaviour until initialised). A config without an `imu:`
/// section is "none".
struct ImuParams {
    std::string mode = "none";
    bool seed_lk = false;           // §10.5: predict LK's starting point from the gyro rotation
    int bias_window = 200;          // tracked pairs in the VO-aided gyro-bias median (attitude.hpp)
    double stationary_s = 2.0;      // PHASE2_PLAN §3 stationary-start test
    double kp = 1.0;                // Mahony et al. 2008 §VI
    double ki = 0.3;
    ImuFullParams full;
};

struct SlamConfig {
    std::string name;
    TriangulationParams triangulation;
    PnpParams pnp;
    KeyframeParams keyframe;
    BaParams ba;
    LostParams lost;
    ImuParams imu;
};

inline void ValidateSlam(const SlamConfig& c) {
    auto fail = [&c](const std::string& why) { throw ConfigError("slam config '" + c.name + "': " + why); };
    if (c.name.empty()) fail("name must not be empty");
    const auto& t = c.triangulation;
    if (!(t.min_depth_m > 0.0)) fail("triangulation min_depth_m must be > 0");
    if (!(t.max_reprojection_px > 0.0) || !(t.sgbm_gate_sigma > 0.0) || !(t.disparity_sigma_px > 0.0) ||
        !(t.max_relative_sigma > 0.0)) {
        fail("triangulation thresholds must be > 0");
    }
    const auto& p = c.pnp;
    if (!(p.ransac_threshold_px > 0.0) || !(p.confidence > 0.0 && p.confidence < 1.0)) fail("bad PnP RANSAC settings");
    if (p.max_iterations < 1 || p.min_correspondences < 4 || p.min_inliers < 4 ||
        p.min_inliers > p.min_correspondences) {
        fail("PnP needs >= 4 points, and min_inliers <= min_correspondences");
    }
    if (!(p.huber_delta > 0.0) || !(p.sigma_px > 0.0)) fail("PnP refinement settings must be > 0");
    const auto& k = c.keyframe;
    if (!(k.ref_ratio > 0.0 && k.ref_ratio <= 1.0)) fail("keyframe ref_ratio must be in (0, 1]");
    if (k.min_tracked_landmarks < p.min_correspondences) {
        fail("keyframe min_tracked_landmarks must be >= pnp.min_correspondences, or PnP starves first");
    }
    if (!(k.max_interval_s > 0.0)) fail("keyframe max_interval_s must be > 0");
    if (!(c.ba.huber_delta > 0.0) || c.ba.max_iterations < 1 || !(c.ba.sigma_px > 0.0)) fail("bad BA settings");
    const LostParams& l = c.lost;
    if (l.motion != "coast" && l.motion != "bridge") fail("lost.motion must be coast or bridge");
    if (l.reanchor != "every_frame" && l.reanchor != "gated") fail("lost.reanchor must be every_frame or gated");
    if (!(l.bridge_threshold_px > 0.0) || l.bridge_min_tracks < 8 || !(l.model_ratio > 0.0 && l.model_ratio < 1.0) ||
        !(l.velocity_decay_s > 0.0) || !(l.max_speed_m_s > 0.0)) {
        fail("lost: bridge_threshold_px, velocity_decay_s, max_speed_m_s > 0; bridge_min_tracks >= 8; model_ratio in (0, 1)");
    }
    if (l.reanchor_min_points < p.min_correspondences) fail("lost.reanchor_min_points must be >= pnp.min_correspondences");
    if (c.imu.mode != "none" && c.imu.mode != "attitude_only" && c.imu.mode != "full")
        fail("imu.mode must be none, attitude_only or full");
    if (c.imu.mode == "full") {
        const ImuFullParams& f = c.imu.full;
        if (c.ba.backend != "window") fail("imu.mode full needs ba.backend window");
        if (!(f.noise_inflation_gyro > 0.0) || !(f.noise_inflation_accel > 0.0) || !(f.bias_prior_inflation >= 1.0) ||
            f.init_keyframes < 3 || f.init_keyframes > c.ba.window_size || !(f.init_accel_bias_sigma > 0.0) ||
            !(f.init_gravity_tolerance > 0.0 && f.init_gravity_tolerance < 1.0) || !(f.init_max_chi2_per_dof > 0.0) ||
            !(f.repropagate_gyro > 0.0) || !std::is_sorted(f.refine_after_s.begin(), f.refine_after_s.end()) ||
            (!f.refine_after_s.empty() && !(f.refine_after_s.front() > 0.0)) ||
            !(f.repropagate_accel > 0.0) || (f.lost_motion != "decay" && f.lost_motion != "propagate")) {
            fail("imu.full: inflations > 0, bias_prior_inflation >= 1, 3 <= init_keyframes <= ba.window_size, "
                 "init_accel_bias_sigma > 0, init_gravity_tolerance in (0, 1), init_max_chi2_per_dof > 0, refine_after_s ascending and > 0, repropagate_* > 0, lost_motion decay|propagate");
        }
    }
    if (c.imu.bias_window < 1 || !(c.imu.stationary_s > 0.0) || !(c.imu.kp > 0.0) || !(c.imu.ki >= 0.0)) {
        fail("imu: bias_window >= 1, stationary_s > 0, kp > 0, ki >= 0");
    }
    if (c.ba.backend == "window" &&
        (c.ba.window_size < 2 || !(c.ba.chi2_reject > 0.0) || c.ba.cull_after_keyframes < 1 ||
         c.ba.min_keyframe_observations < 1 || !(c.ba.rel_energy_tolerance > 0.0) ||
         !(c.ba.step_norm_tolerance > 0.0) || !(c.ba.min_gradient_tolerance > 0.0))) {
        fail("bad window BA settings");
    }
}

inline SlamConfig ParseSlam(StrictConfig& config) {
    SlamConfig c;
    c.name = config.Require<std::string>("name");
    StrictConfig t = config.Section("triangulation");
    c.triangulation.min_depth_m = t.Require<double>("min_depth_m");
    c.triangulation.max_reprojection_px = t.Require<double>("max_reprojection_px");
    c.triangulation.sgbm_gate_sigma = t.Require<double>("sgbm_gate_sigma");
    c.triangulation.disparity_sigma_px = t.Require<double>("disparity_sigma_px");
    c.triangulation.max_relative_sigma = t.Require<double>("max_relative_sigma");
    t.RequireAllConsumed();
    StrictConfig p = config.Section("pnp");
    c.pnp.ransac_threshold_px = p.Require<double>("ransac_threshold_px");
    c.pnp.confidence = p.Require<double>("confidence");
    c.pnp.max_iterations = p.Require<int>("max_iterations");
    c.pnp.min_correspondences = p.Require<int>("min_correspondences");
    c.pnp.min_inliers = p.Require<int>("min_inliers");
    c.pnp.huber_delta = p.Require<double>("huber_delta");
    c.pnp.sigma_px = p.Require<double>("sigma_px");
    p.RequireAllConsumed();
    StrictConfig k = config.Section("keyframe");
    c.keyframe.ref_ratio = k.Require<double>("ref_ratio");
    c.keyframe.min_tracked_landmarks = k.Require<int>("min_tracked_landmarks");
    c.keyframe.max_interval_s = k.Require<double>("max_interval_s");
    k.RequireAllConsumed();
    StrictConfig b = config.Section("ba");
    c.ba.huber_delta = b.Require<double>("huber_delta");
    c.ba.max_iterations = b.Require<int>("max_iterations");
    c.ba.sigma_px = b.Require<double>("sigma_px");
    c.ba.backend = b.Optional<std::string>("backend", "two_frame_stub");
    if (c.ba.backend == "window") {
        c.ba.window_size = b.Require<int>("window_size");
        c.ba.chi2_reject = b.Require<double>("chi2_reject");
        c.ba.cull_after_keyframes = b.Require<int>("cull_after_keyframes");
        c.ba.min_keyframe_observations = b.Require<int>("min_keyframe_observations");
        c.ba.rel_energy_tolerance = b.Require<double>("rel_energy_tolerance");
        c.ba.step_norm_tolerance = b.Require<double>("step_norm_tolerance");
        c.ba.min_gradient_tolerance = b.Require<double>("min_gradient_tolerance");
    } else if (c.ba.backend != "two_frame_stub") {
        throw ConfigError("slam config '" + c.name + "': ba.backend must be two_frame_stub or window");
    }
    b.RequireAllConsumed();
    if (config.Has("lost")) {
        StrictConfig l = config.Section("lost");
        c.lost.motion = l.Require<std::string>("motion");
        c.lost.reanchor = l.Require<std::string>("reanchor");
        c.lost.bridge_threshold_px = l.Require<double>("bridge_threshold_px");
        c.lost.bridge_min_tracks = l.Require<int>("bridge_min_tracks");
        c.lost.model_ratio = l.Require<double>("model_ratio");
        c.lost.velocity_decay_s = l.Require<double>("velocity_decay_s");
        c.lost.max_speed_m_s = l.Require<double>("max_speed_m_s");
        c.lost.reanchor_min_points = l.Require<int>("reanchor_min_points");
        l.RequireAllConsumed();
    }
    if (config.Has("imu")) {
        StrictConfig m = config.Section("imu");
        c.imu.mode = m.Require<std::string>("mode");
        c.imu.seed_lk = m.Require<bool>("seed_lk");
        c.imu.bias_window = m.Require<int>("bias_window");
        c.imu.stationary_s = m.Require<double>("stationary_s");
        c.imu.kp = m.Require<double>("kp");
        c.imu.ki = m.Require<double>("ki");
        if (c.imu.mode == "full") {
            StrictConfig f = m.Section("full");
            c.imu.full.noise_inflation_gyro = f.Require<double>("noise_inflation_gyro");
            c.imu.full.noise_inflation_accel = f.Require<double>("noise_inflation_accel");
            c.imu.full.bias_prior_inflation = f.Require<double>("bias_prior_inflation");
            c.imu.full.init_keyframes = f.Require<int>("init_keyframes");
            c.imu.full.init_accel_bias_sigma = f.Require<double>("init_accel_bias_sigma");
            c.imu.full.init_gravity_tolerance = f.Require<double>("init_gravity_tolerance");
            c.imu.full.init_max_chi2_per_dof = f.Require<double>("init_max_chi2_per_dof");
            c.imu.full.refine_after_s = f.Require<std::vector<double>>("refine_after_s");
            c.imu.full.estimate_gravity = f.Require<bool>("estimate_gravity");
            c.imu.full.repropagate_gyro = f.Require<double>("repropagate_gyro");
            c.imu.full.repropagate_accel = f.Require<double>("repropagate_accel");
            c.imu.full.lost_motion = f.Require<std::string>("lost_motion");
            f.RequireAllConsumed();
        }
        m.RequireAllConsumed();
    }
    config.RequireAllConsumed();
    ValidateSlam(c);
    return c;
}

inline SlamConfig LoadSlam(const std::string& path) {
    StrictConfig config = StrictConfig::FromFile(path);
    return ParseSlam(config);
}

}  // namespace ariadne::slam
