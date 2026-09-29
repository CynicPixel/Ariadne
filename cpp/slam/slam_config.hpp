#pragma once
// ---------------------------------------------------------------------------
// Placeholder stereo-VO parameters (plan 1.5; arch §8.3, §11.2-11.6). Strictly
// loaded from configs/slam/*.yaml and named in the run manifest
// (slam_config), because thresholds that decide which landmarks exist and when
// keyframes are created change every trajectory metric downstream.
// OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"

#include <string>

namespace ariadne::slam {

using core::ConfigError;
using core::StrictConfig;

struct TriangulationParams {
    double min_depth_m = 0.5;            // below the stereo search's Z_min nothing is observable
    double max_depth_baselines = 60.0;   // ORB-SLAM3's ThDepth: close points only (60 B ~ 6.6 m on EuRoC)
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
    int max_frames_between = 20;
};

struct BaParams {
    double huber_delta = 2.0;            // §11.3
    int max_iterations = 15;             // §11.6
    double sigma_px = 1.0;
};

struct SlamConfig {
    std::string name;
    TriangulationParams triangulation;
    PnpParams pnp;
    KeyframeParams keyframe;
    BaParams ba;
};

inline void ValidateSlam(const SlamConfig& c) {
    auto fail = [&c](const std::string& why) { throw ConfigError("slam config '" + c.name + "': " + why); };
    if (c.name.empty()) fail("name must not be empty");
    const auto& t = c.triangulation;
    if (!(t.min_depth_m > 0.0) || !(t.max_depth_baselines > 0.0)) fail("depth limits must be > 0");
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
    if (k.max_frames_between < 1) fail("keyframe max_frames_between must be >= 1");
    if (!(c.ba.huber_delta > 0.0) || c.ba.max_iterations < 1 || !(c.ba.sigma_px > 0.0)) fail("bad BA settings");
}

inline SlamConfig ParseSlam(StrictConfig& config) {
    SlamConfig c;
    c.name = config.Require<std::string>("name");
    StrictConfig t = config.Section("triangulation");
    c.triangulation.min_depth_m = t.Require<double>("min_depth_m");
    c.triangulation.max_depth_baselines = t.Require<double>("max_depth_baselines");
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
    c.keyframe.max_frames_between = k.Require<int>("max_frames_between");
    k.RequireAllConsumed();
    StrictConfig b = config.Section("ba");
    c.ba.huber_delta = b.Require<double>("huber_delta");
    c.ba.max_iterations = b.Require<int>("max_iterations");
    c.ba.sigma_px = b.Require<double>("sigma_px");
    b.RequireAllConsumed();
    config.RequireAllConsumed();
    ValidateSlam(c);
    return c;
}

inline SlamConfig LoadSlam(const std::string& path) {
    StrictConfig config = StrictConfig::FromFile(path);
    return ParseSlam(config);
}

}  // namespace ariadne::slam
