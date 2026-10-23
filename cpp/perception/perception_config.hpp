#pragma once
// ---------------------------------------------------------------------------
// Perception front-end parameters (arch §7.3, §8, §9.3-9.5) and their
// validators. OpenCV-free on purpose: the checks are arithmetic, and keeping
// them here lets the host test suite (no OpenCV) exercise them.
//
// §7.3: "These are manifest fields, not constants." Keypoint caps feed the §15.7
// byte table, and SGBM's disparity range decides the minimum observable range,
// so every value lives in a named config file that the run manifest references.
//
// The rig (focal length, baseline) is NOT in this file. §9.2: calibration is
// immutable per sensor rig, while these are algorithm choices. The §9.4 check
// joins the two, which is why it takes both.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/exposure_control/exposure_config.hpp"
#include "perception/radiometric/falloff_config.hpp"
#include "perception/radiometric/radiometric_config.hpp"

#include <cmath>
#include <string>

namespace ariadne::perception {

using core::ConfigError;
using core::StrictConfig;

/// A rectified pinhole stereo pair: after rectification both cameras share fx,
/// fy, cx, cy, image rows are epipolar lines, and disparity d = x_left - x_right
/// maps to depth by Z = fx * B / d.
struct RectifiedStereoRig {
    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double baseline_m = 0.0;
};

/// §7.3 keypoint selection: top-k per grid cell, never top-N globally.
struct FeatureParams {
    int grid_cols = 8;
    int grid_rows = 6;
    int max_keypoints = 1200;          // §7.3 "stored after filtering <= 1200"
    int detect_oversample = 4;         // raw detections requested = cap * this
    // ORB (the Phase 1 placeholder engine; demoted to an ablation baseline
    // when XFeat lands). Two FAST thresholds, as in ORB-SLAM: cells that the
    // normal threshold leaves starved are re-detected with the lower one,
    // which is what keeps dark, low-texture regions from going empty.
    int orb_fast_threshold = 20;
    int orb_fast_threshold_min = 7;
    int orb_levels = 8;
    double orb_scale_factor = 1.2;
    int orb_patch_size = 31;
    // 2.1 engine choice (PERCEPTION_2.1.md §4.1). "orb" = the Phase 1 placeholder, kept as the ablation baseline.
    std::string engine = "orb";                    // orb | xfeat
    // XFeat (D50 network; upstream verlab/accelerated_features e92685f).
    std::string xfeat_weights;                     // ATB export (infra/scripts/models.sh xfeat); required for xfeat
    std::string xfeat_dnn_engine = "opencv_dnn_graph";   // opencv_dnn_graph | gemm_reference (D50)
    // Network input. resize: bilinear to xfeat_input_width x height, 0 = each side floored to a multiple of 32
    // (upstream preprocess_tensor: 752x480 -> 736x480). pad: replicate-pad right/bottom to the next multiple of 32.
    std::string xfeat_input = "resize";            // resize | pad
    int xfeat_input_width = 0;
    int xfeat_input_height = 0;
    float xfeat_detection_threshold = 0.05f;       // upstream XFeat default
    int xfeat_nms_kernel = 5;                      // upstream default
};

/// §8.1-8.2 pyramidal Lucas-Kanade tracking.
struct TrackerParams {
    int win_size = 21;                 // LK window (px, square)
    int max_level = 3;                 // pyramid levels above the base
    int term_max_iterations = 30;
    double term_epsilon = 0.01;        // px; stop when the update is smaller
    double min_eig_threshold = 1e-4;   // LK drops points whose 2x2 system is this ill-conditioned
    double fb_threshold_px = 0.5;      // forward-backward round-trip tolerance
    int border_px = 1;                 // tracks this close to the edge are dropped
    int target_tracks = 1200;          // §7.3 "1000-1500 tracked points per frame"
    double min_track_distance_px = 20.0;
    // Detections requested per missing track. ORB keypoints cluster, and the
    // spacing mask then rejects most of a cluster, so asking for exactly the
    // shortfall leaves the tracker far below target (measured: 339 of 1200).
    int replenish_request_factor = 4;
    // 2.1 cadence (PERCEPTION_2.1.md §4.2; arch §7.2, C8). every_frame: Process() replenishes whenever tracks are short
    // (Phase 1 behaviour). keyframe: Process() replenishes only below replenish_min_tracks (the off-cadence recovery
    // path); otherwise new tracks start only when the runtime calls Replenish() on a keyframe's image.
    std::string replenish = "every_frame";   // every_frame | keyframe
    int replenish_min_tracks = 0;
    // 2.1 (§9.9): arch §7.2's "image-gradient quality" step for NEW tracks. A detection becomes a track only if its
    // Shi-Tomasi score (cornerMinEigenVal, blockSize 3) exceeds this x the image maximum: goodFeaturesToTrack's
    // qualityLevel convention (OpenCV 5.0.0 featureselect.cpp). 0 = off (Phase 1 behaviour).
    double replenish_min_eig_quality = 0.0;
    // Epipolar outlier rejection on the surviving tracks.
    double f_ransac_threshold_px = 1.0;
    double f_confidence = 0.99;
    int f_max_iterations = 1000;
    int f_min_points = 15;             // below this, rejection is skipped
    int degraded_min_tracks = 50;      // §8.2 "track support collapses"
};

/// §9.3 SGBM and §9.5 depth uncertainty.
struct StereoParams {
    int min_disparity = 0;
    int num_disparities = 256;         // §9.4: the floor at f=640, B=0.10 m
    int block_size = 5;
    double p1_coefficient = 8.0;       // P1 = p1_coefficient * channels * block_size^2
    double p2_coefficient = 32.0;      // P2 = p2_coefficient * channels * block_size^2
    // Top of §9.3's 10-15: measured (D34), 15 halves the false matches SGBM
    // reports for surfaces nearer than its search range, versus 10.
    int uniqueness_ratio = 15;
    // Top of §9.3's 100-200: measured (D34), SGBM's near-range aliases form
    // blobs of ~120-300 px that a 100 px window lets through.
    int speckle_window_size = 200;
    int speckle_range = 2;
    int disp12_max_diff = 1;           // left-right consistency, in px
    int pre_filter_cap = 63;
    std::string mode = "sgbm_3way";    // sgbm | hh | sgbm_3way | hh4
    double sigma_disparity_px = 0.5;   // §9.3 "carry sigma_d = 0.25-0.5 px"
    double required_min_range_m = 0.30;  // §9.4: a narrow void needs Z_min <= 0.3 m
    // Dense-support gate (D34): a depth counts as SUPPORTED only where the
    // fraction of valid pixels in a support_window_px square is at least
    // support_min_density. True surfaces are dense; SGBM's near-range aliases
    // are sparse blobs. Consumers that carve free space must use it.
    int support_window_px = 31;
    double support_min_density = 0.75;
    // 2.1 (PERCEPTION_2.1.md §10): §9.3's confidence from cv::ximgproc::DisparityWLSFilter. none = the placeholder
    // (zeros, no right-matcher cost). wls = a right-matcher pass + the filter's confidence map, the left disparity
    // unchanged unless wls_filtered_disparity.
    std::string confidence = "none";        // none | wls
    bool wls_filtered_disparity = false;    // true: `disparity` becomes the WLS output (changes Z and every gate)
    int wls_lrc_threshold = 24;             // left-right check, 1/16 px (OpenCV default: 1.5 px)
    int wls_discontinuity_radius = 0;       // 0 = ceil(0.5 * block_size), OpenCV's rule for SGBM
    double wls_lambda = 8000.0;             // OpenCV default; shapes only the filtered disparity
    double wls_sigma_color = 1.0;           // OpenCV default; likewise
};

/// Sparse left->right correspondence at keyframes (stereo_lk_matcher.hpp).
struct StereoMatchParams {
    int win_size = 21;
    int max_level = 3;
    double fb_threshold_px = 0.5;
    double max_dy_px = 1.0;           // rectified: a correct match stays on its row
    // Online row refinement (AUDIT_2.3.md, D54): at the first keyframes, fit the residual dy = a + b x + c y of the
    // rectification from round-trip-consistent matches with |dy| <= row_search_px, and fold it into the right map.
    bool row_refinement = true;
    double row_search_px = 4.0;
    int row_min_matches = 100;
    double row_min_span = 0.5;        // fraction of the image the matches must span to fit the slopes b, c
    // Left/right gain-bias match before stereo LK and SGBM (EuRoC per-camera auto-exposure).
    bool exposure_match = true;
};

/// Re-association of untracked landmarks (PERCEPTION_2.1.md §4.3). Absent from a config = off, and the run is the
/// pre-2.1 pipeline byte for byte.
struct ReassociationParams {
    bool enabled = false;
    // Lost mode runs on a frame whose live tracks carry fewer than this many healthy-landmark correspondences,
    // BEFORE that frame's PnP. 2 x pnp.min_correspondences: it starts while PnP still barely passes.
    int trigger_correspondences = 30;
    double ratio = 0.75;              // Lowe's ratio; ORB-SLAM2 Relocalization() uses ORBmatcher(0.75) (Tracking.cc)
    int max_hamming = 50;             // binary descriptors: ORB-SLAM2 ORBmatcher::TH_LOW = 50 (ORBmatcher.cc)
    double max_l2 = 1.0;              // float descriptors (XFeat, unit-norm 64-D): unused until 4.1
    double ransac_threshold_px = 2.0; // reprojection inlier threshold of the verifying PnP
    int min_inliers = 12;             // = slam pnp.min_inliers: below it the relabels are not applied
    bool admit_bridged = false;       // ablation A1b: candidates born at bridged keyframes too
};

struct PerceptionConfig {
    std::string name;
    FeatureParams features;
    TrackerParams tracker;
    StereoParams stereo;
    StereoMatchParams stereo_match;
    radiometric::RadiometricParams radiometric;   // 2.2; mode off unless the config has a radiometric: section
    radiometric::FalloffStageParams falloff;      // 5.2 Stage 3 (D146); mode off unless the config has a falloff: section
    exposure::ExposureControlStage exposure_control;   // 5.3 (D147); mode off unless the config has the section
    ReassociationParams reassociation;
};

/// The largest disparity SGBM searches: disparities run from min_disparity to
/// min_disparity + num_disparities - 1 inclusive.
inline int MaxSearchedDisparity(const StereoParams& s) {
    return s.min_disparity + s.num_disparities - 1;
}

/// §9.4: the nearest depth the configured search can observe. Anything closer
/// has a disparity outside the search range.
inline double MinObservableRange(const StereoParams& s, const RectifiedStereoRig& rig) {
    return rig.fx * rig.baseline_m / static_cast<double>(MaxSearchedDisparity(s));
}

inline void ValidateRig(const RectifiedStereoRig& rig) {
    auto fail = [](const std::string& why) { throw ConfigError("stereo rig: " + why); };
    if (rig.width <= 0 || rig.height <= 0) fail("image size must be positive");
    if (!(rig.fx > 0.0) || !(rig.fy > 0.0)) fail("focal lengths must be > 0");
    if (!(rig.baseline_m > 0.0)) fail("baseline must be > 0");
}

/// §9.4's validator: refuse a (f, B, disparity range) combination whose
/// minimum observable range exceeds what the mission requires. Without it, a
/// baseline change produces a blind zone that looks like sparse depth rather
/// than like a misconfiguration.
inline void ValidateStereoAgainstRig(const StereoParams& s, const RectifiedStereoRig& rig) {
    ValidateRig(rig);
    const double z_min = MinObservableRange(s, rig);
    if (z_min > s.required_min_range_m) {
        throw ConfigError(
            "stereo: min observable range " + std::to_string(z_min) + " m (f=" +
            std::to_string(rig.fx) + " px, B=" + std::to_string(rig.baseline_m) +
            " m, max disparity " + std::to_string(MaxSearchedDisparity(s)) +
            ") exceeds required_min_range_m=" + std::to_string(s.required_min_range_m) +
            " m; raise num_disparities or accept the blind zone explicitly (§9.4)");
    }
    // Columns x < min_disparity + num_disparities have no full search range in
    // the right image, so SGBM cannot produce valid disparity there.
    if (s.min_disparity + s.num_disparities >= rig.width) {
        throw ConfigError("stereo: disparity range covers the whole image width");
    }
}

inline void ValidatePerception(const PerceptionConfig& c) {
    auto fail = [&c](const std::string& why) {
        throw ConfigError("perception config '" + c.name + "': " + why);
    };
    if (c.name.empty()) fail("name must not be empty");
    // Arch §6.2 order: flat field -> VST -> falloff. Without Stage 1 the vignette would be divided into the albedo.
    if (c.falloff.Enabled() && !c.radiometric.Enabled()) fail("falloff needs the radiometric stage (flat field) on");
    // D147: the VST noise model is fitted once and gain changes it; auto/on also hold 8 frames before any decision.
    if (c.exposure_control.Enabled() && c.radiometric.Vst()) {
        fail("exposure_control needs radiometric mode off or flat_only (the VST fit does not follow gain changes)");
    }
    // D146: the proxy exists at keyframes only, so detection on it needs the keyframe cadence.
    if (c.falloff.features == "proxy" && c.tracker.replenish != "keyframe") {
        fail("falloff features: proxy needs tracker replenish: keyframe");
    }

    const FeatureParams& f = c.features;
    if (f.grid_cols < 1 || f.grid_rows < 1) fail("feature grid must be at least 1x1");
    if (f.max_keypoints < f.grid_cols * f.grid_rows) {
        fail("max_keypoints must allow at least one keypoint per grid cell");
    }
    if (f.detect_oversample < 1) fail("detect_oversample must be >= 1");
    if (f.orb_fast_threshold_min < 1 || f.orb_fast_threshold_min > f.orb_fast_threshold) {
        fail("orb_fast_threshold_min must be in [1, orb_fast_threshold]");
    }
    if (f.orb_levels < 1) fail("orb_levels must be >= 1");
    if (!(f.orb_scale_factor > 1.0)) fail("orb_scale_factor must be > 1");
    if (f.orb_patch_size < 2) fail("orb_patch_size must be >= 2");
    if (f.engine != "orb" && f.engine != "xfeat") fail("features.engine must be orb or xfeat");
    if (f.engine == "xfeat" && f.xfeat_weights.empty()) fail("features.engine xfeat needs xfeat_weights");
    if (f.xfeat_dnn_engine != "opencv_dnn_graph" && f.xfeat_dnn_engine != "gemm_reference") {
        fail("xfeat_dnn_engine must be opencv_dnn_graph or gemm_reference");
    }
    if (f.xfeat_input != "resize" && f.xfeat_input != "pad") fail("xfeat_input must be resize or pad");
    if (f.xfeat_input_width < 0 || f.xfeat_input_height < 0 || f.xfeat_input_width % 32 ||
        f.xfeat_input_height % 32) {
        fail("xfeat_input_width/height must be 0 (auto) or positive multiples of 32");
    }
    if (f.xfeat_input == "pad" && (f.xfeat_input_width || f.xfeat_input_height)) {
        fail("xfeat_input pad derives its size; leave xfeat_input_width/height at 0");
    }
    if (!(f.xfeat_detection_threshold >= 0.0f && f.xfeat_detection_threshold < 1.0f)) {
        fail("xfeat_detection_threshold must be in [0, 1)");
    }
    if (f.xfeat_nms_kernel < 1 || f.xfeat_nms_kernel % 2 == 0) fail("xfeat_nms_kernel must be odd and >= 1");

    const TrackerParams& t = c.tracker;
    if (t.win_size < 3 || t.win_size % 2 == 0) fail("tracker win_size must be odd and >= 3");
    if (t.max_level < 0) fail("tracker max_level must be >= 0");
    if (t.term_max_iterations < 1 || !(t.term_epsilon > 0.0)) {
        fail("tracker termination criteria must be positive");
    }
    if (!(t.fb_threshold_px > 0.0)) fail("fb_threshold_px must be > 0");
    if (t.border_px < 0) fail("border_px must be >= 0");
    if (t.target_tracks < 1) fail("target_tracks must be >= 1");
    if (!(t.min_track_distance_px > 0.0)) fail("min_track_distance_px must be > 0");
    if (t.replenish_request_factor < 1) fail("replenish_request_factor must be >= 1");
    if (t.replenish != "every_frame" && t.replenish != "keyframe") fail("tracker replenish must be every_frame or keyframe");
    if (!(t.replenish_min_eig_quality >= 0.0 && t.replenish_min_eig_quality < 1.0)) {
        fail("tracker replenish_min_eig_quality must be in [0, 1)");
    }
    if (t.replenish_min_tracks < 0 || t.replenish_min_tracks > t.target_tracks) {
        fail("tracker replenish_min_tracks must be in [0, target_tracks]");
    }
    if (!(t.f_ransac_threshold_px > 0.0)) fail("f_ransac_threshold_px must be > 0");
    if (!(t.f_confidence > 0.0 && t.f_confidence < 1.0)) fail("f_confidence must be in (0, 1)");
    // Seven points is the minimal fundamental-matrix sample; with fewer there
    // is no model to test against.
    if (t.f_min_points < 8) fail("f_min_points must be >= 8");
    if (t.degraded_min_tracks < 0 || t.degraded_min_tracks > t.target_tracks) {
        fail("degraded_min_tracks must be in [0, target_tracks]");
    }

    const StereoMatchParams& sm = c.stereo_match;
    if (sm.win_size < 3 || sm.win_size % 2 == 0 || sm.max_level < 0 || !(sm.fb_threshold_px > 0.0) ||
        !(sm.max_dy_px > 0.0)) {
        fail("stereo_match: odd win_size >= 3, max_level >= 0, positive thresholds");
    }
    if (!(sm.row_search_px > sm.max_dy_px) || sm.row_min_matches < 3 ||
        !(sm.row_min_span > 0.0 && sm.row_min_span <= 1.0)) {
        fail("stereo_match: row_search_px > max_dy_px, row_min_matches >= 3, row_min_span in (0, 1]");
    }

    const ReassociationParams& ra = c.reassociation;
    if (ra.trigger_correspondences < 1 || !(ra.ratio > 0.0 && ra.ratio <= 1.0) || ra.max_hamming < 0 ||
        !(ra.max_l2 > 0.0) || !(ra.ransac_threshold_px > 0.0) || ra.min_inliers < 4) {
        fail("reassociation: trigger >= 1, ratio in (0, 1], max_hamming >= 0, max_l2 > 0, threshold > 0, "
             "min_inliers >= 4 (the P3P minimal sample + 1)");
    }

    const StereoParams& s = c.stereo;
    // SGBM's own contract: the range is a positive multiple of 16 and the
    // matching block is odd. P2 > P1 is what makes the smoothness penalty
    // tolerate small depth steps (P1) more cheaply than large ones (P2).
    if (s.confidence != "none" && s.confidence != "wls") fail("stereo confidence must be none or wls");
    if (s.wls_filtered_disparity && s.confidence != "wls") fail("wls_filtered_disparity needs confidence: wls");
    if (s.wls_lrc_threshold <= 0 || s.wls_discontinuity_radius < 0 || !(s.wls_lambda > 0.0) ||
        !(s.wls_sigma_color > 0.0)) {
        fail("wls: lrc_threshold > 0, discontinuity_radius >= 0, lambda > 0, sigma_color > 0");
    }
    if (s.num_disparities <= 0 || s.num_disparities % 16 != 0) {
        fail("num_disparities must be a positive multiple of 16");
    }
    if (s.block_size < 1 || s.block_size % 2 == 0) fail("block_size must be odd and >= 1");
    if (!(s.p1_coefficient > 0.0) || !(s.p2_coefficient > s.p1_coefficient)) {
        fail("need 0 < p1_coefficient < p2_coefficient");
    }
    if (s.uniqueness_ratio < 0 || s.speckle_window_size < 0 || s.speckle_range < 0) {
        fail("SGBM filter parameters must be >= 0");
    }
    if (s.mode != "sgbm" && s.mode != "hh" && s.mode != "sgbm_3way" && s.mode != "hh4") {
        fail("stereo mode must be one of sgbm, hh, sgbm_3way, hh4");
    }
    if (!(s.sigma_disparity_px > 0.0)) fail("sigma_disparity_px must be > 0");
    if (!(s.required_min_range_m > 0.0)) fail("required_min_range_m must be > 0");
    if (s.support_window_px < 1 || s.support_window_px % 2 == 0) {
        fail("support_window_px must be odd and >= 1");
    }
    if (!(s.support_min_density > 0.0 && s.support_min_density <= 1.0)) {
        fail("support_min_density must be in (0, 1]");
    }
}

inline PerceptionConfig ParsePerception(StrictConfig& config) {
    PerceptionConfig c;
    c.name = config.Require<std::string>("name");

    StrictConfig fs = config.Section("features");
    FeatureParams& f = c.features;
    f.grid_cols = fs.Require<int>("grid_cols");
    f.grid_rows = fs.Require<int>("grid_rows");
    f.max_keypoints = fs.Require<int>("max_keypoints");
    f.detect_oversample = fs.Optional<int>("detect_oversample", f.detect_oversample);
    f.orb_fast_threshold = fs.Optional<int>("orb_fast_threshold", f.orb_fast_threshold);
    f.orb_fast_threshold_min = fs.Optional<int>("orb_fast_threshold_min", f.orb_fast_threshold_min);
    f.orb_levels = fs.Optional<int>("orb_levels", f.orb_levels);
    f.orb_scale_factor = fs.Optional<double>("orb_scale_factor", f.orb_scale_factor);
    f.orb_patch_size = fs.Optional<int>("orb_patch_size", f.orb_patch_size);
    f.engine = fs.Optional<std::string>("engine", f.engine);
    f.xfeat_weights = fs.Optional<std::string>("xfeat_weights", f.xfeat_weights);
    f.xfeat_dnn_engine = fs.Optional<std::string>("xfeat_dnn_engine", f.xfeat_dnn_engine);
    f.xfeat_input = fs.Optional<std::string>("xfeat_input", f.xfeat_input);
    f.xfeat_input_width = fs.Optional<int>("xfeat_input_width", f.xfeat_input_width);
    f.xfeat_input_height = fs.Optional<int>("xfeat_input_height", f.xfeat_input_height);
    f.xfeat_detection_threshold =
        static_cast<float>(fs.Optional<double>("xfeat_detection_threshold", f.xfeat_detection_threshold));
    f.xfeat_nms_kernel = fs.Optional<int>("xfeat_nms_kernel", f.xfeat_nms_kernel);
    fs.RequireAllConsumed();

    StrictConfig ts = config.Section("tracker");
    TrackerParams& t = c.tracker;
    t.win_size = ts.Optional<int>("win_size", t.win_size);
    t.max_level = ts.Optional<int>("max_level", t.max_level);
    t.term_max_iterations = ts.Optional<int>("term_max_iterations", t.term_max_iterations);
    t.term_epsilon = ts.Optional<double>("term_epsilon", t.term_epsilon);
    t.min_eig_threshold = ts.Optional<double>("min_eig_threshold", t.min_eig_threshold);
    t.fb_threshold_px = ts.Optional<double>("fb_threshold_px", t.fb_threshold_px);
    t.border_px = ts.Optional<int>("border_px", t.border_px);
    t.target_tracks = ts.Require<int>("target_tracks");
    t.min_track_distance_px = ts.Optional<double>("min_track_distance_px", t.min_track_distance_px);
    t.replenish_request_factor =
        ts.Optional<int>("replenish_request_factor", t.replenish_request_factor);
    t.replenish = ts.Optional<std::string>("replenish", t.replenish);
    t.replenish_min_tracks = ts.Optional<int>("replenish_min_tracks", t.replenish_min_tracks);
    t.replenish_min_eig_quality = ts.Optional<double>("replenish_min_eig_quality", t.replenish_min_eig_quality);
    t.f_ransac_threshold_px = ts.Optional<double>("f_ransac_threshold_px", t.f_ransac_threshold_px);
    t.f_confidence = ts.Optional<double>("f_confidence", t.f_confidence);
    t.f_max_iterations = ts.Optional<int>("f_max_iterations", t.f_max_iterations);
    t.f_min_points = ts.Optional<int>("f_min_points", t.f_min_points);
    t.degraded_min_tracks = ts.Optional<int>("degraded_min_tracks", t.degraded_min_tracks);
    ts.RequireAllConsumed();

    StrictConfig ss = config.Section("stereo");
    StereoParams& s = c.stereo;
    s.min_disparity = ss.Optional<int>("min_disparity", s.min_disparity);
    s.num_disparities = ss.Require<int>("num_disparities");
    s.block_size = ss.Require<int>("block_size");
    s.p1_coefficient = ss.Optional<double>("p1_coefficient", s.p1_coefficient);
    s.p2_coefficient = ss.Optional<double>("p2_coefficient", s.p2_coefficient);
    s.uniqueness_ratio = ss.Optional<int>("uniqueness_ratio", s.uniqueness_ratio);
    s.speckle_window_size = ss.Optional<int>("speckle_window_size", s.speckle_window_size);
    s.speckle_range = ss.Optional<int>("speckle_range", s.speckle_range);
    s.disp12_max_diff = ss.Optional<int>("disp12_max_diff", s.disp12_max_diff);
    s.pre_filter_cap = ss.Optional<int>("pre_filter_cap", s.pre_filter_cap);
    s.mode = ss.Optional<std::string>("mode", s.mode);
    s.sigma_disparity_px = ss.Optional<double>("sigma_disparity_px", s.sigma_disparity_px);
    s.required_min_range_m = ss.Require<double>("required_min_range_m");
    s.support_window_px = ss.Optional<int>("support_window_px", s.support_window_px);
    s.support_min_density = ss.Optional<double>("support_min_density", s.support_min_density);
    s.confidence = ss.Optional<std::string>("confidence", s.confidence);
    s.wls_filtered_disparity = ss.Optional<bool>("wls_filtered_disparity", s.wls_filtered_disparity);
    s.wls_lrc_threshold = ss.Optional<int>("wls_lrc_threshold", s.wls_lrc_threshold);
    s.wls_discontinuity_radius = ss.Optional<int>("wls_discontinuity_radius", s.wls_discontinuity_radius);
    s.wls_lambda = ss.Optional<double>("wls_lambda", s.wls_lambda);
    s.wls_sigma_color = ss.Optional<double>("wls_sigma_color", s.wls_sigma_color);
    ss.RequireAllConsumed();

    // Optional section: absent means the defaults above, which are the
    // parameters validated on EuRoC (D35). Present means strictly parsed.
    if (config.Has("stereo_match")) {
        StrictConfig ms = config.Section("stereo_match");
        StereoMatchParams& m = c.stereo_match;
        m.win_size = ms.Optional<int>("win_size", m.win_size);
        m.max_level = ms.Optional<int>("max_level", m.max_level);
        m.fb_threshold_px = ms.Optional<double>("fb_threshold_px", m.fb_threshold_px);
        m.max_dy_px = ms.Optional<double>("max_dy_px", m.max_dy_px);
        m.row_refinement = ms.Require<bool>("row_refinement");
        m.row_search_px = ms.Require<double>("row_search_px");
        m.row_min_matches = ms.Require<int>("row_min_matches");
        m.row_min_span = ms.Require<double>("row_min_span");
        m.exposure_match = ms.Require<bool>("exposure_match");
        ms.RequireAllConsumed();
    }
    // Optional section (2.2): absent means mode off, the radiometric stage does not run.
    if (config.Has("radiometric")) {
        StrictConfig rs = config.Section("radiometric");
        c.radiometric = radiometric::ParseRadiometric(rs);
    }
    // Optional section (5.2, D146): absent means mode off, Stage 3 does not run.
    if (config.Has("falloff")) {
        StrictConfig fs = config.Section("falloff");
        c.falloff = radiometric::ParseFalloff(fs);
    }
    // Optional section (5.3, D147): absent means mode off, the controller does not exist.
    if (config.Has("exposure_control")) {
        StrictConfig es = config.Section("exposure_control");
        c.exposure_control = exposure::ParseExposureControl(es);
    }

    if (config.Has("reassociation")) {
        StrictConfig rs = config.Section("reassociation");
        ReassociationParams& r = c.reassociation;
        r.enabled = rs.Require<bool>("enabled");
        r.trigger_correspondences = rs.Require<int>("trigger_correspondences");
        r.ratio = rs.Require<double>("ratio");
        r.max_hamming = rs.Require<int>("max_hamming");
        r.max_l2 = rs.Optional<double>("max_l2", r.max_l2);
        r.ransac_threshold_px = rs.Require<double>("ransac_threshold_px");
        r.min_inliers = rs.Require<int>("min_inliers");
        r.admit_bridged = rs.Optional<bool>("admit_bridged", r.admit_bridged);
        rs.RequireAllConsumed();
    }

    config.RequireAllConsumed();
    ValidatePerception(c);
    return c;
}

inline PerceptionConfig LoadPerception(const std::string& path) {
    StrictConfig config = StrictConfig::FromFile(path);
    return ParsePerception(config);
}

}  // namespace ariadne::perception
