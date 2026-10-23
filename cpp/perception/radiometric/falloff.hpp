#pragma once
// ---------------------------------------------------------------------------
// Stage 3 of the radiometric front end: light-falloff normalisation (arch §6.2,
// §19.1-19.2; plan 5.2; D134). Output: the ALBEDO-PROXY image.
//
// Under a carried lamp, brightness is mostly geometry: a wall at 1 m is ~10x
// brighter than the same wall at 3 m, and the lamp's lobe darkens the image
// edges. The calibrated lamp model (5.1, D133)
//
//   E(x) = gain_A * p * RID(theta) * [cos(inc)] / (c' + r^2)
//
// predicts that irradiance at every pixel whose depth is known; dividing it out
// leaves (relative) albedo, which is the same surface property from any
// distance. That is what makes a 3DGS map photometrically coherent under a
// moving light (§19.1) and what a feature descriptor would like to see.
//
//   in   linear gray (Stage 4-normalised), depth Z of the same pixels (the §9.6
//        contract: SGBM or RGB-D, NEVER GT depth in production), the camera K,
//        the lamp calibration, the relative lamp power p
//   out  albedo proxy (CV_32F): diffuse reflectance in the calibration's units
//        (the calibration plane reads calib_albedo, 0.5), at the reference
//        exposure; exact only as far as the lamp model and the depth are
//        (D146 corrected this line, which said "relative: 1 = the plane"),
//        and a validity mask
//
// Placement: after rectification and stereo (it needs depth), so it cannot feed
// SGBM. It is a calibrated correction, so §6.3 allows it on the geometry path;
// whether features use it is an ablation row (§31.5).
//
// Noise: dividing by a small E multiplies noise by r^2 / RID. `max_gain` caps
// the division (relative to the brightest predicted irradiance in the frame),
// and pixels outside the lobe (RID < min_rid) are invalid: unlit pixels carry
// no albedo information, and pretending otherwise invents texture from noise.
// ---------------------------------------------------------------------------
#include "perception/radiometric/falloff_params.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ariadne::perception::radiometric {

struct LampCalibration {
    std::string name, rig, render_transfer;
    cv::Vec3d lamp_in_camera{0, 0, 0};   // lamp centre in the (raw) left camera optical frame, metres
    cv::Vec3d axis{0, 0, 1};             // lamp axis in the same frame, unit
    double intensity_ref = 1.0;          // the lamp intensity the calibration was rendered at
    double calib_albedo = 0.5;
    double gain_A = 1.0;
    double c_prime_m2 = 0.0;
    bool cos_incidence = true;
    std::vector<double> rid_theta_deg, rid_value;   // RID table, linear interpolation, 0 beyond the last entry
    double Rid(double theta_deg) const;
};

/// Parse the YAML written by `python -m dataset_tools.lamp_calib fit` (strict: unknown keys fail).
LampCalibration LoadLampCalibration(const std::string& path);


struct AlbedoProxy {
    /// CV_32F. Invalid pixels hold the median valid albedo, not 0, so the image has no step at the valid boundary.
    cv::Mat albedo;
    cv::Mat valid;          // CV_8U, 255 = the proxy is defined here
    /// CV_8U: `valid` eroded by feature_margin_px. THE mask any detector or descriptor on `albedo` must use.
    cv::Mat feature_mask;
    double e_max = 0.0;
};

/// `gray`: CV_32F linear [0, 1] (or CV_8U, read as linear / 255). `z`: CV_32F metres, <= 0 or NaN = unknown.
/// `K`: the camera matrix of these pixels. `R_cam_from_raw`: rotation from the raw left optical frame (where the
/// lamp was calibrated) into these pixels' frame -- the rectification rotation R1, identity if unrectified.
/// `lamp_rel`: lamp power relative to `intensity_ref`.
AlbedoProxy FalloffNormalize(const cv::Mat& gray, const cv::Mat& z, const cv::Matx33d& K, const LampCalibration& lamp,
                             double lamp_rel, const FalloffParams& p = {},
                             const cv::Matx33d& R_cam_from_raw = cv::Matx33d::eye());

}  // namespace ariadne::perception::radiometric
