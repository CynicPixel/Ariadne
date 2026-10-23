#pragma once
// Stage 3's tuning (D134, D140), apart from falloff.hpp so the perception config can parse it without OpenCV (the
// test_unit build has none).
#include <string>

namespace ariadne::perception::radiometric {

struct FalloffParams {
    /// calibrated: divide by cos(incidence) when the calibration says the cosine belongs in the model (D133).
    /// off: never -- the proxy then keeps Lambertian shading. The calibration describes the LIGHT; dividing by an
    /// cos ESTIMATED from stereo depth is a separate estimator choice, and noisy normals can cost more than the
    /// shading they remove. A §31.5 row compares the two.
    std::string incidence = "calibrated";
    /// Surface normals for the incidence term (D140): cv::depthTo3d + cv::RgbdNormals (ptcloud, OpenCV 5),
    /// rgbd_fals | rgbd_linemod | rgbd_sri | rgbd_cross. On a 10 cm-baseline noisy-stereo plane (test_falloff) the
    /// median albedo error is FALS w7 3.9 %, LINEMOD 4.4 %, SRI w7 5.9 %, OpenCV cross 189 %, no incidence 11.6 %. The
    /// hand-rolled 8 px cross product of D134 (4.8 %) lost to FALS and was removed.
    std::string normals = "rgbd_fals";
    int normal_window_px = 7;   // RgbdNormals window size (1, 3, 5 or 7); 7 beat 5 for FALS (3.9 % vs 7.9 %)
    /// LINEMOD's depth-difference threshold, in the depth's units (metres here). OpenCV's default 50 assumes millimetres.
    double linemod_diff_threshold_m = 0.05;
    /// Pixels within this distance of an invalid pixel are excluded from feature_mask (D140, ALGORITHMS §17.4): the
    /// boundary of the valid region (lobe edge, depth holes) moves WITH the camera, and a detector firing on it makes
    /// lamp-anchored keypoints that track perfectly and describe nothing in the world.
    int feature_margin_px = 8;
    double max_gain = 16.0;   // E is floored at max(E) / max_gain before dividing
    double min_rid = 0.05;    // below this the pixel is outside the lamp's lobe: invalid
    double min_cos = 0.15;    // grazing surfaces: incidence too uncertain to divide by
};

}  // namespace ariadne::perception::radiometric
