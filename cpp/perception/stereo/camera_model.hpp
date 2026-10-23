#pragma once
// ---------------------------------------------------------------------------
// Raw (unrectified) stereo calibration: what a dataset or a calibration tool
// hands the perception stage BEFORE rectification. OpenCV-free, so dataset
// adapters and their tests do not need OpenCV.
//
// PINHOLE + RADIAL-TANGENTIAL (Brown-Conrady) is the model EuRoC's Kalibr
// calibration uses and OpenCV's default: a normalized point (x, y) with
// r^2 = x^2 + y^2 is distorted to
//   x' = x (1 + k1 r^2 + k2 r^4) + 2 p1 x y + p2 (r^2 + 2 x^2)
//   y' = y (1 + k1 r^2 + k2 r^4) + p1 (r^2 + 2 y^2) + 2 p2 x y
// then mapped to pixels by u = fu x' + cu, v = fv y' + cv. The coefficient
// order [k1, k2, p1, p2] is also OpenCV's distCoeffs order.
//
// EQUIDISTANT (Kannala-Brandt KB4; Kalibr's "equidistant", cv::fisheye's model)
// is the wide-angle alternative (Hilti's OAK-D Pro W, 5.6, D143). It bends the
// ray angle, not the planar radius: theta = atan(r), and
//   theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
//   x' = (theta_d / r) x,  y' = (theta_d / r) y
// so the image radius grows like the angle (fu * theta_d) instead of like
// tan(theta), which is how a ~127 deg lens fits the sensor. dist then holds
// [k1, k2, k3, k4], cv::fisheye's D order. The struct keeps its old name; the
// model field says which reading of dist applies, and every consumer must
// branch on it or refuse (a KB4 vector read as radtan is silently wrong).
//
// FRAME CONVENTION, the part that silently breaks rectification: T_A_B maps
// coordinates expressed in frame B into frame A (p_A = T_A_B * p_B). So a
// dataset's T_B_C (body <- camera) composes as
//   T_C1_C0 = T_B_C1^-1 * T_B_C0      (right camera <- left camera)
// which is the (R, T) cv::stereoRectify expects: "rotation and translation from
// the first camera's coordinate system to the second's".
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <array>

namespace ariadne::perception {

enum class DistortionModel { kRadtan, kEquidistant };

struct PinholeRadtan {
    int width = 0;
    int height = 0;
    double fu = 0.0, fv = 0.0, cu = 0.0, cv = 0.0;
    std::array<double, 4> dist{};   // radtan: k1, k2, p1, p2; equidistant: k1, k2, k3, k4
    DistortionModel model = DistortionModel::kRadtan;   // last, so positional {w, h, fu, fv, cu, cv, {d}} still reads radtan
};

struct RawStereoCalibration {
    PinholeRadtan left;
    PinholeRadtan right;
    core::SE3 T_right_left;   // p_right = T_right_left * p_left
};

}  // namespace ariadne::perception
