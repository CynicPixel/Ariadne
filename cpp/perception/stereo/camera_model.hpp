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

struct PinholeRadtan {
    int width = 0;
    int height = 0;
    double fu = 0.0, fv = 0.0, cu = 0.0, cv = 0.0;
    std::array<double, 4> dist{};   // k1, k2, p1, p2
};

struct RawStereoCalibration {
    PinholeRadtan left;
    PinholeRadtan right;
    core::SE3 T_right_left;   // p_right = T_right_left * p_left
};

}  // namespace ariadne::perception
