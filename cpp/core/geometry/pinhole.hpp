#pragma once
// ---------------------------------------------------------------------------
// Ideal (rectified, distortion-free) pinhole projection and its analytic
// Jacobians (arch §11.3-11.4). The one projection model the SLAM backend
// differentiates; everything is expressed in Ariadne's single convention.
//
// CONVENTION (§11.3). The optimizer's pose is WORLD -> CAMERA, T_C_W, so
// X_c = T_C_W * X_w, and it is perturbed on the LEFT:
//     T_C_W <- exp(delta) * T_C_W,     delta = [rho; phi]  (SE3Exp's order)
// The reprojection residual is r = pi(T_C_W X_w) - z.
//
// DERIVATION. To first order exp(delta) X_c = X_c + rho + phi x X_c
//                                          = X_c + [I, -[X_c]x] delta,
// so  dX_c/d delta = [ I | -[X_c]x ]   (3x6)   and   dX_c/dX_w = R_C_W (3x3).
// The projection pi(X) = (fx X/Z + cx, fy Y/Z + cy) has
//     dpi/dX = [ fx/Z   0     -fx X/Z^2 ]
//              [ 0      fy/Z  -fy Y/Z^2 ]
// and the chain rule gives both blocks. Finite-difference tests (§11.4) cover
// translation, rotation, point, near-zero depth and wide-FOV rays.
//
// OpenCV's projectPoints is an oracle only for the point and translation
// blocks: its rotation Jacobian is with respect to a Rodrigues vector, not
// this left perturbation.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <Eigen/Core>

namespace ariadne::core {

using Vec2 = Eigen::Vector2d;
using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat26 = Eigen::Matrix<double, 2, 6>;

struct PinholeCamera {
    double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
};

/// A rectified stereo pair: both cameras share `cam`; the right camera sits at
/// +baseline along the left camera's x axis (so its projection of X_c is the
/// left projection of X_c - (baseline, 0, 0)).
struct RectifiedStereo {
    PinholeCamera cam;
    double baseline = 0.0;
};

inline Vec2 Project(const PinholeCamera& k, const Vec3& X_c) {
    return {k.fx * X_c.x() / X_c.z() + k.cx, k.fy * X_c.y() / X_c.z() + k.cy};
}

/// Back-project pixel (u, v) at depth z into the camera frame.
inline Vec3 Unproject(const PinholeCamera& k, const Vec2& uv, double z) {
    return {(uv.x() - k.cx) / k.fx * z, (uv.y() - k.cy) / k.fy * z, z};
}

/// dpi/dX_c, 2x3.
inline Mat23 ProjectJacobian(const PinholeCamera& k, const Vec3& X_c) {
    const double iz = 1.0 / X_c.z();
    const double iz2 = iz * iz;
    Mat23 J;
    J << k.fx * iz, 0.0, -k.fx * X_c.x() * iz2,
         0.0, k.fy * iz, -k.fy * X_c.y() * iz2;
    return J;
}

/// Reprojection Jacobians of r = pi(T_C_W X_w) - z:
///   J_pose  (2x6) w.r.t. the left perturbation [rho; phi] of T_C_W,
///   J_point (2x3) w.r.t. X_w.
/// Returns X_c so callers can check depth without recomputing it.
///
/// `offset` shifts the observing camera along the left camera's frame: the
/// right camera of a rectified pair sees X_c - (baseline, 0, 0). The pose
/// derivative still uses [X_c]x of the LEFT frame, because the perturbation
/// acts on T_C_W, not on the shifted point. Returns X_c (left frame).
inline Vec3 ReprojectionJacobians(const PinholeCamera& k, const SE3& T_C_W, const Vec3& X_w,
                                  Mat26* J_pose, Mat23* J_point,
                                  const Vec3& offset = Vec3::Zero()) {
    const Vec3 X_c = T_C_W * X_w;
    const Mat23 Jp = ProjectJacobian(k, X_c - offset);
    if (J_pose) {
        Eigen::Matrix<double, 3, 6> dX;
        dX.leftCols<3>() = Mat3::Identity();
        dX.rightCols<3>() = -Hat(X_c);
        *J_pose = Jp * dX;
    }
    if (J_point) *J_point = Jp * T_C_W.R;
    return X_c;
}

}  // namespace ariadne::core
