#pragma once
// ---------------------------------------------------------------------------
// SO(3) / SE(3) — the one Lie-group implementation Ariadne is allowed to have.
//
// CONVENTION (arch §11.3, global, non-negotiable):
//   * Perturbations are LEFT-multiplicative:      T <- exp(dxi) * T
//   * The SE(3) tangent vector is ORDERED [rho; phi]  (translation first,
//     rotation second), matching Sophus. GTSAM orders it the other way; when
//     GTSAM is used as a test oracle (§11.6) the ordering must be swapped at
//     that boundary, never here.
//
// Everything is double precision. No OpenCV type appears in this file, by the
// dependency rule of arch §3 — core must compile and test without OpenCV.
// ---------------------------------------------------------------------------
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <cmath>

namespace ariadne::core {

using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// Below this angle the direct trigonometric forms lose too many digits and we
// switch to Taylor series. 1e-2 looks large; it is chosen deliberately -- see
// the cancellation note on `CoeffC` below, which is the binding constraint.
inline constexpr double kSmallAngle = 1e-2;

/// Skew-symmetric matrix ("hat"): Hat(a) * b == a.cross(b).
inline Mat3 Hat(const Vec3& v) {
    Mat3 m;
    m <<     0.0, -v.z(),  v.y(),
          v.z(),     0.0, -v.x(),
         -v.y(),  v.x(),     0.0;
    return m;
}

/// Inverse of Hat, taking the symmetric part so it is well defined on matrices
/// that are only approximately skew (as they are after floating-point work).
inline Vec3 Vee(const Mat3& m) {
    return Vec3(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1)) * 0.5;
}

// --- Series coefficients -------------------------------------------------
// A = sin(t)/t ,  B = (1-cos t)/t^2 ,  C = (t - sin t)/t^3

inline double CoeffA(double t, double t2) {
    if (t < kSmallAngle) return 1.0 - t2 / 6.0 + t2 * t2 / 120.0;
    return std::sin(t) / t;
}

// TRAP 1: computing (1-cos t) directly annihilates digits for small t -- at
// t=1e-8, cos(t) rounds to exactly 1.0 and the result is 0 instead of 0.5.
// The half-angle identity 1-cos t == 2 sin^2(t/2) has no cancellation at all,
// so we use it unconditionally and only guard the removable singularity at 0.
inline double CoeffB(double t, double t2) {
    if (t < kSmallAngle) return 0.5 - t2 / 24.0 + t2 * t2 / 720.0;
    const double h = 0.5 * t;
    const double s = std::sin(h) / h;
    return 0.5 * s * s;
}

// TRAP 2: (t - sin t) is the worst of the three -- the two terms agree to
// ~t^2/6 in relative terms, so at t=1e-3 the direct form already carries only
// ~6 significant digits. Hence kSmallAngle = 1e-2 rather than something
// tighter: at t=1e-2 the 3-term series truncates at O(t^6/362880) ~ 1e-17
// relative, while the direct form carries a cancellation error of
// eps*|t| / |t - sin t| ~ 2.2e-16 * 1e-2 / 1.67e-7 ~ 1.3e-11 relative. The
// series is better everywhere the direct form is bad, and vice versa; 1e-2 is
// the crossover.
inline double CoeffC(double t, double t2) {
    if (t < kSmallAngle) return 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0;
    return (t - std::sin(t)) / (t2 * t);
}

// --- SO(3) ---------------------------------------------------------------

/// Rodrigues: rotation vector (axis * angle) -> rotation matrix.
inline Mat3 SO3Exp(const Vec3& phi) {
    const double t2 = phi.squaredNorm();
    const double t  = std::sqrt(t2);
    const Mat3   K  = Hat(phi);
    return Mat3::Identity() + CoeffA(t, t2) * K + CoeffB(t, t2) * (K * K);
}

/// Rotation matrix -> rotation vector, the inverse of SO3Exp on (-pi, pi].
///
/// Deliberately routed through a quaternion rather than the textbook
/// phi = theta/(2 sin theta) * Vee(R - R^T). That closed form divides by
/// sin(theta), which goes to zero at theta = pi -- exactly the half-turn case
/// that shows up in practice when a robot reverses direction. atan2 on the
/// quaternion is well conditioned across the whole range.
inline Vec3 SO3Log(const Mat3& R) {
    Eigen::Quaterniond q(R);
    q.normalize();
    if (q.w() < 0.0) q.coeffs() *= -1.0;   // pick theta in [0, pi]
    const double nv = q.vec().norm();
    // q ~ (1, phi/2) as phi -> 0, so phi ~ 2*vec with error O(|phi|^3).
    if (nv < 1e-10) return 2.0 * q.vec();
    const double theta = 2.0 * std::atan2(nv, q.w());
    return (theta / nv) * q.vec();
}

// --- SE(3) ---------------------------------------------------------------

/// A rigid transform. Stored as (R, t) rather than a 4x4 so that the bottom
/// row can never drift away from [0 0 0 1] through accumulated arithmetic.
struct SE3 {
    Mat3 R = Mat3::Identity();
    Vec3 t = Vec3::Zero();

    static SE3 Identity() { return SE3{}; }

    Mat4 matrix() const {
        Mat4 m = Mat4::Identity();
        m.topLeftCorner<3, 3>()  = R;
        m.topRightCorner<3, 1>() = t;
        return m;
    }

    SE3 inverse() const {
        SE3 out;
        out.R = R.transpose();
        out.t = -out.R * t;
        return out;
    }

    SE3 operator*(const SE3& rhs) const {
        SE3 out;
        out.R = R * rhs.R;
        out.t = R * rhs.t + t;
        return out;
    }

    Vec3 operator*(const Vec3& p) const { return R * p + t; }

    /// Re-project R onto SO(3). Repeated composition slowly breaks
    /// orthonormality; call this on long chains rather than hoping.
    void normalize() {
        Eigen::Quaterniond q(R);
        q.normalize();
        R = q.toRotationMatrix();
    }
};

/// The left Jacobian V of SE(3): exp maps [rho; phi] to (SO3Exp(phi), V*rho).
inline Mat3 SE3LeftJacobian(const Vec3& phi) {
    const double t2 = phi.squaredNorm();
    const double t  = std::sqrt(t2);
    const Mat3   K  = Hat(phi);
    return Mat3::Identity() + CoeffB(t, t2) * K + CoeffC(t, t2) * (K * K);
}

/// exp: tangent vector [rho; phi] -> rigid transform.
inline SE3 SE3Exp(const Vec6& xi) {
    const Vec3 rho = xi.head<3>();
    const Vec3 phi = xi.tail<3>();
    SE3 out;
    out.R = SO3Exp(phi);
    out.t = SE3LeftJacobian(phi) * rho;
    return out;
}

/// log: rigid transform -> tangent vector [rho; phi].
///
/// V is inverted numerically rather than via its closed form. V is always
/// invertible for |phi| < 2*pi, a 3x3 inverse is a few nanoseconds, and this
/// removes a whole class of transcription bugs from a coefficient formula
/// that is easy to get subtly wrong. If profiling ever shows this on a hot
/// path, the closed form can be added -- with a test against this version.
inline Vec6 SE3Log(const SE3& T) {
    const Vec3 phi = SO3Log(T.R);
    const Vec3 rho = SE3LeftJacobian(phi).inverse() * T.t;
    Vec6 xi;
    xi.head<3>() = rho;
    xi.tail<3>() = phi;
    return xi;
}

/// Adjoint: the operator satisfying  T * exp(xi) * T^-1 == exp(Adj(T) * xi).
/// This is what lets a perturbation be moved between frames, which is why the
/// [rho; phi] ordering has to be stated -- the block layout below depends on it.
inline Mat6 SE3Adjoint(const SE3& T) {
    Mat6 adj = Mat6::Zero();
    adj.topLeftCorner<3, 3>()     = T.R;
    adj.topRightCorner<3, 3>()    = Hat(T.t) * T.R;
    adj.bottomRightCorner<3, 3>() = T.R;
    return adj;
}

}  // namespace ariadne::core
