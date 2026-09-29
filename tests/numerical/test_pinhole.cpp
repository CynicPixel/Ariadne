// Analytic reprojection Jacobians vs central finite differences (arch §11.4):
// translation, rotation, point, near-zero depth, wide-FOV rays.
#include "core/geometry/pinhole.hpp"

#include <gtest/gtest.h>

using namespace ariadne::core;

namespace {

const PinholeCamera kCam{436.2, 436.2, 364.4, 257.0};

SE3 SomePose() {
    SE3 T;
    T.R = SO3Exp(Vec3(0.2, -0.35, 0.1));
    T.t = Vec3(0.3, -0.1, 0.5);
    return T;
}

/// Central differences of r(delta) = pi(exp(delta) T X) over the 6 tangent
/// directions [rho; phi] -- the definition the analytic J_pose must match.
Mat26 NumericPoseJacobian(const SE3& T, const Vec3& X, double h) {
    Mat26 J;
    for (int i = 0; i < 6; ++i) {
        Vec6 d = Vec6::Zero();
        d(i) = h;
        const Vec2 plus = Project(kCam, (SE3Exp(d) * T) * X);
        const Vec2 minus = Project(kCam, (SE3Exp(-d) * T) * X);
        J.col(i) = (plus - minus) / (2.0 * h);
    }
    return J;
}

Mat23 NumericPointJacobian(const SE3& T, const Vec3& X, double h) {
    Mat23 J;
    for (int i = 0; i < 3; ++i) {
        Vec3 d = Vec3::Zero();
        d(i) = h;
        J.col(i) = (Project(kCam, T * (X + d)) - Project(kCam, T * (X - d))) / (2.0 * h);
    }
    return J;
}

/// Relative error, normalized by the Jacobian's magnitude so that large
/// entries near zero depth are judged proportionally.
template <typename M>
double RelErr(const M& a, const M& b) {
    return (a - b).norm() / std::max(1.0, b.norm());
}

/// A world point whose camera-frame coordinates under T are X_c.
Vec3 WorldPointAt(const SE3& T, const Vec3& X_c) { return T.inverse() * X_c; }

void CheckAt(const Vec3& X_c, double h, double tol) {
    const SE3 T = SomePose();
    const Vec3 X = WorldPointAt(T, X_c);
    Mat26 Jpose;
    Mat23 Jpt;
    ReprojectionJacobians(kCam, T, X, &Jpose, &Jpt);
    EXPECT_LT(RelErr(Jpose, NumericPoseJacobian(T, X, h)), tol) << "pose block at X_c=" << X_c.transpose();
    EXPECT_LT(RelErr(Jpt, NumericPointJacobian(T, X, h)), tol) << "point block at X_c=" << X_c.transpose();
    // The translation block of J_pose is dpi/dX_c, and the point block is
    // that times R: two independent routes to the same quantity.
    EXPECT_LT(RelErr(Mat23(Jpose.leftCols<3>()), ProjectJacobian(kCam, X_c)), 1e-12);
}

}  // namespace

TEST(Pinhole, ProjectUnprojectRoundTrip) {
    const Vec2 uv(100.25, 400.5);
    EXPECT_LT((Project(kCam, Unproject(kCam, uv, 3.7)) - uv).norm(), 1e-12);
}

TEST(Pinhole, JacobiansMatchFiniteDifferencesTypicalDepth) { CheckAt(Vec3(0.4, -0.2, 3.0), 1e-6, 1e-6); }

// Depth 1 cm: dpi/dX scales as 1/Z and 1/Z^2, so entries are ~1e6 and any
// sign or factor error in the Z terms is amplified, not hidden.
TEST(Pinhole, JacobiansMatchFiniteDifferencesNearZeroDepth) { CheckAt(Vec3(0.004, -0.003, 0.01), 1e-9, 1e-5); }

// Wide-FOV ray: X/Z = 5 (~79 degrees off axis), where the -X/Z^2 terms dominate.
TEST(Pinhole, JacobiansMatchFiniteDifferencesWideFovRay) { CheckAt(Vec3(5.0, -3.0, 1.0), 1e-6, 1e-6); }

// The right camera of a rectified pair: projection of X_c - (B, 0, 0), with the
// pose derivative still taken through the LEFT frame's X_c.
TEST(Pinhole, RightCameraJacobiansMatchFiniteDifferences) {
    const SE3 T = SomePose();
    const Vec3 X = WorldPointAt(T, Vec3(0.5, -0.3, 2.5));
    const Vec3 off(0.11, 0.0, 0.0);
    Mat26 Jpose;
    Mat23 Jpt;
    ReprojectionJacobians(kCam, T, X, &Jpose, &Jpt, off);
    Mat26 Jn;
    for (int i = 0; i < 6; ++i) {
        Vec6 d = Vec6::Zero();
        d(i) = 1e-6;
        Jn.col(i) = (Project(kCam, (SE3Exp(d) * T) * X - off) - Project(kCam, (SE3Exp(-d) * T) * X - off)) / 2e-6;
    }
    Mat23 Jpn;
    for (int i = 0; i < 3; ++i) {
        Vec3 d = Vec3::Zero();
        d(i) = 1e-6;
        Jpn.col(i) = (Project(kCam, T * (X + d) - off) - Project(kCam, T * (X - d) - off)) / 2e-6;
    }
    EXPECT_LT(RelErr(Jpose, Jn), 1e-6);
    EXPECT_LT(RelErr(Jpt, Jpn), 1e-6);
}

// Pure rotation/translation directions separately: a transposed Hat or a
// swapped [rho; phi] order fails one of these even if a norm test passed.
TEST(Pinhole, EachTangentDirectionMatches) {
    const SE3 T = SomePose();
    const Vec3 X = WorldPointAt(T, Vec3(-0.7, 0.3, 2.2));
    Mat26 Jpose;
    ReprojectionJacobians(kCam, T, X, &Jpose, nullptr);
    const Mat26 Jnum = NumericPoseJacobian(T, X, 1e-6);
    for (int i = 0; i < 6; ++i) {
        EXPECT_LT((Jpose.col(i) - Jnum.col(i)).norm(), 1e-5 * std::max(1.0, Jnum.col(i).norm()))
            << "tangent direction " << i << (i < 3 ? " (rho)" : " (phi)");
    }
}
