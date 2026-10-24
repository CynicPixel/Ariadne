// Arch §37.1 geometry gate: SO(3)/SE(3) exp/log, adjoint, and the small-angle
// and near-pi branches that the closed forms get wrong.
#include "core/math/lie.hpp"

#include <gtest/gtest.h>
#include <random>

using namespace ariadne::core;

namespace {

// Fixed seed, explicitly constructed. Determinism discipline starts in the
// test suite: a test that fails one run in fifty on a random input is worse
// than no test, because it trains people to re-run instead of to look.
std::mt19937 Rng() { return std::mt19937(20260830u); }

Vec3 RandomAxis(std::mt19937& g) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec3 v{n(g), n(g), n(g)};
    return v.normalized();
}

// A random rotation vector: the axis first, then the angle. `RandomRotation(g, ang)` leaves the two draws unsequenced
// (an overloaded operator is a call), so the same seed gave x86 and arm64 different vectors (D123).
Vec3 RandomRotation(std::mt19937& g, std::uniform_real_distribution<double>& ang) {
    const Vec3 axis = RandomAxis(g);
    return axis * ang(g);
}

constexpr double kTight = 1e-12;

}  // namespace

TEST(Hat, MatchesCrossProduct) {
    auto g = Rng();
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    for (int i = 0; i < 100; ++i) {
        Vec3 a{u(g), u(g), u(g)}, b{u(g), u(g), u(g)};
        EXPECT_LT((Hat(a) * b - a.cross(b)).norm(), kTight);
    }
}

TEST(Hat, VeeInverts) {
    auto g = Rng();
    std::uniform_real_distribution<double> u(-3.0, 3.0);
    for (int i = 0; i < 100; ++i) {
        Vec3 a{u(g), u(g), u(g)};
        EXPECT_LT((Vee(Hat(a)) - a).norm(), kTight);
    }
}

TEST(SO3, ExpProducesValidRotation) {
    auto g = Rng();
    std::uniform_real_distribution<double> ang(-3.0 * M_PI, 3.0 * M_PI);
    for (int i = 0; i < 200; ++i) {
        const Mat3 R = SO3Exp(RandomRotation(g, ang));
        EXPECT_LT((R.transpose() * R - Mat3::Identity()).norm(), 1e-13);
        EXPECT_NEAR(R.determinant(), 1.0, 1e-13);
    }
}

TEST(SO3, LogInvertsExp) {
    auto g = Rng();
    // Stay inside (-pi, pi), where log is the true inverse of exp.
    std::uniform_real_distribution<double> ang(-M_PI + 1e-6, M_PI - 1e-6);
    for (int i = 0; i < 500; ++i) {
        const Vec3 phi = RandomRotation(g, ang);
        EXPECT_LT((SO3Log(SO3Exp(phi)) - phi).norm(), 1e-11) << "phi norm " << phi.norm();
    }
}

// The small-angle branch. exp(phi) -> I + Hat(phi) as phi -> 0; if the series
// coefficients were wrong (or if the direct trig forms were used and cancelled)
// this is where it shows up.
TEST(SO3, SmallAngleBranchStaysAccurate) {
    auto g = Rng();
    for (double scale : {1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 0.0}) {
        const Vec3 phi = RandomAxis(g) * scale;
        const Mat3 R = SO3Exp(phi);
        EXPECT_LT((R.transpose() * R - Mat3::Identity()).norm(), 1e-14) << "scale " << scale;
        EXPECT_LT((SO3Log(R) - phi).norm(), 1e-15 + 1e-9 * scale) << "scale " << scale;
        // First-order check: ||exp(phi) - (I + Hat(phi))|| is O(phi^2).
        // The +1e-15 floor is not slack for the algorithm -- it admits the
        // scale == 0 case, where the answer is exactly 0 and a strict bound of
        // exactly 0 would reject a perfect result.
        EXPECT_LT((R - (Mat3::Identity() + Hat(phi))).norm(), scale * scale + 1e-15);
    }
    EXPECT_TRUE(SO3Exp(Vec3::Zero()).isApprox(Mat3::Identity()));
    EXPECT_LT(SO3Log(Mat3::Identity()).norm(), kTight);
}

// The half-turn case the R - R^T closed form cannot handle: at theta = pi the
// matrix is symmetric, so R - R^T is exactly zero and the axis is lost.
TEST(SO3, NearAndAtPiRecoversAxis) {
    auto g = Rng();
    for (int i = 0; i < 100; ++i) {
        const Vec3 axis = RandomAxis(g);
        for (double theta : {M_PI - 1e-4, M_PI - 1e-9, M_PI}) {
            const Vec3 phi = axis * theta;
            const Vec3 rec = SO3Log(SO3Exp(phi));
            EXPECT_NEAR(rec.norm(), theta, 1e-7);
            // At exactly pi, +axis and -axis are both correct answers.
            const double alignment = std::abs(rec.normalized().dot(axis));
            EXPECT_NEAR(alignment, 1.0, 1e-6);
        }
    }
}

TEST(SE3, ExpLogRoundTrip) {
    auto g = Rng();
    std::uniform_real_distribution<double> ang(-M_PI + 1e-6, M_PI - 1e-6);
    std::uniform_real_distribution<double> pos(-10.0, 10.0);
    for (int i = 0; i < 500; ++i) {
        Vec6 xi;
        xi.head<3>() = Vec3{pos(g), pos(g), pos(g)};
        xi.tail<3>() = RandomRotation(g, ang);
        EXPECT_LT((SE3Log(SE3Exp(xi)) - xi).norm(), 1e-9) << "xi " << xi.transpose();
    }
}

TEST(SE3, SmallMotionRoundTrip) {
    auto g = Rng();
    for (double scale : {1e-2, 1e-4, 1e-8, 1e-12, 0.0}) {
        Vec6 xi;
        xi.head<3>() = RandomAxis(g) * scale;
        xi.tail<3>() = RandomAxis(g) * scale;
        EXPECT_LT((SE3Log(SE3Exp(xi)) - xi).norm(), 1e-15 + 1e-6 * scale) << "scale " << scale;
    }
}

TEST(SE3, CompositionAndInverse) {
    auto g = Rng();
    std::uniform_real_distribution<double> ang(-M_PI + 1e-6, M_PI - 1e-6);
    std::uniform_real_distribution<double> pos(-10.0, 10.0);
    for (int i = 0; i < 200; ++i) {
        Vec6 a, b;
        a.head<3>() = Vec3{pos(g), pos(g), pos(g)};  a.tail<3>() = RandomRotation(g, ang);
        b.head<3>() = Vec3{pos(g), pos(g), pos(g)};  b.tail<3>() = RandomRotation(g, ang);
        const SE3 A = SE3Exp(a), B = SE3Exp(b);

        EXPECT_LT(((A * A.inverse()).matrix() - Mat4::Identity()).norm(), 1e-11);
        // Composition of transforms must agree with composition of matrices.
        EXPECT_LT(((A * B).matrix() - A.matrix() * B.matrix()).norm(), 1e-11);
        // Acting on a point must agree with the homogeneous product.
        const Vec3 p{pos(g), pos(g), pos(g)};
        EXPECT_LT(((A * p) - (A.R * p + A.t)).norm(), kTight);
    }
}

// The defining property of the adjoint, and the reason the [rho; phi] tangent
// ordering has to be pinned: get the ordering wrong and this test fails while
// every round-trip test above still passes.
TEST(SE3, AdjointMovesPerturbationsBetweenFrames) {
    auto g = Rng();
    std::uniform_real_distribution<double> ang(-M_PI + 1e-6, M_PI - 1e-6);
    std::uniform_real_distribution<double> pos(-5.0, 5.0);
    for (int i = 0; i < 200; ++i) {
        Vec6 tv;
        tv.head<3>() = Vec3{pos(g), pos(g), pos(g)};
        tv.tail<3>() = RandomRotation(g, ang);
        const SE3 T = SE3Exp(tv);

        Vec6 xi;                      // small perturbation
        xi.head<3>() = RandomAxis(g) * 1e-3;
        xi.tail<3>() = RandomAxis(g) * 1e-3;

        const Mat4 lhs = (T * SE3Exp(xi) * T.inverse()).matrix();
        const Mat4 rhs = SE3Exp(SE3Adjoint(T) * xi).matrix();
        EXPECT_LT((lhs - rhs).norm(), 1e-9);
    }
}

TEST(SE3, AdjointIsMultiplicativeAndIdentityAtIdentity) {
    EXPECT_TRUE(SE3Adjoint(SE3::Identity()).isApprox(Mat6::Identity()));
    auto g = Rng();
    std::uniform_real_distribution<double> ang(-2.0, 2.0);
    std::uniform_real_distribution<double> pos(-5.0, 5.0);
    for (int i = 0; i < 100; ++i) {
        Vec6 a, b;
        a.head<3>() = Vec3{pos(g), pos(g), pos(g)};  a.tail<3>() = RandomRotation(g, ang);
        b.head<3>() = Vec3{pos(g), pos(g), pos(g)};  b.tail<3>() = RandomRotation(g, ang);
        const SE3 A = SE3Exp(a), B = SE3Exp(b);
        EXPECT_LT((SE3Adjoint(A * B) - SE3Adjoint(A) * SE3Adjoint(B)).norm(), 1e-9);
    }
}

// Long chains are where an unnormalized rotation quietly rots. This asserts the
// drift is real but bounded, and that normalize() actually removes it.
TEST(SE3, NormalizeRemovesAccumulatedDrift) {
    auto g = Rng();
    SE3 acc = SE3::Identity();
    for (int i = 0; i < 20000; ++i) {
        Vec6 xi;
        xi.head<3>() = RandomAxis(g) * 1e-2;
        xi.tail<3>() = RandomAxis(g) * 1e-2;
        acc = acc * SE3Exp(xi);
    }
    acc.normalize();
    EXPECT_LT((acc.R.transpose() * acc.R - Mat3::Identity()).norm(), 1e-14);
    EXPECT_NEAR(acc.R.determinant(), 1.0, 1e-14);
}

// --- SE(3) left Jacobian and its inverse (Phase 2 pose graph, POSE_GRAPH_2.md §2) -----------------------------
// The DEFINITION, by central differences: Log(Exp(a) * Exp(xi)) ~ xi + J_l^-1(xi) a. Cases straddle the series
// switch at |phi| = 0.1, include tiny and near-pi rotations, and large translations (Q's rho dependence).
namespace {
Mat6 NumericLeftJacobianInverse(const Vec6& xi) {
    const SE3 T = SE3Exp(xi);
    Mat6 J;
    const double h = 1e-6;
    for (int k = 0; k < 6; ++k) {
        Vec6 a = Vec6::Zero();
        a(k) = h;
        J.col(k) = (SE3Log(SE3Exp(a) * T) - SE3Log(SE3Exp(-a) * T)) / (2 * h);
    }
    return J;
}
}  // namespace

TEST(SE3, LeftJacobianInverseMatchesTheDefinition) {
    const double angles[] = {0.0, 1e-7, 1e-3, 0.05, 0.0999, 0.1001, 0.5, 1.5, 2.8, 3.05};
    std::mt19937 g = Rng();
    std::normal_distribution<double> n(0.0, 1.0);
    for (double a : angles) {
        for (int rep = 0; rep < 4; ++rep) {
            const Vec3 axis = RandomAxis(g);
            Vec6 xi;
            xi.head<3>() = Vec3{n(g), n(g), n(g)} *
                           (rep % 2 ? 5.0 : 0.3);
            xi.tail<3>() = axis * a;
            const Mat6 an = SE3LeftJacobianInverse(xi);
            const Mat6 num = NumericLeftJacobianInverse(xi);
            EXPECT_LT((an - num).cwiseAbs().maxCoeff(), 2e-6 * std::max(1.0, num.cwiseAbs().maxCoeff()))
                << "angle " << a << " rep " << rep;
            EXPECT_LT((SE3LeftJacobian6(xi) * an - Mat6::Identity()).cwiseAbs().maxCoeff(), 1e-10)
                << "angle " << a;
        }
    }
}

// The series branch and the closed form must agree where they meet.
TEST(SE3, LeftJacobianQIsContinuousAtTheSeriesSwitch) {
    const Vec3 rho(0.7, -1.3, 2.1), axis = Vec3(1, 2, -0.5).normalized();
    const Mat3 below = SE3LeftJacobianQ(rho, axis * (0.1 - 1e-12));
    const Mat3 above = SE3LeftJacobianQ(rho, axis * (0.1 + 1e-12));
    // Relative: the closed forms carry ~1e-12 rounding at t = 0.1 (c3's numerator is ~t^5/60 from O(t) terms).
    EXPECT_LT((below - above).cwiseAbs().maxCoeff(), 1e-11 * above.cwiseAbs().maxCoeff());
}
