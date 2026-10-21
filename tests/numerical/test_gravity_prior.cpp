// GRAVITY_PRIOR edge (slam/pose_graph/gravity_prior.hpp, INERTIAL_2.4 §4.10): the Jacobian for the left perturbation
// of T_W_C against central differences (large rotations; the translation block exactly zero), zero residual when the
// pose agrees, blindness to yaw about gravity, and the cost against GTSAM's AttitudeFactor form with isotropic
// information (1 - (R g_C . ĝ)^2).
#include "slam/pose_graph/pose_graph.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

SE3 RandomPose(std::mt19937& g) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << 2 * n(g), 2 * n(g), 2 * n(g), 1.5 * n(g), 1.5 * n(g), 1.5 * n(g);
    return core::SE3Exp(xi);
}

GravityPrior RandomPrior(std::mt19937& g) {
    std::normal_distribution<double> n(0.0, 1.0);
    GravityPrior p;
    p.g_C = Vec3(n(g), n(g), n(g)).normalized();
    p.g_ref_W = Vec3(n(g), n(g), n(g)).normalized();
    p.information << 400, 50, 50, 900;
    return p;
}

}  // namespace

TEST(GravityPrior, JacobianMatchesFiniteDifferences) {
    std::mt19937 g(3);
    for (int trial = 0; trial < 20; ++trial) {
        const GravityPrior p = RandomPrior(g);
        const SE3 T = RandomPose(g);
        const GravityPriorLinearization l = LinearizeGravityPrior(p, T);
        const double h = 1e-6;
        for (int c = 0; c < 6; ++c) {
            core::Vec6 d = core::Vec6::Zero();
            d[c] = h;
            const Eigen::Vector2d num =
                (LinearizeGravityPrior(p, core::SE3Exp(d) * T).r - LinearizeGravityPrior(p, core::SE3Exp(-d) * T).r) / (2 * h);
            EXPECT_LT((num - l.J.col(c)).norm(), 1e-7 * (1 + l.J.col(c).norm())) << "trial " << trial << " col " << c;
        }
        EXPECT_TRUE(l.J.leftCols<3>().isZero(0.0)) << trial;
    }
}

TEST(GravityPrior, ZeroWhenThePoseAgreesAndBlindToYaw) {
    std::mt19937 g(5);
    const SE3 T = RandomPose(g);
    GravityPrior p;
    p.g_ref_W = Vec3(0.1, -0.2, -1.0).normalized();
    p.g_C = T.R.transpose() * p.g_ref_W;   // what this pose predicts
    p.information = Eigen::Matrix2d::Identity();
    EXPECT_LT(LinearizeGravityPrior(p, T).r.norm(), 1e-15);
    SE3 yawed = T;   // rotate about gravity (through the world origin): roll and pitch unchanged
    yawed.R = core::SO3Exp(p.g_ref_W * 1.3) * T.R;
    yawed.t = core::SO3Exp(p.g_ref_W * 1.3) * T.t;
    EXPECT_LT(LinearizeGravityPrior(p, yawed).r.norm(), 1e-12);
    SE3 tilted = T;   // a tilt is seen
    tilted.R = core::SO3Exp(p.g_ref_W.cross(Vec3::UnitX()).normalized() * 0.01) * T.R;
    EXPECT_NEAR(LinearizeGravityPrior(p, tilted).r.norm(), std::sin(0.01), 1e-9);
}

TEST(GravityPrior, CostEqualsGtsamAttitudeFactorFormWithIsotropicInformation) {
    std::mt19937 g(7);
    for (int trial = 0; trial < 20; ++trial) {
        GravityPrior p = RandomPrior(g);
        p.information = Eigen::Matrix2d::Identity() * 37.0;
        const SE3 T = RandomPose(g);
        const double c = (T.R * p.g_C).dot(p.g_ref_W);
        EXPECT_NEAR(GravityPriorChi2(p, T), 37.0 * (1.0 - c * c), 1e-10) << trial;
    }
}

TEST(GravityPrior, PoseGraphKeepsThemApartFromTheEdgeCost) {
    PoseGraph graph;
    PoseNode n;
    n.keyframe = 0;
    graph.AddNode(n);
    GravityPrior p;
    p.keyframe = 0;
    p.g_C = Vec3(0, 1, 0);
    p.g_ref_W = Vec3(0, 0.6, 0.8);
    p.information = Eigen::Matrix2d::Identity();
    const uint64_t v = graph.version();
    graph.AddGravityPrior(p);
    EXPECT_EQ(graph.version(), v + 1);
    EXPECT_EQ(graph.Cost(), 0.0);
    EXPECT_NEAR(graph.GravityCost(), 1.0 - 0.36, 1e-12);
    p.keyframe = 7;
    EXPECT_THROW(graph.AddGravityPrior(p), std::invalid_argument);
    p.keyframe = 0;
    p.g_C = Vec3(0, 2, 0);
    EXPECT_THROW(graph.AddGravityPrior(p), std::invalid_argument);
}
