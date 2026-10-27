// Chordal initialisation and the MST baseline (design robust-backend.md §3, D70): the SO(3) projection and its
// mandatory determinant fix, SE-Sync's isotropic weights, exact recovery on noise-free graphs (anchor anywhere,
// anisotropic information), and refusal of a graph that is not connected to the anchor.
#include "slam/pose_graph/chordal_init.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Mat3;
using core::Vec3;

namespace {

SE3 RandomPose(std::mt19937& g, double rot, double trans) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}

Mat6 RandomInfo(std::mt19937& g) {   // full, correlated, ~100x spread
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Mat6 L = Mat6::Zero();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) L(i, j) = (i == j) ? 1.0 + 9.0 * std::fabs(u(g)) : 0.5 * u(g);
    return L * L.transpose();
}

/// n poses on a random walk (large rotations: the relaxation must not rely on small angles), the chain plus `loops`
/// random loop edges, all measured EXACTLY (noise-free).
PoseGraph NoiseFree(std::mt19937& g, int n, int loops, std::vector<SE3>& truth) {
    PoseGraph pg;
    truth = {RandomPose(g, 1.0, 3.0)};
    for (int i = 1; i < n; ++i) truth.push_back(truth.back() * RandomPose(g, 0.8, 1.0));
    for (int i = 0; i < n; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C = SE3();   // deliberately wrong start: init must not read it (except the anchor's)
        pg.AddNode(nd);
    }
    auto add = [&](int a, int b) {
        PoseEdge e;
        e.type = EdgeType::kLocalLoop;
        e.from = uint64_t(a), e.to = uint64_t(b);
        e.Z = truth[a].inverse() * truth[b];
        e.information = RandomInfo(g);
        pg.AddEdge(e);
    };
    for (int i = 0; i + 1 < n; ++i) add(i, i + 1);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int k = 0; k < loops; ++k) {
        const int a = pick(g), b = pick(g);
        if (a != b) add(a, b);
    }
    return pg;
}

double MaxPoseError(const std::map<uint64_t, SE3>& est, const std::vector<SE3>& truth) {
    double e = 0.0;
    for (const auto& [id, T] : est) e = std::max(e, core::SE3Log(T.inverse() * truth[id]).norm());
    return e;
}

}  // namespace

TEST(ChordalInit, ProjectionIsNearestRotationAndFixesTheDeterminant) {
    std::mt19937 g(20261018u);
    const Mat3 R = RandomPose(g, 1.0, 0.0).R;
    // Near a rotation: projection recovers it, no fix.
    bool fixed = true;
    const Mat3 Mn = R + 1e-3 * Mat3::Random();
    const Mat3 Rn = ProjectToSO3(Mn, &fixed);
    EXPECT_FALSE(fixed);
    EXPECT_NEAR((Rn.transpose() * Rn - Mat3::Identity()).norm(), 0.0, 1e-12);
    EXPECT_NEAR(Rn.determinant(), 1.0, 1e-12);
    EXPECT_LT((Rn - R).norm(), 2e-3);
    // M = R diag(2, 1.5, -0.5): SVD gives U = R, V = diag(1,1,-1), so U V^T is a REFLECTION. The fix must fire and
    // return R itself (the nearest rotation), not R diag(1,1,-1).
    const Mat3 Mr = R * Eigen::Vector3d(2.0, 1.5, -0.5).asDiagonal();
    const Mat3 Rr = ProjectToSO3(Mr, &fixed);
    EXPECT_TRUE(fixed);
    EXPECT_NEAR(Rr.determinant(), 1.0, 1e-12);
    EXPECT_NEAR((Rr - R).norm(), 0.0, 1e-12);
}

TEST(ChordalInit, IsotropicWeightsFollowSeSyncConvention) {
    // sigma_t = 0.1 m, sigma_r = 0.01 rad: tau = 3 / (3 * 0.01) = 100; kappa = 3 / (2 * 3 * 1e-4) = 5000.
    Mat6 I = Mat6::Zero();
    I.topLeftCorner<3, 3>() = Mat3::Identity() / 0.01;
    I.bottomRightCorner<3, 3>() = Mat3::Identity() / 1e-4;
    const IsotropicWeights w = IsotropicFromInformation(I);
    EXPECT_NEAR(w.tau, 100.0, 1e-9);
    EXPECT_NEAR(w.kappa, 5000.0, 1e-9);
    Mat6 bad = I;
    bad(3, 3) = -1.0;
    EXPECT_THROW(IsotropicFromInformation(bad), std::invalid_argument);
}

TEST(ChordalInit, RecoversNoiseFreeGraphsExactly) {
    std::mt19937 g(20261018u);
    for (int trial = 0; trial < 10; ++trial) {
        std::vector<SE3> truth;
        PoseGraph pg = NoiseFree(g, 40, 30, truth);
        const uint64_t anchor = uint64_t(trial * 3);   // the anchor is not always node 0
        pg.UpdateNode(anchor, truth[anchor], Mat6::Zero());
        InitReport rep;
        const auto chordal = InitialPoses(pg, anchor, PgoInit::kChordal, &rep);
        EXPECT_LT(MaxPoseError(chordal, truth), 1e-9) << "trial " << trial;
        EXPECT_EQ(rep.det_fixes, 0);
        EXPECT_LT(rep.max_projection, 1e-9);   // noise-free: the relaxation is already on SO(3)
        EXPECT_LT(MaxPoseError(InitialPoses(pg, anchor, PgoInit::kMst), truth), 1e-9) << "trial " << trial;
    }
}

TEST(ChordalInit, RefusesAGraphNotConnectedToTheAnchor) {
    std::mt19937 g(7u);
    std::vector<SE3> truth;
    PoseGraph pg = NoiseFree(g, 6, 0, truth);
    PoseNode lone;
    lone.keyframe = 100;
    pg.AddNode(lone);
    EXPECT_THROW(InitialPoses(pg, 0, PgoInit::kChordal), std::runtime_error);
    EXPECT_THROW(InitialPoses(pg, 0, PgoInit::kMst), std::runtime_error);
}
