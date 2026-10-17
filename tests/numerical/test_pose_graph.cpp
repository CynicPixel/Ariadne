// Pose graph (arch §12): the §12.2 residual, the left->residual-tangent
// information conversion, versioning, and the ODOMETRY weight invariant.
#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <gtest/gtest.h>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

SE3 Pose(const Vec3& rot, const Vec3& t) {
    SE3 T;
    T.R = core::SO3Exp(rot);
    T.t = t;
    return T;
}

PoseNode Node(uint64_t id, const SE3& T) {
    PoseNode n;
    n.keyframe = id;
    n.T_W_C = T;
    return n;
}

}  // namespace

TEST(PoseGraph, ResidualIsZeroAtTheMeasurement) {
    PoseGraph g;
    const SE3 Xi = Pose(Vec3(0.1, 0.2, -0.3), Vec3(1, 2, 3));
    const SE3 Xj = Pose(Vec3(-0.2, 0.1, 0.4), Vec3(1.5, 2.2, 2.8));
    g.AddNode(Node(0, Xi));
    g.AddNode(Node(1, Xj));
    g.AddEdge(OdometryEdge(0, 1, Xi.inverse() * Xj, core::Mat6::Identity() * 1e-4));
    EXPECT_LT(g.Residual(g.edges()[0]).norm(), 1e-12);
    EXPECT_LT(g.Cost(), 1e-20);
}

// The stored information must describe the §12.2 residual: sample the measured
// relative pose with its LEFT covariance, evaluate r at the TRUE poses, and
// the mean r^T Info r must be ~6 (chi^2, 6 dof). Storing the left covariance
// directly (skipping the Adj(Z^-1) conversion) fails this whenever Z rotates.
TEST(PoseGraph, EdgeInformationMatchesResidualTangentMonteCarlo) {
    const SE3 Xi = Pose(Vec3(0.3, -0.1, 0.2), Vec3(0.5, -1.0, 0.2));
    const SE3 Xj = Pose(Vec3(-0.4, 0.9, 0.3), Vec3(2.0, 0.5, -0.3));   // large relative rotation
    const SE3 Z_true = Xi.inverse() * Xj;
    core::Mat6 L = core::Mat6::Zero();   // an anisotropic, correlated covariance
    L.diagonal() << 0.02, 0.01, 0.03, 0.004, 0.006, 0.002;
    L(3, 0) = 0.003;
    L(5, 1) = -0.002;
    const core::Mat6 cov = L * L.transpose();
    core::PhiloxStream s(3, 0, 0);
    double mean = 0.0;
    const int n = 2000;
    for (int k = 0; k < n; ++k) {
        core::Vec6 z;
        for (int i = 0; i < 6; ++i) z(i) = core::SampleStandardNormal(s);
        const SE3 Z_meas = core::SE3Exp(L * z) * Z_true;   // left-perturbed measurement
        PoseGraph g;
        g.AddNode(Node(0, Xi));
        g.AddNode(Node(1, Xj));
        g.AddEdge(OdometryEdge(0, 1, Z_meas, cov));
        const core::Vec6 r = g.Residual(g.edges()[0]);
        mean += r.dot(g.edges()[0].information * r);
    }
    mean /= n;
    EXPECT_NEAR(mean, 6.0, 0.45);   // sd of the mean ~0.077; +-6 sd, generous for 2nd-order terms
}

TEST(PoseGraph, VersionBumpsOnEveryMutation) {
    PoseGraph g;
    EXPECT_EQ(g.version(), 0u);
    g.AddNode(Node(0, SE3::Identity()));
    g.AddNode(Node(1, SE3::Identity()));
    g.AddEdge(OdometryEdge(0, 1, SE3::Identity(), core::Mat6::Identity()));
    g.UpdateNode(1, SE3::Identity(), core::Mat6::Identity());
    EXPECT_EQ(g.version(), 4u);
}

TEST(PoseGraph, InvariantsAreEnforced) {
    PoseGraph g;
    g.AddNode(Node(0, SE3::Identity()));
    EXPECT_THROW(g.AddNode(Node(0, SE3::Identity())), std::invalid_argument);
    EXPECT_THROW(g.AddEdge(OdometryEdge(0, 7, SE3::Identity(), core::Mat6::Identity())), std::invalid_argument);
    g.AddNode(Node(1, SE3::Identity()));
    PoseEdge e = OdometryEdge(0, 1, SE3::Identity(), core::Mat6::Identity());
    e.gnc_weight = 0.5;
    EXPECT_THROW(g.AddEdge(e), std::invalid_argument);   // ODOMETRY is never down-weighted
}

// An information matrix that is not symmetric PSD makes 1/2 r^T I r unbounded below: AddEdge refuses it. The matrix is
// the SE-Sync cubicle file's first edge (PD diagonal blocks, translation-rotation cross terms that make it indefinite).
TEST(PoseGraph, AddEdgeRefusesInformationThatIsNotPsd) {
    PoseGraph g;
    g.AddNode(Node(0, SE3::Identity()));
    g.AddNode(Node(1, SE3::Identity()));
    const double up[21] = {2.46483e6, 5.3553e6, 0, 0, 0, 0, 1.52034e7, 0, 0, 0, 0, 10, 84022.3, 132748, 0, 10, 0, 0, 10, 0,
                           91520.2};
    core::Mat6 I;
    for (int i = 0, k = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j, ++k) I(i, j) = I(j, i) = up[k];
    PoseEdge e;
    e.from = 0;
    e.to = 1;
    e.information = I;
    EXPECT_THROW(g.AddEdge(e), std::invalid_argument);
    e.information(0, 1) += 1.0;   // not symmetric
    EXPECT_THROW(g.AddEdge(e), std::invalid_argument);
    e.information = core::Mat6::Identity();
    e.information(5, 5) = 0.0;    // PSD and singular: allowed (an uninformative direction)
    EXPECT_NO_THROW(g.AddEdge(e));
    EXPECT_EQ(g.edges().size(), 1u);
}
