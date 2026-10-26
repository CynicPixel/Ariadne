// Pose-graph problem, OpenCV-free (design pose-graph.md §2, §5): finite-difference Jacobians, the gauge null
// space in OUR (left, T_W_C) convention, the gradient, and marginal covariances. Runs in GitHub CI (no OpenCV).
#include "slam/pose_graph/pgo_problem.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec3;

namespace {

std::mt19937 Rng(unsigned s = 20261016u) { return std::mt19937(s); }

SE3 RandomPose(std::mt19937& g, double rot = 1.0, double trans = 3.0) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}

Mat6 RandomInfo(std::mt19937& g) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Mat6 L = Mat6::Zero();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) L(i, j) = (i == j) ? 1.0 + std::fabs(u(g)) : 0.3 * u(g);
    return L * L.transpose();
}

PoseNode Node(uint64_t id, const SE3& T) {
    PoseNode n;
    n.keyframe = id;
    n.T_W_C = T;
    return n;
}

PoseEdge Edge(uint64_t i, uint64_t j, const SE3& Z, const Mat6& info, EdgeType t = EdgeType::kLocalLoop) {
    PoseEdge e;
    e.type = t;
    e.from = i;
    e.to = j;
    e.Z = Z;
    e.information = info;
    return e;
}

/// A connected random graph: a chain plus `loops` extra edges, measurements = truth * noise.
PoseGraph RandomGraph(std::mt19937& g, int n, int loops, double noise, std::vector<SE3>* truth = nullptr) {
    std::vector<SE3> X;
    X.push_back(RandomPose(g));
    for (int i = 1; i < n; ++i) X.push_back(X.back() * RandomPose(g, 0.3, 1.0));
    std::normal_distribution<double> nn(0.0, noise);
    auto noisy = [&](const SE3& Z) {
        core::Vec6 d;
        for (int k = 0; k < 6; ++k) d(k) = nn(g);
        return core::SE3Exp(d) * Z;
    };
    PoseGraph G;
    for (int i = 0; i < n; ++i) G.AddNode(Node(uint64_t(i), X[size_t(i)]));
    for (int i = 0; i + 1 < n; ++i)
        G.AddEdge(Edge(uint64_t(i), uint64_t(i + 1), noisy(X[size_t(i)].inverse() * X[size_t(i + 1)]), RandomInfo(g),
                       EdgeType::kLocalLoop));
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int k = 0; k < loops; ++k) {
        int a = pick(g), b = pick(g);
        if (a == b) b = (a + 2) % n;
        G.AddEdge(Edge(uint64_t(a), uint64_t(b), noisy(X[size_t(a)].inverse() * X[size_t(b)]), RandomInfo(g)));
    }
    if (truth) *truth = X;
    return G;
}

}  // namespace

// The analytic Jacobians against central differences of the residual under LEFT perturbations of each pose.
TEST(PgoProblem, EdgeJacobiansMatchFiniteDifferences) {
    auto g = Rng();
    for (int rep = 0; rep < 50; ++rep) {
        const SE3 Xi = RandomPose(g), Xj = RandomPose(g);
        // Measurements from near-consistent to grossly wrong (large residuals exercise J_l^-1 far from I).
        const SE3 Z = (rep % 3 == 0) ? Xi.inverse() * Xj : RandomPose(g, rep % 3 == 1 ? 0.2 : 1.2);
        for (bool unary : {false, true}) {
            const PoseEdge e = Edge(unary ? 1 : 0, 1, Z, Mat6::Identity());
            const EdgeLinearization l = LinearizeEdge(e, Xi, Xj);
            const double h = 1e-6;
            for (int k = 0; k < 6; ++k) {
                core::Vec6 d = core::Vec6::Zero();
                d(k) = h;
                const auto r = [&](const SE3& A, const SE3& B) { return LinearizeEdge(e, A, B).r; };
                const core::Vec6 nj = (r(Xi, core::SE3Exp(d) * Xj) - r(Xi, core::SE3Exp(-d) * Xj)) / (2 * h);
                EXPECT_LT((nj - l.Jj.col(k)).cwiseAbs().maxCoeff(), 1e-6) << "rep " << rep << " Jj col " << k;
                if (!unary) {
                    const core::Vec6 ni = (r(core::SE3Exp(d) * Xi, Xj) - r(core::SE3Exp(-d) * Xi, Xj)) / (2 * h);
                    EXPECT_LT((ni - l.Ji.col(k)).cwiseAbs().maxCoeff(), 1e-6) << "rep " << rep << " Ji col " << k;
                }
            }
        }
    }
}

// The gauge (catalogue §7.3, re-derived for our convention): the same xi on every node leaves every residual
// unchanged, so the UNANCHORED H annihilates N = [I; I; ...; I] (6 columns), at an arbitrary, non-optimal point.
TEST(PgoProblem, UnanchoredInformationHasExactlyTheGaugeNullSpace) {
    auto g = Rng(7);
    const PoseGraph G = RandomGraph(g, 12, 10, 0.2);
    const PgoProblem p(G, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    p.Linearize(p.poses(), H, grad, /*anchored=*/false);
    const Eigen::MatrixXd Hd(H);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(Hd.rows(), 6);
    for (int i = 0; i < Hd.rows() / 6; ++i) N.block<6, 6>(6 * i, 0) = Eigen::Matrix<double, 6, 6>::Identity();
    EXPECT_LT((Hd * N).norm() / Hd.norm(), 1e-12);
    // ...and nothing else: exactly 6 near-zero eigenvalues.
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hd);
    int zeros = 0;
    for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k) zeros += es.eigenvalues()(k) < 1e-9 * Hd.norm();
    EXPECT_EQ(zeros, 6);
    // Anchored: positive definite, the guard accepts it.
    SparseMat Ha;
    p.Linearize(p.poses(), Ha, grad);
    GuardedLdlt ldlt;
    EXPECT_TRUE(ldlt.Factorize(Ha));
    EXPECT_GT(ldlt.min_pivot(), 0.0);
}

// g = +J^T W r is the gradient of E = 1/2 sum w r^T Omega r with respect to left perturbations.
TEST(PgoProblem, GradientMatchesFiniteDifferenceOfTheEnergy) {
    auto g = Rng(11);
    const PoseGraph G = RandomGraph(g, 8, 6, 0.3);
    const PgoProblem p(G, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    const double E = p.Linearize(p.poses(), H, grad);
    EXPECT_NEAR(E, p.Energy(p.poses()), 1e-12 * std::max(1.0, E));
    const double h = 1e-6;
    for (int k = 0; k < p.dim(); ++k) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(p.dim());
        x(k) = h;
        const double num = (p.Energy(p.OplusX(p.poses(), x)) - p.Energy(p.OplusX(p.poses(), -x))) / (2 * h);
        EXPECT_NEAR(num, grad(k), 1e-5 * std::max(1.0, std::fabs(grad(k)))) << "param " << k;
    }
}

// A node with no edges makes the anchored system singular: the guard must say so instead of returning a step.
TEST(PgoProblem, GuardRejectsADisconnectedNode) {
    auto g = Rng(3);
    PoseGraph G = RandomGraph(g, 5, 0, 0.0);
    G.AddNode(Node(99, RandomPose(g)));
    const PgoProblem p(G, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    p.Linearize(p.poses(), H, grad);
    GuardedLdlt ldlt;
    EXPECT_FALSE(ldlt.Factorize(H));
    EXPECT_THROW(ldlt.Solve(grad), std::logic_error);
}

// Marginals from H X = E vs the dense inverse of H.
TEST(PgoProblem, MarginalCovariancesMatchTheDenseInverse) {
    auto g = Rng(5);
    const PoseGraph G = RandomGraph(g, 7, 5, 0.0);
    const PgoProblem p(G, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    p.Linearize(p.poses(), H, grad);
    const Eigen::MatrixXd Hinv = Eigen::MatrixXd(H).inverse();
    const auto cov = MarginalCovariances(p, {0, 3, 6});
    EXPECT_EQ(cov.at(0), Mat6::Zero());
    for (uint64_t id : {3u, 6u}) {
        const int k = p.IndexOf(id);
        const Mat6 ref = Hinv.block<6, 6>(6 * k, 6 * k);
        EXPECT_LT((cov.at(id) - ref).cwiseAbs().maxCoeff(), 1e-9 * ref.cwiseAbs().maxCoeff()) << id;
    }
}
