// The pose graph on OpenCV 5's LevMarqBase (D51, pose-graph.md §3, §5).
//   1. The DRIVER CONTRACT, pinned with a linear problem: if our reading of levmarq.cpp were wrong (the sign of jtb,
//      the energy scale, solveDecomposed's -right), the first step would go uphill or be mis-scaled.
//   2. A noise-free SE(3) graph from a perturbed start returns the truth.
//   3. A noisy graph reaches a stationary point (gradient ~ 0) and lowers the energy.
#include "slam/pose_graph/pgo_levmarq.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cstring>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;

namespace {

/// f(x) = A x - b. With lambda tiny and no clamping, LM's first step IS the least-squares solution.
class LinearBackend : public cv::detail::LevMarqBackend {
public:
    LinearBackend(Eigen::MatrixXd A, Eigen::VectorXd b) : A_(std::move(A)), b_(std::move(b)) {
        x_ = Eigen::VectorXd::Zero(A_.cols());
        probe_ = x_;
    }
    bool enableGeo() override { return false; }
    bool calcFunc(double& e, bool, bool calcJacobian) override {
        const Eigen::VectorXd r = A_ * probe_ - b_;
        e = 0.5 * r.squaredNorm();
        if (calcJacobian) {
            H_ = A_.transpose() * A_;
            g_ = A_.transpose() * r;
        }
        return true;
    }
    void currentOplusX(const cv::Mat_<double>& d, bool) override {
        probe_ = x_;
        for (int i = 0; i < d.rows; ++i) probe_(i) += d(i);
    }
    void prepareVars() override { probe_ = x_; }
    const cv::Mat_<double> getJtb() override { return Mat(g_); }
    const cv::Mat_<double> getDiag() override { return Mat(H_.diagonal()); }
    void setDiag(const cv::Mat_<double>& d) override {
        for (int i = 0; i < d.rows; ++i) H_(i, i) = d(i);
    }
    void doJacobiScaling(const cv::Mat_<double>&) override {}
    bool decompose() override {
        ldlt_.compute(H_);
        return ldlt_.info() == Eigen::Success;
    }
    bool solveDecomposed(const cv::Mat_<double>& right, cv::Mat_<double>& x) override {
        Eigen::VectorXd r(right.rows);
        for (int i = 0; i < right.rows; ++i) r(i) = right(i);
        x = Mat(ldlt_.solve(-r));
        return true;
    }
    bool calcJtbv(cv::Mat_<double>&) override { return false; }
    void acceptProbe() override {
        x_ = probe_;
        ++accepted;
    }
    Eigen::VectorXd x_;
    int accepted = 0;

private:
    static cv::Mat_<double> Mat(const Eigen::VectorXd& v) {
        cv::Mat_<double> m(int(v.size()), 1);
        for (Eigen::Index i = 0; i < v.size(); ++i) m(int(i)) = v(i);
        return m;
    }
    Eigen::MatrixXd A_, H_;
    Eigen::VectorXd b_, g_, probe_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
};

core::SE3 RandomPose(std::mt19937& g, double rot, double trans) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}

PoseGraph Graph(std::mt19937& g, int n, int loops, double noise, std::vector<core::SE3>& truth) {
    truth.clear();
    truth.push_back(core::SE3::Identity());
    for (int i = 1; i < n; ++i) truth.push_back(truth.back() * RandomPose(g, 0.3, 1.0));
    std::normal_distribution<double> nn(0.0, noise);
    auto noisy = [&](const core::SE3& Z) {
        core::Vec6 d;
        for (int k = 0; k < 6; ++k) d(k) = nn(g);
        return core::SE3Exp(d) * Z;
    };
    PoseGraph G;
    for (int i = 0; i < n; ++i) {
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C = truth[size_t(i)];
        G.AddNode(nd);
    }
    auto add = [&](int a, int b) {
        PoseEdge e;
        e.type = EdgeType::kLocalLoop;
        e.from = uint64_t(a);
        e.to = uint64_t(b);
        e.Z = noisy(truth[size_t(a)].inverse() * truth[size_t(b)]);
        e.information = Mat6::Identity() * 100.0;
        G.AddEdge(e);
    };
    for (int i = 0; i + 1 < n; ++i) add(i, i + 1);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int k = 0; k < loops; ++k) {
        const int a = pick(g), b = pick(g);
        if (a != b) add(a, b);
    }
    return G;
}

void Perturb(PoseGraph& G, std::mt19937& g, double rot, double trans) {
    for (const auto& [id, n] : std::map<uint64_t, PoseNode>(G.nodes())) {
        if (id == 0) continue;
        G.UpdateNode(id, RandomPose(g, rot, trans) * n.T_W_C, n.covariance);
    }
}

}  // namespace

TEST(PgoLevMarq, DriverContractOneStepOnALinearProblem) {
    std::mt19937 g(1);
    std::normal_distribution<double> n(0.0, 1.0);
    Eigen::MatrixXd A(20, 5);
    Eigen::VectorXd b(20);
    for (int i = 0; i < 20; ++i) {
        b(i) = n(g);
        for (int j = 0; j < 5; ++j) A(i, j) = n(g);
    }
    auto be = cv::makePtr<LinearBackend>(A, b);
    PgoSettings s;
    s.initial_lambda = 1e-14;
    s.clamp_diagonal = false;
    s.use_step_quality = false;
    cv::detail::LevMarqBase lm(be, ToOpenCv(s));
    const cv::LevMarq::Report r = lm.optimize();
    const Eigen::VectorXd ls = A.colPivHouseholderQr().solve(b);
    EXPECT_TRUE(r.found);
    EXPECT_LT((be->x_ - ls).cwiseAbs().maxCoeff(), 1e-10);
    EXPECT_NEAR(r.energy, 0.5 * (A * ls - b).squaredNorm(), 1e-10);
    EXPECT_GE(be->accepted, 1);
}

TEST(PgoLevMarq, NoiseFreeGraphReturnsTheTruth) {
    std::mt19937 g(2);
    std::vector<core::SE3> truth;
    PoseGraph G = Graph(g, 30, 25, 0.0, truth);
    Perturb(G, g, 0.05, 0.2);
    PgoSettings s;
    s.rel_energy_delta_tolerance = 1e-14;
    s.step_norm_tolerance = 1e-14;
    s.min_gradient_tolerance = 1e-12;
    const PgoReport rep = SolvePoseGraph(G, 0, s);
    EXPECT_LT(rep.final_energy, 1e-16);
    EXPECT_GT(rep.initial_energy, 1.0);
    for (const auto& [id, n] : G.nodes()) {
        EXPECT_LT((n.T_W_C.t - truth[size_t(id)].t).norm(), 1e-9) << id;
        EXPECT_LT(core::SO3Log(n.T_W_C.R * truth[size_t(id)].R.transpose()).norm(), 1e-9) << id;
    }
    std::printf("noise-free: E %.3e -> %.3e in %d iterations (%d linearizations, %d factorizations, %d failed)\n",
                rep.initial_energy, rep.final_energy, rep.iterations, rep.linearizations, rep.factorizations,
                rep.failed_factorizations);
}

TEST(PgoLevMarq, NoisyGraphReachesAStationaryPoint) {
    std::mt19937 g(3);
    std::vector<core::SE3> truth;
    PoseGraph G = Graph(g, 50, 40, 0.01, truth);
    Perturb(G, g, 0.02, 0.1);
    PgoSettings s;
    s.rel_energy_delta_tolerance = 1e-14;
    s.step_norm_tolerance = 1e-14;
    s.min_gradient_tolerance = 1e-10;
    const PgoReport rep = SolvePoseGraph(G, 0, s);
    EXPECT_LT(rep.final_energy, rep.initial_energy);
    const PgoProblem p(G, 0);
    SparseMat H;
    Eigen::VectorXd grad;
    p.Linearize(p.poses(), H, grad);
    EXPECT_LT(grad.cwiseAbs().maxCoeff(), 1e-6);
    std::printf("noisy: E %.4f -> %.4f in %d iterations; max|grad| %.2e\n", rep.initial_energy, rep.final_energy,
                rep.iterations, grad.cwiseAbs().maxCoeff());
}

// Same binary, same input: the full OpenCV-driven solve is bit-identical (ordered containers, single-threaded algebra).
TEST(PgoLevMarq, SolveIsBitIdenticalTwice) {
    std::vector<core::SE3> truth;
    std::mt19937 g1(9), g2(9);
    PoseGraph A = Graph(g1, 40, 30, 0.01, truth), B = Graph(g2, 40, 30, 0.01, truth);
    Perturb(A, g1, 0.02, 0.1);
    Perturb(B, g2, 0.02, 0.1);
    const PgoReport ra = SolvePoseGraph(A, 0), rb = SolvePoseGraph(B, 0);
    EXPECT_EQ(ra.iterations, rb.iterations);
    EXPECT_EQ(std::memcmp(&ra.final_energy, &rb.final_energy, sizeof(double)), 0);
    for (const auto& [id, n] : A.nodes()) {
        const core::SE3& b = B.nodes().at(id).T_W_C;
        EXPECT_EQ(std::memcmp(n.T_W_C.t.data(), b.t.data(), 3 * sizeof(double)), 0) << id;
        EXPECT_EQ(std::memcmp(n.T_W_C.R.data(), b.R.data(), 9 * sizeof(double)), 0) << id;
        EXPECT_EQ(std::memcmp(n.covariance.data(), B.nodes().at(id).covariance.data(), 36 * sizeof(double)), 0) << id;
    }
}
