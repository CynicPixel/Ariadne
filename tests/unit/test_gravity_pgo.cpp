// GRAVITY_PRIOR consumption in the pose-graph solver (D73 part 2c; design ROBUST_BACKEND_2.5.md §8, D59 handoff):
//   - the analytic gradient (pose + gravity blocks, Huber regime included) matches finite differences of the energy;
//   - disabled (default) the solve is bit-identical to a graph without priors;
//   - a 1.5 deg error in the reference gravity is recovered by the 2-DoF per-agent variable, and tilt drift shrinks;
//   - one 30 deg outlier prior is down-weighted by the Huber kernel.
#include "slam/pose_graph/pgo_levmarq.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec3;
using core::Vec6;

namespace {

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

const Vec3 kDown(0, 0, -1);   // true gravity direction in the world

core::Mat3 Rot(const Vec3& axis_angle) { return core::SO3Exp(axis_angle); }

struct Chain {
    std::vector<SE3> truth;
    PoseGraph g;
};

/// A walk with rotation noise in the odometry (tilt drifts), priors every `every` keyframes measured from the truth
/// with sigma_deg noise, against a reference gravity rotated by ref_err_deg about x.
Chain Make(std::mt19937& gen, int n, int every, double sigma_deg, double ref_err_deg) {
    Chain c;
    std::normal_distribution<double> nz(0.0, 1.0);
    SE3 T;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            Vec6 step;
            step << 0.4, 0.05 * nz(gen), 0.0, 0.02 * nz(gen), 0.02 * nz(gen), 0.15 * nz(gen);
            T = T * core::SE3Exp(step);
        }
        c.truth.push_back(T);
    }
    const Mat6 odo = Diag(0.01, 0.01);
    SE3 est = c.truth[0];
    for (int i = 0; i < n; ++i) {
        Vec6 noise;
        for (int k = 0; k < 6; ++k) noise(k) = nz(gen);
        const SE3 Z = core::SE3Exp(Eigen::LLT<Mat6>(odo).matrixL() * noise) * (i ? c.truth[size_t(i - 1)].inverse() * c.truth[size_t(i)] : SE3());
        if (i > 0) est = est * Z;
        PoseNode nd;
        nd.keyframe = uint64_t(i);
        nd.T_W_C = est;
        c.g.AddNode(nd);
        if (i > 0) c.g.AddEdge(OdometryEdge(uint64_t(i - 1), uint64_t(i), Z, odo));
    }
    const double s = sigma_deg * M_PI / 180.0;
    const Vec3 g_ref = Rot(Vec3(ref_err_deg * M_PI / 180.0, 0, 0)) * kDown;
    for (int i = 0; i < n; i += every) {
        GravityPrior p;
        p.keyframe = uint64_t(i);
        const Vec3 gC = c.truth[size_t(i)].R.transpose() * kDown;
        p.g_C = (Rot(s * Vec3(nz(gen), nz(gen), 0)) * gC).normalized();
        p.g_ref_W = g_ref.normalized();
        p.information = Eigen::Matrix2d::Identity() / (s * s);
        c.g.AddGravityPrior(p);
    }
    return c;
}

double TiltErrorDeg(const Chain& c) {   // RMS angle between estimated and true down, in each camera
    double e = 0;
    for (const auto& [id, n] : c.g.nodes()) {
        const Vec3 a = n.T_W_C.R.transpose() * kDown, b = c.truth[size_t(id)].R.transpose() * kDown;
        e += std::pow(std::acos(std::clamp(a.dot(b), -1.0, 1.0)), 2);
    }
    return std::sqrt(e / double(c.g.nodes().size())) * 180.0 / M_PI;
}

PgoSettings On() {
    PgoSettings s = PgoV2();
    s.recompute_covariance = false;
    s.gravity.enabled = true;
    return s;
}

}  // namespace

TEST(GravityPgo, GradientMatchesFiniteDifferencesIncludingHuber) {
    std::mt19937 gen(1);
    Chain c = Make(gen, 12, 2, 2.3, 1.5);
    GravityPrior big = c.g.gravity_priors()[2];   // one in the Huber (linear) regime
    big.g_C = (Rot(Vec3(0.5, 0, 0)) * big.g_C).normalized();
    c.g.AddGravityPrior(big);
    GravityOptions go;
    go.enabled = true;
    const PgoProblem p(c.g, 0, go);
    SparseMat H;
    Eigen::VectorXd grad;
    p.Linearize(p.poses(), p.gravity(), H, grad);
    ASSERT_EQ(grad.size(), p.dim());
    ASSERT_EQ(p.dim(), p.pose_dim() + 2);
    const double h = 1e-6;
    double worst = 0;
    for (int i = 0; i < p.dim(); ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(p.dim());
        x(i) = h;
        const double ep = p.Energy(p.OplusX(p.poses(), x), p.OplusGravity(p.gravity(), x));
        x(i) = -h;
        const double em = p.Energy(p.OplusX(p.poses(), x), p.OplusGravity(p.gravity(), x));
        worst = std::max(worst, std::fabs((ep - em) / (2 * h) - grad(i)) / std::max(1.0, std::fabs(grad(i))));
    }
    EXPECT_LT(worst, 1e-5);
}

TEST(GravityPgo, DisabledIsBitIdenticalToNoPriors) {
    std::mt19937 g1(2), g2(2);
    Chain with = Make(g1, 30, 3, 2.3, 1.0), without = Make(g2, 30, 3, 2.3, 1.0);
    PoseGraph bare;
    for (const auto& [id, n] : without.g.nodes()) bare.AddNode(n);
    for (const auto& e : without.g.edges()) bare.AddEdge(e);
    PgoSettings off = PgoV2();
    off.recompute_covariance = false;
    SolvePoseGraph(with.g, 0, off);
    SolvePoseGraph(bare, 0, off);
    for (const auto& [id, n] : bare.nodes()) {
        const SE3& T = with.g.nodes().at(id).T_W_C;
        EXPECT_EQ(std::memcmp(n.T_W_C.R.data(), T.R.data(), sizeof(double) * 9), 0);
        EXPECT_EQ(std::memcmp(n.T_W_C.t.data(), T.t.data(), sizeof(double) * 3), 0);
    }
}

TEST(GravityPgo, RecoversTheReferenceGravityErrorAndReducesTilt) {
    std::mt19937 gen(3);
    Chain c = Make(gen, 200, 10, 2.3, 1.5);
    Chain plain = c;
    const double tilt0 = TiltErrorDeg(c);
    const PgoReport r = SolvePoseGraph(c.g, 0, On());
    ASSERT_EQ(r.gravity_W.size(), 1u);
    const double gerr = std::acos(std::clamp(r.gravity_W.begin()->second.dot(kDown), -1.0, 1.0)) * 180.0 / M_PI;
    std::printf("gravity: reference off by 1.50 deg -> estimate off by %.2f deg | tilt RMS %.2f deg (odometry) -> %.2f deg\n",
                gerr, tilt0, TiltErrorDeg(c));
    EXPECT_LT(gerr, 1.0);
    EXPECT_LT(TiltErrorDeg(c), tilt0);
}

TEST(GravityPgo, HuberDownWeightsAnOutlierPrior) {
    std::mt19937 g1(4), g2(4);
    Chain clean = Make(g1, 60, 10, 2.3, 0.0), bad = Make(g2, 60, 10, 2.3, 0.0);
    GravityPrior o = bad.g.gravity_priors()[3];
    o.g_C = (Rot(Vec3(30.0 * M_PI / 180.0, 0, 0)) * o.g_C).normalized();
    bad.g.AddGravityPrior(o);
    SolvePoseGraph(clean.g, 0, On());
    SolvePoseGraph(bad.g, 0, On());
    const uint64_t k = o.keyframe;
    const Vec3 a = clean.g.nodes().at(k).T_W_C.R.transpose() * kDown, b = bad.g.nodes().at(k).T_W_C.R.transpose() * kDown;
    const double shift = std::acos(std::clamp(a.dot(b), -1.0, 1.0)) * 180.0 / M_PI;
    std::printf("a 30 deg outlier prior moves its keyframe's tilt by %.2f deg (quadratic cost would follow it further)\n", shift);
    EXPECT_LT(shift, 3.0);
}
