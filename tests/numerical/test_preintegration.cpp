// IMU preintegration and the inertial factor (D59, slam/inertial/preintegration.hpp): bias Jacobians and every
// residual Jacobian block against finite differences (left pose perturbation, large rotations), zero residual at the
// truth of a closed-form trajectory, and the propagated covariance against a Monte Carlo of the noise model.
#include "slam/inertial/preintegration.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;

namespace {

const Vec3 kG(0, 0, -9.81);

/// A smooth, strongly rotating trajectory with closed-form position and rotation.
SE3 Pose(double t) {
    SE3 T;
    T.R = core::SO3Exp(Vec3(0.8 * std::sin(0.9 * t), 0.6 * t, 0.5 * std::cos(1.3 * t)));
    T.t = Vec3(std::sin(t), std::cos(0.7 * t), 0.3 * t + 0.2 * std::sin(2.0 * t));
    return T;
}
Vec3 Velocity(double t) { return Vec3(std::cos(t), -0.7 * std::sin(0.7 * t), 0.3 + 0.4 * std::cos(2.0 * t)); }
Vec3 Accel(double t) { return Vec3(-std::sin(t), -0.49 * std::cos(0.7 * t), -0.8 * std::sin(2.0 * t)); }
/// Body angular velocity by a symmetric difference of the closed-form rotation (h tiny: exact to ~1e-9).
Vec3 Omega(double t) {
    const double h = 1e-5;
    return core::SO3Log(Pose(t - h).R.transpose() * Pose(t + h).R) / (2 * h);
}

/// Ideal IMU samples at `rate` over [t0, t1], evaluated at interval midpoints (second-order accurate hold).
std::vector<core::ImuMeasurement> IdealImu(double t0, double t1, double rate, const Vec3& bg, const Vec3& ba) {
    std::vector<core::ImuMeasurement> s;
    const int n = int(std::round((t1 - t0) * rate));
    for (int k = 0; k <= n; ++k) {
        const double t = t0 + k / rate, tm = t + 0.5 / rate;
        s.push_back({int64_t(std::llround(t * 1e9)), Omega(tm) + bg,
                     Pose(tm).R.transpose() * (Accel(tm) - kG) + ba});
    }
    return s;
}

Preintegrated RandomPreintegration(std::mt19937& rng, const Vec3& bg_lin, const Vec3& ba_lin) {
    std::normal_distribution<double> N(0, 1);
    Preintegrated p(bg_lin, ba_lin, {1e-3, 1e-2});
    for (int k = 0; k < 200; ++k) {   // gyro, then accel: two draws in one call are unsequenced (D123)
        const Vec3 w{1.5 * N(rng), 1.5 * N(rng), 1.5 * N(rng)};
        const Vec3 a{3 * N(rng), 3 * N(rng), 9.81 + 3 * N(rng)};
        p.Integrate(w, a, 0.005);
    }
    return p;
}

}  // namespace

// ∂ΔR/∂bg, ∂Δv/∂bg, ∂Δv/∂ba, ∂Δp/∂bg, ∂Δp/∂ba: re-integrating at a perturbed bias matches the first-order update.
TEST(Preintegration, BiasJacobiansMatchReintegration) {
    std::mt19937 rng(7);
    std::normal_distribution<double> N(0, 1);
    std::vector<std::pair<Vec3, Vec3>> samples;
    for (int k = 0; k < 200; ++k)
        samples.push_back({Vec3{1.5 * N(rng), 1.5 * N(rng), 1.5 * N(rng)}, Vec3{3 * N(rng), 3 * N(rng), 9.81 + 3 * N(rng)}});
    const Vec3 bg(0.01, -0.02, 0.015), ba(0.05, 0.1, -0.08);
    auto integrate = [&](const Vec3& g, const Vec3& a) {
        Preintegrated p(g, a, {1e-3, 1e-2});
        for (const auto& [w, f] : samples) p.Integrate(w, f, 0.005);
        return p;
    };
    const Preintegrated p0 = integrate(bg, ba);
    const double eps = 1e-6;
    for (int ax = 0; ax < 3; ++ax) {
        const Vec3 e = Vec3::Unit(ax) * eps;
        const Preintegrated pg = integrate(bg + e, ba), pa = integrate(bg, ba + e);
        const Vec3 dRg = core::SO3Log(p0.DeltaR(bg).transpose() * pg.DeltaR(bg + e)) / eps;
        EXPECT_LT((dRg - p0.JRg().col(ax)).norm(), 1e-5 * (1 + p0.JRg().col(ax).norm())) << ax;
        EXPECT_LT(((pg.DeltaV(bg + e, ba) - p0.DeltaV(bg, ba)) / eps - p0.Jvg().col(ax)).norm(), 1e-4) << ax;
        EXPECT_LT(((pa.DeltaV(bg, ba + e) - p0.DeltaV(bg, ba)) / eps - p0.Jva().col(ax)).norm(), 1e-4) << ax;
        EXPECT_LT(((pg.DeltaP(bg + e, ba) - p0.DeltaP(bg, ba)) / eps - p0.Jpg().col(ax)).norm(), 1e-4) << ax;
        EXPECT_LT(((pa.DeltaP(bg, ba + e) - p0.DeltaP(bg, ba)) / eps - p0.Jpa().col(ax)).norm(), 1e-4) << ax;
    }
}

// Every block of the inertial factor's Jacobian against central differences, at random states with large rotations
// and with the bias away from its linearisation point.
TEST(Preintegration, ResidualJacobiansMatchFiniteDifferences) {
    std::mt19937 rng(11);
    std::normal_distribution<double> N(0, 1);
    auto rv = [&](double s) { return Vec3{s * N(rng), s * N(rng), s * N(rng)}; };
    for (int trial = 0; trial < 20; ++trial) {
        const Preintegrated pre = RandomPreintegration(rng, rv(0.02), rv(0.1));
        SE3 Ti, Tj;
        Ti.R = core::SO3Exp(rv(1.5));
        Ti.t = rv(3);
        Tj.R = core::SO3Exp(rv(1.5));
        Tj.t = rv(3);
        const Vec3 vi = rv(1), vj = rv(1), bg = pre.bg_lin() + rv(0.01), ba = pre.ba_lin() + rv(0.05);
        const InertialLinearization L = LinearizeInertial(pre, Ti, vi, bg, ba, Tj, vj, kG);
        const double h = 1e-6;
        auto res = [&](const SE3& a, const Vec3& va, const Vec3& g, const Vec3& acc, const SE3& b, const Vec3& vb) {
            return LinearizeInertial(pre, a, va, g, acc, b, vb, kG).r;
        };
        auto check = [&](const Eigen::MatrixXd& J, auto perturbed, int dim, const char* what) {
            for (int c = 0; c < dim; ++c) {
                const Vec9 num = (perturbed(c, h) - perturbed(c, -h)) / (2 * h);
                EXPECT_LT((num - J.col(c)).norm(), 1e-5 * (1 + J.col(c).norm())) << what << " col " << c << " trial " << trial;
            }
        };
        auto leftPert = [](const SE3& T, int c, double s) {
            core::Vec6 d = core::Vec6::Zero();
            d[c] = s;
            return core::SE3Exp(d) * T;
        };
        check(L.J_Ti, [&](int c, double s) { return res(leftPert(Ti, c, s), vi, bg, ba, Tj, vj); }, 6, "Ti");
        check(L.J_Tj, [&](int c, double s) { return res(Ti, vi, bg, ba, leftPert(Tj, c, s), vj); }, 6, "Tj");
        check(L.J_vi, [&](int c, double s) { return res(Ti, vi + Vec3::Unit(c) * s, bg, ba, Tj, vj); }, 3, "vi");
        check(L.J_vj, [&](int c, double s) { return res(Ti, vi, bg, ba, Tj, vj + Vec3::Unit(c) * s); }, 3, "vj");
        check(L.J_bgi, [&](int c, double s) { return res(Ti, vi, bg + Vec3::Unit(c) * s, ba, Tj, vj); }, 3, "bg");
        check(L.J_bai, [&](int c, double s) { return res(Ti, vi, bg, ba + Vec3::Unit(c) * s, Tj, vj); }, 3, "ba");
    }
}

// Ideal IMU from a closed-form trajectory (biased, known biases): the residual at the true states is ~0.
TEST(Preintegration, ResidualVanishesAtTheTruth) {
    const Vec3 bg(0.02, -0.01, 0.005), ba(0.1, -0.05, 0.08);
    const double t0 = 1.0, t1 = 1.5;
    const auto imu = IdealImu(t0, t1, 1000.0, bg, ba);
    const Preintegrated pre = Preintegrate(imu, imu.front().t_ns, imu.back().t_ns, bg, ba, {1e-3, 1e-2});
    const InertialLinearization L =
        LinearizeInertial(pre, Pose(t0), Velocity(t0), bg, ba, Pose(t1), Velocity(t1), kG);
    EXPECT_LT(L.r.head<3>().norm(), 1e-5);
    EXPECT_LT(L.r.segment<3>(3).norm(), 1e-4);
    EXPECT_LT(L.r.tail<3>().norm(), 1e-5);
}

// The propagated Σ (eqs. 62-63) against the sample covariance of the preintegration error under the discrete noise
// model (per-sample std = density / sqrt(dt)): every diagonal within 10%, and the whitened error has unit covariance.
TEST(Preintegration, CovarianceMatchesMonteCarlo) {
    std::mt19937 rng(23);
    std::normal_distribution<double> N(0, 1);
    const double dt = 0.005;
    const PreintegrationNoise noise{2e-3, 3e-2};
    std::vector<std::pair<Vec3, Vec3>> truth;
    for (int k = 0; k < 100; ++k) truth.push_back({Vec3{0.5 * N(rng), 0.5 * N(rng), 0.5 * N(rng)}, Vec3{N(rng), N(rng), 9.81 + N(rng)}});
    Preintegrated clean(Vec3::Zero(), Vec3::Zero(), noise);
    for (const auto& [w, a] : truth) clean.Integrate(w, a, dt);
    const int runs = 4000;
    const double sg = noise.gyro_density / std::sqrt(dt), sa = noise.accel_density / std::sqrt(dt);
    Mat9 C = Mat9::Zero();
    for (int r = 0; r < runs; ++r) {
        Preintegrated p(Vec3::Zero(), Vec3::Zero(), noise);
        for (const auto& [w, a] : truth)
            {   // gyro noise first, then accel: two draws in one call are unsequenced (D123)
                const Vec3 nw{sg * N(rng), sg * N(rng), sg * N(rng)};
                const Vec3 na{sa * N(rng), sa * N(rng), sa * N(rng)};
                p.Integrate(w + nw, a + na, dt);
            }
        // ΔR̃ = ΔR Exp(-δφ), Δṽ = Δv - δv, Δp̃ = Δp - δp (eqs. 35-37): signs do not matter for the covariance.
        Vec9 e;
        e << core::SO3Log(clean.DeltaR(Vec3::Zero()).transpose() * p.DeltaR(Vec3::Zero())),
            p.DeltaV(Vec3::Zero(), Vec3::Zero()) - clean.DeltaV(Vec3::Zero(), Vec3::Zero()),
            p.DeltaP(Vec3::Zero(), Vec3::Zero()) - clean.DeltaP(Vec3::Zero(), Vec3::Zero());
        C += e * e.transpose() / runs;
    }
    const Mat9& S = clean.Sigma();
    for (int i = 0; i < 9; ++i) EXPECT_NEAR(C(i, i) / S(i, i), 1.0, 0.1) << "diag " << i;
    const Eigen::LLT<Mat9> llt(S);
    const Mat9 L = llt.matrixL();
    const Mat9 W = L.inverse() * C * L.inverse().transpose();
    EXPECT_LT((W - Mat9::Identity()).norm(), 0.25);
}
