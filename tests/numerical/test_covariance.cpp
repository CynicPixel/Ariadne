// Arch §37.1 "covariance transform". The composition and inverse rules are
// checked against Monte Carlo rather than against a re-derivation of the same
// algebra -- an independent check, not a restatement.
#include "core/math/lie.hpp"
#include "core/uncertainty/covariance.hpp"

#include <gtest/gtest.h>
#include <random>

using namespace ariadne::core;

namespace {

std::mt19937 Rng() { return std::mt19937(20260830u); }

// Sampling uses std::normal_distribution, which is implementation-defined
// across standard libraries. That is acceptable HERE because the assertion is
// statistical, not bit-exact. It is explicitly NOT acceptable inside the
// degradation operator D, which must be bit-identical across machines.
Vec6 SampleGaussian(std::mt19937& g, const Eigen::LLT<Mat6>& llt) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 z;
    for (int i = 0; i < 6; ++i) z(i) = n(g);
    return llt.matrixL() * z;
}

Mat6 MakeSpd(double scale) {
    Mat6 m = Mat6::Zero();
    for (int i = 0; i < 6; ++i) m(i, i) = scale * (1.0 + 0.3 * i);
    // Off-diagonal correlation, so a transposed adjoint would actually show up.
    m(0, 4) = m(4, 0) = 0.4 * scale;
    m(1, 3) = m(3, 1) = -0.25 * scale;
    return m;
}

SE3 MakePose(const Vec3& rho, const Vec3& phi) {
    Vec6 xi;
    xi.head<3>() = rho;
    xi.tail<3>() = phi;
    return SE3Exp(xi);
}

// Compare two covariances with a relative Frobenius tolerance.
void ExpectCovNear(const Mat6& got, const Mat6& want, double rel_tol) {
    const double err = (got - want).norm() / want.norm();
    EXPECT_LT(err, rel_tol) << "relative error " << err << "\ngot:\n"
                            << got << "\nwant:\n" << want;
}

constexpr int kSamples = 200000;

}  // namespace

TEST(Covariance, ValidityChecks) {
    EXPECT_TRUE(IsValidCovariance(MakeSpd(1e-2)));
    EXPECT_TRUE(IsValidCovariance(Mat6::Identity()));
    Mat6 indefinite = Mat6::Identity();
    indefinite(2, 2) = -1.0;
    EXPECT_FALSE(IsValidCovariance(indefinite));
    Mat6 asymmetric = Mat6::Identity();
    asymmetric(0, 1) = 0.5;
    EXPECT_FALSE(IsValidCovariance(asymmetric));
}

TEST(Covariance, NearestPsdRepairsSmallNegativeEigenvalues) {
    Mat6 m = MakeSpd(1e-2);
    m(5, 5) = -1e-14;                       // the usual floating-point residue
    EXPECT_FALSE(IsValidCovariance(m, 1e-16));
    const Mat6 repaired = NearestPSD(m);
    EXPECT_TRUE(IsValidCovariance(repaired));
    // Repair must be a small correction, not a rewrite.
    EXPECT_LT((repaired - m).norm() / m.norm(), 1e-6);
}

TEST(Covariance, TransportIsCongruenceNotSimilarity) {
    // Guards the classic bug: J * Sigma * J^-1 instead of J * Sigma * J^T.
    const Mat6 cov = MakeSpd(1e-2);
    const SE3  T   = MakePose(Vec3(1.0, -2.0, 0.5), Vec3(0.3, -0.2, 0.7));
    const Mat6 out = ChangeFrame(T, cov);
    EXPECT_TRUE(IsValidCovariance(out));
    EXPECT_LT((out - out.transpose()).norm(), 1e-12);
}

TEST(Covariance, ComposeMatchesMonteCarlo) {
    auto g = Rng();
    const SE3 A = MakePose(Vec3(0.8, -1.4, 0.3), Vec3(0.25, 0.15, -0.4));
    const SE3 B = MakePose(Vec3(-0.5, 0.9, 1.1), Vec3(-0.1, 0.35, 0.2));

    const Mat6 cov_a = MakeSpd(1e-4);
    const Mat6 cov_b = MakeSpd(4e-5);
    const Eigen::LLT<Mat6> llt_a(cov_a), llt_b(cov_b);

    const SE3 C_mean     = A * B;
    const SE3 C_mean_inv = C_mean.inverse();

    Mat6 acc = Mat6::Zero();
    for (int i = 0; i < kSamples; ++i) {
        const SE3 As = SE3Exp(SampleGaussian(g, llt_a)) * A;
        const SE3 Bs = SE3Exp(SampleGaussian(g, llt_b)) * B;
        const Vec6 xi = SE3Log(As * Bs * C_mean_inv);   // left-perturbation residual
        acc += xi * xi.transpose();
    }
    acc /= static_cast<double>(kSamples);

    ExpectCovNear(acc, ComposeCovariance(A, cov_a, cov_b), 0.05);
}

TEST(Covariance, InverseMatchesMonteCarlo) {
    auto g = Rng();
    const SE3  T   = MakePose(Vec3(1.2, 0.4, -0.7), Vec3(0.2, -0.35, 0.5));
    const Mat6 cov = MakeSpd(1e-4);
    const Eigen::LLT<Mat6> llt(cov);

    const SE3 T_inv_mean = T.inverse();

    Mat6 acc = Mat6::Zero();
    for (int i = 0; i < kSamples; ++i) {
        const SE3 Ts = SE3Exp(SampleGaussian(g, llt)) * T;
        const Vec6 xi = SE3Log(Ts.inverse() * T_inv_mean.inverse());
        acc += xi * xi.transpose();
    }
    acc /= static_cast<double>(kSamples);

    ExpectCovNear(acc, InvertCovariance(T, cov), 0.05);
}

// A wrong-but-plausible rule to prove the Monte Carlo test above has teeth:
// dropping the adjoint (Sigma_a + Sigma_b) must NOT match.
TEST(Covariance, MonteCarloRejectsTheNaiveRule) {
    const SE3  A = MakePose(Vec3(0.8, -1.4, 0.3), Vec3(0.25, 0.15, -0.4));
    const Mat6 cov_a = MakeSpd(1e-4);
    const Mat6 cov_b = MakeSpd(4e-5);
    const Mat6 correct = ComposeCovariance(A, cov_a, cov_b);
    const Mat6 naive   = cov_a + cov_b;
    EXPECT_GT((correct - naive).norm() / correct.norm(), 0.1);
}
