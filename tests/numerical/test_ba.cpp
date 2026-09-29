// Reprojection LM (arch §11.3-11.6): convergence, gauge, robustness, and the
// covariance-consistency (NEES) test that catches convention/transport errors.
#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include "core/uncertainty/covariance.hpp"
#include "slam/bundle_adjustment/reprojection_lm.hpp"

#include <gtest/gtest.h>

#include <cstdio>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

const core::PinholeCamera kCam{436.2, 436.2, 364.4, 257.0};
constexpr double kBaseline = 0.11;

SE3 Pose(const Vec3& rot, const Vec3& t) {
    SE3 T;
    T.R = core::SO3Exp(rot);
    T.t = t;
    return T;
}

/// Points spread in front of a camera at the world origin looking down +z.
std::vector<Vec3> Scene(int n, uint64_t seed) {
    core::PhiloxStream s(seed, 0, 1);
    auto u = [&s] { return (s.NextBits32() >> 8) * (1.0 / 16777216.0); };
    std::vector<Vec3> pts;
    for (int i = 0; i < n; ++i) {
        const double z = 2.0 + 6.0 * u();
        pts.emplace_back((u() - 0.5) * 1.2 * z, (u() - 0.5) * 0.8 * z, z);
    }
    return pts;
}

/// Left-invariant error of an estimate: est = exp(xi) * truth.
core::Vec6 LeftError(const SE3& est, const SE3& truth) { return core::SE3Log(est * truth.inverse()); }

/// Motion-only problem: one free pose, fixed landmarks, left-image observations.
BaProblem MotionOnly(const SE3& truth, const std::vector<Vec3>& pts, core::PhiloxStream* noise, double sigma) {
    BaProblem p;
    p.cam = kCam;
    p.poses = {truth};
    p.pose_fixed = {0};
    p.landmarks = pts;
    p.landmark_fixed.assign(pts.size(), 1);
    for (int j = 0; j < static_cast<int>(pts.size()); ++j) {
        core::Vec2 uv = core::Project(kCam, truth * pts[static_cast<size_t>(j)]);
        if (noise) uv += core::Vec2(core::SampleNormal(*noise, 0, sigma), core::SampleNormal(*noise, 0, sigma));
        p.obs.push_back({0, j, uv, Vec3::Zero(), sigma});
    }
    return p;
}

}  // namespace

TEST(ReprojectionLm, MotionOnlyConvergesToTruthOnCleanData) {
    const SE3 truth = Pose(Vec3(0.05, -0.1, 0.02), Vec3(0.2, -0.05, 0.3));
    BaProblem p = MotionOnly(truth, Scene(120, 1), nullptr, 1.0);
    p.poses[0] = core::SE3Exp((core::Vec6() << 0.05, -0.03, 0.08, 0.02, -0.03, 0.04).finished()) * truth;
    const BaSummary s = SolveReprojection(p);
    EXPECT_LT(LeftError(p.poses[0], truth).norm(), 1e-8);
    EXPECT_LT(s.final_cost, 1e-12);
    EXPECT_EQ(s.inliers, 120);
}

// THE convention test (advisor, 1.5 design). With pure Gaussian pixel noise and
// Huber switched off, the estimator is (to first order) the ML estimate, and
// the reported covariance must describe its actual scatter: the mean of
// xi^T Sigma^-1 xi over trials is chi^2 with 6 dof, i.e. ~6. A transposed
// Jacobian, a [rho; phi] swap, a right-vs-left perturbation mix-up or a wrong
// covariance transport all move this far from 6. It is checked for T_C_W and,
// through InvertCovariance, for the T_W_C form the keyframe stores.
TEST(ReprojectionLm, CovarianceIsConsistentMonteCarloNees) {
    const SE3 truth = Pose(Vec3(0.1, -0.2, 0.05), Vec3(-0.3, 0.1, 0.4));
    const std::vector<Vec3> pts = Scene(60, 2);
    BaOptions o;
    o.huber_delta = 1e9;   // pure least squares
    double nees_cw = 0.0, nees_wc = 0.0;
    const int trials = 2000;
    for (int k = 0; k < trials; ++k) {
        // One stream per trial, keyed (seed, trial, site): one stream for all
        // trials would exhaust its 2^18 words (the D18 guard throws).
        core::PhiloxStream noise(77, static_cast<uint64_t>(k), 0);
        BaProblem p = MotionOnly(truth, pts, &noise, 1.0);
        const BaSummary s = SolveReprojection(p, o);
        ASSERT_EQ(s.pose_covariance.size(), 1u);
        const core::Mat6& cov_cw = s.pose_covariance[0];
        const core::Vec6 e_cw = LeftError(p.poses[0], truth);
        nees_cw += e_cw.dot(cov_cw.ldlt().solve(e_cw));

        const core::Mat6 cov_wc = core::InvertCovariance(p.poses[0], cov_cw);
        const core::Vec6 e_wc = LeftError(p.poses[0].inverse(), truth.inverse());
        nees_wc += e_wc.dot(cov_wc.ldlt().solve(e_wc));
    }
    nees_cw /= trials;
    nees_wc /= trials;
    std::printf("mean NEES over %d trials: T_C_W %.3f, T_W_C %.3f (expected 6)\n", trials, nees_cw, nees_wc);
    // Mean of 2000 chi^2_6 draws: sd = sqrt(12/2000) ~ 0.077; accept +-4.5 sd.
    // (400 trials with +-0.7 would pass a covariance ~12% too small.)
    EXPECT_NEAR(nees_cw, 6.0, 0.35) << "T_C_W covariance inconsistent";
    EXPECT_NEAR(nees_wc, 6.0, 0.35) << "T_W_C covariance (via InvertCovariance) inconsistent";
}

// Two-frame BA stub: keyframe 0 fixed (gauge), keyframe 1 and the landmarks
// free, stereo observations (left + right) in both frames.
TEST(ReprojectionLm, TwoFrameBaRecoversPoseAndLandmarksWithGaugeFixed) {
    const std::vector<Vec3> truth_pts = Scene(80, 3);
    const SE3 kf0 = SE3::Identity();
    const SE3 kf1 = Pose(Vec3(0.02, 0.08, -0.01), Vec3(-0.25, 0.02, -0.1));   // T_C1_W
    BaProblem p;
    p.cam = kCam;
    p.poses = {kf0, core::SE3Exp((core::Vec6() << 0.03, -0.02, 0.04, 0.01, -0.02, 0.015).finished()) * kf1};
    p.pose_fixed = {1, 0};
    core::PhiloxStream s(9, 0, 0);
    for (int j = 0; j < static_cast<int>(truth_pts.size()); ++j) {
        const Vec3& X = truth_pts[static_cast<size_t>(j)];
        p.landmarks.push_back(X + Vec3(core::SampleNormal(s, 0, 0.05), core::SampleNormal(s, 0, 0.05),
                                       core::SampleNormal(s, 0, 0.05)));
        for (int f = 0; f < 2; ++f) {
            const SE3& T = f == 0 ? kf0 : kf1;
            const Vec3 off(kBaseline, 0, 0);
            p.obs.push_back({f, j, core::Project(kCam, T * X), Vec3::Zero(), 1.0});
            p.obs.push_back({f, j, core::Project(kCam, T * X - off), off, 1.0});
        }
    }
    p.landmark_fixed.assign(p.landmarks.size(), 0);
    const BaSummary sum = SolveReprojection(p);
    EXPECT_TRUE(sum.converged);
    EXPECT_LT(LeftError(p.poses[1], kf1).norm(), 1e-6);
    EXPECT_EQ((p.poses[0].matrix() - kf0.matrix()).norm(), 0.0) << "the gauge pose moved";
    for (size_t j = 0; j < truth_pts.size(); ++j) EXPECT_LT((p.landmarks[j] - truth_pts[j]).norm(), 1e-6);
}

// Huber (§11.3): 20% gross outliers. The robust solution stays near truth;
// plain least squares is dragged away. (Robustness to WRONG ASSOCIATION is the
// pose graph's job, §14; this is tracking-noise robustness.)
TEST(ReprojectionLm, HuberLimitsTheInfluenceOfOutliers) {
    const SE3 truth = Pose(Vec3(0.03, 0.05, -0.02), Vec3(0.1, 0.2, -0.1));
    const std::vector<Vec3> pts = Scene(150, 4);
    auto solve = [&](double delta) {
        core::PhiloxStream noise(5, 0, 0);
        BaProblem p = MotionOnly(truth, pts, &noise, 1.0);
        for (size_t k = 0; k < p.obs.size(); k += 5) p.obs[k].uv += core::Vec2(30.0, -25.0);
        BaOptions o;
        o.huber_delta = delta;
        o.max_iterations = 50;
        SolveReprojection(p, o);
        return LeftError(p.poses[0], truth).norm();
    };
    const double robust = solve(2.0), plain = solve(1e9);
    EXPECT_LT(robust, 0.5 * plain);
}

TEST(ReprojectionLm, FreeLandmarksWithoutAFixedPoseIsRejected) {
    BaProblem p;
    p.cam = kCam;
    p.poses = {SE3::Identity()};
    p.pose_fixed = {0};
    p.landmarks = {Vec3(0, 0, 3)};
    p.landmark_fixed = {0};
    EXPECT_THROW(SolveReprojection(p), std::invalid_argument);
}
