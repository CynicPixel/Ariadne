// Loop-closure degeneracy gate (design robust-backend.md §4, D71):
//   - finite differences for the one new analytic Jacobian (stereo back-projection);
//   - the edge information uses OdometryEdge's conversion exactly;
//   - Monte Carlo: NEES of the refined edge in the residual tangent ~ chi^2_6 when pixels AND disparities are noisy;
//   - synthetic scenes, each named for the criterion it physically violates, and a well-conditioned control.
// The scenes are built to be (non-)degenerate on physical grounds; the thresholds are NOT tuned on them.
#include "slam/pose_graph/pose_graph.hpp"
#include "slam/robust/degeneracy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec2;
using core::Vec3;

namespace {

core::RectifiedStereo Rig() {   // 1280x720 rectified stereo [ASSUMPTION: sim-like intrinsics, 12 cm baseline]
    core::RectifiedStereo r;
    r.cam = {700.0, 700.0, 640.0, 360.0};
    r.baseline = 0.12;
    return r;
}

SE3 TruthAB() {   // B is 0.8 m to the side and 0.3 m back of A, yawed 15 deg
    core::Vec6 xi;
    xi << 0.8, 0.05, -0.3, 0.02, 0.26, -0.01;
    return core::SE3Exp(xi);
}

/// Points placed in A's image (pixel sampler) at depths from `depth`, observed by B's stereo with noise, and by A
/// with noise. X_B is what B's stereo would report (noisy disparity), uv_A what A detects.
template <typename PixelFn, typename DepthFn>
LoopGeometry Scene(std::mt19937& g, int n, PixelFn pixel, DepthFn depth, double sigma_px, double sigma_d) {
    LoopGeometry lg;
    lg.rig = Rig();
    const SE3 T = TruthAB(), Tinv = T.inverse();
    std::normal_distribution<double> nz(0.0, 1.0);
    while (int(lg.X_B.size()) < n) {
        const Vec2 uvA = pixel(g);
        const Vec3 XA = core::Unproject(lg.rig.cam, uvA, depth(g));
        const Vec3 XB = Tinv * XA;
        if (XB.z() < 0.5) continue;
        const Vec2 uvB = core::Project(lg.rig.cam, XB);
        if (uvB.x() < 0 || uvB.x() > 1280 || uvB.y() < 0 || uvB.y() > 720) continue;
        const double d = lg.rig.cam.fx * lg.rig.baseline / XB.z() + sigma_d * nz(g);
        if (d <= 0.05) continue;
        const Vec2 uvBn = uvB + sigma_px * Vec2{nz(g), nz(g)};
        lg.X_B.push_back(core::Unproject(lg.rig.cam, uvBn, lg.rig.cam.fx * lg.rig.baseline / d));
        lg.uv_A.push_back(uvA + sigma_px * Vec2{nz(g), nz(g)});
    }
    return lg;
}

SE3 Jitter(std::mt19937& g, double t, double r) {
    std::normal_distribution<double> n(0.0, 1.0);
    core::Vec6 xi;
    xi << t * n(g), t * n(g), t * n(g), r * n(g), r * n(g), r * n(g);
    return core::SE3Exp(xi);
}

auto FullImage() {
    return [](std::mt19937& g) {
        std::uniform_real_distribution<double> u(20, 1260), v(20, 700);
        return Vec2{u(g), v(g)};
    };
}
auto Depth(double lo, double hi) {
    return [lo, hi](std::mt19937& g) { return std::uniform_real_distribution<double>(lo, hi)(g); };
}

bool Has(const DegeneracyReport& r, const std::string& why) {
    return std::find(r.reasons.begin(), r.reasons.end(), why) != r.reasons.end();
}

void Print(const char* name, const DegeneracyReport& r) {
    std::string why;
    for (const auto& s : r.reasons) why += s + " ";
    std::printf("%-13s n %3d | sigma_t %.4f m sigma_r %.5f rad | ratio_t %.1e ratio_r %.1e | near %3d (median %.2f deg)"
                " | line %.1e pixel %.1e planarity %.2f corridor %.2f | s2hat %.2f rms %.2f px | %s\n",
                name, r.points, r.sigma_t_worst, r.sigma_r_worst, r.ratio_t, r.ratio_r, r.near_points,
                r.median_triangulation_deg, r.line_spread_ratio, r.pixel_spread_ratio, r.planarity, r.corridor,
                r.sigma2_hat, r.rms_px,
                r.degenerate ? why.c_str() : "ok");
}

}  // namespace

TEST(Degeneracy, StereoPointJacobianMatchesFiniteDifferences) {
    const core::RectifiedStereo rig = Rig();
    for (const Vec3& X : {Vec3(0.3, -0.2, 2.0), Vec3(-1.5, 0.8, 0.6), Vec3(4.0, 2.0, 30.0)}) {
        const Vec2 uv = core::Project(rig.cam, X);
        const double d = rig.cam.fx * rig.baseline / X.z();
        auto back = [&](double u, double v, double dd) {
            return core::Unproject(rig.cam, Vec2(u, v), rig.cam.fx * rig.baseline / dd);
        };
        core::Mat3 fd;
        const double h = 1e-6;
        fd.col(0) = (back(uv.x() + h, uv.y(), d) - back(uv.x() - h, uv.y(), d)) / (2 * h);
        fd.col(1) = (back(uv.x(), uv.y() + h, d) - back(uv.x(), uv.y() - h, d)) / (2 * h);
        fd.col(2) = (back(uv.x(), uv.y(), d + h) - back(uv.x(), uv.y(), d - h)) / (2 * h);
        const core::Mat3 J = StereoPointJacobian(rig, X);
        EXPECT_LT((J - fd).norm(), 1e-6 * std::max(1.0, fd.norm())) << X.transpose();
    }
}

TEST(Degeneracy, EdgeInformationUsesTheOdometryEdgeConversion) {
    std::mt19937 g(1);
    LoopGeometry lg = Scene(g, 120, FullImage(), Depth(1.5, 6.0), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport r = AssessLoop(lg);
    ASSERT_FALSE(r.degenerate);
    const Mat6 ref = OdometryEdge(0, 1, r.T_A_B, r.cov_left, EdgeType::kLocalLoop).information;
    EXPECT_LT((r.information - ref).norm(), 1e-9 * ref.norm());
}

// The reported covariance must be honest: over Monte Carlo trials with noisy pixels in A and noisy disparities in B,
// the edge error in the residual tangent, r = Log(Z_est^-1 Z_true), has mean NEES r^T Info r = 6 (chi^2_6).
template <typename DepthFn>
double MeanNees(std::mt19937& g, int trials, DepthFn depth) {
    const SE3 truth = TruthAB();
    double nees = 0.0;
    for (int k = 0; k < trials; ++k) {
        LoopGeometry lg = Scene(g, 80, FullImage(), depth, 0.46, 0.65);
        lg.T_A_B = Jitter(g, 0.03, 0.02) * truth;
        const DegeneracyReport r = AssessLoop(lg);
        const core::Vec6 e = core::SE3Log(r.T_A_B.inverse() * truth);
        nees += e.dot(r.information * e);
    }
    return nees / trials;
}

TEST(Degeneracy, MonteCarloNeesMatchesChiSquare6) {
    std::mt19937 g(20261019u);
    const int trials = 500;
    const double nees = MeanNees(g, trials, Depth(1.5, 12.0));
    // Far range (the far-only scene's depths, ~2.8 px of disparity against 0.65 px noise): reported, not gated. It
    // says whether the linearised covariance is still honest where the parallax criterion refuses.
    const double nees_far = MeanNees(g, trials, Depth(25.0, 40.0));
    std::printf("mean NEES over %d trials: %.2f at 1.5-12 m (chi^2_6 mean 6); %.2f at 25-40 m (reported)\n", trials,
                nees, nees_far);
    // Mean of 500 chi^2_6 draws: sd sqrt(12/500) = 0.155. The a-posteriori inflation (max(1, s2hat)) can only lower it.
    EXPECT_GT(nees, 6.0 - 4 * 0.155 - 0.3);
    EXPECT_LT(nees, 6.0 + 4 * 0.155);
}

TEST(Degeneracy, WellConditionedControlPassesEveryCriterion) {
    std::mt19937 g(2);
    LoopGeometry lg = Scene(g, 120, FullImage(), Depth(1.5, 6.0), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport r = AssessLoop(lg);
    Print("control", r);
    EXPECT_FALSE(r.degenerate);
    EXPECT_TRUE(r.reasons.empty());
    EXPECT_LT(core::SE3Log(r.T_A_B.inverse() * TruthAB()).norm(), 0.01);
}

// All inliers on one image row at varied depths: a planar FAN through A's centre, not a 3D line. With known depth it
// is well conditioned (the covariance says mm and mrad), so the gate must NOT refuse it (a pixel-spread gate did).
TEST(Degeneracy, PixelsOnOneRowAreNotRefused) {
    std::mt19937 g(3);
    auto row = [](std::mt19937& gg) {
        std::uniform_real_distribution<double> u(20, 1260), v(-1.0, 1.0);
        return Vec2(u(gg), 360.0 + v(gg));
    };
    LoopGeometry lg = Scene(g, 120, row, Depth(1.5, 6.0), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport r = AssessLoop(lg);
    Print("row-fan", r);
    EXPECT_FALSE(r.degenerate);
}

// A true 3D line (a handrail running away from the camera): rotation about the line is unobservable.
TEST(Degeneracy, ThreeDLineIsRefused) {
    std::mt19937 g(6);
    LoopGeometry lg;
    lg.rig = Rig();
    const SE3 T = TruthAB(), Tinv = T.inverse();
    std::normal_distribution<double> nz(0.0, 1.0);
    std::uniform_real_distribution<double> s(0.0, 1.0);
    const Vec3 a(-0.6, 0.5, 1.5), b(-0.6, 0.5, 9.0);   // in A's frame
    while (lg.X_B.size() < 120) {
        const Vec3 XA = a + s(g) * (b - a), XB = Tinv * XA;
        const Vec2 uvB = core::Project(lg.rig.cam, XB);
        if (XB.z() < 0.5 || uvB.x() < 0 || uvB.x() > 1280 || uvB.y() < 0 || uvB.y() > 720) continue;
        const double d = lg.rig.cam.fx * lg.rig.baseline / XB.z() + 0.65 * nz(g);
        if (d <= 0.05) continue;
        lg.X_B.push_back(core::Unproject(lg.rig.cam, uvB + 0.46 * Vec2{nz(g), nz(g)}, lg.rig.cam.fx * lg.rig.baseline / d));
        lg.uv_A.push_back(core::Project(lg.rig.cam, XA) + 0.46 * Vec2{nz(g), nz(g)});
    }
    lg.T_A_B = Jitter(g, 0.02, 0.01) * T;
    const DegeneracyReport r = AssessLoop(lg);
    Print("3d-line", r);
    EXPECT_TRUE(r.degenerate);
    EXPECT_TRUE(Has(r, "collinear"));
}

// Everything far away (the far end of a long corridor, an open void): stereo depth carries no translation.
TEST(Degeneracy, FarOnlyPointsAreNamedForParallax) {
    std::mt19937 g(4);
    LoopGeometry lg = Scene(g, 120, FullImage(), Depth(25.0, 40.0), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport r = AssessLoop(lg);
    Print("far-only", r);
    EXPECT_TRUE(r.degenerate);
    EXPECT_TRUE(Has(r, "parallax"));
}

// A small object at one depth (25 features, the verifier's minimum, on a ~10 cm patch at 3.5 m): pan and lateral
// translation explain the same pixel motion, so the translation marginal is weak along that coupled direction.
// (120 points on a 40 px patch are NOT degenerate: the gate measured 2.7 cm / 0.4 deg there, a usable loop.)
TEST(Degeneracy, SmallObjectIsNamedForConditioning) {
    std::mt19937 g(5);
    auto patch = [](std::mt19937& gg) {
        std::uniform_real_distribution<double> u(630, 650), v(350, 370);
        return Vec2(u(gg), v(gg));
    };
    LoopGeometry lg = Scene(g, 25, patch, Depth(3.4, 3.6), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport r = AssessLoop(lg);
    Print("small-object", r);
    EXPECT_TRUE(r.degenerate);
    EXPECT_TRUE(Has(r, "conditioning_translation"));
}

// Edge noise model (design §8): the edge information becomes diag(1/var_t I, 1/var_r I) with var = k sigma_PnP^2 +
// floor^2, in the residual tangent; the gate's verdict and the PnP sigmas it reads are unchanged.
TEST(Degeneracy, NoiseModelReplacesTheEdgeCovarianceOnly) {
    std::mt19937 g(9);
    LoopGeometry lg = Scene(g, 120, FullImage(), Depth(1.5, 6.0), 0.46, 0.65);
    lg.T_A_B = Jitter(g, 0.02, 0.01) * TruthAB();
    const DegeneracyReport off = AssessLoop(lg);
    DegeneracyParams p;
    p.noise_model = true;
    p.noise_k_t = 1.33, p.noise_floor_t_m = 0.014, p.noise_k_r = 0.178, p.noise_floor_r_rad = 0.0095;
    const DegeneracyReport on = AssessLoop(lg, p);
    EXPECT_EQ(on.degenerate, off.degenerate);
    EXPECT_EQ(on.sigma_t_worst, off.sigma_t_worst);
    const double vt = 1.33 * off.sigma_t_worst * off.sigma_t_worst + 0.014 * 0.014;
    const double vr = 0.178 * off.sigma_r_worst * off.sigma_r_worst + 0.0095 * 0.0095;
    Mat6 expect = Mat6::Zero();
    expect.topLeftCorner<3, 3>() = core::Mat3::Identity() / vt;
    expect.bottomRightCorner<3, 3>() = core::Mat3::Identity() / vr;
    EXPECT_LT((on.information - expect).norm(), 1e-9 * expect.norm());
    // cov_left maps back to the same residual-tangent blocks (OdometryEdge's conversion).
    const Mat6 info_back = OdometryEdge(0, 1, on.T_A_B, on.cov_left, EdgeType::kLocalLoop).information;
    EXPECT_LT((info_back - expect).norm(), 1e-6 * expect.norm());
}
