// Windowed BA, OpenCV-free (local-slam.md §2-§3): finite-difference Jacobians including near-zero depth and
// wide-FOV rays (Phase 2 exit-gate item, §11.4 / §37.1), the gauge null space [I; I-blocks for cameras, [I, -[X]x]
// for landmarks], the robust gradient, and Schur == dense solve.
#include "slam/bundle_adjustment/window_ba.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ariadne;
using namespace ariadne::slam;

namespace {

const BaCamera kCam{640.0, 640.0, 640.0, 360.0, 0.10};

SE3 RandomPose(std::mt19937& g, double rot, double trans) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}

/// A point at camera-frame position Xc, expressed in the world.
Vec3 PointSeen(const SE3& T_W_C, const Vec3& Xc) { return T_W_C * Xc; }

/// A window of K keyframes looking roughly along +z, M points in front of all of them, left + right observations.
BaWindow MakeWindow(std::mt19937& g, int K, int M, double pix_noise) {
    BaWindow w;
    w.cam = kCam;
    for (int k = 0; k < K; ++k) {
        Vec6 xi;
        xi << 0.15 * k, 0.02 * k, 0.05 * k, 0.01 * k, 0.03 * k, -0.01 * k;
        w.T_W_C.push_back(core::SE3Exp(xi));
        w.fixed.push_back(k == 0);
    }
    std::uniform_real_distribution<double> ux(-1.5, 1.5), uz(1.0, 6.0);
    std::normal_distribution<double> pn(0.0, pix_noise);
    for (int m = 0; m < M; ++m) {
        const Vec3 X{ux(g), 0.6 * ux(g), uz(g)};
        w.X[uint64_t(100 + m)] = X;
        for (int k = 0; k < K; ++k) {
            for (bool right : {false, true}) {
                WindowObservation o;
                o.kf = k;
                o.landmark = uint64_t(100 + m);
                o.right = right;
                Vec3 Xc = w.T_W_C[size_t(k)].R.transpose() * (X - w.T_W_C[size_t(k)].t);
                if (right) Xc.x() -= kCam.baseline;
                if (Xc.z() < 0.2) continue;
                o.z = Vec2{kCam.fx * Xc.x() / Xc.z() + kCam.cx + pn(g), kCam.fy * Xc.y() / Xc.z() + kCam.cy + pn(g)};
                w.obs.push_back(o);
            }
        }
    }
    return w;
}

Eigen::MatrixXd DenseH(const BaSystem& s, const BaLayout& L) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(L.dim(), L.dim());
    H.topLeftCorner(L.dim_c(), L.dim_c()) = s.Hcc;
    for (int l = 0; l < L.n_lms(); ++l) H.block<3, 3>(L.dim_c() + 3 * l, L.dim_c() + 3 * l) = s.Hll[size_t(l)];
    for (const auto& [cl, B] : s.Hcl) {
        H.block<6, 3>(6 * cl.first, L.dim_c() + 3 * cl.second) = B;
        H.block<3, 6>(L.dim_c() + 3 * cl.second, 6 * cl.first) = B.transpose();
    }
    return H;
}

}  // namespace

// §11.4: translation, rotation, point; near-zero depth (z at the stereo floor ~0.28 m and below) and wide-FOV rays
// (|x/z| up to tan 60 deg). Relative tolerance: the Jacobian entries scale with 1/z^2 at the near rays.
TEST(WindowBa, JacobiansMatchFiniteDifferences) {
    std::mt19937 g(1);
    const Vec3 rays[] = {{0.1, -0.2, 3.0},  {0.0, 0.0, 0.30}, {0.05, 0.02, 0.05},   // normal, near floor, very near
                         {1.73, 0.0, 1.0},  {-1.73, 1.2, 1.0}, {2.5, -2.5, 1.5}};     // wide FOV (60 deg and beyond)
    for (const Vec3& Xc : rays) {
        for (bool right : {false, true}) {
            const SE3 T = RandomPose(g, 0.7, 2.0);
            const Vec3 X = PointSeen(T, Xc + (right ? Vec3(kCam.baseline, 0, 0) : Vec3::Zero()));
            WindowObservation o;
            o.right = right;
            o.z = Vec2(600, 300);
            const ObsLinearization l = LinearizeObservation(kCam, T, X, o, 1e-3);
            ASSERT_TRUE(l.valid);
            const double scale = std::max(1.0, l.Jc.cwiseAbs().maxCoeff());
            for (int k = 0; k < 6; ++k) {
                Vec6 d = Vec6::Zero();
                d(k) = 1e-7;
                const Vec2 num = (LinearizeObservation(kCam, core::SE3Exp(d) * T, X, o, 1e-3).r -
                                  LinearizeObservation(kCam, core::SE3Exp(-d) * T, X, o, 1e-3).r) / 2e-7;
                EXPECT_LT((num - l.Jc.col(k)).cwiseAbs().maxCoeff(), 1e-5 * scale) << Xc.transpose() << " pose " << k;
            }
            for (int k = 0; k < 3; ++k) {
                Vec3 d = Vec3::Zero();
                d(k) = 1e-7;
                const Vec2 num = (LinearizeObservation(kCam, T, X + d, o, 1e-3).r -
                                  LinearizeObservation(kCam, T, X - d, o, 1e-3).r) / 2e-7;
                EXPECT_LT((num - l.Jl.col(k)).cwiseAbs().maxCoeff(), 1e-5 * scale) << Xc.transpose() << " point " << k;
            }
        }
    }
}

// SIX gauge directions for stereo (scale is observed through the baseline). Unanchored, at an arbitrary point: H N = 0 with
// camera rows I and landmark rows [I, -[X]x] -- a common left perturbation of everything changes no residual.
TEST(WindowBa, UnanchoredHessianHasTheSixDimensionalGauge) {
    std::mt19937 g(2);
    BaWindow w = MakeWindow(g, 5, 40, 0.5);
    for (auto& T : w.T_W_C) T = RandomPose(g, 0.01, 0.02) * T;   // off the optimum
    std::fill(w.fixed.begin(), w.fixed.end(), false);
    w.huber_delta = 1e9;   // quadratic: the null space is a property of J, the IRLS weights only rescale rows
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    const Eigen::MatrixXd H = DenseH(s, L);
    Eigen::MatrixXd N = Eigen::MatrixXd::Zero(L.dim(), 6);
    for (int c = 0; c < L.n_cams(); ++c) N.block<6, 6>(6 * c, 0) = Mat6::Identity();
    for (const auto& [id, X] : w.X) {
        N.block<3, 3>(L.dim_c() + 3 * L.lm(id), 0) = Mat3::Identity();
        N.block<3, 3>(L.dim_c() + 3 * L.lm(id), 3) = -core::Hat(X);
    }
    EXPECT_LT((H * N).norm() / H.norm(), 1e-12);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    int zeros = 0;
    for (Eigen::Index k = 0; k < es.eigenvalues().size(); ++k) zeros += es.eigenvalues()(k) < 1e-10 * H.norm();
    EXPECT_EQ(zeros, 6);
}

// g = +J^T W r is the gradient of the ROBUST energy (some observations beyond the Huber threshold).
TEST(WindowBa, GradientIsTheRobustCostGradient) {
    std::mt19937 g(3);
    BaWindow w = MakeWindow(g, 4, 25, 3.0);   // 3 px noise: a share of residuals beyond delta = 2 sigma
    const BaLayout L(w);
    BaSystem s;
    const double E = BaLinearize(w, L, s);
    EXPECT_NEAR(E, BaEnergy(w), 1e-9 * E);
    for (int k = 0; k < L.dim(); k += 3) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(L.dim());
        x(k) = 1e-7;
        const double num = (BaEnergy(BaOplus(w, L, x)) - BaEnergy(BaOplus(w, L, -x))) / 2e-7;
        EXPECT_NEAR(num, s.g(k), 1e-4 * std::max(1.0, std::fabs(s.g(k)))) << "param " << k;
    }
}

// The Schur path solves the same damped system as a dense solve of the full H, and returns the FULL step.
TEST(WindowBa, SchurSolveEqualsDenseSolve) {
    std::mt19937 g(4);
    BaWindow w = MakeWindow(g, 6, 50, 0.5);
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    Eigen::VectorXd d = BaDiagonal(s, L);
    d *= 1.0 + 1e-3;   // what the driver's damping does
    BaSetDiagonal(s, L, d);
    BaSchur schur;
    ASSERT_TRUE(schur.Factorize(s, L));
    const Eigen::VectorXd x = schur.Solve(s, L, s.g);
    const Eigen::VectorXd ref = DenseH(s, L).ldlt().solve(-s.g);
    EXPECT_LT((x - ref).cwiseAbs().maxCoeff(), 1e-9 * std::max(1.0, ref.cwiseAbs().maxCoeff()));
}

// A landmark seen by no free camera and no observation cannot be determined: the guard refuses (no silent step).
TEST(WindowBa, GuardRefusesAnUnobservedLandmark) {
    std::mt19937 g(5);
    BaWindow w = MakeWindow(g, 3, 10, 0.5);
    w.X[999] = Vec3(0, 0, 5);
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    BaSchur schur;
    EXPECT_FALSE(schur.Factorize(s, L));
}

// §5 landmark guard: the worst-direction std from the landmark's own information against max_relative_sigma x range.
// Stereo at 3 m (fB = 64 px, sigma 1 px: sigma_Z ~ 0.1 m, 3% of range) passes; at 30 m (~1 of range) it fails; two
// rays from the SAME camera centre (a hovering duplicate keyframe) are parallel and fail outright.
TEST(WindowBa, IllConditionedLandmarksAreFound) {
    BaWindow w;
    w.cam = kCam;
    w.T_W_C = {SE3::Identity(), SE3::Identity()};
    w.fixed = {true, false};
    const auto observe = [&](uint64_t id, const Vec3& X, int kf, bool right) {
        WindowObservation o;
        o.kf = kf;
        o.landmark = id;
        o.right = right;
        Vec3 Xc = w.T_W_C[size_t(kf)].R.transpose() * (X - w.T_W_C[size_t(kf)].t);
        if (right) Xc.x() -= kCam.baseline;
        o.z = Vec2(kCam.fx * Xc.x() / Xc.z() + kCam.cx, kCam.fy * Xc.y() / Xc.z() + kCam.cy);
        w.obs.push_back(o);
    };
    w.X[1] = Vec3(0.2, 0.1, 3.0);
    observe(1, w.X[1], 0, false);
    observe(1, w.X[1], 0, true);
    w.X[2] = Vec3(0.2, 0.1, 30.0);
    observe(2, w.X[2], 0, false);
    observe(2, w.X[2], 0, true);
    w.X[3] = Vec3(-0.3, 0.2, 2.0);
    observe(3, w.X[3], 0, false);
    observe(3, w.X[3], 1, false);   // keyframe 1 sits on keyframe 0: parallel rays
    EXPECT_EQ(IllConditionedLandmarks(w, 0.1), (std::set<uint64_t>{2, 3}));
    RemoveLandmarks(w, {2, 3});
    EXPECT_EQ(w.X.size(), 1u);
    EXPECT_EQ(w.obs.size(), 2u);
}

// §5 anchoring: keyframes {0,1} share 20 landmarks, {2,3} share 20 only with each other -> the second group floats.
// Its oldest keyframe (2) is fixed and anchors 3; the result is a PD system (the MH_05 / V2_03 failure shapes).
TEST(WindowBa, AnchorWindowTiesEveryFreeKeyframeToAFixedOne) {
    std::mt19937 g(6);
    BaWindow w;
    w.cam = kCam;
    for (int k = 0; k < 4; ++k) {
        Vec6 xi;
        xi << 0.1 * k, 0, 0, 0, 0, 0;
        w.T_W_C.push_back(core::SE3Exp(xi));
        w.fixed.push_back(k == 0);
    }
    std::uniform_real_distribution<double> ux(-1.0, 1.0), uz(2.0, 5.0);
    uint64_t id = 0;
    for (const auto& pair : {std::pair<int, int>{0, 1}, std::pair<int, int>{2, 3}})
        for (int m = 0; m < 20; ++m, ++id) {
            w.X[id] = Vec3{ux(g), ux(g), uz(g)};
            for (int k : {pair.first, pair.second})
                for (bool right : {false, true}) {   // stereo: metric scale is observed
                    WindowObservation o;
                    o.kf = k;
                    o.landmark = id;
                    o.right = right;
                    Vec3 Xc = w.T_W_C[size_t(k)].R.transpose() * (w.X[id] - w.T_W_C[size_t(k)].t);
                    if (right) Xc.x() -= kCam.baseline;
                    o.z = Vec2(kCam.fx * Xc.x() / Xc.z() + kCam.cx, kCam.fy * Xc.y() / Xc.z() + kCam.cy);
                    w.obs.push_back(o);
                }
        }
    {   // before anchoring the unanchored pair makes the information singular
        const BaLayout L(w);
        BaSystem s;
        BaLinearize(w, L, s);
        BaSchur schur;
        EXPECT_FALSE(schur.Factorize(s, L));
    }
    int newly = -1;
    const std::vector<int> anchor = AnchorWindow(w, 15, newly);
    EXPECT_EQ(anchor, (std::vector<int>{0, 0, 2, 2}));
    EXPECT_EQ(newly, 1);
    EXPECT_EQ(w.fixed, (std::vector<bool>{true, false, true, false}));
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    BaSchur schur;
    EXPECT_TRUE(schur.Factorize(s, L));

    // A fully covisible window gains no new gauge.
    BaWindow full = MakeWindow(g, 5, 40, 0.0);
    EXPECT_EQ(AnchorWindow(full, 15, newly), (std::vector<int>{0, 0, 0, 0, 0}));
    EXPECT_EQ(newly, 0);
}
