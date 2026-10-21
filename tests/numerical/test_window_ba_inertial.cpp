// Visual-inertial window BA (INERTIAL_2.4.md §4.2, §4.5), OpenCV-free: the assembled gradient and Hessian including
// the inertial, bias random-walk and bias-prior blocks against finite differences of the energy, Schur == dense with
// inertial blocks in the reduced system, the Galilean null space of an inertial-only chain, and AnchorWindow's rule
// that an inertial factor ties only out of a group with two pose-determined keyframes.
#include "inertial_window_fixture.hpp"

#include <gtest/gtest.h>

#include <algorithm>

using namespace ariadne;
using namespace ariadne::slam;
using namespace ariadne::slam::test;

namespace {

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

/// Move every free state off the truth (poses, velocities, biases, landmarks) and give window[0] a bias prior.
void Perturb(BaWindow& w, std::mt19937& g, double scale) {
    std::normal_distribution<double> n(0.0, 1.0);
    const auto rv = [&](double s) { return Vec3(s * n(g), s * n(g), s * n(g)); };
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        if (!w.fixed[k]) {
            Vec6 d;
            d << rv(0.02 * scale), rv(0.01 * scale);
            w.T_W_C[k] = core::SE3Exp(d) * w.T_W_C[k];
        }
        w.imu.v[k] += rv(0.05 * scale);
        w.imu.bg[k] += rv(0.003 * scale);
        w.imu.ba[k] += rv(0.03 * scale);
    }
    for (auto& [id, X] : w.X) X += rv(0.02 * scale);
    w.imu.has_prior = true;
    w.imu.prior_b << kTrueBg + rv(0.002), kTrueBa + rv(0.02);
    Vec6 sd;
    sd << Vec3::Constant(0.01), Vec3::Constant(0.1);
    w.imu.prior_info = sd.cwiseInverse().cwiseAbs2().asDiagonal();
    // Gravity's direction as a variable (§4.8), off the truth by ~1 deg, with a prior.
    w.imu.estimate_gravity = true;
    w.imu.g_W = core::SO3Exp(Vec3(0.01 * scale, -0.015 * scale, 0.0)) * w.imu.g_W;
    w.imu.has_g_prior = true;
    w.imu.g_prior = core::SO3Exp(Vec3(-0.005, 0.004, 0.0)) * kGravity;
    w.imu.g_prior_info = Eigen::Matrix2d::Identity() * 1e4;
}

}  // namespace

// g = J^T W r is the gradient of the energy the driver accepts steps on, with every inertial term in it (§4 of the
// advisor review: BaEnergy and BaLinearize share one function for the Gaussian terms; this would catch a mismatch).
TEST(WindowBaInertial, GradientIsTheEnergyGradient) {
    std::mt19937 g(5);
    BaWindow w = MakeInertialWindow(g, 5, 30, 0.5);
    Perturb(w, g, 1.0);
    const BaLayout L(w);
    ASSERT_EQ(L.n_inertial(), 5);
    ASSERT_GE(L.grav(), 0);
    BaSystem s;
    const double E = BaLinearize(w, L, s);
    EXPECT_NEAR(E, BaEnergy(w), 1e-9 * E);
    const double h = 1e-6;
    for (int i = 0; i < L.dim(); ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(L.dim());
        x(i) = h;
        const double num = (BaEnergy(BaOplus(w, L, x)) - BaEnergy(BaOplus(w, L, -x))) / (2 * h);
        EXPECT_NEAR(num, s.g(i), 1e-5 * (1.0 + std::abs(s.g(i))) + 1e-9 * E / h) << "parameter " << i;
    }
}

// At the noise-free truth every residual is ~0, so the Gauss-Newton H IS the Hessian of the energy: each column of
// H matches the central difference of the assembled gradient. This checks every inertial block of H, the cross
// terms between poses and [v, bg, ba], and their placement in the layout.
TEST(WindowBaInertial, HessianMatchesFiniteDifferencesOfTheGradientAtTheTruth) {
    std::mt19937 g(9);
    BaWindow w = MakeInertialWindow(g, 4, 25, 0.0);
    w.imu.has_prior = true;
    w.imu.prior_b << kTrueBg, kTrueBa;
    w.imu.prior_info = Mat6::Identity() * 100.0;
    w.imu.estimate_gravity = true;   // the tilt block and its cross terms too
    w.imu.has_g_prior = true;
    w.imu.g_prior = kGravity;
    w.imu.g_prior_info = Eigen::Matrix2d::Identity() * 1e4;
    const BaLayout L(w);
    ASSERT_GE(L.grav(), 0);
    BaSystem s;
    BaLinearize(w, L, s);
    const Eigen::MatrixXd H = DenseH(s, L);
    const double h = 1e-6;
    for (int i = 0; i < L.dim(); ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(L.dim());
        x(i) = h;
        BaSystem sp, sm;
        BaLinearize(BaOplus(w, L, x), L, sp);
        BaLinearize(BaOplus(w, L, -x), L, sm);
        const Eigen::VectorXd num = (sp.g - sm.g) / (2 * h);
        EXPECT_LT((num - H.col(i)).norm(), 1e-4 * (1.0 + H.col(i).norm())) << "column " << i;
    }
}

// The Schur path with inertial blocks in the reduced system == the dense damped solve.
TEST(WindowBaInertial, SchurSolveEqualsDenseSolve) {
    std::mt19937 g(13);
    BaWindow w = MakeInertialWindow(g, 5, 30, 0.5);
    Perturb(w, g, 1.0);
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    Eigen::VectorXd d = BaDiagonal(s, L);
    d *= 1.0 + 1e-3;
    BaSetDiagonal(s, L, d);
    BaSchur schur;
    ASSERT_TRUE(schur.Factorize(s, L));
    const Eigen::VectorXd x = schur.Solve(s, L, s.g);
    const Eigen::VectorXd xd = DenseH(s, L).ldlt().solve(-s.g);
    EXPECT_LT((x - xd).norm(), 1e-8 * (1.0 + xd.norm()));
}

// The IMU does not observe velocity: v_k += c, p_k += c (t_k - t_0) leaves every inertial residual unchanged. With
// window[0]'s pose fixed and no vision, that direction is in the null space of H.
TEST(WindowBaInertial, InertialOnlyChainHasTheGalileanNullSpace) {
    std::mt19937 g(17);
    BaWindow w = MakeInertialWindow(g, 4, 0, 0.0);
    w.fixed = {true, false, false, false};
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    const Vec3 c(0.3, -0.2, 0.5);
    Eigen::VectorXd n = Eigen::VectorXd::Zero(L.dim());
    for (int k = 0; k < 4; ++k) {
        if (L.cam(k) >= 0) n.segment<3>(6 * L.cam(k)) = c * (KeyframeTime(k) - KeyframeTime(0));
        n.segment<3>(L.vel(k)) = c;
    }
    const Eigen::MatrixXd H = DenseH(s, L);
    // Compared with a pose direction that is NOT Galilean (move keyframe 2 alone), which the IMU observes. (H's norm
    // is dominated by the stiff gyro-bias blocks, so it is not the scale to compare against.)
    Eigen::VectorXd m = Eigen::VectorXd::Zero(L.dim());
    m.segment<3>(6 * L.cam(2)) = c;
    const double null_gain = (H * n).norm() / n.norm(), observed_gain = (H * m).norm() / m.norm();
    std::printf("Galilean direction |Hn|/|n| %.3e vs a single-keyframe translation %.3e\n", null_gain, observed_gain);
    EXPECT_LT(null_gain, 1e-9 * observed_gain);
}

// AnchorWindow: window[0] without a covisible partner (a dark stretch) has v_0 held, and then the inertial chain ties
// every keyframe (none is held at a predicted pose); with a covisible partner v_0 stays free and the inertial factors
// tie out of the two-keyframe group. Either way H is PD. Without the IMU the untied keyframes are held, as before D59.
TEST(WindowBaInertial, InertialFactorsTieOnlyOutOfAGroupThatPinsTheVelocity) {
    std::mt19937 g(19);
    BaWindow base = MakeInertialWindow(g, 4, 30, 0.0);
    base.imu.has_prior = true;
    base.imu.prior_b << kTrueBg, kTrueBa;
    base.imu.prior_info = Mat6::Identity() * 100.0;
    // H's spread is ~1e9 (the gyro-bias blocks are stiff), so "PD" is judged against the null case below, whose
    // smallest eigenvalue is round-off (~1e-9), not against the largest one.
    const auto min_eigenvalue = [](const BaWindow& w) {
        const BaLayout L(w);
        BaSystem s;
        BaLinearize(w, L, s);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(DenseH(s, L));
        std::printf("H eigenvalues: min %.3e, max %.3e\n", es.eigenvalues()(0), es.eigenvalues()(es.eigenvalues().size() - 1));
        return es.eigenvalues()(0);
    };
    const auto positive_definite = [&](const BaWindow& w) { return min_eigenvalue(w) > 1e-2; };

    BaWindow only0 = base;   // vision in window[0] only
    only0.obs.erase(std::remove_if(only0.obs.begin(), only0.obs.end(), [](const WindowObservation& o) { return o.kf != 0; }),
                    only0.obs.end());
    int held = 0;
    std::vector<int> anchor = AnchorWindow(only0, 8, held);
    EXPECT_TRUE(only0.imu.hold_v0);
    EXPECT_EQ(held, 0);
    EXPECT_EQ(anchor, std::vector<int>({0, 0, 0, 0}));
    EXPECT_TRUE(positive_definite(only0));
    EXPECT_EQ(BaLayout(only0).vel(0), -1);
    BaWindow released = only0;   // the same ties with v_0 free: the Galilean null space is back
    released.imu.hold_v0 = false;
    EXPECT_LT(std::abs(min_eigenvalue(released)), 1e-6);

    BaWindow first2 = base;   // vision in windows[0] and [1]: a covisibility tie, then inertial ties
    first2.obs.erase(std::remove_if(first2.obs.begin(), first2.obs.end(), [](const WindowObservation& o) { return o.kf > 1; }),
                     first2.obs.end());
    first2.imu.hold_v0 = true;   // AnchorWindow decides, whatever the caller left
    anchor = AnchorWindow(first2, 8, held);
    EXPECT_FALSE(first2.imu.hold_v0);
    EXPECT_EQ(held, 0);
    EXPECT_EQ(anchor, std::vector<int>({0, 0, 0, 0}));
    EXPECT_TRUE(positive_definite(first2));

    BaWindow visual = first2;   // the same window without the IMU: windows [2], [3] are held, as before D59
    visual.imu = BaInertial{};
    anchor = AnchorWindow(visual, 8, held);
    EXPECT_EQ(held, 2);
}

// GravityAlignment is a rigid motion about c: it maps g_from's direction to g_to's with no rotation about gravity, and
// every residual of the window (reprojection, inertial, priors) is invariant under it, so the energy is unchanged.
TEST(WindowBaInertial, GravityAlignmentIsARigidMotionThatPreservesTheEnergy) {
    std::mt19937 g(23);
    BaWindow w = MakeInertialWindow(g, 5, 30, 0.5);
    Perturb(w, g, 1.0);
    const Vec3 c = w.T_W_C[0].t;
    const SE3 G = GravityAlignment(w.imu.g_W, kGravity, c);
    BaWindow r = w;
    TransformWindow(r, G);
    EXPECT_LT((r.imu.g_W.normalized() - kGravity.normalized()).norm(), 1e-12);
    EXPECT_LT((G * c - c).norm(), 1e-12);
    EXPECT_LT(std::abs(core::SO3Log(G.R).dot(w.imu.g_W.normalized())), 1e-12);   // no yaw
    EXPECT_NEAR(BaEnergy(r), BaEnergy(w), 1e-9 * BaEnergy(w));
}
