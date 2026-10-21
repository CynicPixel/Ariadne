// Windowed BA on OpenCV 5's LevMarqBase (LOCAL_SLAM_2.3.md §3, §6):
//   1. noise-free window from a perturbed start returns the truth;
//   2. one free pose against FIXED landmarks == cv::solvePnPRefineLM (OpenCV's own LM, left camera, quadratic cost);
//   3. the reported camera covariance == the camera block of the dense inverse of the undamped H;
//   4. visual-inertial (INERTIAL_2.4 §4.5): a noise-free window from a perturbed start returns the truth -- poses,
//      velocities, biases -- and the bias marginal is the dense inverse's block.
#include "slam/bundle_adjustment/window_ba_levmarq.hpp"
#include "../numerical/inertial_window_fixture.hpp"

#include <gtest/gtest.h>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <random>
#include <string>

using namespace ariadne;
using namespace ariadne::slam;

namespace {

const BaCamera kCam{640.0, 640.0, 640.0, 360.0, 0.10};

BaWindow MakeWindow(std::mt19937& g, int K, int M, double pix_noise, bool right_too = true) {
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
        const Vec3 X(ux(g), 0.6 * ux(g), uz(g));
        w.X[uint64_t(m)] = X;
        for (int k = 0; k < K; ++k)
            for (bool right : {false, true}) {
                if (right && !right_too) continue;
                Vec3 Xc = w.T_W_C[size_t(k)].R.transpose() * (X - w.T_W_C[size_t(k)].t);
                if (right) Xc.x() -= kCam.baseline;
                if (Xc.z() < 0.2) continue;
                WindowObservation o;
                o.kf = k;
                o.landmark = uint64_t(m);
                o.right = right;
                o.z = Vec2(kCam.fx * Xc.x() / Xc.z() + kCam.cx + pn(g), kCam.fy * Xc.y() / Xc.z() + kCam.cy + pn(g));
                w.obs.push_back(o);
            }
    }
    return w;
}

SE3 Jitter(std::mt19937& g, double rot, double trans) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 xi;
    xi << trans * n(g), trans * n(g), trans * n(g), rot * n(g), rot * n(g), rot * n(g);
    return core::SE3Exp(xi);
}

PgoSettings Tight() {
    PgoSettings s;
    s.rel_energy_delta_tolerance = 1e-15;
    s.step_norm_tolerance = 1e-15;
    s.min_gradient_tolerance = 1e-12;
    s.max_iterations = 200;
    return s;
}

}  // namespace

TEST(WindowBaLevMarq, NoiseFreeWindowReturnsTheTruth) {
    std::mt19937 g(1);
    BaWindow w = MakeWindow(g, 8, 120, 0.0);
    const BaWindow truth = w;
    for (size_t k = 1; k < w.T_W_C.size(); ++k) w.T_W_C[k] = Jitter(g, 0.01, 0.03) * w.T_W_C[k];
    std::normal_distribution<double> n(0.0, 0.03);
    for (auto& [id, X] : w.X) X += Vec3(n(g), n(g), n(g));
    const WindowBaReport rep = SolveWindowBa(w, Tight());
    EXPECT_LT(rep.final_energy, 1e-18);
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        EXPECT_LT((w.T_W_C[k].t - truth.T_W_C[k].t).norm(), 1e-9) << k;
        EXPECT_LT(core::SO3Log(w.T_W_C[k].R * truth.T_W_C[k].R.transpose()).norm(), 1e-9) << k;
    }
    for (const auto& [id, X] : w.X) EXPECT_LT((X - truth.X.at(id)).norm(), 1e-8) << id;
    std::printf("noise-free window: E %.3e -> %.3e in %d iterations, %d factorizations (%d failed)\n",
                rep.initial_energy, rep.final_energy, rep.iterations, rep.factorizations, rep.failed_factorizations);
}

// OpenCV's own LM (solvePnPRefineLM, quadratic) and ours on the same one-free-pose problem must reach the same pose.
TEST(WindowBaLevMarq, OneFreePoseMatchesSolvePnPRefineLM) {
    std::mt19937 g(2);
    BaWindow w = MakeWindow(g, 2, 150, 0.7, /*right_too=*/false);
    w.fixed = {true, false};
    for (const auto& [id, X] : w.X) w.fixed_landmarks.insert(id);
    w.huber_delta = 1e9;   // quadratic, like solvePnPRefineLM
    w.T_W_C[1] = Jitter(g, 0.02, 0.05) * w.T_W_C[1];
    // OpenCV: object points in the world, image points of keyframe 1, pose as the world -> camera transform.
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    for (const auto& o : w.obs)
        if (o.kf == 1) {
            const Vec3& X = w.X.at(o.landmark);
            obj.emplace_back(X.x(), X.y(), X.z());
            img.emplace_back(o.z.x(), o.z.y());
        }
    const SE3 Tcw0 = w.T_W_C[1].inverse();
    cv::Mat R(3, 3, CV_64F), rvec, tvec(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        tvec.at<double>(i) = Tcw0.t(i);
        for (int j = 0; j < 3; ++j) R.at<double>(i, j) = Tcw0.R(i, j);
    }
    cv::Rodrigues(R, rvec);
    const cv::Matx33d K(kCam.fx, 0, kCam.cx, 0, kCam.fy, kCam.cy, 0, 0, 1);
    cv::solvePnPRefineLM(obj, img, K, cv::noArray(), rvec, tvec,
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 200, 1e-15));
    cv::Rodrigues(rvec, R);
    SE3 Tcw;
    for (int i = 0; i < 3; ++i) {
        Tcw.t(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j) Tcw.R(i, j) = R.at<double>(i, j);
    }
    const SE3 cv_T_W_C = Tcw.inverse();
    SolveWindowBa(w, Tight());
    const double dt = (w.T_W_C[1].t - cv_T_W_C.t).norm();
    const double dr = core::SO3Log(w.T_W_C[1].R * cv_T_W_C.R.transpose()).norm();
    std::printf("vs solvePnPRefineLM: dt %.2e m, dr %.2e rad\n", dt, dr);
    EXPECT_LT(dt, 1e-7);
    EXPECT_LT(dr, 1e-8);
}

TEST(WindowBaLevMarq, CameraCovarianceIsTheDenseInverseBlock) {
    std::mt19937 g(3);
    BaWindow w = MakeWindow(g, 5, 60, 0.5);
    w.huber_delta = 1e9;
    const WindowBaReport rep = SolveWindowBa(w, Tight(), {2, 4});
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(L.dim(), L.dim());
    H.topLeftCorner(L.dim_c(), L.dim_c()) = s.Hcc;
    for (int l = 0; l < L.n_lms(); ++l) H.block<3, 3>(L.dim_c() + 3 * l, L.dim_c() + 3 * l) = s.Hll[size_t(l)];
    for (const auto& [cl, B] : s.Hcl) {
        H.block<6, 3>(6 * cl.first, L.dim_c() + 3 * cl.second) = B;
        H.block<3, 6>(L.dim_c() + 3 * cl.second, 6 * cl.first) = B.transpose();
    }
    const Eigen::MatrixXd Hinv = H.inverse();
    for (int k : {2, 4}) {
        const int c = L.cam(k);
        const Mat6 ref = Hinv.block<6, 6>(6 * c, 6 * c);
        EXPECT_LT((rep.covariance.at(k) - ref).cwiseAbs().maxCoeff(), 1e-8 * ref.cwiseAbs().maxCoeff()) << k;
    }
}

// A singular undamped information is reported, not hidden: WindowBaCovariances throws with the block named, and the
// non-throwing variant returns false with the same diagnostic (the VO then records an uninformative marginal).
TEST(WindowBaLevMarq, SingularInformationIsReportedByBothCovariancePaths) {
    std::mt19937 g(4);
    BaWindow w = MakeWindow(g, 4, 40, 0.0);
    std::fill(w.fixed.begin(), w.fixed.end(), false);   // no gauge: six null directions
    std::map<int, Mat6> cov;
    std::string why;
    EXPECT_FALSE(TryWindowBaCovariances(w, {1}, cov, &why));
    EXPECT_NE(why.find("not PD"), std::string::npos) << why;
    EXPECT_TRUE(cov.empty());
    EXPECT_THROW(WindowBaCovariances(w, {1}), std::runtime_error);
    w.fixed[0] = true;
    EXPECT_TRUE(TryWindowBaCovariances(w, {1, 3}, cov));
    EXPECT_EQ(cov.size(), 2u);
}

// 4. Visual-inertial: perturbed poses, velocities, biases and landmarks return to the truth. The IMU is ideal but
// sampled (1 kHz midpoint hold), so the inertial residual at the truth is ~1e-5, not 0: the tolerances say so.
TEST(WindowBaLevMarq, NoiseFreeVisualInertialWindowReturnsTheTruth) {
    std::mt19937 g(31);
    BaWindow w = test::MakeInertialWindow(g, 6, 60, 0.0);
    const BaWindow truth = w;
    std::normal_distribution<double> n(0.0, 1.0);
    const auto rv = [&](double s) { return Vec3(s * n(g), s * n(g), s * n(g)); };
    for (size_t k = 1; k < w.T_W_C.size(); ++k) w.T_W_C[k] = Jitter(g, 0.01, 0.03) * w.T_W_C[k];
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        w.imu.v[k] += rv(0.1);
        w.imu.bg[k] += rv(0.005);
        w.imu.ba[k] += rv(0.05);
    }
    for (auto& [id, X] : w.X) X += rv(0.03);
    w.imu.has_prior = true;   // a loose prior at the truth (window[0]'s, from a previous solve)
    w.imu.prior_b << test::kTrueBg, test::kTrueBa;
    w.imu.prior_info = Mat6::Identity();
    const WindowBaReport rep = SolveWindowBa(w, Tight());
    std::printf("noise-free VI window: E %.3e -> %.3e in %d iterations, %d factorizations (%d failed)\n",
                rep.initial_energy, rep.final_energy, rep.iterations, rep.factorizations, rep.failed_factorizations);
    EXPECT_LT(rep.final_energy, 1e-3);
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        EXPECT_LT((w.T_W_C[k].t - truth.T_W_C[k].t).norm(), 1e-5) << k;
        EXPECT_LT(core::SO3Log(w.T_W_C[k].R * truth.T_W_C[k].R.transpose()).norm(), 1e-5) << k;
        EXPECT_LT((w.imu.v[k] - truth.imu.v[k]).norm(), 1e-4) << k;
        EXPECT_LT((w.imu.bg[k] - test::kTrueBg).norm(), 1e-4) << k;
        EXPECT_LT((w.imu.ba[k] - test::kTrueBa).norm(), 1e-3) << k;
    }
}

TEST(WindowBaLevMarq, BiasCovarianceIsTheDenseInverseBlock) {
    std::mt19937 g(37);
    BaWindow w = test::MakeInertialWindow(g, 5, 40, 0.5);
    w.imu.has_prior = true;
    w.imu.prior_b << test::kTrueBg, test::kTrueBa;
    w.imu.prior_info = Mat6::Identity() * 10.0;
    w.imu.estimate_gravity = true;
    const BaLayout L(w);
    BaSystem s;
    BaLinearize(w, L, s);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(L.dim(), L.dim());
    H.topLeftCorner(L.dim_c(), L.dim_c()) = s.Hcc;
    for (int l = 0; l < L.n_lms(); ++l) H.block<3, 3>(L.dim_c() + 3 * l, L.dim_c() + 3 * l) = s.Hll[size_t(l)];
    for (const auto& [cl, B] : s.Hcl) {
        H.block<6, 3>(6 * cl.first, L.dim_c() + 3 * cl.second) = B;
        H.block<3, 6>(L.dim_c() + 3 * cl.second, 6 * cl.first) = B.transpose();
    }
    const Eigen::MatrixXd C = H.inverse();
    std::map<int, Mat6> pose, bias;   // one factorization: pose, bias and gravity marginals with inertial blocks in S
    Eigen::Matrix2d gcov;
    ASSERT_TRUE(TryWindowBaCovariances(w, {1, 4}, pose, nullptr, nullptr, &bias, {1, 4}, &gcov));
    const Eigen::Matrix2d gd = C.block<2, 2>(L.grav(), L.grav());
    EXPECT_LT((gcov - gd).norm(), 1e-9 * gd.norm());
    for (int k : {1, 4}) {
        const Mat6 dp = C.block<6, 6>(6 * L.cam(k), 6 * L.cam(k)), db = C.block<6, 6>(L.bias(k), L.bias(k));
        EXPECT_LT((pose.at(k) - dp).norm(), 1e-9 * dp.norm()) << k;
        EXPECT_LT((bias.at(k) - db).norm(), 1e-9 * db.norm()) << k;
    }
}

// A dark window: vision in window[0] only. AnchorWindow holds v_0 (at the previous solve's value: the truth here) and
// the IMU dead-reckons the other keyframes from perturbed starts back to the truth.
TEST(WindowBaLevMarq, DarkWindowIsDeadReckonedWithTheVelocityOfWindow0Held) {
    std::mt19937 g(41);
    BaWindow w = test::MakeInertialWindow(g, 5, 40, 0.0);
    w.obs.erase(std::remove_if(w.obs.begin(), w.obs.end(), [](const WindowObservation& o) { return o.kf != 0; }),
                w.obs.end());
    const BaWindow truth = w;
    int held = 0;
    AnchorWindow(w, 8, held);
    ASSERT_TRUE(w.imu.hold_v0);
    ASSERT_EQ(held, 0);
    std::normal_distribution<double> n(0.0, 1.0);
    const auto rv = [&](double s) { return Vec3(s * n(g), s * n(g), s * n(g)); };
    for (size_t k = 1; k < w.T_W_C.size(); ++k) {
        w.T_W_C[k] = Jitter(g, 0.02, 0.1) * w.T_W_C[k];
        w.imu.v[k] += rv(0.2);
    }
    w.imu.has_prior = true;
    w.imu.prior_b << test::kTrueBg, test::kTrueBa;
    w.imu.prior_info = Mat6::Identity() * 1e4;
    const WindowBaReport rep = SolveWindowBa(w, Tight());
    std::printf("dark VI window: E %.3e -> %.3e in %d iterations\n", rep.initial_energy, rep.final_energy, rep.iterations);
    EXPECT_TRUE((w.imu.v[0] - truth.imu.v[0]).isZero(0.0));
    for (size_t k = 1; k < w.T_W_C.size(); ++k) {
        EXPECT_LT((w.T_W_C[k].t - truth.T_W_C[k].t).norm(), 1e-4) << k;
        EXPECT_LT(core::SO3Log(w.T_W_C[k].R * truth.T_W_C[k].R.transpose()).norm(), 1e-5) << k;
        EXPECT_LT((w.imu.v[k] - truth.imu.v[k]).norm(), 1e-4) << k;
    }
}

// §4.8: a window whose world is tilted 2 deg against gravity (window[0] fixed in the tilted frame, gravity at the
// reference) cannot be fitted without moving gravity; with gravity's direction a variable the solve finds the tilt,
// and GravityAlignment about window[0] rotates the solution back onto the untilted truth.
TEST(WindowBaLevMarq, TiltedWindowIsRotatedBackOntoGravity) {
    std::mt19937 g(43);
    const BaWindow truth = test::MakeInertialWindow(g, 6, 60, 0.0);
    BaWindow w = truth;
    const Vec3 c = truth.T_W_C[0].t;
    SE3 tilt;   // 2 deg about a horizontal axis, about window[0]'s position
    tilt.R = core::SO3Exp(Vec3(1, 1, 0).normalized() * (2.0 * M_PI / 180));
    tilt.t = c - tilt.R * c;
    TransformWindow(w, tilt);
    w.imu.g_W = test::kGravity;   // gravity at the reference, not tilted with the world
    w.imu.g_prior = test::kGravity;
    w.imu.estimate_gravity = true;
    w.imu.has_prior = true;
    w.imu.prior_b << test::kTrueBg, test::kTrueBa;
    w.imu.prior_info = Mat6::Identity();
    const WindowBaReport rep = SolveWindowBa(w, Tight());
    const double tilt_found = std::acos(std::clamp(w.imu.g_W.normalized().dot(test::kGravity.normalized()), -1.0, 1.0));
    std::printf("tilted window: E %.3e -> %.3e; tilt found %.4f deg (applied 2)\n", rep.initial_energy,
                rep.final_energy, tilt_found * 180 / M_PI);
    EXPECT_NEAR(tilt_found * 180 / M_PI, 2.0, 1e-3);
    TransformWindow(w, GravityAlignment(w.imu.g_W, test::kGravity, w.T_W_C[0].t));
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        EXPECT_LT((w.T_W_C[k].t - truth.T_W_C[k].t).norm(), 1e-5) << k;
        EXPECT_LT(core::SO3Log(w.T_W_C[k].R * truth.T_W_C[k].R.transpose()).norm(), 1e-5) << k;
        EXPECT_LT((w.imu.v[k] - truth.imu.v[k]).norm(), 1e-4) << k;
    }
}
