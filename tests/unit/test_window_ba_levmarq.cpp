// Windowed BA on OpenCV 5's LevMarqBase (LOCAL_SLAM_2.3.md §3, §6):
//   1. noise-free window from a perturbed start returns the truth;
//   2. one free pose against FIXED landmarks == cv::solvePnPRefineLM (OpenCV's own LM, left camera, quadratic cost);
//   3. the reported camera covariance == the camera block of the dense inverse of the undamped H.
#include "slam/bundle_adjustment/window_ba_levmarq.hpp"

#include <gtest/gtest.h>
#include <opencv2/geometry.hpp>

#include <random>

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
                BaObservation o;
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
