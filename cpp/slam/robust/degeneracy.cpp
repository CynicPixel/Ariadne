// Loop-closure degeneracy gate (arch §14.4, D71): see degeneracy.hpp for what and why.
#include "slam/robust/degeneracy.hpp"

#include "core/uncertainty/covariance.hpp"

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace ariadne::slam {

namespace {

constexpr double kZFloor = 0.05;   // [m] a point this close to (or behind) either camera is not used

/// Eigen-decomposition of a symmetric 3x3 or 2x2 block, ascending eigenvalues.
template <typename M>
Eigen::SelfAdjointEigenSolver<M> Sym(const M& m) {
    return Eigen::SelfAdjointEigenSolver<M>(0.5 * (m + m.transpose()));
}

template <int D>
Eigen::Matrix<double, D, D> Covariance(const std::vector<Eigen::Matrix<double, D, 1>>& v) {
    Eigen::Matrix<double, D, 1> mean = Eigen::Matrix<double, D, 1>::Zero();
    for (const auto& x : v) mean += x;
    mean /= double(v.size());
    Eigen::Matrix<double, D, D> c = Eigen::Matrix<double, D, D>::Zero();
    for (const auto& x : v) c += (x - mean) * (x - mean).transpose();
    return c / double(v.size());
}

/// Refine with OpenCV's LM (unweighted reprojection, the pose as object(B) -> camera(A)).
SE3 RefineOpenCv(const core::PinholeCamera& k, const std::vector<core::Vec3>& X, const std::vector<core::Vec2>& uv,
                 const SE3& T0) {
    std::vector<cv::Point3d> obj;
    std::vector<cv::Point2d> img;
    for (size_t i = 0; i < X.size(); ++i) {
        obj.emplace_back(X[i].x(), X[i].y(), X[i].z());
        img.emplace_back(uv[i].x(), uv[i].y());
    }
    const cv::Matx33d K(k.fx, 0, k.cx, 0, k.fy, k.cy, 0, 0, 1);
    cv::Matx33d R;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) R(i, j) = T0.R(i, j);
    cv::Mat rvec, tvec = (cv::Mat_<double>(3, 1) << T0.t.x(), T0.t.y(), T0.t.z());
    cv::Rodrigues(R, rvec);
    cv::solvePnPRefineLM(obj, img, K, cv::noArray(), rvec, tvec);
    cv::Mat Rm;
    cv::Rodrigues(rvec, Rm);
    SE3 T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T.R(i, j) = Rm.at<double>(i, j);
        T.t(i) = tvec.at<double>(i);
    }
    return T;
}

}  // namespace

core::Mat3 StereoPointJacobian(const core::RectifiedStereo& rig, const core::Vec3& X_B) {
    // X = ((u - cx) z / fx, (v - cy) z / fy, z) with z = fx b / d, so dz/dd = -z / d and dX/dd = -X / d.
    const double d = rig.cam.fx * rig.baseline / X_B.z();
    core::Mat3 J;
    J << X_B.z() / rig.cam.fx, 0.0, -X_B.x() / d,
         0.0, X_B.z() / rig.cam.fy, -X_B.y() / d,
         0.0, 0.0, -X_B.z() / d;
    return J;
}

Eigen::Matrix2d PixelCovariance(const LoopGeometry& g, const SE3& T_A_B, const core::Vec3& X_B,
                                const DegeneracyParams& p) {
    core::Mat23 J_point;
    core::ReprojectionJacobians(g.rig.cam, T_A_B, X_B, nullptr, &J_point);
    const core::Mat3 Js = StereoPointJacobian(g.rig, X_B);
    const core::Vec3 s(p.sigma_px * p.sigma_px, p.sigma_px * p.sigma_px, p.sigma_disparity_px * p.sigma_disparity_px);
    const core::Mat3 cov_X = Js * s.asDiagonal() * Js.transpose();
    return p.sigma_px * p.sigma_px * Eigen::Matrix2d::Identity() + J_point * cov_X * J_point.transpose();
}

DegeneracyReport AssessLoop(const LoopGeometry& g, const DegeneracyParams& p) {
    DegeneracyReport rep;
    rep.T_A_B = g.T_A_B;
    // Usable points: in front of both cameras.
    std::vector<core::Vec3> X;
    std::vector<core::Vec2> uv;
    for (size_t i = 0; i < std::min(g.X_B.size(), g.uv_A.size()); ++i)
        if (g.X_B[i].z() > kZFloor && (g.T_A_B * g.X_B[i]).z() > kZFloor) X.push_back(g.X_B[i]), uv.push_back(g.uv_A[i]);
    rep.points = int(X.size());
    if (rep.points < 6) {
        rep.reasons.push_back("too_few_points");
        return rep;
    }

    // 1. Refine: OpenCV's LM, then Gauss-Newton on the per-point weighted cost (the ML estimate under the noise
    //    model whose covariance we report).
    SE3 T = RefineOpenCv(g.rig.cam, X, uv, g.T_A_B);
    Mat6 H;
    double chi2 = 0.0, sq_px = 0.0;
    auto linearize = [&](const SE3& Tc, core::Vec6* b) {
        H.setZero();
        chi2 = sq_px = 0.0;
        if (b) b->setZero();
        for (size_t i = 0; i < X.size(); ++i) {
            core::Mat26 J;
            const core::Vec3 Xc = core::ReprojectionJacobians(g.rig.cam, Tc, X[i], &J, nullptr);
            if (!(Xc.z() > kZFloor)) continue;
            const core::Vec2 r = core::Project(g.rig.cam, Xc) - uv[i];
            const Eigen::Matrix2d W = PixelCovariance(g, Tc, X[i], p).inverse();
            H += J.transpose() * W * J;
            if (b) *b += J.transpose() * W * r;
            chi2 += r.dot(W * r);
            sq_px += r.squaredNorm();
        }
    };
    for (int it = 0; it < p.gauss_newton_steps; ++it) {
        core::Vec6 b;
        linearize(T, &b);
        const core::Vec6 dx = -H.ldlt().solve(b);
        if (!dx.allFinite()) break;
        T = core::SE3Exp(dx) * T;
        if (dx.norm() < 1e-12) break;
    }
    linearize(T, nullptr);
    rep.T_A_B = T;
    rep.sigma2_hat = chi2 / double(2 * rep.points - 6);
    rep.rms_px = std::sqrt(sq_px / rep.points);

    // 2-3. Covariance (inflated when the residuals say the noise model is optimistic) and the edge information.
    const Eigen::SelfAdjointEigenSolver<Mat6> eh = Sym(H);
    if (!(eh.eigenvalues()(0) > 1e-12 * std::max(1.0, eh.eigenvalues()(5)))) {
        rep.reasons.push_back("conditioning_singular");
        return rep;
    }
    rep.cov_left = eh.eigenvectors() * eh.eigenvalues().cwiseInverse().asDiagonal() * eh.eigenvectors().transpose() *
                   std::max(1.0, rep.sigma2_hat);
    const Mat6 cov_r = core::ChangeFrame(T.inverse(), rep.cov_left);
    rep.information = cov_r.ldlt().solve(Mat6::Identity());
    rep.information = 0.5 * (rep.information + rep.information.transpose());

    // 4. Criteria. Marginals = blocks of the covariance; residual vector [rho; phi], translation first.
    const auto et = Sym(core::Mat3(cov_r.topLeftCorner<3, 3>()));
    const auto er = Sym(core::Mat3(cov_r.bottomRightCorner<3, 3>()));
    rep.sigma_t_worst = std::sqrt(std::max(0.0, et.eigenvalues()(2)));
    rep.sigma_r_worst = std::sqrt(std::max(0.0, er.eigenvalues()(2)));
    rep.worst_t_dir = et.eigenvectors().col(2);
    rep.worst_r_axis = er.eigenvectors().col(2);
    rep.ratio_t = et.eigenvalues()(0) / et.eigenvalues()(2);
    rep.ratio_r = er.eigenvalues()(0) / er.eigenvalues()(2);
    if (p.noise_model) {   // the edge's covariance, replaced by the calibrated model (criteria below keep the PnP sigmas)
        Mat6 cov_m = Mat6::Zero();
        cov_m.topLeftCorner<3, 3>() = (p.noise_k_t * rep.sigma_t_worst * rep.sigma_t_worst +
                                       p.noise_floor_t_m * p.noise_floor_t_m) * core::Mat3::Identity();
        cov_m.bottomRightCorner<3, 3>() = (p.noise_k_r * rep.sigma_r_worst * rep.sigma_r_worst +
                                           p.noise_floor_r_rad * p.noise_floor_r_rad) * core::Mat3::Identity();
        rep.information = cov_m.ldlt().solve(Mat6::Identity());
        rep.information = 0.5 * (rep.information + rep.information.transpose());
        rep.cov_left = core::ChangeFrame(T, cov_m);   // back to T_A_B's left tangent (inverse of the map above)
    }

    std::vector<double> tri;
    for (const auto& x : X) tri.push_back(2.0 * std::atan(g.rig.baseline / (2.0 * x.z())) * 180.0 / M_PI);
    rep.near_points = int(std::count_if(tri.begin(), tri.end(), [&](double a) { return a >= p.min_triangulation_deg; }));
    std::nth_element(tri.begin(), tri.begin() + long(tri.size() / 2), tri.end());
    rep.median_triangulation_deg = tri[tri.size() / 2];

    const auto ep = Sym(Covariance<2>(uv));
    rep.pixel_spread_ratio = ep.eigenvalues()(1) > 0 ? ep.eigenvalues()(0) / ep.eigenvalues()(1) : 0.0;
    const auto e3 = Sym(Covariance<3>(X));
    rep.planarity = e3.eigenvalues()(2) > 0 ? e3.eigenvalues()(0) / e3.eigenvalues()(2) : 0.0;
    rep.line_spread_ratio = e3.eigenvalues()(2) > 0 ? e3.eigenvalues()(1) / e3.eigenvalues()(2) : 0.0;
    rep.corridor = e3.eigenvalues().sum() > 0 ? e3.eigenvalues()(2) / e3.eigenvalues().sum() : 1.0;

    if (rep.sigma_t_worst > p.max_sigma_t_m) rep.reasons.push_back("conditioning_translation");
    if (rep.sigma_r_worst > p.max_sigma_r_rad) rep.reasons.push_back("conditioning_rotation");
    if (rep.near_points < p.min_near_points) rep.reasons.push_back("parallax");
    if (rep.line_spread_ratio < p.min_line_spread_ratio) rep.reasons.push_back("collinear");
    rep.degenerate = !rep.reasons.empty();
    return rep;
}

}  // namespace ariadne::slam
