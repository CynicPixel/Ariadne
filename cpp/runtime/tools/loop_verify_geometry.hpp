#pragma once
// ---------------------------------------------------------------------------
// Two-view geometry of the loop-verification evaluation (PERCEPTION_2.1.md
// §12.4), shared by ariadne_loop_verify and its convention test.
//
// EstimateRelPose: raw (distorted) pixel matches a -> b, undistorted with the
// camera's radtan model back to pixels (P = K), then the 1.6 verifier's own
// essential-matrix call (collaboration/broker/edge_verifier.hpp):
// findEssentialMat(RANSAC, 0.999, 1.0 px, 1000 iterations). In the pinned
// five-point.cpp that is classic RANSAC with a crisp threshold of 1.0 px over
// the mean focal length, and a registrator RNG seeded (uint64)-1 on every call,
// so each pair is deterministic whatever the processing order. recoverPose
// picks the (R, t) with points in front of both cameras: p_b = R p_a + t, t
// unit-norm (an essential matrix has no scale).
//
// PoseErrorDeg: glue-factory's relative_pose_error (cvg/glue-factory@2d17e3b,
// geometry/epipolar.py): rotation error = angle of R^T R_gt; translation error
// = angle between t and t_gt, folded as min(e, 180 - e) because E fixes t only
// up to sign; the pose error is the max of the two.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "perception/stereo/camera_model.hpp"

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace ariadne::runtime {

struct RelPoseEstimate {
    int matches = 0;
    int inliers = 0;   // essential-matrix RANSAC inliers: the verification score
    bool ok = false;   // a model was found and recoverPose ran
    core::Mat3 R = core::Mat3::Identity();
    core::Vec3 t = core::Vec3::Zero();
};

struct PoseError {
    double r_deg = INFINITY, t_deg = INFINITY, deg = INFINITY;   // INFINITY = estimator failure
};

inline cv::Matx33d CameraMatrix(const perception::PinholeRadtan& c) {
    return {c.fu, 0, c.cu, 0, c.fv, c.cv, 0, 0, 1};
}

inline RelPoseEstimate EstimateRelPose(const std::vector<cv::Point2f>& pa_raw, const std::vector<cv::Point2f>& pb_raw,
                                       const perception::PinholeRadtan& cam) {
    RelPoseEstimate r;
    r.matches = static_cast<int>(pa_raw.size());
    if (r.matches < 5) return r;
    const cv::Matx33d K = CameraMatrix(cam);
    const std::vector<double> D(cam.dist.begin(), cam.dist.end());
    std::vector<cv::Point2f> pa, pb;
    cv::undistortPoints(pa_raw, pa, K, D, cv::noArray(), K);
    cv::undistortPoints(pb_raw, pb, K, D, cv::noArray(), K);
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pa, pb, cv::Mat(K), cv::RANSAC, 0.999, 1.0, 1000, mask);
    if (E.empty() || E.rows < 3) return r;
    if (E.rows > 3) E = E.rowRange(0, 3).clone();   // several five-point solutions: the first is RANSAC's best
    r.inliers = cv::countNonZero(mask);
    cv::Mat R, t;
    cv::recoverPose(E, pa, pb, cv::Mat(K), R, t, mask);
    for (int i = 0; i < 3; ++i) {
        r.t(i) = t.at<double>(i);
        for (int j = 0; j < 3; ++j) r.R(i, j) = R.at<double>(i, j);
    }
    r.ok = true;
    return r;
}

inline double AngleDeg(const core::Vec3& a, const core::Vec3& b) {
    const double n = std::max(a.norm() * b.norm(), 1e-10);
    return std::acos(std::clamp(a.dot(b) / n, -1.0, 1.0)) * 180.0 / M_PI;
}

/// Error of an estimate against the ground truth T_Cb_Ca (p_b = T_Cb_Ca p_a).
inline PoseError PoseErrorDeg(const RelPoseEstimate& e, const core::SE3& T_Cb_Ca) {
    PoseError p;
    if (!e.ok) return p;
    const double te = AngleDeg(e.t, T_Cb_Ca.t);
    p.t_deg = std::min(te, 180.0 - te);
    const double c = std::clamp(((e.R.transpose() * T_Cb_Ca.R).trace() - 1.0) / 2.0, -1.0, 1.0);
    p.r_deg = std::acos(c) * 180.0 / M_PI;
    p.deg = std::max(p.r_deg, p.t_deg);
    return p;
}

}  // namespace ariadne::runtime
