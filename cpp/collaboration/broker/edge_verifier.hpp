#pragma once
// ---------------------------------------------------------------------------
// 1.6 inter-agent loop verification (design collaboration.md#first-collaboration-loop §5): the broker holds its
// own keyframe A (2D ORB keypoints + descriptors) and receives agent B's
// FEATURE RESPONSE (keypoints, descriptors, 3D points in B's rectified-left
// camera frame, camera metadata). Three stages, each a hard gate:
//
//   1. ORB Hamming matching, Lowe ratio test (mutual nearest not required).
//   2. findEssentialMat RANSAC on the matched pixels: the GEOMETRIC gate (plan
//      1.6). It rejects appearance-only matches, but the essential matrix has
//      no scale, so it cannot be the edge.
//   3. solvePnPRansac: B's 3D points against A's pixels gives the METRIC
//      T_A_B (p_A = T_A_B p_B), the SE(3) edge (locked decision: never Sim3).
//
// RANSAC is seeded (cv::setRNGSeed) so a replay reproduces every edge.
//
// Two DIAGNOSTICS (not gates; D43): h_over_e = findHomography inliers / essential
// inliers on the same matches (near 1: one plane or low parallax explains the
// matches as well as the essential matrix does, so the E gate carried little
// geometric evidence); planarity = lambda_min / lambda_max of the PnP inliers' 3D
// covariance (near 0: a planar target, where PnP has a flip ambiguity). The
// Phase 3 degeneracy gate decides on such evidence; here they are only recorded.
// Information matrix: a [PROXY] diagonal, sigma_t 2 cm and sigma_r 0.01 rad at 30
// PnP inliers, shrinking with sqrt(inliers). Real covariance comes in Phase 3.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "core/serialization/serializer.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/geometry.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::collaboration {

/// One keyframe's local features as the broker keeps them (its own) or
/// receives them (FeatureResponseValue carries the same arrays).
struct KeyframeFeatures {
    uint32_t agent = 0;
    uint64_t id = 0;
    int64_t t_ns = 0;
    std::vector<float> keypoints;       // x y per point
    std::vector<uint8_t> descriptors;   // 32 B per point
    std::vector<float> points3d;        // x y z per point (camera frame), NaN = no depth
    std::vector<float> camera;          // fx fy cx cy baseline
    core::SE3 T_O_C;                    // rectified-left camera in the agent's odometry frame
    std::vector<float> global;          // mean-bit descriptor
};

struct VerifyParams {
    double ratio = 0.8;
    int min_matches = 40;
    double essential_threshold_px = 1.0;
    int min_essential_inliers = 30;
    double min_essential_inlier_ratio = 0.4;
    double pnp_reprojection_px = 2.0;
    int min_pnp_inliers = 25;
};

struct VerifyResult {
    bool ok = false;
    std::string reason;   // first gate that failed, or "verified"
    int matches = 0, essential_inliers = 0, pnp_inliers = 0;
    core::SE3 T_A_B;      // p_A = T_A_B p_B (rectified-left cameras)
    double h_over_e = NAN;    // diagnostic, set once PnP ran
    double planarity = NAN;   // diagnostic, set once PnP ran
    // The PnP inlier correspondences (D73, maintainer-approved hunk): B's 3D points and A's pixels, for the 2.5
    // degeneracy gate (slam/robust/degeneracy.hpp). Output only; set once PnP ran; no gate reads them.
    std::vector<cv::Point3f> inlier_points_B;
    std::vector<cv::Point2f> inlier_pixels_A;
};

/// lambda_min / lambda_max of the 3D points' covariance; NaN for fewer than 4.
inline double Planarity(const std::vector<cv::Point3f>& pts) {
    if (pts.size() < 4) return NAN;
    cv::Mat m(int(pts.size()), 3, CV_64F);
    for (int i = 0; i < m.rows; ++i) {
        m.at<double>(i, 0) = pts[size_t(i)].x;
        m.at<double>(i, 1) = pts[size_t(i)].y;
        m.at<double>(i, 2) = pts[size_t(i)].z;
    }
    cv::Mat cov, mean, ev;
    cv::calcCovarMatrix(m, cov, mean, cv::COVAR_NORMAL | cv::COVAR_ROWS);
    cv::eigen(cov, ev);   // descending
    const double hi = ev.at<double>(0), lo = ev.at<double>(2);
    return hi > 0 ? lo / hi : NAN;
}

/// Homography RANSAC inliers on the same pixel matches; NaN for fewer than 4.
inline int HomographyInliers(const std::vector<cv::Point2f>& pa, const std::vector<cv::Point2f>& pb,
                             double threshold_px, uint64_t seed) {
    if (pa.size() < 4) return -1;
    cv::setRNGSeed(int(seed & 0x7fffffff));
    cv::Mat mask;
    const cv::Mat H = cv::findHomography(pb, pa, cv::RANSAC, threshold_px, mask, 2000, 0.999);
    return H.empty() ? 0 : cv::countNonZero(mask);
}

inline cv::Mat DescriptorMat(const std::vector<uint8_t>& d) {
    return cv::Mat(int(d.size() / 32), 32, CV_8U, const_cast<uint8_t*>(d.data()));
}

/// Stages 2-3 on matches made by any matcher (ORB below; ALIKED + LightGlue in the runtime's hard-pair verifier,
/// D171): pa[i] in A matches pb[i] in B, whose feature index ib[i] selects B's 3D point.
inline VerifyResult VerifyMatched(const std::vector<cv::Point2f>& pa, const std::vector<cv::Point2f>& pb,
                                  const std::vector<int>& ib, const core::FeatureResponseValue& b, const VerifyParams& p,
                                  uint64_t seed) {
    VerifyResult r;
    r.matches = int(pa.size());
    if (r.matches < p.min_matches) return r.reason = "few_matches", r;

    const cv::Matx33d K(b.camera[0], 0, b.camera[2], 0, b.camera[1], b.camera[3], 0, 0, 1);
    cv::setRNGSeed(int(seed & 0x7fffffff));
    cv::Mat mask;
    const cv::Mat E = cv::findEssentialMat(pa, pb, cv::Mat(K), cv::RANSAC, 0.999, p.essential_threshold_px, 1000, mask);
    r.essential_inliers = E.empty() ? 0 : cv::countNonZero(mask);
    if (r.essential_inliers < p.min_essential_inliers ||
        r.essential_inliers < p.min_essential_inlier_ratio * r.matches) {
        return r.reason = "essential_gate", r;
    }

    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    for (size_t i = 0; i < pa.size(); ++i) {
        if (!mask.at<uchar>(int(i))) continue;
        const float* q = &b.points3d[3 * size_t(ib[i])];
        if (!std::isfinite(q[0]) || !std::isfinite(q[2])) continue;
        obj.emplace_back(q[0], q[1], q[2]);
        img.push_back(pa[i]);
    }
    if (int(obj.size()) < p.min_pnp_inliers) return r.reason = "few_3d", r;
    cv::setRNGSeed(int(seed & 0x7fffffff));
    cv::Mat rvec, tvec;
    std::vector<int> inl;
    if (!cv::solvePnPRansac(obj, img, cv::Mat(K), cv::noArray(), rvec, tvec, false, 200, p.pnp_reprojection_px, 0.999,
                            inl, cv::SOLVEPNP_EPNP)) {
        return r.reason = "pnp_failed", r;
    }
    r.pnp_inliers = int(inl.size());
    // Diagnostics after the last RANSAC (they consume the RNG; computed last and
    // seeded, so they cannot change the gates' outcome).
    {
        std::vector<cv::Point3f> in3;
        for (int i : inl) in3.push_back(obj[size_t(i)]);
        r.planarity = Planarity(in3);
        r.inlier_points_B = in3;
        for (int i : inl) r.inlier_pixels_A.push_back(img[size_t(i)]);
        const int h = HomographyInliers(pa, pb, p.essential_threshold_px, seed);
        r.h_over_e = h < 0 || r.essential_inliers == 0 ? NAN : double(h) / r.essential_inliers;
    }
    if (r.pnp_inliers < p.min_pnp_inliers) return r.reason = "pnp_gate", r;
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) r.T_A_B.R(i, j) = R.at<double>(i, j);
        r.T_A_B.t(i) = tvec.at<double>(i);
    }
    r.ok = true;
    r.reason = "verified";
    return r;
}

inline VerifyResult VerifyEdge(const KeyframeFeatures& a, const core::FeatureResponseValue& b, const VerifyParams& p,
                               uint64_t seed) {
    const size_t na = a.keypoints.size() / 2, nb = b.keypoints.size() / 2;
    if (na < 8 || nb < 8) {
        VerifyResult r;
        return r.reason = "too_few_features", r;
    }
    std::vector<std::vector<cv::DMatch>> knn;
    cv::BFMatcher(cv::NORM_HAMMING).knnMatch(DescriptorMat(a.descriptors), DescriptorMat(b.descriptors), knn, 2);
    std::vector<cv::Point2f> pa, pb;
    std::vector<int> ib;
    for (const auto& m : knn) {
        if (m.size() == 2 && m[0].distance < p.ratio * m[1].distance) {
            pa.emplace_back(a.keypoints[2 * m[0].queryIdx], a.keypoints[2 * m[0].queryIdx + 1]);
            pb.emplace_back(b.keypoints[2 * m[0].trainIdx], b.keypoints[2 * m[0].trainIdx + 1]);
            ib.push_back(m[0].trainIdx);
        }
    }
    return VerifyMatched(pa, pb, ib, b, p, seed);
}

/// The [PROXY] information matrix, upper triangle (21 floats), translation first.
inline std::vector<float> ProxyInformation(int pnp_inliers) {
    const double s = std::sqrt(30.0 / std::max(pnp_inliers, 1));
    const double it = 1.0 / std::pow(0.02 * s, 2), ir = 1.0 / std::pow(0.01 * s, 2);
    std::vector<float> u(21, 0.0f);
    int k = 0;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j, ++k) {
            if (i == j) u[size_t(k)] = float(i < 3 ? it : ir);
        }
    }
    return u;
}

}  // namespace ariadne::collaboration
