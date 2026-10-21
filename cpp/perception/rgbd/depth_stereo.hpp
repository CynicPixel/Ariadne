#pragma once
// ---------------------------------------------------------------------------
// 2.6 (D90): an RGB-D sensor's depth image as StereoVo's stereo evidence.
//
// StereoVo triangulates a new landmark from a left point and a right point on
// a rectified rig of baseline B, then gates the result with sigma_Z (§9.5)
// and cross-checks it against a dense depth (§9.6). An RGB-D camera has no
// right image, so we synthesise the right point a rig of baseline B WOULD see:
//   u_R = u_L - fx * B / Z,   v_R = v_L
// (ORB-SLAM2's RGB-D "virtual stereo", Mur-Artal & Tardos 2017 §III-A). The
// triangulation then returns the sensor Z exactly, and everything downstream
// (BA right residual, PnP, keyframe records) is unchanged.
//
// sigma_Z model (stated, not tuned): Khoshelham & Elberink, Sensors 12:1437
// (2012), eq. 7: sigma_Z = (m / f b) Z^2 sigma_d', with |m/fb| = 2.85e-5 cm^-1
// from their calibration and sigma_d' = 1/2 px, i.e. sigma_Z = 1.425e-3 Z^2
// (Z, sigma in metres; 3.6 cm at 5 m, the paper's "about 4 cm at maximum
// range"). [PAPER, transcribed 2026-10-21 from the PMC full text]
// The same Z^2 law is the stereo law sigma_Z = Z^2 / (fx B) sigma_d, so the
// model reaches StereoVo's triangulation gate as an equivalent disparity
// noise on the virtual rig: sigma_d = k fx B (DisparitySigmaPx).
//
// Consequences, stated:
// - The §9.6 cross-check compares Z with itself and can never reject here;
//   the depth model acts only through sigma_tri (the relative-sigma gate).
// - BA weights the virtual right point at ba.sigma_px (1 px) while the model
//   implies sigma_d ~ 0.06 px: depth is weighted ~15x looser than the model.
//   ORB-SLAM2 does the same (one sigma per octave for u_R). Deviation recorded
//   in D90.
// ---------------------------------------------------------------------------
#include "perception/stereo/sgbm_stereo.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace ariadne::perception {

struct DepthSensorModel {
    double sigma_k_per_m = 1.425e-3;   // sigma_Z = k Z^2 (Khoshelham & Elberink 2012 eq. 7)
    double z_min_m = 0.4;              // Kinect v1 returns nothing reliable nearer (structured-light minimum range)
    double z_max_m = 4.0;              // sigma_Z(4 m) = 2.3 cm; beyond, the Kinect's 1/8-px disparity steps dominate
    /// 3x3 max/min depth ratio above which the pixel straddles a depth edge (mixed / "flying" pixels). The
    /// counterpart of SGBM's dense-support mask (D34).
    double max_neighbour_ratio = 1.05;
    double virtual_baseline_m = 0.08;   // ORB-SLAM2's TUM bf = 40 at fx ~ 517 px -> b ~ 0.077 m

    double SigmaZ(double z) const { return sigma_k_per_m * z * z; }
    /// The disparity noise on the virtual rig that reproduces SigmaZ through sigma_Z = Z^2 / (fx B) sigma_d.
    double DisparitySigmaPx(double fx) const { return sigma_k_per_m * fx * virtual_baseline_m; }
};

/// CV_8U, 255 where `depth` (CV_32F metres, 0 = no return) is in range and not on a depth edge.
inline cv::Mat DepthSupportMask(const cv::Mat& depth, const DepthSensorModel& m) {
    CV_Assert(depth.type() == CV_32F);
    cv::Mat lo, hi;
    const cv::Mat k = cv::getStructuringElement(cv::MORPH_RECT, {3, 3});
    cv::erode(depth, lo, k);    // 3x3 min: a neighbouring hole (0) masks the pixel too
    cv::dilate(depth, hi, k);   // 3x3 max
    cv::Mat in_range = (depth >= m.z_min_m) & (depth <= m.z_max_m);
    cv::Mat smooth = (lo > 0.0f) & (hi <= lo * m.max_neighbour_ratio);
    return in_range & smooth;
}

/// Virtual-stereo evidence for `points` (rectified left pixels), parallel to them.
struct VirtualStereo {
    std::vector<cv::Point2f> right;
    std::vector<uchar> valid;
    std::vector<float> z;         // NaN where !valid
    std::vector<float> sigma_z;   // NaN where !valid
};

inline VirtualStereo VirtualStereoFromDepth(const cv::Mat& depth, const std::vector<cv::Point2f>& points, double fx,
                                            const DepthSensorModel& m) {
    const cv::Mat ok = DepthSupportMask(depth, m);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    VirtualStereo v;
    v.right.reserve(points.size());
    for (const cv::Point2f& p : points) {
        const int x = cvRound(p.x), y = cvRound(p.y);
        const bool good = x >= 0 && y >= 0 && x < depth.cols && y < depth.rows && ok.at<uchar>(y, x);
        const float z = good ? depth.at<float>(y, x) : nan;
        v.right.emplace_back(good ? static_cast<float>(p.x - fx * m.virtual_baseline_m / z) : p.x, p.y);
        v.valid.push_back(good ? 1 : 0);
        v.z.push_back(z);
        v.sigma_z.push_back(good ? static_cast<float>(m.SigmaZ(z)) : nan);
    }
    return v;
}

/// The keyframe records' dense depth (§9.4 contract, the shape SgbmStereo::Compute returns) from sensor depth:
/// Z and sigma_Z where supported, the virtual rig's disparity, confidence a placeholder. For the reconstruction
/// bundle and loop closure, which consume StereoDepth.
inline StereoDepth DenseDepthFromSensor(const cv::Mat& depth, double fx, const DepthSensorModel& m) {
    StereoDepth d;
    d.valid = DepthSupportMask(depth, m);
    cv::Mat z = cv::Mat::zeros(depth.size(), CV_32F), s = cv::Mat::zeros(depth.size(), CV_32F);
    depth.copyTo(z, d.valid);
    cv::Mat z2;
    cv::multiply(z, z, z2);
    s = z2 * m.sigma_k_per_m;
    z.convertTo(d.Z, CV_16F);
    s.convertTo(d.sigma_Z, CV_16F);
    cv::Mat disp32;
    cv::divide(16.0 * fx * m.virtual_baseline_m, z, disp32, CV_32F);   // 1/16 px, as SGBM
    d.disparity = cv::Mat(depth.size(), CV_16S, cv::Scalar(-16));     // SGBM's invalid value at min_disparity 0
    disp32.convertTo(d.disparity, CV_16S);
    d.disparity.setTo(-16, ~d.valid);
    d.confidence = cv::Mat::zeros(depth.size(), CV_8U);
    d.confidence_is_placeholder = true;
    d.valid_fraction = static_cast<double>(cv::countNonZero(d.valid)) / static_cast<double>(depth.total());
    return d;
}

}  // namespace ariadne::perception
