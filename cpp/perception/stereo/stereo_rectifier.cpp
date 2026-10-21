#include "perception/stereo/stereo_rectifier.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include <cmath>
#include <stdexcept>

namespace ariadne::perception {

namespace {

cv::Mat K(const PinholeRadtan& c) {
    return (cv::Mat_<double>(3, 3) << c.fu, 0, c.cu, 0, c.fv, c.cv, 0, 0, 1);
}

cv::Mat D(const PinholeRadtan& c) {
    return (cv::Mat_<double>(1, 4) << c.dist[0], c.dist[1], c.dist[2], c.dist[3]);
}

}  // namespace

StereoRectifier::StereoRectifier(const RawStereoCalibration& calib, double alpha) {
    if (calib.left.width != calib.right.width || calib.left.height != calib.right.height) {
        throw std::invalid_argument("StereoRectifier: left and right image sizes differ");
    }
    if (!(alpha >= 0.0 && alpha <= 1.0)) throw std::invalid_argument("StereoRectifier: alpha in [0, 1]");
    size_ = cv::Size(calib.left.width, calib.left.height);

    cv::Mat R(3, 3, CV_64F), T(3, 1, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) R.at<double>(r, c) = calib.T_right_left.R(r, c);
        T.at<double>(r) = calib.T_right_left.t(r);
    }
    const cv::Mat K1 = K(calib.left), K2 = K(calib.right);
    const cv::Mat D1 = D(calib.left), D2 = D(calib.right);
    cv::stereoRectify(K1, D1, K2, D2, size_, R, T, R1_, R2_, P1_, P2_, Q_,
                      cv::STEREO_ZERO_DISPARITY, alpha, size_);

    // A horizontal rig puts the baseline in P2(0,3) = -fx * B and nothing in
    // P2(1,3). A vertical result means the rig or the transform direction is
    // wrong; fail here rather than feed SGBM columns that are really rows.
    const double fx = P1_.at<double>(0, 0);
    const double baseline = -P2_.at<double>(0, 3) / P2_.at<double>(0, 0);
    if (std::abs(P2_.at<double>(1, 3)) > 1e-6 * std::abs(P2_.at<double>(0, 3)) || !(baseline > 0.0)) {
        throw std::invalid_argument(
            "StereoRectifier: not a left-right rig after rectification (check T_right_left direction)");
    }
    rig_.width = size_.width;
    rig_.height = size_.height;
    rig_.fx = fx;
    rig_.fy = P1_.at<double>(1, 1);
    rig_.cx = P1_.at<double>(0, 2);
    rig_.cy = P1_.at<double>(1, 2);
    rig_.baseline_m = baseline;

    cv::initUndistortRectifyMap(K1, D1, R1_, P1_(cv::Rect(0, 0, 3, 3)), size_, CV_16SC2,
                                map_left_[0], map_left_[1]);
    cv::initUndistortRectifyMap(K2, D2, R2_, P2_(cv::Rect(0, 0, 3, 3)), size_, CV_16SC2,
                                map_right_[0], map_right_[1]);
    K2_ = K2.clone();
    D2_ = D2.clone();
}

void StereoRectifier::RefineRightRows(double a, double b, double c) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !(std::abs(c) < 0.5)) {
        throw std::invalid_argument("StereoRectifier::RefineRightRows: non-finite or implausible correction");
    }
    const cv::Matx33d A(1.0, 0.0, 0.0, -b, 1.0 - c, -a, 0.0, 0.0, 1.0);
    const cv::Mat P2k = P2_(cv::Rect(0, 0, 3, 3));
    const cv::Mat newK = cv::Mat(A) * P2k;
    cv::initUndistortRectifyMap(K2_, D2_, R2_, newK, size_, CV_16SC2, map_right_[0], map_right_[1]);
    row_a_ = a;
    row_b_ = b;
    row_c_ = c;
}

void StereoRectifier::Rectify(const cv::Mat& left, const cv::Mat& right, cv::Mat& left_out,
                              cv::Mat& right_out) const {
    if (left.size() != size_ || right.size() != size_) {
        throw std::invalid_argument("StereoRectifier: image size does not match the calibration");
    }
    // BORDER_CONSTANT 0: with alpha = 0 almost nothing maps outside the
    // source, and what does is black rather than smeared edge pixels, which
    // would look like texture to the detector.
    cv::remap(left, left_out, map_left_[0], map_left_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::remap(right, right_out, map_right_[0], map_right_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

void StereoRectifier::RectifyLeft(const cv::Mat& left, cv::Mat& left_out) const {
    if (left.size() != size_) throw std::invalid_argument("StereoRectifier: image size does not match the calibration");
    cv::remap(left, left_out, map_left_[0], map_left_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

void StereoRectifier::RectifyLeftNearest(const cv::Mat& left, cv::Mat& left_out) const {
    if (left.size() != size_) throw std::invalid_argument("StereoRectifier: image size does not match the calibration");
    cv::remap(left, left_out, map_left_[0], map_left_[1], cv::INTER_NEAREST, cv::BORDER_CONSTANT);
}

}  // namespace ariadne::perception
