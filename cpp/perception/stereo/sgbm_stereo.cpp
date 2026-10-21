#include "perception/stereo/sgbm_stereo.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ariadne::perception {

namespace {

int SgbmMode(const std::string& mode) {
    if (mode == "sgbm") return cv::StereoSGBM::MODE_SGBM;
    if (mode == "hh") return cv::StereoSGBM::MODE_HH;
    if (mode == "sgbm_3way") return cv::StereoSGBM::MODE_SGBM_3WAY;
    if (mode == "hh4") return cv::StereoSGBM::MODE_HH4;
    throw ConfigError("stereo: unknown mode '" + mode + "'");
}

}  // namespace

SgbmStereo::SgbmStereo(const StereoParams& params, const RectifiedStereoRig& rig)
    : params_(params), rig_(rig) {
    ValidateStereoAgainstRig(params_, rig_);
    // Grayscale input, so channels = 1 in §9.3's P1/P2 formulas.
    const int area = params_.block_size * params_.block_size;
    sgbm_ = cv::StereoSGBM::create(
        params_.min_disparity, params_.num_disparities, params_.block_size,
        static_cast<int>(params_.p1_coefficient * area),
        static_cast<int>(params_.p2_coefficient * area), params_.disp12_max_diff,
        params_.pre_filter_cap, params_.uniqueness_ratio, params_.speckle_window_size,
        params_.speckle_range, SgbmMode(params_.mode));
    if (params_.confidence == "wls") {
        right_ = cv::ximgproc::createRightMatcher(sgbm_);
        wls_ = cv::ximgproc::createDisparityWLSFilterGeneric(/*use_confidence=*/true);
        wls_->setLambda(params_.wls_lambda);
        wls_->setSigmaColor(params_.wls_sigma_color);
        wls_->setLRCthresh(params_.wls_lrc_threshold);
        wls_->setDepthDiscontinuityRadius(params_.wls_discontinuity_radius > 0
                                              ? params_.wls_discontinuity_radius
                                              : static_cast<int>(std::ceil(0.5 * params_.block_size)));
        // The ROI createDisparityWLSFilter sets for an SGBM: columns left of min+num have no full search range.
        const int left_off = std::max(0, params_.min_disparity + params_.num_disparities);
        const int right_off = std::max(0, -params_.min_disparity);
        wls_roi_ = cv::Rect(left_off, 0, rig_.width - left_off - right_off, rig_.height);
    }
}

StereoDepth SgbmStereo::Compute(const cv::Mat& left, const cv::Mat& right) const {
    const cv::Size size(rig_.width, rig_.height);
    if (left.type() != CV_8UC1 || right.type() != CV_8UC1 || left.size() != size ||
        right.size() != size) {
        throw std::invalid_argument("SgbmStereo: expected two CV_8UC1 images of the rig size");
    }
    StereoDepth out;
    sgbm_->compute(left, right, out.disparity);   // CV_16S, d * 16
    cv::Mat wls_conf;
    if (wls_) {
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat right_disp, filtered;
        right_->compute(right, left, right_disp);   // the right view's disparity (negative values)
        wls_->filter(out.disparity, left, filtered, right_disp, wls_roi_);
        wls_->getConfidenceMap().convertTo(wls_conf, CV_8U);   // CV_32F 0..255 -> u8, rounded
        if (params_.wls_filtered_disparity) out.disparity = filtered;
        out.confidence_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // Valid: the value is a real match (at least min_disparity), strictly
    // positive so Z is finite, and outside the blind columns. SGBM marks
    // failures (uniqueness, left-right check, speckle) with
    // (min_disparity - 1) * 16, which the first test excludes.
    out.blind_columns = std::min(rig_.width, params_.min_disparity + params_.num_disparities);
    const int min_raw = std::max(1, params_.min_disparity * cv::StereoMatcher::DISP_SCALE);
    const double fb = rig_.fx * rig_.baseline_m;

    cv::Mat z32(size, CV_32F, cv::Scalar(0));
    cv::Mat s32(size, CV_32F, cv::Scalar(0));
    out.valid = cv::Mat::zeros(size, CV_8U);
    int valid_count = 0;
    for (int y = 0; y < size.height; ++y) {
        const int16_t* d = out.disparity.ptr<int16_t>(y);
        float* z = z32.ptr<float>(y);
        float* s = s32.ptr<float>(y);
        uchar* v = out.valid.ptr<uchar>(y);
        for (int x = out.blind_columns; x < size.width; ++x) {
            if (d[x] < min_raw) continue;
            const double disp = static_cast<double>(d[x]) / cv::StereoMatcher::DISP_SCALE;
            const double depth = fb / disp;
            z[x] = static_cast<float>(depth);
            s[x] = static_cast<float>(DepthSigma(depth, rig_.fx, rig_.baseline_m,
                                                 params_.sigma_disparity_px));
            v[x] = 255;
            ++valid_count;
        }
    }
    z32.convertTo(out.Z, CV_16F);
    s32.convertTo(out.sigma_Z, CV_16F);
    if (wls_) {
        out.confidence = cv::Mat::zeros(size, CV_8U);
        wls_conf.copyTo(out.confidence, out.valid);   // never a confidence for an invalid depth
        out.confidence_is_placeholder = false;
    } else {
        out.confidence = cv::Mat::zeros(size, CV_8U);
        out.confidence_is_placeholder = true;
    }
    out.valid_fraction = static_cast<double>(valid_count) / (size.width * size.height);
    return out;
}

cv::Mat DenseSupportMask(const StereoDepth& depth, const StereoParams& params) {
    cv::Mat valid01, density;
    depth.valid.convertTo(valid01, CV_32F, 1.0 / 255.0);
    // Normalized box filter = fraction of valid pixels in the window. Pixels
    // outside the image count as invalid (BORDER_CONSTANT 0), so support near
    // the edges is judged conservatively.
    const cv::Size win(params.support_window_px, params.support_window_px);
    cv::boxFilter(valid01, density, CV_32F, win, cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
    cv::Mat supported = density >= params.support_min_density;   // CV_8U 0/255
    supported &= depth.valid;
    return supported;
}

}  // namespace ariadne::perception
