#pragma once
// ---------------------------------------------------------------------------
// Sparse left->right stereo correspondence for tracked points: pyramidal LK
// from the rectified left image into the rectified right image (initial guess:
// same pixel), tracked back, kept when the round trip closes within
// fb_threshold_px (the VINS-Fusion idiom), the match stays on the same row
// within max_dy_px (ours: VINS-Fusion has no row test), and the disparity is
// positive.
//
// Validated on EuRoC MH_01 (D35): median |dy| 0.135 px over 96.7k points,
// against 13.1 px on the raw pair and 8.1 px under a 1-degree pitch error.
// Feeds landmark triangulation (§11.2) at keyframes.
//
// The row test is only as good as the rectification. On EuRoC V2 the published
// calibration leaves |dy| ~ 1.7 px (local-slam.md#window-audit): 97% of correct matches fail
// it. EstimateRowCorrection measures that residual (dy = a + b x + c y over
// round-trip-consistent matches, no row test), and the rectifier folds the
// correction into the right map (StereoRectifier::RefineRightRows), so every
// consumer -- this matcher, SGBM, triangulation, BA -- sees aligned rows.
// MatchExposure removes a left/right gain/bias mismatch first (EuRoC's
// per-camera auto-exposure; V2_03's right camera is ~36% darker), which LK's
// brightness-constancy assumption cannot absorb.
// ---------------------------------------------------------------------------
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ariadne::perception {

struct StereoMatches {
    std::vector<cv::Point2f> right;
    std::vector<uchar> valid;
    std::vector<uchar> reason;   // per point: 0 valid, 1 LK lost, 2 forward-backward, 3 off the row, 4 disparity <= 0
    int lk_lost = 0, fb_rejected = 0, row_rejected = 0, nonpositive = 0;
};

StereoMatches MatchLeftToRight(const cv::Mat& left, const cv::Mat& right,
                               const std::vector<cv::Point2f>& points, const StereoMatchParams& params);

/// Residual vertical misalignment of a rectified pair, dy = y_right - y_left = a + b x + c y, in RIGHT image pixel
/// coordinates (so the correction y' = y - (a + b x + c y) is exact in the image it is applied to).
struct RowCorrection {
    bool ok = false;
    double a = 0.0, b = 0.0, c = 0.0;   // px, px/px, px/px
    int matches = 0;                    // round-trip-consistent matches used after trimming
    double mad_px = 0.0;                // median absolute deviation of the fit residual
    bool slopes_fitted = false;         // false: the matches spanned too little of the image, only `a` was fitted
    std::string why;                    // when !ok: the reason (loud, for the log and the manifest)
};

/// Fit the residual from LK matches WITHOUT the row test (round trip <= fb_threshold_px, |dy| <= row_search_px,
/// positive disparity), by least squares trimmed at 3 x 1.4826 MAD, twice. Refuses (ok = false, why set) on fewer
/// than row_min_matches matches, or when the fitted model predicts |dy| > row_search_px anywhere in the image (the
/// search bound would have cut it: the estimate cannot be trusted).
RowCorrection EstimateRowCorrection(const cv::Mat& left, const cv::Mat& right,
                                    const std::vector<cv::Point2f>& points, const StereoMatchParams& params);

/// The same estimate POOLED over several pairs: the residual is a fixed property of the rig, so samples from
/// successive keyframes add up (a dim or sparse first keyframe would otherwise delay the estimate while the flight
/// stays lost -- V2_03 waited until keyframe 52 single-frame). x, y in right image pixels, dy = y_right - y_left.
struct RowSamples {
    std::vector<double> x, y, dy;
    int width = 0, height = 0;
};
void CollectRowSamples(const cv::Mat& left, const cv::Mat& right, const std::vector<cv::Point2f>& points,
                       const StereoMatchParams& params, RowSamples& into);
RowCorrection FitRowCorrection(const RowSamples& samples, const StereoMatchParams& params);

/// The right image with its global gain and bias matched to the left (mean and standard deviation over pixels that
/// are non-zero in both, i.e. inside both rectified footprints), saturated to 8 bits.
cv::Mat MatchExposure(const cv::Mat& left, const cv::Mat& right);

}  // namespace ariadne::perception
