#pragma once
// ---------------------------------------------------------------------------
// Sparse left->right stereo correspondence for tracked points, the VINS-Fusion
// stereo idiom: pyramidal LK from the rectified left image into the rectified
// right image (initial guess: same pixel), tracked back, kept when the round
// trip closes within fb_threshold_px, the match stays on the same row within
// max_dy_px (rectification), and the disparity is positive.
//
// Validated on EuRoC MH_01 (D35): median |dy| 0.135 px over 96.7k points,
// against 13.1 px on the raw pair and 8.1 px under a 1-degree pitch error.
// Feeds landmark triangulation (§11.2) at keyframes.
// ---------------------------------------------------------------------------
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace ariadne::perception {

struct StereoMatches {
    std::vector<cv::Point2f> right;
    std::vector<uchar> valid;
    int lk_lost = 0, fb_rejected = 0, row_rejected = 0, nonpositive = 0;
};

StereoMatches MatchLeftToRight(const cv::Mat& left, const cv::Mat& right,
                               const std::vector<cv::Point2f>& points, const StereoMatchParams& params);

}  // namespace ariadne::perception
