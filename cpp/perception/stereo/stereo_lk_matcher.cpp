#include "perception/stereo/stereo_lk_matcher.hpp"

#include <opencv2/video/tracking.hpp>

#include <cmath>

namespace ariadne::perception {

StereoMatches MatchLeftToRight(const cv::Mat& left, const cv::Mat& right,
                               const std::vector<cv::Point2f>& points, const StereoMatchParams& p) {
    StereoMatches m;
    m.right.assign(points.size(), cv::Point2f(0, 0));
    m.valid.assign(points.size(), 0);
    if (points.empty()) return m;
    const cv::Size win(p.win_size, p.win_size);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<cv::Point2f> fwd = points, back = points;
    std::vector<uchar> st, st_back;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(left, right, points, fwd, st, err, win, p.max_level, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(right, left, fwd, back, st_back, err, win, p.max_level, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t i = 0; i < points.size(); ++i) {
        if (!st[i] || !st_back[i]) { ++m.lk_lost; continue; }
        if (cv::norm(back[i] - points[i]) > p.fb_threshold_px) { ++m.fb_rejected; continue; }
        if (std::abs(fwd[i].y - points[i].y) > p.max_dy_px) { ++m.row_rejected; continue; }
        if (!(points[i].x - fwd[i].x > 0.0f) || fwd[i].x < 0.0f || fwd[i].x > right.cols - 1) {
            ++m.nonpositive;
            continue;
        }
        m.right[i] = fwd[i];
        m.valid[i] = 1;
    }
    return m;
}

}  // namespace ariadne::perception
