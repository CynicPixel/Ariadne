#include "perception/stereo/stereo_lk_matcher.hpp"

#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

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

namespace {

double Median(std::vector<double> v) {
    const auto mid = v.begin() + long(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

}  // namespace

void CollectRowSamples(const cv::Mat& left, const cv::Mat& right, const std::vector<cv::Point2f>& points,
                       const StereoMatchParams& p, RowSamples& into) {
    if (into.width == 0) {
        into.width = right.cols;
        into.height = right.rows;
    } else if (into.width != right.cols || into.height != right.rows) {
        throw std::invalid_argument("CollectRowSamples: image size changed between keyframes");
    }
    if (points.empty()) return;
    const cv::Size win(p.win_size, p.win_size);
    const cv::TermCriteria crit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    std::vector<cv::Point2f> fwd = points, back = points;
    std::vector<uchar> st, st_back;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(left, right, points, fwd, st, err, win, p.max_level, crit, cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(right, left, fwd, back, st_back, err, win, p.max_level, crit,
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t i = 0; i < points.size(); ++i) {
        if (!st[i] || !st_back[i] || cv::norm(back[i] - points[i]) > p.fb_threshold_px) continue;
        const double dy = double(fwd[i].y) - double(points[i].y);
        if (std::abs(dy) > p.row_search_px || !(points[i].x - fwd[i].x > 0.0f) || fwd[i].x < 0.0f ||
            fwd[i].x > right.cols - 1) {
            continue;
        }
        into.x.push_back(fwd[i].x);
        into.y.push_back(fwd[i].y);
        into.dy.push_back(dy);
    }
}

RowCorrection EstimateRowCorrection(const cv::Mat& left, const cv::Mat& right,
                                    const std::vector<cv::Point2f>& points, const StereoMatchParams& p) {
    RowSamples s;
    CollectRowSamples(left, right, points, p, s);
    return FitRowCorrection(s, p);
}

RowCorrection FitRowCorrection(const RowSamples& samples, const StereoMatchParams& p) {
    RowCorrection out;
    const std::vector<double>& xs = samples.x;
    const std::vector<double>& ys = samples.y;
    const std::vector<double>& dys = samples.dy;
    if (dys.empty()) {
        out.why = "no round-trip-consistent matches";
        return out;
    }
    // Slopes only when the matches span enough of the image to determine them.
    const auto span = [](const std::vector<double>& v) {
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()) - *std::min_element(v.begin(), v.end());
    };
    out.slopes_fitted = span(ys) >= p.row_min_span * samples.height && span(xs) >= p.row_min_span * samples.width;
    std::vector<char> use(dys.size(), 1);
    cv::Vec3d coef(0, 0, 0);
    for (int round = 0; round < 3; ++round) {
        const int cols = out.slopes_fitted ? 3 : 1;
        cv::Mat A(0, cols, CV_64F), y(0, 1, CV_64F);
        for (size_t i = 0; i < dys.size(); ++i) {
            if (!use[i]) continue;
            const double row3[3] = {1.0, xs[i], ys[i]};
            A.push_back(cv::Mat(1, cols, CV_64F, const_cast<double*>(row3)).clone());
            y.push_back(dys[i]);
        }
        out.matches = A.rows;
        if (A.rows < p.row_min_matches) {
            std::ostringstream m;
            m << "only " << A.rows << " round-trip-consistent matches (< row_min_matches " << p.row_min_matches << ")";
            out.why = m.str();
            return out;
        }
        cv::Mat sol;
        cv::solve(A, y, sol, cv::DECOMP_SVD);
        coef = cv::Vec3d(sol.at<double>(0), cols == 3 ? sol.at<double>(1) : 0.0, cols == 3 ? sol.at<double>(2) : 0.0);
        std::vector<double> res, absdev;
        for (size_t i = 0; i < dys.size(); ++i)
            if (use[i]) res.push_back(dys[i] - (coef[0] + coef[1] * xs[i] + coef[2] * ys[i]));
        const double med = Median(res);
        for (double r : res) absdev.push_back(std::abs(r - med));
        out.mad_px = Median(absdev);
        if (round == 2) break;
        const double cut = 3.0 * 1.4826 * std::max(out.mad_px, 1e-3);
        for (size_t i = 0; i < dys.size(); ++i)
            use[i] = std::abs(dys[i] - (coef[0] + coef[1] * xs[i] + coef[2] * ys[i]) - med) <= cut;
    }
    out.a = coef[0];
    out.b = coef[1];
    out.c = coef[2];
    double worst = 0.0;
    for (double x : {0.0, double(samples.width - 1)})
        for (double yy : {0.0, double(samples.height - 1)}) worst = std::max(worst, std::abs(out.a + out.b * x + out.c * yy));
    if (worst > p.row_search_px) {
        std::ostringstream m;
        m << "model predicts |dy| " << worst << " px at an image corner, beyond row_search_px " << p.row_search_px;
        out.why = m.str();
        return out;
    }
    out.ok = true;
    return out;
}

cv::Mat MatchExposure(const cv::Mat& left, const cv::Mat& right) {
    if (left.size() != right.size() || left.type() != CV_8UC1 || right.type() != CV_8UC1) {
        throw std::invalid_argument("MatchExposure: two CV_8UC1 images of one size");
    }
    const cv::Mat both = (left > 0) & (right > 0);
    cv::Scalar ml, sl, mr, sr;
    cv::meanStdDev(left, ml, sl, both);
    cv::meanStdDev(right, mr, sr, both);
    if (!(sr[0] > 1e-6)) return right.clone();   // a flat right image: nothing to match
    const double gain = sl[0] / sr[0];
    cv::Mat out;
    right.convertTo(out, CV_8U, gain, ml[0] - gain * mr[0]);   // saturate_cast to [0, 255]
    out.setTo(0, right == 0);                                    // outside the rectified footprint stays black
    return out;
}

}  // namespace ariadne::perception
