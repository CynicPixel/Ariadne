#include "perception/features/orb_engine.hpp"

#include "perception/features/grid_select.hpp"

#include <stdexcept>

namespace ariadne::perception {

namespace {

cv::Ptr<cv::ORB> MakeOrb(const FeatureParams& p, int fast_threshold) {
    // edgeThreshold == patchSize: the border ORB skips must cover the patch
    // the descriptor samples, or compute() silently drops border keypoints.
    return cv::ORB::create(p.max_keypoints * p.detect_oversample,
                           static_cast<float>(p.orb_scale_factor), p.orb_levels,
                           /*edgeThreshold=*/p.orb_patch_size, /*firstLevel=*/0,
                           /*WTA_K=*/2, cv::ORB::HARRIS_SCORE, p.orb_patch_size,
                           fast_threshold);
}

}  // namespace

OrbEngine::OrbEngine(const FeatureParams& params)
    : params_(params),
      orb_(MakeOrb(params, params.orb_fast_threshold)),
      orb_low_(MakeOrb(params, params.orb_fast_threshold_min)) {}

std::vector<cv::KeyPoint> OrbEngine::Detect(const cv::Mat& gray, const cv::Mat& mask,
                                            int max_keypoints) const {
    if (gray.empty()) return {};
    if (gray.type() != CV_8UC1) throw std::invalid_argument("OrbEngine: expected CV_8UC1");
    const int budget = max_keypoints > 0 ? max_keypoints : params_.max_keypoints;
    const int cols = params_.grid_cols;
    const int rows = params_.grid_rows;
    const cv::Size size = gray.size();

    std::vector<cv::KeyPoint> kps;
    orb_->detect(gray, kps, mask);

    // A cell is starved when it cannot fill its share of the budget.
    const int quota = std::max(1, budget / (cols * rows));
    std::vector<int> count(static_cast<size_t>(cols * rows), 0);
    for (const auto& k : kps) ++count[static_cast<size_t>(GridCell(k.pt, size, cols, rows))];

    cv::Mat low_mask = cv::Mat::zeros(size, CV_8U);
    bool any_starved = false;
    for (int cy = 0; cy < rows; ++cy) {
        for (int cx = 0; cx < cols; ++cx) {
            if (count[static_cast<size_t>(cy * cols + cx)] >= quota) continue;
            any_starved = true;
            // Same integer partition GridCell uses, so the rectangles tile the
            // image exactly and every pixel belongs to one cell.
            const int x0 = cx * size.width / cols, x1 = (cx + 1) * size.width / cols;
            const int y0 = cy * size.height / rows, y1 = (cy + 1) * size.height / rows;
            low_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)).setTo(255);
        }
    }
    if (any_starved) {
        if (!mask.empty()) cv::bitwise_and(low_mask, mask, low_mask);
        std::vector<cv::KeyPoint> low;
        orb_low_->detect(gray, low, low_mask);
        // The low-threshold pass re-finds the strong corners in those cells
        // too, so it REPLACES their normal-threshold keypoints rather than
        // adding to them (which would duplicate points).
        std::vector<cv::KeyPoint> merged;
        merged.reserve(kps.size() + low.size());
        for (const auto& k : kps) {
            if (count[static_cast<size_t>(GridCell(k.pt, size, cols, rows))] >= quota) {
                merged.push_back(k);
            }
        }
        for (const auto& k : low) {
            if (count[static_cast<size_t>(GridCell(k.pt, size, cols, rows))] < quota) {
                merged.push_back(k);
            }
        }
        kps = std::move(merged);
    }
    return GridSelect(std::move(kps), size, cols, rows, budget);
}

FeatureSet OrbEngine::DetectAndCompute(const cv::Mat& gray, const cv::Mat& mask) const {
    FeatureSet out;
    out.keypoints = Detect(gray, mask, params_.max_keypoints);
    if (out.keypoints.empty()) return out;
    // compute() may drop keypoints whose patch leaves the image, and rewrites
    // the vector in place: descriptors stay row-aligned with what is returned.
    orb_->compute(gray, out.keypoints, out.descriptors);
    if (out.descriptors.rows != static_cast<int>(out.keypoints.size())) {
        throw std::logic_error("OrbEngine: descriptor rows do not match keypoints");
    }
    return out;
}

cv::Mat OrbEngine::Describe(const cv::Mat& gray, const std::vector<cv::Point2f>& points,
                            std::vector<uchar>& described) const {
    if (gray.type() != CV_8UC1) throw std::invalid_argument("OrbEngine: expected CV_8UC1");
    described.assign(points.size(), 0);
    cv::Mat out = cv::Mat::zeros(static_cast<int>(points.size()), 32, CV_8U);
    if (points.empty()) return out;
    // Provided keypoints get NO orientation from compute() (OpenCV 5.0.0 orb.cpp: ICAngles runs only when detecting)
    // and are dropped within edgeThreshold of the border. So level 0, angle 0: every Describe() descriptor is
    // unrotated BRIEF. Both sides of a re-association match are described this way, so they stay comparable; rotation
    // invariance is not needed to relocalise within a 10-keyframe window. class_id carries the input index through
    // the border filter.
    std::vector<cv::KeyPoint> kps;
    kps.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        kps.emplace_back(points[i], static_cast<float>(params_.orb_patch_size), 0.f, 0.f, 0, static_cast<int>(i));
    }
    cv::Mat d;
    orb_->compute(gray, kps, d);
    if (d.rows != static_cast<int>(kps.size())) throw std::logic_error("OrbEngine: descriptor rows do not match");
    for (int r = 0; r < d.rows; ++r) {
        const int i = kps[static_cast<size_t>(r)].class_id;
        d.row(r).copyTo(out.row(i));
        described[static_cast<size_t>(i)] = 1;
    }
    return out;
}

}  // namespace ariadne::perception
