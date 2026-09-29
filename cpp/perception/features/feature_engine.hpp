#pragma once
// ---------------------------------------------------------------------------
// The local-feature contract (arch §7.1). Every extractor (ORB now; XFeat,
// ALIKED, DISK, SIFT later) implements it, so swapping one is a config change
// and every ablation runs through identical downstream code.
//
// Two entry points rather than §7.1's single detectAndCompute: the tracker
// (§8) only needs POSITIONS to replenish tracks every frame, and computing
// descriptors there would waste most of the per-frame budget. Keyframes call
// DetectAndCompute. An engine may implement Detect as DetectAndCompute minus the
// descriptors; the split exists for callers, not for engines.
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace ariadne::perception {

struct FeatureSet {
    std::vector<cv::KeyPoint> keypoints;
    /// One row per keypoint. CV_8U for binary descriptors (ORB: 32 bytes),
    /// CV_32F for learned ones (XFeat: 64 floats).
    cv::Mat descriptors;
};

class LocalFeatureEngine {
public:
    virtual ~LocalFeatureEngine() = default;

    /// Stable id recorded in the run manifest's feature_engine field.
    virtual std::string Name() const = 0;

    /// Keypoints only. `mask` (CV_8U, nonzero = allowed) may be empty.
    /// At most `max_keypoints`; <= 0 means the engine's configured cap.
    virtual std::vector<cv::KeyPoint> Detect(const cv::Mat& gray, const cv::Mat& mask,
                                             int max_keypoints) const = 0;

    /// Keypoints and descriptors, rows aligned: descriptors.rows == keypoints.size().
    virtual FeatureSet DetectAndCompute(const cv::Mat& gray, const cv::Mat& mask) const = 0;
};

}  // namespace ariadne::perception
