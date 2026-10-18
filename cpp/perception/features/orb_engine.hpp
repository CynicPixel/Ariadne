#pragma once
// ---------------------------------------------------------------------------
// ORB (Rublee et al., ICCV 2011): FAST corners on an image pyramid, oriented by
// intensity centroid, described by a 256-bit rotated BRIEF. The Phase 1
// placeholder engine (plan 1.4); it becomes an ablation baseline when XFeat
// lands (arch §7.1 lists ORBEngine among the engines kept).
//
// Built from OpenCV primitives, selection owned by Ariadne:
//   cv::ORB::detect (normal FAST threshold, oversampled)
//   -> cells left starved are re-detected at the LOWER threshold (ORB-SLAM's
//      iniThFAST/minThFAST idea, done with ORB's own mask argument), because
//      a dark or texture-poor region yields no corners at the normal threshold
//      and would otherwise stay empty
//   -> GridSelect: top-k per cell (§7.3)
//   -> cv::ORB::compute on exactly the selected keypoints.
// ORB's internal nfeatures cap is NOT the selection: it keeps the global top-N,
// which §7.3 rules out. It is set to cap * oversample only to bound the work.
// ---------------------------------------------------------------------------
#include "perception/features/feature_engine.hpp"
#include "perception/perception_config.hpp"

#include <opencv2/features.hpp>

namespace ariadne::perception {

class OrbEngine final : public LocalFeatureEngine {
public:
    explicit OrbEngine(const FeatureParams& params);

    std::string Name() const override { return "orb"; }
    std::vector<cv::KeyPoint> Detect(const cv::Mat& gray, const cv::Mat& mask,
                                     int max_keypoints) const override;
    FeatureSet DetectAndCompute(const cv::Mat& gray, const cv::Mat& mask) const override;
    cv::Mat Describe(const cv::Mat& gray, const std::vector<cv::Point2f>& points,
                     std::vector<uchar>& described) const override;

private:
    FeatureParams params_;
    cv::Ptr<cv::ORB> orb_;       // normal FAST threshold; also computes descriptors
    cv::Ptr<cv::ORB> orb_low_;   // lower FAST threshold, for starved cells only
};

}  // namespace ariadne::perception
