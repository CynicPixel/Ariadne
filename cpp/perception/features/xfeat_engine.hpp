#pragma once
// ---------------------------------------------------------------------------
// XFeat as a §7.1 LocalFeatureEngine (perception.md §4.1), over the D50
// network (xfeat/xfeat_net.hpp: our cv::dnn graph, upstream's weights).
//
// ONE FORWARD PER IMAGE. The network gives dense maps (keypoint logits,
// reliability, 64-D descriptors at 1/8 resolution); keypoints and descriptors
// are both READ from them. A keyframe can call Detect (tracker replenish),
// Describe (re-association) and DetectAndCompute on the same image, so the
// last image's maps are cached: an exact pixel comparison, not a pointer test,
// decides a hit (a reused buffer with new content is a miss).
//
// SELECTION (§7.3). Upstream keeps the global top-k by score. Here every NMS
// survivor with score > 0 that the mask allows becomes a candidate (the mask
// applies BEFORE selection, so masked areas do not spend the budget), then
// GridSelect spreads the budget over the grid, as the ORB engine does.
//
// INPUT SIZE. The network needs sides that are multiples of 32.
//   resize  bilinear in float (upstream preprocess_tensor: F.interpolate,
//           align_corners=False, which cv::resize INTER_LINEAR matches on a
//           float image); keypoints scale back by W/Wn, H/Hn exactly as
//           upstream does (x * rw, not the half-pixel-exact (x+0.5)*rw-0.5:
//           <= 0.011 px at 752 -> 736). 0 x 0 = floor each side to 32.
//   pad     replicate-pad right/bottom to the next multiple of 32; keypoints
//           in the pad are dropped. Scale 1.
// Which one is a measured D6x decision (§4.1); both are here to measure.
//
// NOT THREAD-SAFE (§7.4): the net and the cache are mutable behind const.
// One engine per thread.
// ---------------------------------------------------------------------------
#include "perception/features/feature_engine.hpp"
#include "perception/features/xfeat/xfeat_net.hpp"
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <vector>

namespace ariadne::perception {

class XFeatEngine final : public LocalFeatureEngine {
public:
    explicit XFeatEngine(const FeatureParams& params);

    std::string Name() const override { return "xfeat"; }
    std::vector<cv::KeyPoint> Detect(const cv::Mat& gray, const cv::Mat& mask,
                                     int max_keypoints) const override;
    FeatureSet DetectAndCompute(const cv::Mat& gray, const cv::Mat& mask) const override;
    /// Samples the dense descriptor map at each point (upstream's sampler): no detection, any sub-pixel position.
    /// described[i] = 0 only outside the image.
    cv::Mat Describe(const cv::Mat& gray, const std::vector<cv::Point2f>& points,
                     std::vector<uchar>& described) const override;

    std::map<std::string, std::string> ManifestFields() const override;
    std::string ResultsYaml() const override;

    /// The network input size used for an image of `image` size.
    cv::Size NetSize(cv::Size image) const;
    /// Forward passes actually run, and cache hits (calls that reused them).
    long forwards() const { return static_cast<long>(forward_ms_.size()); }
    long cache_hits() const { return cache_hits_; }

private:
    struct Maps {
        xfeat::NetOutput out;
        cv::Size image, net;
        float sx = 1.f, sy = 1.f;   // image = net * (sx, sy)
    };
    const Maps& Run(const cv::Mat& gray) const;
    /// Candidates in IMAGE pixels, mask applied; class_id = index into `net_pos` (their network-input pixel).
    std::vector<cv::KeyPoint> Select(const Maps& m, const cv::Mat& mask, int budget,
                                     std::vector<cv::Point2f>& net_pos) const;

    FeatureParams params_;
    xfeat::XFeatNet net_;
    mutable cv::Mat cached_input_;
    mutable Maps cached_;
    mutable long cache_hits_ = 0;
    mutable std::vector<double> forward_ms_;   // wall clock per forward (preprocessing + network)
};

}  // namespace ariadne::perception
