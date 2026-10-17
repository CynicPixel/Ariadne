#pragma once
// ---------------------------------------------------------------------------
// Frame-to-frame tracking with pyramidal Lucas-Kanade (arch §8.1-8.2).
//
// LUCAS-KANADE in one paragraph: assume a small window around a point keeps
// its appearance between frames and moves by a displacement d. Linearising the
// image around the point gives a 2x2 linear system G d = b, where G is the
// window's gradient structure tensor (sums of Ix^2, IxIy, Iy^2). It is solvable
// only where G is well conditioned: both eigenvalues large, i.e. a corner, not
// an edge or flat wall (Shi & Tomasi 1994: "good features to track" are exactly
// the points where this system is well posed; min_eig_threshold enforces it).
// The linearisation holds for motions of about a pixel, so Bouguet's PYRAMIDAL
// version solves it coarse-to-fine: estimate at 1/8 resolution, propagate the
// estimate down, refine, and so on to full resolution.
//
// §8.1's split, followed literally. OpenCV does: pyramid construction
// (buildOpticalFlowPyramid), LK (calcOpticalFlowPyrLK), robust epipolar fit
// (findFundamentalMat). Ariadne owns: the forward-backward check, border
// check, epipolar outlier decision, track identity and lifecycle, replenishment
// and the DEGRADED state. Pattern as in VINS-Fusion's feature_tracker and
// OpenVINS's TrackKLT.
//
// Each frame's pyramid is built ONCE and reused for the forward pass, the
// backward pass, and the next frame's forward pass. Pyramid construction is
// also its own timed stage in the COOL benchmark workload (Iteration 14).
//
// NOT HERE: pose estimation, keyframe policy (§8.3) and IMU-predicted search
// windows (§10.5). Those belong to SLAM (plan 1.5, 2.x).
// ---------------------------------------------------------------------------
#include "perception/features/feature_engine.hpp"
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace ariadne::perception {

enum class TrackingState {
    kInitializing,   // first frame: nothing to track from yet
    kTracking,
    kDegraded,       // §8.2 "track support collapses": too few surviving tracks
};

/// Where each track that existed in the previous frame went. Counts are per
/// frame, so a report can show which filter is doing the rejecting.
struct TrackStats {
    int attempted = 0;          // tracks carried in from the previous frame
    int lk_lost = 0;            // LK failed (status 0) forward or backward
    int fb_rejected = 0;        // forward-backward round trip > threshold
    int border_rejected = 0;
    int epipolar_rejected = 0;  // outliers of the fundamental-matrix fit
    int crowded_out = 0;        // within min distance of an older track
    int replenished = 0;        // new tracks started this frame
    bool epipolar_check_ran = false;
    int usac_defect_retries = 0;       // see EpipolarResult
    bool usac_defect_skipped = false;
};

/// The tracker's output for one frame. Parallel arrays, one entry per live
/// track. age == 0 marks a track started this frame; for those,
/// prev_points[i] == points[i] and there is no correspondence yet.
struct TrackedFrame {
    uint64_t frame_id = 0;
    TrackingState state = TrackingState::kInitializing;
    std::vector<uint64_t> ids;
    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> prev_points;
    std::vector<int> ages;
    TrackStats stats;
};

/// A pyramid as calcOpticalFlowPyrLK consumes it. buildOpticalFlowPyramid may
/// build FEWER levels than requested (it stops when a level gets smaller than
/// the LK window), so the level count travels with the images.
struct LkPyramid {
    std::vector<cv::Mat> images;
    int max_level = 0;
    bool empty() const { return images.empty(); }
};

/// Epipolar outlier test (step 2 of the tracker, exposed for testing on
/// non-planar geometry: a fundamental matrix is degenerate when every point
/// lies on one plane, so whole-image shift tests cannot exercise it).
/// `inlier` is parallel to the inputs; `ran` is false when there were too few
/// points or the fit was degenerate, in which case nothing is rejected.
struct EpipolarResult {
    std::vector<uchar> inlier;
    bool ran = false;
    int usac_defect_retries = 0;       // OpenCV 5.0.0 USAC empty-model throws absorbed by a redraw (SMOKE §4q)
    bool usac_defect_skipped = false;  // every draw threw: no epipolar check this frame
};
EpipolarResult EpipolarInliers(const std::vector<cv::Point2f>& p0,
                               const std::vector<cv::Point2f>& p1,
                               const TrackerParams& params, uint64_t seed, uint64_t frame_id);

class KltTracker {
public:
    /// `replenisher` is borrowed and must outlive the tracker. New tracks come
    /// from the §7.1 engine rather than a tracker-private detector, so the
    /// XFeat swap changes one engine for both keyframes and tracking (§7.2).
    KltTracker(const TrackerParams& params, const LocalFeatureEngine& replenisher);

    /// Track into `gray` (CV_8UC1). `seed` and `frame_id` key the robust
    /// estimator's random sampling, so a replay reproduces it exactly.
    TrackedFrame Process(const cv::Mat& gray, uint64_t frame_id, uint64_t seed);

    /// The pyramid LK consumes, exposed so it can be timed on its own.
    static LkPyramid BuildPyramid(const cv::Mat& gray, const TrackerParams& params);

private:
    TrackerParams params_;
    const LocalFeatureEngine& replenisher_;
    LkPyramid prev_pyramid_;
    std::vector<uint64_t> ids_;
    std::vector<cv::Point2f> points_;
    std::vector<int> ages_;
    uint64_t next_id_ = 0;
};

}  // namespace ariadne::perception
