#pragma once
// ---------------------------------------------------------------------------
// Lost-mode re-association (perception.md §4.3): relocalisation against
// the healthy part of the local map, by appearance, verified by geometry.
//
// The situation it serves (§4.0, measured on D56 v2): a lost frame still has
// hundreds of LK tracks, but they were born during the loss and carry fresh
// ids that no landmark holds, so PnP sees 2-13 correspondences where it needs
// 15. The landmarks those pixels belong to were mapped before the loss; their
// tracks died. This matches the orphan tracks back to them:
//
//   1. APPEARANCE. Each orphan track's pixel is described (the engine's
//      Describe); each candidate landmark carries the descriptor of its latest
//      keyframe observation. Brute-force nearest neighbour, Lowe's ratio test
//      on the two nearest, an absolute distance cap, and a MUTUAL check (the
//      landmark's nearest track must be the same track). ORB-SLAM T-RO 2015
//      §V-C relocalises the same way (BoW-restricted there; our candidate set
//      is small enough for brute force).
//   2. GEOMETRY. Descriptor matches alone are not trusted: PnP-RANSAC over the
//      matches PLUS the healthy correspondences the frame already holds
//      ("anchors"). A pose that explains enough of them at a few pixels is the
//      evidence; only matches that are inliers of it are kept. Seeded USAC, so a
//      replay picks the same inliers.
//
// Out: renames (orphan track id -> landmark id) for the tracker to apply. The
// pose is NOT returned: StereoVo then solves PnP itself over the renamed
// tracks, through its normal path, so there is one pose estimator.
//
// Why no projection search here: while lost the pose prior is the 1 m / 1 rad
// placeholder, so "near the projection" means nothing (ORB-SLAM switches to
// relocalisation for the same reason). Tracking-mode projection search is a
// separate step (§4.3, at keyframes).
//
// Plain arrays and OpenCV types only: no SLAM type crosses into perception.
// ---------------------------------------------------------------------------
#include "perception/perception_config.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::perception {

/// Tracks that hold no healthy landmark, described at their current pixel.
struct OrphanTracks {
    std::vector<uint64_t> track_ids;
    std::vector<cv::Point2f> points;
    cv::Mat descriptors;   // one row per track; CV_8U (binary) or CV_32F
};

/// Healthy landmarks no live track holds, with their stored descriptors.
struct CandidateLandmarks {
    std::vector<uint64_t> landmark_ids;
    std::vector<cv::Point3d> X_w;
    cv::Mat descriptors;   // one row per landmark, same type and width as the orphans'
};

/// Correspondences the frame already has (healthy landmarks still tracked).
struct AnchorCorrespondences {
    std::vector<cv::Point3d> X_w;
    std::vector<cv::Point2f> points;
};

struct ReassociationResult {
    std::vector<uint64_t> from, to;   // apply with KltTracker::Relabel(from, to)
    int candidates = 0, orphans = 0, anchors = 0;
    int matched = 0;          // appearance matches surviving ratio + cap + mutual
    int inliers = 0;          // of the PnP model, anchors + matches
    int match_inliers = 0;    // of which matches (= from.size() when accepted)
    bool ran = false;         // PnP was attempted
    bool accepted = false;
    std::string why;          // why not accepted (diagnostic)
};

/// fx, fy, cx, cy: the rectified left camera. `seed`/`frame_id` key USAC.
ReassociationResult ReassociateLost(const OrphanTracks& orphans, const CandidateLandmarks& candidates,
                                    const AnchorCorrespondences& anchors, double fx, double fy, double cx,
                                    double cy, const ReassociationParams& params, uint64_t seed,
                                    uint64_t frame_id);

/// Step 1 alone (exposed for testing): mutual ratio-test matches, query (orphan) row -> train (candidate) row,
/// ascending by query row.
std::vector<cv::DMatch> MutualRatioMatches(const cv::Mat& query, const cv::Mat& train,
                                           const ReassociationParams& params);

}  // namespace ariadne::perception
