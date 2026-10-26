#pragma once
// ---------------------------------------------------------------------------
// Visual rotation bridge (local-slam.md#lost-frames, D56): the camera's rotation between two consecutive frames, measured from
// the 2-D tracks alone -- no landmarks, no depth.
//
// Why it exists. When PnP fails (too few tracked LANDMARKS: dark stereo, blur), the LK tracker usually still holds
// 40-360 2-D tracks (vo_trace.csv on MH_04/MH_05/V1_03/V2_03). Two-view geometry on those tracks observes the
// rotation exactly and the translation only up to scale, so the rotation, which is what a lost stretch corrupts
// worst, stays measured while the map cannot be used.
//
// Two models:
//   E  essential matrix + cheirality (recoverPose). The general model: its rotation is unbiased under any
//      translation. DEGENERATE when the translation vanishes (hover, far scene), where its rotation can be off by
//      ~180 degrees (MH_04, MH_05: measured).
//   H  homography on normalised coordinates, projected to SO(3). Exact for a pure rotation (x1 ~ R x0 for every
//      point) and well-defined in a hover; under translation it absorbs part of it into the "rotation" (biased).
// Selection, ORB-SLAM's initialiser rule as published (Mur-Artal et al., T-RO 2015, §IV-B): H when
// R_H = S_H / (S_H + S_F) > 0.45, with S_M the chi^2-truncated transfer-error scores (rotation_bridge.cpp). H wins at
// small parallax, where its translation bias is small in ABSOLUTE terms; E wins once parallax is real. This holds
// because the caller measures from an ANCHOR frame (stereo_vo.cpp): the error at any frame is ONE two-view estimate,
// so a small H bias never accumulates. Rejected on the way (local-slam.md#lost-frames §4): counting inliers instead of scoring
// (blind to sub-pixel parallax: 0.1 deg/frame H bias, chained); "E unless it disagrees with H by > 5 deg" (fails at
// large baselines, where H's bias itself exceeds 5 deg: 20 deg jumps); ReconstructF's ">= 90% in front" (recoverPose
// ignores points beyond 50 baselines: all of them at 1-2 cm per frame). A rotation larger than half the horizontal field of view
// between consecutive frames cannot leave tracks alive, so such a model is refused.
//
// OpenCV supplies the estimators (findHomography / findEssentialMat UsacParams overloads, seeded, single-threaded,
// so a replay picks the same inliers; recoverPose). Ariadne owns the selection, the SO(3) projection and the
// fallbacks. The OpenCV 5.0.0 USAC `!model.empty()` defect (SMOKE §4q) is absorbed exactly as in the tracker.
//
// The seam for 2.4: this is the rotation source while lost; the preintegrated gyro replaces it there.
// ---------------------------------------------------------------------------
#include "core/geometry/pinhole.hpp"
#include "core/math/lie.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::slam {

struct RotationBridgeParams {
    double threshold_px = 1.5;
    int min_tracks = 20;
    double model_ratio = 0.45;   // R_H threshold (the paper's)
    double sigma_px = 1.0;       // the paper's sigma for the scores
    double confidence = 0.999;
    int max_iterations = 1000;
};

struct FrameRotation {
    bool ok = false;
    core::Mat3 R_prev_cur = core::Mat3::Identity();   // rotation of the current camera in the previous camera's frame
    char model = '-';                                   // 'H' or 'E' when ok
    int tracks = 0;                                     // tracks present in both frames
    int inliers_h = 0, inliers_e = 0;
    int cheiral_e = 0;                                  // of E's inliers, in front of both cameras (recoverPose)
    double disagreement_deg = -1.0;                     // angle between E's and H's rotations, when both exist
    double score_h = 0.0, score_f = 0.0;                // ORB-SLAM scores; R_H = score_h / (score_h + score_f)
    int usac_retries = 0;                               // OpenCV 5.0.0 defect redraws (SMOKE §4q)
    std::string why;                                    // when !ok
};

/// Which of the two estimates per frame a call is (each draws from its own Philox sites).
enum class BridgeStream { kFrameToFrame = 0, kFromAnchor = 1 };

/// p0[i] in an earlier rectified left image and p1[i] in the current one are the same track. Pure function of its
/// inputs and (seed, frame_id, stream).
FrameRotation EstimateFrameRotation(const std::vector<cv::Point2f>& p0, const std::vector<cv::Point2f>& p1,
                                    const core::PinholeCamera& cam, const RotationBridgeParams& params, uint64_t seed,
                                    uint64_t frame_id, BridgeStream stream = BridgeStream::kFrameToFrame);

}  // namespace ariadne::slam
