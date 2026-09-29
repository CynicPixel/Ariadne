#pragma once
// ---------------------------------------------------------------------------
// Placeholder keyframe stereo visual odometry (plan 1.5). The simplest correct
// pipeline behind the real interfaces; Phase 2 replaces its internals with the
// full estimator (windowed Schur-LM BA, IMU, §10-§11) and keeps it as a
// baseline.
//
//   Track(frame)        3D-2D PnP of tracked points against landmarks:
//                       cv::solvePnPRansac (USAC, seeded) for the inlier set,
//                       then Ariadne's own motion-only LM (Huber) for the pose
//                       and its covariance. Decides whether a keyframe is due.
//   AddKeyframe(stereo) given left->right correspondences (and SGBM depth, when
//                       available) for the tracked points: triangulate new
//                       landmarks (cv::triangulatePoints, §11.2 rejections),
//                       run the two-frame BA stub against the previous
//                       keyframe (gauge fixed), and extend the pose graph with
//                       an ODOMETRY edge carrying the BA covariance.
//
// Pattern: stereo landmark creation at keyframes as in ORB-SLAM3 (close
// points only, ThDepth) and VINS-Fusion (stereo DLT at first observation);
// PnP tracking against map points as both do.
//
// Inputs are plain types (§3: slam does not include perception). The runtime
// layer glues the tracker, the stereo matcher and SGBM to this API.
//
// LIMITS (honest, placeholder-level): a landmark is reused only while the LK
// track that created it survives (no descriptor re-association, so no loop
// closure and slow drift is expected); no IMU (imu_mode none); BA covers two
// keyframes, not a window.
// ---------------------------------------------------------------------------
#include "core/geometry/pinhole.hpp"
#include "slam/pose_graph/pose_graph.hpp"
#include "slam/slam_config.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace ariadne::slam {

using core::RectifiedStereo;

struct VoFrame {
    uint64_t frame_id = 0;
    int64_t t_ns = 0;
    std::vector<uint64_t> track_ids;          // tracker ids, parallel to points
    std::vector<cv::Point2f> points;          // rectified left image
};

/// Stereo evidence for the tracked points of the frame just tracked, parallel
/// to VoFrame::points. right[i] is meaningful only when valid[i].
struct VoStereo {
    std::vector<cv::Point2f> right;
    std::vector<uchar> valid;
    /// SGBM depth at the left point where SGBM is valid AND densely supported
    /// (§9.6, D34); NaN otherwise.
    std::vector<float> sgbm_z;
    std::vector<float> sgbm_sigma;
};

enum class VoStatus { kInitializing, kTracking, kLost };

struct VoTrackResult {
    VoStatus status = VoStatus::kInitializing;
    core::SE3 T_W_C;                          // rectified left camera -> world
    /// Left tangent of T_W_C, CONDITIONAL on the current map (landmarks held
    /// fixed in the refinement). Not the absolute uncertainty: keyframe nodes
    /// in graph() carry that, composed along the chain.
    core::Mat6 covariance_given_map = core::Mat6::Zero();
    int correspondences = 0;
    int pnp_inliers = 0;
    bool need_keyframe = false;
};

struct TriangulationStats {
    int candidates = 0, created = 0, no_stereo = 0, nonpositive_disparity = 0, depth_range = 0,
        reprojection = 0, sgbm_inconsistent = 0, uncertain = 0;
};

struct VoKeyframeResult {
    uint64_t keyframe_id = 0;
    TriangulationStats triangulation;
    int ba_landmarks = 0;
    int ba_iterations = 0;
    double ba_initial_cost = 0.0, ba_final_cost = 0.0;
    bool ba_ran = false;
};

class StereoVo {
public:
    StereoVo(const SlamConfig& config, const RectifiedStereo& rig, uint64_t seed, uint32_t agent = 0);

    VoTrackResult Track(const VoFrame& frame);

    /// Call after Track() returned need_keyframe, with stereo for that frame.
    VoKeyframeResult AddKeyframe(const VoStereo& stereo);

    const PoseGraph& graph() const { return graph_; }
    size_t landmark_count() const { return landmarks_.size(); }

private:
    struct Landmark {
        core::Vec3 X_w;
    };
    struct KeyframeState {
        uint64_t id = 0;
        core::SE3 T_W_C;
        // Observations recorded at this keyframe: landmark -> (left, right, has_right).
        struct Obs { cv::Point2f left, right; bool has_right = false; };
        std::map<uint64_t, Obs> obs;
        int tracked_landmarks = 0;    // landmarks visible when it was created (ref for the KF rule)
    };

    SlamConfig cfg_;
    RectifiedStereo rig_;
    uint64_t seed_;
    uint32_t agent_;
    std::map<uint64_t, Landmark> landmarks_;    // keyed by the creating track id
    std::vector<KeyframeState> keyframes_;
    PoseGraph graph_;
    VoFrame last_frame_;
    VoTrackResult last_;
    core::SE3 prev_T_W_C_, prev2_T_W_C_;
    int frames_tracked_ = 0;
    int frames_since_kf_ = 0;
};

}  // namespace ariadne::slam
