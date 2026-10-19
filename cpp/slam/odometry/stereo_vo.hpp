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
#include "core/inertial/imu.hpp"
#include "slam/inertial/initialization.hpp"
#include "slam/inertial/preintegration.hpp"
#include "slam/odometry/pose_source.hpp"
#include "slam/odometry/rotation_bridge.hpp"
#include "slam/pose_graph/pose_graph.hpp"
#include "slam/slam_config.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

using core::RectifiedStereo;
struct BaWindow;

struct VoFrame {
    uint64_t frame_id = 0;
    int64_t t_ns = 0;
    std::vector<uint64_t> track_ids;          // tracker ids, parallel to points
    std::vector<cv::Point2f> points;          // rectified left image
    /// imu_mode attitude_only: the bias-corrected gyro rotation of the camera since the previous frame
    /// (R_C(prev)_C(this)). Used only while lost, where it replaces the visual bridge's rotation.
    bool has_rotation_prior = false;
    core::Mat3 R_prev_cur_prior = core::Mat3::Identity();
    /// imu_mode full: the body (IMU) samples covering [previous frame, this frame] -- the last sample at or before the
    /// previous frame through the first at or after this one (the first frame: through the first at or after it).
    /// Overlap with the previous frame's slice is expected; StereoVo keeps each timestamp once.
    std::vector<core::ImuMeasurement> imu;
};

/// imu_mode full: the IMU's mount and noise model (Kalibr densities, before StereoVo's imu.full inflation).
struct VoImuRig {
    core::SE3 T_B_C;          // rectified left camera in the body (IMU) frame
    core::ImuNoise noise;
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

/// kLost is arch §8.2's DEGRADED: the pose is predicted, not measured against the map.
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
    /// lost.reanchor = gated, frame lost and no keyframe due: give the stereo for this frame to ShouldReanchor(), and
    /// call AddKeyframe() if it says yes.
    bool reanchor_probe = false;
    PoseSource source = PoseSource::kMeasured;
    uint32_t segment = 0;          // map segment: incremented at every loss onset
    FrameRotation bridge;          // lost.motion = bridge: this frame's measured rotation (every frame, diagnostic too)
    FrameRotation anchor_bridge;   // lost + bridge: rotation from the anchor frame (the one used, when ok)
    int anchor_age = -1;           // frames between the anchor and this one
    double anchor_drift_deg = -1;  // anchor snapshot vs the (possibly window-refined) previous pose, when comparable
};

struct TriangulationStats {
    int candidates = 0, created = 0, no_stereo = 0, nonpositive_disparity = 0, depth_range = 0,
        reprojection = 0, sgbm_inconsistent = 0, uncertain = 0;
};

struct VoKeyframeResult {
    uint64_t keyframe_id = 0;
    PoseSource source = PoseSource::kMeasured;   // of the frame it was made from
    uint32_t segment = 0;
    TriangulationStats triangulation;
    int ba_landmarks = 0;
    int ba_iterations = 0;
    double ba_initial_cost = 0.0, ba_final_cost = 0.0;
    bool ba_ran = false;
    int ba_window = 0;            // 2.3: keyframes in the solved window
    int ba_rejected = 0;          // observations dropped by the post-solve chi^2 test
    int ba_culled = 0;            // landmarks culled (design §5)
    bool exit_edge = false;       // an ODOMETRY edge was emitted for the keyframe leaving the window
    bool ba_found = false;        // the LM driver reported convergence
    int ba_observations = 0;      // observations in the solved window
    int ba_invalid_at_entry = 0;  // of them, behind / at a camera at the starting estimate
    int ba_ill_conditioned = 0;        // landmarks left out at entry: own information too weak (window_ba.hpp)
    int ba_unobservable = 0;           // keyframes held at entry: not tied to a fixed one through covisibility
    int ba_ill_conditioned_after = 0;  // the same tests at the solution (left out of the marginals)
    int ba_unobservable_after = 0;
    int ba_marginal_degenerate = 0;    // the information was singular at the solution: marginals uninformative
    double ba_inlier_sq_px = 0.0;      // sum |r|^2 over chi^2 inliers at the solution (sigma_px calibration)
    long ba_inlier_dof = 0;            // their scalar residual count minus the free parameters
    int ba_chi2_failed = 0;       // residuals above chi2_reject at the solution (vs ~5% expected at chi^2_2(0.95))
    // imu_mode full (D59, INERTIAL_2.4 §4).
    bool ba_inertial = false;          // the solve had inertial factors (initialised)
    bool ba_v0_held = false;           // window[0] was the only pose-determined keyframe: its velocity held (§4.2)
    int ba_inertial_factors = 0;       // inertial factors in the solve (9 scalar residuals each)
    double ba_inertial_chi2 = 0.0;     // their whitened squared residuals at the solution (§4.5: chi^2/dof per flight)
    int imu_repropagated = 0;          // intervals re-integrated before the solve (bias moved past imu.full.repropagate_*)
    int imu_prior = 0;                 // the next solve's window[0] bias prior: 1 this solve's marginal, 2 the previous
                                       // one carried forward (degenerate marginal or no solve), 0 unchanged (window
                                       // not full yet, or not initialised)
    bool imu_initialized = false;      // the inertial initialisation succeeded at this keyframe
    bool imu_init_refused = false;     // an attempt was refused at this keyframe (StereoVo::inertial_init().why)
    bool imu_refined = false;          // a scheduled refinement (imu.full.refine_after_s) succeeded at this keyframe
    bool imu_refine_refused = false;   // a refinement attempt was refused at this keyframe
    double imu_tilt_deg = 0.0;         // §4.8: the rotation that took the solved gravity back to the reference
    double imu_g_prior_sigma_deg = 0.0;   // §4.8: the gravity prior's sigma (largest axis) in that solve
};

/// One run of the scheduled inertial refinement (imu.full.refine_after_s): the initialisation's MAP re-solved over every
/// keyframe since the initialisation (ORB-SLAM3, T-RO 2021 §V-B: "visual-inertial BA is performed 5 and 15 seconds
/// after initialization"; here inertial-only, on the window-refined poses).
struct InertialRefinement {
    int64_t t_ns = 0;
    int keyframes = 0;
    bool ok = false;
    double gravity_change_deg = 0.0;   // between the gravity before and after (when ok)
    double chi2_per_dof = 0.0;
    double gravity_norm_unconstrained = 0.0;
    std::string why;                   // when !ok
};

class StereoVo {
public:
    /// `imu` is required when config.imu.mode is full (and ignored otherwise).
    StereoVo(const SlamConfig& config, const RectifiedStereo& rig, uint64_t seed, uint32_t agent = 0,
             const VoImuRig* imu = nullptr);

    VoTrackResult Track(const VoFrame& frame);

    /// Call after Track() returned need_keyframe, with stereo for that frame.
    VoKeyframeResult AddKeyframe(const VoStereo& stereo);

    /// lost.reanchor = gated: would this lost frame, made a keyframe, give the next frame's PnP enough landmarks
    /// (tracked landmarks + the ones it would triangulate >= lost.reanchor_min_points)? Changes nothing.
    bool ShouldReanchor(const VoStereo& stereo) const;

    const PoseGraph& graph() const { return graph_; }
    /// imu_mode full: initialised yet, and the last initialisation attempt (its keyframes' times in init_times()).
    bool inertial_ready() const { return inertial_ready_; }
    const InertialInitResult& inertial_init() const { return init_; }
    const std::vector<int64_t>& init_times() const { return init_t_ns_; }
    int init_attempts() const { return init_attempts_; }
    const std::vector<InertialRefinement>& refinements() const { return refinements_; }
    /// Gravity in the VO world (valid once inertial_ready()).
    const core::Vec3& gravity() const { return g_W_; }
    /// imu_mode full: the newest keyframe's velocity (body, in W) and biases, as last solved.
    void NewestInertialState(core::Vec3& v, core::Vec3& bg, core::Vec3& ba) const {
        if (keyframes_.empty()) throw std::logic_error("StereoVo::NewestInertialState: no keyframe");
        v = keyframes_.back().v;
        bg = keyframes_.back().bg;
        ba = keyframes_.back().ba;
    }
    size_t landmark_count() const { return landmarks_.size(); }
    /// The landmarks observed at keyframe `kf` (§4.2 landmark_ids), ascending; throws for an unknown keyframe.
    std::vector<uint64_t> KeyframeLandmarks(uint64_t kf) const {
        if (kf >= keyframes_.size()) throw std::out_of_range("StereoVo::KeyframeLandmarks: unknown keyframe");
        std::vector<uint64_t> ids;
        for (const auto& [id, obs] : keyframes_[kf].obs) ids.push_back(id);
        return ids;
    }

private:
    struct Landmark {
        core::Vec3 X_w;
        uint64_t born_kf = 0;   // keyframe that created it (2.3 culling)
    };
    struct KeyframeState {
        uint64_t id = 0;
        core::SE3 T_W_C;
        PoseSource source = PoseSource::kMeasured;
        uint32_t segment = 0;
        // Observations recorded at this keyframe: landmark -> (left, right, has_right).
        struct Obs { cv::Point2f left, right; bool has_right = false; };
        std::map<uint64_t, Obs> obs;
        int tracked_landmarks = 0;    // landmarks visible when it was created (ref for the KF rule)
        // imu_mode full.
        int64_t t_ns = 0;
        std::vector<core::ImuMeasurement> imu_from_prev;   // samples covering [previous keyframe, this one]
        Preintegrated pre_from_prev;                        // of imu_from_prev, at the previous keyframe's biases then
        bool has_pre = false;
        core::Vec3 v = core::Vec3::Zero(), bg = core::Vec3::Zero(), ba = core::Vec3::Zero();   // body, in W
    };

    /// 2.3 windowed BA path (LOCAL_SLAM_2.3.md): solve the last window_size keyframes, reject / cull, update the
    /// graph, and emit the exit edge of the keyframe that leaves the window.
    VoKeyframeResult AddKeyframeWindow(KeyframeState kf, VoKeyframeResult out, int64_t t_ns);

    /// §11.2 stereo triangulation of the frame's tracks that are not landmarks yet, every rejection counted:
    /// (track index, point in the camera frame). Const: AddKeyframe commits, ShouldReanchor only counts.
    std::vector<std::pair<size_t, core::Vec3>> TriangulateNew(const VoFrame& f, const VoStereo& st,
                                                              TriangulationStats& stats) const;
    /// imu_mode full, after initialisation: the camera pose at t_ns preintegrated from the last keyframe's state.
    core::SE3 PropagateFromLastKeyframe(int64_t t_ns) const;
    /// imu_mode full: the inertial part of the window keyframes_[first..] (re-integrating stale intervals), and the
    /// window[0] bias prior carried to that keyframe.
    void FillInertial(BaWindow& w, size_t first, VoKeyframeResult& out);
    /// imu_mode full, not initialised: try the initialisation on the newest window keyframes.
    void TryInitializeInertial(size_t first, VoKeyframeResult& out);
    /// imu_mode full, initialised: the scheduled refinement, when due.
    void TryRefineInertial(size_t first, VoKeyframeResult& out);
    /// Apply an initialisation / refinement over keyframes_[s0..]: gravity, the window keyframes' biases and
    /// velocities, and window[0]'s bias prior.
    void ApplyInertialEstimate(const InertialInitResult& r, size_t s0, size_t first);
    /// The next window[0]'s bias prior covariance from a marginal, never looser than the current prior carried over dt.
    core::Mat6 NextBiasPrior(const core::Mat6& marginal, double dt) const;
    /// lost.motion = bridge: the lost frame's pose. Rotation from the ANCHOR -- the last measured frame, or the last
    /// frame enough tracks reach back to -- so the two-view baseline grows instead of per-frame noise accumulating.
    core::SE3 BridgePrediction(const VoFrame& frame, VoTrackResult& res);
    struct BridgeAnchor {
        core::SE3 T_W_C;
        uint64_t frame_id = 0;
        std::map<uint64_t, cv::Point2f> points;
    };
    BridgeAnchor anchor_;
    struct LastAnchorRotation {   // the previous successful anchor estimate (anchor_drift_deg diagnostic)
        uint64_t anchor_frame = ~uint64_t(0), for_frame = ~uint64_t(0);
        core::Mat3 R = core::Mat3::Identity();
    } last_anchor_;

    SlamConfig cfg_;
    RectifiedStereo rig_;
    uint64_t seed_;
    uint32_t agent_;
    std::map<uint64_t, Landmark> landmarks_;    // keyed by the creating track id
    std::vector<KeyframeState> keyframes_;
    PoseGraph graph_;
    VoFrame last_frame_;
    VoFrame prev_frame_;                              // the frame before last_frame_ (the bridge's first view)
    VoTrackResult last_;
    core::SE3 prev_T_W_C_, prev2_T_W_C_;
    int frames_tracked_ = 0;
    core::Vec6 velocity_per_s_ = core::Vec6::Zero();   // last MEASURED twist per second, held while coasting (Track)
    bool coasting_ = false;                           // the previous frame was lost: keep velocity_per_s_
    int64_t prev_t_ns_ = 0, prev2_t_ns_ = 0;          // timestamps of prev_T_W_C_ / prev2_T_W_C_
    int64_t last_kf_t_ns_ = 0;                        // keyframe.max_interval_s is measured from here
    uint32_t segment_ = 0;                            // incremented at every loss onset
    int64_t loss_t0_ns_ = 0;                          // last measured pose's time: the bridge's velocity decays from it
    // Window exit (design §4): the marginal of window[1] from the last solve, in which window[0] was the fixed gauge.
    core::Mat6 pending_exit_cov_ = core::Mat6::Zero();
    bool have_pending_exit_ = false;
    // imu_mode full (D59).
    bool full_imu_ = false;
    VoImuRig imu_rig_;
    core::SE3 T_C_B_;
    PreintegrationNoise pre_noise_;                   // inflated white-noise densities
    double gyro_rw_ = 0.0, accel_rw_ = 0.0;           // inflated random-walk densities
    std::vector<core::ImuMeasurement> imu_since_kf_;  // samples since (the last one at or before) the newest keyframe
    bool inertial_ready_ = false;
    core::Vec3 g_W_ = core::Vec3::Zero();
    InertialInitResult init_;
    std::vector<int64_t> init_t_ns_;
    int init_attempts_ = 0;
    uint64_t init_kf0_ = 0;                           // the initialisation's first keyframe
    int64_t init_done_t_ns_ = 0;                      // the keyframe at which it succeeded
    size_t refinements_done_ = 0;                     // of imu.full.refine_after_s
    std::vector<InertialRefinement> refinements_;
    // window[0]'s bias prior: mean [bg; ba] and covariance, valid for keyframe prior_kf_ (carried forward by the random
    // walk when window[0] moves past it without a new marginal).
    core::Vec6 prior_b_ = core::Vec6::Zero();
    core::Mat6 prior_cov_ = core::Mat6::Zero();
    uint64_t prior_kf_ = 0;
    bool inertial_prior_set_ = false;               // the initialisation has set the first bias prior
    // Gravity's tilt prior (rad^2, GravityTangentBasis(g_W_)), for windows that estimate gravity (§4.8).
    Eigen::Matrix2d g_prior_cov_ = Eigen::Matrix2d::Identity();
};

}  // namespace ariadne::slam
