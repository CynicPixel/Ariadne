#pragma once
// ---------------------------------------------------------------------------
// The canonical keyframe records that combine SLAM and perception outputs.
// They live in runtime because §3 makes slam and perception peers: neither may
// include the other, so the records that join them sit one layer up.
//
//   Keyframe        arch §4.2, the per-agent keyframe
//   KeyframeBundle  arch §18.2, what the reconstruction service consumes
//
// Phase 1 fills what exists. Fields no producer fills yet are listed below and
// left explicitly unset, never defaulted to plausible-looking values:
//   Keyframe: inertial state (velocity, biases, gravity_dir, preintegration:
//     §10, imu_mode none), local_cloud (the 1.7 service fuses depth itself),
//     motion/energy cost (planner, Phase 4), timestamp_age and the uncertainty
//     summary (Phase 2). keypoints/descriptors/global_descriptor are filled by
//     the consumer that extracts them (1.6: ORB + mean-bit in sim_collab).
//   KeyframeBundle: rgb is the rectified RGB (gray where the source has no
//     colour), NOT the falloff-normalised albedo proxy of §19.1 (Phase 5);
//     exposure_state and lamp_state are unset (§6.5, Phase 5).
// The Phase 1 producer is tools::RunStereoVo's keyframe hook; the consumers are
// the 1.6 exchange (features) and the 1.7 spool (bundles), see design PHASE1_GATE.
// ---------------------------------------------------------------------------
#include "core/geometry/pinhole.hpp"
#include "perception/stereo/sgbm_stereo.hpp"
#include "slam/odometry/pose_source.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ariadne::runtime {

/// §4.2 (the subset Phase 1 produces).
struct Keyframe {
    uint64_t id = 0;                 // the agent's keyframe sequence number (its pose-graph node)
    uint64_t frame_id = 0;           // provenance: the §4.1 frame it was made from (the wire's keyframe id)
    uint32_t agent = 0;
    int64_t t_ns = 0;
    core::SE3 T_world_cam;
    core::Mat6 pose_cov = core::Mat6::Zero();       // left tangent of T_world_cam
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    perception::StereoDepth depth;
    std::vector<uint64_t> landmark_ids;
    std::vector<float> global_descriptor;           // empty until a place-recognition consumer fills it
    // Tracking provenance (LOST_FRAMES.md, D56; AUDIT_2.3 §4): a keyframe made while tracking was lost has a
    // PREDICTED pose. Consumers (exchange, reconstruction, planning) weight or exclude it by this, not by guessing
    // from pose_cov. segment counts loss onsets: keyframes of different segments are joined only by predictions.
    slam::PoseSource pose_source = slam::PoseSource::kMeasured;
    uint32_t segment = 0;
    // §4.2 gravity_dir / §10.4 producer (2.4 attitude_only): gravity's direction (down, unit) in this keyframe's
    // rectified left camera, from Mahony's filter. Unset (has_gravity false) without an IMU.
    bool has_gravity = false;
    core::Vec3 gravity_dir_C = core::Vec3::Zero();
};

struct ExposureState { double exposure_s = 0.0; double gain = 0.0; };
struct LampState { bool on = false; double power = 0.0; };

/// §18.2.
struct KeyframeBundle {
    uint64_t kf_id = 0;
    uint32_t agent = 0;
    uint64_t frame_id_of_pose = 0;       // the pose-graph VERSION the pose was exported at
    core::SE3 T_WiC;
    core::PinholeCamera K;
    cv::Mat rgb;                         // [PLACEHOLDER] rectified RGB (gray if no colour), not the §19.1 albedo proxy
    cv::Mat depth;                       // CV_16F metres (StereoDepth::Z), 0 where invalid
    cv::Mat depth_sigma;                 // CV_16F metres (StereoDepth::sigma_Z)
    std::optional<ExposureState> exposure_state;   // unset until §6.5
    std::optional<LampState> lamp_state;           // unset until §6.5
    cv::Mat mask;                        // CV_8U 255 = depth usable (valid AND densely supported, D34)
};

/// Export a keyframe for reconstruction at the graph's CURRENT version, so the
/// service can re-anchor (§18.6) when later optimisation moves the pose.
inline KeyframeBundle BuildKeyframeBundle(const slam::PoseGraph& graph, uint64_t kf_id,
                                          const core::PinholeCamera& K, const cv::Mat& image,
                                          const perception::StereoDepth& depth, const cv::Mat& supported) {
    const auto it = graph.nodes().find(kf_id);
    if (it == graph.nodes().end()) throw std::invalid_argument("BuildKeyframeBundle: keyframe not in the graph");
    if (depth.Z.size() != image.size() || supported.size() != image.size()) {
        throw std::invalid_argument("BuildKeyframeBundle: image, depth and mask sizes differ");
    }
    KeyframeBundle b;
    b.kf_id = kf_id;
    b.agent = it->second.agent;
    b.frame_id_of_pose = graph.version();
    b.T_WiC = it->second.T_W_C;
    b.K = K;
    b.rgb = image;
    b.depth = depth.Z;
    b.depth_sigma = depth.sigma_Z;
    b.mask = supported;
    return b;
}

/// A bundle depth channel (CV_16F) as CV_32F metres with NaN wherever the mask rejects it: what the spool
/// and feature back-projection consume.
inline cv::Mat MaskedDepth32(const cv::Mat& d16, const cv::Mat& mask) {
    if (d16.size() != mask.size() || mask.type() != CV_8U) throw std::invalid_argument("MaskedDepth32: bad mask");
    cv::Mat z;
    d16.convertTo(z, CV_32F);
    z.setTo(std::numeric_limits<float>::quiet_NaN(), mask == 0);
    return z;
}

}  // namespace ariadne::runtime
