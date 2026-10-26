#pragma once
// Where a keyframe's / frame's pose came from (local-slam.md#lost-frames, D56). OpenCV-free, so keyframe records and the wire
// side can carry it without the VO.
#include <cstdint>

namespace ariadne::slam {

/// Measured by PnP against the map, or predicted while lost: by the visual rotation bridge plus a decaying velocity
/// (lost.motion = bridge), or by the frozen constant velocity (coast, the 1.5 behaviour).
/// kGyro (2.4, imu_mode attitude_only): rotation from the bias-corrected gyro, translation as for kBridged.
/// kInertial (D59, imu.mode full, imu.full.lost_motion propagate): the whole pose preintegrated from the last
/// keyframe's solved state (pose, velocity, biases) and gravity.
enum class PoseSource : uint8_t { kMeasured = 0, kBridged = 1, kCoasted = 2, kGyro = 3, kInertial = 4 };

inline const char* ToString(PoseSource s) {
    return s == PoseSource::kMeasured ? "measured"
           : s == PoseSource::kBridged ? "bridged"
           : s == PoseSource::kGyro    ? "gyro"
           : s == PoseSource::kInertial ? "inertial"
                                       : "coasted";
}

}  // namespace ariadne::slam
