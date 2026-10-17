#pragma once
// Where a keyframe's / frame's pose came from (LOST_FRAMES.md, D56). OpenCV-free, so keyframe records and the wire
// side can carry it without the VO.
#include <cstdint>

namespace ariadne::slam {

/// Measured by PnP against the map, or predicted while lost: by the visual rotation bridge plus a decaying velocity
/// (lost.motion = bridge), or by the frozen constant velocity (coast, the 1.5 behaviour).
enum class PoseSource : uint8_t { kMeasured = 0, kBridged = 1, kCoasted = 2 };

inline const char* ToString(PoseSource s) {
    return s == PoseSource::kMeasured ? "measured" : s == PoseSource::kBridged ? "bridged" : "coasted";
}

}  // namespace ariadne::slam
