#pragma once
// ---------------------------------------------------------------------------
// GT depth for the RIGHT sim camera (D142): the recording carries one depth
// camera, co-located with the left camera, but D's dust stage needs the depth
// each camera sees. In the pinhole render the right camera is the left one
// translated by the baseline B along +x (optical), with no rotation, so a left
// pixel (u, v) at depth z lands on the same row at u - f B / z with the same z.
//
// Forward z-buffered splat on the RENDER grid (before the lens; ApplyLensDepth
// then maps it like the image): the nearest surface wins where two land on one
// pixel, and pixels nothing lands on (disocclusions beside near objects) stay 0,
// which D reads as "unknown depth: no dust" -- never an invented distance.
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"

#include <cmath>

namespace ariadne::runtime {

/// `left_z`: render-grid z-depth of the left camera (metres, <= 0 or non-finite = unknown). `f`: render focal, px.
inline core::ImageF32 WarpDepthToRight(const core::ImageF32& left_z, double f, double baseline_m) {
    core::ImageF32 right(left_z.width, left_z.height);
    for (int y = 0; y < left_z.height; ++y)
        for (int x = 0; x < left_z.width; ++x) {
            const float z = left_z.At(x, y);
            if (!(z > 0.0f) || !std::isfinite(z)) continue;
            const long xr = std::lround(x - f * baseline_m / static_cast<double>(z));
            if (xr < 0 || xr >= right.width) continue;
            float& dst = right.At(static_cast<int>(xr), y);
            if (dst == 0.0f || z < dst) dst = z;
        }
    return right;
}

}  // namespace ariadne::runtime
