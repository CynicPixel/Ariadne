// D142: GT depth for the right sim camera, warped from the left depth camera.
#include "runtime/datasets/sim_depth.hpp"

#include <gtest/gtest.h>

using ariadne::core::ImageF32;
using ariadne::runtime::WarpDepthToRight;

// A wall at 4 m with a box at 1 m in columns [100, 120]. f = 640, B = 0.1: the wall shifts 16 px, the box 64 px.
TEST(SimDepth, ForwardSplatKeepsTheNearestAndLeavesDisocclusionsUnknown) {
    ImageF32 left(200, 3);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 200; ++x) left.At(x, y) = (x >= 100 && x <= 120) ? 1.0f : 4.0f;
    left.At(5, 1) = 0.0f;   // an unknown pixel stays unknown
    const ImageF32 r = WarpDepthToRight(left, 640.0, 0.1);
    EXPECT_EQ(r.At(46, 1), 1.0f);    // the box, from left column 110
    EXPECT_EQ(r.At(70, 1), 4.0f);    // the wall, from left column 86
    EXPECT_EQ(r.At(94, 1), 0.0f);    // wall hidden behind the box in the left view: disoccluded, unknown
    EXPECT_EQ(r.At(110, 1), 4.0f);   // the wall again, from left column 126
    EXPECT_EQ(r.At(199, 1), 0.0f);   // nothing lands right of 200 - 16
}
