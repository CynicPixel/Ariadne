// 1.6 edge-verifier diagnostics (D43): planarity is ~0 for coplanar points and
// clearly positive for a spread cloud; a homography explains every match of a
// planar scene (h/E ~ 1) but only a fraction of a non-planar one seen from two
// translated views. Both are recorded, never gated, in 1.6.
#include "collaboration/broker/edge_verifier.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace ariadne;

namespace {
// Pinhole projection of p (camera frame) at f 680, c (640, 360).
cv::Point2f Project(const cv::Point3f& p) { return {680.f * p.x / p.z + 640.f, 680.f * p.y / p.z + 360.f}; }

std::vector<cv::Point3f> Cloud(bool planar, int n) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<cv::Point3f> pts;
    for (int i = 0; i < n; ++i) pts.emplace_back(u(rng), 0.6f * u(rng), planar ? 3.f : 3.f + 1.5f * u(rng));
    return pts;
}
}  // namespace

TEST(EdgeVerifier, PlanaritySeparatesPlaneFromVolume) {
    EXPECT_LT(collaboration::Planarity(Cloud(true, 200)), 1e-9);
    EXPECT_GT(collaboration::Planarity(Cloud(false, 200)), 0.1);
    EXPECT_TRUE(std::isnan(collaboration::Planarity(Cloud(false, 3))));
}

TEST(EdgeVerifier, HomographyExplainsPlanarMatchesOnly) {
    // View A at the origin, view B translated 0.4 m sideways; same points.
    for (const bool planar : {true, false}) {
        const auto pts = Cloud(planar, 200);
        std::vector<cv::Point2f> pa, pb;
        for (const auto& p : pts) {
            pa.push_back(Project(p));
            pb.push_back(Project({p.x - 0.4f, p.y, p.z}));
        }
        const double frac = double(collaboration::HomographyInliers(pa, pb, 1.0, 1)) / double(pts.size());
        if (planar) {
            EXPECT_GT(frac, 0.99);
        } else {
            EXPECT_LT(frac, 0.5);
        }
    }
    EXPECT_EQ(collaboration::HomographyInliers({{0, 0}}, {{0, 0}}, 1.0, 1), -1);
}
