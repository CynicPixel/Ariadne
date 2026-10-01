// The read-time lens stage (D41) against OpenCV's own projection model (oracle
// only: perception/degradation stays OpenCV-free).
#include "perception/degradation/lens.hpp"
#include "perception/degradation/lens_digest.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>

using ariadne::core::ImageF32;
using ariadne::core::ImageU8;
using namespace ariadne::perception;

namespace {

/// configs/sim/rig_stereo_v1.yaml's lens block (D41).
LensModel RigV1() {
    LensModel m;
    m.width = 1280;
    m.height = 720;
    m.fx = m.fy = 680.0;
    m.cx = 639.5;
    m.cy = 359.5;
    m.k1 = -0.05;
    m.k2 = 0.01;
    m.p1 = 0.0005;
    m.p2 = -0.0002;
    m.k3 = 0.0;
    m.render_fx = m.render_fy = 640.0;
    m.render_cx = 639.5;
    m.render_cy = 359.5;
    return m;
}

}  // namespace

TEST(Lens, ModelEqualsOpenCvProjection) {
    // A 3D point seen by the pinhole render at p_render must appear in the output
    // camera exactly where cv::projectPoints (K_out, D) puts it.
    const LensModel m = RigV1();
    const cv::Matx33d K(m.fx, 0, m.cx, 0, m.fy, m.cy, 0, 0, 1);
    const cv::Mat D(cv::Matx<double, 1, 5>(m.k1, m.k2, m.p1, m.p2, m.k3));
    std::vector<cv::Point3d> pts;
    for (double x = -0.95; x <= 0.95; x += 0.05) {
        for (double y = -0.53; y <= 0.53; y += 0.05) pts.emplace_back(x * 2.0, y * 2.0, 2.0);
    }
    std::vector<cv::Point2d> img;
    cv::projectPoints(pts, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, img);
    double worst = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        double x, y;
        LensUndistort(m, (img[i].x - m.cx) / m.fx, (img[i].y - m.cy) / m.fy, x, y);
        const double su = m.render_fx * x + m.render_cx, sv = m.render_fy * y + m.render_cy;
        const double eu = m.render_fx * pts[i].x / pts[i].z + m.render_cx;
        const double ev = m.render_fy * pts[i].y / pts[i].z + m.render_cy;
        worst = std::max(worst, std::hypot(su - eu, sv - ev));
    }
    EXPECT_LT(worst, 1e-9) << "pixels";
}

TEST(Lens, RigMapConvergesAndStaysInsideTheRender) {
    const LensMap map = BuildLensMap(RigV1());
    EXPECT_LT(map.max_inverse_residual, 1e-14);
    EXPECT_GE(map.min_margin_px, 3.0);   // fx 680 vs the 674.05 minimum
}

TEST(Lens, TooShortAFocalLengthIsRefused) {
    LensModel m = RigV1();
    m.fx = m.fy = 670.0;   // barrel: corners would sample outside the render
    EXPECT_THROW(BuildLensMap(m), std::runtime_error);
}

TEST(Lens, IdentityModelReproducesTheImageExactly) {
    LensModel m;
    m.width = 37;
    m.height = 23;
    m.fx = m.fy = m.render_fx = m.render_fy = 30.0;
    m.cx = m.render_cx = 18.0;
    m.cy = m.render_cy = 11.0;
    ImageU8 img(m.width, m.height, 3);
    for (size_t i = 0; i < img.data.size(); ++i) img.data[i] = uint8_t((i * 2654435761u) >> 24);
    EXPECT_EQ(ApplyLens(img, BuildLensMap(m)).data, img.data);
}

TEST(Lens, BilinearIsExactOnALinearImage) {
    LensModel m;
    m.width = 64;
    m.height = 48;
    m.render_fx = m.render_fy = 40.0;
    m.render_cx = m.cx = 31.5;
    m.render_cy = m.cy = 23.5;
    m.fx = m.fy = 46.0;
    m.k1 = -0.08;
    m.p1 = 0.001;
    ImageU8 img(m.width, m.height, 1);
    for (int y = 0; y < m.height; ++y) {
        for (int x = 0; x < m.width; ++x) img.At(x, y) = uint8_t(x + 2 * y);   // <= 157: no wrap
    }
    const LensMap map = BuildLensMap(m);
    const ImageU8 out = ApplyLens(img, map);
    for (int v = 0; v < m.height; ++v) {
        for (int u = 0; u < m.width; ++u) {
            const size_t i = size_t(v) * m.width + u;
            const double want = map.src_u[i] + 2.0 * map.src_v[i];
            ASSERT_LE(std::abs(out.At(u, v) - want), 0.5 + 1e-9) << u << "," << v;
        }
    }
}

TEST(Lens, DepthUsesNearestNeighbour) {
    LensModel m = RigV1();
    m.width = 128;
    m.height = 72;
    m.fx = m.fy = 68.0;
    m.cx = 63.5;
    m.cy = 35.5;
    m.render_fx = m.render_fy = 64.0;
    m.render_cx = 63.5;
    m.render_cy = 35.5;
    ImageF32 d(m.width, m.height);
    for (size_t i = 0; i < d.data.size(); ++i) d.data[i] = float(i);   // value = its own index
    const LensMap map = BuildLensMap(m);
    const ImageF32 out = ApplyLensDepth(d, map);
    for (size_t i = 0; i < out.data.size(); ++i) {
        const int x = int(std::floor(map.src_u[i] + 0.5)), y = int(std::floor(map.src_v[i] + 0.5));
        ASSERT_EQ(out.data[i], float(y * m.width + x));
    }
}

TEST(Lens, OutputIsPinnedByADigest) {
    // Same bytes on every platform (IEEE double, -ffp-contract=off). The digest has
    // ONE implementation (lens_digest.hpp), shared with check_determinism.sh and the
    // run manifest; its constant is EXPECTED_LENS there.
    EXPECT_EQ(LensDigest(), 1253886791886396038ull);
}
