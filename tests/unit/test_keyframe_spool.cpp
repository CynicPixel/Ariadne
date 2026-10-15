// 1.7 spool (design RECON_1.7): the C++ -> Python boundary. fp16-in-PNG depth
// round-trips (NaN preserved, half precision), intrinsics scale with the
// pixel-centre convention, the keyframe stride applies per agent, and the index
// carries every §18.2 field.
#include "reconstruction/spool/keyframe_spool.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>

using namespace ariadne;
namespace fs = std::filesystem;

namespace {
fs::path TempDir(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / ("ariadne_spool_" + name);
    fs::remove_all(d);
    return d;
}
}  // namespace

TEST(KeyframeSpool, HalfDepthRoundTripsWithNan) {
    const fs::path d = TempDir("half");
    reconstruction::KeyframeSpool sp(d, {1.0, 1, 200});
    cv::Mat z(4, 6, CV_32F, cv::Scalar(2.5f)), s(4, 6, CV_32F, cv::Scalar(0.01f));
    z.at<float>(1, 2) = std::numeric_limits<float>::quiet_NaN();
    z.at<float>(3, 5) = 1.2345f;
    const cv::Mat rgb(4, 6, CV_8UC3, cv::Scalar(10, 20, 30));
    sp.AddKeyframe(0, 7, 100, 3, core::SE3(), sp.Camera(10, 10, 2.5, 1.5, 6, 4), 0.1, rgb, z, s);
    const cv::Mat bits = cv::imread((d / "agent0/kf_7_depth.png").string(), cv::IMREAD_UNCHANGED);
    ASSERT_EQ(bits.type(), CV_16U);
    cv::Mat back;
    cv::Mat(bits.size(), CV_16F, bits.data, bits.step).convertTo(back, CV_32F);
    EXPECT_TRUE(std::isnan(back.at<float>(1, 2)));
    EXPECT_NEAR(back.at<float>(3, 5), 1.2345f, 1e-3);   // fp16: ~11 bits of mantissa
    EXPECT_FLOAT_EQ(back.at<float>(0, 0), 2.5f);        // exactly representable
    const cv::Mat bgr = cv::imread((d / "agent0/kf_7_rgb.png").string(), cv::IMREAD_UNCHANGED);
    EXPECT_EQ(bgr.at<cv::Vec3b>(0, 0), cv::Vec3b(30, 20, 10));   // RGB in memory, BGR on disk
}

TEST(KeyframeSpool, IntrinsicsScaleWithPixelCentres) {
    reconstruction::KeyframeSpool sp(TempDir("cam"), {0.5, 1, 200});
    const auto c = sp.Camera(680, 680, 639.5, 359.5, 1280, 720);
    EXPECT_DOUBLE_EQ(c.fx, 340);
    EXPECT_DOUBLE_EQ(c.cx, 319.5);   // the image centre stays the image centre
    EXPECT_DOUBLE_EQ(c.cy, 179.5);
    EXPECT_EQ(c.width, 640);
    EXPECT_EQ(c.height, 360);
}

TEST(KeyframeSpool, StrideIsPerAgentAndIndexHasEveryField) {
    const fs::path d = TempDir("index");
    reconstruction::KeyframeSpool sp(d, {1.0, 2, 10});
    const cv::Mat z(2, 2, CV_32F, cv::Scalar(1.f)), rgb(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto cam = sp.Camera(1, 1, 0.5, 0.5, 2, 2);
    for (uint64_t f = 0; f < 4; ++f) {
        for (uint32_t a = 0; a < 2; ++a) sp.AddKeyframe(a, f, int64_t(f), f, core::SE3(), cam, 0.1, rgb, z, z);
    }
    EXPECT_EQ(sp.size(), 4u);   // frames 0 and 2 for each agent
    EXPECT_TRUE(sp.WantHeldout(5));
    EXPECT_FALSE(sp.WantHeldout(4));
    sp.AddHeldout(1, 5, 5, core::SE3(), cam, rgb);
    sp.Finish("run", "scene", true, core::SE3(), "edge");
    const YAML::Node y = YAML::LoadFile((d / "spool.yaml").string());
    EXPECT_EQ(y["schema_version"].as<int>(), 1);
    EXPECT_EQ(y["merge"]["T_O0_O1"].size(), 12u);
    ASSERT_EQ(y["bundles"].size(), 5u);
    for (const char* k : {"role", "agent", "kf_id", "t_ns", "frame_id_of_pose", "T_WiC", "K", "width", "height",
                          "baseline_m", "files", "rgb_kind", "exposure_state", "lamp_state", "mask"}) {
        EXPECT_TRUE(y["bundles"][0][k].IsDefined()) << k;
    }
    EXPECT_EQ(y["bundles"][4]["role"].as<std::string>(), "heldout");
}
