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

// 6.1 (D195): schema 2 -- the albedo proxy (NaN where it is not defined), the applied photometric state per bundle,
// and the lamp (calibration copied in, R_rect_raw). A spool without any of them stays schema 1 (test above).
TEST(KeyframeSpool, Schema2CarriesTheAlbedoProxyPhotoAndLamp) {
    const fs::path d = TempDir("v2");
    reconstruction::KeyframeSpool sp(d, {1.0, 1, 10});
    const cv::Mat z(4, 6, CV_32F, cv::Scalar(2.0f)), rgb(4, 6, CV_8UC3, cv::Scalar(1, 2, 3));
    cv::Mat albedo(4, 6, CV_32F, cv::Scalar(0.25f)), valid(4, 6, CV_8U, cv::Scalar(255));
    valid.at<uchar>(2, 3) = 0;
    reconstruction::SpoolLamp lamp;
    lamp.calib_source = std::string(ARIADNE_SOURCE_DIR) + "/configs/sim/lamp_calib_v2.yaml";
    lamp.calib_name = "lamp_calib_v2";
    lamp.rig = "rig_stereo_v2";
    lamp.intensity = 3.0;
    lamp.R_rect_raw = cv::Matx33d(0, -1, 0, 1, 0, 0, 0, 0, 1);
    sp.SetLamp(lamp);
    const auto cam = sp.Camera(10, 10, 2.5, 1.5, 6, 4);
    sp.AddKeyframe(0, 4, 40, 1, core::SE3(), cam, 0.1, rgb, z, z, albedo, valid,
                   reconstruction::SpoolPhoto{2.0, 0.5, 7});
    sp.AddHeldout(0, 5, 50, core::SE3(), cam, rgb, reconstruction::SpoolPhoto{1.5, 0.5, 7});
    sp.Finish("run", "scene", false, core::SE3(), "none");

    const YAML::Node y = YAML::LoadFile((d / "spool.yaml").string());
    EXPECT_EQ(y["schema_version"].as<int>(), 2);
    EXPECT_EQ(y["lamp"]["calibration"].as<std::string>(), "lamp_calibration.yaml");
    EXPECT_TRUE(fs::exists(d / "lamp_calibration.yaml"));
    EXPECT_DOUBLE_EQ(y["lamp"]["intensity"].as<double>(), 3.0);
    ASSERT_EQ(y["lamp"]["R_rect_raw"].size(), 9u);
    EXPECT_DOUBLE_EQ(y["lamp"]["R_rect_raw"][1].as<double>(), -1.0);   // row-major
    const YAML::Node kf = y["bundles"][0], ho = y["bundles"][1];
    EXPECT_EQ(kf["rgb_kind"].as<std::string>(), "albedo_proxy");
    EXPECT_DOUBLE_EQ(kf["exposure_state"]["exposure_gain"].as<double>(), 2.0);
    EXPECT_EQ(kf["exposure_state"]["decision_id"].as<uint64_t>(), 7u);
    EXPECT_DOUBLE_EQ(kf["lamp_state"]["lamp_rel"].as<double>(), 0.5);
    EXPECT_EQ(ho["rgb_kind"].as<std::string>(), "raw_radiance");
    EXPECT_DOUBLE_EQ(ho["exposure_state"]["exposure_gain"].as<double>(), 1.5);

    const cv::Mat bits = cv::imread((d / "agent0/kf_4_albedo.png").string(), cv::IMREAD_UNCHANGED);
    ASSERT_EQ(bits.type(), CV_16U);
    cv::Mat back;
    cv::Mat(bits.size(), CV_16F, bits.data, bits.step).convertTo(back, CV_32F);
    EXPECT_FLOAT_EQ(back.at<float>(0, 0), 0.25f);
    EXPECT_TRUE(std::isnan(back.at<float>(2, 3)));   // not defined there: NaN, never a guessed reflectance

    EXPECT_THROW(sp.AddKeyframe(0, 6, 60, 1, core::SE3(), cam, 0.1, rgb, z, z, albedo, cv::Mat()),
                 std::invalid_argument);   // an albedo without its valid mask
}
