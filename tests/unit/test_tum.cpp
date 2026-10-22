// 2.6 (D100): the TUM RGB-D loader and the depth -> virtual-stereo path, on synthetic data.
#include "perception/rgbd/depth_stereo.hpp"
#include "perception/stereo/stereo_rectifier.hpp"
#include "runtime/datasets/tum.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using namespace ariadne;

TEST(Tum, TimestampsParseExactlyWithoutFloatingPoint) {
    EXPECT_EQ(runtime::ParseTumTimestampNs("1305031102.175304"), 1305031102175304000LL);
    EXPECT_EQ(runtime::ParseTumTimestampNs("1305031102.1"), 1305031102100000000LL);
    EXPECT_EQ(runtime::ParseTumTimestampNs("7"), 7000000000LL);
    EXPECT_THROW(runtime::ParseTumTimestampNs("1.2e3"), core::ConfigError);
    EXPECT_THROW(runtime::ParseTumTimestampNs("1.0123456789"), core::ConfigError);
}

TEST(Tum, GroundTruthQuaternionIsXyzwOnFile) {
    // 90 deg about z: qx qy qz qw = 0 0 sin45 cos45. Read w-first it would be a different rotation.
    const auto gt = runtime::ParseTumGroundTruth("# c\n2.0 1 2 3 0 0 0.7071067811865476 0.7071067811865476\n"
                                                 "1.0 0 0 0 0 0 0 1\n");
    ASSERT_EQ(gt.size(), 2u);
    EXPECT_EQ(gt[0].t_ns, 1000000000LL);   // sorted
    const core::Vec3 x = gt[1].T.R * core::Vec3(1, 0, 0);
    EXPECT_NEAR(x.x(), 0.0, 1e-9);
    EXPECT_NEAR(x.y(), 1.0, 1e-9);
    EXPECT_EQ(gt[1].T.t, core::Vec3(1, 2, 3));
}

TEST(Tum, AssociationIsGreedyOneToOneWithin20ms) {
    using runtime::TumStamped;
    const std::vector<TumStamped> rgb{{1000000000, "a"}, {1033000000, "b"}, {1066000000, "c"}, {2000000000, "d"}};
    const std::vector<TumStamped> dep{{1001000000, "A"}, {1030000000, "B"}, {1050000000, "C"}};
    const auto p = runtime::AssociateTum(rgb, dep);
    // a-A (1 ms), b-B (3 ms), c-C (16 ms); d has no depth within 20 ms.
    ASSERT_EQ(p.size(), 3u);
    EXPECT_EQ(p[0], (std::pair<size_t, size_t>{0, 0}));
    EXPECT_EQ(p[1], (std::pair<size_t, size_t>{1, 1}));
    EXPECT_EQ(p[2], (std::pair<size_t, size_t>{2, 2}));
}

TEST(Tum, SigmaModelIsKhoshelhamElberinkAndMatchesTheVirtualRig) {
    const perception::DepthSensorModel m;
    EXPECT_NEAR(m.SigmaZ(5.0), 0.035625, 1e-9);   // eq. 7 with |m/fb| = 2.85e-5 /cm, sigma_d' = 1/2 px
    // The stereo law on the virtual rig reproduces it at every depth: Z^2 / (fx B) * sigma_d = k Z^2.
    const double fx = 517.3, sd = m.DisparitySigmaPx(fx);
    for (double z : {0.5, 1.0, 3.0}) EXPECT_NEAR(z * z / (fx * m.virtual_baseline_m) * sd, m.SigmaZ(z), 1e-12);
}

TEST(Tum, VirtualStereoReturnsSensorDepthAndMasksEdgesAndRange) {
    const perception::DepthSensorModel m;
    cv::Mat d(60, 80, CV_32F, cv::Scalar(2.0f));
    d(cv::Rect(40, 0, 40, 60)) = 3.0f;    // a depth edge at x = 40
    d(cv::Rect(0, 50, 10, 10)) = 6.0f;    // beyond z_max
    d.at<float>(20, 10) = 0.0f;           // a hole
    const double fx = 500.0;
    const auto v = perception::VirtualStereoFromDepth(
        d, {{20.f, 20.f}, {39.f, 20.f}, {60.f, 20.f}, {5.f, 55.f}, {11.f, 20.f}, {-3.f, 4.f}}, fx, m);
    ASSERT_EQ(v.valid.size(), 6u);
    EXPECT_EQ(v.valid[0], 1);
    EXPECT_NEAR(fx * m.virtual_baseline_m / (20.0 - v.right[0].x), 2.0, 1e-5);   // triangulates back to Z
    EXPECT_FLOAT_EQ(v.right[0].y, 20.f);
    EXPECT_NEAR(v.sigma_z[0], m.SigmaZ(2.0), 1e-7);
    EXPECT_EQ(v.valid[1], 0);   // next to the edge (3/2 > 1.05)
    EXPECT_EQ(v.valid[2], 1);
    EXPECT_FLOAT_EQ(v.z[2], 3.0f);
    EXPECT_EQ(v.valid[3], 0);   // out of range
    EXPECT_EQ(v.valid[4], 0);   // next to a hole
    EXPECT_EQ(v.valid[5], 0);   // outside the image
    EXPECT_TRUE(std::isnan(v.z[5]));
}

TEST(Tum, SourceReadsDepthAtFactor5000AndReportsAnIdentityVirtualRig) {
    // freiburg3 in the name: zero distortion, so pixels pass through the source's undistortion unchanged.
    const fs::path dir = fs::temp_directory_path() / "ariadne_test_tum" / "rgbd_dataset_freiburg3_synthetic";
    fs::remove_all(dir.parent_path());
    fs::create_directories(dir / "rgb");
    fs::create_directories(dir / "depth");
    cv::Mat bgr(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat d16(480, 640, CV_16U, cv::Scalar(7500));   // 1.5 m
    ASSERT_TRUE(cv::imwrite((dir / "rgb/1.000000.png").string(), bgr));
    ASSERT_TRUE(cv::imwrite((dir / "depth/1.005000.png").string(), d16));
    std::ofstream(dir / "rgb.txt") << "# rgb\n1.000000 rgb/1.000000.png\n";
    std::ofstream(dir / "depth.txt") << "# depth\n1.005000 depth/1.005000.png\n";
    std::ofstream(dir / "groundtruth.txt") << "# gt\n0.99 0 0 0 0 0 0 1\n1.01 0 0 0 0 0 0 1\n";

    runtime::TumSource src(dir.string());
    ASSERT_EQ(src.FrameCount(), 1u);
    ASSERT_TRUE(src.DepthSensor().has_value());
    EXPECT_EQ(src.GroundTruth().size(), 2u);
    int seen = 0;
    src.ForEachFrame(0, [&](const runtime::StereoFrame& f) {
        ++seen;
        EXPECT_EQ(f.t_ns, 1000000000LL);   // stamped with the RGB time
        EXPECT_TRUE(f.right.empty());
        ASSERT_EQ(f.depth.type(), CV_32F);
        EXPECT_FLOAT_EQ(f.depth.at<float>(240, 320), 1.5f);
        EXPECT_EQ(f.left_rgb.at<cv::Vec3b>(240, 320), cv::Vec3b(30, 20, 10));   // RGB order
    });
    EXPECT_EQ(seen, 1);

    // The virtual rig rectifies to the identity: R1 = I, baseline = B, fx unchanged. A sign error on the
    // translation would rotate the rectified frame or flip the baseline.
    const perception::StereoRectifier rect(src.Calibration(), 0.0);
    const cv::Mat R1 = rect.R1();
    EXPECT_LT(cv::norm(R1 - cv::Mat::eye(3, 3, R1.type())), 1e-9);
    EXPECT_NEAR(rect.Rig().baseline_m, src.DepthSensor()->virtual_baseline_m, 1e-9);
    EXPECT_NEAR(rect.Rig().fx, 535.4, 5.0);   // stereoRectify sets one focal for both axes
    fs::remove_all(dir.parent_path());
}

TEST(Tum, DenseDepthHasTheKeyframeRecordContract) {
    const perception::DepthSensorModel m;
    cv::Mat d(40, 50, CV_32F, cv::Scalar(2.0f));
    d.at<float>(10, 10) = 0.0f;
    const perception::StereoDepth sd = perception::DenseDepthFromSensor(d, 500.0, m);
    EXPECT_EQ(sd.Z.type(), CV_16F);
    EXPECT_EQ(sd.sigma_Z.type(), CV_16F);
    EXPECT_EQ(sd.disparity.type(), CV_16S);
    EXPECT_EQ(sd.valid.at<uchar>(20, 20), 255);
    EXPECT_EQ(sd.valid.at<uchar>(10, 11), 0);   // beside the hole
    cv::Mat z, s;
    sd.Z.convertTo(z, CV_32F);
    sd.sigma_Z.convertTo(s, CV_32F);
    EXPECT_FLOAT_EQ(z.at<float>(20, 20), 2.0f);
    EXPECT_EQ(z.at<float>(10, 11), 0.0f);
    EXPECT_NEAR(s.at<float>(20, 20), m.SigmaZ(2.0), 1e-5);
    EXPECT_EQ(sd.disparity.at<short>(20, 20), cvRound(16.0 * 500.0 * m.virtual_baseline_m / 2.0));
    EXPECT_EQ(sd.disparity.at<short>(10, 10), -16);
}
