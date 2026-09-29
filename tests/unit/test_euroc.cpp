// EuRoC reader: calibration parsing, frame convention, stereo pairing.
// Fixtures are built from the PUBLISHED calibration numbers (the Kalibr
// T_imu_cam in OpenVINS's config/euroc_mav, identical to EuRoC's T_BS), not from
// copied dataset files, so these tests need no download.
#include "runtime/datasets/euroc.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace ariadne;
using namespace ariadne::runtime;

namespace {

std::string CameraYaml(const std::string& t_bs, const std::string& k, const std::string& d) {
    return "%YAML:1.0\n"
           "sensor_type: camera\n"
           "comment: VI-Sensor cam (MT9M034)\n"
           "T_BS:\n  cols: 4\n  rows: 4\n  data: [" + t_bs + "]\n"
           "rate_hz: 20\n"
           "resolution: [752, 480]\n"
           "camera_model: pinhole\n"
           "intrinsics: [" + k + "]\n"
           "distortion_model: radial-tangential\n"
           "distortion_coefficients: [" + d + "]\n";
}

const std::string kCam0TBS =
    "0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,"
    "0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,"
    "-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,"
    "0.0, 0.0, 0.0, 1.0";
const std::string kCam1TBS =
    "0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,"
    "0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,"
    "-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,"
    "0.0, 0.0, 0.0, 1.0";

EurocCamera Cam0() {
    return ParseEurocCamera(CameraYaml(kCam0TBS, "458.654, 457.296, 367.215, 248.375",
                                       "-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05"),
                            "cam0");
}
EurocCamera Cam1() {
    return ParseEurocCamera(CameraYaml(kCam1TBS, "457.587, 456.134, 379.999, 255.238",
                                       "-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05"),
                            "cam1");
}

}  // namespace

TEST(Euroc, ParsesSensorYamlIncludingTheOpenCvDirective) {
    const EurocCamera c = Cam0();
    EXPECT_EQ(c.intrinsics.width, 752);
    EXPECT_EQ(c.intrinsics.height, 480);
    EXPECT_DOUBLE_EQ(c.intrinsics.fu, 458.654);
    EXPECT_DOUBLE_EQ(c.intrinsics.dist[3], 1.76187114e-05);
    EXPECT_DOUBLE_EQ(c.rate_hz, 20.0);
}

// THE convention test. ORB-SLAM3 publishes the same rig as Stereo.T_c1_c2
// (right camera expressed in the left camera's frame). Composing the Kalibr
// T_BS values must reproduce it; a flipped composition would not.
TEST(Euroc, StereoExtrinsicMatchesIndependentlyPublishedValue) {
    const core::SE3 T_c0_c1 = Cam0().T_B_C.inverse() * Cam1().T_B_C;
    core::Mat4 orbslam3;
    orbslam3 << 0.999997256477797, -0.002317135723275, -0.000343393120620, 0.110074137800478,
        0.002312067192432, 0.999898048507103, -0.014090668452683, -0.000156612054392,
        0.000376008102320, 0.014089835846691, 0.999900662638081, 0.000889382785432, 0, 0, 0, 1;
    EXPECT_LT((T_c0_c1.matrix() - orbslam3).cwiseAbs().maxCoeff(), 1e-6);

    // What stereoRectify receives: left -> right. The left camera sits at
    // x = -0.110 m in the right camera's frame.
    const perception::RawStereoCalibration s = EurocStereo(Cam0(), Cam1());
    EXPECT_NEAR(s.T_right_left.t.x(), -0.110, 0.001);
    EXPECT_NEAR(s.T_right_left.t.norm(), 0.1101, 0.0005);
}

TEST(Euroc, RejectsNonRigidOrUnsupportedCalibration) {
    const std::string bad_r =
        "2.0, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1";
    EXPECT_THROW(ParseEurocCamera(CameraYaml(bad_r, "1, 1, 1, 1", "0, 0, 0, 0"), "x"), ConfigError);
    std::string fisheye = CameraYaml(kCam0TBS, "1, 1, 1, 1", "0, 0, 0, 0");
    fisheye.replace(fisheye.find("radial-tangential"), 17, "equidistant");
    EXPECT_THROW(ParseEurocCamera(fisheye, "x"), ConfigError);
    // Strictness: an unexpected key is an error, not silently ignored.
    EXPECT_THROW(ParseEurocCamera(CameraYaml(kCam0TBS, "1, 1, 1, 1", "0, 0, 0, 0") + "rolling_shutter: 1\n",
                                  "x"),
                 ConfigError);
}

TEST(Euroc, PairsByExactTimestampNotByRow) {
    const EurocImageIndex l = ParseEurocImageCsv(
        "#timestamp [ns],filename\n100,100.png\n150,150.png\n200,200.png\n300,300.png\n", "l");
    // The right camera dropped the 150 frame and has an extra one at 250.
    const EurocImageIndex r =
        ParseEurocImageCsv("#timestamp [ns],filename\r\n100,100.png\r\n200,200.png\r\n250,250.png\r\n300,300.png\r\n", "r");
    const StereoPairs p = PairByTimestamp(l, r);
    ASSERT_EQ(p.t_ns.size(), 3u);
    EXPECT_EQ(p.t_ns[1], 200);
    EXPECT_EQ(p.left[1], "200.png");
    EXPECT_EQ(p.right[1], "200.png");
    EXPECT_EQ(p.unmatched_left, 1);
    EXPECT_EQ(p.unmatched_right, 1);
}

TEST(Euroc, RejectsNonIncreasingTimestamps) {
    EXPECT_THROW(ParseEurocImageCsv("1,a.png\n1,b.png\n", "x"), ConfigError);
}
