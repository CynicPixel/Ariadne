// EuRoC reader: calibration parsing, frame convention, stereo pairing.
// Fixtures are built from the PUBLISHED calibration numbers (the Kalibr
// T_imu_cam in OpenVINS's config/euroc_mav, identical to EuRoC's T_BS), not from
// copied dataset files, so these tests need no download.
#include "runtime/datasets/euroc.hpp"
#include "runtime/metrics/trajectory_io.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"

#include <gtest/gtest.h>

#include <fstream>

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

// Kalibr's "equidistant" (KB4) is accepted and tagged, so the rectifier takes the cv::fisheye path (D143); any other
// model name is refused rather than read as radtan.
TEST(Euroc, AcceptsEquidistantAndRefusesUnknownModels) {
    std::string y = CameraYaml(kCam0TBS, "567.76, 567.13, 651.56, 402.91", "-0.00326, 0.00433, 0.00356, -0.00202");
    const auto at = y.find("radial-tangential");
    const EurocCamera c = ParseEurocCamera(y.replace(at, 17, "equidistant"), "cam0");
    EXPECT_EQ(c.intrinsics.model, perception::DistortionModel::kEquidistant);
    EXPECT_DOUBLE_EQ(c.intrinsics.dist[2], 0.00356);
    EXPECT_EQ(Cam0().intrinsics.model, perception::DistortionModel::kRadtan);
    std::string bad = CameraYaml(kCam0TBS, "1, 1, 1, 1", "0, 0, 0, 0");
    EXPECT_THROW(ParseEurocCamera(bad.replace(bad.find("radial-tangential"), 17, "fov"), "cam0"), ConfigError);
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
    // equidistant is supported since D143 (AcceptsEquidistantAndRefusesUnknownModels); an unknown model still throws.
    std::string omni = CameraYaml(kCam0TBS, "1, 1, 1, 1", "0, 0, 0, 0");
    omni.replace(omni.find("radial-tangential"), 17, "omni");
    EXPECT_THROW(ParseEurocCamera(omni, "x"), ConfigError);
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

// --- Ground truth, association, frames (1.5) --------------------------------

TEST(Euroc, ParsesGroundTruthRows) {
    const auto gt = ParseEurocGroundTruth(
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v...\n"
        "1403636580838555648,4.688319,-1.786938,0.783338,0.534108,-0.153029,-0.827383,-0.082152,-0.027876,"
        "0.033207,0.800006,-0.003172,0.021267,0.078502,-0.025266,0.136696,0.075593\n",
        "gt");
    ASSERT_EQ(gt.size(), 1u);
    EXPECT_EQ(gt[0].t_ns, 1403636580838555648LL);
    EXPECT_DOUBLE_EQ(gt[0].T_W_B.t.x(), 4.688319);
    const Eigen::Quaterniond q(gt[0].T_W_B.R);   // w-first in the file, and unit after normalisation
    EXPECT_NEAR(q.w(), 0.534108, 1e-5);
    EXPECT_NEAR(q.y(), -0.827383, 1e-5);
}

TEST(Euroc, AssociatesNearestWithinEvoDefault) {
    std::vector<TimedPose> ref{{0, {}}, {5'000'000, {}}, {10'000'000, {}}, {40'000'000, {}}};
    std::vector<TimedPose> est{{4'000'000, {}}, {9'999'999, {}}, {25'000'001, {}}, {100'000'000, {}}};
    const auto idx = AssociateNearest(est, ref);
    EXPECT_EQ(idx[0], 1);
    EXPECT_EQ(idx[1], 2);
    EXPECT_EQ(idx[2], -1);   // 15 ms from the nearest: beyond evo's 0.01 s
    EXPECT_EQ(idx[3], -1);
}

TEST(Euroc, WritesEvoCompatibleTum) {
    core::SE3 T;
    T.t = core::Vec3(1, 2, 3);
    const std::string path = ::testing::TempDir() + "/traj.tum";
    ASSERT_TRUE(WriteTum(path, {{1403636580838555648LL, T}}));
    std::ifstream in(path);
    std::string header, row;
    std::getline(in, header);
    std::getline(in, row);
    EXPECT_EQ(row, "1403636580.838555648 1.000000000 2.000000000 3.000000000 0.000000000 0.000000000 0.000000000 1.000000000");
}

// The frame trap: poses are of the RECTIFIED left camera. Converting back to
// the body with R1 recovers the body trajectory exactly; omitting R1 leaves a
// constant rotation that SE(3) alignment cannot remove.
TEST(Euroc, RectifiedCameraToBodyRoundTripNeedsR1) {
    const core::SE3 T_B_C0 = Cam0().T_B_C;
    const core::Mat3 R1 = core::SO3Exp(core::Vec3(0.004, -0.012, 0.003));   // ~0.75 deg
    const core::SE3 T_B_Crect = BodyFromRectifiedLeft(T_B_C0, R1);
    std::vector<core::TrajectorySample> truth, with_r1, without_r1;
    for (int k = 0; k < 50; ++k) {
        core::SE3 T_W_B;
        T_W_B.R = core::SO3Exp(core::Vec3(0.1 * std::sin(0.1 * k), 0.2 * std::cos(0.07 * k), 0.05 * k));
        T_W_B.t = core::Vec3(std::sin(0.1 * k), 0.1 * k, 0.3 * std::cos(0.05 * k));
        const core::SE3 T_W_Crect = T_W_B * T_B_Crect;   // what the VO estimates
        truth.push_back({core::Timestamp{k}, 0, T_W_B});
        with_r1.push_back({core::Timestamp{k}, 0, T_W_Crect * T_B_Crect.inverse()});
        without_r1.push_back({core::Timestamp{k}, 0, T_W_Crect * T_B_C0.inverse()});
    }
    EXPECT_LT(ComputeATE(with_r1, truth).rmse, 1e-12);
    EXPECT_LT(ComputeATERotation(with_r1, truth).rmse, 1e-12);
    EXPECT_GT(ComputeATERotation(without_r1, truth).rmse, 0.005);   // the constant R1 offset survives alignment
}

// --- IMU (plan 2.4) ---------------------------------------------------------------------------------------------------

TEST(EurocImu, ParsesRowsAndRejectsMalformedOnes) {
    const std::string csv =
        "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],"
        "a_RS_S_z [m s^-2]\r\n"
        "1403636579758555392,-0.0991,0.1473,0.0272,8.1477,-0.3759,-2.4026\r\n"
        "1403636579763555584,-0.0991,0.1403,0.0293,8.0333,-0.4086,-2.4026\r\n";
    const auto imu = ParseEurocImu(csv, "imu0");
    ASSERT_EQ(imu.size(), 2u);
    EXPECT_EQ(imu[1].t_ns, 1403636579763555584LL);
    EXPECT_DOUBLE_EQ(imu[0].gyro.y(), 0.1473);
    EXPECT_DOUBLE_EQ(imu[1].accel.x(), 8.0333);
    EXPECT_THROW(ParseEurocImu("1,2,3\n", "imu0"), ConfigError);
    EXPECT_THROW(ParseEurocImu("2,0,0,0,0,0,0\n1,0,0,0,0,0,0\n", "imu0"), ConfigError);   // time going back
}

TEST(EurocImu, NoiseModelRequiresIdentityExtrinsic) {
    const std::string ok =
        "T_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]\n"
        "rate_hz: 200\ngyroscope_noise_density: 1.6968e-04\ngyroscope_random_walk: 1.9393e-05\n"
        "accelerometer_noise_density: 2.0000e-3\naccelerometer_random_walk: 3.0000e-3\n";
    const core::ImuNoise n = ParseEurocImuNoise(ok, "imu0");
    EXPECT_DOUBLE_EQ(n.gyro_noise_density, 1.6968e-04);
    EXPECT_DOUBLE_EQ(n.accel_random_walk, 3.0e-3);
    EXPECT_DOUBLE_EQ(n.rate_hz, 200.0);
    std::string moved = ok;
    moved.replace(moved.find("1.0, 0.0, 0.0, 0.0, 0.0"), 3, "0.9");
    EXPECT_THROW(ParseEurocImuNoise(moved, "imu0"), ConfigError);
}

// Constant angular rate: the integral over [t0, t1] is exp(w (t1 - t0)) whatever the sample phase, because the first
// and last intervals are clipped; a bias equal to the rate integrates to identity.
TEST(ImuIntegration, GyroIntegralClipsToTheIntervalAndRemovesBias) {
    const core::Vec3 w(0.3, -0.2, 0.5);
    std::vector<core::ImuMeasurement> s;
    for (int k = 0; k <= 40; ++k) s.push_back({1000000000LL + k * 5000000LL, w, core::Vec3::Zero()});
    const int64_t t0 = 1000000000LL + 12345678, t1 = 1000000000LL + 151234567;
    const core::Mat3 R = core::IntegrateGyro(s, t0, t1, core::Vec3::Zero());
    EXPECT_LT((core::SO3Log(R) - w * (1e-9 * double(t1 - t0))).norm(), 1e-12);
    EXPECT_LT(core::SO3Log(core::IntegrateGyro(s, t0, t1, w)).norm(), 1e-15);
    EXPECT_THROW(core::IntegrateGyro(s, t0 - 100000000LL, t1, core::Vec3::Zero()), std::invalid_argument);
}
