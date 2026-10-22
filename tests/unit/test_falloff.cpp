// 5.2 (D134): Stage 3 falloff normalisation -> albedo proxy, on synthetic scenes lit by a known lamp model.
#include "perception/radiometric/falloff.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace rm = ariadne::perception::radiometric;

namespace {

rm::LampCalibration Lamp(bool cos_incidence = true) {
    rm::LampCalibration l;
    l.lamp_in_camera = {0.05, -0.02, 0.02};
    l.axis = {0, 0, 1};
    l.gain_A = 0.33;
    l.c_prime_m2 = 0.5;
    l.cos_incidence = cos_incidence;
    l.calib_albedo = 0.5;
    l.rid_theta_deg = {0.0, 25.0, 40.0};   // flat top, linear roll-off to 0 at 40 deg (the renderer's shape, D131)
    l.rid_value = {1.0, 1.0, 0.0};
    return l;
}

const cv::Matx33d K(640, 0, 319.5, 0, 640, 239.5, 0, 0, 1);

// A plane z = z0 + slope * y_cam: depth per pixel, and the image the lamp model predicts for an albedo checker.
struct Synthetic {
    cv::Mat z, gray, albedo;
};

Synthetic Plane(double z0, double slope, const rm::LampCalibration& l, double lamp_rel) {
    const int w = 640, h = 480;
    Synthetic s{cv::Mat(h, w, CV_32F), cv::Mat(h, w, CV_32F, cv::Scalar(0)), cv::Mat(h, w, CV_32F)};
    // Plane n.X = d with n = (0, -slope, 1) normalised: Z = d / (n . ray).
    cv::Vec3d n(0, -slope, 1);
    n = cv::normalize(n);
    const double d = n[2] * z0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const cv::Vec3d ray((x - K(0, 2)) / K(0, 0), (y - K(1, 2)) / K(1, 1), 1.0);
            const double zz = d / n.dot(ray);
            const cv::Vec3d X = ray * zz;
            s.z.at<float>(y, x) = static_cast<float>(zz);
            const double a = 0.2 + 0.6 * (((x / 16) + (y / 16)) % 2);
            s.albedo.at<float>(y, x) = static_cast<float>(a);
            const cv::Vec3d L = X - l.lamp_in_camera;
            const double r = cv::norm(L);
            const double rid = l.Rid(std::acos(L.dot(l.axis) / r) * 180.0 / M_PI);
            cv::Vec3d nn = n;
            if (nn.dot(X) > 0) nn = -nn;
            const double ci = l.cos_incidence ? std::max(0.0, nn.dot(-L) / r) : 1.0;
            s.gray.at<float>(y, x) = static_cast<float>(a / l.calib_albedo * l.gain_A * lamp_rel * rid * ci /
                                                        (l.c_prime_m2 + r * r));
        }
    return s;
}

rm::FalloffParams MaxGain(double g) {
    rm::FalloffParams p;
    p.max_gain = g;
    return p;
}

}  // namespace

TEST(Falloff, RecoversAlbedoOnATiltedPlane) {
    const auto lamp = Lamp(true);
    const auto s = Plane(2.0, 0.4, lamp, 1.0);
    const auto out = rm::FalloffNormalize(s.gray, s.z, K, lamp, 1.0, MaxGain(1e6));
    int n = 0, bad = 0;
    for (int y = 2; y < s.z.rows - 2; ++y)
        for (int x = 2; x < s.z.cols - 2; ++x) {
            if (!out.valid.at<uint8_t>(y, x)) continue;
            // Checker edges: the normal is fine (a plane) but skip nothing; compare everywhere valid.
            ++n;
            if (std::abs(out.albedo.at<float>(y, x) - s.albedo.at<float>(y, x)) > 1e-3) ++bad;
        }
    EXPECT_GT(n, 100000);
    EXPECT_EQ(bad, 0);
}

TEST(Falloff, TheSameSurfaceReadsTheSameAtAnyDistanceAndLampPower) {
    const auto lamp = Lamp(true);
    const auto near = rm::FalloffNormalize(Plane(1.0, 0.0, lamp, 1.0).gray, Plane(1.0, 0.0, lamp, 1.0).z, K, lamp, 1.0);
    const auto s = Plane(3.5, 0.0, lamp, 0.25);
    const auto far = rm::FalloffNormalize(s.gray, s.z, K, lamp, 0.25);
    // Centre pixel: raw brightness differs ~40x, the proxy agrees.
    const cv::Point c(320, 240);
    EXPECT_NEAR(near.albedo.at<float>(c), far.albedo.at<float>(c), 1e-4);
}

TEST(Falloff, UnlitAndUnknownPixelsAreInvalidAndTheGainIsCapped) {
    const auto lamp = Lamp(true);
    auto s = Plane(2.0, 0.0, lamp, 1.0);
    s.z.at<float>(240, 320) = 0.0f;                         // no depth
    s.z.at<float>(100, 100) = std::numeric_limits<float>::quiet_NaN();
    const auto out = rm::FalloffNormalize(s.gray, s.z, K, lamp, 1.0, MaxGain(2.0));
    EXPECT_EQ(out.valid.at<uint8_t>(240, 320), 0);
    EXPECT_EQ(out.valid.at<uint8_t>(240, 328), 0);           // normal_step_px (8) away: no normal there
    EXPECT_EQ(out.valid.at<uint8_t>(100, 100), 0);
    // The image corner is ~41 deg off the lamp axis at 2 m: outside the lobe.
    EXPECT_EQ(out.valid.at<uint8_t>(0, 0), 0);
    // With max_gain 2 a pixel lit at under half the frame's peak is divided by the floor instead: biased dark.
    double minr = 1e9;
    for (int y = 0; y < s.z.rows; ++y)
        for (int x = 0; x < s.z.cols; ++x)
            if (out.valid.at<uint8_t>(y, x))
                minr = std::min(minr, double(out.albedo.at<float>(y, x) / s.albedo.at<float>(y, x)));
    EXPECT_LT(minr, 0.99);
    EXPECT_GE(minr, 0.0);
}

TEST(Falloff, LoadsTheCalibrationYaml) {
    const std::string path = ::testing::TempDir() + "/lamp_calib_test.yaml";
    std::ofstream(path) << "name: t\nrig: rig_stereo_v2\nrender_transfer: srgb\n"
                           "lamp_in_camera_optical_m: [0.05, -0.02, 0.02]\nlamp_axis_optical: [0.0, 0.0, 2.0]\n"
                           "intensity_ref: 6.0\ncalib_albedo: 0.5\ngain_A: 0.9\nc_prime_m2: 0.5\ncos_incidence: true\n"
                           "rid: {theta_deg: [0.5, 1.5, 30.0], value: [1.0, 0.9, 0.1]}\nfit: {rms_log_all: 0.01}\n";
    const auto l = rm::LoadLampCalibration(path);
    EXPECT_NEAR(l.axis[2], 1.0, 1e-12);                     // normalised
    EXPECT_DOUBLE_EQ(l.Rid(0.0), 1.0);
    EXPECT_NEAR(l.Rid(1.0), 0.95, 1e-12);
    EXPECT_DOUBLE_EQ(l.Rid(31.0), 0.0);
    std::ofstream(path, std::ios::app) << "typo_key: 1\n";
    EXPECT_THROW(rm::LoadLampCalibration(path), std::exception);
    std::remove(path.c_str());
}

// Stereo depth, not perfect depth: disparity at f = 640 px, B = 0.1 m quantised to SGBM's 1/16 px plus 0.1 px noise.
// Longer normal baselines beat 1 px differences; the numbers go to D134 for the incidence row.
TEST(Falloff, NoisyStereoDepthNormalsAndTheIncidenceOption) {
    const auto lamp = Lamp(true);
    const auto s = Plane(2.0, 0.4, lamp, 1.0);
    cv::Mat zn = s.z.clone();
    cv::RNG rng(7);
    for (int y = 0; y < zn.rows; ++y)
        for (int x = 0; x < zn.cols; ++x) {
            const double d = 640.0 * 0.1 / zn.at<float>(y, x) + rng.gaussian(0.1);
            zn.at<float>(y, x) = static_cast<float>(64.0 / (std::round(d * 16.0) / 16.0));
        }
    const auto med_err = [&](const rm::FalloffParams& fp) {
        const auto out = rm::FalloffNormalize(s.gray, zn, K, lamp, 1.0, fp);
        std::vector<float> e;
        for (int y = 0; y < zn.rows; ++y)
            for (int x = 0; x < zn.cols; ++x)
                if (out.valid.at<uint8_t>(y, x))
                    e.push_back(std::abs(out.albedo.at<float>(y, x) / s.albedo.at<float>(y, x) - 1.0f));
        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
        return double(e[e.size() / 2]);
    };
    auto fp = MaxGain(1e6);
    fp.normal_step_px = 1;
    const double k1 = med_err(fp);
    fp.normal_step_px = 3;
    const double k3 = med_err(fp);
    fp.normal_step_px = 8;
    const double k8 = med_err(fp);
    fp.incidence = "off";
    const double off = med_err(fp);
    std::printf("[ falloff noisy depth ] median |albedo error|: k=1 %.4f  k=3 %.4f  k=8 %.4f  incidence off %.4f\n",
                k1, k3, k8, off);
    EXPECT_LT(k3, k1);
    EXPECT_LT(k8, off);   // with the default step, dividing by the estimated cos beats keeping the shading
}
