// 5.2 (D146): Stage 3 as wired into the runner -- the falloff: config section, the refusals, and the Stage 4 /
// lamp-power arithmetic between the source's photometric state and FalloffNormalize.
#include "perception/perception_config.hpp"
#include "runtime/tools/falloff_glue.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace rm = ariadne::perception::radiometric;
using ariadne::runtime::FramePhoto;
using ariadne::runtime::LampRig;
using ariadne::tools::KeyframeFalloff;

namespace {

const std::string kCalib = std::string(ARIADNE_SOURCE_DIR) + "/configs/sim/lamp_calib_v2.yaml";
const cv::Matx33d K(640, 0, 319.5, 0, 640, 239.5, 0, 0, 1);
const cv::Matx33d I3 = cv::Matx33d::eye();

rm::FalloffStageParams On() {
    rm::FalloffStageParams p;
    p.mode = "keyframe";
    return p;
}

// The calibration's own rig, lamp where the calibration has it (camera = body here), the only light.
LampRig Rig(const rm::LampCalibration& c) {
    LampRig l;
    l.rig = c.rig;
    l.intensity = c.intensity_ref;
    l.in_body_m = ariadne::core::Vec3(c.lamp_in_camera[0], c.lamp_in_camera[1], c.lamp_in_camera[2]);
    l.lamp_only = true;
    return l;
}

rm::FalloffStageParams Parse(const std::string& yaml) {
    auto s = ariadne::core::StrictConfig::FromString(yaml);
    return rm::ParseFalloff(s);
}

}  // namespace

TEST(FalloffGlue, ConfigSection) {
    EXPECT_FALSE(rm::FalloffStageParams{}.Enabled());   // absent section: the stage does not exist
    const auto p = Parse("mode: keyframe\nincidence: off\nnormals: rgbd_linemod\nnormal_window_px: 5\n");
    EXPECT_TRUE(p.Enabled());
    EXPECT_EQ(p.params.incidence, "off");
    EXPECT_EQ(p.params.normals, "rgbd_linemod");
    EXPECT_EQ(p.params.normal_window_px, 5);
    EXPECT_THROW(Parse("mode: every_frame\n"), ariadne::core::ConfigError);   // the ablation row is not built yet
    EXPECT_THROW(Parse("mode: keyframe\nnormal_window_px: 4\n"), ariadne::core::ConfigError);
    EXPECT_THROW(Parse("mode: keyframe\nbogus: 1\n"), ariadne::core::ConfigError);
}

TEST(FalloffGlue, NeedsTheFlatFieldFirst) {
    // A shipped sim config with the radiometric stage on, then the same with it off: falloff needs Stage 1 (§6.2).
    auto c = ariadne::perception::LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/sim_v1_rad_auto.yaml");
    c.falloff = On();
    EXPECT_NO_THROW(ariadne::perception::ValidatePerception(c));
    c.radiometric.mode = "off";
    EXPECT_THROW(ariadne::perception::ValidatePerception(c), ariadne::core::ConfigError);
}

TEST(FalloffGlue, Refusals) {
    const rm::LampCalibration c = rm::LoadLampCalibration(kCalib);
    const ariadne::core::SE3 T;
    const LampRig ok = Rig(c);
    EXPECT_NO_THROW(KeyframeFalloff(On(), kCalib, ok, T, K, I3));
    EXPECT_NO_THROW(KeyframeFalloff(rm::FalloffStageParams{}, "", std::nullopt, T, K, I3));   // off: nothing checked
    EXPECT_THROW(KeyframeFalloff(On(), "", ok, T, K, I3), std::runtime_error);                // no calibration
    EXPECT_THROW(KeyframeFalloff(On(), kCalib, std::nullopt, T, K, I3), std::runtime_error);  // no lamp (EuRoC, Hilti)
    LampRig lit = ok;
    lit.lamp_only = false;   // the fixed-light rows (D145)
    EXPECT_THROW(KeyframeFalloff(On(), kCalib, lit, T, K, I3), std::runtime_error);
    LampRig other = ok;
    other.rig = "rig_stereo_v1";
    EXPECT_THROW(KeyframeFalloff(On(), kCalib, other, T, K, I3), std::runtime_error);
    LampRig moved = ok;
    moved.in_body_m.x() += 0.002;   // 2 mm: another lamp position
    EXPECT_THROW(KeyframeFalloff(On(), kCalib, moved, T, K, I3), std::runtime_error);

    KeyframeFalloff f(On(), kCalib, ok, T, K, I3);
    const cv::Mat g(48, 64, CV_8U, cv::Scalar(100)), z(48, 64, CV_32F, cv::Scalar(2.0f)), s(48, 64, CV_8U, cv::Scalar(255));
    EXPECT_THROW(f.Normalize(g, z, s, std::nullopt), std::runtime_error);   // not known to be linear
}

// A fronto-parallel plane 1.5 m away with an albedo checker, rendered by the calibrated model at lamp intensity
// 3 (half the reference) and D's lamp_scale 0.8, then exposed at exposure x gain = 2 into 8 bits. The glue must
// undo all three: the recovered proxy is the reflectance itself (the calibration plane reads calib_albedo), up to
// 8-bit quantisation.
TEST(FalloffGlue, UndoesExposureAndLampPower) {
    const rm::LampCalibration c = rm::LoadLampCalibration(kCalib);
    LampRig lamp = Rig(c);
    lamp.intensity = 0.5 * c.intensity_ref;
    const FramePhoto photo{2.0, 0.8};
    const double lamp_rel = 0.5 * 0.8;
    const int w = 640, h = 480;
    const double z0 = 1.5;
    cv::Mat gray(h, w, CV_8U), z(h, w, CV_32F, cv::Scalar(z0)), truth(h, w, CV_32F), sup(h, w, CV_8U, cv::Scalar(255));
    const cv::Vec3d axis = c.axis;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const cv::Vec3d X((x - K(0, 2)) / K(0, 0) * z0, (y - K(1, 2)) / K(1, 1) * z0, z0);
            const cv::Vec3d L = X - c.lamp_in_camera;
            const double r = cv::norm(L);
            const double rid = c.Rid(std::acos(L.dot(axis) / r) * 180.0 / M_PI);
            const double ci = c.cos_incidence ? L[2] / r : 1.0;   // plane normal -z, towards the lamp
            const double a = 0.2 + 0.4 * (((x / 16) + (y / 16)) % 2);
            const double I = a / c.calib_albedo * c.gain_A * lamp_rel * rid * ci / (c.c_prime_m2 + r * r);
            gray.at<uchar>(y, x) = cv::saturate_cast<uchar>(255.0 * I * photo.exposure_gain);
            truth.at<float>(y, x) = static_cast<float>(a);
        }
    sup.colRange(0, 40).setTo(0);   // no SGBM support on the left: no depth there, so no albedo either

    KeyframeFalloff f(On(), kCalib, lamp, ariadne::core::SE3(), K, I3);
    const rm::AlbedoProxy a = f.Normalize(gray, z, sup, photo);
    EXPECT_EQ(cv::countNonZero(a.valid.colRange(0, 40)), 0);
    std::vector<double> err;
    for (int y = 0; y < h; ++y)
        for (int x = 40; x < w; ++x)
            if (a.feature_mask.at<uchar>(y, x)) {
                err.push_back(std::abs(a.albedo.at<float>(y, x) / truth.at<float>(y, x) - 1.0));
            }
    ASSERT_GT(err.size(), size_t(w * h / 4));
    std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
    const double med = err[err.size() / 2];
    std::printf("falloff glue: median albedo error %.4f over %zu feature-mask pixels\n", med, err.size());
    EXPECT_LT(med, 0.02);   // 8-bit quantisation of the dim checker squares; ignoring exposure would be 100 %
}
