// 5.3 (D147): the exposure controller as wired into the runner -- the exposure_control: config section, the
// actuator contract (command -> applied report), the seam through the radiometric wrapper, and the keyframe glue.
#include "perception/perception_config.hpp"
#include "runtime/datasets/radiometric_source.hpp"
#include "runtime/tools/exposure_glue.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace ex = ariadne::perception::exposure;
using ariadne::runtime::FramePhoto;
using ariadne::runtime::PhotometricCommand;
using ariadne::tools::KeyframeExposure;

namespace {

ex::ExposureControlStage Parse(const std::string& yaml) {
    auto s = ariadne::core::StrictConfig::FromString(yaml);
    return ex::ParseExposureControl(s);
}

ex::ExposureControlStage On() {
    ex::ExposureControlStage e;
    e.mode = "replay";
    e.params.t_max_s = 0.05;
    return e;
}

// The smallest source that records the actuator calls it receives.
class RecordingSource : public ariadne::runtime::StereoSource {
public:
    std::vector<PhotometricCommand> got;
    std::string RunName() const override { return "fake"; }
    std::string SceneName() const override { return "fake"; }
    std::string CameraConfig() const override { return "fake"; }
    std::string BagChecksum() const override { return "UNRECORDED"; }
    ariadne::perception::RawStereoCalibration Calibration() const override { return {}; }
    ariadne::core::SE3 T_B_C0() const override { return {}; }
    std::vector<ariadne::runtime::TimedPose> GroundTruth() const override { return {}; }
    size_t FrameCount() const override { return 0; }
    uint32_t Agent() const override { return 0; }
    size_t ForEachFrame(size_t, const std::function<void(const ariadne::runtime::StereoFrame&)>&) override { return 0; }
    bool CanActuatePhotometric() const override { return true; }
    void SetPhotometric(const PhotometricCommand& c) override { got.push_back(c); }
};

}  // namespace

TEST(ExposureGlue, ConfigSection) {
    EXPECT_FALSE(ex::ExposureControlStage{}.Enabled());   // absent section: no controller
    const auto e = Parse("mode: replay\nt_max_s: 0.05\nmetric: percentile\n");
    EXPECT_TRUE(e.Enabled());
    EXPECT_DOUBLE_EQ(e.t_ref_s, 0.0125);   // D147's 2 stops of headroom under 50 ms
    EXPECT_EQ(e.params.metric.metric, "percentile");
    EXPECT_THROW(Parse("mode: live\n"), ariadne::core::ConfigError);                   // the live sink is B's
    EXPECT_THROW(Parse("mode: replay\nt_max_s: 0.05\nt_ref_s: 0.1\n"), ariadne::core::ConfigError);   // above t_max
    EXPECT_THROW(Parse("mode: replay\nt_max_s: 0.05\nbogus: 1\n"), ariadne::core::ConfigError);
    EXPECT_THROW(Parse("mode: replay\n"), ariadne::core::ConfigError);   // t_max_s required (the frame period)
}

TEST(ExposureGlue, RefusesTheVstModes) {
    auto c = ariadne::perception::LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/sim_v1_rad_auto.yaml");
    c.exposure_control = On();
    EXPECT_THROW(ariadne::perception::ValidatePerception(c), ariadne::core::ConfigError);   // auto: fitted once
    c.radiometric.mode = "flat_only";
    EXPECT_NO_THROW(ariadne::perception::ValidatePerception(c));
}

TEST(ExposureGlue, CommandBecomesTheAppliedReport) {
    ariadne::perception::DegradationParams d;
    d.radiance_scale = 0.81;
    d.digital_gain = 1.0;
    FramePhoto p = ariadne::runtime::PhotoOf(d);
    EXPECT_EQ(p.decision_id, 0u);
    EXPECT_EQ(p.applied_ns, -1);
    PhotometricCommand cmd{7, 1000, 1.5, 2.0, 0.5};
    ariadne::runtime::ApplyPhotometric(d, cmd, 2000, p);
    EXPECT_DOUBLE_EQ(d.exposure, 1.5);
    EXPECT_DOUBLE_EQ(d.analog_gain, 2.0);
    EXPECT_DOUBLE_EQ(d.lamp_power, 0.5);
    EXPECT_EQ(p.decision_id, 7u);
    EXPECT_EQ(p.applied_ns, 2000);
    EXPECT_FALSE(p.clamped);   // D has no limits of its own
    EXPECT_DOUBLE_EQ(p.Brightness(), 1.5 * 2.0 * 0.81 * 0.5);
}

TEST(ExposureGlue, TheWrapperForwardsTheSeam) {
    // D146's lesson (Lamp()): a wrapper that does not forward makes the stage refuse every real run.
    RecordingSource inner;
    ariadne::runtime::RadiometricSource wrapped(inner, ariadne::perception::radiometric::RadiometricParams{});
    EXPECT_TRUE(wrapped.CanActuatePhotometric());
    wrapped.SetPhotometric(PhotometricCommand{1, 5, 1.0, 1.0, 0.7});
    ASSERT_EQ(inner.got.size(), 1u);
    EXPECT_DOUBLE_EQ(inner.got[0].lamp_rel, 0.7);
}

TEST(ExposureGlue, DecidesFromTheAppliedStateAndStepsDownWhenOverexposed) {
    EXPECT_THROW(KeyframeExposure(On(), false), std::runtime_error);   // a source that cannot actuate
    EXPECT_NO_THROW(KeyframeExposure(ex::ExposureControlStage{}, false));
    KeyframeExposure k(On(), true);
    // Half the frame clipped: the D132 overexposure guard steps down by at most 0.5 stop.
    cv::Mat g(120, 160, CV_8U);
    cv::randu(g, 40, 200);
    g.colRange(0, 80).setTo(255);
    FramePhoto applied;   // nominal, lamp at full (the calibrated P)
    EXPECT_THROW(k.Decide(g, applied, 123, 0.0, 600.0, 40'000'000), std::runtime_error);   // 50 ms > a 25 Hz frame
    const PhotometricCommand c = k.Decide(g, applied, 123, 0.0, 600.0, 50'000'000);
    EXPECT_EQ(c.decision_id, 1u);
    EXPECT_EQ(c.decided_ns, 123);
    const double before = applied.exposure_rel * applied.gain * applied.lamp_rel;
    const double after = c.exposure_rel * c.gain * c.lamp_rel;
    EXPECT_LT(after, before);
    EXPECT_GE(after, before * std::pow(2.0, -0.5) - 1e-9);   // slew limit
    EXPECT_NE(k.Trace().find(",overexposed,"), std::string::npos);
    // Darker is spent gain first, then exposure, then lamp (D132 order reversed): at unit gain, exposure moves.
    EXPECT_LT(c.exposure_rel, 1.0);
    EXPECT_DOUBLE_EQ(c.lamp_rel, 1.0);
}

TEST(ExposureGlue, RadianceEventStepsTheSceneNotTheSensor) {
    // Failure E (D147): 2 stops brighter from 20 s on; nothing before it, nothing without an event.
    const ariadne::runtime::RadianceEvent e{20.0, 2.0};
    EXPECT_DOUBLE_EQ(ariadne::runtime::RadianceEventFactor(std::nullopt, 30'000'000'000), 1.0);
    EXPECT_DOUBLE_EQ(ariadne::runtime::RadianceEventFactor(e, 19'999'999'999), 1.0);
    EXPECT_DOUBLE_EQ(ariadne::runtime::RadianceEventFactor(e, 20'000'000'000), 4.0);
    EXPECT_DOUBLE_EQ(ariadne::runtime::RadianceEventFactor(ariadne::runtime::RadianceEvent{0.0, -1.0}, 0), 0.5);
}
