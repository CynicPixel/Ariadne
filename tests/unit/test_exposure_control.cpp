// 5.3 (D132): the exposure + lamp co-controller (§6.5), Stage 4 normalisation (§6.2), and D's sRGB input (D130).
#include "perception/degradation/degradation.hpp"
#include "perception/exposure_control/exposure_controller.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ex = ariadne::perception::exposure;
using ariadne::perception::DegradationParams;
using ariadne::perception::InputTransfer;
using ariadne::perception::kSrgbToLinear;

namespace {

// A textured linear radiance field in [0.05, 1]: smooth blobs plus fine checker texture.
cv::Mat Scene(int w = 320, int h = 240) {
    cv::Mat r(h, w, CV_32F);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            r.at<float>(y, x) = 0.05f + 0.45f * (1.0f + std::sin(x * 0.07f) * std::cos(y * 0.05f)) * 0.5f +
                                0.5f * (((x / 8) + (y / 8)) % 2);
    return r;
}

// What a linear camera records: radiance x relative exposure, clipped.
cv::Mat Capture(const cv::Mat& radiance, double rel) {
    cv::Mat c;
    cv::min(radiance * rel, 1.0, c);
    return c;
}

// The nominal state (10 ms, gain 1, lamp at its calibrated power) is exposure x1: the actuators reach ~x0.02 to
// x48 around it, as a real camera has headroom both ways.
constexpr double kNominal = 0.01;

}  // namespace

TEST(ExposureControl, GradientInformationIgnoresFlatAndSubThresholdTexture) {
    EXPECT_EQ(ex::GradientInformation(cv::Mat(64, 64, CV_32F, cv::Scalar(0.5))), 0.0);
    // A 1 % ripple is a Sobel magnitude well under delta = 0.06: counted as noise.
    cv::Mat ripple(64, 64, CV_32F);
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x) ripple.at<float>(y, x) = 0.5f + 0.005f * ((x + y) % 2);
    EXPECT_EQ(ex::GradientInformation(ripple), 0.0);
    EXPECT_GT(ex::GradientInformation(Scene()), 0.05);
}

TEST(ExposureControl, AllocatePhotonsFirstWhenBrighterGainFirstWhenDarker) {
    ex::ExposureControlParams p;
    ex::PhotometricState s{0.005, 2.0, 0.5};
    double applied = 0.0;
    // +1 stop: the lamp (0.5 -> 1.0, its budget) takes all of it; t and gain stay.
    auto up = ex::Allocate(s, 1.0, p, p.t_max_s, &applied);
    EXPECT_DOUBLE_EQ(up.lamp, 1.0);
    EXPECT_DOUBLE_EQ(up.exposure_s, 0.005);
    EXPECT_DOUBLE_EQ(up.gain, 2.0);
    EXPECT_NEAR(applied, 1.0, 1e-12);
    // +2 stops: lamp to budget (1 stop), then t (the second).
    up = ex::Allocate(s, 2.0, p, p.t_max_s, &applied);
    EXPECT_DOUBLE_EQ(up.lamp, 1.0);
    EXPECT_NEAR(up.exposure_s, 0.01, 1e-15);
    EXPECT_DOUBLE_EQ(up.gain, 2.0);
    // -1 stop: gain (2 -> 1) goes first; the photons stay.
    auto down = ex::Allocate(s, -1.0, p, p.t_max_s, &applied);
    EXPECT_DOUBLE_EQ(down.gain, 1.0);
    EXPECT_DOUBLE_EQ(down.exposure_s, 0.005);
    EXPECT_DOUBLE_EQ(down.lamp, 0.5);
    // Everything at its limit: nothing applied.
    ex::PhotometricState top{p.t_max_s, p.gain_max, p.lamp_max};
    ex::Allocate(top, 0.5, p, p.t_max_s, &applied);
    EXPECT_NEAR(applied, 0.0, 1e-12);
}

TEST(ExposureControl, BlurCeilingShortensExposureAndKeepsBrightness) {
    ex::ExposureControlParams p;
    // 680 px focal, 2 rad/s: 1 px of blur allows 1 / 1360 s.
    const double tmax = ex::ExposureCeiling(p, 2.0, 680.0);
    EXPECT_NEAR(tmax, 1.0 / 1360.0, 1e-12);
    ex::PhotometricState s{0.01, 1.0, 1.0};   // lamp already at budget
    double applied = 0.0;
    const auto n = ex::Allocate(s, 0.0, p, tmax, &applied);
    EXPECT_NEAR(n.exposure_s, tmax, 1e-15);
    EXPECT_NEAR(n.gain, 0.01 / tmax, 1e-9);   // gain recovers what t gave up
    EXPECT_NEAR(applied, 0.0, 1e-9);
}

TEST(ExposureControl, ClosedLoopConvergesWithinTheSlewLimit) {
    ex::ExposureControlParams p;   // production defaults, including the metric's stride-2 subsample
    const cv::Mat scene = Scene();
    for (double start : {0.004, 40.0}) {   // ~8 stops under, ~5 stops over
        ex::PhotometricState s{0.01, 1.0, 1.0};
        s.lamp = std::clamp(start * kNominal / (s.exposure_s * s.gain), p.lamp_min, p.lamp_max);
        s.gain = start * kNominal / (s.exposure_s * s.lamp);
        double last_metric = 0.0;
        for (int k = 0; k < 40; ++k) {
            const auto d = ex::Decide(Capture(scene, s.Brightness() / kNominal), s, p, 0.0, 680.0);
            EXPECT_LE(std::abs(d.step_stops), p.max_step_stops + 1e-9);
            s = d.next;
            last_metric = d.metric;
        }
        const auto d = ex::Decide(Capture(scene, s.Brightness() / kNominal), s, p, 0.0, 680.0);
        // The best any brightness can do on this scene WITHIN the clipping limit (a 1/16-stop sweep). Shim's metric
        // alone prefers frames that clip many bright pixels; the controller refuses those by design (§6.5 Failure E),
        // so the reference is the constrained optimum.
        double best = 0.0, best_b = 0.0;
        for (double e = -6.0; e <= 6.0; e += 1.0 / 16.0) {
            const cv::Mat c = Capture(scene, std::exp2(e));
            if (cv::countNonZero(c >= p.saturation_level) > p.max_saturated_frac * c.total()) continue;
            const double m = ex::GradientInformation(c, p.metric);
            if (m > best) best = m, best_b = std::exp2(e);
        }
        EXPECT_LE(d.saturated_frac, p.max_saturated_frac) << "start " << start;
        EXPECT_GT(last_metric, 0.95 * best) << "start " << start << " settled at exposure x" << s.Brightness() / kNominal
                                            << ", optimum " << best_b;
        EXPECT_LE(std::abs(d.step_stops), 0.25) << "start " << start;   // settled
    }
}

TEST(ExposureControl, OverexposureStepsDownAtTheSlewLimit) {
    ex::ExposureControlParams p;
    ex::PhotometricState s{0.01, 4.0, 1.0};
    const auto d = ex::Decide(Capture(Scene(), 8.0), s, p, 0.0, 680.0);
    EXPECT_EQ(d.reason, "overexposed");
    EXPECT_NEAR(d.step_stops, -0.5, 1e-12);
    EXPECT_NEAR(d.next.gain, 4.0 / std::sqrt(2.0), 1e-12);   // gain gives up first
}

TEST(ExposureControl, StageFourNormalisesToTheCanonicalState) {
    const cv::Mat r = Scene() * 0.1;
    ex::PhotometricState s{0.02, 4.0, 1.0};
    const cv::Mat n = ex::NormalizeExposure(r * (s.exposure_s * s.gain / 0.01), s, 0.01, 1.0);
    EXPECT_LT(cv::norm(n, r, cv::NORM_INF), 1e-6);
}

// D130: input_transfer srgb decodes before stage 1; with every noise stage off D is decode + quantise.
TEST(DegradationSrgb, DecodesBeforeTheSensorChain) {
    EXPECT_EQ(kSrgbToLinear[0], 0.0);
    EXPECT_EQ(kSrgbToLinear[255], 1.0);
    EXPECT_NEAR(kSrgbToLinear[137], 0.25, 0.002);   // the probe: 8-bit 137 rendered at linear 16394 / 65535
    ariadne::core::ImageU8 in(256, 1, 1);
    for (int x = 0; x < 256; ++x) in.At(x, 0, 0) = static_cast<uint8_t>(x);
    DegradationParams p;
    p.enable_shot_noise = false;
    p.read_noise_electrons = 0.0;
    p.input_transfer = InputTransfer::kSrgb;
    const auto out = ariadne::perception::Degrade(in, p, 1, 0);
    for (int x = 0; x < 256; ++x) {
        EXPECT_EQ(out.At(x, 0, 0), static_cast<uint8_t>(std::lround(kSrgbToLinear[x] * 255.0))) << x;
    }
    p.input_transfer = InputTransfer::kLinear;   // the default path is the identity here
    const auto lin = ariadne::perception::Degrade(in, p, 1, 0);
    for (int x = 0; x < 256; ++x) EXPECT_EQ(lin.At(x, 0, 0), x);
}
