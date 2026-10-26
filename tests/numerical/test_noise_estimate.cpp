// Noise measured, not assumed (2.2, noise_estimate.hpp): the single-image fit recovers the noise model
// the degradation operator D is known to produce, and the stabilised-domain check separates usable data
// from data the VST must not touch (posterised 8-bit darkness).
#include "perception/radiometric/noise_estimate.hpp"

#include "perception/degradation/degradation.hpp"
#include <gtest/gtest.h>

#include <cmath>

using namespace ariadne::core;
using namespace ariadne::perception;
using namespace ariadne::perception::radiometric;

namespace {

/// A test scene: a horizontal ramp (all levels) with a vertical step edge every 64 px and a bright square,
/// so the fit has to survive structure, not just flat noise.
ImageU8 Scene(int w, int h) {
    ImageU8 img(w, h, 1);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int v = (x * 255) / (w - 1);
            if ((x / 64) % 2) v = v * 3 / 4;
            if (x > w / 3 && x < w / 2 && y > h / 3 && y < h / 2) v = 230;
            img.At(x, y) = static_cast<uint8_t>(v);
        }
    return img;
}

/// The shared sensor of configs/degradation/*.yaml with no vignette or PRNU (so K = 1 and no flat-field is needed).
DegradationParams Sensor(double radiance, double gain) {
    DegradationParams p;
    p.radiance_scale = radiance;
    p.photon_scale = 10000.0;
    p.read_noise_electrons = 2.5;
    p.analog_gain = gain;
    p.bit_depth = 8;
    return p;
}

/// The model D is built to produce, in output DN: alpha = gain * 255 / photon_scale per electron,
/// b = (alpha * read)^2 + 1/12 (the 8-bit rounding).
double AlphaDn(const DegradationParams& p) { return p.analog_gain * 255.0 / p.photon_scale; }
double BDn(const DegradationParams& p) { return std::pow(AlphaDn(p) * p.read_noise_electrons, 2) + 1.0 / 12.0; }

struct Frames {
    std::vector<ImageU8> raw;
    std::vector<ImageF32> z;
    std::vector<const ImageU8*> rp;
    std::vector<const ImageF32*> zp;
};

Frames Degraded(const DegradationParams& p, int n) {
    Frames f;
    const ImageU8 scene = Scene(640, 480);
    for (int i = 0; i < n; ++i) {
        f.raw.push_back(Degrade(scene, p, 2026, static_cast<uint64_t>(i)));
        ImageF32 z(640, 480);
        for (size_t k = 0; k < z.data.size(); ++k) z.data[k] = f.raw.back().data[k];
        f.z.push_back(std::move(z));
    }
    for (int i = 0; i < n; ++i) f.rp.push_back(&f.raw[i]), f.zp.push_back(&f.z[i]);
    return f;
}

}  // namespace

// Gain-up darkness (dark2 at analog gain 8): a within 10%, b within 0.1 DN^2, and the GAT built from
// the fit stabilises the noise to std ~ 1.
TEST(NoiseEstimate, RecoversTheModelDProduces) {
    const DegradationParams p = Sensor(0.06, 8.0);
    const Frames f = Degraded(p, 2);
    const NoiseFit fit = FitNoise(f.zp, f.rp);
    ASSERT_TRUE(fit.valid) << fit.why;
    EXPECT_NEAR(fit.a, AlphaDn(p), 0.10 * AlphaDn(p));
    EXPECT_NEAR(fit.b, BDn(p), 0.1);
    const double sz = StabilisedSigma(f.zp, f.rp, FromAffineFit(fit.a, fit.b));
    EXPECT_GT(sz, 0.9);
    EXPECT_LT(sz, 1.1);
}

// The fit tracks the sensor: doubling the analog gain doubles a (within the 10% budget).
TEST(NoiseEstimate, GainScalesA) {
    const Frames g8 = Degraded(Sensor(0.06, 8.0), 2), g16 = Degraded(Sensor(0.06, 16.0), 2);
    const NoiseFit f8 = FitNoise(g8.zp, g8.rp), f16 = FitNoise(g16.zp, g16.rp);
    ASSERT_TRUE(f8.valid) << f8.why;
    ASSERT_TRUE(f16.valid) << f16.why;
    EXPECT_NEAR(f16.a / f8.a, 2.0, 0.2);
}

// [MEASURED 2026-10-21] The intercept b is the weak part of the fit: at dark3 x gain 16 it comes out
// 0.82 DN^2 against D's 1.12 (a within 1%). sigma_z is still 1.01 -- the forward GAT and the inverse use
// the same b, so the error cancels above the lowest counts. Recorded, not hidden (radiometric.md §5).
TEST(NoiseEstimate, InterceptBiasAtDeepDarknessIsBoundedAndHarmless) {
    const DegradationParams p = Sensor(0.015, 16.0);
    const Frames f = Degraded(p, 2);
    const NoiseFit fit = FitNoise(f.zp, f.rp);
    ASSERT_TRUE(fit.valid) << fit.why;
    EXPECT_NEAR(fit.a, AlphaDn(p), 0.10 * AlphaDn(p));
    EXPECT_NEAR(fit.b, BDn(p), 0.5);
    EXPECT_NEAR(StabilisedSigma(f.zp, f.rp, FromAffineFit(fit.a, fit.b)), 1.0, 0.05);
}

// A WRONG model shows in the stabilised domain: half the true gain leaves std well above 1.
TEST(NoiseEstimate, StabilisedSigmaExposesAWrongModel) {
    const DegradationParams p = Sensor(0.06, 8.0);
    const Frames f = Degraded(p, 2);
    const double right = StabilisedSigma(f.zp, f.rp, FromAffineFit(AlphaDn(p), BDn(p)));
    const double wrong = StabilisedSigma(f.zp, f.rp, FromAffineFit(0.5 * AlphaDn(p), BDn(p)));
    EXPECT_NEAR(right, 1.0, 0.1);
    EXPECT_GT(wrong, 1.3);
}

// Unity-gain darkness as shipped (dark2: 0-15 DN, sigma 0.2-0.7 DN) is posterised by the 8-bit
// output: the measurement must say so -- either no valid fit, or a stabilised std far from 1.
TEST(NoiseEstimate, PosterisedDarknessIsNotMistakenForUsableNoise) {
    const Frames f = Degraded(Sensor(0.06, 1.0), 2);
    const NoiseFit fit = FitNoise(f.zp, f.rp);
    if (fit.valid) {
        const double sz = StabilisedSigma(f.zp, f.rp, FromAffineFit(fit.a, fit.b));
        EXPECT_TRUE(sz < 0.8 || sz > 1.25) << "a " << fit.a << " b " << fit.b << " sigma_z " << sz;
    } else {
        SUCCEED() << fit.why;
    }
}

// The sim's real signal path is RGB -> D per channel -> gray (cv::COLOR_RGB2GRAY, fixed point). The gray
// mean equals the channel mean, but the variance is sum(w^2) = 0.447 of one channel's: the gray-path fit
// must find a = 0.447 x D's a (and so an SNR 1.5x the mono one). The gate threshold (radiometric_config.hpp,
// D67) was corrected for exactly this; this test pins the scaling so it cannot drift silently.
TEST(NoiseEstimate, GrayPathScalesTheModel) {
    const DegradationParams p = Sensor(0.06, 8.0);
    ImageU8 rgb(640, 480, 3);
    const ImageU8 mono = Scene(640, 480);
    for (int y = 0; y < 480; ++y)
        for (int x = 0; x < 640; ++x)
            for (int c = 0; c < 3; ++c) rgb.At(x, y, c) = mono.At(x, y);
    Frames f;
    for (int i = 0; i < 2; ++i) {
        const ImageU8 d = Degrade(rgb, p, 2026, static_cast<uint64_t>(i));
        ImageU8 g(640, 480, 1);
        ImageF32 z(640, 480);
        for (size_t k = 0; k < g.data.size(); ++k) {
            // OpenCV's RGB2GRAY for 8U: BT.601 in Q14 fixed point, rounded.
            g.data[k] = static_cast<uint8_t>((d.data[3 * k] * 4899 + d.data[3 * k + 1] * 9617 + d.data[3 * k + 2] * 1868 + 8192) >> 14);
            z.data[k] = g.data[k];
        }
        f.raw.push_back(std::move(g));
        f.z.push_back(std::move(z));
    }
    for (int i = 0; i < 2; ++i) f.rp.push_back(&f.raw[i]), f.zp.push_back(&f.z[i]);
    const NoiseFit fit = FitNoise(f.zp, f.rp);
    ASSERT_TRUE(fit.valid) << fit.why;
    const double w2 = 0.299 * 0.299 + 0.587 * 0.587 + 0.114 * 0.114;
    EXPECT_NEAR(fit.a, w2 * AlphaDn(p), 0.10 * w2 * AlphaDn(p));
    EXPECT_NEAR(StabilisedSigma(f.zp, f.rp, FromAffineFit(fit.a, fit.b)), 1.0, 0.05);
}
