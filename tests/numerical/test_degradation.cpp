// The seeded degradation operator D (arch §5.6) and the Phase 0 determinism
// gate that rests on it.
#include "perception/degradation/degradation.hpp"

#include "perception/degradation/degradation_digest.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ariadne::core;
using namespace ariadne::perception;

namespace {

ImageU8 Ramp(int w, int h, int channels = 1) {
    ImageU8 img(w, h, channels);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < channels; ++c)
                img.At(x, y, c) =
                    static_cast<uint8_t>((x * 255) / (w > 1 ? w - 1 : 1));
    return img;
}

ImageU8 Flat(int w, int h, uint8_t value, int channels = 1) {
    ImageU8 img(w, h, channels);
    std::fill(img.data.begin(), img.data.end(), value);
    return img;
}

ImageF32 FlatDepth(int w, int h, float metres) {
    ImageF32 d(w, h);
    std::fill(d.data.begin(), d.data.end(), metres);
    return d;
}

/// Parameters that make D the identity map, used as the baseline everything
/// else is measured against.
DegradationParams NullParams() {
    DegradationParams p;
    p.enable_shot_noise = false;
    p.read_noise_electrons = 0.0;
    p.vignette_strength = 0.0;
    p.prnu_sigma = 0.0;
    p.enable_dust = false;
    return p;
}

double MeanOf(const ImageU8& img) {
    double sum = 0.0;
    for (uint8_t v : img.data) sum += v;
    return sum / static_cast<double>(img.data.size());
}

double VarianceOf(const ImageU8& img) {
    const double m = MeanOf(img);
    double acc = 0.0;
    for (uint8_t v : img.data) acc += (v - m) * (v - m);
    return acc / static_cast<double>(img.data.size() - 1);
}

}  // namespace

// --- The identity baseline ------------------------------------------------

// With every noise source disabled and unit gain, D must return the input
// unchanged. Any drift here is a rounding bug in the normalize/quantize path
// and would contaminate every other measurement.
TEST(Degradation, NullParamsAreExactlyIdentity) {
    const ImageU8 in = Ramp(64, 48, 3);
    const ImageU8 out = Degrade(in, NullParams(), 1, 0);
    EXPECT_EQ(in.data, out.data);
}

TEST(Degradation, EmptyInputYieldsEmptyOutput) {
    const ImageU8 out = Degrade(ImageU8{}, NullParams(), 1, 0);
    EXPECT_TRUE(out.Empty());
}

// --- Determinism: the Phase 0 gate ---------------------------------------

TEST(Degradation, RepeatedCallsAreBitIdentical) {
    const ImageU8 in = Ramp(96, 64, 3);
    DegradationParams p;
    p.radiance_scale = 0.05;
    p.prnu_sigma = 0.01;
    p.vignette_strength = 0.3;

    const ImageU8 a = Degrade(in, p, 20260906, 42);
    const ImageU8 b = Degrade(in, p, 20260906, 42);
    EXPECT_EQ(a.data, b.data);
}

TEST(Degradation, SeedAndFrameChangeTheNoiseField) {
    const ImageU8 in = Flat(48, 48, 128);
    DegradationParams p;
    p.radiance_scale = 0.05;

    const ImageU8 base  = Degrade(in, p, 1, 0);
    const ImageU8 seed2 = Degrade(in, p, 2, 0);
    const ImageU8 frame2 = Degrade(in, p, 1, 1);
    EXPECT_NE(base.data, seed2.data);
    EXPECT_NE(base.data, frame2.data);
}

// Locality. Changing one input pixel must change only that output pixel.
// This is what makes the operator safe to parallelize over any partition:
// no cross-pixel state exists, so no tiling can change a result.
TEST(Degradation, EachOutputPixelDependsOnlyOnItsOwnInput) {
    ImageU8 in = Flat(32, 32, 100);
    DegradationParams p;
    p.radiance_scale = 0.2;
    p.prnu_sigma = 0.02;

    const ImageU8 before = Degrade(in, p, 7, 3);
    in.At(11, 17) = 200;                       // perturb exactly one pixel
    const ImageU8 after = Degrade(in, p, 7, 3);

    int differing = 0;
    for (size_t i = 0; i < before.data.size(); ++i) {
        if (before.data[i] != after.data[i]) ++differing;
    }
    EXPECT_EQ(differing, 1);
    EXPECT_NE(before.At(11, 17), after.At(11, 17));
}

// --- Fixed-pattern behaviour ---------------------------------------------

// PRNU is a property of the silicon, so it must be identical in every frame.
// If this fails, fixed-pattern noise has become temporal noise, frame
// averaging would remove it for free, and the flat-field stage of §6.2 would
// measure as far more effective than it really is.
TEST(Degradation, PrnuPatternIsIdenticalAcrossFrames) {
    const ImageU8 in = Flat(64, 64, 180);
    DegradationParams p = NullParams();
    p.prnu_sigma = 0.05;                        // only PRNU active

    const ImageU8 f0 = Degrade(in, p, 9, 0);
    const ImageU8 f1 = Degrade(in, p, 9, 1);
    const ImageU8 f2 = Degrade(in, p, 9, 12345);
    EXPECT_EQ(f0.data, f1.data);
    EXPECT_EQ(f0.data, f2.data);
    EXPECT_NE(f0.data, in.data);                // but it did do something
}

TEST(Degradation, PrnuPatternDependsOnSeed) {
    const ImageU8 in = Flat(64, 64, 180);
    DegradationParams p = NullParams();
    p.prnu_sigma = 0.05;
    EXPECT_NE(Degrade(in, p, 1, 0).data, Degrade(in, p, 2, 0).data);
}

// --- Vignette -------------------------------------------------------------

TEST(Degradation, VignetteDarkensTowardTheCorners) {
    const ImageU8 in = Flat(101, 101, 200);
    DegradationParams p = NullParams();
    p.vignette_strength = 0.5;

    const ImageU8 out = Degrade(in, p, 1, 0);
    const int centre = out.At(50, 50);
    const int edge   = out.At(50, 0);
    const int corner = out.At(0, 0);

    EXPECT_EQ(centre, 200);                     // centre is unattenuated
    EXPECT_LT(edge, centre);
    EXPECT_LT(corner, edge);                    // falloff is monotone in radius
    // Corner response should be close to the requested 1 - strength.
    EXPECT_NEAR(corner / 200.0, 0.5, 0.02);
}

TEST(Degradation, VignetteIsRadiallySymmetric) {
    const ImageU8 in = Flat(65, 65, 200);
    DegradationParams p = NullParams();
    p.vignette_strength = 0.4;
    const ImageU8 out = Degrade(in, p, 1, 0);
    EXPECT_EQ(out.At(0, 0), out.At(64, 0));
    EXPECT_EQ(out.At(0, 0), out.At(0, 64));
    EXPECT_EQ(out.At(0, 0), out.At(64, 64));
}

// --- Noise physics --------------------------------------------------------

// Shot noise is signal-dependent: variance grows with the signal, so a bright
// patch carries more absolute noise but better SNR. This is the property that
// distinguishes a Poisson-Gaussian model from "add some noise".
TEST(Degradation, ShotNoiseVarianceGrowsWithSignal) {
    DegradationParams p;
    p.read_noise_electrons = 0.0;
    p.photon_scale = 2000.0;

    const double var_dim    = VarianceOf(Degrade(Flat(128, 128, 20),  p, 5, 0));
    const double var_bright = VarianceOf(Degrade(Flat(128, 128, 200), p, 5, 0));
    EXPECT_GT(var_bright, var_dim);

    // Poisson variance equals the mean in electrons, so the standard deviation
    // in output units scales as sqrt(signal). A 10x brighter signal should give
    // roughly sqrt(10) ~ 3.2x the standard deviation.
    const double ratio = std::sqrt(var_bright / var_dim);
    EXPECT_NEAR(ratio, std::sqrt(10.0), 0.6);
}

// Read noise is constant with signal, which is why it dominates in the dark
// and why low light is hard rather than merely dim.
TEST(Degradation, ReadNoiseDominatesInDarkness) {
    DegradationParams p;
    p.enable_shot_noise = false;
    p.read_noise_electrons = 40.0;
    p.photon_scale = 2000.0;

    const double var_dim    = VarianceOf(Degrade(Flat(128, 128, 20),  p, 5, 0));
    const double var_bright = VarianceOf(Degrade(Flat(128, 128, 200), p, 5, 0));
    // Same additive noise regardless of signal level.
    EXPECT_NEAR(var_dim, var_bright, 0.25 * var_bright);
}

TEST(Degradation, RadianceScaleDarkensTheImage) {
    const ImageU8 in = Flat(64, 64, 200);
    DegradationParams p = NullParams();
    p.radiance_scale = 0.25;
    EXPECT_NEAR(MeanOf(Degrade(in, p, 1, 0)), 50.0, 1.0);
}

TEST(Degradation, GainAndSaturationClip) {
    const ImageU8 in = Flat(32, 32, 200);
    DegradationParams p = NullParams();
    p.analog_gain = 4.0;                        // 200 * 4 overflows the range
    const ImageU8 out = Degrade(in, p, 1, 0);
    EXPECT_EQ(out.At(0, 0), 255);               // saturated, not wrapped

    p.saturation = 0.5;                         // clip earlier
    EXPECT_EQ(Degrade(in, p, 1, 0).At(0, 0), 128);
}

TEST(Degradation, BitDepthQuantizes) {
    const ImageU8 in = Ramp(256, 4);
    DegradationParams p = NullParams();
    p.bit_depth = 4;                            // 16 levels
    const ImageU8 out = Degrade(in, p, 1, 0);
    std::vector<int> distinct(256, 0);
    for (uint8_t v : out.data) distinct[v] = 1;
    int count = 0;
    for (int d : distinct) count += d;
    EXPECT_LE(count, 16);
}

// --- Dust -----------------------------------------------------------------

TEST(Degradation, DustRequiresDepthAndIsOtherwiseInert) {
    const ImageU8 in = Flat(32, 32, 100);
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 1.0;
    // No depth supplied: the image must pass through untouched rather than
    // silently acquiring fog at an invented distance.
    EXPECT_EQ(Degrade(in, p, 1, 0).data, in.data);
}

TEST(Degradation, DustAttenuatesWithDepthTowardAirlight) {
    const ImageU8 in = Flat(32, 32, 60);        // dark subject
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 0.5;
    p.dust_airlight = 0.9;                      // bright airlight
    p.dust_field_strength = 0.0;                // homogeneous, so t is exact

    const int near = Degrade(in, FlatDepth(32, 32, 0.1f), p, 1, 0).At(16, 16);
    const int mid  = Degrade(in, FlatDepth(32, 32, 5.0f), p, 1, 0).At(16, 16);
    const int far  = Degrade(in, FlatDepth(32, 32, 50.0f), p, 1, 0).At(16, 16);

    EXPECT_LT(near, mid);
    EXPECT_LT(mid, far);                        // approaches airlight with depth
    EXPECT_NEAR(far, static_cast<int>(0.9 * 255), 2);
    EXPECT_NEAR(near, 60, 12);                  // barely attenuated up close
}

// The closed form of the transmission term, checked directly:
//   I = J*t + A*(1-t),  t = exp(-beta*d)
TEST(Degradation, DustMatchesTheClosedFormTransmission) {
    const uint8_t J = 100;
    const double beta = 0.3, depth = 4.0, airlight = 0.8;
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = beta;
    p.dust_airlight = airlight;
    p.dust_field_strength = 0.0;

    const double t = std::exp(-beta * depth);
    const double expected = (J / 255.0) * t + airlight * (1.0 - t);
    const int got = Degrade(Flat(8, 8, J), FlatDepth(8, 8, depth), p, 1, 0).At(4, 4);
    EXPECT_NEAR(got, expected * 255.0, 1.5);
}

TEST(Degradation, ZeroOrInvalidDepthLeavesPixelUntouched) {
    const ImageU8 in = Flat(8, 8, 120);
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 2.0;
    p.dust_field_strength = 0.0;

    ImageF32 depth = FlatDepth(8, 8, 0.0f);     // unknown depth
    EXPECT_EQ(Degrade(in, depth, p, 1, 0).At(4, 4), 120);

    depth.At(4, 4) = std::nanf("");
    EXPECT_EQ(Degrade(in, depth, p, 1, 0).At(4, 4), 120);
}

// The particulate field must vary across the image, otherwise dust is just a
// global contrast reduction and the "local density" of §5.6 is fictional.
TEST(Degradation, DustFieldVariesSpatially) {
    const ImageU8 in = Flat(128, 128, 120);
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 0.4;
    p.dust_field_strength = 0.9;
    p.dust_field_frequency = 0.05;
    p.dust_field_octaves = 3;

    const ImageU8 out = Degrade(in, FlatDepth(128, 128, 6.0f), p, 1, 0);
    EXPECT_GT(VarianceOf(out), 1.0);            // not a uniform veil
}

// --- Golden digest --------------------------------------------------------

// Frozen output for a fixed configuration. Committed so the arm64 check is a
// single comparison, and so that an accidental change to any stage is caught
// rather than silently altering every degraded frame in the project.
TEST(Degradation, GoldenDigestForAFixedConfiguration) {
    // Shared implementation: perception/degradation/degradation_digest.hpp.
    EXPECT_EQ(DegradeDigest(), 8535765251477677907ull)
        << "degradation output changed; re-record deliberately or investigate";
}
