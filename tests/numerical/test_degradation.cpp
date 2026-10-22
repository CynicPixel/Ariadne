// The seeded degradation operator D (arch §5.6) and the Phase 0 determinism
// gate that rests on it.
#include "perception/degradation/degradation.hpp"

#include "perception/degradation/degradation_config.hpp"
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

// 2.2: the row-band parallel entry point produces Degrade's bytes for any band count, with every stage on
// (RGB, shot + read noise, PRNU, vignette, gain, dust with its field) and odd sizes that do not divide evenly.
TEST(Degradation, ParallelRowBandsAreByteIdentical) {
    ImageU8 in(97, 61, 3);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x)
            for (int c = 0; c < 3; ++c) in.At(x, y, c) = static_cast<uint8_t>((x * 7 + y * 3 + c * 11) & 0xFF);
    DegradationParams p;
    p.radiance_scale = 0.3;
    p.read_noise_electrons = 2.5;
    p.vignette_strength = 0.28;
    p.prnu_sigma = 0.012;
    p.analog_gain = 4.0;
    p.enable_dust = true;
    p.dust_beta = 0.2;
    const ImageF32 depth = FlatDepth(97, 61, 4.0f);
    const ImageU8 serial = Degrade(in, depth, p, 77, 5);
    for (int threads : {1, 2, 3, 8, 61, 200}) {
        EXPECT_EQ(DegradeParallel(in, depth, p, 77, 5, threads).data, serial.data) << threads << " threads";
    }
    EXPECT_EQ(DegradeParallel(in, p, 77, 5, 4).data, Degrade(in, p, 77, 5).data);
}

// --- L9 / D135: dust fidelity options and the replay lamp actuator ----------

// lamp_power is a pure radiance scale (lamp the only light, linear render): with every noise stage off it is
// indistinguishable from radiance_scale.
TEST(DegradationFidelity, LampPowerIsARadianceScale) {
    const ImageU8 in = Ramp(64, 4);
    DegradationParams a = NullParams(), b = NullParams();
    a.lamp_power = 0.5;
    b.radiance_scale = 0.5;
    EXPECT_EQ(Degrade(in, a, 1, 0).data, Degrade(in, b, 1, 0).data);
}

// D9's objection, made measurable: over a black scene in dense dust, post-sensor airlight is a noise-free constant,
// pre-sensor airlight is light and carries shot noise.
TEST(DegradationFidelity, PreSensorAirlightCarriesShotNoise) {
    const ImageU8 black = Flat(128, 128, 0);
    const ImageF32 far = FlatDepth(128, 128, 50.0f);   // t = exp(-0.3 * 50) ~ 3e-7
    DegradationParams p;
    p.read_noise_electrons = 0.0;
    p.photon_scale = 400.0;                           // few photons: visible shot noise
    p.enable_dust = true;
    p.dust_beta = 0.3;
    p.dust_field_strength = 0.0;
    p.dust_airlight = 0.5;
    const ImageU8 post = Degrade(black, far, p, 3, 0);
    p.dust_order = DustOrder::kPreSensor;
    const ImageU8 pre = Degrade(black, far, p, 3, 0);
    EXPECT_EQ(VarianceOf(post), 0.0);
    EXPECT_GT(VarianceOf(pre), 10.0);                 // ~ (255^2 / 400) * 0.5 ~ 81 before quantisation
    EXPECT_NEAR(MeanOf(pre), MeanOf(post), 2.0);      // same mean veil
}

// D11 option (b): with tau > 1 consecutive frames share most of their dust field.
TEST(DegradationFidelity, CorrelatedDustFieldChangesSlowly) {
    const ImageU8 in = Flat(96, 96, 200);
    const ImageF32 d = FlatDepth(96, 96, 4.0f);
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 0.2;
    p.dust_field_strength = 1.0;
    p.dust_field_frequency = 0.05;
    const auto step_change = [&](int tau) {
        p.dust_field_tau_frames = tau;
        double s = 0.0;
        for (uint64_t k = 3; k < 9; ++k) {
            const ImageU8 a = Degrade(in, d, p, 5, k), b = Degrade(in, d, p, 5, k + 1);
            for (size_t i = 0; i < a.data.size(); ++i) s += std::abs(int(a.data[i]) - int(b.data[i]));
        }
        return s;
    };
    const double white = step_change(0), slow = step_change(16);
    EXPECT_GT(white, 0.0);
    EXPECT_LT(slow, 0.25 * white);
    // Bit-identical replay of any frame, whatever tau.
    p.dust_field_tau_frames = 16;
    EXPECT_EQ(Degrade(in, d, p, 5, 21).data, Degrade(in, d, p, 5, 21).data);
}

// kLamp airlight follows the lamp's lobe: veil in the cone, none outside it, and it scales with lamp power.
TEST(DegradationFidelity, LampAirlightFollowsTheLobeAndThePower) {
    const ImageU8 black = Flat(201, 201, 0);
    const ImageF32 far = FlatDepth(201, 201, 50.0f);
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_beta = 0.3;
    p.dust_field_strength = 0.0;
    p.dust_airlight = 0.5;
    p.dust_airlight_mode = DustAirlight::kLamp;
    p.dust_lamp_focal_px = 100.0;
    p.dust_lamp_cos_inner = 0.906;   // 25 deg
    p.dust_lamp_cos_outer = 0.766;   // 40 deg
    const ImageU8 full = Degrade(black, far, p, 1, 0);
    EXPECT_NEAR(full.At(100, 100, 0), 128, 1);        // centre: dust_airlight
    EXPECT_EQ(full.At(0, 0, 0), 0);                    // corner: 54 deg off axis, unlit
    p.lamp_power = 0.5;
    EXPECT_NEAR(Degrade(black, far, p, 1, 0).At(100, 100, 0), 64, 1);
}

// --- D139: the headlamp dust model (ALGORITHMS §10.2) ------------------------

namespace {
DegradationParams HeadlampParams() {
    DegradationParams p = NullParams();
    p.enable_dust = true;
    p.dust_model = DustModel::kHeadlamp;
    p.dust_order = DustOrder::kPreSensor;
    p.dust_beta = 0.1;
    p.dust_beta_backscatter = 0.1;
    p.dust_backscatter = 0.4;
    p.dust_field_strength = 0.0;
    p.dust_lamp_focal_px = 100.0;
    p.dust_lamp_cos_inner = 0.906;   // 25 deg half-angle
    p.dust_lamp_cos_outer = 0.766;   // 40 deg
    return p;
}

// The same integral, finely and with libm (test-only reference): trapezoid on 200k log-spaced nodes.
double FineVeil(const DegradationParams& p, double ux, double uy, double uz, double lx, double ly, double lz,
                double range) {
    const int n = 200000;
    const double a = std::log(p.dust_near_m), b = std::log(range);
    double acc = 0.0, prev_s = 0.0, prev_f = 0.0;
    for (int i = 0; i <= n; ++i) {
        const double s = std::exp(a + (b - a) * i / n);
        const double vx = s * ux - lx, vy = s * uy - ly, vz = s * uz - lz;
        const double r2 = vx * vx + vy * vy + vz * vz, r = std::sqrt(r2);
        const double f = detail::HeadlampVeil::Lobe(p, vz / r) / r2 * std::exp(-p.dust_beta_backscatter * (s + r));
        if (i > 0) acc += 0.5 * (prev_f + f) * (s - prev_s);
        prev_s = s;
        prev_f = f;
    }
    return acc;
}
}  // namespace

// §10.2 consequence 1: the veil is near-field dominated. Going from a wall at 0.5 m to one at 5 m barely changes it,
// whereas the daylight model's 1 - exp(-beta d) grows ~8x over the same span.
TEST(DegradationHeadlamp, VeilIsNearFieldDominated) {
    const auto p = HeadlampParams();
    detail::HeadlampVeil v(p, 201, 201);
    const double near = v.At(100, 100, 0.5), far = v.At(100, 100, 5.0);
    EXPECT_GT(near, 0.9 * far);
    EXPECT_LE(near, far);
    const double kos_ratio = (1.0 - std::exp(-0.1 * 5.0)) / (1.0 - std::exp(-0.1 * 0.5));
    EXPECT_GT(kos_ratio, 7.0);
}

// §10.2's hardware lever: moving the lamp off the optical axis removes the near-field lobe.
TEST(DegradationHeadlamp, LampOffsetLowersTheVeil) {
    auto p = HeadlampParams();
    const double coax = detail::HeadlampVeil(p, 201, 201).At(100, 100, 3.0);
    p.dust_lamp_x_m = 0.30;
    const double off = detail::HeadlampVeil(p, 201, 201).At(100, 100, 3.0);
    EXPECT_NEAR(coax, 1.0, 0.05);   // the coaxial on-axis reference is ~1 by construction (range 3 m vs ~60 m)
    EXPECT_LT(off, 0.2 * coax);
}

// The 96-node table against a 200k-node libm integral: the deterministic quadrature is accurate.
TEST(DegradationHeadlamp, TabulatedIntegralMatchesAFineReference) {
    auto p = HeadlampParams();
    p.dust_lamp_x_m = 0.05;
    p.dust_lamp_y_m = -0.02;
    p.dust_lamp_z_m = 0.02;
    detail::HeadlampVeil v(p, 201, 201);   // grid nodes at multiples of 8 px: (96, 96) is a node
    const double ref = FineVeil(p, 0, 0, 1, 0, 0, 0, 59.0);
    for (double range : {0.3, 1.0, 3.0}) {
        const double rx = (96 - 100.0) / 100.0, ry = (96 - 100.0) / 100.0, inv = 1.0 / std::sqrt(rx * rx + ry * ry + 1);
        const double fine = FineVeil(p, rx * inv, ry * inv, inv, 0.05, -0.02, 0.02, range) / ref;
        EXPECT_NEAR(v.NodeAt(12, 12, range), fine, 0.03 * fine + 1e-4) << "range " << range;
    }
}

// The direct term is the round trip: with no backscatter and no noise, J exp(-2 beta range).
TEST(DegradationHeadlamp, DirectAttenuationIsOutAndBack) {
    auto p = HeadlampParams();
    p.dust_backscatter = 0.0;
    const ImageU8 in = Flat(201, 201, 200);
    const ImageF32 d = FlatDepth(201, 201, 2.0f);
    const ImageU8 out = Degrade(in, d, p, 1, 0);
    // The centre pixel: range = z = 2 m.
    EXPECT_EQ(out.At(100, 100, 0), static_cast<uint8_t>(std::lround(200.0 * std::exp(-0.1 * 2.0 * 2.0))));
}

TEST(DegradationHeadlamp, DeterministicAndBandSplitInvariant) {
    auto p = HeadlampParams();
    p.enable_shot_noise = true;
    p.read_noise_electrons = 2.0;
    p.dust_field_strength = 0.5;
    p.dust_lamp_x_m = 0.05;
    const ImageU8 in = Ramp(160, 120, 3);
    const ImageF32 d = FlatDepth(160, 120, 1.5f);
    const ImageU8 a = Degrade(in, d, p, 9, 4);
    EXPECT_EQ(a.data, Degrade(in, d, p, 9, 4).data);
    EXPECT_EQ(a.data, DegradeParallel(in, d, p, 9, 4, 3).data);
}

TEST(DegradationHeadlamp, ConfigIsStrict) {
    const std::string base =
        "name: t\nradiance_scale: 1.0\nenable_dust: true\ndust_beta: 0.1\ndust_model: headlamp\n"
        "dust_beta_backscatter: 0.1\ndust_backscatter: 0.3\ndust_lamp_focal_px: 640\n"
        "dust_lamp_cos_inner: 0.906\ndust_lamp_cos_outer: 0.766\ndust_lamp_in_camera_m: [0.05, -0.02, 0.02]\n";
    auto missing_order = StrictConfig::FromString(base);
    EXPECT_THROW(ParseDegradation(missing_order), ConfigError);   // headlamp needs dust_order pre_sensor
    auto ok = StrictConfig::FromString(base + "dust_order: pre_sensor\n");
    const auto d = ParseDegradation(ok);
    EXPECT_EQ(d.params.dust_model, DustModel::kHeadlamp);
    EXPECT_DOUBLE_EQ(d.params.dust_lamp_y_m, -0.02);
}
