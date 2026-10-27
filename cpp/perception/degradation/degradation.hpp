#pragma once
// ---------------------------------------------------------------------------
// The seeded degradation operator D (arch §5.6).
//
//   D(image, depth_gt, params, seed, frame_id) -> degraded image
//
// This is the mechanism that makes the C6 determinism guarantee real. Gazebo
// renders a scene ONCE and writes a clean recording; every perception, SLAM,
// collaboration and ablation run afterwards consumes that recording and applies
// D at read time. Identical pixels across every baseline follow by
// construction rather than by discipline.
//
// DIVISION OF LABOUR, from §5.6 and load-bearing: the lamp's OPTICAL effect
// (falloff, shadows, cone profile) is real Gazebo rendering and is NOT part of
// D. D degrades the SENSOR; the renderer degrades the SCENE. Anything here that
// started reasoning about light transport would be in the wrong file.
//
// DETERMINISM CONTRACT. Every value is produced from a counter-based generator
// keyed on (seed, frame_id, pixel, stage) and from arithmetic that is
// correctly rounded on every platform. No std:: distribution, no libm
// transcendental, no accumulation whose result depends on iteration order.
// The operator is therefore safe to parallelize over any partition of pixels.
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"
#include "core/math/det_math.hpp"
#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include "perception/degradation/srgb.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace ariadne::perception {

using core::DetExp;
using core::DetSqrt;
using core::ImageF32;
using core::ImageU8;
using core::PhiloxStream;

/// Independent random stream per stage, so that disabling one stage does not
/// perturb the values every later stage draws.
///
/// This matters for correctness of the Phase 5 ablations, not merely for
/// tidiness: if all stages shared a stream, turning off shot noise would shift
/// the bit sequence consumed by read noise, and the ablation would measure
/// stage-removal confounded with an unrelated reshuffle of the noise field.
enum class NoiseStage : uint64_t {
    kShot = 0,
    kRead = 1,
    kPrnu = 2,
    kDustField = 3,
    kStageCount = 8,   // padding, so adding a stage does not renumber existing ones
};

/// Frame id used for FIXED-PATTERN effects. PRNU is a property of the silicon,
/// not of the moment: it must be identical in every frame of a sequence.
/// Re-drawing it per frame would turn fixed-pattern noise into temporal noise,
/// which frame averaging removes for free, and the flat-field correction of
/// §6.2 would then appear far more effective than it is.
inline constexpr uint64_t kFixedPatternFrameId = 0xFFFFFFFFFFFFFFFFull;

/// How input pixel values map to radiance before stage 1 (D130).
enum class InputTransfer {
    kLinear,   // value / 255 (the default; every pinned digest and result uses it)
    kSrgb,     // sRGB decode: the gz-sim8/ogre2 render is sRGB-encoded (SMOKE §4u)
};

/// Where dust acts (D9; L9 option c, D135).
enum class DustOrder {
    kPostSensor,   // §5.6 as written: composited after quantization (the default)
    kPreSensor,    // on radiance, before shot noise: attenuation removes photons, airlight adds them (and their noise)
};

/// Which participating-medium model the dust stage uses (D139).
enum class DustModel {
    kKoschmieder,  // I = J t + A (1 - t), t = exp(-beta d): the daylight model of §5.6 (the default)
    kHeadlamp,     // algorithms.md §10.2: round-trip attenuation of the lamp-lit scene plus single-scatter backscatter of
                   // the carried lamp along each pixel's ray (near-field dominated; lamp offset is a parameter)
};

/// What the medium's added radiance is (D135).
enum class DustAirlight {
    kConstant,     // dust_airlight everywhere (the default)
    kLamp,         // backscatter of the carried lamp: dust_airlight * lamp_power * the lamp's lobe at the pixel
};

struct DegradationParams {
    // --- Input transfer (D130) --------------------------------------------
    InputTransfer input_transfer = InputTransfer::kLinear;

    // --- Stage 1: scene radiance scale (the "darkness level") -------------
    /// Multiplies incoming radiance. 1.0 is the clean image; 0.01 is roughly
    /// two decades below nominal exposure.
    double radiance_scale = 1.0;

    // --- Stage 2: Poisson shot noise --------------------------------------
    bool enable_shot_noise = true;
    /// Photons (electrons) collected at full scale. Sets the shot-noise floor:
    /// SNR at full scale is sqrt(photon_scale). 10000 e- is a plausible
    /// small-pixel full well. [ASSUMPTION] not calibrated against a real sensor.
    double photon_scale = 10000.0;
    /// Below this mean the Poisson sampler is exact; above it, Gaussian.
    /// Carried in params so the approximation appears in run manifests.
    double poisson_exact_max_lambda = core::kPoissonExactMaxLambda;

    // --- Stage 3: read noise ----------------------------------------------
    /// Additive Gaussian, in electrons, applied after photon conversion.
    /// Constant with signal, so it dominates in the dark. This is the term
    /// that makes low light hard rather than merely dim.
    double read_noise_electrons = 2.0;

    // --- Stage 4: PRNU and vignette ---------------------------------------
    /// Corner falloff as a fraction: 0.3 means the corner receives 70% of the
    /// centre. Modelled as cos^4, the standard optical law.
    double vignette_strength = 0.0;
    /// Photo-response non-uniformity: per-pixel multiplicative gain, drawn once
    /// per (seed, pixel) and identical across all frames.
    double prnu_sigma = 0.0;

    // --- Stage 5: exposure and gain ---------------------------------------
    double exposure = 1.0;
    double analog_gain = 1.0;
    double digital_gain = 1.0;

    // --- Stage 6: quantization and clipping -------------------------------
    /// Sensor quantization depth before mapping to 8-bit output.
    int bit_depth = 8;
    /// Clipping level in normalized units. Values above this saturate, which is
    /// the failure the overexposure demo of §6.5 depends on.
    double saturation = 1.0;

    // --- Stage 7: dust ----------------------------------------------------
    bool enable_dust = false;
    /// Extinction coefficient per metre in  t = exp(-beta * depth).
    double dust_beta = 0.0;
    /// Airlight A, in normalized units: the radiance the medium itself adds.
    double dust_airlight = 0.7;
    /// Lattice cells per pixel for the particulate field. Smaller means larger
    /// blobs.
    double dust_field_frequency = 0.01;
    int dust_field_octaves = 3;
    /// How strongly the local field modulates beta. 0 gives a homogeneous
    /// medium; 1 varies local density by +/-100%.
    double dust_field_strength = 0.5;

    // --- Dust fidelity options (L9, D135). Defaults are the operator above, byte for byte.
    DustOrder dust_order = DustOrder::kPostSensor;
    /// 0: the field is redrawn every frame (D11's "temporally white"). N > 0: interpolated between fields keyed on
    /// frame / N and frame / N + 1, so it decorrelates over ~N frames instead of 1.
    int dust_field_tau_frames = 0;
    /// Advection: the field moves this many pixels per frame.
    double dust_drift_x_px = 0.0, dust_drift_y_px = 0.0;
    DustAirlight dust_airlight_mode = DustAirlight::kConstant;
    /// The lamp's lobe for kLamp, as cosines of the inner / outer half-angles about the optical axis (cosines, not
    /// degrees: no libm on the per-pixel path), and the render's focal length in px. The 5 cm lamp offset is
    /// neglected: backscatter is a near-field veil along the view ray.
    double dust_lamp_cos_inner = 0.0, dust_lamp_cos_outer = 0.0, dust_lamp_focal_px = 0.0;

    // --- Headlamp dust model (D139; algorithms.md §10.2, after Akkaynak & Treibitz CVPR 2018/2019) -------------
    // Direct: the rendered (lamp-lit) scene J is attenuated out and back, J exp(-2 beta_D range), beta_D = dust_beta.
    // Backscatter, single scatter, phase function folded into the strength:
    //   B(px) = S * Int_{near}^{range} RID(theta(s)) / |s u - L|^2 * exp(-beta_B (s + |s u - L|)) ds  / B_ref
    // u = the pixel's unit ray, L = the lamp centre, theta = the angle from the lamp axis (+z). B_ref is the same
    // integral for a COAXIAL lamp on the optical axis over an infinitely deep scene, so `dust_backscatter` reads as
    // "the veil a coaxial lamp would make on axis", and offsetting the lamp physically lowers the veil.
    // Always pre-sensor (it is light): requires dust_order pre_sensor. Lamp offset in D moves only the veil; the
    // render's direct lighting and shadows stay as recorded (a stated limit of a replay sweep).
    DustModel dust_model = DustModel::kKoschmieder;
    double dust_beta_backscatter = 0.0;   // beta_B, per metre
    double dust_backscatter = 0.0;        // S: coaxial on-axis saturated veil, normalised radiance at lamp_power 1
    double dust_near_m = 0.02;            // particles nearer than this (lens housing) do not scatter into the image
    double dust_lamp_x_m = 0.0, dust_lamp_y_m = 0.0, dust_lamp_z_m = 0.0;   // lamp centre, camera optical frame

    // --- Replay lamp actuator (5.3, D132) ----------------------------------
    /// Lamp power relative to the recording's. With the lamp the only light and a linear(ised) render, scene
    /// radiance is exactly proportional to it (SMOKE §4u), so replay scales stage 1 by it; it also scales kLamp
    /// airlight.
    double lamp_power = 1.0;
};

namespace detail {

/// cos^4 vignetting. `r2_norm` is the squared radius normalized so that the
/// image corner is 1.
///
/// Derivation: for a thin lens, off-axis irradiance falls as cos^4(theta) with
/// tan(theta) = r/f. Writing a = tan^2(theta_corner) gives
/// v(r) = 1/(1 + a*r2_norm)^2, and `a` is chosen so v(1) equals the requested
/// corner response. Purely deterministic, no random draw.
inline double VignetteFactor(double r2_norm, double strength) {
    if (strength <= 0.0) return 1.0;
    const double corner = 1.0 - strength;             // desired v at the corner
    if (corner <= 0.0) return 0.0;
    // (1 + a)^2 = 1/corner  =>  a = 1/sqrt(corner) - 1
    const double a = 1.0 / DetSqrt(corner) - 1.0;
    const double denom = 1.0 + a * r2_norm;
    return 1.0 / (denom * denom);
}

/// Per-pixel fixed-pattern gain. Keyed on kFixedPatternFrameId so it does not
/// vary between frames.
inline double PrnuGain(uint64_t seed, uint64_t site, double sigma) {
    if (sigma <= 0.0) return 1.0;
    PhiloxStream stream(seed, kFixedPatternFrameId, site);
    const double g = 1.0 + sigma * core::SampleStandardNormal(stream);
    return g > 0.0 ? g : 0.0;      // a negative gain is not physical
}

/// Stream site id for one (pixel, channel, stage) triple.
inline uint64_t SiteId(size_t pixel_index, int channel, int channels,
                       NoiseStage stage) {
    const uint64_t per_channel =
        (static_cast<uint64_t>(pixel_index) * static_cast<uint64_t>(channels)) +
        static_cast<uint64_t>(channel);
    return per_channel * static_cast<uint64_t>(NoiseStage::kStageCount) +
           static_cast<uint64_t>(stage);
}

inline double Clamp01(double v) {
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

/// The lamp's lobe at a pixel for kLamp airlight: 1 inside the inner cone, 0 outside the outer, linear in cos(theta)
/// between. r2_px is the squared pixel distance from the image centre.
inline double LampLobe(const DegradationParams& p, double r2_px) {
    const double f = p.dust_lamp_focal_px;
    const double c = f / DetSqrt(f * f + r2_px);
    if (c >= p.dust_lamp_cos_inner) return 1.0;
    if (c <= p.dust_lamp_cos_outer) return 0.0;
    return (c - p.dust_lamp_cos_outer) / (p.dust_lamp_cos_inner - p.dust_lamp_cos_outer);
}

/// The local particulate density multiplier in [1 - strength, 1 + strength] (1 for a homogeneous medium).
inline double DustFieldMultiplier(const DegradationParams& p, uint64_t seed, uint64_t frame_id, int x, int y) {
    if (p.dust_field_strength > 0.0 && p.dust_field_octaves > 0) {
        // frame_id is passed through unmodified and the stage travels in
        // salt_base, so the dust field uses the same keying discipline as every
        // per-pixel draw. Folding the stage into frame_id instead would permute
        // frames against one another.
        const double xs = x - p.dust_drift_x_px * static_cast<double>(frame_id);
        const double ys = y - p.dust_drift_y_px * static_cast<double>(frame_id);
        const auto field = [&](uint64_t key) {
            return core::FractalNoise2D(seed, key, xs * p.dust_field_frequency, ys * p.dust_field_frequency,
                                        p.dust_field_octaves, static_cast<uint64_t>(NoiseStage::kDustField));
        };
        double n;
        if (p.dust_field_tau_frames <= 0) {
            n = field(frame_id);
        } else {
            // Blend the two epoch fields around this frame. Dividing by sqrt(a^2 + b^2) keeps the deviation's
            // variance constant through the blend; a plain lerp would halve it mid-epoch and the haze would pulse.
            const uint64_t tau = static_cast<uint64_t>(p.dust_field_tau_frames);
            const double b = static_cast<double>(frame_id % tau) / static_cast<double>(tau);
            const double a = 1.0 - b;
            const double dev = (a * (field(frame_id / tau) - 0.5) + b * (field(frame_id / tau + 1) - 0.5)) /
                               DetSqrt(a * a + b * b);
            n = Clamp01(0.5 + dev);
        }
        // n in [0,1] -> multiplier in [1-strength, 1+strength]
        return 1.0 + p.dust_field_strength * (2.0 * n - 1.0);
    }
    return 1.0;
}

/// Transmission through the participating medium at one pixel.
///   t = exp(-beta_local * depth),  beta_local modulated by a smooth field.
inline double DustTransmission(const DegradationParams& p, uint64_t seed,
                               uint64_t frame_id, int x, int y, double depth) {
    // Invalid or unknown depth: apply no attenuation rather than inventing a
    // distance. A recording with holes must not silently become foggier there.
    if (!(depth > 0.0)) return 1.0;

    const double beta = p.dust_beta * DustFieldMultiplier(p, seed, frame_id, x, y);
    if (beta <= 0.0) return 1.0;

    const double t = DetExp(-beta * depth);
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

/// The headlamp backscatter integral (D139), tabulated once per call: it depends on the pixel and the range only, so
/// a coarse pixel grid x geometric range nodes holds it, and per-pixel lookups use no transcendental at all. Nodes
/// and weights come from IEEE arithmetic plus DetExp/DetSqrt: same bytes on every platform.
struct HeadlampVeil {
    static constexpr int kStep = 8;            // coarse grid spacing, px (the veil is smooth across the image)
    static constexpr int kNodes = 96;          // range nodes, geometric from dust_near_m
    static constexpr double kRatio = 1.0879;   // 1.0879^95 ~ 2980: 0.02 m -> ~60 m (a literal: no pow at run time)
    int gw = 0, gh = 0;
    double cx = 0.0, cy = 0.0;
    std::vector<double> s;     // range nodes
    std::vector<double> cum;   // gw * gh * kNodes cumulative integrals, normalised by the coaxial reference

    static double Lobe(const DegradationParams& p, double cos_theta) {
        if (cos_theta >= p.dust_lamp_cos_inner) return 1.0;
        if (cos_theta <= p.dust_lamp_cos_outer) return 0.0;
        return (cos_theta - p.dust_lamp_cos_outer) / (p.dust_lamp_cos_inner - p.dust_lamp_cos_outer);
    }
    /// Cumulative trapezoid integral along the unit ray u from a lamp at l, into out[0..kNodes).
    void Integrate(const DegradationParams& p, double ux, double uy, double uz, double lx, double ly, double lz,
                   double* out) const {
        double prev_f = 0.0, acc = 0.0;
        for (int k = 0; k < kNodes; ++k) {
            const double vx = s[k] * ux - lx, vy = s[k] * uy - ly, vz = s[k] * uz - lz;
            const double r2 = vx * vx + vy * vy + vz * vz;
            const double r = DetSqrt(r2);
            const double f = r2 > 0.0 ? Lobe(p, vz / r) / r2 * DetExp(-p.dust_beta_backscatter * (s[k] + r)) : 0.0;
            if (k > 0) acc += 0.5 * (prev_f + f) * (s[k] - s[k - 1]);
            out[k] = acc;
            prev_f = f;
        }
    }
    HeadlampVeil(const DegradationParams& p, int width, int height) {
        s.resize(kNodes);
        s[0] = p.dust_near_m;
        for (int k = 1; k < kNodes; ++k) s[k] = s[k - 1] * kRatio;
        cx = 0.5 * (width - 1);
        cy = 0.5 * (height - 1);
        gw = (width - 1) / kStep + 2;
        gh = (height - 1) / kStep + 2;
        // The coaxial on-axis reference: a lamp at the camera centre, the optical-axis pixel, the whole range.
        std::vector<double> ref(kNodes);
        Integrate(p, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ref.data());
        const double norm = ref[kNodes - 1] > 0.0 ? 1.0 / ref[kNodes - 1] : 0.0;
        cum.resize(static_cast<size_t>(gw) * gh * kNodes);
        const double f = p.dust_lamp_focal_px;
        for (int gy = 0; gy < gh; ++gy)
            for (int gx = 0; gx < gw; ++gx) {
                const double rx = (gx * kStep - cx) / f, ry = (gy * kStep - cy) / f;
                const double inv = 1.0 / DetSqrt(rx * rx + ry * ry + 1.0);
                double* c = &cum[(static_cast<size_t>(gy) * gw + gx) * kNodes];
                Integrate(p, rx * inv, ry * inv, inv, p.dust_lamp_x_m, p.dust_lamp_y_m, p.dust_lamp_z_m, c);
                for (int k = 0; k < kNodes; ++k) c[k] *= norm;
            }
    }
    /// Normalised veil at grid node (gx, gy) for a scene at `range` metres.
    double NodeAt(int gx, int gy, double range) const {
        const double* c = &cum[(static_cast<size_t>(gy) * gw + gx) * kNodes];
        if (range <= s[0]) return 0.0;
        if (range >= s[kNodes - 1]) return c[kNodes - 1];
        const int k = static_cast<int>(std::upper_bound(s.begin(), s.end(), range) - s.begin());   // s[k-1] < range <= s[k]
        const double w = (range - s[k - 1]) / (s[k] - s[k - 1]);
        return c[k - 1] + w * (c[k] - c[k - 1]);
    }
    /// Bilinear in pixel position.
    double At(int x, int y, double range) const {
        const int gx = x / kStep, gy = y / kStep;
        const double fx = static_cast<double>(x - gx * kStep) / kStep, fy = static_cast<double>(y - gy * kStep) / kStep;
        return (1.0 - fy) * ((1.0 - fx) * NodeAt(gx, gy, range) + fx * NodeAt(gx + 1, gy, range)) +
               fy * ((1.0 - fx) * NodeAt(gx, gy + 1, range) + fx * NodeAt(gx + 1, gy + 1, range));
    }
    /// The range along the pixel's ray for a z-depth (GT depth is z, not range [ASSUMPTION: gz depth_camera]).
    double Range(const DegradationParams& p, int x, int y, double z) const {
        const double rx = (x - cx) / p.dust_lamp_focal_px, ry = (y - cy) / p.dust_lamp_focal_px;
        return z * DetSqrt(rx * rx + ry * ry + 1.0);
    }
};

}  // namespace detail

/// Apply the degradation chain.
///
/// `depth_gt` carries the recorded ground-truth depth in metres and may be
/// empty when dust is disabled. Using recorded GT depth is the correct physics
/// for the dust stage and is only possible because GT depth rides in the
/// recording (§5.3).
///
/// STAGE ORDER is exactly as listed in §5.6:
///   radiance scale -> shot noise -> read noise -> PRNU/vignette
///   -> exposure/gain -> quantize/clip -> dust
/// See docs/decisions.md D9: this order places dust after the sensor, so dust
/// does not influence shot-noise statistics. That is the documented order and
/// is implemented literally; the physical alternative is recorded there as an
/// open question rather than silently applied here.
namespace detail {

/// Rows [y0, y1) of Degrade into `out` (already sized). Every pixel depends on
/// its own (seed, frame_id, pixel, stage) keys only, so any row split gives the
/// same bytes.
inline void DegradeRows(const ImageU8& input, const ImageF32& depth_gt,
                        const DegradationParams& params, uint64_t seed,
                        uint64_t frame_id, int y0, int y1, ImageU8& out,
                        const HeadlampVeil* veil = nullptr) {
    const bool use_dust =
        params.enable_dust && !depth_gt.Empty() && SameExtent(input, depth_gt);
    const bool headlamp = use_dust && params.dust_model == DustModel::kHeadlamp && veil != nullptr;

    // Normalization for the vignette radius: 1.0 at the image corner.
    const double cx = 0.5 * (input.width - 1);
    const double cy = 0.5 * (input.height - 1);
    const double corner_r2 = cx * cx + cy * cy;
    const double inv_corner_r2 = corner_r2 > 0.0 ? 1.0 / corner_r2 : 0.0;

    const int levels = (1 << params.bit_depth) - 1;
    const double levels_d = static_cast<double>(levels);
    const double gain_total =
        params.exposure * params.analog_gain * params.digital_gain;
    const bool pre_dust = use_dust && params.dust_order == DustOrder::kPreSensor;
    const double radiance_gain = params.radiance_scale * params.lamp_power;   // lamp_power 1: exactly radiance_scale

    for (int y = y0; y < y1; ++y) {
        for (int x = 0; x < input.width; ++x) {
            const size_t pixel_index =
                static_cast<size_t>(y) * input.width + x;

            const double dx = x - cx;
            const double dy = y - cy;
            const double vignette = detail::VignetteFactor(
                (dx * dx + dy * dy) * inv_corner_r2, params.vignette_strength);

            double transmission = 1.0, veil_radiance = 0.0;
            if (headlamp) {
                const double z = depth_gt.At(x, y);
                if (z > 0.0) {   // unknown depth: no dust at all, as in the daylight model
                    const double range = veil->Range(params, x, y, z);
                    const double m = detail::DustFieldMultiplier(params, seed, frame_id, x, y);
                    const double t = DetExp(-params.dust_beta * m * 2.0 * range);   // out and back
                    transmission = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
                    veil_radiance = params.dust_backscatter * m * params.lamp_power * params.radiance_scale *
                                    veil->At(x, y, range);
                }
            } else if (use_dust) {
                transmission = detail::DustTransmission(params, seed, frame_id, x, y, depth_gt.At(x, y));
            }
            const double airlight =
                params.dust_airlight_mode == DustAirlight::kLamp
                    ? params.dust_airlight * params.lamp_power * detail::LampLobe(params, dx * dx + dy * dy)
                    : params.dust_airlight;

            for (int c = 0; c < input.channels; ++c) {
                // Stage 1: scene radiance scale.
                const uint8_t v = input.At(x, y, c);
                double radiance = params.input_transfer == InputTransfer::kSrgb
                                      ? kSrgbToLinear[v]
                                      : static_cast<double>(v) * (1.0 / 255.0);
                radiance *= radiance_gain;
                if (headlamp) {
                    radiance = radiance * transmission + veil_radiance;   // D139: the medium acts on light
                } else if (pre_dust) {
                    // D135: the medium acts on light. Airlight is light from the same lamp, so it is dimmed by the
                    // same darkness scale as the scene it veils.
                    radiance = radiance * transmission + airlight * params.radiance_scale * (1.0 - transmission);
                }

                // Stage 2: Poisson shot noise, in electrons.
                double electrons = radiance * params.photon_scale;
                if (params.enable_shot_noise) {
                    PhiloxStream s(seed, frame_id,
                                   detail::SiteId(pixel_index, c, input.channels,
                                                  NoiseStage::kShot));
                    electrons = static_cast<double>(core::SamplePoisson(
                        s, electrons, params.poisson_exact_max_lambda));
                }

                // Stage 3: read noise, additive and signal-independent.
                if (params.read_noise_electrons > 0.0) {
                    PhiloxStream s(seed, frame_id,
                                   detail::SiteId(pixel_index, c, input.channels,
                                                  NoiseStage::kRead));
                    electrons += params.read_noise_electrons *
                                 core::SampleStandardNormal(s);
                }

                // Stage 4: PRNU (fixed pattern) and vignette (deterministic).
                const double prnu = detail::PrnuGain(
                    seed,
                    detail::SiteId(pixel_index, c, input.channels,
                                   NoiseStage::kPrnu),
                    params.prnu_sigma);
                electrons *= prnu * vignette;

                // Stage 5: exposure and gain, back to normalized units.
                double signal = (electrons / params.photon_scale) * gain_total;

                // Stage 6: clip, then quantize to the sensor's bit depth.
                if (signal > params.saturation) signal = params.saturation;
                signal = detail::Clamp01(signal);
                const double quantized =
                    static_cast<double>(core::DetRoundToInt64(signal * levels_d)) /
                    levels_d;

                // Stage 7: dust compositing, I = J*t + A*(1-t).
                double value = quantized;
                if (use_dust && !pre_dust && !headlamp) {
                    value = quantized * transmission +
                            airlight * (1.0 - transmission);
                    value = detail::Clamp01(value);
                }

                out.At(x, y, c) = static_cast<uint8_t>(
                    core::DetRoundToInt64(value * 255.0));
            }
        }
    }
}

}  // namespace detail

inline ImageU8 Degrade(const ImageU8& input, const ImageF32& depth_gt,
                       const DegradationParams& params, uint64_t seed,
                       uint64_t frame_id) {
    ImageU8 out(input.width, input.height, input.channels);
    if (input.Empty()) return out;
    std::unique_ptr<detail::HeadlampVeil> veil;
    if (params.enable_dust && params.dust_model == DustModel::kHeadlamp)
        veil = std::make_unique<detail::HeadlampVeil>(params, input.width, input.height);
    detail::DegradeRows(input, depth_gt, params, seed, frame_id, 0, input.height, out, veil.get());
    return out;
}

/// Overload for the no-depth case, so callers that never enable dust do not
/// have to construct an empty depth image.
inline ImageU8 Degrade(const ImageU8& input, const DegradationParams& params,
                       uint64_t seed, uint64_t frame_id) {
    return Degrade(input, ImageF32{}, params, seed, frame_id);
}

/// The same bytes as Degrade, rows split into `threads` bands (2.2: D on a
/// 1280x720 RGB frame is ~1 s on one core). Byte identity is by the contract
/// above and is tested against Degrade, not assumed.
inline ImageU8 DegradeParallel(const ImageU8& input, const ImageF32& depth_gt,
                               const DegradationParams& params, uint64_t seed,
                               uint64_t frame_id, int threads) {
    ImageU8 out(input.width, input.height, input.channels);
    if (input.Empty()) return out;
    const int n = std::max(1, std::min(threads, input.height));
    std::unique_ptr<detail::HeadlampVeil> veil;
    if (params.enable_dust && params.dust_model == DustModel::kHeadlamp)
        veil = std::make_unique<detail::HeadlampVeil>(params, input.width, input.height);
    std::vector<std::thread> pool;
    for (int t = 0; t < n; ++t) {
        const int y0 = input.height * t / n, y1 = input.height * (t + 1) / n;
        pool.emplace_back([&, y0, y1] {
            detail::DegradeRows(input, depth_gt, params, seed, frame_id, y0, y1, out, veil.get());
        });
    }
    for (std::thread& th : pool) th.join();
    return out;
}

inline ImageU8 DegradeParallel(const ImageU8& input, const DegradationParams& params,
                               uint64_t seed, uint64_t frame_id, int threads) {
    return DegradeParallel(input, ImageF32{}, params, seed, frame_id, threads);
}

}  // namespace ariadne::perception
