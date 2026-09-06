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

#include <cstdint>

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

struct DegradationParams {
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

/// Transmission through the participating medium at one pixel.
///   t = exp(-beta_local * depth),  beta_local modulated by a smooth field.
inline double DustTransmission(const DegradationParams& p, uint64_t seed,
                               uint64_t frame_id, int x, int y, double depth) {
    // Invalid or unknown depth: apply no attenuation rather than inventing a
    // distance. A recording with holes must not silently become foggier there.
    if (!(depth > 0.0)) return 1.0;

    double beta = p.dust_beta;
    if (p.dust_field_strength > 0.0 && p.dust_field_octaves > 0) {
        // frame_id is passed through unmodified and the stage travels in
        // salt_base, so the dust field uses the same keying discipline as every
        // per-pixel draw. Folding the stage into frame_id instead would permute
        // frames against one another.
        const double n = core::FractalNoise2D(
            seed, frame_id, x * p.dust_field_frequency,
            y * p.dust_field_frequency, p.dust_field_octaves,
            static_cast<uint64_t>(NoiseStage::kDustField));
        // n in [0,1] -> multiplier in [1-strength, 1+strength]
        beta *= 1.0 + p.dust_field_strength * (2.0 * n - 1.0);
    }
    if (beta <= 0.0) return 1.0;

    const double t = DetExp(-beta * depth);
    return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

inline double Clamp01(double v) {
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

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
/// See docs/DECISIONS.md D9: this order places dust after the sensor, so dust
/// does not influence shot-noise statistics. That is the documented order and
/// is implemented literally; the physical alternative is recorded there as an
/// open question rather than silently applied here.
inline ImageU8 Degrade(const ImageU8& input, const ImageF32& depth_gt,
                       const DegradationParams& params, uint64_t seed,
                       uint64_t frame_id) {
    ImageU8 out(input.width, input.height, input.channels);
    if (input.Empty()) return out;

    const bool use_dust =
        params.enable_dust && !depth_gt.Empty() && SameExtent(input, depth_gt);

    // Normalization for the vignette radius: 1.0 at the image corner.
    const double cx = 0.5 * (input.width - 1);
    const double cy = 0.5 * (input.height - 1);
    const double corner_r2 = cx * cx + cy * cy;
    const double inv_corner_r2 = corner_r2 > 0.0 ? 1.0 / corner_r2 : 0.0;

    const int levels = (1 << params.bit_depth) - 1;
    const double levels_d = static_cast<double>(levels);
    const double gain_total =
        params.exposure * params.analog_gain * params.digital_gain;

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            const size_t pixel_index =
                static_cast<size_t>(y) * input.width + x;

            const double dx = x - cx;
            const double dy = y - cy;
            const double vignette = detail::VignetteFactor(
                (dx * dx + dy * dy) * inv_corner_r2, params.vignette_strength);

            const double transmission =
                use_dust ? detail::DustTransmission(params, seed, frame_id, x, y,
                                                    depth_gt.At(x, y))
                         : 1.0;

            for (int c = 0; c < input.channels; ++c) {
                // Stage 1: scene radiance scale.
                double radiance =
                    static_cast<double>(input.At(x, y, c)) * (1.0 / 255.0);
                radiance *= params.radiance_scale;

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
                if (use_dust) {
                    value = quantized * transmission +
                            params.dust_airlight * (1.0 - transmission);
                    value = detail::Clamp01(value);
                }

                out.At(x, y, c) = static_cast<uint8_t>(
                    core::DetRoundToInt64(value * 255.0));
            }
        }
    }
    return out;
}

/// Overload for the no-depth case, so callers that never enable dust do not
/// have to construct an empty depth image.
inline ImageU8 Degrade(const ImageU8& input, const DegradationParams& params,
                       uint64_t seed, uint64_t frame_id) {
    return Degrade(input, ImageF32{}, params, seed, frame_id);
}

}  // namespace ariadne::perception
