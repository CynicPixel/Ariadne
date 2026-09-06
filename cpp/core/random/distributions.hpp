#pragma once
// ---------------------------------------------------------------------------
// Deterministic samplers built on the counter-based generator and the
// deterministic transcendentals.
//
// Every sampler here is bit-identical across machines, which rules out the
// std:: distributions: their ALGORITHMS are implementation-defined, so
// libstdc++ and libc++ produce different values from the same bits. Nothing in
// <random> may be used inside the degradation operator.
//
// The physical model these serve (arch §5.6) is Poisson-Gaussian:
//   photon arrival is Poisson (signal-dependent, so bright pixels carry more
//   absolute noise but better SNR), read noise is additive Gaussian (constant,
//   so it dominates in the dark). That asymmetry is why dark scenes degrade
//   the way they do, and why "add some noise" would not reproduce it.
// ---------------------------------------------------------------------------
#include "core/math/det_math.hpp"
#include "core/random/philox.hpp"

#include <cstdint>

namespace ariadne::core {

/// Round to nearest int64, ties away from zero, without libm.
inline int64_t DetRoundToInt64(double x) {
    return static_cast<int64_t>(x >= 0.0 ? x + 0.5 : x - 0.5);
}

/// Standard normal sample by the Marsaglia polar method.
///
/// Chosen over Box-Muller deliberately: Box-Muller needs sin and cos, which are
/// transcendentals we would have to implement and validate as well. The polar
/// method needs only log and sqrt, both of which already exist here.
///
/// The method draws a point uniformly in the unit square and REJECTS it unless
/// it falls inside the unit circle (acceptance probability pi/4, so ~1.27 draws
/// per accepted sample on average). Rejection is safe for determinism because
/// the rejected draws come from this site's own stream: the sequence of
/// candidates is itself a deterministic function of (seed, frame_id, site), so
/// the accepted value is too, however many rejections it took.
///
/// Generates two independent normals; the second is discarded. Caching it would
/// couple consecutive calls on one stream, which is a correctness hazard for no
/// meaningful gain in a per-pixel workload.
inline double SampleStandardNormal(PhiloxStream& stream) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const double u = 2.0 * stream.NextUniform() - 1.0;
        const double v = 2.0 * stream.NextUniform() - 1.0;
        const double s = u * u + v * v;
        if (s < 1.0 && s > 0.0) {
            // z = u * sqrt(-2 ln(s) / s)
            return u * DetSqrt(-2.0 * DetLog(s) / s);
        }
    }
    // Unreachable in practice: rejection probability is (1 - pi/4) ~ 0.215, so
    // 100 consecutive rejections has probability ~1e-66. Returning 0 rather
    // than looping forever keeps the function total.
    return 0.0;
}

inline double SampleNormal(PhiloxStream& stream, double mean, double stddev) {
    return mean + stddev * SampleStandardNormal(stream);
}

/// Above this mean, Poisson sampling switches from exact to a Gaussian
/// approximation. Two independent reasons for a threshold in this region:
///   - COST: Knuth's method takes O(lambda) uniform draws. At 720p with
///     lambda ~ 1000 photons/pixel that is ~9e8 draws per frame, which is not
///     viable. The approximation is O(1).
///   - VALIDITY: the Poisson distribution converges to Normal(lambda, lambda)
///     as lambda grows; by lambda = 30 the relative error in the tails is
///     already small compared with the other approximations in the sensor chain.
/// [PROXY] The Gaussian branch is an approximation, not exact Poisson. It is
/// stated here and carried in DegradationParams so it appears in run manifests
/// rather than being buried.
inline constexpr double kPoissonExactMaxLambda = 30.0;

/// Poisson sample with mean `lambda`.
///
/// Exact (Knuth) below the threshold, Gaussian-approximated above it. Both
/// branches are deterministic; the branch taken is a function of lambda alone,
/// so it does not vary between machines.
inline int64_t SamplePoisson(PhiloxStream& stream, double lambda,
                             double exact_max_lambda = kPoissonExactMaxLambda) {
    if (!(lambda > 0.0)) return 0;          // also catches NaN

    if (lambda < exact_max_lambda) {
        // Knuth: multiply uniforms until the product drops below e^-lambda.
        // Exact for the Poisson distribution. e^-lambda is representable down
        // to lambda ~ 745, so no underflow concern in this branch.
        const double target = DetExp(-lambda);
        double product = 1.0;
        int64_t k = 0;
        // The loop is bounded in practice at ~lambda + O(sqrt(lambda)); the
        // hard cap prevents an unbounded loop if target underflows to zero.
        while (k < 10000) {
            product *= stream.NextUniform();
            if (product <= target) break;
            ++k;
        }
        return k;
    }

    // Gaussian approximation: Poisson(lambda) -> Normal(lambda, lambda).
    // Clamped at zero because a photon count cannot be negative.
    const double z = SampleStandardNormal(stream);
    const int64_t n = DetRoundToInt64(lambda + DetSqrt(lambda) * z);
    return n > 0 ? n : 0;
}

/// Value-noise lattice sample in [0, 1), used to build the spatially varying
/// particulate field of §5.6.
///
/// Keyed on (seed, frame_id, lattice cell) exactly like every other random
/// draw, so the field is reproducible and does not depend on evaluation order.
inline double LatticeNoise(uint64_t seed, uint64_t frame_id, int64_t ix,
                           int64_t iy, uint64_t salt) {
    // Fold the 2D cell index into one site id. The multipliers are large odd
    // constants; their only job is to keep distinct (ix, iy) pairs from
    // colliding for the ranges an image uses.
    const uint64_t site = static_cast<uint64_t>(ix) * 0x9E3779B97F4A7C15ull ^
                          static_cast<uint64_t>(iy) * 0xC2B2AE3D27D4EB4Full ^
                          salt * 0x165667B19E3779F9ull;
    return PhiloxStream(seed, frame_id, site).NextUniform();
}

/// Smoothstep, the standard 3t^2 - 2t^3 interpolant. Continuous first
/// derivative, so the interpolated field has no visible lattice creases.
inline double SmoothStep(double t) { return t * t * (3.0 - 2.0 * t); }

/// Bilinearly interpolated, smoothstep-eased value noise at (x, y), where the
/// unit of (x, y) is one lattice cell.
///
/// This is value noise rather than true Perlin gradient noise. §5.6 asks for "a
/// Perlin local particulate field"; the visual role is a smooth, spatially
/// correlated multiplier on dust density, which value noise fills at lower cost
/// and with fewer random draws per pixel. [PROXY] Recorded as a deviation from
/// the literal wording of §5.6.
inline double ValueNoise2D(uint64_t seed, uint64_t frame_id, double x, double y,
                           uint64_t salt = 0) {
    const double fx = x >= 0.0 ? x : x - 1.0;   // deterministic floor
    const double fy = y >= 0.0 ? y : y - 1.0;
    const int64_t ix = static_cast<int64_t>(fx);
    const int64_t iy = static_cast<int64_t>(fy);
    const double tx = SmoothStep(x - static_cast<double>(ix));
    const double ty = SmoothStep(y - static_cast<double>(iy));

    const double n00 = LatticeNoise(seed, frame_id, ix,     iy,     salt);
    const double n10 = LatticeNoise(seed, frame_id, ix + 1, iy,     salt);
    const double n01 = LatticeNoise(seed, frame_id, ix,     iy + 1, salt);
    const double n11 = LatticeNoise(seed, frame_id, ix + 1, iy + 1, salt);

    const double a = n00 + tx * (n10 - n00);
    const double b = n01 + tx * (n11 - n01);
    return a + ty * (b - a);
}

/// Maximum octaves a single `salt_base` may span before two stages could
/// collide in the salt dimension. Fractal noise beyond 64 octaves is
/// meaningless anyway (amplitude has long since underflowed).
inline constexpr uint64_t kMaxNoiseOctaves = 64;

/// Fractal (multi-octave) value noise in [0, 1). Successive octaves double the
/// frequency and halve the amplitude, which is what gives a particulate field
/// structure at more than one scale.
///
/// `salt_base` identifies the CALLER (which noise field this is). The octave
/// index is folded in beneath it rather than beside it, so a caller's salt and
/// an octave index can never collide. Callers must not encode themselves into
/// `frame_id`: mixing a stage id into the frame dimension permutes frames
/// against each other, which is both fragile and hard to reason about.
inline double FractalNoise2D(uint64_t seed, uint64_t frame_id, double x,
                             double y, int octaves, uint64_t salt_base = 0,
                             double lacunarity = 2.0, double gain = 0.5) {
    double sum = 0.0;
    double amplitude = 1.0;
    double norm = 0.0;
    double fx = x;
    double fy = y;
    for (int o = 0; o < octaves; ++o) {
        const uint64_t salt =
            salt_base * kMaxNoiseOctaves + static_cast<uint64_t>(o);
        sum += amplitude * ValueNoise2D(seed, frame_id, fx, fy, salt);
        norm += amplitude;
        amplitude *= gain;
        fx *= lacunarity;
        fy *= lacunarity;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}

}  // namespace ariadne::core
