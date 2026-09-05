#pragma once
// ---------------------------------------------------------------------------
// Deterministic transcendental functions.
//
// WHY THESE EXIST (and why calling libm here would be a bug):
// IEEE 754 requires +, -, *, / and sqrt to be CORRECTLY ROUNDED -- given the
// same inputs they produce the same bits on every conforming platform. The
// standard imposes no such requirement on exp, log, sin, cos or pow. libm
// implementations therefore differ, legitimately, between glibc versions and
// between x86-64 and AArch64 (different polynomial degrees, different
// range-reduction, sometimes different vector paths for the same call).
//
// The Phase 0 gate requires D(image, params, seed, frame_id) to be
// bit-identical across machines. D needs exp(-lambda) for Poisson noise and
// exp(-beta*depth) for dust transmission, so a libm call would silently break
// the guarantee -- silently, because the difference is one ulp and only shows
// up as a checksum mismatch months later.
//
// Everything below is built from correctly-rounded operations only, so its
// output is reproducible by construction. Accuracy is verified against libm in
// the tests: libm is the ACCURACY oracle, never the runtime path.
//
// Compiled under -ffp-contract=off (set project-wide), so no multiply-add is
// fused into an FMA -- fusion would change results on targets that have FMA.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <limits>

namespace ariadne::core {

// ln(2) split into two exactly-representable halves. The reduction
// r = x - k*LN2_HI - k*LN2_LO is far more accurate than subtracting a single
// rounded ln2, because k*LN2_HI is exact (LN2_HI has trailing zero bits).
inline constexpr double kLn2Hi = 6.93147180369123816490e-01;
inline constexpr double kLn2Lo = 1.90821492927058770002e-10;
inline constexpr double kInvLn2 = 1.44269504088896338700e+00;   // 1/ln(2)

/// Reinterpret the bits of a double as a uint64. memcpy is the only
/// strict-aliasing-safe spelling and compiles to a register move.
inline uint64_t DoubleToBits(double x) {
    uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

inline double BitsToDouble(uint64_t bits) {
    double x;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

/// Construct 2^k exactly, for |k| <= 1023, by writing the exponent field.
/// Exact: a power of two is representable with a zero mantissa, so this
/// introduces no rounding at all.
inline double Exp2Int(int k) {
    if (k > 1023) return std::numeric_limits<double>::infinity();
    if (k < -1074) return 0.0;
    if (k >= -1022) {
        // Normal range: exponent field is biased by 1023.
        return BitsToDouble(static_cast<uint64_t>(k + 1023) << 52);
    }
    // Subnormal range: build 2^-1022 then halve repeatedly. Each halving is
    // exact until the value underflows to zero.
    double v = BitsToDouble(uint64_t(1) << 52);   // 2^-1022
    for (int i = 0; i < (-1022 - k); ++i) v *= 0.5;
    return v;
}

/// Round to nearest integer, ties away from zero, without libm.
/// Used only for the exp range reduction, where the exact tie behaviour is
/// irrelevant (any nearby integer k gives a valid reduction) but DETERMINISM
/// is essential.
inline int RoundToInt(double x) {
    return static_cast<int>(x >= 0.0 ? x + 0.5 : x - 0.5);
}

/// e^x, accurate to a few ulp and bit-identical on every platform.
///
/// Method: range-reduce x = k*ln2 + r with |r| <= ln2/2, evaluate e^r by a
/// Taylor series in Horner form, then scale by 2^k. The series is carried to
/// r^15/15!, which bounds the truncation error at ~1e-19 relative for
/// |r| <= 0.347 -- below double precision, so the polynomial is not the
/// limiting factor.
///
/// A minimax polynomial would need fewer terms; Taylor is used because its
/// coefficients are self-evidently correct on inspection, and this function is
/// audited far more often than it is optimized. If profiling ever puts this on
/// a hot path, replace it WITH a test against this version.
inline double DetExp(double x) {
    if (x != x) return x;                                        // NaN
    if (x > 709.782712893384) return std::numeric_limits<double>::infinity();
    if (x < -745.133219101941) return 0.0;                       // underflow
    if (x == 0.0) return 1.0;

    const int k = RoundToInt(x * kInvLn2);
    const double kd = static_cast<double>(k);
    const double r = (x - kd * kLn2Hi) - kd * kLn2Lo;

    // Horner: 1 + r(1 + r/2(1 + r/3(...)))  ==  sum r^n/n!
    double poly = 1.0 / 1307674368000.0;              // 1/15!
    poly = poly * r + 1.0 / 87178291200.0;            // 1/14!
    poly = poly * r + 1.0 / 6227020800.0;             // 1/13!
    poly = poly * r + 1.0 / 479001600.0;              // 1/12!
    poly = poly * r + 1.0 / 39916800.0;               // 1/11!
    poly = poly * r + 1.0 / 3628800.0;                // 1/10!
    poly = poly * r + 1.0 / 362880.0;                 // 1/9!
    poly = poly * r + 1.0 / 40320.0;                  // 1/8!
    poly = poly * r + 1.0 / 5040.0;                   // 1/7!
    poly = poly * r + 1.0 / 720.0;                    // 1/6!
    poly = poly * r + 1.0 / 120.0;                    // 1/5!
    poly = poly * r + 1.0 / 24.0;                     // 1/4!
    poly = poly * r + 1.0 / 6.0;                      // 1/3!
    poly = poly * r + 1.0 / 2.0;                      // 1/2!
    poly = poly * r + 1.0;                            // 1/1!
    poly = poly * r + 1.0;                            // 1/0!

    return poly * Exp2Int(k);
}

/// ln(x), accurate to a few ulp and bit-identical on every platform.
///
/// Method: split x = m * 2^e with m in [sqrt(1/2), sqrt(2)), so that
/// s = (m-1)/(m+1) satisfies |s| <= 0.1716, then use the rapidly converging
/// odd series ln(m) = 2*(s + s^3/3 + s^5/5 + ...). Carried to s^21/21, whose
/// truncation error is ~8e-18 relative -- again below double precision.
///
/// The atanh form is used rather than a series in (m-1) because it converges
/// in odd powers of a much smaller argument: |s| <= 0.17 against |m-1| <= 0.41.
inline double DetLog(double x) {
    if (x != x) return x;                                        // NaN
    if (x < 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (x == 0.0) return -std::numeric_limits<double>::infinity();
    if (x == std::numeric_limits<double>::infinity()) return x;
    if (x == 1.0) return 0.0;

    uint64_t bits = DoubleToBits(x);
    int exponent = static_cast<int>((bits >> 52) & 0x7FF);

    if (exponent == 0) {
        // Subnormal: scale up by 2^64 (exact) and correct the exponent after.
        x *= 18446744073709551616.0;
        bits = DoubleToBits(x);
        exponent = static_cast<int>((bits >> 52) & 0x7FF);
        exponent -= 64;
    }

    int e = exponent - 1023;
    // Mantissa with the exponent field forced to 0 => m in [1, 2).
    double m = BitsToDouble((bits & 0x000FFFFFFFFFFFFFull) | (uint64_t(1023) << 52));

    // Re-centre to [sqrt(1/2), sqrt(2)) so |s| is minimized. The comparison
    // constant is sqrt(2); the halving is exact.
    if (m > 1.4142135623730951) {
        m *= 0.5;
        e += 1;
    }

    const double s = (m - 1.0) / (m + 1.0);
    const double s2 = s * s;

    // Horner in s^2 for  s + s^3/3 + s^5/5 + ... + s^21/21.
    double poly = 1.0 / 21.0;
    poly = poly * s2 + 1.0 / 19.0;
    poly = poly * s2 + 1.0 / 17.0;
    poly = poly * s2 + 1.0 / 15.0;
    poly = poly * s2 + 1.0 / 13.0;
    poly = poly * s2 + 1.0 / 11.0;
    poly = poly * s2 + 1.0 / 9.0;
    poly = poly * s2 + 1.0 / 7.0;
    poly = poly * s2 + 1.0 / 5.0;
    poly = poly * s2 + 1.0 / 3.0;
    poly = poly * s2 + 1.0;

    const double log_m = 2.0 * s * poly;
    const double ed = static_cast<double>(e);
    // Two-part ln2 again, so the exponent contribution is accurate.
    return ed * kLn2Hi + (log_m + ed * kLn2Lo);
}

/// sqrt is IEEE-754 correctly rounded, so the library version is already
/// bit-identical everywhere. Wrapped only so that call sites in D read
/// uniformly as "deterministic math" and nobody has to re-derive why sqrt is
/// allowed here while exp is not.
inline double DetSqrt(double x) {
    return __builtin_sqrt(x);
}

}  // namespace ariadne::core
