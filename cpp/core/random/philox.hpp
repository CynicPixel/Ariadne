#pragma once
// ---------------------------------------------------------------------------
// Philox4x32-10 -- a counter-based ("stateless") pseudo-random generator.
// Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3",
// SC'11 (the Random123 paper).
//
// WHY THIS AND NOT std::mt19937:
// A conventional PRNG carries state, so its output depends on HOW MANY TIMES it
// has been called. That makes results depend on iteration order and thread
// count -- fatal for the Phase 0 gate, which requires D(image, params, seed,
// frame_id) to be bit-identical across machines.
//
// Philox is a keyed hash: value = f(key, counter). Pixel 5000's noise is the
// same bits whether it is computed first, last, on 1 thread or on 64, on amd64
// or on Graviton. It uses ONLY 32-bit integer multiply/xor/add, so there is no
// floating-point and therefore no architectural variation at all.
//
// The (seed, frame_id, pixel_block) keying that §5.6 specifies maps directly
// onto the key/counter split below.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstring>

namespace ariadne::core {

/// Philox round constants, verbatim from the reference implementation.
inline constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
inline constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
/// Key-bump constants: the golden ratio and sqrt(3)-1 in 32-bit fixed point.
/// They are arbitrary odd constants; their only job is to decorrelate rounds.
inline constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
inline constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

struct Counter4x32 {
    uint32_t v[4] = {0, 0, 0, 0};
};

struct Key2x32 {
    uint32_t v[2] = {0, 0};
};

/// Full 64-bit product of two 32-bit values, split into halves.
/// Written through uint64_t so it is exact and identical on every platform.
inline void MulHiLo32(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *hi = static_cast<uint32_t>(product >> 32);
    *lo = static_cast<uint32_t>(product & 0xFFFFFFFFu);
}

/// One Philox round: a Feistel-like mix of the four counter words.
inline Counter4x32 PhiloxRound(const Counter4x32& ctr, const Key2x32& key) {
    uint32_t hi0, lo0, hi1, lo1;
    MulHiLo32(kPhiloxM0, ctr.v[0], &hi0, &lo0);
    MulHiLo32(kPhiloxM1, ctr.v[2], &hi1, &lo1);

    Counter4x32 out;
    out.v[0] = hi1 ^ ctr.v[1] ^ key.v[0];
    out.v[1] = lo1;
    out.v[2] = hi0 ^ ctr.v[3] ^ key.v[1];
    out.v[3] = lo0;
    return out;
}

inline Key2x32 PhiloxBumpKey(const Key2x32& key) {
    Key2x32 out;
    out.v[0] = key.v[0] + kPhiloxW0;
    out.v[1] = key.v[1] + kPhiloxW1;
    return out;
}

/// Philox4x32 with 10 rounds -- the standard, well-analysed variant.
/// Returns 4 x 32 bits of high-quality randomness for the given (key, counter).
inline Counter4x32 Philox4x32_10(Counter4x32 ctr, Key2x32 key) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) key = PhiloxBumpKey(key);
        ctr = PhiloxRound(ctr, key);
    }
    return ctr;
}

// --- Convenience layer over the raw generator ----------------------------

/// Convert 32 random bits to a double in [0, 1).
///
/// Uses a multiply by 2^-32 rather than a modulo: the multiply is exact (2^-32
/// is a power of two, so this only adjusts the exponent) and therefore
/// bit-identical everywhere, while modulo would introduce bias.
inline double UniformFromBits32(uint32_t bits) {
    return static_cast<double>(bits) * (1.0 / 4294967296.0);   // 2^-32
}

/// 53 random bits -> a double in [0, 1) with full mantissa resolution.
/// Needed where 32 bits of resolution would visibly quantize a distribution.
inline double UniformFromBits64(uint64_t bits) {
    // Take the top 53 bits; scale by 2^-53. Both steps are exact.
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

/// A small, explicit stream of random values for one (seed, frame, index) site.
///
/// This is the type the degradation operator uses. It deliberately does NOT
/// look like a std::random engine: there is no global state and no
/// "advance the engine" semantics that could couple two call sites together.
class PhiloxStream {
public:
    /// `seed` is the run seed, `frame_id` the frame, `site` a per-frame index
    /// (typically a pixel-block id). Together they name a unique, reproducible
    /// stream -- exactly the (seed, frame_id, pixel-block) keying of §5.6.
    PhiloxStream(uint64_t seed, uint64_t frame_id, uint64_t site) {
        key_.v[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFu);
        key_.v[1] = static_cast<uint32_t>(seed >> 32);
        ctr_.v[0] = static_cast<uint32_t>(frame_id & 0xFFFFFFFFu);
        ctr_.v[1] = static_cast<uint32_t>(frame_id >> 32);
        ctr_.v[2] = static_cast<uint32_t>(site & 0xFFFFFFFFu);
        ctr_.v[3] = static_cast<uint32_t>(site >> 32);
        Refill();
    }

    /// Next 32 random bits.
    uint32_t NextBits32() {
        if (used_ == 4) {
            // Advance the counter's high word rather than re-keying, so a
            // single site can draw an unlimited number of values without ever
            // colliding with a neighbouring site's stream.
            ++block_;
            Refill();
        }
        return buffer_.v[used_++];
    }

    uint64_t NextBits64() {
        const uint64_t hi = NextBits32();
        const uint64_t lo = NextBits32();
        return (hi << 32) | lo;
    }

    /// Uniform double in [0, 1) with 53-bit resolution.
    double NextUniform() { return UniformFromBits64(NextBits64()); }

    /// Uniform double in [lo, hi).
    double NextUniform(double lo, double hi) {
        return lo + (hi - lo) * NextUniform();
    }

private:
    void Refill() {
        Counter4x32 c = ctr_;
        // Mix the block index into a word the caller does not control, so that
        // successive refills for one site are independent.
        c.v[3] ^= static_cast<uint32_t>(block_);
        buffer_ = Philox4x32_10(c, key_);
        used_ = 0;
    }

    Key2x32 key_;
    Counter4x32 ctr_;
    Counter4x32 buffer_;
    uint64_t block_ = 0;
    int used_ = 0;
};

}  // namespace ariadne::core
