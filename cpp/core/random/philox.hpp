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
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>

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
    /// Largest addressable site: the counter reserves 48 bits for it.
    static constexpr uint64_t kMaxSite  = (uint64_t(1) << 48) - 1;
    /// Largest refill block index: 16 bits, so 262144 words per site.
    static constexpr uint64_t kMaxBlock = 0xFFFFull;


    /// `seed` is the run seed, `frame_id` the frame, `site` a per-frame index
    /// (typically a pixel-block id). Together they name a unique, reproducible
    /// stream -- exactly the (seed, frame_id, pixel-block) keying of §5.6.
    ///
    /// DOMAIN: `site` must be < 2^48 (see kMaxSite). Philox4x32's counter is
    /// 128 bits and must hold frame_id, site AND the refill block index, so
    /// one of the three has to be bounded. frame_id keeps all 64 bits because
    /// kFixedPatternFrameId deliberately uses the all-ones sentinel; block
    /// takes 16 bits; site takes the remaining 48. The map from
    /// (frame_id, site, block) to the counter is then INJECTIVE, which is what
    /// makes the no-collision property structural rather than probable.
    PhiloxStream(uint64_t seed, uint64_t frame_id, uint64_t site) {
        // The assert is caller hygiene ONLY -- it is compiled out under NDEBUG,
        // which this project's Release build defines. The masking below is what
        // actually makes the behaviour defined, in every build: `site` IS a
        // 48-bit identifier, and passing more bits is a documented truncation
        // rather than an aliasing surprise. Callers holding a 64-bit id must
        // map it in with SiteFromHash64() rather than relying on truncation.
        assert(site <= kMaxSite && "site exceeds the 48-bit stream domain");
        site &= kMaxSite;
        key_.v[0] = static_cast<uint32_t>(seed & 0xFFFFFFFFu);
        key_.v[1] = static_cast<uint32_t>(seed >> 32);
        ctr_.v[0] = static_cast<uint32_t>(frame_id & 0xFFFFFFFFu);
        ctr_.v[1] = static_cast<uint32_t>(frame_id >> 32);
        ctr_.v[2] = static_cast<uint32_t>(site & 0xFFFFFFFFu);
        // Low 16 bits of v[3] carry site's high half; the top 16 are reserved
        // for the block index, which Refill() writes. Nothing overlaps.
        ctr_.v[3] = static_cast<uint32_t>((site >> 32) & 0xFFFFu);
        Refill();
    }

    /// Next 32 random bits.
    ///
    /// A site may draw up to 4 * 2^16 = 262144 words. Beyond that the block
    /// index would wrap into another site's stream, so it is an assert rather
    /// than a silent wrap: a wrapped stream reuses numbers, which is exactly
    /// the failure this class exists to prevent.
    uint32_t NextBits32() {
        if (used_ == 4) {
            ++block_;
            // NOT an assert. Wrapping past kMaxBlock would alias back onto
            // block 0 and silently REUSE this site's numbers -- the exact
            // failure this class exists to prevent, and invisible in the
            // output. A determinism substrate that quietly repeats itself is
            // worse than one that stops, so this is checked in every build.
            // Cost is one comparison per four draws.
            if (block_ > kMaxBlock) {
                throw std::logic_error(
                    "PhiloxStream: site exhausted after 262144 words; draw "
                    "from a new site rather than wrapping the block index");
            }
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
        // block_ is bounded by the check in NextBits32(), so this shift cannot
        // collide with the site bits below it.

        // PARTITION, not XOR-fold. The previous version did
        // `c.v[3] ^= block_`, which let (site=0, block=1) produce byte-for-byte
        // the same stream as (site=2^32, block=0) -- the counter words were
        // identical, so the "never collides" claim was false. Writing block
        // into its own reserved bit-field keeps the mapping injective.
        c.v[3] |= static_cast<uint32_t>(block_) << 16;
        buffer_ = Philox4x32_10(c, key_);
        used_ = 0;
    }

    Key2x32 key_;
    Counter4x32 ctr_;
    Counter4x32 buffer_;
    uint64_t block_ = 0;
    int used_ = 0;
};

/// Map an arbitrary 64-bit identifier into PhiloxStream's 48-bit site domain.
///
/// Phase 1+ ids (keyframe, landmark, submap) are naturally 64-bit hashes, and
/// passing one straight in would truncate to its low 48 bits -- so two ids
/// differing only above bit 47 would share a stream. This mixes all 64 bits
/// down first (splitmix64 finalizer) so every input bit affects the result.
/// Collisions are then birthday-bounded at ~2^-48 per pair rather than
/// systematic.
inline uint64_t SiteFromHash64(uint64_t id) {
    uint64_t z = id;
    z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ull;
    z ^= z >> 27; z *= 0x94D049BB133111EBull;
    z ^= z >> 31;
    return z & PhiloxStream::kMaxSite;
}

}  // namespace ariadne::core
