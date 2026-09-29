#pragma once
// ---------------------------------------------------------------------------
// Determinism digests: ONE implementation, used by the tests, by
// infra/scripts/check_determinism.sh, and by the run-manifest writer.
//
// Previously each digest was written out twice (once in the shell script's
// inline C++, once in a test) and the two agreed only because they were kept
// in sync by hand. A digest that exists twice can drift silently, and then the
// committed constant guards one copy while the other is what actually runs.
//
// The run manifest calls these at RUN TIME rather than copying the committed
// constants: the point of recording them (D21) is that a result produced by
// drifted math is identifiable after the fact, which only works if the value
// in the manifest was measured on the machine that produced the result.
// ---------------------------------------------------------------------------
#include "core/math/det_math.hpp"
#include "core/random/philox.hpp"

#include <cstdint>

namespace ariadne::core {

/// FNV-1a over the 8 little-endian bytes of a 64-bit value.
class Fnv1a64 {
public:
    void AbsorbU64(uint64_t v) {
        for (int i = 0; i < 8; ++i) { h_ ^= (v >> (i * 8)) & 0xFFull; h_ *= kPrime; }
    }
    void AbsorbByte(uint8_t b) { h_ ^= b; h_ *= kPrime; }
    uint64_t value() const { return h_; }

private:
    static constexpr uint64_t kOffsetBasis = 1469598103934665603ull;
    static constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t h_ = kOffsetBasis;
};

/// Raw IEEE bits of DetExp/DetLog over a fixed grid.
///
/// BOUNDARY COVERAGE (D18). The first two sweeps span only |x| <= 500, and on
/// their own they stayed byte-identical through a real DetExp bug: the range
/// reduction produced k=1024 (resp. -1075) only near the overflow/underflow
/// limits, where Exp2Int saturated. A digest that never evaluates the boundary
/// of a function's domain cannot protect it.
inline uint64_t DetMathDigest() {
    Fnv1a64 h;
    for (int i = -2000; i <= 2000; ++i) h.AbsorbU64(DoubleToBits(DetExp(i * 0.25)));
    for (int i = 1; i <= 4000; ++i) h.AbsorbU64(DoubleToBits(DetLog(i * 0.125)));
    for (int i = 0; i <= 400; ++i) h.AbsorbU64(DoubleToBits(DetExp(708.0 + i * 0.00445)));
    for (int i = 0; i <= 400; ++i) h.AbsorbU64(DoubleToBits(DetExp(-745.1332191019411 + i * 0.0028)));
    return h.value();
}

/// Philox output over sites, high site bits, the domain ends, and a draw deep
/// enough to reach the last legal block index.
///
/// As above (D18): sites 0..20000 drawing 2 words each leave block=0 and
/// site>>32 == 0, so that sweep alone was blind to the counter layout and
/// stayed byte-identical through the repartition that fixed a real collision.
inline uint64_t PhiloxDigest() {
    Fnv1a64 h;
    for (uint64_t site = 0; site < 20000; ++site) {
        h.AbsorbU64(PhiloxStream(20260906ull, 17ull, site).NextBits64());
    }
    for (uint64_t hi = 0; hi < 64; ++hi) {
        h.AbsorbU64(PhiloxStream(20260906ull, 17ull, (hi << 32) | 0x1234ull).NextBits64());
    }
    h.AbsorbU64(PhiloxStream(20260906ull, 17ull, PhiloxStream::kMaxSite).NextBits64());
    h.AbsorbU64(PhiloxStream(20260906ull, 17ull, PhiloxStream::kMaxSite - 1).NextBits64());
    PhiloxStream deep(20260906ull, 17ull, 99ull);
    for (uint64_t i = 0; i < 2 * (PhiloxStream::kMaxBlock + 1); ++i) h.AbsorbU64(deep.NextBits64());
    return h.value();
}

}  // namespace ariadne::core
