// Philox4x32-10 correctness and determinism.
//
// The known-answer tests below are the authoritative vectors from the
// Random123 reference implementation (DEShawResearch/random123,
// tests/kat_vectors), fetched 2026-09-06. Passing them proves our generator
// produces the SAME stream as the published algorithm -- so the noise field is
// reproducible by anyone, not merely self-consistent.
#include "core/random/philox.hpp"

#include <gtest/gtest.h>

#include <map>
#include <set>
#include <thread>
#include <vector>

using namespace ariadne::core;

namespace {

Counter4x32 Ctr(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    Counter4x32 x;
    x.v[0] = a; x.v[1] = b; x.v[2] = c; x.v[3] = d;
    return x;
}

Key2x32 Key(uint32_t a, uint32_t b) {
    Key2x32 k;
    k.v[0] = a; k.v[1] = b;
    return k;
}

void ExpectCounterEq(const Counter4x32& got, uint32_t a, uint32_t b, uint32_t c,
                     uint32_t d) {
    EXPECT_EQ(got.v[0], a) << "word 0";
    EXPECT_EQ(got.v[1], b) << "word 1";
    EXPECT_EQ(got.v[2], c) << "word 2";
    EXPECT_EQ(got.v[3], d) << "word 3";
}

}  // namespace

// --- Known-answer tests (authoritative) ----------------------------------

TEST(Philox, KatAllZeros) {
    ExpectCounterEq(Philox4x32_10(Ctr(0, 0, 0, 0), Key(0, 0)),
                    0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u);
}

TEST(Philox, KatAllOnes) {
    ExpectCounterEq(Philox4x32_10(Ctr(0xffffffffu, 0xffffffffu, 0xffffffffu,
                                      0xffffffffu),
                                  Key(0xffffffffu, 0xffffffffu)),
                    0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu);
}

TEST(Philox, KatPiDigits) {
    ExpectCounterEq(Philox4x32_10(Ctr(0x243f6a88u, 0x85a308d3u, 0x13198a2eu,
                                      0x03707344u),
                                  Key(0xa4093822u, 0x299f31d0u)),
                    0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u);
}

TEST(Philox, MulHiLoIsExact) {
    uint32_t hi, lo;
    MulHiLo32(0xFFFFFFFFu, 0xFFFFFFFFu, &hi, &lo);
    EXPECT_EQ(hi, 0xFFFFFFFEu);
    EXPECT_EQ(lo, 0x00000001u);
    MulHiLo32(0, 12345, &hi, &lo);
    EXPECT_EQ(hi, 0u);
    EXPECT_EQ(lo, 0u);
}

// --- The properties the Phase 0 gate actually depends on ------------------

// Statelessness: the value at a site does not depend on what was computed
// before it. This is the property mt19937 cannot provide and the reason a
// counter-based generator was chosen.
TEST(Philox, ValueAtSiteIsIndependentOfAccessOrder) {
    const uint64_t seed = 42, frame = 7;
    std::vector<double> forward, backward;
    for (uint64_t site = 0; site < 256; ++site) {
        forward.push_back(PhiloxStream(seed, frame, site).NextUniform());
    }
    for (int site = 255; site >= 0; --site) {
        backward.push_back(PhiloxStream(seed, frame, site).NextUniform());
    }
    for (size_t i = 0; i < forward.size(); ++i) {
        EXPECT_EQ(forward[i], backward[backward.size() - 1 - i]) << "site " << i;
    }
}

// The same property under real concurrency: thread count must not change a
// single bit. D will be parallelized over pixel blocks, so this is the test
// that keeps that safe.
TEST(Philox, ThreadCountDoesNotChangeResults) {
    const uint64_t seed = 99, frame = 3;
    const int kSites = 4096;

    std::vector<double> serial(kSites);
    for (int i = 0; i < kSites; ++i) {
        serial[i] = PhiloxStream(seed, frame, i).NextUniform();
    }

    std::vector<double> parallel(kSites);
    const int kThreads = 8;
    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t) {
        pool.emplace_back([&, t] {
            for (int i = t; i < kSites; i += kThreads) {
                parallel[i] = PhiloxStream(seed, frame, i).NextUniform();
            }
        });
    }
    for (auto& th : pool) th.join();

    EXPECT_EQ(serial, parallel);   // exact equality, not approximate
}

// Distinct (seed, frame, site) triples must not collide -- otherwise two pixels
// or two frames would share a noise value and the noise field would show
// structure.
TEST(Philox, DistinctSitesGiveDistinctStreams) {
    std::set<uint64_t> seen;
    for (uint64_t frame = 0; frame < 16; ++frame) {
        for (uint64_t site = 0; site < 512; ++site) {
            seen.insert(PhiloxStream(1234, frame, site).NextBits64());
        }
    }
    EXPECT_EQ(seen.size(), 16u * 512u);   // zero collisions
}

TEST(Philox, ChangingSeedChangesStream) {
    const double a = PhiloxStream(1, 0, 0).NextUniform();
    const double b = PhiloxStream(2, 0, 0).NextUniform();
    EXPECT_NE(a, b);
}

// A long draw from one site must not wander into a neighbouring site's stream.
TEST(Philox, LongDrawFromOneSiteStaysDistinct) {
    PhiloxStream s0(7, 0, 0);
    std::set<uint32_t> values;
    for (int i = 0; i < 4000; ++i) values.insert(s0.NextBits32());
    // 4000 draws from a 32-bit space: birthday collisions are possible but
    // should be very few. Anything close to 4000 unique values is healthy;
    // a refill bug would show up as heavy repetition.
    EXPECT_GT(values.size(), 3990u);
}

// --- Distributional sanity ------------------------------------------------
// Not a substitute for a test suite like TestU01; enough to catch a broken
// bit-extraction or scaling step.

TEST(Philox, UniformIsInRangeAndRoughlyFlat) {
    const int kN = 200000;
    const int kBins = 10;
    std::vector<int> hist(kBins, 0);
    double sum = 0.0;
    for (int i = 0; i < kN; ++i) {
        const double u = PhiloxStream(2024, 0, i).NextUniform();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
        ++hist[static_cast<int>(u * kBins)];
    }
    EXPECT_NEAR(sum / kN, 0.5, 0.01);
    const int expected = kN / kBins;
    for (int b = 0; b < kBins; ++b) {
        EXPECT_NEAR(hist[b], expected, expected * 0.05) << "bin " << b;
    }
}

TEST(Philox, UniformRangeOverload) {
    for (int i = 0; i < 1000; ++i) {
        const double u = PhiloxStream(5, 1, i).NextUniform(-3.0, 7.0);
        ASSERT_GE(u, -3.0);
        ASSERT_LT(u, 7.0);
    }
}

TEST(Philox, BitsToUniformConversionsAreExact) {
    EXPECT_EQ(UniformFromBits32(0u), 0.0);
    EXPECT_EQ(UniformFromBits32(0x80000000u), 0.5);
    EXPECT_EQ(UniformFromBits64(0u), 0.0);
    // 2^63 >> 11 = 2^52, scaled by 2^-53 = 0.5, exactly.
    EXPECT_EQ(UniformFromBits64(uint64_t(1) << 63), 0.5);
}

// Committed digest, mirroring infra/scripts/check_determinism.sh. Philox is
// integer-only, so this must hold on every platform and every optimization
// level -- including under -ffast-math, which cannot touch integer arithmetic.
// Recorded 2026-09-06; verified identical under gcc 13.3.0 and clang 18.1.3
// at -O0/-O2/-O3.
TEST(Philox, DigestOverFixedGridIsStable) {
    uint64_t digest = 1469598103934665603ull;
    for (uint64_t site = 0; site < 20000; ++site) {
        const uint64_t bits = PhiloxStream(20260906ull, 17ull, site).NextBits64();
        for (int byte = 0; byte < 8; ++byte) {
            digest ^= (bits >> (byte * 8)) & 0xFFull;
            digest *= 1099511628211ull;
        }
    }
    EXPECT_EQ(digest, 215287054852263674ull);
}
