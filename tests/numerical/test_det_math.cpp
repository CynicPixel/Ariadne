// Deterministic exp/log.
//
// Two distinct things are being tested and they must not be confused:
//   1. ACCURACY -- checked against libm, which is a good approximation oracle.
//   2. DETERMINISM -- the property libm cannot provide. Checked structurally
//      (exact round-trips, exact special cases) and, in CI, by comparing a
//      committed digest across toolchains and architectures.
// libm is the oracle here; it is never the runtime path.
#include "core/math/det_math.hpp"
#include "core/random/philox.hpp"

#include "core/determinism/digests.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ariadne::core;

namespace {

double RelError(double got, double want) {
    if (want == 0.0) return std::abs(got);
    return std::abs(got - want) / std::abs(want);
}

// A few ulp of double precision. DetExp/DetLog are not claimed to be
// correctly rounded -- only accurate and reproducible.
constexpr double kTol = 1e-14;

}  // namespace

TEST(DetMath, Exp2IntIsExact) {
    EXPECT_EQ(Exp2Int(0), 1.0);
    EXPECT_EQ(Exp2Int(1), 2.0);
    EXPECT_EQ(Exp2Int(10), 1024.0);
    EXPECT_EQ(Exp2Int(-1), 0.5);
    EXPECT_EQ(Exp2Int(-10), 1.0 / 1024.0);
    // Exactness across the whole normal range: compare against std::ldexp,
    // which is also exact (it only manipulates the exponent).
    for (int k = -1022; k <= 1023; ++k) {
        EXPECT_EQ(Exp2Int(k), std::ldexp(1.0, k)) << "k=" << k;
    }
}

TEST(DetMath, Exp2IntHandlesSubnormalAndOverflow) {
    EXPECT_EQ(Exp2Int(-1074), std::ldexp(1.0, -1074));   // smallest subnormal
    EXPECT_EQ(Exp2Int(-1075), 0.0);
    EXPECT_TRUE(std::isinf(Exp2Int(1024)));
}

TEST(DetMath, ExpSpecialValues) {
    EXPECT_EQ(DetExp(0.0), 1.0);          // exact, not approximate
    EXPECT_EQ(DetExp(-1000.0), 0.0);      // underflow to exactly zero
    EXPECT_TRUE(std::isinf(DetExp(1000.0)));
    EXPECT_TRUE(std::isnan(DetExp(std::nan(""))));
}

TEST(DetMath, ExpMatchesLibmAcrossRange) {
    double worst = 0.0;
    double worst_at = 0.0;
    for (double x = -700.0; x <= 700.0; x += 0.37) {
        const double err = RelError(DetExp(x), std::exp(x));
        if (err > worst) { worst = err; worst_at = x; }
    }
    EXPECT_LT(worst, kTol) << "worst relative error " << worst << " at x=" << worst_at;
}

// REGRESSION. The sweep above stops at +-700, and the bug lived entirely
// OUTSIDE it: the range reduction produces k=1024 for x >= ~709.44 and k=-1075
// for x <= ~-745.13, and Exp2Int saturates to inf / 0 at those k. DetExp
// therefore returned inf (resp. 0) for a whole band of representable results,
// inside its own declared domain, with every test green. Scaling is now split
// so neither factor leaves the normal range.
TEST(DetMath, ExpIsFiniteAndAccurateAtTheOverflowBoundary) {
    for (double x = 708.0; x <= 709.782712893384; x += 0.0007) {
        const double got = DetExp(x);
        ASSERT_TRUE(std::isfinite(got)) << "DetExp overflowed early at x=" << x;
        EXPECT_LT(RelError(got, std::exp(x)), kTol) << "at x=" << x;
    }
    // Just past the guard, infinity is the correct answer.
    EXPECT_TRUE(std::isinf(DetExp(709.9)));
}

TEST(DetMath, ExpReachesTheSmallestSubnormalBeforeUnderflowing) {
    for (double x = -745.1332191019411; x <= -744.0; x += 0.0007) {
        const double got = DetExp(x);
        ASSERT_GT(got, 0.0) << "DetExp underflowed early at x=" << x;
        EXPECT_LT(RelError(got, std::exp(x)), kTol) << "at x=" << x;
    }
    // The smallest subnormal must be reachable, not rounded away to zero.
    EXPECT_EQ(DetExp(-745.13), std::exp(-745.13));
    EXPECT_GT(DetExp(-745.13), 0.0);
    // Below the guard, zero is the correct answer.
    EXPECT_EQ(DetExp(-746.0), 0.0);
}

// The values D actually uses: exp(-lambda) for Poisson, exp(-beta*depth) for
// dust transmission. Both live in a narrow negative range, so it is worth
// pinning accuracy there specifically.
TEST(DetMath, ExpAccurateOnTheRangeDegradationUses) {
    double worst = 0.0;
    for (int i = 0; i <= 20000; ++i) {
        const double x = -30.0 * (static_cast<double>(i) / 20000.0);
        worst = std::max(worst, RelError(DetExp(x), std::exp(x)));
    }
    EXPECT_LT(worst, 1e-15) << "worst relative error " << worst;
}

TEST(DetMath, ExpNearZeroIsAccurate) {
    for (double x : {1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, -1e-9, -1e-15}) {
        EXPECT_LT(RelError(DetExp(x), std::exp(x)), 1e-15) << "x=" << x;
    }
}

TEST(DetMath, LogSpecialValues) {
    EXPECT_EQ(DetLog(1.0), 0.0);          // exact
    EXPECT_TRUE(std::isinf(DetLog(0.0)) && DetLog(0.0) < 0.0);
    EXPECT_TRUE(std::isnan(DetLog(-1.0)));
    EXPECT_TRUE(std::isnan(DetLog(std::nan(""))));
    EXPECT_TRUE(std::isinf(DetLog(std::numeric_limits<double>::infinity())));
}

TEST(DetMath, LogMatchesLibmAcrossRange) {
    double worst = 0.0;
    double worst_at = 0.0;
    // Sweep exponents as well as mantissas, so the exponent-recombination path
    // is exercised, not just m in [1,2).
    for (int e = -300; e <= 300; e += 7) {
        for (double m = 1.0; m < 2.0; m += 0.013) {
            const double x = std::ldexp(m, e);
            const double err = RelError(DetLog(x), std::log(x));
            if (err > worst) { worst = err; worst_at = x; }
        }
    }
    EXPECT_LT(worst, kTol) << "worst relative error " << worst << " at x=" << worst_at;
}

TEST(DetMath, LogHandlesSubnormals) {
    const double tiny = std::ldexp(1.0, -1060);   // subnormal
    ASSERT_GT(tiny, 0.0);
    EXPECT_LT(RelError(DetLog(tiny), std::log(tiny)), kTol);
}

TEST(DetMath, LogNearOneIsAccurate) {
    // The regime where a naive series in (m-1) would lose precision.
    //
    // The oracle is std::log of the SAME double, not std::log1p(d). Comparing
    // against log1p would compare different inputs: `1.0 + d` rounds to the
    // nearest double before DetLog sees it, and for d = 1e-12 that rounding is
    // ~1e-4 of the increment. log1p exists precisely to avoid that rounding, so
    // using it as the oracle here measures the test's own arithmetic rather
    // than DetLog's accuracy.
    for (double d : {1e-1, 1e-3, 1e-6, 1e-9, 1e-12}) {
        const double y = 1.0 + d;      // the value DetLog actually receives
        EXPECT_LT(RelError(DetLog(y), std::log(y)), 1e-13) << "d=" << d;
    }
}

// Round-tripping is an internal-consistency check that does not depend on libm
// at all -- it would still pass on a platform whose libm was badly wrong.
TEST(DetMath, ExpLogRoundTrip) {
    for (int i = 1; i <= 5000; ++i) {
        const double x = static_cast<double>(i) * 0.01;
        EXPECT_LT(RelError(DetExp(DetLog(x)), x), 1e-14) << "x=" << x;
    }
}

TEST(DetMath, SqrtIsExactPowersOfTwo) {
    EXPECT_EQ(DetSqrt(4.0), 2.0);
    EXPECT_EQ(DetSqrt(0.25), 0.5);
    EXPECT_EQ(DetSqrt(0.0), 0.0);
}

// --- The determinism assertion -------------------------------------------

// A digest over a fixed grid of inputs. This value is committed: CI recomputes
// it on arm64 and on a second toolchain and compares. A mismatch means the
// bit-identity guarantee has been lost, which invalidates every named
// degradation config that has already produced published results.
//
// The digest is built from raw BITS, so it cannot be fooled by a comparison
// that tolerates a one-ulp drift.
TEST(DetMath, DigestOverFixedGridIsStable) {
    // Computed by the shared implementation (core/determinism/digests.hpp), the
    // same code check_determinism.sh and the run manifest call.
    // Recorded 2026-09-08 (the value is an OUTPUT of the code, not a spec).
    // Fails after a deliberate change to det_math.hpp: re-record it and every
    // dependent golden digest. Fails WITHOUT such a change, on another compiler
    // or architecture: the cross-platform guarantee is broken; investigate.
    EXPECT_EQ(DetMathDigest(), 11806531094783593156ull)
        << "det_math digest changed -- see comment above";
}
