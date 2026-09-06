// Deterministic samplers.
//
// Two classes of property are checked, and they are not interchangeable:
//   1. STATISTICAL correctness -- the samplers produce the distributions they
//      claim, checked against closed-form Poisson probabilities and against
//      moments, not merely "looks random".
//   2. DETERMINISM -- identical bits regardless of access order, thread count,
//      or how many rejections a rejection sampler happened to take.
#include "core/random/distributions.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <thread>
#include <vector>

using namespace ariadne::core;

namespace {

// Closed-form Poisson pmf, used as the oracle for the exact branch.
double PoissonPmf(int k, double lambda) {
    double log_p = -lambda + k * std::log(lambda);
    for (int i = 2; i <= k; ++i) log_p -= std::log(static_cast<double>(i));
    return std::exp(log_p);
}

struct Moments {
    double mean;
    double variance;
};

Moments MomentsOf(const std::vector<double>& xs) {
    double sum = 0.0;
    for (double x : xs) sum += x;
    const double mean = sum / static_cast<double>(xs.size());
    double acc = 0.0;
    for (double x : xs) acc += (x - mean) * (x - mean);
    return {mean, acc / static_cast<double>(xs.size() - 1)};
}

}  // namespace

// --- Determinism ----------------------------------------------------------

// The property that makes a rejection sampler safe here: the accepted value at
// a site is a function of (seed, frame, site) alone, no matter how many
// candidates were rejected on the way. If rejection consumed shared state this
// would fail.
TEST(Distributions, NormalIsIndependentOfAccessOrder) {
    std::vector<double> forward, backward;
    for (uint64_t site = 0; site < 512; ++site) {
        PhiloxStream s(11, 5, site);
        forward.push_back(SampleStandardNormal(s));
    }
    for (int site = 511; site >= 0; --site) {
        PhiloxStream s(11, 5, site);
        backward.push_back(SampleStandardNormal(s));
    }
    for (size_t i = 0; i < forward.size(); ++i) {
        EXPECT_EQ(forward[i], backward[backward.size() - 1 - i]) << "site " << i;
    }
}

TEST(Distributions, PoissonIsIndependentOfThreadCount) {
    const int kSites = 4096;
    const double lambda = 12.0;   // exact branch

    std::vector<int64_t> serial(kSites);
    for (int i = 0; i < kSites; ++i) {
        PhiloxStream s(77, 2, i);
        serial[i] = SamplePoisson(s, lambda);
    }

    std::vector<int64_t> parallel(kSites);
    std::vector<std::thread> pool;
    for (int t = 0; t < 8; ++t) {
        pool.emplace_back([&, t] {
            for (int i = t; i < kSites; i += 8) {
                PhiloxStream s(77, 2, i);
                parallel[i] = SamplePoisson(s, lambda);
            }
        });
    }
    for (auto& th : pool) th.join();
    EXPECT_EQ(serial, parallel);
}

// --- Gaussian correctness -------------------------------------------------

TEST(Distributions, StandardNormalMomentsAreCorrect) {
    std::vector<double> xs;
    xs.reserve(200000);
    for (uint64_t i = 0; i < 200000; ++i) {
        PhiloxStream s(3, 0, i);
        xs.push_back(SampleStandardNormal(s));
    }
    const Moments m = MomentsOf(xs);
    EXPECT_NEAR(m.mean, 0.0, 0.01);
    EXPECT_NEAR(m.variance, 1.0, 0.02);

    // Tail fractions, which a broken scale factor would move even when the
    // variance happened to look right.
    int within1 = 0, within2 = 0, within3 = 0;
    for (double x : xs) {
        const double a = std::abs(x);
        if (a < 1.0) ++within1;
        if (a < 2.0) ++within2;
        if (a < 3.0) ++within3;
    }
    const double n = static_cast<double>(xs.size());
    EXPECT_NEAR(within1 / n, 0.6827, 0.01);
    EXPECT_NEAR(within2 / n, 0.9545, 0.005);
    EXPECT_NEAR(within3 / n, 0.9973, 0.002);
}

TEST(Distributions, NormalAppliesMeanAndStddev) {
    std::vector<double> xs;
    for (uint64_t i = 0; i < 100000; ++i) {
        PhiloxStream s(4, 1, i);
        xs.push_back(SampleNormal(s, 5.0, 2.0));
    }
    const Moments m = MomentsOf(xs);
    EXPECT_NEAR(m.mean, 5.0, 0.05);
    EXPECT_NEAR(m.variance, 4.0, 0.1);
}

// --- Poisson correctness --------------------------------------------------

// The exact branch checked against the closed-form pmf, bin by bin. This is a
// real distributional test, not a moment check: a sampler can match mean and
// variance while having the wrong shape.
TEST(Distributions, ExactPoissonMatchesClosedFormPmf) {
    for (double lambda : {0.5, 2.0, 8.0, 20.0}) {
        ASSERT_LT(lambda, kPoissonExactMaxLambda);
        const int kN = 400000;
        std::map<int64_t, int> hist;
        for (int i = 0; i < kN; ++i) {
            PhiloxStream s(static_cast<uint64_t>(lambda * 100), 9, i);
            ++hist[SamplePoisson(s, lambda)];
        }
        for (int k = 0; k <= static_cast<int>(lambda * 3) + 5; ++k) {
            const double expected = PoissonPmf(k, lambda);
            if (expected < 1e-4) continue;      // too rare to test reliably
            const double observed =
                static_cast<double>(hist[k]) / static_cast<double>(kN);
            // 4 sigma of the binomial standard error, plus a small floor.
            const double sigma = std::sqrt(expected * (1.0 - expected) / kN);
            EXPECT_NEAR(observed, expected, 4.0 * sigma + 1e-4)
                << "lambda=" << lambda << " k=" << k;
        }
    }
}

// For Poisson, variance equals the mean. That identity is the sharpest cheap
// check on the approximate branch, because it fails if the sqrt(lambda) scaling
// is wrong even though the mean stays right.
TEST(Distributions, ApproximatePoissonHasVarianceEqualToMean) {
    for (double lambda : {50.0, 500.0, 5000.0}) {
        ASSERT_GE(lambda, kPoissonExactMaxLambda);
        std::vector<double> xs;
        xs.reserve(100000);
        for (uint64_t i = 0; i < 100000; ++i) {
            PhiloxStream s(6, 3, i);
            xs.push_back(static_cast<double>(SamplePoisson(s, lambda)));
        }
        const Moments m = MomentsOf(xs);
        EXPECT_NEAR(m.mean, lambda, lambda * 0.01) << "lambda=" << lambda;
        EXPECT_NEAR(m.variance, lambda, lambda * 0.05) << "lambda=" << lambda;
    }
}

// The two branches must agree near the switchover, otherwise the sensor model
// would show a discontinuity at a brightness that has no physical meaning.
TEST(Distributions, BranchesAgreeAcrossTheThreshold) {
    const double lambda = kPoissonExactMaxLambda;   // 30
    auto sample_with = [&](double threshold) {
        std::vector<double> xs;
        xs.reserve(150000);
        for (uint64_t i = 0; i < 150000; ++i) {
            PhiloxStream s(8, 4, i);
            xs.push_back(static_cast<double>(SamplePoisson(s, lambda, threshold)));
        }
        return MomentsOf(xs);
    };
    const Moments exact = sample_with(1e9);    // force the exact branch
    const Moments approx = sample_with(0.0);   // force the Gaussian branch
    EXPECT_NEAR(exact.mean, approx.mean, 0.3);
    EXPECT_NEAR(exact.variance, approx.variance, 1.5);
}

TEST(Distributions, PoissonEdgeCases) {
    PhiloxStream s(1, 1, 1);
    EXPECT_EQ(SamplePoisson(s, 0.0), 0);
    EXPECT_EQ(SamplePoisson(s, -1.0), 0);
    EXPECT_EQ(SamplePoisson(s, std::nan("")), 0);
    // Never negative, however large the mean.
    for (uint64_t i = 0; i < 5000; ++i) {
        PhiloxStream t(2, 2, i);
        EXPECT_GE(SamplePoisson(t, 100000.0), 0);
    }
}

// --- Noise field ----------------------------------------------------------

TEST(Distributions, ValueNoiseIsInRangeAndDeterministic) {
    for (int i = 0; i < 500; ++i) {
        const double x = i * 0.37;
        const double y = i * 0.21;
        const double a = ValueNoise2D(12, 3, x, y);
        const double b = ValueNoise2D(12, 3, x, y);
        ASSERT_GE(a, 0.0);
        ASSERT_LE(a, 1.0);
        EXPECT_EQ(a, b);                       // bit-identical on repeat
    }
}

// Continuity: the field must not jump at lattice boundaries, which is what the
// smoothstep easing buys. A jump would appear as a visible grid in the dust.
TEST(Distributions, ValueNoiseIsContinuousAcrossLatticeCells) {
    const double eps = 1e-6;
    for (int cell = -3; cell <= 3; ++cell) {
        const double x = static_cast<double>(cell);
        const double left  = ValueNoise2D(21, 0, x - eps, 0.5);
        const double right = ValueNoise2D(21, 0, x + eps, 0.5);
        EXPECT_LT(std::abs(left - right), 1e-4) << "cell boundary x=" << x;
    }
}

TEST(Distributions, ValueNoiseVariesAcrossSpaceAndSeed) {
    EXPECT_NE(ValueNoise2D(1, 0, 0.5, 0.5), ValueNoise2D(1, 0, 7.5, 3.5));
    EXPECT_NE(ValueNoise2D(1, 0, 0.5, 0.5), ValueNoise2D(2, 0, 0.5, 0.5));
    EXPECT_NE(ValueNoise2D(1, 0, 0.5, 0.5), ValueNoise2D(1, 1, 0.5, 0.5));
}

TEST(Distributions, FractalNoiseStaysInRange) {
    for (int i = 0; i < 2000; ++i) {
        const double v = FractalNoise2D(5, 2, i * 0.13, i * 0.07, 4);
        ASSERT_GE(v, 0.0);
        ASSERT_LE(v, 1.0);
    }
}

TEST(Distributions, RoundToInt64TiesAwayFromZero) {
    EXPECT_EQ(DetRoundToInt64(0.5), 1);
    EXPECT_EQ(DetRoundToInt64(-0.5), -1);
    EXPECT_EQ(DetRoundToInt64(2.4), 2);
    EXPECT_EQ(DetRoundToInt64(-2.6), -3);
    EXPECT_EQ(DetRoundToInt64(0.0), 0);
}
