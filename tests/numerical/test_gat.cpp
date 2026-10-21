// The Generalized Anscombe Transformation and its exact unbiased inverse (2.2, gat.hpp), checked against
// Mäkitalo & Foi's formulas and published accuracy figures (thesis eqs. 4.3-4.10, 4.35-4.40), and against
// Monte Carlo draws from an exact Poisson sampler written here, independent of the table code.
#include "perception/radiometric/gat.hpp"

#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ariadne::perception::radiometric;
using ariadne::core::PhiloxStream;

namespace {

/// Exact Poisson draw by CDF inversion (fine for the y <= 100 used here). Deliberately not
/// core::SamplePoisson, which switches to a Gaussian above lambda = 30.
long ExactPoisson(PhiloxStream& s, double y) {
    const double u = s.NextUniform();
    long k = 0;
    double p = std::exp(-y), cdf = p;
    while (u > cdf && k < 100000) {
        ++k;
        p *= y / k;
        cdf += p;
    }
    return k;
}

struct Moments {
    double mean = 0, var = 0;
    int n = 0;
};

/// Sample mean and variance of f_s(p + n), p ~ P(y), n ~ N(0, s^2), over n draws.
Moments MonteCarloGat(double y, double s, int n, uint64_t seed) {
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; ++i) {
        PhiloxStream st(seed, static_cast<uint64_t>(i), 0);
        const double z = static_cast<double>(ExactPoisson(st, y)) + s * ariadne::core::SampleStandardNormal(st);
        const double f = GatReduced(z, s);
        sum += f;
        sum2 += f * f;
    }
    Moments m;
    m.n = n;
    m.mean = sum / n;
    m.var = (sum2 - n * m.mean * m.mean) / (n - 1);
    return m;
}

}  // namespace

// eq. 4.3 (the GAT in image units) equals eq. 4.7 applied after the reduction 4.4, for alpha != 1 and mu != 0.
// The two are written independently in gat.hpp, so this checks the algebra of the parameter reduction.
TEST(Gat, ForwardEqualsReducedFormForAnyGainAndOffset) {
    for (const PoissonGaussian m : {PoissonGaussian{0.0255, 0.064, 0.0}, PoissonGaussian{0.204, 0.59, 3.0},
                                    PoissonGaussian{2.5, 4.0, -7.0}, PoissonGaussian{1.0, 0.0, 0.0}}) {
        for (double z = -20.0; z <= 300.0; z += 0.37) {
            const double direct = GatForward(z, m);
            const double reduced = GatReduced(ReducedValue(z, m), ReducedSigma(m));
            ASSERT_NEAR(direct, reduced, 1e-12 * std::max(1.0, direct)) << "alpha " << m.alpha << " z " << z;
        }
    }
}

// alpha = 1, sigma = 0, mu = 0 is Anscombe's 2 sqrt(z + 3/8) (eq. 2.15), zero below -3/8.
TEST(Gat, PurePoissonIsAnscombe) {
    for (double z = 0.0; z <= 50.0; z += 0.5) EXPECT_DOUBLE_EQ(GatReduced(z, 0.0), 2.0 * std::sqrt(z + 0.375));
    EXPECT_EQ(GatReduced(-0.5, 0.0), 0.0);
    EXPECT_EQ(GatForward(-100.0, {0.5, 1.0, 0.0}), 0.0);
}

// The point of the transform: std{f(z)} ~ 1 regardless of y once y is moderate (thesis fig. 4.1(b)).
TEST(Gat, StabilisesTheVarianceToUnity) {
    for (double s : {0.3, 1.0, 2.0, 5.0}) {
        for (double y : {5.0, 20.0, 80.0}) {
            const Moments m = MonteCarloGat(y, s, 100000, 11);
            EXPECT_GT(m.var, 0.9) << "s " << s << " y " << y;
            EXPECT_LT(m.var, 1.1) << "s " << s << " y " << y;
        }
    }
}

// eq. 4.10 by quadrature against the Monte Carlo mean: within 4 standard errors.
TEST(Gat, ExpectedValueMatchesMonteCarlo) {
    for (double s : {0.05, 0.4, 1.0, 3.0}) {
        for (double y : {0.0, 0.3, 1.0, 4.0, 15.0, 60.0}) {
            const Moments m = MonteCarloGat(y, s, 200000, 23);
            const double se = std::sqrt(m.var / m.n);
            EXPECT_NEAR(ExpectedGat(y, s), m.mean, 4.0 * se + 1e-9) << "s " << s << " y " << y;
        }
    }
}

// s -> 0 is continuous into the pure-Poisson sum of eq. 3.4, and E{f|0,0} = 2 sqrt(3/8).
TEST(Gat, ExpectedValueIsContinuousIntoPurePoisson) {
    EXPECT_NEAR(ExpectedGat(0.0, 0.0), 2.0 * std::sqrt(0.375), 1e-12);
    for (double y : {0.0, 0.5, 3.0, 50.0}) EXPECT_NEAR(ExpectedGat(y, 1e-3), ExpectedGat(y, 0.0), 2e-6) << y;
}

// The exact inverse is unbiased: fed the Monte Carlo mean of f, it returns y within the Monte Carlo error
// propagated through dy/dD = D/2. The NEGATIVE CONTROL: the algebraic inverse, fed the same means, misses by
// far more than that tolerance at low counts -- so the test can tell an unbiased inverse from a biased one.
TEST(Gat, ExactInverseIsUnbiasedAndTheAlgebraicInverseIsNot) {
    for (double s : {0.2, 1.0, 3.0}) {
        const ExactInverse inv(s);
        for (double y : {0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0}) {
            const Moments m = MonteCarloGat(y, s, 400000, 37);
            const double tol = 4.0 * std::sqrt(m.var / m.n) * 0.5 * m.mean + 2e-3;   // + table interpolation
            EXPECT_NEAR(inv(m.mean), y, tol) << "exact, s " << s << " y " << y;
            if (y <= 5.0 && s <= 1.0) {
                EXPECT_GT(std::abs(AlgebraicInverse(m.mean, s) - y), 5.0 * tol) << "algebraic, s " << s << " y " << y;
            }
        }
    }
}

// Round trip in image units with alpha != 1 and mu != 0: DN -> reduced -> E{f} -> exact inverse -> DN.
TEST(Gat, RoundTripInImageUnits) {
    const PoissonGaussian m{0.204, 0.59, 2.0};
    const ExactInverse inv(ReducedSigma(m));
    for (double mean_dn : {2.5, 4.0, 10.0, 40.0}) {
        const double y = ReducedValue(mean_dn, m);
        EXPECT_NEAR(FromReduced(inv(ExpectedGat(y, ReducedSigma(m))), m), mean_dn, 1e-3 * m.alpha + 1e-9);
    }
    EXPECT_EQ(inv(inv.DMin() - 0.1), 0.0);   // eq. 4.14: below E{f|0,s} the ML inverse is 0
}

// The paper's accuracy figures, reproduced: eq. 4.40 (closed form, L_inf 0.0468 at s = 0.4, y = 0) and
// eq. 4.36 (I_0 - s^2, L_inf 0.0470, same place). The maximum is at y = 0, so the sweep is over s there.
TEST(Gat, ClosedFormAndAsymptoticErrorsMatchThePaperLinf) {
    const ExactInverse i0(0.0, 400.0, 2400);
    double cf = 0, cf_s = 0, asy = 0, asy_s = 0;
    for (double s = 0.20; s <= 0.70 + 1e-9; s += 0.01) {
        const double D = ExpectedGat(0.0, s);
        const double e1 = std::abs(ClosedFormInverse(D, s)), e2 = std::abs(i0(D) - s * s);
        if (e1 > cf) cf = e1, cf_s = s;
        if (e2 > asy) asy = e2, asy_s = s;
    }
    EXPECT_NEAR(cf, 0.0468, 1e-3);
    EXPECT_NEAR(asy, 0.0470, 1e-3);
    EXPECT_GT(cf_s, 0.3);
    EXPECT_LT(cf_s, 0.5);
    EXPECT_GT(asy_s, 0.3);
    EXPECT_LT(asy_s, 0.5);
    // And nowhere on a global grid (s in (0, 50], y in [0, 200]) is the closed form worse than that.
    for (double s : {0.05, 1.0, 2.0, 5.0, 20.0, 50.0}) {
        for (double y : {0.0, 0.5, 2.0, 10.0, 50.0, 200.0}) {
            EXPECT_LT(std::abs(ClosedFormInverse(ExpectedGat(y, s), s) - y), 0.0475) << s << " " << y;
        }
    }
}

// eq. 4.39: the variance-normalised integral criterion of the closed form, 0.0069, by trapezoid on a
// quadratic grid over s in [0, 50], y in [0, 200] (the paper's table range). 4% tolerance for the coarse grid
// (the 120 x 160 grid gives 0.00690; this one is coarser to keep the test fast).
TEST(Gat, ClosedFormWeightedL2MatchesThePaper) {
    const int ns = 60, ny = 80;
    std::vector<double> sg(ns + 1), yg(ny + 1);
    for (int i = 0; i <= ns; ++i) sg[i] = 50.0 * (double(i) / ns) * (double(i) / ns);
    for (int j = 0; j <= ny; ++j) yg[j] = 200.0 * (double(j) / ny) * (double(j) / ny);
    std::vector<double> e((ns + 1) * (ny + 1));
    for (int i = 0; i <= ns; ++i)
        for (int j = 0; j <= ny; ++j) e[i * (ny + 1) + j] = ClosedFormInverse(ExpectedGat(yg[j], sg[i]), sg[i]) - yg[j];
    double l2 = 0;
    for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < ny; ++j) {
            double a = 0;
            int n = 0;
            for (int di = 0; di < 2; ++di) {
                for (int dj = 0; dj < 2; ++dj) {
                    const double w = yg[j + dj] + sg[i + di] * sg[i + di];
                    if (w <= 0) continue;   // the (0, 0) corner: the error is 0 there and the weight singular
                    const double v = e[(i + di) * (ny + 1) + j + dj];
                    a += v * v / w;
                    ++n;
                }
            }
            l2 += a / n * (sg[i + 1] - sg[i]) * (yg[j + 1] - yg[j]);
        }
    }
    EXPECT_NEAR(l2, 0.0069, 0.04 * 0.0069) << "weighted L2^2 = " << l2;
}
