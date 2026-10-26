#pragma once
// ---------------------------------------------------------------------------
// The Generalized Anscombe Transformation (GAT) and its inverses (arch §6.2
// stage 2, §6.4). OpenCV-free: plain arithmetic, testable on the host.
//
// Noise model, per pixel, in output DN (Mäkitalo & Foi, TIP 2013 = thesis ch. 4,
// eq. 4.1):  z = alpha * p + n,  p ~ Poisson(y),  n ~ N(mu, sigma^2).
// Its variance is affine in the mean: var = alpha * E[z] + (sigma^2 - alpha*mu),
// which is what the noise fit (noise_estimate.hpp) measures as a*m + b.
//
// The three-step denoiser is: f = GAT(z) (noise std ~ 1 everywhere), any
// Gaussian denoiser at sigma = 1, then an inverse back to the mean. Which
// inverse matters: f is concave, so E{f(z)} < f(E{z}) and the ALGEBRAIC inverse
// f^-1 returns a biased (too dark) mean at low counts. The EXACT UNBIASED
// inverse (eq. 4.9) maps E{f(z) | y} back to y, using a table of
// E{f(z) | y, sigma} computed from the noise model (eq. 4.10).
//
// Source: M. Mäkitalo, A. Foi, "Optimal inversion of the generalized Anscombe
// transformation for Poisson-Gaussian noise", IEEE TIP 22(1), 2013; equations
// as numbered in M. Mäkitalo's doctoral thesis (Tampere UT, 2013), ch. 3-4.
// Implemented from the paper. The authors' Matlab package (invansc) is under a
// non-commercial licence and none of it is used or shipped (algorithms.md §14.6).
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ariadne::perception::radiometric {

/// z = alpha * Poisson(y) + N(mu, sigma^2), in the units of the image (DN).
struct PoissonGaussian {
    double alpha = 1.0;   // gain: DN per Poisson count
    double sigma = 0.0;   // std of the Gaussian component, DN
    double mu = 0.0;      // mean of the Gaussian component (black level), DN
};

/// From a fit var = a * m + b: with no separate black level (mu = 0, as in D),
/// alpha = a and sigma^2 = b. A slightly negative b (fit noise) is clamped to 0.
inline PoissonGaussian FromAffineFit(double a, double b) {
    return {a, std::sqrt(std::max(b, 0.0)), 0.0};
}

// --- Forward ---------------------------------------------------------------

/// eq. 4.3: the GAT in the image's own units,
///   f(z) = (2/alpha) * sqrt(alpha z + 3/8 alpha^2 + sigma^2 - alpha mu), 0 below the root.
inline double GatForward(double z, const PoissonGaussian& m) {
    const double arg = m.alpha * z + 0.375 * m.alpha * m.alpha + m.sigma * m.sigma - m.alpha * m.mu;
    return arg > 0.0 ? (2.0 / m.alpha) * std::sqrt(arg) : 0.0;
}

/// eq. 4.4: the affine reduction to a unit-gain, zero-mean problem with one
/// parameter, z_r = (z - mu) / alpha and s = sigma / alpha.
inline double ReducedValue(double z, const PoissonGaussian& m) { return (z - m.mu) / m.alpha; }
inline double ReducedSigma(const PoissonGaussian& m) { return m.sigma / m.alpha; }
/// ...and back: the estimate of E{z} is alpha * y + mu (affine, so unbiased).
inline double FromReduced(double y, const PoissonGaussian& m) { return m.alpha * y + m.mu; }

/// eq. 4.7: the GAT of the reduced variable, f_s(z) = 2 sqrt(z + 3/8 + s^2), 0 below the root.
/// s = 0 is Anscombe's transformation 2 sqrt(z + 3/8) (eq. 2.15).
inline double GatReduced(double z, double s) {
    const double arg = z + 0.375 + s * s;
    return arg > 0.0 ? 2.0 * std::sqrt(arg) : 0.0;
}

// --- Closed-form inverses (reduced units: D -> y) ----------------------------

/// The algebraic inverse of eq. 4.7, I_alg(D) = D^2/4 - 3/8 - s^2. Biased at low counts;
/// kept as the ablation row that shows why the exact inverse exists.
inline double AlgebraicInverse(double D, double s) { return 0.25 * D * D - 0.375 - s * s; }

/// eq. 4.38: the asymptotically unbiased inverse, I_asy(D) = D^2/4 - 1/8 - s^2.
inline double AsymptoticInverse(double D, double s) { return 0.25 * D * D - 0.125 - s * s; }

/// eq. 3.7: the closed-form approximation of the exact unbiased Anscombe inverse I_0,
///   I~_0(D) = D^2/4 + (1/4) sqrt(3/2) D^-1 - (11/8) D^-2 + (5/8) sqrt(3/2) D^-3 - 1/8,
/// set to 0 for D < 2 sqrt(3/8) (thesis §3.2.2, below eq. 3.5; the invansc .m omits this branch).
inline double ClosedFormAnscombeInverse(double D) {
    const double d_min = 2.0 * std::sqrt(0.375);
    if (D < d_min) return 0.0;
    const double r = std::sqrt(1.5);
    const double inv = 1.0 / D;
    return 0.25 * D * D + 0.25 * r * inv - 1.375 * inv * inv + 0.625 * r * inv * inv * inv - 0.125;
}

/// eq. 4.37: the closed-form approximation of the exact unbiased GAT inverse,
/// I~_s(D) = I~_0(D) - s^2. Accuracy against the exact inverse: L_inf 0.0468 at
/// (s = 0.4, y = 0) and weighted L2^2 0.0069 (eqs. 4.39-4.40).
inline double ClosedFormInverse(double D, double s) { return ClosedFormAnscombeInverse(D) - s * s; }

// --- The exact unbiased inverse ---------------------------------------------

namespace detail {

/// 20-point Gauss-Legendre nodes/weights on [-1, 1] (positive half; symmetric).
inline constexpr double kGl20X[10] = {0.0765265211334973, 0.2277858511416451, 0.3737060887154195,
                                      0.5108670019508271, 0.6360536807265150, 0.7463319064601508,
                                      0.8391169718222188, 0.9122344282513259, 0.9639719272779138,
                                      0.9931285991850949};
inline constexpr double kGl20W[10] = {0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
                                      0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
                                      0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
                                      0.0176140071391521};

/// int_lo^hi g(u) du by composite 20-point Gauss-Legendre over `panels` panels.
template <class G>
double Integrate(const G& g, double lo, double hi, int panels) {
    const double h = (hi - lo) / panels;
    double sum = 0.0;
    for (int p = 0; p < panels; ++p) {
        const double c = lo + (p + 0.5) * h, r = 0.5 * h;
        for (int i = 0; i < 10; ++i) sum += kGl20W[i] * r * (g(c - r * kGl20X[i]) + g(c + r * kGl20X[i]));
    }
    return sum;
}

/// E{ f_s(k + n) }, n ~ N(0, s^2): the Gaussian half of eq. 4.10 for one Poisson outcome k.
/// f_s has a square-root kink at t0 = -3/8 - s^2, so the integral is taken over u with
/// z = t0 + u^2 (dz = 2u du), where the integrand 2u * 2u * phi(t0 + u^2) is smooth.
inline double GaussianPart(double k, double s) {
    const double t0 = -0.375 - s * s;
    const double lo_z = std::max(t0, k - 12.0 * s), hi_z = k + 12.0 * s;
    if (hi_z <= t0) return 0.0;
    const double norm = 1.0 / (s * std::sqrt(2.0 * M_PI));
    const auto g = [&](double u) {
        const double d = (t0 + u * u - k) / s;
        return 4.0 * u * u * norm * std::exp(-0.5 * d * d);
    };
    return Integrate(g, std::sqrt(lo_z - t0), std::sqrt(hi_z - t0), 24);
}

}  // namespace detail

/// eq. 4.10: E{ f_s(z) | y, s } = sum_k P(k; y) * int f_s(z) N(z; k, s^2) dz.
/// s == 0 is the pure-Poisson sum of eq. 3.4. The Poisson sum is truncated where the
/// remaining mass is below 1e-16 (12 standard deviations either side of y).
inline double ExpectedGat(double y, double s) {
    if (y < 0.0) throw std::invalid_argument("ExpectedGat: y < 0");
    const double sd = std::sqrt(y);
    const long k_lo = std::max(0L, static_cast<long>(std::floor(y - 12.0 * sd - 10.0)));
    const long k_hi = static_cast<long>(std::ceil(y + 12.0 * sd + 20.0));
    double sum = 0.0;
    for (long k = k_lo; k <= k_hi; ++k) {
        // log P(k; y) = k ln y - y - ln k!; y = 0 puts all mass at k = 0.
        const double logp = (y > 0.0 ? k * std::log(y) - y : (k == 0 ? 0.0 : -INFINITY)) - std::lgamma(k + 1.0);
        const double p = std::exp(logp);
        if (p == 0.0) continue;
        sum += p * (s > 0.0 ? detail::GaussianPart(static_cast<double>(k), s) : GatReduced(static_cast<double>(k), 0.0));
    }
    return sum;
}

/// The exact unbiased inverse I_s for one s (eq. 4.9), as a table: E{f_s | y_i, s} on a
/// grid y_i in [0, y_max] with quadratic spacing (dense near 0, where the bias lives; thesis
/// §3.3 footnote and §4.3.5), linear interpolation in between. Beyond y_max it uses the
/// closed form (4.37), whose error there is O(D^-4) (§4.3.3). Below E{f_s | 0, s} it returns
/// 0: the ML inverse of eq. 4.14. s is fixed per camera and run, so one table serves a run.
class ExactInverse {
public:
    explicit ExactInverse(double s, double y_max = 200.0, int intervals = 1200) : s_(s) {
        if (!(s >= 0.0) || !(y_max > 0.0) || intervals < 2) throw std::invalid_argument("ExactInverse: bad grid");
        y_.resize(static_cast<size_t>(intervals) + 1);
        d_.resize(y_.size());
        for (size_t i = 0; i < y_.size(); ++i) {
            const double t = static_cast<double>(i) / intervals;
            y_[i] = y_max * t * t;
            d_[i] = ExpectedGat(y_[i], s);
            if (i > 0 && !(d_[i] > d_[i - 1])) throw std::runtime_error("ExactInverse: E{f|y} not increasing");
        }
    }

    double s() const { return s_; }
    double DMin() const { return d_.front(); }   // E{f_s | 0, s}
    double DMax() const { return d_.back(); }
    const std::vector<double>& y() const { return y_; }
    const std::vector<double>& expected() const { return d_; }

    /// D (a denoised value of f_s, reduced units) -> y.
    double operator()(double D) const {
        if (D <= d_.front()) return 0.0;
        if (D >= d_.back()) return ClosedFormInverse(D, s_);
        const size_t hi = static_cast<size_t>(std::upper_bound(d_.begin(), d_.end(), D) - d_.begin());
        const size_t lo = hi - 1;
        const double w = (D - d_[lo]) / (d_[hi] - d_[lo]);
        return y_[lo] + w * (y_[hi] - y_[lo]);
    }

private:
    double s_;
    std::vector<double> y_, d_;
};

}  // namespace ariadne::perception::radiometric
