#pragma once
// ---------------------------------------------------------------------------
// Noise measured, not assumed (arch §6.2, 2.2 acceptance): fit the signal-
// dependent noise model var = a * mean + b from the images themselves, then
// check that the GAT built from the fit really stabilises the noise to ~1.
// OpenCV-free.
//
// Method: a simplified single-image estimator after Foi, Trimeche, Katkovnik,
// Egiazarian, "Practical Poissonian-Gaussian noise modeling and fitting for
// single-image raw-data", IEEE TIP 17(10), 2008 [PROXY: no ML fit and no
// clipped-data likelihood; clipped neighbourhoods are excluded instead].
// Per pixel (on a stride grid):
//   level  m = 3x3 mean                       (what brightness this pixel is at)
//   detail d = (3x3 second difference) / 6    (white noise of variance v -> var(d) = v;
//                                              exactly 0 on a linear ramp)
//   edge   q = curvature at the 5-px scale: cx, cy = second differences of 5x5 box means
//               5 px apart (B(x+5) - 2B(x) + B(x-5)). Exactly 0 on any linear shading, so
//               smooth gradients are kept (d ignores them too); large at edges and texture.
//               For white noise of variance v: var(cx) = var(cy) = 0.24 v and
//               cov(cx, cy) = 0.16 v (they share the centre box), so the Mahalanobis form
//               q = (7.5 cx^2 - 10 cx cy + 7.5 cy^2) / v is chi^2_2 under noise alone.
// The mean and second-difference filters are orthogonal (their dot product is 0), and d is
// uncorrelated with cx, cy (d sums to 0 over the centre box), so for Gaussian noise the bin a
// pixel falls in and its edge test are independent of its detail sample.
// Pass 1 bins by level and takes v0 = (1.4826 MAD(d))^2 per bin; pass 2 drops pixels
// whose q at v0 exceeds the 99% point of chi^2_2 (structure, not noise) and re-takes
// the per-bin MAD; a weighted least-squares line through (level, variance) gives a, b.
// Then the stabilised-domain check (the RPG-VST idea, arXiv 2607.24291): apply the GAT
// with the fitted model and measure the detail std again; ~1 means the fit is usable.
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"
#include "perception/radiometric/gat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::perception::radiometric {

struct NoiseFitParams {
    int stride = 2;                  // sample every stride-th pixel in x and y
    double bin_width_dn = 4.0;       // level bins, DN
    int min_pixels_per_bin = 200;    // a bin with fewer accepted pixels is not used
    int min_bins = 6;
    double min_level_span_dn = 16.0; // the used bins must span at least this many DN, or a and b are not separable
    double edge_chi2 = 9.21;         // chi^2_2 99%: an edge statistic q above this is structure, not noise
    size_t max_per_bin = 200000;     // memory cap; later samples of a full bin are dropped
};

struct NoiseBin {
    double level = 0.0;   // mean of the 3x3 means in the bin, DN
    double var = 0.0;     // (1.4826 MAD of the detail)^2, DN^2
    long n = 0;
};

struct NoiseFit {
    bool valid = false;
    std::string why;               // empty when valid
    double a = 0.0, b = 0.0;       // var = a * level + b
    std::vector<NoiseBin> bins;    // the bins used
    long sampled = 0, unclipped = 0, accepted = 0;
    double median_level = 0.0;     // median 3x3 level over all sampled pixels, DN
    double snr_at_median = 0.0;    // median_level / sqrt(a median_level + b)
};

namespace detail {

inline double Median(std::vector<float>& v) {
    if (v.empty()) return 0.0;
    const size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
    return v[k];
}

/// 1.4826 * median|x| of zero-mean samples: the robust std. Reorders v.
inline double RobustSigma(std::vector<float>& v) {
    for (float& x : v) x = std::abs(x);
    return 1.4826 * Median(v);
}

struct Features {
    float level, detail, q;   // q: the edge statistic times v (compare against chi2 * v)
};

/// The margin the three quantities need around (x, y).
inline constexpr int kMargin = 7;

/// The three per-pixel quantities at (x, y). `px(x, y)` reads the signal.
template <class Px>
Features At(const Px& px, int x, int y) {
    double s3 = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) s3 += px(x + dx, y + dy);
    // Outer product of [1 -2 1] with itself: corners 1, edges -2, centre 4; sum of squares 36.
    const double d = (px(x - 1, y - 1) + px(x + 1, y - 1) + px(x - 1, y + 1) + px(x + 1, y + 1) -
                      2.0 * (px(x, y - 1) + px(x, y + 1) + px(x - 1, y) + px(x + 1, y)) + 4.0 * px(x, y)) / 6.0;
    const auto box5 = [&](int cx, int cy) {
        double s = 0.0;
        for (int dy = -2; dy <= 2; ++dy)
            for (int dx = -2; dx <= 2; ++dx) s += px(cx + dx, cy + dy);
        return s / 25.0;
    };
    const double b0 = box5(x, y);
    const double cx = box5(x + 5, y) - 2.0 * b0 + box5(x - 5, y);
    const double cy = box5(x, y + 5) - 2.0 * b0 + box5(x, y - 5);
    const double q = 7.5 * cx * cx - 10.0 * cx * cy + 7.5 * cy * cy;
    return {static_cast<float>(s3 / 9.0), static_cast<float>(d), static_cast<float>(q)};
}

/// True if the 3x3 raw neighbourhood touches 0 or 255: clipping shrinks the variance there.
inline bool Clipped(const core::ImageU8& raw, int x, int y) {
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const uint8_t v = raw.At(x + dx, y + dy);
            if (v == 0 || v == 255) return true;
        }
    return false;
}

}  // namespace detail

/// Fit var = a * level + b over one or more frames. `z[i]` is the signal the model is for (the
/// flat-fielded image, or the raw image as float); `raw[i]` is the 8-bit image it came from (for the
/// clipping test). All frames must share one size.
inline NoiseFit FitNoise(const std::vector<const core::ImageF32*>& z, const std::vector<const core::ImageU8*>& raw,
                         const NoiseFitParams& p = {}) {
    NoiseFit out;
    if (z.empty() || z.size() != raw.size()) {
        out.why = "no frames";
        return out;
    }
    struct Sample {
        float detail, q;
    };
    std::vector<std::vector<Sample>> bins;
    std::vector<std::vector<float>> levels;
    std::vector<float> all_levels;
    for (size_t f = 0; f < z.size(); ++f) {
        const core::ImageF32& im = *z[f];
        const auto px = [&](int x, int y) { return static_cast<double>(im.At(x, y)); };
        for (int y = detail::kMargin; y < im.height - detail::kMargin; y += p.stride) {
            for (int x = detail::kMargin; x < im.width - detail::kMargin; x += p.stride) {
                const detail::Features ft = detail::At(px, x, y);
                ++out.sampled;
                all_levels.push_back(ft.level);
                if (detail::Clipped(*raw[f], x, y)) continue;
                ++out.unclipped;
                const size_t bin = static_cast<size_t>(std::max(0.0, ft.level / p.bin_width_dn));
                if (bin >= bins.size()) bins.resize(bin + 1), levels.resize(bin + 1);
                if (bins[bin].size() >= p.max_per_bin) continue;
                bins[bin].push_back({ft.detail, ft.q});
                levels[bin].push_back(ft.level);
            }
        }
    }
    out.median_level = detail::Median(all_levels);

    // Pass 1: per-bin variance with structure included (MAD is robust to a minority of edges);
    // pass 2: drop pixels whose gradient is implausible for noise of that variance, re-estimate.
    std::vector<double> xs, vs, ns;
    for (size_t i = 0; i < bins.size(); ++i) {
        if (static_cast<int>(bins[i].size()) < p.min_pixels_per_bin) continue;
        std::vector<float> d0;
        for (const auto& s : bins[i]) d0.push_back(s.detail);
        const double v0 = std::pow(detail::RobustSigma(d0), 2);
        std::vector<float> d1;
        double level_sum = 0.0;
        for (size_t k = 0; k < bins[i].size(); ++k) {
            if (bins[i][k].q > p.edge_chi2 * v0) continue;
            d1.push_back(bins[i][k].detail);
            level_sum += levels[i][k];
        }
        if (static_cast<int>(d1.size()) < p.min_pixels_per_bin) continue;
        const long n = static_cast<long>(d1.size());
        const double level = level_sum / n;
        const double v = std::pow(detail::RobustSigma(d1), 2);
        out.accepted += n;
        if (!(v > 0.0)) continue;   // posterised: the detail is mostly exactly 0
        out.bins.push_back({level, v, n});
        xs.push_back(level);
        vs.push_back(v);
        ns.push_back(static_cast<double>(n));
    }
    if (static_cast<int>(out.bins.size()) < p.min_bins) {
        out.why = "only " + std::to_string(out.bins.size()) + " usable level bins (need " + std::to_string(p.min_bins) + ")";
        return out;
    }
    if (xs.back() - xs.front() < p.min_level_span_dn) {
        out.why = "level span " + std::to_string(xs.back() - xs.front()) + " DN too narrow to separate a and b";
        return out;
    }
    // Weighted least squares; var(v_hat) ~ 2.7 v^2 / n for the MAD estimate, so weight n / v_model^2.
    // First pass weights by the measured v, second by the first fit's prediction.
    double a = 0.0, b = 0.0;
    for (int it = 0; it < 3; ++it) {
        double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < xs.size(); ++i) {
            const double model = it == 0 ? vs[i] : std::max(a * xs[i] + b, 1e-6);
            const double w = ns[i] / (model * model);
            sw += w;
            sx += w * xs[i];
            sy += w * vs[i];
            sxx += w * xs[i] * xs[i];
            sxy += w * xs[i] * vs[i];
        }
        const double det = sw * sxx - sx * sx;
        if (!(det > 0.0)) {
            out.why = "degenerate level distribution";
            return out;
        }
        a = (sw * sxy - sx * sy) / det;
        b = (sxx * sy - sx * sxy) / det;
    }
    out.a = a;
    out.b = b;
    out.snr_at_median = a * out.median_level + b > 0.0 ? out.median_level / std::sqrt(a * out.median_level + b) : 0.0;
    if (!(a > 0.0)) {
        out.why = "fitted a = " + std::to_string(a) + " is not positive";
        return out;
    }
    out.valid = true;
    return out;
}

/// The stabilised-domain check: GAT the frames with `model`, then the robust std of the detail over
/// unclipped, structure-free sample sites. ~1 when the model is right; posterised or mis-modelled data
/// lands far from 1. The edge mask is scaled by the MEASURED stabilised variance (two passes, as in
/// FitNoise), never by the hoped-for 1: a mask at v = 1 would cut away the large-noise pixels of a wrong
/// model and pull the answer back toward 1, hiding the failure this check exists to catch.
inline double StabilisedSigma(const std::vector<const core::ImageF32*>& z, const std::vector<const core::ImageU8*>& raw,
                              const PoissonGaussian& model, const NoiseFitParams& p = {}) {
    std::vector<float> d, q;
    for (size_t f = 0; f < z.size(); ++f) {
        const core::ImageF32& im = *z[f];
        const auto fz = [&](int x, int y) { return GatForward(im.At(x, y), model); };
        for (int y = detail::kMargin; y < im.height - detail::kMargin; y += p.stride) {
            for (int x = detail::kMargin; x < im.width - detail::kMargin; x += p.stride) {
                if (detail::Clipped(*raw[f], x, y)) continue;
                const detail::Features ft = detail::At(fz, x, y);
                d.push_back(ft.detail);
                q.push_back(ft.q);
            }
        }
    }
    if (d.size() < 1000) return 0.0;
    std::vector<float> d0 = d;
    const double v0 = std::pow(detail::RobustSigma(d0), 2);
    std::vector<float> d1;
    for (size_t i = 0; i < d.size(); ++i)
        if (q[i] <= p.edge_chi2 * v0) d1.push_back(d[i]);
    return d1.size() < 1000 ? 0.0 : detail::RobustSigma(d1);
}

}  // namespace ariadne::perception::radiometric
