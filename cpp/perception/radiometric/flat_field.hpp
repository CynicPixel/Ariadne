#pragma once
// ---------------------------------------------------------------------------
// Flat-field correction (arch §6.2 stage 1): divide each pixel by its relative
// response K(x), the product of optical vignetting (cos^4 falloff) and the
// sensor's photo-response non-uniformity (PRNU, a fixed per-pixel gain).
// OpenCV-free.
//
// K is measured the way a lab measures it: average many frames of a uniformly
// lit target, so temporal noise averages out and the fixed pattern remains. In
// the simulator the target is D applied to a flat input with the camera's own
// D seed (the PRNU field is keyed on the seed, so it is the same silicon).
// D's analytic VignetteFactor x PrnuGain is the ground truth it is tested
// against. Sensor only: the lamp's beam profile is Phase 5 (falloff, §6.3).
//
// Where it sits relative to the noise (recorded in RADIOMETRIC_2.2.md §3): D
// applies K AFTER shot and read noise, so a raw pixel has var = K a m' + K^2 b;
// dividing by K first restores one homogeneous (a, b) for the whole frame,
// which is what the GAT needs. This is arch §6.2's order (flat-field, then VST).
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"
#include "perception/degradation/degradation.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::perception::radiometric {

/// Frame ids used for calibration frames, far from any run's frame ids, so a calibration frame never
/// shares a temporal-noise stream with a frame the estimator sees.
inline constexpr uint64_t kFlatFieldFrameBase = 0xC0FFEE0000000000ull;
/// K below this is treated as dead (the division would amplify noise without bound).
inline constexpr double kMinRelativeResponse = 0.1;
/// cv::COLOR_RGB2GRAY's weights (ITU-R BT.601).
inline constexpr double kGrayWeights[3] = {0.299, 0.587, 0.114};

struct FlatField {
    int width = 0, height = 0;
    std::vector<float> k;   // relative response; median of the central block = 1; >= kMinRelativeResponse
    std::string source = "none";

    bool Empty() const { return k.empty(); }
    float At(int x, int y) const { return k[static_cast<size_t>(y) * width + x]; }
    double KMin() const { return k.empty() ? 1.0 : *std::min_element(k.begin(), k.end()); }
};

namespace detail {

/// Scale so the median over the central 64 x 64 block is 1 (the centre keeps its brightness), clamp low.
inline void NormaliseToCentre(FlatField& f) {
    std::vector<float> c;
    const int x0 = std::max(0, f.width / 2 - 32), y0 = std::max(0, f.height / 2 - 32);
    for (int y = y0; y < std::min(f.height, y0 + 64); ++y)
        for (int x = x0; x < std::min(f.width, x0 + 64); ++x) c.push_back(f.At(x, y));
    std::nth_element(c.begin(), c.begin() + static_cast<long>(c.size() / 2), c.end());
    const float med = c[c.size() / 2];
    for (float& v : f.k) v = std::max(v / med, static_cast<float>(kMinRelativeResponse));
}

/// The gray value the pipeline sees: the BT.601 weighted sum of the channels (or the one channel).
inline double Gray(const core::ImageU8& im, int x, int y) {
    if (im.channels == 1) return im.At(x, y);
    double g = 0.0;
    for (int c = 0; c < 3; ++c) g += kGrayWeights[c] * im.At(x, y, c);
    return g;
}

}  // namespace detail

/// Calibrate K for a camera whose sensor is `sensor` (its D parameters) with D seed `seed`: `frames` D
/// frames of a flat input at `level`, under the sensor's own noise but at unity radiance and gain (a
/// bright, unclipped flat), averaged in gray.
inline FlatField CalibrateFlatField(const DegradationParams& sensor, uint64_t seed, int width, int height, int channels,
                                    int frames = 16, uint8_t level = 200, int threads = 8) {
    DegradationParams p = sensor;
    p.radiance_scale = 1.0;
    p.exposure = p.analog_gain = p.digital_gain = 1.0;
    p.enable_dust = false;
    const core::ImageU8 flat = [&] {
        core::ImageU8 im(width, height, channels);
        std::fill(im.data.begin(), im.data.end(), level);
        return im;
    }();
    FlatField f;
    f.width = width;
    f.height = height;
    std::vector<double> sum(static_cast<size_t>(width) * height, 0.0);
    for (int i = 0; i < frames; ++i) {
        const core::ImageU8 d = DegradeParallel(flat, p, seed, kFlatFieldFrameBase + static_cast<uint64_t>(i), threads);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) sum[static_cast<size_t>(y) * width + x] += detail::Gray(d, x, y);
    }
    f.k.resize(sum.size());
    for (size_t i = 0; i < sum.size(); ++i) f.k[i] = static_cast<float>(sum[i] / frames);
    detail::NormaliseToCentre(f);
    f.source = "calibrated: " + std::to_string(frames) + " D frames of flat " + std::to_string(level) + ", seed " +
               std::to_string(seed);
    return f;
}

/// The ground truth for the simulator: D's own vignette x PRNU, gray-weighted, normalised the same way.
inline FlatField AnalyticFlatField(const DegradationParams& sensor, uint64_t seed, int width, int height, int channels) {
    FlatField f;
    f.width = width;
    f.height = height;
    f.k.resize(static_cast<size_t>(width) * height);
    const double cx = 0.5 * (width - 1), cy = 0.5 * (height - 1);
    const double inv_r2 = 1.0 / (cx * cx + cy * cy);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t pix = static_cast<size_t>(y) * width + x;
            const double v = perception::detail::VignetteFactor(((x - cx) * (x - cx) + (y - cy) * (y - cy)) * inv_r2,
                                                                sensor.vignette_strength);
            double prnu = 0.0;
            for (int c = 0; c < channels; ++c) {
                const double w = channels == 1 ? 1.0 : kGrayWeights[c];
                prnu += w * perception::detail::PrnuGain(
                                seed, perception::detail::SiteId(pix, c, channels, NoiseStage::kPrnu), sensor.prnu_sigma);
            }
            f.k[pix] = static_cast<float>(v * prnu);
        }
    }
    detail::NormaliseToCentre(f);
    f.source = "analytic (D vignette x PRNU)";
    return f;
}

/// z / K, the flat-fielded signal, as float. An empty FlatField is K = 1.
inline core::ImageF32 ApplyFlatField(const core::ImageU8& gray, const FlatField& f) {
    core::ImageF32 z(gray.width, gray.height);
    for (int y = 0; y < gray.height; ++y)
        for (int x = 0; x < gray.width; ++x) z.At(x, y) = f.Empty() ? gray.At(x, y) : gray.At(x, y) / f.At(x, y);
    return z;
}

}  // namespace ariadne::perception::radiometric
