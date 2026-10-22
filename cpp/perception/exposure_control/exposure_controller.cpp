#include "perception/exposure_control/exposure_controller.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ariadne::perception::exposure {
namespace {

cv::Mat ToUnitFloat(const cv::Mat& gray) {
    if (gray.channels() != 1) throw std::invalid_argument("exposure: expects a single-channel image");
    cv::Mat f;
    if (gray.depth() == CV_8U) {
        gray.convertTo(f, CV_32F, 1.0 / 255.0);
    } else if (gray.depth() == CV_32F) {
        f = gray;
    } else {
        throw std::invalid_argument("exposure: expects CV_8U or CV_32F");
    }
    return f;
}

// The metric on an already-subsampled unit-float image.
double GradientInfoOf(const cv::Mat& f, const GradientInfoParams& p) {
    cv::Mat gx, gy, mag;
    cv::Sobel(f, gx, CV_32F, 1, 0, 3);
    cv::Sobel(f, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    // Unscaled 3x3 Sobel gives 4 across a full 0 -> 1 step: divide by 4 so m is in [0, 1] (clamped for diagonals).
    const double n = std::log(p.lambda * (1.0 - p.delta) + 1.0);
    double sum = 0.0;
    for (int y = 0; y < mag.rows; ++y) {
        const float* row = mag.ptr<float>(y);
        for (int x = 0; x < mag.cols; ++x) {
            const double m = std::min(1.0, row[x] / 4.0);
            if (m >= p.delta) sum += std::log(p.lambda * (m - p.delta) + 1.0) / n;
        }
    }
    return mag.total() ? sum / static_cast<double>(mag.total()) : 0.0;
}

cv::Mat Subsample(const cv::Mat& f, int stride) {
    if (stride <= 1) return f;
    cv::Mat s;
    cv::resize(f, s, cv::Size(std::max(1, f.cols / stride), std::max(1, f.rows / stride)), 0, 0, cv::INTER_AREA);
    return s;
}

}  // namespace

double GradientInformation(const cv::Mat& gray, const GradientInfoParams& p) {
    return GradientInfoOf(Subsample(ToUnitFloat(gray), p.stride), p);
}

double ExposureCeiling(const ExposureControlParams& p, double angular_rate_rad_s, double focal_px) {
    double t = p.t_max_s;
    const double w = std::abs(angular_rate_rad_s);
    // A point at the image centre moves f * |omega| px/s, so exposing for t smears it f |omega| t px.
    if (w > 0.0 && focal_px > 0.0) t = std::min(t, p.max_blur_px / (focal_px * w));
    return std::max(t, p.t_min_s);
}

PhotometricState Allocate(const PhotometricState& s, double stops, const ExposureControlParams& p, double t_max_s,
                          double* applied_stops) {
    PhotometricState n = s;
    double want = stops;
    // The blur ceiling binds even when no change is requested: shorten t, and ask for the lost stops back from the
    // other actuators so the frame's brightness does not drop with it.
    if (n.exposure_s > t_max_s) {
        want += std::log2(n.exposure_s / t_max_s);
        n.exposure_s = t_max_s;
    }
    // Move one actuator in log2 units within [lo, hi], consuming `want`.
    const auto spend = [&want](double& v, double lo, double hi) {
        if (want > 0.0 && v < hi) {
            const double use = std::min(want, std::log2(hi / v));
            v *= std::exp2(use);
            want -= use;
        } else if (want < 0.0 && v > lo) {
            const double use = std::max(want, std::log2(lo / v));
            v *= std::exp2(use);
            want -= use;
        }
    };
    if (want > 0.0) {   // brighter: photons first (lamp, then t), gain last
        spend(n.lamp, p.lamp_min, p.lamp_max);
        spend(n.exposure_s, p.t_min_s, t_max_s);
        spend(n.gain, p.gain_min, p.gain_max);
    } else if (want < 0.0) {   // darker: give up gain first, photons last
        spend(n.gain, p.gain_min, p.gain_max);
        spend(n.exposure_s, p.t_min_s, t_max_s);
        spend(n.lamp, p.lamp_min, p.lamp_max);
    }
    if (applied_stops) *applied_stops = std::log2(n.Brightness() / s.Brightness());
    return n;
}

ControlDecision Decide(const cv::Mat& gray, const PhotometricState& cur, const ExposureControlParams& p,
                       double angular_rate_rad_s, double focal_px) {
    const cv::Mat f = ToUnitFloat(gray);
    ControlDecision d;
    d.saturated_frac = cv::countNonZero(f >= p.saturation_level) / static_cast<double>(f.total());
    d.shadow_frac = cv::countNonZero(f < p.shadow_level) / static_cast<double>(f.total());

    const cv::Mat s = Subsample(f, p.metric.stride);
    d.metric = GradientInfoOf(s, p.metric);

    double best = 0.0;
    if (d.saturated_frac > p.max_saturated_frac) {
        // Clipped pixels carry no information about how far over they are, so a synthesised darker frame
        // underestimates the gain from stepping down: step down at the slew limit instead of trusting the metric.
        best = -p.max_step_stops;
        d.reason = "overexposed";
    } else {
        // Linear sensor: a frame at +delta stops is this frame times 2^delta, clipped at 1 (Zhang et al. 2017).
        double best_m = d.metric;
        const int k = static_cast<int>(std::floor(p.max_step_stops / p.grid_step_stops + 1e-9));
        for (int i = -k; i <= k; ++i) {
            if (i == 0) continue;
            const double delta = i * p.grid_step_stops;
            cv::Mat c;
            cv::min(s * std::exp2(delta), 1.0, c);
            // A candidate the overexposure guard would undo next frame is not a candidate: without this the metric
            // (which barely penalises clipping) and the guard alternate +-0.5 stop forever (found by the closed-loop
            // test: a limit cycle with 20 % of the frame clipped at one end).
            if (delta > 0.0 &&
                cv::countNonZero(c >= p.saturation_level) > p.max_saturated_frac * static_cast<double>(c.total())) {
                continue;
            }
            const double m = GradientInfoOf(c, p.metric);
            // Strictly better by 0.5 % of the current score: no dithering between near-equal candidates.
            if (m > best_m + 0.005 * std::max(best_m, 1e-6) ||
                (m > best_m && std::abs(delta) < std::abs(best))) {
                best_m = m;
                best = delta;
            }
        }
        d.reason = "metric";
        if (!(best_m > 0.0)) {
            // No candidate shows a single gradient above delta: the metric is blind (Shim's objective is flat at 0
            // in a frame many stops under or over). Fall back to the frame's mean, stepping at the slew limit
            // towards mid-grey (0.18 linear), until texture appears and the metric takes over.
            best = cv::mean(f)[0] < 0.18 ? p.max_step_stops : -p.max_step_stops;
            d.reason = "blind";
        }
    }
    double applied = 0.0;
    d.next = Allocate(cur, best, p, ExposureCeiling(p, angular_rate_rad_s, focal_px), &applied);
    d.step_stops = applied;
    if (std::abs(applied - best) > 1e-9 && d.reason != "overexposed") d.reason = "limited";
    return d;
}

cv::Mat NormalizeExposure(const cv::Mat& gray_f32, const PhotometricState& s, double t0_s, double g0) {
    if (gray_f32.depth() != CV_32F) throw std::invalid_argument("NormalizeExposure: expects CV_32F");
    return gray_f32 * ((t0_s * g0) / (s.exposure_s * s.gain));
}

}  // namespace ariadne::perception::exposure
