#pragma once
// The exposure controller's parameters and photometric state (D132), apart from exposure_controller.hpp so the
// perception config can parse them without OpenCV (the test_unit build has none; D147).
#include <string>

namespace ariadne::perception::exposure {

/// Shim et al. eq. 1: per-pixel gradient magnitude m in [0, 1] mapped to
///   m_bar = log(lambda (m - delta) + 1) / N   for m >= delta, else 0,   N = log(lambda (1 - delta) + 1).
/// Gradients below delta count as noise; lambda sets how much weak texture counts. The paper's values.
struct GradientInfoParams {
    /// shim: the mapped mean above. percentile: the p-th percentile of gradient magnitude, Zhang, Forster & Scaramuzza
    /// ICRA 2017's M_perc (best in 13 of 18 of their datasets, algorithms.md §9.4); the §6.5 comparator row (D132).
    std::string metric = "shim";
    double percentile = 0.5;   // [ASSUMPTION] p for `percentile`: Zhang's value not transcribed; a sweep row
    double delta = 0.06;
    double lambda = 1e3;
    int stride = 2;          // evaluate every stride-th pixel: the metric is a sum, subsampling keeps its argmax
};

/// The camera's photometric state: what Stage 4 divides out and what the controller actuates.
struct PhotometricState {
    double exposure_s = 0.01;
    double gain = 1.0;
    double lamp = 1.0;       // relative lamp power, 1 = the calibrated P
    double Brightness() const { return exposure_s * gain * lamp; }
};

struct ExposureControlParams {
    double max_step_stops = 0.5;        // §6.5 slew limit: at most half a stop per decision, so LK keeps tracking
    double grid_step_stops = 0.125;     // candidates in [-max_step, +max_step] at this spacing
    double t_min_s = 0.0005;            // from the noise model (§6.2 Stage 2); a config value until 5.4 fits it
    double t_max_s = 0.03;              // hard ceiling (frame period at 30 Hz)
    double max_blur_px = 1.0;           // t_max from blur: t <= max_blur_px / (f * |omega|)
    double gain_min = 1.0, gain_max = 16.0;
    double lamp_min = 0.05, lamp_max = 1.0;   // lamp_max is the energy budget, relative to the calibrated P
    double saturation_level = 254.5 / 255.0;  // a pixel at or above this is clipped
    double max_saturated_frac = 0.02;   // above this the frame is overexposed: step down regardless of the metric
    double shadow_level = 2.0 / 255.0;  // below this a pixel is crushed (logged, Failure E evidence)
    GradientInfoParams metric;
};

}  // namespace ariadne::perception::exposure
