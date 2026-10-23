#pragma once
// ---------------------------------------------------------------------------
// Active exposure + lamp co-control (arch §6.5; plan 5.3) and Stage 4 exposure
// normalisation (§6.2).
//
// Three actuators set how bright the next frame is: exposure time t, gain g,
// lamp power P. On a linear sensor with the lamp as the only light, the mean
// signal is proportional to t * g * P, so the controller decides ONE quantity
// -- the brightness change in stops -- and then spends it across actuators by
// what each one costs:
//   lamp  more photons (better SNR); costs energy and, in dust, backscatter
//   t     more photons; costs motion blur (t_max from the angular rate, §10.5)
//   gain  no new photons, amplifies read noise; the last resort
// Brighter: lamp up to its budget, then t up to the blur limit, then gain.
// Darker: the reverse (gain first, so photons are given up last).
//
// The objective is Shim, Lee & Kweon's gradient information (IROS 2014, arXiv
// 1708.07338 eq. 1) [PAPER]. Candidate exposures are synthesised by scaling
// the current linear frame and clipping, which is Zhang, Forster & Scaramuzza
// (ICRA 2017)'s use of a known response; Shim's gamma-mapping stands in for an
// UNKNOWN response, and ours is known (linear, D's model).
//
// In replay the decision drives D's exposure/gain (and lamp power through the
// radiance scale, if the render is linear in lamp power); live, the lamp goes
// through gz's light_config. Exposure time never adds motion blur in D: a stated
// replay approximation (§5.6).
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <string>

namespace ariadne::perception::exposure {

/// Shim et al. eq. 1: per-pixel gradient magnitude m in [0, 1] mapped to
///   m_bar = log(lambda (m - delta) + 1) / N   for m >= delta, else 0,   N = log(lambda (1 - delta) + 1).
/// Gradients below delta count as noise; lambda sets how much weak texture counts. The paper's values.
struct GradientInfoParams {
    /// shim: the mapped mean above. percentile: the p-th percentile of gradient magnitude, Zhang, Forster & Scaramuzza
    /// ICRA 2017's M_perc (best in 13 of 18 of their datasets, ALGORITHMS §9.4); the §6.5 comparator row (D132).
    std::string metric = "shim";
    double percentile = 0.5;   // [ASSUMPTION] p for `percentile`: Zhang's value not transcribed; a sweep row
    double delta = 0.06;
    double lambda = 1e3;
    int stride = 2;          // evaluate every stride-th pixel: the metric is a sum, subsampling keeps its argmax
};

/// Mean of m_bar over the image (the paper's M divided by the pixel count, so it does not depend on resolution).
/// `gray` is CV_32F in [0, 1], linear.
double GradientInformation(const cv::Mat& gray, const GradientInfoParams& p = {});

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

struct ControlDecision {
    PhotometricState next;
    double step_stops = 0.0;       // the brightness change decided (after limits)
    double metric = 0.0;           // gradient information of the current frame
    double saturated_frac = 0.0, shadow_frac = 0.0;
    std::string reason;            // metric | overexposed | blind (no gradient anywhere: step toward mid-grey) | limited
};

/// Effective t_max: the blur limit at the current angular rate, capped by t_max_s and floored at t_min_s.
double ExposureCeiling(const ExposureControlParams& p, double angular_rate_rad_s, double focal_px);

/// Spend `stops` of brightness change across lamp / t / gain in the order above, within limits.
/// Returns the new state; `applied_stops` gets what the limits allowed.
PhotometricState Allocate(const PhotometricState& s, double stops, const ExposureControlParams& p, double t_max_s,
                          double* applied_stops);

/// One decision at keyframe rate. `gray` is the current frame (CV_8U or CV_32F in [0, 1], linear).
ControlDecision Decide(const cv::Mat& gray, const PhotometricState& cur, const ExposureControlParams& p,
                       double angular_rate_rad_s, double focal_px);

/// Stage 4 (§6.2): scale a linear working image to the canonical (t0, g0) so cross-frame photometric comparisons
/// see scene change, not actuator change. Lamp power is NOT divided here: Stage 3 owns P.
cv::Mat NormalizeExposure(const cv::Mat& gray_f32, const PhotometricState& s, double t0_s, double g0);

}  // namespace ariadne::perception::exposure
