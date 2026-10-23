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
#include "perception/exposure_control/exposure_params.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace ariadne::perception::exposure {

/// Mean of m_bar over the image (the paper's M divided by the pixel count, so it does not depend on resolution).
/// `gray` is CV_32F in [0, 1], linear.
double GradientInformation(const cv::Mat& gray, const GradientInfoParams& p = {});

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
