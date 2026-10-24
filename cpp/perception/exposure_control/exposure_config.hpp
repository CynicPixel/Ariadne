#pragma once
// The perception config's optional `exposure_control:` section (plan 5.3, D147). Absent = mode off = the
// controller does not exist, and no output changes.
#include "core/config/config.hpp"
#include "perception/exposure_control/exposure_params.hpp"

#include <string>

namespace ariadne::perception::exposure {

struct ExposureControlStage {
    // off     the controller is absent (default)
    // replay  a D132 decision at every keyframe, commanded to the source (D's exposure, analog gain and lamp_power
    //         from the next frame on); everything downstream reads the state each frame reports as applied
    std::string mode = "off";
    /// The seconds that D exposure 1 stands for. D's exposure is relative and adds no blur, so this only sets the
    /// headroom log2(t_max / t_ref) below the ceiling: [ASSUMPTION] 12.5 ms = 2 stops under a 50 ms frame period,
    /// derived from the D145 brightness measurement (D147).
    double t_ref_s = 0.0125;
    ExposureControlParams params;

    bool Enabled() const { return mode != "off"; }
};

inline void ValidateExposureControl(const ExposureControlStage& e) {
    const auto fail = [](const std::string& m) { throw core::ConfigError("exposure_control: " + m); };
    const ExposureControlParams& p = e.params;
    if (e.mode != "off" && e.mode != "replay") fail("mode off|replay");
    if (p.metric.metric != "shim" && p.metric.metric != "percentile") fail("metric shim|percentile");
    if (!(p.metric.percentile > 0.0 && p.metric.percentile < 1.0)) fail("0 < percentile < 1");
    if (!(e.t_ref_s > 0.0) || !(p.t_min_s > 0.0) || !(p.t_max_s > p.t_min_s) || !(e.t_ref_s >= p.t_min_s) ||
        !(e.t_ref_s <= p.t_max_s)) {
        fail("0 < t_min_s <= t_ref_s <= t_max_s");
    }
    if (!(p.max_blur_px > 0.0) || !(p.gain_min >= 1.0) || !(p.gain_max >= p.gain_min) || !(p.lamp_min > 0.0) ||
        !(p.lamp_max >= p.lamp_min) || !(p.max_saturated_frac > 0.0 && p.max_saturated_frac < 1.0)) {
        fail("max_blur_px > 0, 1 <= gain_min <= gain_max, 0 < lamp_min <= lamp_max, 0 < max_saturated_frac < 1");
    }
}

inline ExposureControlStage ParseExposureControl(core::StrictConfig& s) {
    ExposureControlStage e;
    ExposureControlParams& p = e.params;
    e.mode = s.Require<std::string>("mode");
    e.t_ref_s = s.Optional<double>("t_ref_s", e.t_ref_s);
    p.metric.metric = s.Optional<std::string>("metric", p.metric.metric);
    p.metric.percentile = s.Optional<double>("percentile", p.metric.percentile);
    p.t_min_s = s.Optional<double>("t_min_s", p.t_min_s);
    // Required (D147): the hard ceiling is the recording's frame period, and D132's 30 ms default would silently
    // shrink the derived headroom. The runner refuses a t_max_s longer than the measured frame interval.
    p.t_max_s = s.Require<double>("t_max_s");
    p.max_blur_px = s.Optional<double>("max_blur_px", p.max_blur_px);
    p.gain_max = s.Optional<double>("gain_max", p.gain_max);
    p.lamp_min = s.Optional<double>("lamp_min", p.lamp_min);
    p.lamp_max = s.Optional<double>("lamp_max", p.lamp_max);
    p.max_saturated_frac = s.Optional<double>("max_saturated_frac", p.max_saturated_frac);
    s.RequireAllConsumed();
    ValidateExposureControl(e);
    return e;
}

}  // namespace ariadne::perception::exposure
