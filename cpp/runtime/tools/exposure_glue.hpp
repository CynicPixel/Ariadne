#pragma once
// ---------------------------------------------------------------------------
// 5.3 glue (arch §6.5; D132, D147): the exposure + lamp co-controller at
// keyframe rate, in replay. Owned by the runtime because it reads the frame's
// applied photometric state and commands the source. Kept out of
// stereo_vo_runner.hpp so that shared file only gains call sites.
//
//   in   the keyframe's rectified left gray (after Stage 1, linear in
//        radiance) and the state that frame was ACTUALLY taken with
//        (`frame.photo`, the report half of the D147 contract), the angular
//        rate (blur ceiling, §10.5)
//   out  a PhotometricCommand for the source (applied from a later frame),
//        and one exposure_trace.csv row per decision
//
// The controller never reads its own commands back: the next decision starts
// from whatever the frames report as applied (late, clamped, or dropped
// commands then cost nothing but what they really cost).
//
// Disabled (no `exposure_control:` section): nothing is called, nothing is
// written.
// ---------------------------------------------------------------------------
#include "perception/exposure_control/exposure_config.hpp"
#include "perception/exposure_control/exposure_controller.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ariadne::tools {

class KeyframeExposure {
public:
    KeyframeExposure(const perception::exposure::ExposureControlStage& p, bool source_can_actuate) : p_(p) {
        if (!p_.Enabled()) return;
        if (!source_can_actuate) {
            throw std::runtime_error("exposure_control: the source cannot actuate exposure/gain/lamp (sim needs D with "
                                     "input_transfer: srgb)");
        }
        trace_ << "decision,decided_ns,metric,saturated_frac,shadow_frac,reason,step_stops,applied_decision,"
                  "applied_ns,exposure_rel,gain,lamp_rel,omega_rad_s,cmd_exposure_rel,cmd_gain,cmd_lamp_rel\n";
    }

    bool enabled() const { return p_.Enabled(); }

    /// One decision at a keyframe. `photo` is the keyframe's applied state; `omega_rad_s` its angular rate (0 when
    /// unknown: the ceiling is then t_max_s). Returns the command to send to the source.
    runtime::PhotometricCommand Decide(const cv::Mat& gray, const runtime::FramePhoto& photo, int64_t t_ns,
                                       double omega_rad_s, double focal_px, int64_t frame_interval_ns) {
        // An exposure cannot outlast the frame: t_max_s must fit the recording's frame interval (D147).
        if (frame_interval_ns > 0 && p_.params.t_max_s > 1.001e-9 * double(frame_interval_ns)) {
            throw std::runtime_error("exposure_control: t_max_s " + std::to_string(p_.params.t_max_s) +
                                     " s exceeds the frame interval " + std::to_string(1e-9 * double(frame_interval_ns)) +
                                     " s");
        }
        perception::exposure::PhotometricState cur;
        cur.exposure_s = photo.exposure_rel * p_.t_ref_s;
        cur.gain = photo.gain;
        cur.lamp = photo.lamp_rel;
        const perception::exposure::ControlDecision d =
            perception::exposure::Decide(gray, cur, p_.params, omega_rad_s, focal_px);
        runtime::PhotometricCommand cmd;
        cmd.decision_id = ++decisions_;
        cmd.decided_ns = t_ns;
        cmd.exposure_rel = d.next.exposure_s / p_.t_ref_s;
        cmd.gain = d.next.gain;
        cmd.lamp_rel = d.next.lamp;
        ++reasons_[d.reason];
        sat_max_ = std::max(sat_max_, d.saturated_frac);
        step_abs_max_ = std::max(step_abs_max_, std::abs(d.step_stops));
        if (photo.decision_id > 0 && photo.applied_ns >= 0) {
            latency_ns_max_ = std::max(latency_ns_max_, photo.applied_ns - decided_ns_[photo.decision_id]);
        }
        decided_ns_[cmd.decision_id] = t_ns;
        last_ = photo;
        trace_ << cmd.decision_id << ',' << t_ns << ',' << d.metric << ',' << d.saturated_frac << ',' << d.shadow_frac
               << ',' << d.reason << ',' << d.step_stops << ',' << photo.decision_id << ',' << photo.applied_ns << ','
               << photo.exposure_rel << ',' << photo.gain << ',' << photo.lamp_rel << ',' << omega_rad_s << ','
               << cmd.exposure_rel << ',' << cmd.gain << ',' << cmd.lamp_rel << '\n';
        return cmd;
    }

    /// The keyframe bundle's §18.2 states (5.3 owns them now): t in seconds through t_ref_s.
    double ExposureSeconds(const runtime::FramePhoto& photo) const { return photo.exposure_rel * p_.t_ref_s; }

    std::string Trace() const { return trace_.str(); }

    std::string Summary() const {
        return p_.mode + ": " + p_.params.metric.metric + ", t_ref " + std::to_string(p_.t_ref_s) + " s, t " +
               std::to_string(p_.params.t_min_s) + "-" + std::to_string(p_.params.t_max_s) + " s, gain <= " +
               std::to_string(p_.params.gain_max) + ", lamp " + std::to_string(p_.params.lamp_min) + "-" +
               std::to_string(p_.params.lamp_max) + " (D132 fixed order lamp > t > gain, D147)";
    }

    /// results.yaml block; nothing when the controller is off.
    void WriteResults(std::ostream& out) const {
        if (!p_.Enabled()) return;
        out << "exposure_control: {mode: " << p_.mode << ", metric: " << p_.params.metric.metric
            << ", decisions: " << decisions_ << ", reasons: {";
        bool first = true;
        for (const auto& [r, n] : reasons_) out << (first ? "" : ", ") << r << ": " << n, first = false;
        out << "}, saturated_frac_max: " << sat_max_ << ", step_abs_max_stops: " << step_abs_max_
            << ", latency_ms_max: " << 1e-6 * double(latency_ns_max_) << ",\n"
            << "  last_applied: {exposure_rel: " << last_.exposure_rel << ", gain: " << last_.gain
            << ", lamp_rel: " << last_.lamp_rel << "}}\n";
    }

private:
    perception::exposure::ExposureControlStage p_;
    uint64_t decisions_ = 0;
    std::map<std::string, int> reasons_;
    std::map<uint64_t, int64_t> decided_ns_;
    double sat_max_ = 0.0, step_abs_max_ = 0.0;
    int64_t latency_ns_max_ = 0;
    runtime::FramePhoto last_;
    std::ostringstream trace_;
};

}  // namespace ariadne::tools
