#pragma once
// ---------------------------------------------------------------------------
// The §30.2 core-metric registry: the single list every run report is checked
// against (arch §30.2, Phase 1 exit gate "every metric in §30.2 emits a number,
// however bad").
//
// WHY A REGISTRY AND NOT A CHECKLIST: "every metric emits a number" is only a
// gate if something mechanical fails when one is missing. A report is a map
// from metric id to value; MissingMetrics() names every registered id that is
// absent OR non-finite (NaN is not a number, and a metric stuck at NaN would
// otherwise pass a presence check), and UnknownMetrics() names every id that is
// not registered, so a typo'd key is an error rather than a silently orphaned
// value. The same philosophy as the strict config loader.
//
// Ids are stable snake_case with the unit as a suffix where one exists. The
// registry is the contract: renaming an id is a breaking change to every
// stored report and figure script, so ids are never renamed, only added.
//
// [ASSUMPTION] §30.2's localization line lists "ATE RMSE, RPE, rotation error,
// translation error" without saying which quantity each error refers to. It is
// read the way the evo toolchain reports trajectories: translation AND rotation
// parts of BOTH the absolute (ATE) and relative (RPE) error. Those four ids
// cover all four listed items. Recorded in DECISIONS D31.
// ---------------------------------------------------------------------------
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ariadne::runtime {

enum class MetricGroup { kLocalization, kRegistration, kMapping, kCommunication, kMission };

struct MetricSpec {
    std::string_view id;
    MetricGroup group;
    std::string_view unit;
    std::string_view source;   // the §30.2 wording this id implements
};

// clang-format off
inline constexpr std::array<MetricSpec, 27> kCoreMetrics = {{
    // Localization (§30.2)
    {"ate_rmse_m",                    MetricGroup::kLocalization,  "m",     "ATE RMSE"},
    {"ate_rotation_rmse_rad",         MetricGroup::kLocalization,  "rad",   "rotation error (absolute, after alignment)"},
    {"rpe_translation_rmse_m",        MetricGroup::kLocalization,  "m",     "RPE / translation error (relative)"},
    {"rpe_rotation_rmse_rad",         MetricGroup::kLocalization,  "rad",   "RPE / rotation error (relative)"},
    // Registration
    {"inter_agent_rpe_m",             MetricGroup::kRegistration,  "m",     "inter-agent relative pose error"},
    {"closure_precision",             MetricGroup::kRegistration,  "ratio", "accepted/rejected closure precision"},
    {"false_closure_rate",            MetricGroup::kRegistration,  "ratio", "false closure rate"},
    // Mapping
    {"chamfer_m",                     MetricGroup::kMapping,       "m",     "Chamfer distance"},
    {"completeness",                  MetricGroup::kMapping,       "ratio", "completeness"},
    {"coverage",                      MetricGroup::kMapping,       "ratio", "coverage"},
    {"heldout_psnr_db",               MetricGroup::kMapping,       "dB",    "held-out-view PSNR"},
    {"heldout_ssim",                  MetricGroup::kMapping,       "ratio", "SSIM"},
    {"heldout_lpips",                 MetricGroup::kMapping,       "ratio", "LPIPS"},
    // Communication (bytes counted at the serializer, Invariant 6)
    {"bytes_total",                   MetricGroup::kCommunication, "B",     "total bytes"},
    {"bytes_per_keyframe",            MetricGroup::kCommunication, "B",     "bytes/keyframe"},
    {"bytes_per_verified_loop",       MetricGroup::kCommunication, "B",     "bytes/verified loop"},
    {"bytes_per_map_area_m2",         MetricGroup::kCommunication, "B/m2",  "bytes/map-area-reconstructed"},
    {"descriptor_bytes",              MetricGroup::kCommunication, "B",     "descriptor bytes"},
    {"local_feature_bytes",           MetricGroup::kCommunication, "B",     "local feature bytes"},
    {"pose_graph_bytes",              MetricGroup::kCommunication, "B",     "pose graph bytes"},
    // Mission (egress quantities at rho_responder, Invariant 4)
    {"time_to_first_usable_map_s",    MetricGroup::kMission,       "s",     "time to first usable map"},
    {"time_to_first_valid_egress_s",  MetricGroup::kMission,       "s",     "time to first valid egress route"},
    {"route_validity",                MetricGroup::kMission,       "ratio", "route validity vs GT traversability"},
    {"route_confidence",              MetricGroup::kMission,       "ratio", "route confidence"},
    {"route_confidence_calibration",  MetricGroup::kMission,       "ratio", "route-confidence calibration"},
    {"distance_travelled_m",          MetricGroup::kMission,       "m",     "distance travelled"},
    {"energy_proxy",                  MetricGroup::kMission,       "proxy", "energy proxy (unitless; defined with the planner, 4.3)"},
}};
// clang-format on

/// A run report: metric id -> value. Ordered, so serialized reports are stable.
using MetricReport = std::map<std::string, double>;

inline const MetricSpec* FindMetric(std::string_view id) {
    for (const MetricSpec& m : kCoreMetrics) {
        if (m.id == id) return &m;
    }
    return nullptr;
}

/// Registered ids that are absent from `report` or whose value is not finite.
/// Empty result == the §30.2 exit-gate condition holds.
inline std::vector<std::string> MissingMetrics(const MetricReport& report) {
    std::vector<std::string> missing;
    for (const MetricSpec& m : kCoreMetrics) {
        const auto it = report.find(std::string(m.id));
        if (it == report.end() || !std::isfinite(it->second)) {
            missing.emplace_back(m.id);
        }
    }
    return missing;
}

/// Ids present in `report` that the registry does not know. A typo'd key must
/// be an error, or its value is orphaned and the real metric reads as missing.
inline std::vector<std::string> UnknownMetrics(const MetricReport& report) {
    std::vector<std::string> unknown;
    for (const auto& [id, value] : report) {
        (void)value;
        if (FindMetric(id) == nullptr) unknown.push_back(id);
    }
    return unknown;
}

}  // namespace ariadne::runtime
