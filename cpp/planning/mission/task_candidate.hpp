#pragma once
// ---------------------------------------------------------------------------
// Phase 4's task option: what the planner (4.3 NBV, 4.4 allocation) hands to the runtime's chooser through
// runtime/live/planner.hpp's TaskOption (B's L1 / 7.x seam, maintainer's option B, 2026-10-23). Planning-level:
// no runtime, ROS or AWS type. Deterministic: every field is a pure function of the planner's inputs.
//
// The runtime reads: task_id (exactly-once across agents), goal_xy, path_m, standoff_m, centroid_xy, cells, risk,
// risk_reason. The rest is the planner's evidence, logged in the trace.
// ---------------------------------------------------------------------------
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ariadne::planning {

/// §21.2 candidate generators.
enum class TaskSource : char {
    kFrontier = 'A',       // frontier cluster viewpoint (FUEL-style per-cluster sampling)
    kUncertainty = 'B',    // uncertainty hotspot
    kConnectivity = 'C',   // bridge two disconnected map regions / a peer's trajectory
    kAlias = 'D',          // alias-resolution view
    kEgress = 'E',         // validate the egress route's clearance / obstruction
    kReturn = 'R',         // §21.4 hard rule: return to the nearest briefed entrance (supervisor event)
    kHold = 'H',           // §21.4 hard rule: hold position
};

/// §21.3 IG terms, in this order (raw, then normalised by the running p90 over the candidate set).
enum IgTerm { kCoverage = 0, kEntropy = 1, kPoseConnectivity = 2, kEgressConfidence = 3, kLoopDisambiguation = 4 };

/// D175 R1: one 32x32 planner block of a view's footprint (the SUBMAP_PATCH block keys) and how many of the view's
/// coverage voxels (own-unknown, not peer-known) fall in it, in units of kFootprintUnit voxels (rounded up, capped at
/// 255): a 1.6 m block holds thousands of voxels, so raw counts would saturate a uint8.
constexpr int kFootprintUnit = 16;
struct BlockGain {
    int16_t bx = 0, by = 0;
    uint8_t w = 0;
};

struct TaskCandidate {
    int64_t task_id = -1;      // stable across replans for the same task: a hash of the generator and its anchor
                               // voxel (e.g. a frontier cluster's minimum voxel), so exactly-once survives re-ranking
    TaskSource source = TaskSource::kFrontier;
    std::pair<double, double> goal_xy{0, 0};       // world m: the viewpoint position the agent drives to
    double goal_yaw = 0;                            // rad, world: the viewing direction at the goal
    double path_m = 0;                              // agent path length to the goal at rho_agent (D111 lattice)
    double standoff_m = 0;                          // goal to the observed target (frontier / hotspot), m
    std::pair<double, double> centroid_xy{0, 0};   // world m: what the view observes
    int cells = 0;                                  // the target's size (frontier voxels; info proxy for the runtime)
    std::array<double, 5> ig_raw{};                 // §21.3 terms, raw units
    std::array<double, 5> ig_norm{};                // divided by the candidate set's running p90 (self-normalised)
    double gain = 0;                                // G(a) = sum of the normalised terms (weights 1, §21.3)
    std::array<double, 5> cost{};                   // §21.4 C_motion, C_bandwidth, C_energy, C_time, C_risk
    double utility = 0;                             // U(a) = G - sum lambda_k C_k (λ scheduled, §21.4)
    // §23 gate input. risk = max(geometric, structural):
    //   geometric  the D158 bottleneck rule (what a responder would need to follow the agent there), from D111's
    //              egress clearance: always available;
    //   structural D111's [PROXY], only once D112's calibration passes its bar; until then 0, and the reason says so.
    double risk = 0;
    double risk_geometric = 0, risk_structural = 0;
    std::string risk_reason;
    std::vector<BlockGain> footprint;   // D175 R1, per-agent planning only (empty in the centralised planner): <= 32, by w
};

/// Exactly-once helper (the runtime's OtherCluster, by task id): the best-ranked option of `options` whose task is
/// not `taken_task_id` and whose risk is at or below `threshold`; -1 if none.
inline int OtherTask(const std::vector<TaskCandidate>& options, int64_t taken_task_id, double threshold) {
    for (size_t i = 0; i < options.size(); ++i) {
        if (options[i].task_id != taken_task_id && options[i].risk <= threshold) return int(i);
    }
    return -1;
}

}  // namespace ariadne::planning
