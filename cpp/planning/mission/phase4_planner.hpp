#pragma once
// ---------------------------------------------------------------------------
// Phase 4's planner behind B's seam (runtime/live/planner.hpp MissionPlanner, AGENTIC_7 §0b option B), at the
// planning level: no runtime, ROS or AWS type, so replay and live drive the same object. B's MissionPlanner adapter
// copies KeyframeInput -> DepthFrame, TaskCandidate -> TaskOption and these maps -> ReplanMaps, field by field.
//
//   Integrate(agent, t, frame)  4.1: TSDF -> incremental ESDF -> frontier update; the agent's track for 4.3.
//   Plan(t, agents, directives) 4.2 maps, 4.5 egress replan (D* Lite), 4.3 NBV options per active agent, then the
//                               operator directives and the mover's refusals: keep_out masks the plan space (no
//                               path or goal through it; B's review) and drops goals in it; assign/explore keep
//                               only goals inside, when one exists (B's semantics); priority_region scales G of
//                               every task that OBSERVES the region (goal or observed centroid inside, or the
//                               centroid within the depth gate of it: a region in unexplored space still pulls the
//                               frontier views that look into it, B's LLM-run lesson); a joint exactly-once pick (greedy: the agent
//                               with the higher best utility picks first; 4.4 CBBA replaces it).
//   Route(plan, agent, i, from) A* on the agent's plan space, line-of-sight string-pulled (4-connected chords, so
//                               no corner is cut), cut at max_command_m; never a raw-cell fallback.
// Deterministic given its inputs: sim time only, no wall clock in any decision.
// ---------------------------------------------------------------------------
#include "planning/mission/nbv.hpp"
#include "planning/route/egress.hpp"
#include "planning/volumetric/esdf.hpp"
#include "planning/volumetric/frontiers3d.hpp"
#include "planning/volumetric/traversability.hpp"
#include "planning/volumetric/tsdf_map.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ariadne::planning {

struct Phase4Briefing {
    GridFrame grid;
    Cell target;
    std::vector<Cell> entrances;   // = exits
    double z_max_m = 3.0;          // the briefing's ceiling bound (mission_v1 grid.z_max_m): IG extent
};

struct Phase4Params {
    TsdfParams tsdf;
    FrontierParams frontiers;
    TravParams trav;
    EgressWeights egress;
    double support_band_lo = -0.31, support_band_hi = 2.31;   // D113's pre-registered band (= the full envelope)
    NbvParams nbv;
    double refused_radius_m = 0.5;   // B's PlaceholderParams default: a refused goal excludes options this close
    double max_command_m = 3.0;      // B's PlaceholderParams default
    bool risk_calibrated = false;    // D112's bar
};

/// The region kinds of B's Directive (planning/mission/directives.hpp, feat/ph7-agentic), mirrored field by field.
/// goto / hold / release / cancel are runtime-level and never reach the planner.
struct RegionDirective {
    std::string id, kind;   // priority_region | keep_out | assign | explore
    int agent = -1;         // -1 = every agent
    std::vector<std::pair<double, double>> polygon;
    double weight = 1.0;    // priority_region
    bool Covers(double px, double py) const;     // even-odd, as B's
    double Distance(double px, double py) const;  // 0 inside, else to the nearest edge (as B's)
};

struct Phase4AgentInput {
    bool active = false;
    core::Vec3 position = core::Vec3::Zero();
    double flown_m = 0;
    std::vector<std::pair<double, double>> refused_goals;
};

struct Phase4AgentPlan {
    cv::Mat plan;                      // the agent's plan space at Plan time (trav_a + escape disc - keep_out), CV_8U
    cv::Mat keep_out;                  // CV_8U, nonzero = a keep_out region for this agent (Route subtracts it)
    NbvPlan nbv;                       // options after directives and refusals, sorted by utility
    std::vector<std::string> applied;  // ids of the directives that shaped them
    int excluded_refused = 0, excluded_directive = 0;
};

struct Phase4Plan {
    double t_s = 0;
    TraversabilityMaps maps;
    cv::Mat support;
    EgressReplan egress;
    Phase4AgentPlan agents[2];
    int joint[2] = {-1, -1};   // index into agents[a].nbv.options; exactly-once by task_id
};

/// Filter and rescale one agent's options by the active region directives and the mover's refusals (in place).
void ApplyDirectives(int agent, const std::vector<RegionDirective>& dirs,
                     const std::vector<std::pair<double, double>>& refused, double refused_radius_m, double observe_m,
                     Phase4AgentPlan* ap);

/// Zero `plan` inside every keep_out polygon that applies to `agent` (cv::fillPoly on the grid).
void MaskKeepOut(int agent, const std::vector<RegionDirective>& dirs, const GridFrame& g, cv::Mat* plan);

/// Exactly-once greedy joint pick: the agent whose best option has the higher utility picks first (ties: agent 0);
/// the other takes its best option with a different task_id. -1 = none.
void JointPick(Phase4Plan* p);

/// Line-of-sight string pulling of a grid path over `plan` (CV_8U): from each kept cell, the farthest later path
/// cell whose 4-connected chord is all traversable. Returns the kept cells, first and last included.
std::vector<Cell> StringPull(const cv::Mat& plan, const std::vector<Cell>& path);

class Phase4Planner {
public:
    Phase4Planner(const Phase4Briefing& b, const Phase4Params& p);
    /// Returns the number of dirty blocks.
    size_t Integrate(int agent, int64_t t_ns, const DepthFrame& f);
    Phase4Plan Plan(double t_s, const Phase4AgentInput agents[2], const std::vector<RegionDirective>& dirs);
    std::pair<std::vector<std::pair<double, double>>, double> Route(const Phase4Plan& p, int agent, int i,
                                                                   const Cell& from) const;
    const TsdfMap& Map() const { return map_; }
    const Esdf& Distance() const { return esdf_; }
    const FrontierSet3D& Frontiers() const { return fr_; }

private:
    cv::Mat PlanSpace(const TraversabilityMaps& m, const Cell& at) const;   // trav_a + the escape disc (D158)
    Cell CellOf(const core::Vec3& p) const;

    Phase4Briefing b_;
    Phase4Params p_;
    TsdfMap map_;
    Esdf esdf_;
    FrontierSet3D fr_;
    EgressPlanner egress_;
    std::vector<core::Vec3> track_[2];
    bool detected_ = false;
};

}  // namespace ariadne::planning
