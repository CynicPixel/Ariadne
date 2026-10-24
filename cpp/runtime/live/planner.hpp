#pragma once
// ---------------------------------------------------------------------------
// The live runtime's planner seam (design AGENTIC_7 §0b, option B). Phase 4's planner (A1: NBV 4.3, CBBA 4.4,
// route graph 4.5/4.6) implements MissionPlanner; until it lands, PlaceholderPlanner is the 1.8 layer plus the L1
// standoff rule (D153b), an ablation baseline under the standing rule. Everything here is deterministic.
//
//   Integrate(keyframe)   one keyframe's depth into the planner's map. KeyframeInput mirrors A1's
//                         planning::MissionInput keyframe (feat/ph4-mission, VOLUMETRIC_4.1 §4); swap = one adapter.
//   Plan(context)         Plane E's products for this replan: per agent the ranked task options (each with risk and
//                         the reason for it), the per-agent classical pick and the joint (allocated) pick, and the
//                         map summary the console and the LLM's images are drawn from.
//   Route(agent, option)  the waypoints to execute an option, computed by the planner, never by the chooser.
// The chooser (classical | LLM) picks among options; the runtime validates; the mover executes Route().
// ---------------------------------------------------------------------------
#include "planning/mission/directives.hpp"
#include "planning/mission/frontier_mission.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <limits>
#include <vector>

namespace ariadne::runtime::live {

namespace mis = planning::mission;
using planning::Cell;

/// One keyframe for the map, at the spool's scale (A1's MissionInput keyframe carries the same fields).
struct KeyframeInput {
    int agent = 0;
    uint64_t id = 0;
    int64_t t_ns = 0;
    core::SE3 T_W_C;          // world <- camera
    cv::Mat Z, S;             // CV_32F depth and sigma (m)
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

struct PlanContext {   // sized by the scene's agent count (P8 D213: N agents; 2 everywhere before that)
    explicit PlanContext(int n_agents = 2)
        : active(size_t(n_agents), false), at(size_t(n_agents)),
          position(size_t(n_agents), core::Vec3::Zero()), yaw(size_t(n_agents), std::numeric_limits<double>::quiet_NaN()),
          flown_m(size_t(n_agents), 0) {}
    int n() const { return int(active.size()); }
    int64_t T = 0;
    std::vector<char> active;                 // agent has an estimated pose before T (char: vector<bool> has no refs)
    std::vector<Cell> at;                     // its cell
    std::vector<core::Vec3> position;         // its estimated position in W
    std::vector<double> yaw;                  // its estimated heading in W (camera optical axis, rad), NaN unknown (D117)
    std::vector<double> flown_m;   // metres flown to T: the ground-truth path length (the rig's executed motion)
    std::vector<std::pair<int, std::pair<double, double>>> refused_goals;  // the mover's refusals (agent, goal xy)
    std::vector<mis::Directive> directives;   // active at T (recorded inputs; AGENTIC_7 §0b): region kinds reach here
};

/// What the runtime reads of a task option. Field names follow A1's planning::TaskCandidate
/// (cpp/planning/mission/task_candidate.hpp, feat/ph4-mission e2e538d), so A1's adapter is a field copy.
struct TaskOption {
    int64_t task_id = -1;                 // stable task identity (exactly-once); placeholder: the frontier cluster id
    std::pair<double, double> goal_xy{};  // world metres
    double goal_yaw = 0;                  // facing the task (radians)
    double path_m = 0, standoff_m = 0;
    std::pair<double, double> centroid_xy{};
    int cells = 0;                        // information proxy
    double risk = 0;                      // §23, in [0, 1]
    std::string risk_reason;
    // §21.4 energy-reserve return ordered by the planner's supervisor: never blocked by the §23 gate (it is taken,
    // logged and alerted). Maintainer, 2026-10-23: holding at the reserve strands the agent, and the return's risk is
    // 1.0 whenever the way in was flown but not observed (A1, NBV_4.3 §13).
    bool supervisor_return = false;
    double utility = 0;                   // the planner's own value (A1: U = G - sum lambda C); options are sorted by it
    mis::Candidate c;                     // PlaceholderPlanner's own (grid cell goal for Route); not read by the runtime
};

struct AgentOptions {
    std::vector<TaskOption> options;   // eligible, nearest first
    int classical = -1;                // the nearest option at or below the risk threshold, per agent
    int excluded_refused = 0, excluded_near = 0, excluded_directive = 0;
    std::vector<std::string> applied;  // ids of the directives that shaped these options
    cv::Mat plan;                      // the planning space the routes use
};

struct PlanOutput {
    explicit PlanOutput(int n_agents = 2) : agents(size_t(n_agents)), joint(size_t(n_agents), -1) {}
    mis::ReplanMaps maps;              // map summary: slab, egress route, frontiers (console, LLM images)
    std::vector<AgentOptions> agents;
    std::vector<int> joint;            // the allocated picks (exactly-once across agents)
    std::string trace_json = "{}";     // the planner's own record for the trace (which planner, allocation event, ...)
};

/// The task id of agent k's option i (-1 for hold): the exactly-once identity, for any planner.
inline int64_t TaskOf(const PlanOutput& p, int k, int i) { return i >= 0 ? p.agents[k].options[size_t(i)].task_id : -1; }

/// Agent k's best-ranked option whose task is not `taken` and whose risk is within the gate; -1 if none.
inline int OtherTask(const PlanOutput& p, int k, int64_t taken, double threshold) {
    const auto& v = p.agents[k].options;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i].risk <= threshold && v[i].task_id != taken) return int(i);
    }
    return -1;
}

class MissionPlanner {
public:
    virtual ~MissionPlanner() = default;
    virtual size_t Integrate(const KeyframeInput& k) = 0;
    virtual PlanOutput Plan(const PlanContext& ctx) = 0;
    /// Waypoints (x, y) for agent's option i, and the final yaw (facing the task), from the planner's own space.
    /// Empty = hold. One (NaN, NaN) waypoint = turn here to the yaw (D125). A NaN yaw = no final turn (D119).
    virtual std::pair<std::vector<std::pair<double, double>>, double> Route(const PlanOutput& p, int agent, int i,
                                                                           const Cell& from) const = 0;
};

struct PlaceholderParams {
    double min_goal_m = 1.0, escape_m = 0.5, max_standoff_m = 2.0, refused_radius_m = 0.5;
    double autonomous_threshold = 1.01, eps_cells = 1.0, max_command_m = 3.0;
};

/// True if every cell the 8-connected segment a-b visits is traversable.
inline bool ChordFree(const cv::Mat& trav, cv::Point a, cv::Point b) {
    cv::LineIterator it(trav, a, b, 8);
    for (int k = 0; k < it.count; ++k, ++it) {
        if (trav.at<uint8_t>(it.pos()) == 0) return false;
    }
    return true;
}

/// Appends the vertices that replace pts[i..j] after pts[i]: the chord itself if it is free, else the two halves split
/// at the cell farthest from the chord (Douglas-Peucker's split point), each refined the same way. Terminates: a chord
/// between consecutive A* cells is free (A* stepped between them).
inline void RefineChord(const cv::Mat& trav, const std::vector<cv::Point>& pts, size_t i, size_t j, std::vector<cv::Point>& out) {
    if (j <= i + 1 || ChordFree(trav, pts[i], pts[j])) {
        out.push_back(pts[j]);
        return;
    }
    const cv::Point2d a = pts[i], d = cv::Point2d(pts[j]) - a;
    const double n = std::max(1e-9, std::hypot(d.x, d.y));
    size_t m = i + 1;
    double best = -1;
    for (size_t k = i + 1; k < j; ++k) {
        const cv::Point2d v = cv::Point2d(pts[k]) - a;
        const double dist = std::abs(d.x * v.y - d.y * v.x) / n;
        if (dist > best) best = dist, m = k;
    }
    RefineChord(trav, pts, i, m, out);
    RefineChord(trav, pts, m, j, out);
}

/// The A* path to `goal`, simplified (approxPolyDP; a chord that crosses a non-traversable cell is split where it
/// deviates most and refined, never replaced by the raw cells), in W, without the start cell, cut at max_m. Empty if
/// no path. Before 2026-10-23 one failing chord replaced the WHOLE path by its 5 cm cells (a 2.6 m path became 45
/// waypoints and 85 s of turn-and-move segments in the first LLM run); with every chord free the output is unchanged.
inline std::vector<std::pair<double, double>> CommandPath(const mis::Briefing& b, const cv::Mat& trav, const Cell& from,
                                                          const Cell& goal, double eps_cells, double max_m) {
    const planning::GridPath p = planning::AStar(trav, from, {goal});
    if (!p.found || p.cells.size() < 2) return {};
    std::vector<cv::Point> pts, dp;
    for (const auto& c : p.cells) pts.emplace_back(c.c, c.r);
    cv::approxPolyDP(pts, dp, eps_cells, false);
    // approxPolyDP returns a subsequence of pts (in order; an A* path never repeats a cell): recover the indices
    std::vector<size_t> idx;
    for (size_t k = 0, q = 0; k < dp.size(); ++k) {
        while (q < pts.size() && pts[q] != dp[k]) ++q;
        idx.push_back(q);
    }
    std::vector<cv::Point> simp{pts.front()};
    for (size_t k = 1; k < idx.size(); ++k) RefineChord(trav, pts, idx[k - 1], idx[k], simp);
    std::vector<std::pair<double, double>> out;
    auto [px, py] = b.CellCentre(Cell{simp[0].y, simp[0].x});
    double left = max_m;
    for (size_t i = 1; i < simp.size() && left > 1e-9; ++i) {
        auto [x, y] = b.CellCentre(Cell{simp[i].y, simp[i].x});
        const double d = std::hypot(x - px, y - py);
        if (d > left) {
            x = px + (x - px) * left / d;
            y = py + (y - py) * left / d;
        }
        out.emplace_back(x, y);
        left -= std::min(d, left);
        px = x, py = y;
    }
    return out;
}

/// PLACEHOLDER (ablation baseline): 1.8 frontiers, standoff goals (D153b), bottleneck risk (D158), greedy
/// exactly-once allocation. Its parameters are placeholder knobs, not part of the seam.
class PlaceholderPlanner : public MissionPlanner {
public:
    PlaceholderPlanner(const mis::MissionConfig& mc, const mis::Briefing& b, const PlaceholderParams& p)
        : mc_(mc), b_(b), p_(p), map_(mc, b) {}

    size_t Integrate(const KeyframeInput& k) override {
        return map_.Integrate(k.T_W_C, k.Z, k.S, k.fx, k.fy, k.cx, k.cy);
    }

    PlanOutput Plan(const PlanContext& ctx) override {
        PlanOutput out(ctx.n());
        out.maps = map_.Replan();
        const double rho_a = mc_.rho_a, rho_r = mc_.rho_r;
        for (int k = 0; k < ctx.n(); ++k) {
            if (!ctx.active[k]) continue;
            AgentOptions& ao = out.agents[k];
            const Cell ac = ctx.at[k];
            ao.plan = map_.PlanSpace(out.maps, ac, p_.escape_m);
            std::vector<mis::Candidate> cands;
            for (const auto& c : map_.StandoffCandidates(out.maps, ao.plan, ac, p_.max_standoff_m)) {
                const auto [gx, gy] = b_.CellCentre(c.goal);
                bool refused = false;
                for (const auto& [ra, rg] : ctx.refused_goals) {
                    refused = refused || (ra == k && std::hypot(rg.first - gx, rg.second - gy) < p_.refused_radius_m);
                }
                if (refused) ++ao.excluded_refused;
                else if (c.path_cells * b_.gs.cell < p_.min_goal_m) ++ao.excluded_near;
                else cands.push_back(c);
            }
            // Directives (operator / confirmed LLM proposals): keep_out removes, assign / explore restrict to their
            // region when it holds an option, priority_region divides the path cost by its weight (task value x w).
            std::vector<double> cost;
            {
                std::vector<mis::Candidate> kept;
                for (const auto& c : cands) {
                    const auto [gx, gy] = b_.CellCentre(c.goal);
                    bool out_ = false;
                    for (const auto& d : ctx.directives) {
                        if (d.kind == "keep_out" && (d.agent < 0 || d.agent == k) && d.Covers(gx, gy)) out_ = true;
                    }
                    if (out_) ++ao.excluded_directive;
                    else kept.push_back(c);
                }
                for (const auto& d : ctx.directives) {
                    if (d.kind == "keep_out" && (d.agent < 0 || d.agent == k)) ao.applied.push_back(d.id);
                    if ((d.kind == "assign" || d.kind == "explore") && d.agent == k) {
                        std::vector<mis::Candidate> in;
                        for (const auto& c : kept) {
                            const auto [gx, gy] = b_.CellCentre(c.goal);
                            if (d.Covers(gx, gy)) in.push_back(c);
                        }
                        if (!in.empty()) {
                            ao.excluded_directive += int(kept.size() - in.size());
                            kept = in;
                            ao.applied.push_back(d.id);
                        }
                    }
                }
                std::vector<std::pair<double, size_t>> order;
                for (size_t i = 0; i < kept.size(); ++i) {
                    const auto [gx, gy] = b_.CellCentre(kept[i].goal);
                    double w = 1.0;
                    for (const auto& d : ctx.directives) {
                        if (d.kind == "priority_region" && (d.agent < 0 || d.agent == k) && d.Covers(gx, gy)) {
                            w = std::max(w, d.weight);
                            if (std::find(ao.applied.begin(), ao.applied.end(), d.id) == ao.applied.end()) ao.applied.push_back(d.id);
                        }
                    }
                    order.emplace_back(kept[i].path_cells / w, i);
                }
                std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                cands.clear();
                for (const auto& [cst, i] : order) cands.push_back(kept[i]);
            }
            // §23 risk (D158): the widest way to ARRIVE within escape_m of the goal (the standoff goal itself sits at
            // rho_agent from unknown space by construction), against what a responder needs.
            const cv::Mat B = map_.Bottleneck(out.maps, ao.plan, ac, p_.escape_m);
            const int rad = int(std::ceil(p_.escape_m / b_.gs.cell));
            for (const auto& c : cands) {
                double bott = -1;
                for (int dy = -rad; dy <= rad; ++dy) {
                    for (int dx = -rad; dx <= rad; ++dx) {
                        const int y = c.goal.r + dy, x = c.goal.c + dx;
                        if (dx * dx + dy * dy > rad * rad || y < 0 || x < 0 || y >= B.rows || x >= B.cols) continue;
                        bott = std::max(bott, B.at<double>(y, x));
                    }
                }
                TaskOption o;
                o.c = c;
                o.task_id = c.cluster;
                o.goal_xy = b_.CellCentre(c.goal);
                o.goal_yaw = std::atan2(c.centroid_y - o.goal_xy.second, c.centroid_x - o.goal_xy.first);
                o.path_m = c.path_cells * b_.gs.cell;
                o.standoff_m = c.standoff_m;
                o.centroid_xy = {c.centroid_x, c.centroid_y};
                o.cells = c.cells;
                o.risk = std::isfinite(bott) ? std::clamp((rho_r - bott) / (rho_r - rho_a), 0.0, 1.0) : 0.0;
                std::ostringstream why;
                why.precision(3);
                if (std::isfinite(bott)) why << "narrowest unavoidable gap " << bott << " m; a responder needs " << rho_r << " m";
                else why << "goal inside the agent's own escape disc";
                o.risk_reason = why.str();
                if (ao.classical < 0 && o.risk <= p_.autonomous_threshold) ao.classical = int(ao.options.size());
                ao.options.push_back(o);
            }
        }
        // Allocation (placeholder for A1's CBBA, 4.4): greedy, agent 0 first, exactly-once (ALGORITHMS line 1165).
        // Exactly-once across agents, ascending index (P8 D213: N agents; at N=2 this is the old pairwise rule:
        // agent 1 moves to its best other cluster when it collides with agent 0's pick).
        for (int k = 0; k < ctx.n(); ++k) out.joint[size_t(k)] = out.agents[size_t(k)].classical;
        std::set<int64_t> taken;
        for (int k = 0; k < ctx.n(); ++k) {
            if (!ctx.active[size_t(k)] || out.joint[size_t(k)] < 0) continue;
            int64_t cl = ClusterOf(out, k, out.joint[size_t(k)]);
            if (taken.count(cl)) {
                const int alt = OtherCluster(out, k, cl);
                if (alt >= 0) out.joint[size_t(k)] = alt, cl = ClusterOf(out, k, alt);
            }
            taken.insert(cl);
        }
        return out;
    }

    std::pair<std::vector<std::pair<double, double>>, double> Route(const PlanOutput& p, int agent, int i,
                                                                   const Cell& from) const override {
        const auto& o = p.agents[agent].options[size_t(i)];
        auto path = CommandPath(b_, p.agents[agent].plan, from, o.c.goal, p_.eps_cells, p_.max_command_m);
        double yaw = 0;
        if (!path.empty()) yaw = std::atan2(o.c.centroid_y - path.back().second, o.c.centroid_x - path.back().first);
        return {path, yaw};
    }

    static int64_t ClusterOf(const PlanOutput& p, int k, int i) { return TaskOf(p, k, i); }
    /// The nearest option of agent k at or below the threshold whose task is not `taken`; -1 if none (A1: OtherTask).
    int OtherCluster(const PlanOutput& p, int k, int64_t taken) const { return OtherTask(p, k, taken, p_.autonomous_threshold); }

private:
    mis::MissionConfig mc_;
    mis::Briefing b_;
    PlaceholderParams p_;
    mis::FrontierMap map_;
};

}  // namespace ariadne::runtime::live
