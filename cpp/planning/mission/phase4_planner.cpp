#include "planning/mission/phase4_planner.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace ariadne::planning {

bool RegionDirective::Covers(double px, double py) const {
    bool in = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const auto [xi, yi] = polygon[i];
        const auto [xj, yj] = polygon[j];
        if ((yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) in = !in;
    }
    return in;
}

double RegionDirective::Distance(double px, double py) const {
    if (Covers(px, py)) return 0;
    double best = 1e300;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const double ax = polygon[j].first, ay = polygon[j].second, bx = polygon[i].first, by = polygon[i].second;
        const double dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
        const double t = L2 > 0 ? std::clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0) : 0.0;
        best = std::min(best, std::hypot(px - (ax + t * dx), py - (ay + t * dy)));
    }
    return best;
}

void MaskKeepOut(int agent, const std::vector<RegionDirective>& dirs, const GridFrame& g, cv::Mat* plan) {
    for (const auto& d : dirs) {
        if (d.kind != "keep_out" || (d.agent >= 0 && d.agent != agent)) continue;
        std::vector<cv::Point> poly;   // grid (col, row) of each vertex: row 0 = south, as every layer here
        for (const auto& [x, y] : d.polygon)
            poly.emplace_back(int(std::lround((x - g.ox) / g.cell - 0.5)), int(std::lround((y - g.oy) / g.cell - 0.5)));
        cv::fillPoly(*plan, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(0));
    }
}

void ApplyDirectives(int agent, const std::vector<RegionDirective>& dirs,
                     const std::vector<std::pair<double, double>>& refused, double refused_radius_m, double observe_m,
                     Phase4AgentPlan* ap) {
    auto& opts = ap->nbv.options;
    if (ap->nbv.reserve_rule) return;   // the hard rule's {return, hold} is outside every directive (§21.4)
    std::vector<TaskCandidate> kept;
    std::vector<char> used(dirs.size(), 0);
    // assign / explore restrict only when at least one option's goal lies inside (B's semantics): an agent is never
    // left without a task because its region holds no candidate yet.
    std::vector<char> restrict_ok(dirs.size(), 0);
    for (size_t k = 0; k < dirs.size(); ++k) {
        const auto& d = dirs[k];
        if ((d.kind != "assign" && d.kind != "explore") || (d.agent >= 0 && d.agent != agent)) continue;
        for (const auto& c : opts) restrict_ok[k] |= d.Covers(c.goal_xy.first, c.goal_xy.second);
    }
    for (auto c : opts) {
        const auto [x, y] = c.goal_xy;
        bool drop = false;
        for (const auto& r : refused) {
            if (std::hypot(x - r.first, y - r.second) <= refused_radius_m) {
                drop = true;
                ++ap->excluded_refused;
                break;
            }
        }
        if (drop) continue;
        for (size_t k = 0; k < dirs.size() && !drop; ++k) {
            const auto& d = dirs[k];
            if (d.agent >= 0 && d.agent != agent) continue;
            if (d.kind == "keep_out" && d.Covers(x, y)) drop = true, used[k] = 1;
            if ((d.kind == "assign" || d.kind == "explore") && restrict_ok[k] && !d.Covers(x, y)) drop = true, used[k] = 1;
            const auto [ox, oy] = c.centroid_xy;
            if (d.kind == "priority_region" && (d.Covers(x, y) || d.Distance(ox, oy) <= observe_m)) {
                c.utility += (d.weight - 1.0) * c.gain;   // the task value G scales; the costs do not
                c.gain *= d.weight;
                used[k] = 1;
            }
        }
        if (drop) {
            ++ap->excluded_directive;
            continue;
        }
        kept.push_back(c);
    }
    std::stable_sort(kept.begin(), kept.end(), [](const TaskCandidate& a, const TaskCandidate& b) {
        if (a.utility != b.utility) return a.utility > b.utility;
        if (a.path_m != b.path_m) return a.path_m < b.path_m;
        return a.task_id < b.task_id;
    });
    opts = std::move(kept);
    for (size_t k = 0; k < dirs.size(); ++k)
        if (used[k]) ap->applied.push_back(dirs[k].id);
}

void JointPick(Phase4Plan* p) {
    const auto& o0 = p->agents[0].nbv.options;
    const auto& o1 = p->agents[1].nbv.options;
    const double u0 = o0.empty() ? -1e300 : o0[0].utility, u1 = o1.empty() ? -1e300 : o1[0].utility;
    const int first = u1 > u0 ? 1 : 0, second = 1 - first;
    const auto& of = p->agents[first].nbv.options;
    if (!of.empty()) p->joint[first] = 0;
    const int64_t taken = of.empty() ? -1 : of[0].task_id;
    p->joint[second] = OtherTask(p->agents[second].nbv.options, taken, 1e300);
}

std::vector<Cell> StringPull(const cv::Mat& plan, const std::vector<Cell>& path) {
    std::vector<Cell> out;
    if (path.empty()) return out;
    auto Clear = [&](const Cell& a, const Cell& b) {
        cv::LineIterator it(plan, cv::Point(a.c, a.r), cv::Point(b.c, b.r), 4);   // 4-connected: no corner cut
        for (int k = 0; k < it.count; ++k, ++it)
            if (plan.at<uint8_t>(it.pos()) == 0) return false;
        return true;
    };
    size_t i = 0;
    out.push_back(path[0]);
    while (i + 1 < path.size()) {
        size_t j = i + 1;   // adjacent path cells are always mutually visible (A*'s corner rule)
        for (size_t k = path.size() - 1; k > i + 1; --k)
            if (Clear(path[i], path[k])) {
                j = k;
                break;
            }
        out.push_back(path[j]);
        i = j;
    }
    return out;
}

Phase4Planner::Phase4Planner(const Phase4Briefing& b, const Phase4Params& p)
    : b_(b), p_(p), map_(p.tsdf), esdf_(p.tsdf.voxel, p.trav.responder.rho + 2 * p.tsdf.voxel), fr_(p.frontiers),
      egress_(b.target, b.entrances, b.grid.cell) {}

size_t Phase4Planner::Integrate(int agent, int64_t /*t_ns*/, const DepthFrame& f) {
    const std::vector<Idx3> dirty = map_.Integrate(f);
    if (esdf_.Empty()) esdf_.Rebuild(map_);
    else esdf_.Update(map_, dirty);
    if (!dirty.empty()) {
        Idx3 lo{1 << 30, 1 << 30, 1 << 30}, hi{-(1 << 30), -(1 << 30), -(1 << 30)};
        for (const auto& bk : dirty) {
            lo = {std::min(lo.x, bk.x * kBlockSide), std::min(lo.y, bk.y * kBlockSide), std::min(lo.z, bk.z * kBlockSide)};
            hi = {std::max(hi.x, bk.x * kBlockSide + kBlockSide - 1), std::max(hi.y, bk.y * kBlockSide + kBlockSide - 1),
                  std::max(hi.z, bk.z * kBlockSide + kBlockSide - 1)};
        }
        if (!detected_) fr_.Detect(map_), detected_ = true;
        else fr_.Update(map_, lo, hi);
    }
    track_[agent].push_back(f.T_W_C.t);
    return dirty.size();
}

Cell Phase4Planner::CellOf(const core::Vec3& p) const {
    return Cell{int(std::floor((p.y() - b_.grid.oy) / b_.grid.cell)), int(std::floor((p.x() - b_.grid.ox) / b_.grid.cell))};
}

cv::Mat Phase4Planner::PlanSpace(const TraversabilityMaps& m, const Cell& at) const {
    cv::Mat plan = m.trav_a.clone();
    const double esc = p_.nbv.escape_m / b_.grid.cell;
    for (int r = std::max(0, at.r - int(esc) - 1); r <= std::min(plan.rows - 1, at.r + int(esc) + 1); ++r)
        for (int c = std::max(0, at.c - int(esc) - 1); c <= std::min(plan.cols - 1, at.c + int(esc) + 1); ++c)
            if (std::hypot(r - at.r, c - at.c) <= esc && m.slab.at<uint8_t>(r, c) != kOccupied) plan.at<uint8_t>(r, c) = 1;
    return plan;
}

Phase4Plan Phase4Planner::Plan(double t_s, const Phase4AgentInput agents[2], const std::vector<RegionDirective>& dirs) {
    Phase4Plan out;
    out.t_s = t_s;
    out.maps = ComputeTraversability(map_, esdf_, b_.grid, p_.trav);
    out.support = BodySupport(map_, b_.grid, p_.trav.responder, p_.support_band_lo, p_.support_band_hi);
    EgressWeights w = p_.egress;
    if (!p_.risk_calibrated) w.w_r = 0;   // invariant: the [PROXY] enters only after D112's bar
    const cv::Mat pen = EgressPenalty(out.maps.clear_r, out.support, out.maps.risk, w);
    out.egress = egress_.Replan(out.maps.trav_r, pen, out.support, out.maps.risk);
    const CoarseLayer coarse = BuildCoarse(map_);
    for (int a = 0; a < 2; ++a) {
        if (!agents[a].active) continue;
        NbvInputs ni;
        ni.map = &map_, ni.frontiers = &fr_, ni.coarse = &coarse, ni.trav = &out.maps, ni.grid = b_.grid;
        ni.route = &out.egress, ni.route_support = &out.support, ni.responder = p_.trav.responder;
        ni.entrances = b_.entrances, ni.risk_calibrated = p_.risk_calibrated, ni.t_s = t_s;
        for (size_t i = 0; i < track_[1 - a].size(); ++i) ni.peers.push_back({1 - a, int64_t(i), track_[1 - a][i]});
        ni.bounds.lo = core::Vec3(b_.grid.ox, b_.grid.oy, 0.0);
        ni.bounds.hi = core::Vec3(b_.grid.ox + b_.grid.cols * b_.grid.cell, b_.grid.oy + b_.grid.rows * b_.grid.cell, b_.z_max_m);
        // keep_out masks the agent's plan space before NBV, so neither a view nor a path to it can use the region.
        cv::Mat ones(out.maps.trav_a.size(), CV_8U, cv::Scalar(1));
        MaskKeepOut(a, dirs, b_.grid, &ones);
        out.agents[a].keep_out = ones == 0;
        TraversabilityMaps masked = out.maps;
        masked.trav_a = out.maps.trav_a.clone();
        masked.trav_a.setTo(0, out.agents[a].keep_out);
        ni.trav = &masked;
        AgentState st;
        st.position = agents[a].position, st.e_used_m = agents[a].flown_m, st.own_keyframes = track_[a];
        out.agents[a].nbv = PlanNbv(ni, a, st, p_.nbv);
        out.agents[a].plan = PlanSpace(out.maps, CellOf(agents[a].position));
        out.agents[a].plan.setTo(0, out.agents[a].keep_out);   // the escape disc never re-opens a keep_out
        ApplyDirectives(a, dirs, agents[a].refused_goals, p_.refused_radius_m, p_.nbv.frustum.max_range_m, &out.agents[a]);
    }
    JointPick(&out);
    return out;
}

std::pair<std::vector<std::pair<double, double>>, double> Phase4Planner::Route(const Phase4Plan& p, int agent, int i,
                                                                              const Cell& from) const {
    const auto& opts = p.agents[agent].nbv.options;
    if (i < 0 || size_t(i) >= opts.size()) return {{}, 0.0};
    const TaskCandidate& o = opts[size_t(i)];
    const GridFrame& g = b_.grid;
    const Cell goal{int(std::floor((o.goal_xy.second - g.oy) / g.cell)), int(std::floor((o.goal_xy.first - g.ox) / g.cell))};
    cv::Mat plan = PlanSpace(p.maps, from);   // trav_a + the escape disc at `from` (D158)
    if (!p.agents[agent].keep_out.empty()) plan.setTo(0, p.agents[agent].keep_out);
    const GridPath gp = AStar(plan, from, {goal});
    if (!gp.found || gp.cells.size() < 2) return {{}, o.goal_yaw};
    const std::vector<Cell> keep = StringPull(plan, gp.cells);
    std::vector<std::pair<double, double>> wp;
    double px = g.ox + (keep[0].c + 0.5) * g.cell, py = g.oy + (keep[0].r + 0.5) * g.cell, left = p_.max_command_m;
    for (size_t k = 1; k < keep.size() && left > 1e-9; ++k) {
        double x = g.ox + (keep[k].c + 0.5) * g.cell, y = g.oy + (keep[k].r + 0.5) * g.cell;
        const double d = std::hypot(x - px, y - py);
        if (d > left) x = px + (x - px) * left / d, y = py + (y - py) * left / d;   // on a clear chord: still clear
        wp.emplace_back(x, y);
        left -= std::min(d, left);
        px = x, py = y;
    }
    return {wp, o.goal_yaw};
}

}  // namespace ariadne::planning
