#include "planning/mission/nbv.hpp"

#include "core/random/philox.hpp"
#include "planning/volumetric/occupancy_grid.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>

namespace ariadne::planning {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

uint64_t Mix(uint64_t x) {   // splitmix64 finaliser: a stable task id from (generator, anchor)
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

int64_t TaskId(TaskSource s, int64_t anchor) {
    return int64_t(Mix(uint64_t(anchor) ^ (uint64_t(uint8_t(s)) << 56)) >> 1);   // non-negative
}

double Hb(double p) { return p <= 0 || p >= 1 ? 0.0 : -(p * std::log2(p) + (1 - p) * std::log2(1 - p)); }

struct Target {
    TaskSource src = TaskSource::kFrontier;
    core::Vec3 p = core::Vec3::Zero();
    int64_t anchor = 0;
    int size = 0;
    double gain_conn = 0, gain_alias = 0;   // producer-supplied terms (C, D)
};

Cell CellOf(const GridFrame& g, double x, double y) {
    return Cell{int(std::floor((y - g.oy) / g.cell)), int(std::floor((x - g.ox) / g.cell))};
}

bool In(const cv::Mat& m, const Cell& c) { return c.r >= 0 && c.c >= 0 && c.r < m.rows && c.c < m.cols; }

// Generator A: frontier clusters, split along the first principal axis while the AABB extent exceeds `ext` (FUEL).
void SplitCluster(const TsdfMap& map, std::vector<Idx3> vox, double ext, int depth, std::vector<Target>* out) {
    if (vox.empty()) return;
    Idx3 lo = vox[0], hi = vox[0];
    for (const auto& v : vox) {
        lo = {std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
        hi = {std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
    }
    const double e = map.Voxel() * double(std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z}) + 1);
    if (e > ext && vox.size() >= 2 && depth < 8) {
        cv::Mat pts(int(vox.size()), 3, CV_64F);
        for (size_t i = 0; i < vox.size(); ++i) {
            pts.at<double>(int(i), 0) = vox[i].x, pts.at<double>(int(i), 1) = vox[i].y, pts.at<double>(int(i), 2) = vox[i].z;
        }
        const cv::PCA pca(pts, cv::Mat(), cv::PCA::DATA_AS_ROW, 1);
        std::vector<Idx3> a, b;
        for (size_t i = 0; i < vox.size(); ++i) {
            const cv::Mat proj = pca.project(pts.row(int(i)));
            (proj.at<double>(0, 0) < 0 ? a : b).push_back(vox[i]);
        }
        if (!a.empty() && !b.empty()) {
            SplitCluster(map, std::move(a), ext, depth + 1, out);
            SplitCluster(map, std::move(b), ext, depth + 1, out);
            return;
        }
    }
    std::sort(vox.begin(), vox.end());
    core::Vec3 c = core::Vec3::Zero();
    for (const auto& v : vox) c += map.Centre(v);
    Target t;
    t.src = TaskSource::kFrontier, t.p = c / double(vox.size()), t.anchor = Pack(vox[0]), t.size = int(vox.size());
    out->push_back(t);
}

// The route's body envelope (D113's support definition): voxels within rho of each route column's capsule axis.
std::unordered_set<int64_t> RouteEnvelope(const TsdfMap& map, const GridFrame& g, const BodyCapsule& b,
                                          const std::vector<Cell>& cells) {
    std::unordered_set<int64_t> env;
    const double vox = map.Voxel();
    const int cx0 = int(std::lround(g.ox / vox)), cy0 = int(std::lround(g.oy / vox)), R = int(std::ceil(b.rho / vox));
    const int z0 = int(std::floor((b.axis_lo - b.rho) / vox)), z1 = int(std::floor((b.axis_hi + b.rho) / vox));
    for (const auto& q : cells)
        for (int z = z0; z <= z1; ++z) {
            const double zc = (z + 0.5) * vox;
            const double dz = zc < b.axis_lo ? b.axis_lo - zc : (zc > b.axis_hi ? zc - b.axis_hi : 0.0);
            const double r2 = b.rho * b.rho - dz * dz;
            if (r2 < 0) continue;
            for (int i = -R; i <= R; ++i)
                for (int j = -R; j <= R; ++j)
                    if ((i * i + j * j) * vox * vox <= r2 + 1e-12) env.insert(Pack({cx0 + q.c + j, cy0 + q.r + i, z}));
        }
    return env;
}

}  // namespace

double VoxelEntropy(const TsdfVoxel* v, double w_min) {
    if (!v || v->w <= w_min) return 1.0;
    return Hb(0.5 * std::exp(-double(v->w)));   // free 0.5 e^-w and occupied 1 - 0.5 e^-w have the same entropy
}

IgCounts RaycastIg(const TsdfMap& map, const core::Vec3& eye, double yaw, const RigFrustum& f,
                   const std::unordered_set<int64_t>* route_env) {
    IgCounts ig;
    const double vox = map.Voxel();
    const core::Vec3 fw(std::cos(yaw), std::sin(yaw), 0), right(std::sin(yaw), -std::cos(yaw), 0), down(0, 0, -1);
    std::unordered_set<int64_t> seen;
    for (int i = 0; i < f.grid; ++i)
        for (int j = 0; j < f.grid; ++j) {
            const double u = (j + 0.5) * f.width / f.grid, v = (i + 0.5) * f.height / f.grid;
            const core::Vec3 d = (fw + ((u - f.cx) / f.fx) * right + ((v - f.cy) / f.fy) * down).normalized();
            const core::Vec3 a = eye / vox, b = (eye + d * f.max_range_m) / vox;
            bool stop = false;
            TraverseVoxels(a, b, [&](int x, int y, int z) {
                if (stop) return;
                const Idx3 q{x, y, z};
                const TsdfVoxel* t = map.Find(q);
                const uint8_t st = map.State(q);
                if (seen.insert(Pack(q)).second) {
                    ++ig.voxels;
                    ig.entropy += VoxelEntropy(t, map.Params().w_min);
                    if (st == kUnknown) {
                        ++ig.coverage;
                        if (route_env && route_env->count(Pack(q))) ++ig.egress;
                    }
                }
                if (st == kOccupied) stop = true;
            });
        }
    return ig;
}

std::vector<View> SampleViews(const TsdfMap& map, const GridFrame& g, const cv::Mat& trav_a, const core::Vec3& target,
                              const SamplerParams& sp, const RigFrustum& f, const std::unordered_set<int64_t>* route_env) {
    std::vector<View> views;
    const double vox = map.Voxel();
    const Idx3 tv = map.ToVoxel(target);
    for (double r : sp.radii)
        for (int k = 0; k < sp.azimuths; ++k) {
            const double az = 2.0 * M_PI * k / sp.azimuths;
            const core::Vec3 eye(target.x() + r * std::cos(az), target.y() + r * std::sin(az), sp.z);
            const Cell c = CellOf(g, eye.x(), eye.y());
            if (!In(trav_a, c) || !trav_a.at<uint8_t>(c.r, c.c)) continue;
            bool blocked = false;
            TraverseVoxels(eye / vox, target / vox, [&](int x, int y, int z) {
                if (blocked) return;
                if (std::max({std::abs(x - tv.x), std::abs(y - tv.y), std::abs(z - tv.z)}) <= 1) return;
                blocked = map.State({x, y, z}) == kOccupied;
            });
            if (blocked) continue;
            View v;
            v.eye = eye, v.cell = c, v.yaw = std::atan2(target.y() - eye.y(), target.x() - eye.x());
            v.ig = RaycastIg(map, eye, v.yaw, f, route_env);
            views.push_back(v);
        }
    std::stable_sort(views.begin(), views.end(), [](const View& a, const View& b) { return a.ig.coverage > b.ig.coverage; });
    if (int(views.size()) > sp.keep) views.resize(size_t(sp.keep));
    return views;
}

double P90Scale(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const double p90 = v[size_t(std::floor(0.9 * double(v.size() - 1)))];
    if (p90 > 0) return p90;
    return v.back() > 0 ? v.back() : 0.0;
}

cv::Mat Bottleneck(const cv::Mat& plan, const cv::Mat& clr, const Cell& from, double escape_cells) {
    const int H = plan.rows, W = plan.cols;
    cv::Mat B(H, W, CV_64F, cv::Scalar(-1.0));
    auto ok = [&](int y, int x) { return y >= 0 && x >= 0 && y < H && x < W && plan.at<uint8_t>(y, x); };
    if (!ok(from.r, from.c)) return B;
    auto c = [&](int y, int x) { return std::hypot(y - from.r, x - from.c) <= escape_cells ? kInf : double(clr.at<float>(y, x)); };
    using Node = std::pair<double, int64_t>;
    std::priority_queue<Node> open;   // max-heap on the bottleneck
    B.at<double>(from.r, from.c) = kInf;
    open.emplace(kInf, int64_t(from.r) * W + from.c);
    static const int dr[8] = {0, 0, 1, -1, 1, 1, -1, -1}, dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    while (!open.empty()) {
        const auto [g, u] = open.top();
        open.pop();
        const int uy = int(u / W), ux = int(u % W);
        if (g < B.at<double>(uy, ux)) continue;
        for (int k = 0; k < 8; ++k) {
            const int y = uy + dr[k], x = ux + dc[k];
            if (!ok(y, x)) continue;
            if (k >= 4 && (!ok(uy + dr[k], ux) || !ok(uy, ux + dc[k]))) continue;
            const double nb = std::min(g, c(y, x));
            if (nb > B.at<double>(y, x)) {
                B.at<double>(y, x) = nb;
                open.emplace(nb, int64_t(y) * W + x);
            }
        }
    }
    return B;
}

NbvPlan PlanNbv(const NbvInputs& in, int agent, const AgentState& st, const NbvParams& p) {
    if (!in.map || !in.frontiers || !in.coarse || !in.trav) throw std::invalid_argument("PlanNbv: map inputs");
    const TsdfMap& map = *in.map;
    const GridFrame& g = in.grid;
    const TraversabilityMaps& tm = *in.trav;
    NbvPlan out;

    // Plan space (D158): trav_a, plus the agent's own escape disc where the slab is not occupied.
    const Cell me = CellOf(g, st.position.x(), st.position.y());
    cv::Mat plan = tm.trav_a.clone();
    const double esc = p.escape_m / g.cell;
    for (int r = std::max(0, me.r - int(esc) - 1); r <= std::min(plan.rows - 1, me.r + int(esc) + 1); ++r)
        for (int c = std::max(0, me.c - int(esc) - 1); c <= std::min(plan.cols - 1, me.c + int(esc) + 1); ++c)
            if (std::hypot(r - me.r, c - me.c) <= esc && tm.slab.at<uint8_t>(r, c) != kOccupied) plan.at<uint8_t>(r, c) = 1;
    const cv::Mat dist = GridDistances(plan, me);
    const cv::Mat bott = Bottleneck(plan, tm.clear_r, me, esc);

    // lambda schedule (D114): the path-proportional costs counted once.
    const double fe = std::min(1.0, st.e_used_m / p.e_total_m), ft = std::min(1.0, in.t_s / p.mission_T_s);
    out.lambda_path = (1.0 + (1.0 + fe * fe * fe) + (1.0 + ft * ft * ft)) / 3.0;

    auto Risk = [&](const Cell& c, TaskCandidate* t) {
        const double b = bott.at<double>(c.r, c.c);
        t->risk_geometric = std::isinf(b) ? 0.0 : std::clamp((p.rho_r - b) / (p.rho_r - p.rho_a), 0.0, 1.0);
        t->risk_structural = in.risk_calibrated ? double(tm.risk.at<float>(c.r, c.c)) : 0.0;
        t->risk = std::max(t->risk_geometric, t->risk_structural);
        char buf[200];
        std::snprintf(buf, sizeof buf, "bottleneck %.2f m (D158); structural %s", std::isinf(b) ? 99.0 : b,
                      in.risk_calibrated ? "[PROXY] calibrated (D112)" : "[PROXY] not calibrated (D112): 0");
        t->risk_reason = buf;
    };

    // --- the hard rule, outside the utility (§21.4) ----------------------------------------------------------------
    int best_e = -1;
    double d_ret = kInf;
    for (size_t i = 0; i < in.entrances.size(); ++i) {
        const Cell& e = in.entrances[i];
        if (In(dist, e) && dist.at<double>(e.r, e.c) < d_ret) d_ret = dist.at<double>(e.r, e.c), best_e = int(i);
    }
    const double e_return = p.e_per_m * d_ret * g.cell;
    if (p.e_total_m - st.e_used_m < p.reserve * e_return) {
        out.reserve_rule = true;
        char buf[200];
        std::snprintf(buf, sizeof buf, "energy reserve: E_remaining %.2f < %.2f x E(return) %.2f -> {return, hold}",
                      p.e_total_m - st.e_used_m, p.reserve, e_return);
        out.supervisor_event = buf;
        if (best_e >= 0) {
            TaskCandidate t;
            const Cell& e = in.entrances[size_t(best_e)];
            t.source = TaskSource::kReturn, t.task_id = TaskId(TaskSource::kReturn, best_e);
            t.goal_xy = {g.ox + (e.c + 0.5) * g.cell, g.oy + (e.r + 0.5) * g.cell}, t.centroid_xy = t.goal_xy;
            t.path_m = d_ret * g.cell;
            Risk(e, &t);
            t.utility = 1;   // the supervisor's order, not a score: return before hold
            out.options.push_back(t);
        }
        TaskCandidate h;
        h.source = TaskSource::kHold, h.task_id = TaskId(TaskSource::kHold, agent);
        h.goal_xy = h.centroid_xy = {st.position.x(), st.position.y()};
        h.risk_reason = "hold";
        out.options.push_back(h);
        return out;
    }

    // --- targets -----------------------------------------------------------------------------------------------------
    std::vector<Target> targets;
    for (const auto& k : in.frontiers->Clusters()) SplitCluster(map, k.voxels, p.split_extent_m, 0, &targets);
    out.targets[0] = int(targets.size());
    {   // B: surface coarse cells in the bottom decile of mean weight (rank-based)
        std::vector<const CoarseCell*> surf;
        for (const auto& c : in.coarse->cells)
            if (c.occupied > 0 && c.known > 0) surf.push_back(&c);
        std::stable_sort(surf.begin(), surf.end(), [](const CoarseCell* a, const CoarseCell* b) { return a->mean_w < b->mean_w; });
        const size_t n = surf.empty() ? 0 : std::max<size_t>(1, surf.size() / 10);
        for (size_t i = 0; i < n; ++i) {
            const CoarseCell& c = *surf[i];
            Target t;
            t.src = TaskSource::kUncertainty;
            t.p = core::Vec3((c.c.x + 0.5) * in.coarse->cell_m, (c.c.y + 0.5) * in.coarse->cell_m, (c.c.z + 0.5) * in.coarse->cell_m);
            t.anchor = Pack(c.c), t.size = 1;
            targets.push_back(t);
        }
        out.targets[1] = int(n);
    }
    {   // C: peers' keyframes near this agent's trajectory but not visited
        int n = 0;
        for (const auto& pk : in.peers) {
            if (pk.agent == agent) continue;
            double dmin = kInf;
            for (const auto& o : st.own_keyframes) dmin = std::min(dmin, (o - pk.p).head<2>().norm());
            if (!(dmin <= p.peer_near_m) || dmin < p.peer_seen_m) continue;
            Target t;
            t.src = TaskSource::kConnectivity, t.p = pk.p, t.anchor = pk.kf_id * 4 + pk.agent, t.size = 1;
            t.gain_conn = in.connectivity ? in.connectivity->Gain(agent, pk.kf_id) : 0.0;
            targets.push_back(t);
            ++n;
        }
        out.targets[2] = n;
        if (!in.connectivity) out.notes.push_back("pose_connectivity = 0: no ConnectivityOracle (3.4 selector not merged)");
    }
    if (in.alias) {   // D
        for (const auto& a : in.alias->Places()) {
            Target t;
            t.src = TaskSource::kAlias, t.p = a.p, t.anchor = a.id, t.size = 1, t.gain_alias = a.value;
            targets.push_back(t);
            ++out.targets[3];
        }
    } else {
        out.notes.push_back("loop_disambiguation = 0: no AliasSource (no §17.3 producer)");
    }
    std::unordered_set<int64_t> env;
    if (in.route && in.route->found && in.route_support) {   // E: runs of route cells with support < 1
        env = RouteEnvelope(map, g, in.responder, in.route->cells);
        const auto& rc = in.route->cells;
        for (size_t i = 0; i < rc.size();) {
            if (in.route_support->at<float>(rc[i].r, rc[i].c) >= 1.f) { ++i; continue; }
            size_t j = i, arg = i;
            for (; j < rc.size() && in.route_support->at<float>(rc[j].r, rc[j].c) < 1.f; ++j)
                if (in.route_support->at<float>(rc[j].r, rc[j].c) < in.route_support->at<float>(rc[arg].r, rc[arg].c)) arg = j;
            Target t;
            t.src = TaskSource::kEgress;
            t.p = core::Vec3(g.ox + (rc[arg].c + 0.5) * g.cell, g.oy + (rc[arg].r + 0.5) * g.cell,
                             0.5 * (in.responder.axis_lo + in.responder.axis_hi));
            t.anchor = int64_t(rc[arg].r) * 100003 + rc[arg].c, t.size = int(j - i);
            targets.push_back(t);
            ++out.targets[4];
            i = j;
        }
    }
    const std::unordered_set<int64_t>* envp = env.empty() ? nullptr : &env;

    // --- views, raw terms, costs ---------------------------------------------------------------------------------
    for (const auto& t : targets) {
        for (const View& v : SampleViews(map, g, tm.trav_a, t.p, p.sampler, p.frustum, envp)) {
            if (!In(dist, v.cell) || !std::isfinite(dist.at<double>(v.cell.r, v.cell.c))) continue;
            TaskCandidate c;
            c.task_id = TaskId(t.src, t.anchor), c.source = t.src;
            c.goal_xy = {v.eye.x(), v.eye.y()}, c.goal_yaw = v.yaw, c.centroid_xy = {t.p.x(), t.p.y()}, c.cells = t.size;
            c.path_m = dist.at<double>(v.cell.r, v.cell.c) * g.cell;
            c.standoff_m = (v.eye - t.p).head<2>().norm();
            c.ig_raw[kCoverage] = v.ig.coverage, c.ig_raw[kEntropy] = v.ig.entropy, c.ig_raw[kEgressConfidence] = v.ig.egress;
            c.ig_raw[kPoseConnectivity] = t.gain_conn, c.ig_raw[kLoopDisambiguation] = t.gain_alias;
            c.cost = {c.path_m, t.src == TaskSource::kConnectivity ? p.exchange_bytes : 0.0, p.e_per_m * c.path_m,
                      c.path_m / p.speed_mps, 0.0};
            Risk(v.cell, &c);
            c.cost[4] = c.risk;
            out.options.push_back(c);
        }
    }

    // --- self-normalisation and utility (§21.3/§21.4, D114) ------------------------------------------------------
    std::array<double, 5> scale{};
    for (int k = 0; k < 5; ++k) {
        std::vector<double> v;
        for (const auto& c : out.options) v.push_back(c.ig_raw[size_t(k)]);
        scale[size_t(k)] = P90Scale(v);
    }
    std::vector<double> vp, vb, vr;
    for (const auto& c : out.options) vp.push_back(c.path_m), vb.push_back(c.cost[1]), vr.push_back(c.cost[4]);
    const double sp = P90Scale(vp), sb = P90Scale(vb), sr = P90Scale(vr);
    core::PhiloxStream rng(p.random_seed, uint64_t(std::llround(in.t_s * 1000.0)), uint64_t(agent));
    for (auto& c : out.options) {
        c.gain = 0;
        for (int k = 0; k < 5; ++k) {
            c.ig_norm[size_t(k)] = scale[size_t(k)] > 0 ? c.ig_raw[size_t(k)] / scale[size_t(k)] : 0.0;
            c.gain += c.ig_norm[size_t(k)];
        }
        const double cp = sp > 0 ? c.path_m / sp : 0.0, cb = sb > 0 ? c.cost[1] / sb : 0.0, cr = sr > 0 ? c.cost[4] / sr : 0.0;
        switch (p.policy) {
            case NbvPolicy::kAriadne: c.utility = c.gain - out.lambda_path * cp - cb - cr; break;
            case NbvPolicy::kCoverageGreedy: c.utility = c.ig_norm[kCoverage]; break;
            case NbvPolicy::kUncertaintyOnly: c.utility = c.ig_norm[kEntropy]; break;
            case NbvPolicy::kRandom: c.utility = rng.NextUniform(); break;
        }
    }
    std::stable_sort(out.options.begin(), out.options.end(), [](const TaskCandidate& a, const TaskCandidate& b) {
        if (a.utility != b.utility) return a.utility > b.utility;
        if (a.path_m != b.path_m) return a.path_m < b.path_m;
        return a.task_id < b.task_id;
    });
    return out;
}

}  // namespace ariadne::planning
