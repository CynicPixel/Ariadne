// ---------------------------------------------------------------------------
// ariadne_volumetric_eval: 4.1's evaluation on a 1.7 spool (design volumetric.md §6, D110). Replay only, no GT.
//
// The same keyframes, anchor and merge timing as the 1.8 tool's merged run (D45: agent 0's briefed entry pose
// anchors W; agent 1 joins through the 1.6 loop at t_merge, sharing its history then; a schema-3 spool carries one
// merge per non-anchor agent, Phase 8 §42, and each agent k joins at its own t_merge). Each keyframe goes through
// the Phase 4 volumetric stack: TsdfMap::Integrate -> Esdf::Update (incremental) -> FrontierSet3D::Update; each
// stage is timed. At the end the map is projected onto the briefing's 2D grid with 1.8's slab rule over the same z
// band (occupied if any voxel in the band is, free if any is free and none occupied, else unknown), so
// python/analysis/volumetric_eval.py scores it exactly as mission_eval.py scores 1.8's slab.
//
//   ariadne_volumetric_eval --spool DIR --scene DIR --config configs/mission/mission_v1.yaml --out DIR
//                           [--egress 1 [--support-band LO,HI]] [--nbv 1]
//
// --egress 1 (4.5, design egress.md §7, D113): every replan_period_s of mission time (1.8's schedule: t0 + k T,
// then once after the last keyframe), D111's maps -> body support -> the §22.1 penalty -> EgressPlanner (D* Lite,
// target -> any briefed entrance at rho_responder). One JSON line per replan in egress_routes.jsonl; the final
// route's support range in results.yaml (the §7 pre-registered range check, read before any GT scoring).
// --nbv 1 (4.3, next-best-view.md §7, D114; implies the replan schedule): at every replan, PlanNbv per integrated agent
// (pose = its latest keyframe, flown = summed keyframe displacement, peers = the other agent's keyframes); one JSON
// line per (replan, agent) in nbv_decisions.jsonl: targets per generator, notes, timing, the top 10 options. The
// trajectory is the spool's: this logs what the planner WOULD choose (machinery evidence, not policy value).
// --phase4 1 (4.4, allocation.md §8, D115): the real planning::Phase4Planner fed the same keyframes (Integrate) and
// planned on the same schedule (Plan, both agents once active, linked after the merge); per replan one line in
// phase4_allocation.jsonl: its CBBA event (trigger, rounds, messages, TASK_CONSENSUS wire bytes), and on the SAME
// auction instance the totals of CBBA, broker SGA, the greedy joint pick and the exact optimum.
// --route-compare 1 (4.6, exploration.md §3, D116; with --egress 1): at every egress replan, on the same
// rho_responder traversable set, D* Lite SHORTEST (all weights 0, incremental), the FAR-style visibility graph
// (incremental update + query) and a fresh A* (oracle): each structure's own update and query time only (the shared
// map work is excluded), Euclidean length of each executed polyline (lattice paths string-pulled), validity.
// --support-band restricts support to voxel layers with centre z in [LO, HI] m (the §7 fallback).
//
// Writes slab.png (uint8 0 unknown / 1 free / 2 occupied, row 0 = south), results.yaml, manifest.yaml.
// The spool reading duplicates ariadne_mission's; it moves onto L1's shared briefing/spool library when that lands.
// ---------------------------------------------------------------------------
#include "planning/mission/nbv.hpp"
#include "planning/mission/allocation_wire.hpp"
#include "planning/mission/phase4_planner.hpp"
#include "planning/route/egress.hpp"
#include "planning/route/visibility_graph.hpp"
#include "planning/volumetric/esdf.hpp"
#include "planning/volumetric/frontiers3d.hpp"
#include "planning/volumetric/traversability.hpp"
#include "planning/volumetric/tsdf_map.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace ariadne;
namespace fs = std::filesystem;

namespace {

core::SE3 RowMajorPose(const YAML::Node& n) {
    if (!n || n.size() != 12) throw std::runtime_error("pose: expected 12 numbers");
    core::SE3 T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T.R(i, j) = n[i * 4 + j].as<double>();
        T.t(i) = n[i * 4 + 3].as<double>();
    }
    return T;
}

cv::Mat ReadHalf(const fs::path& p, int w, int h) {
    const cv::Mat bits = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    if (bits.empty() || bits.type() != CV_16U || bits.cols != w || bits.rows != h) {
        throw std::runtime_error("spool: bad fp16 png " + p.string());
    }
    cv::Mat f;
    cv::Mat(bits.size(), CV_16F, bits.data, bits.step).convertTo(f, CV_32F);
    return f;
}

struct Kf {
    int agent;
    uint64_t kf_id;
    int64_t t_ns;
    core::SE3 T_O_C;
    double fx, fy, cx, cy;
    int w, h;
    fs::path stem;
};

struct Stats {
    std::vector<double> ms;
    void Add(double x) { ms.push_back(x); }
    double Mean() const { double s = 0; for (double x : ms) s += x; return ms.empty() ? 0 : s / double(ms.size()); }
    double Pct(double q) const {
        if (ms.empty()) return 0;
        std::vector<double> v = ms;
        std::sort(v.begin(), v.end());
        return v[size_t(q * double(v.size() - 1))];
    }
};

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string spool_dir, scene_dir, config_path, out;
        bool egress_on = false, nbv_on = false, phase4_on = false, compare_on = false;
        double band_lo = -1e9, band_hi = 1e9, risk_weight = 0;
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--spool") spool_dir = v;
            else if (k == "--scene") scene_dir = v;
            else if (k == "--config") config_path = v;
            else if (k == "--out") out = v;
            else if (k == "--egress") egress_on = v == "1";
            else if (k == "--nbv") nbv_on = v == "1";
            else if (k == "--phase4") phase4_on = v == "1";
            else if (k == "--route-compare") compare_on = v == "1";
            else if (k == "--risk-weight") risk_weight = std::stod(v);   // D112 passed: the risk-aware egress rows
            else if (k == "--support-band") {
                if (std::sscanf(v.c_str(), "%lf,%lf", &band_lo, &band_hi) != 2 || !(band_hi > band_lo)) {
                    throw std::invalid_argument("--support-band LO,HI");
                }
            }
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (spool_dir.empty() || scene_dir.empty() || config_path.empty() || out.empty()) {
            throw std::invalid_argument("usage: ariadne_volumetric_eval --spool DIR --scene DIR --config Y --out DIR");
        }
        const YAML::Node cfg = YAML::LoadFile(config_path);
        const double z_lo = cfg["slab"]["z_band_m"][0].as<double>(), z_hi = cfg["slab"]["z_band_m"][1].as<double>();
        planning::TsdfParams tp;   // D110 defaults: 5 cm, tau 3 voxels
        tp.pixel_stride = cfg["carving"]["pixel_stride"].as<int>();
        tp.max_depth = cfg["carving"]["max_depth_m"].as<double>();
        tp.max_sigma = cfg["carving"]["max_sigma_m"].as<double>();
        const double rho_r = cfg["clearance"]["rho_responder_m"].as<double>();

        // --- briefing (no GT layers) -----------------------------------------------------------------------------
        const YAML::Node scene = YAML::LoadFile(scene_dir + "/scene.json");
        const YAML::Node mission = YAML::LoadFile(scene_dir + "/mission_default.json");
        const YAML::Node route0 = YAML::LoadFile(scene_dir + "/routes/agent_0.json");
        const double cell = scene["esdf"]["cell_m"].as<double>();
        const double ox = scene["esdf"]["origin_m"][0].as<double>(), oy = scene["esdf"]["origin_m"][1].as<double>();
        const int cols = scene["esdf"]["cols"].as<int>(), rows = scene["esdf"]["rows"].as<int>();
        if (std::abs(cell - tp.voxel) > 1e-12) throw std::runtime_error("the briefing grid cell must equal the voxel");
        const YAML::Node e0 = scene["entrances"][0];
        const double yaw = e0["start_yaw_deg"].as<double>() * M_PI / 180.0;
        core::SE3 T_W_B0;
        T_W_B0.R = core::Mat3::Identity();
        T_W_B0.R(0, 0) = std::cos(yaw), T_W_B0.R(0, 1) = -std::sin(yaw);
        T_W_B0.R(1, 0) = std::sin(yaw), T_W_B0.R(1, 1) = std::cos(yaw);
        T_W_B0.t = core::Vec3(e0["start_xy_m"][0].as<double>(), e0["start_xy_m"][1].as<double>(),
                              route0["segments"][0]["start"][2].as<double>());
        core::SE3 T_B_C;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) T_B_C.R(i, j) = mission["R_B_optical_rows"][i][j].as<double>();
            T_B_C.t(i) = mission["t_B_camera_m"]["left"][i].as<double>();
        }
        const core::SE3 T_W_O0 = T_W_B0 * T_B_C;

        // --- spool -----------------------------------------------------------------------------------------------
        const YAML::Node sp = YAML::LoadFile(spool_dir + "/spool.yaml");
        const int schema = sp["schema_version"].as<int>();
        if (schema < 1 || schema > 3) throw std::runtime_error("spool: schema_version not 1, 2 or 3");
        std::vector<Kf> kfs;
        std::map<std::pair<int, uint64_t>, int64_t> t_of;
        for (const auto& b : sp["bundles"]) {
            if (b["role"].as<std::string>() != "keyframe") continue;
            Kf k{b["agent"].as<int>(), b["kf_id"].as<uint64_t>(), b["t_ns"].as<int64_t>(), RowMajorPose(b["T_WiC"]),
                 b["K"][0].as<double>(), b["K"][1].as<double>(), b["K"][2].as<double>(), b["K"][3].as<double>(),
                 b["width"].as<int>(), b["height"].as<int>(), fs::path(spool_dir) / b["files"].as<std::string>()};
            t_of[{k.agent, k.kf_id}] = k.t_ns;
            kfs.push_back(k);
        }
        std::sort(kfs.begin(), kfs.end(), [](const Kf& a, const Kf& b) {
            return std::tie(a.t_ns, a.agent, a.kf_id) < std::tie(b.t_ns, b.agent, b.kf_id);
        });
        // One merge per non-anchor agent: its T_O0_Ok and t_merge = the later keyframe of its merge edge
        // ([ASSUMPTION]: found then). Schemas 1 and 2 (A2's D195 photometric fields; two agents) have one `merge` block,
        // agent 1's; schema 3 (Phase 8 §42, N agents) = schema 2's fields + a `merges` list of {agent, ok, T_O0_O, edge}
        // [DESIGN]. The photometric fields are not read here. Agent 0 anchors W (D45).
        int n_agents = 2;
        if (schema == 3)
            for (const auto& k : kfs) n_agents = std::max(n_agents, k.agent + 1);
        struct Merge {
            bool ok = false;
            core::SE3 T_O0_O;
            int64_t t = std::numeric_limits<int64_t>::max();
        };
        std::vector<Merge> merges(static_cast<size_t>(n_agents));
        merges[0].ok = true, merges[0].t = std::numeric_limits<int64_t>::min();
        auto ReadMerge = [&](const YAML::Node& mg, int agent) {
            Merge& M = merges.at(size_t(agent));
            M.ok = mg["ok"].as<bool>();
            M.T_O0_O = RowMajorPose(schema < 3 ? mg["T_O0_O1"] : mg["T_O0_O"]);
            if (!M.ok) return;
            std::smatch m;
            const std::string edge = mg["edge"].as<std::string>();
            if (!std::regex_search(edge, m, std::regex(R"(agent(\d+) kf (\d+) <- agent(\d+) kf (\d+))")) ||
                std::stoi(m[3]) != agent) {
                throw std::runtime_error("spool: cannot parse merge edge '" + edge + "' for agent " + std::to_string(agent));
            }
            M.t = std::max(t_of.at({std::stoi(m[1]), std::stoull(m[2])}), t_of.at({agent, std::stoull(m[4])}));
        };
        if (schema < 3) {
            ReadMerge(sp["merge"], 1);
        } else {
            if (!sp["merges"] || !sp["merges"].IsSequence()) throw std::runtime_error("spool: schema 3 needs a merges list");
            if (sp["merge"]) throw std::runtime_error("spool: schema 3 carries merges, not a merge block");
            for (const auto& mg : sp["merges"]) ReadMerge(mg, mg["agent"].as<int>());
        }

        // --- the volumetric stack, per keyframe ----------------------------------------------------------------------
        planning::TsdfMap map(tp);
        planning::Esdf esdf(tp.voxel, rho_r + 2 * tp.voxel);
        planning::FrontierParams fp;
        fp.z_lo = z_lo, fp.z_hi = z_hi;
        planning::FrontierSet3D fr(fp);
        Stats s_int, s_esdf, s_fr;
        std::vector<size_t> integrated(size_t(n_agents), 0);
        size_t incremental = 0, rebuilds = 0, esdf_voxels = 0;
        std::vector<std::vector<Kf>> held(static_cast<size_t>(n_agents));   // per agent, before its merge
        std::vector<std::vector<core::Vec3>> track(static_cast<size_t>(n_agents));
        std::vector<double> flown(size_t(n_agents), 0.0);
        std::unique_ptr<planning::Phase4Planner> p4;
        std::ofstream p4_log;
        auto Integrate = [&](const Kf& k) {
            const core::SE3 T_W_C = T_W_O0 * (k.agent == 0 ? k.T_O_C : merges[size_t(k.agent)].T_O0_O * k.T_O_C);
            if (!track[k.agent].empty()) flown[k.agent] += (T_W_C.t - track[k.agent].back()).norm();
            track[k.agent].push_back(T_W_C.t);
            const cv::Mat Z = ReadHalf(k.stem.string() + "_depth.png", k.w, k.h);
            const cv::Mat S = ReadHalf(k.stem.string() + "_sigma.png", k.w, k.h);
            planning::DepthFrame f;
            f.T_W_C = T_W_C, f.fx = k.fx, f.fy = k.fy, f.cx = k.cx, f.cy = k.cy, f.width = k.w, f.height = k.h;
            f.depth = Z.ptr<float>(), f.sigma = S.ptr<float>();
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<planning::Idx3> dirty = map.Integrate(f);
            s_int.Add(Ms(t0));
            if (p4) p4->Integrate(k.agent, k.t_ns, f);   // the real Phase4Planner, fed the identical frame (untimed)
            t0 = std::chrono::steady_clock::now();
            if (esdf.Empty()) esdf.Rebuild(map), ++rebuilds;
            else if (esdf.Update(map, dirty)) ++incremental;
            else ++rebuilds;
            esdf_voxels += esdf.LastUpdatedVoxels();
            s_esdf.Add(Ms(t0));
            t0 = std::chrono::steady_clock::now();
            if (!dirty.empty()) {
                planning::Idx3 lo{1 << 30, 1 << 30, 1 << 30}, hi{-(1 << 30), -(1 << 30), -(1 << 30)};
                for (const auto& b : dirty) {
                    lo = {std::min(lo.x, b.x * 8), std::min(lo.y, b.y * 8), std::min(lo.z, b.z * 8)};
                    hi = {std::max(hi.x, b.x * 8 + 7), std::max(hi.y, b.y * 8 + 7), std::max(hi.z, b.z * 8 + 7)};
                }
                if (std::accumulate(integrated.begin(), integrated.end(), size_t(0)) == 0) fr.Detect(map);
                else fr.Update(map, lo, hi);
            }
            s_fr.Add(Ms(t0));
            ++integrated[k.agent];
        };
        // --- 4.5 egress replans (egress.md §4/§7) ----------------------------------------------------------------
        planning::GridFrame gf;
        gf.ox = ox, gf.oy = oy, gf.cell = cell, gf.cols = cols, gf.rows = rows;
        planning::TravParams trp;
        trp.responder.rho = rho_r;
        trp.agent.rho = cfg["clearance"]["rho_agent_m"].as<double>();
        auto ToCell = [&](double x, double y) {   // 1.8's (ariadne_mission)
            return planning::Cell{int(std::floor((y - oy) / cell)), int(std::floor((x - ox) / cell))};
        };
        const planning::Cell target = ToCell(scene["target_xy_m"][0].as<double>(), scene["target_xy_m"][1].as<double>());
        std::vector<planning::Cell> exits;
        for (const auto& e : scene["entrances"]) exits.push_back(ToCell(e["start_xy_m"][0].as<double>(), e["start_xy_m"][1].as<double>()));
        planning::EgressWeights ew;   // w_u = w_c = 1; w_r from --risk-weight (0 default; 1 = the D112-calibrated rows)
        ew.w_r = risk_weight;
        planning::EgressPlanner egress(target, exits, cell);
        planning::EgressPlanner egress_short(target, exits, cell);   // §3 comparator: all weights 0 (length only)
        planning::VisibilityGraph far_graph;
        std::ofstream cmp_log;
        if (compare_on) fs::create_directories(out), cmp_log.open(fs::path(out) / "route_compare.jsonl");
        std::ofstream eg_log;
        if (egress_on || nbv_on || phase4_on) fs::create_directories(out), eg_log.open(fs::path(out) / "egress_routes.jsonl");
        std::ofstream nbv_log;
        if (nbv_on) nbv_log.open(fs::path(out) / "nbv_decisions.jsonl");
        if (phase4_on) {
            planning::Phase4Briefing pb;
            pb.grid = gf, pb.target = target, pb.entrances = exits, pb.z_max_m = cfg["grid"]["z_max_m"].as<double>();
            planning::Phase4Params pp;
            pp.tsdf = tp;
            pp.trav = trp;
            if (!kfs.empty()) {
                pp.nbv.frustum.fx = kfs[0].fx, pp.nbv.frustum.fy = kfs[0].fy, pp.nbv.frustum.cx = kfs[0].cx;
                pp.nbv.frustum.cy = kfs[0].cy, pp.nbv.frustum.width = kfs[0].w, pp.nbv.frustum.height = kfs[0].h;
            }
            pp.nbv.frustum.max_range_m = tp.max_depth;
            pp.nbv.rho_r = rho_r, pp.nbv.rho_a = trp.agent.rho;
            p4 = std::make_unique<planning::Phase4Planner>(pb, pp, n_agents);
            fs::create_directories(out);
            p4_log.open(fs::path(out) / "phase4_allocation.jsonl");
        }
        Stats s_p4;
        const core::WireCodec p4_codec;
        Stats s_nbv;
        planning::NbvParams nbp;   // replay: no energy budget or mission clock (the hard rule cannot fire; stated)
        nbp.rho_r = rho_r, nbp.rho_a = trp.agent.rho;
        if (!kfs.empty()) {
            nbp.frustum.fx = kfs[0].fx, nbp.frustum.fy = kfs[0].fy, nbp.frustum.cx = kfs[0].cx, nbp.frustum.cy = kfs[0].cy;
            nbp.frustum.width = kfs[0].w, nbp.frustum.height = kfs[0].h;
        }
        nbp.frustum.max_range_m = tp.max_depth;
        double first_valid_s = -1;
        int replans = 0, reroutes = 0;
        Stats s_replan;
        planning::EgressReplan last_route;
        double fresh_cost = -1;
        size_t fresh_expanded = 0, incremental_expanded = 0;
        cv::Mat last_support;
        auto Replan = [&](int64_t t_ns, bool final_replan) {
            const auto w0 = std::chrono::steady_clock::now();
            const planning::TraversabilityMaps m = planning::ComputeTraversability(map, esdf, gf, trp);
            const cv::Mat support = planning::BodySupport(map, gf, trp.responder, band_lo, band_hi);
            const cv::Mat pen = planning::EgressPenalty(m.clear_r, support, m.risk, ew);
            const planning::EgressReplan r = egress.Replan(m.trav_r, pen, support, m.risk);
            if (compare_on) {
                auto PolyLen = [&](const std::vector<planning::Cell>& q) {
                    double L = 0;
                    for (size_t i = 1; i < q.size(); ++i) L += std::hypot(double(q[i].r - q[i - 1].r), double(q[i].c - q[i - 1].c));
                    return L * cell;
                };
                auto Valid = [&](const std::vector<planning::Cell>& q) {
                    for (size_t i = 1; i < q.size(); ++i)
                        if (!planning::LineOfSight(m.trav_r, q[i - 1], q[i])) return false;
                    return true;
                };
                const cv::Mat zero = cv::Mat::zeros(m.trav_r.size(), CV_32F);
                auto t1 = std::chrono::steady_clock::now();
                const planning::EgressReplan ds = egress_short.Replan(m.trav_r, zero, support, cv::Mat());
                const double ms_ds = Ms(t1);
                t1 = std::chrono::steady_clock::now();
                const size_t los = far_graph.Update(m.trav_r);
                const double ms_far_upd = Ms(t1);
                t1 = std::chrono::steady_clock::now();
                const planning::VisRoute fr_r = far_graph.Query(target, exits);
                const double ms_far_q = Ms(t1);
                t1 = std::chrono::steady_clock::now();
                const planning::GridPath as = planning::AStar(m.trav_r, target, exits);
                const double ms_as = Ms(t1);
                const auto ds_pull = ds.found ? planning::StringPull(m.trav_r, ds.cells) : std::vector<planning::Cell>{};
                const auto as_pull = as.found ? planning::StringPull(m.trav_r, as.cells) : std::vector<planning::Cell>{};
                cmp_log << "{\"t_s\":" << double(t_ns - kfs.front().t_ns) * 1e-9 << ",\"dstar_short\":{\"found\":" << ds.found
                        << ",\"ms\":" << ms_ds << ",\"expanded\":" << ds.expanded << ",\"lattice_m\":" << ds.length_m
                        << ",\"pulled_m\":" << PolyLen(ds_pull) << ",\"valid\":" << Valid(ds_pull) << "},\"far\":{\"found\":"
                        << fr_r.found << ",\"update_ms\":" << ms_far_upd << ",\"query_ms\":" << ms_far_q << ",\"los_tests\":" << los
                        << ",\"nodes\":" << far_graph.Nodes().size() << ",\"edges\":" << far_graph.Edges() << ",\"length_m\":"
                        << fr_r.length_cells * cell << ",\"valid\":" << Valid(fr_r.cells) << "},\"astar\":{\"found\":" << as.found
                        << ",\"ms\":" << ms_as << ",\"expanded\":" << as.expanded << ",\"lattice_m\":" << as.length_cells * cell
                        << ",\"pulled_m\":" << PolyLen(as_pull) << "}}\n";
            }
            s_replan.Add(Ms(w0));
            const double ts = double(t_ns - kfs.front().t_ns) * 1e-9;
            if (r.found && first_valid_s < 0) first_valid_s = ts;
            reroutes += !r.reroute.empty();
            incremental_expanded += r.expanded;
            ++replans;
            std::string trig;
            for (const auto& t : r.triggers) trig += (trig.empty() ? "\"" : ",\"") + t + "\"";
            eg_log << "{\"t_s\":" << ts << ",\"final\":" << (final_replan ? "true" : "false") << ",\"found\":"
                   << (r.found ? "true" : "false") << ",\"exit\":" << r.exit << ",\"length_m\":" << r.length_m
                   << ",\"cost_cells\":" << (r.found ? r.cost_cells : -1) << ",\"confidence\":" << r.confidence
                   << ",\"changed_cells\":" << r.changed_cells << ",\"expanded\":" << r.expanded << ",\"ms\":"
                   << Ms(w0) << ",\"triggers\":[" << trig << "],\"reroute\":\"" << r.reroute << "\",\"cells\":[";
            for (size_t i = 0; i < r.cells.size(); ++i) eg_log << (i ? "," : "") << "[" << r.cells[i].r << "," << r.cells[i].c << "]";
            eg_log << "]}\n";
            if (final_replan) {   // run-level check of the incremental search: a fresh D* Lite on the final grids
                const planning::DStarLite fresh(m.trav_r, pen, target, exits);
                fresh_cost = fresh.Cost(), fresh_expanded = fresh.Expanded();
            }
            last_route = r, last_support = support;
            if (p4) {
                std::vector<planning::Phase4AgentInput> ag(static_cast<size_t>(n_agents));
                int n_active = 0;
                for (int a = 0; a < n_agents; ++a) {
                    ag[size_t(a)].active = !track[size_t(a)].empty();
                    if (ag[size_t(a)].active) ag[size_t(a)].position = track[size_t(a)].back(), ag[size_t(a)].flown_m = flown[size_t(a)], ++n_active;
                }
                // An agent integrates only after its merge, so every active agent shares the merged frame: all of them
                // are linked once two are active (N = 2: linked = both active).
                for (auto& x : ag) x.linked = x.active && n_active >= 2;
                const auto t1 = std::chrono::steady_clock::now();
                const planning::Phase4Plan P = p4->Plan(ts, ag, {});
                const double ms = Ms(t1);
                s_p4.Add(ms);
                // The rows, on the SAME auction instance (allocation.md §4/§8): CBBA, broker SGA, greedy, optimum.
                std::vector<float> sc;
                std::vector<std::vector<int>> view;
                const planning::AuctionInstance in = planning::BuildAuction(P, gf, 1, &sc, &view);
                planning::CbbaParams cp;
                cp.bytes = planning::WireBytes(p4_codec, sc);
                cp.transport = planning::WireTransport(p4_codec, sc);
                const bool linked = n_active >= 2;
                std::vector<std::vector<int>> G(size_t(n_agents), std::vector<int>(size_t(n_agents), 0));
                for (int i = 0; i < n_agents; ++i)
                    for (int j = 0; j < n_agents; ++j) G[size_t(i)][size_t(j)] = i != j && ag[size_t(i)].linked && ag[size_t(j)].linked;
                const planning::Allocation cb = planning::RunCbba(in, [&](int) { return G; }, cp);
                const planning::Allocation sg = planning::SequentialGreedy(in);
                planning::Phase4Plan Pg = P;
                planning::JointPick(&Pg);
                double greedy = 0;
                for (int a = 0; a < n_agents; ++a) {
                    if (Pg.joint[size_t(a)] < 0) continue;
                    for (int j = 0; j < in.Tasks(); ++j)
                        if (view[size_t(a)][size_t(j)] == Pg.joint[size_t(a)]) greedy += planning::PathScore(in, a, {j});
                }
                const double opt = n_agents != 2 ? -1.0 : in.Tasks() ? planning::TwoAgentOptimum(in) : 0.0;   // 2-agent only
                // The §31.7 L_t = 2 row on the same plan (D115 correction: CBBA may differ from SGA there).
                std::vector<float> sc2;
                std::vector<std::vector<int>> view2;
                const auto t2 = std::chrono::steady_clock::now();
                const planning::AuctionInstance in2 = planning::BuildAuction(P, gf, 2, &sc2, &view2);
                const double ms_legs = Ms(t2);
                planning::CbbaParams cp2;
                cp2.bytes = planning::WireBytes(p4_codec, sc2);
                cp2.transport = planning::WireTransport(p4_codec, sc2);
                const planning::Allocation cb2 = planning::RunCbba(in2, [&](int) { return G; }, cp2);
                const planning::Allocation sg2 = planning::SequentialGreedy(in2);
                const double opt2 = n_agents != 2 ? -1.0 : in2.Tasks() ? planning::TwoAgentOptimum(in2) : 0.0;
                p4_log << "{\"t_s\":" << ts << ",\"plan_ms\":" << ms << ",\"active\":[";
                for (int a = 0; a < n_agents; ++a) p4_log << (a ? "," : "") << ag[size_t(a)].active;
                p4_log << "],\"linked\":" << linked << ",\"options\":[";
                for (int a = 0; a < n_agents; ++a) p4_log << (a ? "," : "") << P.agents[size_t(a)].nbv.options.size();
                p4_log << "],\"event\":{\"method\":\"" << P.allocation.method
                       << "\",\"trigger\":\"" << P.allocation.trigger << "\",\"rounds\":" << P.allocation.rounds
                       << ",\"messages\":" << P.allocation.messages << ",\"bytes\":" << P.allocation.bytes << "},\"tasks\":"
                       << in.Tasks() << ",\"cbba\":{\"total\":" << cb.total << ",\"rounds\":" << cb.rounds << ",\"messages\":"
                       << cb.messages << ",\"bytes\":" << cb.bytes << "},\"sga\":" << sg.total << ",\"greedy\":" << greedy
                       << ",\"optimum\":" << opt << ",\"cbba_eq_sga\":" << (cb.winner == sg.winner)
                       << ",\"lt2\":{\"cbba\":" << cb2.total << ",\"rounds\":" << cb2.rounds << ",\"messages\":" << cb2.messages
                       << ",\"bytes\":" << cb2.bytes << ",\"tail_releases\":" << cb2.tail_releases << ",\"sga\":" << sg2.total
                       << ",\"optimum\":" << opt2 << ",\"cbba_eq_sga\":" << (cb2.winner == sg2.winner) << ",\"legs_ms\":" << ms_legs
                       << "},\"joint\":[";
                for (int a = 0; a < n_agents; ++a) {
                    const int i = P.joint[size_t(a)];
                    p4_log << (a ? "," : "");
                    if (i < 0) {
                        p4_log << "null";
                        continue;
                    }
                    const auto& o = P.agents[a].nbv.options[size_t(i)];
                    p4_log << "{\"src\":\"" << char(o.source) << "\",\"task_id\":" << o.task_id << ",\"goal\":["
                           << o.goal_xy.first << "," << o.goal_xy.second << "]}";
                }
                p4_log << "]}\n";
            }
            if (nbv_on) {
                const planning::CoarseLayer coarse_now = planning::BuildCoarse(map);
                for (int a = 0; a < n_agents; ++a) {
                    if (track[size_t(a)].empty()) continue;
                    planning::NbvInputs ni;
                    ni.map = &map, ni.frontiers = &fr, ni.coarse = &coarse_now, ni.trav = &m, ni.grid = gf;
                    ni.route = &r, ni.route_support = &support, ni.responder = trp.responder, ni.entrances = exits;
                    for (int b = 0; b < n_agents; ++b)   // every other agent's keyframes, ascending (N = 2: the other one)
                        if (b != a)
                            for (size_t i = 0; i < track[size_t(b)].size(); ++i) ni.peers.push_back({b, int64_t(i), track[size_t(b)][i]});
                    ni.t_s = ts;
                    ni.bounds.lo = core::Vec3(ox, oy, 0.0);
                    ni.bounds.hi = core::Vec3(ox + cols * cell, oy + rows * cell, cfg["grid"]["z_max_m"].as<double>());
                    planning::AgentState as;
                    as.position = track[size_t(a)].back(), as.e_used_m = flown[size_t(a)], as.own_keyframes = track[size_t(a)];
                    const auto t1 = std::chrono::steady_clock::now();
                    const planning::NbvPlan pl = planning::PlanNbv(ni, a, as, nbp);
                    const double ms = Ms(t1);
                    s_nbv.Add(ms);
                    nbv_log << "{\"t_s\":" << ts << ",\"agent\":" << a << ",\"mode\":\"replay\",\"executed\":false,\"ms\":" << ms
                            << ",\"lambda_path\":" << pl.lambda_path << ",\"targets\":[" << pl.targets[0] << "," << pl.targets[1]
                            << "," << pl.targets[2] << "," << pl.targets[3] << "," << pl.targets[4] << "],\"n_options\":"
                            << pl.options.size() << ",\"notes\":[";
                    for (size_t i = 0; i < pl.notes.size(); ++i) nbv_log << (i ? "," : "") << "\"" << pl.notes[i] << "\"";
                    nbv_log << "],\"options\":[";
                    for (size_t i = 0; i < std::min<size_t>(10, pl.options.size()); ++i) {
                        const auto& o = pl.options[i];
                        nbv_log << (i ? "," : "") << "{\"src\":\"" << char(o.source) << "\",\"task_id\":" << o.task_id
                                << ",\"goal\":[" << o.goal_xy.first << "," << o.goal_xy.second << "],\"yaw\":" << o.goal_yaw
                                << ",\"path_m\":" << o.path_m << ",\"ig_raw\":[" << o.ig_raw[0] << "," << o.ig_raw[1] << ","
                                << o.ig_raw[2] << "," << o.ig_raw[3] << "," << o.ig_raw[4] << "],\"gain\":" << o.gain
                                << ",\"utility\":" << o.utility << ",\"risk\":" << o.risk << "}";
                    }
                    nbv_log << "]}\n";
                }
            }
        };
        int64_t next_replan = kfs.front().t_ns + int64_t(cfg["replan_period_s"].as<double>() * 1e9);
        for (const auto& k : kfs) {
            while ((egress_on || nbv_on || phase4_on) && k.t_ns >= next_replan) {
                Replan(next_replan, false);
                next_replan += int64_t(cfg["replan_period_s"].as<double>() * 1e9);
            }
            const Merge& M = merges.at(size_t(k.agent));
            if (!M.ok) continue;
            if (k.t_ns < M.t) {
                held[size_t(k.agent)].push_back(k);
                continue;
            }
            for (int j = 1; j < n_agents; ++j)   // a merge: that agent shares its history (ascending agent index)
                if (!held[size_t(j)].empty() && k.t_ns >= merges[size_t(j)].t) {
                    for (const auto& h : held[size_t(j)]) Integrate(h);
                    held[size_t(j)].clear();
                }
            Integrate(k);
        }
        if (egress_on || nbv_on || phase4_on) Replan(kfs.back().t_ns, true);

        // --- slab on the briefing grid (1.8's rule, same band) -------------------------------------------------------
        const int zv0 = int(std::floor(z_lo / tp.voxel)), zv1 = int(std::floor(z_hi / tp.voxel));
        const int cx0 = int(std::lround(ox / tp.voxel)), cy0 = int(std::lround(oy / tp.voxel));
        cv::Mat slab(rows, cols, CV_8U, cv::Scalar(planning::kUnknown));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                bool any_free = false, any_occ = false;
                for (int z = zv0; z <= zv1 && !any_occ; ++z) {
                    const uint8_t st = map.State({cx0 + c, cy0 + r, z});
                    any_free |= st == planning::kFree;
                    any_occ |= st == planning::kOccupied;
                }
                slab.at<uint8_t>(r, c) = any_occ ? planning::kOccupied : (any_free ? planning::kFree : planning::kUnknown);
            }
        }
        fs::create_directories(out);
        cv::imwrite((fs::path(out) / "slab.png").string(), slab);
        // 4.2 (D111): the capsule traversability at both rho and the [PROXY] risk, on the same grid.
        auto t_trav = std::chrono::steady_clock::now();
        const planning::TraversabilityMaps tm = planning::ComputeTraversability(map, esdf, gf, trp);
        const double trav_ms = Ms(t_trav);
        cv::imwrite((fs::path(out) / "trav_responder.png").string(), tm.trav_r);
        cv::imwrite((fs::path(out) / "trav_agent.png").string(), tm.trav_a);
        cv::imwrite((fs::path(out) / "trav_responder_strict3d.png").string(), tm.trav_r_strict);
        cv::Mat risk16;
        tm.risk.convertTo(risk16, CV_16U, 65535.0);
        cv::imwrite((fs::path(out) / "risk_proxy.png").string(), risk16);
        tm.overhang.convertTo(risk16, CV_16U, 65535.0);
        cv::imwrite((fs::path(out) / "risk_overhang.png").string(), risk16);
        tm.unsupported.convertTo(risk16, CV_16U, 65535.0);
        cv::imwrite((fs::path(out) / "risk_unsupported.png").string(), risk16);
        double risk_max;
        cv::minMaxLoc(tm.risk, nullptr, &risk_max);
        std::string risk_where;   // per term: max and its world xy (diagnosis of the [PROXY]'s response)
        for (const auto& [name, m] : {std::pair{"overhang", &tm.overhang}, std::pair{"unsupported", &tm.unsupported}}) {
            double mx;
            cv::Point at;
            cv::minMaxLoc(*m, nullptr, &mx, nullptr, &at);
            char buf[160];
            std::snprintf(buf, sizeof buf, "%s%s: {max: %.3f, at_xy_m: [%.3f, %.3f], cells_over_0.2: %d}",
                          risk_where.empty() ? "" : ", ", name, mx, ox + (at.x + 0.5) * cell, oy + (at.y + 0.5) * cell,
                          cv::countNonZero(*m > 0.2));
            risk_where += buf;
        }
        // Diagnostics (D111 §5): the responder capsule with unknown treated as free (agent field), per axis height,
        // so the cause of a closed column can be split into "unobserved" vs "occupied", and at which height.
        {
            const int zl = int(std::floor(trp.responder.axis_lo / cell)), zh = int(std::floor(trp.responder.axis_hi / cell));
            cv::Mat occ_only(rows, cols, CV_8U), bind_z(rows, cols, CV_8U, cv::Scalar(255));
            const int cx0 = int(std::lround(ox / cell)), cy0 = int(std::lround(oy / cell));
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c) {
                    bool ok = true;
                    for (int z = zl; z <= zh; ++z) {
                        if (esdf.Agent({cx0 + c, cy0 + r, z}) < trp.responder.rho - 1e-9) ok = false;
                        if (bind_z.at<uint8_t>(r, c) == 255 && esdf.Egress({cx0 + c, cy0 + r, z}) < trp.responder.rho - 1e-9)
                            bind_z.at<uint8_t>(r, c) = uint8_t(z - zl);   // lowest axis voxel that closes the column
                    }
                    occ_only.at<uint8_t>(r, c) = ok;
                }
            cv::imwrite((fs::path(out) / "diag_trav_responder_occupied_only.png").string(), occ_only);
            cv::imwrite((fs::path(out) / "diag_bind_z.png").string(), bind_z);
            // Unknown voxels per z layer, 0 .. 1.9 m, for every column of the grid (the evaluator masks by GT-free).
            cv::Mat unk(rows * 38, cols, CV_8U);
            for (int z = 0; z < 38; ++z)
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < cols; ++c)
                        unk.at<uint8_t>(z * rows + r, c) = map.State({cx0 + c, cy0 + r, z}) == planning::kUnknown;
            cv::imwrite((fs::path(out) / "diag_unknown_by_layer.png").string(), unk);
        }
        // next-best-view.md §3 check (no GT): known-voxel weight quantiles, and the binary entropy each would carry under the
        // proposed p_occ (free 0.5 e^-w, occupied 1 - 0.5 e^-w), so a saturated Delta-entropy is caught before adoption.
        std::string wq;
        {
            std::vector<float> wf, wo;
            map.ForEachBlock([&](const planning::Idx3& bi, const planning::TsdfBlock& b) {
                for (int i = 0; i < planning::kBlockVoxels; ++i) {
                    const auto& v = b.v[size_t(i)];
                    if (v.w <= tp.w_min) continue;
                    (v.sdf < 0.5 * tp.voxel ? wo : wf).push_back(v.w);
                }
                (void)bi;
            });
            auto H = [](double p) { return p <= 0 || p >= 1 ? 0.0 : -(p * std::log2(p) + (1 - p) * std::log2(1 - p)); };
            for (auto [name, vv] : {std::pair{"free", &wf}, std::pair{"occupied", &wo}}) {
                std::sort(vv->begin(), vv->end());
                auto Q = [&](double q) { return vv->empty() ? 0.0 : double((*vv)[size_t(q * double(vv->size() - 1))]); };
                double hsum = 0, hlo = 0;
                for (float w : *vv) {
                    const double h = H(0.5 * std::exp(-double(w)));
                    hsum += h, hlo += h > 0.9;
                }
                char buf[320];
                std::snprintf(buf, sizeof buf, "%s%s: {n: %zu, w_p10: %.4g, w_p50: %.4g, w_p90: %.4g, w_max: %.4g, "
                              "H_mean_bits: %.4f, frac_H_over_0.9: %.4f}", wq.empty() ? "" : ", ", name, vv->size(), Q(0.1),
                              Q(0.5), Q(0.9), Q(1.0), vv->empty() ? 0.0 : hsum / double(vv->size()),
                              vv->empty() ? 0.0 : hlo / double(vv->size()));
                wq += buf;
            }
        }
        const planning::CoarseLayer coarse = planning::BuildCoarse(map);
        const size_t rle = planning::EncodeCoarseRle(coarse).size();
        std::ofstream res(fs::path(out) / "results.yaml");
        res.precision(6);
        res << "# ariadne_volumetric_eval (volumetric.md §6, D110). Generated; host timings are indicative.\n"
            << "keyframes_integrated: {" << [&] {
                   std::ostringstream o;
                   for (int a = 0; a < n_agents; ++a) o << (a ? ", " : "") << "agent" << a << ": " << integrated[size_t(a)];
                   return o.str();
               }() << "}\n"
            << "voxel_m: " << tp.voxel << "\ntruncation_voxels: " << tp.truncation_voxels
            << "\nblocks: " << map.Blocks() << "\nmap_bytes: " << map.Blocks() * sizeof(planning::TsdfBlock)
            << "\nesdf: {incremental: " << incremental << ", rebuilds: " << rebuilds
            << ", mean_voxels_recomputed: " << (incremental + rebuilds ? esdf_voxels / (incremental + rebuilds) : 0)
            << ", box: [" << esdf.Dims().x << ", " << esdf.Dims().y << ", " << esdf.Dims().z << "]}\n"
            << "frontier_clusters_final: " << fr.Clusters().size() << "\nfrontier_rejected_small: " << fr.RejectedSmall()
            << "\ncoarse: {cells: " << coarse.cells.size() << ", rle_bytes: " << rle << "}\n"
            << "ms_per_keyframe:\n"
            << "  integrate: {mean: " << s_int.Mean() << ", p95: " << s_int.Pct(0.95) << "}\n"
            << "  esdf_update: {mean: " << s_esdf.Mean() << ", p95: " << s_esdf.Pct(0.95) << "}\n"
            << "  frontier_update: {mean: " << s_fr.Mean() << ", p95: " << s_fr.Pct(0.95) << "}\n"
            << "traversability: {ms: " << trav_ms << ", responder_cells: " << cv::countNonZero(tm.trav_r)
            << ", agent_cells: " << cv::countNonZero(tm.trav_a) << ", risk_proxy_max: " << risk_max << "}   # [PROXY] risk, uncalibrated\n"
            << "risk_terms: {" << risk_where << "}\n";
        res << "known_weights: {" << wq << "}   # next-best-view.md §3 entropy-saturation check\n";
        if (phase4_on) res << "phase4: {replans: " << s_p4.ms.size() << ", plan_ms: {mean: " << s_p4.Mean() << ", p95: " << s_p4.Pct(0.95) << "}}\n";
        if (nbv_on) res << "nbv: {replans_x_agents: " << s_nbv.ms.size() << ", ms: {mean: " << s_nbv.Mean() << ", p95: " << s_nbv.Pct(0.95) << "}}\n";
        if (egress_on) {
            // §7 range check (no GT): support along the final route. Flat (spread < 0.1) => the band fallback.
            std::vector<float> sv;
            for (const auto& q : last_route.cells) sv.push_back(last_support.at<float>(q.r, q.c));
            std::sort(sv.begin(), sv.end());
            auto Q = [&](double p) { return sv.empty() ? 0.0 : double(sv[size_t(std::floor(p * double(sv.size() - 1)))]); };
            res << "egress:   # 4.5 (egress.md, D113): D* Lite at rho_responder, w_u = w_c = 1, w_r = " << ew.w_r << "\n"
                << "  support_band_m: [" << band_lo << ", " << band_hi << "]\n"
                << "  replans: " << replans << "\n  reroutes: " << reroutes << "\n  first_route_s: " << first_valid_s
                << "\n  final: {found: " << (last_route.found ? "true" : "false") << ", exit: " << last_route.exit
                << ", length_m: " << last_route.length_m << ", cells: " << last_route.cells.size()
                << ", confidence_p5: " << last_route.confidence << "}\n"
                << "  final_route_support: {min: " << Q(0) << ", p5: " << Q(0.05) << ", median: " << Q(0.5)
                << ", max: " << Q(1) << ", spread: " << (Q(1) - Q(0)) << "}\n"
                << "  final_vs_fresh: {incremental_cost_cells: " << (last_route.found ? last_route.cost_cells : -1)
                << ", fresh_cost_cells: " << (std::isfinite(fresh_cost) ? fresh_cost : -1)
                << ", equal: " << ((last_route.found ? last_route.cost_cells : -1) == (std::isfinite(fresh_cost) ? fresh_cost : -1) ? "true" : "false")
                << ", expanded_incremental_all_replans: " << incremental_expanded << ", expanded_fresh_once: " << fresh_expanded << "}\n"
                << "  ms_per_replan: {mean: " << s_replan.Mean() << ", p95: " << s_replan.Pct(0.95) << "}\n";
            cv::Mat sup16;
            last_support.convertTo(sup16, CV_16U, 65535.0);
            cv::imwrite((fs::path(out) / "egress_support.png").string(), sup16);
        }
        std::ofstream man(fs::path(out) / "manifest.yaml");
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        auto Env = [](const char* k) { return std::getenv(k) ? std::string(std::getenv(k)) : std::string("UNRECORDED"); };
        man << "# ariadne_volumetric_eval manifest. Generated.\ntool: ariadne_volumetric_eval\ncommand: \"" << cmd
            << "\"\nscene_id: " << scene["scene_id"].as<std::string>() << "\nspool_run_id: " << sp["run_id"].as<std::string>()
            << "\nspool_sha256: " << Env("ARIADNE_SPOOL_SHA256") << "\nconfig: " << cfg["name"].as<std::string>()
            << "\nlayer: phase4_volumetric (TsdfMap D110)\ngit_commit: " << Env("ARIADNE_GIT_COMMIT")
            << "\ngit_dirty: " << Env("ARIADNE_GIT_DIRTY") << "\nimage_digest: " << Env("ARIADNE_IMAGE_DIGEST") << '\n';
        std::string per_agent;
        for (int a = 0; a < n_agents; ++a) per_agent += (a ? " + " : "") + std::to_string(integrated[size_t(a)]);
        std::printf("integrated %s keyframes; %zu blocks; ESDF %zu incremental / %zu rebuilds; %zu frontier clusters\n",
                    per_agent.c_str(), map.Blocks(), incremental, rebuilds, fr.Clusters().size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_volumetric_eval: %s\n", e.what());
        return 1;
    }
}
