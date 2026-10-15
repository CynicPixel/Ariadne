// ---------------------------------------------------------------------------
// ariadne_mission: 1.8 placeholder mission layer in replay (design MISSION_1.8).
// Reads the 1.7 spool (keyframe depth + sigma + poses, the 1.6 merge) and the
// mission briefing (entrances = exits, target, rig extrinsic), integrates depth
// into an OctoMap-model voxel grid in the world frame, and every replan period
// computes frontiers, a nearest-frontier decision per agent, the egress route at
// rho_responder (clearance to anything NOT observed free) and the agent route at
// rho_agent (clearance to occupied). No ground truth is read here.
//
//   ariadne_mission --spool DIR --scene DIR --config mission_v1.yaml --out DIR [--agents merged|0|1]
//
// --agents merged (default): agent 0's entry pose anchors W, agent 1 joins through the 1.6 loop.
// --agents 0 / 1: that agent alone, anchored by ITS OWN briefed entry pose (the comparator rows of
// the collaboration ablation; the merged run never uses agent 1's entry pose).
// Outputs: trace.jsonl (one record per decision, mode replay / executed false),
// routes.jsonl (both routes at every replan), slab.png + support.png (final map),
// results.yaml, manifest.yaml.
// ---------------------------------------------------------------------------
#include "planning/frontier/frontiers.hpp"
#include "planning/route/astar.hpp"
#include "planning/volumetric/clearance.hpp"
#include "planning/volumetric/occupancy_grid.hpp"
#include "runtime/manifest/build_identity.hpp"
#include "runtime/manifest/determinism.hpp"
#include "runtime/manifest/opencv_identity.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ariadne;
using planning::Cell;

namespace {

std::string UtcNow(const char* fmt) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

core::SE3 RowMajorPose(const YAML::Node& n) {
    if (!n || n.size() != 12) throw std::runtime_error("pose: expected 12 numbers");
    core::SE3 T;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T.R(i, j) = n[i * 4 + j].as<double>();
        T.t(i) = n[i * 4 + 3].as<double>();
    }
    return T;
}

struct Keyframe {
    int agent;
    uint64_t kf_id;
    int64_t t_ns;
    core::SE3 T_O_C;   // camera in its agent's odometry frame
    double fx, fy, cx, cy;
    int w, h;
    fs::path stem;
};

cv::Mat ReadHalf(const fs::path& p, int w, int h) {
    const cv::Mat bits = cv::imread(p.string(), cv::IMREAD_UNCHANGED);
    if (bits.empty() || bits.type() != CV_16U || bits.cols != w || bits.rows != h) {
        throw std::runtime_error("spool: bad fp16 png " + p.string());
    }
    cv::Mat f;
    cv::Mat(bits.size(), CV_16F, bits.data, bits.step).convertTo(f, CV_32F);
    return f;
}

struct Config {
    std::string name;
    double z_max, max_sigma, max_depth, margin_k, z_lo, z_hi, rho_a, rho_r, replan_s;
    int stride, min_frontier;
};

Config LoadConfig(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    Config c;
    c.name = n["name"].as<std::string>();
    c.z_max = n["grid"]["z_max_m"].as<double>();
    c.stride = n["carving"]["pixel_stride"].as<int>();
    c.max_sigma = n["carving"]["max_sigma_m"].as<double>();
    c.max_depth = n["carving"]["max_depth_m"].as<double>();
    c.margin_k = n["carving"]["surface_margin_k"].as<double>();
    c.z_lo = n["slab"]["z_band_m"][0].as<double>();
    c.z_hi = n["slab"]["z_band_m"][1].as<double>();
    c.rho_a = n["clearance"]["rho_agent_m"].as<double>();
    c.rho_r = n["clearance"]["rho_responder_m"].as<double>();
    c.min_frontier = n["frontier"]["min_cells"].as<int>();
    c.replan_s = n["replan_period_s"].as<double>();
    if (c.stride < 1 || !(c.replan_s > 0) || !(c.z_hi > c.z_lo)) throw std::invalid_argument("mission config: bad values");
    return c;
}

std::string CellsJson(const std::vector<Cell>& cs) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < cs.size(); ++i) o << (i ? "," : "") << "[" << cs[i].r << "," << cs[i].c << "]";
    o << "]";
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string spool_dir, scene_dir, config_path, out;
        std::string agents = "merged";
        for (int i = 1; i < argc; ++i) {
            const std::string k = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
            const std::string v = argv[++i];
            if (k == "--spool") spool_dir = v;
            else if (k == "--scene") scene_dir = v;
            else if (k == "--config") config_path = v;
            else if (k == "--out") out = v;
            else if (k == "--agents") agents = v;
            else throw std::invalid_argument("unknown argument " + k);
        }
        if (spool_dir.empty() || scene_dir.empty() || config_path.empty() || out.empty()) {
            throw std::invalid_argument("usage: ariadne_mission --spool DIR --scene DIR --config Y --out DIR [--agents merged|0|1]");
        }
        if (agents != "merged" && agents != "0" && agents != "1") throw std::invalid_argument("--agents merged|0|1");
        const int anchor_agent = agents == "1" ? 1 : 0;
        auto Uses = [&](int a) { return agents == "merged" || agents == std::to_string(a); };
        const Config cfg = LoadConfig(config_path);

        // --- the mission briefing (no GT layers): grid extent, clearance, exits, target, rig -------------
        const YAML::Node scene = YAML::LoadFile(scene_dir + "/scene.json");
        const YAML::Node mission = YAML::LoadFile(scene_dir + "/mission_default.json");
        const YAML::Node route_a =
            YAML::LoadFile(scene_dir + "/routes/agent_" + std::to_string(anchor_agent) + ".json");
        const std::string scene_id = scene["scene_id"].as<std::string>();
        if (std::abs(scene["clearance"]["rho_agent_m"].as<double>() - cfg.rho_a) > 1e-12 ||
            std::abs(scene["clearance"]["rho_responder_m"].as<double>() - cfg.rho_r) > 1e-12) {
            throw std::runtime_error("config rho values differ from the briefing's scene.json clearance");
        }
        planning::GridSpec gs;
        gs.cell = scene["esdf"]["cell_m"].as<double>();
        gs.origin_x = scene["esdf"]["origin_m"][0].as<double>();
        gs.origin_y = scene["esdf"]["origin_m"][1].as<double>();
        gs.origin_z = 0.0;
        gs.nx = scene["esdf"]["cols"].as<int>();
        gs.ny = scene["esdf"]["rows"].as<int>();
        gs.nz = int(std::ceil(cfg.z_max / gs.cell));
        auto ToCell = [&](double x, double y) {
            return Cell{int(std::floor((y - gs.origin_y) / gs.cell)), int(std::floor((x - gs.origin_x) / gs.cell))};
        };
        // World anchor: the anchor agent's entry pose x rig extrinsic (body <- rectified-left camera, R1 ~ I on
        // the ideal sim rig; the evaluator's 10 s check licenses that). In the merged run that is agent 0 and
        // agent 1's entry pose is NOT used: agent 1 enters W only through the verified loop.
        const YAML::Node e0 = scene["entrances"][anchor_agent];
        if (e0["agent"].as<int>() != anchor_agent) throw std::runtime_error("briefing: entrance order != agent id");
        const double yaw = e0["start_yaw_deg"].as<double>() * M_PI / 180.0;
        core::SE3 T_W_B0;
        T_W_B0.R = core::Mat3::Identity();
        T_W_B0.R(0, 0) = std::cos(yaw);
        T_W_B0.R(0, 1) = -std::sin(yaw);
        T_W_B0.R(1, 0) = std::sin(yaw);
        T_W_B0.R(1, 1) = std::cos(yaw);
        T_W_B0.t = core::Vec3(e0["start_xy_m"][0].as<double>(), e0["start_xy_m"][1].as<double>(),
                              route_a["segments"][0]["start"][2].as<double>());
        core::SE3 T_B_C;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) T_B_C.R(i, j) = mission["R_B_optical_rows"][i][j].as<double>();
            T_B_C.t(i) = mission["t_B_camera_m"]["left"][i].as<double>();
        }
        const core::SE3 T_W_Oa = T_W_B0 * T_B_C;   // the anchor agent's odometry frame in W
        const auto tg = scene["target_xy_m"];
        const Cell target = ToCell(tg[0].as<double>(), tg[1].as<double>());
        std::vector<Cell> exits;
        for (const auto& e : scene["entrances"]) exits.push_back(ToCell(e["start_xy_m"][0].as<double>(), e["start_xy_m"][1].as<double>()));

        // --- the spool ------------------------------------------------------------------------------------
        const YAML::Node sp = YAML::LoadFile(spool_dir + "/spool.yaml");
        if (sp["schema_version"].as<int>() != 1) throw std::runtime_error("spool: schema_version != 1");
        const bool merge_ok = sp["merge"]["ok"].as<bool>();
        const core::SE3 T_O0_O1 = RowMajorPose(sp["merge"]["T_O0_O1"]);
        std::vector<Keyframe> kfs;
        std::map<std::pair<int, uint64_t>, int64_t> t_of;
        for (const auto& b : sp["bundles"]) {
            if (b["role"].as<std::string>() != "keyframe") continue;
            Keyframe k;
            k.agent = b["agent"].as<int>();
            k.kf_id = b["kf_id"].as<uint64_t>();
            k.t_ns = b["t_ns"].as<int64_t>();
            k.T_O_C = RowMajorPose(b["T_WiC"]);
            k.fx = b["K"][0].as<double>();
            k.fy = b["K"][1].as<double>();
            k.cx = b["K"][2].as<double>();
            k.cy = b["K"][3].as<double>();
            k.w = b["width"].as<int>();
            k.h = b["height"].as<int>();
            k.stem = fs::path(spool_dir) / b["files"].as<std::string>();
            t_of[{k.agent, k.kf_id}] = k.t_ns;
            kfs.push_back(k);
        }
        std::sort(kfs.begin(), kfs.end(), [](const Keyframe& a, const Keyframe& b) {
            return std::tie(a.t_ns, a.agent, a.kf_id) < std::tie(b.t_ns, b.agent, b.kf_id);
        });
        // t_merge: the later of the two keyframes of the 1.6 merge edge ([ASSUMPTION]: found then).
        int64_t t_merge = std::numeric_limits<int64_t>::max();
        if (merge_ok && agents == "merged") {
            std::smatch m;
            const std::string edge = sp["merge"]["edge"].as<std::string>();
            if (!std::regex_search(edge, m, std::regex(R"(agent0 kf (\d+) <- agent1 kf (\d+))"))) {
                throw std::runtime_error("spool: cannot parse merge edge '" + edge + "'");
            }
            t_merge = std::max(t_of.at({0, std::stoull(m[1])}), t_of.at({1, std::stoull(m[2])}));
        }
        const int64_t t0 = kfs.front().t_ns;
        auto Sec = [&](int64_t t) { return double(t - t0) * 1e-9; };

        // --- integrate + replan --------------------------------------------------------------------------
        planning::OccupancyGrid grid(gs);
        const int z0 = int(std::floor(cfg.z_lo / gs.cell)), z1 = int(std::ceil(cfg.z_hi / gs.cell));
        fs::create_directories(out);
        const fs::path run_dir = fs::path(out) / (scene_id + (agents == "merged" ? "-" : "-agent" + agents + "only-") +
                                                  UtcNow("%Y%m%dT%H%M%SZ"));
        fs::create_directories(run_dir);
        std::ofstream trace(run_dir / "trace.jsonl"), routes(run_dir / "routes.jsonl");
        trace.precision(6);
        routes.precision(6);
        std::map<int, core::Vec3> agent_pos;   // latest integrated camera position per agent (world)
        std::vector<Keyframe> held;            // agent 1 before the merge
        size_t integrated[2] = {0, 0}, rays = 0;
        double first_valid_s = -1, first_agent_route_s = -1;
        int replans = 0;
        double replan_ms_total = 0;

        auto Integrate = [&](const Keyframe& k) {
            const core::SE3 T_W_C = T_W_Oa * (k.agent == anchor_agent ? k.T_O_C : T_O0_O1 * k.T_O_C);
            const cv::Mat Z = ReadHalf(k.stem.string() + "_depth.png", k.w, k.h);
            const cv::Mat S = ReadHalf(k.stem.string() + "_sigma.png", k.w, k.h);
            const core::Vec3 c = T_W_C.t;
            for (int v = 0; v < k.h; v += cfg.stride) {
                for (int u = 0; u < k.w; u += cfg.stride) {
                    const float z = Z.at<float>(v, u), s = S.at<float>(v, u);
                    if (!std::isfinite(z) || !std::isfinite(s) || z <= 0 || z > cfg.max_depth || s > cfg.max_sigma) continue;
                    const core::Vec3 Xc((u - k.cx) / k.fx * z, (v - k.cy) / k.fy * z, z);
                    const core::Vec3 p = T_W_C * Xc;
                    const double frac = std::max(0.0, (z - cfg.margin_k * s) / z);
                    grid.IntegrateRay(c, c + (p - c) * frac, p);
                    ++rays;
                }
            }
            agent_pos[k.agent] = c;
            ++integrated[k.agent];
        };

        auto Replan = [&](int64_t t_now, bool final_replan) {
            const auto w0 = std::chrono::steady_clock::now();
            const double ts = Sec(t_now);
            const cv::Mat slab = grid.Slab(z0, z1);
            cv::Mat free = slab == planning::kFree, occ = slab == planning::kOccupied;
            free /= 255;
            occ /= 255;
            const cv::Mat not_free = 1 - free;
            const double cap = cfg.rho_r + 2 * gs.cell;
            const cv::Mat clr_egress = planning::EdgeClearance(not_free, gs.cell, cap);
            const cv::Mat clr_agent = planning::EdgeClearance(occ, gs.cell, cap);
            const cv::Mat trav_r = planning::Traversable(free, clr_egress, cfg.rho_r);
            const cv::Mat trav_a = planning::Traversable(free, clr_agent, cfg.rho_a);
            const cv::Mat support = grid.SlabSupport(z0, z1, slab);

            auto Route = [&](const char* kind, const cv::Mat& trav, double rho) {
                const planning::GridPath p = planning::AStar(trav, target, exits);
                double p5 = 0;
                if (p.found) {
                    std::vector<float> sv;
                    for (const auto& q : p.cells) sv.push_back(support.at<float>(q.r, q.c));
                    std::sort(sv.begin(), sv.end());
                    p5 = sv[size_t(std::floor(0.05 * double(sv.size() - 1)))];
                }
                routes << "{\"t_s\":" << ts << ",\"kind\":\"" << kind << "\",\"rho_m\":" << rho << ",\"found\":"
                       << (p.found ? "true" : "false") << ",\"length_m\":" << p.length_cells * gs.cell
                       << ",\"exit\":" << p.goal_index << ",\"support_p5\":" << p5 << ",\"final\":"
                       << (final_replan ? "true" : "false") << ",\"cells\":" << CellsJson(p.cells) << "}\n";
                return p;
            };
            const auto egress = Route("egress", trav_r, cfg.rho_r);
            const auto agent_route = Route("agent", trav_a, cfg.rho_a);
            if (egress.found && first_valid_s < 0) first_valid_s = ts;
            if (agent_route.found && first_agent_route_s < 0) first_agent_route_s = ts;

            const planning::FrontierSet fr = planning::Frontiers(slab, cfg.min_frontier);
            for (int a = 0; a < 2; ++a) {
                trace << "{\"t_s\":" << ts << ",\"agent\":" << a << ",\"mode\":\"replay\",\"executed\":false,"
                      << "\"decision\":\"nearest_frontier\",";
                if (!Uses(a)) {
                    trace << "\"result\":\"no decision: agent not in this run\"}\n";
                    continue;
                }
                if (agents == "merged" && a == 1 && t_now < t_merge) {
                    trace << "\"result\":\"no decision: not merged\"}\n";
                    continue;
                }
                if (!agent_pos.count(a)) {
                    trace << "\"result\":\"no decision: no keyframe yet\"}\n";
                    continue;
                }
                const core::Vec3 pw = agent_pos.at(a);
                const Cell ac = ToCell(pw.x(), pw.y());
                trace << "\"agent_cell\":[" << ac.r << "," << ac.c << "],\"frontier_clusters\":" << fr.clusters.size()
                      << ",\"rejected_small\":" << fr.rejected_small << ",";
                const cv::Mat D = planning::GridDistances(trav_a, ac);
                int best = -1;
                double best_d = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < fr.clusters.size(); ++i) {
                    for (const auto& m : fr.clusters[i].members) {
                        const double d = D.at<double>(m.r, m.c);
                        if (d < best_d) best_d = d, best = int(i);
                    }
                }
                if (best < 0) {
                    trace << "\"result\":\"no reachable frontier at rho_agent\"}\n";
                } else {
                    const auto& k = fr.clusters[size_t(best)];
                    const double cx = gs.origin_x + (k.centroid_c + 0.5) * gs.cell;
                    const double cy = gs.origin_y + (k.centroid_r + 0.5) * gs.cell;
                    trace << "\"result\":\"go\",\"frontier_xy_m\":[" << cx << "," << cy << "],\"frontier_cells\":" << k.cells
                          << ",\"path_m\":" << best_d * gs.cell << "}\n";
                }
            }
            ++replans;
            replan_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
            return std::make_tuple(slab, support, egress, agent_route, fr);
        };

        int64_t next_replan = t0 + int64_t(cfg.replan_s * 1e9);
        for (const auto& k : kfs) {
            while (k.t_ns >= next_replan) {
                Replan(next_replan, false);
                next_replan += int64_t(cfg.replan_s * 1e9);
            }
            if (!Uses(k.agent) || (agents == "merged" && k.agent == 1 && !merge_ok)) continue;
            if (agents == "merged" && k.agent == 1 && k.t_ns < t_merge) {
                held.push_back(k);
                continue;
            }
            if (!held.empty() && k.t_ns >= t_merge) {   // the merge happens: agent 1 shares its history
                for (const auto& h : held) Integrate(h);
                held.clear();
            }
            Integrate(k);
        }
        const auto [slab, support, egress, agent_route, fr] = Replan(kfs.back().t_ns, true);

        cv::imwrite((run_dir / "slab.png").string(), slab);   // 0 unknown, 1 free, 2 occupied; row 0 = south
        cv::Mat sup16;
        support.convertTo(sup16, CV_16U, 65535.0);
        cv::imwrite((run_dir / "support.png").string(), sup16);

        std::ofstream res(run_dir / "results.yaml");
        res.precision(10);
        res << "# ariadne_mission results (design MISSION_1.8). Generated; do not hand-edit.\n"
            << "scene_id: " << scene_id << "\nconfig: " << cfg.name << "\nagents: " << agents
            << "\nanchor_agent: " << anchor_agent << "\n"
            << "keyframes_integrated: {agent0: " << integrated[0] << ", agent1: " << integrated[1] << "}\n"
            << "rays: " << rays << "\n"
            << "t_merge_s: " << (t_merge == std::numeric_limits<int64_t>::max() ? -1.0 : Sec(t_merge)) << "\n"
            << "replans: " << replans << "\nreplan_ms_mean: " << replan_ms_total / std::max(replans, 1) << "\n"
            << "time_to_first_valid_route_s: " << first_valid_s << "   # egress, observed free at rho_responder; -1 = none\n"
            << "time_to_first_agent_route_s: " << first_agent_route_s << "\n"
            << "final_egress: {found: " << egress.found << ", length_m: " << egress.length_cells * gs.cell
            << ", exit: " << egress.goal_index << "}\n"
            << "final_agent_route: {found: " << agent_route.found << ", length_m: " << agent_route.length_cells * gs.cell
            << ", exit: " << agent_route.goal_index << "}\n"
            << "final_frontiers: {clusters: " << fr.clusters.size() << ", rejected_small: " << fr.rejected_small << "}\n"
            << "target_cell: [" << target.r << ", " << target.c << "]\nexit_cells: " << CellsJson(exits) << "\n"
            << "T_W_Oanchor: [";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) res << T_W_Oa.R(i, j) << ", ";
            res << T_W_Oa.t(i) << (i < 2 ? ", " : "]\n");
        }
        res << "T_W_Oanchor_source: briefing (the anchor agent's entry pose x mission t_B_camera_m/R_B_optical_rows)\n"
            << "T_O0_O1: [";
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) res << T_O0_O1.R(i, j) << ", ";
            res << T_O0_O1.t(i) << (i < 2 ? ", " : "]\n");
        }
        res.close();

        runtime::RunManifest man;
        man.run_id = run_dir.filename().string();
        man.created_utc = UtcNow("%Y-%m-%dT%H:%M:%SZ");
        man.git_commit = std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : runtime::kUnknownField;
        man.git_dirty = std::getenv("ARIADNE_GIT_DIRTY") && std::string(std::getenv("ARIADNE_GIT_DIRTY")) == "true";
        man.image_digest = std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : runtime::kUnknownField;
        man.scene = "sim/" + scene_id;
        man.seed_recorded = false;
        man.agent_count = 2;
        man.execution_mode = "replay";
        man.planner_config = cfg.name;
        man.extra["planner"] = "nearest_frontier+astar_placeholder";
        man.extra["spool"] = spool_dir;
        man.extra["spool_run_id"] = sp["run_id"].as<std::string>();
        man.extra["spool_sha256"] = std::getenv("ARIADNE_SPOOL_SHA256") ? std::getenv("ARIADNE_SPOOL_SHA256") : "unknown";
        man.extra["agents"] = agents;
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        man.extra["command"] = cmd;
        runtime::RecordBuildIdentity(man);
        runtime::RecordOpenCvIdentity(man);
        runtime::RecordDeterminismDigests(man);
        std::ofstream(run_dir / "manifest.yaml") << runtime::ToYaml(man);
        std::printf("mission[%s]: merge at %.1f s; first valid egress route %.1f s; final egress %s %.2f m; "
                    "agent route %s %.2f m -> %s\n",
                    agents.c_str(), t_merge == std::numeric_limits<int64_t>::max() ? -1.0 : Sec(t_merge),
                    first_valid_s, egress.found ? "found" : "NOT found", egress.length_cells * gs.cell,
                    agent_route.found ? "found" : "NOT found", agent_route.length_cells * gs.cell, run_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_mission: %s\n", e.what());
        return 1;
    }
}
