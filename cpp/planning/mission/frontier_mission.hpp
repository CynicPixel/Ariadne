#pragma once
// ---------------------------------------------------------------------------
// The 1.8 placeholder mission layer as a library (design MISSION_1.8; extracted for the live runtime, design
// LIVE_RUNTIME_L1 D153). One implementation, two callers:
//   ariadne_mission  (batch, replay of a spool; its outputs are byte-identical to before the extraction)
//   ariadne_live     (incremental: per keyframe, per replan, on a sim-time barrier)
//
//   Briefing      the mission briefing (no GT layers): grid extent from scene.json's esdf block, clearance, exits,
//                 target, each agent's entry pose x the rig extrinsic (its odometry frame in W)
//   FrontierMap   OctoMap-model voxel grid; Integrate(keyframe depth + sigma) carves free space up to
//                 margin_k sigma short of the surface; Replan() = slab -> clearance at both rho -> routes at
//                 rho_responder (egress, clearance to anything NOT observed free) and rho_agent (clearance to
//                 occupied) -> frontier clusters
//   Candidates()  per agent, every frontier cluster reachable at rho_agent with its path distance, nearest first:
//                 the deterministic candidate set the chooser picks from (§25.1). Element 0 is the 1.8 policy
//                 (nearest_frontier). Phase 4 replaces this generator; the chooser interface stays.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "planning/frontier/frontiers.hpp"
#include "planning/route/astar.hpp"
#include "planning/volumetric/clearance.hpp"
#include "planning/volumetric/occupancy_grid.hpp"

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::planning::mission {

struct MissionConfig {
    std::string name;
    double z_max = 0, max_sigma = 0, max_depth = 0, margin_k = 0, z_lo = 0, z_hi = 0, rho_a = 0, rho_r = 0,
           replan_s = 0;
    int stride = 1, min_frontier = 1;
};

inline MissionConfig LoadMissionConfig(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    MissionConfig c;
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

struct Briefing {
    std::string scene_id;
    GridSpec gs;
    Cell target;
    std::vector<Cell> exits;           // every entrance, in scene.json order
    std::vector<core::SE3> T_W_O;      // per agent: its odometry frame (rectified left camera at entry) in W
    std::vector<double> entry_z;       // per agent: entry altitude (its route's first pose)

    Cell ToCell(double x, double y) const {
        return Cell{int(std::floor((y - gs.origin_y) / gs.cell)), int(std::floor((x - gs.origin_x) / gs.cell))};
    }
    /// Cell centre in W (x, y).
    std::pair<double, double> CellCentre(const Cell& c) const {
        return {gs.origin_x + (c.c + 0.5) * gs.cell, gs.origin_y + (c.r + 0.5) * gs.cell};
    }
};

/// The briefing of `scene_dir` (scene.json, mission_<mission>.json, routes/agent_<k>.json). World anchor of agent k:
/// its entry pose x the rig extrinsic (body <- rectified-left camera, R1 ~ I on the ideal sim rig).
inline Briefing LoadBriefing(const std::string& scene_dir, const MissionConfig& cfg, int n_agents = 2,
                             const std::string& mission = "default") {
    const YAML::Node scene = YAML::LoadFile(scene_dir + "/scene.json");
    const YAML::Node m = YAML::LoadFile(scene_dir + "/mission_" + mission + ".json");
    Briefing b;
    b.scene_id = scene["scene_id"].as<std::string>();
    if (std::abs(scene["clearance"]["rho_agent_m"].as<double>() - cfg.rho_a) > 1e-12 ||
        std::abs(scene["clearance"]["rho_responder_m"].as<double>() - cfg.rho_r) > 1e-12) {
        throw std::runtime_error("config rho values differ from the briefing's scene.json clearance");
    }
    b.gs.cell = scene["esdf"]["cell_m"].as<double>();
    b.gs.origin_x = scene["esdf"]["origin_m"][0].as<double>();
    b.gs.origin_y = scene["esdf"]["origin_m"][1].as<double>();
    b.gs.origin_z = 0.0;
    b.gs.nx = scene["esdf"]["cols"].as<int>();
    b.gs.ny = scene["esdf"]["rows"].as<int>();
    b.gs.nz = int(std::ceil(cfg.z_max / b.gs.cell));
    core::SE3 T_B_C;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) T_B_C.R(i, j) = m["R_B_optical_rows"][i][j].as<double>();
        T_B_C.t(i) = m["t_B_camera_m"]["left"][i].as<double>();
    }
    for (int k = 0; k < n_agents; ++k) {
        const YAML::Node e = scene["entrances"][k];
        if (e["agent"].as<int>() != k) throw std::runtime_error("briefing: entrance order != agent id");
        const YAML::Node route = YAML::LoadFile(scene_dir + "/routes/agent_" + std::to_string(k) + ".json");
        const double yaw = e["start_yaw_deg"].as<double>() * M_PI / 180.0;
        core::SE3 T_W_B;
        T_W_B.R = core::Mat3::Identity();
        T_W_B.R(0, 0) = std::cos(yaw);
        T_W_B.R(0, 1) = -std::sin(yaw);
        T_W_B.R(1, 0) = std::sin(yaw);
        T_W_B.R(1, 1) = std::cos(yaw);
        T_W_B.t = core::Vec3(e["start_xy_m"][0].as<double>(), e["start_xy_m"][1].as<double>(),
                             route["segments"][0]["start"][2].as<double>());
        b.T_W_O.push_back(T_W_B * T_B_C);
        b.entry_z.push_back(T_W_B.t.z());
    }
    const auto tg = scene["target_xy_m"];
    b.target = b.ToCell(tg[0].as<double>(), tg[1].as<double>());
    for (const auto& e : scene["entrances"]) b.exits.push_back(b.ToCell(e["start_xy_m"][0].as<double>(), e["start_xy_m"][1].as<double>()));
    return b;
}

/// One replan's maps and routes (everything the 1.8 tool writes or decides from).
struct ReplanMaps {
    cv::Mat slab, support, trav_r, trav_a;
    GridPath egress, agent_route;
    FrontierSet frontiers;
};

/// A deterministic candidate: one frontier cluster reachable at rho_agent from the agent's cell.
struct Candidate {
    int cluster = -1;
    Cell goal;                  // the cluster member nearest by path (the cell a "go" drives to)
    double path_cells = 0;      // grid path distance to `goal`
    double centroid_x = 0, centroid_y = 0;
    int cells = 0;
};

class FrontierMap {
public:
    FrontierMap(const MissionConfig& cfg, const Briefing& b)
        : cfg_(cfg), b_(b), grid_(b.gs), z0_(int(std::floor(cfg.z_lo / b.gs.cell))), z1_(int(std::ceil(cfg.z_hi / b.gs.cell))) {}

    /// Carves one keyframe: depth Z and its sigma S (CV_32F, metres), pinhole K, the camera pose in W. Returns rays.
    size_t Integrate(const core::SE3& T_W_C, const cv::Mat& Z, const cv::Mat& S, double fx, double fy, double cx,
                     double cy) {
        const core::Vec3 c = T_W_C.t;
        size_t rays = 0;
        for (int v = 0; v < Z.rows; v += cfg_.stride) {
            for (int u = 0; u < Z.cols; u += cfg_.stride) {
                const float z = Z.at<float>(v, u), s = S.at<float>(v, u);
                if (!std::isfinite(z) || !std::isfinite(s) || z <= 0 || z > cfg_.max_depth || s > cfg_.max_sigma) continue;
                const core::Vec3 Xc((u - cx) / fx * z, (v - cy) / fy * z, z);
                const core::Vec3 p = T_W_C * Xc;
                const double frac = std::max(0.0, (z - cfg_.margin_k * s) / z);
                grid_.IntegrateRay(c, c + (p - c) * frac, p);
                ++rays;
            }
        }
        return rays;
    }

    ReplanMaps Replan() const {
        ReplanMaps r;
        const GridSpec& gs = b_.gs;
        r.slab = grid_.Slab(z0_, z1_);
        cv::Mat free = r.slab == kFree, occ = r.slab == kOccupied;
        free /= 255;
        occ /= 255;
        const cv::Mat not_free = 1 - free;
        const double cap = cfg_.rho_r + 2 * gs.cell;
        const cv::Mat clr_egress = EdgeClearance(not_free, gs.cell, cap);
        const cv::Mat clr_agent = EdgeClearance(occ, gs.cell, cap);
        r.trav_r = Traversable(free, clr_egress, cfg_.rho_r);
        r.trav_a = Traversable(free, clr_agent, cfg_.rho_a);
        r.support = grid_.SlabSupport(z0_, z1_, r.slab);
        r.egress = AStar(r.trav_r, b_.target, b_.exits);
        r.agent_route = AStar(r.trav_a, b_.target, b_.exits);
        r.frontiers = Frontiers(r.slab, cfg_.min_frontier);
        return r;
    }

    /// Every cluster reachable at rho_agent from `from`, nearest first (stable: ties keep cluster order, so element 0
    /// is exactly the 1.8 nearest_frontier choice).
    std::vector<Candidate> Candidates(const ReplanMaps& r, const Cell& from) const {
        const cv::Mat D = GridDistances(r.trav_a, from);
        std::vector<Candidate> out;
        for (size_t i = 0; i < r.frontiers.clusters.size(); ++i) {
            const auto& k = r.frontiers.clusters[i];
            Candidate c;
            c.cluster = int(i);
            c.path_cells = std::numeric_limits<double>::infinity();
            for (const auto& m : k.members) {
                const double d = D.at<double>(m.r, m.c);
                if (d < c.path_cells) c.path_cells = d, c.goal = m;
            }
            if (!std::isfinite(c.path_cells)) continue;
            c.centroid_x = b_.gs.origin_x + (k.centroid_c + 0.5) * b_.gs.cell;
            c.centroid_y = b_.gs.origin_y + (k.centroid_r + 0.5) * b_.gs.cell;
            c.cells = k.cells;
            out.push_back(c);
        }
        std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.path_cells < b.path_cells; });
        return out;
    }

    const OccupancyGrid& grid() const { return grid_; }
    const Briefing& briefing() const { return b_; }
    const MissionConfig& config() const { return cfg_; }

private:
    MissionConfig cfg_;
    Briefing b_;
    OccupancyGrid grid_;
    int z0_, z1_;
};

}  // namespace ariadne::planning::mission
