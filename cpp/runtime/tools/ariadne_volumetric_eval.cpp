// ---------------------------------------------------------------------------
// ariadne_volumetric_eval: 4.1's evaluation on a 1.7 spool (design VOLUMETRIC_4.1 §6, D110). Replay only, no GT.
//
// The same keyframes, anchor and merge timing as the 1.8 tool's merged run (D45: agent 0's briefed entry pose
// anchors W; agent 1 joins through the 1.6 loop at t_merge, sharing its history then). Each keyframe goes through
// the Phase 4 volumetric stack: TsdfMap::Integrate -> Esdf::Update (incremental) -> FrontierSet3D::Update; each
// stage is timed. At the end the map is projected onto the briefing's 2D grid with 1.8's slab rule over the same z
// band (occupied if any voxel in the band is, free if any is free and none occupied, else unknown), so
// python/analysis/volumetric_eval.py scores it exactly as mission_eval.py scores 1.8's slab.
//
//   ariadne_volumetric_eval --spool DIR --scene DIR --config configs/mission/mission_v1.yaml --out DIR
//
// Writes slab.png (uint8 0 unknown / 1 free / 2 occupied, row 0 = south), results.yaml, manifest.yaml.
// The spool reading duplicates ariadne_mission's; it moves onto L1's shared briefing/spool library when that lands.
// ---------------------------------------------------------------------------
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
#include <regex>
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
        for (int i = 1; i + 1 < argc; i += 2) {
            const std::string k = argv[i], v = argv[i + 1];
            if (k == "--spool") spool_dir = v;
            else if (k == "--scene") scene_dir = v;
            else if (k == "--config") config_path = v;
            else if (k == "--out") out = v;
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
        if (sp["schema_version"].as<int>() != 1) throw std::runtime_error("spool: schema_version != 1");
        const bool merge_ok = sp["merge"]["ok"].as<bool>();
        const core::SE3 T_O0_O1 = RowMajorPose(sp["merge"]["T_O0_O1"]);
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
        int64_t t_merge = std::numeric_limits<int64_t>::max();
        if (merge_ok) {
            std::smatch m;
            const std::string edge = sp["merge"]["edge"].as<std::string>();
            if (!std::regex_search(edge, m, std::regex(R"(agent0 kf (\d+) <- agent1 kf (\d+))"))) {
                throw std::runtime_error("spool: cannot parse merge edge '" + edge + "'");
            }
            t_merge = std::max(t_of.at({0, std::stoull(m[1])}), t_of.at({1, std::stoull(m[2])}));
        }

        // --- the volumetric stack, per keyframe ----------------------------------------------------------------------
        planning::TsdfMap map(tp);
        planning::Esdf esdf(tp.voxel, rho_r + 2 * tp.voxel);
        planning::FrontierParams fp;
        fp.z_lo = z_lo, fp.z_hi = z_hi;
        planning::FrontierSet3D fr(fp);
        Stats s_int, s_esdf, s_fr;
        size_t integrated[2] = {0, 0}, incremental = 0, rebuilds = 0, esdf_voxels = 0;
        std::vector<Kf> held;
        auto Integrate = [&](const Kf& k) {
            const core::SE3 T_W_C = T_W_O0 * (k.agent == 0 ? k.T_O_C : T_O0_O1 * k.T_O_C);
            const cv::Mat Z = ReadHalf(k.stem.string() + "_depth.png", k.w, k.h);
            const cv::Mat S = ReadHalf(k.stem.string() + "_sigma.png", k.w, k.h);
            planning::DepthFrame f;
            f.T_W_C = T_W_C, f.fx = k.fx, f.fy = k.fy, f.cx = k.cx, f.cy = k.cy, f.width = k.w, f.height = k.h;
            f.depth = Z.ptr<float>(), f.sigma = S.ptr<float>();
            auto t0 = std::chrono::steady_clock::now();
            const std::vector<planning::Idx3> dirty = map.Integrate(f);
            s_int.Add(Ms(t0));
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
                if (integrated[0] + integrated[1] == 0) fr.Detect(map);
                else fr.Update(map, lo, hi);
            }
            s_fr.Add(Ms(t0));
            ++integrated[k.agent];
        };
        for (const auto& k : kfs) {
            if (k.agent == 1 && !merge_ok) continue;
            if (k.agent == 1 && k.t_ns < t_merge) {
                held.push_back(k);
                continue;
            }
            if (!held.empty() && k.t_ns >= t_merge) {   // the merge: agent 1 shares its history
                for (const auto& h : held) Integrate(h);
                held.clear();
            }
            Integrate(k);
        }

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
        planning::GridFrame gf;
        gf.ox = ox, gf.oy = oy, gf.cell = cell, gf.cols = cols, gf.rows = rows;
        planning::TravParams trp;
        trp.responder.rho = rho_r;
        trp.agent.rho = cfg["clearance"]["rho_agent_m"].as<double>();
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
        const planning::CoarseLayer coarse = planning::BuildCoarse(map);
        const size_t rle = planning::EncodeCoarseRle(coarse).size();
        std::ofstream res(fs::path(out) / "results.yaml");
        res.precision(6);
        res << "# ariadne_volumetric_eval (VOLUMETRIC_4.1 §6, D110). Generated; host timings are indicative.\n"
            << "keyframes_integrated: {agent0: " << integrated[0] << ", agent1: " << integrated[1] << "}\n"
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
        std::ofstream man(fs::path(out) / "manifest.yaml");
        std::string cmd;
        for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
        auto Env = [](const char* k) { return std::getenv(k) ? std::string(std::getenv(k)) : std::string("UNRECORDED"); };
        man << "# ariadne_volumetric_eval manifest. Generated.\ntool: ariadne_volumetric_eval\ncommand: \"" << cmd
            << "\"\nscene_id: " << scene["scene_id"].as<std::string>() << "\nspool_run_id: " << sp["run_id"].as<std::string>()
            << "\nspool_sha256: " << Env("ARIADNE_SPOOL_SHA256") << "\nconfig: " << cfg["name"].as<std::string>()
            << "\nlayer: phase4_volumetric (TsdfMap D110)\ngit_commit: " << Env("ARIADNE_GIT_COMMIT")
            << "\ngit_dirty: " << Env("ARIADNE_GIT_DIRTY") << "\nimage_digest: " << Env("ARIADNE_IMAGE_DIGEST") << '\n';
        std::printf("integrated %zu + %zu keyframes; %zu blocks; ESDF %zu incremental / %zu rebuilds; %zu frontier clusters\n",
                    integrated[0], integrated[1], map.Blocks(), incremental, rebuilds, fr.Clusters().size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_volumetric_eval: %s\n", e.what());
        return 1;
    }
}
