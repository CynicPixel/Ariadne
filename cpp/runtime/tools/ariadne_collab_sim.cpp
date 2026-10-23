// ---------------------------------------------------------------------------
// ariadne_collab_sim: pass B of the 3.2 sim integration (design SIMCOLLAB_3.2, D171). The v2 protocol (CollabNode)
// over the 3.3 comms node (CommsNode: RFComms-derived links, walls, scripted partitions), as a seeded discrete-event
// run on the recording clock, fed by the pass-A keyframe stores (ariadne_collab_passa).
//
//   ariadne_collab_sim --stores DIR --rec RECDIR --scene SCENEDIR --config YAML --out DIR --seed S
//       --aliked-onnx ONNX --lightglue-onnx ONNX
//
// Each step of dt sim seconds: deliver the network up to t, release every store record whose t_final <= t whole
// (CollabNode::AddKeyframe), tick every node. After the last release, `drain_s` more, then the COLLAB_3.2 §6 outcome.
//
// Hooks (design §2):
//   retrieve    cosine over the broker's RECEIVED global descriptors (EigenPlaces R50-256), other agents only,
//               >= retrieval.min_score, top retrieval.top_k by score; the scores are cached for `similarity`
//   features    the agent's own store record: ALIKED-n16 keypoints, descriptors int8 + per-keyframe scale (arch §15.7;
//               fp16 the ablation, `local_encoding`), 3D points, camera
//   verify      LightGlue matches -> VerifyMatched (E gate, PnP) -> the D71 gate with the loop noise model
//               (GatedInterAgentEdge) -> the edge A -> B with its information
//   solve       RobustMultiAgentSolve (PCM -> chordal re-init -> GNC-TLS, D76) or MultiAgentSolve (robust: false)
//   select      arm: similarity (unset: request every candidate) | greedy (SelectBySimilarity) | spectral
//
// Outputs in DIR: ledger.csv (every message; first transmission vs retransmit), ate_vs_bytes.csv (one row per sim
// second: cumulative wire bytes, first-transmission and all, and the merged ATE of what the agents adopted),
// results.yaml (per-class bytes and predictions, edges, solves, ATE solo and merged, timings, link stats),
// outcome.yaml (consistent | explicitly_incomplete | corrupt, with reasons), manifest.yaml.
// GROUND TRUTH IS EVALUATION ONLY: it is read when scoring a row, never by a hook.
// ---------------------------------------------------------------------------
#include "collaboration/broker/broker_solve.hpp"
#include "collaboration/broker/edge_verifier.hpp"
#include "collaboration/candidate_selection/selection.hpp"
#include "collaboration/candidate_selection/spectral_selector.hpp"
#include "collaboration/comms/comms_node.hpp"
#include "collaboration/protocol/collab_node.hpp"
#include "collaboration/protocol/outcome.hpp"
#include "collaboration/retrieval/local_descriptor_codec.hpp"
#include "perception/association/hard_pair_matcher.hpp"
#include "runtime/agent/collab_store.hpp"
#include "runtime/datasets/scene_walls.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"
#include "slam/robust/joint_robust.hpp"

#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace ariadne;
namespace ac = ariadne::collaboration;
namespace fs = std::filesystem;

namespace {

struct Config {
    std::string arm = "spectral";             // similarity | greedy | spectral
    double budget_bytes_per_s = 0, budget_burst_bytes = 0;
    std::string budget_source;                // how the level was derived (D171 §4): recorded, never defaulted
    double response_ratio = 0;                // calibrated MATCH_RESPONSE compression ratio; <= 0 = uncalibrated
    double edge_ratio = 0, graph_ratio = 0;   // POSE_EDGE / POSE_GRAPH ratios, same calibration run (D171)
    std::string calibration_run;
    std::string comms_scenario;
    double duplicate_p = 0, reorder_p = 0, jitter_s = 0;
    double dt_s = 0.02, drain_s = 30.0, ate_every_s = 1.0;
    double min_score = 0.5;
    int top_k = 3;
    ac::VerifyParams verify;
    slam::DegeneracyParams gate;
    bool robust = true;
    double selection_interval_s = 5.0, candidate_ttl_s = 120.0;
    int refine_passes = 2;
    std::string local_encoding = "int8";      // arch §15.7 default; fp16 = the ablation row (D171)
    std::string verify_calibration;           // where verify_min_pnp_inliers came from (manifest)
};

Config LoadConfig(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    Config c;
    const auto get = [&](const YAML::Node& m, const char* k, auto& v) {
        if (m[k]) v = m[k].as<std::decay_t<decltype(v)>>();
    };
    get(n, "arm", c.arm);
    if (c.arm != "similarity" && c.arm != "greedy" && c.arm != "spectral")
        throw std::invalid_argument(path + ": arm: similarity | greedy | spectral");
    get(n, "budget_bytes_per_s", c.budget_bytes_per_s);
    get(n, "budget_burst_bytes", c.budget_burst_bytes);
    get(n, "budget_source", c.budget_source);
    if (c.arm != "similarity" && (c.budget_bytes_per_s <= 0 || c.budget_burst_bytes <= 0 || c.budget_source.empty()))
        throw std::invalid_argument(path + ": a budgeted arm needs budget_bytes_per_s, budget_burst_bytes, budget_source");
    get(n, "response_ratio", c.response_ratio);
    get(n, "calibration_run", c.calibration_run);
    get(n, "edge_ratio", c.edge_ratio);
    get(n, "graph_ratio", c.graph_ratio);
    get(n, "comms_scenario", c.comms_scenario);
    if (c.comms_scenario.empty()) throw std::invalid_argument(path + ": comms_scenario is required");
    if (const YAML::Node f = n["faults"]) get(f, "duplicate_p", c.duplicate_p), get(f, "reorder_p", c.reorder_p),
                                          get(f, "jitter_s", c.jitter_s);
    get(n, "dt_s", c.dt_s);
    get(n, "drain_s", c.drain_s);
    get(n, "ate_every_s", c.ate_every_s);
    if (const YAML::Node r = n["retrieval"]) get(r, "min_score", c.min_score), get(r, "top_k", c.top_k);
    if (const YAML::Node v = n["verify"]) {
        get(v, "min_matches", c.verify.min_matches);
        get(v, "essential_threshold_px", c.verify.essential_threshold_px);
        get(v, "min_essential_inliers", c.verify.min_essential_inliers);
        get(v, "min_essential_inlier_ratio", c.verify.min_essential_inlier_ratio);
        get(v, "pnp_reprojection_px", c.verify.pnp_reprojection_px);
        get(v, "min_pnp_inliers", c.verify.min_pnp_inliers);
    }
    // The calibrated PnP floor (D171 §5 procedure, from the OTHER scene); the sweep writes it as one top-level key.
    if (n["verify_min_pnp_inliers"]) c.verify.min_pnp_inliers = n["verify_min_pnp_inliers"].as<int>();
    get(n, "verify_calibration", c.verify_calibration);
    if (const YAML::Node g = n["gate"]) {
        get(g, "noise_model", c.gate.noise_model);
        get(g, "noise_k_t", c.gate.noise_k_t);
        get(g, "noise_floor_t_m", c.gate.noise_floor_t_m);
        get(g, "noise_k_r", c.gate.noise_k_r);
        get(g, "noise_floor_r_rad", c.gate.noise_floor_r_rad);
        get(g, "sigma_px", c.gate.sigma_px);
        get(g, "sigma_disparity_px", c.gate.sigma_disparity_px);
    }
    get(n, "robust", c.robust);
    get(n, "selection_interval_s", c.selection_interval_s);
    get(n, "candidate_ttl_s", c.candidate_ttl_s);
    get(n, "refine_passes", c.refine_passes);
    get(n, "local_encoding", c.local_encoding);
    ac::ParseLocalEncoding(c.local_encoding);
    return c;
}

/// The response's descriptors as the broker decodes them (arch §15.7 encodings; D171: int8 + per-keyframe scale).
perception::HardPairFeatures ToHardPair(const core::FeatureResponseValue& r, ac::LocalEncoding enc) {
    perception::HardPairFeatures f;
    f.size = cv::Size(1280, 720);
    for (size_t i = 0; i + 1 < r.keypoints.size(); i += 2) f.keypoints.emplace_back(r.keypoints[i], r.keypoints[i + 1], 1.0f);
    const std::vector<float> d = ac::DecodeLocal(r.descriptors, enc);
    cv::Mat(int(d.size() / 128), 128, CV_32F, const_cast<float*>(d.data())).copyTo(f.descriptors);
    return f;
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += double(a[i]) * b[i], na += double(a[i]) * a[i], nb += double(b[i]) * b[i];
    return na > 0 && nb > 0 ? d / std::sqrt(na * nb) : 0.0;
}

core::Timestamp At(int64_t ns) { return core::Timestamp{ns}; }

}  // namespace

int main(int argc, char** argv) {
    std::string stores_dir, rec, scene, config_path, out, aliked_onnx, lightglue_onnx;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); return 2; }
        const std::string v = argv[++i];
        if (k == "--stores") stores_dir = v;
        else if (k == "--rec") rec = v;
        else if (k == "--scene") scene = v;
        else if (k == "--config") config_path = v;
        else if (k == "--out") out = v;
        else if (k == "--seed") seed = std::stoull(v);
        else if (k == "--aliked-onnx") aliked_onnx = v;
        else if (k == "--lightglue-onnx") lightglue_onnx = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (stores_dir.empty() || rec.empty() || scene.empty() || config_path.empty() || out.empty() || aliked_onnx.empty() ||
        lightglue_onnx.empty()) {
        std::fprintf(stderr, "usage: see the header of ariadne_collab_sim.cpp\n");
        return 2;
    }
    try {
        const Config cfg = LoadConfig(config_path);
        fs::create_directories(out);

        // --- the stores (pass A), one per agent -------------------------------------------------------------
        std::map<ac::AgentId, runtime::CollabStore> stores;
        for (const auto& e : fs::directory_iterator(stores_dir)) {
            const std::string name = e.path().filename().string();
            if (name.rfind("store_agent", 0) != 0 || e.path().extension() != ".bin") continue;
            runtime::CollabStore s = runtime::ReadCollabStore(e.path().string());
            stores[s.agent] = std::move(s);
        }
        if (stores.size() < 2) throw std::runtime_error("need at least two agent stores in " + stores_dir);
        std::vector<ac::AgentId> fleet;
        for (const auto& [a, s] : stores) fleet.push_back(a);
        std::map<ac::AgentId, std::map<uint64_t, const runtime::CollabStoreRecord*>> rec_of;
        struct Release { int64_t t; ac::AgentId agent; const runtime::CollabStoreRecord* r; };
        std::vector<Release> releases;
        int64_t t_first = INT64_MAX, t_last = INT64_MIN;
        for (const auto& [a, s] : stores)
            for (const auto& r : s.records) {
                rec_of[a][r.id] = &r;
                releases.push_back({r.t_final_ns, a, &r});
                t_first = std::min(t_first, r.t_create_ns);
                t_last = std::max(t_last, r.t_final_ns);
            }
        std::stable_sort(releases.begin(), releases.end(), [](const Release& x, const Release& y) {
            return std::tie(x.t, x.agent, x.r->id) < std::tie(y.t, y.agent, y.r->id);
        });

        // --- the link: CommsNode over the scenario, GT body tracks and walls (the radios' true positions) ------
        const ac::CommsScenario scenario = ac::LoadCommsScenario(cfg.comms_scenario);
        const auto tracks = runtime::LoadAgentTracks(rec, scene);
        int64_t origin = INT64_MAX;
        for (const auto& [a, tr] : tracks) origin = std::min(origin, tr.t_ns.front());
        const std::unique_ptr<ac::LinkModel> base = ac::MakeLinkModel(scenario, tracks, runtime::LoadSceneWalls(scene));
        ac::CommsNode comms(*base, seed, At(origin), {cfg.duplicate_p, cfg.reorder_p, 0.05, cfg.jitter_s});
        comms.Apply(scenario);

        // --- hooks ----------------------------------------------------------------------------------------------
        static const core::WireCodec codec(core::Framing::kFragmented);
        perception::HardPairParams hp;
        hp.aliked_onnx = aliked_onnx;
        hp.lightglue_onnx = lightglue_onnx;
        hp.width = 1280, hp.height = 720;
        const perception::HardPairMatcher lightglue(hp);
        std::map<std::pair<ac::KeyframeRef, ac::KeyframeRef>, double> score_cache;
        struct VerifyStats { uint64_t calls = 0, verified = 0, gated = 0; std::map<std::string, uint64_t> reasons;
                             double ms = 0; } vs;
        double solve_ms = 0;
        // Every verify call with its ground-truth edge error: EVALUATION ONLY (threshold calibration on the other
        // scene, accepted-edge precision). Written after the decision; nothing reads it back.
        std::ofstream ecsv(fs::path(out) / "edges.csv");
        ecsv << "agent_a,kf_a,agent_b,kf_b,score,matches,essential_inliers,pnp_inliers,reason,gt_err_t_m,gt_err_r_deg,"
                "gt_baseline_m,pose_err_deg\n";
        const auto gt_rel = [&](const ac::KeyframeRef& a, const ac::KeyframeRef& b) -> std::optional<core::SE3> {
            const auto* ra = rec_of.at(a.agent).at(a.keyframe);
            const auto* rb = rec_of.at(b.agent).at(b.keyframe);
            if (!ra->has_gt || !rb->has_gt) return std::nullopt;
            return (ra->gt_T_W_B * ra->T_B_C).inverse() * (rb->gt_T_W_B * rb->T_B_C);   // T_Ca_Cb, common GT world
        };
        const auto log_edge = [&](const ac::KeyframeRef& a, const ac::KeyframeRef& b, const ac::VerifyResult* v,
                                  const std::string& reason, const core::SE3* Z) {
            // pose_err_deg: D63's protocol for this matcher (LightGlue/SuperGlue relative-pose AUC): the larger of the
            // rotation error and the angle between the estimated and true translation directions; rotation only when
            // the true baseline is under 0.1 m (the direction is undefined there, D63's tiny-baseline caveat).
            double et = NAN, er = NAN, base = NAN, pe = NAN;
            if (const auto g = gt_rel(a, b); g && Z) {
                const core::SE3 d = g->inverse() * *Z;
                et = d.t.norm();
                er = std::acos(std::clamp((d.R.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
                base = g->t.norm();
                pe = er;
                if (base >= 0.1 && Z->t.norm() > 1e-9)
                    pe = std::max(pe, std::acos(std::clamp(g->t.normalized().dot(Z->t.normalized()), -1.0, 1.0)) *
                                          180.0 / M_PI);
            }
            const auto sc = score_cache.find({a, b});
            ecsv << a.agent << ',' << a.keyframe << ',' << b.agent << ',' << b.keyframe << ','
                 << (sc == score_cache.end() ? NAN : sc->second) << ',' << (v ? v->matches : 0) << ','
                 << (v ? v->essential_inliers : 0) << ',' << (v ? v->pnp_inliers : 0) << ',' << reason << ',' << et << ','
                 << er << ',' << base << ',' << pe << '\n';
        };
        uint64_t solve_calls = 0, robust_inter = 0, robust_inliers = 0;

        ac::NodeParams np;
        np.selection_interval_s = cfg.selection_interval_s;
        np.candidate_ttl_s = cfg.candidate_ttl_s;
        if (cfg.arm != "similarity") np.budget_bytes_per_s = cfg.budget_bytes_per_s, np.budget_burst_bytes = cfg.budget_burst_bytes;
        const ac::LocalEncoding enc = ac::ParseLocalEncoding(cfg.local_encoding);
        const uint16_t feature_bytes = uint16_t(128 * ac::BytesPerDim(enc));
        if (cfg.response_ratio > 0) np.response_cost.response_ratio[feature_bytes] = cfg.response_ratio,
                                    np.response_cost.calibration_run = cfg.calibration_run;
        if (cfg.edge_ratio > 0) np.response_cost.class_ratio[int(core::MessageClass::POSE_EDGE)] = cfg.edge_ratio;
        if (cfg.graph_ratio > 0) np.response_cost.class_ratio[int(core::MessageClass::POSE_GRAPH)] = cfg.graph_ratio;
        np.default_feature_bytes = feature_bytes;

        const slam::JointRobustParams robust_params{};
        auto solve_robust = ac::RobustMultiAgentSolve(robust_params, [&](const slam::JointRobustReport& r) {
            robust_inter += r.inter_agent.size();
            robust_inliers += uint64_t(r.inter_inliers);
        });

        std::vector<std::unique_ptr<ac::CollabNode>> nodes;
        for (ac::AgentId self : fleet) {
            ac::NodeHooks h;
            h.retrieve = [&, self](const ac::KeyframeRef& q, const std::map<ac::KeyframeRef, core::GlobalDescriptorValue>& db) {
                std::vector<std::pair<double, ac::KeyframeRef>> scored;
                const auto qi = db.find(q);
                if (qi == db.end()) return std::vector<ac::KeyframeRef>{};
                for (const auto& [k, d] : db) {
                    if (k.agent == q.agent) continue;
                    const double s = Cosine(qi->second.descriptor, d.descriptor);
                    if (s >= cfg.min_score) scored.push_back({s, k});
                }
                std::stable_sort(scored.begin(), scored.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
                if (int(scored.size()) > cfg.top_k) scored.resize(size_t(cfg.top_k));
                std::vector<ac::KeyframeRef> outk;
                for (const auto& [s, k] : scored) {
                    score_cache[{q, k}] = score_cache[{k, q}] = s;
                    outk.push_back(k);
                }
                return outk;
            };
            h.similarity = [&](const ac::KeyframeRef& q, const ac::KeyframeRef& c) {
                const auto it = score_cache.find({q, c});
                return it == score_cache.end() ? 1.0 : std::clamp(it->second, 1e-6, 1.0);
            };
            h.features = [&, self](uint64_t kf) {
                const runtime::CollabStoreRecord& r = *rec_of.at(self).at(kf);
                core::FeatureResponseValue v;
                v.keyframe_id = kf;
                v.keypoints = r.keypoints;
                v.descriptors = ac::EncodeLocal(r.descriptors, enc);
                v.feature_bytes = feature_bytes;
                v.header_bytes = enc == ac::LocalEncoding::kInt8 ? 4 : 0;
                v.points3d = r.points3d;
                v.camera.assign(r.camera.begin(), r.camera.end());
                return v;
            };
            h.verify = [&, self](const ac::KeyframeRef& a, const core::FeatureResponseValue& fa, const ac::KeyframeRef& b,
                                 const core::FeatureResponseValue& fb) -> std::optional<core::LoopEdgeValue> {
                const auto t0 = std::chrono::steady_clock::now();
                ++vs.calls;
                const auto done = [&](const std::string& why) {
                    ++vs.reasons[why];
                    vs.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                };
                const perception::HardPairFeatures A = ToHardPair(fa, enc), B = ToHardPair(fb, enc);
                if (A.keypoints.size() < 8 || B.keypoints.size() < 8)
                    return log_edge(a, b, nullptr, "too_few_features", nullptr), done("too_few_features"), std::nullopt;
                const std::vector<cv::DMatch> m = lightglue.Match(A, B);
                std::vector<cv::Point2f> pa, pb;
                std::vector<int> ib;
                for (const auto& d : m) {
                    pa.push_back(A.keypoints[size_t(d.queryIdx)].pt);
                    pb.push_back(B.keypoints[size_t(d.trainIdx)].pt);
                    ib.push_back(d.trainIdx);
                }
                const uint64_t vseed = seed * 1000003ull + a.agent * 104729ull + a.keyframe * 7919ull + b.agent * 31ull + b.keyframe;
                const ac::VerifyResult v = ac::VerifyMatched(pa, pb, ib, fb, cfg.verify, vseed);
                if (!v.ok) return log_edge(a, b, &v, v.reason, v.reason == "pnp_gate" ? &v.T_A_B : nullptr),
                                  done(v.reason), std::nullopt;
                ++vs.verified;
                slam::LoopGeometry geo;
                geo.rig.cam = {fb.camera[0], fb.camera[1], fb.camera[2], fb.camera[3]};
                geo.rig.baseline = fb.camera[4];
                for (const auto& q : v.inlier_points_B) geo.X_B.push_back(core::Vec3(q.x, q.y, q.z));
                for (const auto& q : v.inlier_pixels_A) geo.uv_A.push_back(core::Vec2(q.x, q.y));
                geo.T_A_B = v.T_A_B;
                const auto edge = slam::GatedInterAgentEdge(0, 1, geo, cfg.gate);
                if (!edge) return log_edge(a, b, &v, "degenerate", &v.T_A_B), done("degenerate"), std::nullopt;
                log_edge(a, b, &v, "verified", &edge->Z);
                ++vs.gated;
                done("verified");
                core::LoopEdgeValue e;
                e.agent_a = a.agent;
                e.keyframe_a = a.keyframe;
                e.agent_b = b.agent;
                e.keyframe_b = b.keyframe;
                e.measurement = ac::ToPoseValue(edge->Z);
                e.information = ac::PackInformation(edge->information);
                e.frame_a = {a.agent, 0};
                e.frame_b = {b.agent, 0};
                e.status = wire::EdgeStatus::GEOMETRY_VERIFIED;
                e.inlier_count = uint32_t(v.pnp_inliers);
                return e;
            };
            h.solve = [&](const ac::SolveInput& in) {
                const auto t0 = std::chrono::steady_clock::now();
                ac::SolveOutput o = cfg.robust ? solve_robust(in) : ac::MultiAgentSolve(in);
                solve_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                ++solve_calls;
                return o;
            };
            if (cfg.arm == "greedy") h.select = [](const ac::SelectionProblem& P) { return ac::SelectBySimilarity(P); };
            if (cfg.arm == "spectral") {
                ac::SelectorParams sp;
                sp.refine_passes = cfg.refine_passes;
                h.select = [sp](const ac::SelectionProblem& P) { return ac::SelectSpectral(P, sp); };
            }
            nodes.push_back(std::make_unique<ac::CollabNode>(self, fleet, codec, comms.ledger(), std::move(h), np));
        }
        std::map<ac::AgentId, ac::CollabNode*> node_of;
        for (auto& n : nodes) node_of[n->self()] = n.get();

        // --- scoring (evaluation only) -------------------------------------------------------------------------
        std::map<ac::AgentId, std::vector<const runtime::CollabStoreRecord*>> released;
        // Agents share a frame only if they adopted estimates of the same anchor (G = W_anchor, held across epochs,
        // COLLAB_3.2 §5). After a split brain each side has its own G: scored is the frame holding the most released
        // keyframes (ties: the lower anchor), never a mix of frames.
        auto merged_ate = [&](size_t& merged_agents, size_t& kf_merged, size_t& kf_total, ac::AgentId* frame = nullptr) {
            std::map<ac::AgentId, size_t> kf_in;   // anchor -> released keyframes of agents in that frame
            kf_total = 0;
            for (const auto& [a, rs] : released) {
                kf_total += rs.size();
                if (node_of.at(a)->T_G_W()) kf_in[node_of.at(a)->T_G_W_anchor()] += rs.size();
            }
            merged_agents = kf_merged = 0;
            if (kf_in.empty()) return runtime::ErrorStats{};
            ac::AgentId best = kf_in.begin()->first;
            for (const auto& [anc, n] : kf_in) if (n > kf_in.at(best)) best = anc;
            if (frame) *frame = best;
            std::vector<core::TrajectorySample> est, gt;
            for (const auto& [a, rs] : released) {
                const auto T = node_of.at(a)->T_G_W();
                if (!T || node_of.at(a)->T_G_W_anchor() != best) continue;
                ++merged_agents;
                for (const auto* r : rs) {
                    if (!r->has_gt) continue;
                    const core::SE3 T_G_B = *T * r->T_W_C * r->T_B_C.inverse();
                    est.push_back({core::Timestamp{r->t_create_ns}, r->id, T_G_B});
                    gt.push_back({core::Timestamp{r->t_create_ns}, r->id, r->gt_T_W_B});
                    ++kf_merged;
                }
            }
            return runtime::ComputeATE(est, gt);
        };

        // --- the run -------------------------------------------------------------------------------------------
        std::ofstream ab(fs::path(out) / "ate_vs_bytes.csv");
        ab << "t_s,first_tx_wire_bytes,all_wire_bytes,merged_agents,kf_merged,kf_released,ate_rmse_m\n";
        const auto cum = [&](bool first_only) {
            uint64_t s = 0;
            for (const auto& r : comms.ledger().Records())
                if (!first_only || !r.retransmit) s += r.wire_bytes;
            return s;
        };
        const int64_t dt = int64_t(std::llround(cfg.dt_s * 1e9));
        const int64_t t_end = t_last + int64_t(std::llround(cfg.drain_s * 1e9));
        int64_t next_ate = t_first;
        size_t next_rel = 0;
        core::Timestamp now = At(t_first);
        for (int64_t t = t_first; t <= t_end; t += dt) {
            now = At(t);
            for (const auto& d : comms.network().DeliverUntil(now)) node_of.at(d.to)->OnDatagram(d, now);
            for (; next_rel < releases.size() && releases[next_rel].t <= t; ++next_rel) {
                const Release& rl = releases[next_rel];
                const runtime::CollabStoreRecord& r = *rl.r;
                std::optional<ac::OdometryEdge> odo;
                if (r.odo_from) {
                    core::Mat6 I;
                    for (int q = 0; q < 36; ++q) I(q / 6, q % 6) = r.odo_information[size_t(q)];
                    odo = ac::OdometryEdge{*r.odo_from, r.odo_Z, I};
                }
                core::GlobalDescriptorValue g;
                g.keyframe_id = r.id;
                g.stamp_ns = r.t_create_ns;
                g.descriptor = r.global;
                g.n_features = uint32_t(r.keypoints.size() / 2);
                g.feature_bytes = feature_bytes;
                g.fp16 = true;   // arch §15.7's encoding (D171)
                node_of.at(rl.agent)->AddKeyframe(r.id, r.t_create_ns, r.T_W_C, odo, g, now);
                released[rl.agent].push_back(&r);
            }
            for (auto& n : nodes) n->Tick(now, comms.network());
            if (t >= next_ate) {
                size_t ma = 0, km = 0, kt = 0;
                const runtime::ErrorStats e = merged_ate(ma, km, kt);
                ab << 1e-9 * double(t - t_first) << ',' << cum(true) << ',' << cum(false) << ',' << ma << ',' << km << ','
                   << kt << ',' << (e.count ? e.rmse : NAN) << '\n';
                next_ate += int64_t(std::llround(cfg.ate_every_s * 1e9));
            }
        }

        // --- outputs -------------------------------------------------------------------------------------------
        {
            std::ofstream l(fs::path(out) / "ledger.csv");
            l << "t_s,sender,receiver,class,payload_bytes,transmitted_bytes,wire_bytes,compressed,retransmit,resync,"
                 "predicted_wire_bytes\n";
            for (const auto& r : comms.ledger().Records())
                l << 1e-9 * double(r.t.ns - t_first) << ',' << r.sender << ',' << r.receiver << ',' << int(r.type) << ','
                  << r.payload_bytes << ',' << r.transmitted_bytes << ',' << r.wire_bytes << ',' << int(r.compressed)
                  << ',' << int(r.retransmit) << ',' << int(r.resync) << ',' << r.predicted_wire_bytes << '\n';
        }
        std::vector<const ac::CollabNode*> cn;
        for (const auto& n : nodes) cn.push_back(n.get());
        const ac::OutcomeReport oc = ac::EvaluateOutcome(cn, now);
        {
            std::ofstream o(fs::path(out) / "outcome.yaml");
            o << "outcome: " << ac::OutcomeName(oc.outcome) << "\ncorrupt:\n";
            for (const auto& s : oc.corrupt) o << "  - \"" << s << "\"\n";
            o << "incomplete:\n";
            for (const auto& s : oc.incomplete) o << "  - \"" << s << "\"\n";
        }
        size_t ma = 0, km = 0, kt = 0;
        ac::AgentId frame = 0;
        const runtime::ErrorStats merged = merged_ate(ma, km, kt, &frame);
        std::ofstream y(fs::path(out) / "results.yaml");
        y << "tool: ariadne_collab_sim\nprotocol: v2\narm: " << cfg.arm << "\nseed: " << seed
          << "\ncomms_scenario: " << scenario.name << "\nbudget: {bytes_per_s: " << cfg.budget_bytes_per_s
          << ", burst_bytes: " << cfg.budget_burst_bytes << ", source: \"" << cfg.budget_source << "\"}"
          << "\nresponse_ratio: {value: " << cfg.response_ratio << ", calibration_run: \"" << cfg.calibration_run << "\"}"
          << "\nrobust: " << (cfg.robust ? "true" : "false") << "\nlocal_encoding: " << cfg.local_encoding
          << "\nverify: {min_pnp_inliers: " << cfg.verify.min_pnp_inliers << ", calibration: \"" << cfg.verify_calibration
          << "\"}" << "\nagents: " << fleet.size()
          << "\nduration_s: " << 1e-9 * double(t_end - t_first) << "\n";
        // Per class (COLLAB_3.2 §7, D171): first transmission outside resync (the I6 measured side) with the senders'
        // per-message predictions; overhead rows apart: retransmissions and resends to a new broker (resync).
        struct ClassBytes { uint64_t msgs = 0, wire = 0, transmitted = 0, retx = 0, resync = 0, predicted_msgs = 0;
                            double predicted = 0; };
        std::map<int, ClassBytes> cls;
        for (const auto& r : comms.ledger().Records()) {
            ClassBytes& c = cls[int(r.type)];
            if (r.retransmit) c.retx += r.wire_bytes;
            else if (r.resync) c.resync += r.wire_bytes;
            else {
                ++c.msgs, c.wire += r.wire_bytes, c.transmitted += r.transmitted_bytes;
                if (r.predicted_wire_bytes >= 0) ++c.predicted_msgs, c.predicted += r.predicted_wire_bytes;
            }
        }
        y << "bytes:\n";
        for (const auto& [k, c] : cls)
            y << "  - {class: " << k << ", messages: " << c.msgs << ", wire_first: " << c.wire << ", transmitted_first: "
              << c.transmitted << ", predicted_first: " << (c.predicted_msgs == c.msgs && c.msgs ? c.predicted : -1.0)
              << ", wire_retransmit: " << c.retx << ", wire_resync: " << c.resync << "}\n";
        y << "total_wire_bytes: {first: " << cum(true) << ", all: " << cum(false) << "}\n";
        double pred_req = 0, pred_resp = 0;
        uint64_t edges_verified = 0, solves = 0, cands = 0, selected = 0, expired = 0, uncal = 0, unadv = 0, changes = 0,
                 uncal_cls = 0;
        for (const auto& n : nodes) {   // summed over every broker the run had (D171 §5)
            const auto& s = n->stats();
            pred_req += s.predicted_request_bytes, pred_resp += s.predicted_response_bytes;
            edges_verified += s.edges_verified, solves += s.solves, cands += s.candidates;
            selected += s.candidates_selected, expired += s.candidates_expired, uncal += s.uncalibrated_prices;
            unadv += s.unadvertised_prices, changes += s.broker_changes, uncal_cls += s.uncalibrated_class_prices;
        }
        y << "predicted: {match_request: " << pred_req << ", match_response: " << pred_resp
          << ", uncalibrated_prices: " << uncal << ", uncalibrated_class_prices: " << uncal_cls
          << ", unadvertised_prices: " << unadv << "}\n";
        y << "protocol: {candidates: " << cands << ", selected: " << selected << ", expired: " << expired
          << ", edges_verified: " << edges_verified << ", solves: " << solves << ", broker_changes: " << changes << "}\n";
        y << "verify: {calls: " << vs.calls << ", geometric: " << vs.verified << ", gated: " << vs.gated
          << ", ms_total: " << vs.ms << ", reasons: {";
        bool first = true;
        for (const auto& [k, c] : vs.reasons) y << (first ? "" : ", ") << k << ": " << c, first = false;
        y << "}}\nsolve: {calls: " << solve_calls << ", ms_total: " << solve_ms << ", robust_inter_agent: " << robust_inter
          << ", robust_inliers: " << robust_inliers << "}\n";
        y << "ate_merged: {rmse_m: " << (merged.count ? merged.rmse : NAN) << ", frame_anchor: " << frame << ", merged_agents: " << ma
          << ", kf_merged: " << km << ", kf_released: " << kt << "}\n";
        y << "ate_solo:\n";
        for (const auto& [a, rs] : released) {
            std::vector<core::TrajectorySample> est, gt;
            for (const auto* r : rs)
                if (r->has_gt) est.push_back({core::Timestamp{r->t_create_ns}, r->id, r->T_W_C * r->T_B_C.inverse()}),
                               gt.push_back({core::Timestamp{r->t_create_ns}, r->id, r->gt_T_W_B});
            const runtime::ErrorStats e = runtime::ComputeATE(est, gt);
            y << "  - {agent: " << a << ", rmse_m: " << (e.count ? e.rmse : NAN) << ", keyframes: " << rs.size()
              << ", prefix_unreleased: " << stores.at(a).prefix_unreleased << "}\n";
        }
        y << "endpoints:\n";
        for (const auto& n : nodes) {
            const auto& e = n->endpoint().stats();
            y << "  - {agent: " << n->self() << ", first_tx: " << e.first_tx_messages << ", retx_fragments: "
              << e.retransmitted_fragments << ", acked: " << e.acked << ", delivered: " << e.delivered << ", duplicates: "
              << e.duplicates << ", decode_rejected: " << e.decode_rejected << ", expired: " << e.expired
              << ", superseded: " << e.superseded << ", abandoned_reliable: " << e.abandoned_reliable << "}\n";
        }
        const auto& ns = comms.network().stats();
        y << "network: {sent: " << ns.sent << ", lost_link_down: " << ns.lost_link_down << ", lost_random: " << ns.lost_random
          << ", reordered: " << ns.reordered << "}\noutcome: " << ac::OutcomeName(oc.outcome) << "\n";
        {
            std::ofstream m(fs::path(out) / "manifest.yaml");
            const auto env = [](const char* k) { const char* v = std::getenv(k); return std::string(v ? v : "UNRECORDED"); };
            m << "tool: ariadne_collab_sim\ngit_commit: " << env("ARIADNE_GIT_COMMIT") << "\ngit_dirty: " << env("ARIADNE_GIT_DIRTY")
              << "\nimage_digest: " << env("ARIADNE_IMAGE_DIGEST") << "\nopencv_build_hash: " << env("ARIADNE_OPENCV_BUILD_HASH")
              << "\nconfig: " << config_path << "\nstores: " << stores_dir << "\nrec: " << rec << "\nscene: " << scene
              << "\nseed: " << seed << "\naliked: " << lightglue.Name() << "\n";
        }
        std::printf("collab_sim %s seed %llu: %s; merged ATE %.4f m (%zu agents), %llu edges, %llu first-tx wire bytes\n",
                    cfg.arm.c_str(), (unsigned long long)seed, ac::OutcomeName(oc.outcome), merged.count ? merged.rmse : NAN,
                    ma, (unsigned long long)edges_verified, (unsigned long long)cum(true));
        return oc.outcome == ac::Outcome::kCorrupt ? 3 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_collab_sim: %s\n", e.what());
        return 1;
    }
}
