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
#include "planning/mission/frontier_mission.hpp"
#include "runtime/agent/collab_fleet.hpp"
#include "runtime/agent/collab_store.hpp"
#include "runtime/datasets/scene_walls.hpp"
#include "runtime/metrics/trajectory_metrics.hpp"
#include "slam/robust/joint_robust.hpp"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

using namespace ariadne;
namespace ac = ariadne::collaboration;
namespace fs = std::filesystem;

namespace {

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
        const runtime::CollabConfig cfg = runtime::LoadCollabConfig(config_path);
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

        // --- the fleet (runtime/agent/collab_fleet.hpp: the same class the live parent runs, D174) --------------
        perception::HardPairParams hp;
        hp.aliked_onnx = aliked_onnx;
        hp.lightglue_onnx = lightglue_onnx;
        hp.width = 1280, hp.height = 720;
        const perception::HardPairMatcher lightglue(hp);
        // Every verify call with its ground-truth edge error: EVALUATION ONLY (threshold calibration on the other
        // scene, accepted-edge precision). Written after the decision; nothing reads it back.
        std::ofstream ecsv(fs::path(out) / "edges.csv");
        const ac::LocalEncoding enc = ac::ParseLocalEncoding(cfg.local_encoding);
        runtime::CollabFleet fleet_(
            cfg, fleet, comms, lightglue, seed, t_first,
            [&](ac::AgentId self, uint64_t kf) -> std::optional<core::FeatureResponseValue> {   // the store: always ready
                const runtime::CollabStoreRecord& r = *rec_of.at(self).at(kf);
                return runtime::CollabFleet::Response(kf, r.keypoints, r.descriptors, r.points3d, r.camera, enc);
            },
            [&](const ac::KeyframeRef& k) -> std::optional<core::SE3> {   // EVALUATION ONLY
                const auto* r = rec_of.at(k.agent).at(k.keyframe);
                if (!r->has_gt) return std::nullopt;
                return r->gt_T_W_B * r->T_B_C;
            },
            &ecsv);
        if (cfg.entry_gate.enabled) {   // D178 known starts: the briefing's entry placements (scene entrances), not GT
            const auto T_W_O = planning::mission::EntryPoses(scene, int(fleet.size()));
            std::map<ac::AgentId, core::SE3> entry;
            for (ac::AgentId a : fleet) entry[a] = T_W_O.at(size_t(a));
            fleet_.SetEntryPoses(entry);
        }
        const auto& nodes = fleet_.nodes();
        std::map<ac::AgentId, ac::CollabNode*> node_of;
        for (const auto& n : nodes) node_of[n->self()] = n.get();
        const uint16_t feature_bytes = fleet_.feature_bytes();
        (void)feature_bytes;

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
            fleet_.Deliver(now);
            for (; next_rel < releases.size() && releases[next_rel].t <= t; ++next_rel) {
                const Release& rl = releases[next_rel];
                fleet_.Release(rl.agent, *rl.r, now);
                released[rl.agent].push_back(rl.r);
            }
            fleet_.Tick(now);
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
          << "\nrobust: " << (cfg.robust ? "true" : "false") << "\nentry_gate: {enabled: "
          << (cfg.entry_gate.enabled ? "true" : "false") << ", chi2: " << cfg.entry_gate.chi2 << ", entry_sigma_m: "
          << cfg.entry_gate.entry_sigma_m << ", entry_sigma_rad: " << cfg.entry_gate.entry_sigma_rad << ", kappa: "
          << cfg.entry_gate.kappa << ", tested: " << fleet_.entry_gate_stats().tested << ", rejected: "
          << fleet_.entry_gate_stats().rejected << ", ungated: " << fleet_.entry_gate_stats().ungated << "}" << "\nlocal_encoding: " << cfg.local_encoding
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
        const auto& vs = fleet_.verify_stats();
        y << "verify: {calls: " << vs.calls << ", geometric: " << vs.verified << ", gated: " << vs.gated
          << ", ms_total: " << vs.ms << ", reasons: {";
        bool first = true;
        for (const auto& [k, c] : vs.reasons) y << (first ? "" : ", ") << k << ": " << c, first = false;
        y << "}}\nsolve: {calls: " << fleet_.solve_calls() << ", ms_total: " << fleet_.solve_ms()
          << ", robust_inter_agent: " << fleet_.robust_inter() << ", robust_inliers: " << fleet_.robust_inliers() << "}\n";
        // Per agent: the frame it adopted (G = W of `anchor`) and T_G_W, row-major 3x4; null = never merged.
        y << "estimates:\n";
        for (const auto& n : nodes) {
            y << "  - {agent: " << n->self();
            if (const auto T = n->T_G_W()) {
                y << ", anchor: " << n->T_G_W_anchor() << ", T_G_W: [";
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c)
                        y << (r || c ? ", " : "") << std::setprecision(17) << (c < 3 ? T->R(r, c) : T->t(r));
                y << "]}\n" << std::setprecision(6);
            } else {
                y << ", anchor: null, T_G_W: null}\n";
            }
        }
        std::set<std::tuple<ac::AgentId, uint64_t, ac::AgentId, uint64_t>> injected_keys;
        for (const auto& q : cfg.inject)
            injected_keys.insert(q.agent_a < q.agent_b ? std::make_tuple(q.agent_a, q.kf_a, q.agent_b, q.kf_b)
                                                       : std::make_tuple(q.agent_b, q.kf_b, q.agent_a, q.kf_a));
        y << "kept_inter_edges:   # kept by the last solve that contained them; never injected (merge evidence)\n";
        for (const auto& [k, v] : fleet_.kept_edges())
            if (v.last_weight >= 0.5 && !injected_keys.count(k))
                y << "  - {agent_a: " << std::get<0>(k) << ", kf_a: " << std::get<1>(k) << ", agent_b: " << std::get<2>(k)
                  << ", kf_b: " << std::get<3>(k) << ", t_s: " << v.first_t_s << ", weight: " << v.last_weight << "}\n";
        y << "injected:\n";
        for (size_t i = 0; i < cfg.inject.size(); ++i) {
            const auto& q = cfg.inject[i];
            const auto& st = fleet_.injections()[i];
            // rejected: the robust stage gave it weight < 0.5 in the latest solve that contained it
            y << "  - {t_s: " << q.t_s << ", agent_a: " << q.agent_a << ", kf_a: " << q.kf_a << ", agent_b: " << q.agent_b
              << ", kf_b: " << q.kf_b << ", planted: " << (st.planted ? "true" : "false") << ", planted_t_s: "
              << st.planted_t_s << ", gt_err_t_m: " << st.gt_t << ", gt_err_r_deg: " << st.gt_r << ", solves_seen: "
              << st.solves_seen << ", kept_solves: " << st.kept_solves << ", final_weight: " << st.weight << ", rejected: "
              << (st.solves_seen ? (st.weight < 0.5 ? "true" : "false") : "unsolved") << "}\n";
        }
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
