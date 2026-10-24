#pragma once
// ---------------------------------------------------------------------------
// CollabFleet: the fleet side of the 3.x collaboration (design LIVE_COLLAB §1, D174). One CollabNode per agent over one
// CommsNode, with the 3.2 hooks: retrieval, similarity, features, verification, the robust broker solve and the
// selection arm. It also plants false closures (§42 item 7) and keeps the merge evidence. Extracted from
// ariadne_collab_sim (pass B, SIMCOLLAB_3.2 / D171 / D173), so the replay and the live parent run the same code.
//
// The driver owns the clock. Each step at sim time `now`: Deliver(now) (the network up to now), Release() every
// keyframe whose t_final <= now, then Tick(now) (every node, then the scheduled injections). Features come from a
// FeatureSource: the pass-A store in replay (always ready), the estimator worker in live (nullopt = not ready yet,
// answered on a later tick: CollabNode's deferred answers).
//
// Hooks (SIMCOLLAB_3.2 §2):
//   retrieve    cosine over the broker's RECEIVED global descriptors (EigenPlaces R50-256), other agents only,
//               >= retrieval.min_score, top retrieval.top_k by score; the scores are cached for `similarity`
//   features    FeatureSource -> MATCH_RESPONSE: ALIKED-n16 keypoints, descriptors int8 + per-keyframe scale (arch
//               §15.7; fp16 the ablation, `local_encoding`), 3D points, camera
//   verify      LightGlue matches -> VerifyMatched (E gate, PnP) -> the D71 gate with the loop noise model
//               (GatedInterAgentEdge) -> the edge A -> B with its information
//   solve       RobustMultiAgentSolve (PCM -> chordal re-init -> GNC-TLS, D76) or MultiAgentSolve (robust: false)
//   select      arm: similarity (unset: request every candidate) | greedy (SelectBySimilarity) | spectral
// GROUND TRUTH IS EVALUATION ONLY: GroundTruthCamera feeds edges.csv's error columns and an injection's true pose,
// never a hook's decision.
// ---------------------------------------------------------------------------
#include "collaboration/broker/broker_solve.hpp"
#include "collaboration/broker/edge_verifier.hpp"
#include "collaboration/candidate_selection/selection.hpp"
#include "collaboration/candidate_selection/spectral_selector.hpp"
#include "collaboration/comms/comms_node.hpp"
#include "collaboration/protocol/collab_node.hpp"
#include "collaboration/retrieval/local_descriptor_codec.hpp"
#include "perception/association/hard_pair_matcher.hpp"
#include "runtime/agent/collab_store.hpp"
#include "slam/robust/joint_robust.hpp"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace ariadne::runtime {

namespace ac = ariadne::collaboration;

struct CollabConfig {
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
    // Arch §42 item 7 (EVALUATION ONLY; default none): false closures planted at the broker as if verified, so only
    // the robust stage can reject them. Z = (the true T_Ca_Cb) * [R_z(yaw_deg) | offset_m]; the information is copied
    // from the last edge verified on the same agent pair, and a planting waits until that pair HAS a verified edge
    // (a lone closure on a pair is accepted by PCM as "untested", pcm.hpp: the demo claim needs a competitor).
    struct Injection {
        double t_s = 0;
        ac::AgentId agent_a = 0, agent_b = 1;
        uint64_t kf_a = 0, kf_b = 0;
        core::Vec3 offset_m = core::Vec3::Zero();
        double yaw_deg = 0;
    };
    std::vector<Injection> inject;
};

inline CollabConfig LoadCollabConfig(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    // yaml-cpp keeps a duplicated key's FIRST value silently; a later override would be ignored (sweep 0345a5).
    std::set<std::string> seen;
    for (const auto& kv : n)
        if (!seen.insert(kv.first.as<std::string>()).second)
            throw std::invalid_argument(path + ": duplicate key " + kv.first.as<std::string>());
    CollabConfig c;
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
    for (const YAML::Node& j : n["inject_false_edges"]) {
        CollabConfig::Injection q;
        q.t_s = j["t_s"].as<double>();
        q.agent_a = j["agent_a"].as<ac::AgentId>(), q.kf_a = j["kf_a"].as<uint64_t>();
        q.agent_b = j["agent_b"].as<ac::AgentId>(), q.kf_b = j["kf_b"].as<uint64_t>();
        if (q.agent_a == q.agent_b) throw std::invalid_argument(path + ": inject_false_edges must join two agents");
        if (const YAML::Node o = j["offset_m"]) q.offset_m = core::Vec3(o[0].as<double>(), o[1].as<double>(), o[2].as<double>());
        if (j["yaw_deg"]) q.yaw_deg = j["yaw_deg"].as<double>();
        c.inject.push_back(q);
    }
    return c;
}

class CollabFleet {
public:
    /// Agent k's local features of keyframe kf as a MATCH_RESPONSE body; nullopt = not ready (live) or gone.
    using FeatureSource = std::function<std::optional<core::FeatureResponseValue>(ac::AgentId, uint64_t)>;
    /// EVALUATION ONLY: keyframe's true camera pose T_World_C (common GT world); nullopt = no ground truth.
    using GroundTruthCamera = std::function<std::optional<core::SE3>(const ac::KeyframeRef&)>;

    struct VerifyStats {
        uint64_t calls = 0, verified = 0, gated = 0;
        std::map<std::string, uint64_t> reasons;
        double ms = 0;
    };
    struct InjectionState {
        bool planted = false;
        double planted_t_s = NAN;
        uint64_t solves_seen = 0, kept_solves = 0;
        double weight = NAN;
        double gt_t = NAN, gt_r = NAN;
    };
    struct KeptEdge {   // an inter-agent loop the robust stage kept (weight >= 0.5) at least once
        double first_t_s = 0, last_weight = 0;
        bool injected = false;
    };
    using EdgeKey = std::tuple<ac::AgentId, uint64_t, ac::AgentId, uint64_t>;

    /// The MATCH_RESPONSE body for local features (descriptors float, 128 per feature) in the configured encoding.
    static core::FeatureResponseValue Response(uint64_t kf, const std::vector<float>& keypoints,
                                               const std::vector<float>& descriptors, const std::vector<float>& points3d,
                                               const std::array<float, 5>& camera, ac::LocalEncoding enc) {
        core::FeatureResponseValue v;
        v.keyframe_id = kf;
        v.keypoints = keypoints;
        v.descriptors = ac::EncodeLocal(descriptors, enc);
        v.feature_bytes = uint16_t(128 * ac::BytesPerDim(enc));
        v.header_bytes = enc == ac::LocalEncoding::kInt8 ? 4 : 0;
        v.points3d = points3d;
        v.camera.assign(camera.begin(), camera.end());
        return v;
    }

    CollabFleet(const CollabConfig& cfg, const std::vector<ac::AgentId>& fleet, ac::CommsNode& comms,
                const perception::HardPairMatcher& lightglue, uint64_t seed, int64_t t_first_ns, FeatureSource features,
                GroundTruthCamera gt_camera, std::ostream* edges_csv,
                std::function<void(ac::AgentId, ac::NodeHooks&)> customize = {})
        : cfg_(cfg), comms_(comms), lightglue_(lightglue), seed_(seed), t_first_ns_(t_first_ns),
          features_(std::move(features)), gt_camera_(std::move(gt_camera)), ecsv_(edges_csv),
          enc_(ac::ParseLocalEncoding(cfg.local_encoding)), feature_bytes_(uint16_t(128 * ac::BytesPerDim(enc_))),
          inject_(cfg.inject), inj_(cfg.inject.size()) {
        if (ecsv_)
            *ecsv_ << "agent_a,kf_a,agent_b,kf_b,score,matches,essential_inliers,pnp_inliers,reason,gt_err_t_m,"
                      "gt_err_r_deg,gt_baseline_m,pose_err_deg\n";
        ac::NodeParams np;
        np.selection_interval_s = cfg_.selection_interval_s;
        np.candidate_ttl_s = cfg_.candidate_ttl_s;
        if (cfg_.arm != "similarity")
            np.budget_bytes_per_s = cfg_.budget_bytes_per_s, np.budget_burst_bytes = cfg_.budget_burst_bytes;
        if (cfg_.response_ratio > 0) np.response_cost.response_ratio[feature_bytes_] = cfg_.response_ratio,
                                     np.response_cost.calibration_run = cfg_.calibration_run;
        if (cfg_.edge_ratio > 0) np.response_cost.class_ratio[int(core::MessageClass::POSE_EDGE)] = cfg_.edge_ratio;
        if (cfg_.graph_ratio > 0) np.response_cost.class_ratio[int(core::MessageClass::POSE_GRAPH)] = cfg_.graph_ratio;
        np.default_feature_bytes = feature_bytes_;

        const slam::JointRobustParams robust_params{};
        solve_robust_ = ac::RobustMultiAgentSolve(
            robust_params,
            [this](const slam::JointRobustReport& r) {
                robust_inter_ += r.inter_agent.size();
                robust_inliers_ += uint64_t(r.inter_inliers);
            },
            [this](const slam::PoseGraph& joint) { ObserveSolve(joint); });

        for (ac::AgentId self : fleet) {
            ac::NodeHooks h = Hooks(self);
            if (customize) customize(self, h);   // live: the D175 coordination mailbox (ariadne_live, B)
            nodes_.push_back(std::make_unique<ac::CollabNode>(self, fleet, codec_, comms_.ledger(), std::move(h), np));
            node_of_[self] = nodes_.back().get();
        }
    }
    CollabFleet(const CollabFleet&) = delete;
    CollabFleet& operator=(const CollabFleet&) = delete;

    /// The network up to `now`. Datagrams to an id outside the fleet are dropped (counted).
    void Deliver(core::Timestamp now) {
        sim_t_s_ = 1e-9 * double(now.ns - t_first_ns_);
        for (const auto& d : comms_.network().DeliverUntil(now)) {
            const auto it = node_of_.find(d.to);
            if (it == node_of_.end()) ++undeliverable_;
            else it->second->OnDatagram(d, now);
        }
    }
    uint64_t undeliverable() const { return undeliverable_; }

    /// One keyframe of `agent`, whole, at its t_final (its odometry became final). `n_features` advertises the
    /// local feature count when the record carries none (live, lazy); otherwise the record's own count.
    void Release(ac::AgentId agent, const CollabStoreRecord& r, core::Timestamp now,
                 std::optional<uint32_t> n_features = std::nullopt) {
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
        g.n_features = n_features ? *n_features : uint32_t(r.keypoints.size() / 2);
        g.feature_bytes = feature_bytes_;
        g.fp16 = true;   // arch §15.7's encoding (D171)
        node_of_.at(agent)->AddKeyframe(r.id, r.t_create_ns, r.T_W_C, odo, g, now);
        released_[agent].insert(r.id);
    }

    /// Every node, then the scheduled injections (cfg.inject) that are due and plantable.
    void Tick(core::Timestamp now) {
        for (auto& n : nodes_) n->Tick(now, comms_.network());
        for (size_t i = 0; i < cfg_.inject.size(); ++i) {
            const auto& q = inject_[i];
            if (inj_[i].planted || now.ns < t_first_ns_ + int64_t(std::llround(q.t_s * 1e9))) continue;
            Plant(i, now);
        }
    }

    /// D174 panel "inject false closure" (LIVE_COLLAB §5): plant one NOW on the agent pair with the most kept true
    /// loops, between each agent's latest released keyframe, offset `offset_m` and `yaw_deg` (fixed by the caller:
    /// every press is the same experiment). Armed only when that pair has a kept true loop and both keyframes have
    /// ground truth. Returns the injection's index in injections(), or nullopt when not armed.
    std::optional<size_t> InjectNow(core::Timestamp now, const core::Vec3& offset_m, double yaw_deg) {
        std::map<std::pair<ac::AgentId, ac::AgentId>, size_t> kept;
        for (const auto& [k, v] : kept_edges_)
            if (v.last_weight >= 0.5 && !IsInjected(k)) ++kept[PairKey(std::get<0>(k), std::get<2>(k))];
        if (kept.empty()) return std::nullopt;
        auto best = kept.begin();
        for (auto it = kept.begin(); it != kept.end(); ++it)
            if (it->second > best->second) best = it;
        const auto [a, b] = best->first;
        const auto ra = released_.find(a), rb = released_.find(b);
        if (ra == released_.end() || rb == released_.end() || ra->second.empty() || rb->second.empty()) return std::nullopt;
        CollabConfig::Injection q;
        q.t_s = 1e-9 * double(now.ns - t_first_ns_);
        q.agent_a = a, q.kf_a = *ra->second.rbegin(), q.agent_b = b, q.kf_b = *rb->second.rbegin();
        q.offset_m = offset_m, q.yaw_deg = yaw_deg;
        if (!GtRel({a, q.kf_a}, {b, q.kf_b})) return std::nullopt;
        inject_.push_back(q);
        inj_.emplace_back();
        if (!Plant(inject_.size() - 1, now)) {
            inject_.pop_back(), inj_.pop_back();
            return std::nullopt;
        }
        return inject_.size() - 1;
    }

    // --- state and evidence ---
    const std::vector<std::unique_ptr<ac::CollabNode>>& nodes() const { return nodes_; }
    ac::CollabNode& node(ac::AgentId a) const { return *node_of_.at(a); }
    const VerifyStats& verify_stats() const { return vs_; }
    double solve_ms() const { return solve_ms_; }
    uint64_t solve_calls() const { return solve_calls_; }
    uint64_t robust_inter() const { return robust_inter_; }
    uint64_t robust_inliers() const { return robust_inliers_; }
    const std::vector<InjectionState>& injections() const { return inj_; }
    const std::vector<CollabConfig::Injection>& injection_specs() const { return inject_; }
    /// Every inter-agent loop the robust stage ever kept, when first kept, its weight in the LAST solve containing it.
    const std::map<EdgeKey, KeptEdge>& kept_edges() const { return kept_edges_; }
    ac::LocalEncoding encoding() const { return enc_; }
    uint16_t feature_bytes() const { return feature_bytes_; }

private:
    /// Plants injection i at `now` if its pair has a verified edge and both keyframes are released (else false).
    bool Plant(size_t i, core::Timestamp now) {
        const auto& q = inject_[i];
        const auto info = pair_info_.find(PairKey(q.agent_a, q.agent_b));
        const auto has = [&](ac::AgentId ag, uint64_t kf) {
            const auto it = released_.find(ag);
            return it != released_.end() && it->second.count(kf);
        };
        if (info == pair_info_.end() || !has(q.agent_a, q.kf_a) || !has(q.agent_b, q.kf_b)) return false;
        const ac::KeyframeRef a{q.agent_a, q.kf_a}, b{q.agent_b, q.kf_b};
        const auto truth = GtRel(a, b);
        if (!truth) throw std::runtime_error("inject_false_edges: no ground truth for the planted pair");
        core::SE3 D;
        D.R = Eigen::AngleAxisd(q.yaw_deg * M_PI / 180.0, core::Vec3::UnitZ()).toRotationMatrix();
        D.t = q.offset_m;
        const core::SE3 Z = *truth * D;
        core::LoopEdgeValue e;
        e.agent_a = a.agent, e.keyframe_a = a.keyframe, e.agent_b = b.agent, e.keyframe_b = b.keyframe;
        e.measurement = ac::ToPoseValue(Z);
        e.information = info->second;
        e.frame_a = {a.agent, 0}, e.frame_b = {b.agent, 0};
        e.status = wire::EdgeStatus::GEOMETRY_VERIFIED;
        for (auto& n : nodes_)
            if (n->InjectVerifiedEdge(e, now)) {
                inj_[i].planted = true, inj_[i].planted_t_s = 1e-9 * double(now.ns - t_first_ns_);
                LogEdge(a, b, nullptr, "injected", &Z);
                const core::SE3 d = truth->inverse() * Z;
                inj_[i].gt_t = d.t.norm();
                inj_[i].gt_r = std::acos(std::clamp((d.R.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
                return true;
            }
        return false;
    }

    bool IsInjected(const EdgeKey& k) const {
        for (const auto& q : inject_) {
            const EdgeKey e = q.agent_a < q.agent_b ? std::make_tuple(q.agent_a, q.kf_a, q.agent_b, q.kf_b)
                                                    : std::make_tuple(q.agent_b, q.kf_b, q.agent_a, q.kf_a);
            if (e == k) return true;
        }
        return false;
    }

    static std::pair<ac::AgentId, ac::AgentId> PairKey(ac::AgentId x, ac::AgentId y) {
        return std::make_pair(std::min(x, y), std::max(x, y));
    }

    /// The response's descriptors as the broker decodes them (arch §15.7 encodings; D171: int8 + per-keyframe scale).
    static perception::HardPairFeatures ToHardPair(const core::FeatureResponseValue& r, ac::LocalEncoding enc) {
        perception::HardPairFeatures f;
        f.size = cv::Size(1280, 720);
        for (size_t i = 0; i + 1 < r.keypoints.size(); i += 2) f.keypoints.emplace_back(r.keypoints[i], r.keypoints[i + 1], 1.0f);
        const std::vector<float> d = ac::DecodeLocal(r.descriptors, enc);
        cv::Mat(int(d.size() / 128), 128, CV_32F, const_cast<float*>(d.data())).copyTo(f.descriptors);
        return f;
    }

    static double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
        double d = 0, na = 0, nb = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) d += double(a[i]) * b[i], na += double(a[i]) * a[i], nb += double(b[i]) * b[i];
        return na > 0 && nb > 0 ? d / std::sqrt(na * nb) : 0.0;
    }

    std::optional<core::SE3> GtRel(const ac::KeyframeRef& a, const ac::KeyframeRef& b) const {
        if (!gt_camera_) return std::nullopt;
        const auto ca = gt_camera_(a), cb = gt_camera_(b);
        if (!ca || !cb) return std::nullopt;
        return ca->inverse() * *cb;   // T_Ca_Cb, common GT world
    }

    void LogEdge(const ac::KeyframeRef& a, const ac::KeyframeRef& b, const ac::VerifyResult* v, const std::string& reason,
                 const core::SE3* Z) {
        if (!ecsv_) return;
        // pose_err_deg: D63's protocol for this matcher (LightGlue/SuperGlue relative-pose AUC): the larger of the
        // rotation error and the angle between the estimated and true translation directions; rotation only when the
        // true baseline is under 0.1 m (the direction is undefined there, D63's tiny-baseline caveat).
        double et = NAN, er = NAN, base = NAN, pe = NAN;
        if (const auto g = GtRel(a, b); g && Z) {
            const core::SE3 d = g->inverse() * *Z;
            et = d.t.norm();
            er = std::acos(std::clamp((d.R.trace() - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / M_PI;
            base = g->t.norm();
            pe = er;
            if (base >= 0.1 && Z->t.norm() > 1e-9)
                pe = std::max(pe, std::acos(std::clamp(g->t.normalized().dot(Z->t.normalized()), -1.0, 1.0)) * 180.0 /
                                      M_PI);
        }
        const auto sc = score_cache_.find({a, b});
        *ecsv_ << a.agent << ',' << a.keyframe << ',' << b.agent << ',' << b.keyframe << ','
               << (sc == score_cache_.end() ? NAN : sc->second) << ',' << (v ? v->matches : 0) << ','
               << (v ? v->essential_inliers : 0) << ',' << (v ? v->pnp_inliers : 0) << ',' << reason << ',' << et << ','
               << er << ',' << base << ',' << pe << '\n';
    }

    void ObserveSolve(const slam::PoseGraph& joint) {
        for (const auto& e : joint.edges()) {
            const auto ag = [](uint64_t id) { return ac::AgentId(id >> slam::kAgentIdShift); };
            const auto kf = [](uint64_t id) { return id & ((1ull << slam::kAgentIdShift) - 1); };
            if (ag(e.from) == ag(e.to)) continue;
            const bool fwd = ag(e.from) < ag(e.to);
            const auto key = fwd ? std::make_tuple(ag(e.from), kf(e.from), ag(e.to), kf(e.to))
                                 : std::make_tuple(ag(e.to), kf(e.to), ag(e.from), kf(e.from));
            const auto it = kept_edges_.find(key);
            if (it != kept_edges_.end()) it->second.last_weight = e.gnc_weight;
            else if (e.gnc_weight >= 0.5) kept_edges_.emplace(key, KeptEdge{sim_t_s_, e.gnc_weight, false});
        }
        for (size_t i = 0; i < inject_.size(); ++i) {
            if (!inj_[i].planted) continue;
            const auto& q = inject_[i];
            const uint64_t A = slam::JointId(q.agent_a, q.kf_a), B = slam::JointId(q.agent_b, q.kf_b);
            for (const auto& e : joint.edges())
                if ((e.from == A && e.to == B) || (e.from == B && e.to == A)) {
                    ++inj_[i].solves_seen, inj_[i].weight = e.gnc_weight;
                    if (e.gnc_weight >= 0.5) ++inj_[i].kept_solves;
                    break;
                }
        }
    }

    ac::NodeHooks Hooks(ac::AgentId self) {
        ac::NodeHooks h;
        h.retrieve = [this](const ac::KeyframeRef& q, const std::map<ac::KeyframeRef, core::GlobalDescriptorValue>& db) {
            std::vector<std::pair<double, ac::KeyframeRef>> scored;
            const auto qi = db.find(q);
            if (qi == db.end()) return std::vector<ac::KeyframeRef>{};
            for (const auto& [k, d] : db) {
                if (k.agent == q.agent) continue;
                const double s = Cosine(qi->second.descriptor, d.descriptor);
                if (s >= cfg_.min_score) scored.push_back({s, k});
            }
            std::stable_sort(scored.begin(), scored.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            if (int(scored.size()) > cfg_.top_k) scored.resize(size_t(cfg_.top_k));
            std::vector<ac::KeyframeRef> outk;
            for (const auto& [s, k] : scored) {
                score_cache_[{q, k}] = score_cache_[{k, q}] = s;
                outk.push_back(k);
            }
            return outk;
        };
        h.similarity = [this](const ac::KeyframeRef& q, const ac::KeyframeRef& c) {
            const auto it = score_cache_.find({q, c});
            return it == score_cache_.end() ? 1.0 : std::clamp(it->second, 1e-6, 1.0);
        };
        h.features = [this, self](uint64_t kf) { return features_(self, kf); };
        h.verify = [this](const ac::KeyframeRef& a, const core::FeatureResponseValue& fa, const ac::KeyframeRef& b,
                          const core::FeatureResponseValue& fb) -> std::optional<core::LoopEdgeValue> {
            const auto t0 = std::chrono::steady_clock::now();
            ++vs_.calls;
            const auto done = [&](const std::string& why) {
                ++vs_.reasons[why];
                vs_.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            };
            const perception::HardPairFeatures A = ToHardPair(fa, enc_), B = ToHardPair(fb, enc_);
            if (A.keypoints.size() < 8 || B.keypoints.size() < 8)
                return LogEdge(a, b, nullptr, "too_few_features", nullptr), done("too_few_features"), std::nullopt;
            const std::vector<cv::DMatch> m = lightglue_.Match(A, B);
            std::vector<cv::Point2f> pa, pb;
            std::vector<int> ib;
            for (const auto& d : m) {
                pa.push_back(A.keypoints[size_t(d.queryIdx)].pt);
                pb.push_back(B.keypoints[size_t(d.trainIdx)].pt);
                ib.push_back(d.trainIdx);
            }
            const uint64_t vseed = seed_ * 1000003ull + a.agent * 104729ull + a.keyframe * 7919ull + b.agent * 31ull + b.keyframe;
            const ac::VerifyResult v = ac::VerifyMatched(pa, pb, ib, fb, cfg_.verify, vseed);
            if (!v.ok) return LogEdge(a, b, &v, v.reason, v.reason == "pnp_gate" ? &v.T_A_B : nullptr),
                              done(v.reason), std::nullopt;
            ++vs_.verified;
            slam::LoopGeometry geo;
            geo.rig.cam = {fb.camera[0], fb.camera[1], fb.camera[2], fb.camera[3]};
            geo.rig.baseline = fb.camera[4];
            for (const auto& q : v.inlier_points_B) geo.X_B.push_back(core::Vec3(q.x, q.y, q.z));
            for (const auto& q : v.inlier_pixels_A) geo.uv_A.push_back(core::Vec2(q.x, q.y));
            geo.T_A_B = v.T_A_B;
            const auto edge = slam::GatedInterAgentEdge(0, 1, geo, cfg_.gate);
            if (!edge) return LogEdge(a, b, &v, "degenerate", &v.T_A_B), done("degenerate"), std::nullopt;
            LogEdge(a, b, &v, "verified", &edge->Z);
            ++vs_.gated;
            done("verified");
            core::LoopEdgeValue e;
            pair_info_[PairKey(a.agent, b.agent)] = ac::PackInformation(edge->information);
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
        h.solve = [this](const ac::SolveInput& in) {
            const auto t0 = std::chrono::steady_clock::now();
            ac::SolveOutput o = cfg_.robust ? solve_robust_(in) : ac::MultiAgentSolve(in);
            solve_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            ++solve_calls_;
            return o;
        };
        if (cfg_.arm == "greedy") h.select = [](const ac::SelectionProblem& P) { return ac::SelectBySimilarity(P); };
        if (cfg_.arm == "spectral") {
            ac::SelectorParams sp;
            sp.refine_passes = cfg_.refine_passes;
            h.select = [sp](const ac::SelectionProblem& P) { return ac::SelectSpectral(P, sp); };
        }
        return h;
    }

    const CollabConfig cfg_;
    ac::CommsNode& comms_;
    const perception::HardPairMatcher& lightglue_;
    const uint64_t seed_;
    const int64_t t_first_ns_;
    FeatureSource features_;
    GroundTruthCamera gt_camera_;
    std::ostream* ecsv_;
    const ac::LocalEncoding enc_;
    const uint16_t feature_bytes_;
    const core::WireCodec codec_{core::Framing::kFragmented};
    std::map<std::pair<ac::KeyframeRef, ac::KeyframeRef>, double> score_cache_;
    VerifyStats vs_;
    double solve_ms_ = 0;
    uint64_t solve_calls_ = 0, robust_inter_ = 0, robust_inliers_ = 0;
    // The last verified edge's information per unordered agent pair (an injection copies it).
    std::map<std::pair<ac::AgentId, ac::AgentId>, std::vector<float>> pair_info_;
    std::vector<CollabConfig::Injection> inject_;   // cfg.inject, then the panel's InjectNow presses
    std::vector<InjectionState> inj_;
    double sim_t_s_ = 0;
    std::map<EdgeKey, KeptEdge> kept_edges_;
    std::map<ac::AgentId, std::set<uint64_t>> released_;
    uint64_t undeliverable_ = 0;
    std::function<ac::SolveOutput(const ac::SolveInput&)> solve_robust_;
    std::vector<std::unique_ptr<ac::CollabNode>> nodes_;
    std::map<ac::AgentId, ac::CollabNode*> node_of_;
};

}  // namespace ariadne::runtime
