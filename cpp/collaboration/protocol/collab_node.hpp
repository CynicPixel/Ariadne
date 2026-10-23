#pragma once
// ---------------------------------------------------------------------------
// One agent's collaboration node (design COLLAB_3.2 §3-§6): the transport, the
// membership state machine, the §15.1 two-stage exchange and, when elected,
// the broker duties. OpenCV-free: retrieval, geometric verification and the
// broker solve are injected (`NodeHooks`), so the protocol is tested with
// oracles while the runtime plugs in XFeat/PnP and the multi-agent PGO.
//
// DATA FLOW (agent a, broker b):
//   a: keyframe -> GLOBAL_DESCRIPTOR to b            (best effort, 120 s TTL)
//   b: retrieval over descriptors from all agents -> candidate pairs
//   b: MATCH_REQUEST to each endpoint whose features b lacks
//   a: MATCH_RESPONSE (features)                     (multi-datagram, ARQ, TTL)
//   b: verify -> inter-agent POSE_EDGE to both endpoints (reliable): the agents
//      keep it, so a new broker can be re-sent it (broker state is soft)
//   a: POSE_GRAPH snapshot of its own subgraph to b  (reliable, superseding)
//   b: solve over the members' snapshots + edges -> estimate POSE_GRAPH with
//      epoch and every member's T_G_W, to each member (reliable)
//   a: accept iff member, from my broker, strictly newer (Membership::Check)
//
// FRAMES (§15.5). Nodes and edges are in agents' odometry frames W_i, tagged
// {i, 0}. G is W of the anchor agent; a pose "in G as estimated by epoch c" is
// tagged {anchor, c}. A message carrying a pose in a frame this node does not
// know is rejected and counted, never composed.
//
// The node also logs what the outcome harness (§6) needs: every estimate it
// issued (as broker) or adopted, keyed by the full epoch, with a digest over
// the wire-encoded values.
// ---------------------------------------------------------------------------
#include "collaboration/candidate_selection/response_cost.hpp"
#include "collaboration/candidate_selection/selection.hpp"
#include "collaboration/protocol/membership.hpp"
#include "collaboration/protocol/transport.hpp"
#include "core/math/lie.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace ariadne::collaboration {

using core::FrameTag;
using core::LoopEdgeValue;
using core::PoseGraphValue;
using core::PoseValue;

// --- SE(3) <-> wire pose (float32, §15.7) ---------------------------------------

inline PoseValue ToPoseValue(const core::SE3& T) {
    const Eigen::Quaterniond q(T.R);
    return {float(q.w()), float(q.x()), float(q.y()), float(q.z()), float(T.t.x()), float(T.t.y()), float(T.t.z())};
}
inline core::SE3 FromPoseValue(const PoseValue& p) {
    core::SE3 T;
    T.R = Eigen::Quaterniond(p.qw, p.qx, p.qy, p.qz).normalized().toRotationMatrix();
    T.t = core::Vec3(p.tx, p.ty, p.tz);
    return T;
}

/// §4.3 wire information: the upper triangle of the 6x6, row-major, 21 floats.
inline std::vector<float> PackInformation(const core::Mat6& I) {
    std::vector<float> v;
    v.reserve(21);
    for (int r = 0; r < 6; ++r)
        for (int c = r; c < 6; ++c) v.push_back(float(I(r, c)));
    return v;
}
inline core::Mat6 UnpackInformation(const std::vector<float>& v) {
    core::Mat6 I = core::Mat6::Zero();
    size_t k = 0;
    for (int r = 0; r < 6; ++r)
        for (int c = r; c < 6; ++c, ++k) I(r, c) = I(c, r) = k < v.size() ? double(v[k]) : 0.0;
    return I;
}

/// An agent's odometry measurement into a new keyframe: Z = T_Cfrom_Cto, with
/// its information in the PoseEdge convention (slam/pose_graph/pose_graph.hpp).
struct OdometryEdge {
    uint64_t from_keyframe = 0;
    core::SE3 Z;
    core::Mat6 information = core::Mat6::Identity();
};

// --- hooks -----------------------------------------------------------------------

struct KeyframeRef {
    AgentId agent = 0;
    uint64_t keyframe = 0;
    bool operator<(const KeyframeRef& o) const { return std::tie(agent, keyframe) < std::tie(o.agent, o.keyframe); }
    bool operator==(const KeyframeRef& o) const { return agent == o.agent && keyframe == o.keyframe; }
};

struct SolveInput {
    std::vector<AgentId> members;
    std::map<AgentId, PoseGraphValue> graphs;   // each member's latest snapshot (nodes + odometry edges in W_member)
    std::vector<LoopEdgeValue> edges;           // inter-agent edges whose endpoints the snapshots cover
    AgentId anchor = 0;                         // G = W_anchor
    AgentId gauge_agent = 0;                    // whose T_G_W is held fixed
    core::SE3 gauge_T_G_W;                      // identity when gauge_agent == anchor
};
/// T_G_W per member placed by the solve; members with no edge path to the gauge are absent.
using SolveOutput = std::map<AgentId, core::SE3>;

struct NodeHooks {
    /// Broker: candidate keyframes (of other agents) for a newly received descriptor.
    std::function<std::vector<KeyframeRef>(const KeyframeRef& query,
                                           const std::map<KeyframeRef, core::GlobalDescriptorValue>& db)>
        retrieve;
    /// Agent: local features of one of its keyframes, for a MATCH_RESPONSE. nullopt = not ready yet (D174: the live
    /// worker extracts them lazily, on request); the node keeps the request and asks again on every Tick, answering
    /// when the hook is ready, until response_ttl_s. Replay hooks are always ready, so their bytes are unchanged.
    std::function<std::optional<core::FeatureResponseValue>(uint64_t keyframe)> features;
    /// Broker: geometric verification of a candidate pair, given both feature sets.
    std::function<std::optional<LoopEdgeValue>(const KeyframeRef& a, const core::FeatureResponseValue& fa,
                                               const KeyframeRef& b, const core::FeatureResponseValue& fb)>
        verify;
    std::function<SolveOutput(const SolveInput&)> solve;
    /// Broker, 3.4 (D170): retrieval similarity of a candidate pair, in (0, 1]: the candidate edge's weight. 1 if unset.
    std::function<double(const KeyframeRef& query, const KeyframeRef& candidate)> similarity;
    /// Broker, 3.4: the selection policy over pending candidates under the byte budget. UNSET = request every
    /// candidate as it appears (the Phase-1 behaviour: the "similarity" baseline, and the default). SelectBySimilarity
    /// is the "greedy" baseline; SelectSpectral (spectral_selector.hpp, links OpenCV) is the policy.
    std::function<SelectionReport(const SelectionProblem&)> select;
    /// D175: coordination messages (SUBMAP_PATCH, TASK_BID, TASK_CONSENSUS) addressed to this agent, delivered as they
    /// arrive (sim time), for the agent's planner mailbox. Unset = such messages are dropped.
    std::function<void(const core::DecodedMessage&, core::Timestamp)> coordination;
};

struct NodeParams {
    // D175 coordination TTLs: a patch is superseded by later ones (it carries everything since the acked version), a
    // bid or a consensus older than one replan period is worthless.
    double patch_ttl_s = 30.0, bid_ttl_s = 5.0, consensus_ttl_s = 5.0;
    MembershipParams membership;
    EndpointParams endpoint;
    double descriptor_ttl_s = 120.0;
    double response_ttl_s = 120.0;
    double request_retry_s = 10.0;      // broker re-requests features not received by then
    double request_give_up_s = 120.0;
    uint64_t snapshot_every_keyframes = 10;
    double snapshot_max_interval_s = 10.0;
    double solve_interval_s = 2.0;
    // 3.4 selection (only with hooks.select): a token bucket of predicted fetch bytes, refilled at budget_bytes_per_s
    // and capped at budget_burst_bytes, spent once per selection_interval_s.
    double selection_interval_s = 5.0;       // Swarm-SLAM inter_robot_detection_period_sec default (cslam)
    double budget_bytes_per_s = 20000.0;     // [PLACEHOLDER] unsourced; replaced by the derived level (25 % of the
    double budget_burst_bytes = 200000.0;    // similarity arm's fetch rate, burst = one period) in D171's sim runs
    double candidate_ttl_s = 120.0;          // [ASSUMPTION] a pending candidate never selected by then is dropped
                                             // (counted); retrieval proposes a pair once, so this is final. The sim
                                             // runs decide it (D170).
    ResponseCostModel response_cost;
    uint32_t default_features = 800;         // the price of a keyframe whose descriptor did not advertise its size
    uint16_t default_feature_bytes = 32;
};

/// One estimate as issued (broker) or adopted (member), for the §6 harness.
struct EstimateRecord {
    core::EpochValue epoch;
    std::vector<AgentId> members;
    AgentId anchor = 0;
    uint64_t digest = 0;                    // over the wire-encoded (float32) values
    std::set<std::tuple<AgentId, uint64_t, AgentId, uint64_t>> edges;   // solved over (broker log only)
};

struct NodeStats {
    uint64_t frame_rejects = 0, stale_estimates = 0, requests_sent = 0, requests_given_up = 0;
    uint64_t edges_verified = 0, candidates = 0, solves = 0, broker_changes = 0;
    // 3.4: selection rounds, candidates selected / dropped unselected, predicted first-transmission bytes per class
    uint64_t selection_rounds = 0, candidates_selected = 0, candidates_expired = 0;
    double predicted_request_bytes = 0, predicted_response_bytes = 0;
    uint64_t edges_injected = 0;            // §42 item 7 evaluation: false closures planted through InjectVerifiedEdge
    uint64_t answers_deferred = 0, answers_given_up = 0;   // D174: features not ready when asked / never ready by the TTL
    uint64_t unadvertised_prices = 0;     // keyframes priced at the default size (expected 0 on v2 runs)
    uint64_t uncalibrated_prices = 0;     // fetches priced without a calibrated ratio: such a run is not I6 evidence
    uint64_t uncalibrated_class_prices = 0;   // edges / graph batches priced at ratio 1.0 (D171): likewise
    uint64_t max_pending = 0;             // largest candidate pool handed to the policy
    double select_ms_total = 0, select_ms_max = 0;   // policy wall time per round (not used for decisions)
};

using EdgeKey = std::tuple<AgentId, uint64_t, AgentId, uint64_t>;
inline EdgeKey KeyOf(const LoopEdgeValue& e) { return {e.agent_a, e.keyframe_a, e.agent_b, e.keyframe_b}; }

/// FNV-1a over an estimate's wire values (the float32 an agent decodes).
inline uint64_t EstimateDigest(const PoseGraphValue& g) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) {
            h ^= b[i];
            h *= 1099511628211ull;
        }
    };
    mix(&g.epoch.counter, 8);
    mix(&g.epoch.members_hash, 8);
    mix(&g.epoch.broker, 4);
    mix(&g.anchor.agent, 4);
    mix(&g.anchor.epoch, 4);
    for (const auto& f : g.agent_frames) {
        mix(&f.agent, 4);
        const float v[7] = {f.T_G_W.qw, f.T_G_W.qx, f.T_G_W.qy, f.T_G_W.qz, f.T_G_W.tx, f.T_G_W.ty, f.T_G_W.tz};
        mix(v, sizeof(v));
    }
    return h;
}

class CollabNode {
public:
    CollabNode(AgentId self, std::vector<AgentId> fleet, const core::WireCodec& codec, core::ByteLedger& ledger,
               NodeHooks hooks, NodeParams p = {})
        : self_(self), fleet_(std::move(fleet)), codec_(codec), ep_(self, codec, ledger, p.endpoint),
          mem_(self, p.membership), hooks_(std::move(hooks)), p_(p), cost_(codec, p.response_cost) {
        // 3.4: a policy without similarities would weigh every candidate 1, silently "spectral on topology only".
        if (hooks_.select && !hooks_.similarity)
            throw std::invalid_argument("CollabNode: NodeHooks::select needs NodeHooks::similarity");
    }

    // --- application input ---

    /// A keyframe of this agent's odometry: pose in W_self, the odometry edge
    /// into it (none for the first keyframe), and its global descriptor.
    void AddKeyframe(uint64_t keyframe, int64_t stamp_ns, const core::SE3& T_W_C,
                     const std::optional<OdometryEdge>& odometry, const core::GlobalDescriptorValue& descriptor,
                     core::Timestamp t) {
        if (odometry) {
            LoopEdgeValue e;
            e.agent_a = e.agent_b = self_;
            e.keyframe_a = odometry->from_keyframe;
            e.keyframe_b = keyframe;
            e.measurement = ToPoseValue(odometry->Z);
            e.information = PackInformation(odometry->information);
            e.frame_a = e.frame_b = {self_, 0};
            e.status = wire::EdgeStatus::ODOMETRY;
            odometry_.push_back(e);
        }
        core::PoseNodeValue n;
        n.agent = self_;
        n.keyframe = keyframe;
        n.stamp_ns = stamp_ns;
        n.pose = ToPoseValue(T_W_C);
        n.frame = {self_, 0};
        nodes_.push_back(n);
        descriptors_[keyframe] = descriptor;
        if (mem_.IsBroker()) {
            BrokerOnDescriptor({self_, keyframe}, descriptor, t);
        } else {
            SendDescriptor(keyframe, t);
        }
    }

    // --- driven by the harness / runtime ---

    void OnDatagram(const Datagram& d, core::Timestamp t) {
        for (auto& m : ep_.Receive(d, t)) Handle(std::move(m), t);
    }

    void Tick(core::Timestamp t, Network& net) {
        // 1. membership and election
        const AgentId old_broker = mem_.broker();
        if (mem_.Update(t)) OnBrokerChange(old_broker, t);
        for (AgentId a : fleet_)
            if (a != self_) ep_.SetReachable(a, mem_.IsNeighbor(a));
        // 2. heartbeats to every fleet agent (discovery does not wait for reachability)
        if (!next_heartbeat_ || t >= *next_heartbeat_) {
            SendHeartbeats(t, net);
            next_heartbeat_ = t + core::Duration::SecondsFromDouble(p_.membership.period_s);
        }
        // 3. snapshot cadence to my broker; feature requests whose answer was not ready (D174)
        if (!mem_.IsBroker()) MaybeSnapshot(t);
        AnswerDeferred(t);
        // 4. broker duties
        if (mem_.IsBroker()) {
            ResolveOwnFeatures(t);
            MaybeSelect(t);
            RetryRequests(t);
            MaybeSolve(t);
        }
        ep_.Pump(t, net);
    }

    // --- state for the harness ---

    AgentId self() const { return self_; }
    const Membership& membership() const { return mem_; }
    const Endpoint& endpoint() const { return ep_; }
    const NodeStats& stats() const { return stats_; }
    /// The last 3.4 selection round's report (broker only; empty before the first).
    const std::optional<SelectionReport>& last_selection() const { return last_selection_; }
    const std::optional<EstimateRecord>& adopted() const { return adopted_; }
    const std::map<core::EpochValue, EstimateRecord, bool (*)(const core::EpochValue&, const core::EpochValue&)>&
    issued() const {
        return issued_;
    }
    std::optional<core::SE3> T_G_W() const { return T_G_W_; }
    /// The anchor whose G the current T_G_W is expressed in (it can predate the latest adopted estimate, which may
    /// list this agent as a member it did not place). For scoring (D171); never read by the protocol.
    AgentId T_G_W_anchor() const { return T_G_W_anchor_; }
    AgentId anchor() const { return mem_.anchor(); }
    const std::map<EdgeKey, LoopEdgeValue>& edges() const { return edges_; }
    /// Edges the broker holds back because a snapshot does not yet cover an endpoint (§3.4).
    size_t HeldEdges() const { return mem_.IsBroker() ? held_count_ : 0; }

private:
    struct Request {
        uint64_t id = 0;
        core::Timestamp first, last;
    };

    static bool EpochLess(const core::EpochValue& a, const core::EpochValue& b) {
        return std::tie(a.counter, a.members_hash, a.broker) < std::tie(b.counter, b.members_hash, b.broker);
    }

    // --- sending helpers ---

    void SendDescriptor(uint64_t kf, core::Timestamp t) {
        const AgentId to = mem_.broker();
        if (to == self_) return;
        OutMessage m;
        m.cls = core::MessageClass::GLOBAL_DESCRIPTOR;
        m.key = kf;
        m.created = t;
        m.ttl = core::Duration::SecondsFromDouble(p_.descriptor_ttl_s);
        const auto g = descriptors_.at(kf);
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeGlobalDescriptor(g, self, to, ts, seq); };
        Plan(m, t);
        ep_.Enqueue(to, std::move(m));
    }

    void SendEdge(AgentId to, const LoopEdgeValue& e, core::Timestamp t) {
        OutMessage m;
        m.cls = core::MessageClass::POSE_EDGE;
        m.key = ++edge_msg_key_;
        m.created = t;
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeLoopEdge(e, self, to, ts, seq); };
        Plan(m, t);
        ep_.Enqueue(to, std::move(m));
    }

    void SendGraph(AgentId to, const PoseGraphValue& g, core::Timestamp t, bool supersedes) {
        OutMessage m;
        m.cls = core::MessageClass::POSE_GRAPH;
        m.key = ++graph_msg_key_;
        m.created = t;
        m.supersedes = supersedes;
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodePoseGraph(g, self, to, ts, seq); };
        Plan(m, t);
        ep_.Enqueue(to, std::move(m));
    }

    /// COLLAB_3.2 §7 / D171: price a planned message (a sample encode with a representative reliable sequence, as
    /// FetchCostPredictor) and tag a resend to a new broker. Prediction and tag reach the ledger with the first
    /// transmission; a message that expires unsent never does.
    void Plan(OutMessage& m, core::Timestamp t) {
        const uint64_t seq = (1ull << 40) | 1ull;
        m.predicted_wire_bytes = cost_.PredictWire(m.encode(seq, t), m.cls);
        m.resync = resyncing_;
        if (core::IsCompressedClass(m.cls) && !cost_.ClassCalibrated(m.cls)) ++stats_.uncalibrated_class_prices;
    }

    PoseGraphValue Snapshot() const {
        PoseGraphValue g;
        g.nodes = nodes_;
        g.edges = odometry_;
        // `anchor` names the frame of agent_frames: my T_G_W as last adopted.
        if (T_G_W_ && adopted_) {
            g.anchor = {mem_.anchor(), uint32_t(adopted_->epoch.counter)};
            g.agent_frames.push_back({self_, ToPoseValue(*T_G_W_)});
        } else {
            g.anchor = {self_, 0};
        }
        return g;
    }

    void MaybeSnapshot(core::Timestamp t) {
        const bool enough_kf = nodes_.size() >= snapshot_nodes_ + p_.snapshot_every_keyframes;
        const bool stale = nodes_.size() > snapshot_nodes_ &&
                           (!last_snapshot_ || (t - *last_snapshot_).toSeconds() >= p_.snapshot_max_interval_s);
        if (!enough_kf && !stale) return;
        resyncing_ = resync_snapshot_;
        SendGraph(mem_.broker(), Snapshot(), t, /*supersedes=*/true);
        resyncing_ = resync_snapshot_ = false;
        snapshot_nodes_ = nodes_.size();
        last_snapshot_ = t;
    }

    void SendHeartbeats(core::Timestamp t, Network& net) {
        for (AgentId to : fleet_) {
            if (to == self_) continue;
            core::HeartbeatValue h;
            h.agent = self_;
            h.stamp_ns = t.ns;
            if (!nodes_.empty()) h.pose = nodes_.back().pose;
            h.frame = {self_, 0};
            h.keyframe_watermark = nodes_.empty() ? 0 : nodes_.back().keyframe;
            h.neighbors = mem_.Heard();
            h.anchor = mem_.anchor();
            h.broker = mem_.broker();
            h.epoch = mem_.accepted();
            h.ack = ep_.TakeAck(to);
            ep_.SendImmediate(to, core::MessageClass::HEARTBEAT,
                              [&](uint64_t seq, core::Timestamp ts) { return codec_.EncodeHeartbeat(h, self_, to, ts, seq); },
                              t, net);
        }
    }

    /// Election change (§3.2, §4): abandon reliable traffic to the old broker,
    /// then give the new one everything it needs (broker state is soft).
    void OnBrokerChange(AgentId old_broker, core::Timestamp t) {
        ++stats_.broker_changes;
        if (old_broker != self_) ep_.AbandonReliable(old_broker);
        const AgentId b = mem_.broker();
        if (b == self_) {
            // I became broker: my own descriptors enter my database.
            for (const auto& [kf, g] : descriptors_) BrokerOnDescriptor({self_, kf}, g, t);
            for (const auto& [k, e] : edges_) broker_edges_.emplace(k, e);
            dirty_ = true;
            return;
        }
        const auto& sent = ep_.Sent(b, core::MessageClass::GLOBAL_DESCRIPTOR);
        resyncing_ = true;
        for (const auto& [kf, g] : descriptors_)
            if (!sent.count(kf)) SendDescriptor(kf, t);
        for (const auto& [k, e] : edges_) SendEdge(b, e, t);
        resyncing_ = false;
        snapshot_nodes_ = 0;   // force a fresh snapshot to the new broker
        last_snapshot_.reset();
        resync_snapshot_ = true;
    }

    // --- receiving ---

    /// §15.5: a W frame of a fleet agent, or a G frame this node adopted.
    bool KnownFrame(const FrameTag& f) const {
        if (f.epoch == 0) return std::find(fleet_.begin(), fleet_.end(), f.agent) != fleet_.end();
        return known_g_.count({f.agent, f.epoch}) > 0;
    }

    template <typename Fill>
    void DeliverLocal(core::MessageClass cls, core::Timestamp t, Fill fill) {
        if (!hooks_.coordination) return;
        core::DecodedMessage d;
        d.cls = cls, d.sender = self_, d.receiver = self_, d.stamp_ns = t.ns;
        fill(d);
        hooks_.coordination(d, t);
    }

    void Handle(core::DecodedMessage m, core::Timestamp t) {
        using MC = core::MessageClass;
        if (m.has_heartbeat) {
            mem_.OnHeartbeat(m.heartbeat, t);
            return;
        }
        switch (m.cls) {
            case MC::GLOBAL_DESCRIPTOR:
                if (mem_.IsBroker()) BrokerOnDescriptor({m.sender, m.global_descriptor.keyframe_id}, m.global_descriptor, t);
                break;
            case MC::MATCH_REQUEST: AnswerRequest(m, t); break;
            case MC::MATCH_RESPONSE:
                if (mem_.IsBroker()) BrokerOnResponse(m, t);
                break;
            case MC::POSE_EDGE: {
                const auto& e = m.edge;
                if (e.frame_a.epoch != 0 || e.frame_b.epoch != 0 || e.frame_a.agent != e.agent_a ||
                    e.frame_b.agent != e.agent_b || !KnownFrame(e.frame_a) || !KnownFrame(e.frame_b)) {
                    ++stats_.frame_rejects;
                    break;
                }
                edges_.emplace(KeyOf(e), e);   // every agent keeps the edges it is part of
                if (mem_.IsBroker() && broker_edges_.emplace(KeyOf(e), e).second) dirty_ = true;
                break;
            }
            case MC::POSE_GRAPH:
                if (m.pose_graph.has_epoch) {
                    OnEstimate(m.pose_graph, m.sender, t);
                } else if (mem_.IsBroker()) {
                    for (const auto& n : m.pose_graph.nodes)
                        if (n.frame.agent != m.sender || n.frame.epoch != 0) {
                            ++stats_.frame_rejects;
                            return;
                        }
                    for (const auto& e : m.pose_graph.edges)
                        if (e.agent_a != m.sender || e.agent_b != m.sender || e.frame_a.agent != m.sender ||
                            e.frame_b.agent != m.sender || e.frame_a.epoch != 0 || e.frame_b.epoch != 0) {
                            ++stats_.frame_rejects;
                            return;
                        }
                    snapshots_[m.sender] = std::move(m.pose_graph);
                    dirty_ = true;
                }
                break;
            case MC::SUBMAP_PATCH:
            case MC::TASK_BID:
            case MC::TASK_CONSENSUS:
                if (hooks_.coordination) hooks_.coordination(m, t);
                break;
            default: break;
        }
    }

    void AnswerRequest(const core::DecodedMessage& m, core::Timestamp t) {
        if (!hooks_.features || !descriptors_.count(m.match_request.candidate_keyframe)) return;
        const uint64_t id = m.match_request.request_id, kf = m.match_request.candidate_keyframe;
        if (deferred_answers_.count(id)) return;   // a retry of a request still being prepared
        auto f = hooks_.features(kf);
        if (!f) {
            deferred_answers_.emplace(id, DeferredAnswer{m.sender, kf, t});
            ++stats_.answers_deferred;
            return;
        }
        SendResponse(std::move(*f), id, kf, m.sender, t);
    }

    /// D174: answer the requests whose features have become ready, in request-id order (deterministic); drop those
    /// still not ready after response_ttl_s (the broker gives up on its side after request_give_up_s).
    void AnswerDeferred(core::Timestamp t) {
        for (auto it = deferred_answers_.begin(); it != deferred_answers_.end();) {
            const DeferredAnswer& d = it->second;
            if ((t - d.since).toSeconds() >= p_.response_ttl_s) {
                ++stats_.answers_given_up;
                it = deferred_answers_.erase(it);
                continue;
            }
            auto f = hooks_.features(d.keyframe);
            if (!f) {
                ++it;
                continue;
            }
            SendResponse(std::move(*f), it->first, d.keyframe, d.to, t);
            it = deferred_answers_.erase(it);
        }
    }

    void SendResponse(core::FeatureResponseValue f, uint64_t request_id, uint64_t keyframe, AgentId to,
                      core::Timestamp t) {
        f.request_id = request_id;
        f.keyframe_id = keyframe;
        OutMessage o;
        o.cls = core::MessageClass::MATCH_RESPONSE;
        o.key = f.request_id;
        o.created = t;
        o.ttl = core::Duration::SecondsFromDouble(p_.response_ttl_s);
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        o.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeFeatureResponse(f, self, to, ts, seq); };
        ep_.Enqueue(to, std::move(o));
    }

    void OnEstimate(const PoseGraphValue& g, AgentId sender, core::Timestamp t) {
        std::vector<AgentId> members;
        // The epoch's member list travels as agent_frames; an unplaced member carries qw = 0.
        for (const auto& f : g.agent_frames) members.push_back(f.agent);
        std::sort(members.begin(), members.end());
        const auto verdict = mem_.Check(g.epoch, members, sender);
        if (verdict != Membership::Verdict::kAccept) {
            ++stats_.stale_estimates;
            return;
        }
        Adopt(g, members);
        (void)t;
    }

    void Adopt(const PoseGraphValue& g, const std::vector<AgentId>& members) {
        mem_.Accept(g.epoch);
        EstimateRecord r;
        r.epoch = g.epoch;
        r.members = members;
        r.anchor = g.anchor.agent;
        r.digest = EstimateDigest(g);
        adopted_ = r;
        for (const auto& f : g.agent_frames) {
            if (f.agent != self_ || f.T_G_W.qw == 0.0f && f.T_G_W.qx == 0.0f && f.T_G_W.qy == 0.0f &&
                                        f.T_G_W.qz == 0.0f)
                continue;
            // Placed: adopt the anchor and my T_G_W (anchor propagation, §15.5).
            mem_.SetAnchor(g.anchor.agent);
            T_G_W_ = FromPoseValue(f.T_G_W);
            T_G_W_anchor_ = g.anchor.agent;
            known_g_.insert({g.anchor.agent, g.anchor.epoch});
        }
    }

    // --- broker ---

    void BrokerOnDescriptor(const KeyframeRef& q, const core::GlobalDescriptorValue& g, core::Timestamp t) {
        if (!descriptor_db_.emplace(q, g).second || !hooks_.retrieve) return;
        for (const KeyframeRef& c : hooks_.retrieve(q, descriptor_db_)) {
            if (c.agent == q.agent) continue;
            const std::pair<KeyframeRef, KeyframeRef> pair = std::minmax(q, c);
            if (hooks_.select) {   // 3.4: queue it; MaybeSelect decides under the budget
                if (candidates_.count(pair) || pending_.count(pair)) continue;
                ++stats_.candidates;
                pending_.emplace(pair, Pending{++pending_id_, hooks_.similarity ? hooks_.similarity(q, c) : 1.0, t});
                continue;
            }
            if (!candidates_.insert(pair).second) continue;
            ++stats_.candidates;
            for (const KeyframeRef& end : {q, c}) RequestFeatures(end, t);
            TryVerify(pair.first, pair.second, t);
        }
    }

    /// 3.4 (design SPECTRAL_3.4): every selection_interval_s, hand the pending candidates, the broker's keyframe graph
    /// and the predicted fetch bytes to the policy; request what it selects; charge the predicted bytes per class.
    void MaybeSelect(core::Timestamp t) {
        if (!hooks_.select) return;
        if (last_select_ && (t - *last_select_).toSeconds() < p_.selection_interval_s) return;
        const double dt = last_select_ ? (t - *last_select_).toSeconds() : p_.selection_interval_s;
        last_select_ = t;
        tokens_ = std::min(p_.budget_burst_bytes, tokens_ + p_.budget_bytes_per_s * dt);
        for (auto it = pending_.begin(); it != pending_.end();) {
            if ((t - it->second.since).toSeconds() >= p_.candidate_ttl_s) {
                ++stats_.candidates_expired;
                it = pending_.erase(it);
            } else {
                ++it;
            }
        }
        if (pending_.empty()) return;
        // Nodes and odometry as Swarm-SLAM builds them (cslam algebraic_connectivity_maximization.py, fill_odometry):
        // every keyframe id min..max seen per agent, consecutive ids joined at the fixed weight 1. Gaps in what the
        // broker has seen are full chains, not one collapsed edge. cslam starts at 0 because every keyframe is sent;
        // ours start at the first one sent, since a VO startup prefix is never released (SIMCOLLAB_3.2 §1, D87 carry).
        std::map<AgentId, std::pair<uint64_t, uint64_t>> span;
        auto see = [&](const KeyframeRef& k) {
            const auto [it, fresh] = span.try_emplace(k.agent, k.keyframe, k.keyframe);
            if (!fresh) it->second = {std::min(it->second.first, k.keyframe), std::max(it->second.second, k.keyframe)};
        };
        for (const auto& [k, g] : descriptor_db_) see(k);
        for (const auto& [a, g] : snapshots_)
            for (const auto& n : g.nodes) see({a, n.keyframe});
        for (const auto& [pr, c] : pending_) see(pr.first), see(pr.second);
        std::map<KeyframeRef, int> idx;
        SelectionProblem P;
        for (const auto& [a, lohi] : span)
            for (uint64_t kf = lohi.first; kf <= lohi.second; ++kf) {
                idx.emplace(KeyframeRef{a, kf}, P.n);
                if (kf > lohi.first) P.fixed.push_back({P.n - 1, P.n, 1.0});
                ++P.n;
            }
        for (const auto& [key, e] : broker_edges_) {
            const auto a = idx.find({e.agent_a, e.keyframe_a}), b = idx.find({e.agent_b, e.keyframe_b});
            if (a != idx.end() && b != idx.end() && e.agent_a != e.agent_b) P.fixed.push_back({a->second, b->second, 1.0});
        }
        auto fetch_cost = [&](const KeyframeRef& k) {
            const auto d = descriptor_db_.find(k);
            const bool adv = d != descriptor_db_.end() && d->second.n_features > 0;
            return std::make_pair(cost_.Predict(adv ? d->second.n_features : p_.default_features,
                                                adv ? d->second.feature_bytes : p_.default_feature_bytes),
                                  std::make_pair(adv, cost_.Calibrated(adv ? d->second.feature_bytes
                                                                           : p_.default_feature_bytes)));
        };
        auto held = [&](const KeyframeRef& k) { return k.agent == self_ || features_.count(k) || requests_.count(k); };
        P.node_bytes.assign(size_t(P.n), 0.0);
        for (const auto& [k, v] : idx)
            if (!held(k)) P.node_bytes[size_t(v)] = fetch_cost(k).first.total();
        std::map<uint64_t, std::pair<KeyframeRef, KeyframeRef>> by_id;
        for (const auto& [pr, c] : pending_) {
            P.candidates.push_back({c.id, idx.at(pr.first), idx.at(pr.second), c.kappa});
            by_id.emplace(c.id, pr);
        }
        P.budget_bytes = tokens_;
        stats_.max_pending = std::max<uint64_t>(stats_.max_pending, pending_.size());
        const auto w0 = std::chrono::steady_clock::now();
        SelectionReport r = hooks_.select(P);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        stats_.select_ms_total += ms;
        stats_.select_ms_max = std::max(stats_.select_ms_max, ms);
        ++stats_.selection_rounds;
        for (uint64_t id : r.selected) {
            const auto pr = by_id.at(id);
            pending_.erase(pr);
            for (const KeyframeRef& end : {pr.first, pr.second}) {
                if (!held(end)) {
                    const auto [c, flags] = fetch_cost(end);
                    stats_.predicted_request_bytes += c.request;
                    stats_.predicted_response_bytes += c.response;
                    stats_.unadvertised_prices += !flags.first;
                    stats_.uncalibrated_prices += !flags.second;
                }
                RequestFeatures(end, t);
            }
            candidates_.insert(pr);
            ++stats_.candidates_selected;
            TryVerify(pr.first, pr.second, t);
        }
        tokens_ = std::max(0.0, tokens_ - r.bytes);
        last_selection_ = std::move(r);
    }

    void RequestFeatures(const KeyframeRef& k, core::Timestamp t) {
        if (features_.count(k)) return;
        if (k.agent == self_) {
            if (!hooks_.features) return;
            if (auto f = hooks_.features(k.keyframe)) {
                features_[k] = std::move(*f);
            } else {
                own_waiting_.emplace(k, t);   // D174: verified once ResolveOwnFeatures finds it ready
            }
            return;
        }
        if (requests_.count(k)) return;
        Request& r = requests_[k];
        r.id = (uint64_t(self_) << 48) | ++request_counter_;
        r.first = t;
        SendRequest(k, r, t);
    }

    void SendRequest(const KeyframeRef& k, Request& r, core::Timestamp t) {
        r.last = t;
        request_owner_[r.id] = k;
        core::MatchRequestValue v{r.id, 0, k.keyframe, 0};
        OutMessage o;
        o.cls = core::MessageClass::MATCH_REQUEST;
        o.key = r.id;
        o.created = t;
        o.ttl = core::Duration::SecondsFromDouble(p_.request_retry_s);
        const AgentId self = self_, to = k.agent;
        const core::WireCodec* codec = &codec_;
        o.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeMatchRequest(v, self, to, ts, seq); };
        ep_.Enqueue(to, std::move(o));
        ++stats_.requests_sent;
    }

    void RetryRequests(core::Timestamp t) {
        for (auto it = requests_.begin(); it != requests_.end();) {
            Request& r = it->second;
            if ((t - r.first).toSeconds() >= p_.request_give_up_s) {
                ++stats_.requests_given_up;
                it = requests_.erase(it);
                continue;
            }
            if ((t - r.last).toSeconds() >= p_.request_retry_s) SendRequest(it->first, r, t);
            ++it;
        }
    }

    /// D174: the broker's own keyframes whose features were not ready when a selected pair needed them.
    void ResolveOwnFeatures(core::Timestamp t) {
        for (auto it = own_waiting_.begin(); it != own_waiting_.end();) {
            const KeyframeRef k = it->first;
            if ((t - it->second).toSeconds() >= p_.request_give_up_s) {
                ++stats_.requests_given_up;
                it = own_waiting_.erase(it);
                continue;
            }
            auto f = hooks_.features(k.keyframe);
            if (!f) {
                ++it;
                continue;
            }
            features_[k] = std::move(*f);
            it = own_waiting_.erase(it);
            for (const auto& [a, b] : candidates_)
                if (a == k || b == k) TryVerify(a, b, t);
        }
    }

    void BrokerOnResponse(const core::DecodedMessage& m, core::Timestamp t) {
        const auto owner = request_owner_.find(m.feature_response.request_id);
        if (owner == request_owner_.end()) return;   // unknown or given up: stale
        const KeyframeRef k = owner->second;
        if (k.agent != m.sender || k.keyframe != m.feature_response.keyframe_id) return;
        features_[k] = m.feature_response;
        requests_.erase(k);
        for (const auto& [a, b] : candidates_)
            if (a == k || b == k) TryVerify(a, b, t);
    }

    void TryVerify(const KeyframeRef& a, const KeyframeRef& b, core::Timestamp t) {
        if (!hooks_.verify || verified_.count({a, b})) return;
        const auto fa = features_.find(a), fb = features_.find(b);
        if (fa == features_.end() || fb == features_.end()) return;
        verified_.insert({a, b});
        const auto e = hooks_.verify(a, fa->second, b, fb->second);
        if (!e) return;
        ++stats_.edges_verified;
        AcceptEdge(*e, t);
    }

    void AcceptEdge(const LoopEdgeValue& e, core::Timestamp t) {
        broker_edges_.emplace(KeyOf(e), e);
        // The endpoints keep the edge so any later broker can be re-sent it.
        for (AgentId to : {e.agent_a, e.agent_b}) {
            if (to == self_) {
                edges_.emplace(KeyOf(e), e);
            } else {
                SendEdge(to, e, t);
            }
        }
        dirty_ = true;
    }

public:
    // --- D175: coordination through the radio (MULTIAGENT_PIPELINE) ---------------------------------------------
    struct ComponentView {
        AgentId broker = 0;
        std::vector<AgentId> members;
        core::EpochValue epoch;
    };
    /// This agent's connected component as its membership sees it now (§15.4: lowest id brokers, epoch accepted).
    ComponentView Component() const { return {mem_.broker(), mem_.ComponentMembers(), mem_.accepted()}; }
    /// The newest heartbeat received from `a` (D175 item 4: peer poses reach the planner only through what was
    /// heard on the radio), or nothing if none arrived. Read-only.
    std::optional<core::HeartbeatValue> LastHeartbeat(AgentId a) const { return mem_.LastHeartbeat(a); }
    /// The highest SUBMAP_PATCH version `to` has acknowledged (0 = none): the next patch carries what changed since.
    uint64_t AckedPatchVersion(AgentId to) {
        const auto& d = ep_.Delivered(to, core::MessageClass::SUBMAP_PATCH);
        return d.empty() ? 0 : *d.rbegin();
    }
    void SendPatch(AgentId to, const core::SubmapPatchValue& p, core::Timestamp t) {
        if (to == self_) return;
        OutMessage m;
        m.cls = core::MessageClass::SUBMAP_PATCH;
        m.key = p.version;
        m.created = t;
        m.ttl = core::Duration::SecondsFromDouble(p_.patch_ttl_s);
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeSubmapPatch(p, self, to, ts, seq); };
        Plan(m, t);
        ep_.Enqueue(to, std::move(m));
    }
    /// To this agent's component broker; the broker's own bid is delivered locally at once (no radio hop).
    void SendBid(const core::TaskBidValue& b, core::Timestamp t) {
        const AgentId to = mem_.broker();
        if (to == self_) return DeliverLocal(core::MessageClass::TASK_BID, t, [&](core::DecodedMessage& d) {
            d.has_task_bid = true, d.task_bid = b;
        });
        OutMessage m;
        m.cls = core::MessageClass::TASK_BID;
        m.key = b.seq;
        m.created = t;
        m.ttl = core::Duration::SecondsFromDouble(p_.bid_ttl_s);
        const AgentId self = self_;
        const core::WireCodec* codec = &codec_;
        m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeTaskBid(b, self, to, ts, seq); };
        Plan(m, t);
        ep_.Enqueue(to, std::move(m));
    }
    /// Broker: the assignment to every member of `c` (self locally).
    void SendConsensus(const core::TaskConsensusValue& c, core::Timestamp t) {
        for (AgentId to : c.members) {
            if (to == self_) {
                DeliverLocal(core::MessageClass::TASK_CONSENSUS, t, [&](core::DecodedMessage& d) {
                    d.has_task_consensus = true, d.task_consensus = c;
                });
                continue;
            }
            OutMessage m;
            m.cls = core::MessageClass::TASK_CONSENSUS;
            m.key = c.seq;
            m.created = t;
            m.ttl = core::Duration::SecondsFromDouble(p_.consensus_ttl_s);
            const AgentId self = self_;
            const core::WireCodec* codec = &codec_;
            m.encode = [=](uint64_t seq, core::Timestamp ts) { return codec->EncodeTaskConsensus(c, self, to, ts, seq); };
            Plan(m, t);
            ep_.Enqueue(to, std::move(m));
        }
    }

    /// EVALUATION ONLY (arch §42 item 7: "at least one false closure is generated intentionally"): plant `e` at the
    /// broker exactly as if its verify hook had returned it, so it travels and is solved like any verified edge and
    /// only the robust stage can reject it. Returns false (nothing planted) unless this node is the broker.
    bool InjectVerifiedEdge(const LoopEdgeValue& e, core::Timestamp t) {
        if (!mem_.IsBroker()) return false;
        ++stats_.edges_injected;
        AcceptEdge(e, t);
        return true;
    }

private:

    void MaybeSolve(core::Timestamp t) {
        if (!hooks_.solve) return;
        const std::vector<AgentId> members = mem_.Members();
        if (!dirty_ && members == last_solve_members_) return;
        if (last_solve_ && (t - *last_solve_).toSeconds() < p_.solve_interval_s) return;
        last_solve_members_ = members;
        SolveInput in;
        in.members = members;
        for (AgentId a : members) {
            if (a == self_) {
                in.graphs[a] = Snapshot();
            } else if (snapshots_.count(a)) {
                in.graphs[a] = snapshots_.at(a);
            }
        }
        // §3.4 prerequisite hold: an edge waits until both endpoints' snapshots cover it.
        held_count_ = 0;
        auto covers = [&](AgentId a, uint64_t kf) {
            const auto it = in.graphs.find(a);
            if (it == in.graphs.end()) return false;
            for (const auto& n : it->second.nodes)
                if (n.keyframe == kf) return true;
            return false;
        };
        std::set<EdgeKey> used;
        for (const auto& [k, e] : broker_edges_) {
            const bool in_members = std::binary_search(members.begin(), members.end(), e.agent_a) &&
                                    std::binary_search(members.begin(), members.end(), e.agent_b);
            if (!in_members) continue;
            if (!covers(e.agent_a, e.keyframe_a) || !covers(e.agent_b, e.keyframe_b)) {
                ++held_count_;
                continue;
            }
            in.edges.push_back(e);
            used.insert(k);
        }
        // Gauge (§5): the anchor if present; else the lowest member already in G(anchor).
        in.anchor = mem_.anchor();
        if (std::binary_search(members.begin(), members.end(), in.anchor)) {
            in.gauge_agent = in.anchor;
        } else {
            bool found = false;
            for (AgentId a : members) {
                const auto it = in.graphs.find(a);
                if (it == in.graphs.end() || it->second.agent_frames.empty() || it->second.anchor.agent != in.anchor)
                    continue;
                in.gauge_agent = a;
                in.gauge_T_G_W = FromPoseValue(it->second.agent_frames.front().T_G_W);
                found = true;
                break;
            }
            if (!found) {   // no member is in G(anchor): this component starts its own frame
                in.anchor = self_;
                in.gauge_agent = self_;
            }
        }
        const SolveOutput placed = hooks_.solve(in);
        ++stats_.solves;
        last_solve_ = t;
        dirty_ = false;

        PoseGraphValue est;
        est.has_epoch = true;
        est.epoch = mem_.NextEpoch(members);
        mem_.Issued(est.epoch);
        est.anchor = {in.anchor, uint32_t(est.epoch.counter)};
        for (AgentId a : members) {
            const auto it = placed.find(a);
            PoseValue v{0, 0, 0, 0, 0, 0, 0};   // qw = 0 marks "member, not placed"
            if (it != placed.end()) v = ToPoseValue(it->second);
            est.agent_frames.push_back({a, v});
        }
        EstimateRecord r;
        r.epoch = est.epoch;
        r.members = members;
        r.anchor = in.anchor;
        r.digest = EstimateDigest(est);
        r.edges = used;
        issued_.emplace(est.epoch, r);
        for (AgentId a : members) {
            if (a == self_) continue;
            SendGraph(a, est, t, /*supersedes=*/true);   // a newer estimate replaces an unsent older one
        }
        known_g_.insert({est.anchor.agent, est.anchor.epoch});
        Adopt(est, members);
    }

    AgentId self_;
    std::vector<AgentId> fleet_;
    const core::WireCodec& codec_;
    Endpoint ep_;
    Membership mem_;
    NodeHooks hooks_;
    NodeParams p_;
    NodeStats stats_;

    // agent state
    std::vector<core::PoseNodeValue> nodes_;
    std::vector<LoopEdgeValue> odometry_;   // intra-agent, in W_self
    std::map<uint64_t, core::GlobalDescriptorValue> descriptors_;
    std::map<EdgeKey, LoopEdgeValue> edges_;
    size_t snapshot_nodes_ = 0;
    std::optional<core::Timestamp> last_snapshot_, next_heartbeat_;
    std::optional<core::SE3> T_G_W_;
    std::optional<EstimateRecord> adopted_;
    std::set<std::pair<AgentId, uint32_t>> known_g_;
    uint64_t edge_msg_key_ = 0, graph_msg_key_ = 0;
    bool resyncing_ = false, resync_snapshot_ = false;
    struct DeferredAnswer {   // D174: a MATCH_REQUEST whose features were not ready when it arrived
        AgentId to = 0;
        uint64_t keyframe = 0;
        core::Timestamp since;
    };
    std::map<uint64_t, DeferredAnswer> deferred_answers_;   // by request id
    AgentId T_G_W_anchor_ = 0;   // D171: sends to a new broker are tagged resync

    // broker state (soft: rebuilt from agents after an election change)
    std::map<KeyframeRef, core::GlobalDescriptorValue> descriptor_db_;
    std::set<std::pair<KeyframeRef, KeyframeRef>> candidates_, verified_;
    std::map<KeyframeRef, core::FeatureResponseValue> features_;
    std::map<KeyframeRef, Request> requests_;
    std::map<KeyframeRef, core::Timestamp> own_waiting_;   // D174: own keyframes whose features were not ready
    std::map<uint64_t, KeyframeRef> request_owner_;
    uint64_t request_counter_ = 0;
    std::map<EdgeKey, LoopEdgeValue> broker_edges_;
    std::map<AgentId, PoseGraphValue> snapshots_;
    bool dirty_ = false;
    std::optional<core::Timestamp> last_solve_;
    std::vector<AgentId> last_solve_members_;
    size_t held_count_ = 0;
    std::map<core::EpochValue, EstimateRecord, bool (*)(const core::EpochValue&, const core::EpochValue&)> issued_{
        &CollabNode::EpochLess};
    // 3.4 selection (soft broker state, like the above)
    struct Pending {
        uint64_t id = 0;
        double kappa = 1.0;
        core::Timestamp since;
    };
    FetchCostPredictor cost_;
    std::map<std::pair<KeyframeRef, KeyframeRef>, Pending> pending_;
    uint64_t pending_id_ = 0;
    double tokens_ = 0.0;
    std::optional<core::Timestamp> last_select_;
    std::optional<SelectionReport> last_selection_;
};

}  // namespace ariadne::collaboration
