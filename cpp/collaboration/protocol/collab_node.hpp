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
#include "collaboration/protocol/membership.hpp"
#include "collaboration/protocol/transport.hpp"
#include "core/math/lie.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
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

// --- hooks -----------------------------------------------------------------------

struct KeyframeRef {
    AgentId agent = 0;
    uint64_t keyframe = 0;
    bool operator<(const KeyframeRef& o) const { return std::tie(agent, keyframe) < std::tie(o.agent, o.keyframe); }
    bool operator==(const KeyframeRef& o) const { return agent == o.agent && keyframe == o.keyframe; }
};

struct SolveInput {
    std::vector<AgentId> members;
    std::map<AgentId, PoseGraphValue> graphs;   // each member's latest snapshot (nodes in W_member)
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
    /// Agent: local features of one of its keyframes, for a MATCH_RESPONSE.
    std::function<core::FeatureResponseValue(uint64_t keyframe)> features;
    /// Broker: geometric verification of a candidate pair, given both feature sets.
    std::function<std::optional<LoopEdgeValue>(const KeyframeRef& a, const core::FeatureResponseValue& fa,
                                               const KeyframeRef& b, const core::FeatureResponseValue& fb)>
        verify;
    std::function<SolveOutput(const SolveInput&)> solve;
};

struct NodeParams {
    MembershipParams membership;
    EndpointParams endpoint;
    double descriptor_ttl_s = 120.0;
    double response_ttl_s = 120.0;
    double request_retry_s = 10.0;      // broker re-requests features not received by then
    double request_give_up_s = 120.0;
    uint64_t snapshot_every_keyframes = 10;
    double snapshot_max_interval_s = 10.0;
    double solve_interval_s = 2.0;
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
          mem_(self, p.membership), hooks_(std::move(hooks)), p_(p) {}

    // --- application input ---

    /// A keyframe of this agent's odometry: pose in W_self, plus its global descriptor.
    void AddKeyframe(uint64_t keyframe, int64_t stamp_ns, const core::SE3& T_W_C,
                     const core::GlobalDescriptorValue& descriptor, core::Timestamp t) {
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
        // 3. snapshot cadence to my broker
        if (!mem_.IsBroker()) MaybeSnapshot(t);
        // 4. broker duties
        if (mem_.IsBroker()) {
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
    const std::optional<EstimateRecord>& adopted() const { return adopted_; }
    const std::map<core::EpochValue, EstimateRecord, bool (*)(const core::EpochValue&, const core::EpochValue&)>&
    issued() const {
        return issued_;
    }
    std::optional<core::SE3> T_G_W() const { return T_G_W_; }
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
        ep_.Enqueue(to, std::move(m));
    }

    PoseGraphValue Snapshot() const {
        PoseGraphValue g;
        g.nodes = nodes_;
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
        SendGraph(mem_.broker(), Snapshot(), t, /*supersedes=*/true);
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
        for (const auto& [kf, g] : descriptors_)
            if (!sent.count(kf)) SendDescriptor(kf, t);
        for (const auto& [k, e] : edges_) SendEdge(b, e, t);
        snapshot_nodes_ = 0;   // force a fresh snapshot to the new broker
        last_snapshot_.reset();
    }

    // --- receiving ---

    /// §15.5: a W frame of a fleet agent, or a G frame this node adopted.
    bool KnownFrame(const FrameTag& f) const {
        if (f.epoch == 0) return std::find(fleet_.begin(), fleet_.end(), f.agent) != fleet_.end();
        return known_g_.count({f.agent, f.epoch}) > 0;
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
                    snapshots_[m.sender] = std::move(m.pose_graph);
                    dirty_ = true;
                }
                break;
            default: break;
        }
    }

    void AnswerRequest(const core::DecodedMessage& m, core::Timestamp t) {
        if (!hooks_.features || !descriptors_.count(m.match_request.candidate_keyframe)) return;
        core::FeatureResponseValue f = hooks_.features(m.match_request.candidate_keyframe);
        f.request_id = m.match_request.request_id;
        f.keyframe_id = m.match_request.candidate_keyframe;
        OutMessage o;
        o.cls = core::MessageClass::MATCH_RESPONSE;
        o.key = f.request_id;
        o.created = t;
        o.ttl = core::Duration::SecondsFromDouble(p_.response_ttl_s);
        const AgentId self = self_, to = m.sender;
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
            known_g_.insert({g.anchor.agent, g.anchor.epoch});
        }
    }

    // --- broker ---

    void BrokerOnDescriptor(const KeyframeRef& q, const core::GlobalDescriptorValue& g, core::Timestamp t) {
        if (!descriptor_db_.emplace(q, g).second || !hooks_.retrieve) return;
        for (const KeyframeRef& c : hooks_.retrieve(q, descriptor_db_)) {
            if (c.agent == q.agent) continue;
            const auto pair = std::minmax(q, c);
            if (!candidates_.insert(pair).second) continue;
            ++stats_.candidates;
            for (const KeyframeRef& end : {q, c}) RequestFeatures(end, t);
            TryVerify(pair.first, pair.second, t);
        }
    }

    void RequestFeatures(const KeyframeRef& k, core::Timestamp t) {
        if (features_.count(k)) return;
        if (k.agent == self_) {
            if (hooks_.features) features_[k] = hooks_.features(k.keyframe);
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
        broker_edges_.emplace(KeyOf(*e), *e);
        // The endpoints keep the edge so any later broker can be re-sent it.
        for (AgentId to : {e->agent_a, e->agent_b}) {
            if (to == self_) {
                edges_.emplace(KeyOf(*e), *e);
            } else {
                SendEdge(to, *e, t);
            }
        }
        dirty_ = true;
    }

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
    std::map<uint64_t, core::GlobalDescriptorValue> descriptors_;
    std::map<EdgeKey, LoopEdgeValue> edges_;
    size_t snapshot_nodes_ = 0;
    std::optional<core::Timestamp> last_snapshot_, next_heartbeat_;
    std::optional<core::SE3> T_G_W_;
    std::optional<EstimateRecord> adopted_;
    std::set<std::pair<AgentId, uint32_t>> known_g_;
    uint64_t edge_msg_key_ = 0, graph_msg_key_ = 0;

    // broker state (soft: rebuilt from agents after an election change)
    std::map<KeyframeRef, core::GlobalDescriptorValue> descriptor_db_;
    std::set<std::pair<KeyframeRef, KeyframeRef>> candidates_, verified_;
    std::map<KeyframeRef, core::FeatureResponseValue> features_;
    std::map<KeyframeRef, Request> requests_;
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
};

}  // namespace ariadne::collaboration
