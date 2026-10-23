#pragma once
// ---------------------------------------------------------------------------
// The link seam and a discrete-event datagram network (design COLLAB_3.2 §2).
//
// WHAT THIS IS. The protocol (outbox, ARQ, election) never touches a socket. It
// hands datagrams to `Network`, which decides, on the simulated clock, whether
// and when each one arrives. That decision comes from a `LinkModel`:
//   - 3.3's comms broker node implements it from RFComms link state plus
//     ray-cast wall attenuation (arch §5.7);
//   - tests use `ScriptedLinkModel`, wrapped by `FaultPlan` for partitions and
//     disconnects.
// Byte accounting stays at the serializer (§15.7); this layer only moves the
// framed payloads the serializer already counted.
//
// DATAGRAM, NOT MESSAGE. Loss is drawn per datagram (one MTU-sized fragment).
// A 15-datagram pose graph at 30% loss arrives whole with p = 0.7^15 ≈ 0.5%,
// so reassembly and selective retransmission live in the protocol above.
//
// TIMING. A directed link transmits one datagram at a time: a datagram departs
// at max(send time, link free) + size/bandwidth, then arrives `latency` later.
// So queueing behind a large message is modelled, which is what makes byte
// budgets cost time as well as bytes.
//
// DETERMINISM. The loss/duplicate/reorder draws are Philox keyed by
// (seed, per-link datagram index, link). Two links never share a stream, and
// the outcome of link A->B does not depend on how traffic on C->D interleaves.
// ---------------------------------------------------------------------------
#include "core/random/philox.hpp"
#include "core/random/seed.hpp"
#include "core/serialization/serializer.hpp"
#include "core/time/timestamp.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ariadne::collaboration {

using core::AgentId;
using core::Duration;
using core::Timestamp;

/// One directed link at one instant. `bandwidth_Bps == 0` means unlimited.
struct LinkState {
    bool up = false;
    double latency_s = 0.0;
    double loss_p = 0.0;
    double bandwidth_Bps = 0.0;
};

/// The seam 3.3 implements. Directed: Sample(a, b) may differ from Sample(b, a)
/// (asymmetric links are real, and the election must survive them, §4).
class LinkModel {
public:
    virtual ~LinkModel() = default;
    virtual LinkState Sample(AgentId from, AgentId to, Timestamp t) const = 0;
    /// The state one datagram of `wire_bytes` (payload + the 28 B UDP/IP header) sees. A physical model's loss
    /// depends on the size (RFComms: PER over the datagram's bytes); the default ignores it (D171, SIMCOLLAB_3.2 §3).
    virtual LinkState SampleSized(AgentId from, AgentId to, Timestamp t, double /*wire_bytes*/) const {
        return Sample(from, to, t);
    }
};

/// A default state for every pair, with per-directed-link overrides.
class ScriptedLinkModel : public LinkModel {
public:
    explicit ScriptedLinkModel(LinkState all = {true, 0.01, 0.0, 0.0}) : all_(all) {}

    void Set(AgentId from, AgentId to, LinkState s) { directed_[{from, to}] = s; }
    void SetBoth(AgentId a, AgentId b, LinkState s) { Set(a, b, s); Set(b, a, s); }

    LinkState Sample(AgentId from, AgentId to, Timestamp) const override {
        const auto it = directed_.find({from, to});
        return it == directed_.end() ? all_ : it->second;
    }

private:
    LinkState all_;
    std::map<std::pair<AgentId, AgentId>, LinkState> directed_;
};

/// Scripted faults over any LinkModel: partitions, disconnects and extra
/// latency in time windows [t0, t1). A link is up only if the base model says
/// so and no fault cuts it.
class FaultPlan : public LinkModel {
public:
    explicit FaultPlan(const LinkModel& base) : base_(base) {}

    /// During [t0, t1) agents in different groups cannot talk. Agents listed
    /// in no group are isolated.
    void Partition(Timestamp t0, Timestamp t1, std::vector<std::vector<AgentId>> groups) {
        partitions_.push_back({t0, t1, std::move(groups)});
    }
    /// During [t0, t1) `agent` can neither send nor receive.
    void Disconnect(AgentId agent, Timestamp t0, Timestamp t1) { disconnects_.push_back({agent, t0, t1}); }
    /// During [t0, t1) every link gains `extra_s` of latency.
    void ExtraLatency(Timestamp t0, Timestamp t1, double extra_s) { latency_.push_back({t0, t1, extra_s}); }

    LinkState Sample(AgentId from, AgentId to, Timestamp t) const override {
        return Faulted(base_.Sample(from, to, t), from, to, t);
    }
    LinkState SampleSized(AgentId from, AgentId to, Timestamp t, double wire_bytes) const override {
        return Faulted(base_.SampleSized(from, to, t, wire_bytes), from, to, t);
    }

private:
    LinkState Faulted(LinkState s, AgentId from, AgentId to, Timestamp t) const {
        for (const auto& p : partitions_)
            if (In(t, p.t0, p.t1) && GroupOf(p.groups, from) != GroupOf(p.groups, to)) s.up = false;
        for (const auto& d : disconnects_)
            if (In(t, d.t0, d.t1) && (d.agent == from || d.agent == to)) s.up = false;
        for (const auto& l : latency_)
            if (In(t, l.t0, l.t1)) s.latency_s += l.extra_s;
        return s;
    }

    struct PartitionWindow { Timestamp t0, t1; std::vector<std::vector<AgentId>> groups; };
    struct DisconnectWindow { AgentId agent; Timestamp t0, t1; };
    struct LatencyWindow { Timestamp t0, t1; double extra_s; };

    static bool In(Timestamp t, Timestamp t0, Timestamp t1) { return t0 <= t && t < t1; }
    /// Index of the group holding `a`; agents in no group get a unique negative id.
    static int64_t GroupOf(const std::vector<std::vector<AgentId>>& groups, AgentId a) {
        for (size_t g = 0; g < groups.size(); ++g)
            if (std::find(groups[g].begin(), groups[g].end(), a) != groups[g].end()) return int64_t(g);
        return -1 - int64_t(a);
    }

    const LinkModel& base_;
    std::vector<PartitionWindow> partitions_;
    std::vector<DisconnectWindow> disconnects_;
    std::vector<LatencyWindow> latency_;
};

/// Per-datagram operators the link model does not express: duplication and
/// reordering (a datagram held back so later ones overtake it).
struct DatagramFaults {
    double duplicate_p = 0.0;
    double reorder_p = 0.0;
    double reorder_delay_s = 0.05;
    double jitter_s = 0.0;   // uniform extra delay in [0, jitter_s)
};

struct Datagram {
    AgentId from = 0;
    AgentId to = 0;
    std::vector<uint8_t> payload;
    Timestamp sent;
    Timestamp arrive;
    uint64_t id = 0;   // network-wide, in send order; breaks arrival-time ties
};

struct NetworkStats {
    uint64_t sent = 0, lost_link_down = 0, lost_random = 0, duplicated = 0, reordered = 0, delivered = 0;
};

class Network {
public:
    Network(const LinkModel& links, uint64_t root_seed, DatagramFaults faults = {})
        : links_(links), seed_(core::DeriveSeed(root_seed, core::seed_purpose::kCommsLoss)), faults_(faults) {}

    /// Queue one datagram. The payload must fit one MTU (the protocol fragments).
    void Send(AgentId from, AgentId to, std::vector<uint8_t> payload, Timestamp t) {
        if (payload.size() > core::kDatagramPayloadBytes)
            throw std::invalid_argument("Network::Send: payload exceeds one datagram; fragment it first");
        ++stats_.sent;
        // Bytes on the wire include the 28 B UDP/IP header; a size-aware model prices the loss at that size.
        const double wire = double(payload.size() + core::kDatagramOverheadBytes);
        const LinkState s = links_.SampleSized(from, to, t, wire);
        LinkTx& tx = tx_[{from, to}];
        const uint64_t index = tx.next_index++;
        if (!s.up) {
            ++stats_.lost_link_down;
            return;
        }
        // Serialization delay on a busy directed link.
        const Timestamp start = std::max(t, tx.free_at);
        const Duration tx_time =
            s.bandwidth_Bps > 0.0 ? Duration::SecondsFromDouble(wire / s.bandwidth_Bps) : Duration{0};
        tx.free_at = start + tx_time;

        core::PhiloxStream rng(seed_, index, core::SiteFromHash64((uint64_t(from) << 32) | to));
        const double u_loss = rng.NextUniform(), u_dup = rng.NextUniform(), u_reo = rng.NextUniform(),
                     u_jit = rng.NextUniform();
        if (u_loss < s.loss_p) {
            ++stats_.lost_random;
            return;
        }
        double delay = s.latency_s + u_jit * faults_.jitter_s;
        if (u_reo < faults_.reorder_p) {
            delay += faults_.reorder_delay_s;
            ++stats_.reordered;
        }
        Datagram d{from, to, std::move(payload), t, tx.free_at + Duration::SecondsFromDouble(delay), next_id_++};
        if (u_dup < faults_.duplicate_p) {
            Datagram copy = d;
            copy.arrive = d.arrive + Duration::SecondsFromDouble(faults_.reorder_delay_s);
            copy.id = next_id_++;
            ++stats_.duplicated;
            in_flight_.push(std::move(copy));
        }
        in_flight_.push(std::move(d));
    }

    /// Every datagram arriving at or before `t`, in (arrival, id) order.
    std::vector<Datagram> DeliverUntil(Timestamp t) {
        std::vector<Datagram> out;
        while (!in_flight_.empty() && in_flight_.top().arrive <= t) {
            out.push_back(in_flight_.top());
            in_flight_.pop();
        }
        stats_.delivered += out.size();
        return out;
    }

    bool idle() const { return in_flight_.empty(); }
    const NetworkStats& stats() const { return stats_; }
    const LinkModel& links() const { return links_; }

private:
    struct LinkTx { Timestamp free_at; uint64_t next_index = 0; };
    struct Later {
        bool operator()(const Datagram& a, const Datagram& b) const {
            return a.arrive != b.arrive ? a.arrive > b.arrive : a.id > b.id;
        }
    };

    const LinkModel& links_;
    uint64_t seed_;
    DatagramFaults faults_;
    std::map<std::pair<AgentId, AgentId>, LinkTx> tx_;
    std::priority_queue<Datagram, std::vector<Datagram>, Later> in_flight_;
    uint64_t next_id_ = 0;
    NetworkStats stats_;
};

}  // namespace ariadne::collaboration
