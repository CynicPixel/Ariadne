#pragma once
// ---------------------------------------------------------------------------
// The 3.2 transport: one `Endpoint` per agent (design COLLAB_3.2 §3).
//
// WHAT IT DOES, data in -> data out:
//   Enqueue(dest, OutMessage)  -- an application message, NOT yet encoded
//   Pump(t, network)           -- transmit what priority, reachability and the
//                                 send budget allow; resend unacked fragments
//   Receive(datagram, t)       -- reassemble, decode, apply per-class delivery;
//                                 returns the messages the application may act on
//
// WHY EACH PIECE EXISTS
// - Encoding happens at FIRST TRANSMISSION, with the reliable sequence number
//   assigned there. A queued message that is superseded, expires or is
//   cancelled therefore never consumed a number and cannot open a hole in a
//   reliable stream (the Phase-1 Inbox's failure mode, COLLAB_3.2 §1).
// - Fragments carry (msg_id, index, count). Loss is per datagram; whole-message
//   retransmission would not converge (0.7^15 ≈ 0.5% for a 15-datagram graph
//   at 30% loss). Acks name message ids: `done` ids plus a fragment bitmap per
//   partially received message, so only missing fragments are resent.
// - Which messages are retransmitted: reliable classes (POSE_EDGE, POSE_GRAPH)
//   until acked or abandoned on a broker change; best-effort messages that span
//   several datagrams (feature responses) until acked, expired or cancelled.
//   Single-datagram best-effort messages (descriptors, requests) are
//   fire-and-forget: the application re-requests on timeout. Heartbeats and
//   acks are periodic and never tracked.
// - Delivery per class (§3.1 table): heartbeats latest-wins; edges and graphs
//   in order per (sender, class); descriptors/requests/responses/submaps as
//   idempotent sets keyed by content id. A hole in an unordered set stalls
//   nothing, which is what makes TTLs and cancellation safe.
// - Reliable sequence = (incarnation << 40) | n. Abandoning reliable messages
//   (broker change, §3.2) would otherwise leave the receiver waiting forever
//   for sequence numbers that were spent and then dropped; instead the sender
//   starts a new incarnation and the receiver, on seeing it, resets the stream.
//   The abandoned content is re-sent by the application (broker state is soft).
//
// Every first transmission and every retransmission passes through the codec's
// accounting row into the ByteLedger (retransmit rows flagged, §7).
// ---------------------------------------------------------------------------
#include "collaboration/protocol/network.hpp"
#include "core/serialization/serializer.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ariadne::collaboration {

using core::AckValue;
using core::ByteLedger;
using core::DecodedMessage;
using core::MessageClass;
using core::SerializedMessage;
using core::WireCodec;

// --- Fragments ---------------------------------------------------------------

struct FragmentHeader {
    uint32_t msg_id = 0;
    uint8_t index = 0;
    uint8_t count = 0;
};

/// Split a framed message (codec output) into datagram payloads, each led by
/// the 6 B fragment header. The sizes match the codec's kFragmented charge.
inline std::vector<std::vector<uint8_t>> Fragment(const std::vector<uint8_t>& framed, uint32_t msg_id) {
    const uint64_t n = core::FragmentCount(framed.size());
    if (n > core::kMaxFragments) throw std::invalid_argument("Fragment: message exceeds 255 fragments");
    std::vector<std::vector<uint8_t>> out(n);
    for (uint64_t i = 0; i < n; ++i) {
        const size_t lo = i * core::kFragmentPayloadBytes;
        const size_t hi = std::min<size_t>(framed.size(), lo + core::kFragmentPayloadBytes);
        auto& d = out[i];
        d.reserve(core::kFragmentHeaderBytes + (hi - lo));
        for (int b = 0; b < 4; ++b) d.push_back(uint8_t((msg_id >> (8 * b)) & 0xFF));
        d.push_back(uint8_t(i));
        d.push_back(uint8_t(n));
        d.insert(d.end(), framed.begin() + lo, framed.begin() + hi);
    }
    return out;
}

inline bool ParseFragment(const std::vector<uint8_t>& d, FragmentHeader* h) {
    if (d.size() < core::kFragmentHeaderBytes) return false;
    h->msg_id = uint32_t(d[0]) | (uint32_t(d[1]) << 8) | (uint32_t(d[2]) << 16) | (uint32_t(d[3]) << 24);
    h->index = d[4];
    h->count = d[5];
    return h->count > 0 && h->index < h->count;
}

inline bool BitGet(const std::vector<uint8_t>& bits, size_t i) {
    return i / 8 < bits.size() && (bits[i / 8] >> (i % 8)) & 1;
}
inline void BitSet(std::vector<uint8_t>& bits, size_t i) {
    if (bits.size() <= i / 8) bits.resize(i / 8 + 1, 0);
    bits[i / 8] |= uint8_t(1u << (i % 8));
}

// --- Classes -----------------------------------------------------------------

/// §15.3 strict priority (1 = highest). MATCH_REQUEST rides with descriptors.
inline int PriorityOf(MessageClass c) {
    switch (c) {
        case MessageClass::HEARTBEAT:
        case MessageClass::ACK: return 1;
        case MessageClass::POSE_EDGE:
        case MessageClass::POSE_GRAPH: return 2;
        case MessageClass::GLOBAL_DESCRIPTOR:
        case MessageClass::MATCH_REQUEST: return 3;
        case MessageClass::MATCH_RESPONSE:
        case MessageClass::LOCAL_DESCRIPTOR: return 4;
        default: return 5;
    }
}
inline bool IsReliableClass(MessageClass c) { return c == MessageClass::POSE_EDGE || c == MessageClass::POSE_GRAPH; }
inline bool IsLatestWinsClass(MessageClass c) { return c == MessageClass::HEARTBEAT || c == MessageClass::ACK; }
inline constexpr int kPriorities = 5;
inline constexpr int kIncarnationShift = 40;

// --- Endpoint ----------------------------------------------------------------

/// An application message waiting for its first transmission. `encode` runs
/// then, with the sequence number assigned then.
struct OutMessage {
    MessageClass cls = MessageClass::HEARTBEAT;
    std::function<SerializedMessage(uint64_t sequence, core::Timestamp t)> encode;
    core::Timestamp created;
    std::optional<core::Duration> ttl;   // best-effort expiry (§3.3); none for edges/graphs
    uint64_t key = 0;                    // content id: keyframe / request / submap id; bookkeeping + cancel
    bool supersedes = false;             // a POSE_GRAPH snapshot: replaces a queued, unsent one
    double predicted_wire_bytes = -1.0;  // the sender's prediction (COLLAB_3.2 §7, D171); ledger only
    bool resync = false;                 // a resend to a new broker (overhead row, D171)
};

struct EndpointParams {
    double send_budget_Bps = 0.0;      // per destination; 0 = unlimited (the network still models bandwidth)
    double send_burst_bytes = 64 * 1024;
    double rto_min_s = 1.0;
    double rto_max_s = 8.0;            // bounds the drain time under heavy loss; unreachable peers are not sent to at all
    double ack_delay_s = 0.2;          // standalone ACK if no heartbeat carried it by then
    // An ack rides a heartbeat or a standalone ACK, which must fit one datagram
    // (SendImmediate throws otherwise): budget its encoded bytes, leave the rest pending.
    size_t max_ack_bytes = 900;        // heartbeat body ~200 B + ack <= 1466 B fragment payload
    int ack_repeats = 3;               // each done id rides this many acks: 1 - 0.3^3 = 97% at 30% loss
    size_t reorder_buffer = 4096;      // per reliable stream
    double reassembly_timeout_s = 120.0;   // an abandoned partial message is forgotten (matches the 120 s TTL)
};

struct EndpointStats {
    uint64_t first_tx_messages = 0, retransmitted_fragments = 0, acks_sent = 0;
    uint64_t delivered = 0, duplicates = 0, stale_heartbeats = 0, decode_rejected = 0;
    uint64_t expired = 0, cancelled = 0, superseded = 0, abandoned_reliable = 0, abandoned_best_effort = 0;
    uint64_t acked = 0;
    uint64_t abandoned_reliable_rx = 0;   // buffered messages dropped on a new incarnation
};

class Endpoint {
public:
    Endpoint(AgentId self, const WireCodec& codec, ByteLedger& ledger, EndpointParams p = {})
        : self_(self), codec_(codec), ledger_(ledger), p_(p) {
        if (codec.framing() != core::Framing::kFragmented)
            throw std::invalid_argument("Endpoint: the codec must charge fragment headers (Framing::kFragmented)");
    }

    AgentId self() const { return self_; }

    // --- send side ---

    void Enqueue(AgentId dest, OutMessage m) {
        auto& q = peer(dest).queue[PriorityOf(m.cls) - 1];
        if (m.supersedes) {
            for (auto it = q.begin(); it != q.end();) {
                if (it->supersedes && it->cls == m.cls) {
                    it = q.erase(it);
                    ++stats_.superseded;
                } else {
                    ++it;
                }
            }
        }
        q.push_back(std::move(m));
    }

    /// The policy hook (§16): cancel a queued or in-flight best-effort message
    /// of class 4-5 by its content key. Reliable classes cannot be cancelled.
    void Cancel(AgentId dest, MessageClass cls, uint64_t key) {
        if (IsReliableClass(cls) || PriorityOf(cls) < 4) throw std::invalid_argument("Cancel: only classes 4-5");
        Peer& pr = peer(dest);
        auto& q = pr.queue[PriorityOf(cls) - 1];
        const auto n = q.size();
        q.erase(std::remove_if(q.begin(), q.end(), [&](const OutMessage& m) { return m.cls == cls && m.key == key; }),
                q.end());
        stats_.cancelled += n - q.size();
        for (auto it = pr.inflight.begin(); it != pr.inflight.end();) {
            if (it->second.cls == cls && it->second.key == key) {
                it = pr.inflight.erase(it);
                ++stats_.cancelled;
            } else {
                ++it;
            }
        }
    }

    /// Only reachable destinations drain (store-and-forward = hold until
    /// reachable, §3.3). Set by the membership layer from symmetric neighbours.
    void SetReachable(AgentId dest, bool reachable) { peer(dest).reachable = reachable; }

    /// Election change (§3.2): unacked and queued reliable messages to `dest`
    /// are abandoned and counted; broker state is soft, they are re-sent to the
    /// new broker by the application.
    void AbandonReliable(AgentId dest) {
        Peer& pr = peer(dest);
        for (auto it = pr.inflight.begin(); it != pr.inflight.end();) {
            if (it->second.reliable) {
                it = pr.inflight.erase(it);
                ++stats_.abandoned_reliable;
            } else {
                ++it;
            }
        }
        auto& q = pr.queue[PriorityOf(MessageClass::POSE_EDGE) - 1];
        stats_.abandoned_reliable += q.size();
        q.clear();
        // New stream incarnation: the receiver skips the numbers spent on abandoned messages.
        ++pr.incarnation;
        for (MessageClass c : {MessageClass::POSE_EDGE, MessageClass::POSE_GRAPH}) pr.next_seq[uint8_t(c)] = 0;
    }

    /// Send a message now, outside the queues and the send budget: heartbeats
    /// (discovery must not wait behind data) and standalone acks.
    void SendImmediate(AgentId dest, MessageClass cls,
                       const std::function<SerializedMessage(uint64_t, core::Timestamp)>& encode, core::Timestamp t,
                       Network& net) {
        Peer& pr = peer(dest);
        SerializedMessage m = encode(pr.next_seq[uint8_t(cls)]++, t);
        if (core::FragmentCount(m.bytes.size()) != 1)
            throw std::invalid_argument("SendImmediate: periodic messages must fit one datagram");
        ledger_.Record(m.record);
        for (auto& d : Fragment(m.bytes, pr.next_msg_id++)) net.Send(self_, dest, std::move(d), t);
        if (cls == MessageClass::ACK) ++stats_.acks_sent;
    }

    /// The pending ack for `peer`, consumed (a heartbeat to `peer` carries it).
    AckValue TakeAck(AgentId from) {
        Peer& pr = peer(from);
        AckValue a;
        // Conservative FlatBuffers sizes: 4 B per done id; a partial entry is a
        // table (vtable + offsets ~ 24 B) plus its bitmap rounded to 4 B.
        size_t bytes = 32;
        for (auto it = pr.ack_done.begin(); it != pr.ack_done.end() && bytes + 4 <= p_.max_ack_bytes;) {
            a.done.push_back(it->first);
            bytes += 4;
            it = --it->second == 0 ? pr.ack_done.erase(it) : std::next(it);
        }
        for (const auto& [id, r] : pr.reassembly) {
            const size_t cost = 24 + 4 * ((r.received.size() + 3) / 4);
            if (bytes + cost > p_.max_ack_bytes) break;
            a.partial.push_back({id, r.received});
            bytes += cost;
        }
        pr.ack_due.reset();
        return a;
    }

    void Pump(core::Timestamp t, Network& net) {
        for (auto& [dest, pr] : peers_) {
            // Standalone ack when no heartbeat carried it in time.
            if (pr.ack_due && (t - *pr.ack_due).toSeconds() >= p_.ack_delay_s) {
                const AckValue a = TakeAck(dest);
                if (!a.empty())
                    SendImmediate(dest, MessageClass::ACK,
                                  [&](uint64_t seq, core::Timestamp ts) { return codec_.EncodeAck(a, self_, dest, ts, seq); },
                                  t, net);
            }
            ExpireQueued(pr, t);
            ExpireInflight(pr, t);
            PruneReassembly(pr, t);
            if (!pr.reachable) continue;
            Refill(pr, t);
            for (int p = 0; p < kPriorities; ++p) {
                // 1. retransmissions due at this priority
                for (auto& [id, m] : pr.inflight) {
                    if (PriorityOf(m.cls) != p + 1 || m.next_first < m.fragments.size() || !m.tracked) continue;
                    if ((t - m.last_tx).toSeconds() < m.rto_s) continue;
                    if (!Resend(pr, m, t, net, dest)) goto next_peer;
                }
                // 2. first transmission of partially sent messages, then new ones
                for (auto& [id, m] : pr.inflight)
                    if (PriorityOf(m.cls) == p + 1 && !SendFirst(dest, pr, m, t, net)) goto next_peer;
                while (!pr.queue[p].empty()) {
                    OutMessage om = std::move(pr.queue[p].front());
                    pr.queue[p].pop_front();
                    const uint32_t id = Start(pr, std::move(om), t);
                    if (!SendFirst(dest, pr, pr.inflight.at(id), t, net)) goto next_peer;
                }
            }
        next_peer:
            // Untracked messages are done once their first transmission is complete.
            for (auto it = pr.inflight.begin(); it != pr.inflight.end();)
                it = (!it->second.tracked && it->second.next_first == it->second.fragments.size())
                         ? pr.inflight.erase(it)
                         : std::next(it);
        }
    }

    // --- receive side ---

    std::vector<DecodedMessage> Receive(const Datagram& d, core::Timestamp t) {
        std::vector<DecodedMessage> out;
        FragmentHeader h;
        if (!ParseFragment(d.payload, &h)) {
            ++stats_.decode_rejected;
            return out;
        }
        Peer& pr = peer(d.from);
        if (pr.done_ids.count(h.msg_id)) {   // a resend of a tracked message already complete: re-ack it
            ++stats_.duplicates;
            AckDone(pr, h.msg_id, t);
            return out;
        }
        Reassembly& r = pr.reassembly[h.msg_id];
        if (r.parts.empty()) {
            r.parts.resize(h.count);
            r.first_seen = t;
        }
        if (h.count != r.parts.size()) {
            ++stats_.decode_rejected;
            return out;
        }
        if (BitGet(r.received, h.index)) {
            ++stats_.duplicates;
        } else {
            BitSet(r.received, h.index);
            r.parts[h.index].assign(d.payload.begin() + core::kFragmentHeaderBytes, d.payload.end());
            ++r.have;
        }
        if (r.have < r.parts.size()) {
            if (r.parts.size() > 1 && !pr.ack_due) pr.ack_due = t;   // the bitmap goes out with the next ack
            return out;
        }
        std::vector<uint8_t> framed;
        for (auto& part : r.parts) framed.insert(framed.end(), part.begin(), part.end());
        const size_t nfrag = r.parts.size();
        pr.reassembly.erase(h.msg_id);

        DecodedMessage m;
        if (codec_.Decode(framed, &m) != WireCodec::Status::kOk || m.sender != d.from) {
            ++stats_.decode_rejected;
            return out;
        }
        // Only messages the sender tracks are acked (the same rule as Start's).
        if (IsTracked(m.cls, nfrag)) {
            pr.done_ids.insert(h.msg_id);
            AckDone(pr, h.msg_id, t);
        }
        if (m.has_ack) ApplyAck(pr, m.ack);
        if (m.has_heartbeat) ApplyAck(pr, m.heartbeat.ack);
        Deliver(pr, std::move(m), &out);
        stats_.delivered += out.size();
        return out;
    }

    // --- outcome accounting (COLLAB_3.2 §6) ---

    /// Reliable messages not yet acked by `dest` (queued or in flight).
    size_t UnackedReliable(AgentId dest) const {
        const auto it = peers_.find(dest);
        if (it == peers_.end()) return 0;
        size_t n = it->second.queue[PriorityOf(MessageClass::POSE_EDGE) - 1].size();
        for (const auto& [id, m] : it->second.inflight) n += m.reliable;
        return n;
    }
    /// Unacked reliable messages to `dest`: (class, key, seconds since last transmission; -1 = still queued).
    std::vector<std::tuple<MessageClass, uint64_t, double>> UnackedReliableDetail(AgentId dest, core::Timestamp t) const {
        std::vector<std::tuple<MessageClass, uint64_t, double>> out;
        const auto it = peers_.find(dest);
        if (it == peers_.end()) return out;
        for (const auto& m : it->second.queue[PriorityOf(MessageClass::POSE_EDGE) - 1]) out.emplace_back(m.cls, m.key, -1.0);
        for (const auto& [id, m] : it->second.inflight)
            if (m.reliable) out.emplace_back(m.cls, m.key, (t - m.last_tx).toSeconds());
        return out;
    }
    std::vector<AgentId> Peers() const {
        std::vector<AgentId> out;
        for (const auto& [id, pr] : peers_) out.push_back(id);
        return out;
    }
    size_t UnackedReliableTotal() const {
        size_t n = 0;
        for (const auto& [dest, pr] : peers_) n += UnackedReliable(dest);
        return n;
    }
    /// Content keys first transmitted to `dest` per class (§15.3 bookkeeping).
    const std::set<uint64_t>& Sent(AgentId dest, MessageClass cls) { return peer(dest).sent[uint8_t(cls)]; }
    /// Reliable messages acked by `dest`, by class and key.
    const std::set<uint64_t>& Delivered(AgentId dest, MessageClass cls) { return peer(dest).delivered[uint8_t(cls)]; }
    const EndpointStats& stats() const { return stats_; }

private:
    /// Sender and receiver agree on what is acked: reliable classes, global
    /// descriptors (single datagram, but the broker cannot re-request one it
    /// never heard of), and any multi-datagram message that is not periodic.
    static bool IsTracked(MessageClass c, size_t fragments) {
        return !IsLatestWinsClass(c) &&
               (IsReliableClass(c) || c == MessageClass::GLOBAL_DESCRIPTOR || fragments > 1);
    }

    struct Inflight {
        MessageClass cls;
        uint64_t key = 0;
        bool reliable = false;
        bool tracked = false;
        std::optional<core::Timestamp> expires;
        std::vector<std::vector<uint8_t>> fragments;
        std::vector<uint8_t> acked;   // bitmap
        size_t next_first = 0;        // fragments [0, next_first) have had their first transmission
        core::Timestamp last_tx;
        double rto_s = 1.0;
        core::ByteRecord record;      // the first transmission's row, for retransmit rows
    };
    struct Reassembly {
        std::vector<std::vector<uint8_t>> parts;
        std::vector<uint8_t> received;
        size_t have = 0;
        core::Timestamp first_seen;
    };
    struct Peer {
        bool reachable = false;
        std::deque<OutMessage> queue[kPriorities];
        std::map<uint32_t, Inflight> inflight;
        uint32_t next_msg_id = 1;   // 0 = "none"
        std::map<uint8_t, uint64_t> next_seq;
        uint64_t incarnation = 0;   // reliable streams to this peer
        double tokens = 0.0;
        std::optional<core::Timestamp> last_refill;
        std::map<uint8_t, std::set<uint64_t>> sent, delivered;
        // receive side
        std::map<uint32_t, Reassembly> reassembly;
        std::set<uint32_t> done_ids;
        std::map<uint32_t, int> ack_done;   // id -> acks it still rides
        std::optional<core::Timestamp> ack_due;
        std::map<uint8_t, uint64_t> latest_seq;   // latest-wins classes
        std::map<uint8_t, uint64_t> next_reliable;
        std::map<uint8_t, std::map<uint64_t, DecodedMessage>> reorder;
        std::map<uint8_t, std::set<uint64_t>> seen_keys;   // idempotent classes
    };

    Peer& peer(AgentId id) {
        auto it = peers_.find(id);
        if (it == peers_.end()) {
            it = peers_.emplace(id, Peer{}).first;
            it->second.tokens = p_.send_burst_bytes;
        }
        return it->second;
    }

    void Refill(Peer& pr, core::Timestamp t) {
        if (p_.send_budget_Bps <= 0.0) return;
        if (pr.last_refill) pr.tokens += (t - *pr.last_refill).toSeconds() * p_.send_budget_Bps;
        pr.tokens = std::min(pr.tokens, p_.send_burst_bytes);
        pr.last_refill = t;
    }
    bool Spend(Peer& pr, size_t datagram_bytes) {
        if (p_.send_budget_Bps <= 0.0) return true;
        const double cost = double(datagram_bytes + core::kDatagramOverheadBytes);
        if (pr.tokens < cost) return false;
        pr.tokens -= cost;
        return true;
    }

    void ExpireQueued(Peer& pr, core::Timestamp t) {
        for (auto& q : pr.queue) {
            const auto n = q.size();
            q.erase(std::remove_if(q.begin(), q.end(),
                                   [&](const OutMessage& m) { return m.ttl && t - m.created >= *m.ttl; }),
                    q.end());
            stats_.expired += n - q.size();
        }
    }
    void PruneReassembly(Peer& pr, core::Timestamp t) {
        for (auto it = pr.reassembly.begin(); it != pr.reassembly.end();)
            it = (t - it->second.first_seen).toSeconds() >= p_.reassembly_timeout_s ? pr.reassembly.erase(it)
                                                                                      : std::next(it);
    }

    void ExpireInflight(Peer& pr, core::Timestamp t) {
        for (auto it = pr.inflight.begin(); it != pr.inflight.end();) {
            if (it->second.expires && t >= *it->second.expires) {
                // Expired unacked: no longer "sent", so a later broker is given it again.
                pr.sent[uint8_t(it->second.cls)].erase(it->second.key);
                it = pr.inflight.erase(it);
                ++stats_.abandoned_best_effort;
            } else {
                ++it;
            }
        }
    }

    /// Encode now (sequence assigned now), fragment, and register. Untracked
    /// messages leave the table once fully sent (end of Pump).
    uint32_t Start(Peer& pr, OutMessage om, core::Timestamp t) {
        uint64_t seq = pr.next_seq[uint8_t(om.cls)]++;
        if (IsReliableClass(om.cls)) seq |= pr.incarnation << kIncarnationShift;
        SerializedMessage m = om.encode(seq, t);
        const uint32_t id = pr.next_msg_id++;
        Inflight f;
        f.cls = om.cls;
        f.key = om.key;
        f.reliable = IsReliableClass(om.cls);
        if (om.ttl) f.expires = om.created + *om.ttl;
        f.fragments = Fragment(m.bytes, id);
        f.tracked = IsTracked(om.cls, f.fragments.size());
        f.rto_s = p_.rto_min_s;
        f.record = m.record;
        f.record.predicted_wire_bytes = om.predicted_wire_bytes;
        f.record.resync = om.resync;
        ledger_.Record(f.record);
        ++stats_.first_tx_messages;
        pr.sent[uint8_t(om.cls)].insert(om.key);
        pr.inflight.emplace(id, std::move(f));
        return id;
    }

    /// Continue first transmission. False when the send budget is exhausted.
    bool SendFirst(AgentId dest, Peer& pr, Inflight& m, core::Timestamp t, Network& net) {
        while (m.next_first < m.fragments.size()) {
            if (!Spend(pr, m.fragments[m.next_first].size())) return false;
            net.Send(self_, dest, m.fragments[m.next_first], t);
            ++m.next_first;
            m.last_tx = t;
        }
        return true;
    }

    bool Resend(Peer& pr, Inflight& m, core::Timestamp t, Network& net, AgentId dest) {
        core::ByteRecord row = m.record;
        row.t = t;
        row.retransmit = true;
        row.predicted_wire_bytes = -1.0;
        row.payload_bytes = row.transmitted_bytes = row.wire_bytes = 0;
        row.compressed = m.record.compressed;
        bool budget_ok = true;
        for (size_t i = 0; i < m.fragments.size(); ++i) {
            if (BitGet(m.acked, i)) continue;
            if (!Spend(pr, m.fragments[i].size())) {
                budget_ok = false;
                break;
            }
            net.Send(self_, dest, m.fragments[i], t);
            row.transmitted_bytes += m.fragments[i].size() - core::kFragmentHeaderBytes;
            row.wire_bytes += m.fragments[i].size() + core::kDatagramOverheadBytes;
            ++stats_.retransmitted_fragments;
        }
        if (row.wire_bytes > 0) ledger_.Record(row);
        m.last_tx = t;
        m.rto_s = std::min(2.0 * m.rto_s, p_.rto_max_s);
        return budget_ok;
    }

    void AckDone(Peer& pr, uint32_t id, core::Timestamp t) {
        pr.ack_done[id] = p_.ack_repeats;
        if (!pr.ack_due) pr.ack_due = t;
    }

    void ApplyAck(Peer& pr, const AckValue& a) {
        for (uint32_t id : a.done) {
            const auto it = pr.inflight.find(id);
            if (it == pr.inflight.end()) continue;
            pr.delivered[uint8_t(it->second.cls)].insert(it->second.key);
            pr.inflight.erase(it);
            ++stats_.acked;
        }
        for (const auto& p : a.partial) {
            const auto it = pr.inflight.find(p.msg_id);
            if (it == pr.inflight.end()) continue;
            for (size_t i = 0; i < it->second.fragments.size(); ++i)
                if (BitGet(p.received, i)) BitSet(it->second.acked, i);
        }
    }

    static uint64_t ContentKey(const DecodedMessage& m) {
        if (m.has_global_descriptor) return m.global_descriptor.keyframe_id;
        if (m.has_match_request) return m.match_request.request_id;
        if (m.has_feature_response) return m.feature_response.request_id;
        return m.sequence;
    }

    void Deliver(Peer& pr, DecodedMessage m, std::vector<DecodedMessage>* out) {
        const uint8_t c = uint8_t(m.cls);
        if (IsLatestWinsClass(m.cls)) {
            auto it = pr.latest_seq.find(c);
            if (it != pr.latest_seq.end() && m.sequence <= it->second) {
                ++stats_.stale_heartbeats;
                return;
            }
            pr.latest_seq[c] = m.sequence;
            if (m.cls == MessageClass::HEARTBEAT) out->push_back(std::move(m));
            return;
        }
        if (IsReliableClass(m.cls)) {
            uint64_t& next = pr.next_reliable[c];
            auto& buf = pr.reorder[c];
            const uint64_t inc = m.sequence >> kIncarnationShift, cur = next >> kIncarnationShift;
            if (inc < cur) {   // from an abandoned incarnation: the sender re-sends its content
                ++stats_.duplicates;
                return;
            }
            if (inc > cur) {
                next = inc << kIncarnationShift;
                stats_.abandoned_reliable_rx += buf.size();
                buf.clear();
            }
            if (m.sequence < next || buf.count(m.sequence)) {
                ++stats_.duplicates;
                return;
            }
            if (m.sequence > next) {
                if (buf.size() >= p_.reorder_buffer) throw std::runtime_error("Endpoint: reorder buffer overflow");
                buf.emplace(m.sequence, std::move(m));
                return;
            }
            out->push_back(std::move(m));
            ++next;
            for (auto it = buf.find(next); it != buf.end(); it = buf.find(next)) {
                out->push_back(std::move(it->second));
                buf.erase(it);
                ++next;
            }
            return;
        }
        if (!pr.seen_keys[c].insert(ContentKey(m)).second) {
            ++stats_.duplicates;
            return;
        }
        out->push_back(std::move(m));
    }

    AgentId self_;
    const WireCodec& codec_;
    ByteLedger& ledger_;
    EndpointParams p_;
    std::map<AgentId, Peer> peers_;
    EndpointStats stats_;
};

}  // namespace ariadne::collaboration
