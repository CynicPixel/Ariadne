#pragma once
// ---------------------------------------------------------------------------
// Membership, broker election and epochs (design collaboration.md §4).
//
// WHAT IT IS. A per-agent state machine fed by heartbeats. It answers three
// questions on every tick:
//   - who are my symmetric neighbours (I hear them AND they hear me)?
//   - who is my broker?
//   - if I am a broker, which epoch would my next solve carry, and should I
//     accept an estimate stamped with some epoch?
//
// HEARD vs NEIGHBOUR. A heartbeat lists the agents its sender HEARS (with
// hysteresis). `j` is a symmetric neighbour of `i` when i hears j and j's
// latest heartbeat lists i. Listing only symmetric neighbours would deadlock
// (each waits for the other to list it first); requiring the reverse listing is
// what stops an asymmetric link from making two agents elect different brokers
// for the same agent.
//
// HYSTERESIS. Heard: in after `m_in` of the last `window` heartbeat periods,
// out after `k_out` periods with nothing heard (likewise for "j lists me").
// Without it, 30% heartbeat loss flaps links and churns brokers and epochs;
// k_out is sized from the loss rate (MembershipParams).
//
// ELECTION (Swarm-SLAM's `is_optimizer()`, verified in lajoiepy/cslam): the
// lexicographic minimum of (anchor_id, agent_id) over me and my symmetric
// neighbours. The anchor (the agent whose frame defines G) wins first, so an
// already-merged group keeps its frame when it meets a newcomer.
//
// EPOCHS. A broker solve is stamped (counter, members_hash, broker) with
// members = me + the symmetric neighbours that elected me, and counter =
// 1 + max(counters I have seen from members, including mine). After two
// partitions each reach 5, the merged solve is 6: newer than both sides.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"
#include "core/time/timestamp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace ariadne::collaboration {

using core::AgentId;
using core::EpochValue;
using core::HeartbeatValue;

/// Sized for heartbeat loss up to 30%: a link is falsely dropped when k_out
/// consecutive heartbeats are lost, p^k_out per period. 0.3^12 = 5e-7, i.e.
/// ~4e-3 expected false drops over 300 s x 12 directed links; k_out = 4 at 1 Hz
/// gave ~1 per link per run (measured: 40 broker changes). The price is a 6 s
/// partition-detection time.
struct MembershipParams {
    double period_s = 0.5;   // heartbeat period
    int window = 3;          // periods looked at for "in"
    int m_in = 2;            // heard in >= m_in of the last `window` periods
    int k_out = 12;          // out after k_out periods with nothing heard (6 s)
};

/// FNV-1a over the sorted member ids: the epoch's member-set identity.
inline uint64_t MembersHash(const std::vector<AgentId>& sorted_members) {
    uint64_t h = 1469598103934665603ull;
    for (AgentId a : sorted_members)
        for (int b = 0; b < 4; ++b) {
            h ^= (a >> (8 * b)) & 0xFF;
            h *= 1099511628211ull;
        }
    return h;
}

class Membership {
public:
    Membership(AgentId self, MembershipParams p = {}) : self_(self), anchor_(self), p_(p), broker_(self) {}

    AgentId self() const { return self_; }

    /// A delivered heartbeat (latest-wins already applied by the transport).
    void OnHeartbeat(const HeartbeatValue& h, core::Timestamp t) {
        if (h.agent == self_) return;
        Peer& pr = peers_[h.agent];
        const int64_t period = Period(t);
        pr.heard_periods.push_back(period);
        pr.last_heard = period;
        pr.last = h;
        if (std::find(h.neighbors.begin(), h.neighbors.end(), self_) != h.neighbors.end()) pr.last_listed_me = period;
        max_seen_counter_ = std::max(max_seen_counter_, h.epoch.counter);
    }

    /// Re-evaluate heard/neighbour state and the election at time t. Returns
    /// true when my broker changed.
    bool Update(core::Timestamp t) {
        const int64_t now = Period(t);
        for (auto& [id, pr] : peers_) {
            // Keep only the periods inside the window.
            pr.heard_periods.erase(std::remove_if(pr.heard_periods.begin(), pr.heard_periods.end(),
                                                  [&](int64_t q) { return q <= now - p_.window; }),
                                   pr.heard_periods.end());
            std::vector<int64_t> distinct = pr.heard_periods;
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            if (!pr.heard && int(distinct.size()) >= p_.m_in) pr.heard = true;
            if (pr.heard && pr.last_heard && now - *pr.last_heard >= p_.k_out) pr.heard = false;
            const bool lists_me = pr.last_listed_me && now - *pr.last_listed_me < p_.k_out;
            pr.neighbor = pr.heard && lists_me;
        }
        const AgentId before = broker_;
        broker_ = Elect();
        return broker_ != before;
    }

    /// Agents I hear (what my heartbeat lists).
    std::vector<AgentId> Heard() const {
        std::vector<AgentId> out;
        for (const auto& [id, pr] : peers_)
            if (pr.heard) out.push_back(id);
        return out;
    }
    std::vector<AgentId> Neighbors() const {
        std::vector<AgentId> out;
        for (const auto& [id, pr] : peers_)
            if (pr.neighbor) out.push_back(id);
        return out;
    }
    bool IsNeighbor(AgentId a) const {
        const auto it = peers_.find(a);
        return it != peers_.end() && it->second.neighbor;
    }

    AgentId broker() const { return broker_; }
    bool IsBroker() const { return broker_ == self_; }
    AgentId anchor() const { return anchor_; }
    void SetAnchor(AgentId a) { anchor_ = a; }

    /// Members a solve I run now would cover: me + symmetric neighbours that
    /// elected me (sorted).
    std::vector<AgentId> Members() const {
        std::vector<AgentId> m{self_};
        for (const auto& [id, pr] : peers_)
            if (pr.neighbor && pr.last.broker == self_) m.push_back(id);
        std::sort(m.begin(), m.end());
        return m;
    }

    /// D175: my connected component as seen from here, at any node: me + symmetric neighbours that elected the same
    /// broker as I did (sorted). At the broker this equals Members().
    std::vector<AgentId> ComponentMembers() const {
        std::vector<AgentId> m{self_};
        for (const auto& [id, pr] : peers_)
            if (pr.neighbor && (pr.last.broker == broker_ || id == broker_)) m.push_back(id);
        std::sort(m.begin(), m.end());
        m.erase(std::unique(m.begin(), m.end()), m.end());
        return m;
    }

    /// The epoch for a solve I run now over `members` (call only as broker).
    EpochValue NextEpoch(const std::vector<AgentId>& members) const {
        uint64_t c = accepted_.counter;
        for (AgentId a : members) {
            const auto it = peers_.find(a);
            if (it != peers_.end()) c = std::max(c, it->second.last.epoch.counter);
        }
        c = std::max(c, issued_counter_);
        return {c + 1, MembersHash(members), self_};
    }
    void Issued(const EpochValue& e) { issued_counter_ = std::max(issued_counter_, e.counter); }

    enum class Verdict { kAccept, kNotMember, kNotMyBroker, kNotNewer };

    /// The acceptance rule of §4: member, from my broker, strictly newer.
    Verdict Check(const EpochValue& e, const std::vector<AgentId>& members, AgentId sender) const {
        if (!std::binary_search(members.begin(), members.end(), self_) ||
            MembersHash(members) != e.members_hash)
            return Verdict::kNotMember;
        if (sender != broker_ || e.broker != broker_) return Verdict::kNotMyBroker;
        if (e.counter <= accepted_.counter) return Verdict::kNotNewer;
        return Verdict::kAccept;
    }
    void Accept(const EpochValue& e) {
        accepted_ = e;
        max_seen_counter_ = std::max(max_seen_counter_, e.counter);
    }
    const EpochValue& accepted() const { return accepted_; }

    /// The anchor id a peer advertised (self's own for self).
    AgentId AnchorOf(AgentId a) const {
        if (a == self_) return anchor_;
        const auto it = peers_.find(a);
        return it == peers_.end() ? a : it->second.last.anchor;
    }
    std::optional<HeartbeatValue> LastHeartbeat(AgentId a) const {
        const auto it = peers_.find(a);
        if (it == peers_.end()) return std::nullopt;
        return it->second.last;
    }

private:
    struct Peer {
        std::vector<int64_t> heard_periods;
        std::optional<int64_t> last_heard, last_listed_me;
        bool heard = false, neighbor = false;
        HeartbeatValue last;
    };

    int64_t Period(core::Timestamp t) const { return int64_t(std::floor(t.toSeconds() / p_.period_s)); }

    AgentId Elect() const {
        std::tuple<AgentId, AgentId> best{anchor_, self_};
        for (const auto& [id, pr] : peers_)
            if (pr.neighbor) best = std::min(best, std::make_tuple(pr.last.anchor, id));
        return std::get<1>(best);
    }

    AgentId self_;
    AgentId anchor_;
    MembershipParams p_;
    std::map<AgentId, Peer> peers_;
    AgentId broker_ = 0;
    EpochValue accepted_;
    uint64_t issued_counter_ = 0;
    uint64_t max_seen_counter_ = 0;
};

}  // namespace ariadne::collaboration
