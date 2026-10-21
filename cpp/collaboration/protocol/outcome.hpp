#pragma once
// ---------------------------------------------------------------------------
// The I5 exit verdict (design COLLAB_3.2 §6): every run ends `consistent` or
// `explicitly incomplete` (with the exact list of what is missing), never
// silently `corrupt`.
//
// CORRUPT (a failure, never acceptable):
//   - an agent adopted an estimate whose digest differs from what that epoch's
//     broker issued (digest over the wire-encoded float32 values);
//   - an agent adopted an epoch it is not a member of, or one no broker issued;
//   - two agents adopted different estimates under the same full epoch.
// INCOMPLETE (acceptable, reported): unacked reliable messages, edges held for
// a missing snapshot, members of an adopted epoch that adopted something else,
// an edge a member holds that its adopted solve did not include, or an agent
// whose broker's current neighbourhood is larger than the epoch it adopted.
// CONSISTENT: neither.
// ---------------------------------------------------------------------------
#include "collaboration/protocol/collab_node.hpp"

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::collaboration {

enum class Outcome { kConsistent, kIncomplete, kCorrupt };

inline const char* OutcomeName(Outcome o) {
    switch (o) {
        case Outcome::kConsistent: return "consistent";
        case Outcome::kIncomplete: return "explicitly_incomplete";
        case Outcome::kCorrupt: return "corrupt";
    }
    return "unknown";
}

struct OutcomeReport {
    Outcome outcome = Outcome::kConsistent;
    std::vector<std::string> corrupt, incomplete;
};

inline OutcomeReport EvaluateOutcome(const std::vector<const CollabNode*>& nodes, core::Timestamp now) {
    OutcomeReport r;
    std::map<AgentId, const CollabNode*> by_id;
    for (const auto* n : nodes) by_id[n->self()] = n;
    auto say = [](std::vector<std::string>& v, auto&&... parts) {
        std::ostringstream os;
        (os << ... << parts);
        v.push_back(os.str());
    };
    auto epoch_str = [](const core::EpochValue& e) {
        std::ostringstream os;
        os << "(" << e.counter << "," << std::hex << e.members_hash << std::dec << ",b" << e.broker << ")";
        return os.str();
    };

    std::map<std::tuple<uint64_t, uint64_t, AgentId>, uint64_t> digest_by_epoch;
    for (const auto* n : nodes) {
        const auto& a = n->adopted();
        if (!a) continue;
        // Corruption checks.
        const auto broker = by_id.find(a->epoch.broker);
        const EstimateRecord* issued = nullptr;
        if (broker != by_id.end()) {
            const auto it = broker->second->issued().find(a->epoch);
            if (it != broker->second->issued().end()) issued = &it->second;
        }
        if (!issued) {
            say(r.corrupt, "agent ", n->self(), " adopted epoch ", epoch_str(a->epoch), " that no broker issued");
            continue;
        }
        if (issued->digest != a->digest)
            say(r.corrupt, "agent ", n->self(), " adopted a different estimate than broker ", a->epoch.broker,
                " issued for ", epoch_str(a->epoch));
        if (!std::binary_search(issued->members.begin(), issued->members.end(), n->self()))
            say(r.corrupt, "agent ", n->self(), " adopted epoch ", epoch_str(a->epoch), " without being a member");
        const auto key = std::make_tuple(a->epoch.counter, a->epoch.members_hash, a->epoch.broker);
        const auto [it, fresh] = digest_by_epoch.emplace(key, a->digest);
        if (!fresh && it->second != a->digest)
            say(r.corrupt, "two agents adopted different estimates under epoch ", epoch_str(a->epoch));

        // Completeness checks.
        for (AgentId m : issued->members) {
            const auto mn = by_id.find(m);
            if (mn == by_id.end() || !mn->second->adopted() || mn->second->adopted()->epoch != a->epoch)
                say(r.incomplete, "agent ", m, " (member of ", epoch_str(a->epoch), " adopted by agent ", n->self(),
                    ") adopted a different epoch");
        }
        for (const auto& [k, e] : n->edges()) {
            const bool both_members = std::binary_search(issued->members.begin(), issued->members.end(), e.agent_a) &&
                                      std::binary_search(issued->members.begin(), issued->members.end(), e.agent_b);
            if (both_members && !issued->edges.count(k))
                say(r.incomplete, "agent ", n->self(), " holds edge (", e.agent_a, ":", e.keyframe_a, " - ", e.agent_b,
                    ":", e.keyframe_b, ") that its adopted solve ", epoch_str(a->epoch), " did not include");
        }
        if (broker->second->membership().IsBroker() && broker->second->membership().Members() != issued->members)
            say(r.incomplete, "agent ", n->self(), ": broker ", a->epoch.broker,
                "'s current members differ from the adopted epoch ", epoch_str(a->epoch));
    }
    for (const auto* n : nodes) {
        if (!n->adopted() && !n->membership().Neighbors().empty())
            say(r.incomplete, "agent ", n->self(), " has neighbours but adopted no estimate");
        for (AgentId dest : n->endpoint().Peers())
            for (const auto& [cls, key, age] : n->endpoint().UnackedReliableDetail(dest, now))
                say(r.incomplete, "agent ", n->self(), ": reliable class ", int(cls), " key ", key, " to ", dest,
                    " unacked (", age < 0 ? "queued" : "last sent " + std::to_string(age) + " s ago", ")");
        if (n->HeldEdges()) say(r.incomplete, "broker ", n->self(), ": ", n->HeldEdges(), " edges held for snapshots");
    }
    r.outcome = !r.corrupt.empty() ? Outcome::kCorrupt : !r.incomplete.empty() ? Outcome::kIncomplete : Outcome::kConsistent;
    return r;
}

}  // namespace ariadne::collaboration
