#pragma once
// ---------------------------------------------------------------------------
// D115: a CBBA message's bytes at the serializer (invariant 6). The CbbaMessage snapshot is encoded as the wire's
// TaskConsensus table (class TASK_CONSENSUS, through WireCodec::FinishEnvelope, so a ByteLedger counts it exactly
// as it counts every other class). Kept out of allocation.cpp so the auction core stays free of FlatBuffers.
//
// Sparse (D115a): only tasks with a winner (z != none) are sent. Every CBBA event starts from y = 0, z = none, so the
// omitted entries are exactly the sender's state, and the receiver's expansion (missing -> y 0, z none) feeds Table I
// the identical inputs: the encoding cannot change the assignment (round trip tested). Replay measured 2.4 KB per
// message with every task (~160) sent.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"
#include "planning/mission/allocation.hpp"

#include <algorithm>
#include <vector>

namespace ariadne::planning {

inline core::CbbaConsensusValue ToWire(const CbbaMessage& m, const std::vector<float>& scales) {
    core::CbbaConsensusValue v;
    for (size_t j = 0; j < m.task_ids.size(); ++j) {
        if (m.z[j] < 0) continue;   // sparse: (0, none) is implied for every task not listed
        v.tasks.push_back(uint64_t(m.task_ids[j]));
        v.y.push_back(m.y[j]);
        v.z.push_back(int8_t(m.z[j]));
    }
    v.s = m.s;
    v.scales = scales;
    return v;
}

/// The receiver's expansion of a decoded sparse message onto the event's task list (ids not listed: y 0, z none).
inline CbbaMessage FromWire(const core::CbbaConsensusValue& v, int sender, const std::vector<int64_t>& task_ids) {
    CbbaMessage m;
    m.sender = sender, m.task_ids = task_ids;
    m.y.assign(task_ids.size(), 0.0), m.z.assign(task_ids.size(), -1);
    for (size_t e = 0; e < v.tasks.size(); ++e) {
        const auto it = std::lower_bound(task_ids.begin(), task_ids.end(), int64_t(v.tasks[e]));
        if (it == task_ids.end() || *it != int64_t(v.tasks[e])) continue;   // a task this receiver does not know
        const size_t j = size_t(it - task_ids.begin());
        m.y[j] = v.y[e], m.z[j] = int(v.z[e]);
    }
    m.s = v.s;
    return m;
}

/// RunCbba's transport through the wire: encode (sparse), decode, expand. CBBA then runs on exactly what was sent.
inline std::function<CbbaMessage(const CbbaMessage&)> WireTransport(const core::WireCodec& codec, std::vector<float> scales) {
    return [&codec, scales](const CbbaMessage& m) {
        const auto enc = codec.EncodeCbbaConsensus(ToWire(m, scales), core::AgentId(m.sender), 0, core::Timestamp{0}, 0);
        core::DecodedMessage d;
        if (codec.Decode(enc.bytes, &d) != core::WireCodec::Status::kOk || !d.has_cbba_consensus) return CbbaMessage{};
        return FromWire(d.cbba_consensus, m.sender, m.task_ids);
    };
}

/// The function object RunCbba takes: wire bytes (payload + framing) of the message, as the ledger would record.
inline std::function<size_t(const CbbaMessage&)> WireBytes(const core::WireCodec& codec, std::vector<float> scales) {
    return [&codec, scales](const CbbaMessage& m) {
        return size_t(codec.EncodeCbbaConsensus(ToWire(m, scales), core::AgentId(m.sender), 0, core::Timestamp{0}, 0)
                          .record.wire_bytes);
    };
}

}  // namespace ariadne::planning
