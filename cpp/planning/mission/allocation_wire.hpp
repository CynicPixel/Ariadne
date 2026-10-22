#pragma once
// ---------------------------------------------------------------------------
// D115: a CBBA message's bytes at the serializer (invariant 6). The CbbaMessage snapshot is encoded as the wire's
// TaskConsensus table (class TASK_CONSENSUS, through WireCodec::FinishEnvelope, so a ByteLedger counts it exactly
// as it counts every other class). Kept out of allocation.cpp so the auction core stays free of FlatBuffers.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"
#include "planning/mission/allocation.hpp"

#include <vector>

namespace ariadne::planning {

inline core::TaskConsensusValue ToWire(const CbbaMessage& m, const std::vector<float>& scales) {
    core::TaskConsensusValue v;
    for (int64_t t : m.task_ids) v.tasks.push_back(uint64_t(t));
    for (double y : m.y) v.y.push_back(float(y));
    for (int z : m.z) v.z.push_back(int8_t(z));
    v.s = m.s;
    v.scales = scales;
    return v;
}

/// The function object RunCbba takes: wire bytes (payload + framing) of the message, as the ledger would record.
inline std::function<size_t(const CbbaMessage&)> WireBytes(const core::WireCodec& codec, std::vector<float> scales) {
    return [&codec, scales](const CbbaMessage& m) {
        return size_t(codec.EncodeTaskConsensus(ToWire(m, scales), core::AgentId(m.sender), 0, core::Timestamp{0}, 0)
                          .record.wire_bytes);
    };
}

}  // namespace ariadne::planning
