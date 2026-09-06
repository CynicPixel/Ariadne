#pragma once
// ---------------------------------------------------------------------------
// Event schema and log (arch §41).
//
// Every event carries timestamp, agent, run_id, sequence_id and state_hash.
//
// WHAT state_hash BUYS. It is a running hash chaining each event into the last,
// so two runs of the same configuration produce identical hash sequences. When
// they do not, comparing the streams gives the exact sequence_id where they
// first diverged, rather than only the fact that they did. Without it, a
// determinism failure in a long multi-agent mission is a needle hunt; with it,
// it is a bisection.
//
// Format is JSONL: one self-describing object per line, appendable, greppable,
// and readable by the Python analysis environment without a schema compiler.
// The wire format uses FlatBuffers because bytes are counted there; logs are
// not on the wire and favour inspectability instead.
// ---------------------------------------------------------------------------
#include "core/time/timestamp.hpp"

#include <cstdint>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::runtime {

using core::Timestamp;

/// The 20 event types of §41, in the order the architecture lists them.
/// Values are pinned: a recorded log is only readable if these are stable.
enum class EventType : uint8_t {
    FRAME = 0,
    KEYFRAME = 1,
    TRACK_LOSS = 2,
    LOOP_CANDIDATE = 3,
    LOOP_REJECT = 4,
    LOOP_ACCEPT = 5,
    GNC_UPDATE = 6,
    PGO = 7,
    MESSAGE_SENT = 8,
    MESSAGE_DROPPED = 9,
    AGENT_DISCONNECT = 10,
    AGENT_RECONNECT = 11,
    TASK_CREATED = 12,
    TASK_BID = 13,
    TASK_ASSIGNED = 14,
    ROUTE_CREATED = 15,
    ROUTE_INVALIDATED = 16,
    HUMAN_REQUEST = 17,
    FAILURE = 18,
    RECOVERY = 19,
    kCount = 20
};

inline const char* EventTypeName(EventType type) {
    switch (type) {
        case EventType::FRAME: return "FRAME";
        case EventType::KEYFRAME: return "KEYFRAME";
        case EventType::TRACK_LOSS: return "TRACK_LOSS";
        case EventType::LOOP_CANDIDATE: return "LOOP_CANDIDATE";
        case EventType::LOOP_REJECT: return "LOOP_REJECT";
        case EventType::LOOP_ACCEPT: return "LOOP_ACCEPT";
        case EventType::GNC_UPDATE: return "GNC_UPDATE";
        case EventType::PGO: return "PGO";
        case EventType::MESSAGE_SENT: return "MESSAGE_SENT";
        case EventType::MESSAGE_DROPPED: return "MESSAGE_DROPPED";
        case EventType::AGENT_DISCONNECT: return "AGENT_DISCONNECT";
        case EventType::AGENT_RECONNECT: return "AGENT_RECONNECT";
        case EventType::TASK_CREATED: return "TASK_CREATED";
        case EventType::TASK_BID: return "TASK_BID";
        case EventType::TASK_ASSIGNED: return "TASK_ASSIGNED";
        case EventType::ROUTE_CREATED: return "ROUTE_CREATED";
        case EventType::ROUTE_INVALIDATED: return "ROUTE_INVALIDATED";
        case EventType::HUMAN_REQUEST: return "HUMAN_REQUEST";
        case EventType::FAILURE: return "FAILURE";
        case EventType::RECOVERY: return "RECOVERY";
        case EventType::kCount: return "INVALID";
    }
    return "INVALID";
}

/// One key/value pair of event-specific detail. Kept as flat strings so the
/// schema never needs a migration to carry a new field, and so a log written by
/// an older build stays readable.
struct EventField {
    std::string key;
    std::string value;
};

struct Event {
    EventType type = EventType::FRAME;
    Timestamp t;
    uint32_t agent = 0;
    uint64_t sequence_id = 0;
    uint64_t state_hash = 0;
    std::vector<EventField> fields;
};

/// Escapes the characters JSON forbids in a string. Small and explicit rather
/// than pulling a JSON library into the runtime for one function.
inline std::string EscapeJson(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

/// Appends events, maintaining the sequence counter and the state-hash chain.
///
/// The log is the only writer of sequence_id and state_hash: a caller supplying
/// its own would be able to produce a stream that looks reproducible without
/// being so.
class EventLog {
public:
    EventLog(std::string run_id, std::ostream* sink)
        : run_id_(std::move(run_id)), sink_(sink) {}

    /// Record an event. Returns the completed record, including the fields the
    /// log itself owns, so a caller can assert on them in tests.
    Event Append(EventType type, Timestamp t, uint32_t agent,
                 std::vector<EventField> fields = {}) {
        Event event;
        event.type = type;
        event.t = t;
        event.agent = agent;
        event.sequence_id = next_sequence_++;
        event.fields = std::move(fields);
        event.state_hash = ChainHash(event);
        state_hash_ = event.state_hash;
        if (sink_ != nullptr) (*sink_) << ToJson(event) << "\n";
        return event;
    }

    /// The running hash after the most recent event. Two runs of the same
    /// configuration must end with the same value.
    uint64_t StateHash() const { return state_hash_; }
    uint64_t EventCount() const { return next_sequence_; }
    const std::string& RunId() const { return run_id_; }

    std::string ToJson(const Event& event) const {
        std::ostringstream out;
        out << "{\"run_id\":\"" << EscapeJson(run_id_) << "\""
            << ",\"seq\":" << event.sequence_id
            << ",\"t_ns\":" << event.t.ns
            << ",\"agent\":" << event.agent
            << ",\"type\":\"" << EventTypeName(event.type) << "\""
            << ",\"state_hash\":\"" << HexOf(event.state_hash) << "\"";
        for (const auto& f : event.fields) {
            out << ",\"" << EscapeJson(f.key) << "\":\""
                << EscapeJson(f.value) << "\"";
        }
        out << "}";
        return out.str();
    }

    static std::string HexOf(uint64_t value) {
        char buf[19];
        std::snprintf(buf, sizeof(buf), "0x%016llx",
                      static_cast<unsigned long long>(value));
        return std::string(buf);
    }

private:
    /// FNV-1a over the previous hash and this event's identifying content. The
    /// chain means a divergence at event N changes every hash from N onward,
    /// so the FIRST mismatch localizes the divergence exactly.
    uint64_t ChainHash(const Event& event) const {
        uint64_t hash = state_hash_ != 0 ? state_hash_ : 1469598103934665603ull;
        auto absorb_u64 = [&hash](uint64_t value) {
            for (int i = 0; i < 8; ++i) {
                hash ^= (value >> (i * 8)) & 0xFFull;
                hash *= 1099511628211ull;
            }
        };
        auto absorb_str = [&hash](const std::string& s) {
            for (char c : s) {
                hash ^= static_cast<unsigned char>(c);
                hash *= 1099511628211ull;
            }
        };
        absorb_u64(static_cast<uint64_t>(event.type));
        absorb_u64(static_cast<uint64_t>(event.t.ns));
        absorb_u64(event.agent);
        absorb_u64(event.sequence_id);
        for (const auto& f : event.fields) {
            absorb_str(f.key);
            absorb_str(f.value);
        }
        return hash;
    }

    std::string run_id_;
    std::ostream* sink_ = nullptr;
    uint64_t next_sequence_ = 0;
    uint64_t state_hash_ = 0;
};

}  // namespace ariadne::runtime
