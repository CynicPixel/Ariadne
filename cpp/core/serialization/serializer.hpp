#pragma once
// ---------------------------------------------------------------------------
// The single serialization boundary (arch §4.4, §15.7).
//
// Every message crosses exactly one serializer, and bytes are counted HERE, at
// serialization, never as "logical payload size". That is not bookkeeping
// pedantry: the spectral prioritization of §16 optimizes gain/predicted_cost,
// and Phase 3 puts a hard <=15% error gate on predicted-vs-measured wire bytes
// (§15.7). A byte count taken anywhere else would be optimizing against a
// number the link does not actually carry.
//
// Wire accounting:
//   payload_bytes = serialized FlatBuffer size
//   framed_bytes  = kFrameHeaderBytes
//                   + transmitted payload (compressed where applicable)
//   wire_bytes    = framed_bytes
//                   + kDatagramOverheadBytes * DatagramCount(framed_bytes)
//
// Framing is charged PER DATAGRAM, matching arch §15.7's wording ("28 B
// emulated UDP/IP header per datagram"). Anything over the 1472 B datagram
// payload fragments and pays the header again.
//
// Compression covers MATCH_RESPONSE and POSE_EDGE (§15.7's classes 4-5) plus
// POSE_GRAPH, which §15.7 leaves uncompressed. See DECISIONS.md D14: pose
// graphs measure 4.93x under zstd-3 because keyframe ids, timestamps and poses
// are near-sequential, and compressing them takes a 200-node/40-edge batch from
// 21.0 KB to ~4.3 KB against §15.7's 12 KB prediction.
//
// Descriptors are high-entropy, so the achieved ratio is MEASURED and reported
// rather than assumed; CompressionRatio() exists to make that measurement, and
// a ratio near 1.0 is a legitimate result to report rather than a bug to hide.
// A compressed form is kept only when it is actually smaller.
// ---------------------------------------------------------------------------
#include "core/time/timestamp.hpp"

#include <flatbuffers/flatbuffers.h>
#include <zstd.h>
#include <wire_generated.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ariadne::core {

using AgentId = uint32_t;
using MessageClass = wire::MessageClass;

/// Emulated UDP/IP framing charged to every datagram: 20 B IPv4 header + 8 B
/// UDP header. Included so the metric is honest about framing rather than
/// flattering, exactly as §15.7 requires. Small messages are dominated by it,
/// which is itself a result worth showing.
inline constexpr uint64_t kDatagramOverheadBytes = 28;

/// Path MTU assumed for the emulated link (standard Ethernet). A message
/// larger than one datagram's payload is fragmented and each fragment carries
/// its own 20+8 B header, so framing is charged PER DATAGRAM, not per message.
///
/// This matters because §15.7's byte table feeds a hard <=15% prediction-error
/// gate in Phase 3, and the classes that exceed the MTU are exactly the large
/// ones the spectral policy has to price: an uncompressed 21 KB pose graph is
/// ~15 datagrams (420 B of framing), not one (28 B). Charging once per message
/// would understate framing precisely where the table is load-bearing.
inline constexpr uint64_t kPathMtuBytes = 1500;
inline constexpr uint64_t kDatagramPayloadBytes =
    kPathMtuBytes - kDatagramOverheadBytes;   // 1472

/// Number of datagrams needed to carry `on_wire` bytes of framed payload.
/// A zero-length message still costs one datagram.
inline constexpr uint64_t DatagramCount(uint64_t on_wire) {
    return on_wire == 0 ? 1
                        : (on_wire + kDatagramPayloadBytes - 1) /
                              kDatagramPayloadBytes;
}

/// COLLAB_3.2 §3.2: the 3.2 transport splits a framed message into datagrams,
/// each carrying a 6 B fragment header (msg_id u32, index u8, count u8) inside
/// the 1472 B datagram payload. Selective retransmission needs it; the Phase-1
/// exchange sends whole messages and is charged without it (`Framing`).
inline constexpr uint64_t kFragmentHeaderBytes = 6;
inline constexpr uint64_t kFragmentPayloadBytes = kDatagramPayloadBytes - kFragmentHeaderBytes;   // 1466
inline constexpr uint64_t kMaxFragments = 255;

inline constexpr uint64_t FragmentCount(uint64_t on_wire) {
    return on_wire == 0 ? 1 : (on_wire + kFragmentPayloadBytes - 1) / kFragmentPayloadBytes;
}

/// How the codec charges datagram framing: Phase 1's whole messages, or the
/// 3.2 transport's fragments (6 B more per datagram, 6 B fewer payload bytes).
enum class Framing { kWholeMessage, kFragmented };

/// Ariadne frame header, carried UNCOMPRESSED ahead of every payload:
///   [0]    message class
///   [1]    flags, bit 0 = zstd compressed
///   [2..6) payload length, uint32 little-endian
///
/// This exists so a receiver can read the class WITHOUT decompressing or even
/// parsing the payload. §15.3's store-and-forward triages queued messages by
/// priority class and TTL, and must be able to drop or forward a message it
/// never decompresses. Carrying the class only inside the envelope would force
/// every queue operation to pay a decompression, and would make the class
/// unknowable out-of-band.
inline constexpr uint64_t kFrameHeaderBytes = 6;
inline constexpr uint8_t kFlagCompressed = 0x01;

/// zstd level from §15.7.
inline constexpr int kCompressionLevel = 3;

/// Which classes are compressed. Stated once, here, rather than as a scattered
/// condition, so the policy is auditable and appears in one place in the byte
/// table. POSE_GRAPH extends §15.7 deliberately; see DECISIONS.md D14.
inline bool IsCompressedClass(MessageClass cls) {
    return cls == MessageClass::MATCH_RESPONSE ||
           cls == MessageClass::POSE_EDGE ||
           cls == MessageClass::POSE_GRAPH;
}

/// §4.4. One row per message actually sent.
struct ByteRecord {
    Timestamp t;
    AgentId sender = 0;
    AgentId receiver = 0;
    MessageClass type = MessageClass::HEARTBEAT;

    uint64_t payload_bytes = 0;      // serialized, before compression
    uint64_t transmitted_bytes = 0;  // after compression, before framing
    uint64_t wire_bytes = 0;         // transmitted + framing
    uint64_t cumulative_bytes = 0;   // running total on this ledger
    bool compressed = false;
    bool retransmit = false;         // 3.2 ARQ: resent fragments; not "first transmission" (COLLAB_3.2 §7)

    /// Achieved compression ratio. 1.0 means compression did not help, which
    /// for high-entropy descriptors is the expected outcome and is reported.
    double CompressionRatio() const {
        if (transmitted_bytes == 0) return 1.0;
        return static_cast<double>(payload_bytes) /
               static_cast<double>(transmitted_bytes);
    }
};

/// A serialized message plus its accounting row. The bytes and the record are
/// produced together so a caller cannot transmit one without recording the other.
struct SerializedMessage {
    std::vector<uint8_t> bytes;   // frame header + payload: what goes on the wire
    ByteRecord record;
};

/// Read the message class and compression flag from a frame header without
/// touching the payload. This is the operation §15.3's queue needs.
inline bool PeekFrameHeader(const std::vector<uint8_t>& wire_bytes,
                            MessageClass* cls, bool* compressed,
                            uint32_t* payload_bytes) {
    if (wire_bytes.size() < kFrameHeaderBytes) return false;
    const uint8_t raw_class = wire_bytes[0];
    if (raw_class > static_cast<uint8_t>(MessageClass::ACK)) return false;
    *cls = static_cast<MessageClass>(raw_class);
    *compressed = (wire_bytes[1] & kFlagCompressed) != 0;
    *payload_bytes = static_cast<uint32_t>(wire_bytes[2]) |
                     (static_cast<uint32_t>(wire_bytes[3]) << 8) |
                     (static_cast<uint32_t>(wire_bytes[4]) << 16) |
                     (static_cast<uint32_t>(wire_bytes[5]) << 24);
    return true;
}

/// Accumulates byte totals, overall and per class. This is what the §15.7 byte
/// table and the Phase 3 gate are computed from.
class ByteLedger {
public:
    void Record(ByteRecord& record) {
        cumulative_ += record.wire_bytes;
        record.cumulative_bytes = cumulative_;
        per_class_[record.type] += record.wire_bytes;
        per_class_count_[record.type] += 1;
        records_.push_back(record);
    }

    uint64_t Total() const { return cumulative_; }
    uint64_t TotalForClass(MessageClass cls) const {
        auto it = per_class_.find(cls);
        return it == per_class_.end() ? 0 : it->second;
    }
    uint64_t CountForClass(MessageClass cls) const {
        auto it = per_class_count_.find(cls);
        return it == per_class_count_.end() ? 0 : it->second;
    }
    const std::vector<ByteRecord>& Records() const { return records_; }
    void Clear() {
        cumulative_ = 0;
        per_class_.clear();
        per_class_count_.clear();
        records_.clear();
    }

private:
    uint64_t cumulative_ = 0;
    std::map<MessageClass, uint64_t> per_class_;
    std::map<MessageClass, uint64_t> per_class_count_;
    std::vector<ByteRecord> records_;
};

// --- Plain C++ mirrors of the wire types ---------------------------------
// Callers build these; the codec owns every FlatBuffers detail. Keeping
// generated types out of call sites is what allows the schema to evolve
// without a rewrite, and keeps the "one boundary" property enforceable.

struct FrameTag {
    uint32_t agent = 0;
    uint32_t epoch = 0;
};

struct PoseValue {
    // Quaternion (w, x, y, z) then translation. Stored as float on the wire per
    // the §15.7 size table; the estimator uses double internally.
    float qw = 1.0f, qx = 0.0f, qy = 0.0f, qz = 0.0f;
    float tx = 0.0f, ty = 0.0f, tz = 0.0f;
};

struct PoseNodeValue {
    AgentId agent = 0;
    uint64_t keyframe = 0;
    int64_t stamp_ns = 0;
    PoseValue pose;
    FrameTag frame;
};

struct LoopEdgeValue {
    AgentId agent_a = 0;
    uint64_t keyframe_a = 0;
    AgentId agent_b = 0;
    uint64_t keyframe_b = 0;
    PoseValue measurement;
    /// Upper triangle of the 6x6 information matrix, 21 entries.
    std::vector<float> information = std::vector<float>(21, 0.0f);
    FrameTag frame_a;
    FrameTag frame_b;
    wire::EdgeStatus status = wire::EdgeStatus::CANDIDATE;
    uint32_t inlier_count = 0;
    float retrieval_score = 0.0f;
    float pcm_score = 0.0f;
    float gnc_weight = 1.0f;
    bool degeneracy_flag = false;
};

/// COLLAB_3.2 §4: a broker solve's identity; the full triple identifies it.
struct EpochValue {
    uint64_t counter = 0;
    uint64_t members_hash = 0;
    AgentId broker = 0;
    bool operator==(const EpochValue& o) const {
        return counter == o.counter && members_hash == o.members_hash && broker == o.broker;
    }
    bool operator!=(const EpochValue& o) const { return !(*this == o); }
};

/// T_G_W of one agent in a broker estimate (G = the estimate's anchor frame).
struct AgentFrameValue {
    AgentId agent = 0;
    PoseValue T_G_W;
};

struct PoseGraphValue {
    std::vector<PoseNodeValue> nodes;
    std::vector<LoopEdgeValue> edges;
    FrameTag anchor;
    // 3.2: a broker estimate carries its epoch and every member's T_G_W.
    bool has_epoch = false;
    EpochValue epoch;
    std::vector<AgentFrameValue> agent_frames;
};

/// COLLAB_3.2 §3.2: transport acks by message id.
struct PartialAckValue {
    uint32_t msg_id = 0;
    std::vector<uint8_t> received;   // bitmap, bit i = fragment i
};
struct AckValue {
    std::vector<uint32_t> done;
    std::vector<PartialAckValue> partial;
    bool empty() const { return done.empty() && partial.empty(); }
};

/// §15.2 heartbeat plus the 3.2 membership fields (COLLAB_3.2 §4).
struct HeartbeatValue {
    AgentId agent = 0;
    int64_t stamp_ns = 0;
    PoseValue pose;
    FrameTag frame;
    float battery = 1.0f;
    float link_state = 0.0f;
    uint64_t keyframe_watermark = 0;
    std::vector<AgentId> neighbors;
    AgentId anchor = 0;
    AgentId broker = 0;
    EpochValue epoch;
    AckValue ack;
};

struct GlobalDescriptorValue {
    uint64_t keyframe_id = 0;
    int64_t stamp_ns = 0;
    std::vector<float> descriptor;
    float quality = 0.0f;
    uint32_t n_features = 0;     // 3.4: the feature response's size, advertised for pricing; 0 = not advertised
    uint16_t feature_bytes = 0;
};

struct MatchRequestValue {
    uint64_t request_id = 0, query_keyframe = 0, candidate_keyframe = 0;
    uint8_t feature_type = 0;
};

struct FeatureResponseValue {
    uint64_t request_id = 0, keyframe_id = 0;
    std::vector<float> keypoints;     // x0 y0 x1 y1 ...
    std::vector<uint8_t> descriptors; // 32 bytes per keypoint
    std::vector<float> points3d;      // x y z per keypoint, NaN = no depth
    std::vector<float> camera;        // fx fy cx cy baseline_m
};

struct DecodedMessage {
    MessageClass cls = MessageClass::HEARTBEAT;
    AgentId sender = 0;
    AgentId receiver = 0;
    int64_t stamp_ns = 0;
    uint64_t sequence = 0;
    bool has_pose_graph = false;
    PoseGraphValue pose_graph;
    bool has_edge = false;
    LoopEdgeValue edge;
    bool has_global_descriptor = false;
    GlobalDescriptorValue global_descriptor;
    bool has_match_request = false;
    MatchRequestValue match_request;
    bool has_feature_response = false;
    FeatureResponseValue feature_response;
    bool has_heartbeat = false;
    HeartbeatValue heartbeat;
    bool has_ack = false;   // a standalone ACK; a heartbeat's ack is in `heartbeat.ack`
    AckValue ack;
};

namespace detail {

inline wire::Pose ToWire(const PoseValue& p) {
    return wire::Pose(wire::Quaternion(p.qw, p.qx, p.qy, p.qz),
                      wire::Translation(p.tx, p.ty, p.tz));
}

inline PoseValue FromWire(const wire::Pose& p) {
    PoseValue v;
    v.qw = p.q().w(); v.qx = p.q().x(); v.qy = p.q().y(); v.qz = p.q().z();
    v.tx = p.t().x(); v.ty = p.t().y(); v.tz = p.t().z();
    return v;
}

inline wire::FrameTag ToWire(const FrameTag& f) {
    return wire::FrameTag(f.agent, f.epoch);
}

inline FrameTag FromWire(const wire::FrameTag& f) {
    FrameTag v;
    v.agent = f.agent();
    v.epoch = f.epoch();
    return v;
}

}  // namespace detail

/// The codec. Every encode returns bytes AND the accounting row.
class WireCodec {
public:
    explicit WireCodec(Framing framing = Framing::kWholeMessage) : framing_(framing) {}
    Framing framing() const { return framing_; }

    /// Reasons a decode can fail. Returned rather than thrown so a malformed
    /// or hostile message degrades one link instead of the process.
    enum class Status {
        kOk,
        kMalformed,        // failed FlatBuffers verification
        kUntaggedPose,     // §15.5: a pose arrived without a frame tag
        kBadInformation,   // information matrix was not 21 entries
        kDecompressFailed,
    };

    static const char* StatusName(Status s) {
        switch (s) {
            case Status::kOk: return "ok";
            case Status::kMalformed: return "malformed";
            case Status::kUntaggedPose: return "untagged_pose";
            case Status::kBadInformation: return "bad_information";
            case Status::kDecompressFailed: return "decompress_failed";
        }
        return "unknown";
    }

    SerializedMessage EncodePoseGraph(const PoseGraphValue& graph,
                                      AgentId sender, AgentId receiver,
                                      Timestamp t, uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(1024);

        std::vector<flatbuffers::Offset<wire::PoseNode>> nodes;
        nodes.reserve(graph.nodes.size());
        for (const auto& n : graph.nodes) {
            const auto pose = detail::ToWire(n.pose);
            const auto frame = detail::ToWire(n.frame);
            nodes.push_back(wire::CreatePoseNode(fbb, n.agent, n.keyframe,
                                                 n.stamp_ns, &pose, &frame));
        }
        std::vector<flatbuffers::Offset<wire::LoopEdge>> edges;
        edges.reserve(graph.edges.size());
        for (const auto& e : graph.edges) edges.push_back(BuildEdge(fbb, e));

        const auto anchor = detail::ToWire(graph.anchor);
        // 3.2 fields are written only when set (a broker estimate, or a 3.2
        // snapshot carrying its sender's T_G_W), so a Phase-1 graph encodes
        // byte-identically to before the schema grew.
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<wire::AgentFrame>>> frames_off = 0;
        if (graph.has_epoch || !graph.agent_frames.empty()) {
            std::vector<flatbuffers::Offset<wire::AgentFrame>> frames;
            for (const auto& f : graph.agent_frames) {
                const auto T = detail::ToWire(f.T_G_W);
                frames.push_back(wire::CreateAgentFrame(fbb, f.agent, &T));
            }
            frames_off = fbb.CreateVector(frames);
        }
        const wire::Epoch epoch(graph.epoch.counter, graph.epoch.members_hash, graph.epoch.broker);
        const auto pg = wire::CreatePoseGraph(fbb, fbb.CreateVector(nodes), fbb.CreateVector(edges), &anchor,
                                              graph.has_epoch ? &epoch : nullptr, frames_off);

        wire::EnvelopeBuilder eb(fbb);
        eb.add_cls(MessageClass::POSE_GRAPH);
        eb.add_sender(sender);
        eb.add_receiver(receiver);
        eb.add_stamp_ns(t.ns);
        eb.add_sequence(sequence);
        eb.add_pose_graph(pg);
        fbb.Finish(eb.Finish());

        return Finalize(fbb, MessageClass::POSE_GRAPH, sender, receiver, t);
    }

    SerializedMessage EncodeLoopEdge(const LoopEdgeValue& edge, AgentId sender,
                                     AgentId receiver, Timestamp t,
                                     uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(512);
        const auto e = BuildEdge(fbb, edge);
        wire::EnvelopeBuilder eb(fbb);
        eb.add_cls(MessageClass::POSE_EDGE);
        eb.add_sender(sender);
        eb.add_receiver(receiver);
        eb.add_stamp_ns(t.ns);
        eb.add_sequence(sequence);
        eb.add_edge(e);
        fbb.Finish(eb.Finish());
        return Finalize(fbb, MessageClass::POSE_EDGE, sender, receiver, t);
    }

    SerializedMessage EncodeGlobalDescriptor(const GlobalDescriptorValue& g, AgentId sender, AgentId receiver,
                                             Timestamp t, uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(1024 + 4 * g.descriptor.size());
        const auto d = fbb.CreateVector(g.descriptor);
        const auto off = wire::CreateGlobalDescriptor(fbb, g.keyframe_id, g.stamp_ns, d, g.quality, g.n_features,
                                                        g.feature_bytes);
        return FinishEnvelope(fbb, MessageClass::GLOBAL_DESCRIPTOR, sender, receiver, t, sequence,
                              [&](wire::EnvelopeBuilder& eb) { eb.add_global_descriptor(off); });
    }

    SerializedMessage EncodeMatchRequest(const MatchRequestValue& r, AgentId sender, AgentId receiver,
                                         Timestamp t, uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(128);
        const auto off = wire::CreateMatchRequest(fbb, r.request_id, r.query_keyframe, r.candidate_keyframe,
                                                  r.feature_type);
        return FinishEnvelope(fbb, MessageClass::MATCH_REQUEST, sender, receiver, t, sequence,
                              [&](wire::EnvelopeBuilder& eb) { eb.add_match_request(off); });
    }

    SerializedMessage EncodeFeatureResponse(const FeatureResponseValue& f, AgentId sender, AgentId receiver,
                                            Timestamp t, uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(1024 + f.descriptors.size() + 4 * (f.keypoints.size() + f.points3d.size()));
        const auto kp = fbb.CreateVector(f.keypoints);
        const auto de = fbb.CreateVector(f.descriptors);
        const auto p3 = fbb.CreateVector(f.points3d);
        const auto ca = fbb.CreateVector(f.camera);
        const auto off = wire::CreateFeatureResponse(fbb, f.request_id, f.keyframe_id, kp, de, p3, ca);
        return FinishEnvelope(fbb, MessageClass::MATCH_RESPONSE, sender, receiver, t, sequence,
                              [&](wire::EnvelopeBuilder& eb) { eb.add_feature_response(off); });
    }

    SerializedMessage EncodeHeartbeat(const HeartbeatValue& h, AgentId sender, AgentId receiver, Timestamp t,
                                      uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(256);
        const auto neighbors = fbb.CreateVector(h.neighbors);
        const auto ack = BuildAck(fbb, h.ack);
        const auto pose = detail::ToWire(h.pose);
        const auto frame = detail::ToWire(h.frame);
        const wire::Epoch epoch(h.epoch.counter, h.epoch.members_hash, h.epoch.broker);
        const auto off = wire::CreateHeartbeat(fbb, h.agent, h.stamp_ns, &pose, &frame, h.battery, h.link_state,
                                               h.keyframe_watermark, neighbors, h.anchor, h.broker, &epoch, ack);
        return FinishEnvelope(fbb, MessageClass::HEARTBEAT, sender, receiver, t, sequence,
                              [&](wire::EnvelopeBuilder& eb) { eb.add_heartbeat(off); });
    }

    SerializedMessage EncodeAck(const AckValue& a, AgentId sender, AgentId receiver, Timestamp t,
                                uint64_t sequence) const {
        flatbuffers::FlatBufferBuilder fbb(128);
        const auto off = BuildAck(fbb, a);
        return FinishEnvelope(fbb, MessageClass::ACK, sender, receiver, t, sequence,
                              [&](wire::EnvelopeBuilder& eb) { eb.add_ack(off); });
    }

    /// Decode. The message class comes from the frame header, so no
    /// out-of-band knowledge is required. Returns kOk only when the message is
    /// well formed AND every pose carries a frame tag (§15.5).
    Status Decode(const std::vector<uint8_t>& wire_bytes,
                  DecodedMessage* out) const {
        MessageClass cls;
        bool compressed = false;
        uint32_t declared = 0;
        if (!PeekFrameHeader(wire_bytes, &cls, &compressed, &declared)) {
            return Status::kMalformed;
        }
        const std::vector<uint8_t> body(wire_bytes.begin() + kFrameHeaderBytes,
                                        wire_bytes.end());
        std::vector<uint8_t> payload;
        if (compressed) {
            if (!Decompress(body, &payload)) return Status::kDecompressFailed;
        } else {
            payload = body;
        }
        if (payload.size() != declared) return Status::kMalformed;

        flatbuffers::Verifier verifier(payload.data(), payload.size());
        if (!wire::VerifyEnvelopeBuffer(verifier)) return Status::kMalformed;

        const auto* env = wire::GetEnvelope(payload.data());
        if (env == nullptr) return Status::kMalformed;

        out->cls = env->cls();
        out->sender = env->sender();
        out->receiver = env->receiver();
        out->stamp_ns = env->stamp_ns();
        out->sequence = env->sequence();
        out->has_pose_graph = false;
        out->has_edge = false;

        if (const auto* pg = env->pose_graph()) {
            out->has_pose_graph = true;
            out->pose_graph.nodes.clear();
            out->pose_graph.edges.clear();
            if (pg->anchor() == nullptr) return Status::kUntaggedPose;
            out->pose_graph.anchor = detail::FromWire(*pg->anchor());

            if (const auto* nodes = pg->nodes()) {
                for (const auto* n : *nodes) {
                    // A pose without a frame tag cannot be composed with
                    // anything, so it is rejected rather than defaulted.
                    if (n->pose() == nullptr || n->frame() == nullptr) {
                        return Status::kUntaggedPose;
                    }
                    PoseNodeValue v;
                    v.agent = n->agent();
                    v.keyframe = n->keyframe();
                    v.stamp_ns = n->stamp_ns();
                    v.pose = detail::FromWire(*n->pose());
                    v.frame = detail::FromWire(*n->frame());
                    out->pose_graph.nodes.push_back(v);
                }
            }
            if (const auto* edges = pg->edges()) {
                for (const auto* e : *edges) {
                    LoopEdgeValue v;
                    const Status s = ReadEdge(e, &v);
                    if (s != Status::kOk) return s;
                    out->pose_graph.edges.push_back(v);
                }
            }
            out->pose_graph.has_epoch = pg->epoch() != nullptr;
            out->pose_graph.agent_frames.clear();
            if (const auto* ep = pg->epoch()) out->pose_graph.epoch = {ep->counter(), ep->members_hash(), ep->broker()};
            if (const auto* frames = pg->agent_frames()) {
                for (const auto* f : *frames) {
                    if (f->T_G_W() == nullptr) return Status::kUntaggedPose;
                    out->pose_graph.agent_frames.push_back({f->agent(), detail::FromWire(*f->T_G_W())});
                }
            }
        }

        if (const auto* e = env->edge()) {
            out->has_edge = true;
            const Status s = ReadEdge(e, &out->edge);
            if (s != Status::kOk) return s;
        }
        out->has_global_descriptor = out->has_match_request = out->has_feature_response = false;
        if (const auto* g = env->global_descriptor()) {
            if (!g->descriptor()) return Status::kMalformed;
            out->has_global_descriptor = true;
            auto& v = out->global_descriptor;
            v.keyframe_id = g->keyframe_id();
            v.stamp_ns = g->stamp_ns();
            v.descriptor.assign(g->descriptor()->begin(), g->descriptor()->end());
            v.quality = g->quality();
            v.n_features = g->n_features();
            v.feature_bytes = g->feature_bytes();
        }
        if (const auto* r = env->match_request()) {
            out->has_match_request = true;
            out->match_request = {r->request_id(), r->query_keyframe(), r->candidate_keyframe(), r->feature_type()};
        }
        if (const auto* f = env->feature_response()) {
            if (!f->keypoints() || !f->descriptors() || !f->points3d() || !f->camera()) return Status::kMalformed;
            const size_t n = f->keypoints()->size() / 2;
            if (f->keypoints()->size() % 2 || f->descriptors()->size() != 32 * n || f->points3d()->size() != 3 * n ||
                f->camera()->size() != 5) {
                return Status::kMalformed;
            }
            out->has_feature_response = true;
            auto& v = out->feature_response;
            v.request_id = f->request_id();
            v.keyframe_id = f->keyframe_id();
            v.keypoints.assign(f->keypoints()->begin(), f->keypoints()->end());
            v.descriptors.assign(f->descriptors()->begin(), f->descriptors()->end());
            v.points3d.assign(f->points3d()->begin(), f->points3d()->end());
            v.camera.assign(f->camera()->begin(), f->camera()->end());
        }
        out->has_heartbeat = out->has_ack = false;
        if (const auto* h = env->heartbeat()) {
            // §15.5: the heartbeat's pose summary is a pose like any other.
            if (h->pose() == nullptr || h->frame() == nullptr) return Status::kUntaggedPose;
            out->has_heartbeat = true;
            auto& v = out->heartbeat;
            v.agent = h->agent();
            v.stamp_ns = h->stamp_ns();
            v.pose = detail::FromWire(*h->pose());
            v.frame = detail::FromWire(*h->frame());
            v.battery = h->battery();
            v.link_state = h->link_state();
            v.keyframe_watermark = h->keyframe_watermark();
            v.neighbors.clear();
            if (h->neighbors()) v.neighbors.assign(h->neighbors()->begin(), h->neighbors()->end());
            v.anchor = h->anchor();
            v.broker = h->broker();
            v.epoch = {};
            if (const auto* ep = h->epoch()) v.epoch = {ep->counter(), ep->members_hash(), ep->broker()};
            if (!ReadAck(h->ack(), &v.ack)) return Status::kMalformed;
        }
        if (const auto* a = env->ack()) {
            out->has_ack = true;
            if (!ReadAck(a, &out->ack)) return Status::kMalformed;
        }
        return Status::kOk;
    }

private:
    template <typename AddPayload>
    SerializedMessage FinishEnvelope(flatbuffers::FlatBufferBuilder& fbb, MessageClass cls, AgentId sender,
                                     AgentId receiver, Timestamp t, uint64_t sequence, AddPayload add) const {
        wire::EnvelopeBuilder eb(fbb);
        eb.add_cls(cls);
        eb.add_sender(sender);
        eb.add_receiver(receiver);
        eb.add_stamp_ns(t.ns);
        eb.add_sequence(sequence);
        add(eb);
        fbb.Finish(eb.Finish());
        return Finalize(fbb, cls, sender, receiver, t);
    }

    static flatbuffers::Offset<wire::Ack> BuildAck(flatbuffers::FlatBufferBuilder& fbb, const AckValue& a) {
        std::vector<flatbuffers::Offset<wire::PartialAck>> partial;
        for (const auto& p : a.partial)
            partial.push_back(wire::CreatePartialAck(fbb, p.msg_id, fbb.CreateVector(p.received)));
        return wire::CreateAck(fbb, fbb.CreateVector(a.done), fbb.CreateVector(partial));
    }

    /// A missing ack table is an empty ack; a malformed partial entry is not.
    static bool ReadAck(const wire::Ack* a, AckValue* out) {
        *out = {};
        if (a == nullptr) return true;
        if (a->done()) out->done.assign(a->done()->begin(), a->done()->end());
        if (const auto* parts = a->partial()) {
            for (const auto* p : *parts) {
                if (p->received() == nullptr) return false;
                out->partial.push_back({p->msg_id(), {p->received()->begin(), p->received()->end()}});
            }
        }
        return true;
    }

    static flatbuffers::Offset<wire::LoopEdge> BuildEdge(
        flatbuffers::FlatBufferBuilder& fbb, const LoopEdgeValue& e) {
        const auto info = fbb.CreateVector(e.information);
        const auto measurement = detail::ToWire(e.measurement);
        const auto fa = detail::ToWire(e.frame_a);
        const auto fb = detail::ToWire(e.frame_b);
        wire::LoopEdgeBuilder lb(fbb);
        lb.add_agent_a(e.agent_a);
        lb.add_keyframe_a(e.keyframe_a);
        lb.add_agent_b(e.agent_b);
        lb.add_keyframe_b(e.keyframe_b);
        lb.add_measurement(&measurement);
        lb.add_information(info);
        lb.add_frame_a(&fa);
        lb.add_frame_b(&fb);
        lb.add_status(e.status);
        lb.add_inlier_count(e.inlier_count);
        lb.add_retrieval_score(e.retrieval_score);
        lb.add_pcm_score(e.pcm_score);
        lb.add_gnc_weight(e.gnc_weight);
        lb.add_degeneracy_flag(e.degeneracy_flag);
        return lb.Finish();
    }

    static Status ReadEdge(const wire::LoopEdge* e, LoopEdgeValue* out) {
        if (e->measurement() == nullptr) return Status::kUntaggedPose;
        if (e->frame_a() == nullptr || e->frame_b() == nullptr) {
            return Status::kUntaggedPose;
        }
        const auto* info = e->information();
        if (info == nullptr || info->size() != 21) return Status::kBadInformation;

        out->agent_a = e->agent_a();
        out->keyframe_a = e->keyframe_a();
        out->agent_b = e->agent_b();
        out->keyframe_b = e->keyframe_b();
        out->measurement = detail::FromWire(*e->measurement());
        out->information.assign(info->begin(), info->end());
        out->frame_a = detail::FromWire(*e->frame_a());
        out->frame_b = detail::FromWire(*e->frame_b());
        out->status = e->status();
        out->inlier_count = e->inlier_count();
        out->retrieval_score = e->retrieval_score();
        out->pcm_score = e->pcm_score();
        out->gnc_weight = e->gnc_weight();
        out->degeneracy_flag = e->degeneracy_flag();
        return Status::kOk;
    }

    /// Compress where the class calls for it, prepend the frame header, then
    /// charge datagram framing. This is the ONE place wire_bytes is computed.
    SerializedMessage Finalize(flatbuffers::FlatBufferBuilder& fbb,
                               MessageClass cls, AgentId sender,
                               AgentId receiver, Timestamp t) const {
        SerializedMessage msg;
        const uint8_t* data = fbb.GetBufferPointer();
        const size_t size = fbb.GetSize();

        msg.record.t = t;
        msg.record.sender = sender;
        msg.record.receiver = receiver;
        msg.record.type = cls;
        msg.record.payload_bytes = size;

        std::vector<uint8_t> body;
        if (IsCompressedClass(cls)) {
            const size_t bound = ZSTD_compressBound(size);
            std::vector<uint8_t> compressed(bound);
            const size_t written = ZSTD_compress(compressed.data(), bound, data,
                                                 size, kCompressionLevel);
            // Only keep the compressed form if it actually helped. High-entropy
            // descriptors can expand under zstd; sending the larger form would
            // make the byte table worse and the claim dishonest.
            if (!ZSTD_isError(written) && written < size) {
                compressed.resize(written);
                body = std::move(compressed);
                msg.record.compressed = true;
            } else {
                body.assign(data, data + size);
            }
        } else {
            body.assign(data, data + size);
        }

        msg.bytes.reserve(kFrameHeaderBytes + body.size());
        msg.bytes.push_back(static_cast<uint8_t>(cls));
        msg.bytes.push_back(msg.record.compressed ? kFlagCompressed : 0);
        const uint32_t declared = static_cast<uint32_t>(size);
        msg.bytes.push_back(static_cast<uint8_t>(declared & 0xFF));
        msg.bytes.push_back(static_cast<uint8_t>((declared >> 8) & 0xFF));
        msg.bytes.push_back(static_cast<uint8_t>((declared >> 16) & 0xFF));
        msg.bytes.push_back(static_cast<uint8_t>((declared >> 24) & 0xFF));
        msg.bytes.insert(msg.bytes.end(), body.begin(), body.end());

        msg.record.transmitted_bytes = body.size();
        const uint64_t framed = kFrameHeaderBytes + body.size();
        msg.record.wire_bytes =
            framing_ == Framing::kWholeMessage
                ? framed + kDatagramOverheadBytes * DatagramCount(framed)
                : framed + (kDatagramOverheadBytes + kFragmentHeaderBytes) * FragmentCount(framed);
        return msg;
    }

    static bool Decompress(const std::vector<uint8_t>& in,
                           std::vector<uint8_t>* out) {
        const unsigned long long declared =
            ZSTD_getFrameContentSize(in.data(), in.size());
        if (declared == ZSTD_CONTENTSIZE_ERROR) {
            // Not a zstd frame. The encoder falls back to uncompressed on a
            // compression failure, so accept the payload as-is.
            *out = in;
            return true;
        }
        if (declared == ZSTD_CONTENTSIZE_UNKNOWN) return false;

        out->resize(static_cast<size_t>(declared));
        const size_t written =
            ZSTD_decompress(out->data(), out->size(), in.data(), in.size());
        if (ZSTD_isError(written) || written != out->size()) return false;
        return true;
    }

    Framing framing_;
};

}  // namespace ariadne::core
