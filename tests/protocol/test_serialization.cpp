// The single serialization boundary (arch §4.4, §15.7, §37.1 round-trip).
#include "core/serialization/serializer.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace ariadne::core;
// `wire::` resolves inside namespace ariadne::core by enclosing-namespace
// lookup, but not at global scope; alias it explicitly here.
namespace wire = ariadne::wire;

namespace {

PoseValue MakePose(float tx, float ty, float tz) {
    PoseValue p;
    // A non-identity rotation, normalized, so the round-trip has something to
    // preserve beyond zeros.
    const float n = std::sqrt(1.0f + 0.1f * 0.1f + 0.2f * 0.2f + 0.3f * 0.3f);
    p.qw = 1.0f / n; p.qx = 0.1f / n; p.qy = 0.2f / n; p.qz = 0.3f / n;
    p.tx = tx; p.ty = ty; p.tz = tz;
    return p;
}

LoopEdgeValue MakeEdge(uint32_t a, uint32_t b) {
    LoopEdgeValue e;
    e.agent_a = a; e.keyframe_a = 100 + a;
    e.agent_b = b; e.keyframe_b = 200 + b;
    e.measurement = MakePose(1.5f, -2.5f, 0.25f);
    for (int i = 0; i < 21; ++i) e.information[i] = static_cast<float>(i) * 0.5f;
    e.frame_a = FrameTag{a, 1};
    e.frame_b = FrameTag{b, 1};
    e.status = wire::EdgeStatus::GNC_INLIER;
    e.inlier_count = 87;
    e.retrieval_score = 0.91f;
    e.pcm_score = 0.77f;
    e.gnc_weight = 0.99f;
    e.degeneracy_flag = false;
    return e;
}

PoseGraphValue MakeGraph(int nodes, int edges) {
    PoseGraphValue g;
    g.anchor = FrameTag{0, 1};
    for (int i = 0; i < nodes; ++i) {
        PoseNodeValue n;
        n.agent = static_cast<uint32_t>(i % 3);
        n.keyframe = static_cast<uint64_t>(i);
        n.stamp_ns = 1000000LL * i;
        n.pose = MakePose(static_cast<float>(i), 0.5f * i, -0.25f * i);
        n.frame = FrameTag{n.agent, 1};
        g.nodes.push_back(n);
    }
    for (int i = 0; i < edges; ++i) {
        g.edges.push_back(MakeEdge(static_cast<uint32_t>(i % 3),
                                   static_cast<uint32_t>((i + 1) % 3)));
    }
    return g;
}

void ExpectPoseEq(const PoseValue& a, const PoseValue& b) {
    EXPECT_EQ(a.qw, b.qw); EXPECT_EQ(a.qx, b.qx);
    EXPECT_EQ(a.qy, b.qy); EXPECT_EQ(a.qz, b.qz);
    EXPECT_EQ(a.tx, b.tx); EXPECT_EQ(a.ty, b.ty); EXPECT_EQ(a.tz, b.tz);
}

}  // namespace

// --- Round-trip -----------------------------------------------------------

TEST(Serialization, PoseGraphRoundTripsExactly) {
    const WireCodec codec;
    const PoseGraphValue graph = MakeGraph(40, 7);
    const SerializedMessage msg =
        codec.EncodePoseGraph(graph, 1, 2, Timestamp::Nanoseconds(12345), 9);

    DecodedMessage decoded;
    ASSERT_EQ(codec.Decode(msg.bytes, &decoded),
              WireCodec::Status::kOk);

    EXPECT_EQ(decoded.cls, MessageClass::POSE_GRAPH);
    EXPECT_EQ(decoded.sender, 1u);
    EXPECT_EQ(decoded.receiver, 2u);
    EXPECT_EQ(decoded.stamp_ns, 12345);
    EXPECT_EQ(decoded.sequence, 9u);
    ASSERT_TRUE(decoded.has_pose_graph);
    ASSERT_EQ(decoded.pose_graph.nodes.size(), graph.nodes.size());
    ASSERT_EQ(decoded.pose_graph.edges.size(), graph.edges.size());

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& a = graph.nodes[i];
        const auto& b = decoded.pose_graph.nodes[i];
        EXPECT_EQ(a.agent, b.agent);
        EXPECT_EQ(a.keyframe, b.keyframe);
        EXPECT_EQ(a.stamp_ns, b.stamp_ns);
        EXPECT_EQ(a.frame.agent, b.frame.agent);
        EXPECT_EQ(a.frame.epoch, b.frame.epoch);
        ExpectPoseEq(a.pose, b.pose);   // float32 in, float32 out: exact
    }
}

TEST(Serialization, LoopEdgeRoundTripsExactly) {
    const WireCodec codec;
    const LoopEdgeValue edge = MakeEdge(0, 1);
    const SerializedMessage msg =
        codec.EncodeLoopEdge(edge, 0, 1, Timestamp::Nanoseconds(77), 3);

    DecodedMessage decoded;
    ASSERT_EQ(codec.Decode(msg.bytes, &decoded),
              WireCodec::Status::kOk);
    ASSERT_TRUE(decoded.has_edge);

    const auto& d = decoded.edge;
    EXPECT_EQ(d.agent_a, edge.agent_a);
    EXPECT_EQ(d.keyframe_b, edge.keyframe_b);
    EXPECT_EQ(d.status, edge.status);
    EXPECT_EQ(d.inlier_count, edge.inlier_count);
    EXPECT_EQ(d.degeneracy_flag, edge.degeneracy_flag);
    EXPECT_FLOAT_EQ(d.gnc_weight, edge.gnc_weight);
    ExpectPoseEq(d.measurement, edge.measurement);
    ASSERT_EQ(d.information.size(), 21u);
    for (int i = 0; i < 21; ++i) EXPECT_EQ(d.information[i], edge.information[i]);
}

// Serialization must be a pure function of its input: the same graph must
// produce the same bytes every time, or replay would not be reproducible.
TEST(Serialization, EncodingIsDeterministic) {
    const WireCodec codec;
    const PoseGraphValue graph = MakeGraph(25, 5);
    const auto a = codec.EncodePoseGraph(graph, 1, 2, Timestamp::Nanoseconds(5), 1);
    const auto b = codec.EncodePoseGraph(graph, 1, 2, Timestamp::Nanoseconds(5), 1);
    EXPECT_EQ(a.bytes, b.bytes);
    EXPECT_EQ(a.record.wire_bytes, b.record.wire_bytes);
}

// --- Frame tagging (§15.5) ------------------------------------------------

// A pose whose reference frame is unknown cannot be composed with anything, so
// the deserializer rejects it rather than silently defaulting to frame 0. A
// silent default is how one agent's map ends up expressed in another's frame.
TEST(Serialization, UntaggedPoseIsRejected) {
    // Hand-build an envelope with a pose but no frame tag.
    flatbuffers::FlatBufferBuilder fbb(256);
    const auto pose = detail::ToWire(MakePose(1, 2, 3));
    std::vector<flatbuffers::Offset<wire::PoseNode>> nodes;
    {
        wire::PoseNodeBuilder nb(fbb);
        nb.add_agent(1);
        nb.add_keyframe(5);
        nb.add_pose(&pose);
        // deliberately no add_frame(...)
        nodes.push_back(nb.Finish());
    }
    const auto anchor = detail::ToWire(FrameTag{0, 1});
    const auto pg = wire::CreatePoseGraph(fbb, fbb.CreateVector(nodes), 0, &anchor);
    wire::EnvelopeBuilder eb(fbb);
    eb.add_cls(MessageClass::POSE_GRAPH);
    eb.add_pose_graph(pg);
    fbb.Finish(eb.Finish());

    // Prepend a valid, uncompressed frame header so the message reaches the
    // frame-tag check rather than being rejected as malformed framing.
    const uint32_t payload_size = fbb.GetSize();
    std::vector<uint8_t> bytes;
    bytes.push_back(static_cast<uint8_t>(MessageClass::POSE_GRAPH));
    bytes.push_back(0);
    bytes.push_back(static_cast<uint8_t>(payload_size & 0xFF));
    bytes.push_back(static_cast<uint8_t>((payload_size >> 8) & 0xFF));
    bytes.push_back(static_cast<uint8_t>((payload_size >> 16) & 0xFF));
    bytes.push_back(static_cast<uint8_t>((payload_size >> 24) & 0xFF));
    bytes.insert(bytes.end(), fbb.GetBufferPointer(),
                 fbb.GetBufferPointer() + payload_size);
    const WireCodec codec;
    DecodedMessage decoded;
    EXPECT_EQ(codec.Decode(bytes, &decoded),
              WireCodec::Status::kUntaggedPose);
}

// --- Malformed input ------------------------------------------------------

// A corrupt or hostile message must degrade one link, not the process.
TEST(Serialization, MalformedInputIsRejectedNotCrashed) {
    const WireCodec codec;
    DecodedMessage decoded;

    EXPECT_NE(codec.Decode({}, &decoded),
              WireCodec::Status::kOk);

    std::vector<uint8_t> garbage(64, 0xAB);
    EXPECT_NE(codec.Decode(garbage, &decoded),
              WireCodec::Status::kOk);

    // Truncation of a valid buffer.
    auto msg = codec.EncodePoseGraph(MakeGraph(10, 2), 1, 2,
                                     Timestamp::Nanoseconds(1), 1);
    msg.bytes.resize(msg.bytes.size() / 2);
    EXPECT_NE(codec.Decode(msg.bytes, &decoded),
              WireCodec::Status::kOk);
}

// --- Byte accounting (§15.7) ---------------------------------------------

TEST(Serialization, WireBytesIncludeDatagramFraming) {
    const WireCodec codec;
    const auto msg = codec.EncodePoseGraph(MakeGraph(10, 1), 1, 2,
                                           Timestamp::Nanoseconds(1), 1);
    const uint64_t framed = kFrameHeaderBytes + msg.record.transmitted_bytes;
    EXPECT_EQ(msg.bytes.size(), framed);
    EXPECT_EQ(msg.record.wire_bytes,
              framed + kDatagramOverheadBytes * DatagramCount(framed));
    // This one fits in a single datagram, so the old per-message arithmetic
    // happens to agree here. The multi-datagram case below is where they part.
    EXPECT_LE(framed, kDatagramPayloadBytes);
    // Framing is always charged on top of what is transmitted. Note this is
    // NOT the same as wire > payload: with POSE_GRAPH compression (D14) the
    // wire form is routinely far smaller than the uncompressed payload.
    EXPECT_GT(msg.record.wire_bytes, msg.record.transmitted_bytes);
}

// §15.7 charges "28 B emulated UDP/IP header PER DATAGRAM". A message larger
// than one datagram's payload fragments, and each fragment carries its own
// header. Charging once per message understated framing on exactly the large
// classes the Phase 3 <=15% prediction gate has to price.
TEST(Serialization, LargeMessagesAreChargedFramingPerDatagram) {
    const WireCodec codec;
    const auto msg = codec.EncodePoseGraph(MakeGraph(200, 40), 1, 2,
                                           Timestamp::Nanoseconds(1), 1);
    const uint64_t framed = kFrameHeaderBytes + msg.record.transmitted_bytes;
    ASSERT_GT(framed, kDatagramPayloadBytes) << "test needs a multi-MTU message";

    const uint64_t datagrams = DatagramCount(framed);
    EXPECT_GT(datagrams, 1u);
    EXPECT_EQ(msg.record.wire_bytes,
              framed + kDatagramOverheadBytes * datagrams);
    // Strictly more than the old per-message arithmetic would have charged.
    EXPECT_GT(msg.record.wire_bytes, framed + kDatagramOverheadBytes);
}

TEST(Serialization, DatagramCountRoundsUpAndNeverReturnsZero) {
    EXPECT_EQ(DatagramCount(0), 1u);            // an empty message still costs one
    EXPECT_EQ(DatagramCount(1), 1u);
    EXPECT_EQ(DatagramCount(kDatagramPayloadBytes), 1u);
    EXPECT_EQ(DatagramCount(kDatagramPayloadBytes + 1), 2u);
    EXPECT_EQ(DatagramCount(2 * kDatagramPayloadBytes), 2u);
    EXPECT_EQ(DatagramCount(2 * kDatagramPayloadBytes + 1), 3u);
}

// The compression policy is stated once in IsCompressedClass and asserted here
// so it cannot drift. POSE_GRAPH extends §15.7's classes 4-5; see D14.
TEST(Serialization, CompressionPolicyCoversTheIntendedClasses) {
    EXPECT_TRUE(IsCompressedClass(MessageClass::MATCH_RESPONSE));
    EXPECT_TRUE(IsCompressedClass(MessageClass::POSE_EDGE));
    EXPECT_TRUE(IsCompressedClass(MessageClass::POSE_GRAPH));
    EXPECT_FALSE(IsCompressedClass(MessageClass::HEARTBEAT));
    EXPECT_FALSE(IsCompressedClass(MessageClass::GLOBAL_DESCRIPTOR));

    const WireCodec codec;
    const auto edge = codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1,
                                           Timestamp::Nanoseconds(1), 1);
    EXPECT_TRUE(edge.record.compressed);
    const auto graph = codec.EncodePoseGraph(MakeGraph(200, 40), 0, 1,
                                             Timestamp::Nanoseconds(1), 1);
    EXPECT_TRUE(graph.record.compressed);

    // The whole point of D14: a realistic batch must now land under §15.7's
    // 12 KB prediction rather than 75% over it.
    EXPECT_LT(graph.record.wire_bytes, 12288u)
        << "pose-graph batch no longer fits the §15.7 budget";
    std::cout << "[ MEASURED ] pose graph 200n/40e COMPRESSED: payload="
              << graph.record.payload_bytes
              << " transmitted=" << graph.record.transmitted_bytes
              << " wire=" << graph.record.wire_bytes
              << " ratio=" << graph.record.CompressionRatio() << "x\n";
}

TEST(Serialization, LedgerAccumulatesTotalAndPerClass) {
    const WireCodec codec;
    ByteLedger ledger;

    uint64_t expected = 0;
    for (int i = 0; i < 5; ++i) {
        auto m = codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1,
                                      Timestamp::Nanoseconds(i), i);
        expected += m.record.wire_bytes;
        ledger.Record(m.record);
        EXPECT_EQ(m.record.cumulative_bytes, expected);
    }
    for (int i = 0; i < 3; ++i) {
        auto m = codec.EncodePoseGraph(MakeGraph(20, 2), 0, 1,
                                       Timestamp::Nanoseconds(i), i);
        expected += m.record.wire_bytes;
        ledger.Record(m.record);
    }

    EXPECT_EQ(ledger.Total(), expected);
    EXPECT_EQ(ledger.CountForClass(MessageClass::POSE_EDGE), 5u);
    EXPECT_EQ(ledger.CountForClass(MessageClass::POSE_GRAPH), 3u);
    EXPECT_EQ(ledger.Total(), ledger.TotalForClass(MessageClass::POSE_EDGE) +
                                  ledger.TotalForClass(MessageClass::POSE_GRAPH));

    ledger.Clear();
    EXPECT_EQ(ledger.Total(), 0u);
}

// §15.7's size table is a PREDICTION, and Phase 3 gates predicted-vs-measured
// at 15%. This records what the schema actually costs today so the prediction
// can be checked rather than assumed. Reported, not silently accepted.
TEST(Serialization, MeasuredSizesAreReportedAgainstTheSection157Table) {
    const WireCodec codec;

    const auto edge = codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1,
                                           Timestamp::Nanoseconds(1), 1);
    std::cout << "[ MEASURED ] pose edge: payload=" << edge.record.payload_bytes
              << " transmitted=" << edge.record.transmitted_bytes
              << " wire=" << edge.record.wire_bytes
              << " ratio=" << edge.record.CompressionRatio()
              << "  (§15.7 predicts ~150 B)\n";

    // 200 nodes + 40 edges, the worked example in §15.7 (~12 KB predicted).
    const auto graph = codec.EncodePoseGraph(MakeGraph(200, 40), 0, 1,
                                             Timestamp::Nanoseconds(1), 1);
    const double per_node =
        static_cast<double>(graph.record.payload_bytes) / 240.0;
    std::cout << "[ MEASURED ] pose graph 200n/40e: payload="
              << graph.record.payload_bytes
              << " wire=" << graph.record.wire_bytes
              << " (~" << per_node << " B per element"
              << ", §15.7 predicts ~12 KB total)\n";

    // Per-element cost, isolated by differencing against an empty graph so the
    // envelope overhead does not contaminate the figure. See DECISIONS.md D12.
    const auto empty = codec.EncodePoseGraph(MakeGraph(0, 0), 0, 1,
                                             Timestamp::Nanoseconds(1), 1);
    const auto nodes_only = codec.EncodePoseGraph(MakeGraph(200, 0), 0, 1,
                                                  Timestamp::Nanoseconds(1), 1);
    const auto edges_only = codec.EncodePoseGraph(MakeGraph(0, 40), 0, 1,
                                                  Timestamp::Nanoseconds(1), 1);
    const double per_node_b =
        (nodes_only.record.payload_bytes - empty.record.payload_bytes) / 200.0;
    const double per_edge_b =
        (edges_only.record.payload_bytes - empty.record.payload_bytes) / 40.0;
    std::cout << "[ MEASURED ] per node=" << per_node_b << " B (§15.7: 28 B)"
              << ", per edge=" << per_edge_b << " B (§15.7: 150 B)"
              << ", envelope=" << empty.record.payload_bytes << " B\n";

    // Both elements exceed the §15.7 prediction; DECISIONS.md D12 records the
    // measured values and why the node figure is unreachable under §15.5.
    // These bounds hold the MEASURED values so a further regression is caught,
    // and must be revisited together with §15.7 when D12 is decided.
    //
    // NOTE: measure on POPULATED messages. FlatBuffers omits scalars equal to
    // their schema default, so default-constructed objects understate the wire
    // size -- which is exactly how the first version of this measurement
    // reported a falsely reassuring 148.8 B for an edge.
    EXPECT_LT(per_node_b, 80.0) << "pose-graph node size regressed beyond D12";
    EXPECT_LT(per_edge_b, 220.0) << "pose-edge size regressed beyond D12";

    // Loose bounds: catch an order-of-magnitude schema regression without
    // freezing numbers the design may revise.
    EXPECT_LT(edge.record.wire_bytes, 600u);
    EXPECT_LT(graph.record.wire_bytes, 60000u);
}

// --- Frame header (§15.3 store-and-forward) ------------------------------

// A queue must be able to triage by class and TTL without decompressing, so
// the class travels in an uncompressed header, not only inside the envelope.
TEST(Serialization, ClassIsReadableWithoutDecompressing) {
    const WireCodec codec;
    const auto msg = codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1,
                                          Timestamp::Nanoseconds(1), 1);
    ASSERT_TRUE(msg.record.compressed);   // this class is compressed

    MessageClass cls;
    bool compressed = false;
    uint32_t declared = 0;
    ASSERT_TRUE(PeekFrameHeader(msg.bytes, &cls, &compressed, &declared));
    EXPECT_EQ(cls, MessageClass::POSE_EDGE);
    EXPECT_TRUE(compressed);
    EXPECT_EQ(declared, msg.record.payload_bytes);
}

TEST(Serialization, TruncatedOrInvalidHeaderIsRejected) {
    MessageClass cls;
    bool compressed = false;
    uint32_t declared = 0;
    EXPECT_FALSE(PeekFrameHeader({1, 2, 3}, &cls, &compressed, &declared));
    // Class byte outside the enum.
    EXPECT_FALSE(PeekFrameHeader({99, 0, 0, 0, 0, 0}, &cls, &compressed, &declared));
}

// D14 rests on pose graphs being highly compressible. Assert that directly, so
// a schema change that destroys compressibility (for example switching ids to
// random uuids) is caught here rather than as a blown byte budget in Phase 3.
TEST(Serialization, PoseGraphsRemainHighlyCompressible) {
    const WireCodec codec;
    const auto graph = codec.EncodePoseGraph(MakeGraph(200, 40), 0, 1,
                                             Timestamp::Nanoseconds(1), 1);
    ASSERT_TRUE(graph.record.compressed);
    EXPECT_GT(graph.record.CompressionRatio(), 3.0)
        << "pose-graph compressibility regressed; D14's byte budget assumes ~5x";
}

// --- Pre-D115 bytes are unchanged by the CbbaConsensus append (master's rule 2) ---------------------------------
// Size and FNV-1a of fixed messages, recorded with the PRE-change tree (feat/ph4-mission at 1ca0457, whose wire.fbs
// and serializer.hpp equal main's), so appending TaskConsensus must leave every existing Envelope message
// byte-identical.
namespace {
uint64_t Fnv(const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : b) h ^= x, h *= 1099511628211ull;
    return h;
}
HeartbeatValue FixedHeartbeat() {
    HeartbeatValue h;
    h.agent = 1, h.stamp_ns = 123456789, h.pose = MakePose(1, 2, 3), h.frame = FrameTag{1, 1};
    h.battery = 0.8f, h.link_state = 0.5f, h.keyframe_watermark = 42, h.neighbors = {0, 2};
    h.anchor = 0, h.broker = 0, h.epoch = {3, 0xABCDEFull, 0};
    h.ack.done = {7, 8};
    return h;
}
}  // namespace

TEST(Serialization, PreTaskConsensusBytesUnchanged) {
    const WireCodec codec(Framing::kWholeMessage);
    const auto hb = codec.EncodeHeartbeat(FixedHeartbeat(), 1, 0, Timestamp{987654321}, 11);
    const auto ed = codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1, Timestamp{987654321}, 12);
    const auto pg = codec.EncodePoseGraph(MakeGraph(5, 3), 0, 1, Timestamp{987654321}, 13);
    AckValue a;
    a.done = {1, 2, 3};
    const auto ak = codec.EncodeAck(a, 1, 0, Timestamp{987654321}, 14);
    EXPECT_EQ(hb.bytes.size(), 238u);
    EXPECT_EQ(Fnv(hb.bytes), 0x5d85433fd5441b89ull) << "heartbeat bytes changed";
    EXPECT_EQ(ed.bytes.size(), 194u);
    EXPECT_EQ(Fnv(ed.bytes), 0x49bb277520bc078bull) << "loop-edge bytes changed";
    EXPECT_EQ(pg.bytes.size(), 417u);
    EXPECT_EQ(Fnv(pg.bytes), 0xb6f2ab38c40f0161ull) << "pose-graph bytes changed";
    EXPECT_EQ(ak.bytes.size(), 110u);
    EXPECT_EQ(Fnv(ak.bytes), 0x011ede3f1e7c8e79ull) << "ack bytes changed";
}

TEST(Serialization, CbbaConsensusRoundTripsAndIsCounted) {   // D115: class TASK_CONSENSUS through FinishEnvelope
    const WireCodec codec(Framing::kWholeMessage);
    CbbaConsensusValue v;
    v.tasks = {101, 202, 303};
    v.y = {0.5f, 0.0f, 1.25f};
    v.z = {1, -1, 0};
    v.s = {7, 9};
    v.scales = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto m = codec.EncodeCbbaConsensus(v, 1, 0, Timestamp{5}, 3);
    EXPECT_EQ(m.record.type, MessageClass::TASK_CONSENSUS);
    EXPECT_FALSE(m.record.compressed) << "§15.7 compresses classes 4-5 only";
    DecodedMessage d;
    ASSERT_EQ(codec.Decode(m.bytes, &d), WireCodec::Status::kOk);
    ASSERT_TRUE(d.has_cbba_consensus);
    EXPECT_EQ(d.cbba_consensus.tasks, v.tasks);
    EXPECT_EQ(d.cbba_consensus.y, v.y);
    EXPECT_EQ(d.cbba_consensus.z, v.z);
    EXPECT_EQ(d.cbba_consensus.s, v.s);
    EXPECT_EQ(d.cbba_consensus.scales, v.scales);
    ByteLedger ledger;
    auto rec = m.record;
    ledger.Record(rec);
    EXPECT_EQ(ledger.TotalForClass(MessageClass::TASK_CONSENSUS), m.record.wire_bytes);
    EXPECT_GE(m.record.wire_bytes, m.bytes.size()) << "framing is charged on top";
    CbbaConsensusValue bad = v;   // per-task vectors of unequal length are malformed, not silently truncated
    bad.y.pop_back();
    const auto mb = codec.EncodeCbbaConsensus(bad, 1, 0, Timestamp{5}, 4);
    EXPECT_EQ(codec.Decode(mb.bytes, &d), WireCodec::Status::kMalformed);
}

TEST(Serialization, LargeFeatureResponseRoundTripsThroughFragmentation) {
    // D171: an ALIKED-n16 response at its real size (1024 features, int8 128-D + scale, 3D points): ~93 fragments.
    // The protocol tests' scenes use ~200 features; the sim integration first exercised this size.
    using namespace ariadne::core;
    for (auto framing : {Framing::kWholeMessage, Framing::kFragmented}) {
        const WireCodec codec(framing);
        FeatureResponseValue f;
        f.request_id = 7;
        f.keyframe_id = 99;
        for (int i = 0; i < 1024; ++i) {
            f.keypoints.push_back(float(i % 1280)), f.keypoints.push_back(float(i % 720));
            for (int k = 0; k < 3; ++k) f.points3d.push_back(i % 5 == 0 ? NAN : 0.01f * float(i + k));
        }
        f.descriptors.resize(4 + 1024 * 128);
        for (size_t i = 0; i < f.descriptors.size(); ++i) f.descriptors[i] = uint8_t((i * 2654435761u) >> 13);
        f.camera = {672.f, 672.f, 640.f, 360.f, 0.1f};
        f.feature_bytes = 128;
        f.header_bytes = 4;
        const auto m = codec.EncodeFeatureResponse(f, 1, 0, Timestamp{123456789}, (1ull << 40) | 5);
        DecodedMessage d;
        ASSERT_EQ(codec.Decode(m.bytes, &d), WireCodec::Status::kOk) << "framing " << int(framing) << ", "
                                                                     << m.bytes.size() << " B";
        ASSERT_TRUE(d.has_feature_response);
        EXPECT_EQ(d.feature_response.descriptors, f.descriptors);
        EXPECT_EQ(d.feature_response.keypoints.size(), f.keypoints.size());
        EXPECT_EQ(d.feature_response.feature_bytes, 128u);
        EXPECT_EQ(d.feature_response.header_bytes, 4u);
        f.feature_bytes = 0;   // a v1 decoder's rule: 32 B per feature, so this layout is malformed without the fields
        EXPECT_EQ(codec.Decode(codec.EncodeFeatureResponse(f, 1, 0, Timestamp{123456789}, 5).bytes, &d),
                  WireCodec::Status::kMalformed);
    }
}
