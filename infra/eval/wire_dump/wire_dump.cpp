// DIAGNOSTIC (not a gate; post-demo backlog): hex dump of the four fixed messages of
// Serialization.PreTaskConsensusBytesUnchanged (tests/protocol/test_serialization.cpp), byte for byte, so the x86 and
// arm64 encodings can be compared offset by offset. The fixtures below are copied verbatim from that test.
//   infra/eval/wire_dump/run.sh   (configure + build the core deps, compile this file, print the dump)
#include "core/serialization/serializer.hpp"

#include <cmath>
#include <cstdio>

using namespace ariadne::core;
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

HeartbeatValue FixedHeartbeat() {
    HeartbeatValue h;
    h.agent = 1, h.stamp_ns = 123456789, h.pose = MakePose(1, 2, 3), h.frame = FrameTag{1, 1};
    h.battery = 0.8f, h.link_state = 0.5f, h.keyframe_watermark = 42, h.neighbors = {0, 2};
    h.anchor = 0, h.broker = 0, h.epoch = {3, 0xABCDEFull, 0};
    h.ack.done = {7, 8};
    return h;
}

uint64_t Fnv(const std::vector<uint8_t>& b) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t x : b) h ^= x, h *= 1099511628211ull;
    return h;
}

void Dump(const char* name, const SerializedMessage& m) {
    std::printf("%s size %zu fnv 0x%016llx\n", name, m.bytes.size(), (unsigned long long)Fnv(m.bytes));
    for (size_t i = 0; i < m.bytes.size(); ++i) std::printf("%s%02x", i % 32 ? " " : (i ? "\n  " : "  "), m.bytes[i]);
    std::printf("\n");
}

}  // namespace

int main() {
    const WireCodec codec(Framing::kWholeMessage);
    AckValue a;
    a.done = {1, 2, 3};
    Dump("heartbeat", codec.EncodeHeartbeat(FixedHeartbeat(), 1, 0, Timestamp{987654321}, 11));
    Dump("loop_edge", codec.EncodeLoopEdge(MakeEdge(0, 1), 0, 1, Timestamp{987654321}, 12));
    Dump("pose_graph", codec.EncodePoseGraph(MakeGraph(5, 3), 0, 1, Timestamp{987654321}, 13));
    Dump("ack", codec.EncodeAck(a, 1, 0, Timestamp{987654321}, 14));
    return 0;
}
