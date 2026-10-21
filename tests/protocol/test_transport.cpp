// COLLAB_3.2 §3: the 3.2 transport (fragments, per-class delivery, outbox, selective ARQ).
#include "collaboration/protocol/transport.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace ac = ariadne::collaboration;
namespace core = ariadne::core;
using core::Duration;
using core::MessageClass;
using core::Timestamp;

namespace {

Timestamp At(double s) { return Timestamp{int64_t(std::llround(s * 1e9))}; }

const core::WireCodec& Codec() {
    static const core::WireCodec c(core::Framing::kFragmented);
    return c;
}

core::GlobalDescriptorValue Descriptor(uint64_t kf) {
    core::GlobalDescriptorValue g;
    g.keyframe_id = kf;
    g.descriptor.assign(8, 0.5f);
    return g;
}

/// A pose graph that does not compress away: ~`n` nodes with pseudo-random poses.
core::PoseGraphValue Graph(uint32_t agent, size_t n, uint32_t salt = 0) {
    core::PoseGraphValue g;
    g.anchor = {agent, 0};
    uint64_t x = 0x9E3779B97F4A7C15ull ^ salt;
    auto rnd = [&] {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        return float(x % 100000) * 1e-3f;
    };
    for (size_t i = 0; i < n; ++i) {
        core::PoseNodeValue v;
        v.agent = agent;
        v.keyframe = i;
        v.stamp_ns = int64_t(i) * 1000003;
        v.pose = {rnd(), rnd(), rnd(), rnd(), rnd(), rnd(), rnd()};
        v.frame = {agent, 0};
        g.nodes.push_back(v);
    }
    return g;
}

ac::OutMessage DescriptorMsg(uint32_t from, uint32_t to, uint64_t kf, Timestamp created,
                             std::optional<Duration> ttl = std::nullopt) {
    ac::OutMessage m;
    m.cls = MessageClass::GLOBAL_DESCRIPTOR;
    m.key = kf;
    m.created = created;
    m.ttl = ttl;
    m.encode = [=](uint64_t seq, Timestamp t) { return Codec().EncodeGlobalDescriptor(Descriptor(kf), from, to, t, seq); };
    return m;
}

ac::OutMessage GraphMsg(uint32_t from, uint32_t to, size_t n, uint32_t salt, Timestamp created) {
    ac::OutMessage m;
    m.cls = MessageClass::POSE_GRAPH;
    m.key = salt;
    m.created = created;
    m.supersedes = true;
    const auto g = Graph(from, n, salt);
    m.encode = [=](uint64_t seq, Timestamp t) { return Codec().EncodePoseGraph(g, from, to, t, seq); };
    return m;
}

ac::OutMessage EdgeMsg(uint32_t from, uint32_t to, uint64_t key, Timestamp created) {
    ac::OutMessage m;
    m.cls = MessageClass::POSE_EDGE;
    m.key = key;
    m.created = created;
    core::LoopEdgeValue e;
    e.agent_a = from;
    e.keyframe_a = key;
    e.agent_b = to;
    e.frame_a = {from, 0};
    e.frame_b = {to, 0};
    m.encode = [=](uint64_t seq, Timestamp t) { return Codec().EncodeLoopEdge(e, from, to, t, seq); };
    return m;
}

ac::OutMessage ResponseMsg(uint32_t from, uint32_t to, uint64_t request, size_t keypoints, Timestamp created) {
    ac::OutMessage m;
    m.cls = MessageClass::MATCH_RESPONSE;
    m.key = request;
    m.created = created;
    m.ttl = Duration::SecondsFromDouble(120.0);
    core::FeatureResponseValue f;
    f.request_id = request;
    for (size_t i = 0; i < keypoints; ++i) {
        f.keypoints.insert(f.keypoints.end(), {float(i), float(i * 7 % 640)});
        for (int b = 0; b < 32; ++b) f.descriptors.push_back(uint8_t((i * 131 + b * 17) & 0xFF));
        f.points3d.insert(f.points3d.end(), {float(i), 1.0f, 2.0f});
    }
    f.camera = {400, 400, 320, 240, 0.1f};
    m.encode = [=](uint64_t seq, Timestamp t) { return Codec().EncodeFeatureResponse(f, from, to, t, seq); };
    return m;
}

/// Two endpoints over one network, stepped on the sim clock. Heartbeats every
/// `hb_s` carry the pending acks both ways.
struct Pair {
    ac::ScriptedLinkModel links;
    ac::Network net;
    core::ByteLedger ledger;
    ac::Endpoint a, b;
    std::vector<core::DecodedMessage> at_a, at_b;
    double hb_s;
    double next_hb = 0.0;

    Pair(ac::LinkState s, uint64_t seed, double hb = 1.0, ac::EndpointParams p = {})
        : links(s), net(links, seed), a(0, Codec(), ledger, p), b(1, Codec(), ledger, p), hb_s(hb) {
        a.SetReachable(1, true);
        b.SetReachable(0, true);
    }

    void Heartbeat(ac::Endpoint& from, uint32_t to, Timestamp t) {
        core::HeartbeatValue h;
        h.agent = from.self();
        h.frame = {from.self(), 0};
        h.ack = from.TakeAck(to);
        from.SendImmediate(to, MessageClass::HEARTBEAT,
                           [&](uint64_t seq, Timestamp ts) { return Codec().EncodeHeartbeat(h, from.self(), to, ts, seq); },
                           t, net);
    }

    void RunUntil(double t_end, double dt = 0.01) {
        for (double t = 0.0; t <= t_end + 1e-9; t += dt) Step(At(t));
    }
    void Step(Timestamp t) {
        for (const auto& d : net.DeliverUntil(t)) {
            auto got = (d.to == 0 ? a : b).Receive(d, t);
            auto& sink = d.to == 0 ? at_a : at_b;
            for (auto& m : got) sink.push_back(std::move(m));
        }
        if (t.toSeconds() + 1e-9 >= next_hb) {
            Heartbeat(a, 1, t);
            Heartbeat(b, 0, t);
            next_hb += hb_s;
        }
        a.Pump(t, net);
        b.Pump(t, net);
    }
    size_t CountAtB(MessageClass c) const {
        size_t n = 0;
        for (const auto& m : at_b) n += m.cls == c;
        return n;
    }
};

/// Hand-built datagrams straight into an endpoint (no network), for delivery semantics.
ac::Datagram Dgram(const core::SerializedMessage& m, uint32_t msg_id, uint32_t from, uint32_t to) {
    auto frags = ac::Fragment(m.bytes, msg_id);
    EXPECT_EQ(frags.size(), 1u);
    ac::Datagram d;
    d.from = from;
    d.to = to;
    d.payload = frags[0];
    return d;
}

}  // namespace

// --- codec additions -----------------------------------------------------------

TEST(Transport, HeartbeatAckAndEstimateRoundTrip) {
    core::HeartbeatValue h;
    h.agent = 3;
    h.frame = {3, 0};
    h.keyframe_watermark = 42;
    h.neighbors = {1, 2};
    h.anchor = 1;
    h.broker = 1;
    h.epoch = {6, 0xABCDu, 1};
    h.ack.done = {5, 9};
    h.ack.partial = {{7, {0x05}}};
    core::DecodedMessage m;
    ASSERT_EQ(Codec().Decode(Codec().EncodeHeartbeat(h, 3, 1, At(1), 4).bytes, &m), core::WireCodec::Status::kOk);
    ASSERT_TRUE(m.has_heartbeat);
    EXPECT_EQ(m.heartbeat.neighbors, h.neighbors);
    EXPECT_EQ(m.heartbeat.epoch, h.epoch);
    EXPECT_EQ(m.heartbeat.ack.done, h.ack.done);
    ASSERT_EQ(m.heartbeat.ack.partial.size(), 1u);
    EXPECT_EQ(m.heartbeat.ack.partial[0].received, std::vector<uint8_t>{0x05});

    core::PoseGraphValue est = Graph(1, 3);
    est.has_epoch = true;
    est.epoch = {6, 0xABCDu, 1};
    est.agent_frames = {{1, {}}, {3, {1, 0, 0, 0, 1, 2, 3}}};
    ASSERT_EQ(Codec().Decode(Codec().EncodePoseGraph(est, 1, 3, At(1), 0).bytes, &m), core::WireCodec::Status::kOk);
    EXPECT_TRUE(m.pose_graph.has_epoch);
    EXPECT_EQ(m.pose_graph.epoch, est.epoch);
    ASSERT_EQ(m.pose_graph.agent_frames.size(), 2u);
    EXPECT_EQ(m.pose_graph.agent_frames[1].T_G_W.tz, 3.0f);

    // A Phase-1 graph (no epoch) encodes exactly as before the schema grew: no 3.2 field present.
    const core::WireCodec phase1;
    ASSERT_EQ(phase1.Decode(phase1.EncodePoseGraph(Graph(1, 3), 1, 0, At(1), 0).bytes, &m),
              core::WireCodec::Status::kOk);
    EXPECT_FALSE(m.pose_graph.has_epoch);
    EXPECT_TRUE(m.pose_graph.agent_frames.empty());
}

TEST(Transport, FragmentedChargeMatchesTheDatagramsSent) {
    const auto m = Codec().EncodePoseGraph(Graph(0, 400), 0, 1, At(0), 0);
    const auto frags = ac::Fragment(m.bytes, 1);
    ASSERT_GT(frags.size(), 10u);
    uint64_t wire = 0;
    for (const auto& f : frags) {
        EXPECT_LE(f.size(), core::kDatagramPayloadBytes);
        wire += f.size() + core::kDatagramOverheadBytes;
    }
    EXPECT_EQ(wire, m.record.wire_bytes);
}

// --- per-class delivery ----------------------------------------------------------

TEST(Transport, DescriptorHoleDoesNotStallLaterDescriptors) {
    core::ByteLedger ledger;
    ac::Endpoint rx(1, Codec(), ledger);
    size_t delivered = 0;
    for (uint64_t seq : {0u, 2u, 3u, 4u}) {   // seq 1 expired at the sender: never arrives
        const auto m = Codec().EncodeGlobalDescriptor(Descriptor(10 + seq), 0, 1, At(0), seq);
        delivered += rx.Receive(Dgram(m, uint32_t(seq + 1), 0, 1), At(0)).size();
    }
    EXPECT_EQ(delivered, 4u);
    // A duplicate (same keyframe, resent by the application) is dropped.
    const auto again = Codec().EncodeGlobalDescriptor(Descriptor(12), 0, 1, At(0), 9);
    EXPECT_TRUE(rx.Receive(Dgram(again, 99, 0, 1), At(0)).empty());
}

TEST(Transport, ReliableClassHoldsBackUntilTheGapFills) {
    core::ByteLedger ledger;
    ac::Endpoint rx(1, Codec(), ledger);
    auto edge = [&](uint64_t seq) {
        core::LoopEdgeValue e;
        e.keyframe_a = seq;
        e.frame_a = {0, 0};
        e.frame_b = {1, 0};
        return Codec().EncodeLoopEdge(e, 0, 1, At(0), seq);
    };
    EXPECT_EQ(rx.Receive(Dgram(edge(0), 1, 0, 1), At(0)).size(), 1u);
    EXPECT_TRUE(rx.Receive(Dgram(edge(2), 3, 0, 1), At(0)).empty());
    const auto got = rx.Receive(Dgram(edge(1), 2, 0, 1), At(0));
    ASSERT_EQ(got.size(), 2u);
    EXPECT_EQ(got[0].edge.keyframe_a, 1u);
    EXPECT_EQ(got[1].edge.keyframe_a, 2u);
}

TEST(Transport, LostHeartbeatDoesNotFreezeLaterOnes) {
    core::ByteLedger ledger;
    ac::Endpoint rx(1, Codec(), ledger);
    auto hb = [&](uint64_t seq) {
        core::HeartbeatValue h;
        h.agent = 0;
        h.frame = {0, 0};
        h.keyframe_watermark = seq;
        return Codec().EncodeHeartbeat(h, 0, 1, At(0), seq);
    };
    EXPECT_EQ(rx.Receive(Dgram(hb(0), 1, 0, 1), At(0)).size(), 1u);
    EXPECT_EQ(rx.Receive(Dgram(hb(2), 3, 0, 1), At(0)).size(), 1u);   // seq 1 lost
    EXPECT_TRUE(rx.Receive(Dgram(hb(1), 2, 0, 1), At(0)).empty());    // arrives late: stale
    EXPECT_EQ(rx.stats().stale_heartbeats, 1u);
}

// --- outbox ------------------------------------------------------------------------

TEST(Transport, HeldWhileUnreachableThenSent) {
    Pair p({true, 0.01, 0.0, 0.0}, 1);
    p.a.SetReachable(1, false);
    p.a.Enqueue(1, EdgeMsg(0, 1, 7, At(0)));
    p.RunUntil(3.0);
    EXPECT_EQ(p.CountAtB(MessageClass::POSE_EDGE), 0u);
    EXPECT_EQ(p.a.UnackedReliable(1), 1u);
    p.a.SetReachable(1, true);
    for (double t = 3.0; t < 6.0; t += 0.01) p.Step(At(t));
    EXPECT_EQ(p.CountAtB(MessageClass::POSE_EDGE), 1u);
    EXPECT_EQ(p.a.UnackedReliable(1), 0u);
}

TEST(Transport, DescriptorExpiresAtItsTtlWhileHeld) {
    Pair p({true, 0.01, 0.0, 0.0}, 1, 10.0);
    p.a.SetReachable(1, false);
    p.a.Enqueue(1, DescriptorMsg(0, 1, 5, At(0), Duration::SecondsFromDouble(120.0)));
    p.a.Enqueue(1, DescriptorMsg(0, 1, 6, At(100.0), Duration::SecondsFromDouble(120.0)));
    p.Step(At(121.0));
    p.a.SetReachable(1, true);
    p.Step(At(121.01));
    p.Step(At(122.0));
    ASSERT_EQ(p.CountAtB(MessageClass::GLOBAL_DESCRIPTOR), 1u);
    EXPECT_EQ(p.a.stats().expired, 1u);
}

TEST(Transport, CancelRemovesAQueuedFeatureResponse) {
    Pair p({true, 0.01, 0.0, 0.0}, 1);
    p.a.SetReachable(1, false);
    p.a.Enqueue(1, ResponseMsg(0, 1, 77, 50, At(0)));
    p.a.Cancel(1, MessageClass::MATCH_RESPONSE, 77);
    EXPECT_THROW(p.a.Cancel(1, MessageClass::POSE_EDGE, 1), std::invalid_argument);
    p.a.SetReachable(1, true);
    p.RunUntil(3.0);
    EXPECT_EQ(p.CountAtB(MessageClass::MATCH_RESPONSE), 0u);
    EXPECT_EQ(p.a.stats().cancelled, 1u);
}

TEST(Transport, SupersededSnapshotLeavesNoSequenceHole) {
    Pair p({true, 0.01, 0.0, 0.0}, 1);
    p.a.SetReachable(1, false);
    p.a.Enqueue(1, GraphMsg(0, 1, 20, 1, At(0)));
    p.a.Enqueue(1, GraphMsg(0, 1, 30, 2, At(0)));   // replaces the unsent first snapshot
    p.a.SetReachable(1, true);
    p.RunUntil(2.0);
    p.a.Enqueue(1, GraphMsg(0, 1, 40, 3, At(2.0)));
    for (double t = 2.0; t < 5.0; t += 0.01) p.Step(At(t));
    // Delivered in order with sequences 0, 1: the superseded one never consumed a number.
    ASSERT_EQ(p.CountAtB(MessageClass::POSE_GRAPH), 2u);
    EXPECT_EQ(p.a.stats().superseded, 1u);
    std::vector<size_t> sizes;
    for (const auto& m : p.at_b)
        if (m.cls == MessageClass::POSE_GRAPH) sizes.push_back(m.pose_graph.nodes.size());
    EXPECT_EQ(sizes, (std::vector<size_t>{30, 40}));
}

TEST(Transport, StrictPriorityUnderASendBudget) {
    ac::EndpointParams prm;
    prm.send_budget_Bps = 20000;
    prm.send_burst_bytes = 1500;   // one datagram per refill
    Pair p({true, 0.01, 0.0, 0.0}, 1, 100.0, prm);
    p.a.Enqueue(1, ResponseMsg(0, 1, 5, 400, At(0)));   // ~40 datagrams, class 4
    p.a.Enqueue(1, EdgeMsg(0, 1, 1, At(0)));             // class 2, enqueued after
    p.RunUntil(5.0);
    ASSERT_GE(p.at_b.size(), 2u);
    size_t edge_at = 0, resp_at = 0;
    for (size_t i = 0; i < p.at_b.size(); ++i) {
        if (p.at_b[i].cls == MessageClass::POSE_EDGE) edge_at = i;
        if (p.at_b[i].cls == MessageClass::MATCH_RESPONSE) resp_at = i;
    }
    EXPECT_LT(edge_at, resp_at);
}

// --- ARQ -----------------------------------------------------------------------------

TEST(Transport, SelectiveRetransmitDeliversEverythingAt30PercentLoss) {
    Pair p({true, 0.02, 0.3, 0.0}, 4);
    const int n_graphs = 10;
    uint64_t first_fragments = 0;
    for (int i = 0; i < n_graphs; ++i) {
        ac::OutMessage m = GraphMsg(0, 1, 560, uint32_t(100 + i), At(0));
        m.supersedes = false;   // ten distinct reliable messages, not snapshots
        first_fragments += core::FragmentCount(m.encode(0, At(0)).bytes.size());
        p.a.Enqueue(1, std::move(m));
    }
    ASSERT_GE(first_fragments / n_graphs, 15u);
    p.RunUntil(240.0);
    EXPECT_EQ(p.CountAtB(MessageClass::POSE_GRAPH), size_t(n_graphs));
    EXPECT_EQ(p.a.UnackedReliable(1), 0u);
    for (int i = 0; i < n_graphs; ++i) EXPECT_TRUE(p.a.Delivered(1, MessageClass::POSE_GRAPH).count(100 + i));
    // Selective: roughly loss/(1-loss) of the fragments are resent (plus ack losses), far from whole-message resends.
    const uint64_t resent = p.a.stats().retransmitted_fragments;
    EXPECT_GT(resent, 0u);
    EXPECT_LT(resent, first_fragments);
    uint64_t retransmit_rows = 0;
    for (const auto& r : p.ledger.Records()) retransmit_rows += r.retransmit;
    EXPECT_GT(retransmit_rows, 0u);
}

TEST(Transport, FeatureResponseRecoversItsMissingFragments) {
    Pair p({true, 0.02, 0.1, 0.0}, 9);
    p.a.Enqueue(1, ResponseMsg(0, 1, 5, 800, At(0)));   // ~ 39 datagrams
    p.RunUntil(60.0);
    EXPECT_EQ(p.CountAtB(MessageClass::MATCH_RESPONSE), 1u);
}

TEST(Transport, ElectionChangeAbandonsReliableMessagesToTheOldBroker) {
    Pair p({true, 0.01, 0.0, 0.0}, 1);
    p.links.Set(0, 1, {false, 0, 0, 0});   // the old broker became unreachable after we decided to send
    p.a.Enqueue(1, EdgeMsg(0, 1, 1, At(0)));
    p.a.Enqueue(1, GraphMsg(0, 1, 30, 1, At(0)));
    p.RunUntil(2.0);
    EXPECT_EQ(p.a.UnackedReliable(1), 2u);
    p.a.AbandonReliable(1);
    EXPECT_EQ(p.a.UnackedReliable(1), 0u);
    EXPECT_EQ(p.a.stats().abandoned_reliable, 2u);
}

TEST(Transport, SameSeedSameLedger) {
    auto run = [](uint64_t seed) {
        Pair p({true, 0.02, 0.3, 0.0}, seed);
        for (int i = 0; i < 3; ++i) p.a.Enqueue(1, GraphMsg(0, 1, 200, uint32_t(i), At(0)));
        p.a.Enqueue(1, ResponseMsg(0, 1, 5, 300, At(0)));
        p.RunUntil(60.0);
        std::vector<uint64_t> rows;
        for (const auto& r : p.ledger.Records()) rows.push_back(r.wire_bytes * 2 + r.retransmit);
        return rows;
    };
    EXPECT_EQ(run(3), run(3));
}
