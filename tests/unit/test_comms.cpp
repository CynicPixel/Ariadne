// 3.3 (D102): the RFComms-derived link model, the wall ray-cast, the scenario configs, and the comms node's
// ledger rule.
#include "collaboration/comms/comms_node.hpp"
#include "collaboration/protocol/transport.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <random>

namespace ac = ariadne::collaboration;
namespace core = ariadne::core;
using core::Timestamp;

namespace {

Timestamp At(double s) { return Timestamp{int64_t(std::llround(s * 1e9))}; }

ac::AgentTrack Still(core::Vec3 p) { return {{0, int64_t(1e12)}, {p, p}}; }

/// An axis-aligned quad as two triangles, normal into free space.
void Quad(std::vector<ac::WallTriangle>& w, core::Vec3 a, core::Vec3 b, core::Vec3 c, core::Vec3 d, core::Vec3 n,
          uint64_t id) {
    w.push_back({a, b, c, n, id});
    w.push_back({a, c, d, n, id});
}

/// Two rooms split by a 0.2 m solid wall block at x in [5, 5.2], with a doorway at y in [2, 3]: the block's west
/// face (normal -x) and east face (normal +x), each in two pieces around the door.
std::vector<ac::WallTriangle> TwoRooms() {
    std::vector<ac::WallTriangle> w;
    for (double x : {5.0, 5.2}) {
        const core::Vec3 n(x == 5.0 ? -1 : 1, 0, 0);
        const uint64_t base = x == 5.0 ? 10 : 20;
        Quad(w, {x, 0, 0}, {x, 2, 0}, {x, 2, 3}, {x, 0, 3}, n, base);       // y in [0, 2]
        Quad(w, {x, 3, 0}, {x, 6, 0}, {x, 6, 3}, {x, 3, 3}, n, base + 1);   // y in [3, 6]
    }
    return w;
}

}  // namespace

// RFComms decides each packet by drawing P_rx ~ N(mean, sigma^2) then a Bernoulli(PER). Our loss_p must be that
// marginal, across the transition band where PER is nearly a step.
TEST(Comms, LossMatchesRfCommsPerPacketMonteCarlo) {
    const double noise = -90.0, sigma = 10.0, bytes = 1500.0;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    ac::RfParams p;
    p.sigma_db = sigma;
    const ac::RfLinkModel model(p, {}, {});
    for (double margin : {-10.0, 0.0, 8.0, 14.0, 20.0, 30.0}) {
        const double mean = noise + margin;
        const int n = 200000;
        int dropped = 0;
        for (int i = 0; i < n; ++i) dropped += u(rng) < ac::rf::Per(mean + sigma * g(rng), noise, bytes);
        const double mc = double(dropped) / n, se = std::sqrt(std::max(mc * (1 - mc), 1e-6) / n);
        const double ours = ac::rf::ExpectedPer(mean, sigma, noise, bytes);
        EXPECT_NEAR(ours, mc, 4.0 * se + 1e-4) << "margin " << margin;
        EXPECT_NEAR(model.LossAt(mean), ours, 2e-3) << "table, margin " << margin;
    }
}

TEST(Comms, LossIsPricedAtTheDatagramsOwnSize) {
    // D171: Network asks SampleSized(.., wire_bytes); RFComms as modelled prices PER over the datagram's bytes, so a
    // heartbeat near the edge drops less often than a full fragment. The unsized Sample keeps the 1500 B table.
    ac::RfParams p;
    p.sigma_db = 10.0;
    p.tx_power_dbm = 0.0;
    p.fading_exponent = 3.0;   // ~16 m: in the transition band
    const ac::RfLinkModel m(p, {{0, Still({0, 0, 1})}, {1, Still({16, 0, 1})}}, {});
    const auto full = m.Sample(0, 1, At(1)), at1500 = m.SampleSized(0, 1, At(1), 1500.0),
               small = m.SampleSized(0, 1, At(1), 68.0);
    const double rx = m.Detail(0, 1, At(1)).rx_mean_dbm;
    ASSERT_TRUE(full.up);
    EXPECT_EQ(at1500.loss_p, full.loss_p) << "the MTU size reads the existing table";
    EXPECT_GT(full.loss_p, 0.05);
    EXPECT_LT(small.loss_p, full.loss_p);
    EXPECT_NEAR(small.loss_p, ac::rf::ExpectedPer(rx, p.sigma_db, p.noise_floor_dbm, 68.0), 2e-3)
        << "grid interpolation at a new size";
    EXPECT_EQ(m.SampleSized(0, 1, At(1), 68.0).loss_p, small.loss_p) << "cached cells are deterministic";

    // The fault plan forwards the size, and the network uses it: 68 B datagrams are lost less than 1472 B ones.
    ac::FaultPlan plan(m);
    EXPECT_EQ(plan.SampleSized(0, 1, At(1), 68.0).loss_p, small.loss_p);
    ac::Network small_net(plan, 9), big_net(plan, 9);
    const int n = 4000;
    for (int i = 0; i < n; ++i) {
        small_net.Send(0, 1, std::vector<uint8_t>(40, 1), At(1));
        big_net.Send(0, 1, std::vector<uint8_t>(core::kDatagramPayloadBytes, 1), At(1));
    }
    const double ls = double(small_net.stats().lost_random) / n, lb = double(big_net.stats().lost_random) / n;
    EXPECT_NEAR(ls, small.loss_p, 4.0 * std::sqrt(small.loss_p * (1 - small.loss_p) / n) + 1e-3);
    EXPECT_NEAR(lb, full.loss_p, 4.0 * std::sqrt(full.loss_p * (1 - full.loss_p) / n) + 1e-3);
    EXPECT_LT(ls, lb);
}

TEST(Comms, PathLossRangeCutAndRate) {
    ac::RfParams p;   // gz-sim8 defaults
    p.sigma_db = 0.0;
    p.wall_loss_db = 6.9;
    const ac::RfLinkModel m(p, {{0, Still({0, 0, 1})}, {1, Still({10, 0, 1})}, {2, Still({60, 0, 1})}}, {});
    const ac::RfLinkDetail d = m.Detail(0, 1, At(0));
    EXPECT_NEAR(d.distance_m, 10.0, 1e-12);
    EXPECT_NEAR(d.rx_mean_dbm, 27.0 - (40.0 + 25.0), 1e-9);   // tx - (l0 + 10 n log10 d)
    EXPECT_TRUE(d.state.up);
    EXPECT_LT(d.state.loss_p, 1e-9);                            // ~52 dB above the noise floor
    EXPECT_DOUBLE_EQ(d.state.bandwidth_Bps, 54e6 / 8.0);
    EXPECT_FALSE(m.Sample(0, 2, At(0)).up);                     // 60 m > max_range 50 m
    EXPECT_FALSE(m.Sample(0, 7, At(0)).up);                     // unknown agent: no radio
}

TEST(Comms, WallsCountBlocksNotFacesAndDoorwaysAreFree) {
    const auto w = TwoRooms();
    using ac::rf::WallsCrossed;
    EXPECT_EQ(WallsCrossed({1, 1, 1}, {9, 1, 1}, w), 1);     // through the block: two faces, one block
    EXPECT_EQ(WallsCrossed({9, 1, 1}, {1, 1, 1}, w), 1);     // either direction
    EXPECT_EQ(WallsCrossed({1, 2.5, 1}, {9, 2.5, 1}, w), 0); // straight through the doorway
    EXPECT_EQ(WallsCrossed({1, 1, 1}, {4, 5, 2}, w), 0);     // same room
    // Through a quad's diagonal (both triangles touch the hit point's neighbourhood): the surface counts once.
    EXPECT_EQ(WallsCrossed({1, 0.0, 0.0}, {9, 2.0, 3.0}, w), 1);
    // With walls, the attenuation lands in the received power.
    ac::RfParams p;
    p.sigma_db = 0.0;
    p.wall_loss_db = 6.9;
    const ac::RfLinkModel m(p, {{0, Still({1, 1, 1})}, {1, Still({9, 1, 1})}, {2, Still({9, 2.5, 1})},
                                {3, Still({1, 2.5, 1})}}, w);
    EXPECT_EQ(m.Detail(0, 1, At(0)).walls, 1);
    EXPECT_EQ(m.Detail(3, 2, At(0)).walls, 0);   // the same 8 m, through the doorway
    EXPECT_NEAR(m.Detail(3, 2, At(0)).rx_mean_dbm - m.Detail(0, 1, At(0)).rx_mean_dbm, 6.9, 1e-9);
}

TEST(Comms, TracksInterpolateAndClamp) {
    ac::AgentTrack t{{1000000000, 3000000000}, {{0, 0, 0}, {2, 0, 0}}};
    EXPECT_NEAR(t.At(2000000000).x(), 1.0, 1e-12);
    EXPECT_NEAR(t.At(0).x(), 0.0, 1e-12);
    EXPECT_NEAR(t.At(9000000000).x(), 2.0, 1e-12);
}

TEST(Comms, EveryShippedScenarioLoadsAndBuilds) {
    int n = 0;
    for (const auto& e : std::filesystem::directory_iterator(std::string(ARIADNE_SOURCE_DIR) + "/configs/comms")) {
        const ac::CommsScenario s = ac::LoadCommsScenario(e.path().string());
        const auto base = ac::MakeLinkModel(s, {{0, Still({0, 0, 0})}, {1, Still({3, 0, 0})}}, TwoRooms());
        ac::CommsNode node(*base, 1);
        node.Apply(s);
        EXPECT_TRUE(node.links().Sample(0, 1, At(0)).up) << e.path();
        ++n;
    }
    EXPECT_GE(n, 4);
}

// The ledger rule (COLLAB_3.2 §2): one ledger, written only by Endpoints. Over a lossy RF link with a scripted
// partition, every non-retransmit row is an Endpoint first transmission or immediate send, and draining the
// network -- forwarding and dropping everything in flight -- adds no row.
TEST(Comms, OneLedgerAndForwardedDatagramsAreNeverRecorded) {
    static const core::WireCodec codec(core::Framing::kFragmented);
    ac::RfParams p;
    p.sigma_db = 10.0;
    p.tx_power_dbm = 0.0;
    p.fading_exponent = 3.0;   // ~16 m: in the transition band, so fragments are lost and retransmitted
    const ac::RfLinkModel rf(p, {{0, Still({0, 0, 1})}, {1, Still({16, 0, 1})}}, {});
    ac::CommsNode node(rf, 3);
    ac::CommsScenario faults;
    faults.partitions.push_back({10.0, 20.0, {{0}, {1}}});
    node.Apply(faults);

    ac::Endpoint a(0, codec, node.ledger()), b(1, codec, node.ledger());
    a.SetReachable(1, true);
    b.SetReachable(0, true);
    for (uint32_t k = 0; k < 5; ++k) {
        core::PoseGraphValue g;
        g.anchor = {0, 0};
        for (uint64_t i = 0; i < 80; ++i) {
            core::PoseNodeValue v;
            v.agent = 0;
            v.keyframe = i;
            v.stamp_ns = int64_t(i * 1000003 + k);
            v.pose = {float(i), float(k), 1, 0, 0, 0, 1};
            v.frame = {0, 0};
            g.nodes.push_back(v);
        }
        ac::OutMessage m;
        m.cls = core::MessageClass::POSE_GRAPH;
        m.key = k;
        m.created = At(0);
        m.encode = [=](uint64_t seq, Timestamp t) { return codec.EncodePoseGraph(g, 0, 1, t, seq); };
        a.Enqueue(1, std::move(m));
    }
    uint64_t immediate = 0;
    for (double t = 0.0; t <= 60.0 + 1e-9; t += 0.01) {
        const Timestamp ts = At(t);
        for (const auto& d : node.network().DeliverUntil(ts)) (d.to == 0 ? a : b).Receive(d, ts);
        if (std::fmod(t + 1e-9, 1.0) < 0.01) {   // 1 Hz heartbeats both ways, carrying the acks
            for (auto* e : {&a, &b}) {
                const uint32_t to = e->self() == 0 ? 1 : 0;
                core::HeartbeatValue h;
                h.agent = e->self();
                h.frame = {e->self(), 0};
                h.ack = e->TakeAck(to);
                e->SendImmediate(to, core::MessageClass::HEARTBEAT,
                                 [&](uint64_t seq, Timestamp tt) { return codec.EncodeHeartbeat(h, e->self(), to, tt, seq); },
                                 ts, node.network());
                ++immediate;
            }
        }
        a.Pump(ts, node.network());
        b.Pump(ts, node.network());
    }
    const auto& st = node.network().stats();
    EXPECT_GT(st.lost_random, 0u);      // the RF link dropped fragments
    EXPECT_GT(st.lost_link_down, 0u);   // the partition cut the link
    uint64_t first = 0, resend = 0;
    for (const auto& r : node.ledger().Records()) (r.retransmit ? resend : first) += 1;
    EXPECT_GT(resend, 0u);
    EXPECT_EQ(first, a.stats().first_tx_messages + b.stats().first_tx_messages + immediate + a.stats().acks_sent +
                         b.stats().acks_sent);
    const size_t rows = node.ledger().Records().size();
    node.network().DeliverUntil(At(1e6));   // forward / drop everything still in flight
    EXPECT_EQ(node.ledger().Records().size(), rows);
}
