// COLLAB_3.2 §2: the link seam and the discrete-event datagram network.
#include "collaboration/protocol/network.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <tuple>
#include <vector>

namespace ac = ariadne::collaboration;
using ariadne::core::Duration;
using ariadne::core::Timestamp;

namespace {

Timestamp At(double s) { return Timestamp{int64_t(std::llround(s * 1e9))}; }

std::vector<uint8_t> Bytes(size_t n, uint8_t v = 0) { return std::vector<uint8_t>(n, v); }

/// (from, to, arrival ns, first payload byte) per delivered datagram.
using Log = std::vector<std::tuple<uint32_t, uint32_t, int64_t, uint8_t>>;

Log RunLinks(const ac::LinkModel& links, uint64_t seed, ac::DatagramFaults f, int n) {
    ac::Network net(links, seed, f);
    for (int i = 0; i < n; ++i) {
        net.Send(0, 1, Bytes(100, uint8_t(i)), At(0.01 * i));
        net.Send(1, 2, Bytes(100, uint8_t(i)), At(0.01 * i));
    }
    Log log;
    for (const auto& d : net.DeliverUntil(At(1e6))) log.emplace_back(d.from, d.to, d.arrive.ns, d.payload[0]);
    return log;
}

}  // namespace

TEST(Network, SameSeedSameEventLog) {
    ac::ScriptedLinkModel links({true, 0.02, 0.3, 0.0});
    const ac::DatagramFaults f{0.1, 0.1, 0.05, 0.01};
    EXPECT_EQ(RunLinks(links, 7, f, 500), RunLinks(links, 7, f, 500));
    EXPECT_NE(RunLinks(links, 7, f, 500), RunLinks(links, 8, f, 500));
}

TEST(Network, PerDatagramLossMatchesProbability) {
    const double p = 0.3;
    const int n = 20000;
    ac::ScriptedLinkModel links({true, 0.0, p, 0.0});
    ac::Network net(links, 3);
    for (int i = 0; i < n; ++i) net.Send(0, 1, Bytes(10), At(0.0));
    const double lost = double(net.stats().lost_random) / n;
    const double sigma = std::sqrt(p * (1 - p) / n);
    EXPECT_NEAR(lost, p, 5 * sigma);
    EXPECT_EQ(net.DeliverUntil(At(1.0)).size(), size_t(n) - net.stats().lost_random);
}

TEST(Network, LossOnOneLinkIndependentOfOtherTraffic) {
    // Philox is keyed per directed link: adding traffic on 1->2 must not change what 0->1 loses.
    ac::ScriptedLinkModel links({true, 0.0, 0.5, 0.0});
    auto losses_0_1 = [&](bool other_traffic) {
        ac::Network net(links, 11);
        for (int i = 0; i < 200; ++i) {
            net.Send(0, 1, Bytes(4, uint8_t(i)), At(0.0));
            if (other_traffic) net.Send(1, 2, Bytes(4), At(0.0));
        }
        std::vector<uint8_t> got;
        for (const auto& d : net.DeliverUntil(At(1.0)))
            if (d.from == 0) got.push_back(d.payload[0]);
        return got;
    };
    EXPECT_EQ(losses_0_1(false), losses_0_1(true));
}

TEST(Network, PartitionCutsOnlyAcrossGroupsAndOnlyInWindow) {
    ac::ScriptedLinkModel base({true, 0.01, 0.0, 0.0});
    ac::FaultPlan plan(base);
    plan.Partition(At(1.0), At(2.0), {{0, 1}, {2, 3}});
    EXPECT_TRUE(plan.Sample(0, 1, At(1.5)).up);
    EXPECT_TRUE(plan.Sample(2, 3, At(1.5)).up);
    EXPECT_FALSE(plan.Sample(1, 2, At(1.5)).up);
    EXPECT_FALSE(plan.Sample(4, 0, At(1.5)).up);   // in no group: isolated
    EXPECT_TRUE(plan.Sample(1, 2, At(0.5)).up);
    EXPECT_TRUE(plan.Sample(1, 2, At(2.0)).up);    // window is [t0, t1)

    ac::Network net(plan, 1);
    net.Send(1, 2, Bytes(10), At(1.5));
    net.Send(1, 2, Bytes(10), At(2.5));
    EXPECT_EQ(net.DeliverUntil(At(10.0)).size(), 1u);
    EXPECT_EQ(net.stats().lost_link_down, 1u);
}

// D174 panel "cut comms": an open-ended partition from the press, closed at the release; a second release is a no-op.
TEST(Network, OpenPartitionLastsUntilEnded) {
    ac::ScriptedLinkModel base({true, 0.01, 0.0, 0.0});
    ac::FaultPlan plan(base);
    plan.PartitionOpen(At(1.0), {{0}, {1, 2}});
    EXPECT_TRUE(plan.Sample(0, 1, At(0.5)).up);
    EXPECT_FALSE(plan.Sample(0, 1, At(1.0)).up);
    EXPECT_FALSE(plan.Sample(0, 1, At(1e6)).up);   // until ended
    EXPECT_TRUE(plan.Sample(1, 2, At(5.0)).up);
    EXPECT_EQ(plan.EndOpenPartitions(At(7.0)), 1u);
    EXPECT_FALSE(plan.Sample(0, 1, At(6.9)).up);
    EXPECT_TRUE(plan.Sample(0, 1, At(7.0)).up);
    EXPECT_EQ(plan.EndOpenPartitions(At(9.0)), 0u);
    EXPECT_TRUE(plan.Sample(0, 1, At(8.0)).up);
}

TEST(Network, DisconnectAndExtraLatency) {
    ac::ScriptedLinkModel base({true, 0.01, 0.0, 0.0});
    ac::FaultPlan plan(base);
    plan.Disconnect(2, At(0.0), At(5.0));
    plan.ExtraLatency(At(5.0), At(6.0), 0.5);
    EXPECT_FALSE(plan.Sample(2, 0, At(1.0)).up);
    EXPECT_FALSE(plan.Sample(0, 2, At(1.0)).up);
    EXPECT_TRUE(plan.Sample(0, 1, At(1.0)).up);
    EXPECT_NEAR(plan.Sample(0, 2, At(5.5)).latency_s, 0.51, 1e-12);
}

TEST(Network, AsymmetricLinkIsDirected) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    links.Set(0, 1, {false, 0.0, 0.0, 0.0});
    EXPECT_FALSE(links.Sample(0, 1, At(0)).up);
    EXPECT_TRUE(links.Sample(1, 0, At(0)).up);
}

TEST(Network, BandwidthSerializesDatagramsOnALink) {
    // 1472 B payload + 28 B header = 1500 B per datagram; at 15 kB/s that is 0.1 s each.
    ac::ScriptedLinkModel links({true, 0.05, 0.0, 15000.0});
    ac::Network net(links, 1);
    for (int i = 0; i < 3; ++i) net.Send(0, 1, Bytes(1472, uint8_t(i)), At(0.0));
    net.Send(1, 0, Bytes(1472), At(0.0));   // the reverse direction has its own transmitter
    const auto got = net.DeliverUntil(At(10.0));
    ASSERT_EQ(got.size(), 4u);
    std::vector<int64_t> fwd;
    for (const auto& d : got)
        if (d.from == 0) fwd.push_back(d.arrive.ns);
    ASSERT_EQ(fwd.size(), 3u);
    EXPECT_EQ(fwd[0], At(0.15).ns);
    EXPECT_EQ(fwd[1], At(0.25).ns);
    EXPECT_EQ(fwd[2], At(0.35).ns);
    for (const auto& d : got)
        if (d.from == 1) EXPECT_EQ(d.arrive.ns, At(0.15).ns);
}

TEST(Network, ReorderAndDuplicate) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    ac::Network net(links, 5, {0.2, 0.2, 0.05, 0.0});
    for (int i = 0; i < 1000; ++i) net.Send(0, 1, Bytes(4, uint8_t(i % 256)), At(0.001 * i));
    const auto got = net.DeliverUntil(At(100.0));
    EXPECT_GT(net.stats().duplicated, 100u);
    EXPECT_GT(net.stats().reordered, 100u);
    EXPECT_EQ(got.size(), 1000u + net.stats().duplicated);
    bool out_of_order = false;
    for (size_t i = 1; i < got.size(); ++i) out_of_order |= got[i].sent < got[i - 1].sent;
    EXPECT_TRUE(out_of_order);
    for (size_t i = 1; i < got.size(); ++i) EXPECT_LE(got[i - 1].arrive, got[i].arrive);
}

TEST(Network, OversizePayloadIsRejected) {
    ac::ScriptedLinkModel links;
    ac::Network net(links, 1);
    EXPECT_THROW(net.Send(0, 1, Bytes(1473), At(0)), std::invalid_argument);
}
