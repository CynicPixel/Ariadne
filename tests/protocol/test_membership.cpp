// collaboration.md §4: membership with hysteresis, Swarm-SLAM's election, epochs.
#include "collaboration/protocol/membership.hpp"
#include "collaboration/protocol/transport.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace ac = ariadne::collaboration;
namespace core = ariadne::core;
using core::MessageClass;
using core::Timestamp;

namespace {

Timestamp At(double s) { return Timestamp{int64_t(std::llround(s * 1e9))}; }

const core::WireCodec& Codec() {
    static const core::WireCodec c(core::Framing::kFragmented);
    return c;
}

/// N agents exchanging heartbeats over a network; membership updated every tick.
struct Fleet {
    const ac::LinkModel& links;
    ac::Network net;
    core::ByteLedger ledger;
    std::vector<std::unique_ptr<ac::Endpoint>> ep;
    std::vector<ac::Membership> mem;
    std::vector<int> broker_changes;
    double next_hb = 0.0;

    Fleet(const ac::LinkModel& l, size_t n, uint64_t seed) : links(l), net(l, seed) {
        for (size_t i = 0; i < n; ++i) {
            ep.push_back(std::make_unique<ac::Endpoint>(uint32_t(i), Codec(), ledger));
            mem.emplace_back(uint32_t(i));
            broker_changes.push_back(0);
        }
    }
    void Step(Timestamp t) {
        for (const auto& d : net.DeliverUntil(t))
            for (const auto& m : ep[d.to]->Receive(d, t))
                if (m.has_heartbeat) mem[d.to].OnHeartbeat(m.heartbeat, t);
        for (size_t i = 0; i < mem.size(); ++i) broker_changes[i] += mem[i].Update(t);
        if (t.toSeconds() + 1e-9 >= next_hb) {
            for (size_t i = 0; i < mem.size(); ++i)
                for (size_t j = 0; j < mem.size(); ++j) {
                    if (i == j) continue;
                    core::HeartbeatValue h;
                    h.agent = uint32_t(i);
                    h.frame = {uint32_t(i), 0};
                    h.neighbors = mem[i].Heard();
                    h.anchor = mem[i].anchor();
                    h.broker = mem[i].broker();
                    h.epoch = mem[i].accepted();
                    h.ack = ep[i]->TakeAck(uint32_t(j));
                    ep[i]->SendImmediate(uint32_t(j), MessageClass::HEARTBEAT,
                                         [&](uint64_t seq, Timestamp ts) {
                                             return Codec().EncodeHeartbeat(h, uint32_t(i), uint32_t(j), ts, seq);
                                         },
                                         t, net);
                }
            next_hb += 0.5;
        }
    }
    void RunUntil(double t0, double t1) {
        for (double t = t0; t <= t1 + 1e-9; t += 0.05) Step(At(t));
    }
};

}  // namespace

TEST(Membership, FullMeshElectsLowestAgent) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    Fleet f(links, 3, 1);
    f.RunUntil(0, 10);
    for (auto& m : f.mem) EXPECT_EQ(m.broker(), 0u);
    EXPECT_EQ(f.mem[0].Members(), (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(f.mem[1].Neighbors(), (std::vector<uint32_t>{0, 2}));
}

TEST(Membership, AsymmetricLinkIsNotANeighbourLink) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    links.Set(0, 1, {false, 0, 0, 0});   // 1 never hears 0; 0 hears 1
    Fleet f(links, 2, 1);
    f.RunUntil(0, 10);
    EXPECT_EQ(f.mem[0].Heard(), (std::vector<uint32_t>{1}));
    EXPECT_TRUE(f.mem[0].Neighbors().empty());
    EXPECT_TRUE(f.mem[1].Neighbors().empty());
    // Each elects itself; neither claims the other.
    EXPECT_EQ(f.mem[0].broker(), 0u);
    EXPECT_EQ(f.mem[1].broker(), 1u);
    EXPECT_EQ(f.mem[0].Members(), (std::vector<uint32_t>{0}));
    EXPECT_EQ(f.mem[1].Members(), (std::vector<uint32_t>{1}));
}

TEST(Membership, ChainElectsPerNeighbourhood) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    links.SetBoth(0, 2, {false, 0, 0, 0});
    Fleet f(links, 3, 1);
    f.RunUntil(0, 10);
    EXPECT_EQ(f.mem[1].broker(), 0u);
    EXPECT_EQ(f.mem[2].broker(), 1u);   // 1 is not a broker: 2 holds its data until it reaches one
    EXPECT_FALSE(f.mem[1].IsBroker());
    EXPECT_EQ(f.mem[0].Members(), (std::vector<uint32_t>{0, 1}));
}

TEST(Membership, AnchorWinsBeforeAgentId) {
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    Fleet f(links, 4, 1);
    // Agents 0 and 1 are out of range; 3 already merged into anchor 1's frame, 2 is fresh.
    links.SetBoth(0, 1, {false, 0, 0, 0});
    for (uint32_t a : {0u, 1u})
        for (uint32_t b : {2u, 3u}) links.SetBoth(a, b, {false, 0, 0, 0});
    f.mem[3].SetAnchor(1);
    f.RunUntil(0, 10);
    EXPECT_EQ(f.mem[2].broker(), 3u);   // (1, 3) < (2, 2)
    EXPECT_EQ(f.mem[3].broker(), 3u);
}

TEST(Membership, ThirtyPercentLossDoesNotFlapTheBroker) {
    ac::ScriptedLinkModel links({true, 0.02, 0.3, 0.0});
    Fleet f(links, 4, 7);
    f.RunUntil(0, 10);
    for (auto& m : f.mem) ASSERT_EQ(m.broker(), 0u);
    const auto before = f.broker_changes;
    f.RunUntil(10.05, 300);
    EXPECT_EQ(f.broker_changes, before);
    EXPECT_EQ(f.mem[0].Members(), (std::vector<uint32_t>{0, 1, 2, 3}));
}

TEST(Membership, LinkLossDropsTheNeighbourAfterKMissedPeriods) {
    ac::ScriptedLinkModel base({true, 0.01, 0.0, 0.0});
    ac::FaultPlan plan(base);
    plan.Disconnect(1, At(20), At(1000));
    Fleet f(plan, 2, 1);
    f.RunUntil(0, 22);
    EXPECT_TRUE(f.mem[0].IsNeighbor(1));   // not yet: hysteresis holds the link through short gaps
    f.RunUntil(22.05, 30);
    EXPECT_FALSE(f.mem[0].IsNeighbor(1));
    EXPECT_EQ(f.mem[1].broker(), 1u);
}

TEST(Membership, EpochCounterIsNewerThanEveryMemberAndAcceptanceRules) {
    ac::Membership m(0);
    core::HeartbeatValue h1;
    h1.agent = 1;
    h1.neighbors = {0};
    h1.epoch = {5, 11, 1};
    core::HeartbeatValue h2 = h1;
    h2.agent = 2;
    h2.epoch = {3, 12, 2};
    for (int k = 0; k < 3; ++k) {
        m.OnHeartbeat(h1, At(0.5 * k));
        m.OnHeartbeat(h2, At(0.5 * k));
        h1.broker = h2.broker = 0;
    }
    m.Update(At(1.2));
    const auto members = m.Members();
    EXPECT_EQ(members, (std::vector<uint32_t>{0, 1, 2}));
    const auto e = m.NextEpoch(members);
    EXPECT_EQ(e.counter, 6u);   // 1 + max(5, 3, mine)
    EXPECT_EQ(e.broker, 0u);
    EXPECT_EQ(e.members_hash, ac::MembersHash(members));

    // Acceptance at agent 1 whose broker is 0.
    ac::Membership a1(1);
    core::HeartbeatValue h0;
    h0.agent = 0;
    h0.neighbors = {1};
    for (int k = 0; k < 3; ++k) a1.OnHeartbeat(h0, At(0.5 * k));
    a1.Update(At(1.2));
    ASSERT_EQ(a1.broker(), 0u);
    using V = ac::Membership::Verdict;
    EXPECT_EQ(a1.Check(e, members, 0), V::kAccept);
    EXPECT_EQ(a1.Check(e, {0, 2}, 0), V::kNotMember);
    EXPECT_EQ(a1.Check({6, ac::MembersHash(members), 2}, members, 2), V::kNotMyBroker);
    a1.Accept(e);
    EXPECT_EQ(a1.Check(e, members, 0), V::kNotNewer);
}
