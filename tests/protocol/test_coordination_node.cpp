// D175: coordination messages travel through CollabNode over the simulated link: patches to a peer (acked, so the
// sender learns the receiver's version), bids to the component broker, the broker's consensus to every member; under
// a partition each component's own broker receives its members' bids.
#include "collab_scenario.hpp"

#include <gtest/gtest.h>

#include <map>
#include <vector>

namespace ac = ariadne::collaboration;
namespace act = ariadne::collaboration::testing;
namespace core = ariadne::core;
using act::At;

namespace {

struct Mailboxes {
    std::map<ac::AgentId, std::vector<core::DecodedMessage>> box;
    int next = 0;
    std::function<void(ac::NodeHooks&)> Customize() {
        return [this](ac::NodeHooks& h) {
            const ac::AgentId self = ac::AgentId(next++);
            h.coordination = [this, self](const core::DecodedMessage& m, core::Timestamp) { box[self].push_back(m); };
        };
    }
    size_t Count(ac::AgentId a, core::MessageClass c) const {
        const auto it = box.find(a);
        size_t n = 0;
        if (it != box.end())
            for (const auto& m : it->second) n += m.cls == c;
        return n;
    }
};

core::SubmapPatchValue Patch(int sender, uint64_t version) {
    core::SubmapPatchValue p;
    p.sender = sender, p.version = version, p.frame = {uint32_t(sender), 0};
    core::SlabBlockValue b;
    b.cells.assign(core::kSlabBlockCells, 3);
    p.blocks.push_back(b);
    return p;
}

core::TaskBidValue Bid(int sender, uint64_t seq) {
    core::TaskBidValue b;
    b.sender = sender, b.seq = seq;
    core::BidOptionValue o;
    o.task_key = 100 + sender;
    b.options.push_back(o);
    return b;
}

}  // namespace

TEST(CoordinationNode, PatchBidConsensusDeliveredAndPatchAcked) {
    Mailboxes mb;
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 3, {}, {}, mb.Customize());
    s.RunUntil(0, 5);   // membership forms: one component, broker 0
    ASSERT_EQ(s.nodes[2]->Component().broker, 0u);
    EXPECT_EQ(s.nodes[2]->Component().members.size(), 4u);

    s.nodes[1]->SendPatch(2, Patch(1, 5), At(5));
    s.nodes[2]->SendBid(Bid(2, 1), At(5));
    s.nodes[0]->SendBid(Bid(0, 1), At(5));   // the broker's own bid: delivered locally
    s.RunUntil(5, 8);
    EXPECT_EQ(mb.Count(2, core::MessageClass::SUBMAP_PATCH), 1u);
    EXPECT_EQ(s.nodes[1]->AckedPatchVersion(2), 5u);
    EXPECT_EQ(mb.Count(0, core::MessageClass::TASK_BID), 2u);

    core::TaskConsensusValue c;
    c.broker = 0, c.seq = 1, c.members = {0, 1, 2, 3};
    c.assignments = {{2, 102, 1.0, 2.0}};
    s.nodes[0]->SendConsensus(c, At(8));
    s.RunUntil(8, 11);
    for (ac::AgentId a = 0; a < 4; ++a) EXPECT_EQ(mb.Count(a, core::MessageClass::TASK_CONSENSUS), 1u) << a;
    const auto& got = mb.box[2].back();
    ASSERT_TRUE(got.has_task_consensus);
    EXPECT_EQ(got.task_consensus.assignments.at(0).task_key, 102);
}

TEST(CoordinationNode, PartitionRoutesBidsToEachComponentsBroker) {
    Mailboxes mb;
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 4, {}, {}, mb.Customize());
    s.plan->Partition(At(2), At(60), {{0, 1}, {2, 3}});
    s.RunUntil(0, 15);   // heartbeats time out across the cut: two components, brokers 0 and 2
    ASSERT_EQ(s.nodes[3]->Component().broker, 2u);
    ASSERT_EQ(s.nodes[1]->Component().broker, 0u);
    s.nodes[3]->SendBid(Bid(3, 1), At(15));
    s.nodes[1]->SendBid(Bid(1, 1), At(15));
    s.RunUntil(15, 18);
    EXPECT_EQ(mb.Count(2, core::MessageClass::TASK_BID), 1u);
    EXPECT_EQ(mb.Count(0, core::MessageClass::TASK_BID), 1u);
    EXPECT_EQ(mb.box[2].back().task_bid.sender, 3);
}
