// COLLAB_3.2 §6, §8: the I5 exit — every run ends consistent or explicitly
// incomplete, never corrupt; split-brain merges under a new epoch; frames.
#include "collab_scenario.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <set>

namespace ac = ariadne::collaboration;
namespace act = ariadne::collaboration::testing;
namespace core = ariadne::core;
using act::At;

namespace {

/// T_G_W of agent a against ground truth, G = W_anchor.
void ExpectFrameMatchesGroundTruth(const act::Swarm& s, size_t a, size_t anchor) {
    const auto T = s.nodes[a]->T_G_W();
    ASSERT_TRUE(T.has_value()) << "agent " << a << " was never placed";
    const core::SE3 gt = s.sc.T_World_W(anchor).inverse() * s.sc.T_World_W(a);
    EXPECT_LT((T->t - gt.t).norm(), 2e-3) << "agent " << a;
    EXPECT_LT((T->R.transpose() * gt.R - core::Mat3::Identity()).norm(), 2e-3) << "agent " << a;
}

std::set<ac::EdgeKey> FinalEdges(const act::Swarm& s) {
    const auto& a = s.nodes[0]->adopted();
    if (!a) return {};
    const auto& issued = s.nodes[a->epoch.broker]->issued();
    const auto it = issued.find(a->epoch);
    return it == issued.end() ? std::set<ac::EdgeKey>{} : it->second.edges;
}

void PrintIfNotConsistent(const ac::OutcomeReport& r, const std::string& label) {
    if (r.outcome == ac::Outcome::kConsistent) return;
    std::cout << "[" << label << "] " << ac::OutcomeName(r.outcome) << "\n";
    for (const auto& c : r.corrupt) std::cout << "  corrupt: " << c << "\n";
    size_t shown = 0, edges = 0;
    for (const auto& c : r.incomplete) {
        if (c.find("holds edge") != std::string::npos) {
            ++edges;
            continue;
        }
        if (shown++ < 6) std::cout << "  incomplete: " << c << "\n";
    }
    if (edges) std::cout << "  incomplete: " << edges << " held edges missing from the adopted solve\n";
}

}  // namespace

TEST(CollabV2, NoFaultsEndsConsistentWithGroundTruthFrames) {
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 1);
    s.RunUntil(0, 90);
    const auto r = s.Outcome();
    PrintIfNotConsistent(r, "no faults");
    ASSERT_EQ(r.outcome, ac::Outcome::kConsistent);
    const auto e = s.nodes[0]->adopted()->epoch;
    for (const auto& n : s.nodes) {
        EXPECT_EQ(n->adopted()->epoch, e);
        EXPECT_EQ(n->adopted()->members, (std::vector<uint32_t>{0, 1, 2, 3}));
        EXPECT_EQ(n->anchor(), 0u);
    }
    for (size_t a = 0; a < 4; ++a) ExpectFrameMatchesGroundTruth(s, a, 0);
    EXPECT_GT(s.nodes[0]->stats().edges_verified, 4u);
    EXPECT_EQ(s.nodes[0]->stats().frame_rejects, 0u);
}

TEST(CollabV2, SplitBrainMergesUnderANewEpoch) {
    act::Swarm nopart({}, {true, 0.02, 0.0, 0.0}, 2);
    nopart.RunUntil(0, 110);

    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 2);
    s.plan->Partition(At(5), At(40), {{0, 1}, {2, 3}});
    s.RunUntil(0, 39);
    // Two brokers, each solving its own side under its own epochs.
    ASSERT_TRUE(s.nodes[0]->adopted() && s.nodes[2]->adopted());
    const auto e01 = s.nodes[0]->adopted()->epoch, e23 = s.nodes[2]->adopted()->epoch;
    EXPECT_EQ(e01.broker, 0u);
    EXPECT_EQ(e23.broker, 2u);
    EXPECT_EQ(s.nodes[1]->adopted()->epoch, e01);
    EXPECT_EQ(s.nodes[3]->adopted()->epoch, e23);
    EXPECT_EQ(s.nodes[2]->adopted()->members, (std::vector<uint32_t>{2, 3}));
    EXPECT_EQ(s.nodes[3]->anchor(), 2u);   // side {2,3} anchored at its own lowest member

    s.RunUntil(39.05, 110);
    const auto r = s.Outcome();
    PrintIfNotConsistent(r, "split brain");
    ASSERT_EQ(r.outcome, ac::Outcome::kConsistent);
    const auto e = s.nodes[0]->adopted()->epoch;
    EXPECT_EQ(e.broker, 0u);
    EXPECT_GT(e.counter, std::max(e01.counter, e23.counter));
    for (const auto& n : s.nodes) {
        EXPECT_EQ(n->adopted()->epoch, e);
        EXPECT_EQ(n->anchor(), 0u);   // anchor propagation: the lower anchor won
    }
    for (size_t a = 0; a < 4; ++a) ExpectFrameMatchesGroundTruth(s, a, 0);
    // Graph contents equal the no-partition run's exactly.
    EXPECT_EQ(FinalEdges(s), FinalEdges(nopart));
    EXPECT_FALSE(FinalEdges(s).empty());
}

TEST(CollabV2, PoseInAnUnknownFrameIsRejectedNotComposed) {
    act::Scenario sc;
    sc.agents = 2;
    core::ByteLedger ledger;
    ac::ScriptedLinkModel links({true, 0.01, 0.0, 0.0});
    ac::Network net(links, 1);
    ac::CollabNode node(0, {0, 1}, act::FragCodec(), ledger, sc.Hooks(0));
    ac::Endpoint rogue(1, act::FragCodec(), ledger);
    rogue.SetReachable(0, true);
    for (const core::FrameTag bad : {core::FrameTag{7, 0}, core::FrameTag{1, 3}}) {   // unknown agent; unadopted G
        ac::OutMessage m;
        m.cls = core::MessageClass::POSE_EDGE;
        m.key = bad.agent * 10 + bad.epoch;
        core::LoopEdgeValue e;
        e.agent_a = 0;
        e.agent_b = bad.agent;
        e.frame_a = {0, 0};
        e.frame_b = bad;
        m.encode = [=](uint64_t seq, core::Timestamp t) { return act::FragCodec().EncodeLoopEdge(e, 1, 0, t, seq); };
        rogue.Enqueue(0, std::move(m));
    }
    for (double t = 0; t < 2; t += 0.05) {
        rogue.Pump(At(t), net);
        for (const auto& d : net.DeliverUntil(At(t)))
            if (d.to == 0) node.OnDatagram(d, At(t));
    }
    EXPECT_EQ(node.stats().frame_rejects, 2u);
    EXPECT_TRUE(node.edges().empty());
}

TEST(CollabV2, FaultCampaignNeverEndsCorrupt) {
    struct Fault {
        const char* name;
        std::function<void(act::Swarm&)> apply;
        ac::DatagramFaults df;
    };
    const std::vector<Fault> faults = {
        {"none", [](act::Swarm&) {}, {}},
        {"latency_jitter", [](act::Swarm& s) { s.plan->ExtraLatency(At(10), At(50), 0.2); }, {0, 0, 0.05, 0.05}},
        {"reorder_duplicate", [](act::Swarm&) {}, {0.2, 0.2, 0.05, 0.0}},
        {"partition", [](act::Swarm& s) { s.plan->Partition(At(10), At(40), {{0, 1}, {2, 3}}); }, {}},
        {"disconnect_agent", [](act::Swarm& s) { s.plan->Disconnect(2, At(15), At(45)); }, {}},
        {"broker_disconnect", [](act::Swarm& s) { s.plan->Disconnect(0, At(20), At(50)); }, {}},
    };
    int runs = 0, consistent = 0, incomplete = 0, corrupt = 0;
    for (const auto& f : faults)
        for (double loss : {0.0, 0.1, 0.3})
            for (uint64_t seed : {1u, 2u, 3u}) {
                act::Swarm s({}, {true, 0.02, loss, 0.0}, seed, f.df);
                f.apply(s);
                s.RunUntil(0, 150);
                const auto r = s.Outcome();
                ++runs;
                consistent += r.outcome == ac::Outcome::kConsistent;
                incomplete += r.outcome == ac::Outcome::kIncomplete;
                corrupt += r.outcome == ac::Outcome::kCorrupt;
                PrintIfNotConsistent(r, std::string(f.name) + " loss=" + std::to_string(loss) +
                                            " seed=" + std::to_string(seed));
                if (loss == 0.0 && std::string(f.name) == "none") EXPECT_EQ(r.outcome, ac::Outcome::kConsistent);
            }
    std::cout << "campaign: runs=" << runs << " consistent=" << consistent << " incomplete=" << incomplete
              << " corrupt=" << corrupt << "\n";
    EXPECT_EQ(corrupt, 0);
    EXPECT_EQ(runs, 54);
}

TEST(CollabV2, ClassifierFlagsARogueEstimateAsCorrupt) {
    act::Scenario sc;
    sc.agents = 2;
    act::Swarm s(sc, {true, 0.02, 0.0, 0.0}, 5);
    s.RunUntil(0, 100);   // keyframes stop at 60 s; 40 s drain
    ASSERT_EQ(s.Outcome().outcome, ac::Outcome::kConsistent);
    const auto good = *s.nodes[1]->adopted();
    // A forged estimate "from broker 0": a counter it never issued, a perturbed T_G_W.
    core::PoseGraphValue est;
    est.has_epoch = true;
    est.epoch = {good.epoch.counter + 1, ac::MembersHash({0, 1}), 0};
    est.anchor = {0, uint32_t(est.epoch.counter)};
    est.agent_frames = {{0, ac::ToPoseValue(core::SE3{})}, {1, ac::ToPoseValue(act::Pose(0.3, 5, 5))}};
    const uint64_t seq = uint64_t(1000) << ac::kIncarnationShift;   // a fresh incarnation: delivered at once
    const auto m = act::FragCodec().EncodePoseGraph(est, 0, 1, At(100), seq);
    for (auto& frag : ac::Fragment(m.bytes, 0xFFFFFF00u)) {
        ac::Datagram d;
        d.from = 0;
        d.to = 1;
        d.payload = frag;
        s.nodes[1]->OnDatagram(d, At(100));
    }
    ASSERT_EQ(s.nodes[1]->adopted()->epoch, est.epoch);   // it passed the acceptance rule: only the audit can see it
    const auto r = s.Outcome();
    EXPECT_EQ(r.outcome, ac::Outcome::kCorrupt);
    ASSERT_FALSE(r.corrupt.empty());
    EXPECT_NE(r.corrupt.front().find("no broker issued"), std::string::npos);
}

TEST(CollabV2, RunEndingPartitionedIsExplicitlyIncomplete) {
    act::Swarm s({}, {true, 0.02, 0.0, 0.0}, 6);
    s.plan->Partition(At(30), At(1e6), {{0, 1}, {2, 3}});
    s.RunUntil(0, 60);   // no drain: the sides never reconnect
    const auto r = s.Outcome();
    PrintIfNotConsistent(r, "ends partitioned");
    ASSERT_EQ(r.outcome, ac::Outcome::kIncomplete);
    bool names_members = false, names_unacked = false;
    for (const auto& c : r.incomplete) {
        names_members |= c.find("adopted a different epoch") != std::string::npos ||
                         c.find("current members differ") != std::string::npos;
        names_unacked |= c.find("unacked") != std::string::npos;
    }
    EXPECT_TRUE(names_members || names_unacked);
}

TEST(CollabV2, SameSeedSameLedgerAndOutcome) {
    auto run = [] {
        act::Swarm s({}, {true, 0.02, 0.1, 0.0}, 9, {0.1, 0.1, 0.05, 0.01});
        s.plan->Partition(At(10), At(30), {{0, 1}, {2, 3}});
        s.RunUntil(0, 80);
        std::vector<std::tuple<int64_t, uint32_t, uint32_t, int, uint64_t, bool>> rows;
        for (const auto& r : s.ledger.Records())
            rows.emplace_back(r.t.ns, r.sender, r.receiver, int(r.type), r.wire_bytes, r.retransmit);
        const auto o = s.Outcome();
        return std::make_tuple(rows, int(o.outcome), o.incomplete, o.corrupt);
    };
    EXPECT_EQ(run(), run());
}
