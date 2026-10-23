// D175 per-agent planning (planning/mission/multiagent.hpp): the exchanged cell code; the exporter sends only blocks
// changed since the receiver's ack; received layers place patches at T_G_W(j), refuse an agent's own patch, and merge
// conservatively; the broker equals the centralised SGA on the same instance under full delivery, is exactly-once by
// r_same, and ignores bids from another epoch or a non-member; a member's pick follows the consensus it received, or
// falls back "uncoordinated" for each stated reason; PlanOwn without received data equals Plan's own-agent output; a
// peer-known column is no gain.
#include "planning/mission/multiagent.hpp"
#include "planning/mission/phase4_planner.hpp"
#include "planning/volumetric/occupancy_grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ariadne;
using namespace ariadne::planning;
using namespace ariadne::planning::multi;

namespace {

GridFrame Grid(int rows = 64, int cols = 96) {
    GridFrame g;
    g.ox = 0, g.oy = 0, g.cell = 0.05, g.rows = rows, g.cols = cols;
    return g;
}

TraversabilityMaps Maps(const GridFrame& g) {
    TraversabilityMaps m;
    m.slab = cv::Mat(g.rows, g.cols, CV_8U, cv::Scalar(kUnknown));
    m.trav_r = cv::Mat::zeros(g.rows, g.cols, CV_8U);
    return m;
}

TaskCandidate Opt(int64_t id, double x, double y, double cov, double path, double util) {
    TaskCandidate c;
    c.task_id = id, c.goal_xy = {x, y}, c.utility = util, c.path_m = path;
    c.ig_raw = {cov, 0.5 * cov, 0, 0, 0};
    return c;
}

DepthFrame WallFrame(std::vector<float>* Z, std::vector<float>* S, double y) {
    const int W = 160, H = 90;
    Z->assign(size_t(W * H), 2.0f), S->assign(size_t(W * H), 0.02f);
    DepthFrame f;
    f.T_W_C.R << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    f.T_W_C.t = core::Vec3(0.5, y, 1.0);
    f.fx = 84, f.fy = 84, f.cx = 80, f.cy = 45, f.width = W, f.height = H, f.depth = Z->data(), f.sigma = S->data();
    return f;
}

Phase4Briefing Brief() {
    Phase4Briefing b;
    b.grid.ox = 0, b.grid.oy = 0, b.grid.cell = 0.05, b.grid.cols = 80, b.grid.rows = 80;
    b.target = {40, 70};
    b.entrances = {{40, 10}};
    return b;
}

}  // namespace

TEST(MultiAgentPlanning, EncodeCellsTriStatePlusResponderBit) {
    const GridFrame g = Grid(4, 4);
    TraversabilityMaps m = Maps(g);
    m.slab.at<uint8_t>(0, 0) = kOccupied;
    m.slab.at<uint8_t>(0, 1) = kFree;
    m.slab.at<uint8_t>(0, 2) = kFree, m.trav_r.at<uint8_t>(0, 2) = 1;
    const cv::Mat c = EncodeCells(m);
    EXPECT_EQ(c.at<uint8_t>(0, 0), kCellOccupied);
    EXPECT_EQ(c.at<uint8_t>(0, 1), kCellFree);
    EXPECT_EQ(c.at<uint8_t>(0, 2), kCellFreeResponder);
    EXPECT_EQ(c.at<uint8_t>(0, 3), kCellUnknown);
}

TEST(MultiAgentPlanning, ExporterSendsOnlyBlocksChangedSinceTheAck) {
    const GridFrame g = Grid();   // 96 x 64 cells = 3 x 2 blocks
    SlabExporter ex(1, g);
    TraversabilityMaps m = Maps(g);
    EXPECT_TRUE(ex.Export(0, 0, 0).blocks.empty()) << "nothing observed yet";
    m.slab.at<uint8_t>(5, 5) = kFree;         // block (0, 0)
    m.slab.at<uint8_t>(40, 70) = kOccupied;   // block (2, 1)
    const uint64_t v1 = ex.Update(m);
    EXPECT_EQ(v1, 1u);
    const SubmapPatch all = ex.Export(0, 3, 5000);
    ASSERT_EQ(all.blocks.size(), 2u);
    EXPECT_EQ(all.sender, 1);
    EXPECT_EQ(all.epoch, 3u);
    EXPECT_EQ(all.version, 1u);
    EXPECT_EQ(ex.Update(m), 1u) << "no change, no new version";
    m.slab.at<uint8_t>(6, 5) = kFree;         // block (0, 0) again
    EXPECT_EQ(ex.Update(m), 2u);
    const SubmapPatch inc = ex.Export(v1, 3, 10000);
    ASSERT_EQ(inc.blocks.size(), 1u) << "only the block changed after the acked version";
    EXPECT_EQ(inc.blocks[0].bx, 0);
    EXPECT_EQ(inc.blocks[0].by, 0);
    EXPECT_EQ(inc.blocks[0].cells[size_t(6 * kBlock + 5)], kCellFree);
    EXPECT_TRUE(ex.Export(2, 3, 10000).blocks.empty());
}

TEST(MultiAgentPlanning, ReceivedLayersPlaceMergeAndRefuseOwn) {
    const GridFrame g = Grid();
    SlabExporter e1(1, g), e2(2, g);
    TraversabilityMaps m1 = Maps(g), m2 = Maps(g);
    m1.slab.at<uint8_t>(3, 3) = kFree, m1.trav_r.at<uint8_t>(3, 3) = 1;
    m1.slab.at<uint8_t>(4, 4) = kFree;
    m2.slab.at<uint8_t>(4, 4) = kOccupied;   // the peers disagree: the merge is conservative
    e1.Update(m1), e2.Update(m2);
    ReceivedLayers R(0, 3, g);
    R.Integrate(e1.Export(0, 0, 0));
    R.Integrate(e2.Export(0, 0, 0));
    EXPECT_EQ(R.AckedVersion(1), 1u);
    EXPECT_NE(R.KnownMask().at<uint8_t>(3, 3), 0);
    EXPECT_EQ(R.KnownMask().at<uint8_t>(10, 10), 0);
    const cv::Mat M = R.Merged();
    EXPECT_EQ(M.at<uint8_t>(3, 3), kCellFreeResponder);
    EXPECT_EQ(M.at<uint8_t>(4, 4), kCellOccupied) << "occupied wins";
    EXPECT_THROW(R.Integrate(SlabExporter(0, g).Export(0, 0, 0)), std::invalid_argument);
    ReceivedLayers S(0, 3, g);   // T_G_W(1) = one block (1.6 m) along +x: the patch lands one block over
    S.Integrate(e1.Export(0, 0, 0), Rigid2{1.6, 0, 0});
    EXPECT_EQ(S.KnownMask().at<uint8_t>(3, 3), 0);
    EXPECT_NE(S.KnownMask().at<uint8_t>(3, 3 + kBlock), 0);
}

TEST(MultiAgentPlanning, BrokerEqualsCentralisedSgaUnderFullDelivery) {
    // Two agents, every option bid (M = all), distinct tasks farther apart than r_same: the broker's instance is
    // BuildAuction's (same scales, same values), so its sequential greedy picks what the centralised SGA picks.
    Phase4Plan p;
    p.agents.resize(2);
    p.agents[0].nbv.options = {Opt(11, 1.0, 1.0, 50, 2, 0.9), Opt(12, 4.0, 1.0, 40, 3, 0.5), Opt(13, 7.0, 1.0, 5, 1, 0.1)};
    p.agents[1].nbv.options = {Opt(11, 1.0, 1.0, 60, 4, 0.8), Opt(14, 4.0, 4.0, 45, 1, 0.7), Opt(12, 4.0, 1.0, 30, 2, 0.2)};
    for (auto& a : p.agents) a.nbv.lambda_path = 0.5;
    std::vector<float> sc;
    std::vector<std::vector<int>> view;
    GridFrame g;
    g.cell = 0.05;
    const AuctionInstance in = BuildAuction(p, g, 1, &sc, &view);
    const Allocation central = SequentialGreedy(in);
    std::vector<TaskBid> bids;
    for (int a = 0; a < 2; ++a) bids.push_back(BuildBid(a, 7, 1, 5000000000, p.agents[size_t(a)].nbv, 100));
    const TaskConsensus c = BrokerAssign(bids, ComponentInfo{0, {0, 1}, 7}, 1.0, 5000000000, 1);
    ASSERT_EQ(c.assignments.size(), 2u);
    for (const Assignment& x : c.assignments) {
        const int j = central.agents[size_t(x.agent)].path.front();
        EXPECT_EQ(x.task_key, in.task_ids[size_t(j)]) << "agent " << x.agent;
    }
    EXPECT_EQ(c.bids_t_ns, 5000000000);
}

TEST(MultiAgentPlanning, BrokerIsExactlyOnceByRSameAndIgnoresStrangers) {
    // Both agents' best goals are 0.3 m apart: one task. A bid from another epoch and one from a non-member are ignored.
    NbvPlan n0, n1, n2, n9;
    n0.options = {Opt(101, 2.0, 2.0, 50, 1, 0.9), Opt(102, 6.0, 2.0, 10, 1, 0.2)};
    n1.options = {Opt(201, 2.3, 2.0, 50, 1, 0.9), Opt(202, 2.0, 6.0, 10, 1, 0.2)};
    n2.options = {Opt(301, 9.0, 9.0, 99, 1, 0.9)};
    n9.options = {Opt(901, 9.0, 9.0, 99, 1, 0.9)};
    std::vector<TaskBid> bids = {BuildBid(0, 4, 1, 10, n0, 8), BuildBid(1, 4, 1, 10, n1, 8), BuildBid(2, 3, 1, 10, n2, 8),
                                 BuildBid(9, 4, 1, 10, n9, 8)};
    const TaskConsensus c = BrokerAssign(bids, ComponentInfo{0, {0, 1, 2}, 4}, 1.0, 20, 2);
    std::vector<std::pair<double, double>> goals;
    for (const Assignment& a : c.assignments) {
        EXPECT_NE(a.agent, 9) << "not a member";
        EXPECT_NE(a.agent, 2) << "its bid is from another epoch";
        for (const auto& [x, y] : goals) EXPECT_GE(std::hypot(a.goal_x - x, a.goal_y - y), 1.0) << "exactly-once by r_same";
        goals.push_back({a.goal_x, a.goal_y});
    }
    EXPECT_EQ(c.assignments.size(), 2u);
}

TEST(MultiAgentPlanning, ApplyConsensusFollowsWhatArrivedOrSaysWhyNot) {
    const std::vector<TaskCandidate> o = {Opt(5, 1, 1, 9, 1, 0.9), Opt(6, 3, 3, 5, 1, 0.5), Opt(8, 6, 6, 1, 1, 0.1)};
    TaskConsensus c;
    c.epoch = 2, c.t_ns = 5000000000, c.assignments = {{1, 6, 3, 3}};
    Pick p = ApplyConsensus(&c, 1, 2, 10000000000, 12.0, o, 1.0);
    EXPECT_TRUE(p.coordinated);
    EXPECT_EQ(p.option, 1);
    c.assignments = {{1, 77, 6.2, 6.1}};   // re-keyed since the bid: the option within r_same of the goal
    p = ApplyConsensus(&c, 1, 2, 10000000000, 12.0, o, 1.0);
    EXPECT_TRUE(p.coordinated);
    EXPECT_EQ(p.option, 2);
    const struct { const TaskConsensus* c; int self; uint32_t epoch; int64_t now; const char* why; } cases[] = {
        {nullptr, 1, 2, 10000000000, "uncoordinated: no consensus received"},
        {&c, 1, 3, 10000000000, "uncoordinated: consensus from another epoch"},
        {&c, 1, 2, 30000000000, "uncoordinated: consensus older than the timeout"},
        {&c, 0, 2, 10000000000, "uncoordinated: no assignment for this agent"},
    };
    for (const auto& k : cases) {
        p = ApplyConsensus(k.c, k.self, k.epoch, k.now, 12.0, o, 1.0);
        EXPECT_FALSE(p.coordinated);
        EXPECT_EQ(p.option, 0) << "its own best";
        EXPECT_EQ(p.why, k.why);
    }
    c.assignments = {{1, 77, 20, 20}};
    EXPECT_EQ(ApplyConsensus(&c, 1, 2, 10000000000, 12.0, o, 1.0).why, "uncoordinated: the assigned task is gone");
}

TEST(MultiAgentPlanning, PlanOwnWithoutReceivedDataEqualsPlansOwnAgent) {
    // The refactor guard: an agent's own planner with nothing received plans exactly what the centralised planner
    // plans for that agent when it is the only active one.
    std::vector<float> Z, S;
    const DepthFrame f = WallFrame(&Z, &S, 2.0);
    Phase4Params prm;
    Phase4Planner A(Brief(), prm, 2), B(Brief(), prm, 2);
    A.Integrate(0, 0, f), B.Integrate(0, 0, f);
    std::vector<Phase4AgentInput> ag(2);
    ag[0].active = true, ag[0].position = f.T_W_C.t;
    const Phase4Plan pc = A.Plan(5.0, ag, {});
    const Phase4Plan po = B.PlanOwn(5.0, 0, ag[0], {}, nullptr, {});
    ASSERT_EQ(po.agents.size(), 2u);
    EXPECT_EQ(po.allocation.method, "broker");
    const auto& a = pc.agents[0].nbv.options;
    const auto& b = po.agents[0].nbv.options;
    ASSERT_EQ(a.size(), b.size());
    ASSERT_FALSE(a.empty());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].task_id, b[i].task_id);
        EXPECT_EQ(a[i].utility, b[i].utility);
        EXPECT_EQ(a[i].goal_xy, b[i].goal_xy);
    }
    EXPECT_EQ(cv::countNonZero(pc.maps.trav_r != po.maps.trav_r), 0);
}

TEST(MultiAgentPlanning, PeerKnownColumnsAreNoGain) {
    std::vector<float> Z, S;
    const DepthFrame f = WallFrame(&Z, &S, 2.0);
    Phase4Params prm;
    Phase4Planner P(Brief(), prm, 2);
    P.Integrate(0, 0, f);
    Phase4AgentInput in;
    in.active = true, in.position = f.T_W_C.t;
    const Phase4Plan free_plan = P.PlanOwn(5.0, 0, in, {}, nullptr, {});
    const cv::Mat all(80, 80, CV_8U, cv::Scalar(1));   // a peer reported every column
    const Phase4Plan told = P.PlanOwn(5.0, 0, in, {}, &all, {});
    int frontier_before = 0, frontier_after = 0;
    for (const auto& o : free_plan.agents[0].nbv.options) frontier_before += o.source == TaskSource::kFrontier;
    for (const auto& o : told.agents[0].nbv.options) {
        frontier_after += o.source == TaskSource::kFrontier;
        EXPECT_EQ(o.ig_raw[0], 0) << "no coverage gain left anywhere";
    }
    EXPECT_GT(frontier_before, 0);
    EXPECT_EQ(frontier_after, 0) << "frontier views that only see peer-known space are no task";
    // RaycastIg directly: the same ray, with and without the mask.
    PeerKnown pk;
    pk.mask = &all, pk.grid = Brief().grid;
    const RigFrustum fr;
    Box3 box;   // the briefing's extent: the mask covers the grid, and rays leaving it are no gain anyway
    box.lo = core::Vec3(0, 0, 0), box.hi = core::Vec3(4, 4, 3);
    const core::Vec3 eye(1.0, 2.0, 1.0);   // looking back (-x) at the unobserved space behind the camera
    EXPECT_GT(RaycastIg(P.Map(), eye, M_PI, fr, nullptr, box).coverage, 0);
    EXPECT_EQ(RaycastIg(P.Map(), eye, M_PI, fr, nullptr, box, -1, &pk).coverage, 0);
}
