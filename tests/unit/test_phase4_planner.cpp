// Phase4Planner (the planning side of B's MissionPlanner seam): string pulling never cuts a blocked cell and keeps few
// waypoints; directives (keep_out drops, assign restricts, priority_region scales G and reorders) and refusals; the
// keep_out masks paths, not only goals (B's review); the joint pick is exactly-once and the higher-utility agent
// picks first; an Integrate -> Plan -> Route smoke on a
// synthetic wall, deterministic across two planners.
#include "planning/mission/phase4_planner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace ariadne;
using namespace ariadne::planning;

namespace {

TaskCandidate Opt(int64_t id, double x, double y, double gain, double util) {
    TaskCandidate c;
    c.task_id = id, c.goal_xy = {x, y}, c.gain = gain, c.utility = util;
    return c;
}

bool SegmentClear(const cv::Mat& plan, const Cell& a, const Cell& b) {   // dense sampling of the continuous chord
    const int n = 400;
    for (int k = 0; k <= n; ++k) {
        const double t = double(k) / n;
        const int r = int(std::floor(a.r + 0.5 + t * (b.r - a.r))), c = int(std::floor(a.c + 0.5 + t * (b.c - a.c)));
        if (!plan.at<uint8_t>(r, c)) return false;
    }
    return true;
}

}  // namespace

TEST(Phase4Planner, StringPullLCorridor) {
    cv::Mat plan = cv::Mat::zeros(40, 40, CV_8U);
    plan(cv::Rect(2, 2, 4, 30)).setTo(1);    // vertical leg, columns 2..5
    plan(cv::Rect(2, 28, 30, 4)).setTo(1);   // horizontal leg, rows 28..31
    const GridPath p = AStar(plan, {3, 3}, {{29, 30}});
    ASSERT_TRUE(p.found);
    const auto k = StringPull(plan, p.cells);
    EXPECT_LE(k.size(), 4u) << "an L needs one corner";
    EXPECT_EQ(k.front(), p.cells.front());
    EXPECT_EQ(k.back(), p.cells.back());
    for (size_t i = 1; i < k.size(); ++i) EXPECT_TRUE(SegmentClear(plan, k[i - 1], k[i])) << i;
}

TEST(Phase4Planner, DirectivesAndRefusals) {
    Phase4AgentPlan ap;
    ap.nbv.options = {Opt(1, 1, 1, 1.0, 0.9), Opt(2, 5, 5, 1.0, 0.8), Opt(3, 9, 9, 1.0, 0.7), Opt(4, 2, 8, 1.0, 0.6)};
    RegionDirective keep{"k1", "keep_out", -1, {{0, 0}, {2, 0}, {2, 2}, {0, 2}}, 1.0};
    RegionDirective prio{"p1", "priority_region", 0, {{8, 8}, {10, 8}, {10, 10}, {8, 10}}, 3.0};
    RegionDirective other{"x1", "keep_out", 1, {{4, 4}, {6, 4}, {6, 6}, {4, 6}}, 1.0};   // agent 1 only
    ApplyDirectives(0, {keep, prio, other}, {{2.1, 8.1}}, 0.5, 0.0, &ap);
    ASSERT_EQ(ap.nbv.options.size(), 2u);
    EXPECT_EQ(ap.nbv.options[0].task_id, 3) << "priority x3: utility 0.7 + 2 * 1.0";
    EXPECT_DOUBLE_EQ(ap.nbv.options[0].utility, 2.7);
    EXPECT_EQ(ap.nbv.options[1].task_id, 2);
    EXPECT_EQ(ap.excluded_directive, 1);
    EXPECT_EQ(ap.excluded_refused, 1);
    EXPECT_EQ(ap.applied, (std::vector<std::string>{"k1", "p1"}));
    Phase4AgentPlan as;
    as.nbv.options = {Opt(1, 1, 1, 1, 0.9), Opt(2, 5, 5, 1, 0.8)};
    ApplyDirectives(1, {RegionDirective{"a1", "assign", 1, {{4, 4}, {6, 4}, {6, 6}, {4, 6}}, 1.0}}, {}, 0.5, 0.0, &as);
    ASSERT_EQ(as.nbv.options.size(), 1u);
    EXPECT_EQ(as.nbv.options[0].task_id, 2);
    Phase4AgentPlan none;   // an assign region holding no option restricts nothing (B's semantics)
    none.nbv.options = {Opt(1, 1, 1, 1, 0.9)};
    ApplyDirectives(1, {RegionDirective{"a2", "assign", 1, {{4, 4}, {6, 4}, {6, 6}, {4, 6}}, 1.0}}, {}, 0.5, 0.0, &none);
    EXPECT_EQ(none.nbv.options.size(), 1u);
    Phase4AgentPlan pull;   // a priority region pulls a view whose observed centroid is within observe_m of it
    TaskCandidate near = Opt(1, 0, 0, 1.0, 0.5), far = Opt(2, 0, 0, 1.0, 0.6);
    near.centroid_xy = {7.0, 9.0}, far.centroid_xy = {0.0, 0.0};
    pull.nbv.options = {far, near};
    ApplyDirectives(0, {prio}, {}, 0.5, 2.84, &pull);
    EXPECT_EQ(pull.nbv.options[0].task_id, 1) << "1 m from the region, inside the 2.84 m depth gate: x3";
    EXPECT_DOUBLE_EQ(RegionDirective(prio).Distance(7.0, 9.0), 1.0);
    EXPECT_DOUBLE_EQ(RegionDirective(prio).Distance(9.0, 9.0), 0.0);
    Phase4AgentPlan hr;   // the hard rule's set is outside every directive
    hr.nbv.reserve_rule = true;
    hr.nbv.options = {Opt(9, 1, 1, 0, 1)};
    ApplyDirectives(0, {keep}, {}, 0.5, 0.0, &hr);
    EXPECT_EQ(hr.nbv.options.size(), 1u);
}

TEST(Phase4Planner, KeepOutMasksPaths) {
    // A 2 m-wide corridor with a keep_out band across its full width: no path crosses it. The same directive aimed
    // at agent 1 leaves agent 0's space open.
    GridFrame g;
    g.ox = 0, g.oy = 0, g.cell = 0.05, g.cols = 40, g.rows = 100;
    cv::Mat plan(100, 40, CV_8U, cv::Scalar(1));
    MaskKeepOut(0, {RegionDirective{"k", "keep_out", -1, {{-1, 2.0}, {3, 2.0}, {3, 2.5}, {-1, 2.5}}, 1.0}}, g, &plan);
    EXPECT_FALSE(AStar(plan, {10, 20}, {{90, 20}}).found);
    cv::Mat plan1(100, 40, CV_8U, cv::Scalar(1));   // the same directive for agent 1 only leaves agent 0 unmasked
    MaskKeepOut(0, {RegionDirective{"k", "keep_out", 1, {{-1, 2.0}, {3, 2.0}, {3, 2.5}, {-1, 2.5}}, 1.0}}, g, &plan1);
    EXPECT_TRUE(AStar(plan1, {10, 20}, {{90, 20}}).found);
}

TEST(Phase4Planner, JointPickExactlyOnce) {
    Phase4Plan p;
    p.agents[0].nbv.options = {Opt(7, 0, 0, 1, 0.5), Opt(8, 0, 0, 1, 0.4)};
    p.agents[1].nbv.options = {Opt(7, 0, 0, 1, 0.9), Opt(9, 0, 0, 1, 0.1)};
    JointPick(&p);
    EXPECT_EQ(p.joint[1], 0) << "agent 1's best utility is higher: it picks task 7";
    EXPECT_EQ(p.joint[0], 1) << "agent 0 takes its best other task";
    Phase4Plan q;
    q.agents[0].nbv.options = {Opt(5, 0, 0, 1, 1.0)};
    JointPick(&q);
    EXPECT_EQ(q.joint[0], 0);
    EXPECT_EQ(q.joint[1], -1);
}

TEST(Phase4Planner, IntegratePlanRouteSmoke) {
    // A 4 x 4 m briefing; one camera at (0.5, 2.0, 1.0) facing +x sees a wall at x = 2.5 m (depth = 2 m on the axis).
    Phase4Briefing b;
    b.grid.ox = 0, b.grid.oy = 0, b.grid.cell = 0.05, b.grid.cols = 80, b.grid.rows = 80;
    b.target = {40, 70};
    b.entrances = {{40, 10}};
    Phase4Params prm;
    const int W = 160, H = 90;
    const double fx = 84, fy = 84, cx = 80, cy = 45;
    std::vector<float> Z(size_t(W * H)), S(size_t(W * H), 0.02f);
    for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u) Z[size_t(v * W + u)] = 2.0f;   // a fronto-parallel plane: z_cam = 2 m everywhere
    DepthFrame f;
    f.T_W_C.R << 0, 0, 1, -1, 0, 0, 0, -1, 0;   // optical z -> world +x, optical x -> world -y, optical y -> world -z
    f.T_W_C.t = core::Vec3(0.5, 2.0, 1.0);
    f.fx = fx, f.fy = fy, f.cx = cx, f.cy = cy, f.width = W, f.height = H, f.depth = Z.data(), f.sigma = S.data();
    Phase4Planner A(b, prm), B(b, prm);
    EXPECT_GT(A.Integrate(0, 0, f), 0u);
    B.Integrate(0, 0, f);
    Phase4AgentInput ag[2];
    ag[0].active = true, ag[0].position = f.T_W_C.t;
    const Phase4Plan pa = A.Plan(5.0, ag, {}), pb = B.Plan(5.0, ag, {});
    ASSERT_EQ(pa.agents[0].nbv.options.size(), pb.agents[0].nbv.options.size());
    for (size_t i = 0; i < pa.agents[0].nbv.options.size(); ++i) {
        EXPECT_EQ(pa.agents[0].nbv.options[i].task_id, pb.agents[0].nbv.options[i].task_id);
        EXPECT_EQ(pa.agents[0].nbv.options[i].utility, pb.agents[0].nbv.options[i].utility);
    }
    EXPECT_FALSE(pa.egress.found) << "the target is behind the wall, unobserved";
    if (!pa.agents[0].nbv.options.empty()) {
        EXPECT_EQ(pa.joint[0], 0);
        const Cell from{40, 10};
        const auto [wp, yaw] = A.Route(pa, 0, 0, from);
        EXPECT_DOUBLE_EQ(yaw, pa.agents[0].nbv.options[0].goal_yaw);
        double len = 0, px = (from.c + 0.5) * 0.05, py = (from.r + 0.5) * 0.05;
        for (const auto& [x, y] : wp) len += std::hypot(x - px, y - py), px = x, py = y;
        EXPECT_LE(len, prm.max_command_m + 1e-9);
    }
}
