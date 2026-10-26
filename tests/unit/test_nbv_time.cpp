// 4.3's time-axis fixes (next-best-view.md §15.4):
//   - D125: an at-cell view (its goal in the agent's own cell) routes as ONE "turn here" waypoint (NaN x, y) with the
//     view's yaw, not as an empty path (which the live loop sends as hold: the ci_tiny seed-2 livelock, t = 20-45 s);
//     PlanCommand turns it into exactly one turn from any nearby pose, and executing that turn collects the view's
//     gain (closed loop in miniature: integrate the frame the turn sees, the view's coverage collapses);
//   - D119: a command cut at max_command_m carries no final yaw (NaN); an uncut one keeps the view's yaw;
//   - an unreachable goal still routes empty (hold); the reserve rule's hold stays a hold;
//   - D117: C_time is the mover's PlanCommand duration (monotone in turn and path; no final turn when cut);
//   - D118: arrival twins carry no final yaw and never cost more; with a baseline policy the keys change nothing.
#include "core/trajectory/command_planner.hpp"
#include "planning/mission/phase4_planner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace ariadne;
using namespace ariadne::planning;

namespace {

constexpr double kCell = 0.05;

// A 6 x 4 m briefing at 5 cm.
Phase4Briefing Briefing() {
    Phase4Briefing b;
    b.grid.ox = 0, b.grid.oy = 0, b.grid.cell = kCell, b.grid.cols = 120, b.grid.rows = 80;
    b.target = {40, 110};
    b.entrances = {{40, 10}};
    return b;
}

// A fronto-parallel plane `depth` m ahead of a camera at `eye` looking along world yaw 0 (+x) or pi (-x).
struct Frame {
    std::vector<float> Z, S;
    DepthFrame f;
    Frame(const core::Vec3& eye, bool facing_minus_x, float depth) {
        const int W = 160, H = 90;
        Z.assign(size_t(W * H), depth), S.assign(size_t(W * H), 0.02f);
        if (facing_minus_x) f.T_W_C.R << 0, 0, -1, 1, 0, 0, 0, -1, 0;   // optical z -> -x, x -> +y, y -> -z
        else f.T_W_C.R << 0, 0, 1, -1, 0, 0, 0, -1, 0;                  // optical z -> +x, x -> -y, y -> -z
        f.T_W_C.t = eye;
        f.fx = 84, f.fy = 84, f.cx = 80, f.cy = 45, f.width = W, f.height = H, f.depth = Z.data(), f.sigma = S.data();
    }
};

Cell CellAt(double x, double y) { return Cell{int(std::floor(y / kCell)), int(std::floor(x / kCell))}; }

TaskCandidate ViewOpt(double x, double y, double yaw) {
    TaskCandidate c;
    c.task_id = 1, c.goal_xy = {x, y}, c.goal_yaw = yaw, c.utility = 1.0;
    return c;
}

}  // namespace

TEST(NbvTime, AtCellViewRoutesAsOneTurnHereWaypoint) {
    const core::Vec3 eye(3.0, 2.0, 1.0);
    const Frame seen(eye, false, 2.0f);   // the agent has looked +x at a wall 2 m ahead
    Phase4Planner P(Briefing(), Phase4Params{});
    P.Integrate(0, 0, seen.f);
    std::vector<Phase4AgentInput> ag(2);
    ag[0].active = true, ag[0].position = eye;
    Phase4Plan pa = P.Plan(5.0, ag, {});
    // The livelock's geometry: the view's eye 2 cm from the agent, inside its cell, facing the unseen side.
    pa.agents[0].nbv.options = {ViewOpt(3.02, 2.01, M_PI)};
    const auto [wp, yaw] = P.Route(pa, 0, 0, CellAt(eye.x(), eye.y()));
    ASSERT_EQ(wp.size(), 1u) << "an at-cell view is executed, not held";
    EXPECT_TRUE(std::isnan(wp[0].first) && std::isnan(wp[0].second)) << "turn here: no position to drive to";
    EXPECT_DOUBLE_EQ(yaw, M_PI);
    TaskCandidate hold = ViewOpt(eye.x(), eye.y(), 0.0);   // the reserve rule's hold sits at the agent: still a hold
    hold.source = TaskSource::kHold;
    pa.agents[0].nbv.options = {hold};
    EXPECT_TRUE(P.Route(pa, 0, 0, CellAt(eye.x(), eye.y())).first.empty());
}

TEST(NbvTime, TurnHereIsOnePureTurnFromAnyNearbyPose) {
    // The mover executes from its GT pose, which differs from the planner's estimate by centimetres. A finite
    // waypoint there would first turn to face that offset (any direction); the NaN waypoint never does.
    using namespace core::scripted;
    const ClearanceFn wide = [](double, double, double, double, double, double) { return 10.0; };
    const MotionLimits L;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int k = 0; k < 8; ++k) {
        const double a = k * M_PI / 4;
        const Vec5 from{3.0 + 0.03 * std::cos(a), 2.0 + 0.03 * std::sin(a), 1.0, 0.3, 0.0};
        const auto segs = PlanCommand(from, 7.0, {{nan, nan, 1.0, true, M_PI}}, L, wide, "turn");
        ASSERT_EQ(segs.size(), 1u) << k;
        EXPECT_EQ(segs[0].kind, Segment::Kind::kTurn);
        EXPECT_DOUBLE_EQ(segs[0].duration, PlanProfile(std::abs(WrapPi(M_PI - 0.3)), L.omega_max, L.alpha_max).T());
        EXPECT_EQ(segs[0].end[0], from[0]);
        EXPECT_EQ(segs[0].end[1], from[1]);
    }
    const Vec5 at{3.0, 2.0, 1.0, 0.3, 0.0};
    EXPECT_TRUE(PlanCommand(at, 7.0, {{nan, nan, 1.0, false, 0.0}}, L, wide, "x").empty()) << "no yaw: nothing to do";
    // Every finite waypoint is as before (M's check): 1e-12 m away is the existing coincident rule (the final turn
    // only), 1 cm away is a face-turn, a move and the final turn.
    const auto same = PlanCommand(at, 7.0, {{3.0 + 1e-12, 2.0, 1.0, true, M_PI}}, L, wide, "x");
    ASSERT_EQ(same.size(), 1u);
    EXPECT_EQ(same[0].kind, Segment::Kind::kTurn);
    const auto cm = PlanCommand(at, 7.0, {{3.01, 2.0, 1.0, true, M_PI}}, L, wide, "x");
    ASSERT_EQ(cm.size(), 3u);
    EXPECT_EQ(cm[1].kind, Segment::Kind::kMove);
}

TEST(NbvTime, ExecutingTheTurnCollectsTheGainSoTheViewStopsWinning) {
    // The livelock persisted because the at-cell view's gain was never collected. Execute the D125 turn: integrate
    // the frame the rig sees after it (a wall 1.5 m behind), and the same view's coverage collapses.
    const core::Vec3 eye(3.0, 2.0, 1.0);
    const Frame ahead(eye, false, 2.0f), behind(eye, true, 1.5f);
    Phase4Planner P(Briefing(), Phase4Params{});
    P.Integrate(0, 0, ahead.f);
    const RigFrustum fr;
    const int before = RaycastIg(P.Map(), eye, M_PI, fr, nullptr).coverage;
    P.Integrate(0, 1, behind.f);
    const int after = RaycastIg(P.Map(), eye, M_PI, fr, nullptr).coverage;
    EXPECT_GT(before, 1000) << "the unseen side is the view's gain";
    EXPECT_LT(after, before / 5) << before << " -> " << after;
}

TEST(NbvTime, CutCommandCarriesNoFinalYawAndUnreachableHolds) {
    const core::Vec3 eye(3.0, 2.0, 1.0);
    const Frame seen(eye, false, 2.0f);
    Phase4Params prm;
    prm.max_command_m = 0.5;
    Phase4Planner P(Briefing(), prm);
    P.Integrate(0, 0, seen.f);
    std::vector<Phase4AgentInput> ag(2);
    ag[0].active = true, ag[0].position = eye;
    Phase4Plan pa = P.Plan(5.0, ag, {});
    const Cell from = CellAt(eye.x(), eye.y());
    pa.agents[0].nbv.options = {ViewOpt(3.75, 2.0, 0.7), ViewOpt(3.3, 2.0, 0.7), ViewOpt(5.5, 2.0, 0.7)};
    const auto [far, far_yaw] = P.Route(pa, 0, 0, from);   // 0.75 m through known free space: cut at 0.5 m
    ASSERT_FALSE(far.empty());
    EXPECT_TRUE(std::isnan(far_yaw)) << "D119: the view's yaw only where the command ends at the view";
    double len = 0, px = (from.c + 0.5) * kCell, py = (from.r + 0.5) * kCell;
    for (const auto& [x, y] : far) len += std::hypot(x - px, y - py), px = x, py = y;
    EXPECT_NEAR(len, 0.5, 1e-9);
    const auto [near, near_yaw] = P.Route(pa, 0, 1, from);   // 0.3 m: ends at the view
    ASSERT_FALSE(near.empty());
    EXPECT_DOUBLE_EQ(near_yaw, 0.7);
    const auto [none, none_yaw] = P.Route(pa, 0, 2, from);   // behind the wall, never seen: no path, hold
    EXPECT_TRUE(none.empty());
}

TEST(NbvTime, CommandTimeIsTheMoversProfileMonotoneInTurnAndPath) {
    // D117: C_time is PlanCommand's duration, so it grows with the turn from the current heading and with the path,
    // and a cut command (D119) or an arrival twin (D118) prices no final turn.
    using core::scripted::MotionLimits;
    using core::scripted::PlanProfile;
    GridFrame g;
    g.ox = 0, g.oy = 0, g.cell = kCell, g.cols = 120, g.rows = 80;
    const cv::Mat wide(80, 120, CV_32F, cv::Scalar(2.0f));   // 2 m of clearance everywhere: v = v_cap
    const MotionLimits L;
    ViewCommand one;
    one.ok = true, one.wp = {{4.0, 2.0}};
    auto T = [&](const ViewCommand& c, double heading, bool yaw, double final_yaw, double* turn = nullptr) {
        return CommandTime(c, 3.0, 2.0, 1.0, heading, yaw, final_yaw, L, wide, g, 0.2, 3.0, turn);
    };
    const double t0 = T(one, 0.0, false, 0.0), t90 = T(one, M_PI / 2, false, 0.0), t180 = T(one, M_PI - 1e-6, false, 0.0);
    EXPECT_LT(t0, t90);
    EXPECT_LT(t90, t180);
    EXPECT_DOUBLE_EQ(t0, PlanProfile(1.0, L.v_cap, L.a_max).T()) << "facing the waypoint: the move alone";
    ViewCommand two = one;
    two.wp = {{5.0, 2.0}};
    EXPECT_LT(t0, T(two, 0.0, false, 0.0));
    double turn = 0;
    const double with_yaw = T(one, 0.0, true, M_PI / 2, &turn);
    EXPECT_DOUBLE_EQ(turn, PlanProfile(M_PI / 2, L.omega_max, L.alpha_max).T());
    EXPECT_DOUBLE_EQ(with_yaw, t0 + turn);
    ViewCommand cut = one;
    cut.cut = true;
    EXPECT_DOUBLE_EQ(T(cut, 0.0, true, M_PI / 2), t0) << "a cut command carries no final turn";
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(T(one, nan, false, 0.0), t0) << "unknown heading: no initial turn priced";
    ViewCommand here;
    here.ok = true, here.wp = {{nan, nan}};
    EXPECT_DOUBLE_EQ(T(here, 0.3, true, M_PI), PlanProfile(M_PI - 0.3, L.omega_max, L.alpha_max).T()) << "a pure turn";
    // Unknown clearance (0) is floored at rho_a: the mover refuses anything closer, so no executed move has less.
    const cv::Mat none(80, 120, CV_32F, cv::Scalar(0.0f));
    EXPECT_DOUBLE_EQ(CommandTime(one, 3.0, 2.0, 1.0, 0.0, false, 0.0, L, none, g, 0.2, 3.0),
                     PlanProfile(1.0, L.v_lk_per_m * 0.2, L.a_max).T());
}

TEST(NbvTime, ArrivalTwinsAndTheKeysOffIdentity) {
    const core::Vec3 eye(3.0, 2.0, 1.0);
    const Frame seen(eye, false, 2.0f);
    std::vector<Phase4AgentInput> ag(2);
    ag[0].active = true, ag[0].position = eye, ag[0].yaw = 0.0;
    auto Run = [&](bool cost_time, bool twins, NbvPolicy pol) {
        Phase4Params prm;
        prm.nbv.policy = pol, prm.nbv.cost_time = cost_time, prm.nbv.arrival_twins = twins;
        prm.allocator = Phase4Params::Allocator::kGreedy;
        auto P = std::make_unique<Phase4Planner>(Briefing(), prm);
        P->Integrate(0, 0, seen.f);
        Phase4Plan pl = P->Plan(5.0, ag, {});
        return std::make_pair(std::move(P), std::move(pl));
    };
    // Keys on for a baseline policy = keys off: D117/D118 touch the Ariadne row only.
    const auto off = Run(false, false, NbvPolicy::kCoverageGreedy), on = Run(true, true, NbvPolicy::kCoverageGreedy);
    const auto& a = off.second.agents[0].nbv.options;
    const auto& b = on.second.agents[0].nbv.options;
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].utility, b[i].utility) << i;
    // Ariadne with both keys: every option is priced by its command; the twins carry no final yaw and their time is
    // the original's minus its final turn.
    const auto ta = Run(true, true, NbvPolicy::kAriadne);
    const auto& o = ta.second.agents[0].nbv.options;
    ASSERT_FALSE(o.empty());
    int twins = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        EXPECT_GT(o[i].cost[3], 0.0) << i;
        if (!o[i].keep_heading) continue;
        ++twins;
        const auto [wp, yaw] = ta.first->Route(ta.second, 0, int(i), CellAt(eye.x(), eye.y()));
        EXPECT_FALSE(wp.empty());
        EXPECT_TRUE(std::isnan(yaw)) << "an arrival twin keeps its heading";
        for (const auto& c : o)
            if (!c.keep_heading && c.task_id == o[i].task_id && c.goal_xy == o[i].goal_xy) EXPECT_LE(o[i].cost[3], c.cost[3]);
    }
    EXPECT_GT(twins, 0);
}
