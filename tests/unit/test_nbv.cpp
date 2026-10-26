// 4.3 NBV (design next-best-view.md §6, D114): the IG raycast on a wall with and without a hole (exact zero behind a solid
// wall; the entropy identity), self-normalisation's scale invariance and fallbacks, the lambda schedule by hand, the
// energy-reserve rule unreachable from the optimizer (alias views of a 1e6-gain place are emitted above the reserve and all dropped below it), the sampler's
// trav_a / line-of-sight / determinism contract, generator E's target at the route's lowest support.
#include "planning/mission/nbv.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace ariadne;
using namespace ariadne::planning;

namespace {

// A 3 x 3 m patch, z 0..2 m: voxels with x < 1.0 m observed free (w = 1), a wall plane at voxel x = 20 (1.00 m)
// occupied (w = 1), optionally a 0.3 x 0.3 m hole centred at (y, z) = (1.5, 1.0); everything else unknown.
TsdfMap WallMap(bool hole) {
    TsdfParams tp;
    TsdfMap m(tp);
    for (int y = 0; y < 60; ++y)
        for (int z = 0; z < 40; ++z) {
            for (int x = 0; x < 20; ++x) m.SetVoxel({x, y, z}, TsdfVoxel{0.1f, 1.f});
            const bool in_hole = hole && std::abs((y + 0.5) * 0.05 - 1.5) < 0.15 && std::abs((z + 0.5) * 0.05 - 1.0) < 0.15;
            m.SetVoxel({20, y, z}, in_hole ? TsdfVoxel{0.1f, 1.f} : TsdfVoxel{-0.01f, 1.f});
        }
    return m;
}

struct Scene {
    TsdfMap map;
    FrontierSet3D fr{FrontierParams{}};
    CoarseLayer coarse;
    TraversabilityMaps tm;
    GridFrame g;
    explicit Scene(TsdfMap m) : map(std::move(m)) {
        fr.Detect(map);
        coarse = BuildCoarse(map);
        g.ox = 0, g.oy = 0, g.cell = 0.05, g.cols = 60, g.rows = 60;
        tm.trav_a = cv::Mat::zeros(60, 60, CV_8U);
        tm.trav_a(cv::Rect(2, 2, 16, 56)).setTo(1);   // x 0.1..0.9 m: the observed free side
        tm.trav_r = tm.trav_a.clone();
        tm.slab = cv::Mat(60, 60, CV_8U, cv::Scalar(kFree));
        tm.clear_r = cv::Mat(60, 60, CV_32F, cv::Scalar(0.5f));
        tm.risk = cv::Mat::zeros(60, 60, CV_32F);
    }
    NbvInputs Inputs() const {
        NbvInputs in;
        in.map = &map, in.frontiers = &fr, in.coarse = &coarse, in.trav = &tm, in.grid = g;
        in.entrances = {{30, 3}};
        return in;
    }
};

class FakeAlias : public AliasSource {
public:
    std::vector<AliasPlace> Places() const override { return {{7, core::Vec3(0.5, 1.5, 1.0), 1e6}}; }
};

}  // namespace

TEST(Nbv, RaycastIgWallAndHole) {
    const RigFrustum f;
    const core::Vec3 eye(0.4, 1.5, 1.0);
    const TsdfMap solid = WallMap(false), holed = WallMap(true);
    const IgCounts a = RaycastIg(solid, eye, 0.0, f, nullptr);
    EXPECT_EQ(a.coverage, 0) << "everything in front of a solid wall is known";
    const double h1 = -(0.5 * std::exp(-1.0) * std::log2(0.5 * std::exp(-1.0)) +
                        (1 - 0.5 * std::exp(-1.0)) * std::log2(1 - 0.5 * std::exp(-1.0)));
    EXPECT_NEAR(a.entropy, a.voxels * h1, 1e-9 * a.voxels) << "every reached voxel has w = 1";
    const IgCounts b = RaycastIg(holed, eye, 0.0, f, nullptr);
    EXPECT_GT(b.coverage, 0) << "the hole lets rays into the unknown";
    EXPECT_NEAR(b.entropy, (b.voxels - b.coverage) * h1 + b.coverage, 1e-9 * b.voxels);
    const IgCounts c = RaycastIg(holed, eye, M_PI, f, nullptr);   // facing away: the unknown beyond the patch
    EXPECT_GT(c.coverage, 0);
    Box3 patch;   // the same view clipped to the observed patch's extent: nothing unknown is inside it
    patch.lo = core::Vec3(0, 0, 0), patch.hi = core::Vec3(1.0, 3.0, 2.0);
    EXPECT_EQ(RaycastIg(holed, eye, M_PI, f, nullptr, patch).coverage, 0);
    EXPECT_LT(RaycastIg(holed, eye, M_PI, f, nullptr, Box3{}, 8).voxels, c.voxels) << "a 64-ray bundle reaches fewer voxels";
    // The route envelope counts only its own voxels: here everything behind the wall, out to the rays' 2.84 m reach.
    std::unordered_set<int64_t> env;
    for (int x = 21; x < 70; ++x)
        for (int y = -30; y < 90; ++y)
            for (int z = -40; z < 100; ++z) env.insert(Pack({x, y, z}));
    EXPECT_EQ(RaycastIg(holed, eye, 0.0, f, &env).egress, b.coverage) << "behind the wall every unknown is in env";
    EXPECT_EQ(RaycastIg(solid, eye, 0.0, f, &env).egress, 0);
}

TEST(Nbv, P90ScaleInvarianceAndFallbacks) {
    std::vector<double> v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100};
    const double s = P90Scale(v);
    EXPECT_DOUBLE_EQ(s, 9.0);
    for (double k : {0.001, 7.0, 1e6}) {
        std::vector<double> w;
        for (double x : v) w.push_back(k * x);
        const double sk = P90Scale(w);
        for (size_t i = 0; i < v.size(); ++i) EXPECT_NEAR(w[i] / sk, v[i] / s, 1e-12);
    }
    EXPECT_DOUBLE_EQ(P90Scale({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5}), 5.0) << "p90 = 0: fall back to the max";
    EXPECT_DOUBLE_EQ(P90Scale({0, 0, 0}), 0.0);
    EXPECT_DOUBLE_EQ(P90Scale({}), 0.0);
}

TEST(Nbv, LambdaScheduleByHand) {
    Scene sc(WallMap(true));
    NbvParams p;
    p.e_total_m = 100, p.mission_T_s = 200;
    AgentState st;
    st.position = core::Vec3(0.4, 1.5, 1.0);
    NbvInputs in = sc.Inputs();
    EXPECT_DOUBLE_EQ(PlanNbv(in, 0, st, p).lambda_path, 1.0);
    st.e_used_m = 50;
    EXPECT_NEAR(PlanNbv(in, 0, st, p).lambda_path, (1 + 1.125 + 1) / 3.0, 1e-12);
    st.e_used_m = 0, in.t_s = 200;
    EXPECT_NEAR(PlanNbv(in, 0, st, p).lambda_path, (1 + 1 + 2) / 3.0, 1e-12);
}

TEST(Nbv, EnergyReserveRuleIsOutsideTheOptimizer) {
    Scene sc(WallMap(true));
    FakeAlias alias;
    NbvInputs in = sc.Inputs();
    in.alias = &alias;
    AgentState st;
    st.position = core::Vec3(0.4, 1.5, 1.0);   // 1.35 m of path from the entrance cell (30, 3) is ~0.25 m away in x
    NbvParams p;
    p.e_total_m = 100;
    const NbvPlan free_plan = PlanNbv(in, 0, st, p);
    ASSERT_FALSE(free_plan.reserve_rule);
    ASSERT_FALSE(free_plan.options.empty());
    int alias_views = 0;   // a 1e6 gain self-normalises to 1 (p90 = 0 -> max): it is a candidate, not a trump
    for (const auto& c : free_plan.options) alias_views += c.source == TaskSource::kAlias;
    EXPECT_GT(alias_views, 0);
    st.e_used_m = 100 - 0.2;   // remaining 0.2 m < 1.3 x the return path
    const NbvPlan low = PlanNbv(in, 0, st, p);
    ASSERT_TRUE(low.reserve_rule);
    ASSERT_EQ(low.options.size(), 2u);
    EXPECT_EQ(low.options[0].source, TaskSource::kReturn);
    EXPECT_EQ(low.options[1].source, TaskSource::kHold);
    EXPECT_NE(low.supervisor_event.find("energy reserve"), std::string::npos);
}

TEST(Nbv, SamplerContract) {
    Scene sc(WallMap(true));
    const RigFrustum f;
    SamplerParams sp;
    const core::Vec3 target(1.0, 1.5, 1.0);   // the hole
    const auto v1 = SampleViews(sc.map, sc.g, sc.tm.trav_a, target, sp, f, nullptr);
    const auto v2 = SampleViews(sc.map, sc.g, sc.tm.trav_a, target, sp, f, nullptr);
    ASSERT_FALSE(v1.empty());
    ASSERT_EQ(v1.size(), v2.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        EXPECT_EQ(v1[i].eye, v2[i].eye);
        EXPECT_TRUE(sc.tm.trav_a.at<uint8_t>(v1[i].cell.r, v1[i].cell.c));
        EXPECT_NEAR(std::atan2(target.y() - v1[i].eye.y(), target.x() - v1[i].eye.x()), v1[i].yaw, 1e-12);
    }
    EXPECT_LE(int(v1.size()), sp.keep);
    // A target behind the solid wall has no line of sight from the free side.
    const Scene solid(WallMap(false));
    EXPECT_TRUE(SampleViews(solid.map, solid.g, solid.tm.trav_a, core::Vec3(1.5, 1.5, 1.0), sp, f, nullptr).empty());
}

TEST(Nbv, EgressTargetAtLowestSupport) {
    Scene sc(WallMap(true));
    EgressReplan route;
    route.found = true;
    for (int r = 10; r < 50; ++r) route.cells.push_back({r, 8});
    cv::Mat support(60, 60, CV_32F, cv::Scalar(1.f));
    for (int r = 20; r < 26; ++r) support.at<float>(r, 8) = 0.9f;
    support.at<float>(23, 8) = 0.5f;   // the run's minimum
    NbvInputs in = sc.Inputs();
    in.route = &route, in.route_support = &support;
    AgentState st;
    st.position = core::Vec3(0.4, 1.5, 1.0);
    const NbvPlan pl = PlanNbv(in, 0, st, NbvParams{});
    EXPECT_EQ(pl.targets[4], 1);
    bool found = false;
    for (const auto& c : pl.options) {
        if (c.source != TaskSource::kEgress) continue;
        found = true;
        EXPECT_NEAR(c.centroid_xy.first, (8 + 0.5) * 0.05, 1e-12);
        EXPECT_NEAR(c.centroid_xy.second, (23 + 0.5) * 0.05, 1e-12);
        EXPECT_EQ(c.cells, 6);
    }
    EXPECT_TRUE(found) << "generator E emitted no view";
    EXPECT_EQ(pl.notes.size(), 2u) << "both missing producers are named";
}

TEST(Nbv, OpenTourAgainstExactOracle) {
    std::mt19937 rng(416);
    std::uniform_real_distribution<double> u(0, 10);
    double worst = 1;
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 2 + trial % 7;   // up to 8 nodes incl. the start
        std::vector<std::pair<double, double>> pts(static_cast<size_t>(n));
        for (auto& q : pts) q = {u(rng), u(rng)};
        std::vector<std::vector<double>> D(static_cast<size_t>(n), std::vector<double>(static_cast<size_t>(n)));
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b) D[size_t(a)][size_t(b)] = std::hypot(pts[size_t(a)].first - pts[size_t(b)].first, pts[size_t(a)].second - pts[size_t(b)].second);
        const auto h = OpenTour(D), e = ExactOpenTour(D);
        ASSERT_EQ(int(h.size()), n - 1);
        const double lh = TourLength(D, h), le = TourLength(D, e);
        EXPECT_GE(lh, le - 1e-9);
        if (lh > 0) worst = std::min(worst, le / lh);
    }
    std::printf("[tare] NN+2-opt vs exact open tour, 200 instances <= 8 nodes: worst exact/heuristic %.3f\n", worst);
}

TEST(Nbv, TareSchedulesLocalThenGlobal) {
    Scene sc(WallMap(true));
    NbvInputs in = sc.Inputs();
    AgentState st;
    st.position = core::Vec3(0.4, 1.5, 1.0);
    NbvParams p;
    p.policy = NbvPolicy::kTare;
    const NbvPlan local = PlanNbv(in, 0, st, p);
    ASSERT_FALSE(local.options.empty());
    EXPECT_EQ(local.tare.rfind("local", 0), 0u) << local.tare;
    const auto& f = local.options.front();
    EXPECT_LE(std::max(std::abs(f.goal_xy.first - 0.4), std::abs(f.goal_xy.second - 1.5)), p.frustum.max_range_m + 0.05);
    p.frustum.max_range_m = 0.05;   // a horizon too small to hold any view: the global tour schedules
    const NbvPlan global = PlanNbv(in, 0, st, p);
    if (!global.options.empty()) EXPECT_EQ(global.tare.rfind("global", 0), 0u) << global.tare;
    const NbvPlan again = PlanNbv(in, 0, st, p);   // deterministic
    ASSERT_EQ(again.options.size(), global.options.size());
    if (!again.options.empty()) EXPECT_EQ(again.options.front().task_id, global.options.front().task_id);
}

TEST(Nbv, ArrivalBottleneckIsTheWidestWayIntoTheGoalDisc) {
    // D158 as B built it: a goal cell with clearance 0 (at the edge of unknown space) whose 0.5 m disc holds a cell
    // reachable through 0.45 m must read 0.45 m, not 0.
    cv::Mat B(20, 20, CV_64F, cv::Scalar(0.0));
    B.at<double>(10, 13) = 0.45;   // 3 cells (0.15 m) from the goal
    B.at<double>(10, 19) = 0.9;    // 9 cells away: inside a 10-cell disc, outside a 5-cell one
    EXPECT_DOUBLE_EQ(ArrivalBottleneck(B, {10, 10}, 5), 0.45);
    EXPECT_DOUBLE_EQ(ArrivalBottleneck(B, {10, 10}, 10), 0.9);
    EXPECT_DOUBLE_EQ(ArrivalBottleneck(B, {10, 10}, 0), 0.0);
    cv::Mat none(5, 5, CV_64F, cv::Scalar(-1.0));
    EXPECT_DOUBLE_EQ(ArrivalBottleneck(none, {2, 2}, 2), -1.0);
}

TEST(Nbv, ReturnOverTheFlownTrackWhenTheWayInWasNeverObserved) {
    // B's live finding (2026-10-23): the forward rig never observes the space behind it, so the way back to the entrance
    // is not known-free; E(return) = inf fired the hard rule on every replan and left only {hold}. The return space adds
    // the swept body of the flown track (GBPlanner's homing over the robot's own path; next-best-view.md §13).
    Scene sc(WallMap(true));
    sc.tm.trav_a(cv::Rect(2, 2, 11, 56)).setTo(0);   // x 0.10..0.65 m (the way in, entrance cell (30, 3)): unobserved
    NbvInputs in = sc.Inputs();
    AgentState st;
    st.position = core::Vec3(0.85, 1.5, 1.0);   // the escape disc (0.5 m) reaches x 0.35 m: not the entrance
    NbvParams p;
    p.e_total_m = 100;

    const NbvPlan lost = PlanNbv(in, 0, st, p);   // no track: no route home at all
    EXPECT_FALSE(lost.reserve_rule);
    EXPECT_NE(lost.supervisor_event.find("return route lost"), std::string::npos);
    EXPECT_GT(lost.options.size(), 1u);   // a lost route does not freeze the agent on {hold}

    st.own_keyframes = {core::Vec3(0.20, 1.5, 1.0), core::Vec3(0.50, 1.5, 1.0), core::Vec3(0.85, 1.5, 1.0)};
    const NbvPlan ok = PlanNbv(in, 0, st, p);
    EXPECT_FALSE(ok.reserve_rule);
    EXPECT_TRUE(ok.supervisor_event.empty());
    EXPECT_GT(ok.options.size(), 1u);

    st.e_used_m = 100 - 0.5;   // 0.5 m left < 1.3 x ~0.7 m of track
    const NbvPlan low = PlanNbv(in, 0, st, p);
    ASSERT_TRUE(low.reserve_rule);
    ASSERT_EQ(low.options.size(), 2u);
    EXPECT_EQ(low.options[0].source, TaskSource::kReturn);
    EXPECT_NEAR(low.options[0].path_m, 0.70, 0.08);   // entrance centre (0.175, 1.525) to (0.85, 1.5), on the lattice
    EXPECT_LT(low.options[0].risk_geometric, 1.0);    // scored on the return space, not the known-free plan (b < 0)

    // A collapse across the track (the slab now marks it occupied): the swept columns no longer count.
    sc.tm.slab(cv::Rect(8, 0, 1, 60)).setTo(kOccupied);
    const NbvPlan cut = PlanNbv(in, 0, st, p);
    EXPECT_FALSE(cut.reserve_rule);
    EXPECT_NE(cut.supervisor_event.find("return route lost"), std::string::npos);

    // FlownReturnSpace itself: the swept band is the body disc rounded down (0.20 m -> 4 cells each side).
    cv::Mat swept;
    const cv::Mat ret = FlownReturnSpace(cv::Mat::zeros(60, 60, CV_8U), cv::Mat(60, 60, CV_8U, cv::Scalar(kFree)), sc.g,
                                         {core::Vec3(0.20, 1.5, 1.0), core::Vec3(0.85, 1.5, 1.0)}, {{30, 3}}, 0.20, &swept);
    EXPECT_EQ(ret.at<uint8_t>(30, 3), 1);
    EXPECT_EQ(ret.at<uint8_t>(34, 10), 1);
    EXPECT_EQ(ret.at<uint8_t>(35, 10), 0);
    EXPECT_EQ(cv::countNonZero(swept), cv::countNonZero(ret));
}
