// The live mover's command planner (core/trajectory/command_planner.hpp; design LIVE_RUNTIME_L1 D151).
//   - PlanProfile equals motion.py's plan_profile on every turn and move of the Python golden scripts
//     (tests/fixtures/scripted/): turns at the preset's (omega_max, alpha_max), moves at their own cruise speed;
//   - a commanded path appended to a script keeps it contiguous (Script::Validate), ends at the waypoint and the
//     final yaw, and never exceeds the speed / rate limits or the clearance-scaled speed.
#include "core/trajectory/command_planner.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <string>

using namespace ariadne::core::scripted;

namespace {

Vec5 V5(const YAML::Node& n) {
    Vec5 v{};
    for (int i = 0; i < 5; ++i) v[i] = n[i].as<double>();
    return v;
}

const ClearanceFn kWide = [](double, double, double, double, double, double) { return 10.0; };

}  // namespace

TEST(CommandPlanner, PlanProfileMatchesThePythonScripts) {
    const YAML::Node golden =
        YAML::LoadFile(std::string(ARIADNE_SOURCE_DIR) + "/tests/fixtures/scripted/scripted_golden.json");
    const MotionLimits L;   // params.py Motion defaults; the golden presets do not override them
    int turns = 0, moves = 0;
    for (const auto& c : golden) {
        for (const auto& s : c["script"]["segments"]) {
            const std::string kind = s["kind"].as<std::string>();
            const double D = s["D"].as<double>(), Ta = s["Ta"].as<double>(), vc = s["vc"].as<double>();
            if (kind == "turn") {
                const Profile p = PlanProfile(D, L.omega_max, L.alpha_max);
                EXPECT_EQ(p.Ta, Ta) << c["case"].as<std::string>();
                EXPECT_EQ(p.vc, vc);
                ++turns;
            } else if (kind == "move") {
                // The cruise cap v is not stored; a long move cruises at it (vc = v), a short one is re-timed and
                // independent of it, so v = max(vc, v_cap) reproduces both branches.
                const Profile p = PlanProfile(D, std::max(vc, L.v_cap), L.a_max);
                const Profile q = PlanProfile(D, vc, L.a_max);
                EXPECT_TRUE((p.Ta == Ta && p.vc == vc) || (q.Ta == Ta && q.vc == vc)) << D << " " << Ta << " " << vc;
                ++moves;
            }
        }
    }
    EXPECT_GT(turns, 20);
    EXPECT_GT(moves, 20);
}

TEST(CommandPlanner, AppendedPathIsContiguousAndReachesTheGoal) {
    // A script ending at rest, then a commanded 3-waypoint path with a final yaw.
    Segment hold;
    hold.kind = Segment::Kind::kHold;
    hold.t0 = 0;
    hold.duration = 2.0;
    hold.start = hold.end = {1.0, 2.0, 1.0, 0.3, 0.0};
    const std::vector<Waypoint> path{{3.0, 2.0, 1.0, false, 0}, {3.0, 4.5, 1.1, false, 0}, {1.0, 4.5, 1.0, true, -2.5}};
    const MotionLimits L;
    // Clearance 0.4 m: v = min(0.6, 0.9375 * 0.4) = 0.375 m/s.
    const ClearanceFn clr = [](double, double, double, double, double, double) { return 0.4; };
    std::vector<Segment> segs{hold};
    for (const auto& s : PlanCommand(hold.end, 2.0, path, L, clr, "cmd")) segs.push_back(s);
    const Script sc(segs);
    EXPECT_NO_THROW(sc.Validate());
    const State end = sc.Eval(sc.duration());
    EXPECT_NEAR(end.pose[0], 1.0, 1e-12);
    EXPECT_NEAR(end.pose[1], 4.5, 1e-12);
    EXPECT_NEAR(end.pose[2], 1.0, 1e-12);
    EXPECT_NEAR(WrapPi(end.pose[3] - (-2.5)), 0.0, 1e-12);
    double vmax = 0, wmax = 0;
    for (double t = 0; t <= sc.duration(); t += 0.005) {
        const State s = sc.Eval(t);
        vmax = std::max(vmax, std::sqrt(s.rate[0] * s.rate[0] + s.rate[1] * s.rate[1] + s.rate[2] * s.rate[2]));
        wmax = std::max(wmax, std::abs(s.rate[3]));
        EXPECT_EQ(s.pose[4], 0.0);   // pitch untouched
    }
    EXPECT_LE(vmax, 0.375 + 1e-9);
    EXPECT_LE(wmax, L.omega_max + 1e-9);
    // Every segment ends at rest (any later command can follow it).
    for (const auto& s : segs) {
        const State e = s.Eval(s.t0 + s.duration);
        for (int i = 0; i < 5; ++i) EXPECT_NEAR(e.rate[i], 0.0, 1e-12);
    }
}

TEST(CommandPlanner, HoldIsNoSegmentAndCoincidentWaypointsAreSkipped) {
    const Vec5 at{0, 0, 1, 0, 0};
    EXPECT_TRUE(PlanCommand(at, 5.0, {}, MotionLimits{}, kWide, "hold").empty());
    // Same position, a yaw: one turn only.
    const auto segs = PlanCommand(at, 5.0, {{0, 0, 1, true, 1.0}}, MotionLimits{}, kWide, "goto");
    ASSERT_EQ(segs.size(), 1u);
    EXPECT_EQ(segs[0].kind, Segment::Kind::kTurn);
    EXPECT_DOUBLE_EQ(segs[0].t0, 5.0);
}

TEST(CommandPlanner, ClearanceRefusalPropagates) {
    const ClearanceFn wall = [](double, double, double, double, double, double) -> double {
        throw std::runtime_error("leaves the envelope");
    };
    EXPECT_THROW(PlanCommand({0, 0, 1, 0, 0}, 0.0, {{2, 0, 1, false, 0}}, MotionLimits{}, wall, "x"),
                 std::runtime_error);
}
