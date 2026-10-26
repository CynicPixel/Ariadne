// The live runtime's command path (runtime/live/planner.hpp CommandPath; design live-runtime.md D153, fixed 2026-10-23):
//   - on random obstacle fields, every chord of the command is traversable (the safety rule the old code enforced by
//     falling back to the raw A* cells), the last waypoint is the goal's centre, and the command is short: far fewer
//     waypoints than A* cells, even where approxPolyDP's own chords clip an obstacle (the old code then sent one
//     waypoint per 5 cm cell: 45 waypoints, 85 s of turn-and-move for 2.6 m in the first LLM run);
//   - the fields include such cases (counted), so the refinement branch is exercised, not just the happy path.
#include "runtime/live/planner.hpp"

#include <gtest/gtest.h>

#include <random>

using namespace ariadne::runtime::live;
using ariadne::planning::Cell;

TEST(LivePlanner, CommandPathChordsFreeAndShort) {
    ariadne::planning::mission::Briefing b;
    b.gs.origin_x = 0, b.gs.origin_y = 0, b.gs.cell = 0.05;
    int dp_clipped = 0, checked = 0;
    for (int seed = 0; seed < 200; ++seed) {
        std::mt19937 rng(seed);
        cv::Mat trav(80, 80, CV_8U, cv::Scalar(1));
        std::uniform_int_distribution<int> pos(8, 71), rad(3, 9);
        for (int k = 0; k < 8; ++k) {   // filled discs: staircase boundaries A* hugs, where a 1-cell chord tolerance cuts corners
            const cv::Point c{pos(rng), pos(rng)};   // centre, then radius: sequenced (D123)
            const int r = rad(rng);
            cv::circle(trav, c, r, cv::Scalar(0), cv::FILLED);
        }
        const Cell from{2, 2}, goal{77, 77};
        const auto p = ariadne::planning::AStar(trav, from, {goal});
        if (!p.found) continue;
        ++checked;
        // does approxPolyDP alone produce a chord through an obstacle here?
        std::vector<cv::Point> pts, dp;
        for (const auto& c : p.cells) pts.emplace_back(c.c, c.r);
        cv::approxPolyDP(pts, dp, 1.0, false);
        for (size_t i = 1; i < dp.size(); ++i) {
            if (!ChordFree(trav, dp[i - 1], dp[i])) { ++dp_clipped; break; }
        }
        const auto w = CommandPath(b, trav, from, goal, 1.0, 1e9);
        ASSERT_FALSE(w.empty()) << "seed " << seed;
        cv::Point prev(from.c, from.r);
        for (const auto& [x, y] : w) {
            const Cell c = b.ToCell(x, y);
            const cv::Point q(c.c, c.r);
            EXPECT_TRUE(ChordFree(trav, prev, q)) << "seed " << seed << ": chord through an obstacle";
            prev = q;
        }
        const auto [gx, gy] = b.CellCentre(goal);
        EXPECT_NEAR(w.back().first, gx, 1e-9) << "seed " << seed;
        EXPECT_NEAR(w.back().second, gy, 1e-9) << "seed " << seed;
        EXPECT_LT(w.size() * 4, p.cells.size()) << "seed " << seed << ": " << w.size() << " waypoints for "
                                                << p.cells.size() << " cells";
    }
    EXPECT_GT(checked, 150);
    EXPECT_GT(dp_clipped, 0) << "no field made approxPolyDP clip an obstacle: the refinement branch is untested";
    std::printf("[ LivePlanner ] %d fields, %d where approxPolyDP alone clipped an obstacle\n", checked, dp_clipped);
}
