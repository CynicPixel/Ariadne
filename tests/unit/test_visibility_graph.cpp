// 4.6 FAR-style visibility graph (design EXPLORATION_4.6 §4, D116): on random rectangle maps whose free space is a
// HOLE of the blocked border (so the orientation test matters), the convex-vertex graph's route is within the derived
// per-bend node offset (2 sqrt 2 cells) of a reference graph over ALL polygon vertices and of the string-pulled A* path, never
// crosses a blocked cell, and the incremental update equals a full rebuild.
#include "planning/mission/phase4_planner.hpp"
#include "planning/route/visibility_graph.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace ariadne::planning;

namespace {

cv::Mat RandomMap(std::mt19937& rng, int H, int W, int n_rects) {
    cv::Mat free(H, W, CV_8U, cv::Scalar(0));
    free(cv::Rect(1, 1, W - 2, H - 2)).setTo(1);   // a blocked border: the free space is a hole of it
    std::uniform_int_distribution<int> X(2, W - 12), Y(2, H - 12), S(2, 9);
    for (int i = 0; i < n_rects; ++i) free(cv::Rect(X(rng), Y(rng), S(rng), S(rng))).setTo(0);
    return free;
}

Cell RandomFree(std::mt19937& rng, const cv::Mat& free) {
    std::uniform_int_distribution<int> R(0, free.rows - 1), C(0, free.cols - 1);
    for (;;) {
        const Cell c{R(rng), C(rng)};
        if (free.at<uint8_t>(c.r, c.c)) return c;
    }
}

double PolyLen(const std::vector<Cell>& p) {
    double s = 0;
    for (size_t i = 1; i < p.size(); ++i) s += std::hypot(double(p[i].r - p[i - 1].r), double(p[i].c - p[i - 1].c));
    return s;
}

}  // namespace

TEST(VisibilityGraph, ConvexGraphMatchesAllVertexReferenceAndBeatsLattice) {
    std::mt19937 rng(46);
    int routed = 0;
    for (int trial = 0; trial < 40; ++trial) {
        const cv::Mat free = RandomMap(rng, 60, 80, 8);
        VisibilityGraph vg, ref({1.0, 1, true});
        vg.Build(free), ref.Build(free);
        const Cell s = RandomFree(rng, free), g = RandomFree(rng, free);
        const VisRoute a = vg.Query(s, {g}), b = ref.Query(s, {g});
        const GridPath lat = AStar(free, s, {g});
        ASSERT_EQ(a.found, lat.found) << trial;
        if (!a.found) continue;
        ++routed;
        // Each node sits delta = offset * sqrt(2) cells from its true corner, so routing through it instead of the
        // corner adds at most 2 delta per bend (triangle inequality): the derived per-bend allowance.
        const size_t bends = a.cells.size() >= 2 ? a.cells.size() - 2 : 0;
        const double eps = 2.0 * std::sqrt(2.0) * 1.0 * double(bends) + 1e-9;
        EXPECT_LE(a.length_cells, b.length_cells + eps) << "convex filter lost a needed corner, trial " << trial;
        const auto pulled = StringPull(free, lat.cells);
        EXPECT_LE(a.length_cells, PolyLen(pulled) + eps) << trial;
        for (size_t i = 1; i < a.cells.size(); ++i) EXPECT_TRUE(LineOfSight(free, a.cells[i - 1], a.cells[i])) << trial;
        EXPECT_LT(vg.Nodes().size(), ref.Nodes().size() + 1) << "the filter keeps a subset";
    }
    EXPECT_GT(routed, 30);
}

TEST(VisibilityGraph, IncrementalEqualsRebuild) {
    std::mt19937 rng(47);
    for (int trial = 0; trial < 20; ++trial) {
        cv::Mat free = RandomMap(rng, 60, 80, 6);
        VisibilityGraph inc;
        inc.Build(free);
        const size_t full_tests = inc.LastLosTests();
        size_t inc_tests = 0;
        for (int round = 0; round < 5; ++round) {
            std::uniform_int_distribution<int> X(2, 70), Y(2, 50);
            free(cv::Rect(X(rng), Y(rng), 3, 3)).setTo(round % 2);   // a local change: block or clear a 3x3 patch
            free.row(0).setTo(0), free.row(free.rows - 1).setTo(0), free.col(0).setTo(0), free.col(free.cols - 1).setTo(0);
            inc_tests += inc.Update(free);
            VisibilityGraph fresh;
            fresh.Build(free);
            ASSERT_EQ(inc.Nodes().size(), fresh.Nodes().size());
            ASSERT_EQ(inc.Edges(), fresh.Edges()) << trial << "/" << round;
            const Cell s = RandomFree(rng, free), g = RandomFree(rng, free);
            const VisRoute a = inc.Query(s, {g}), b = fresh.Query(s, {g});
            ASSERT_EQ(a.found, b.found);
            if (a.found) EXPECT_DOUBLE_EQ(a.length_cells, b.length_cells);
        }
        EXPECT_LT(inc_tests, 5 * full_tests) << "local changes re-test fewer edges than 5 rebuilds";
    }
}

TEST(VisibilityGraph, MultiGoalAndNoRoute) {
    cv::Mat free(20, 40, CV_8U, cv::Scalar(0));
    free(cv::Rect(1, 1, 38, 18)).setTo(1);
    free(cv::Rect(20, 0, 2, 20)).setTo(0);   // a full wall: the east half is unreachable
    VisibilityGraph vg;
    vg.Build(free);
    const VisRoute r = vg.Query({10, 5}, {{10, 30}, {15, 15}});
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.goal_index, 1);
    EXPECT_FALSE(vg.Query({10, 5}, {{10, 30}}).found);
}
