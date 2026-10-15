// 1.8 planning placeholders (design MISSION_1.8): exact voxel traversal, OctoMap
// tri-state carving, GT-convention clearance, A* optimality and frontier clusters,
// each against a brute-force or hand-computed answer.
#include "planning/frontier/frontiers.hpp"
#include "planning/route/astar.hpp"
#include "planning/volumetric/clearance.hpp"
#include "planning/volumetric/occupancy_grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <tuple>

using namespace ariadne;
using namespace ariadne::planning;

namespace {
// Brute force: every voxel whose cube intersects segment a->b (slab test), as a set.
std::set<std::tuple<int, int, int>> BruteVoxels(const core::Vec3& a, const core::Vec3& b) {
    std::set<std::tuple<int, int, int>> s;
    const core::Vec3 lo = a.cwiseMin(b), hi = a.cwiseMax(b), d = b - a;
    for (int x = int(std::floor(lo.x())); x <= int(std::floor(hi.x())); ++x)
        for (int y = int(std::floor(lo.y())); y <= int(std::floor(hi.y())); ++y)
            for (int z = int(std::floor(lo.z())); z <= int(std::floor(hi.z())); ++z) {
                double t0 = 0, t1 = 1;
                bool hit = true;
                const double mn[3] = {double(x), double(y), double(z)};
                for (int k = 0; k < 3 && hit; ++k) {
                    if (std::abs(d(k)) < 1e-15) {
                        hit = a(k) >= mn[k] && a(k) <= mn[k] + 1;
                    } else {
                        double ta = (mn[k] - a(k)) / d(k), tb = (mn[k] + 1 - a(k)) / d(k);
                        if (ta > tb) std::swap(ta, tb);
                        t0 = std::max(t0, ta);
                        t1 = std::min(t1, tb);
                        hit = t0 < t1 - 1e-12;   // positive-length overlap: touching a corner does not count
                    }
                }
                if (hit) s.insert({x, y, z});
            }
    return s;
}
}  // namespace

TEST(Planning, VoxelTraversalMatchesBruteForce) {
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(0.05, 9.95);
    std::vector<std::pair<core::Vec3, core::Vec3>> rays = {{{0.5, 0.5, 0.5}, {7.5, 0.5, 0.5}},     // axis-aligned
                                                           {{0.2, 0.3, 0.4}, {6.7, 5.1, 2.9}},     // oblique
                                                           {{5.5, 5.5, 5.5}, {0.5, 9.2, 1.7}}};    // negative steps
    for (int i = 0; i < 300; ++i) rays.push_back({{u(rng), u(rng), u(rng)}, {u(rng), u(rng), u(rng)}});
    for (const auto& [a, b] : rays) {
        std::vector<std::tuple<int, int, int>> walk;
        TraverseVoxels(a, b, [&](int x, int y, int z) { walk.push_back({x, y, z}); });
        const std::set<std::tuple<int, int, int>> ws(walk.begin(), walk.end());
        ASSERT_EQ(ws.size(), walk.size()) << "a voxel visited twice";
        EXPECT_EQ(ws, BruteVoxels(a, b));
        EXPECT_EQ(walk.front(), std::make_tuple(int(std::floor(a.x())), int(std::floor(a.y())), int(std::floor(a.z()))));
    }
}

TEST(Planning, CarvingGivesTriStateAndKeepsTheSurface) {
    GridSpec g{0, 0, 0, 0.1, 40, 10, 10};   // 4 m x 1 m x 1 m
    OccupancyGrid m(g);
    const core::Vec3 c(0.05, 0.55, 0.55), p(3.05, 0.55, 0.55);   // a wall at x = 3.0..3.1
    for (int i = 0; i < 6; ++i) m.IntegrateRay(c, c + (p - c) * ((3.0 - 0.2) / 3.0), p);   // margin 0.2 m
    EXPECT_EQ(m.State(0, 5, 5), kFree);
    EXPECT_EQ(m.State(27, 5, 5), kFree);       // ends at x = 2.85 -> voxel 28 is the last carved
    EXPECT_EQ(m.State(29, 5, 5), kUnknown);    // inside the margin: not carved
    EXPECT_EQ(m.State(30, 5, 5), kOccupied);
    EXPECT_EQ(m.State(35, 5, 5), kUnknown);    // behind the wall
    const cv::Mat s = m.Slab(0, 10);
    EXPECT_EQ(s.at<uint8_t>(5, 30), kOccupied);
    EXPECT_EQ(s.at<uint8_t>(5, 10), kFree);
    EXPECT_EQ(s.at<uint8_t>(0, 10), kUnknown);
    // One hit outweighs one miss (OctoMap: logit(.7) + logit(.4) > 0): occupied stays occupied.
    m.IntegrateRay(c, p + core::Vec3(0.3, 0, 0), p + core::Vec3(0.4, 0, 0));
    EXPECT_EQ(m.State(30, 5, 5), kOccupied);
}

TEST(Planning, ClearanceMatchesBruteForceEdgeDistance) {
    std::mt19937 rng(5);
    cv::Mat b(30, 40, CV_8U);
    for (int r = 0; r < b.rows; ++r)
        for (int c = 0; c < b.cols; ++c) b.at<uint8_t>(r, c) = rng() % 9 == 0;
    const double cell = 0.05, cap = 0.5;
    const cv::Mat d = EdgeClearance(b, cell, cap);
    for (int r = 0; r < b.rows; ++r)
        for (int c = 0; c < b.cols; ++c) {
            double best = std::numeric_limits<double>::infinity();
            for (int rr = 0; rr < b.rows; ++rr)
                for (int cc = 0; cc < b.cols; ++cc) {
                    if (!b.at<uint8_t>(rr, cc)) continue;
                    const double dx = std::max(std::abs(cc - c) - 0.5, 0.0), dy = std::max(std::abs(rr - r) - 0.5, 0.0);
                    best = std::min(best, (rr == r && cc == c) ? 0.0 : std::sqrt(dx * dx + dy * dy));
                }
            EXPECT_NEAR(d.at<double>(r, c), std::min(best, cap / cell) * cell, 1e-12) << r << "," << c;
        }
}

TEST(Planning, DoorwayClearanceByHand) {
    // Walls on rows 0 and 22 (a 21-cell = 1.05 m opening at 0.05 m, like the 1.05 m north entrance).
    cv::Mat b = cv::Mat::zeros(23, 5, CV_8U);
    b.row(0).setTo(1);
    b.row(22).setTo(1);
    const cv::Mat d = EdgeClearance(b, 0.05, 1.0);
    // Centre row 11: 11 cells from each wall centre -> 10.5 cells to the edge = 0.525 m.
    EXPECT_NEAR(d.at<double>(11, 2), 0.525, 1e-12);
    const cv::Mat free = 1 - b;
    const cv::Mat t = Traversable(free, d, 0.40);
    int n = 0;
    for (int r = 0; r < 23; ++r) n += t.at<uint8_t>(r, 2);
    // Row r is r - 0.5 and 21.5 - r cells from the wall edges; >= 8 cells (0.40 m) holds for rows 9..13:
    // a 5-cell (0.25 m) traversable band, 2.5 cells (0.125 m) of slack either side of the centre.
    EXPECT_EQ(n, 5);
}

TEST(Planning, AStarFindsTheOptimumAndRespectsCorners) {
    cv::Mat t = cv::Mat::ones(5, 5, CV_8U);
    t.at<uint8_t>(1, 1) = 0;   // diagonal (0,0)->(1,1) blocked; (0,1)/(1,0) still free
    t.at<uint8_t>(2, 1) = 0;
    t.at<uint8_t>(1, 2) = 0;
    const GridPath p = AStar(t, {0, 0}, {{4, 4}});
    ASSERT_TRUE(p.found);
    const cv::Mat d = GridDistances(t, {0, 0});
    EXPECT_NEAR(p.length_cells, d.at<double>(4, 4), 1e-12);   // A* = Dijkstra
    for (size_t i = 1; i < p.cells.size(); ++i) {               // no corner squeezing
        const Cell a = p.cells[i - 1], b = p.cells[i];
        if (a.r != b.r && a.c != b.c) {
            EXPECT_TRUE(t.at<uint8_t>(a.r, b.c) && t.at<uint8_t>(b.r, a.c));
        }
    }
    const GridPath q = AStar(t, {0, 0}, {{4, 4}, {0, 3}});      // multi-goal: the nearer goal wins
    ASSERT_TRUE(q.found);
    EXPECT_EQ(q.goal_index, 1);
    EXPECT_NEAR(q.length_cells, 3.0, 1e-12);
    t.at<uint8_t>(4, 4) = 0;
    EXPECT_FALSE(AStar(t, {0, 0}, {{4, 4}}).found);
    // An untraversable goal earlier in the list must not shift the reported index.
    const GridPath r = AStar(t, {0, 0}, {{4, 4}, {0, 3}});
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.goal_index, 1);
}

TEST(Planning, FrontiersClusterAndDropSmallOnes) {
    cv::Mat s(10, 20, CV_8U, cv::Scalar(kUnknown));
    s(cv::Rect(0, 0, 10, 10)).setTo(kFree);   // free left half: a 10-cell frontier column at c = 9
    s.at<uint8_t>(2, 15) = kFree;             // a lone free cell in the unknown: a 1-cell cluster
    const FrontierSet f = Frontiers(s, 3);
    ASSERT_EQ(f.clusters.size(), 1u);
    EXPECT_EQ(f.clusters[0].cells, 10);
    EXPECT_NEAR(f.clusters[0].centroid_c, 9.0, 1e-12);
    EXPECT_EQ(f.rejected_small, 1);
}
