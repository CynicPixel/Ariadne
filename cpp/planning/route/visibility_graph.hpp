#pragma once
// ---------------------------------------------------------------------------
// 4.6 FAR-style visibility graph (arch §21.6; FAR, Yang et al. IROS 2022; design exploration.md, D116) on a 2D
// traversable mask (CV_8U, nonzero = free; row 0 = south, as every planning layer).
//
//   polygons  cv::findContours(blocked, RETR_CCOMP) + cv::approxPolyDP(eps); holes (free pockets inside blocked
//             components) have the opposite winding, so a vertex is kept iff its interior angle on the BLOCKED side
//             is < 180 deg: convex w.r.t. an outer contour's region, reflex w.r.t. a hole's;
//   nodes     each kept vertex offset `offset` cells along its outward bisector into free space;
//   edges     node pairs with line of sight (4-connected cv::LineIterator on the mask: no corner cut), Euclidean;
//   update    contours are re-extracted globally (one O(cells) pass); the O(V^2) visibility is incremental: edges
//             between unchanged nodes are kept unless their box meets a changed region, the rest are re-tested.
//   query     start and goals joined to every node they see, Dijkstra (multi-goal), the polyline back.
// ---------------------------------------------------------------------------
#include "planning/route/astar.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>

namespace ariadne::planning {

struct VisGraphParams {
    double eps_cells = 1.0;   // approxPolyDP tolerance
    int offset_cells = 1;     // node offset from its polygon vertex into free space
    bool all_vertices = false;   // test reference only: keep every polygon vertex (no convexity filter)
};

struct VisRoute {
    bool found = false;
    std::vector<Cell> cells;   // polyline vertices: start ... goal
    double length_cells = 0;   // Euclidean along the polyline
    int goal_index = -1;
};

/// 4-connected line of sight between two cells over `free` (both endpoints included), clear in BOTH raster
/// directions (a 4-connected raster is not symmetric).
bool LineOfSight(const cv::Mat& free, const Cell& a, const Cell& b);

class VisibilityGraph {
public:
    explicit VisibilityGraph(VisGraphParams p = {}) : p_(p) {}
    /// Full build on `free`.
    void Build(const cv::Mat& free);
    /// Incremental: diff against the previous mask; returns the number of line-of-sight tests performed.
    size_t Update(const cv::Mat& free);
    VisRoute Query(const Cell& start, const std::vector<Cell>& goals) const;
    const std::vector<Cell>& Nodes() const { return nodes_; }
    size_t Edges() const;
    size_t LastLosTests() const { return los_tests_; }

private:
    std::vector<Cell> ExtractNodes(const cv::Mat& free) const;
    VisGraphParams p_;
    cv::Mat free_;
    std::vector<Cell> nodes_;
    std::vector<std::vector<int>> adj_;   // symmetric adjacency (node indices)
    size_t los_tests_ = 0;
};

}  // namespace ariadne::planning
