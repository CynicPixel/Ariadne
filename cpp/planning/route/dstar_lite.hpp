#pragma once
// ---------------------------------------------------------------------------
// D* Lite on the egress lattice (arch §22.1/§22.3; design egress.md, D113).
//
// Koenig & Likhachev, AAAI 2002, Figure 3, transcribed from the rendered page:
//   CalculateKey(s)     = [min(g, rhs) + h(s_start, s) + k_m ; min(g, rhs)]
//   UpdateVertex(u)     : u != goal -> rhs(u) = min_{s' in Succ(u)} c(u, s') + g(s'); re-queue iff g != rhs
//   ComputeShortestPath : while TopKey < Key(s_start) or rhs(s_start) != g(s_start): pop u;
//                         overconsistent (g > rhs) -> g = rhs, UpdateVertex(preds);
//                         else g = inf, UpdateVertex(preds and u)
// The search runs FROM the goals, so g is cost-to-goal and the route is read greedily from s_start.
//
// Our case (egress.md §1): s_start = the target cell, fixed (nobody walks the route while the agents map), so
// k_m = 0 and this is the incremental core. Goals = every exit: each traversable exit is seeded rhs = 0 (a
// multi-source backward search), and h(s) = octile(target, s) is consistent because every edge cost is at
// least its octile length. Costs are in cells.
//
// Lattice: 1.8's (route/astar.hpp): 8-connected over a traversable mask, a diagonal only if both orthogonal
// neighbours are traversable. Edge cost c(x, y) = |x - y| * (1 + (m(x) + m(y)) / 2), m >= 0 a per-cell
// penalty (the §22.1 terms, built by the caller). Costs are symmetric, so Succ = Pred.
//
// Update(): the caller's new (traversable, penalty) grids are diffed against the current ones; every changed
// cell re-runs UpdateVertex on its whole 3x3 block (a cost or traversability change at x alters all 8 edges
// at x and, through the corner rule, the diagonals between x's neighbours, all of whose endpoints are in the
// block). Deterministic: vertices are updated in index order and the queue breaks ties by vertex index.
// ---------------------------------------------------------------------------
#include "planning/route/astar.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <set>
#include <tuple>
#include <vector>

namespace ariadne::planning {

class DStarLite {
public:
    /// `trav` CV_8U (nonzero = traversable); `penalty` CV_32F, >= 0, same size. Runs the first search.
    DStarLite(const cv::Mat& trav, const cv::Mat& penalty, const Cell& start, const std::vector<Cell>& goals);

    /// Diff against the current grids, 3x3 UpdateVertex per changed cell, recompute. Returns the changed cells.
    std::vector<Cell> Update(const cv::Mat& trav, const cv::Mat& penalty);

    bool Found() const;                     // g(start) finite
    double Cost() const;                    // g(start), cells (inf if none)
    std::vector<Cell> Path() const;         // start ... a goal, by greedy descent of c + g; empty if none
    int GoalIndex(const Cell& c) const;     // index into the caller's goal list, -1 if not a goal
    size_t Expanded() const { return expanded_; }   // vertices popped, summed over every search so far
    double G(const Cell& c) const { return g_[Idx(c.r, c.c)]; }

    /// Edge cost on the current grids (inf if the move is not allowed). Exposed for the oracle tests.
    double EdgeCost(int r0, int c0, int r1, int c1) const;

private:
    using Key = std::pair<double, double>;
    size_t Idx(int r, int c) const { return size_t(r) * size_t(W_) + size_t(c); }
    bool Ok(int r, int c) const { return r >= 0 && c >= 0 && r < H_ && c < W_ && trav_.at<uint8_t>(r, c); }
    Key CalculateKey(size_t s) const;
    void UpdateVertex(size_t u);
    void ComputeShortestPath();

    int H_ = 0, W_ = 0;
    cv::Mat trav_, penalty_;
    Cell start_;
    std::vector<Cell> goals_;
    std::vector<uint8_t> is_goal_;
    std::vector<double> g_, rhs_;
    std::vector<Key> qkey_;           // the key a queued vertex was inserted with
    std::vector<uint8_t> queued_;
    std::set<std::tuple<double, double, size_t>> U_;   // (k1, k2, vertex): lexicographic, ties by vertex index
    size_t expanded_ = 0;
};

}  // namespace ariadne::planning
