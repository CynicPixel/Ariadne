#pragma once
// ---------------------------------------------------------------------------
// Grid A* (arch §22.1 placeholder until D* Lite; design MISSION_1.8): 8-connected
// over a traversable mask, step costs 1 and sqrt(2) cells, a diagonal step only if
// both orthogonal neighbours are traversable (worldgen's "no squeezing between two
// corners", so lengths match the GT geodesic). Multi-goal: the heuristic is the
// octile distance to the nearest goal (admissible and consistent), so the result is
// the shortest path to ANY goal. Deterministic: ties broken by (f, g, cell index).
// ---------------------------------------------------------------------------
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

namespace ariadne::planning {

struct Cell {
    int r = 0, c = 0;   // row (north index), column (east index)
    bool operator==(const Cell& o) const { return r == o.r && c == o.c; }
};

struct GridPath {
    bool found = false;
    std::vector<Cell> cells;   // start ... goal
    double length_cells = 0;
    int goal_index = -1;       // which goal was reached: an index into the CALLER's goal list
    size_t expanded = 0;
};

inline double Octile(const Cell& a, const Cell& b) {
    const double dr = std::abs(a.r - b.r), dc = std::abs(a.c - b.c);
    return std::max(dr, dc) + (std::sqrt(2.0) - 1.0) * std::min(dr, dc);
}

inline GridPath AStar(const cv::Mat& trav, const Cell& start, const std::vector<Cell>& goals) {
    GridPath out;
    const int H = trav.rows, W = trav.cols;
    auto ok = [&](int r, int c) { return r >= 0 && c >= 0 && r < H && c < W && trav.at<uint8_t>(r, c); };
    if (!ok(start.r, start.c)) return out;
    std::vector<Cell> gs;
    std::vector<int> gidx;   // gs[k] is goals[gidx[k]]: untraversable goals are dropped
    for (size_t i = 0; i < goals.size(); ++i) {
        if (ok(goals[i].r, goals[i].c)) gs.push_back(goals[i]), gidx.push_back(int(i));
    }
    if (gs.empty()) return out;
    auto h = [&](const Cell& a) {
        double m = std::numeric_limits<double>::infinity();
        for (const auto& g : gs) m = std::min(m, Octile(a, g));
        return m;
    };
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> gcost(size_t(H) * size_t(W), inf);
    std::vector<int64_t> parent(size_t(H) * size_t(W), -1);
    std::vector<uint8_t> closed(size_t(H) * size_t(W), 0);
    using Node = std::tuple<double, double, int64_t>;   // f, g, index: lexicographic tie-break
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    const int64_t s = int64_t(start.r) * W + start.c;
    gcost[size_t(s)] = 0;
    open.emplace(h(start), 0.0, s);
    static const int dr[8] = {0, 0, 1, -1, 1, 1, -1, -1}, dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    while (!open.empty()) {
        const auto [f, g, u] = open.top();
        open.pop();
        if (closed[size_t(u)]) continue;
        closed[size_t(u)] = 1;
        ++out.expanded;
        const Cell cu{int(u / W), int(u % W)};
        for (size_t k = 0; k < gs.size(); ++k) {
            if (gs[k] == cu) {
                out.found = true;
                out.goal_index = gidx[k];
                out.length_cells = g;
                for (int64_t v = u; v >= 0; v = parent[size_t(v)]) out.cells.push_back({int(v / W), int(v % W)});
                std::reverse(out.cells.begin(), out.cells.end());
                return out;
            }
        }
        for (int k = 0; k < 8; ++k) {
            const int r = cu.r + dr[k], c = cu.c + dc[k];
            if (!ok(r, c)) continue;
            if (k >= 4 && (!ok(cu.r + dr[k], cu.c) || !ok(cu.r, cu.c + dc[k]))) continue;   // corner rule
            const int64_t v = int64_t(r) * W + c;
            if (closed[size_t(v)]) continue;
            const double ng = g + (k >= 4 ? std::sqrt(2.0) : 1.0);
            if (ng < gcost[size_t(v)]) {
                gcost[size_t(v)] = ng;
                parent[size_t(v)] = u;
                open.emplace(ng + h({r, c}), ng, v);
            }
        }
    }
    return out;
}

/// Dijkstra distances (cells) from `start` over `trav`, same moves and corner rule as AStar
/// (worldgen's `geodesic`). Unreachable = +inf. CV_64F.
inline cv::Mat GridDistances(const cv::Mat& trav, const Cell& start) {
    const int H = trav.rows, W = trav.cols;
    const double inf = std::numeric_limits<double>::infinity();
    cv::Mat d(H, W, CV_64F, cv::Scalar(inf));
    auto ok = [&](int r, int c) { return r >= 0 && c >= 0 && r < H && c < W && trav.at<uint8_t>(r, c); };
    if (!ok(start.r, start.c)) return d;
    using Node = std::pair<double, int64_t>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
    d.at<double>(start.r, start.c) = 0;
    open.emplace(0.0, int64_t(start.r) * W + start.c);
    static const int dr[8] = {0, 0, 1, -1, 1, 1, -1, -1}, dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
    while (!open.empty()) {
        const auto [g, u] = open.top();
        open.pop();
        const int ur = int(u / W), uc = int(u % W);
        if (g > d.at<double>(ur, uc)) continue;
        for (int k = 0; k < 8; ++k) {
            const int r = ur + dr[k], c = uc + dc[k];
            if (!ok(r, c)) continue;
            if (k >= 4 && (!ok(ur + dr[k], uc) || !ok(ur, uc + dc[k]))) continue;
            const double ng = g + (k >= 4 ? std::sqrt(2.0) : 1.0);
            if (ng < d.at<double>(r, c)) {
                d.at<double>(r, c) = ng;
                open.emplace(ng, int64_t(r) * W + c);
            }
        }
    }
    return d;
}

}  // namespace ariadne::planning
