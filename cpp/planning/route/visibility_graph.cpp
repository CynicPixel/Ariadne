#include "planning/route/visibility_graph.hpp"

#include <opencv2/geometry/2d.hpp>   // OpenCV 5: approxPolyDP, contourArea moved here from imgproc
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>

namespace ariadne::planning {
namespace {

bool Free(const cv::Mat& m, int r, int c) { return r >= 0 && c >= 0 && r < m.rows && c < m.cols && m.at<uint8_t>(r, c); }

double Dist(const Cell& a, const Cell& b) { return std::hypot(double(a.r - b.r), double(a.c - b.c)); }

}  // namespace

bool LineOfSight(const cv::Mat& free, const Cell& a, const Cell& b) {
    if (!Free(free, a.r, a.c) || !Free(free, b.r, b.c)) return false;
    // A 4-connected raster is not symmetric (a -> b and b -> a can visit different cells): a segment must be clear
    // whichever way it is drawn, so both are tested.
    for (const auto& [p, q] : {std::pair{a, b}, std::pair{b, a}}) {
        cv::LineIterator it(free, cv::Point(p.c, p.r), cv::Point(q.c, q.r), 4);
        for (int k = 0; k < it.count; ++k, ++it)
            if (free.at<uint8_t>(it.pos()) == 0) return false;
    }
    return true;
}

std::vector<Cell> VisibilityGraph::ExtractNodes(const cv::Mat& free) const {
    cv::Mat blocked = free == 0;   // 255 where blocked
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(blocked, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);
    std::vector<Cell> nodes;
    for (size_t ci = 0; ci < contours.size(); ++ci) {
        const bool hole = hierarchy[ci][3] >= 0;   // RETR_CCOMP: a contour with a parent is a hole
        std::vector<cv::Point> poly;
        cv::approxPolyDP(contours[ci], poly, p_.eps_cells, true);
        if (poly.size() < 3) {   // a 1-2 vertex "polygon" (a single blocked cell or a thin line): every point bends
            for (const auto& v : poly) {
                for (int dr = -1; dr <= 1; dr += 2)
                    for (int dc = -1; dc <= 1; dc += 2) {
                        const int r = v.y + dr * p_.offset_cells, c = v.x + dc * p_.offset_cells;
                        if (Free(free, r, c)) nodes.push_back({r, c});
                    }
            }
            continue;
        }
        // The polygon's own orientation (image coordinates), so the convexity sign means the same for every contour.
        const double area = cv::contourArea(poly, true);
        const double orient = area >= 0 ? 1.0 : -1.0;
        const size_t n = poly.size();
        for (size_t i = 0; i < n; ++i) {
            const cv::Point2d p = poly[(i + n - 1) % n], v = poly[i], q = poly[(i + 1) % n];
            const double cross = (v.x - p.x) * (q.y - v.y) - (v.y - p.y) * (q.x - v.x);
            if (cross == 0) continue;   // collinear: not a corner
            const bool convex_wrt_enclosed = cross * orient > 0;
            // Outer contour encloses BLOCKED cells: keep its convex corners. A hole encloses FREE cells: its corners
            // that bulge into free space are the reflex ones w.r.t. the hole.
            if (!p_.all_vertices && hole == convex_wrt_enclosed) continue;
            cv::Point2d u1 = v - p, u2 = v - q;
            u1 /= std::hypot(u1.x, u1.y), u2 /= std::hypot(u2.x, u2.y);
            cv::Point2d b = u1 + u2;   // the outward bisector at a corner bulging into free space
            const double bn = std::hypot(b.x, b.y);
            if (bn < 1e-9) continue;
            b /= bn;
            const int r = int(std::lround(v.y + b.y * p_.offset_cells * std::sqrt(2.0)));
            const int c = int(std::lround(v.x + b.x * p_.offset_cells * std::sqrt(2.0)));
            if (Free(free, r, c)) nodes.push_back({r, c});
        }
    }
    std::sort(nodes.begin(), nodes.end(), [](const Cell& a, const Cell& b) { return a.r != b.r ? a.r < b.r : a.c < b.c; });
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    return nodes;
}

void VisibilityGraph::Build(const cv::Mat& free) {
    free_ = free.clone();
    nodes_ = ExtractNodes(free_);
    adj_.assign(nodes_.size(), {});
    los_tests_ = 0;
    for (size_t i = 0; i < nodes_.size(); ++i)
        for (size_t j = i + 1; j < nodes_.size(); ++j) {
            ++los_tests_;
            if (LineOfSight(free_, nodes_[i], nodes_[j])) adj_[i].push_back(int(j)), adj_[j].push_back(int(i));
        }
}

size_t VisibilityGraph::Update(const cv::Mat& free) {
    if (free_.empty() || free.size() != free_.size()) {
        Build(free);
        return los_tests_;
    }
    // Changed regions: bounding boxes of the connected components of the diff.
    cv::Mat diff = free != free_;
    cv::Mat labels, stats, cents;
    const int nl = cv::connectedComponentsWithStats(diff, labels, stats, cents, 8);
    std::vector<cv::Rect> boxes;
    for (int l = 1; l < nl; ++l)
        boxes.emplace_back(stats.at<int>(l, cv::CC_STAT_LEFT), stats.at<int>(l, cv::CC_STAT_TOP),
                           stats.at<int>(l, cv::CC_STAT_WIDTH), stats.at<int>(l, cv::CC_STAT_HEIGHT));
    const std::vector<Cell> old_nodes = nodes_;
    const std::vector<std::vector<int>> old_adj = adj_;
    free_ = free.clone();
    nodes_ = ExtractNodes(free_);
    adj_.assign(nodes_.size(), {});
    los_tests_ = 0;
    std::map<std::pair<int, int>, int> old_index;
    for (size_t i = 0; i < old_nodes.size(); ++i) old_index[{old_nodes[i].r, old_nodes[i].c}] = int(i);
    std::vector<int> was(nodes_.size(), -1);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto it = old_index.find({nodes_[i].r, nodes_[i].c});
        if (it != old_index.end()) was[i] = it->second;
    }
    auto Touches = [&](const Cell& a, const Cell& b) {   // the edge's box meets a changed region
        const cv::Rect e(std::min(a.c, b.c), std::min(a.r, b.r), std::abs(a.c - b.c) + 1, std::abs(a.r - b.r) + 1);
        for (const auto& bx : boxes)
            if ((e & bx).area() > 0) return true;
        return false;
    };
    for (size_t i = 0; i < nodes_.size(); ++i)
        for (size_t j = i + 1; j < nodes_.size(); ++j) {
            bool vis;
            if (was[i] >= 0 && was[j] >= 0 && !Touches(nodes_[i], nodes_[j])) {
                const auto& oa = old_adj[size_t(was[i])];
                vis = std::find(oa.begin(), oa.end(), was[j]) != oa.end();   // unchanged: reuse
            } else {
                ++los_tests_;
                vis = LineOfSight(free_, nodes_[i], nodes_[j]);
            }
            if (vis) adj_[i].push_back(int(j)), adj_[j].push_back(int(i));
        }
    return los_tests_;
}

size_t VisibilityGraph::Edges() const {
    size_t e = 0;
    for (const auto& a : adj_) e += a.size();
    return e / 2;
}

VisRoute VisibilityGraph::Query(const Cell& start, const std::vector<Cell>& goals) const {
    VisRoute out;
    const int V = int(nodes_.size()), S = V, G0 = V + 1, N = V + 1 + int(goals.size());
    auto At = [&](int u) -> Cell { return u < V ? nodes_[size_t(u)] : (u == S ? start : goals[size_t(u - G0)]); };
    if (!Free(free_, start.r, start.c)) return out;
    // Edges of the query nodes: start/goals to every node they see, and start to goals directly.
    std::vector<std::vector<int>> extra(static_cast<size_t>(N));
    for (int g = G0; g < N; ++g) {
        if (!Free(free_, At(g).r, At(g).c)) continue;
        if (LineOfSight(free_, start, At(g))) extra[size_t(S)].push_back(g);
        for (int v = 0; v < V; ++v)
            if (LineOfSight(free_, At(g), nodes_[size_t(v)])) extra[size_t(v)].push_back(g);
    }
    for (int v = 0; v < V; ++v)
        if (LineOfSight(free_, start, nodes_[size_t(v)])) extra[size_t(S)].push_back(v);
    // Endpoints that see no node (nor each other) join through the LATTICE, as FAR joins start / goal through its local
    // planner (EXPLORATION_4.6 §10: hazard seed 0's exit sat where no node was in sight; FAR routed to the far exit,
    // 11.1 m vs 4.4 m). Edge cost = the octile lattice distance (>= the executed string-pulled length); the segment is
    // spliced in as the string-pulled A* path at reconstruction, so every executed segment stays line-of-sight valid.
    const double inf = std::numeric_limits<double>::infinity();
    std::map<std::pair<int, int>, double> lattice;   // (endpoint, node) -> octile cells
    auto Isolated = [&](int u) {
        if (!extra[size_t(u)].empty()) return false;
        for (int x = 0; x < N; ++x)
            for (int y : extra[size_t(x)])
                if (y == u) return false;
        return true;
    };
    for (int u = S; u < N; ++u) {
        if (!Free(free_, At(u).r, At(u).c) || !Isolated(u)) continue;
        const cv::Mat d = GridDistances(free_, At(u));
        for (int v = 0; v < V; ++v) {
            const double dv = d.at<double>(nodes_[size_t(v)].r, nodes_[size_t(v)].c);
            if (!std::isfinite(dv)) continue;
            lattice[{u, v}] = dv;
            if (u == S) extra[size_t(S)].push_back(v);
            else extra[size_t(v)].push_back(u);
        }
    }
    std::vector<double> d(size_t(N), inf);
    std::vector<int> par(size_t(N), -1);
    using Q = std::pair<double, int>;
    std::priority_queue<Q, std::vector<Q>, std::greater<Q>> pq;
    d[size_t(S)] = 0, pq.emplace(0.0, S);
    while (!pq.empty()) {
        const auto [du, u] = pq.top();
        pq.pop();
        if (du > d[size_t(u)]) continue;
        if (u >= G0) {   // the nearest goal by path length
            out.found = true, out.goal_index = u - G0;
            std::vector<int> chain;
            for (int x = u; x >= 0; x = par[size_t(x)]) chain.push_back(x);
            std::reverse(chain.begin(), chain.end());
            out.cells.push_back(At(chain[0]));
            for (size_t i = 1; i < chain.size(); ++i) {
                const int a = chain[i - 1], b = chain[i];
                const bool lat = lattice.count({a, b}) || lattice.count({b, a});
                if (!lat) {
                    out.cells.push_back(At(b));
                    continue;
                }
                const GridPath gp = AStar(free_, At(a), {At(b)});   // the lattice approach, string-pulled
                std::vector<Cell> pulled;
                if (gp.found) {
                    pulled.push_back(gp.cells[0]);
                    size_t k = 0;
                    while (k + 1 < gp.cells.size()) {
                        size_t j = k + 1;
                        for (size_t q = gp.cells.size() - 1; q > k + 1; --q)
                            if (LineOfSight(free_, gp.cells[k], gp.cells[q])) { j = q; break; }
                        pulled.push_back(gp.cells[j]);
                        k = j;
                    }
                }
                for (size_t q = 1; q < pulled.size(); ++q) out.cells.push_back(pulled[q]);
            }
            for (size_t i = 1; i < out.cells.size(); ++i) out.length_cells += Dist(out.cells[i - 1], out.cells[i]);
            return out;
        }
        auto Relax = [&](int v) {
            const auto it = lattice.find({u, v}), it2 = lattice.find({v, u});
            const double w = it != lattice.end() ? it->second : (it2 != lattice.end() ? it2->second : Dist(At(u), At(v)));
            const double nd = du + w;
            if (nd < d[size_t(v)]) d[size_t(v)] = nd, par[size_t(v)] = u, pq.emplace(nd, v);
        };
        if (u < V)
            for (int v : adj_[size_t(u)]) Relax(v);
        for (int v : extra[size_t(u)]) Relax(v);
    }
    return out;
}

}  // namespace ariadne::planning
