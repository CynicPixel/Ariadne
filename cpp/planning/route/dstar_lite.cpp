#include "planning/route/dstar_lite.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ariadne::planning {
namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
const int kDr[8] = {0, 0, 1, -1, 1, 1, -1, -1}, kDc[8] = {1, -1, 0, 0, 1, -1, 1, -1};
}  // namespace

DStarLite::DStarLite(const cv::Mat& trav, const cv::Mat& penalty, const Cell& start, const std::vector<Cell>& goals)
    : H_(trav.rows), W_(trav.cols), start_(start), goals_(goals) {
    if (trav.type() != CV_8U || penalty.type() != CV_32F || penalty.size() != trav.size()) {
        throw std::invalid_argument("DStarLite: trav CV_8U and penalty CV_32F of the same size");
    }
    if (start.r < 0 || start.c < 0 || start.r >= H_ || start.c >= W_) throw std::invalid_argument("DStarLite: start");
    trav_ = trav.clone(), penalty_ = penalty.clone();
    const size_t n = size_t(H_) * size_t(W_);
    g_.assign(n, kInf), rhs_.assign(n, kInf), qkey_.assign(n, {kInf, kInf}), queued_.assign(n, 0), is_goal_.assign(n, 0);
    for (const auto& q : goals_) {
        if (q.r < 0 || q.c < 0 || q.r >= H_ || q.c >= W_) throw std::invalid_argument("DStarLite: goal outside the grid");
        is_goal_[Idx(q.r, q.c)] = 1;
    }
    // Initialize(): rhs(goal) = 0 for every traversable goal, queued with its key.
    for (size_t s = 0; s < n; ++s) {
        if (is_goal_[s]) UpdateVertex(s);
    }
    ComputeShortestPath();
}

double DStarLite::EdgeCost(int r0, int c0, int r1, int c1) const {
    if (!Ok(r0, c0) || !Ok(r1, c1)) return kInf;
    const int dr = r1 - r0, dc = c1 - c0;
    if (std::abs(dr) > 1 || std::abs(dc) > 1 || (dr == 0 && dc == 0)) return kInf;
    const bool diag = dr != 0 && dc != 0;
    if (diag && (!Ok(r0 + dr, c0) || !Ok(r0, c0 + dc))) return kInf;   // corner rule
    const double len = diag ? std::sqrt(2.0) : 1.0;
    return len * (1.0 + 0.5 * (double(penalty_.at<float>(r0, c0)) + double(penalty_.at<float>(r1, c1))));
}

DStarLite::Key DStarLite::CalculateKey(size_t s) const {
    const double m = std::min(g_[s], rhs_[s]);
    const Cell c{int(s / size_t(W_)), int(s % size_t(W_))};
    return {m + Octile(start_, c), m};   // k_m = 0: the start never moves (EGRESS_4.5 §1)
}

void DStarLite::UpdateVertex(size_t u) {
    const int r = int(u / size_t(W_)), c = int(u % size_t(W_));
    if (!Ok(r, c)) {
        rhs_[u] = kInf;
    } else if (is_goal_[u]) {
        rhs_[u] = 0;
    } else {
        double best = kInf;
        for (int k = 0; k < 8; ++k) {
            const int rr = r + kDr[k], cc = c + kDc[k];
            const double e = EdgeCost(r, c, rr, cc);
            if (std::isfinite(e)) best = std::min(best, e + g_[Idx(rr, cc)]);
        }
        rhs_[u] = best;
    }
    if (queued_[u]) {
        U_.erase({qkey_[u].first, qkey_[u].second, u});
        queued_[u] = 0;
    }
    if (g_[u] != rhs_[u]) {
        qkey_[u] = CalculateKey(u);
        U_.insert({qkey_[u].first, qkey_[u].second, u});
        queued_[u] = 1;
    }
}

void DStarLite::ComputeShortestPath() {
    const size_t s0 = Idx(start_.r, start_.c);
    while (!U_.empty()) {
        const auto [k1, k2, u] = *U_.begin();
        if (!(Key{k1, k2} < CalculateKey(s0)) && rhs_[s0] == g_[s0]) break;
        U_.erase(U_.begin());
        queued_[u] = 0;
        ++expanded_;
        const Key k_new = CalculateKey(u);
        if (Key{k1, k2} < k_new) {   // cannot happen with k_m = 0 and keys refreshed on every update; kept per Fig. 3
            qkey_[u] = k_new;
            U_.insert({k_new.first, k_new.second, u});
            queued_[u] = 1;
            continue;
        }
        const int r = int(u / size_t(W_)), c = int(u % size_t(W_));
        if (g_[u] > rhs_[u]) {
            g_[u] = rhs_[u];
        } else {
            g_[u] = kInf;
            UpdateVertex(u);
        }
        for (int k = 0; k < 8; ++k) {   // Pred(u) = Succ(u): symmetric costs
            const int rr = r + kDr[k], cc = c + kDc[k];
            if (rr >= 0 && cc >= 0 && rr < H_ && cc < W_) UpdateVertex(Idx(rr, cc));
        }
    }
}

std::vector<Cell> DStarLite::Update(const cv::Mat& trav, const cv::Mat& penalty) {
    if (trav.size() != trav_.size() || penalty.size() != trav_.size() || trav.type() != CV_8U || penalty.type() != CV_32F) {
        throw std::invalid_argument("DStarLite::Update: grids changed shape or type");
    }
    std::vector<Cell> changed;
    std::vector<uint8_t> touch(size_t(H_) * size_t(W_), 0);
    for (int r = 0; r < H_; ++r) {
        for (int c = 0; c < W_; ++c) {
            const bool t0 = trav_.at<uint8_t>(r, c) != 0, t1 = trav.at<uint8_t>(r, c) != 0;
            if (t0 == t1 && penalty_.at<float>(r, c) == penalty.at<float>(r, c)) continue;
            changed.push_back({r, c});
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc) {
                    const int rr = r + dr, cc = c + dc;
                    if (rr >= 0 && cc >= 0 && rr < H_ && cc < W_) touch[Idx(rr, cc)] = 1;
                }
        }
    }
    if (changed.empty()) return changed;
    trav.copyTo(trav_), penalty.copyTo(penalty_);
    for (size_t s = 0; s < touch.size(); ++s) {
        if (touch[s]) UpdateVertex(s);
    }
    ComputeShortestPath();
    return changed;
}

bool DStarLite::Found() const { return std::isfinite(g_[Idx(start_.r, start_.c)]); }

double DStarLite::Cost() const { return g_[Idx(start_.r, start_.c)]; }

int DStarLite::GoalIndex(const Cell& c) const {
    for (size_t i = 0; i < goals_.size(); ++i) {
        if (goals_[i] == c) return int(i);
    }
    return -1;
}

std::vector<Cell> DStarLite::Path() const {
    std::vector<Cell> p;
    if (!Found()) return p;
    Cell s = start_;
    p.push_back(s);
    const size_t cap = size_t(H_) * size_t(W_);
    while (!is_goal_[Idx(s.r, s.c)] || g_[Idx(s.r, s.c)] != 0) {
        double best = kInf;
        Cell nxt = s;
        for (int k = 0; k < 8; ++k) {   // argmin c(s, s') + g(s'); ties by neighbour order (deterministic)
            const int rr = s.r + kDr[k], cc = s.c + kDc[k];
            const double e = EdgeCost(s.r, s.c, rr, cc);
            if (!std::isfinite(e)) continue;
            const double v = e + g_[Idx(rr, cc)];
            if (v < best) best = v, nxt = {rr, cc};
        }
        if (!std::isfinite(best) || p.size() > cap) return {};   // inconsistent state: never return a broken route
        s = nxt;
        p.push_back(s);
    }
    return p;
}

}  // namespace ariadne::planning
