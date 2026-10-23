#include "planning/mission/multiagent.hpp"

#include "planning/volumetric/occupancy_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>

namespace ariadne::planning::multi {

cv::Mat EncodeCells(const TraversabilityMaps& m) {
    cv::Mat out(m.slab.size(), CV_8U, cv::Scalar(kCellUnknown));
    for (int r = 0; r < m.slab.rows; ++r)
        for (int c = 0; c < m.slab.cols; ++c) {
            const uint8_t s = m.slab.at<uint8_t>(r, c);
            if (s == kOccupied) out.at<uint8_t>(r, c) = kCellOccupied;
            else if (s == kFree) out.at<uint8_t>(r, c) = m.trav_r.at<uint8_t>(r, c) ? kCellFreeResponder : kCellFree;
        }
    return out;
}

SlabExporter::SlabExporter(int sender, const GridFrame& g)
    : sender_(sender), g_(g), bw_((g.cols + kBlock - 1) / kBlock), bh_((g.rows + kBlock - 1) / kBlock),
      codes_(g.rows, g.cols, CV_8U, cv::Scalar(kCellUnknown)), changed_(size_t(bw_ * bh_), 0) {}

uint64_t SlabExporter::Update(const TraversabilityMaps& m) {
    const cv::Mat now = EncodeCells(m);
    if (now.size() != codes_.size()) throw std::invalid_argument("SlabExporter: maps are not on the briefing grid");
    bool any = false;
    for (int by = 0; by < bh_; ++by)
        for (int bx = 0; bx < bw_; ++bx) {
            const cv::Rect rc(bx * kBlock, by * kBlock, std::min(kBlock, g_.cols - bx * kBlock), std::min(kBlock, g_.rows - by * kBlock));
            if (cv::countNonZero(now(rc) != codes_(rc)) == 0) continue;
            if (!any) ++version_, any = true;
            now(rc).copyTo(codes_(rc));
            changed_[size_t(by * bw_ + bx)] = version_;
        }
    return version_;
}

SubmapPatch SlabExporter::Export(uint64_t since, uint32_t epoch, int64_t t_ns) const {
    SubmapPatch p;
    p.sender = sender_, p.epoch = epoch, p.version = version_, p.t_ns = t_ns;
    for (int by = 0; by < bh_; ++by)
        for (int bx = 0; bx < bw_; ++bx) {
            if (changed_[size_t(by * bw_ + bx)] <= since) continue;
            SlabBlock b;
            b.bx = bx, b.by = by;
            b.cells.assign(size_t(kBlock * kBlock), kCellUnknown);   // cells past the grid's edge stay unknown
            for (int r = 0; r < kBlock && by * kBlock + r < g_.rows; ++r)
                for (int c = 0; c < kBlock && bx * kBlock + c < g_.cols; ++c)
                    b.cells[size_t(r * kBlock + c)] = codes_.at<uint8_t>(by * kBlock + r, bx * kBlock + c);
            p.blocks.push_back(std::move(b));
        }
    return p;
}

ReceivedLayers::ReceivedLayers(int self, int n_agents, const GridFrame& g)
    : self_(self), g_(g), acked_(size_t(n_agents), 0), known_(g.rows, g.cols, CV_8U, cv::Scalar(0)) {
    for (int k = 0; k < n_agents; ++k) layer_.emplace_back(g.rows, g.cols, CV_8U, cv::Scalar(kCellUnknown));
}

void ReceivedLayers::Integrate(const SubmapPatch& p, const Rigid2& T) {
    if (p.sender == self_) throw std::invalid_argument("ReceivedLayers: an agent does not receive its own patch");
    cv::Mat& L = layer_.at(size_t(p.sender));
    const double cs = std::cos(T.yaw), sn = std::sin(T.yaw);
    for (const SlabBlock& b : p.blocks) {
        if (b.cells.size() != size_t(kBlock * kBlock)) throw std::invalid_argument("ReceivedLayers: bad block");
        for (int r = 0; r < kBlock; ++r)
            for (int c = 0; c < kBlock; ++c) {
                int rr = b.by * kBlock + r, cc = b.bx * kBlock + c;
                if (!T.Identity()) {   // nearest cell of the cell centre in G [DESIGN: small corrections, no resampling]
                    const double x = g_.ox + (cc + 0.5) * g_.cell, y = g_.oy + (rr + 0.5) * g_.cell;
                    const double gx = cs * x - sn * y + T.x, gy = sn * x + cs * y + T.y;
                    cc = int(std::floor((gx - g_.ox) / g_.cell)), rr = int(std::floor((gy - g_.oy) / g_.cell));
                }
                if (rr < 0 || cc < 0 || rr >= g_.rows || cc >= g_.cols) continue;
                L.at<uint8_t>(rr, cc) = b.cells[size_t(r * kBlock + c)];
                uint8_t k = 0;
                for (size_t j = 0; j < layer_.size() && !k; ++j) k = layer_[j].at<uint8_t>(rr, cc) != kCellUnknown;
                known_.at<uint8_t>(rr, cc) = k;
            }
    }
    acked_.at(size_t(p.sender)) = std::max(acked_.at(size_t(p.sender)), p.version);
}

cv::Mat ReceivedLayers::Merged(const cv::Mat* own) const {
    cv::Mat out(g_.rows, g_.cols, CV_8U, cv::Scalar(kCellUnknown));
    // Precedence: occupied > free with the responder bit > free > unknown.
    static constexpr uint8_t kRank[4] = {0, 3, 1, 2};   // indexed by CellCode
    auto Fold = [&](const cv::Mat& L) {
        for (int r = 0; r < g_.rows; ++r)
            for (int c = 0; c < g_.cols; ++c) {
                const uint8_t v = L.at<uint8_t>(r, c);
                uint8_t& o = out.at<uint8_t>(r, c);
                if (kRank[v & 3] > kRank[o & 3]) o = v;
            }
    };
    if (own) Fold(*own);
    for (const auto& L : layer_) Fold(L);
    return out;
}

TaskBid BuildBid(int sender, uint32_t epoch, uint64_t seq, int64_t t_ns, const NbvPlan& nbv, int max_options,
                 const Rigid2& T) {
    TaskBid b;
    b.sender = sender, b.epoch = epoch, b.seq = seq, b.t_ns = t_ns, b.lambda_path = float(nbv.lambda_path);
    const double cs = std::cos(T.yaw), sn = std::sin(T.yaw);
    for (const TaskCandidate& o : nbv.options) {   // sorted by utility: the first view of a task is its best
        if (int(b.options.size()) >= max_options) break;
        if (o.source == TaskSource::kReturn || o.source == TaskSource::kHold) continue;
        BidOption x;
        x.task_key = o.task_id;
        x.goal_x = cs * o.goal_xy.first - sn * o.goal_xy.second + T.x;
        x.goal_y = sn * o.goal_xy.first + cs * o.goal_xy.second + T.y;
        for (int k = 0; k < 5; ++k) x.ig_raw[k] = float(o.ig_raw[size_t(k)]);
        x.path_m = float(o.path_m), x.cost_bandwidth = float(o.cost[1]), x.cost_risk = float(o.cost[4]);
        x.risk = float(o.risk), x.source = uint8_t(o.source);
        b.options.push_back(x);
    }
    return b;
}

TaskConsensus BrokerAssign(const std::vector<TaskBid>& bids, const ComponentInfo& comp, double r_same, int64_t t_ns,
                           uint64_t seq) {
    TaskConsensus out;
    out.broker = comp.broker, out.epoch = comp.epoch, out.seq = seq, out.t_ns = t_ns, out.members = comp.members;
    // One bid per member, the newest; the auction's agent index = the member's position in comp.members.
    std::vector<const TaskBid*> by(comp.members.size(), nullptr);
    for (const TaskBid& b : bids) {
        if (b.epoch != comp.epoch) continue;
        const auto it = std::find(comp.members.begin(), comp.members.end(), b.sender);
        if (it == comp.members.end()) continue;
        const TaskBid*& slot = by[size_t(it - comp.members.begin())];
        if (!slot || b.t_ns > slot->t_ns || (b.t_ns == slot->t_ns && b.seq > slot->seq)) slot = &b;
    }
    for (const TaskBid* b : by) if (b) out.bids_t_ns = std::max(out.bids_t_ns, b->t_ns);
    // Common scales (BuildAuction's rule): per term the max over bidders of that bidder's p90.
    std::array<double, 8> P{};
    for (const TaskBid* b : by) {
        if (!b) continue;
        for (int k = 0; k < 8; ++k) {
            std::vector<double> v;
            for (const BidOption& o : b->options)
                v.push_back(k < 5 ? o.ig_raw[k] : (k == 5 ? o.path_m : (k == 6 ? o.cost_bandwidth : o.cost_risk)));
            P[size_t(k)] = std::max(P[size_t(k)], P90Scale(v));
        }
    }
    // Tasks: greedy spatial clustering in bid order (member order, then each bid's utility order).
    std::vector<std::pair<double, double>> task_goal;
    const size_t A = comp.members.size();
    std::vector<std::vector<int>> task_of(A);
    for (size_t a = 0; a < A; ++a) {
        if (!by[a]) continue;
        for (const BidOption& o : by[a]->options) {
            int j = -1;
            for (size_t t = 0; t < task_goal.size() && j < 0; ++t)
                if (std::hypot(o.goal_x - task_goal[t].first, o.goal_y - task_goal[t].second) < r_same) j = int(t);
            if (j < 0) j = int(task_goal.size()), task_goal.push_back({o.goal_x, o.goal_y});
            task_of[a].push_back(j);
        }
    }
    AuctionInstance in;
    in.n_agents = int(A), in.max_bundle = 1;
    for (size_t t = 0; t < task_goal.size(); ++t) in.task_ids.push_back(int64_t(t));
    const size_t T = task_goal.size();
    in.value.assign(A, std::vector<double>(T, 0.0));
    in.d0.assign(A, std::vector<double>(T, 0.0));
    in.leg.assign(A, std::vector<std::vector<double>>(T, std::vector<double>(T, 0.0)));
    std::vector<std::vector<int>> view(A, std::vector<int>(T, -1));
    for (size_t a = 0; a < A; ++a) {
        in.rate.push_back(by[a] && P[5] > 0 ? by[a]->lambda_path / P[5] : 0.0);
        if (!by[a]) continue;
        for (size_t i = 0; i < by[a]->options.size(); ++i) {
            const size_t j = size_t(task_of[a][i]);
            if (view[a][j] >= 0) continue;   // the first (best) of this member's options in the task
            const BidOption& o = by[a]->options[i];
            double g = 0;
            for (int k = 0; k < 5; ++k) g += P[size_t(k)] > 0 ? o.ig_raw[k] / P[size_t(k)] : 0.0;
            const double cb = P[6] > 0 ? o.cost_bandwidth / P[6] : 0.0, cr = P[7] > 0 ? o.cost_risk / P[7] : 0.0;
            in.value[a][j] = std::max(0.0, g - cb - cr);
            in.d0[a][j] = o.path_m;
            view[a][j] = int(i);
        }
    }
    const Allocation al = SequentialGreedy(in);
    for (size_t a = 0; a < A; ++a) {
        if (!by[a] || al.agents[a].path.empty()) continue;
        const BidOption& o = by[a]->options[size_t(view[a][size_t(al.agents[a].path.front())])];
        out.assignments.push_back({comp.members[a], o.task_key, o.goal_x, o.goal_y});
    }
    return out;
}

Pick ApplyConsensus(const TaskConsensus* c, int self, uint32_t epoch, int64_t now_ns, double timeout_s,
                    const std::vector<TaskCandidate>& options, double r_same) {
    Pick p;
    auto Own = [&](const std::string& why) {
        p.option = options.empty() ? -1 : 0, p.coordinated = false, p.why = "uncoordinated: " + why;
        return p;
    };
    if (!c) return Own("no consensus received");
    if (c->epoch != epoch) return Own("consensus from another epoch");
    if (double(now_ns - c->t_ns) * 1e-9 > timeout_s) return Own("consensus older than the timeout");
    const Assignment* mine = nullptr;
    for (const Assignment& a : c->assignments)
        if (a.agent == self) mine = &a;
    if (!mine) return Own("no assignment for this agent");
    for (size_t i = 0; i < options.size(); ++i)
        if (options[i].task_id == mine->task_key) {
            p.option = int(i), p.coordinated = true, p.why = "coordinated";
            return p;
        }
    for (size_t i = 0; i < options.size(); ++i)   // the task re-keyed since the bid (map changed): the nearest goal
        if (std::hypot(options[i].goal_xy.first - mine->goal_x, options[i].goal_xy.second - mine->goal_y) < r_same) {
            p.option = int(i), p.coordinated = true, p.why = "coordinated";
            return p;
        }
    return Own("the assigned task is gone");
}

}  // namespace ariadne::planning::multi
