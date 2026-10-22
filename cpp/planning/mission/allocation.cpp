#include "planning/mission/allocation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ariadne::planning {
namespace {
constexpr double kNegInf = -std::numeric_limits<double>::infinity();
}

double PathScore(const AuctionInstance& in, int a, const std::vector<int>& path) {
    double s = 0, d = 0;
    for (size_t n = 0; n < path.size(); ++n) {
        d += n == 0 ? in.d0[size_t(a)][size_t(path[0])] : in.leg[size_t(a)][size_t(path[n - 1])][size_t(path[n])];
        s += in.value[size_t(a)][size_t(path[n])] * std::exp(-in.rate[size_t(a)] * d);
    }
    return s;
}

double Marginal(const AuctionInstance& in, int a, const std::vector<int>& path, int j, int* best_pos) {
    const double base = PathScore(in, a, path);
    double best = kNegInf;
    int pos = 0;
    for (size_t n = 0; n <= path.size(); ++n) {   // eq. 3: insert after the n-th element (n = 0: first)
        std::vector<int> q = path;
        q.insert(q.begin() + std::ptrdiff_t(n), j);
        const double v = PathScore(in, a, q) - base;
        if (v > best) best = v, pos = int(n);
    }
    if (best_pos) *best_pos = pos;
    return best;
}

bool Outbids(double bid_a, int agent_a, double bid_b, int agent_b) {
    if (agent_a < 0) return false;
    if (agent_b < 0) return bid_a > 0;
    if (bid_a != bid_b) return bid_a > bid_b;
    return agent_a < agent_b;
}

Allocation SequentialGreedy(const AuctionInstance& in) {
    const int U = in.n_agents, N = in.Tasks();
    Allocation out;
    out.agents.assign(size_t(U), {});
    out.winner.assign(size_t(N), -1);
    for (;;) {
        double best = 0;
        int ba = -1, bj = -1, bpos = 0;
        for (int a = 0; a < U; ++a) {
            if (int(out.agents[size_t(a)].bundle.size()) >= in.max_bundle) continue;
            for (int j = 0; j < N; ++j) {
                if (out.winner[size_t(j)] >= 0 || in.value[size_t(a)][size_t(j)] <= 0) continue;
                int pos;
                const double c = Marginal(in, a, out.agents[size_t(a)].path, j, &pos);
                if (c <= 0) continue;
                // the comparator, then the lower task index
                if (ba < 0 || Outbids(c, a, best, ba) || (c == best && a == ba && j < bj)) best = c, ba = a, bj = j, bpos = pos;
            }
        }
        if (ba < 0) break;
        auto& A = out.agents[size_t(ba)];
        A.bundle.push_back(bj), A.bids.push_back(best);
        A.path.insert(A.path.begin() + bpos, bj);
        out.winner[size_t(bj)] = ba;
    }
    out.rounds = 1, out.converged = true;
    for (int a = 0; a < U; ++a) out.total += PathScore(in, a, out.agents[size_t(a)].path);
    return out;
}

CbbaAction TableI(int i, int k, int z_kj, int z_ij, double y_kj, double y_ij, const std::vector<int64_t>& s_k,
                  const std::vector<int64_t>& s_i) {
    using A = CbbaAction;
    auto newer = [&](int m) { return s_k[size_t(m)] > s_i[size_t(m)]; };   // s_km > s_im
    const bool kbid = Outbids(y_kj, z_kj, y_ij, z_ij);                      // y_kj > y_ij, with the tie-break
    if (z_kj == k) {
        if (z_ij == i) return kbid ? A::kUpdate : A::kLeave;
        if (z_ij == k) return A::kUpdate;
        if (z_ij < 0) return A::kUpdate;
        return newer(z_ij) || kbid ? A::kUpdate : A::kLeave;               // z_ij = m not in {i, k}
    }
    if (z_kj == i) {
        if (z_ij == i) return A::kLeave;
        if (z_ij == k) return A::kReset;
        if (z_ij < 0) return A::kLeave;
        return newer(z_ij) ? A::kReset : A::kLeave;                        // m
    }
    if (z_kj >= 0) {                                                         // z_kj = m not in {i, k}
        const int m = z_kj;
        if (z_ij == i) return newer(m) && kbid ? A::kUpdate : A::kLeave;
        if (z_ij == k) return newer(m) ? A::kUpdate : A::kReset;
        if (z_ij == m) return newer(m) ? A::kUpdate : A::kLeave;
        if (z_ij < 0) return newer(m) ? A::kUpdate : A::kLeave;
        const int n = z_ij;                                                  // n not in {i, k, m}
        if (newer(m) && newer(n)) return A::kUpdate;
        if (newer(m) && kbid) return A::kUpdate;
        if (newer(n) && s_i[size_t(m)] > s_k[size_t(m)]) return A::kReset;
        return A::kLeave;
    }
    // z_kj = none
    if (z_ij == i) return A::kLeave;
    if (z_ij == k) return A::kUpdate;
    if (z_ij < 0) return A::kLeave;
    return newer(z_ij) ? A::kUpdate : A::kLeave;                            // m
}

namespace {

struct AgentState {
    std::vector<double> y;
    std::vector<int> z;
    std::vector<int64_t> s;
    std::vector<int> b, p;
};

void BuildBundle(const AuctionInstance& in, int i, AgentState* st) {   // Algorithm 3
    const int N = in.Tasks();
    while (int(st->b.size()) < in.max_bundle) {
        double best = 0;
        int J = -1, pos = 0;
        for (int j = 0; j < N; ++j) {
            if (std::find(st->b.begin(), st->b.end(), j) != st->b.end() || in.value[size_t(i)][size_t(j)] <= 0) continue;
            int n;
            const double c = Marginal(in, i, st->p, j, &n);
            if (c <= 0 || !Outbids(c, i, st->y[size_t(j)], st->z[size_t(j)])) continue;   // h_ij
            if (J < 0 || c > best) best = c, J = j, pos = n;                                // ties: lower task index
        }
        if (J < 0) break;
        st->b.push_back(J);
        st->p.insert(st->p.begin() + pos, J);
        st->y[size_t(J)] = best, st->z[size_t(J)] = i;
    }
}

size_t Release(int i, AgentState* st) {   // eq. 6; returns the tail tasks reset (n > n_bar)
    size_t nbar = st->b.size();
    for (size_t n = 0; n < st->b.size(); ++n)
        if (st->z[size_t(st->b[n])] != i) {
            nbar = n;
            break;
        }
    if (nbar == st->b.size()) return 0;
    const size_t tail = st->b.size() - nbar - 1;
    for (size_t n = nbar + 1; n < st->b.size(); ++n) st->y[size_t(st->b[n])] = 0, st->z[size_t(st->b[n])] = -1;
    std::vector<int> released(st->b.begin() + std::ptrdiff_t(nbar), st->b.end());
    st->b.resize(nbar);
    std::vector<int> p;
    for (int t : st->p)
        if (std::find(released.begin(), released.end(), t) == released.end()) p.push_back(t);
    st->p = p;
    return tail;
}

}  // namespace

Allocation RunCbba(const AuctionInstance& in, const std::function<std::vector<std::vector<int>>(int)>& graph,
                   const CbbaParams& prm) {
    const int U = in.n_agents, N = in.Tasks();
    std::vector<AgentState> st(static_cast<size_t>(U));
    for (auto& a : st) a.y.assign(size_t(N), 0), a.z.assign(size_t(N), -1), a.s.assign(size_t(U), 0);
    Allocation out;
    int stable = 0;
    for (int t = 1; t <= prm.max_rounds; ++t) {
        for (int i = 0; i < U; ++i) BuildBundle(in, i, &st[size_t(i)]);
        const auto G = graph(t);
        std::vector<CbbaMessage> snap(static_cast<size_t>(U));
        for (int k = 0; k < U; ++k) {
            snap[size_t(k)].sender = k, snap[size_t(k)].task_ids = in.task_ids;
            snap[size_t(k)].y = st[size_t(k)].y, snap[size_t(k)].z = st[size_t(k)].z, snap[size_t(k)].s = st[size_t(k)].s;
        }
        bool changed = false;
        for (int i = 0; i < U; ++i) {
            AgentState& me = st[size_t(i)];
            const std::vector<double> y0 = me.y;
            const std::vector<int> z0 = me.z;
            for (int k = 0; k < U; ++k) {
                if (k == i || !G[size_t(i)][size_t(k)]) continue;
                const CbbaMessage& m = snap[size_t(k)];
                ++out.messages;
                if (prm.bytes) out.bytes += prm.bytes(m);
                for (int j = 0; j < N; ++j) {
                    // A snapshot's own-entry timestamp is the round it was sent (eq. 5's tau_r on a direct link).
                    std::vector<int64_t> sk = m.s;
                    sk[size_t(k)] = t;
                    switch (TableI(i, k, m.z[size_t(j)], me.z[size_t(j)], m.y[size_t(j)], me.y[size_t(j)], sk, me.s)) {
                        case CbbaAction::kUpdate: me.y[size_t(j)] = m.y[size_t(j)], me.z[size_t(j)] = m.z[size_t(j)]; break;
                        case CbbaAction::kReset: me.y[size_t(j)] = 0, me.z[size_t(j)] = -1; break;
                        case CbbaAction::kLeave: break;
                    }
                }
            }
            // eq. 5: s_ik = tau_r for neighbours, else the max over neighbours' s_mk.
            std::vector<int64_t> s_new = me.s;
            for (int k = 0; k < U; ++k) {
                if (k == i) continue;
                if (G[size_t(i)][size_t(k)]) {
                    s_new[size_t(k)] = t;
                } else {
                    for (int m = 0; m < U; ++m)
                        if (m != i && G[size_t(i)][size_t(m)]) s_new[size_t(k)] = std::max(s_new[size_t(k)], snap[size_t(m)].s[size_t(k)]);
                }
            }
            me.s = s_new;
            out.tail_releases += Release(i, &me);
            changed |= me.y != y0 || me.z != z0;
        }
        out.rounds = t;
        // Converged: every agent agrees on (y, z), and a round passed with no change.
        bool agree = true;
        for (int i = 1; i < U && agree; ++i) agree = st[size_t(i)].z == st[0].z && st[size_t(i)].y == st[0].y;
        stable = (!changed && agree) ? stable + 1 : 0;
        if (stable >= 1) {
            out.converged = true;
            break;
        }
    }
    out.agents.assign(size_t(U), {});
    out.winner.assign(size_t(N), -1);
    for (int i = 0; i < U; ++i) {
        auto& A = out.agents[size_t(i)];
        A.bundle = st[size_t(i)].b, A.path = st[size_t(i)].p;
        for (int j : A.bundle) A.bids.push_back(st[size_t(i)].y[size_t(j)]);
        for (int j : A.bundle) out.winner[size_t(j)] = i;
        out.total += PathScore(in, i, A.path);
    }
    return out;
}

double BruteForceOptimum(const AuctionInstance& in) {
    const int U = in.n_agents, N = in.Tasks();
    std::vector<int> owner(static_cast<size_t>(N), -1);
    double best = 0;
    auto BestOrder = [&](int a, std::vector<int> t) {
        std::sort(t.begin(), t.end());
        double v = 0;
        do v = std::max(v, PathScore(in, a, t));
        while (std::next_permutation(t.begin(), t.end()));
        return v;
    };
    std::function<void(int)> rec = [&](int j) {
        if (j == N) {
            double tot = 0;
            for (int a = 0; a < U; ++a) {
                std::vector<int> t;
                for (int q = 0; q < N; ++q)
                    if (owner[size_t(q)] == a) t.push_back(q);
                if (int(t.size()) > in.max_bundle) return;
                tot += t.empty() ? 0.0 : BestOrder(a, t);
            }
            best = std::max(best, tot);
            return;
        }
        for (int a = -1; a < U; ++a) {
            if (a >= 0 && in.value[size_t(a)][size_t(j)] <= 0) continue;
            owner[size_t(j)] = a;
            rec(j + 1);
        }
        owner[size_t(j)] = -1;
    };
    rec(0);
    return best;
}

double TwoAgentOptimum(const AuctionInstance& in) {
    if (in.n_agents != 2 || in.max_bundle > 2) throw std::invalid_argument("TwoAgentOptimum: 2 agents, L_t <= 2");
    const int N = in.Tasks();
    struct B {
        double v;
        int t0, t1;   // -1 = empty slot
    };
    std::vector<B> bundles[2];
    for (int a = 0; a < 2; ++a) {
        bundles[a].push_back({0.0, -1, -1});
        for (int j = 0; j < N; ++j) {
            if (in.value[size_t(a)][size_t(j)] <= 0) continue;
            bundles[a].push_back({PathScore(in, a, {j}), j, -1});
            if (in.max_bundle < 2) continue;
            for (int k = j + 1; k < N; ++k) {
                if (in.value[size_t(a)][size_t(k)] <= 0) continue;
                bundles[a].push_back({std::max(PathScore(in, a, {j, k}), PathScore(in, a, {k, j})), j, k});
            }
        }
    }
    std::sort(bundles[1].begin(), bundles[1].end(), [](const B& x, const B& y) { return x.v > y.v; });
    double best = 0;
    for (const B& b0 : bundles[0]) {
        for (const B& b1 : bundles[1]) {   // the first disjoint one is the best partner for b0
            if ((b1.t0 >= 0 && (b1.t0 == b0.t0 || b1.t0 == b0.t1)) || (b1.t1 >= 0 && (b1.t1 == b0.t0 || b1.t1 == b0.t1))) continue;
            best = std::max(best, b0.v + b1.v);
            break;
        }
    }
    return best;
}

size_t TestCodecBytes(const CbbaMessage& m) { return 16 + m.task_ids.size() * (8 + 4 + 1) + m.s.size() * 8; }

}  // namespace ariadne::planning
