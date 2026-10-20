// Pairwise Consistency Maximization (arch §14.5, D72): see pcm.hpp for what and why.
#include "slam/robust/pcm.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>

namespace ariadne::slam {

namespace {

bool IsOdometry(EdgeType t) { return t == EdgeType::kOdometry || t == EdgeType::kStereoOdometry; }

Mat6 Inverse(const Mat6& info) {
    Mat6 c = info.ldlt().solve(Mat6::Identity());
    return 0.5 * (c + c.transpose());
}

/// The odometry-only chains of a pose graph: per node, its chain, position, prefix pose C (chain start -> node) and
/// prefix covariance S (sum of the edges' left covariances transported to the chain start's frame).
struct Chains {
    struct Place {
        uint64_t chain = 0;   // smallest keyframe id in the component
        size_t pos = 0;
    };
    std::map<uint64_t, Place> place;
    std::map<uint64_t, std::vector<SE3>> C;    // per chain, per position
    std::map<uint64_t, std::vector<Mat6>> S;
    int parallel = 0;

    explicit Chains(const PoseGraph& g) {
        // Deduplicated odometry adjacency: node -> (neighbour, edge index).
        std::map<uint64_t, std::vector<std::pair<uint64_t, size_t>>> adj;
        std::map<std::pair<uint64_t, uint64_t>, size_t> seen;
        for (size_t i = 0; i < g.edges().size(); ++i) {
            const PoseEdge& e = g.edges()[i];
            if (!IsOdometry(e.type) || e.from == e.to) continue;
            const auto key = std::minmax(e.from, e.to);
            if (!seen.emplace(key, i).second) {
                ++parallel;
                continue;
            }
            adj[e.from].push_back({e.to, i});
            adj[e.to].push_back({e.from, i});
        }
        for (const auto& [id, nd] : g.nodes()) {
            if (place.count(id)) continue;
            // Walk to one end of the component, then along it. Components must be chains (StereoVo emits one per
            // agent); anything else is refused loudly rather than composed wrongly.
            std::vector<uint64_t> comp{id};
            std::map<uint64_t, bool> in{{id, true}};
            for (size_t k = 0; k < comp.size(); ++k)
                for (const auto& [nb, ei] : adj[comp[k]])
                    if (!in[nb]) in[nb] = true, comp.push_back(nb);
            const uint64_t chain = *std::min_element(comp.begin(), comp.end());
            std::vector<uint64_t> ends;
            for (uint64_t v : comp) {
                if (adj[v].size() > 2)
                    throw std::runtime_error("Pcm: odometry component is not a chain at node " + std::to_string(v));
                if (adj[v].size() < 2) ends.push_back(v);
            }
            if (ends.empty()) throw std::runtime_error("Pcm: odometry component has a cycle at node " + std::to_string(chain));
            const uint64_t start = *std::min_element(ends.begin(), ends.end());   // the smaller-id end
            auto& Cv = C[chain];
            auto& Sv = S[chain];
            Cv = {SE3()};
            Sv = {Mat6::Zero()};
            uint64_t prev = std::numeric_limits<uint64_t>::max(), cur = start;
            place[cur] = {chain, 0};
            while (true) {
                const std::pair<uint64_t, size_t>* next = nullptr;
                for (const auto& nb : adj[cur])
                    if (nb.first != prev) next = &nb;
                if (!next) break;
                const PoseEdge& e = g.edges()[next->second];
                // Edge k from position k to k+1: Z_k and its LEFT covariance. Stored forward: Ad(Z) Sigma_r Ad(Z)^T.
                // Stored backward: Z^-1, whose left covariance is Sigma_r of Z itself.
                const Mat6 cov_r = Inverse(e.information);
                SE3 Zk;
                Mat6 cov_k;
                if (e.from == cur) {
                    Zk = e.Z;
                    const Mat6 A = core::SE3Adjoint(e.Z);
                    cov_k = A * cov_r * A.transpose();
                } else {
                    Zk = e.Z.inverse();
                    cov_k = cov_r;
                }
                const Mat6 Ac = core::SE3Adjoint(Cv.back());
                Sv.push_back(Sv.back() + Ac * cov_k * Ac.transpose());
                Cv.push_back(Cv.back() * Zk);
                prev = cur;
                cur = next->first;
                place[cur] = {chain, Cv.size() - 1};
            }
        }
    }

    /// Covariance contribution of odometry edges in [lo, hi) of `chain` with coefficient M (root-frame vectors in).
    Mat6 Piece(uint64_t chain, size_t lo, size_t hi, const Mat6& M) const {
        if (hi <= lo) return Mat6::Zero();
        const auto& Sv = S.at(chain);
        return M * (Sv[hi] - Sv[lo]) * M.transpose();
    }

    /// Sum over pieces of one chain where edge k carries coefficient sum_j sigma_j(k) M_j, with
    /// sigma_j(k) = [k < end_j] - [k < start_j] (+1 forward, -1 backward, 0 outside the leg).
    Mat6 Legs(uint64_t chain, const std::vector<std::pair<size_t, size_t>>& legs, const std::vector<Mat6>& M) const {
        std::vector<size_t> cuts{0, C.at(chain).size() - 1};
        for (const auto& [s, t] : legs) cuts.push_back(s), cuts.push_back(t);
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        Mat6 out = Mat6::Zero();
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            const size_t k = cuts[i];   // representative edge of [cuts[i], cuts[i+1])
            Mat6 coef = Mat6::Zero();
            for (size_t j = 0; j < legs.size(); ++j)
                coef += (double(k < legs[j].second) - double(k < legs[j].first)) * M[j];
            if (!coef.isZero(0.0)) out += Piece(chain, cuts[i], cuts[i + 1], coef);
        }
        return out;
    }

    bool Leg(uint64_t a, uint64_t c, SE3& O, Mat6& cov) const {
        const auto ia = place.find(a), ic = place.find(c);
        if (ia == place.end() || ic == place.end() || ia->second.chain != ic->second.chain) return false;
        const auto& Cv = C.at(ia->second.chain);
        const SE3 Cai = Cv[ia->second.pos].inverse();
        O = Cai * Cv[ic->second.pos];
        const Mat6 M = core::SE3Adjoint(Cai);
        cov = Legs(ia->second.chain, {{ia->second.pos, ic->second.pos}}, {M});
        return true;
    }

    double Pair(const PoseGraph& g, size_t l1, size_t l2) const {
        const PoseEdge& e1 = g.edges()[l1];
        const PoseEdge& e2 = g.edges()[l2];
        const auto pa = place.find(e1.from), pb = place.find(e1.to);
        const auto p2f = place.find(e2.from), p2t = place.find(e2.to);
        if (pa == place.end() || pb == place.end() || p2f == place.end() || p2t == place.end()) return NAN;
        // Loop 2 oriented so that c shares a's chain and d shares b's chain.
        SE3 Z1 = e1.Z, Z2 = e2.Z;
        Mat6 S1 = core::SE3Adjoint(Z1) * Inverse(e1.information) * core::SE3Adjoint(Z1).transpose();
        uint64_t c = e2.from, d = e2.to;
        Mat6 S2;
        if (p2f->second.chain == pa->second.chain && p2t->second.chain == pb->second.chain) {
            S2 = core::SE3Adjoint(Z2) * Inverse(e2.information) * core::SE3Adjoint(Z2).transpose();
        } else if (p2t->second.chain == pa->second.chain && p2f->second.chain == pb->second.chain) {
            std::swap(c, d);
            S2 = Inverse(e2.information);   // left covariance of Z2^-1
            Z2 = Z2.inverse();
        } else {
            return NAN;
        }
        const Place& Pa = pa->second;
        const Place& Pb = pb->second;
        const Place& Pc = place.at(c);
        const Place& Pd = place.at(d);
        const auto& Ca = C.at(Pa.chain);
        const auto& Cb = C.at(Pb.chain);
        const SE3 O_ac = Ca[Pa.pos].inverse() * Ca[Pc.pos];
        const SE3 O_db = Cb[Pd.pos].inverse() * Cb[Pb.pos];
        const SE3 Z1i = Z1.inverse();
        const core::Vec6 e0 = core::SE3Log(Z1i * O_ac * Z2 * O_db);
        // Jacobians of e (before Jl^-1) w.r.t. the left perturbation of each factor.
        const Mat6 A_Z1 = -core::SE3Adjoint(Z1i);
        const Mat6 A_Oac = core::SE3Adjoint(Z1i);
        const Mat6 A_Z2 = core::SE3Adjoint(Z1i * O_ac);
        const Mat6 A_Odb = core::SE3Adjoint(Z1i * O_ac * Z2);
        Mat6 cov = A_Z1 * S1 * A_Z1.transpose() + A_Z2 * S2 * A_Z2.transpose();
        const Mat6 M1 = A_Oac * core::SE3Adjoint(Ca[Pa.pos].inverse());
        const Mat6 M2 = A_Odb * core::SE3Adjoint(Cb[Pd.pos].inverse());
        if (Pa.chain == Pb.chain) {
            cov += Legs(Pa.chain, {{Pa.pos, Pc.pos}, {Pd.pos, Pb.pos}}, {M1, M2});
        } else {
            cov += Legs(Pa.chain, {{Pa.pos, Pc.pos}}, {M1});
            cov += Legs(Pb.chain, {{Pd.pos, Pb.pos}}, {M2});
        }
        const Mat6 Ji = core::SE3LeftJacobianInverse(e0);
        const Mat6 cov_e = Ji * cov * Ji.transpose();
        return e0.dot(cov_e.ldlt().solve(e0));
    }
};

std::vector<int> GreedyClique(const std::vector<std::vector<bool>>& adj) {
    const int n = int(adj.size());
    std::vector<int> deg(size_t(n), 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) deg[size_t(i)] += (i != j && adj[size_t(i)][size_t(j)]);
    std::vector<int> order(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) order[size_t(i)] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return deg[size_t(a)] > deg[size_t(b)]; });
    std::vector<int> best;
    for (int v : order) {
        if (deg[size_t(v)] + 1 <= int(best.size())) continue;   // cannot beat the best (CLIQUEHEU pruning)
        std::vector<int> clique{v}, U;
        for (int u = 0; u < n; ++u)
            if (u != v && adj[size_t(v)][size_t(u)] && deg[size_t(u)] + 1 > int(best.size())) U.push_back(u);
        while (!U.empty()) {
            int pick = U[0];
            for (int u : U)
                if (deg[size_t(u)] > deg[size_t(pick)]) pick = u;   // U is ascending: ties keep the smallest index
            clique.push_back(pick);
            std::vector<int> next;
            for (int u : U)
                if (u != pick && adj[size_t(pick)][size_t(u)]) next.push_back(u);
            U.swap(next);
        }
        std::sort(clique.begin(), clique.end());
        if (clique.size() > best.size()) best = clique;
    }
    return best;
}

std::vector<int> ExactClique(const std::vector<std::vector<bool>>& adj) {
    // Bron-Kerbosch with Tomita pivoting; among maximum cliques the lexicographically smallest index list.
    const int n = int(adj.size());
    std::vector<int> best;
    std::function<void(std::vector<int>&, std::vector<int>, std::vector<int>)> bk = [&](std::vector<int>& R,
                                                                                         std::vector<int> P,
                                                                                         std::vector<int> X) {
        if (P.empty() && X.empty()) {
            std::vector<int> c = R;
            std::sort(c.begin(), c.end());
            if (c.size() > best.size() || (c.size() == best.size() && c < best)) best = c;
            return;
        }
        if (R.size() + P.size() < best.size()) return;
        int pivot = -1, most = -1;
        for (const auto* S : {&P, &X})
            for (int u : *S) {
                int k = 0;
                for (int v : P) k += adj[size_t(u)][size_t(v)];
                if (k > most) most = k, pivot = u;
            }
        const std::vector<int> cand = P;
        for (int v : cand) {
            if (pivot >= 0 && adj[size_t(pivot)][size_t(v)]) continue;
            std::vector<int> P2, X2;
            for (int u : P)
                if (adj[size_t(v)][size_t(u)]) P2.push_back(u);
            for (int u : X)
                if (adj[size_t(v)][size_t(u)]) X2.push_back(u);
            R.push_back(v);
            bk(R, P2, X2);
            R.pop_back();
            P.erase(std::find(P.begin(), P.end(), v));
            X.push_back(v);
        }
    };
    std::vector<int> R, P(static_cast<size_t>(n)), X;
    for (int i = 0; i < n; ++i) P[size_t(i)] = i;
    bk(R, P, X);
    return best;
}

}  // namespace

double ChiSquare6Survival(double x) {
    if (x <= 0) return 1.0;
    const double h = 0.5 * x;   // 6 dof: P(X > x) = e^-h (1 + h + h^2 / 2)
    return std::exp(-h) * (1.0 + h + 0.5 * h * h);
}

double ChiSquare6Quantile(double p) {
    double lo = 0.0, hi = 1.0;
    while (1.0 - ChiSquare6Survival(hi) < p) hi *= 2.0;
    for (int i = 0; i < 200; ++i) {
        const double mid = 0.5 * (lo + hi);
        (1.0 - ChiSquare6Survival(mid) < p ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

bool OdometryLeg(const PoseGraph& g, uint64_t a, uint64_t c, SE3& O, Mat6& cov_left) {
    return Chains(g).Leg(a, c, O, cov_left);
}

double PairConsistency(const PoseGraph& g, size_t loop1, size_t loop2) { return Chains(g).Pair(g, loop1, loop2); }

std::vector<int> MaxClique(const std::vector<std::vector<bool>>& adj, bool exact) {
    return exact ? ExactClique(adj) : GreedyClique(adj);
}

PcmReport Pcm(const PoseGraph& g, const PcmParams& p) {
    const Chains ch(g);
    PcmReport rep;
    rep.parallel_odometry_edges = ch.parallel;
    std::map<std::pair<uint64_t, uint64_t>, std::vector<size_t>> groups;
    for (size_t i = 0; i < g.edges().size(); ++i) {
        const PoseEdge& e = g.edges()[i];
        if (!IsLoopType(e.type)) continue;
        const auto f = ch.place.find(e.from), t = ch.place.find(e.to);
        if (f == ch.place.end() || t == ch.place.end()) throw std::runtime_error("Pcm: loop endpoint not in the graph");
        groups[std::minmax(f->second.chain, t->second.chain)].push_back(i);
    }
    for (auto& [key, loops] : groups) {
        std::stable_sort(loops.begin(), loops.end(), [&](size_t x, size_t y) {
            return std::pair(g.edges()[x].from, g.edges()[x].to) < std::pair(g.edges()[y].from, g.edges()[y].to);
        });
        PcmGroup gr;
        gr.component_from = key.first;
        gr.component_to = key.second;
        gr.loops = loops;
        const int n = int(loops.size());
        if (n == 1) {
            gr.accepted = loops;
            gr.verdict = "untested";
        } else {
            gr.threshold = p.alpha_loop > 0 ? ChiSquare6Quantile(1.0 - p.alpha_loop / double(n - 1)) : p.chi2_threshold;
            std::vector<std::vector<bool>> adj(size_t(n), std::vector<bool>(size_t(n), false));
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) {
                    const double c = ch.Pair(g, loops[size_t(i)], loops[size_t(j)]);
                    const bool ok = std::isfinite(c) && c <= gr.threshold;
                    adj[size_t(i)][size_t(j)] = adj[size_t(j)][size_t(i)] = ok;
                    ++gr.pairs;
                    gr.consistent_pairs += ok;
                }
            gr.exact = p.exact && n <= std::min(p.exact_max, 64);
            const std::vector<int> clique = MaxClique(adj, gr.exact);
            if (clique.size() >= 2) {
                for (int k : clique) gr.accepted.push_back(loops[size_t(k)]);
                gr.verdict = "consensus";
            } else {
                gr.verdict = "no_consensus";
            }
        }
        for (size_t l : gr.loops) {
            const bool acc = std::find(gr.accepted.begin(), gr.accepted.end(), l) != gr.accepted.end();
            (acc ? rep.accepted : rep.rejected).push_back(l);
        }
        rep.groups.push_back(std::move(gr));
    }
    std::sort(rep.accepted.begin(), rep.accepted.end());
    std::sort(rep.rejected.begin(), rep.rejected.end());
    return rep;
}

}  // namespace ariadne::slam
