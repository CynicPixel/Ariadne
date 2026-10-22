#pragma once
// ---------------------------------------------------------------------------
// Spectral prioritisation of feature requests (design SPECTRAL_3.4, arch §16, D170).
//
// The broker holds a graph of every keyframe it knows of: odometry, intra-agent loops and accepted inter-agent edges
// (fixed, kappa = 1), plus retrieval candidates (kappa = similarity). Verifying a candidate costs the feature
// responses of its endpoints the broker does not hold. Under a byte budget, pick the candidates whose edges raise the
// algebraic connectivity lambda_2 of the graph Laplacian the most per byte: lambda_2 controls the worst-case error of
// the pose-graph estimate (Rosen et al.; MAC, Doherty et al.), so this is "which evidence buys the most accuracy".
//
//   - Fiedler space: block inverse iteration on L^+ (grounded SimplicialLDLT), Rayleigh-Ritz, warm-startable (§3).
//   - Selection: a connectivity bootstrap, then greedy kappa_e |V^T (e_a - e_b)|^2 / marginal bytes (§4).
//   - Bound: MAC's post-hoc dual bound over an LP relaxation of the SHARED-cost feasible set (cv::solveLP), so
//     "near-optimal" is a per-instance certificate, not a claim (§4).
// No protocol or wire type here: node indices and byte costs in, a selection and its report out.
// ---------------------------------------------------------------------------
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <opencv2/core.hpp>
#include <opencv2/core/optim.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <vector>

namespace ariadne::collaboration {

struct GraphEdge {
    int a = 0, b = 0;
    double kappa = 1.0;
};

struct CandidateEdge {
    uint64_t id = 0;        // stable id: the tie-break, so the selection is the same on every host
    int a = 0, b = 0;       // node indices
    double kappa = 1.0;     // retrieval similarity in (0, 1]
};

struct SelectionProblem {
    int n = 0;                          // nodes 0..n-1
    std::vector<GraphEdge> fixed;
    std::vector<CandidateEdge> candidates;
    std::vector<double> node_bytes;     // predicted bytes to fetch a node's features; 0 = held (broker's own, fetched)
    double budget_bytes = 0;
};

struct SelectorParams {
    int block = 4;                      // Ritz block size p
    int max_iterations = 200;
    double tolerance = 1e-9;            // Ritz residual / ||L||
    double degenerate_rel = 1e-6;       // lambda_i within this (relative) of lambda_2: one cluster (§3)
    bool bound = true;                  // the dual bound (one LP)
};

/// lambda_2 and an orthonormal basis of its (possibly repeated) eigenspace, of one connected graph.
struct FiedlerSpace {
    double lambda2 = 0;
    double rayleigh = 0;                // v^T L v of the first basis vector (the bound uses this, not lambda2)
    Eigen::MatrixXd V;                  // n x c, c = cluster size
    Eigen::MatrixXd block;              // n x p Ritz block, for warm starts
    int iterations = 0;
    double residual = 0;                // max Ritz residual over the cluster, / ||L||
    bool converged = false;
};

struct SelectionReport {
    std::vector<uint64_t> selected;     // in selection order
    double bytes = 0;
    int bootstrap = 0;                  // picks that joined components
    bool connected = false;
    double lambda2 = 0;                 // of the final graph (0 if disconnected)
    std::optional<double> bound;        // >= the optimum's lambda_2; absent when not computable (stated)
    int solves = 0, iterations = 0;     // Fiedler solves and their total iterations
};

namespace spectral_detail {

using SparseMat = Eigen::SparseMatrix<double>;

inline SparseMat Laplacian(int n, const std::vector<GraphEdge>& edges) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(edges.size() * 4);
    for (const GraphEdge& e : edges) {
        if (e.a == e.b) continue;
        t.emplace_back(e.a, e.a, e.kappa);
        t.emplace_back(e.b, e.b, e.kappa);
        t.emplace_back(e.a, e.b, -e.kappa);
        t.emplace_back(e.b, e.a, -e.kappa);
    }
    SparseMat L(n, n);
    L.setFromTriplets(t.begin(), t.end());
    return L;
}

struct UnionFind {
    std::vector<int> p;
    explicit UnionFind(int n) : p(size_t(n)) { std::iota(p.begin(), p.end(), 0); }
    int Find(int x) { return p[size_t(x)] == x ? x : p[size_t(x)] = Find(p[size_t(x)]); }
    bool Join(int a, int b) {
        a = Find(a), b = Find(b);
        if (a == b) return false;
        p[size_t(std::max(a, b))] = std::min(a, b);
        return true;
    }
};

/// Deterministic start block: a fixed integer hash per (row, column) mapped to [-1, 1]. No <random> distribution, whose
/// output is implementation-defined.
inline Eigen::MatrixXd StartBlock(int n, int p) {
    Eigen::MatrixXd X(n, p);
    for (int j = 0; j < p; ++j)
        for (int i = 0; i < n; ++i) {
            uint64_t h = (uint64_t(i) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(j + 1) * 0xC2B2AE3D27D4EB4Full);
            h ^= h >> 31;
            h *= 0xBF58476D1CE4E5B9ull;
            h ^= h >> 29;
            X(i, j) = double(h >> 11) / double(1ull << 53) * 2.0 - 1.0;
        }
    return X;
}

/// Orthonormalise X against 1 and itself (modified Gram-Schmidt, twice).
inline void OrthonormalizePerp1(Eigen::MatrixXd& X) {
    for (int pass = 0; pass < 2; ++pass)
        for (int j = 0; j < X.cols(); ++j) {
            X.col(j).array() -= X.col(j).mean();
            for (int k = 0; k < j; ++k) X.col(j) -= X.col(k).dot(X.col(j)) * X.col(k);
            const double nrm = X.col(j).norm();
            if (nrm > 1e-300) X.col(j) /= nrm;
        }
}

}  // namespace spectral_detail

/// The Fiedler space of a CONNECTED graph with Laplacian L (n >= 2). L^+ is applied by grounding node 0: solve the
/// SPD system on nodes 1..n-1, put 0 at node 0, remove the mean (the minimum-norm solution of L x = b for b _|_ 1).
/// Block inverse iteration converges at lambda_i / lambda_{p+1}; with a warm block from the previous round (one edge
/// changed) it takes a few iterations (arch §16 "warm-started").
inline FiedlerSpace SolveFiedler(const spectral_detail::SparseMat& L, const SelectorParams& prm,
                                 const Eigen::MatrixXd* warm = nullptr) {
    using namespace spectral_detail;
    FiedlerSpace out;
    const int n = int(L.rows());
    const int p = std::min(prm.block, n - 1);
    if (n == 2) {   // closed form: lambda_2 = 2 kappa, v = (1, -1)/sqrt 2
        out.lambda2 = out.rayleigh = L.coeff(0, 0) * 2.0;
        out.V = Eigen::MatrixXd(2, 1);
        out.V << M_SQRT1_2, -M_SQRT1_2;
        out.block = out.V;
        out.converged = true;
        return out;
    }
    const SparseMat Lg = L.bottomRightCorner(n - 1, n - 1);
    Eigen::SimplicialLDLT<SparseMat> ldlt(Lg);
    double norm_L = 0;   // ||L||_2 <= 2 max weighted degree
    for (int i = 0; i < n; ++i) norm_L = std::max(norm_L, 2.0 * L.coeff(i, i));
    Eigen::MatrixXd X = warm && warm->rows() == n && warm->cols() == p ? *warm : StartBlock(n, p);
    OrthonormalizePerp1(X);
    Eigen::VectorXd theta;
    Eigen::MatrixXd R;
    for (int it = 1; it <= prm.max_iterations; ++it) {
        Eigen::MatrixXd Y(n, p);
        for (int j = 0; j < p; ++j) {
            Eigen::VectorXd b = X.col(j);
            b.array() -= b.mean();
            Y(0, j) = 0.0;
            Y.col(j).tail(n - 1) = ldlt.solve(b.tail(n - 1));
        }
        OrthonormalizePerp1(Y);
        const Eigen::MatrixXd LY = L * Y;
        const Eigen::MatrixXd T = Y.transpose() * LY;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (T + T.transpose()));
        theta = es.eigenvalues();
        X = Y * es.eigenvectors();
        R = L * X - X * theta.asDiagonal();
        out.iterations = it;
        // Converged when every Ritz pair in lambda_2's cluster, and the first one past it, is accurate: the cluster
        // boundary is then known.
        int c = 1;
        while (c < p && theta(c) - theta(0) <= prm.degenerate_rel * std::max(theta(0), 1e-300)) ++c;
        double res = 0;
        for (int j = 0; j < std::min(c + 1, p); ++j) res = std::max(res, R.col(j).norm() / std::max(norm_L, 1e-300));
        out.residual = res;
        if (res <= prm.tolerance) {
            out.converged = true;
            break;
        }
    }
    int c = 1;
    while (c < p && theta(c) - theta(0) <= prm.degenerate_rel * std::max(theta(0), 1e-300)) ++c;
    out.lambda2 = theta(0);
    out.V = X.leftCols(c);
    out.rayleigh = X.col(0).dot(L * X.col(0));
    out.block = X;
    return out;
}

/// Greedy spectral selection under a shared byte budget, plus the dual bound (design §4).
inline SelectionReport SelectSpectral(const SelectionProblem& P, const SelectorParams& prm = {}) {
    using namespace spectral_detail;
    SelectionReport rep;
    const size_t m = P.candidates.size();
    std::vector<char> held(size_t(P.n), 0), chosen(m, 0);
    for (int i = 0; i < P.n; ++i) held[size_t(i)] = P.node_bytes[size_t(i)] <= 0.0;
    auto delta_cost = [&](const CandidateEdge& e) {
        double c = 0;
        if (!held[size_t(e.a)]) c += P.node_bytes[size_t(e.a)];
        if (e.b != e.a && !held[size_t(e.b)]) c += P.node_bytes[size_t(e.b)];
        return c;
    };
    std::vector<GraphEdge> edges = P.fixed;
    auto take = [&](size_t k) {
        const CandidateEdge& e = P.candidates[k];
        rep.bytes += delta_cost(e);
        held[size_t(e.a)] = held[size_t(e.b)] = 1;
        chosen[k] = 1;
        edges.push_back({e.a, e.b, e.kappa});
        rep.selected.push_back(e.id);
    };
    auto affordable = [&](size_t k) { return !chosen[k] && rep.bytes + delta_cost(P.candidates[k]) <= P.budget_bytes; };
    // Ratio order with the id tie-break: larger score per byte first; free candidates (0 bytes) before any paid one.
    auto better = [&](double sa, double ca, uint64_t ia, double sb, double cb, uint64_t ib) {
        const double lhs = sa * std::max(cb, 0.0), rhs = sb * std::max(ca, 0.0);
        if (ca <= 0.0 || cb <= 0.0) {
            if ((ca <= 0.0) != (cb <= 0.0)) return ca <= 0.0;
            if (sa != sb) return sa > sb;
            return ia < ib;
        }
        if (lhs != rhs) return lhs > rhs;
        return ia < ib;
    };

    // 1 + 2 in one loop: a bridging candidate, whenever one is affordable, comes first (connectivity bootstrap: lambda_2
    // means nothing on a disconnected graph, arch §16.2; a pick can make a node held and a bridge affordable later).
    // Otherwise greedy on lambda_2, per connected component (one that cannot be joined is still improved).
    std::map<int, Eigen::MatrixXd> warm;   // keyed by the component's smallest node
    for (;;) {
        UnionFind uf(P.n);
        for (const GraphEdge& e : edges) uf.Join(e.a, e.b);
        std::optional<size_t> bridge;
        for (size_t k = 0; k < m; ++k) {
            const CandidateEdge& e = P.candidates[k];
            if (!affordable(k) || uf.Find(e.a) == uf.Find(e.b)) continue;
            if (!bridge || better(e.kappa, delta_cost(e), e.id, P.candidates[*bridge].kappa,
                                  delta_cost(P.candidates[*bridge]), P.candidates[*bridge].id))
                bridge = k;
        }
        if (bridge) {
            take(*bridge);
            ++rep.bootstrap;
            continue;
        }
        std::map<int, std::vector<int>> comps;
        for (int i = 0; i < P.n; ++i) comps[uf.Find(i)].push_back(i);
        std::optional<size_t> best;
        double best_gain = 0;
        for (const auto& [root, nodes] : comps) {
            if (nodes.size() < 2) continue;
            bool any = false;
            for (size_t k = 0; k < m && !any; ++k)
                any = affordable(k) && uf.Find(P.candidates[k].a) == root;
            if (!any) continue;
            std::vector<int> local(size_t(P.n), -1);
            for (size_t i = 0; i < nodes.size(); ++i) local[size_t(nodes[i])] = int(i);
            std::vector<GraphEdge> sub;
            for (const GraphEdge& e : edges)
                if (local[size_t(e.a)] >= 0) sub.push_back({local[size_t(e.a)], local[size_t(e.b)], e.kappa});
            const SparseMat L = Laplacian(int(nodes.size()), sub);
            const auto w = warm.find(root);
            const FiedlerSpace F = SolveFiedler(L, prm, w != warm.end() ? &w->second : nullptr);
            warm[root] = F.block;
            ++rep.solves;
            rep.iterations += F.iterations;
            const double eps = 1e-12 * std::max(F.lambda2, 1e-300);
            for (size_t k = 0; k < m; ++k) {
                const CandidateEdge& e = P.candidates[k];
                if (!affordable(k) || uf.Find(e.a) != root) continue;
                const double gain = e.kappa * (F.V.row(local[size_t(e.a)]) - F.V.row(local[size_t(e.b)])).squaredNorm();
                if (gain <= eps) continue;
                if (!best || better(gain, delta_cost(e), e.id, best_gain, delta_cost(P.candidates[*best]),
                                    P.candidates[*best].id)) {
                    best = k;
                    best_gain = gain;
                }
            }
        }
        if (!best) break;
        take(*best);
    }

    // 3. Final lambda_2 and the dual bound (connected graphs only; otherwise both are stated absent / 0).
    UnionFind uf(P.n);
    int ncomp = P.n;
    for (const GraphEdge& e : edges) ncomp -= uf.Join(e.a, e.b);
    rep.connected = ncomp == 1 && P.n >= 2;
    if (!rep.connected) return rep;
    const SparseMat L = Laplacian(P.n, edges);
    const auto w = warm.find(0);
    const FiedlerSpace F = SolveFiedler(L, prm, w != warm.end() ? &w->second : nullptr);
    ++rep.solves;
    rep.iterations += F.iterations;
    rep.lambda2 = F.lambda2;
    if (!prm.bound) return rep;
    // Supergradient at x from one unit vector v _|_ 1: lambda_2(x') <= v^T L(x') v = rho + g^T (x' - x) for EVERY x'
    // (Rayleigh), so the bound holds even if v is not fully converged.
    const Eigen::VectorXd v = F.V.col(0);
    std::vector<double> g(m);
    double gx = 0;
    for (size_t k = 0; k < m; ++k) {
        const CandidateEdge& e = P.candidates[k];
        g[k] = e.kappa * std::pow(v(e.a) - v(e.b), 2);
        if (chosen[k]) gx += g[k];
    }
    // LP over the shared-cost relaxation: variables s (candidates) then y (payable nodes). Nodes held before this
    // round are free and have no y. maximise g^T s; s_k <= y_j per payable endpoint j; sum bytes_j y_j <= B; s, y <= 1.
    std::map<int, int> ycol;
    for (const CandidateEdge& e : P.candidates)
        for (int j : {e.a, e.b})
            if (P.node_bytes[size_t(j)] > 0.0 && !ycol.count(j)) ycol.emplace(j, int(m + ycol.size()));
    const int N = int(m + ycol.size());
    std::vector<std::vector<double>> rows;
    for (size_t k = 0; k < m; ++k) {
        const CandidateEdge& e = P.candidates[k];
        for (int j : std::set<int>{e.a, e.b}) {
            if (!ycol.count(j)) continue;
            std::vector<double> r(size_t(N) + 1, 0.0);
            r[k] = 1.0;
            r[size_t(ycol[j])] = -1.0;
            rows.push_back(std::move(r));
        }
        std::vector<double> r(size_t(N) + 1, 0.0);
        r[k] = 1.0;
        r[size_t(N)] = 1.0;
        rows.push_back(std::move(r));
    }
    std::vector<double> budget(size_t(N) + 1, 0.0);
    for (const auto& [j, col] : ycol) {
        budget[size_t(col)] = P.node_bytes[size_t(j)];
        std::vector<double> r(size_t(N) + 1, 0.0);
        r[size_t(col)] = 1.0;
        r[size_t(N)] = 1.0;
        rows.push_back(std::move(r));
    }
    budget[size_t(N)] = P.budget_bytes;
    rows.push_back(budget);
    cv::Mat func(1, N, CV_64F), constr(int(rows.size()), N + 1, CV_64F), z;
    for (int k = 0; k < N; ++k) func.at<double>(0, k) = k < int(m) ? g[size_t(k)] : 0.0;
    for (size_t i = 0; i < rows.size(); ++i)
        for (int k = 0; k <= N; ++k) constr.at<double>(int(i), k) = rows[i][size_t(k)];
    const int status = cv::solveLP(func, constr, z, 1e-9);
    if (status != cv::SOLVELP_SINGLE && status != cv::SOLVELP_MULTI) return rep;   // bound absent, never guessed
    double gs = 0;
    for (size_t k = 0; k < m; ++k) gs += g[k] * z.at<double>(int(k));
    rep.bound = F.rayleigh + std::max(0.0, gs - gx);
    return rep;
}

/// The "greedy" baseline of the gate plot: highest similarity per marginal byte, same budget, no spectrum.
inline SelectionReport SelectBySimilarity(const SelectionProblem& P) {
    SelectionReport rep;
    std::vector<char> held(size_t(P.n), 0);
    for (int i = 0; i < P.n; ++i) held[size_t(i)] = P.node_bytes[size_t(i)] <= 0.0;
    std::vector<size_t> order(P.candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        const auto& a = P.candidates[x];
        const auto& b = P.candidates[y];
        return a.kappa != b.kappa ? a.kappa > b.kappa : a.id < b.id;
    });
    for (size_t k : order) {
        const CandidateEdge& e = P.candidates[k];
        double c = (held[size_t(e.a)] ? 0.0 : P.node_bytes[size_t(e.a)]) +
                   (e.b == e.a || held[size_t(e.b)] ? 0.0 : P.node_bytes[size_t(e.b)]);
        if (rep.bytes + c > P.budget_bytes) continue;
        rep.bytes += c;
        held[size_t(e.a)] = held[size_t(e.b)] = 1;
        rep.selected.push_back(e.id);
    }
    return rep;
}

}  // namespace ariadne::collaboration
