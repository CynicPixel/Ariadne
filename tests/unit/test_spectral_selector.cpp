// Spectral prioritisation (design SPECTRAL_3.4 §7, D170), synthetic graphs, no runtime:
//   - the block inverse-iteration Fiedler solver against a dense eigensolver, cold and warm;
//   - a repeated lambda_2 (a cycle): the cluster is detected and the selection does not depend on input order;
//   - the exact oracle (every subset under the SHARED byte cost, |C| <= 12): the dual bound is >= the optimum on
//     every instance, the selection never beats the optimum and never exceeds the budget; greedy/optimum reported;
//   - bootstrap: a bridging candidate first; sharing: one fetched keyframe serves every candidate on it;
//   - maximality: nothing affordable with positive gain is left unselected.
#include "collaboration/candidate_selection/spectral_selector.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <random>

using namespace ariadne::collaboration;

namespace {

double DenseLambda2(int n, const std::vector<GraphEdge>& edges) {
    spectral_detail::UnionFind uf(n);
    int comps = n;
    for (const auto& e : edges) comps -= uf.Join(e.a, e.b);
    if (comps != 1) return 0.0;
    Eigen::MatrixXd L = Eigen::MatrixXd(spectral_detail::Laplacian(n, edges));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(L, Eigen::EigenvaluesOnly);
    return es.eigenvalues()(1);
}

/// Agents' odometry chains (kappa 1) laid out back to back; agent 0 is the broker (its nodes are free).
struct Chains {
    int n = 0;
    std::vector<int> first, len;
    std::vector<GraphEdge> fixed;
};
Chains MakeChains(const std::vector<int>& lens) {
    Chains c;
    for (int l : lens) {
        c.first.push_back(c.n);
        c.len.push_back(l);
        for (int i = 1; i < l; ++i) c.fixed.push_back({c.n + i - 1, c.n + i, 1.0});
        c.n += l;
    }
    return c;
}

/// Random instance: chains, a few intra-agent loops, inter-agent candidates with similarity in [0.3, 1], node costs
/// 0 (agent 0) or 1..3 units, a budget that affords part of it.
SelectionProblem RandomInstance(std::mt19937& g, int n_cand) {
    std::uniform_int_distribution<int> nag(2, 3), len(5, 9);
    std::vector<int> lens;
    for (int a = nag(g); a > 0; --a) lens.push_back(len(g));
    const Chains c = MakeChains(lens);
    SelectionProblem P;
    P.n = c.n;
    P.fixed = c.fixed;
    std::uniform_int_distribution<int> agent(0, int(lens.size()) - 1), cost(1, 3);
    std::uniform_real_distribution<double> sim(0.3, 1.0);
    if (std::uniform_int_distribution<int>(0, 1)(g)) P.fixed.push_back({c.first[0], c.first[0] + c.len[0] - 1, 1.0});
    P.node_bytes.assign(size_t(P.n), 0.0);
    for (size_t a = 1; a < lens.size(); ++a)
        for (int i = 0; i < c.len[a]; ++i) P.node_bytes[size_t(c.first[a] + i)] = 1000.0 * cost(g);
    std::set<std::pair<int, int>> seen;
    while (int(P.candidates.size()) < n_cand) {
        int x = agent(g), y = agent(g);
        if (x == y) continue;
        const int a = c.first[size_t(x)] + std::uniform_int_distribution<int>(0, c.len[size_t(x)] - 1)(g);
        const int b = c.first[size_t(y)] + std::uniform_int_distribution<int>(0, c.len[size_t(y)] - 1)(g);
        if (!seen.insert(std::minmax(a, b)).second) continue;
        P.candidates.push_back({uint64_t(P.candidates.size() + 1), a, b, sim(g)});
    }
    double all = 0;
    for (double b : P.node_bytes) all += b;
    P.budget_bytes = std::uniform_real_distribution<double>(0.15, 0.6)(g) * all;
    return P;
}

double SharedCost(const SelectionProblem& P, const std::vector<size_t>& pick) {
    std::set<int> need;
    for (size_t k : pick)
        for (int j : {P.candidates[k].a, P.candidates[k].b})
            if (P.node_bytes[size_t(j)] > 0.0) need.insert(j);
    double c = 0;
    for (int j : need) c += P.node_bytes[size_t(j)];
    return c;
}

double Lambda2Of(const SelectionProblem& P, const std::vector<uint64_t>& ids) {
    std::vector<GraphEdge> e = P.fixed;
    for (const auto& c : P.candidates)
        if (std::find(ids.begin(), ids.end(), c.id) != ids.end()) e.push_back({c.a, c.b, c.kappa});
    return DenseLambda2(P.n, e);
}

}  // namespace

TEST(SpectralSelector, FiedlerMatchesDenseColdAndWarm) {
    std::mt19937 g(7);
    for (int trial = 0; trial < 20; ++trial) {
        const Chains c = MakeChains({40, 35, 25});
        std::vector<GraphEdge> e = c.fixed;
        std::uniform_int_distribution<int> node(0, c.n - 1);
        std::uniform_real_distribution<double> w(0.2, 1.0);
        e.push_back({c.first[0] + 3, c.first[1] + 10, w(g)});   // connect the three chains
        e.push_back({c.first[1] + 30, c.first[2] + 5, w(g)});
        for (int k = 0; k < 6; ++k) e.push_back({node(g), node(g), w(g)});
        const auto L = spectral_detail::Laplacian(c.n, e);
        SelectorParams prm;
        const FiedlerSpace cold = SolveFiedler(L, prm);
        ASSERT_TRUE(cold.converged) << "residual " << cold.residual;
        const double ref = DenseLambda2(c.n, e);
        EXPECT_NEAR(cold.lambda2, ref, 1e-9 * std::max(1.0, ref)) << trial;
        EXPECT_LE(cold.rayleigh, ref * (1 + 1e-8) + 1e-15);
        EXPECT_NEAR(cold.V.col(0).sum(), 0.0, 1e-9);
        // One more edge, warm from the previous block: same answer, fewer iterations.
        e.push_back({node(g), node(g), w(g)});
        const auto L2 = spectral_detail::Laplacian(c.n, e);
        const FiedlerSpace warm = SolveFiedler(L2, prm, &cold.block);
        const FiedlerSpace cold2 = SolveFiedler(L2, prm);
        EXPECT_NEAR(warm.lambda2, cold2.lambda2, 1e-9 * std::max(1.0, cold2.lambda2));
        EXPECT_LE(warm.iterations, cold2.iterations) << trial;
    }
}

TEST(SpectralSelector, RepeatedLambda2IsAClusterAndOrderDoesNotMatter) {
    // A 12-cycle: lambda_2 = lambda_3 = 2 - 2 cos(2 pi / 12). Any single Fiedler vector is arbitrary there.
    const int n = 12;
    SelectionProblem P;
    P.n = n;
    for (int i = 0; i < n; ++i) P.fixed.push_back({i, (i + 1) % n, 1.0});
    const FiedlerSpace F = SolveFiedler(spectral_detail::Laplacian(n, P.fixed), SelectorParams{});
    EXPECT_NEAR(F.lambda2, 2.0 - 2.0 * std::cos(2.0 * M_PI / n), 1e-9);
    EXPECT_EQ(F.V.cols(), 2) << "the repeated eigenvalue must be one cluster";
    P.node_bytes.assign(size_t(n), 1000.0);
    for (int i = 0; i < 6; ++i) P.candidates.push_back({uint64_t(10 + i), i, (i + 5) % n, 0.8});
    P.budget_bytes = 4000;
    const SelectionReport a = SelectSpectral(P);
    std::reverse(P.candidates.begin(), P.candidates.end());
    const SelectionReport b = SelectSpectral(P);
    EXPECT_EQ(a.selected, b.selected);
    EXPECT_FALSE(a.selected.empty());
}

TEST(SpectralSelector, ExactOracleUnderSharedCost) {
    std::mt19937 g(11);
    int with_bound = 0;
    double ratio_sum = 0;
    int ratio_n = 0;
    double worst = 1.0;
    for (int trial = 0; trial < 60; ++trial) {
        const SelectionProblem P = RandomInstance(g, 6 + trial % 7);   // 6..12 candidates
        const size_t m = P.candidates.size();
        double opt = 0;
        for (uint32_t mask = 0; mask < (1u << m); ++mask) {
            std::vector<size_t> pick;
            std::vector<uint64_t> ids;
            for (size_t k = 0; k < m; ++k)
                if (mask >> k & 1u) pick.push_back(k), ids.push_back(P.candidates[k].id);
            if (SharedCost(P, pick) > P.budget_bytes) continue;
            opt = std::max(opt, Lambda2Of(P, ids));
        }
        const SelectionReport r = SelectSpectral(P);
        ASSERT_LE(r.bytes, P.budget_bytes + 1e-9) << trial;
        std::vector<size_t> pick;
        for (size_t k = 0; k < m; ++k)
            if (std::find(r.selected.begin(), r.selected.end(), P.candidates[k].id) != r.selected.end()) pick.push_back(k);
        EXPECT_NEAR(SharedCost(P, pick), r.bytes, 1e-9) << "reported bytes must be the shared cost";
        const double got = Lambda2Of(P, r.selected);
        EXPECT_NEAR(got, r.lambda2, 1e-8 * std::max(1.0, got)) << trial;
        EXPECT_LE(got, opt + 1e-9) << trial;
        if (r.bound) {
            ++with_bound;
            EXPECT_GE(*r.bound, opt - 1e-9) << "INVALID dual bound, trial " << trial;
        }
        if (opt > 0) {
            ratio_sum += got / opt;
            ++ratio_n;
            worst = std::min(worst, got / opt);
        }
    }
    std::printf("oracle: %d instances with a connected optimum; selected/optimum mean %.4f, worst %.4f; bound present "
                "on %d\n", ratio_n, ratio_sum / std::max(1, ratio_n), worst, with_bound);
    EXPECT_GT(with_bound, 0);
}

TEST(SpectralSelector, BootstrapBridgesBeforeImprovingAComponent) {
    const Chains c = MakeChains({10, 10});
    SelectionProblem P;
    P.n = c.n;
    P.fixed = c.fixed;
    P.node_bytes.assign(size_t(P.n), 0.0);
    for (int i = 10; i < 20; ++i) P.node_bytes[size_t(i)] = 1000;
    P.fixed.push_back({10, 19, 1.0});                       // agent 1 has a loop; still two components
    P.candidates.push_back({1, 2, 15, 0.35});               // the only bridge, low similarity
    P.candidates.push_back({2, 0, 9, 0.95});                // free, inside agent 0 (a high-gain chord)
    P.budget_bytes = 1000;
    const SelectionReport r = SelectSpectral(P);
    ASSERT_FALSE(r.selected.empty());
    EXPECT_EQ(r.selected.front(), 1u);
    EXPECT_EQ(r.bootstrap, 1);
    EXPECT_TRUE(r.connected);
}

TEST(SpectralSelector, OneFetchServesEveryCandidateOnIt) {
    const Chains c = MakeChains({8, 8});
    SelectionProblem P;
    P.n = c.n;
    P.fixed = c.fixed;
    P.node_bytes.assign(size_t(P.n), 0.0);
    for (int i = 8; i < 16; ++i) P.node_bytes[size_t(i)] = 5000;
    for (int i = 0; i < 4; ++i) P.candidates.push_back({uint64_t(i + 1), i * 2, 12, 0.7});   // a star on node 12
    P.budget_bytes = 5000;
    const SelectionReport r = SelectSpectral(P);
    EXPECT_DOUBLE_EQ(r.bytes, 5000);
    EXPECT_GE(r.selected.size(), 2u) << "after the first, the star's other edges are free";
    const SelectionReport s = SelectBySimilarity(P);
    EXPECT_DOUBLE_EQ(s.bytes, 5000);
    EXPECT_EQ(s.selected.size(), 4u);
}

TEST(SpectralSelector, NothingAffordableWithGainIsLeft) {
    std::mt19937 g(5);
    for (int trial = 0; trial < 20; ++trial) {
        const SelectionProblem P = RandomInstance(g, 12);
        const SelectionReport r = SelectSpectral(P);
        if (!r.connected) continue;
        std::vector<GraphEdge> e = P.fixed;
        std::set<int> held;
        for (int i = 0; i < P.n; ++i)
            if (P.node_bytes[size_t(i)] <= 0) held.insert(i);
        for (const auto& c : P.candidates)
            if (std::find(r.selected.begin(), r.selected.end(), c.id) != r.selected.end())
                e.push_back({c.a, c.b, c.kappa}), held.insert(c.a), held.insert(c.b);
        const FiedlerSpace F = SolveFiedler(spectral_detail::Laplacian(P.n, e), SelectorParams{});
        for (const auto& c : P.candidates) {
            if (std::find(r.selected.begin(), r.selected.end(), c.id) != r.selected.end()) continue;
            double dc = 0;
            for (int j : {c.a, c.b})
                if (!held.count(j)) dc += P.node_bytes[size_t(j)];
            const double gain = c.kappa * (F.V.row(c.a) - F.V.row(c.b)).squaredNorm();
            EXPECT_TRUE(r.bytes + dc > P.budget_bytes || gain <= 1e-9 * F.lambda2)
                << "trial " << trial << ": candidate " << c.id << " affordable with gain " << gain;
        }
    }
}
