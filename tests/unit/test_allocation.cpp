// 4.4 allocation (design ALLOCATION_4.4 §7, D115 + correction): Table I row by row (expected actions written from
// ALGORITHMS §22.3.1, not from the code); CBBA = SGA exactly at L_t = 1 (Theorem 1) within N_min D rounds; at L_t = 2
// the insertion marginal is not DMG (pinned), so at L_t = 2 Johnson's bid warping: eq. 9, conflict-free, N_min D;
// conflict-free convergence on a jointly connected switching graph within rho N_min; Theorem 2's 50 % bound against
// brute force; the two-agent sorted scan equals brute force; eq. 7 holds for the time-discounted score and fails for
// an additive one (the reason for D115's score).
#include "planning/mission/allocation.hpp"
#include "planning/mission/allocation_wire.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <queue>
#include <random>

using namespace ariadne::planning;

namespace {

using A = CbbaAction;

// Agents i = 0 (receiver), k = 1 (sender), m = 2, n = 3; "none" = -1. s vectors over the 4 agents.
struct Row {
    int z_kj, z_ij;
    double y_kj, y_ij;
    std::vector<int64_t> s_k, s_i;
    A want;
    const char* what;
};

AuctionInstance RandomInstance(std::mt19937& rng, int U, int N, int L) {
    std::uniform_real_distribution<double> u(0, 1);
    AuctionInstance in;
    in.n_agents = U, in.max_bundle = L;
    std::vector<std::pair<double, double>> ap(static_cast<size_t>(U)), tp(static_cast<size_t>(N));
    for (auto& p : ap) p = {10 * u(rng), 10 * u(rng)};
    for (auto& p : tp) p = {10 * u(rng), 10 * u(rng)};
    for (int j = 0; j < N; ++j) in.task_ids.push_back(1000 + j);
    in.value.assign(size_t(U), std::vector<double>(size_t(N)));
    in.d0 = in.value;
    in.leg.assign(size_t(U), std::vector<std::vector<double>>(size_t(N), std::vector<double>(size_t(N))));
    for (int a = 0; a < U; ++a) {
        in.rate.push_back(0.05 + 0.3 * u(rng));
        for (int j = 0; j < N; ++j) {
            in.value[size_t(a)][size_t(j)] = u(rng) < 0.1 ? 0.0 : 0.2 + u(rng);   // some agents cannot do some tasks
            in.d0[size_t(a)][size_t(j)] = std::hypot(ap[size_t(a)].first - tp[size_t(j)].first, ap[size_t(a)].second - tp[size_t(j)].second);
            for (int k = 0; k < N; ++k)   // Euclidean legs: the triangle inequality, hence DMG
                in.leg[size_t(a)][size_t(j)][size_t(k)] = std::hypot(tp[size_t(j)].first - tp[size_t(k)].first, tp[size_t(j)].second - tp[size_t(k)].second);
        }
    }
    return in;
}

std::vector<std::vector<int>> RandomConnected(std::mt19937& rng, int U, int* diameter) {
    std::vector<std::vector<int>> G(size_t(U), std::vector<int>(size_t(U), 0));
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 1; i < U; ++i) {   // a random spanning tree, plus random extra edges
        const int p = std::uniform_int_distribution<int>(0, i - 1)(rng);
        G[size_t(i)][size_t(p)] = G[size_t(p)][size_t(i)] = 1;
    }
    for (int i = 0; i < U; ++i)
        for (int j = i + 1; j < U; ++j)
            if (u(rng) < 0.2) G[size_t(i)][size_t(j)] = G[size_t(j)][size_t(i)] = 1;
    *diameter = 0;
    for (int s = 0; s < U; ++s) {   // BFS eccentricities
        std::vector<int> d(size_t(U), -1);
        std::queue<int> q;
        d[size_t(s)] = 0, q.push(s);
        while (!q.empty()) {
            const int x = q.front();
            q.pop();
            for (int y = 0; y < U; ++y)
                if (G[size_t(x)][size_t(y)] && d[size_t(y)] < 0) d[size_t(y)] = d[size_t(x)] + 1, q.push(y);
        }
        *diameter = std::max(*diameter, *std::max_element(d.begin(), d.end()));
    }
    return G;
}

}  // namespace

TEST(Cbba, TableIRowByRow) {
    const std::vector<int64_t> old{0, 0, 1, 1}, nw{0, 0, 2, 2};   // nw: newer info about m (2) and n (3)
    const std::vector<Row> rows = {
        // z_kj = k
        {1, 0, 0.9, 0.5, old, old, A::kUpdate, "k|i y_kj > y_ij"},
        {1, 0, 0.4, 0.5, old, old, A::kLeave, "k|i y_kj < y_ij"},
        {1, 1, 0.1, 0.9, old, old, A::kUpdate, "k|k"},
        {1, 2, 0.4, 0.5, nw, old, A::kUpdate, "k|m s_km > s_im"},
        {1, 2, 0.9, 0.5, old, old, A::kUpdate, "k|m y_kj > y_ij"},
        {1, 2, 0.4, 0.5, old, old, A::kLeave, "k|m neither"},
        {1, -1, 0.4, 0.0, old, old, A::kUpdate, "k|none"},
        // z_kj = i
        {0, 0, 0.9, 0.5, old, old, A::kLeave, "i|i"},
        {0, 1, 0.9, 0.5, old, old, A::kReset, "i|k"},
        {0, 2, 0.9, 0.5, nw, old, A::kReset, "i|m s_km > s_im"},
        {0, 2, 0.9, 0.5, old, old, A::kLeave, "i|m not newer"},
        {0, -1, 0.9, 0.0, old, old, A::kLeave, "i|none"},
        // z_kj = m
        {2, 0, 0.9, 0.5, nw, old, A::kUpdate, "m|i newer and y_kj > y_ij"},
        {2, 0, 0.4, 0.5, nw, old, A::kLeave, "m|i newer, lower bid"},
        {2, 0, 0.9, 0.5, old, old, A::kLeave, "m|i higher bid, not newer"},
        {2, 1, 0.4, 0.5, nw, old, A::kUpdate, "m|k newer"},
        {2, 1, 0.4, 0.5, old, old, A::kReset, "m|k else reset"},
        {2, 2, 0.4, 0.5, nw, old, A::kUpdate, "m|m newer"},
        {2, 2, 0.4, 0.5, old, old, A::kLeave, "m|m not newer"},
        {2, 3, 0.4, 0.5, nw, old, A::kUpdate, "m|n s_km > s_im and s_kn > s_in"},
        {2, 3, 0.9, 0.5, {0, 0, 2, 1}, old, A::kUpdate, "m|n s_km > s_im and y_kj > y_ij"},
        {2, 3, 0.4, 0.5, {0, 0, 2, 1}, old, A::kLeave, "m|n newer m only, lower bid"},
        {2, 3, 0.9, 0.5, {0, 0, 1, 2}, {0, 0, 2, 1}, A::kReset, "m|n s_kn > s_in and s_im > s_km"},
        {2, 3, 0.9, 0.5, old, old, A::kLeave, "m|n nothing newer"},
        {2, -1, 0.4, 0.0, nw, old, A::kUpdate, "m|none newer"},
        {2, -1, 0.4, 0.0, old, old, A::kLeave, "m|none not newer"},
        // z_kj = none
        {-1, 0, 0.0, 0.5, nw, old, A::kLeave, "none|i"},
        {-1, 1, 0.0, 0.5, old, old, A::kUpdate, "none|k"},
        {-1, 2, 0.0, 0.5, nw, old, A::kUpdate, "none|m newer"},
        {-1, 2, 0.0, 0.5, old, old, A::kLeave, "none|m not newer"},
        {-1, -1, 0.0, 0.0, nw, old, A::kLeave, "none|none"},
    };
    for (const auto& r : rows) EXPECT_EQ(TableI(0, 1, r.z_kj, r.z_ij, r.y_kj, r.y_ij, r.s_k, r.s_i), r.want) << r.what;
}

TEST(Cbba, EqualsSequentialGreedyAtBundleOne) {
    // L_t = 1: every bid is a marginal on the empty bundle, so no stale bid exists and Theorem 1's equality is exact.
    std::mt19937 rng(115);
    for (int trial = 0; trial < 2000; ++trial) {
        const int U = 2 + trial % 7, N = 3 + trial % 10;
        const AuctionInstance in = RandomInstance(rng, U, N, 1);
        int D = 0;
        const auto G = RandomConnected(rng, U, &D);
        CbbaParams p;
        p.bytes = TestCodecBytes;
        const Allocation c = RunCbba(in, [&](int) { return G; }, p);
        const Allocation g = SequentialGreedy(in);
        ASSERT_TRUE(c.converged) << trial;
        ASSERT_EQ(c.winner, g.winner) << "trial " << trial << " U " << U << " N " << N;
        for (int a = 0; a < U; ++a) ASSERT_EQ(c.agents[size_t(a)].bids, g.agents[size_t(a)].bids) << trial;
        EXPECT_EQ(c.tail_releases, 0u);
        EXPECT_LE(c.rounds, std::min(N, U) * D + 1) << "Theorem 1(2): N_min D, +1 detection round; trial " << trial;
        EXPECT_GT(c.bytes, 0u);
    }
}

TEST(Cbba, BundleTwoWithBidWarping) {
    // L_t = 2 (the reported row): the insertion marginal is not DMG (DiscountedInsertionMarginalIsNotDmg), so
    // Theorem 1's SGA equality does not apply. With Johnson et al.'s bid warping (CDC 2012, eq. 6) the broadcast bids
    // never increase along a bundle (Choi eq. 9), which is asserted; conflict-free convergence within N_min D (+1
    // detection round) is asserted; the divergence from raw-score SGA and the ratio to brute force are reported.
    std::mt19937 rng(116);
    int diverged = 0, total = 0;
    double worst = 1;
    for (int trial = 0; trial < 1000; ++trial) {
        const int U = 2 + trial % 3, N = 3 + trial % 6;   // small enough for brute force
        const AuctionInstance in = RandomInstance(rng, U, N, 2);
        int D = 0;
        const auto G = RandomConnected(rng, U, &D);
        const Allocation c = RunCbba(in, [&](int) { return G; }, CbbaParams{});
        const Allocation g = SequentialGreedy(in);
        ASSERT_TRUE(c.converged) << trial;
        std::vector<int> owners(size_t(N), 0);
        for (const auto& a : c.agents) {
            for (int j : a.bundle) ++owners[size_t(j)];
            for (size_t n = 1; n < a.bids.size(); ++n) ASSERT_LE(a.bids[n], a.bids[n - 1]) << "eq. 9, trial " << trial;
        }
        for (int o : owners) ASSERT_LE(o, 1) << "conflict, trial " << trial;
        ASSERT_LE(c.rounds, std::min(N, U * 2) * D + 1) << "N_min D, trial " << trial;
        ++total;
        diverged += c.winner != g.winner;
        const double opt = BruteForceOptimum(in);
        if (opt > 0) worst = std::min(worst, c.total / opt);
        EXPECT_GE(c.total, 0.5 * opt - 1e-12) << "empirical at L_t = 2; trial " << trial;
    }
    std::printf("[cbba] L_t=2 warped: %d / %d differ from raw-score SGA; worst CBBA/opt %.3f\n", diverged, total, worst);
}

TEST(Cbba, StaleSecondSlotBidRegression) {
    // Hand-built (ALLOCATION_4.4 §7, worked from ALGORITHMS §22.3.1). Line topology B - A - C; positions on a line:
    // A 0, B 10, C 1.5; tasks X 1, Y 2, W -1, P 11, Z 12. Values: A {X 10, Y 15, W 8} rate 1; B {Y 3, P 10, Z 1}
    // rate 0.1; C {X 20} rate 1.
    //  round 1: A builds [X 3.679, Y 2.030]; B [P 9.048, Y 1.104]; C [X 12.13]. A hears C: loses X, eq. 6 resets Y
    //           (tail). B hears A's (stale) Y 2.030 > 1.104: loses Y.
    //  round 2: A builds [W 2.943, Y 0.275]; B, seeing Y held at 2.030, fills [P, Z 0.819] - full. B then hears A's
    //           Y 0.275 but never re-enters BuildBundle. Final CBBA: A {W, Y}, B {P, Z}, C {X}.
    //  SGA: C-X 12.13, B-P 9.048, A-W 2.943, B-Y 1.104 (> A's 0.275 and B's Z 0.819). Final: A {W}, B {P, Y}, C {X}.
    const std::vector<double> pos_t = {1, 2, -1, 11, 12}, pos_a = {0, 10, 1.5};
    AuctionInstance in;
    in.n_agents = 3, in.max_bundle = 2, in.task_ids = {0, 1, 2, 3, 4};
    in.value = {{10, 15, 8, 0, 0}, {0, 3, 0, 10, 1}, {20, 0, 0, 0, 0}};
    in.rate = {1.0, 0.1, 1.0};
    in.d0.assign(3, std::vector<double>(5));
    in.leg.assign(3, std::vector<std::vector<double>>(5, std::vector<double>(5)));
    for (int a = 0; a < 3; ++a)
        for (int j = 0; j < 5; ++j) {
            in.d0[size_t(a)][size_t(j)] = std::abs(pos_a[size_t(a)] - pos_t[size_t(j)]);
            for (int k = 0; k < 5; ++k) in.leg[size_t(a)][size_t(j)][size_t(k)] = std::abs(pos_t[size_t(j)] - pos_t[size_t(k)]);
        }
    const std::vector<std::vector<int>> line = {{0, 1, 1}, {1, 0, 0}, {1, 0, 0}};   // A=0 linked to B=1 and C=2
    const Allocation c = RunCbba(in, [&](int) { return line; }, CbbaParams{});
    const Allocation g = SequentialGreedy(in);
    ASSERT_TRUE(c.converged);
    EXPECT_EQ(c.winner, (std::vector<int>{2, 0, 0, 1, 1})) << "CBBA: X->C, Y->A, W->A, P->B, Z->B";
    EXPECT_EQ(g.winner, (std::vector<int>{2, 1, 0, 1, -1})) << "SGA: X->C, Y->B, W->A, P->B, Z unassigned";
    EXPECT_GE(c.tail_releases, 1u);
    EXPECT_NEAR(Marginal(in, 0, {0}, 1), 15 * std::exp(-2.0), 1e-12) << "A's stale second-slot bid on Y";
    EXPECT_NEAR(Marginal(in, 0, {2}, 1), 15 * std::exp(-4.0), 1e-12) << "A's re-bid on Y behind W";
}

TEST(Cbba, DiscountedInsertionMarginalIsNotDmg) {
    // Choi eq. 7 needs c_ij[b] >= c_ij[b (+)end {k}] for EVERY b; eqs. 11-12's argument (each task arrives later on a
    // longer path) covers the path score, not its insertion marginal. The stale-bid instance's agent A: Y behind W
    // (path W) earns 15 e^-4 = 0.275; once X is in the path (X, W), Y slots between them and earns more.
    const std::vector<double> pos_t = {1, 2, -1, 11, 12};
    AuctionInstance in;
    in.n_agents = 1, in.max_bundle = 3, in.task_ids = {0, 1, 2, 3, 4};
    in.value = {{10, 15, 8, 0, 0}};
    in.rate = {1.0};
    in.d0.assign(1, std::vector<double>(5));
    in.leg.assign(1, std::vector<std::vector<double>>(5, std::vector<double>(5)));
    for (int j = 0; j < 5; ++j) {
        in.d0[0][size_t(j)] = std::abs(pos_t[size_t(j)]);
        for (int k = 0; k < 5; ++k) in.leg[0][size_t(j)][size_t(k)] = std::abs(pos_t[size_t(j)] - pos_t[size_t(k)]);
    }
    const double before = Marginal(in, 0, {2}, 1), after = Marginal(in, 0, {0, 2}, 1);
    EXPECT_NEAR(before, 15 * std::exp(-4.0), 1e-12);
    EXPECT_NEAR(after, 1.6856362775989622, 1e-9);   // scratchpad dmg check (Python port of PathScore / Marginal)
    EXPECT_GT(after, before) << "eq. 7 fails: the marginal rose when X joined the bundle";
}

TEST(Cbba, ConflictFreeOnSwitchingGraph) {
    // Two graphs, each disconnected, jointly connected (rho = 2): {0-1, 2-3} and {1-2}.
    std::mt19937 rng(7);
    const auto Ga = std::vector<std::vector<int>>{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
    const auto Gb = std::vector<std::vector<int>>{{0, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}};
    for (int trial = 0; trial < 20; ++trial) {
        const AuctionInstance in = RandomInstance(rng, 4, 6, 2);
        const Allocation c = RunCbba(in, [&](int t) { return t % 2 ? Ga : Gb; }, CbbaParams{});
        ASSERT_TRUE(c.converged);
        std::vector<int> owners(6, 0);
        for (const auto& a : c.agents)
            for (int j : a.bundle) ++owners[size_t(j)];
        for (int o : owners) EXPECT_LE(o, 1) << "a task held twice";
        // The union graph is a path of diameter 3; rho = 2 => bound rho * N_min * D_union, +1 detection round.
        EXPECT_LE(c.rounds, 2 * std::min(6, 4 * 2) * 3 + 1);
        if (c.tail_releases == 0) EXPECT_EQ(c.winner, SequentialGreedy(in).winner) << "no stale bid: SGA's assignment";
    }
}

TEST(Cbba, HalfOptimalAndTwoAgentScanIsExact) {
    std::mt19937 rng(2);
    double worst = 1;
    for (int trial = 0; trial < 40; ++trial) {
        const int U = 2 + trial % 2, N = 3 + trial % 4;
        const AuctionInstance in = RandomInstance(rng, U, N, 2);
        const double opt = BruteForceOptimum(in);
        const Allocation g = SequentialGreedy(in);
        EXPECT_GE(g.total, 0.5 * opt - 1e-12) << "Theorem 2";
        worst = std::min(worst, opt > 0 ? g.total / opt : 1.0);
        if (U == 2) EXPECT_NEAR(TwoAgentOptimum(in), opt, 1e-12) << trial;
    }
    std::printf("[cbba] worst SGA/optimum over 40 instances: %.3f\n", worst);
}

TEST(Cbba, DiscountedScoreIsDmgAdditiveIsNot) {
    std::mt19937 rng(9);
    for (int trial = 0; trial < 50; ++trial) {
        const AuctionInstance in = RandomInstance(rng, 1, 5, 3);
        // eq. 7: c_ij[b] >= c_ij[b (+)end {k}] for every j, k not in b (here b = {} and {0}).
        for (int k = 0; k < 5; ++k)
            for (int j = 0; j < 5; ++j) {
                if (j == k || in.value[0][size_t(j)] <= 0 || in.value[0][size_t(k)] <= 0) continue;
                EXPECT_GE(Marginal(in, 0, {}, j) + 1e-12, Marginal(in, 0, {k}, j)) << trial;
            }
    }
    // Additive (gain - path) on a line: agent at 0, tasks at 10 and 11, gains 12 each. Alone, task 11 nets
    // 12 - 11 = 1; after task 10 is in the path its detour is 1, so it nets 12 - 1 = 11 > 1: eq. 7 fails.
    const double alone = 12 - 11, after = 12 - 1;
    EXPECT_GT(after, alone);
}

TEST(Cbba, WireBytesAtTheSerializer) {
    // D115 / invariant 6: a CBBA message measured as the TASK_CONSENSUS envelope it is on the wire; the test codec
    // (16 + 13 per task + 8 per agent) is a different number and never a claim.
    std::mt19937 rng(3);
    const AuctionInstance in = RandomInstance(rng, 2, 6, 1);
    const ariadne::core::WireCodec codec(ariadne::core::Framing::kWholeMessage);
    CbbaParams p;
    p.bytes = WireBytes(codec, std::vector<float>(8, 1.0f));
    const std::vector<std::vector<int>> G = {{0, 1}, {1, 0}};
    const Allocation c = RunCbba(in, [&](int) { return G; }, p);
    ASSERT_TRUE(c.converged);
    EXPECT_GT(c.messages, 0u);
    CbbaMessage m;
    m.sender = 1, m.task_ids = in.task_ids, m.y.assign(6, 0.5), m.z.assign(6, 0), m.s = {1, 1};
    const size_t one = WireBytes(codec, std::vector<float>(8, 1.0f))(m);
    EXPECT_LE(c.bytes, c.messages * one) << "sparse messages are never larger than a full one";
    std::printf("[cbba] TASK_CONSENSUS wire bytes for 6 tasks, all with winners: %zu (test codec %zu)\n", one, TestCodecBytes(m));
}

TEST(Cbba, SparseWireRoundTripIsExact) {
    // D115a: decode(encode(sparse)) expanded onto the task list reproduces the sender's snapshot exactly, so Table I
    // sees identical inputs; float y is the wire's precision, and the snapshot's y must survive it unchanged when
    // CBBA runs on decoded messages (checked below by running both).
    const ariadne::core::WireCodec codec(ariadne::core::Framing::kWholeMessage);
    CbbaMessage m;
    m.sender = 1, m.task_ids = {3, 8, 21, 40, 55}, m.y = {0, 0.75, 0, 1.5, 0}, m.z = {-1, 1, -1, 0, -1}, m.s = {4, 7};
    const auto enc = codec.EncodeCbbaConsensus(ToWire(m, {1.f, 2.f}), 1, 0, ariadne::core::Timestamp{0}, 0);
    ariadne::core::DecodedMessage d;
    ASSERT_EQ(codec.Decode(enc.bytes, &d), ariadne::core::WireCodec::Status::kOk);
    EXPECT_EQ(d.cbba_consensus.tasks.size(), 2u) << "only the two tasks with a winner travel";
    const CbbaMessage back = FromWire(d.cbba_consensus, 1, m.task_ids);
    EXPECT_EQ(back.z, m.z);
    EXPECT_EQ(back.y, m.y);   // doubles on the wire: exact
    EXPECT_EQ(back.s, m.s);
    // Size: a 160-task event where 2 tasks have winners.
    CbbaMessage big;
    big.sender = 0, big.s = {1, 1};
    for (int j = 0; j < 160; ++j) big.task_ids.push_back(1000 + j), big.y.push_back(j < 2 ? 0.5 : 0.0), big.z.push_back(j < 2 ? 0 : -1);
    const size_t sparse = codec.EncodeCbbaConsensus(ToWire(big, std::vector<float>(8, 1.f)), 0, 1, ariadne::core::Timestamp{0}, 0).record.wire_bytes;
    std::printf("[cbba] sparse TASK_CONSENSUS for 160 tasks, 2 with winners: %zu B\n", sparse);
    EXPECT_LT(sparse, 300u);
}

TEST(Cbba, ConvergesWithoutAgreementAcrossComponents) {
    // Two agents with no link: each converges on its own view (a disconnected fleet cannot agree); rounds stay small.
    std::mt19937 rng(11);
    const AuctionInstance in = RandomInstance(rng, 2, 5, 1);
    const std::vector<std::vector<int>> none = {{0, 0}, {0, 0}};
    const Allocation c = RunCbba(in, [&](int) { return none; }, CbbaParams{});
    EXPECT_TRUE(c.converged);
    EXPECT_LE(c.rounds, 2);
    EXPECT_EQ(c.messages, 0u);
}

TEST(Cbba, WireTransportLeavesTheAssignmentUnchanged) {
    // D115a end to end: CBBA whose every message is encoded (sparse), decoded and expanded gives the same assignment
    // and bids as CBBA on the in-memory snapshots, and is never larger on the wire than the dense encoding.
    std::mt19937 rng(117);
    const ariadne::core::WireCodec codec(ariadne::core::Framing::kWholeMessage);
    for (int trial = 0; trial < 300; ++trial) {
        const int U = 2 + trial % 5, N = 3 + trial % 12, L = 1 + trial % 2;
        const AuctionInstance in = RandomInstance(rng, U, N, L);
        int D = 0;
        const auto G = RandomConnected(rng, U, &D);
        const Allocation a = RunCbba(in, [&](int) { return G; }, CbbaParams{});
        CbbaParams w;
        w.transport = WireTransport(codec, std::vector<float>(8, 1.f));
        w.bytes = WireBytes(codec, std::vector<float>(8, 1.f));
        const Allocation b = RunCbba(in, [&](int) { return G; }, w);
        ASSERT_EQ(a.winner, b.winner) << trial;
        for (int i = 0; i < U; ++i) ASSERT_EQ(a.agents[size_t(i)].bids, b.agents[size_t(i)].bids) << trial;
        ASSERT_EQ(a.rounds, b.rounds) << trial;
    }
}
