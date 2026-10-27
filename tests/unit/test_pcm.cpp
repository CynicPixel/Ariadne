// PCM (design robust-backend.md §5, D72):
//   - the odometry leg: pose and LEFT covariance from the chain (forward and backward storage), Monte Carlo NEES;
//   - the pair statistic: Monte Carlo mean ~ 6 (chi^2_6) for disjoint, nested, crossing and backward legs on ONE
//     chain (shared odometry edges counted once) and for inter-agent pairs, including a loop stored reversed;
//   - clique solvers: greedy is a maximal clique, exact is maximum, on planted-clique graphs;
//   - group rules (1 loop: untested; no pair consistent: no consensus), determinism under edge reordering;
//   - acceptance fixed before measuring: recall >= 0.95 on every seed at <= 40 % outliers; precision reported;
//   - the known blind spot (catalogue §17.2): a self-consistent wrong cluster passes. Asserted, labelled.
#include "slam/robust/pcm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <random>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec6;

namespace {

Vec6 Draw(std::mt19937& g, const Mat6& cov) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 z;
    for (int i = 0; i < 6; ++i) z(i) = n(g);
    return Eigen::LLT<Mat6>(cov).matrixL() * z;
}

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

struct World {
    std::map<uint64_t, SE3> truth;
    PoseGraph g;
};

/// One agent's chain: ids first..first+n-1, a random walk (large turns), noisy odometry with left covariance `odo`.
void AddChain(World& w, std::mt19937& g, uint64_t first, int n, const Mat6& odo, bool store_backward = false) {
    std::normal_distribution<double> nz(0.0, 1.0);
    SE3 T;
    T.t = core::Vec3{nz(g), nz(g), 0.0} * 5.0;
    for (int i = 0; i < n; ++i) {
        if (i > 0) {
            Vec6 step;
            step << 0.5 + 0.1 * nz(g), 0.1 * nz(g), 0.02 * nz(g), 0.02 * nz(g), 0.02 * nz(g), 0.3 * nz(g);
            T = T * core::SE3Exp(step);
        }
        w.truth[first + uint64_t(i)] = T;
        PoseNode nd;
        nd.keyframe = first + uint64_t(i);
        nd.T_W_C = T;
        w.g.AddNode(nd);
    }
    for (int i = 0; i + 1 < n; ++i) {
        uint64_t a = first + uint64_t(i), b = a + 1;
        if (store_backward && i % 2) std::swap(a, b);
        const SE3 Zt = w.truth[a].inverse() * w.truth[b];
        // OdometryEdge's covariance is LEFT on Z; draw the noise the same way.
        w.g.AddEdge(OdometryEdge(a, b, core::SE3Exp(Draw(g, odo)) * Zt, odo));
    }
}

size_t AddLoop(World& w, std::mt19937& g, uint64_t a, uint64_t b, const Mat6& cov, const SE3* wrong = nullptr) {
    const SE3 Zt = w.truth[a].inverse() * w.truth[b];
    const SE3 Z = wrong ? *wrong : core::SE3Exp(Draw(g, cov)) * Zt;
    w.g.AddEdge(OdometryEdge(a, b, Z, cov, EdgeType::kLocalLoop));
    return w.g.edges().size() - 1;
}

SE3 RandomPose(std::mt19937& g) {   // uniform rotation, translation within +-5 m (the outlier protocol)
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    Eigen::Quaterniond q{n(g), n(g), n(g), n(g)};
    q.normalize();
    SE3 T;
    T.R = q.toRotationMatrix();
    T.t = core::Vec3{u(g), u(g), u(g)};
    return T;
}

const Mat6 kOdo = Diag(0.02, 0.005);
const Mat6 kLoop = Diag(0.05, 0.01);

}  // namespace

TEST(Pcm, OdometryLegIsTheEdgeForwardAndBackward) {
    World w;
    std::mt19937 g(1);
    AddChain(w, g, 0, 2, kOdo);
    SE3 O;
    Mat6 cov;
    ASSERT_TRUE(OdometryLeg(w.g, 0, 1, O, cov));
    EXPECT_LT(core::SE3Log(O.inverse() * w.g.edges()[0].Z).norm(), 1e-12);
    EXPECT_LT((cov - kOdo).norm(), 1e-12 * kOdo.norm());   // left covariance in = left covariance out
    ASSERT_TRUE(OdometryLeg(w.g, 1, 0, O, cov));            // the inverse: left covariance Ad(Z^-1) S Ad(Z^-1)^T
    const Mat6 A = core::SE3Adjoint(w.g.edges()[0].Z.inverse());
    EXPECT_LT((cov - A * kOdo * A.transpose()).norm(), 1e-9 * kOdo.norm());
}

TEST(Pcm, OdometryLegMonteCarloNees) {
    std::mt19937 g(2);
    const int trials = 400;
    for (const bool backward_storage : {false, true}) {
        double nees_f = 0, nees_b = 0;
        for (int k = 0; k < trials; ++k) {
            World w;
            AddChain(w, g, 0, 40, kOdo, backward_storage);
            SE3 O;
            Mat6 cov;
            for (const auto& [a, c, acc] : {std::tuple{5ull, 35ull, &nees_f}, std::tuple{35ull, 5ull, &nees_b}}) {
                ASSERT_TRUE(OdometryLeg(w.g, a, c, O, cov));
                const Vec6 e = core::SE3Log(O * (w.truth[a].inverse() * w.truth[c]).inverse());   // left error
                *acc += e.dot(cov.ldlt().solve(e));
            }
        }
        std::printf("odometry leg NEES (storage %s): forward %.2f backward %.2f\n", backward_storage ? "mixed" : "forward",
                    nees_f / trials, nees_b / trials);
        EXPECT_NEAR(nees_f / trials, 6.0, 4 * std::sqrt(12.0 / trials) + 0.2);
        EXPECT_NEAR(nees_b / trials, 6.0, 4 * std::sqrt(12.0 / trials) + 0.2);
    }
}

// The pair statistic of two TRUE loops is chi^2_6: its mean over trials must be 6 whatever the legs' geometry. The
// crossing case shares odometry edges [20, 90) between the legs: summing the legs' covariances would double count.
TEST(Pcm, PairStatisticIsChiSquare6ForEveryLegConfiguration) {
    struct Cfg {
        const char* name;
        uint64_t a, b, c, d;
    };
    const Cfg cfgs[] = {{"disjoint", 10, 40, 60, 100}, {"nested", 10, 100, 30, 60}, {"crossing", 10, 100, 90, 20},
                        {"backward", 100, 10, 20, 90}, {"shared-end", 10, 100, 10, 60}};
    std::mt19937 g(3);
    const int trials = 400;
    const double tol = 4 * std::sqrt(12.0 / trials) + 0.2;
    for (const Cfg& c : cfgs) {
        double m = 0;
        for (int k = 0; k < trials; ++k) {
            World w;
            AddChain(w, g, 0, 120, kOdo);
            const size_t l1 = AddLoop(w, g, c.a, c.b, kLoop), l2 = AddLoop(w, g, c.c, c.d, kLoop);
            m += PairConsistency(w.g, l1, l2);
        }
        std::printf("pair statistic, one chain, %-10s mean %.2f (chi^2_6: 6)\n", c.name, m / trials);
        EXPECT_NEAR(m / trials, 6.0, tol) << c.name;
    }
    // Inter-agent: agent 0 = ids 0.., agent 1 = ids 1000..; loop 2 stored a->b and reversed b->a.
    for (const bool reversed : {false, true}) {
        double m = 0;
        for (int k = 0; k < trials; ++k) {
            World w;
            AddChain(w, g, 0, 60, kOdo);
            AddChain(w, g, 1000, 60, kOdo);
            const size_t l1 = AddLoop(w, g, 10, 1030, kLoop);
            const size_t l2 = reversed ? AddLoop(w, g, 1005, 45, kLoop) : AddLoop(w, g, 45, 1005, kLoop);
            m += PairConsistency(w.g, l1, l2);
        }
        std::printf("pair statistic, two agents, loop 2 %-8s mean %.2f\n", reversed ? "reversed" : "forward", m / trials);
        EXPECT_NEAR(m / trials, 6.0, tol);
    }
}

TEST(Pcm, CliqueSolversOnPlantedCliques) {
    std::mt19937 g(4);
    int same = 0;
    for (int t = 0; t < 200; ++t) {
        const int n = 8 + int(g() % 17), k = 3 + int(g() % 6);
        std::bernoulli_distribution edge(0.25);
        std::vector<std::vector<bool>> adj(size_t(n), std::vector<bool>(size_t(n), false));
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) adj[size_t(i)][size_t(j)] = adj[size_t(j)][size_t(i)] = edge(g);
        std::vector<int> planted(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) planted[size_t(i)] = i;
        std::shuffle(planted.begin(), planted.end(), g);
        planted.resize(size_t(k));
        for (int i : planted)
            for (int j : planted)
                if (i != j) adj[size_t(i)][size_t(j)] = true;
        auto is_clique = [&](const std::vector<int>& c) {
            for (int i : c)
                for (int j : c)
                    if (i != j && !adj[size_t(i)][size_t(j)]) return false;
            return true;
        };
        const auto ex = MaxClique(adj, true), gr = MaxClique(adj, false);
        ASSERT_TRUE(is_clique(ex));
        ASSERT_TRUE(is_clique(gr));
        EXPECT_GE(int(ex.size()), k);
        EXPECT_LE(gr.size(), ex.size());
        for (int v = 0; v < n; ++v) {   // greedy result is maximal: no vertex extends it
            if (std::find(gr.begin(), gr.end(), v) != gr.end()) continue;
            std::vector<int> ext = gr;
            ext.push_back(v);
            EXPECT_FALSE(is_clique(ext));
        }
        same += ex.size() == gr.size();
    }
    std::printf("greedy clique size == exact on %d / 200 planted-clique graphs (8-24 vertices)\n", same);
}

TEST(Pcm, GroupRules) {
    std::mt19937 g(5);
    World w;
    AddChain(w, g, 0, 80, kOdo);
    const SE3 bad = RandomPose(g);
    AddLoop(w, g, 5, 60, kLoop);
    {
        const PcmReport r = Pcm(w.g);
        ASSERT_EQ(r.groups.size(), 1u);
        EXPECT_EQ(r.groups[0].verdict, "untested");
        EXPECT_EQ(r.accepted.size(), 1u);
    }
    AddLoop(w, g, 20, 70, kLoop, &bad);   // inconsistent with the first
    const PcmReport r = Pcm(w.g);
    EXPECT_EQ(r.groups[0].verdict, "no_consensus");
    EXPECT_TRUE(r.accepted.empty());
    EXPECT_EQ(r.rejected.size(), 2u);
}

TEST(Pcm, RefusesAnOdometryGraphThatIsNotAChain) {
    std::mt19937 g(6);
    World w;
    AddChain(w, g, 0, 10, kOdo);
    w.g.AddEdge(OdometryEdge(3, 7, w.truth[3].inverse() * w.truth[7], kOdo));   // a branch: node 3 has degree 3
    EXPECT_THROW(Pcm(w.g), std::runtime_error);
}

// Acceptance (fixed before measuring, design §5): synthetic single chain, true loops between revisits, outliers =
// random node pairs with uniformly random SE(3) measurements; recall of true loops >= 0.95 on every seed at <= 40 %,
// gated for the default threshold. The fixed quantiles (arch §14.5's chi^2_6(0.95), and 0.999) are reported.
TEST(Pcm, OutlierRatesRecallAndPrecision) {
    struct Mode {
        const char* name;
        PcmParams p;
        bool gated;
    };
    PcmParams q95, q999;
    q95.alpha_loop = q999.alpha_loop = 0.0;
    q95.chi2_threshold = 12.592;
    q999.chi2_threshold = 22.458;
    PcmParams a05;
    a05.alpha_loop = 0.05;
    const Mode modes[] = {{"per-loop alpha 0.001 (default)", PcmParams{}, true},
                          {"per-loop alpha 0.05", a05, false},
                          {"fixed chi2_6(0.95) (arch)", q95, false},
                          {"fixed chi2_6(0.999)", q999, false}};
    for (const Mode& m : modes)
        for (const double rate : {0.1, 0.2, 0.3, 0.4}) {
            double worst_recall = 1, sum_recall = 0, sum_prec = 0, worst_prec = 1;
            for (int seed = 0; seed < 20; ++seed) {
                std::mt19937 g(1000u * unsigned(rate * 10) + unsigned(seed));
                World w;
                AddChain(w, g, 0, 300, kOdo);
                const int n_true = 30, n_out = int(std::lround(n_true * rate / (1 - rate)));
                std::vector<size_t> truth_loops;
                std::uniform_int_distribution<int> node(0, 299);
                for (int k = 0; k < n_true; ++k) {
                    const int a = node(g) % 250, b = a + 20 + int(g() % 30);
                    truth_loops.push_back(AddLoop(w, g, uint64_t(a), uint64_t(b), kLoop));
                }
                for (int k = 0; k < n_out; ++k) {
                    int a = node(g), b = node(g);
                    if (a == b) b = (a + 7) % 300;
                    const SE3 bad = RandomPose(g);
                    AddLoop(w, g, uint64_t(a), uint64_t(b), kLoop, &bad);
                }
                const PcmReport r = Pcm(w.g, m.p);
                int tp = 0;
                for (size_t l : r.accepted)
                    tp += std::find(truth_loops.begin(), truth_loops.end(), l) != truth_loops.end();
                const double recall = double(tp) / n_true;
                const double prec = r.accepted.empty() ? 1.0 : double(tp) / double(r.accepted.size());
                worst_recall = std::min(worst_recall, recall);
                worst_prec = std::min(worst_prec, prec);
                sum_recall += recall;
                sum_prec += prec;
                if (m.gated) EXPECT_GE(recall, 0.95) << "rate " << rate << " seed " << seed;
            }
            std::printf("%-30s outliers %2.0f%%: recall mean %.3f worst %.3f | precision mean %.3f worst %.3f\n", m.name,
                        rate * 100, sum_recall / 20, worst_recall, sum_prec / 20, worst_prec);
        }
}

// Near-miss outliers (aliasing-like: the true relative pose off by err_t m and err_r rad in a random direction) are
// what a wider chi^2 bound could let through. Reported, not gated: the default against the arch quantile, 30 true
// loops + 10 near misses, 20 seeds.
TEST(Pcm, NearMissOutliersPrecisionReported) {
    PcmParams q95;
    q95.alpha_loop = 0.0;
    q95.chi2_threshold = 12.592;
    for (const auto& [err_t, err_r] : {std::pair{0.2, 0.02}, {0.5, 0.087}, {1.0, 0.17}})
        for (const auto& [name, p] : {std::pair{"default", PcmParams{}}, {"arch 0.95", q95}}) {
            double sum_prec = 0, sum_recall = 0;
            for (int seed = 0; seed < 20; ++seed) {
                std::mt19937 g(77u + unsigned(seed));
                World w;
                AddChain(w, g, 0, 300, kOdo);
                std::vector<size_t> truth_loops;
                for (int k = 0; k < 30; ++k) {
                    const int a = int(g() % 250), b = a + 20 + int(g() % 30);
                    truth_loops.push_back(AddLoop(w, g, uint64_t(a), uint64_t(b), kLoop));
                }
                std::normal_distribution<double> n(0.0, 1.0);
                for (int k = 0; k < 10; ++k) {
                    const int a = int(g() % 250), b = a + 20 + int(g() % 30);
                    const core::Vec3 dt{n(g), n(g), n(g)}, dr{n(g), n(g), n(g)};
                    Vec6 xi;
                    xi << err_t * dt.normalized(), err_r * dr.normalized();
                    const SE3 bad = core::SE3Exp(xi) * (w.truth[uint64_t(a)].inverse() * w.truth[uint64_t(b)]);
                    AddLoop(w, g, uint64_t(a), uint64_t(b), kLoop, &bad);
                }
                const PcmReport r = Pcm(w.g, p);
                int tp = 0;
                for (size_t l : r.accepted)
                    tp += std::find(truth_loops.begin(), truth_loops.end(), l) != truth_loops.end();
                sum_prec += r.accepted.empty() ? 1.0 : double(tp) / double(r.accepted.size());
                sum_recall += tp / 30.0;
            }
            std::printf("near-miss outliers %.1f m / %2.0f deg, %-9s: precision mean %.3f, recall mean %.3f\n", err_t,
                        err_r * 180 / M_PI, name, sum_prec / 20, sum_recall / 20);
        }
}

TEST(Pcm, ChiSquare6QuantileIsTheInverseOfItsSurvival) {
    EXPECT_NEAR(ChiSquare6Quantile(0.95), 12.5916, 1e-3);   // table value
    EXPECT_NEAR(ChiSquare6Quantile(0.99), 16.8119, 1e-3);
    for (const double p : {0.5, 0.9, 0.999, 1 - 1e-6}) EXPECT_NEAR(1 - ChiSquare6Survival(ChiSquare6Quantile(p)), p, 1e-12);
}

// Known blind spot (catalogue §17.2, arch §14.5): loops that alias the same wrong place agree with each other. Five
// loops into a region displaced by one rigid G are mutually consistent; with only three true loops, PCM keeps the
// wrong five. This asserts the CURRENT behaviour; the defence is the degeneracy gate before PCM (and Iteration 7's
// aliasing work), not PCM.
TEST(Pcm, BlindSpotSelfConsistentWrongClusterPasses) {
    std::mt19937 g(7);
    World w;
    AddChain(w, g, 0, 200, kOdo);
    std::vector<size_t> good, wrong;
    for (const auto& [a, b] : {std::pair{10ull, 120ull}, {30ull, 140ull}, {50ull, 160ull}})
        good.push_back(AddLoop(w, g, a, b, kLoop));
    core::Vec6 gx;
    gx << 3.0, -1.0, 0.0, 0.0, 0.0, 0.8;
    const SE3 G = core::SE3Exp(gx);   // the aliased copy of the region around nodes 170-190
    for (uint64_t k = 0; k < 5; ++k) {
        const uint64_t a = 60 + 5 * k, b = 170 + 4 * k;
        const SE3 Z = w.truth[a].inverse() * G * w.truth[b];
        wrong.push_back(AddLoop(w, g, a, b, kLoop, &Z));
    }
    const PcmReport r = Pcm(w.g);
    std::printf("blind spot: accepted %zu loops, the %s cluster\n", r.accepted.size(),
                r.accepted == wrong ? "WRONG (aliased)" : "other");
    EXPECT_EQ(r.accepted, wrong);   // labelled: this is the known failure, not a goal
}

TEST(Pcm, DeterministicUnderEdgeReordering) {
    std::mt19937 g(8);
    World w;
    AddChain(w, g, 0, 150, kOdo);
    for (int k = 0; k < 15; ++k) AddLoop(w, g, uint64_t(5 * k), uint64_t(5 * k + 40), kLoop);
    for (int k = 0; k < 6; ++k) {
        const SE3 bad = RandomPose(g);
        AddLoop(w, g, uint64_t(7 * k + 3), uint64_t(7 * k + 90), kLoop, &bad);
    }
    auto ids = [](const PoseGraph& pg, const std::vector<size_t>& idx) {
        std::vector<std::pair<uint64_t, uint64_t>> out;
        for (size_t i : idx) out.push_back({pg.edges()[i].from, pg.edges()[i].to});
        std::sort(out.begin(), out.end());
        return out;
    };
    const auto ref = ids(w.g, Pcm(w.g).accepted);
    std::vector<PoseEdge> edges = w.g.edges();
    for (int rep = 0; rep < 5; ++rep) {
        std::shuffle(edges.begin(), edges.end(), g);
        PoseGraph s;
        for (const auto& [id, nd] : w.g.nodes()) s.AddNode(nd);
        for (const auto& e : edges) s.AddEdge(e);
        EXPECT_EQ(ids(s, Pcm(s).accepted), ref);
    }
}

TEST(Pcm, TimingAtThreeHundredLoops) {
    std::mt19937 g(9);
    World w;
    AddChain(w, g, 0, 3000, kOdo);
    for (int k = 0; k < 240; ++k) {
        const int a = int(g() % 2900);
        AddLoop(w, g, uint64_t(a), uint64_t(a + 20 + int(g() % 80)), kLoop);
    }
    for (int k = 0; k < 60; ++k) {
        const SE3 bad = RandomPose(g);
        AddLoop(w, g, uint64_t(g() % 3000), uint64_t(g() % 3000), kLoop, &bad);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const PcmReport r = Pcm(w.g);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("PCM: 3000 nodes, 300 loops (60 outliers): %.1f ms, accepted %zu (reported, not gated)\n", ms,
                r.accepted.size());
}
