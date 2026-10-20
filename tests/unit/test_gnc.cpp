// GNC-TLS and per-component anchoring (design ROBUST_BACKEND_2.5.md §6, D73 part 1).
// Definitions, fixed before the first measured run:
//   - inlier = final weight > 0.5; precision and recall are over LOOP edges only;
//   - every run starts from the odometry-composed poses (never a chordal init that includes the outliers);
//   - outliers (paper protocol) = random node pairs with uniformly random SE(3) measurements;
//   - near-miss gate (D72 decision): GNC on the loops PCM accepted, with 0.5 m / 5 deg near misses; precision
//     TP / (TP + accepted near misses), pooled over 20 seeds, >= 0.99. 0.2 m / 1 deg near misses: ATE reported.
// Gates: precision >= 0.99 and recall >= 0.95 on every seed at <= 40 % outliers, grid and sphere topologies.
#include "slam/robust/gnc.hpp"
#include "slam/robust/pcm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <random>
#include <tuple>

using namespace ariadne;
using namespace ariadne::slam;
using core::Vec3;
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

const Mat6 kOdo = Diag(0.02, 0.005);
const Mat6 kLoop = Diag(0.05, 0.01);

SE3 At(const Vec3& t, double yaw) {
    SE3 T;
    T.R = Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix();
    T.t = t;
    return T;
}

SE3 RandomPose(std::mt19937& g, double half_extent) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(-half_extent, half_extent);
    Eigen::Quaterniond q(n(g), n(g), n(g), n(g));
    q.normalize();
    SE3 T;
    T.R = q.toRotationMatrix();
    T.t = Vec3(u(g), u(g), u(g));
    return T;
}

struct Scenario {
    std::vector<SE3> truth;                          // index = keyframe id - first
    std::vector<std::pair<int, int>> loop_pairs;     // true loop candidates (indices into truth)
    PoseGraph g;
    std::vector<size_t> true_loops, outlier_loops;   // edge indices
    uint64_t first = 0;

    /// Nodes at the odometry-composed estimate; odometry edges with left covariance kOdo.
    void Build(std::mt19937& gen) {
        SE3 est = truth[0];
        for (size_t i = 0; i < truth.size(); ++i) {
            if (i > 0) est = est * (core::SE3Exp(Draw(gen, kOdo)) * (truth[i - 1].inverse() * truth[i]));
            PoseNode nd;
            nd.keyframe = first + i;
            nd.T_W_C = est;
            g.AddNode(nd);
            if (i > 0) {
                const SE3 Z = g.nodes().at(first + i - 1).T_W_C.inverse() * est;
                g.AddEdge(OdometryEdge(first + i - 1, first + i, Z, kOdo));
            }
        }
    }
    size_t Loop(std::mt19937& gen, int a, int b, const SE3* wrong = nullptr) {
        const SE3 Zt = truth[size_t(a)].inverse() * truth[size_t(b)];
        const SE3 Z = wrong ? *wrong : core::SE3Exp(Draw(gen, kLoop)) * Zt;
        g.AddEdge(OdometryEdge(first + uint64_t(a), first + uint64_t(b), Z, kLoop, EdgeType::kLocalLoop));
        return g.edges().size() - 1;
    }
    double Ate() const {
        double s = 0;
        for (size_t i = 0; i < truth.size(); ++i) s += (g.nodes().at(first + i).T_W_C.t - truth[i].t).squaredNorm();
        return std::sqrt(s / double(truth.size()));
    }
};

/// Boustrophedon walk over an nx x ny x nz lattice (1 m spacing); loop candidates between lattice neighbours that
/// are not consecutive on the walk (the g2o "grid" topology).
Scenario Grid(int nx, int ny, int nz) {
    Scenario s;
    std::map<std::tuple<int, int, int>, int> at;
    for (int z = 0; z < nz; ++z)
        for (int yy = 0; yy < ny; ++yy) {
            const int y = (z % 2) ? ny - 1 - yy : yy;
            for (int xx = 0; xx < nx; ++xx) {
                const int x = (yy % 2) ? nx - 1 - xx : xx;
                at[{x, y, z}] = int(s.truth.size());
                s.truth.push_back(At(Vec3(x, y, z), 0.3 * x - 0.2 * y + 0.5 * z));
            }
        }
    for (const auto& [p, i] : at) {
        const auto [x, y, z] = p;
        for (const auto& q : {std::tuple{x + 1, y, z}, std::tuple{x, y + 1, z}, std::tuple{x, y, z + 1}}) {
            const auto it = at.find(q);
            if (it != at.end() && std::abs(it->second - i) > 1) s.loop_pairs.push_back({std::min(i, it->second), std::max(i, it->second)});
        }
    }
    return s;
}

/// Rings on a sphere of radius 5 m, walked ring by ring; loop candidates between vertical neighbours.
Scenario Sphere(int rings, int per_ring) {
    Scenario s;
    for (int r = 0; r < rings; ++r)
        for (int j = 0; j < per_ring; ++j) {
            const double th = M_PI * (r + 1) / (rings + 1), ph = 2 * M_PI * j / per_ring;
            s.truth.push_back(At(5.0 * Vec3(std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)),
                                 ph + M_PI / 2));
        }
    for (int r = 0; r + 1 < rings; ++r)
        for (int j = 0; j < per_ring; ++j) s.loop_pairs.push_back({r * per_ring + j, (r + 1) * per_ring + j});
    return s;
}

struct Score {
    double precision = 1, recall = 1;
};
Score Classify(const Scenario& s, const GncReport& r) {
    int tp = 0;
    for (size_t l : r.inliers) tp += std::find(s.true_loops.begin(), s.true_loops.end(), l) != s.true_loops.end();
    Score sc;
    sc.precision = r.inliers.empty() ? 1.0 : double(tp) / double(r.inliers.size());
    sc.recall = s.true_loops.empty() ? 1.0 : double(tp) / double(s.true_loops.size());
    return sc;
}

/// A scenario with `n_true` true loops (sampled from the candidates) and outliers at `rate` of all loops.
Scenario WithOutliers(Scenario s, std::mt19937& g, int n_true, double rate) {
    s.Build(g);
    std::shuffle(s.loop_pairs.begin(), s.loop_pairs.end(), g);
    n_true = std::min<int>(n_true, int(s.loop_pairs.size()));
    for (int k = 0; k < n_true; ++k) s.true_loops.push_back(s.Loop(g, s.loop_pairs[size_t(k)].first, s.loop_pairs[size_t(k)].second));
    const int n_out = int(std::lround(n_true * rate / (1.0 - rate)));
    std::uniform_int_distribution<int> node(0, int(s.truth.size()) - 1);
    for (int k = 0; k < n_out; ++k) {
        int a = node(g), b = node(g);
        if (a == b) b = (a + 3) % int(s.truth.size());
        const SE3 bad = RandomPose(g, 5.0);
        s.outlier_loops.push_back(s.Loop(g, std::min(a, b), std::max(a, b), &bad));
    }
    return s;
}

}  // namespace

// Prop. 4's closed form is the minimiser over w in [0, 1] of the Black-Rangarajan dual w r^2 + mu c^2 (1-w)/(mu+w).
TEST(Gnc, TlsWeightIsTheArgminOfTheDual) {
    const double c2 = 16.812;
    for (const double mu : {0.01, 0.3, 1.0, 5.0, 100.0})
        for (const double r2 : {0.5, 5.0, 12.0, 16.812, 20.0, 40.0, 400.0}) {
            double best_w = 0, best = 1e300;
            for (int i = 0; i <= 200000; ++i) {
                const double w = i / 200000.0, f = w * r2 + mu * c2 * (1 - w) / (mu + w);
                if (f < best) best = f, best_w = w;
            }
            EXPECT_NEAR(TlsWeight(r2, mu, c2), best_w, 1e-4) << "mu " << mu << " r2 " << r2;
        }
    EXPECT_EQ(TlsWeight(1.0, 1.0, c2), 1.0);   // r2 <= mu/(mu+1) c2
    EXPECT_EQ(TlsWeight(40.0, 1.0, c2), 0.0);  // r2 >= (mu+1)/mu c2
}

TEST(Gnc, NoOutliersStopsAtOnceAndMatchesPlainLm) {
    std::mt19937 g(1);
    Scenario s = WithOutliers(Grid(5, 5, 2), g, 30, 0.0);
    PoseGraph plain = s.g;
    PgoSettings lm = PgoV2();
    lm.recompute_covariance = false;
    const PgoReport pr = SolvePoseGraph(plain, 0, lm);
    const GncReport r = SolveGnc(s.g, 0);
    std::printf("no outliers: stop %s, %zu inliers / %zu loops, energy %.6f vs plain %.6f\n", r.stop.c_str(),
                r.inliers.size(), r.loops.size(), r.final_energy, pr.final_energy);
    EXPECT_EQ(r.inliers.size(), r.loops.size());
    EXPECT_NEAR(r.final_energy, pr.final_energy, 1e-9 * std::max(1.0, pr.final_energy));
}

// Gate (§6): precision >= 0.99 and recall >= 0.95 on every seed at <= 40 % outliers; grid and sphere.
TEST(Gnc, OutlierRatesOnGridAndSphere) {
    for (const char* topo : {"grid", "sphere"})
        for (const double rate : {0.1, 0.2, 0.3, 0.4}) {
            double worst_p = 1, worst_r = 1, ate = 0, ms = 0;
            int outer = 0;
            for (int seed = 0; seed < 20; ++seed) {
                std::mt19937 g(100u * unsigned(rate * 10) + unsigned(seed) + (topo[0] == 's' ? 7000u : 0u));
                Scenario s = WithOutliers(topo[0] == 'g' ? Grid(6, 6, 3) : Sphere(8, 20), g, 60, rate);
                const auto t0 = std::chrono::steady_clock::now();
                const GncReport r = SolveGnc(s.g, 0);
                ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                const Score sc = Classify(s, r);
                worst_p = std::min(worst_p, sc.precision);
                worst_r = std::min(worst_r, sc.recall);
                ate += s.Ate();
                outer += int(r.mu.size());
                EXPECT_GE(sc.precision, 0.99) << topo << " rate " << rate << " seed " << seed;
                EXPECT_GE(sc.recall, 0.95) << topo << " rate " << rate << " seed " << seed;
            }
            std::printf("%-6s outliers %2.0f%%: precision worst %.3f, recall worst %.3f | ATE mean %.3f m | %.1f outer its,"
                        " %.0f ms mean\n", topo, rate * 100, worst_p, worst_r, ate / 20, outer / 20.0, ms / 20);
        }
}

// The 10 -> 90 % ramp: reported, no gate. Success = precision >= 0.99 and recall >= 0.95.
TEST(Gnc, OutlierRampReported) {
    for (const double rate : {0.5, 0.6, 0.7, 0.8, 0.9}) {
        int ok = 0;
        double ate = 0;
        for (int seed = 0; seed < 10; ++seed) {
            std::mt19937 g(9000u + 100u * unsigned(rate * 10) + unsigned(seed));
            Scenario s = WithOutliers(Grid(6, 6, 3), g, 60, rate);
            const GncReport r = SolveGnc(s.g, 0);
            const Score sc = Classify(s, r);
            ok += sc.precision >= 0.99 && sc.recall >= 0.95;
            ate += s.Ate();
        }
        std::printf("ramp grid outliers %2.0f%%: success %2d / 10, ATE mean %.3f m\n", rate * 100, ok, ate / 10);
    }
}

// D72 decision: GNC on the loops PCM accepted, with near misses mixed in.
TEST(Gnc, NearMissesAfterPcm) {
    for (const auto& [err_t, err_r, gated] : {std::tuple{0.5, 0.087, true}, {0.2, 0.0175, false}}) {
        int tp = 0, nm_accepted = 0, nm_after_pcm = 0;
        double ate_with = 0, ate_clean = 0;
        for (int seed = 0; seed < 20; ++seed) {
            std::mt19937 g(500u + unsigned(seed));
            Scenario s = WithOutliers(Grid(6, 6, 3), g, 60, 0.0);
            Scenario clean = s;
            std::vector<size_t> near;
            std::normal_distribution<double> n(0.0, 1.0);
            for (int k = 0; k < 10; ++k) {
                const auto [a, b] = s.loop_pairs[size_t(60 + k)];
                Vec6 xi;
                xi << err_t * Vec3(n(g), n(g), n(g)).normalized(), err_r * Vec3(n(g), n(g), n(g)).normalized();
                const SE3 bad = core::SE3Exp(xi) * (s.truth[size_t(a)].inverse() * s.truth[size_t(b)]);
                near.push_back(s.Loop(g, a, b, &bad));
            }
            const PcmReport pcm = Pcm(s.g);
            std::vector<bool> active(s.g.edges().size(), false);
            for (size_t l : pcm.accepted) active[l] = true;
            for (size_t l : near) nm_after_pcm += active[l];
            const GncReport r = SolveGnc(s.g, 0, &active);
            for (size_t l : r.inliers) {
                tp += std::find(s.true_loops.begin(), s.true_loops.end(), l) != s.true_loops.end();
                nm_accepted += std::find(near.begin(), near.end(), l) != near.end();
            }
            SolveGnc(clean.g, 0);
            ate_with += s.Ate();
            ate_clean += clean.Ate();
        }
        const double prec = double(tp) / double(tp + nm_accepted);
        std::printf("near misses %.1f m / %.0f deg: %d / 200 passed PCM, %d accepted by GNC | precision (pooled) %.4f |"
                    " ATE %.4f m with them, %.4f m without\n", err_t, err_r * 180 / M_PI, nm_after_pcm, nm_accepted, prec,
                    ate_with / 20, ate_clean / 20);
        if (gated) EXPECT_GE(prec, 0.99);
    }
}

// A second agent linked only by outlier loops: GNC rejects them, the agents fall apart into two components, each is
// solved with its own anchor, nothing throws. Also the plain solve of two unlinked agents.
TEST(Gnc, ComponentsAnchoredWhenRejectionsSplitTheGraph) {
    std::mt19937 g(3);
    Scenario a = Grid(5, 4, 1), b = Grid(5, 4, 1);
    for (auto& T : b.truth) T.t += Vec3(20, 0, 0);
    a.Build(g);
    b.first = 1000;
    b.g = a.g;   // agent B's chain goes into the same graph
    {
        SE3 est = b.truth[0];
        for (size_t i = 0; i < b.truth.size(); ++i) {
            if (i > 0) est = est * (core::SE3Exp(Draw(g, kOdo)) * (b.truth[i - 1].inverse() * b.truth[i]));
            PoseNode nd;
            nd.keyframe = 1000 + i;
            nd.T_W_C = est;
            b.g.AddNode(nd);
            if (i > 0) b.g.AddEdge(OdometryEdge(999 + i, 1000 + i, b.g.nodes().at(999 + i).T_W_C.inverse() * est, kOdo));
        }
    }
    PoseGraph unlinked = b.g;
    PgoSettings lm = PgoV2();
    lm.recompute_covariance = true;
    PgoReport pr;
    EXPECT_NO_THROW(pr = SolvePoseGraph(unlinked, 0, lm));
    EXPECT_EQ(pr.components, 2);
    EXPECT_TRUE(unlinked.nodes().at(1005).covariance.hasNaN());   // outside the anchor's component
    EXPECT_FALSE(unlinked.nodes().at(5).covariance.hasNaN());
    for (int k = 0; k < 3; ++k) {
        const SE3 bad = RandomPose(g, 5.0);
        b.g.AddEdge(OdometryEdge(uint64_t(2 + 5 * k), 1003 + uint64_t(4 * k), bad, kLoop, EdgeType::kInterAgentLoop));
    }
    const SE3 b0 = b.g.nodes().at(1000).T_W_C;
    // Any ONE loop between two otherwise unconnected agents fits exactly (nothing else constrains it), so GNC alone
    // cannot reject it: measured and reported. The pipeline's defence is PCM: three mutually inconsistent loops ->
    // "no_consensus" -> none reach GNC -> two components.
    {
        PoseGraph alone = b.g;
        const GncReport ra = SolveGnc(alone, 0);
        std::printf("GNC alone on 3 random inter-agent loops: %zu accepted (a lone link is unverifiable without "
                    "redundancy; reported)\n", ra.inliers.size());
    }
    const PcmReport pcm = Pcm(b.g);
    std::vector<bool> active(b.g.edges().size(), false);
    for (size_t l : pcm.accepted) active[l] = true;
    GncReport r;
    EXPECT_NO_THROW(r = SolveGnc(b.g, 0, &active));
    std::printf("PCM %s -> GNC: inliers %zu / %zu, components %d\n", pcm.groups.at(0).verdict.c_str(), r.inliers.size(),
                r.loops.size(), r.components);
    EXPECT_EQ(pcm.groups.at(0).verdict, "no_consensus");
    EXPECT_TRUE(r.inliers.empty());
    EXPECT_EQ(r.components, 2);
    EXPECT_LT(core::SE3Log(b.g.nodes().at(1000).T_W_C.inverse() * b0).norm(), 1e-12);   // B held at its own anchor
}

// GNC on its own against 0.5 m / 5 deg near misses (in the PCM -> GNC test PCM removed them all, so GNC's own
// discrimination there was not exercised). Reported.
TEST(Gnc, NearMissesWithoutPcmReported) {
    int tp = 0, nm = 0;
    for (int seed = 0; seed < 20; ++seed) {
        std::mt19937 g(600u + unsigned(seed));
        Scenario s = WithOutliers(Grid(6, 6, 3), g, 60, 0.0);
        std::vector<size_t> near;
        std::normal_distribution<double> n(0.0, 1.0);
        for (int k = 0; k < 10; ++k) {
            const auto [a, b] = s.loop_pairs[size_t(60 + k)];
            Vec6 xi;
            xi << 0.5 * Vec3(n(g), n(g), n(g)).normalized(), 0.087 * Vec3(n(g), n(g), n(g)).normalized();
            const SE3 bad = core::SE3Exp(xi) * (s.truth[size_t(a)].inverse() * s.truth[size_t(b)]);
            near.push_back(s.Loop(g, a, b, &bad));
        }
        const GncReport r = SolveGnc(s.g, 0);
        for (size_t l : r.inliers) {
            tp += std::find(s.true_loops.begin(), s.true_loops.end(), l) != s.true_loops.end();
            nm += std::find(near.begin(), near.end(), l) != near.end();
        }
    }
    std::printf("GNC alone, near misses 0.5 m / 5 deg: %d / 200 accepted, true loops kept %d / 1200 (reported)\n", nm, tp);
}

TEST(Gnc, BitIdenticalTwice) {
    std::mt19937 g1(4), g2(4);
    Scenario s1 = WithOutliers(Grid(5, 5, 2), g1, 40, 0.3), s2 = WithOutliers(Grid(5, 5, 2), g2, 40, 0.3);
    SolveGnc(s1.g, 0);
    SolveGnc(s2.g, 0);
    for (const auto& [id, n] : s1.g.nodes()) {
        const SE3& T = s2.g.nodes().at(id).T_W_C;
        EXPECT_EQ(std::memcmp(n.T_W_C.R.data(), T.R.data(), sizeof(double) * 9), 0);
        EXPECT_EQ(std::memcmp(n.T_W_C.t.data(), T.t.data(), sizeof(double) * 3), 0);
    }
}

// Self-consistent wrong cluster (catalogue §17.2): reported whatever it is; GNC is not expected to beat it.
TEST(Gnc, SelfConsistentClusterReported) {
    std::mt19937 g(5);
    Scenario s = WithOutliers(Grid(6, 6, 3), g, 30, 0.0);
    Vec6 gx;
    gx << 2.0, -1.0, 0.0, 0.0, 0.0, 0.6;
    const SE3 G = core::SE3Exp(gx);
    std::vector<size_t> wrong;
    for (int k = 0; k < 8; ++k) {
        const int a = 5 + 3 * k, b = 80 + 2 * k;
        const SE3 Z = s.truth[size_t(a)].inverse() * G * s.truth[size_t(b)];
        wrong.push_back(s.Loop(g, a, b, &Z));
    }
    const GncReport r = SolveGnc(s.g, 0);
    int wrong_in = 0;
    for (size_t l : r.inliers) wrong_in += std::find(wrong.begin(), wrong.end(), l) != wrong.end();
    std::printf("self-consistent cluster: %d / 8 wrong loops accepted, %zu inliers total (reported)\n", wrong_in,
                r.inliers.size());
}

// Demo scale (design §9 follow-up): the maze-like graph of the timing probe, 10 % random outliers.
TEST(Gnc, TimingAtDemoScale) {
    for (const int N : {1000, 3000}) {
        std::mt19937 g(static_cast<unsigned>(N));
        Scenario s;
        double x = 0, y = 0, yaw = 0;
        std::uniform_int_distribution<int> turn(-1, 1);
        for (int i = 0; i < N; ++i) {
            s.truth.push_back(At(Vec3(x, y, 0), yaw));
            x += 0.1 * std::cos(yaw);
            y += 0.1 * std::sin(yaw);
            if (std::fabs(std::fmod(x + 100, 4.0) - 2.0) < 0.051 && std::fabs(std::fmod(y + 100, 4.0) - 2.0) < 0.051)
                yaw += turn(g) * M_PI / 2;
            if (std::fabs(x) > 6.01 || std::fabs(y) > 6.01) yaw += M_PI, x += 0.2 * std::cos(yaw), y += 0.2 * std::sin(yaw);
        }
        s.Build(g);
        int last = -100;
        for (int i = 0; i < N; ++i) {
            if (i - last < 10) continue;
            for (int j = 0; j + 60 < i; ++j)
                if ((s.truth[size_t(j)].t - s.truth[size_t(i)].t).norm() < 0.5 &&
                    (s.truth[size_t(j)].R.transpose() * s.truth[size_t(i)].R - core::Mat3::Identity()).norm() < 0.3) {
                    s.true_loops.push_back(s.Loop(g, j, i));
                    last = i;
                    break;
                }
        }
        const int n_out = int(s.true_loops.size() / 9);
        std::uniform_int_distribution<int> node(0, N - 1);
        for (int k = 0; k < n_out; ++k) {
            const int a = node(g), b = node(g);
            const SE3 bad = RandomPose(g, 5.0);
            if (a != b) s.outlier_loops.push_back(s.Loop(g, std::min(a, b), std::max(a, b), &bad));
        }
        const auto t0 = std::chrono::steady_clock::now();
        const GncReport r = SolveGnc(s.g, 0);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const Score sc = Classify(s, r);
        int lm_total = 0;
        for (int it : r.lm_iterations) lm_total += it;
        std::printf("GNC timing N=%d: %zu true + %zu outlier loops | %.0f ms, %zu outer its, %d LM its | precision %.3f"
                    " recall %.3f (reported)\n", N, s.true_loops.size(), s.outlier_loops.size(), ms, r.mu.size(), lm_total,
                    sc.precision, sc.recall);
    }
}
