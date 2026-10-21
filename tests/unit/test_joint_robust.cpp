// The robust stage on a joint multi-agent graph (D76; joint_robust.hpp), and the gravity gauge in a joint graph.
//   - two agents (JointId nodes), true inter-agent loops plus one inter-agent ALIASING group (aliasing_generator),
//     agent 1 placed along a false edge (multi_agent's unfiltered spanning tree): PCM -> chordal re-init -> GNC keeps
//     every true inter-agent loop and rejects the group;
//   - gravity: inter-agent edges carrying a 2 deg tilt bias; per-agent gravity variables leave agent 1 tilted, the
//     shared variable (GravityOptions::shared) pulls the tilt out.
#include "slam/pose_graph/multi_agent.hpp"
#include "slam/pose_graph/pgo_levmarq.hpp"
#include "slam/robust/aliasing_generator.hpp"
#include "slam/robust/joint_robust.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <map>
#include <random>
#include <set>

using namespace ariadne;
using namespace ariadne::slam;
using core::SE3;
using core::Vec3;

namespace {

Mat6 Diag(double st, double sr) {
    Vec6 d;
    d << st * st, st * st, st * st, sr * sr, sr * sr, sr * sr;
    return d.asDiagonal();
}

const Mat6 kOdo = Diag(0.02, 0.005);
const Mat6 kLoop = Diag(0.05, 0.01);
const Vec3 kDown(0, 0, -1);

struct Joint {
    std::map<uint64_t, SE3> truth;
    std::vector<std::vector<uint64_t>> chains;   // one per agent
    std::map<uint32_t, std::map<uint64_t, SE3>> local;   // dead reckoning per agent, own frame (first pose = truth)
    PoseGraph g;
};

/// Two agents walking parallel 40-keyframe tracks 2 m apart, heading along x, with small yaw wiggles.
Joint MakeAgents(std::mt19937& gen, int n = 40) {
    Joint j;
    for (uint32_t a = 0; a < 2; ++a) {
        j.chains.emplace_back();
        for (int k = 0; k < n; ++k) {
            SE3 T;
            T.R = Eigen::AngleAxisd(0.2 * std::sin(0.3 * k + a), Vec3::UnitZ()).toRotationMatrix();
            T.t = Vec3(0.5 * k, 2.0 * a, 0.1 * std::sin(0.2 * k));
            const uint64_t id = JointId(a, uint64_t(k));
            j.truth[id] = T;
            j.chains.back().push_back(id);
        }
        SE3 est = j.truth[j.chains.back().front()];
        for (int k = 0; k < n; ++k) {
            const uint64_t id = j.chains.back()[size_t(k)];
            if (k > 0) {
                const uint64_t prev = j.chains.back()[size_t(k - 1)];
                est = est * (core::SE3Exp(DrawLeftNoise(gen, kOdo)) * (j.truth[prev].inverse() * j.truth[id]));
            }
            j.local[a][id] = est;
        }
    }
    return j;
}

/// Nodes and odometry into j.g; agent 1's nodes placed by T_G_W1 (applied to its dead reckoning).
void Assemble(Joint& j, const SE3& T_G_W1) {
    for (uint32_t a = 0; a < 2; ++a)
        for (size_t k = 0; k < j.chains[a].size(); ++k) {
            const uint64_t id = j.chains[a][k];
            PoseNode n;
            n.keyframe = id;
            n.agent = a;
            n.T_W_C = a == 0 ? j.local[0][id] : T_G_W1 * j.local[1][id];
            j.g.AddNode(n);
            if (k > 0) {
                const uint64_t prev = j.chains[a][k - 1];
                j.g.AddEdge(OdometryEdge(prev, id, j.local[a][prev].inverse() * j.local[a][id], kOdo));
            }
        }
}

double Ate(const Joint& j) {
    double s = 0;
    for (const auto& [id, T] : j.truth) s += (j.g.nodes().at(id).T_W_C.t - T.t).squaredNorm();
    return std::sqrt(s / double(j.truth.size()));
}

}  // namespace

TEST(JointRobust, RejectsAnInterAgentAliasingGroupPlacedFirst) {
    for (int seed = 0; seed < 5; ++seed) {
        std::mt19937 gen(760u + unsigned(seed));
        Joint j = MakeAgents(gen);
        // The aliasing group first (its first edge places agent 1), then 10 true inter-agent loops.
        PoseGraph scratch;   // plant into a scratch graph to get the group's measurements before placement
        for (const auto& [id, T] : j.truth) {
            PoseNode n;
            n.keyframe = id;
            scratch.AddNode(n);
        }
        AliasingGroupSpec spec;
        spec.size = 6;
        spec.loop_cov = kLoop;
        spec.type = EdgeType::kInterAgentLoop;
        spec.cross_chain_only = true;
        const auto grp = PlantAliasingGroup(scratch, j.truth, j.chains, spec, gen);
        ASSERT_TRUE(grp);
        const PoseEdge& first = scratch.edges()[grp->edges.front()];
        // multi_agent's placement along that edge: zero residual on it.
        const uint64_t a = first.from >> kAgentIdShift ? first.to : first.from;   // agent-0 end
        const uint64_t b = a == first.from ? first.to : first.from;
        const SE3 Zab = a == first.from ? first.Z : first.Z.inverse();
        const SE3 T_G_W1 = j.local[0][a] * Zab * j.local[1][b].inverse();
        Assemble(j, T_G_W1);
        std::set<size_t> bad, good;
        for (size_t e : grp->edges) {
            j.g.AddEdge(scratch.edges()[e]);
            bad.insert(j.g.edges().size() - 1);
        }
        std::vector<int> ks(40);
        std::iota(ks.begin(), ks.end(), 0);
        std::shuffle(ks.begin(), ks.end(), gen);
        for (int i = 0; i < 10; ++i)
            good.insert(AddTrueLoop(j.g, j.truth, JointId(0, uint64_t(ks[size_t(i)])), JointId(1, uint64_t(ks[size_t(i)])),
                                    kLoop, EdgeType::kInterAgentLoop, gen));
        Joint oracle = j;   // reference: the same graph with the false group's edges removed (true loops only)
        {
            PoseGraph g;
            for (const auto& [id, n] : j.g.nodes()) g.AddNode(n);
            for (size_t e = 0; e < j.g.edges().size(); ++e)
                if (!bad.count(e)) g.AddEdge(j.g.edges()[e]);
            oracle.g = g;
            RobustJointSolve(oracle.g, JointId(0, 0));
        }
        Joint naive = j;
        JointRobustParams off;
        off.reinit_chordal = false;
        const JointRobustReport rn = RobustJointSolve(naive.g, JointId(0, 0), off);
        const JointRobustReport r = RobustJointSolve(j.g, JointId(0, 0));
        int tp = 0, fp = 0;
        for (size_t e : r.gnc.inliers) tp += int(good.count(e)), fp += int(bad.count(e));
        std::printf("seed %d: PCM kept %d / %zu inter-agent, GNC inliers %d true + %d false | ATE %.3f m (true loops "
                    "only: %.3f m; re-init %d nodes) | without re-init: %d inter inliers, ATE %.3f m\n",
                    seed, r.inter_pcm_accepted, r.inter_agent.size(), tp, fp, Ate(j), Ate(oracle), r.reinit_nodes,
                    rn.inter_inliers, Ate(naive));
        EXPECT_EQ(tp, 10) << "seed " << seed;
        EXPECT_EQ(fp, 0) << "seed " << seed;
        EXPECT_TRUE(r.reinitialised);
        EXPECT_NEAR(Ate(j), Ate(oracle), 1e-3) << "seed " << seed << ": the rejected group still moved the map";
    }
}

// Inter-agent edges all carry the same 2 deg tilt bias. Per-agent gravity cannot pull it out (agent 1's priors are
// satisfied by its own variable); one shared variable can, in proportion to the priors' confidence against the
// edges'. Two edge confidences: 0.01 rad (each edge much more certain than a 1 deg prior) and 0.05 rad.
TEST(JointRobust, SharedGravityRemovesAnInterAgentTiltBias) {
    double reduction[2];
    int c = 0;
    for (const double sr : {0.01, 0.05}) {
        std::mt19937 gen(77);
        Joint j = MakeAgents(gen, 60);
        Assemble(j, SE3());
        const Mat6 cov = Diag(0.05, sr);
        const SE3 bias = core::SE3Exp((Vec6() << 0, 0, 0, 2.0 * M_PI / 180.0, 0, 0).finished());
        for (int k = 0; k < 60; k += 3) {
            const uint64_t a = JointId(0, uint64_t(k)), b = JointId(1, uint64_t(k));
            const SE3 Z = bias * core::SE3Exp(DrawLeftNoise(gen, cov)) * (j.truth[a].inverse() * j.truth[b]);
            j.g.AddEdge(OdometryEdge(a, b, Z, cov, EdgeType::kInterAgentLoop));
        }
        const double s = M_PI / 180.0;   // priors: 1 deg, both agents, every 2 keyframes, g_ref in G
        std::normal_distribution<double> nz(0.0, 1.0);
        for (const auto& chain : j.chains)
            for (size_t k = 0; k < chain.size(); k += 2) {
                GravityPrior p;
                p.keyframe = chain[k];
                p.g_C = (core::SO3Exp(s * Vec3(nz(gen), nz(gen), 0)) * (j.truth[chain[k]].R.transpose() * kDown))
                            .normalized();
                p.g_ref_W = kDown;
                p.information = Eigen::Matrix2d::Identity() / (s * s);
                j.g.AddGravityPrior(p);
            }
        auto tilt1 = [&](const PoseGraph& g) {   // RMS tilt error of agent 1's cameras, degrees
            double e = 0;
            for (uint64_t id : j.chains[1]) {
                const Vec3 x = g.nodes().at(id).T_W_C.R.transpose() * kDown, y = j.truth[id].R.transpose() * kDown;
                e += std::pow(std::acos(std::clamp(x.dot(y), -1.0, 1.0)), 2);
            }
            return std::sqrt(e / double(j.chains[1].size())) * 180.0 / M_PI;
        };
        double out[2];
        for (const bool shared : {false, true}) {
            PoseGraph g = j.g;
            PgoSettings st = PgoV2();
            st.recompute_covariance = false;
            st.gravity.enabled = true;
            st.gravity.shared = shared;
            const PgoReport r = SolvePoseGraph(g, JointId(0, 0), st);
            out[shared] = tilt1(g);
            EXPECT_EQ(r.gravity_W.size(), shared ? 1u : 2u);
        }
        reduction[c++] = 1.0 - out[1] / out[0];
        std::printf("inter-agent sigma_r %.2f rad, 2 deg tilt bias: agent-1 tilt RMS per-agent gravity %.2f deg, shared "
                    "%.2f deg (%.0f%% less)\n", sr, out[0], out[1], 100.0 * reduction[c - 1]);
        EXPECT_GT(out[0], 1.5) << "per-agent gravity should leave the inter-agent tilt in";
        EXPECT_LT(out[1], out[0]);
    }
    EXPECT_GT(reduction[1], reduction[0]) << "weaker edges: the shared gravity should correct more";
}
