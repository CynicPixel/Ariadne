// COLLAB_3.2 §5: the broker's multi-agent pose-graph solve and its protocol adapter.
#include "collaboration/broker/broker_solve.hpp"
#include "slam/pose_graph/multi_agent.hpp"

#include "../protocol/collab_scenario.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace ac = ariadne::collaboration;
namespace act = ariadne::collaboration::testing;
namespace core = ariadne::core;
namespace slam = ariadne::slam;

namespace {

core::SE3 Exp(const core::Vec3& rho, const core::Vec3& phi) {
    core::SE3 T;
    const double a = phi.norm();
    T.R = a < 1e-12 ? core::Mat3::Identity() : Eigen::AngleAxisd(a, phi / a).toRotationMatrix();
    T.t = rho;
    return T;
}

/// Deterministic small noise.
core::Vec3 Noise(uint64_t k, double s) {
    auto h = [](uint64_t x) {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33;
        return double(x % 20001) / 10000.0 - 1.0;
    };
    return s * core::Vec3(h(3 * k + 1), h(3 * k + 2), h(3 * k + 3));
}

/// Three agents on the shared loop; odometry drifts, inter-agent edges are near-exact.
slam::MultiAgentInput DriftingSwarm(const act::Scenario& sc, double drift_t, double drift_r,
                                    std::map<uint32_t, std::vector<core::SE3>>* gt_W) {
    slam::MultiAgentInput in;
    in.gauge_agent = 0;
    const int K = 60;
    for (uint32_t a = 0; a < 3; ++a) {
        slam::AgentSubgraph sub;
        sub.agent = a;
        core::SE3 est = sc.T_W_C(a, 0);
        for (int k = 0; k < K; ++k) {
            const core::SE3 gt = sc.T_W_C(a, sc.KfTime(k));
            (*gt_W)[a].push_back(gt);
            if (k > 0) {
                const core::SE3 Zgt = sc.T_W_C(a, sc.KfTime(k - 1)).inverse() * gt;
                const core::SE3 Z = Zgt * Exp(Noise(1000 * a + k, drift_t), Noise(5000 * a + k, drift_r));
                est = est * Z;
                sub.edges.push_back(slam::OdometryEdge(uint64_t(k - 1), uint64_t(k), Z,
                                                       act::Scenario::Information().inverse()));
            }
            slam::PoseNode n;
            n.agent = a;
            n.keyframe = uint64_t(k);
            n.T_W_C = est;
            sub.nodes.push_back(n);
        }
        in.agents.push_back(sub);
    }
    // Inter-agent edges wherever two agents' keyframes are within 0.8 m (exact relative pose).
    for (uint32_t a = 0; a < 3; ++a)
        for (uint32_t b = a + 1; b < 3; ++b)
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < K; ++j) {
                    const core::SE3 Pa = sc.T_World_C(a, sc.KfTime(i)), Pb = sc.T_World_C(b, sc.KfTime(j));
                    if ((Pa.t - Pb.t).norm() < sc.overlap_m)
                        in.edges.push_back({a, uint64_t(i), b, uint64_t(j), Pa.inverse() * Pb,
                                            act::Scenario::Information()});
                }
    return in;
}

double RotErr(const core::Mat3& A, const core::Mat3& B) {
    return Eigen::AngleAxisd(A.transpose() * B).angle();
}

}  // namespace

TEST(MultiAgent, FrameFitRecoversAnExactTransform) {
    const core::SE3 T = act::Pose(0.7, 1.0, -2.0, 0.3);
    std::vector<core::SE3> w, g;
    for (int k = 0; k < 5; ++k) {
        w.push_back(act::Pose(0.1 * k, k, 0.5 * k));   // collinear positions: Kabsch alone would be ambiguous
        g.push_back(T * w.back());
    }
    const core::SE3 F = slam::FrameFit(w, g);
    EXPECT_LT((F.t - T.t).norm(), 1e-9);
    EXPECT_LT(RotErr(F.R, T.R), 1e-9);
    EXPECT_LT(RotErr(slam::FrameFit({w[0]}, {g[0]}).R, T.R), 1e-9);   // one node is enough
}

TEST(MultiAgent, LoopClosuresAcrossAgentsCorrectDrift) {
    act::Scenario sc;
    std::map<uint32_t, std::vector<core::SE3>> gt_W;
    slam::MultiAgentInput in = DriftingSwarm(sc, 0.01, 0.004, &gt_W);
    ASSERT_GT(in.edges.size(), 20u);
    const slam::MultiAgentResult r = slam::SolveMultiAgent(in);
    EXPECT_TRUE(r.unplaced.empty());
    EXPECT_LT(r.final_cost, r.initial_cost);

    // ATE in G = W_0 against ground truth, before (dead reckoning aligned at the first edge) and after.
    const core::SE3 T_G_World = sc.T_World_W(0).inverse();
    double before = 0, after = 0;
    int n = 0;
    std::map<uint32_t, core::SE3> T0;
    for (uint32_t a = 0; a < 3; ++a) {
        const core::SE3 gt_T_G_W = T_G_World * sc.T_World_W(a);
        for (size_t k = 0; k < in.agents[a].nodes.size(); ++k) {
            const core::Vec3 gt = (gt_T_G_W * gt_W[a][k]).t;
            before += (gt_T_G_W * in.agents[a].nodes[k].T_W_C).t.operator-(gt).squaredNorm();
            after += (r.T_G_C.at(a).at(k).t - gt).squaredNorm();
            ++n;
        }
        // With drift the rigid T_G_W is a fit, so its reference is the best rigid fit of the agent's drifted
        // W-frame poses onto their ground-truth G poses, not the ground-truth frame itself.
        std::vector<core::SE3> w, g;
        for (size_t k = 0; k < in.agents[a].nodes.size(); ++k) {
            w.push_back(in.agents[a].nodes[k].T_W_C);
            g.push_back(gt_T_G_W * gt_W[a][k]);
        }
        const core::SE3 ideal = a == 0 ? core::SE3{} : slam::FrameFit(w, g);
        EXPECT_LT((r.T_G_W.at(a).t - ideal.t).norm(), 0.05) << "agent " << a;
        EXPECT_LT(RotErr(r.T_G_W.at(a).R, ideal.R), 0.01) << "agent " << a;
    }
    before = std::sqrt(before / n);
    after = std::sqrt(after / n);
    std::cout << "[ MEASURED ] multi-agent PGO node ATE: dead reckoning " << before << " m -> " << after << " m\n";
    EXPECT_LT(after, 0.5 * before);
}

TEST(MultiAgent, GaugeIsHeldAndUnconnectedAgentsAreUnplaced) {
    act::Scenario sc;
    std::map<uint32_t, std::vector<core::SE3>> gt_W;
    slam::MultiAgentInput in = DriftingSwarm(sc, 0.0, 0.0, &gt_W);
    in.gauge_T_G_W = act::Pose(0.4, 3, 4, 1);
    // Agent 2 loses every inter-agent edge.
    std::vector<slam::InterAgentEdge> kept;
    for (const auto& e : in.edges)
        if (e.agent_a != 2 && e.agent_b != 2) kept.push_back(e);
    in.edges = kept;
    const auto r = slam::SolveMultiAgent(in);
    EXPECT_EQ(r.unplaced, (std::vector<uint32_t>{2}));
    EXPECT_FALSE(r.T_G_W.count(2));
    const core::SE3 first = in.gauge_T_G_W * in.agents[0].nodes[0].T_W_C;
    EXPECT_LT((r.T_G_C.at(0).at(0).t - first.t).norm(), 1e-9);
    EXPECT_LT(RotErr(r.T_G_C.at(0).at(0).R, first.R), 1e-9);
    EXPECT_LT(RotErr(r.T_G_W.at(0).R, in.gauge_T_G_W.R), 1e-12);
}

TEST(MultiAgent, ProtocolWithTheProductionSolveEndsConsistent) {
    act::Scenario sc;
    act::Swarm s(sc, {true, 0.02, 0.1, 0.0}, 4);
    // Swap the oracle alignment for the production solve on every node.
    for (size_t a = 0; a < sc.agents; ++a) {
        ac::NodeHooks h = sc.Hooks(a);
        h.solve = &ac::MultiAgentSolve;
        std::vector<ac::AgentId> fleet{0, 1, 2, 3};
        s.nodes[a] = std::make_unique<ac::CollabNode>(ac::AgentId(a), fleet, act::FragCodec(), s.ledger, h);
    }
    s.plan->Partition(act::At(10), act::At(30), {{0, 1}, {2, 3}});
    s.RunUntil(0, 100);
    const auto r = s.Outcome();
    for (const auto& c : r.incomplete) std::cout << "  incomplete: " << c << "\n";
    ASSERT_EQ(r.outcome, ac::Outcome::kConsistent);
    for (size_t a = 0; a < sc.agents; ++a) {
        const auto T = s.nodes[a]->T_G_W();
        ASSERT_TRUE(T.has_value());
        const core::SE3 gt = sc.T_World_W(0).inverse() * sc.T_World_W(a);
        EXPECT_LT((T->t - gt.t).norm(), 5e-3) << "agent " << a;
        EXPECT_LT(RotErr(T->R, gt.R), 5e-3) << "agent " << a;
    }
}
