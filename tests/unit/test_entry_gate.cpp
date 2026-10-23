// D178: the known-entry Mahalanobis gate keeps an inter-agent loop consistent with the entry prior within the composed
// odometry covariance, rejects a one-period alias (~5 m / 100 deg), and leaves agents without a prior ungated.
#include "collaboration/broker/broker_solve.hpp"

#include <gtest/gtest.h>

namespace ac = ariadne::collaboration;
namespace core = ariadne::core;

namespace {

core::SE3 Pose(double x, double y, double yaw_deg) {
    core::SE3 T;
    T.R = Eigen::AngleAxisd(yaw_deg * M_PI / 180.0, core::Vec3::UnitZ()).toRotationMatrix();
    T.t = core::Vec3(x, y, 0);
    return T;
}

/// Agent k's snapshot: n keyframes along +x in its own frame, odometry edges with 1 cm / 0.5 deg sigmas.
core::PoseGraphValue Chain(uint32_t agent, int n) {
    core::PoseGraphValue g;
    core::Mat6 I = core::Mat6::Zero();
    I.topLeftCorner<3, 3>().diagonal().setConstant(1.0 / (0.01 * 0.01));
    I.bottomRightCorner<3, 3>().diagonal().setConstant(1.0 / std::pow(0.5 * M_PI / 180.0, 2));
    for (int i = 0; i < n; ++i) {
        core::PoseNodeValue v;
        v.keyframe = uint64_t(i);
        v.pose = ac::ToPoseValue(Pose(0.5 * i, 0, 0));
        g.nodes.push_back(v);
        if (i) {
            core::LoopEdgeValue o;
            o.agent_a = o.agent_b = agent, o.keyframe_a = uint64_t(i - 1), o.keyframe_b = uint64_t(i);
            o.measurement = ac::ToPoseValue(Pose(0.5, 0, 0));
            o.information = ac::PackInformation(I);
            g.edges.push_back(o);
        }
    }
    return g;
}

core::LoopEdgeValue Loop(uint64_t ka, uint64_t kb, const core::SE3& Z) {
    core::LoopEdgeValue e;
    e.agent_a = 0, e.keyframe_a = ka, e.agent_b = 1, e.keyframe_b = kb;
    e.measurement = ac::ToPoseValue(Z);
    core::Mat6 I = core::Mat6::Identity() * 1e4;   // 1 cm / ~0.6 deg
    e.information = ac::PackInformation(I);
    return e;
}

}  // namespace

TEST(EntryGate, KeepsTheTrueLoopRejectsTheAliasAndLeavesUnknownAgentsUngated) {
    // World: agent 0 enters at the origin facing +x; agent 1 enters at (0, 3) facing +x. Both walk 10 m along +x.
    const core::SE3 W0 = Pose(0, 0, 0), W1 = Pose(0, 3, 0);
    ac::SolveInput in;
    in.graphs[0] = Chain(0, 21);
    in.graphs[1] = Chain(1, 21);
    // True loop: agent 0 kf 10 (x=5, y=0) sees agent 1 kf 10 (x=5, y=3): T_Ca_Cb = T_World_Ca^-1 T_World_Cb.
    const core::SE3 Ca = W0 * Pose(5, 0, 0), Cb = W1 * Pose(5, 0, 0);
    in.edges.push_back(Loop(10, 10, Ca.inverse() * Cb));
    // Alias one 6 m lattice period away and rotated 100 deg (the D176 measurement).
    in.edges.push_back(Loop(10, 10, Ca.inverse() * Cb * Pose(5.0, 0, 100.0)));
    ac::EntryGate gate;
    gate.enabled = true;
    gate.T_World_W = std::make_shared<const std::map<ac::AgentId, core::SE3>>(std::map<ac::AgentId, core::SE3>{{0, W0}, {1, W1}});
    ac::EntryGateStats st;
    auto out = ac::GateEdgesByEntry(in, gate, &st);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(st.tested, 2u);
    EXPECT_EQ(st.rejected, 1u);
    EXPECT_LT((ac::FromPoseValue(out[0].measurement).t - (Ca.inverse() * Cb).t).norm(), 1e-6);
    // A 0.3 m / 3 deg disagreement (inside drift + entry sigmas over 5 m) is kept.
    in.edges = {Loop(10, 10, Ca.inverse() * Cb * Pose(0.3, 0.0, 3.0))};
    EXPECT_EQ(ac::GateEdgesByEntry(in, gate).size(), 1u);
    // No prior for agent 1: ungated.
    gate.T_World_W = std::make_shared<const std::map<ac::AgentId, core::SE3>>(std::map<ac::AgentId, core::SE3>{{0, W0}});
    in.edges = {Loop(10, 10, Ca.inverse() * Cb * Pose(5.0, 0, 100.0))};
    out = ac::GateEdgesByEntry(in, gate, &st);
    EXPECT_EQ(out.size(), 1u);
    EXPECT_EQ(st.ungated, 1u);
}

TEST(EntryGate, ChainCovarianceGrowsAlongTheOdometry) {
    ac::SolveInput in;
    in.graphs[0] = Chain(0, 11);
    const auto c = ac::ChainCovariances(in);
    EXPECT_TRUE(c.at({0, 0}).second.isZero(0));
    EXPECT_GT(c.at({0, 10}).second.trace(), c.at({0, 5}).second.trace());
    EXPECT_GT(c.at({0, 5}).second.trace(), 0.0);
}
