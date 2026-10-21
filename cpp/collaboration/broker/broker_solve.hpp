#pragma once
// ---------------------------------------------------------------------------
// The production broker solve (design COLLAB_3.2 §5): adapts the protocol's
// SolveInput (wire values, float32, frame-tagged) to the multi-agent pose graph
// (slam/pose_graph/multi_agent.hpp) and back. Plugged into NodeHooks::solve by
// the runtime; links slam_opencv. The protocol tests use an oracle instead.
//
// The robust stage (degeneracy gate -> PCM -> GNC-TLS, 2.5) is not here yet:
// inter-agent edges enter the solve as verified. It slots in between the edge
// list and SolveMultiAgent once 2.5 merges (COLLAB_3.2 §5).
// ---------------------------------------------------------------------------
#include "collaboration/protocol/collab_node.hpp"
#include "slam/pose_graph/multi_agent.hpp"

namespace ariadne::collaboration {

struct BrokerSolveReport {
    slam::MultiAgentResult result;
    size_t skipped_edges = 0;   // edges whose information was not usable (zero / non-PSD)
};

inline slam::MultiAgentInput ToMultiAgentInput(const SolveInput& in, size_t* skipped = nullptr) {
    slam::MultiAgentInput out;
    out.gauge_agent = in.gauge_agent;
    out.gauge_T_G_W = in.gauge_T_G_W;
    auto usable = [](const core::Mat6& I) {
        return I.allFinite() && I.cwiseAbs().maxCoeff() > 0.0 &&
               Eigen::SelfAdjointEigenSolver<core::Mat6>(I, Eigen::EigenvaluesOnly).eigenvalues()(0) >= 0.0;
    };
    size_t skip = 0;
    for (const auto& [agent, g] : in.graphs) {
        slam::AgentSubgraph sub;
        sub.agent = agent;
        for (const auto& n : g.nodes) {
            slam::PoseNode p;
            p.agent = agent;
            p.keyframe = n.keyframe;
            p.t_ns = n.stamp_ns;
            p.T_W_C = FromPoseValue(n.pose);
            sub.nodes.push_back(p);
        }
        for (const auto& e : g.edges) {
            const core::Mat6 I = UnpackInformation(e.information);
            if (!usable(I)) {
                ++skip;
                continue;
            }
            slam::PoseEdge pe;
            pe.type = slam::EdgeType::kOdometry;
            pe.from = e.keyframe_a;
            pe.to = e.keyframe_b;
            pe.Z = FromPoseValue(e.measurement);
            pe.information = I;
            sub.edges.push_back(pe);
        }
        out.agents.push_back(std::move(sub));
    }
    for (const auto& e : in.edges) {
        const core::Mat6 I = UnpackInformation(e.information);
        if (!usable(I)) {
            ++skip;
            continue;
        }
        out.edges.push_back({e.agent_a, e.keyframe_a, e.agent_b, e.keyframe_b, FromPoseValue(e.measurement), I});
    }
    if (skipped) *skipped = skip;
    return out;
}

/// NodeHooks::solve for the runtime: the multi-agent PGO over the members' snapshots.
inline SolveOutput MultiAgentSolve(const SolveInput& in) {
    // The gauge agent's snapshot may not have arrived yet: nothing can be placed.
    const auto g = in.graphs.find(in.gauge_agent);
    if (g == in.graphs.end() || g->second.nodes.empty()) return {};
    const slam::MultiAgentResult r = slam::SolveMultiAgent(ToMultiAgentInput(in));
    return SolveOutput(r.T_G_W.begin(), r.T_G_W.end());
}

}  // namespace ariadne::collaboration
