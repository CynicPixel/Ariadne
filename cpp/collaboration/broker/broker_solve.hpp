#pragma once
// ---------------------------------------------------------------------------
// The production broker solve (design COLLAB_3.2 §5): adapts the protocol's
// SolveInput (wire values, float32, frame-tagged) to the multi-agent pose graph
// (slam/pose_graph/multi_agent.hpp) and back. Plugged into NodeHooks::solve by
// the runtime; links slam_opencv. The protocol tests use an oracle instead.
//
// The robust stage (D171, COLLAB_3.2 §5): RobustMultiAgentSolve runs 2.5's
// joint-graph entry (slam/robust/joint_robust.hpp, D76: PCM -> chordal re-init
// -> GNC-TLS) in place of the plain PGO. The D71 degeneracy gate runs earlier,
// in the verifier that holds the geometry (COLLAB_3.2 §11). An agent whose
// inter-agent edges were all rejected is reported unplaced, never placed on
// them. MultiAgentSolve (no robust stage) stays as the ablation row.
// ---------------------------------------------------------------------------
#include "collaboration/protocol/collab_node.hpp"
#include "slam/pose_graph/multi_agent.hpp"
#include "slam/robust/joint_robust.hpp"

#include <functional>
#include <set>

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

/// The agents tied to `gauge` through inter-agent edges the robust stage kept (GNC inliers among the PCM-accepted).
inline std::set<uint32_t> AgentsTiedToGauge(const slam::PoseGraph& g, const std::vector<size_t>& inliers,
                                            uint32_t gauge) {
    std::map<uint32_t, std::set<uint32_t>> adj;
    for (size_t k : inliers)
        if (const auto& e = g.edges()[k]; e.type == slam::EdgeType::kInterAgentLoop) {
            const uint32_t a = uint32_t(e.from >> slam::kAgentIdShift), b = uint32_t(e.to >> slam::kAgentIdShift);
            adj[a].insert(b);
            adj[b].insert(a);
        }
    std::set<uint32_t> seen{gauge};
    std::vector<uint32_t> st{gauge};
    while (!st.empty()) {
        const uint32_t a = st.back();
        st.pop_back();
        for (uint32_t b : adj[a])
            if (seen.insert(b).second) st.push_back(b);
    }
    return seen;
}

/// NodeHooks::solve with the robust stage (D171). `last` (optional) receives the robust report of each call.
inline std::function<SolveOutput(const SolveInput&)> RobustMultiAgentSolve(
    slam::JointRobustParams params, std::function<void(const slam::JointRobustReport&)> last = {}) {
    return [params, last](const SolveInput& in) -> SolveOutput {
        const auto g = in.graphs.find(in.gauge_agent);
        if (g == in.graphs.end() || g->second.nodes.empty()) return {};
        slam::MultiAgentInput mi = ToMultiAgentInput(in);
        mi.solver = [&](slam::PoseGraph& joint, uint64_t anchor) {
            // A cold GncState: the broker rebuilds the joint graph per solve, so edge indices are not stable.
            const slam::JointRobustReport r = slam::RobustJointSolve(joint, anchor, params, nullptr);
            if (last) last(r);
            return AgentsTiedToGauge(joint, r.gnc.inliers, in.gauge_agent);
        };
        const slam::MultiAgentResult r = slam::SolveMultiAgent(mi);
        return SolveOutput(r.T_G_W.begin(), r.T_G_W.end());
    };
}

}  // namespace ariadne::collaboration
