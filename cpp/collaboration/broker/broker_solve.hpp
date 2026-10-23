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

#include <cmath>
#include <functional>
#include <map>
#include <memory>
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

/// D178 known-entry gate (LAMP / LAMP 2.0: known entry frame, proximity before consistency; Olson RAS 2009: Mahalanobis
/// gate with covariance projected along the odometry path). Agents enter at known placements T_World_Wk. An inter-agent
/// edge (a,ka)->(b,kb) with Z = T_Ca_Cb implies T_Wa_Wb = T_Wa_Ca Z T_Wb_Cb^-1; the prior is P = T_World_Wa^-1
/// T_World_Wb; r = Log(P^-1 implied). Left perturbations of every input map to
///   r ~ Ad(P^-1) [da + Ad(T_Wa_Ca) dz - Ad(implied) db - Ad(T_World_Wa^-1)(eb - ea)],
/// with da, db the odometry chains' drift (each agent's odometry covariances composed from its first snapshot keyframe,
/// Sum_k += Ad(T_W_Cj) C_jk Ad^T, scaled by kappa from the NEES calibration), dz the edge's own covariance and ea, eb the
/// entry placement errors. The edge is kept iff r^T Sigma_r^-1 r <= chi2. Edges of an agent without an entry pose, or
/// whose endpoint has no composed covariance, pass ungated (counted).
struct EntryGate {
    bool enabled = false;
    double chi2 = 22.4577;                   // chi^2_6 at 0.999 (D178, pre-registered)
    double entry_sigma_m = 0.10;             // operator placement, per axis [ASSUMPTION, physical]
    double entry_sigma_rad = 2.0 * M_PI / 180.0;
    double kappa = 1.0;                      // odometry covariance scale from the scene-1 NEES calibration (>= 1)
    std::shared_ptr<const std::map<AgentId, core::SE3>> T_World_W;
};
struct EntryGateStats {
    uint64_t tested = 0, rejected = 0, ungated = 0;
};

/// Each agent's keyframe poses and their composed odometry covariance (left perturbation in W_k).
inline std::map<std::pair<AgentId, uint64_t>, std::pair<core::SE3, core::Mat6>> ChainCovariances(const SolveInput& in) {
    std::map<std::pair<AgentId, uint64_t>, std::pair<core::SE3, core::Mat6>> out;
    for (const auto& [agent, g] : in.graphs) {
        std::map<uint64_t, core::SE3> pose;
        for (const auto& n : g.nodes) pose[n.keyframe] = FromPoseValue(n.pose);
        if (pose.empty()) continue;
        std::map<uint64_t, const LoopEdgeValue*> into;   // the odometry edge ending at each keyframe
        for (const auto& e : g.edges) into[e.keyframe_b] = &e;
        std::map<uint64_t, core::Mat6> cov;
        cov[pose.begin()->first] = core::Mat6::Zero();
        for (const auto& [kf, T] : pose) {
            if (cov.count(kf)) continue;
            const auto e = into.find(kf);
            if (e == into.end() || !cov.count(e->second->keyframe_a) || !pose.count(e->second->keyframe_a)) continue;
            const core::Mat6 I = UnpackInformation(e->second->information);
            Eigen::FullPivLU<core::Mat6> lu(I);
            if (!lu.isInvertible()) continue;
            const core::Mat6 Ad = core::SE3Adjoint(pose.at(e->second->keyframe_a));
            cov[kf] = cov.at(e->second->keyframe_a) + Ad * lu.inverse() * Ad.transpose();
        }
        for (const auto& [kf, C] : cov) out[{agent, kf}] = {pose.at(kf), C};
    }
    return out;
}

inline std::vector<LoopEdgeValue> GateEdgesByEntry(const SolveInput& in, const EntryGate& gate,
                                                   EntryGateStats* stats = nullptr) {
    EntryGateStats st;
    if (!gate.T_World_W) return in.edges;
    const auto chain = ChainCovariances(in);
    core::Mat6 E = core::Mat6::Zero();
    E.topLeftCorner<3, 3>().diagonal().setConstant(gate.entry_sigma_m * gate.entry_sigma_m);
    E.bottomRightCorner<3, 3>().diagonal().setConstant(gate.entry_sigma_rad * gate.entry_sigma_rad);
    std::vector<LoopEdgeValue> out;
    for (const auto& e : in.edges) {
        const auto wa = gate.T_World_W->find(e.agent_a), wb = gate.T_World_W->find(e.agent_b);
        const auto ca = chain.find({e.agent_a, e.keyframe_a}), cb = chain.find({e.agent_b, e.keyframe_b});
        const core::Mat6 Iz = UnpackInformation(e.information);
        Eigen::FullPivLU<core::Mat6> luz(Iz);
        if (wa == gate.T_World_W->end() || wb == gate.T_World_W->end() || ca == chain.end() || cb == chain.end() ||
            !luz.isInvertible()) {
            ++st.ungated;
            out.push_back(e);
            continue;
        }
        ++st.tested;
        const core::SE3& Ta = ca->second.first;
        const core::SE3& Tb = cb->second.first;
        const core::SE3 implied = Ta * FromPoseValue(e.measurement) * Tb.inverse();
        const core::SE3 P = wa->second.inverse() * wb->second;
        const core::Vec6 r = core::SE3Log(P.inverse() * implied);
        const core::Mat6 AdTa = core::SE3Adjoint(Ta), AdI = core::SE3Adjoint(implied),
                         AdWa = core::SE3Adjoint(wa->second.inverse()), A = core::SE3Adjoint(P.inverse());
        const core::Mat6 inner = gate.kappa * ca->second.second + AdTa * luz.inverse() * AdTa.transpose() +
                                 gate.kappa * AdI * cb->second.second * AdI.transpose() +
                                 AdWa * (2.0 * E) * AdWa.transpose();
        const core::Mat6 S = A * inner * A.transpose();
        const double d2 = r.dot(S.ldlt().solve(r));
        if (std::isfinite(d2) && d2 <= gate.chi2) out.push_back(e);
        else ++st.rejected;
    }
    if (stats) *stats = st;
    return out;
}

/// NodeHooks::solve with the robust stage (D171). `last` (optional) receives the robust report of each call;
/// `observe` (optional, evaluation) the solved joint graph with its loop weights (nodes are JointId(agent, keyframe)).
inline std::function<SolveOutput(const SolveInput&)> RobustMultiAgentSolve(
    slam::JointRobustParams params, std::function<void(const slam::JointRobustReport&)> last = {},
    std::function<void(const slam::PoseGraph&)> observe = {}, EntryGate entry = {},
    std::function<void(const EntryGateStats&)> gate_stats = {}) {
    return [params, last, observe, entry, gate_stats](const SolveInput& in0) -> SolveOutput {
        const auto g = in0.graphs.find(in0.gauge_agent);
        if (g == in0.graphs.end() || g->second.nodes.empty()) return {};
        const bool gated = entry.enabled && entry.T_World_W && !entry.T_World_W->empty();
        SolveInput filtered;
        if (gated) {
            filtered = in0;
            EntryGateStats st;
            filtered.edges = GateEdgesByEntry(in0, entry, &st);
            if (gate_stats) gate_stats(st);
        }
        const SolveInput& in = gated ? filtered : in0;
        slam::MultiAgentInput mi = ToMultiAgentInput(in);
        slam::JointRobustParams p = params;
        if (gated && entry.T_World_W->count(in.gauge_agent)) {
            // D178 (Kimera-Multi: GNC needs an outlier-free initial guess): start every agent at its entry prior,
            // G = W_gauge, instead of chordal over PCM's accepted set.
            const core::SE3 T_G_World = in.gauge_T_G_W * entry.T_World_W->at(in.gauge_agent).inverse();
            for (const auto& [k, T] : *entry.T_World_W) mi.prior_T_G_W[k] = T_G_World * T;
            p.reinit_chordal = false;
        }
        mi.solver = [&](slam::PoseGraph& joint, uint64_t anchor) {
            // A cold GncState: the broker rebuilds the joint graph per solve, so edge indices are not stable.
            const slam::JointRobustReport r = slam::RobustJointSolve(joint, anchor, p, nullptr);
            if (last) last(r);
            if (observe) observe(joint);
            return AgentsTiedToGauge(joint, r.gnc.inliers, in.gauge_agent);
        };
        const slam::MultiAgentResult r = slam::SolveMultiAgent(mi);
        return SolveOutput(r.T_G_W.begin(), r.T_G_W.end());
    };
}

}  // namespace ariadne::collaboration
