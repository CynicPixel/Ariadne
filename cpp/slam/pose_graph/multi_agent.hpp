#pragma once
// ---------------------------------------------------------------------------
// The broker's multi-agent pose-graph solve (arch §15.4, design COLLAB_3.2 §5).
//
// WHAT IT IS. Each member agent contributes its own subgraph in its odometry
// frame W_a (keyframe nodes + intra-agent odometry edges with information).
// Verified inter-agent edges tie the subgraphs together. The solve puts every
// node into one global frame G = W_anchor and returns, per agent, the rigid
// T_G_W that composes that agent's stored geometry into G (§15.5).
//
// HOW.
//   1. Agents reachable from the gauge agent through inter-agent edges are
//      placed; the rest are not (they stay in their own frame, reported unplaced).
//   2. Start: a spanning-tree alignment along inter-agent edges gives each
//      placed agent an initial T_G_W; its nodes start at T_G_W * T_W_C. Then the
//      chosen initialiser (chordal by default, D70) refines the start.
//   3. One PGO (Schur-free LM on OpenCV's LevMarqBase, D51) with the gauge held
//      fixed: the gauge agent's first node at T_G_W_gauge * T_W_C (§15.4: fix
//      the anchor's first pose; never solve the free-gauge system).
//   4. T_G_W per agent: the gauge agent reports the gauge value (G is W_anchor
//      by definition); every other agent reports the frame fit of its W-frame
//      keyframe poses onto their optimised G poses: R = proj_SO3(sum R_G_C
//      R_W_C^T) (chordal mean), t = mean(t_G_C - R t_W_C). Unlike a positions-
//      only Kabsch fit it stays well-posed on a straight trajectory and with one
//      node. No scale: stereo makes it observable (§1). Per-keyframe drift
//      corrections are in `T_G_C` for consumers that store keyframe-local data.
//
// Node ids in the joint graph: (agent << 40) | keyframe. Part of slam_opencv
// (it calls SolvePoseGraph); the protocol layer reaches it only through the
// injected NodeHooks::solve, so cpp/collaboration/protocol stays OpenCV-free.
// ---------------------------------------------------------------------------
#include "slam/pose_graph/chordal_init.hpp"
#include "slam/pose_graph/pgo_levmarq.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <cstdint>
#include <map>
#include <vector>

namespace ariadne::slam {

inline constexpr int kAgentIdShift = 40;
inline uint64_t JointId(uint32_t agent, uint64_t keyframe) { return (uint64_t(agent) << kAgentIdShift) | keyframe; }

struct AgentSubgraph {
    uint32_t agent = 0;
    std::vector<PoseNode> nodes;   // T_W_C in W_agent; keyframe = the agent's own id
    std::vector<PoseEdge> edges;   // intra-agent (odometry / local loop), keyframe ids local
};

struct InterAgentEdge {
    uint32_t agent_a = 0;
    uint64_t keyframe_a = 0;
    uint32_t agent_b = 0;
    uint64_t keyframe_b = 0;
    SE3 Z;                                  // T_Ca_Cb
    Mat6 information = Mat6::Identity();    // in the tangent of the edge residual (PoseEdge convention)
};

struct MultiAgentInput {
    std::vector<AgentSubgraph> agents;
    std::vector<InterAgentEdge> edges;
    uint32_t gauge_agent = 0;
    SE3 gauge_T_G_W;                        // identity when the gauge agent is the anchor
    PgoInit init = PgoInit::kChordal;
    PgoSettings settings = PgoV2();
};

struct MultiAgentResult {
    std::map<uint32_t, SE3> T_G_W;                          // placed agents only
    std::map<uint32_t, std::map<uint64_t, SE3>> T_G_C;      // optimised keyframe poses in G, placed agents
    std::vector<uint32_t> unplaced;
    PgoReport report;
    double initial_cost = 0.0, final_cost = 0.0;            // PoseGraph::Cost (chi-square, no 1/2)
};

/// Throws std::invalid_argument on an inter-agent edge naming an unknown node, or a gauge agent with no nodes.
MultiAgentResult SolveMultiAgent(const MultiAgentInput& in);

/// The frame T with T * T_W_C[i] ≈ T_G_C[i]: chordal rotation mean, then the mean translation residual.
SE3 FrameFit(const std::vector<SE3>& T_W_C, const std::vector<SE3>& T_G_C);

}  // namespace ariadne::slam
