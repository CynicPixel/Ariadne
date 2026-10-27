#pragma once
// ---------------------------------------------------------------------------
// The robust stage on a JOINT multi-agent pose graph (D76; design robust-backend.md §9): the entry point the
// broker's solve (collaboration.md §5 "Robust stage", owned by main) calls instead of a plain SolvePoseGraph.
//
// INPUT. A PoseGraph whose node ids are JointId(agent, keyframe) (multi_agent.hpp), each agent's keyframes and
// odometry plus its own accepted local loops, and the verified inter-agent edges as kInterAgentLoop. Preconditions:
//   - each inter-agent edge already passed the D71 gate where its geometry lives (collaboration.md §11: the verifying
//     agent). A broker that holds the geometry can gate here with GatedInterAgentEdge;
//   - edge information already includes the loop noise model (var = k sigma_PnP^2 + floor^2, design §8). Raw PnP
//     information is ~1000x overconfident on real data, and PCM's Bonferroni thresholds would be off by that much;
//   - the node poses are an initial placement in G (multi_agent's spanning tree is fine; see ORDER).
//
// ORDER. PCM first (it needs no placement: each pair test composes odometry legs, pcm.hpp). Then, by default, a
// chordal re-initialisation over the anchor's component of the graph PCM left (odometry + accepted loops only). That
// removes the placement hazard: multi_agent places agents along UNFILTERED inter-agent edges, so a false edge placed
// first would start GNC at zero residual, the chordal-init-with-outliers trap. Then GNC-TLS on the PCM-accepted
// loops, with per-component anchoring when rejections split the graph (pgo_levmarq.hpp).
//
// SCOPE NOTES.
//   - GNC decides EVERY loop-type edge in the joint graph, the agents' local loops included (Yang et al.: one TLS
//     problem). Agents send only the local loops they accepted.
//   - PCM tests consistency within one pair of odometry components (here: one agent pair). Cycles through three or
//     more agents are left to GNC.
//   - GncState is keyed by edge INDEX (gnc.hpp). Pass it only if the broker's joint graph is append-only across
//     epochs (same edges, same order, new ones at the end); a rebuilt graph must run cold (state = nullptr).
//
// GRAVITY (the yaw-only gauge). With GravityOptions::enabled, PgoProblem gives each AGENT its own 2-DoF gravity
// variable by default. In a joint graph that decouples the agents' tilt: an agent's priors are satisfied by its own
// variable, so gravity cannot correct a tilt error the inter-agent edges carry. One physical gravity means one
// variable in G: set GravityOptions::shared (every prior's g_ref_W must then be expressed in G and agree). A fixed
// 6-DoF anchor plus a free shared 2-DoF gravity is the yaw-only (x, y, z, yaw) gauge with gravity fixed, up to a
// global rotation (a cost-invariant gauge transform): do not also fix gravity, or the anchor's tilt is fixed twice.
// Measured in tests/unit/test_joint_robust.cpp.
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pose_graph.hpp"
#include "slam/robust/degeneracy.hpp"
#include "slam/robust/gnc.hpp"
#include "slam/robust/pcm.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace ariadne::slam {

struct JointRobustParams {
    PcmParams pcm;
    GncParams gnc;                 // gnc.lm.gravity: priors in the joint solve (use .shared, see header)
    bool reinit_chordal = true;    // chordal init over the PCM-accepted graph before GNC (see ORDER)
};

struct JointRobustReport {
    PcmReport pcm;
    GncReport gnc;
    std::vector<size_t> inter_agent;     // kInterAgentLoop edge indices
    int inter_pcm_accepted = 0, inter_inliers = 0;
    bool reinitialised = false;
    int reinit_nodes = 0;                // nodes in the anchor's PCM-accepted component (re-initialised)
};

/// The D71 gate on an inter-agent candidate (for a broker that holds the geometry): the edge from -> to (A pixels,
/// B points; Z = the gate's refined T_A_B, information with the noise model), or nullopt if degenerate.
std::optional<PoseEdge> GatedInterAgentEdge(uint64_t from, uint64_t to, const LoopGeometry& geometry,
                                            const DegeneracyParams& gate, DegeneracyReport* report = nullptr);

/// PCM -> (chordal re-init) -> GNC on `g` with `anchor` fixed; writes poses and loop weights into `g`.
JointRobustReport RobustJointSolve(PoseGraph& g, uint64_t anchor, const JointRobustParams& p = {},
                                   GncState* state = nullptr);

}  // namespace ariadne::slam
