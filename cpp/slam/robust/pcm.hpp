#pragma once
// ---------------------------------------------------------------------------
// Pairwise Consistency Maximization (arch §14.5, D72; design ROBUST_BACKEND_2.5.md §5). Runs AFTER the degeneracy
// gate (D71) and BEFORE GNC (D73); the order is inviolable (arch §14.5).
//
// Idea (Mangelson, Dominic, Eustice, Vasudevan, ICRA 2018). Two loop closures a->b (Z1) and c->d (Z2) close a cycle
// with the odometry: X_a^-1 X_b = (X_a^-1 X_c)(X_c^-1 X_d)(X_d^-1 X_b), so
//     e = Log( Z1^-1 * O_ac * Z2 * O_db )
// must be ~0 if both loops are right. The pair is consistent if e^T Sigma_e^-1 e <= chi^2_6(0.95) = 12.59 (config).
// The accepted set is a maximum clique of the consistency graph: wrong loops rarely agree with the right ones, or
// with each other.
//
// The odometry legs O_ac, O_db and their covariance come from the ODOMETRY-ONLY chain (kOdometry / kStereoOdometry
// edges), composed by adjoints, never from the solved graph, which already contains the loops under test.
//
// COVARIANCE (left perturbations, Ariadne's convention). With every factor perturbed on the left, exp(d) F,
//     e ~ e0 + Jl^-1(e0) [ -Ad(Z1^-1) d_Z1 + Ad(Z1^-1) d_Oac + Ad(Z1^-1 O_ac) d_Z2 + Ad(Z1^-1 O_ac Z2) d_Odb ].
// A leg's perturbation is the sum over its odometry edges; edge k (prefix C_k from the chain root) enters a leg
// starting at position p as  s * Ad(C_p^-1) * Ad(C_k) d_k  (s = -1 for a leg traversed backwards). Prefix sums
// S_p = sum_{k<p} Ad(C_k) Sigma_k Ad(C_k)^T make each leg O(1). When both legs run along ONE chain (intra-agent),
// they can share edges: the chain is split at the four endpoints and each piece gets the summed coefficient
// (M1 + M2), so shared edges are counted once with the right sign (opposite directions cancel).
//
// GROUPS. Odometry components (one chain per agent; D56 bridges a lost stretch with an uninformative edge, so pairs
// across it pass trivially: conservative, it blunts PCM there). Loops are grouped by the unordered pair of
// components of their endpoints; only loops in one group are compared (Mangelson: per robot pair).
// Per group: 1 candidate -> accepted, tagged untested (the gate ran before, GNC runs after; otherwise the first loop
// of a mission could never be accepted). >= 2 candidates whose maximum clique is 1 -> nothing accepted ("no
// consensus").
//
// CLIQUE. Default: the greedy heuristic of Pattabiraman et al. (arXiv 1209.5818, CLIQUEHEU): from each vertex, grow
// a clique through the highest-degree remaining neighbour. Deterministic (ties by edge index). Exact Bron-Kerbosch
// with pivoting behind a flag, for groups of <= exact_max candidates. Loops in a group are ordered by (from, to)
// before the clique, so ties do not depend on edge insertion order.
//
// THRESHOLD. Arch §14.5 fixes chi^2_6(0.95) per pair. Measured (design §5): with a maximum clique that needs every
// pair of the true set to pass, a 5 % per-pair rejection drops true loops (recall 0.60-0.93 on 30-loop groups). The
// default controls the false rejection PER LOOP instead (alpha_loop / (n - 1) per pair); the fixed quantile is the
// ablation baseline (§49 deviation, D72).
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pose_graph.hpp"

#include <Eigen/Core>

#include <string>
#include <vector>

namespace ariadne::slam {

struct PcmParams {
    /// Per-loop false-rejection rate. A true loop survives the maximum clique only if ALL its n-1 pairs pass, so the
    /// pair test runs at alpha_pair = alpha_loop / (n - 1) (Bonferroni), threshold = chi^2_6 quantile(1 - alpha_pair).
    /// <= 0: the fixed quantile below (arch §14.5's chi^2_6(0.95), kept as the ablation baseline).
    /// 0.001 is derived from the recall requirement (design §5): ~n alpha_loop / 2 true loops are lost per group, and
    /// recall >= 0.95 on every seed of 30-loop groups allows one; P(>= 2 lost) ~ 1e-4 per group at 0.001. A dropped
    /// true loop is unrecoverable; an outlier that passes is GNC's job (D73).
    double alpha_loop = 0.001;
    double chi2_threshold = 12.592;   // chi^2_6(0.95), used when alpha_loop <= 0
    bool exact = false;               // exact maximum clique (Bron-Kerbosch) for groups of <= exact_max loops
    int exact_max = 25;
};

struct PcmGroup {
    uint64_t component_from = 0, component_to = 0;   // odometry components (smallest keyframe id of each)
    std::vector<size_t> loops;                       // edge indices into PoseGraph::edges()
    std::vector<size_t> accepted;
    std::string verdict;                             // "consensus", "untested" (1 loop), "no_consensus"
    bool exact = false;                              // the clique was exact
    int pairs = 0, consistent_pairs = 0;
    double threshold = 0;                            // the chi^2 bound the pairs were tested at
};

struct PcmReport {
    std::vector<size_t> accepted, rejected;   // edge indices, ascending
    std::vector<PcmGroup> groups;
    int parallel_odometry_edges = 0;          // duplicates between one node pair: the first in edge order is used
};

/// Cycle statistic of two loops (edge indices into g.edges()); NaN when their endpoints are not connected by
/// odometry. Exposed for tests; Pcm() calls it for every pair in a group.
double PairConsistency(const PoseGraph& g, size_t loop1, size_t loop2);

/// Odometry-only relative pose X_a^-1 X_c and its LEFT covariance, composed along the chain; false if a and c are
/// not in one odometry component. Exposed for the Monte Carlo test.
bool OdometryLeg(const PoseGraph& g, uint64_t a, uint64_t c, SE3& O, Mat6& cov_left);

/// chi^2_6 survival function (closed form for 6 dof) and its inverse by bisection.
double ChiSquare6Survival(double x);
double ChiSquare6Quantile(double p);

/// Maximum (exact) or maximal (greedy) clique of a symmetric adjacency matrix; vertex indices ascending.
std::vector<int> MaxClique(const std::vector<std::vector<bool>>& adj, bool exact);

PcmReport Pcm(const PoseGraph& g, const PcmParams& p = {});

}  // namespace ariadne::slam
