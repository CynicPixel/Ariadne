#pragma once
// ---------------------------------------------------------------------------
// GNC-TLS: graduated non-convexity with a truncated-least-squares loss (arch §14.6, D73 part 1; design
// ROBUST_BACKEND_2.5.md §6). Runs AFTER PCM (D72) on the loops PCM accepted.
//
// Idea (Yang, Antonante, Tzoumas, Carlone, RA-L 2020). TLS cost per loop: min(r^2, c^2), with r^2 = r^T Omega r
// (full, no 1/2) and c^2 = chi^2_6(0.99) = 16.81: an outlier costs a constant, so it stops pulling. TLS is
// non-convex, so start from a convex surrogate (mu small), solve, make it less convex (mu *= 1.4), re-solve, ...
// Each step is a weighted least-squares solve; the weights come in closed form (Prop. 4, the argmin over w of the
// Black-Rangarajan dual  w r^2 + mu c^2 (1 - w) / (mu + w)):
//     w = 1                              if r^2 <= mu/(mu+1) c^2
//     w = 0                              if r^2 >= (mu+1)/mu c^2
//     w = (c/r) sqrt(mu (mu + 1)) - mu   otherwise.
// mu_0 = c^2 / (2 r^2_max - c^2) after a first solve with every loop at w = 1 (Remark 5); if 2 r^2_max <= c^2 every
// loop is an inlier and GNC stops at once. GTSAM's GncOptimizer (4.3.0, read at the source) uses the same rule with
// barcSq = c^2 / 2, because its factor error carries the 1/2.
//
// Scope: only loop-type edges get weights (odometry stays at 1, §12.1). Loops rejected upstream (PCM) enter with
// `active[edge] = false`: weight 0 throughout, so edge indices stay the same from the gate through GNC.
// Runs on a working copy; the input graph gets the final poses and weights once (one version bump per node update
// and per weight, never per GNC iteration). Inner solves never recompute marginals.
// If rejections split the graph, SolvePoseGraph solves each component with its own anchor (pgo_levmarq.hpp).
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pgo_levmarq.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <string>
#include <vector>

namespace ariadne::slam {

struct GncParams {
    double cbar2 = 16.812;            // chi^2_6(0.99) on the full r^T Omega r
    double mu_step = 1.4;             // Yang et al. §VI-B
    int max_outer = 100;
    double rel_cost_tol = 1e-5;       // stop when sum w r^2 over loops changes less than this, relatively
    double binary_tol = 1e-4;         // ... or every weight is within this of 0 or 1
    double inlier_weight = 0.5;       // final w > this = inlier
    PgoSettings lm = PgoV2();         // inner solves (recompute_covariance forced off)
    bool marginals_at_end = false;    // one covariance pass on the final weights
};

struct GncReport {
    std::vector<size_t> loops;                   // loop edge indices (all loop-type edges, active or not)
    std::vector<double> r2_initial, r2_final;    // per loop: r^T Omega r after the all-ones solve / at the end
    std::vector<double> w_final;
    std::vector<std::vector<double>> w_history;  // [outer iteration][loop]
    std::vector<double> mu;                      // per outer iteration
    std::vector<int> lm_iterations;              // per solve (the first is the all-ones solve)
    std::vector<size_t> inliers, outliers;       // loop edge indices; inactive loops count as outliers
    int non_binary = 0;                          // final weights not within binary_tol of {0, 1}
    int components = 1;                          // of the final active graph
    std::string stop;                            // "all_inliers", "cost", "weights", "max_outer", "no_loops"
    double final_energy = 0;
};

/// Prop. 4 closed form (exposed for the test against the dual's argmin).
double TlsWeight(double r2, double mu, double cbar2);

/// GNC-TLS on `g` with `anchor` fixed. `active` (optional, indexed by edge) marks the loops allowed in; the rest are
/// held at weight 0. Writes the final poses and loop weights into `g`.
GncReport SolveGnc(PoseGraph& g, uint64_t anchor, const std::vector<bool>* active = nullptr, const GncParams& p = {});

}  // namespace ariadne::slam
