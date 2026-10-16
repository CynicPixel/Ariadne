#pragma once
// ---------------------------------------------------------------------------
// Pose-graph optimization on OpenCV 5's LM driver (D51, design POSE_GRAPH_2.md §3).
//
// cv::detail::LevMarqBase runs the Levenberg-Marquardt loop: Marquardt damping
// (J^T J + lambda diag(J^T J), clamped), step-quality-adaptive lambda, optional
// Jacobi column scaling, four termination tests. It knows nothing about poses:
// it calls a LevMarqBackend, which is this file's PgoBackend over the
// OpenCV-free PgoProblem (residuals, exact Jacobians, sparse LDLT).
//
// THE CONTRACT (read from OpenCV's levmarq.cpp and its own pose-graph backend):
//   calcFunc        energy 1/2 sum w r^T Omega r at the PROBE poses; with the
//                   Jacobian, H = J^T W J and jtb = +J^T W r
//   solveDecomposed solves (H + D) x = -right, so x is the descent step
//   currentOplusX   probe <- exp(x) * current (current = last accepted)
//   acceptProbe     current <- probe
//   getDiag/setDiag the driver rewrites H's diagonal on every damping attempt
// Geodesic acceleration (Transtrum & Sethna) is NOT implemented: enableGeo()
// returns false, so asking for it makes the driver refuse, loudly.
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pgo_problem.hpp"

#include <opencv2/geometry.hpp>
#include <opencv2/geometry/detail/optimizer.hpp>

#include <string>

namespace ariadne::slam {

/// Every LevMarqBase setting, explicitly (configs/slam/pgo_v1.yaml; recorded in the manifest).
struct PgoSettings {
    std::string name = "pgo_v1";
    bool jacobi_scaling = false;
    bool clamp_diagonal = true;
    bool use_step_quality = true;
    bool up_double = true;
    bool step_norm_inf = false;
    bool check_rel_energy_change = true;
    bool check_min_gradient = true;
    bool check_step_norm = true;
    double step_norm_tolerance = 1e-6;
    double rel_energy_delta_tolerance = 1e-6;
    double min_gradient_tolerance = 1e-6;
    double small_energy_tolerance = 0.0;
    int max_iterations = 500;
    double initial_lambda = 1e-4;
    double initial_lm_up_factor = 2.0;
    double initial_lm_down_factor = 3.0;
    /// After the solve, every node's covariance is the marginal at the optimum (H X = E, the anchor's is zero).
    /// false: covariances are set to NaN (never left stale: a consumer reading them must fail visibly).
    bool recompute_covariance = true;
};

cv::LevMarq::Settings ToOpenCv(const PgoSettings& s);

struct PgoReport {
    bool found = false;          // the driver's verdict: converged by a tolerance, not by iterations / lambda
    int iterations = 0;          // the driver's count of LM attempts
    double initial_energy = 0.0, final_energy = 0.0;
    int linearizations = 0, factorizations = 0, failed_factorizations = 0;
};

/// Optimise `g` in place (node poses) with `anchor` held fixed; returns the report. The graph's version bumps.
PgoReport SolvePoseGraph(PoseGraph& g, uint64_t anchor, const PgoSettings& s = {});

}  // namespace ariadne::slam
