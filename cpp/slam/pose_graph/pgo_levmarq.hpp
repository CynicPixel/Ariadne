#pragma once
// ---------------------------------------------------------------------------
// Pose-graph optimization on OpenCV 5's LM driver (D51, design pose-graph.md §3).
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

/// Every LevMarqBase setting, explicitly. The defaults are pgo_v1, which the windowed BA (stereo_vo.cpp) also builds
/// on; the pose graph runs PgoV2() below. No runtime path solves the pose graph yet: its config and manifest fields
/// land with the first caller (D70).
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
    /// GRAVITY_PRIOR edges in the solve (D73 part 2c, pgo_problem.hpp); off: they are ignored, as before.
    GravityOptions gravity;
};

/// pgo_v2 (D70): pgo_v1 with clamp_diagonal off. OpenCV clamps the DAMPING lambda*diag(H) to an absolute [1e-6, 1e32];
/// Ceres, whose min/max_lm_diagonal it mirrors, clamps diag(J^T J) before applying lambda, so its damping still vanishes
/// as lambda -> 0. OpenCV's absolute floor throttles the last Gauss-Newton steps once lambda has shrunk: parking-garage
/// took 95 iterations and stopped 3.7e-5 above the minimum with it, 15 and at the minimum without (design
/// robust-backend.md §2). The floor also regularised a rank-deficient H (a component cut off from the anchor); that
/// case is handled where it arises instead: GNC anchors every connected component of the active graph (design §6).
inline PgoSettings PgoV2() {
    PgoSettings s;
    s.name = "pgo_v2";
    s.clamp_diagonal = false;
    return s;
}

cv::LevMarq::Settings ToOpenCv(const PgoSettings& s);

struct PgoReport {
    bool found = false;          // the driver's verdict: converged by a tolerance, not by iterations / lambda
    int iterations = 0;          // the driver's count of LM attempts
    double initial_energy = 0.0, final_energy = 0.0;
    int linearizations = 0, factorizations = 0, failed_factorizations = 0;
    int components = 1;          // connected components of the active graph, each solved with its own anchor
    std::map<uint32_t, core::Vec3> gravity_W;   // estimated gravity direction per agent (gravity priors on)
};

/// A loop edge whose GNC weight is at or below this is inactive: it neither connects components nor enters a solve.
constexpr double kInactiveGncWeight = 1e-4;

/// Optimise `g` in place (node poses) with `anchor` held fixed; returns the report. The graph's version bumps.
/// COMPONENTS (D73). The active graph (every edge except loops at weight <= kInactiveGncWeight) can fall apart: a
/// second agent before its first inter-agent loop, or GNC rejecting the only loops between two agents. Without the
/// OpenCV damping floor (pgo_v2) that system is singular, so each component is solved on its own: `anchor` holds the
/// component containing it; every other component is held at its lowest keyframe id's current pose. Covariances of
/// nodes outside the anchor's component are NaN (their uncertainty relative to the world frame is not defined).
PgoReport SolvePoseGraph(PoseGraph& g, uint64_t anchor, const PgoSettings& s = PgoV2());

}  // namespace ariadne::slam
