#pragma once
// ---------------------------------------------------------------------------
// Windowed BA on OpenCV 5's LM driver (D51; design LOCAL_SLAM_2.3.md §3): the
// same contract as the pose graph (pgo_levmarq.hpp) -- energy = the robust
// cost at the PROBE state, jtb = +J^T W r, solveDecomposed solves
// (H + D) x = -right with the Schur complement inside, currentOplusX writes
// the probe from the last accepted state, acceptProbe commits it.
// ---------------------------------------------------------------------------
#include "slam/bundle_adjustment/window_ba.hpp"
#include "slam/pose_graph/pgo_levmarq.hpp"   // PgoSettings / ToOpenCv: one settings type for both solvers

#include <opencv2/geometry/detail/optimizer.hpp>

namespace ariadne::slam {

struct WindowBaReport {
    bool found = false;
    int iterations = 0;
    double initial_energy = 0, final_energy = 0;
    int factorizations = 0, failed_factorizations = 0;
    std::map<int, Mat6> covariance;   // free keyframe index -> marginal (left tangent of T_W_C), at the solution
};

/// Optimise `w` in place. `covariance_for`: keyframe indices whose marginals are wanted (free ones only).
WindowBaReport SolveWindowBa(BaWindow& w, const PgoSettings& s, const std::vector<int>& covariance_for = {});

}  // namespace ariadne::slam
