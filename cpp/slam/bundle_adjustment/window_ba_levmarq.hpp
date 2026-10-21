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

/// Marginals (left tangent of T_W_C) of the given FREE keyframes from the undamped information at `w`. Throws, with
/// the singular block named, when that information is not PD -- the caller's observability guards (window_ba.hpp)
/// are meant to make that impossible. `solve`, when given, adds the LM run to the diagnostic.
std::map<int, Mat6> WindowBaCovariances(const BaWindow& w, const std::vector<int>& keyframes,
                                        const WindowBaReport* solve = nullptr);

/// The same, without throwing: false (and `why`, the singular block named) when the information is not PD. For a
/// caller that can say "no information" honestly -- a JOINT degeneracy the per-landmark and covisibility guards cannot
/// see (EuRoC V1_03: the null vector is a common scale of two keyframes' translations, with far / right-less landmarks).
/// `bias_cov`, inertial windows only: also the marginals of the biases [bg; ba] of `bias_keyframes` (fixed poses
/// allowed: biases are always variables) from the same factorization (window[1]'s becomes the next solve's window[0]
/// bias prior, INERTIAL_2.4 §4.2). `gravity_cov`: the tilt marginal (estimate_gravity windows; GravityTangentBasis of
/// the window's g_W).
bool TryWindowBaCovariances(const BaWindow& w, const std::vector<int>& keyframes, std::map<int, Mat6>& cov,
                            std::string* why = nullptr, const WindowBaReport* solve = nullptr,
                            std::map<int, Mat6>* bias_cov = nullptr, const std::vector<int>& bias_keyframes = {},
                            Eigen::Matrix2d* gravity_cov = nullptr);

}  // namespace ariadne::slam
