#pragma once
// ---------------------------------------------------------------------------
// Global initialisation of a pose graph (arch §13.1; design robust-backend.md §3; D70).
//
// WHY. LM only finds the minimum whose basin it starts in. Starting from the
// current node poses (dead reckoning) carries the accumulated odometry drift;
// on hard graphs that start lies in a wrong basin (Carlone, Tron, Daniilidis,
// Dellaert, ICRA 2015, Table I: torus, cube, sphere-a, cubicle, rim).
//
// CHORDAL RELAXATION (Martinec & Pajdla 2007; Carlone et al. 2015 §III-A).
// The nonconvexity of pose-graph optimisation is in the rotations. Edge
// i -> j measures X_i^-1 X_j = Z, i.e. R_j = R_i R_Z. Forget that R must be a
// rotation: the equation is LINEAR in the 9 entries of every R. Row k of it is
//     (R_j row k)^T - R_Z^T (R_i row k)^T = 0,
// so all three rows share one sparse normal matrix: one factorisation, three
// right-hand sides. The anchor's rotation is fixed (it is the gauge). Each
// solved 3x3 block is then projected onto SO(3) with an SVD,
//     R = U diag(1, 1, det(U V^T)) V^T,
// and the determinant term is MANDATORY: without it the nearest orthogonal
// matrix can be a reflection. Rotations fixed, translations are linear too:
// t_j - t_i = R_i t_Z, a weighted graph Laplacian with three right-hand sides.
//
// WEIGHTS. SE-Sync's isotropic weights from each edge's information (Rosen et
// al. IJRR 2019, the g2o convention of its reference code), also used by the
// certificate (D74): tau = 3 / tr(I_t^-1), kappa = 3 / (2 tr(I_R^-1)), from the
// translation and rotation DIAGONAL BLOCKS of the [rho; phi] information.
//
// MST BASELINE (ablation). Compose edge measurements outward from the anchor
// along a spanning tree of maximum kappa (cv::buildMST on weight -kappa). No
// averaging, so every tree edge's noise is kept: the classic "odometry-like"
// initialiser, but with the most certain edges.
//
// cv::SVDecomp does the projection and cv::buildMST the tree (OpenCV 5
// geometry/core); the sparse least squares is Eigen, like the rest of the PGO.
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pose_graph.hpp"

#include <map>

namespace ariadne::slam {

/// How the optimiser's starting poses are produced (configs/slam/pgo_*.yaml `init`).
enum class PgoInit { kCurrent, kMst, kChordal };

/// SE-Sync isotropic weights of one edge (see header). Throws if a diagonal block is not positive definite.
struct IsotropicWeights {
    double kappa = 0.0, tau = 0.0;
};
IsotropicWeights IsotropicFromInformation(const Mat6& information);

/// Nearest rotation in Frobenius norm, with the determinant fix. `det_fixed` reports whether det(U V^T) was -1.
core::Mat3 ProjectToSO3(const core::Mat3& M, bool* det_fixed = nullptr);

struct InitReport {
    PgoInit method = PgoInit::kCurrent;
    int det_fixes = 0;            // chordal: blocks whose projection needed the determinant fix
    double max_projection = 0.0;  // chordal: max ||M - R||_F over blocks (how far the relaxation was from SO(3))
};

/// Starting poses for every node (the anchor keeps its current pose). Throws std::runtime_error if the graph is not
/// connected to the anchor (the linear systems would be singular). kCurrent returns the current node poses.
std::map<uint64_t, SE3> InitialPoses(const PoseGraph& g, uint64_t anchor, PgoInit method,
                                     InitReport* report = nullptr);

/// Writes `poses` into the graph's nodes (covariances set to NaN: they are stale until the solve recomputes them).
void ApplyPoses(PoseGraph& g, const std::map<uint64_t, SE3>& poses);

}  // namespace ariadne::slam
