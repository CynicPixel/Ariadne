#pragma once
// ---------------------------------------------------------------------------
// Global-optimality certificate for a pose graph (arch §14.8, D74; design robust-backend.md §7). Behind a flag:
// nothing in the runtime calls it unless `loop_closure.certificate` is set; it is a verification artifact, never a
// solver on the hot path.
//
// WHAT IS CERTIFIED. SE-Sync's isotropic chordal cost (Rosen, Carlone, Bandeira, Leonard, IJRR 2019, eq. 12),
//     f(X) = sum_e w_e [ kappa_e ||R_j - R_i R~_e||_F^2 + tau_e ||t_j - t_i - R_i t~_e||^2 ],
// not the runtime anisotropic Log cost. kappa/tau come from each edge's information diagonal blocks
// (IsotropicFromInformation, D70); w_e is the edge's GNC weight (1 for odometry; loops at or under kInactiveGncWeight
// are dropped, as in SolvePoseGraph). So a `certified` result reads "globally optimal for the isotropic chordal
// surrogate of this edge set". It proves optimality of the GIVEN cost, not correctness: an outlier kept in the cost
// can be certified (local-slam.md#estimator-plan §3; reported, never tuned away). The certificate is ONE-DIRECTIONAL: `not certified`
// means unknown, not wrong (§14.8).
//
// THE MATRIX. Stack the unknowns as X = [t_1 .. t_n | R_1 .. R_n] (3 x 4n). Every residual above is linear in X:
// R_j - R_i R~ = X B_e and t_j - t_i - R_i t~ = X b_e with sparse selectors, so f = tr(X M X^T) with the
// translation-explicit data matrix M = sum w (kappa B B^T + tau b b^T) (4n x 4n, PSD; SE-Sync eqs. 13-18, built here
// from the residuals rather than transcribed). Relax R_i^T R_i = I with symmetric multipliers Lambda_i. First-order
// stationarity X (M - Lambda~) = 0, Lambda~ = blkdiag(0 on translations, Lambda_i on rotations), gives
//     Lambda_i = sym( (M X^T)_{R_i} R_i ),    S = M - Lambda~   (SE-Sync's sign, eq. 119 / Thm 7; arch §14.8's
// "Lambda - Q~" is the other sign, corrected per local-slam.md#estimator-plan §4 and fixed by the noise-free test).
// If S >= 0 at a stationary point, X is a global minimiser of the O(3) relaxation, hence of the SE(3) problem.
//
// THE TEST (§49 substitution: arch §14.8 says shifted Lanczos; SE-Sync's own practice is a Cholesky). Ground one
// translation (the anchor's t row/column is deleted: translation shifts are an exact null direction of S) and shift
// the ROTATION diagonal only: S_g + eta [0 0; 0 I]. By the Schur complement on the (positive definite, grounded)
// translation Laplacian, that factorises iff the reduced rotation matrix Q~ - Lambda + eta I > 0, SE-Sync's test.
// Shifting the translation blocks too would be a weaker test (it can over-certify: (A + eta I)^-1 <= A^+).
// Success then BOUNDS the suboptimality: f(X) - f* <= |gap| + 3 n eta, gap = f(X) - sum tr(Lambda_i), which is
// reported next to the verdict. eta is RELATIVE, eta = rel_eta * ||M||_inf, because SE-Sync's absolute tolerance is
// meaningless at parking-garage's information scale; the absolute-eta verdict and the smallest passing decade of
// ||M|| are reported too. On failure, Lanczos on the reduced matrix (full reorthogonalisation) gives the smallest
// Ritz value, an upper bound on lambda_min: `not certified (lambda_min <= ...)`.
//
// THE POINT. Lambda is only right at a stationary point of f, so by default the input poses are refined first: LM
// (OpenCV LevMarqBase, as pgo_levmarq) on the chordal residuals with the anchor fixed, retraction t <- t + dt,
// R <- Exp(theta) R (an internal product-manifold step, never exported), then an exact linear solve of the
// translations at the refined rotations. The distance moved from the input is reported.
//
// Uses only Eigen sparse algebra plus OpenCV's LevMarqBase.
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pgo_levmarq.hpp"
#include "slam/pose_graph/pgo_problem.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace ariadne::slam {

struct CertificateOptions {
    bool enabled = false;       // the flag (loop_closure.certificate); CertifyPoseGraph itself always runs
    bool refine = true;         // move to a stationary point of f first (see header); false certifies the input as is
    double rel_eta = 1e-10;     // eta = rel_eta * ||M||_inf (design §7; value fixed in D74 before the SE-Sync runs)
    double abs_eta = 1e-3;      // SE-Sync's absolute tolerance [UNVERIFIED default], reported alongside
    int lanczos_steps = 300;    // only when not certified
    PgoSettings lm = [] {
        PgoSettings s = PgoV2();
        s.name = "chordal_refine";
        s.rel_energy_delta_tolerance = 1e-14;
        s.step_norm_tolerance = 1e-14;
        s.min_gradient_tolerance = 1e-12;
        s.max_iterations = 200;
        s.recompute_covariance = false;
        return s;
    }();
};

struct CertificateReport {
    bool certified = false;       // S_g + eta I_R factorises at eta = rel_eta ||M||_inf
    bool certified_abs = false;   // ... at eta = abs_eta
    double norm_M = 0, eta = 0;
    double eta_min_passing = std::numeric_limits<double>::quiet_NaN();   // smallest ||M|| 10^-k (k = 2..16) passing
    double objective_input = 0;   // f at the given poses (SE-Sync's units, no 1/2)
    double objective = 0;         // f at the certified point
    double duality_gap = 0;       // f - sum tr(Lambda_i)
    double suboptimality_bound = std::numeric_limits<double>::infinity();   // |gap| + 3 n eta when certified
    double max_rot_gradient = 0;  // max_i ||skew((M X^T)_{R_i} R_i)||_F: zero at a stationary point
    double max_trans_gradient = 0;
    double lambda_min_ritz = std::numeric_limits<double>::quiet_NaN();     // not certified: lambda_min <= this
    double moved_t = 0, moved_r = 0;   // max over nodes of the refinement's change (m, rad)
    int iterations = 0;
    bool refine_found = false;
    int nodes = 0, edges = 0;
    std::string verdict;          // "certified for the isotropic chordal surrogate (...)" | "not certified (...)"
};

/// The chordal surrogate of a graph's active edges. Node order: the anchor first, then ascending keyframe id.
/// Throws std::invalid_argument on a unary edge (the homogeneous form has no constant term), an unknown anchor, or
/// an active graph that is not connected.
class ChordalSurrogate {
public:
    ChordalSurrogate(const PoseGraph& g, uint64_t anchor);

    int n() const { return int(order_.size()); }
    const std::vector<uint64_t>& order() const { return order_; }
    const SparseMat& M() const { return M_; }   // 4n x 4n, layout [t_1 .. t_n | R_1 .. R_n]

    /// X = [t | R] for `poses` (every node of order()).
    Eigen::MatrixXd StackX(const std::map<uint64_t, SE3>& poses) const;
    /// f from the residuals (equals tr(X M X^T); tested).
    double Objective(const std::map<uint64_t, SE3>& poses) const;
    /// LM pieces, internal energy f/2: J^T J and J^T r over every node but the anchor, [dt; theta] per node.
    double Linearize(const std::map<uint64_t, SE3>& poses, SparseMat& H, Eigen::VectorXd& grad) const;
    std::map<uint64_t, SE3> Oplus(const std::map<uint64_t, SE3>& poses, const Eigen::VectorXd& x) const;
    /// The translations minimising f at the given rotations (anchor translation fixed): one Laplacian solve.
    std::map<uint64_t, SE3> SolveTranslations(const std::map<uint64_t, SE3>& poses) const;
    /// S = M - Lambda~ at X (4n x 4n); `Lambda` (optional) receives the 3 x 3 multipliers, unsymmetrised.
    SparseMat CertificateMatrix(const Eigen::MatrixXd& X, std::vector<core::Mat3>* Lambda = nullptr) const;

private:
    struct Edge {
        int i, j;
        core::Mat3 R;
        core::Vec3 t;
        double kappa, tau;
    };
    std::vector<uint64_t> order_;
    std::map<uint64_t, int> index_;
    std::vector<Edge> edges_;
    SparseMat M_;
};

/// Refine (unless o.refine is false), then certify. `point` (optional) receives the certified poses.
CertificateReport CertifyPoseGraph(const PoseGraph& g, uint64_t anchor, const CertificateOptions& o = {},
                                   std::map<uint64_t, SE3>* point = nullptr);

}  // namespace ariadne::slam
