#pragma once
// ---------------------------------------------------------------------------
// Reprojection least squares by Levenberg-Marquardt with Schur elimination
// (arch §11.3, §11.5, §11.6). ONE optimizer, two uses in Phase 1:
//   * motion-only refinement: one free pose, all landmarks fixed (PnP polish);
//   * the two-frame BA stub:  the older keyframe fixed (the gauge, §11.3),
//                             the newer keyframe and shared landmarks free.
// Phase 2's windowed BA is the same code with more free poses and a sparse
// solver for the reduced camera system.
//
// THE PROBLEM. Minimise sum_k rho(|r_k| / sigma_k) over free poses T_C_W and
// landmarks X_w, where r_k = pi(T_C_W X_w - offset_k) - z_k and rho is Huber
// (quadratic within delta, linear beyond, so one bad track cannot dominate).
// Solved by iteratively reweighted least squares: each iteration fixes the
// Huber weights, builds the normal equations H delta = -g, and takes a damped
// step.
//
// SCHUR COMPLEMENT (§11.5), in one paragraph. The normal equations have block
// structure [H_cc H_cl; H_lc H_ll] over cameras (6 each) and landmarks (3
// each). H_ll is block-diagonal, because a landmark's residuals involve only
// that landmark, so it inverts cheaply block by block. Eliminating landmarks:
//     S = H_cc - H_cl H_ll^-1 H_lc,     S delta_c = -g_c + H_cl H_ll^-1 g_l
// leaves a system in the cameras alone: 6N unknowns instead of 6N + 3M. Then
//     delta_l = H_ll^-1 (-g_l - H_lc delta_c)
// back-substitutes the landmarks. Exact, not an approximation.
//
// LM (§11.6): (H + lambda D) delta = -g with D = I and lambda_0 = 1e-4 max diag(H)
// (the Madsen-Nielsen form: lambda is absolute, so its initial value is scaled
// by the problem's largest curvature). A step is kept only if it lowers the
// robust cost (then lambda /= 3), otherwise it is discarded (lambda *= 2).
// Small lambda -> Gauss-Newton; large lambda -> short gradient-descent steps.
// Termination: relative cost change < 1e-6, a negligible step, or 15 iterations.
//
// COVARIANCE (§11.6): the free poses' covariance is S^-1 at the solution,
// undamped, obtained by solving S X = I, never by forming an explicit inverse
// of the full H. It is in the LEFT-perturbation tangent of T_C_W; use
// core::InvertCovariance for the T_W_C form a Keyframe stores (§4.2).
//
// Dense S: Phase 1 has at most a few free poses. Eigen sparse Cholesky
// replaces the dense LDLT when the window grows (§11.6).
// ---------------------------------------------------------------------------
#include "core/geometry/pinhole.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

using core::Mat3;
using core::Mat6;
using core::PinholeCamera;
using core::SE3;
using core::Vec2;
using core::Vec3;
using core::Vec6;

struct BaObservation {
    int pose = 0;
    int landmark = 0;
    Vec2 uv = Vec2::Zero();
    /// Observing camera relative to the pose's (left) camera, in that frame:
    /// zero for the left image, (baseline, 0, 0) for the right.
    Vec3 offset = Vec3::Zero();
    double sigma_px = 1.0;
};

struct BaProblem {
    PinholeCamera cam;
    std::vector<SE3> poses;          // T_C_W
    std::vector<char> pose_fixed;
    std::vector<Vec3> landmarks;     // X_w
    std::vector<char> landmark_fixed;
    std::vector<BaObservation> obs;
};

struct BaOptions {
    double huber_delta = 2.0;        // §11.3: 1.5-2.5, in units of sigma
    int max_iterations = 15;         // §11.6
    double relative_cost_tolerance = 1e-6;
    // Also stop when the accepted step is negligible: on exactly consistent
    // data the cost goes to ~0 quadratically and its RELATIVE change never
    // becomes small, so a cost-only test would run to max_iterations.
    double step_tolerance = 1e-10;
    double lambda_init_scale = 1e-4;
    double min_depth = 1e-3;         // observations behind/at the camera are skipped
    bool compute_covariance = true;
};

struct BaSummary {
    int iterations = 0;
    int accepted = 0;
    int rejected = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    bool converged = false;
    int observations_used = 0;
    int observations_skipped = 0;    // non-positive depth at the final estimate
    int inliers = 0;                 // |r|/sigma <= huber_delta at the final estimate
    /// Covariance of each free pose (index into problem.poses), left tangent of
    /// T_C_W. Empty when not computed or when the system was singular.
    std::vector<int> cov_pose_index;
    std::vector<Mat6> pose_covariance;
};

namespace detail {

inline double Huber(double s, double delta) {
    return s <= delta ? 0.5 * s * s : delta * (s - 0.5 * delta);
}

/// Robust cost of the current estimate.
inline double Cost(const BaProblem& p, const BaOptions& o) {
    double c = 0.0;
    for (const auto& ob : p.obs) {
        const Vec3 Xc = p.poses[static_cast<size_t>(ob.pose)] * p.landmarks[static_cast<size_t>(ob.landmark)];
        if (Xc.z() - ob.offset.z() <= o.min_depth) continue;
        const double s = (core::Project(p.cam, Xc - ob.offset) - ob.uv).norm() / ob.sigma_px;
        c += Huber(s, o.huber_delta);
    }
    return c;
}

struct Normal {
    Eigen::MatrixXd Hcc;                       // 6N x 6N (block diagonal here)
    Eigen::VectorXd gc;
    std::vector<Mat3> Hll;                     // per landmark
    std::vector<Vec3> gl;
    // H_cl blocks, keyed by (camera slot, landmark), stored per observation.
    struct Cross { int cam; int lm; Eigen::Matrix<double, 6, 3> H; };
    std::vector<Cross> Hcl;
    int used = 0, skipped = 0;
};

}  // namespace detail

/// Solve in place. Throws std::invalid_argument on an ill-posed problem.
inline BaSummary SolveReprojection(BaProblem& p, const BaOptions& o = {}) {
    const int np = static_cast<int>(p.poses.size());
    const int nl = static_cast<int>(p.landmarks.size());
    if (static_cast<int>(p.pose_fixed.size()) != np || static_cast<int>(p.landmark_fixed.size()) != nl) {
        throw std::invalid_argument("SolveReprojection: fixed flags do not match sizes");
    }
    // Slots: free poses get consecutive camera slots, free landmarks theirs.
    std::vector<int> cam_slot(static_cast<size_t>(np), -1), lm_slot(static_cast<size_t>(nl), -1);
    int nc = 0, nfl = 0;
    for (int i = 0; i < np; ++i) if (!p.pose_fixed[static_cast<size_t>(i)]) cam_slot[static_cast<size_t>(i)] = nc++;
    for (int j = 0; j < nl; ++j) if (!p.landmark_fixed[static_cast<size_t>(j)]) lm_slot[static_cast<size_t>(j)] = nfl++;
    // Gauge (§11.3): with free landmarks, a rigid motion of everything leaves
    // the cost unchanged unless something is held fixed.
    if (nfl > 0 && nc == np) {
        throw std::invalid_argument("SolveReprojection: free landmarks need at least one fixed pose (gauge)");
    }

    auto build = [&](detail::Normal& N) {
        N.Hcc = Eigen::MatrixXd::Zero(6 * nc, 6 * nc);
        N.gc = Eigen::VectorXd::Zero(6 * nc);
        N.Hll.assign(static_cast<size_t>(nfl), Mat3::Zero());
        N.gl.assign(static_cast<size_t>(nfl), Vec3::Zero());
        N.Hcl.clear();
        N.used = N.skipped = 0;
        for (const auto& ob : p.obs) {
            const int cs = cam_slot[static_cast<size_t>(ob.pose)];
            const int ls = lm_slot[static_cast<size_t>(ob.landmark)];
            if (cs < 0 && ls < 0) continue;   // nothing free in this residual
            core::Mat26 Jc;
            core::Mat23 Jl;
            const Vec3 Xc = core::ReprojectionJacobians(p.cam, p.poses[static_cast<size_t>(ob.pose)],
                                                        p.landmarks[static_cast<size_t>(ob.landmark)],
                                                        &Jc, &Jl, ob.offset);
            if (Xc.z() - ob.offset.z() <= o.min_depth) {
                ++N.skipped;
                continue;
            }
            ++N.used;
            const Vec2 r = (core::Project(p.cam, Xc - ob.offset) - ob.uv) / ob.sigma_px;
            Jc /= ob.sigma_px;
            Jl /= ob.sigma_px;
            // IRLS Huber weight: 1 inside delta, delta/|r| outside.
            const double s = r.norm();
            const double w = s <= o.huber_delta ? 1.0 : o.huber_delta / s;
            if (cs >= 0) {
                N.Hcc.block<6, 6>(6 * cs, 6 * cs) += w * Jc.transpose() * Jc;
                N.gc.segment<6>(6 * cs) += w * Jc.transpose() * r;
            }
            if (ls >= 0) {
                N.Hll[static_cast<size_t>(ls)] += w * Jl.transpose() * Jl;
                N.gl[static_cast<size_t>(ls)] += w * Jl.transpose() * r;
            }
            if (cs >= 0 && ls >= 0) N.Hcl.push_back({cs, ls, w * Jc.transpose() * Jl});
        }
    };

    // Reduced camera system for damping lambda (0 = undamped, for covariance).
    auto schur = [&](const detail::Normal& N, double lambda, Eigen::MatrixXd& S, Eigen::VectorXd& b,
                     std::vector<Mat3>& Hll_inv) {
        S = N.Hcc;
        b = -N.gc;
        for (int i = 0; i < 6 * nc; ++i) S(i, i) += lambda;
        Hll_inv.resize(static_cast<size_t>(nfl));
        for (int j = 0; j < nfl; ++j) {
            Mat3 H = N.Hll[static_cast<size_t>(j)];
            for (int i = 0; i < 3; ++i) H(i, i) += lambda;
            // A landmark seen from a single ray has rank-2 H_ll; the tiny
            // ridge keeps the inverse finite, and LM damping does the rest.
            H += 1e-12 * Mat3::Identity();
            Hll_inv[static_cast<size_t>(j)] = H.inverse();
        }
        // Group cross blocks by landmark: S -= sum_l H_cl Hll^-1 H_lc.
        std::vector<std::vector<int>> by_lm(static_cast<size_t>(nfl));
        for (int k = 0; k < static_cast<int>(N.Hcl.size()); ++k) by_lm[static_cast<size_t>(N.Hcl[static_cast<size_t>(k)].lm)].push_back(k);
        for (int j = 0; j < nfl; ++j) {
            const Mat3& Hinv = Hll_inv[static_cast<size_t>(j)];
            const auto& ks = by_lm[static_cast<size_t>(j)];
            for (int a : ks) {
                const auto& A = N.Hcl[static_cast<size_t>(a)];
                const Eigen::Matrix<double, 6, 3> AHinv = A.H * Hinv;
                b.segment<6>(6 * A.cam) += AHinv * N.gl[static_cast<size_t>(j)];
                for (int c : ks) {
                    const auto& C = N.Hcl[static_cast<size_t>(c)];
                    S.block<6, 6>(6 * A.cam, 6 * C.cam) -= AHinv * C.H.transpose();
                }
            }
        }
    };

    BaSummary sum;
    sum.initial_cost = detail::Cost(p, o);
    double cost = sum.initial_cost;
    detail::Normal N;
    build(N);
    double max_diag = 0.0;
    for (int i = 0; i < 6 * nc; ++i) max_diag = std::max(max_diag, N.Hcc(i, i));
    for (const auto& H : N.Hll) max_diag = std::max(max_diag, H.diagonal().maxCoeff());
    double lambda = o.lambda_init_scale * std::max(max_diag, 1e-12);

    for (int it = 0; it < o.max_iterations; ++it) {
        ++sum.iterations;
        Eigen::MatrixXd S;
        Eigen::VectorXd b;
        std::vector<Mat3> Hll_inv;
        schur(N, lambda, S, b, Hll_inv);
        Eigen::VectorXd dc = Eigen::VectorXd::Zero(6 * nc);
        if (nc > 0) dc = S.ldlt().solve(b);
        // Back-substitute landmarks.
        std::vector<Vec3> dl(static_cast<size_t>(nfl));
        for (int j = 0; j < nfl; ++j) dl[static_cast<size_t>(j)] = -N.gl[static_cast<size_t>(j)];
        for (const auto& C : N.Hcl) dl[static_cast<size_t>(C.lm)] -= C.H.transpose() * dc.segment<6>(6 * C.cam);
        for (int j = 0; j < nfl; ++j) dl[static_cast<size_t>(j)] = Hll_inv[static_cast<size_t>(j)] * dl[static_cast<size_t>(j)];

        // Candidate: left update T <- exp(delta) T; X <- X + delta.
        const std::vector<SE3> poses0 = p.poses;
        const std::vector<Vec3> lms0 = p.landmarks;
        for (int i = 0; i < np; ++i) {
            const int cs = cam_slot[static_cast<size_t>(i)];
            if (cs >= 0) p.poses[static_cast<size_t>(i)] = core::SE3Exp(Vec6(dc.segment<6>(6 * cs))) * p.poses[static_cast<size_t>(i)];
        }
        for (int j = 0; j < nl; ++j) {
            const int ls = lm_slot[static_cast<size_t>(j)];
            if (ls >= 0) p.landmarks[static_cast<size_t>(j)] += dl[static_cast<size_t>(ls)];
        }
        const double new_cost = detail::Cost(p, o);
        if (std::isfinite(new_cost) && new_cost < cost) {
            ++sum.accepted;
            const double rel = (cost - new_cost) / std::max(cost, 1e-300);
            cost = new_cost;
            lambda /= 3.0;
            build(N);
            double step2 = dc.squaredNorm();
            for (const auto& d : dl) step2 += d.squaredNorm();
            if (rel < o.relative_cost_tolerance || std::sqrt(step2) < o.step_tolerance) {
                sum.converged = true;
                break;
            }
        } else {
            ++sum.rejected;
            p.poses = poses0;
            p.landmarks = lms0;
            lambda *= 2.0;
            if (lambda > 1e12 * std::max(max_diag, 1.0)) break;   // no descent direction left
        }
    }
    sum.final_cost = cost;

    // Final bookkeeping at the solution.
    for (const auto& ob : p.obs) {
        const Vec3 Xc = p.poses[static_cast<size_t>(ob.pose)] * p.landmarks[static_cast<size_t>(ob.landmark)];
        if (Xc.z() - ob.offset.z() <= o.min_depth) { ++sum.observations_skipped; continue; }
        ++sum.observations_used;
        sum.inliers += (core::Project(p.cam, Xc - ob.offset) - ob.uv).norm() / ob.sigma_px <= o.huber_delta;
    }
    if (o.compute_covariance && nc > 0) {
        build(N);
        Eigen::MatrixXd S;
        Eigen::VectorXd b;
        std::vector<Mat3> Hll_inv;
        schur(N, 0.0, S, b, Hll_inv);
        const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
        if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
            for (int i = 0; i < np; ++i) {
                const int cs = cam_slot[static_cast<size_t>(i)];
                if (cs < 0) continue;
                Eigen::MatrixXd E = Eigen::MatrixXd::Zero(6 * nc, 6);
                E.block<6, 6>(6 * cs, 0).setIdentity();
                const Eigen::MatrixXd X = ldlt.solve(E);   // S X = E, not inv(S)
                sum.cov_pose_index.push_back(i);
                sum.pose_covariance.push_back(Mat6(X.block<6, 6>(6 * cs, 0)));
            }
        }
    }
    return sum;
}

}  // namespace ariadne::slam
