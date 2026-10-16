#pragma once
// ---------------------------------------------------------------------------
// The pose-graph least-squares problem (arch §12.2, design POSE_GRAPH_2.md),
// OpenCV-free: residuals, exact Jacobians, sparse normal equations, a guarded
// sparse LDLT, and marginal covariances. The LM loop that drives it is OpenCV
// 5's cv::detail::LevMarqBase (D51), through the thin adapter in
// pgo_levmarq.hpp; everything numerical lives here so it builds and tests
// without OpenCV (GitHub CI, cross-architecture).
//
// PROBLEM. Nodes X_i = T_W_C; for each edge e = (i -> j, Z, Omega, w):
//     r_e = Log(Z^-1 X_i^-1 X_j)                  ([rho; phi])
//     E   = 1/2 sum_e w_e r_e^T Omega_e r_e
// A unary edge (from == to: ANCHOR) uses X_i = I: r = Log(Z^-1 X_j).
//
// JACOBIANS (left perturbation X <- exp(d) X, derived in the design §2):
//     dr/dd_j = +J_l^-1(r) Ad(Z^-1 X_i^-1),   dr/dd_i = -J_l^-1(r) Ad(Z^-1 X_i^-1)
// A common perturbation of every node cancels: the gauge null space is "the
// same xi on every node" (asserted in test_pgo.cpp).
//
// GAUGE. The anchor node is not a variable (the oldest pose, §11.3). Every
// free node owns 6 consecutive parameters, in ascending keyframe id: the
// ordering is deterministic and so is the assembled matrix.
//
// THE GUARD. After weighting (GNC can drive weights to 0) the anchored system
// must stay positive definite. SimplicialLDLT reports success on some
// singular systems, so every pivot D_k is checked > 0: a non-positive pivot
// means a disconnected component or collapsed weights, an ERROR, never a
// silently returned step (catalogue §7.8 "weight collapse").
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <map>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

struct EdgeLinearization {
    Vec6 r;
    Mat6 Ji, Jj;   // Ji is zero for a unary edge
};

/// Residual and both Jacobians of one edge at the given poses.
inline EdgeLinearization LinearizeEdge(const PoseEdge& e, const SE3& Xi, const SE3& Xj) {
    const bool unary = e.from == e.to;
    const SE3 A = unary ? e.Z.inverse() : e.Z.inverse() * Xi.inverse();   // Z^-1 X_i^-1
    EdgeLinearization l;
    l.r = core::SE3Log(A * Xj);
    const Mat6 Jj = core::SE3LeftJacobianInverse(l.r) * core::SE3Adjoint(A);
    l.Jj = Jj;
    l.Ji = unary ? Mat6::Zero() : Mat6(-Jj);
    return l;
}

using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

/// The variables of one solve: every node except the anchor, 6 parameters each.
class PgoProblem {
public:
    PgoProblem(const PoseGraph& g, uint64_t anchor) : edges_(g.edges()), anchor_(anchor) {
        if (!g.nodes().count(anchor)) throw std::invalid_argument("PgoProblem: anchor is not a node");
        for (const auto& [id, n] : g.nodes()) {
            poses_[id] = n.T_W_C;
            if (id != anchor) {
                index_[id] = int(order_.size());
                order_.push_back(id);
            }
        }
        if (order_.empty()) throw std::invalid_argument("PgoProblem: no free nodes");
        if (edges_.empty()) throw std::invalid_argument("PgoProblem: no edges");
    }

    int dim() const { return 6 * int(order_.size()); }
    const std::vector<uint64_t>& order() const { return order_; }
    uint64_t anchor() const { return anchor_; }
    const std::map<uint64_t, SE3>& poses() const { return poses_; }
    std::map<uint64_t, SE3>& poses() { return poses_; }
    int IndexOf(uint64_t id) const { return index_.count(id) ? index_.at(id) : -1; }

    /// E = 1/2 sum w r^T Omega r at `poses`.
    double Energy(const std::map<uint64_t, SE3>& poses) const {
        double E = 0.0;
        for (const auto& e : edges_) {
            const Vec6 r = core::SE3Log((e.from == e.to ? e.Z.inverse() : e.Z.inverse() * poses.at(e.from).inverse()) *
                                        poses.at(e.to));
            E += 0.5 * e.gnc_weight * r.dot(e.information * r);
        }
        return E;
    }

    /// Energy, H = sum J^T (w Omega) J over the free variables, and g = +sum J^T (w Omega) r (the LevMarqBase
    /// convention: the driver solves (H + D) x = -g). With `anchored == false` the anchor is a variable too
    /// (only for the null-space test).
    double Linearize(const std::map<uint64_t, SE3>& poses, SparseMat& H, Eigen::VectorXd& g,
                     bool anchored = true) const {
        std::map<uint64_t, int> idx = index_;
        int n = dim();
        if (!anchored) {
            idx.clear();
            int k = 0;
            for (const auto& [id, T] : poses) idx[id] = k++;
            n = 6 * k;
        }
        // Accumulate dense 6x6 blocks in an ordered map (deterministic), then build the sparse matrix once.
        std::map<std::pair<int, int>, Mat6> blocks;
        g = Eigen::VectorXd::Zero(n);
        double E = 0.0;
        for (const auto& e : edges_) {
            const SE3& Xi = poses.at(e.from);
            const SE3& Xj = poses.at(e.to);
            const EdgeLinearization l = LinearizeEdge(e, Xi, Xj);
            const Mat6 W = e.gnc_weight * e.information;
            E += 0.5 * l.r.dot(W * l.r);
            const auto vi = idx.find(e.from), vj = idx.find(e.to);
            const bool fi = e.from != e.to && vi != idx.end(), fj = vj != idx.end();
            if (fi) g.segment<6>(6 * vi->second) += l.Ji.transpose() * W * l.r;
            if (fj) g.segment<6>(6 * vj->second) += l.Jj.transpose() * W * l.r;
            if (fi) Add(blocks, vi->second, vi->second, l.Ji.transpose() * W * l.Ji);
            if (fj) Add(blocks, vj->second, vj->second, l.Jj.transpose() * W * l.Jj);
            if (fi && fj) {
                Add(blocks, vi->second, vj->second, l.Ji.transpose() * W * l.Jj);
                Add(blocks, vj->second, vi->second, l.Jj.transpose() * W * l.Ji);
            }
        }
        // Every free node's diagonal block exists (an isolated node then fails the pivot guard, loudly).
        for (int k = 0; k < n / 6; ++k) Add(blocks, k, k, Mat6::Zero());
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(blocks.size() * 36);
        for (const auto& [rc, B] : blocks)
            for (int a = 0; a < 6; ++a)
                for (int b = 0; b < 6; ++b) trip.emplace_back(6 * rc.first + a, 6 * rc.second + b, B(a, b));
        H.resize(n, n);
        H.setFromTriplets(trip.begin(), trip.end());
        H.makeCompressed();
        return E;
    }

    /// X_i <- exp(x_i) X_i for every free node (the left update).
    std::map<uint64_t, SE3> OplusX(const std::map<uint64_t, SE3>& poses, const Eigen::VectorXd& x) const {
        if (x.size() != dim()) throw std::invalid_argument("PgoProblem::OplusX: wrong step size");
        std::map<uint64_t, SE3> out = poses;
        for (size_t k = 0; k < order_.size(); ++k) {
            SE3& T = out.at(order_[k]);
            T = core::SE3Exp(x.segment<6>(6 * Eigen::Index(k))) * T;
            T.normalize();
        }
        return out;
    }

private:
    static void Add(std::map<std::pair<int, int>, Mat6>& m, int r, int c, const Mat6& B) {
        auto it = m.find({r, c});
        if (it == m.end()) m.emplace(std::make_pair(r, c), B);
        else it->second += B;
    }

    std::vector<PoseEdge> edges_;
    uint64_t anchor_;
    std::map<uint64_t, SE3> poses_;
    std::map<uint64_t, int> index_;
    std::vector<uint64_t> order_;
};

/// Sparse LDLT with the positive-pivot guard. analyzePattern once per sparsity pattern, factorize per LM attempt.
class GuardedLdlt {
public:
    /// Returns false (never throws) when H is not positive definite, so the LM driver treats it as a failed step.
    bool Factorize(const SparseMat& H) {
        if (!analyzed_ || H.nonZeros() != nnz_) {
            ldlt_.analyzePattern(H);
            analyzed_ = true;
            nnz_ = H.nonZeros();
        }
        ldlt_.factorize(H);
        ok_ = ldlt_.info() == Eigen::Success && (ldlt_.vectorD().array() > 0.0).all();
        min_pivot_ = ldlt_.info() == Eigen::Success ? ldlt_.vectorD().minCoeff() : 0.0;
        return ok_;
    }
    Eigen::VectorXd Solve(const Eigen::VectorXd& b) const {
        if (!ok_) throw std::logic_error("GuardedLdlt::Solve without a successful factorization");
        return ldlt_.solve(b);
    }
    bool ok() const { return ok_; }
    double min_pivot() const { return min_pivot_; }

private:
    Eigen::SimplicialLDLT<SparseMat> ldlt_;
    bool analyzed_ = false, ok_ = false;
    Eigen::Index nnz_ = -1;
    double min_pivot_ = 0.0;
};

/// Marginal covariances (left tangent of T_W_C, the PoseNode convention) of the requested free nodes at the
/// optimum: solve H X = E for each node's 6 columns, never an explicit inverse (§11.6). The anchor's is zero.
inline std::map<uint64_t, Mat6> MarginalCovariances(const PgoProblem& p, const std::vector<uint64_t>& nodes) {
    SparseMat H;
    Eigen::VectorXd g;
    p.Linearize(p.poses(), H, g);
    GuardedLdlt ldlt;
    if (!ldlt.Factorize(H)) throw std::runtime_error("MarginalCovariances: the information matrix is not PD");
    std::map<uint64_t, Mat6> out;
    for (uint64_t id : nodes) {
        if (id == p.anchor()) {
            out[id] = Mat6::Zero();
            continue;
        }
        const int k = p.IndexOf(id);
        if (k < 0) throw std::invalid_argument("MarginalCovariances: unknown node");
        Mat6 C;
        for (int c = 0; c < 6; ++c) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(p.dim());
            e(6 * k + c) = 1.0;
            C.col(c) = ldlt.Solve(e).segment<6>(6 * k);
        }
        out[id] = 0.5 * (C + C.transpose());
    }
    return out;
}

}  // namespace ariadne::slam
