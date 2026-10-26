#pragma once
// ---------------------------------------------------------------------------
// The local pose graph (arch §12): keyframe poses as nodes, typed relative-pose
// measurements as edges. Phase 1 builds and versions it; the optimizer
// (chordal init -> SE(3) refinement -> GNC, §13-§14) arrives in Phase 2/3.
//
// NODES store T_W_C (camera -> world, the §4.2 convention) with a covariance
// in the LEFT tangent of T_W_C (T = exp(xi) That), Ariadne's one convention.
//
// EDGES follow §12.2 verbatim: for i -> j with measurement Z_ij (the pose of j
// seen from i, Z = X_i^-1 X_j),
//     r_ij = Log( Z_ij^-1 * X_i^-1 * X_j )
// Note r is a RIGHT-side error on Z: X_i^-1 X_j = Z exp(r). A left-tangent
// covariance Sigma_z of Z (exp(xi) Z) maps to it by exp(xi) Z = Z exp(r) =>
// r = Adj(Z^-1) xi, so the edge stores its information matrix in r's tangent:
//     Info_r = ( Adj(Z^-1) Sigma_z Adj(Z^-1)^T )^-1
// OdometryEdge() does that conversion; a Monte Carlo test checks it.
//
// VERSION. Every mutation bumps version(). The reconstruction service stamps
// exported poses with it (KeyframeBundle.frame_id_of_pose, §18.2) and re-anchors
// on change instead of retraining.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "core/uncertainty/covariance.hpp"

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "slam/pose_graph/gravity_prior.hpp"

namespace ariadne::slam {

using core::Mat6;
using core::SE3;
using core::Vec6;

/// §12.1. Only loop-type edges are robustified; ODOMETRY always has GNC weight 1.
enum class EdgeType { kOdometry, kStereoOdometry, kLocalLoop, kInterAgentLoop, kAnchor, kGravityPrior };

inline bool IsLoopType(EdgeType t) { return t == EdgeType::kLocalLoop || t == EdgeType::kInterAgentLoop; }

struct PoseNode {
    uint64_t keyframe = 0;
    uint32_t agent = 0;
    int64_t t_ns = 0;
    SE3 T_W_C;
    /// ABSOLUTE uncertainty of T_W_C (left tangent) relative to the graph's
    /// gauge (the first keyframe). Producers must not store a conditional
    /// (neighbour-fixed) covariance here: that belongs on the edge.
    Mat6 covariance = Mat6::Zero();
};

struct PoseEdge {
    EdgeType type = EdgeType::kOdometry;
    uint64_t from = 0;                // keyframe ids
    uint64_t to = 0;
    SE3 Z;                            // measured X_from^-1 X_to
    Mat6 information = Mat6::Identity();   // in the tangent of r_ij (see header)
    double gnc_weight = 1.0;
};

/// An odometry edge i -> j from the relative-pose measurement Z and its LEFT
/// covariance.
inline PoseEdge OdometryEdge(uint64_t from, uint64_t to, const SE3& Z, const Mat6& cov_z_left,
                             EdgeType type = EdgeType::kOdometry) {
    PoseEdge e;
    e.type = type;
    e.from = from;
    e.to = to;
    e.Z = Z;
    const Mat6 cov_r = core::ChangeFrame(Z.inverse(), cov_z_left);
    e.information = cov_r.ldlt().solve(Mat6::Identity());
    e.information = 0.5 * (e.information + e.information.transpose());
    return e;
}

class PoseGraph {
public:
    void AddNode(const PoseNode& n) {
        if (nodes_.count(n.keyframe)) throw std::invalid_argument("PoseGraph: duplicate keyframe node");
        nodes_[n.keyframe] = n;
        ++version_;
    }

    void UpdateNode(uint64_t keyframe, const SE3& T_W_C, const Mat6& covariance) {
        auto& n = nodes_.at(keyframe);
        n.T_W_C = T_W_C;
        n.covariance = covariance;
        ++version_;
    }

    void AddEdge(const PoseEdge& e) {
        if (!nodes_.count(e.from) || !nodes_.count(e.to)) {
            throw std::invalid_argument("PoseGraph: edge references an unknown keyframe");
        }
        if (e.type == EdgeType::kOdometry && e.gnc_weight != 1.0) {
            throw std::invalid_argument("PoseGraph: ODOMETRY edges always carry GNC weight 1 (§12.1)");
        }
        // The cost is 1/2 r^T I r: an information that is not symmetric PSD makes it unbounded below and the normal
        // equations indefinite (the SE-Sync cubicle / rim files carry such matrices, pose-graph.md §5). Refuse it
        // here, where it enters, rather than as a failed factorisation far downstream.
        const double scale = e.information.cwiseAbs().maxCoeff();
        if (!e.information.allFinite() || !(scale > 0.0) ||
            (e.information - e.information.transpose()).cwiseAbs().maxCoeff() > 1e-9 * scale) {
            throw std::invalid_argument("PoseGraph: edge information must be finite, non-zero and symmetric");
        }
        const double lmin = Eigen::SelfAdjointEigenSolver<Mat6>(e.information, Eigen::EigenvaluesOnly).eigenvalues()(0);
        if (lmin < -1e-12 * scale) {
            throw std::invalid_argument("PoseGraph: edge information is not positive semidefinite (min eigenvalue " +
                                        std::to_string(lmin) + ")");
        }
        edges_.push_back(e);
        ++version_;
    }

    /// GNC weight of a LOOP-type edge (§12.1: only loops are robustified), in [0, 1]. GNC (D73) sets weights on a
    /// working copy and writes the final ones back once, so a loop event bumps the version once, not per iteration.
    void SetGncWeight(size_t edge, double w) {
        if (edge >= edges_.size()) throw std::out_of_range("PoseGraph::SetGncWeight: no such edge");
        if (!IsLoopType(edges_[edge].type)) {
            throw std::invalid_argument("PoseGraph::SetGncWeight: only loop-type edges carry a GNC weight (§12.1)");
        }
        if (!(w >= 0.0 && w <= 1.0)) throw std::invalid_argument("PoseGraph::SetGncWeight: weight must be in [0, 1]");
        edges_[edge].gnc_weight = w;
        ++version_;
    }

    /// §12.2 residual of an edge at the current node estimates.
    Vec6 Residual(const PoseEdge& e) const {
        const SE3& Xi = nodes_.at(e.from).T_W_C;
        const SE3& Xj = nodes_.at(e.to).T_W_C;
        return core::SE3Log(e.Z.inverse() * Xi.inverse() * Xj);
    }

    /// sum_e w_e r_e^T Info_e r_e: what the Phase 2 optimizer minimises.
    double Cost() const {
        double c = 0.0;
        for (const auto& e : edges_) {
            const Vec6 r = Residual(e);
            c += e.gnc_weight * r.dot(e.information * r);
        }
        return c;
    }

    /// GRAVITY_PRIOR edges (§12.1, unary; gravity_prior.hpp). Kept apart from edges(): Cost() and the solver are the
    /// pose-graph owner's (2.5), who decides how to consume them.
    void AddGravityPrior(const GravityPrior& g) {
        if (!nodes_.count(g.keyframe)) throw std::invalid_argument("PoseGraph: gravity prior on an unknown keyframe");
        if (std::abs(g.g_C.norm() - 1.0) > 1e-9 || std::abs(g.g_ref_W.norm() - 1.0) > 1e-9)
            throw std::invalid_argument("PoseGraph: gravity prior directions must be unit vectors");
        const double scale = g.information.cwiseAbs().maxCoeff();
        if (!g.information.allFinite() || !(scale > 0.0) ||
            std::abs(g.information(0, 1) - g.information(1, 0)) > 1e-9 * scale ||
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(g.information, Eigen::EigenvaluesOnly).eigenvalues()(0) <
                -1e-12 * scale)
            throw std::invalid_argument("PoseGraph: gravity prior information must be finite, non-zero, symmetric PSD");
        gravity_priors_.push_back(g);
        ++version_;
    }

    /// sum over gravity priors of rᵀ Λ r at the current node estimates (not part of Cost()).
    double GravityCost() const {
        double c = 0.0;
        for (const auto& g : gravity_priors_) c += GravityPriorChi2(g, nodes_.at(g.keyframe).T_W_C);
        return c;
    }

    const std::map<uint64_t, PoseNode>& nodes() const { return nodes_; }
    const std::vector<PoseEdge>& edges() const { return edges_; }
    const std::vector<GravityPrior>& gravity_priors() const { return gravity_priors_; }
    uint64_t version() const { return version_; }

private:
    std::map<uint64_t, PoseNode> nodes_;   // ordered by keyframe id: stable iteration
    std::vector<PoseEdge> edges_;
    std::vector<GravityPrior> gravity_priors_;
    uint64_t version_ = 0;
};

}  // namespace ariadne::slam
