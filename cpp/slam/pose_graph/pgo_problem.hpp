#pragma once
// ---------------------------------------------------------------------------
// The pose-graph least-squares problem (arch §12.2, design pose-graph.md),
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

/// GRAVITY_PRIOR consumption (D73 part 2c; design robust-backend.md §8, the D59 handoff). Off by default.
/// Each agent with priors gets a 2-DoF gravity-direction variable ĝ_W (initialised at its priors' common g_ref_W,
/// updated on the sphere along GravityTangentBasis(ĝ)), shared by every prior of that agent; each prior costs a
/// Huber-robustified 1/2 rho(s), s = r^T (scale Lambda) r, r = B(g_C)^T R_WC^T ĝ_W. Not in GNC (§12.1: loops only).
struct GravityOptions {
    bool enabled = false;
    double huber_k2 = 9.21;      // chi^2_2(0.99): Mahony's measured tail is p99 ~ 18 (D59)
    double info_scale = 0.6;     // effective-N: Mahony priors' lag-1 error autocorrelation 0.12-0.36 (design §8)
    // D76: ONE gravity variable for every agent (a joint multi-agent graph in G: one physical gravity; every prior's
    // g_ref_W must then be in G and agree). Off: one per agent, which decouples the agents' tilt (joint_robust.hpp).
    bool shared = false;
};

/// The gravity_agents() / PgoReport::gravity_W key of the shared variable (GravityOptions::shared).
constexpr uint32_t kSharedGravityAgent = 0xFFFFFFFFu;

/// The variables of one solve: every node except the anchor, 6 parameters each, then 2 per gravity agent.
class PgoProblem {
public:
    PgoProblem(const PoseGraph& g, uint64_t anchor, const GravityOptions& go = {})
        : edges_(g.edges()), anchor_(anchor), go_(go) {
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
        if (go_.enabled) {
            for (const GravityPrior& pr : g.gravity_priors()) {
                const auto n = g.nodes().find(pr.keyframe);
                if (n == g.nodes().end()) continue;
                const uint32_t agent = go_.shared ? kSharedGravityAgent : n->second.agent;
                auto it = agent_.find(agent);
                if (it == agent_.end()) {
                    it = agent_.emplace(agent, int(gravity_.size())).first;
                    gravity_.push_back(pr.g_ref_W);
                } else if ((gravity_[size_t(it->second)] - pr.g_ref_W).norm() > 1e-9) {
                    throw std::invalid_argument("PgoProblem: one gravity variable's priors disagree on g_ref_W");
                }
                priors_.push_back({pr, it->second});
            }
        }
    }

    int dim() const { return 6 * int(order_.size()) + 2 * int(gravity_.size()); }
    int pose_dim() const { return 6 * int(order_.size()); }
    const std::vector<Vec3>& gravity() const { return gravity_; }   // per gravity agent (agent_ order)
    std::vector<Vec3>& gravity() { return gravity_; }
    const std::map<uint32_t, int>& gravity_agents() const { return agent_; }
    const std::vector<uint64_t>& order() const { return order_; }
    uint64_t anchor() const { return anchor_; }
    const std::map<uint64_t, SE3>& poses() const { return poses_; }
    std::map<uint64_t, SE3>& poses() { return poses_; }
    int IndexOf(uint64_t id) const { return index_.count(id) ? index_.at(id) : -1; }

    /// E = 1/2 sum w r^T Omega r at `poses` (+ the gravity priors at the problem's current gravity()).
    double Energy(const std::map<uint64_t, SE3>& poses) const { return Energy(poses, gravity_); }
    double Energy(const std::map<uint64_t, SE3>& poses, const std::vector<Vec3>& grav) const {
        double E = 0.0;
        for (const auto& [pr, a] : priors_) E += 0.5 * Huber(PriorS(pr, poses.at(pr.keyframe), grav[size_t(a)]));
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
        return Linearize(poses, gravity_, H, g, anchored);
    }
    double Linearize(const std::map<uint64_t, SE3>& poses, const std::vector<Vec3>& grav, SparseMat& H,
                     Eigen::VectorXd& g, bool anchored = true) const {
        std::map<uint64_t, int> idx = index_;
        int n = pose_dim();   // the gravity variables are appended below
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
        // Gravity priors (IRLS weight rho'(s) on the Huber cost; the energy uses rho itself).
        const int ng = anchored ? int(gravity_.size()) : 0;
        if (ng > 0) {
            const int g0 = n;
            n += 2 * ng;
            g.conservativeResize(n);
            g.tail(2 * ng).setZero();
            for (int a = 0; a < 2 * ng; ++a) trip.emplace_back(g0 + a, g0 + a, 0.0);   // the diagonal always exists
            for (const auto& [pr, ai] : priors_) {
                const Vec3& gh = grav[size_t(ai)];
                GravityPrior q = pr;
                q.g_ref_W = gh;
                const GravityPriorLinearization l = LinearizeGravityPrior(q, poses.at(pr.keyframe));
                const Eigen::Matrix2d L = go_.info_scale * pr.information;
                const double s = l.r.dot(L * l.r);
                E += 0.5 * Huber(s);
                const Eigen::Matrix2d W = HuberWeight(s) * L;
                const Eigen::Matrix2d Jg =
                    GravityTangentBasis(pr.g_C).transpose() * poses.at(pr.keyframe).R.transpose() * GravityTangentBasis(gh);
                const int gi = g0 + 2 * ai;
                g.segment<2>(gi) += Jg.transpose() * W * l.r;
                const Eigen::Matrix2d Hgg = Jg.transpose() * W * Jg;
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) trip.emplace_back(gi + a, gi + b, Hgg(a, b));
                const auto vi = idx.find(pr.keyframe);
                if (vi == idx.end()) continue;   // the anchor: only ĝ moves
                const int pi = 6 * vi->second;
                g.segment<6>(pi) += l.J.transpose() * W * l.r;
                const Mat6 Hpp = l.J.transpose() * W * l.J;
                const Eigen::Matrix<double, 6, 2> Hpg = l.J.transpose() * W * Jg;
                for (int a = 0; a < 6; ++a) {
                    for (int b = 0; b < 6; ++b) trip.emplace_back(pi + a, pi + b, Hpp(a, b));
                    for (int b = 0; b < 2; ++b) {
                        trip.emplace_back(pi + a, gi + b, Hpg(a, b));
                        trip.emplace_back(gi + b, pi + a, Hpg(a, b));
                    }
                }
            }
        } else if (!anchored) {
            for (const auto& [pr, ai] : priors_) E += 0.5 * Huber(PriorS(pr, poses.at(pr.keyframe), grav[size_t(ai)]));
        }
        H.resize(n, n);
        H.setFromTriplets(trip.begin(), trip.end());
        H.makeCompressed();
        return E;
    }

    /// ĝ <- the sphere retraction of ĝ along B(ĝ) x_g (the gravity part of a step).
    std::vector<Vec3> OplusGravity(const std::vector<Vec3>& grav, const Eigen::VectorXd& x) const {
        std::vector<Vec3> out = grav;
        for (size_t a = 0; a < out.size(); ++a) {
            const Vec3 v = GravityTangentBasis(grav[a]) * x.segment<2>(pose_dim() + 2 * Eigen::Index(a));
            const double t = v.norm();
            if (t > 0) out[a] = (std::cos(t) * grav[a] + std::sin(t) * v / t).normalized();
        }
        return out;
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
    double Huber(double s) const {
        const double k2 = go_.huber_k2;
        return s <= k2 ? s : 2.0 * std::sqrt(k2 * s) - k2;
    }
    double HuberWeight(double s) const { return s <= go_.huber_k2 ? 1.0 : std::sqrt(go_.huber_k2 / s); }
    double PriorS(const GravityPrior& pr, const SE3& T, const Vec3& gh) const {
        GravityPrior q = pr;
        q.g_ref_W = gh;
        const Eigen::Vector2d r = LinearizeGravityPrior(q, T).r;
        return r.dot(go_.info_scale * pr.information * r);
    }
    struct Prior {
        GravityPrior p;
        int agent;
    };
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
    GravityOptions go_;
    std::map<uint32_t, int> agent_;
    std::vector<Vec3> gravity_;
    std::vector<Prior> priors_;
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
