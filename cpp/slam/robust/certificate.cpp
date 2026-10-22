// The D74 certificate: see certificate.hpp for what is certified, the matrix and the test.
#include "slam/robust/certificate.hpp"

#include "slam/pose_graph/chordal_init.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ariadne::slam {

using core::Mat3;
using core::Vec3;

namespace {

using Triplets = std::vector<Eigen::Triplet<double>>;

bool Active(const PoseEdge& e) { return !IsLoopType(e.type) || e.gnc_weight > kInactiveGncWeight; }
double Weight(const PoseEdge& e) { return IsLoopType(e.type) ? e.gnc_weight : 1.0; }

cv::Mat_<double> ToMat(const Eigen::VectorXd& v) {
    cv::Mat_<double> m(int(v.size()), 1);
    for (Eigen::Index i = 0; i < v.size(); ++i) m(int(i)) = v(i);
    return m;
}

Eigen::VectorXd ToVec(const cv::Mat_<double>& m) {
    Eigen::VectorXd v(m.rows);
    for (int i = 0; i < m.rows; ++i) v(i) = m(i);
    return v;
}

/// LevMarqBase over the chordal residuals: pgo_levmarq's PgoBackend with the surrogate's energy, no gravity.
class ChordalBackend : public cv::detail::LevMarqBackend {
public:
    ChordalBackend(const ChordalSurrogate& s, std::map<uint64_t, SE3> start)
        : s_(s), current_(std::move(start)), probe_(current_) {}

    bool enableGeo() override { return false; }
    bool calcFunc(double& energy, bool calcEnergy, bool calcJacobian) override {
        if (calcJacobian) energy = s_.Linearize(probe_, H_, g_);
        else if (calcEnergy) energy = 0.5 * s_.Objective(probe_);
        return std::isfinite(energy);
    }
    void currentOplusX(const cv::Mat_<double>& x, bool geo) override {
        if (geo) CV_Error(cv::Error::StsBadArg, "ChordalBackend: geodesic acceleration is not implemented");
        probe_ = s_.Oplus(current_, ToVec(x));
    }
    void prepareVars() override { probe_ = current_; }
    const cv::Mat_<double> getJtb() override { return ToMat(g_); }
    const cv::Mat_<double> getDiag() override { return ToMat(H_.diagonal()); }
    void setDiag(const cv::Mat_<double>& d) override {
        for (int i = 0; i < d.rows; ++i) H_.coeffRef(i, i) = d(i);   // diagonal entries always exist (Linearize)
    }
    void doJacobiScaling(const cv::Mat_<double>& di) override {
        const Eigen::VectorXd d = ToVec(di);
        for (int k = 0; k < H_.outerSize(); ++k)
            for (SparseMat::InnerIterator it(H_, k); it; ++it) it.valueRef() *= d(it.row()) * d(it.col());
        g_ = g_.cwiseProduct(d);
    }
    bool decompose() override { return ldlt_.Factorize(H_); }
    bool solveDecomposed(const cv::Mat_<double>& right, cv::Mat_<double>& x) override {
        if (!ldlt_.ok()) return false;
        const Eigen::VectorXd sol = ldlt_.Solve(-ToVec(right));
        if (!sol.allFinite()) return false;
        x = ToMat(sol);
        return true;
    }
    bool calcJtbv(cv::Mat_<double>&) override { return false; }
    void acceptProbe() override { current_ = probe_; }

    const std::map<uint64_t, SE3>& current() const { return current_; }

private:
    const ChordalSurrogate& s_;
    std::map<uint64_t, SE3> current_, probe_;
    SparseMat H_;
    Eigen::VectorXd g_;
    GuardedLdlt ldlt_;
};

/// Smallest Ritz value of a symmetric operator: Lanczos with full reorthogonalisation from a seeded start.
template <class Op>
double SmallestRitz(const Op& op, int dim, int steps) {
    const int m = std::min(steps, dim);
    Eigen::MatrixXd V(dim, m + 1);
    std::mt19937 gen(1);
    std::normal_distribution<double> nz(0.0, 1.0);
    for (int i = 0; i < dim; ++i) V(i, 0) = nz(gen);
    V.col(0).normalize();
    std::vector<double> alpha, beta;
    for (int k = 0; k < m; ++k) {
        Eigen::VectorXd w = op(V.col(k));
        alpha.push_back(w.dot(V.col(k)));
        for (int pass = 0; pass < 2; ++pass) w -= V.leftCols(k + 1) * (V.leftCols(k + 1).transpose() * w);
        const double b = w.norm();
        if (k + 1 == m || b < 1e-12 * std::max(1.0, std::fabs(alpha.back()))) break;
        beta.push_back(b);
        V.col(k + 1) = w / b;
    }
    const int k = int(alpha.size());
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
        T(i, i) = alpha[size_t(i)];
        if (i + 1 < k) T(i, i + 1) = T(i + 1, i) = beta[size_t(i)];
    }
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(T, Eigen::EigenvaluesOnly).eigenvalues()(0);
}

}  // namespace

ChordalSurrogate::ChordalSurrogate(const PoseGraph& g, uint64_t anchor) {
    if (!g.nodes().count(anchor)) throw std::invalid_argument("ChordalSurrogate: anchor is not a node");
    order_.push_back(anchor);
    for (const auto& [id, nd] : g.nodes())
        if (id != anchor) order_.push_back(id);
    for (int k = 0; k < n(); ++k) index_[order_[size_t(k)]] = k;

    std::vector<int> parent(static_cast<size_t>(n()));
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](int x) {
        while (parent[size_t(x)] != x) x = parent[size_t(x)] = parent[size_t(parent[size_t(x)])];
        return x;
    };
    for (const PoseEdge& e : g.edges()) {
        if (!Active(e)) continue;
        if (e.from == e.to) throw std::invalid_argument("ChordalSurrogate: unary edges are not in the homogeneous form");
        const IsotropicWeights w = IsotropicFromInformation(e.information);
        const double s = Weight(e);
        edges_.push_back({index_.at(e.from), index_.at(e.to), e.Z.R, e.Z.t, s * w.kappa, s * w.tau});
        parent[size_t(find(edges_.back().i))] = find(edges_.back().j);
    }
    for (int k = 1; k < n(); ++k)
        if (find(k) != find(0)) throw std::invalid_argument("ChordalSurrogate: the active graph is not connected");

    // M = sum kappa B B^T + tau b b^T, with X B = R_j - R_i R~ and X b = t_j - t_i - R_i t~ (see header).
    const int N = 4 * n();
    Triplets T;
    auto add = [&T](int r, int c, double v) { T.emplace_back(r, c, v); };
    for (const Edge& e : edges_) {
        const int ri = n() + 3 * e.i, rj = n() + 3 * e.j;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                const double id = a == b ? e.kappa : 0.0;
                add(rj + a, rj + b, id);
                add(ri + a, ri + b, id + e.tau * e.t(a) * e.t(b));
                add(rj + a, ri + b, -e.kappa * e.R(b, a));
                add(ri + a, rj + b, -e.kappa * e.R(a, b));
            }
        add(e.j, e.j, e.tau);
        add(e.i, e.i, e.tau);
        add(e.j, e.i, -e.tau);
        add(e.i, e.j, -e.tau);
        for (int a = 0; a < 3; ++a) {
            add(e.j, ri + a, -e.tau * e.t(a));
            add(ri + a, e.j, -e.tau * e.t(a));
            add(e.i, ri + a, e.tau * e.t(a));
            add(ri + a, e.i, e.tau * e.t(a));
        }
    }
    M_.resize(N, N);
    M_.setFromTriplets(T.begin(), T.end());
}

Eigen::MatrixXd ChordalSurrogate::StackX(const std::map<uint64_t, SE3>& poses) const {
    Eigen::MatrixXd X(3, 4 * n());
    for (int k = 0; k < n(); ++k) {
        const SE3& P = poses.at(order_[size_t(k)]);
        X.col(k) = P.t;
        X.block<3, 3>(0, n() + 3 * k) = P.R;
    }
    return X;
}

double ChordalSurrogate::Objective(const std::map<uint64_t, SE3>& poses) const {
    std::vector<const SE3*> P;
    for (uint64_t id : order_) P.push_back(&poses.at(id));
    double f = 0;
    for (const Edge& e : edges_) {
        const SE3 &Pi = *P[size_t(e.i)], &Pj = *P[size_t(e.j)];
        f += e.kappa * (Pj.R - Pi.R * e.R).squaredNorm() + e.tau * (Pj.t - Pi.t - Pi.R * e.t).squaredNorm();
    }
    return f;
}

double ChordalSurrogate::Linearize(const std::map<uint64_t, SE3>& poses, SparseMat& H, Eigen::VectorXd& grad) const {
    const int dim = 6 * (n() - 1);
    std::vector<const SE3*> P;
    for (uint64_t id : order_) P.push_back(&poses.at(id));
    grad = Eigen::VectorXd::Zero(dim);
    Triplets T;
    for (int k = 0; k < dim; ++k) T.emplace_back(k, k, 0.0);
    double energy = 0;
    for (const Edge& e : edges_) {
        const SE3 &Pi = *P[size_t(e.i)], &Pj = *P[size_t(e.j)];
        const double sk = std::sqrt(e.kappa), st = std::sqrt(e.tau);
        // Rows: 9 rotation (column c of R_j - R_i R~ at rows 3c..3c+2), 3 translation. Cols: [dt_i th_i dt_j th_j].
        Eigen::Matrix<double, 12, 12> J = Eigen::Matrix<double, 12, 12>::Zero();
        Eigen::Matrix<double, 12, 1> r;
        const Mat3 RiRz = Pi.R * e.R;
        for (int c = 0; c < 3; ++c) {
            r.segment<3>(3 * c) = sk * (Pj.R.col(c) - RiRz.col(c));
            J.block<3, 3>(3 * c, 3) = sk * core::Hat(RiRz.col(c));      // d(-R_i R~ e_c)/d th_i
            J.block<3, 3>(3 * c, 9) = -sk * core::Hat(Pj.R.col(c));     // d(R_j e_c)/d th_j
        }
        const Vec3 Rit = Pi.R * e.t;
        r.segment<3>(9) = st * (Pj.t - Pi.t - Rit);
        J.block<3, 3>(9, 0) = -st * Mat3::Identity();
        J.block<3, 3>(9, 3) = st * core::Hat(Rit);
        J.block<3, 3>(9, 6) = st * Mat3::Identity();
        energy += 0.5 * r.squaredNorm();
        const Eigen::Matrix<double, 12, 12> JtJ = J.transpose() * J;
        const Eigen::Matrix<double, 12, 1> Jtr = J.transpose() * r;
        const int off[2] = {6 * (e.i - 1), 6 * (e.j - 1)};   // node 0 is the anchor: no columns
        for (int a = 0; a < 2; ++a) {
            if (off[a] < 0) continue;
            grad.segment<6>(off[a]) += Jtr.segment<6>(6 * a);
            for (int b = 0; b < 2; ++b) {
                if (off[b] < 0) continue;
                for (int p = 0; p < 6; ++p)
                    for (int q = 0; q < 6; ++q) T.emplace_back(off[a] + p, off[b] + q, JtJ(6 * a + p, 6 * b + q));
            }
        }
    }
    H.resize(dim, dim);
    H.setFromTriplets(T.begin(), T.end());
    return energy;
}

std::map<uint64_t, SE3> ChordalSurrogate::Oplus(const std::map<uint64_t, SE3>& poses, const Eigen::VectorXd& x) const {
    std::map<uint64_t, SE3> out = poses;
    for (int k = 1; k < n(); ++k) {
        SE3& P = out.at(order_[size_t(k)]);
        P.t += x.segment<3>(6 * (k - 1));
        P.R = core::SO3Exp(x.segment<3>(6 * (k - 1) + 3)) * P.R;
    }
    return out;
}

std::map<uint64_t, SE3> ChordalSurrogate::SolveTranslations(const std::map<uint64_t, SE3>& poses) const {
    // Residual t_j - t_i - c, c = R_i t~, weight tau: a Laplacian in the n - 1 free translations, three right sides.
    const int m = n() - 1;
    std::map<uint64_t, SE3> out = poses;
    if (m == 0) return out;
    const Vec3 t0 = poses.at(order_[0]).t;
    Triplets T;
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(m, 3);
    for (const Edge& e : edges_) {
        const Vec3 c = poses.at(order_[size_t(e.i)]).R * e.t;
        const int i = e.i - 1, j = e.j - 1;   // -1: the anchor, known
        if (j >= 0) {
            T.emplace_back(j, j, e.tau);
            b.row(j) += e.tau * c.transpose();
            if (i >= 0) T.emplace_back(j, i, -e.tau);
            else b.row(j) += e.tau * t0.transpose();
        }
        if (i >= 0) {
            T.emplace_back(i, i, e.tau);
            b.row(i) -= e.tau * c.transpose();
            if (j >= 0) T.emplace_back(i, j, -e.tau);
            else b.row(i) += e.tau * t0.transpose();
        }
    }
    SparseMat L(m, m);
    L.setFromTriplets(T.begin(), T.end());
    Eigen::SimplicialLDLT<SparseMat> ldlt(L);
    if (ldlt.info() != Eigen::Success) throw std::runtime_error("ChordalSurrogate: translation Laplacian is singular");
    const Eigen::MatrixXd t = ldlt.solve(b);
    for (int k = 1; k < n(); ++k) out.at(order_[size_t(k)]).t = t.row(k - 1).transpose();
    return out;
}

SparseMat ChordalSurrogate::CertificateMatrix(const Eigen::MatrixXd& X, std::vector<Mat3>* Lambda) const {
    const Eigen::MatrixXd Y = M_ * X.transpose();   // 4n x 3
    Triplets T;
    if (Lambda) Lambda->clear();
    for (int k = 0; k < n(); ++k) {
        const int r = n() + 3 * k;
        const Mat3 L = Y.block<3, 3>(r, 0) * X.block<3, 3>(0, r);
        if (Lambda) Lambda->push_back(L);
        const Mat3 Ls = 0.5 * (L + L.transpose());
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) T.emplace_back(r + a, r + b, -Ls(a, b));
    }
    SparseMat D(M_.rows(), M_.cols());
    D.setFromTriplets(T.begin(), T.end());
    return SparseMat(M_ + D);
}

CertificateReport CertifyPoseGraph(const PoseGraph& g, uint64_t anchor, const CertificateOptions& o,
                                   std::map<uint64_t, SE3>* point) {
    const ChordalSurrogate cs(g, anchor);
    CertificateReport rep;
    rep.nodes = cs.n();
    rep.edges = 0;
    for (const auto& e : g.edges()) rep.edges += Active(e);
    std::map<uint64_t, SE3> poses;
    for (const auto& [id, nd] : g.nodes()) poses[id] = nd.T_W_C;
    rep.objective_input = cs.Objective(poses);

    if (o.refine && cs.n() > 1) {
        const cv::Ptr<ChordalBackend> backend(new ChordalBackend(cs, poses));
        cv::detail::LevMarqBase lm(backend, ToOpenCv(o.lm));
        const cv::LevMarq::Report r = lm.optimize();
        rep.iterations = r.iters;
        rep.refine_found = r.found;
        std::map<uint64_t, SE3> refined = cs.SolveTranslations(backend->current());
        for (const auto& [id, P] : refined) {
            rep.moved_t = std::max(rep.moved_t, (P.t - poses.at(id).t).norm());
            rep.moved_r = std::max(rep.moved_r, core::SO3Log(P.R * poses.at(id).R.transpose()).norm());
        }
        poses = std::move(refined);
    }
    rep.objective = cs.Objective(poses);

    const Eigen::MatrixXd X = cs.StackX(poses);
    std::vector<Mat3> Lambda;
    const SparseMat S = cs.CertificateMatrix(X, &Lambda);
    const Eigen::MatrixXd Y = cs.M() * X.transpose();
    double trace = 0;
    for (const Mat3& L : Lambda) {
        trace += L.trace();
        rep.max_rot_gradient = std::max(rep.max_rot_gradient, (0.5 * (L - L.transpose())).norm());
    }
    for (int k = 0; k < cs.n(); ++k) rep.max_trans_gradient = std::max(rep.max_trans_gradient, Y.row(k).norm());
    rep.duality_gap = rep.objective - trace;
    for (int k = 0; k < cs.M().outerSize(); ++k) {   // symmetric: column sums are row sums
        double s = 0;
        for (SparseMat::InnerIterator it(cs.M(), k); it; ++it) s += std::fabs(it.value());
        rep.norm_M = std::max(rep.norm_M, s);
    }
    rep.eta = o.rel_eta * rep.norm_M;

    // Ground the anchor's translation (index 0), shift the rotation diagonal only (see header).
    const int N = int(S.rows()), nt = cs.n() - 1;
    const SparseMat Sg = S.bottomRightCorner(N - 1, N - 1);
    GuardedLdlt ldlt;
    auto passes = [&](double eta) {
        SparseMat A = Sg;
        for (int i = nt; i < N - 1; ++i) A.coeffRef(i, i) += eta;
        return ldlt.Factorize(A);
    };
    rep.certified = passes(rep.eta);
    rep.certified_abs = passes(o.abs_eta);
    for (int k = 2; k <= 16; ++k) {
        const double eta = rep.norm_M * std::pow(10.0, -k);
        if (!passes(eta)) break;
        rep.eta_min_passing = eta;
    }
    char buf[256];
    if (rep.certified) {
        rep.suboptimality_bound = std::fabs(rep.duality_gap) + 3.0 * cs.n() * rep.eta;
        std::snprintf(buf, sizeof buf,
                      "certified for the isotropic chordal surrogate (eta %.2e = %.0e ||M||; f - f* <= %.3g, %.2g%% of "
                      "f = %.6g)",
                      rep.eta, o.rel_eta, rep.suboptimality_bound,
                      100.0 * rep.suboptimality_bound / std::max(rep.objective, 1e-300), rep.objective);
    } else {
        // Reduced rotation matrix C - B^T A^-1 B (SE-Sync's Q~ - Lambda), A the grounded translation Laplacian.
        const SparseMat A = Sg.topLeftCorner(nt, nt), B = Sg.topRightCorner(nt, N - 1 - nt),
                        C = Sg.bottomRightCorner(N - 1 - nt, N - 1 - nt);
        Eigen::SimplicialLDLT<SparseMat> la;
        if (nt > 0) la.compute(A);
        auto op = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
            Eigen::VectorXd w = C * v;
            if (nt > 0) w -= B.transpose() * la.solve(B * v);
            return w;
        };
        rep.lambda_min_ritz = SmallestRitz(op, int(C.rows()), o.lanczos_steps);
        std::snprintf(buf, sizeof buf, "not certified (lambda_min <= %.3e at eta %.2e): unknown, not wrong",
                      rep.lambda_min_ritz, rep.eta);
    }
    rep.verdict = buf;
    if (point) *point = poses;
    return rep;
}

}  // namespace ariadne::slam
