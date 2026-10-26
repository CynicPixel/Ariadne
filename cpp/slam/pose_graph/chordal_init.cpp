// Chordal initialisation, MST baseline (design robust-backend.md §3; the math is in the header).
#include "slam/pose_graph/chordal_init.hpp"

#include "slam/pose_graph/pgo_problem.hpp"

#include <opencv2/core.hpp>
#include <opencv2/geometry/mst.hpp>

#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace ariadne::slam {

namespace {

using core::Mat3;
using core::Vec3;
using Triplets = std::vector<Eigen::Triplet<double>>;

/// Free-node index of every non-anchor node, ascending id (the PgoProblem ordering).
std::map<uint64_t, int> FreeIndex(const PoseGraph& g, uint64_t anchor) {
    std::map<uint64_t, int> idx;
    for (const auto& [id, n] : g.nodes())
        if (id != anchor) idx[id] = int(idx.size());
    return idx;
}

/// Solve the SPD system A X = B (A from triplets); throws if A is singular (a component not linked to the anchor).
Eigen::MatrixXd SolveSpd(int n, const Triplets& t, const Eigen::MatrixXd& B, const char* what) {
    SparseMat A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    GuardedLdlt ldlt;
    if (!ldlt.Factorize(A))
        throw std::runtime_error(std::string("InitialPoses: ") + what +
                                 " system is singular: some node is not connected to the anchor");
    Eigen::MatrixXd X(n, B.cols());
    for (int c = 0; c < B.cols(); ++c) X.col(c) = ldlt.Solve(B.col(c));
    return X;
}

std::map<uint64_t, SE3> Chordal(const PoseGraph& g, uint64_t anchor, InitReport* rep) {
    const auto idx = FreeIndex(g, anchor);
    const int m = int(idx.size());
    const SE3& Xa = g.nodes().at(anchor).T_W_C;
    std::vector<IsotropicWeights> w;
    for (const auto& e : g.edges()) w.push_back(IsotropicFromInformation(e.information));

    // Rotations: unknown x_i^k = (row k of R_i)^T. Residual of edge i -> j: x_j - R_Z^T x_i, weight kappa.
    // Normal blocks: (j,j) += I, (i,i) += R_Z R_Z^T = I, (j,i) += -R_Z^T, (i,j) += -R_Z. Known anchor rows go right.
    Triplets t;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(3 * m, 3);
    auto block = [&t](int r, int c, const Mat3& M, double s) {
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) t.emplace_back(3 * r + a, 3 * c + b, s * M(a, b));
    };
    for (size_t k = 0; k < g.edges().size(); ++k) {
        const PoseEdge& e = g.edges()[k];
        const double kap = w[k].kappa;
        const Mat3& RZ = e.Z.R;
        const bool unary = e.from == e.to;
        const int j = e.to == anchor ? -1 : idx.at(e.to);
        if (unary) {   // X_i = I: R_j = R_Z, i.e. x_j^k = (row k of R_Z)^T
            if (j < 0) continue;
            block(j, j, Mat3::Identity(), kap);
            B.block<3, 3>(3 * j, 0) += kap * RZ.transpose();   // column k = (row k of R_Z)^T
            continue;
        }
        const int i = e.from == anchor ? -1 : idx.at(e.from);
        if (i < 0 && j < 0) continue;
        if (j >= 0) block(j, j, Mat3::Identity(), kap);
        if (i >= 0) block(i, i, Mat3::Identity(), kap);
        if (i >= 0 && j >= 0) {
            block(j, i, RZ.transpose(), -kap);
            block(i, j, RZ, -kap);
        } else if (i < 0) {   // x_i known: anchor rows; column k of Ra^T is row k of Ra
            B.block<3, 3>(3 * j, 0) += kap * RZ.transpose() * Xa.R.transpose();
        } else {              // x_j known: (i, j) block times x_j moves right with a minus sign
            B.block<3, 3>(3 * i, 0) += kap * RZ * Xa.R.transpose();
        }
    }
    const Eigen::MatrixXd Xr = SolveSpd(3 * m, t, B, "chordal rotation");

    std::map<uint64_t, SE3> out;
    out[anchor] = Xa;
    for (const auto& [id, i] : idx) {
        Mat3 M;
        for (int k = 0; k < 3; ++k) M.row(k) = Xr.col(k).segment<3>(3 * i).transpose();
        bool fixed = false;
        SE3 T;
        T.R = ProjectToSO3(M, &fixed);
        if (rep) {
            rep->det_fixes += fixed;
            rep->max_projection = std::max(rep->max_projection, (M - T.R).norm());
        }
        out[id] = T;
    }

    // Translations, rotations fixed: t_j - t_i = R_i t_Z, weight tau. A scalar weighted Laplacian, 3 right-hand sides.
    t.clear();
    Eigen::MatrixXd Bt = Eigen::MatrixXd::Zero(m, 3);
    for (size_t k = 0; k < g.edges().size(); ++k) {
        const PoseEdge& e = g.edges()[k];
        const double tau = w[k].tau;
        const bool unary = e.from == e.to;
        const int j = e.to == anchor ? -1 : idx.at(e.to);
        if (unary) {   // X_i = I: t_j = t_Z
            if (j < 0) continue;
            t.emplace_back(j, j, tau);
            Bt.row(j) += tau * e.Z.t.transpose();
            continue;
        }
        const int i = e.from == anchor ? -1 : idx.at(e.from);
        if (i < 0 && j < 0) continue;
        const Vec3 c = out.at(e.from).R * e.Z.t;
        if (j >= 0) {
            t.emplace_back(j, j, tau);
            Bt.row(j) += tau * c.transpose();
        }
        if (i >= 0) {
            t.emplace_back(i, i, tau);
            Bt.row(i) -= tau * c.transpose();
        }
        if (i >= 0 && j >= 0) {
            t.emplace_back(i, j, -tau);
            t.emplace_back(j, i, -tau);
        } else if (i < 0) {
            Bt.row(j) += tau * Xa.t.transpose();
        } else {
            Bt.row(i) += tau * Xa.t.transpose();
        }
    }
    const Eigen::MatrixXd Xt = SolveSpd(m, t, Bt, "chordal translation");
    for (const auto& [id, i] : idx) out[id].t = Xt.row(i).transpose();
    return out;
}

std::map<uint64_t, SE3> Mst(const PoseGraph& g, uint64_t anchor) {
    std::map<uint64_t, int> all;
    std::vector<uint64_t> ids;
    for (const auto& [id, n] : g.nodes()) {
        all[id] = int(ids.size());
        ids.push_back(id);
    }
    // One candidate per unordered pair: the highest kappa (buildMST keeps the lowest weight of duplicates; we keep
    // the same edge so the tree edge maps back to exactly one measurement). Ties: the first edge, deterministically.
    std::map<std::pair<int, int>, size_t> best;
    std::vector<double> kappa;
    for (size_t k = 0; k < g.edges().size(); ++k) {
        const PoseEdge& e = g.edges()[k];
        kappa.push_back(e.from == e.to ? 0.0 : IsotropicFromInformation(e.information).kappa);
        if (e.from == e.to) continue;
        const int a = all.at(e.from), b = all.at(e.to);
        const auto key = std::minmax(a, b);
        auto it = best.find(key);
        if (it == best.end() || kappa[k] > kappa[it->second]) best[key] = k;
    }
    std::vector<cv::MSTEdge> in, tree;
    for (const auto& [key, k] : best) in.push_back({key.first, key.second, -kappa[k]});
    if (!cv::buildMST(int(ids.size()), in, tree, cv::MST_PRIM, all.at(anchor)))
        throw std::runtime_error("InitialPoses: the graph is not connected (cv::buildMST failed)");
    std::vector<std::vector<int>> adj(ids.size());
    for (const auto& te : tree) {
        adj[te.source].push_back(te.target);
        adj[te.target].push_back(te.source);
    }
    std::map<uint64_t, SE3> out;
    out[anchor] = g.nodes().at(anchor).T_W_C;
    std::queue<int> q;
    q.push(all.at(anchor));
    std::vector<bool> seen(ids.size(), false);
    seen[all.at(anchor)] = true;
    while (!q.empty()) {
        const int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (seen[v]) continue;
            seen[v] = true;
            const PoseEdge& e = g.edges()[best.at(std::minmax(u, v))];
            // Z = X_from^-1 X_to: walking forward X_to = X_from Z, backward X_from = X_to Z^-1.
            out[ids[v]] = e.from == ids[u] ? out.at(ids[u]) * e.Z : out.at(ids[u]) * e.Z.inverse();
            q.push(v);
        }
    }
    if (out.size() != ids.size()) throw std::runtime_error("InitialPoses: spanning tree does not reach every node");
    return out;
}

}  // namespace

IsotropicWeights IsotropicFromInformation(const Mat6& I) {
    // [rho; phi]: translation block first. Traces of the INVERSE diagonal blocks (SE-Sync's g2o reader convention).
    const Eigen::LDLT<Mat3> lt(I.topLeftCorner<3, 3>()), lr(I.bottomRightCorner<3, 3>());
    if (lt.info() != Eigen::Success || lr.info() != Eigen::Success || !lt.isPositive() || !lr.isPositive() ||
        (lt.vectorD().array() <= 0.0).any() || (lr.vectorD().array() <= 0.0).any())
        throw std::invalid_argument("IsotropicFromInformation: a diagonal information block is not positive definite");
    IsotropicWeights w;
    w.tau = 3.0 / lt.solve(Mat3::Identity()).trace();
    w.kappa = 3.0 / (2.0 * lr.solve(Mat3::Identity()).trace());
    return w;
}

Mat3 ProjectToSO3(const Mat3& M, bool* det_fixed) {
    cv::Matx33d m;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) m(r, c) = M(r, c);
    cv::Mat w, u, vt;
    cv::SVDecomp(m, w, u, vt, cv::SVD::FULL_UV);
    const cv::Matx33d U(u), Vt(vt);
    const double d = cv::determinant(U * Vt) < 0.0 ? -1.0 : 1.0;
    const cv::Matx33d R = U * cv::Matx33d::diag(cv::Vec3d(1.0, 1.0, d)) * Vt;
    if (det_fixed) *det_fixed = d < 0.0;
    Mat3 out;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out(r, c) = R(r, c);
    return out;
}

std::map<uint64_t, SE3> InitialPoses(const PoseGraph& g, uint64_t anchor, PgoInit method, InitReport* report) {
    if (!g.nodes().count(anchor)) throw std::invalid_argument("InitialPoses: anchor is not a node");
    if (report) *report = InitReport{method};
    switch (method) {
        case PgoInit::kChordal: return Chordal(g, anchor, report);
        case PgoInit::kMst: return Mst(g, anchor);
        case PgoInit::kCurrent: break;
    }
    std::map<uint64_t, SE3> out;
    for (const auto& [id, n] : g.nodes()) out[id] = n.T_W_C;
    return out;
}

void ApplyPoses(PoseGraph& g, const std::map<uint64_t, SE3>& poses) {
    const Mat6 unknown = Mat6::Constant(std::numeric_limits<double>::quiet_NaN());
    for (const auto& [id, T] : poses) g.UpdateNode(id, T, unknown);
}

}  // namespace ariadne::slam
