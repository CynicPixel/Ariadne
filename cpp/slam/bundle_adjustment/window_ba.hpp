#pragma once
// ---------------------------------------------------------------------------
// Windowed stereo bundle adjustment (arch §11.3-§11.6; design LOCAL_SLAM_2.3.md),
// OpenCV-free: residuals, analytic Jacobians, Huber/IRLS assembly, and the
// Schur elimination of landmarks. OpenCV 5's cv::detail::LevMarqBase drives
// it through window_ba_levmarq.hpp (D51), exactly as for the pose graph.
//
// STATE. Keyframe poses T_W_C (left perturbation T <- exp(d) T, d = [rho; phi],
// the PoseNode convention) and landmarks as world points X. The oldest
// keyframe (and any other marked fixed) is not a variable: the gauge (§11.3).
//
// RESIDUAL of an observation z of X in keyframe k (left camera, or the right
// camera of the rectified pair, offset by the baseline along x):
//     Xc = R^T (X - t) - (right ? (B,0,0) : 0),   r = pi(Xc) - z
// JACOBIANS (design §2):
//     dXc/drho = -R^T,   dXc/dphi = R^T [X]x,   dXc/dX = R^T
//     dpi/dXc  = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]
// HUBER on s = |r| / sigma (delta = 2.0, §11.3): rho(s) = s^2/2 (s <= delta),
// delta (s - delta/2) otherwise; IRLS weight w = 1 or delta/s. The energy the
// LM driver sees IS the robust cost, and H, g are the IRLS-weighted ones at
// the linearisation point, so g is exactly the robust cost's gradient.
//
// SCHUR (§11.5). H = [[Hcc, Hcl], [Hlc, Hll]] with Hll block-diagonal (3x3 per
// landmark). The LM driver damps the FULL diagonal; damping is applied first,
// then landmarks are eliminated:
//     S = Hcc - sum_l Hcl Hll^-1 Hlc,  S dc = -(gc - sum_l Hcl Hll^-1 gl),
//     dl = Hll^-1 (-gl - Hlc dc)
// and the FULL step (cameras + landmarks) is returned, because the driver's
// predicted-decrease and step-norm tests use the whole vector. S is dense
// (<= 6 * 14 = 84 rows at window 15): Eigen LDLT with the positive-pivot guard.
//
// A landmark that projects behind a camera (z <= z_floor) makes the energy
// enormous (finite), so the driver rejects that step and raises lambda,
// instead of evaluating pi at a singularity.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <Eigen/Dense>

#include <algorithm>

#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

using core::Mat3;
using core::Mat6;
using core::SE3;
using core::Vec3;
using core::Vec6;
using Vec2 = Eigen::Vector2d;

struct BaCamera {
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline = 0;
};

struct BaObservation {
    int kf = 0;            // index into the window's keyframes
    uint64_t landmark = 0;
    Vec2 z;                // rectified pixel
    bool right = false;    // observed in the right camera
    double sigma = 1.0;    // pixel standard deviation
};

struct BaWindow {
    BaCamera cam;
    std::vector<SE3> T_W_C;          // keyframe poses, oldest first
    std::vector<bool> fixed;         // fixed[k]: not a variable (the gauge is fixed[0])
    std::map<uint64_t, Vec3> X;      // landmarks
    std::set<uint64_t> fixed_landmarks;   // not variables (motion-only refinement; the PnP oracle)
    std::vector<BaObservation> obs;
    double huber_delta = 2.0;
    double z_floor = 1e-3;           // m: behind / at the camera
};

struct ObsLinearization {
    bool valid = false;              // in front of the camera
    Vec2 r;
    Eigen::Matrix<double, 2, 6> Jc;
    Eigen::Matrix<double, 2, 3> Jl;
};

/// Residual and Jacobians of one observation at the given pose and point.
inline ObsLinearization LinearizeObservation(const BaCamera& cam, const SE3& T_W_C, const Vec3& X,
                                             const BaObservation& o, double z_floor) {
    ObsLinearization l;
    const Mat3 Rt = T_W_C.R.transpose();
    Vec3 Xc = Rt * (X - T_W_C.t);
    if (o.right) Xc.x() -= cam.baseline;
    if (!(Xc.z() > z_floor)) return l;
    l.valid = true;
    const double iz = 1.0 / Xc.z();
    l.r = Vec2(cam.fx * Xc.x() * iz + cam.cx, cam.fy * Xc.y() * iz + cam.cy) - o.z;
    Eigen::Matrix<double, 2, 3> P;
    P << cam.fx * iz, 0.0, -cam.fx * Xc.x() * iz * iz, 0.0, cam.fy * iz, -cam.fy * Xc.y() * iz * iz;
    l.Jc.leftCols<3>() = -P * Rt;
    l.Jc.rightCols<3>() = P * Rt * core::Hat(X);
    l.Jl = P * Rt;
    return l;
}

/// Huber cost and IRLS weight of a whitened residual norm s.
inline void Huber(double s, double delta, double& cost, double& weight) {
    if (s <= delta) {
        cost = 0.5 * s * s;
        weight = 1.0;
    } else {
        cost = delta * (s - 0.5 * delta);
        weight = delta / s;
    }
}

/// The parameter layout: free cameras (6 each, window order), then landmarks (3 each, ascending id).
class BaLayout {
public:
    explicit BaLayout(const BaWindow& w) {
        if (w.T_W_C.size() != w.fixed.size()) throw std::invalid_argument("BaLayout: poses and fixed flags differ");
        for (size_t k = 0; k < w.T_W_C.size(); ++k) cam_index_.push_back(w.fixed[k] ? -1 : n_cams_++);
        for (const auto& [id, X] : w.X) lm_index_[id] = w.fixed_landmarks.count(id) ? -1 : n_lms_++;
        if (n_cams_ == 0 && n_lms_ == 0) throw std::invalid_argument("BaLayout: nothing to optimise");
    }
    int n_cams() const { return n_cams_; }
    int n_lms() const { return n_lms_; }
    int dim_c() const { return 6 * n_cams_; }
    int dim() const { return 6 * n_cams_ + 3 * n_lms_; }
    int cam(int k) const { return cam_index_[size_t(k)]; }
    int lm(uint64_t id) const { return lm_index_.at(id); }

private:
    int n_cams_ = 0, n_lms_ = 0;
    std::vector<int> cam_index_;
    std::map<uint64_t, int> lm_index_;
};

/// Robust energy of the whole window. Behind-camera observations make it huge (finite): the step is rejected.
inline double BaEnergy(const BaWindow& w) {
    double E = 0.0;
    for (const auto& o : w.obs) {
        const ObsLinearization l = LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor);
        if (!l.valid) return 1e300;
        double c, wt;
        Huber(l.r.norm() / o.sigma, w.huber_delta, c, wt);
        E += c;
    }
    return E;
}

/// The normal equations in block form (IRLS-weighted at the current point).
struct BaSystem {
    Eigen::MatrixXd Hcc;                                   // dim_c x dim_c
    std::map<std::pair<int, int>, Eigen::Matrix<double, 6, 3>> Hcl;   // (camera, landmark) -> block
    std::vector<Mat3> Hll;                                 // per landmark
    Eigen::VectorXd g;                                     // [gc; gl], +J^T W r
    std::vector<std::vector<int>> lm_cams;                 // per landmark: cameras that see it (sorted)
};

inline double BaLinearize(const BaWindow& w, const BaLayout& L, BaSystem& s) {
    s.Hcc = Eigen::MatrixXd::Zero(L.dim_c(), L.dim_c());
    s.Hcl.clear();
    s.Hll.assign(size_t(L.n_lms()), Mat3::Zero());
    s.g = Eigen::VectorXd::Zero(L.dim());
    s.lm_cams.assign(size_t(L.n_lms()), {});
    double E = 0.0;
    for (const auto& o : w.obs) {
        const ObsLinearization l = LinearizeObservation(w.cam, w.T_W_C[size_t(o.kf)], w.X.at(o.landmark), o, w.z_floor);
        if (!l.valid) return 1e300;
        double c, wt;
        Huber(l.r.norm() / o.sigma, w.huber_delta, c, wt);
        E += c;
        const double W = wt / (o.sigma * o.sigma);
        const int ci = L.cam(o.kf), li = L.lm(o.landmark);
        if (li >= 0) {
            s.Hll[size_t(li)] += W * l.Jl.transpose() * l.Jl;
            s.g.segment<3>(L.dim_c() + 3 * li) += W * l.Jl.transpose() * l.r;
        }
        if (ci >= 0) {
            s.Hcc.block<6, 6>(6 * ci, 6 * ci) += W * l.Jc.transpose() * l.Jc;
            s.g.segment<6>(6 * ci) += W * l.Jc.transpose() * l.r;
        }
        if (ci >= 0 && li >= 0) {
            const size_t lis = size_t(li);
            auto it = s.Hcl.find({ci, li});
            const Eigen::Matrix<double, 6, 3> B = W * l.Jc.transpose() * l.Jl;
            if (it == s.Hcl.end()) {
                s.Hcl.emplace(std::make_pair(ci, li), B);
                s.lm_cams[lis].push_back(ci);
            } else {
                it->second += B;
            }
        }
    }
    for (auto& v : s.lm_cams) std::sort(v.begin(), v.end());
    return E;
}

/// Diagonal of the full H (cameras then landmarks), and writing it back (the driver's damping).
inline Eigen::VectorXd BaDiagonal(const BaSystem& s, const BaLayout& L) {
    Eigen::VectorXd d(L.dim());
    d.head(L.dim_c()) = s.Hcc.diagonal();
    for (int l = 0; l < L.n_lms(); ++l) d.segment<3>(L.dim_c() + 3 * l) = s.Hll[size_t(l)].diagonal();
    return d;
}

inline void BaSetDiagonal(BaSystem& s, const BaLayout& L, const Eigen::VectorXd& d) {
    s.Hcc.diagonal() = d.head(L.dim_c());
    for (int l = 0; l < L.n_lms(); ++l) s.Hll[size_t(l)].diagonal() = d.segment<3>(L.dim_c() + 3 * l);
}

/// Schur elimination with guards. Factorize() returns false (never throws) on a non-PD system so the LM driver
/// treats it as a failed step; Solve() returns the FULL step x for (H) x = -right.
class BaSchur {
public:
    bool Factorize(const BaSystem& s, const BaLayout& L) {
        ok_ = false;
        Hll_inv_.assign(size_t(L.n_lms()), Mat3::Zero());
        for (int l = 0; l < L.n_lms(); ++l) {
            const Eigen::LDLT<Mat3> ld(s.Hll[size_t(l)]);
            if (ld.info() != Eigen::Success || !(ld.vectorD().array() > 0.0).all()) return false;
            Hll_inv_[size_t(l)] = ld.solve(Mat3::Identity());
        }
        S_ = s.Hcc;
        for (int l = 0; l < L.n_lms(); ++l) {
            const auto& cams = s.lm_cams[size_t(l)];
            for (int a : cams) {
                const Eigen::Matrix<double, 6, 3> Ba = s.Hcl.at({a, l}) * Hll_inv_[size_t(l)];
                for (int b : cams) S_.block<6, 6>(6 * a, 6 * b) -= Ba * s.Hcl.at({b, l}).transpose();
            }
        }
        if (L.n_cams() > 0) {
            ldlt_.compute(S_);
            if (ldlt_.info() != Eigen::Success || !(ldlt_.vectorD().array() > 0.0).all()) return false;
        }
        ok_ = true;
        return true;
    }

    Eigen::VectorXd Solve(const BaSystem& s, const BaLayout& L, const Eigen::VectorXd& right) const {
        if (!ok_) throw std::logic_error("BaSchur::Solve without a successful factorization");
        const int nc = L.dim_c();
        Eigen::VectorXd b_c = -right.head(nc);
        // b_c -= sum_l Hcl Hll^-1 (-r_l)  ==  rhs of the reduced system for -right
        for (int l = 0; l < L.n_lms(); ++l) {
            const Vec3 bl = -right.segment<3>(nc + 3 * l);
            const Vec3 y = Hll_inv_[size_t(l)] * bl;
            for (int a : s.lm_cams[size_t(l)]) b_c.segment<6>(6 * a) -= s.Hcl.at({a, l}) * y;
        }
        Eigen::VectorXd x(L.dim());
        if (nc > 0) x.head(nc) = ldlt_.solve(b_c);
        for (int l = 0; l < L.n_lms(); ++l) {
            Vec3 bl = -right.segment<3>(nc + 3 * l);
            for (int a : s.lm_cams[size_t(l)]) bl -= s.Hcl.at({a, l}).transpose() * x.segment<6>(6 * a);
            x.segment<3>(nc + 3 * l) = Hll_inv_[size_t(l)] * bl;
        }
        return x;
    }

    /// Marginal covariance (left tangent of T_W_C) of free camera c at the solution: the 6x6 block of S^-1, from
    /// S X = E (never an explicit inverse, §11.6). Call on an UNDAMPED factorization.
    Mat6 CameraCovariance(int c) const {
        if (!ok_) throw std::logic_error("BaSchur::CameraCovariance without a factorization");
        Eigen::MatrixXd E = Eigen::MatrixXd::Zero(S_.rows(), 6);
        E.block<6, 6>(6 * c, 0) = Mat6::Identity();
        const Mat6 C = ldlt_.solve(E).block<6, 6>(6 * c, 0);
        return 0.5 * (C + C.transpose());
    }

    const Eigen::MatrixXd& S() const { return S_; }

private:
    bool ok_ = false;
    std::vector<Mat3> Hll_inv_;
    Eigen::MatrixXd S_;
    Eigen::LDLT<Eigen::MatrixXd> ldlt_;
};

/// Apply a full step: cameras by the left exp, landmarks additively.
inline BaWindow BaOplus(const BaWindow& w, const BaLayout& L, const Eigen::VectorXd& x) {
    if (x.size() != L.dim()) throw std::invalid_argument("BaOplus: wrong step size");
    BaWindow out = w;
    for (size_t k = 0; k < w.T_W_C.size(); ++k) {
        const int c = L.cam(int(k));
        if (c < 0) continue;
        out.T_W_C[k] = core::SE3Exp(x.segment<6>(6 * c)) * w.T_W_C[k];
        out.T_W_C[k].normalize();
    }
    for (auto& [id, X] : out.X)
        if (L.lm(id) >= 0) X += x.segment<3>(L.dim_c() + 3 * L.lm(id));
    return out;
}

}  // namespace ariadne::slam
