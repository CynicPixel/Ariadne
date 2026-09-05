#pragma once
// ---------------------------------------------------------------------------
// Covariance transport on SE(3) (arch §37.1 "covariance transform").
//
// A pose covariance is expressed in the pose's own TANGENT space, so moving one
// between frames is conjugation by the SE(3) adjoint -- not by the rotation.
// Both rules below are derived for Ariadne's global LEFT-multiplicative
// convention (T = exp(xi) * That) and are verified by Monte Carlo in the tests,
// because a covariance rule that is wrong by a transpose still produces
// plausible-looking numbers forever.
//
//   Composition  C = A * B, with A and B independent:
//     exp(xi_c) Chat = exp(xi_a) Ahat exp(xi_b) Bhat
//                    = exp(xi_a) exp(Adj(Ahat) xi_b) Chat
//     => xi_c ~ xi_a + Adj(Ahat) xi_b        (first order)
//     => Sigma_c = Sigma_a + Adj(Ahat) Sigma_b Adj(Ahat)^T
//
//   Inverse:
//     exp(xi_inv) That^-1 = (exp(xi) That)^-1 = exp(-Adj(That^-1) xi) That^-1
//     => Sigma_inv = Adj(That^-1) Sigma Adj(That^-1)^T
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace ariadne::core {

/// General first-order transport through a linear map: Sigma_out = J Sigma J^T.
template <int Rows, int Cols>
inline Eigen::Matrix<double, Rows, Rows> TransportCovariance(
    const Eigen::Matrix<double, Rows, Cols>& J,
    const Eigen::Matrix<double, Cols, Cols>& cov) {
    return J * cov * J.transpose();
}

/// Covariance of A * B where A and B are independent. `A` is the MEAN of the
/// left factor; the adjoint is evaluated at the mean, which is what makes this
/// a first-order (not exact) result.
inline Mat6 ComposeCovariance(const SE3& A, const Mat6& cov_a, const Mat6& cov_b) {
    const Mat6 adj = SE3Adjoint(A);
    return cov_a + adj * cov_b * adj.transpose();
}

/// Covariance of T^-1 given the covariance of T.
inline Mat6 InvertCovariance(const SE3& T, const Mat6& cov) {
    const Mat6 adj = SE3Adjoint(T.inverse());
    return adj * cov * adj.transpose();
}

/// Re-express a tangent-space covariance in another frame related by `T`.
inline Mat6 ChangeFrame(const SE3& T, const Mat6& cov) {
    const Mat6 adj = SE3Adjoint(T);
    return adj * cov * adj.transpose();
}

/// True if `m` is symmetric and positive semi-definite to within `tol`.
/// Used as an assertion at interfaces: an indefinite covariance means an
/// upstream bug, and silently continuing on one corrupts every later estimate.
inline bool IsValidCovariance(const Mat6& m, double tol = 1e-9) {
    if ((m - m.transpose()).cwiseAbs().maxCoeff() > tol) return false;
    Eigen::SelfAdjointEigenSolver<Mat6> es(m);
    if (es.info() != Eigen::Success) return false;
    return es.eigenvalues().minCoeff() >= -tol;
}

/// Nearest symmetric PSD matrix, by clamping negative eigenvalues to `floor`.
/// Small negative eigenvalues are a normal consequence of accumulated
/// floating-point work; this repairs them without hiding a genuinely broken
/// matrix, since the caller can compare before and after.
inline Mat6 NearestPSD(const Mat6& m, double floor_value = 0.0) {
    const Mat6 sym = 0.5 * (m + m.transpose());
    Eigen::SelfAdjointEigenSolver<Mat6> es(sym);
    Vec6 d = es.eigenvalues();
    for (int i = 0; i < 6; ++i) d(i) = std::max(d(i), floor_value);
    return es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();
}

}  // namespace ariadne::core
