#pragma once
// ---------------------------------------------------------------------------
// On-manifold IMU preintegration (Forster, Carlone, Dellaert, Scaramuzza, T-RO 2017, arXiv 1512.02363) and the
// inertial factor between two keyframes, in Ariadne's conventions (arch §10.2, design INERTIAL_2.4.md §4).
//
// WHAT IT IS. Between keyframes i and j the IMU gives hundreds of samples. Integrating them once, in the frame of
// body i and with the bias held at a linearisation point b̄, gives three "relative motion measurements"
//     ΔR_ij, Δv_ij, Δp_ij                       (eqs. 35-38)
// that do not depend on the (unknown, changing) world pose of i. When the optimiser later moves the bias by δb,
// they are corrected to first order through stored bias Jacobians (eq. 44) instead of re-integrating. Their 9x9
// covariance Σ_ij [δφ, δv, δp] is propagated per sample (eqs. 59-63).
//
// RESIDUAL (eq. 45), body poses R_i, p_i (world <- body), velocities v_i, v_j in the world, gravity g_W:
//     r_R = Log(ΔR̃(b)^T R_i^T R_j)
//     r_v = R_i^T (v_j - v_i - g Δt) - Δṽ(b)
//     r_p = R_i^T (p_j - p_i - v_i Δt - ½ g Δt²) - Δp̃(b)
// with the bias of keyframe i. Energy ½ r^T Σ^-1 r (the reprojection term's Huber quadratic is s²/2 too).
//
// CONVENTIONS. Forster perturbs rotations on the RIGHT; Ariadne's window perturbs T_W_C on the LEFT:
// T_W_C <- Exp(δ) T_W_C, δ = [ρ; φ]. Since T_W_B = T_W_C T_C_B, the same δ left-perturbs T_W_B: R <- Exp(φ) R,
// p <- Exp(φ) p + V(φ) ρ ≈ p - [p]x φ + ρ. The pose Jacobians below are derived for THAT perturbation (not copied
// from the paper); velocities and biases are additive. Every block is finite-difference tested
// (tests/numerical/test_preintegration.cpp), including large rotations.
//
// NOISE. Kalibr/EuRoC densities are continuous-time; the discrete per-sample covariance is density² / Δt (Forster,
// after eq. 47). EuRoC measured (INERTIAL_2.4 §4.1): the datasheet underestimates the white noise ~3x (gyro) and ~10x
// (accel), so the caller passes inflated densities.
//
// OpenCV-free; the window BA (window_ba.hpp) assembles these blocks next to the reprojection ones.
// ---------------------------------------------------------------------------
#include "core/inertial/imu.hpp"
#include "core/math/lie.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace ariadne::slam {

using core::Mat3;
using core::Vec3;
using Mat9 = Eigen::Matrix<double, 9, 9>;
using Vec9 = Eigen::Matrix<double, 9, 1>;

/// Right Jacobian of SO(3) and its inverse (Forster eqs. 8, 9).
inline Mat3 RightJacobianSO3(const Vec3& phi) {
    const double th = phi.norm();
    const Mat3 K = core::Hat(phi);
    if (th < 1e-6) return Mat3::Identity() - 0.5 * K + K * K / 6.0;
    return Mat3::Identity() - (1.0 - std::cos(th)) / (th * th) * K + (th - std::sin(th)) / (th * th * th) * K * K;
}

inline Mat3 RightJacobianInvSO3(const Vec3& phi) {
    const double th = phi.norm();
    const Mat3 K = core::Hat(phi);
    if (th < 1e-6) return Mat3::Identity() + 0.5 * K + K * K / 12.0;
    return Mat3::Identity() + 0.5 * K + (1.0 / (th * th) - (1.0 + std::cos(th)) / (2.0 * th * std::sin(th))) * K * K;
}

/// Inverse of a small SPD covariance (LDLT; never an explicit inverse of an indefinite matrix).
template <int N>
Eigen::Matrix<double, N, N> InformationOf(const Eigen::Matrix<double, N, N>& cov) {
    const Eigen::LDLT<Eigen::Matrix<double, N, N>> ld(cov);
    if (ld.info() != Eigen::Success || !(ld.vectorD().array() > 0.0).all())
        throw std::invalid_argument("InformationOf: covariance is not positive definite");
    return ld.solve(Eigen::Matrix<double, N, N>::Identity());
}

/// White-noise densities (already inflated by the caller) for the per-sample covariance.
struct PreintegrationNoise {
    double gyro_density = 0.0;    // rad/s/sqrt(Hz)
    double accel_density = 0.0;   // m/s^2/sqrt(Hz)
};

class Preintegrated {
public:
    Preintegrated() = default;
    Preintegrated(const Vec3& bg_lin, const Vec3& ba_lin, const PreintegrationNoise& noise)
        : bg_(bg_lin), ba_(ba_lin), noise_(noise) {}

    /// One IMU interval of length dt with the sample (gyro, accel) held over it (zero-order hold). Order matters:
    /// p and v use the rotation BEFORE this step (eqs. 59-61, and the bias-Jacobian sums of the appendix).
    void Integrate(const Vec3& gyro, const Vec3& accel, double dt) {
        if (!(dt > 0.0)) return;
        const Vec3 w = gyro - bg_, a = accel - ba_;
        const Mat3 dRk = core::SO3Exp(w * dt);
        const Mat3 Jr = RightJacobianSO3(w * dt);
        const Mat3 aHat = core::Hat(a);
        // The accelerometer sample is rotated by the MID-interval orientation dRmid = ΔR_ik Exp(½ ω dt) (as VINS-Mono
        // does). The paper's start-of-interval rotation is first order in ω dt: ~1 cm/s of Δv error over 0.5 s at
        // 1 rad/s and 200 Hz, as large as EuRoC's measured residuals (INERTIAL_2.4 §4.1). Covariance and bias
        // Jacobians below are re-derived for this mean (the appendix's sums with ΔR_ik -> dRmid), FD-tested.
        const Mat3 Eh = core::SO3Exp(0.5 * w * dt);
        const Mat3 dRmid = dR_ * Eh;
        // ∂dRmid/∂bg (right perturbation): Eh^T ∂ΔR_ik/∂bg - ½ Jr(½ ω dt) dt.
        const Mat3 Jmid = Eh.transpose() * JRg_ - 0.5 * RightJacobianSO3(0.5 * w * dt) * dt;
        // Covariance (eqs. 62-63 with dRmid), order [δφ, δv, δp]; δφ_ik enters dRmid as Eh^T δφ_ik.
        Mat9 A = Mat9::Identity();
        A.block<3, 3>(0, 0) = dRk.transpose();
        A.block<3, 3>(3, 0) = -dRmid * aHat * Eh.transpose() * dt;
        A.block<3, 3>(6, 0) = -0.5 * dRmid * aHat * Eh.transpose() * dt * dt;
        A.block<3, 3>(6, 3) = Mat3::Identity() * dt;
        Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
        B.block<3, 3>(0, 0) = Jr * dt;
        B.block<3, 3>(3, 3) = dRmid * dt;
        B.block<3, 3>(6, 3) = 0.5 * dRmid * dt * dt;
        Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = Mat3::Identity() * (noise_.gyro_density * noise_.gyro_density / dt);
        Q.block<3, 3>(3, 3) = Mat3::Identity() * (noise_.accel_density * noise_.accel_density / dt);
        Sigma_ = A * Sigma_ * A.transpose() + B * Q * B.transpose();
        // Bias Jacobians, with the pre-step values on the right-hand side.
        Jpa_ += Jva_ * dt - 0.5 * dRmid * dt * dt;
        Jpg_ += Jvg_ * dt - 0.5 * dRmid * aHat * Jmid * dt * dt;
        Jva_ += -dRmid * dt;
        Jvg_ += -dRmid * aHat * Jmid * dt;
        JRg_ = dRk.transpose() * JRg_ - Jr * dt;
        // Mean.
        dp_ += dv_ * dt + 0.5 * dRmid * a * dt * dt;
        dv_ += dRmid * a * dt;
        dR_ = dR_ * dRk;
        dt_ += dt;
    }

    /// The measurements at bias (bg, ba), first order in δb = b - b_lin (eq. 44).
    Mat3 DeltaR(const Vec3& bg) const { return dR_ * core::SO3Exp(JRg_ * (bg - bg_)); }
    Vec3 DeltaV(const Vec3& bg, const Vec3& ba) const { return dv_ + Jvg_ * (bg - bg_) + Jva_ * (ba - ba_); }
    Vec3 DeltaP(const Vec3& bg, const Vec3& ba) const { return dp_ + Jpg_ * (bg - bg_) + Jpa_ * (ba - ba_); }

    double dt() const { return dt_; }
    const Mat9& Sigma() const { return Sigma_; }
    const Vec3& bg_lin() const { return bg_; }
    const Vec3& ba_lin() const { return ba_; }
    const Mat3& JRg() const { return JRg_; }
    const Mat3& Jvg() const { return Jvg_; }
    const Mat3& Jva() const { return Jva_; }
    const Mat3& Jpg() const { return Jpg_; }
    const Mat3& Jpa() const { return Jpa_; }

private:
    Vec3 bg_ = Vec3::Zero(), ba_ = Vec3::Zero();
    PreintegrationNoise noise_;
    Mat3 dR_ = Mat3::Identity();
    Vec3 dv_ = Vec3::Zero(), dp_ = Vec3::Zero();
    double dt_ = 0.0;
    Mat9 Sigma_ = Mat9::Zero();
    Mat3 JRg_ = Mat3::Zero(), Jvg_ = Mat3::Zero(), Jva_ = Mat3::Zero(), Jpg_ = Mat3::Zero(), Jpa_ = Mat3::Zero();
};

/// Preintegrate the samples over [t0, t1] (clipped zero-order hold, as core::IntegrateGyro). Throws if not covered.
inline Preintegrated Preintegrate(const std::vector<core::ImuMeasurement>& imu, int64_t t0, int64_t t1,
                                  const Vec3& bg, const Vec3& ba, const PreintegrationNoise& noise) {
    if (t1 < t0 || imu.empty() || imu.front().t_ns > t0 || imu.back().t_ns < t1) {
        throw std::invalid_argument("Preintegrate: samples do not cover [t0, t1]");
    }
    Preintegrated p(bg, ba, noise);
    for (size_t k = 0; k + 1 < imu.size(); ++k) {
        const int64_t a = std::max(imu[k].t_ns, t0), b = std::min(imu[k + 1].t_ns, t1);
        if (b <= a) continue;
        p.Integrate(imu[k].gyro, imu[k].accel, 1e-9 * double(b - a));
    }
    return p;
}

/// The inertial factor's residual and Jacobians. Pose blocks are w.r.t. the LEFT perturbation [ρ; φ] of the body
/// pose (equivalently of T_W_C, see header); velocity and bias blocks are additive.
struct InertialLinearization {
    Vec9 r = Vec9::Zero();
    Eigen::Matrix<double, 9, 6> J_Ti, J_Tj;
    Eigen::Matrix<double, 9, 3> J_vi, J_vj, J_bgi, J_bai;
};

inline InertialLinearization LinearizeInertial(const Preintegrated& pre, const core::SE3& T_W_Bi, const Vec3& v_i,
                                               const Vec3& bg_i, const Vec3& ba_i, const core::SE3& T_W_Bj,
                                               const Vec3& v_j, const Vec3& g_W) {
    InertialLinearization L;
    const Mat3& Ri = T_W_Bi.R;
    const Mat3& Rj = T_W_Bj.R;
    const Vec3& pi = T_W_Bi.t;
    const Vec3& pj = T_W_Bj.t;
    const double T = pre.dt();
    const Mat3 dR = pre.DeltaR(bg_i);
    const Vec3 u = v_j - v_i - g_W * T;
    const Vec3 w = pj - pi - v_i * T - 0.5 * g_W * T * T;
    const Vec3 rR = core::SO3Log(dR.transpose() * Ri.transpose() * Rj);
    L.r << rR, Ri.transpose() * u - pre.DeltaV(bg_i, ba_i), Ri.transpose() * w - pre.DeltaP(bg_i, ba_i);

    const Mat3 JrInv = RightJacobianInvSO3(rR);
    const Mat3 RiT = Ri.transpose();
    L.J_Ti.setZero();
    L.J_Tj.setZero();
    L.J_vi.setZero();
    L.J_vj.setZero();
    L.J_bgi.setZero();
    L.J_bai.setZero();
    // Rotation residual: only the φ parts (columns 3..5) of the poses; ρ does not enter.
    L.J_Ti.block<3, 3>(0, 3) = -JrInv * Rj.transpose();
    L.J_Tj.block<3, 3>(0, 3) = JrInv * Rj.transpose();
    // ∂/∂bg through ΔR̃(b) = ΔR̄ Exp(J δbg): -Jr^-1(r) Exp(r)^T Jr(J δbg) J (Forster's bias Jacobian of r_R).
    L.J_bgi.block<3, 3>(0, 0) =
        -JrInv * core::SO3Exp(rR).transpose() * RightJacobianSO3(pre.JRg() * (bg_i - pre.bg_lin())) * pre.JRg();
    // Velocity residual.
    L.J_Ti.block<3, 3>(3, 3) = RiT * core::Hat(u);
    L.J_vi.block<3, 3>(3, 0) = -RiT;
    L.J_vj.block<3, 3>(3, 0) = RiT;
    L.J_bgi.block<3, 3>(3, 0) = -pre.Jvg();
    L.J_bai.block<3, 3>(3, 0) = -pre.Jva();
    // Position residual: p_i, p_j move with ρ and with φ (p <- p - [p]x φ + ρ).
    L.J_Ti.block<3, 3>(6, 0) = -RiT;
    L.J_Ti.block<3, 3>(6, 3) = RiT * (core::Hat(w) + core::Hat(pi));
    L.J_Tj.block<3, 3>(6, 0) = RiT;
    L.J_Tj.block<3, 3>(6, 3) = -RiT * core::Hat(pj);
    L.J_vi.block<3, 3>(6, 0) = -RiT * T;
    L.J_bgi.block<3, 3>(6, 0) = -pre.Jpg();
    L.J_bai.block<3, 3>(6, 0) = -pre.Jpa();
    return L;
}

}  // namespace ariadne::slam
