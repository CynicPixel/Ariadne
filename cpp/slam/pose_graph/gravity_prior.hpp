#pragma once
// ---------------------------------------------------------------------------
// GRAVITY_PRIOR edge (arch §10.4, §12.1; design inertial.md §4.10): a unary pose-graph factor on a keyframe's
// roll and pitch. Produced by the inertial layer (2.4); consumed by the pose-graph solver (2.5), which this header
// does not touch.
//
// WHAT IT SAYS. "In this keyframe's rectified left camera, gravity points along g_C." With the agent's gravity
// direction ĝ_W in its world, the residual compares the down direction the node's pose predicts with the measured
// one, on the measured direction's tangent plane:
//     r = B(g_C)^T R_WC^T ĝ_W  in R^2,   B = GravityTangentBasis (orthonormal, perpendicular to g_C)
// zero when the camera's tilt agrees and blind to yaw (a rotation about ĝ_W leaves R_WC^T ĝ_W unchanged).
// Literature: GTSAM's AttitudeFactor<Pose3> (navigation/AttitudeFactor.h) states the same constraint on the world
// side, error = nRef.errorVector(nRb * bMeasured); with isotropic information both costs are 1 - (R_WC g_C . ĝ_W)^2.
// The body-side form is used here because the measurement's covariance lives in the body.
//
// JACOBIAN for the left perturbation T_W_C <- Exp([ρ; φ]) T_W_C (Ariadne's convention):
//     R^T ĝ <- R^T Exp(-φ) ĝ ≈ R^T ĝ + R^T [ĝ]x φ   =>   ∂r/∂φ = B(g_C)^T R^T [ĝ_W]x,   ∂r/∂ρ = 0.
// Finite-difference tested (tests/numerical/test_gravity_prior.cpp).
//
// OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "slam/inertial/preintegration.hpp"   // GravityTangentBasis

#include <Eigen/Dense>

#include <cstdint>

namespace ariadne::slam {

struct GravityPrior {
    uint64_t keyframe = 0;
    core::Vec3 g_C = core::Vec3::Zero();       // measured down direction in the keyframe's camera (unit)
    core::Vec3 g_ref_W = core::Vec3::Zero();   // the agent's gravity direction in its world (unit)
    Eigen::Matrix2d information = Eigen::Matrix2d::Zero();   // rad^-2, in GravityTangentBasis(g_C)
};

struct GravityPriorLinearization {
    Eigen::Vector2d r = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 2, 6> J = Eigen::Matrix<double, 2, 6>::Zero();   // [∂r/∂ρ = 0 | ∂r/∂φ]
};

inline GravityPriorLinearization LinearizeGravityPrior(const GravityPrior& p, const core::SE3& T_W_C) {
    GravityPriorLinearization l;
    const Eigen::Matrix<double, 3, 2> B = GravityTangentBasis(p.g_C);
    const core::Mat3 Rt = T_W_C.R.transpose();
    l.r = B.transpose() * Rt * p.g_ref_W;
    l.J.rightCols<3>() = B.transpose() * Rt * core::Hat(p.g_ref_W);
    return l;
}

/// rᵀ Λ r: the edge's chi-square (2 dof), the same convention as PoseGraph::Cost (no ½).
inline double GravityPriorChi2(const GravityPrior& p, const core::SE3& T_W_C) {
    const Eigen::Vector2d r = LinearizeGravityPrior(p, T_W_C).r;
    return r.dot(p.information * r);
}

}  // namespace ariadne::slam
