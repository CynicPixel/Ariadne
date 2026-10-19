#pragma once
// ---------------------------------------------------------------------------
// Loop-closure degeneracy gate (arch §14.4, D71; design ROBUST_BACKEND_2.5.md §4). Runs BEFORE PCM (§14.5).
//
// A verified loop A <- B is a PnP problem: B's stereo 3D points, seen as pixels in A, fix T_A_B (p_A = T_A_B p_B).
// When the scene cannot constrain all six degrees of freedom, PnP still returns a pose, and a degenerate edge with an
// honest but huge covariance is consistent with almost anything, so PCM cannot discriminate it. The gate refuses
// such hypotheses before they reach PCM.
//
// What it computes, data in -> data out:
//   1. Refine T_A_B: cv::solvePnPRefineLM (OpenCV's LM, unweighted), then weighted Gauss-Newton steps where each
//      point's pixel covariance includes B's stereo depth noise propagated into A's image (Vakhitov et al., CVPR 2021:
//      3D points are uncertain too; ignoring it makes far points look precise).
//   2. H = sum J_i^T Sigma_i^-1 J_i with J_i = d pixel / d xi for the LEFT perturbation T_A_B <- exp(xi) T_A_B
//      (translation first, Ariadne's one convention: core::ReprojectionJacobians with T_C_W = T_A_B); covariance of T_A_B = H^-1, inflated by the a-posteriori
//      variance factor when the residuals say the noise model is optimistic.
//   3. The edge information in the pose-graph residual tangent, by the SAME conversion OdometryEdge uses
//      (Info_r = (Adj(Z^-1) Sigma_z Adj(Z^-1)^T)^-1, Z = T_A_B, edge from A to B).
//   4. Criteria on that residual-tangent covariance (the matrix the PGO consumes):
//      - conditioning: worst-direction 1-sigma of the translation marginal [m] and of the rotation marginal [rad],
//        each against an absolute bound. Marginals (blocks of the covariance = Schur complements of H) keep the
//        rotation/translation coupling, where visual degeneracy lives (pan vs lateral translation); metres and
//        radians never share one eigenvalue ratio. The within-block ratios lambda_min/lambda_max are reported (arch
//        §14.4's ratio criterion, made unit-consistent).
//      - parallax: too few inliers with a stereo triangulation angle >= the bound (depth too poor to fix
//        translation);
//      - collinearity: the inliers' 3D points lie on a line (lambda_mid/lambda_max of their covariance below the
//        bound): rotation about the line is unobservable, and stereo noise scattering the points off the line
//        fabricates the missing information, so conditioning alone misses it (measured, design §4). Collinear
//        PIXELS are not the criterion: a planar fan through A's centre is well conditioned with known depth.
//      Pixel spread, planarity and a corridor score are REPORTED, not gated (planar PnP with known 3D points is not
//      degenerate).
// The criteria are physically coupled (far-only points usually also fail conditioning); every failing one is named.
// Thresholds are config, [ASSUMPTION] until calibrated on real candidates (design §4 protocol, D73).
// ---------------------------------------------------------------------------
#include "core/geometry/pinhole.hpp"
#include "core/math/lie.hpp"

#include <string>
#include <vector>

namespace ariadne::slam {

using core::Mat6;
using core::SE3;

struct LoopGeometry {
    core::RectifiedStereo rig;               // shared by A and B; B's 3D points came from its disparity
    std::vector<core::Vec3> X_B;             // inlier 3D points in B's rectified-left camera frame
    std::vector<core::Vec2> uv_A;            // their pixels in A
    SE3 T_A_B;                               // initial estimate (e.g. the verifier's EPnP RANSAC pose)
};

struct DegeneracyParams {
    double sigma_px = 0.46;             // [ASSUMPTION] the 2.3 EuRoC reprojection sigma, carried to the sim
    double sigma_disparity_px = 0.65;   // [ASSUMPTION] sqrt(2) * sigma_px: left and right detections independent
    int gauss_newton_steps = 5;
    double max_sigma_t_m = 0.10;        // [ASSUMPTION] worst-direction 1-sigma, translation marginal
    double max_sigma_r_rad = 0.02;      // [ASSUMPTION] worst-direction 1-sigma, rotation marginal (~1.1 deg)
    double min_triangulation_deg = 1.0; // [ASSUMPTION] stereo angle 2 atan(b / 2z) counted as "near"
    int min_near_points = 10;           // [ASSUMPTION]
    double min_line_spread_ratio = 0.01;    // [ASSUMPTION] lambda_mid/lambda_max of the 3D inliers' covariance
};

struct DegeneracyReport {
    bool degenerate = true;
    std::vector<std::string> reasons;   // "too_few_points", "conditioning_translation", "conditioning_rotation",
                                        // "parallax", "collinear"; empty iff !degenerate
    int points = 0;
    SE3 T_A_B;                          // refined
    Mat6 cov_left = Mat6::Zero();       // of T_A_B, left tangent
    Mat6 information = Mat6::Zero();    // pose-graph edge A -> B, residual tangent (PoseEdge::information)
    double sigma2_hat = 0;              // chi^2 / (2N - 6) at the refined pose (1 = noise model right)
    double rms_px = 0;
    double sigma_t_worst = 0, sigma_r_worst = 0;   // [m], [rad]
    core::Vec3 worst_t_dir = core::Vec3::Zero();   // B's frame (residual tangent)
    core::Vec3 worst_r_axis = core::Vec3::Zero();
    double ratio_t = 0, ratio_r = 0;    // lambda_min/lambda_max within each marginal block (reported)
    int near_points = 0;
    double median_triangulation_deg = 0;
    double line_spread_ratio = 0;       // gated: lambda_mid/lambda_max of the 3D inliers' covariance (0 = a line)
    double pixel_spread_ratio = 0;      // reported: lambda_min/lambda_max of A's inlier pixel covariance
    double planarity = 0;               // reported: lambda_min/lambda_max of the 3D inliers' covariance
    double corridor = 0;                // reported: lambda_max / trace of the 3D inliers' covariance (1 = a line)
};

/// d X_B / d (u_B, v_B, disparity) of stereo back-projection z = fx b / d at B's own pixel; 3x3. Exposed for the FD
/// test.
core::Mat3 StereoPointJacobian(const core::RectifiedStereo& rig, const core::Vec3& X_B);

/// Pixel covariance in A of one point: detection noise plus B's stereo noise (u, v, disparity) carried through.
Eigen::Matrix2d PixelCovariance(const LoopGeometry& g, const SE3& T_A_B, const core::Vec3& X_B, const DegeneracyParams& p);

DegeneracyReport AssessLoop(const LoopGeometry& g, const DegeneracyParams& p = {});

}  // namespace ariadne::slam
