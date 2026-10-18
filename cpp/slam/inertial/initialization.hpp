#pragma once
// ---------------------------------------------------------------------------
// Visual-inertial initialisation (INERTIAL_2.4.md §4.3, D59): from N keyframes whose poses the stereo window has
// already solved (metric: stereo gives scale), estimate what the IMU factor needs and vision cannot give -- the gyro
// bias, every keyframe's velocity, gravity in the VO world, and the accelerometer bias.
//
// WHY THIS ORDER. With the poses held, the problem separates:
//   1. Gyro bias b_g. Only the rotation residual depends on it, r_R = Log(ΔR(b_g)^T R_i^T R_j): a small nonlinear
//      least squares (Gauss-Newton, the preintegrator's bias Jacobian), as VINS-Mono's solveGyroscopeBias (Qin et al.,
//      T-RO 2018, §V-A) but weighted by the propagated rotation covariance. Then the intervals are re-integrated at
//      that b_g, so step 2 does not see its first-order error.
//   2. v_0..v_{N-1}, g, b_a. The velocity and position residuals are LINEAR in them once the poses are fixed:
//          R_i^T (v_j - v_i - g T)            = Δv + J_va b_a
//          R_i^T (-v_i T - ½ g T²)            = Δp + J_pa b_a - R_i^T (p_j - p_i)
//      weighted by the preintegration information (Δv, Δp block), plus a loose zero-mean prior on b_a (arch §10.3: it
//      is weakly observable over a few seconds and trades off against gravity's direction). This is the MAP of
//      ORB-SLAM3's inertial-only optimisation (Campos et al., T-RO 2021, §V-B) with the scale known, which makes it
//      linear and closed form.
//   3. Gravity's norm is known (9.81): the unconstrained |g| from step 2 is a CHECK (refused outside tolerance:
//      wrong frames, bad poses, too little excitation), then g is re-solved on the sphere, g = |g|(ĝ + B w) with B a
//      tangent basis at ĝ, a few iterations (VINS-Mono's RefineGravity). The fit's chi2/dof is the second check: a
//      scale or frame error moves |g| only by (motion acceleration / 9.81) but leaves accelerations a constant g
//      cannot explain (tests/numerical/test_inertial_init.cpp: 1.5x scale, |g| +8%, chi2/dof x1000).
//
// OBSERVABILITY. b_a trades off against gravity's tilt (a rotation of g by θ looks like a bias of 9.81 θ) until the
// body has rotated enough; over ~2 s of mild motion the zero-mean b_a prior decides that direction. The window BA
// keeps refining b_a afterwards (its bias prior is inflated, INERTIAL_2.4 §4.2).
// Refusals are returned (why), never thrown: the caller retries at the next keyframe.
//
// The poses are BODY poses (T_W_B = T_W_C T_C_B). OpenCV-free.
// ---------------------------------------------------------------------------
#include "slam/inertial/preintegration.hpp"

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <vector>

namespace ariadne::slam {

struct InertialInitParams {
    double gravity_norm = 9.81;            // m/s^2
    double gravity_norm_tolerance = 0.1;   // relative: | |g| - 9.81 | / 9.81 above this refuses
    double accel_bias_sigma = 0.2;         // m/s^2, the b_a prior (zero mean)
    double max_condition = 1e10;           // step-2 normal matrix condition number above this refuses
    double max_chi2_per_dof = std::numeric_limits<double>::infinity();   // step-2 fit worse than this refuses
    int gyro_iterations = 5;
    int gravity_iterations = 4;
};

struct InertialInitResult {
    bool ok = false;
    std::string why;                       // when !ok
    Vec3 bg = Vec3::Zero(), ba = Vec3::Zero(), g_W = Vec3::Zero();
    std::vector<Vec3> v;                   // per keyframe, body velocity in W
    Eigen::Matrix<double, 6, 6> bias_cov = Eigen::Matrix<double, 6, 6>::Zero();   // [bg; ba] marginal of the solves
    double gravity_norm_unconstrained = 0.0;
    double condition = 0.0;                // of step 2's normal matrix
    double chi2_per_dof = 0.0;             // step 2's whitened residual at the solution, per degree of freedom
    double rotation_rms_deg = 0.0;         // step 1's rotation residual at b_g (diagnostic)
};

/// T_W_B[k], t_ns[k]: N >= 3 keyframes; imu[k]: samples covering [t_ns[k], t_ns[k+1]] (N - 1 intervals).
inline InertialInitResult InitializeInertial(const std::vector<core::SE3>& T_W_B, const std::vector<int64_t>& t_ns,
                                             const std::vector<std::vector<core::ImuMeasurement>>& imu,
                                             const PreintegrationNoise& noise, const InertialInitParams& p) {
    InertialInitResult out;
    const int N = int(T_W_B.size());
    if (N < 3 || int(t_ns.size()) != N || int(imu.size()) != N - 1)
        throw std::invalid_argument("InitializeInertial: need N >= 3 poses, N times and N - 1 sample intervals");
    const auto integrate = [&](const Vec3& bg) {
        std::vector<Preintegrated> pre;
        for (int k = 0; k + 1 < N; ++k)
            pre.push_back(Preintegrate(imu[size_t(k)], t_ns[size_t(k)], t_ns[size_t(k + 1)], bg, Vec3::Zero(), noise));
        return pre;
    };

    // --- 1. Gyro bias ---------------------------------------------------------------------------------------------
    std::vector<Preintegrated> pre = integrate(Vec3::Zero());
    Vec3 bg = Vec3::Zero();
    Mat3 Hg = Mat3::Zero();
    for (int it = 0; it < p.gyro_iterations; ++it) {
        Hg.setZero();
        Vec3 gg = Vec3::Zero();
        double sq = 0.0;
        for (int k = 0; k + 1 < N; ++k) {
            const InertialLinearization l = LinearizeInertial(pre[size_t(k)], T_W_B[size_t(k)], Vec3::Zero(), bg,
                                                              Vec3::Zero(), T_W_B[size_t(k + 1)], Vec3::Zero(), Vec3::Zero());
            const Mat3 Om = pre[size_t(k)].Sigma().topLeftCorner<3, 3>().inverse();
            const Mat3 J = l.J_bgi.topRows<3>();
            const Vec3 r = l.r.head<3>();
            Hg += J.transpose() * Om * J;
            gg += J.transpose() * Om * r;
            sq += r.squaredNorm();
        }
        const Vec3 d = Hg.ldlt().solve(-gg);
        bg += d;
        out.rotation_rms_deg = std::sqrt(sq / (N - 1)) * 180.0 / M_PI;
        if (d.norm() < 1e-9) break;
    }
    out.bg = bg;
    pre = integrate(bg);   // step 2 at the estimated b_g

    // --- 2./3. Velocities, gravity, accelerometer bias ---------------------------------------------------------------
    // Unknowns x = [v_0 .. v_{N-1} (3N) | gravity parameters (3, or 2 on the sphere) | b_a (3)].
    const auto solve = [&](bool on_sphere, const Vec3& g_dir, Eigen::VectorXd& x, Eigen::MatrixXd& H) {
        const int ng = on_sphere ? 2 : 3, n = 3 * N + ng + 3;
        H = Eigen::MatrixXd::Zero(n, n);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        double yOy = 0.0;   // y^T Ω y, for the residual at the solution: |r|^2_Ω = y^T Ω y - x^T b (normal equations)
        // Tangent basis at g_dir (on the sphere: g = |g| g_dir + B w, linear in w).
        Eigen::Matrix<double, 3, 2> B;
        {
            const Vec3 a = std::abs(g_dir.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
            const Vec3 b1 = (a - g_dir * g_dir.dot(a)).normalized();
            B << b1, g_dir.cross(b1);
        }
        const Vec3 g0 = on_sphere ? Vec3(p.gravity_norm * g_dir) : Vec3::Zero();
        const Eigen::Matrix<double, 3, Eigen::Dynamic> G =
            on_sphere ? Eigen::Matrix<double, 3, Eigen::Dynamic>(B) : Eigen::Matrix<double, 3, Eigen::Dynamic>(Mat3::Identity());
        for (int k = 0; k + 1 < N; ++k) {
            const Preintegrated& q = pre[size_t(k)];
            const Mat3 RiT = T_W_B[size_t(k)].R.transpose();
            const double T = q.dt();
            Eigen::Matrix<double, 6, Eigen::Dynamic> A = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, n);
            Eigen::Matrix<double, 6, 1> y;
            // Velocity rows.
            A.block<3, 3>(0, 3 * k) = -RiT;
            A.block<3, 3>(0, 3 * (k + 1)) = RiT;
            A.block(0, 3 * N, 3, ng) = -RiT * T * G;
            A.block<3, 3>(0, 3 * N + ng) = -q.Jva();
            y.head<3>() = q.DeltaV(bg, Vec3::Zero()) + RiT * g0 * T;
            // Position rows.
            A.block<3, 3>(3, 3 * k) = -RiT * T;
            A.block(3, 3 * N, 3, ng) = -0.5 * RiT * T * T * G;
            A.block<3, 3>(3, 3 * N + ng) = -q.Jpa();
            y.tail<3>() = q.DeltaP(bg, Vec3::Zero()) - RiT * (T_W_B[size_t(k + 1)].t - T_W_B[size_t(k)].t) +
                          0.5 * RiT * g0 * T * T;
            const Eigen::Matrix<double, 6, 6> Om = InformationOf<6>(q.Sigma().bottomRightCorner<6, 6>());
            H += A.transpose() * Om * A;
            b += A.transpose() * Om * y;
            yOy += y.dot(Om * y);
        }
        H.block<3, 3>(3 * N + ng, 3 * N + ng) += Mat3::Identity() / (p.accel_bias_sigma * p.accel_bias_sigma);
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H, Eigen::EigenvaluesOnly);
        out.condition = es.eigenvalues()(n - 1) / std::max(es.eigenvalues()(0), 1e-300);
        x = H.ldlt().solve(b);
        // The minimum of |A x - y|^2_Ω + |b_a|^2 / sigma^2 is y^T Ω y - x^T b (the prior's rows have zero right-hand
        // side, so it is already in H and not in b); dof = 6 (N - 1) + 3 rows - n unknowns.
        out.chi2_per_dof = std::max(yOy - x.dot(b), 0.0) / std::max(6 * (N - 1) + 3 - n, 1);
        return B;
    };
    Eigen::VectorXd x;
    Eigen::MatrixXd H;
    solve(false, Vec3::UnitZ(), x, H);
    if (!(out.condition < p.max_condition)) {
        out.why = "step-2 system ill-conditioned (condition " + std::to_string(out.condition) + ")";
        return out;
    }
    Vec3 g = x.segment<3>(3 * N);
    out.gravity_norm_unconstrained = g.norm();
    if (!(std::abs(g.norm() - p.gravity_norm) <= p.gravity_norm_tolerance * p.gravity_norm)) {
        out.why = "|g| " + std::to_string(g.norm()) + " outside tolerance";
        return out;
    }
    Vec3 g_dir = g.normalized();
    for (int it = 0; it < p.gravity_iterations; ++it) {
        const Eigen::Matrix<double, 3, 2> B = solve(true, g_dir, x, H);
        g_dir = (p.gravity_norm * g_dir + B * x.segment<2>(3 * N)).normalized();
    }
    if (!(out.chi2_per_dof <= p.max_chi2_per_dof)) {
        out.why = "chi2/dof " + std::to_string(out.chi2_per_dof) + " above " + std::to_string(p.max_chi2_per_dof);
        return out;
    }
    out.g_W = p.gravity_norm * g_dir;
    out.ba = x.segment<3>(3 * N + 2);
    for (int k = 0; k < N; ++k) out.v.push_back(x.segment<3>(3 * k));
    // Bias marginals: b_g from step 1's information, b_a from step 2's (its block of H^-1).
    const Eigen::MatrixXd C = H.ldlt().solve(Eigen::MatrixXd::Identity(H.rows(), H.cols()));
    out.bias_cov.topLeftCorner<3, 3>() = Hg.inverse();
    out.bias_cov.bottomRightCorner<3, 3>() = C.block<3, 3>(3 * N + 2, 3 * N + 2);
    out.ok = true;
    return out;
}

}  // namespace ariadne::slam
