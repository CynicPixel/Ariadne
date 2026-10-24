#pragma once
// A visual-inertial BA window with a known truth (INERTIAL_2.4.md §4.5), shared by the numerical and OpenCV tests:
// a closed-form body trajectory, a camera rigidly mounted with a lever arm (T_C_B), ideal biased IMU samples at
// 1 kHz preintegrated between keyframes 0.25 s apart, and landmarks in front of the cameras observed in stereo.
#include "slam/bundle_adjustment/window_ba.hpp"

#include <cmath>
#include <random>

namespace ariadne::slam::test {

inline const BaCamera kInertialCam{640.0, 640.0, 640.0, 360.0, 0.10};
inline const Vec3 kGravity(0, 0, -9.81);
inline const Vec3 kTrueBg(0.02, -0.01, 0.005), kTrueBa(0.1, -0.05, 0.08);

/// Body pose, velocity and acceleration in the world, closed form; rotation moderate so the scene stays in view.
inline SE3 BodyPose(double t) {
    SE3 T;
    T.R = core::SO3Exp(Vec3(0.15 * std::sin(0.9 * t), 0.2 * t, 0.1 * std::cos(1.3 * t)));
    T.t = Vec3(std::sin(t), 0.5 * std::cos(0.7 * t), 0.3 * t + 0.2 * std::sin(2.0 * t));
    return T;
}
inline Vec3 BodyVelocity(double t) {
    return Vec3(std::cos(t), -0.35 * std::sin(0.7 * t), 0.3 + 0.4 * std::cos(2.0 * t));
}
inline Vec3 BodyAccel(double t) { return Vec3(-std::sin(t), -0.245 * std::cos(0.7 * t), -0.8 * std::sin(2.0 * t)); }
inline Vec3 BodyOmega(double t) {   // body angular velocity, symmetric difference of the closed form (~1e-9)
    const double h = 1e-5;
    return core::SO3Log(BodyPose(t - h).R.transpose() * BodyPose(t + h).R) / (2 * h);
}

/// The camera's mount: T_C_B (body in camera), a small rotation and a lever arm of a few cm.
inline SE3 CameraFromBody() {
    SE3 T;
    T.R = core::SO3Exp(Vec3(0.05, -0.03, 0.02));
    T.t = Vec3(0.05, -0.02, 0.01);
    return T;
}

/// Ideal biased IMU over [t0, t1] at `rate`, each sample the value at its interval's midpoint.
inline std::vector<core::ImuMeasurement> IdealImu(double t0, double t1, double rate) {
    std::vector<core::ImuMeasurement> s;
    const int n = int(std::round((t1 - t0) * rate));
    for (int k = 0; k <= n; ++k) {
        const double t = t0 + k / rate, tm = t + 0.5 / rate;
        s.push_back({int64_t(std::llround(t * 1e9)), BodyOmega(tm) + kTrueBg,
                     BodyPose(tm).R.transpose() * (BodyAccel(tm) - kGravity) + kTrueBa});
    }
    return s;
}

inline double KeyframeTime(int k) { return 1.0 + 0.25 * k; }

/// The true window: K keyframes, M landmarks; window[0] fixed. Pixel noise `pix_noise` on the observations only
/// (the IMU is ideal; its densities below only weight it).
inline BaWindow MakeInertialWindow(std::mt19937& g, int K, int M, double pix_noise) {
    BaWindow w;
    w.cam = kInertialCam;
    w.imu.T_C_B = CameraFromBody();
    w.imu.g_W = kGravity;
    w.imu.gyro_random_walk = 6e-5;
    w.imu.accel_random_walk = 3e-2;
    const SE3 T_B_C = w.imu.T_C_B.inverse();
    const PreintegrationNoise noise{5e-4, 2e-2};
    for (int k = 0; k < K; ++k) {
        const double t = KeyframeTime(k);
        w.T_W_C.push_back(BodyPose(t) * T_B_C);
        w.fixed.push_back(k == 0);
        w.imu.v.push_back(BodyVelocity(t));
        w.imu.bg.push_back(kTrueBg);
        w.imu.ba.push_back(kTrueBa);
        if (k + 1 < K) {
            const auto imu = IdealImu(t, KeyframeTime(k + 1), 1000.0);
            w.imu.pre.push_back(Preintegrate(imu, imu.front().t_ns, imu.back().t_ns, kTrueBg, kTrueBa, noise));
        }
    }
    // Landmarks in front of the middle keyframe's camera; observed wherever they are in front (z > 0.5 m).
    const SE3& mid = w.T_W_C[size_t(K / 2)];
    std::uniform_real_distribution<double> ux(-1.5, 1.5), uz(2.0, 6.0);
    std::normal_distribution<double> pn(0.0, pix_noise);
    for (int m = 0; m < M; ++m) {
        const double z = uz(g);
        const Vec3 X = mid * Vec3{ux(g) * z / 3.0, 0.6 * ux(g) * z / 3.0, z};
        const uint64_t id = uint64_t(100 + m);
        w.X[id] = X;
        for (int k = 0; k < K; ++k)
            for (bool right : {false, true}) {
                Vec3 Xc = w.T_W_C[size_t(k)].R.transpose() * (X - w.T_W_C[size_t(k)].t);
                if (right) Xc.x() -= w.cam.baseline;
                if (Xc.z() < 0.5) continue;
                WindowObservation o;
                o.kf = k;
                o.landmark = id;
                o.right = right;
                o.z = Vec2{w.cam.fx * Xc.x() / Xc.z() + w.cam.cx + pn(g), w.cam.fy * Xc.y() / Xc.z() + w.cam.cy + pn(g)};
                w.obs.push_back(o);
            }
    }
    return w;
}

}  // namespace ariadne::slam::test
