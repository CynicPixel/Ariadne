#pragma once
// ---------------------------------------------------------------------------
// IMU measurements and the gyro rotation integral (arch §10, plan 2.4).
//
// The IMU frame IS the body frame B (EuRoC imu0 T_BS = identity; the sim's body is its IMU). A gyro sample is the
// body's angular velocity in B (rad/s); an accel sample is the specific force in B (m/s^2): a = R_WB^T (a_W - g_W).
// OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ariadne::core {

struct ImuMeasurement {
    int64_t t_ns = 0;
    Vec3 gyro = Vec3::Zero();    // rad/s, body frame
    Vec3 accel = Vec3::Zero();   // m/s^2, specific force, body frame
};

/// Continuous-time noise densities (Kalibr / EuRoC sensor.yaml convention). The discrete per-sample std of white
/// noise at rate f is density * sqrt(f); a bias random walk adds density * sqrt(dt) per step.
struct ImuNoise {
    double gyro_noise_density = 0.0;     // rad / s / sqrt(Hz)
    double gyro_random_walk = 0.0;       // rad / s^2 / sqrt(Hz)
    double accel_noise_density = 0.0;    // m / s^2 / sqrt(Hz)
    double accel_random_walk = 0.0;      // m / s^3 / sqrt(Hz)
    double rate_hz = 0.0;
};

/// R_B(t0)_B(t1): the body rotation from t0 to t1, integrating (gyro - bias) piecewise-constant over the samples
/// (zero-order hold on the sample at the START of each interval, the Forster et al. 2017 convention), with the
/// first and last intervals clipped to [t0, t1]. `samples` must be time-sorted and cover [t0, t1]; throws otherwise.
inline Mat3 IntegrateGyro(const std::vector<ImuMeasurement>& samples, int64_t t0, int64_t t1, const Vec3& bias) {
    if (t1 < t0) throw std::invalid_argument("IntegrateGyro: t1 < t0");
    if (samples.empty() || samples.front().t_ns > t0 || samples.back().t_ns < t1) {
        throw std::invalid_argument("IntegrateGyro: samples do not cover [t0, t1]");
    }
    Mat3 R = Mat3::Identity();
    for (size_t k = 0; k + 1 < samples.size(); ++k) {
        const int64_t a = std::max(samples[k].t_ns, t0), b = std::min(samples[k + 1].t_ns, t1);
        if (b <= a) continue;
        R = R * SO3Exp((samples[k].gyro - bias) * (1e-9 * double(b - a)));
    }
    return R;
}

}  // namespace ariadne::core
