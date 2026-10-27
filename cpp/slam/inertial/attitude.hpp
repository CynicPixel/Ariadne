#pragma once
// ---------------------------------------------------------------------------
// imu_mode attitude_only (arch §10.7, plan 2.4 step 1): the inertial products that need no velocity or accel-bias
// states.
//
//   rotation prediction   the camera's rotation between two frames from the bias-corrected gyro integral. While
//                         tracking is lost it replaces the visual rotation bridge (D56); every frame it seeds LK
//                         (§10.5, Hwangbo et al. IROS 2009).
//   gravity direction     Mahony's explicit complementary filter (IEEE TAC 2008, Theorem 5.1), the §10.4 producer.
//
// GYRO BIAS. Validated on EuRoC (2.4 step 2, all 11 flights, 26,530 tracked frame pairs): the raw gyro integral is
// off by a median 0.23 deg per pair; with the ground-truth bias removed, by 0.019 deg. The bias dominates, and
// Mahony's filter cannot see it about the gravity axis (an accelerometer does not observe yaw). So it is estimated
// from VO: on every pair of consecutive TRACKED frames, (Log(R_gyro,raw) - Log(R_vo)) / dt is a sample of the bias
// (first order, both rotations small at 20 Hz); the estimate is the per-axis median over the last `bias_window`
// samples. Before any sample: the mean gyro if the first `stationary_s` seconds were at rest, else zero.
//
// Pure: the caller (runtime) owns the IMU stream and the clock and hands in what it needs. OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/inertial/imu.hpp"
#include "core/math/lie.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

using core::Mat3;
using core::Vec3;

struct AttitudeParams {
    int bias_window = 200;          // tracked pairs in the bias median (10 s at 20 Hz)
    double stationary_s = 2.0;      // local-slam.md#estimator-plan §3: the first 2 s decide whether the start is at rest
    double kp = 1.0;                // Mahony et al. 2008 §VI: kP = 1 rad/s (filter crossover)
    double ki = 0.3;                // kI = 0.3 rad/s
};

/// Robust running gyro bias from VO rotations (see header).
class GyroBiasEstimator {
public:
    explicit GyroBiasEstimator(int window) : window_(window) {
        if (window < 1) throw std::invalid_argument("GyroBiasEstimator: window >= 1");
    }
    void Seed(const Vec3& b) { seed_ = b; }
    /// R_gyro_raw: the gyro integral with ZERO bias over [t0, t1], body frame; R_vo: the VO's body rotation.
    void Add(const Mat3& R_gyro_raw, const Mat3& R_vo, double dt) {
        if (!(dt > 0.0)) return;
        samples_.push_back((core::SO3Log(R_gyro_raw) - core::SO3Log(R_vo)) / dt);
        if (int(samples_.size()) > window_) samples_.pop_front();
    }
    Vec3 Bias() const {
        if (samples_.empty()) return seed_;
        Vec3 b;
        for (int a = 0; a < 3; ++a) {
            std::vector<double> v;
            for (const Vec3& s : samples_) v.push_back(s[a]);
            std::nth_element(v.begin(), v.begin() + long(v.size() / 2), v.end());
            b[a] = v[v.size() / 2];
        }
        return b;
    }
    size_t Count() const { return samples_.size(); }

private:
    int window_;
    Vec3 seed_ = Vec3::Zero();
    std::deque<Vec3> samples_;
};

/// The stationary test of local-slam.md#estimator-plan §3: over the first `seconds`, every gyro axis's std below 3x the white-noise
/// std at the sample rate (Kalibr density * sqrt(rate)). Returns the mean gyro if at rest.
inline bool StationaryGyroMean(const std::vector<core::ImuMeasurement>& imu, const core::ImuNoise& noise, double seconds,
                               Vec3& mean) {
    if (imu.empty() || !(noise.rate_hz > 0.0)) return false;
    const int64_t end = imu.front().t_ns + int64_t(seconds * 1e9);
    Vec3 s = Vec3::Zero(), s2 = Vec3::Zero();
    int n = 0;
    for (const auto& m : imu) {
        if (m.t_ns > end) break;
        s += m.gyro;
        s2 += m.gyro.cwiseProduct(m.gyro);
        ++n;
    }
    if (n < 10) return false;
    mean = s / n;
    const Vec3 var = s2 / n - mean.cwiseProduct(mean);
    const double limit = 3.0 * noise.gyro_noise_density * std::sqrt(noise.rate_hz);
    return var.maxCoeff() <= limit * limit;
}

/// Mahony et al., IEEE TAC 2008, Theorem 5.1 (explicit complementary filter with bias correction), one vector
/// measurement: the accelerometer direction v = a / |a| against the predicted "up" in the body, v_hat = R^T e3 (world
/// z up, g_W = (0, 0, -9.81)). omega_mes = k1 v x v_hat; R <- R exp((omega_y - b + kP omega_mes) dt);
/// b <- b - kI omega_mes dt. Roll and pitch converge; yaw stays where it started (unobservable from gravity).
class MahonyFilter {
public:
    MahonyFilter(double kp, double ki) : kp_(kp), ki_(ki) {}
    /// Initialise roll/pitch from one accelerometer reading at rest (yaw = 0, arch §10.3).
    void Init(const Vec3& accel, const Vec3& bias) {
        const Vec3 up = accel.normalized();                  // body-frame "up" at rest
        const Vec3 z(0, 0, 1);
        const Vec3 axis = up.cross(z);                       // rotate body-up onto world z
        const double s = axis.norm(), c = up.dot(z);
        R_ = s < 1e-12 ? (c > 0 ? Mat3::Identity() : Mat3(core::SO3Exp(Vec3(M_PI, 0, 0))))
                       : Mat3(core::SO3Exp(axis / s * std::atan2(s, c)));
        b_ = bias;
        init_ = true;
    }
    void Update(const Vec3& gyro, const Vec3& accel, double dt) {
        if (!init_) throw std::logic_error("MahonyFilter: Init first");
        Vec3 w_mes = Vec3::Zero();
        if (accel.norm() > 1e-6) w_mes = accel.normalized().cross(R_.transpose() * Vec3(0, 0, 1));
        R_ = R_ * core::SO3Exp((gyro - b_ + kp_ * w_mes) * dt);
        b_ -= ki_ * w_mes * dt;
    }
    const Mat3& R_WB() const { return R_; }
    /// Gravity's direction in the body frame (unit, pointing DOWN).
    Vec3 GravityInBody() const { return R_.transpose() * Vec3(0, 0, -1); }
    bool initialised() const { return init_; }

private:
    double kp_, ki_;
    Mat3 R_ = Mat3::Identity();
    Vec3 b_ = Vec3::Zero();
    bool init_ = false;
};

/// attitude_only glue: owns the bias estimator and the Mahony filter; the caller feeds IMU samples and VO rotations.
class AttitudeTracker {
public:
    AttitudeTracker(const AttitudeParams& p, const std::vector<core::ImuMeasurement>& imu, const core::ImuNoise& noise,
                    const Mat3& R_BC)
        : p_(p), imu_(imu), R_BC_(R_BC), bias_(p.bias_window), mahony_(p.kp, p.ki) {
        Vec3 mean;
        stationary_start_ = StationaryGyroMean(imu_, noise, p.stationary_s, mean);
        if (stationary_start_) bias_.Seed(mean);
        if (!imu_.empty()) mahony_.Init(imu_.front().accel, stationary_start_ ? mean : Vec3::Zero());
    }
    bool Covers(int64_t t0, int64_t t1) const {
        return !imu_.empty() && imu_.front().t_ns <= t0 && imu_.back().t_ns >= t1 && t1 >= t0;
    }
    /// The camera's rotation from t0 to t1 (R_C(t0)_C(t1)), bias-corrected. Caller checks Covers().
    Mat3 PredictCameraRotation(int64_t t0, int64_t t1) const {
        return R_BC_.transpose() * core::IntegrateGyro(imu_, t0, t1, bias_.Bias()) * R_BC_;
    }
    /// A pair of consecutive tracked frames: the VO camera rotation R_C(t0)_C(t1) updates the bias.
    void ObserveTracked(int64_t t0, int64_t t1, const Mat3& R_vo_camera) {
        if (!Covers(t0, t1)) return;
        bias_.Add(core::IntegrateGyro(imu_, t0, t1, Vec3::Zero()), R_BC_ * R_vo_camera * R_BC_.transpose(),
                  1e-9 * double(t1 - t0));
    }
    /// Advance the gravity filter through every IMU sample up to t (monotone).
    void AdvanceTo(int64_t t) {
        while (next_ + 1 < imu_.size() && imu_[next_ + 1].t_ns <= t) {
            mahony_.Update(imu_[next_].gyro, imu_[next_].accel, 1e-9 * double(imu_[next_ + 1].t_ns - imu_[next_].t_ns));
            ++next_;
        }
    }
    /// Gravity's direction (down) in the rectified left camera at the last AdvanceTo time.
    Vec3 GravityInCamera() const { return R_BC_.transpose() * mahony_.GravityInBody(); }
    Vec3 Bias() const { return bias_.Bias(); }
    size_t BiasSamples() const { return bias_.Count(); }
    bool StationaryStart() const { return stationary_start_; }

private:
    AttitudeParams p_;
    const std::vector<core::ImuMeasurement>& imu_;
    Mat3 R_BC_;
    GyroBiasEstimator bias_;
    MahonyFilter mahony_;
    size_t next_ = 0;
    bool stationary_start_ = false;
};

}  // namespace ariadne::slam
