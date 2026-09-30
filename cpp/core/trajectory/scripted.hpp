#pragma once
// ---------------------------------------------------------------------------
// Scripted agent trajectories: the C++ evaluator of the scripts worldgen writes
// (python/dataset_tools/worldgen/{motion,routes}.py; design SIM_1.1-1.3 §4.2).
//
// ONE implementation, two consumers: the gz kinematic-mover plugin (sim image)
// and the gtest suite (core image), which checks it against golden samples
// produced by the Python generator (tests/fixtures/scripted/). Dependency-free
// (<cmath> only) so the plugin can include it without Eigen.
//
// Semantics, exactly as in Python:
//   Profile: an S-curve moving u from 0 to D. Ramps use
//     v = vc*sigma(tau), sigma = 10 tau^3 - 15 tau^4 + 6 tau^5,
//     u = vc*Ta*Sigma(tau), Sigma = 2.5 tau^4 - 3 tau^5 + tau^6,
//   then a cruise at vc, then the mirror-image ramp. The evaluator never
//   re-plans: (D, Ta, vc) come from the script.
//   Pose = (x, y, z, yaw, pitch), roll 0, R = Rz(yaw) * Ry(pitch) (positive
//   pitch = nose down). hold: rest at start. turn: (yaw, pitch) along a straight
//   line in angle space (end angles stored unwrapped). move: straight 3D line at
//   the start orientation.
//   Body rate:        omega_B = (-yaw' sin(pitch), pitch', yaw' cos(pitch))
//   Specific force:   f_B = Ry(pitch)^T Rz(yaw)^T (a_W + g e_z)
// ---------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::core::scripted {

struct Profile {
    double D = 0.0, Ta = 0.0, vc = 0.0;

    double Tc() const { return vc == 0.0 ? 0.0 : std::max(D / vc - Ta, 0.0); }
    double T() const { return D == 0.0 ? 0.0 : 2.0 * Ta + Tc(); }

    static double Sigma(double t) { return t * t * t * (10.0 + t * (-15.0 + 6.0 * t)); }
    static double DSigma(double t) { return 30.0 * t * t * (1.0 + t * (-2.0 + t)); }
    static double BigSigma(double t) { return t * t * t * t * (2.5 + t * (-3.0 + t)); }

    /// u, du/dt, d2u/dt2 at t (clamped to [0, T]).
    void Eval(double t, double& u, double& du, double& ddu) const {
        if (D == 0.0) {
            u = du = ddu = 0.0;
            return;
        }
        const double T_ = T();
        t = std::min(std::max(t, 0.0), T_);
        if (t < Ta) {
            const double tau = t / Ta;
            u = vc * Ta * BigSigma(tau);
            du = vc * Sigma(tau);
            ddu = vc * DSigma(tau) / Ta;
        } else if (t <= Ta + Tc()) {
            u = 0.5 * vc * Ta + vc * (t - Ta);
            du = vc;
            ddu = 0.0;
        } else {
            const double tau = (T_ - t) / Ta;
            u = D - vc * Ta * BigSigma(tau);
            du = vc * Sigma(tau);
            ddu = -vc * DSigma(tau) / Ta;
        }
    }
};

using Vec5 = std::array<double, 5>;       // (x, y, z, yaw, pitch)

struct State {
    Vec5 pose{};      // x, y, z, yaw, pitch
    Vec5 rate{};      // their first time derivatives
    Vec5 acc{};       // their second time derivatives
};

struct Segment {
    enum class Kind { kHold, kTurn, kMove };
    Kind kind = Kind::kHold;
    double t0 = 0.0, duration = 0.0;
    Profile profile;
    Vec5 start{}, end{};
    std::string tag;

    static Kind ParseKind(const std::string& k) {
        if (k == "hold") return Kind::kHold;
        if (k == "turn") return Kind::kTurn;
        if (k == "move") return Kind::kMove;
        throw std::invalid_argument("scripted: unknown segment kind '" + k + "'");
    }

    State Eval(double t) const {
        State s;
        const double tl = std::min(std::max(t - t0, 0.0), duration);
        if (kind == Kind::kHold) {
            s.pose = start;
            return s;
        }
        double u, du, ddu;
        profile.Eval(tl, u, du, ddu);
        const double k = u / profile.D, kd = du / profile.D, kdd = ddu / profile.D;
        Vec5 d;
        for (int i = 0; i < 5; ++i) d[i] = end[i] - start[i];
        s.pose = start;
        if (kind == Kind::kTurn) {
            for (int i = 3; i < 5; ++i) {
                s.pose[i] = start[i] + k * d[i];
                s.rate[i] = kd * d[i];
                s.acc[i] = kdd * d[i];
            }
        } else {
            for (int i = 0; i < 3; ++i) {
                s.pose[i] = start[i] + k * d[i];
                s.rate[i] = kd * d[i];
                s.acc[i] = kdd * d[i];
            }
        }
        return s;
    }
};

class Script {
public:
    Script() = default;
    explicit Script(std::vector<Segment> segs) : segs_(std::move(segs)) { Index(); }

    const std::vector<Segment>& segments() const { return segs_; }
    double duration() const { return segs_.empty() ? 0.0 : ends_.back(); }

    /// The first segment whose end is >= t (the Python bisect_left choice).
    State Eval(double t) const {
        if (segs_.empty()) throw std::logic_error("scripted: empty script");
        const auto it = std::lower_bound(ends_.begin(), ends_.end(), t);
        const size_t i = std::min<size_t>(size_t(it - ends_.begin()), segs_.size() - 1);
        return segs_[i].Eval(t);
    }

    /// Structural checks: kinds valid, durations consistent with profiles, segments
    /// contiguous in time and in pose (a jump would teleport the rig). Throws.
    void Validate(double tol = 1e-6) const {
        for (size_t i = 0; i < segs_.size(); ++i) {
            const Segment& s = segs_[i];
            if (!(s.duration >= 0.0)) throw std::invalid_argument("scripted: negative duration");
            if (s.kind != Segment::Kind::kHold && std::abs(s.profile.T() - s.duration) > tol) {
                throw std::invalid_argument("scripted: segment duration != profile duration");
            }
            if (i > 0) {
                const Segment& p = segs_[i - 1];
                if (std::abs(p.t0 + p.duration - s.t0) > tol) throw std::invalid_argument("scripted: time gap");
                const State a = p.Eval(p.t0 + p.duration), b = s.Eval(s.t0);
                for (int j = 0; j < 3; ++j) {
                    if (std::abs(a.pose[j] - b.pose[j]) > tol) throw std::invalid_argument("scripted: position jump");
                }
                for (int j = 3; j < 5; ++j) {
                    const double dd = std::remainder(a.pose[j] - b.pose[j], 2.0 * M_PI);
                    if (std::abs(dd) > tol) throw std::invalid_argument("scripted: orientation jump");
                }
            }
        }
    }

private:
    void Index() {
        ends_.clear();
        for (const auto& s : segs_) ends_.push_back(s.t0 + s.duration);
    }
    std::vector<Segment> segs_;
    std::vector<double> ends_;
};

struct ImuSample {
    std::array<double, 3> gyro{};    // rad/s, body frame
    std::array<double, 3> accel{};   // specific force, m/s^2, body frame
};

inline ImuSample Imu(const State& s, double gravity) {
    const double h = s.pose[3], p = s.pose[4], hd = s.rate[3], pd = s.rate[4];
    const double ch = std::cos(h), sh = std::sin(h), cp = std::cos(p), sp = std::sin(p);
    ImuSample m;
    m.gyro = {-hd * sp, pd, hd * cp};
    const double wx = ch * s.acc[0] + sh * s.acc[1];          // Rz^T (a + g e_z)
    const double wy = -sh * s.acc[0] + ch * s.acc[1];
    const double wz = s.acc[2] + gravity;
    m.accel = {cp * wx - sp * wz, wy, sp * wx + cp * wz};     // Ry^T
    return m;
}

/// Unit quaternion (w, x, y, z) of R = Rz(yaw) * Ry(pitch).
inline std::array<double, 4> Quaternion(double yaw, double pitch) {
    const double cy = std::cos(0.5 * yaw), sy = std::sin(0.5 * yaw);
    const double cp = std::cos(0.5 * pitch), sp = std::sin(0.5 * pitch);
    // q = qz(yaw) * qy(pitch) = (cy, 0, 0, sy) * (cp, 0, sp, 0)
    return {cy * cp, -sy * sp, cy * sp, sy * cp};
}

}  // namespace ariadne::core::scripted
