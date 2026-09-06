#pragma once
// ---------------------------------------------------------------------------
// Deterministic synthetic trajectories (arch §33 Iteration 0).
//
// Exists so the Phase 0 exit gate can run before any simulator does: a
// trajectory that is generated, serialized, replayed and scored, with a known
// ground truth to score against. Every value derives from (seed) through the
// counter-based generator, so a trajectory is reproducible on any machine.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "core/random/distributions.hpp"
#include "core/random/philox.hpp"
#include "core/time/timestamp.hpp"

#include <cstdint>
#include <vector>

namespace ariadne::core {

struct TrajectoryParams {
    int keyframes = 200;
    /// Interval between keyframes. Integer nanoseconds, so timestamps stay exact.
    Duration interval = Duration::Milliseconds(50);

    // A Lissajous path: non-repeating in general, self-intersecting in
    // projection, and smooth. Self-intersection matters because it produces
    // genuine revisits, which is what loop closure and the aliasing work need.
    double radius_x = 6.0;
    double radius_y = 4.0;
    double radius_z = 1.2;
    double freq_x = 1.0;
    double freq_y = 2.0;
    double freq_z = 3.0;

    /// Odometry noise applied to the ESTIMATE, per step, in the body frame.
    /// Drift accumulates because each step's error compounds through the chain,
    /// which is the behaviour the metrics need to be able to see.
    double translation_noise_m = 0.01;
    double rotation_noise_rad = 0.002;
};

/// One pose on a trajectory.
struct TrajectorySample {
    Timestamp t;
    uint64_t keyframe = 0;
    SE3 pose;
};

/// Ground-truth trajectory. A closed-form function of the parameters, with no
/// randomness at all: the same params always give the same path.
inline std::vector<TrajectorySample> GenerateGroundTruth(
    const TrajectoryParams& p) {
    std::vector<TrajectorySample> out;
    out.reserve(p.keyframes);
    for (int i = 0; i < p.keyframes; ++i) {
        const double u = static_cast<double>(i) /
                         static_cast<double>(p.keyframes > 1 ? p.keyframes - 1 : 1);
        const double theta = 2.0 * M_PI * u;

        TrajectorySample s;
        s.keyframe = static_cast<uint64_t>(i);
        s.t = Timestamp::Nanoseconds(p.interval.ns * i);

        // Position on the Lissajous curve. std::sin/cos are acceptable here
        // because this is test-fixture generation, not the degradation
        // operator: the trajectory is regenerated identically within a run and
        // is not required to be bit-identical across architectures.
        const double x = p.radius_x * std::sin(p.freq_x * theta);
        const double y = p.radius_y * std::sin(p.freq_y * theta);
        const double z = p.radius_z * std::sin(p.freq_z * theta);
        s.pose.t = Vec3(x, y, z);

        // Orientation: yaw follows the path tangent, so the camera looks where
        // it is going, with a small roll so all three axes are exercised.
        const double dx = p.radius_x * p.freq_x * std::cos(p.freq_x * theta);
        const double dy = p.radius_y * p.freq_y * std::cos(p.freq_y * theta);
        const double yaw = std::atan2(dy, dx);
        const double roll = 0.15 * std::sin(3.0 * theta);
        Vec3 rotvec(roll, 0.0, yaw);
        s.pose.R = SO3Exp(rotvec);

        out.push_back(s);
    }
    return out;
}

/// A noisy ESTIMATE of the ground truth, built by corrupting each relative
/// motion and re-integrating. Errors compound along the chain, which is what
/// produces realistic drift rather than independent per-pose jitter.
///
/// The first pose is left exact so the trajectories share an origin; alignment
/// is still applied when scoring, because a real estimator has no such anchor.
inline std::vector<TrajectorySample> GenerateNoisyEstimate(
    const std::vector<TrajectorySample>& gt, const TrajectoryParams& p,
    uint64_t seed) {
    std::vector<TrajectorySample> out;
    if (gt.empty()) return out;
    out.reserve(gt.size());

    SE3 accumulated = gt.front().pose;
    out.push_back({gt.front().t, gt.front().keyframe, accumulated});

    for (size_t i = 1; i < gt.size(); ++i) {
        // True relative motion between consecutive ground-truth poses.
        const SE3 relative = gt[i - 1].pose.inverse() * gt[i].pose;

        // Corrupt it in the body frame. Site index is the step, so step i's
        // perturbation does not depend on how many steps preceded it.
        PhiloxStream stream(seed, 0, i);
        Vec6 xi;
        xi(0) = p.translation_noise_m * SampleStandardNormal(stream);
        xi(1) = p.translation_noise_m * SampleStandardNormal(stream);
        xi(2) = p.translation_noise_m * SampleStandardNormal(stream);
        xi(3) = p.rotation_noise_rad * SampleStandardNormal(stream);
        xi(4) = p.rotation_noise_rad * SampleStandardNormal(stream);
        xi(5) = p.rotation_noise_rad * SampleStandardNormal(stream);

        accumulated = accumulated * (relative * SE3Exp(xi));
        accumulated.normalize();   // long chains drift off SO(3)
        out.push_back({gt[i].t, gt[i].keyframe, accumulated});
    }
    return out;
}

}  // namespace ariadne::core
