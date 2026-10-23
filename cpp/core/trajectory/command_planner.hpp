#pragma once
// ---------------------------------------------------------------------------
// Commanded motion -> script segments (design LIVE_RUNTIME_L1 D151; arch §5.8).
//
// The live kinematic mover turns goto(pose) / follow(path) into the SAME segment
// family the worldgen scripts use (python/dataset_tools/worldgen/{motion,routes}.py):
// per waypoint, a turn in place to face it, then a straight move; after the last
// waypoint, an optional turn to the commanded final yaw. Profiles are motion.py's
// plan_profile; the cruise speed is routes.py's move_to rule,
//   v = min(v_cap, v_lk * clearance_along(move), v_limit),
// so a commanded path keeps the per-frame image motion inside the tracker's budget
// exactly as a scripted one does. hold() is no segment at all: the mover simply
// stops appending (see the plugin).
//
// Dependency-free (<cmath>, scripted.hpp) so the gz plugin can include it; the
// core suite tests it (tests/numerical/test_command_planner.cpp).
// ---------------------------------------------------------------------------
#include "core/trajectory/scripted.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::core::scripted {

/// motion.py plan_profile: the fastest profile of the family within v_max and a_max.
/// Long moves ramp over Ta = 1.875 v_max / a_max and cruise at v_max; short ones have
/// no cruise and are re-timed so the peak acceleration is exactly a_max.
inline Profile PlanProfile(double D, double v_max, double a_max) {
    if (!(D >= 0) || !(v_max > 0) || !(a_max > 0)) throw std::invalid_argument("PlanProfile: D >= 0, v_max, a_max > 0");
    if (D == 0) return Profile{0.0, 0.0, 0.0};
    const double Ta = 1.875 * v_max / a_max;
    if (D >= v_max * Ta) return Profile{D, Ta, v_max};
    const double vc = std::sqrt(a_max * D / 1.875);
    return Profile{D, D / vc, vc};
}

/// Python's (a + pi) % (2 pi) - pi (floored modulo), in [-pi, pi).
inline double WrapPi(double a) { return a - 2.0 * M_PI * std::floor((a + M_PI) / (2.0 * M_PI)); }

/// The worldgen Motion preset values the mover is limited by (params.py Motion).
struct MotionLimits {
    double v_cap = 0.6;          // m/s
    double a_max = 0.4;          // m/s^2
    double omega_max = 0.44;     // rad/s, already capped by the image-edge flow budget (routes.py w_max)
    double alpha_max = 0.6;      // rad/s^2
    double v_lk_per_m = 0.9375;  // (m/s) per metre of clearance: lk_budget_px / (focal_px * frame_dt_s)
};

struct Waypoint {
    double x = 0, y = 0, z = 0;
    bool has_yaw = false;   // the final waypoint's yaw is honoured; intermediate yaws are not (face the next one)
    double yaw = 0;
};

/// Minimum clearance (m) along the straight move p -> q at the given heights; throws if the
/// move leaves the collision envelope. Supplied by the caller (the plugin: the scene ESDF).
using ClearanceFn = std::function<double(double x0, double y0, double z0, double x1, double y1, double z1)>;

/// Segments that take the rig from rest at `from` (time t0) through every waypoint. All
/// segments start and end at rest, so they can follow any segment of a script.
inline std::vector<Segment> PlanCommand(const Vec5& from, double t0, const std::vector<Waypoint>& path,
                                        const MotionLimits& L, const ClearanceFn& clearance,
                                        const std::string& tag) {
    std::vector<Segment> segs;
    double t = t0;
    double x = from[0], y = from[1], z = from[2], h = from[3];
    const double pt = from[4];   // pitch is kept: commands steer in the horizontal plane
    auto add = [&](Segment::Kind kind, const Profile& prof, const Vec5& a, const Vec5& b) {
        Segment s;
        s.kind = kind;
        s.t0 = t;
        s.duration = prof.T();
        s.profile = prof;
        s.start = a;
        s.end = b;
        s.tag = tag;
        segs.push_back(s);
        t += s.duration;
    };
    auto turn_to = [&](double target) {
        const double dh = WrapPi(target - h);
        if (std::abs(dh) < 1e-9) return;
        add(Segment::Kind::kTurn, PlanProfile(std::abs(dh), L.omega_max, L.alpha_max), {x, y, z, h, pt},
            {x, y, z, h + dh, pt});
        h = WrapPi(h + dh);
    };
    for (size_t i = 0; i < path.size(); ++i) {
        const Waypoint& w = path[i];
        const double d2 = std::hypot(w.x - x, w.y - y);
        if (d2 >= 1e-9 || std::abs(w.z - z) >= 1e-9) {
            if (d2 >= 1e-9) turn_to(std::atan2(w.y - y, w.x - x));
            const double c = clearance(x, y, z, w.x, w.y, w.z);
            const double v = std::min(L.v_cap, L.v_lk_per_m * c);
            if (!(v > 0)) throw std::runtime_error("PlanCommand: zero clearance along a move");
            const double D = std::sqrt(d2 * d2 + (w.z - z) * (w.z - z));
            add(Segment::Kind::kMove, PlanProfile(D, v, L.a_max), {x, y, z, h, pt}, {w.x, w.y, w.z, h, pt});
            x = w.x, y = w.y, z = w.z;
        }
        if (i + 1 == path.size() && w.has_yaw) turn_to(w.yaw);
    }
    return segs;
}

}  // namespace ariadne::core::scripted
