#pragma once
// ---------------------------------------------------------------------------
// Time types (arch §10.6).
//
// Timestamps are INTEGER nanoseconds, never double seconds. This is a
// determinism requirement, not a style preference: floating-point time
// accumulates rounding that depends on evaluation order and on whether the
// compiler contracted a multiply-add, so two machines can disagree on the last
// bits of a timestamp. Integer nanoseconds are exact, and exactness is what the
// Phase 0 cross-machine bit-identity gate rests on.
//
// A strong type (not a bare int64_t) so that nanoseconds, microseconds and
// frame indices cannot be silently swapped at a call site.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <ratio>

namespace ariadne::core {

/// A signed duration in nanoseconds.
struct Duration {
    int64_t ns = 0;

    static constexpr Duration Nanoseconds(int64_t v)  { return Duration{v}; }
    static constexpr Duration Microseconds(int64_t v) { return Duration{v * 1'000}; }
    static constexpr Duration Milliseconds(int64_t v) { return Duration{v * 1'000'000}; }
    /// Seconds from a double is the ONE lossy entry point, for config files
    /// only. Never use it on a per-sample path.
    static Duration SecondsFromDouble(double s) {
        return Duration{static_cast<int64_t>(s * 1e9)};
    }

    constexpr double toSeconds() const { return static_cast<double>(ns) * 1e-9; }

    constexpr Duration operator+(Duration o) const { return Duration{ns + o.ns}; }
    constexpr Duration operator-(Duration o) const { return Duration{ns - o.ns}; }
    constexpr bool operator<(Duration o)  const { return ns < o.ns; }
    constexpr bool operator>(Duration o)  const { return ns > o.ns; }
    constexpr bool operator<=(Duration o) const { return ns <= o.ns; }
    constexpr bool operator>=(Duration o) const { return ns >= o.ns; }
    constexpr bool operator==(Duration o) const { return ns == o.ns; }
    constexpr bool operator!=(Duration o) const { return ns != o.ns; }
    constexpr Duration abs() const { return Duration{ns < 0 ? -ns : ns}; }
};

/// A point in time on the simulated clock (§5.4). In simulation this is the
/// /clock topic; on a bag it is the recorded stamp. There is exactly one clock.
struct Timestamp {
    int64_t ns = 0;

    static constexpr Timestamp Nanoseconds(int64_t v) { return Timestamp{v}; }
    constexpr double toSeconds() const { return static_cast<double>(ns) * 1e-9; }

    constexpr Timestamp operator+(Duration d) const { return Timestamp{ns + d.ns}; }
    constexpr Timestamp operator-(Duration d) const { return Timestamp{ns - d.ns}; }
    constexpr Duration  operator-(Timestamp o) const { return Duration{ns - o.ns}; }
    constexpr bool operator<(Timestamp o)  const { return ns < o.ns; }
    constexpr bool operator>(Timestamp o)  const { return ns > o.ns; }
    constexpr bool operator<=(Timestamp o) const { return ns <= o.ns; }
    constexpr bool operator>=(Timestamp o) const { return ns >= o.ns; }
    constexpr bool operator==(Timestamp o) const { return ns == o.ns; }
    constexpr bool operator!=(Timestamp o) const { return ns != o.ns; }
};

/// Camera/IMU time-offset calibration.
///
/// STATED SIMPLIFICATION (§10.6): both streams come from the same simulated
/// clock, so this is 0 in simulation and NO online time-offset estimation
/// exists. It is carried explicitly, rather than omitted, so that a reader can
/// see the real-hardware path is absent instead of discovering it later.
struct TimeOffsetCalibration {
    Duration t_offset_cam_imu{0};   // [ASSUMPTION] exactly 0 in simulation
    bool estimated_online = false;  // always false; see §10.6
};

}  // namespace ariadne::core
