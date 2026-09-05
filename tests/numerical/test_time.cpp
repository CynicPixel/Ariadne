// Time types (arch §10.6). The point of these tests is exactness: integer
// nanosecond arithmetic must never round.
#include "core/time/timestamp.hpp"

#include <gtest/gtest.h>

using namespace ariadne::core;

TEST(Time, ArithmeticIsExact) {
    const Timestamp t0 = Timestamp::Nanoseconds(1'700'000'000'123'456'789LL);
    const Duration  d  = Duration::Milliseconds(33);
    EXPECT_EQ((t0 + d).ns, 1'700'000'000'156'456'789LL);
    EXPECT_EQ(((t0 + d) - t0).ns, 33'000'000LL);
    // Round-tripping a large timestamp through the type loses nothing, which a
    // double-seconds representation could not promise at nanosecond resolution.
    EXPECT_EQ(((t0 + d) - d).ns, t0.ns);
}

TEST(Time, AccumulationDoesNotDrift) {
    // 30 minutes of 200 Hz IMU samples added one at a time. With double
    // seconds this accumulates visible error; with int64 ns it is exact.
    Timestamp t = Timestamp::Nanoseconds(0);
    const Duration step = Duration::Microseconds(5000);   // 5 ms = 200 Hz
    const int64_t n = 30 * 60 * 200;
    for (int64_t i = 0; i < n; ++i) t = t + step;
    EXPECT_EQ(t.ns, n * 5'000'000LL);
}

TEST(Time, Ordering) {
    const Timestamp a = Timestamp::Nanoseconds(10);
    const Timestamp b = Timestamp::Nanoseconds(20);
    EXPECT_TRUE(a < b);
    EXPECT_TRUE(b > a);
    EXPECT_TRUE(a <= a);
    EXPECT_TRUE(a == a);
    EXPECT_TRUE(a != b);
    EXPECT_EQ((b - a).ns, 10);
    EXPECT_EQ((a - b).abs().ns, 10);
}

TEST(Time, OffsetCalibrationDefaultsToTheStatedSimplification) {
    const TimeOffsetCalibration cal;
    EXPECT_EQ(cal.t_offset_cam_imu.ns, 0);
    EXPECT_FALSE(cal.estimated_online);   // §10.6: no online estimation exists
}
