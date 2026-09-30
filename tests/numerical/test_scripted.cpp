// The C++ script evaluator (what the kinematic mover runs) against golden samples
// from the Python generator that wrote the scripts (tests/fixtures/scripted/).
// Tolerance 1e-9: both sides compute the same formulas, but libm sin/cos may
// differ in the last bit between the Python and C++ builds.
#include "core/trajectory/scripted.hpp"

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <string>

using namespace ariadne::core::scripted;

namespace {

Vec5 V5(const YAML::Node& n) {
    Vec5 v{};
    for (int i = 0; i < 5; ++i) v[i] = n[i].as<double>();
    return v;
}

Script FromJson(const YAML::Node& js) {
    std::vector<Segment> segs;
    for (const auto& d : js["segments"]) {
        Segment s;
        s.kind = Segment::ParseKind(d["kind"].as<std::string>());
        s.t0 = d["t0"].as<double>();
        s.duration = d["duration"].as<double>();
        s.profile = {d["D"].as<double>(), d["Ta"].as<double>(), d["vc"].as<double>()};
        s.start = V5(d["start"]);
        s.end = V5(d["end"]);
        s.tag = d["tag"].as<std::string>();
        segs.push_back(s);
    }
    return Script(std::move(segs));
}

YAML::Node Golden() {
    return YAML::LoadFile(std::string(ARIADNE_SOURCE_DIR) + "/tests/fixtures/scripted/scripted_golden.json");
}

}  // namespace

TEST(Scripted, MatchesPythonGoldenSamples) {
    const YAML::Node golden = Golden();
    ASSERT_EQ(golden.size(), 2u);
    int pitched = 0;
    for (const auto& c : golden) {
        const Script sc = FromJson(c["script"]);
        EXPECT_NO_THROW(sc.Validate()) << c["case"].as<std::string>();
        EXPECT_NEAR(sc.duration(), c["script"]["duration_s"].as<double>(), 1e-9);
        const double g = c["gravity"].as<double>();
        for (const auto& smp : c["samples"]) {
            const double t = smp["t"].as<double>();
            const State s = sc.Eval(t);
            const ImuSample m = Imu(s, g);
            for (int i = 0; i < 5; ++i) {
                ASSERT_NEAR(s.pose[i], smp["pose"][i].as<double>(), 1e-9) << c["case"] << " t=" << t << " i=" << i;
                ASSERT_NEAR(s.rate[i], smp["rate"][i].as<double>(), 1e-9) << " t=" << t;
                ASSERT_NEAR(s.acc[i], smp["acc"][i].as<double>(), 1e-9) << " t=" << t;
            }
            for (int i = 0; i < 3; ++i) {
                ASSERT_NEAR(m.gyro[i], smp["gyro"][i].as<double>(), 1e-9) << " t=" << t;
                ASSERT_NEAR(m.accel[i], smp["accel"][i].as<double>(), 1e-9) << " t=" << t;
            }
            if (std::abs(s.pose[4]) > 1e-6) ++pitched;
        }
    }
    EXPECT_GT(pitched, 50);   // the pitch path (calibration views) is really exercised
}

TEST(Scripted, QuaternionIsRzTimesRy) {
    // Rotate e_x by q and compare with Rz(yaw) Ry(pitch) e_x = (cy cp, sy cp, -sp).
    for (double yaw : {-2.5, -0.3, 0.0, 1.2, 3.0}) {
        for (double pitch : {-0.4, 0.0, 0.25}) {
            const auto q = Quaternion(yaw, pitch);
            const double w = q[0], x = q[1], y = q[2], z = q[3];
            EXPECT_NEAR(w * w + x * x + y * y + z * z, 1.0, 1e-15);
            const double rx = 1 - 2 * (y * y + z * z), ry = 2 * (x * y + w * z), rz = 2 * (x * z - w * y);
            EXPECT_NEAR(rx, std::cos(yaw) * std::cos(pitch), 1e-12);
            EXPECT_NEAR(ry, std::sin(yaw) * std::cos(pitch), 1e-12);
            EXPECT_NEAR(rz, -std::sin(pitch), 1e-12);
        }
    }
}

TEST(Scripted, ValidateRejectsJumpsAndGaps) {
    Segment a;
    a.kind = Segment::Kind::kHold;
    a.duration = 1.0;
    a.start = a.end = {0, 0, 1, 0, 0};
    Segment b = a;
    b.t0 = 1.0;
    EXPECT_NO_THROW(Script({a, b}).Validate());
    b.start = b.end = {0.5, 0, 1, 0, 0};                 // teleport
    EXPECT_THROW(Script({a, b}).Validate(), std::invalid_argument);
    b.start = b.end = a.start;
    b.t0 = 1.5;                                           // time gap
    EXPECT_THROW(Script({a, b}).Validate(), std::invalid_argument);
    EXPECT_THROW(Segment::ParseKind("jump"), std::invalid_argument);
}
