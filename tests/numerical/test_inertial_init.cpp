// Visual-inertial initialisation (slam/inertial/initialization.hpp, inertial.md §4.3): from exact body poses and an
// ideal biased IMU it recovers b_g, b_a, gravity and every velocity; a scale error in the poses (what a wrong frame or
// a monocular map gives) is refused by the |g| check instead of producing a wrong gravity.
#include "slam/inertial/initialization.hpp"
#include "inertial_window_fixture.hpp"

#include <gtest/gtest.h>

using namespace ariadne;
using namespace ariadne::slam;
using namespace ariadne::slam::test;

namespace {

struct InitData {
    std::vector<SE3> T_W_B;
    std::vector<int64_t> t_ns;
    std::vector<std::vector<core::ImuMeasurement>> imu;
};

InitData MakeInitData(int N, double position_scale) {
    InitData d;
    for (int k = 0; k < N; ++k) {
        SE3 T = BodyPose(KeyframeTime(k));
        T.t *= position_scale;
        d.T_W_B.push_back(T);
        if (k + 1 < N) {
            d.imu.push_back(IdealImu(KeyframeTime(k), KeyframeTime(k + 1), 1000.0));
            d.t_ns.push_back(d.imu.back().front().t_ns);
        } else {
            d.t_ns.push_back(d.imu.back().back().t_ns);
        }
    }
    return d;
}

const PreintegrationNoise kNoise{5e-4, 2e-2};

}  // namespace

double GravityAngle(const Vec3& g) { return std::acos(std::min(1.0, g.normalized().dot(kGravity.normalized()))); }

// Noise-free and a negligible b_a prior: the algebra recovers the truth exactly.
TEST(InertialInit, RecoversBiasesGravityAndVelocities) {
    const InitData d = MakeInitData(10, 1.0);
    InertialInitParams p;
    p.accel_bias_sigma = 1e3;
    const InertialInitResult r = InitializeInertial(d.T_W_B, d.t_ns, d.imu, kNoise, p);
    ASSERT_TRUE(r.ok) << r.why;
    std::printf("init (flat b_a prior): |g| %.5f, condition %.2e, chi2/dof %.2e; bg err %.2e, ba err %.2e, g angle %.2e deg\n",
                r.gravity_norm_unconstrained, r.condition, r.chi2_per_dof, (r.bg - kTrueBg).norm(), (r.ba - kTrueBa).norm(),
                GravityAngle(r.g_W) * 180 / M_PI);
    EXPECT_LT((r.bg - kTrueBg).norm(), 1e-5);
    EXPECT_LT((r.ba - kTrueBa).norm(), 1e-3);
    EXPECT_NEAR(r.g_W.norm(), 9.81, 1e-12);
    EXPECT_LT(GravityAngle(r.g_W), 1e-4);
    ASSERT_EQ(r.v.size(), 10u);
    for (int k = 0; k < 10; ++k) EXPECT_LT((r.v[size_t(k)] - BodyVelocity(KeyframeTime(k))).norm(), 1e-3) << k;
    EXPECT_GT(r.bias_cov.diagonal().minCoeff(), 0.0);
}

// With the default zero-mean b_a prior over 2.25 s of mild rotation, b_a and gravity's tilt trade off (arch §10.3:
// weakly observable): the b_a error is the tilt times 9.81, and the velocities are unaffected.
TEST(InertialInit, AccelBiasTradesOffAgainstGravityTiltUnderThePrior) {
    const InitData d = MakeInitData(10, 1.0);
    const InertialInitResult r = InitializeInertial(d.T_W_B, d.t_ns, d.imu, kNoise, {});
    ASSERT_TRUE(r.ok) << r.why;
    std::printf("init (prior 0.2): chi2/dof %.2e; ba err %.3f, g angle %.3f deg (x 9.81 = %.3f)\n", r.chi2_per_dof,
                (r.ba - kTrueBa).norm(), GravityAngle(r.g_W) * 180 / M_PI, GravityAngle(r.g_W) * 9.81);
    EXPECT_NEAR((r.ba - kTrueBa).norm(), GravityAngle(r.g_W) * 9.81, 0.2 * (r.ba - kTrueBa).norm() + 1e-3);
    EXPECT_LT(GravityAngle(r.g_W), 1.0 * M_PI / 180);
    for (int k = 0; k < 10; ++k) EXPECT_LT((r.v[size_t(k)] - BodyVelocity(KeyframeTime(k))).norm(), 0.02) << k;
}

// A scale error (a wrong frame or a monocular map would give one) moves |g| only by the motion's acceleration over
// 9.81 (~8% at 1.5x here), but a constant g cannot fit the inconsistent accelerations: chi2/dof grows by orders of
// magnitude. The |g| check catches gross errors; the chi2 threshold (imu.full.init_max_chi2_per_dof) the rest.
TEST(InertialInit, WrongScaleIsVisibleInTheFit) {
    std::vector<InertialInitResult> r;
    for (double scale : {1.0, 1.05, 1.5, 3.0}) {
        const InitData d = MakeInitData(10, scale);
        r.push_back(InitializeInertial(d.T_W_B, d.t_ns, d.imu, kNoise, {}));
        std::printf("scale %.2f: ok %d, |g| unconstrained %.3f, chi2/dof %.3e (%s)\n", scale, r.back().ok,
                    r.back().gravity_norm_unconstrained, r.back().chi2_per_dof, r.back().why.c_str());
    }
    EXPECT_GT(r[1].chi2_per_dof, 10 * r[0].chi2_per_dof);
    EXPECT_GT(r[2].chi2_per_dof, 1000 * r[0].chi2_per_dof);
    EXPECT_FALSE(r[3].ok);
    InertialInitParams p;
    p.max_chi2_per_dof = 1.0;
    const InitData d = MakeInitData(10, 1.5);
    const InertialInitResult refused = InitializeInertial(d.T_W_B, d.t_ns, d.imu, kNoise, p);
    EXPECT_FALSE(refused.ok);
    EXPECT_NE(refused.why.find("chi2"), std::string::npos) << refused.why;
}
