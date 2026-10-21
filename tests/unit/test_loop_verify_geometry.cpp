// PERCEPTION_2.1.md §12.4 convention test: a synthetic pair with KNOWN T_Cb_Ca through the exact evaluation pipeline
// (radtan distortion -> undistort -> RANSAC E -> recoverPose -> glue-factory error). An inverted R or a swapped point
// order would otherwise surface on EuRoC as a "bad pose AUC" for every method.
#include "core/random/philox.hpp"
#include "runtime/tools/loop_verify_geometry.hpp"

#include <gtest/gtest.h>
#include <opencv2/geometry.hpp>

using namespace ariadne;

namespace {

perception::PinholeRadtan EurocCam0() {   // mav0/cam0/sensor.yaml
    perception::PinholeRadtan c;
    c.width = 752;
    c.height = 480;
    c.fu = 458.654;
    c.fv = 457.296;
    c.cu = 367.215;
    c.cv = 248.375;
    c.dist = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
    return c;
}

/// Points in camera a, projected (with distortion) into a and into b = T_Cb_Ca a; only those inside both images.
void Project(const core::SE3& T_Cb_Ca, const perception::PinholeRadtan& cam, std::vector<cv::Point2f>& pa,
             std::vector<cv::Point2f>& pb) {
    core::PhiloxStream s(11, 0, 0);
    auto u = [&]() { return (s.NextBits32() >> 8) * (1.0 / 16777216.0); };
    std::vector<cv::Point3d> Xa, Xb;
    for (int i = 0; i < 400; ++i) {
        const core::Vec3 x((u() - 0.5) * 4.0, (u() - 0.5) * 3.0, 2.0 + 4.0 * u());
        const core::Vec3 y = T_Cb_Ca.R * x + T_Cb_Ca.t;
        if (y.z() < 0.5) continue;
        Xa.emplace_back(x.x(), x.y(), x.z());
        Xb.emplace_back(y.x(), y.y(), y.z());
    }
    const cv::Matx33d K = runtime::CameraMatrix(cam);
    const std::vector<double> D(cam.dist.begin(), cam.dist.end());
    std::vector<cv::Point2d> ua, ub;
    cv::projectPoints(Xa, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, ua);
    cv::projectPoints(Xb, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), K, D, ub);
    for (size_t i = 0; i < ua.size(); ++i) {
        auto in = [&](const cv::Point2d& p) { return p.x > 5 && p.y > 5 && p.x < cam.width - 5 && p.y < cam.height - 5; };
        if (in(ua[i]) && in(ub[i])) {
            pa.emplace_back(float(ua[i].x), float(ua[i].y));
            pb.emplace_back(float(ub[i].x), float(ub[i].y));
        }
    }
}

}  // namespace

TEST(LoopVerifyGeometry, RecoversAKnownRelativePose) {
    const auto cam = EurocCam0();
    core::SE3 T;   // p_b = T p_a: 12 deg about y, 5 deg about x, a 0.35 m baseline
    T.R = core::SO3Exp(core::Vec3(5.0, 12.0, 0.0) * M_PI / 180.0);
    T.t = core::Vec3(0.3, 0.05, 0.15);
    std::vector<cv::Point2f> pa, pb;
    Project(T, cam, pa, pb);
    ASSERT_GT(pa.size(), 100u);
    const auto e = runtime::EstimateRelPose(pa, pb, cam);
    const auto err = runtime::PoseErrorDeg(e, T);
    std::printf("synthetic: %d matches, %d inliers, error R %.4f t %.4f deg\n", e.matches, e.inliers, err.r_deg, err.t_deg);
    ASSERT_TRUE(e.ok);
    EXPECT_GE(e.inliers, int(pa.size()) * 95 / 100);
    EXPECT_LT(err.deg, 0.1);
    // The swapped pair estimates T^-1: scored against T it must be clearly wrong (the convention is not symmetric).
    const auto s = runtime::EstimateRelPose(pb, pa, cam);
    EXPECT_GT(runtime::PoseErrorDeg(s, T).deg, 5.0);
    EXPECT_LT(runtime::PoseErrorDeg(s, T.inverse()).deg, 0.1);
}

TEST(LoopVerifyGeometry, FailureIsInfiniteError) {
    const auto e = runtime::EstimateRelPose({{1, 1}, {2, 2}}, {{1, 1}, {2, 2}}, EurocCam0());
    EXPECT_FALSE(e.ok);
    EXPECT_EQ(e.matches, 2);
    EXPECT_TRUE(std::isinf(runtime::PoseErrorDeg(e, core::SE3()).deg));
}
