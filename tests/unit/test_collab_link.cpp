// D174 (LIVE_COLLAB §2): the live collaboration's messages round-trip exactly, and a damaged message is refused.
#include "runtime/live/collab_link.hpp"
#include "runtime/live/collab_live.hpp"   // compiled here until ariadne_live wires it (B)
#include "runtime/live/collab_tee.hpp"
#include "runtime/live/collab_worker_core.hpp"

#include <gtest/gtest.h>

namespace rl = ariadne::runtime::live;
namespace rt = ariadne::runtime;
namespace core = ariadne::core;

namespace {

core::SE3 Pose(double a, double x) {
    core::SE3 T;
    T.R = Eigen::AngleAxisd(a, core::Vec3(0.3, 0.5, 0.8).normalized()).toRotationMatrix();
    T.t = core::Vec3(x, -2.0 * x, 0.25);
    return T;
}
bool Same(const core::SE3& a, const core::SE3& b) { return a.R == b.R && a.t == b.t; }
bool SameMat(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type() &&
           std::memcmp(a.ptr(), b.ptr(), a.total() * a.elemSize()) == 0;
}

}  // namespace

TEST(CollabLink, KeyframeInRoundTrips) {
    rl::CollabKeyframeIn k;
    k.id = 41, k.t_ns = 123456789, k.fx = 600.25, k.fy = 600.5, k.cx = 639.5, k.cy = 359.5, k.baseline_m = 0.12;
    k.T_B_C = Pose(0.4, 1.5), k.gt_T_W_B = Pose(-1.1, 3.0);
    k.rgb = cv::Mat(4, 6, CV_8UC3), k.depth16 = cv::Mat(4, 6, CV_16F), k.mask = cv::Mat(4, 6, CV_8U);
    cv::randu(k.rgb, 0, 255), cv::randu(k.mask, 0, 255);
    cv::Mat d32(4, 6, CV_32F);
    cv::randu(d32, 0.1, 9.0);
    d32.convertTo(k.depth16, CV_16F);
    const rl::CollabKeyframeIn r = rl::DecodeKeyframeIn(rl::EncodeKeyframeIn(k));
    EXPECT_EQ(r.id, k.id);
    EXPECT_EQ(r.t_ns, k.t_ns);
    EXPECT_EQ(r.fx, k.fx);
    EXPECT_EQ(r.cy, k.cy);
    EXPECT_EQ(r.baseline_m, k.baseline_m);
    EXPECT_TRUE(Same(r.T_B_C, k.T_B_C));
    ASSERT_TRUE(r.gt_T_W_B.has_value());
    EXPECT_TRUE(Same(*r.gt_T_W_B, *k.gt_T_W_B));
    EXPECT_TRUE(SameMat(r.rgb, k.rgb));
    EXPECT_TRUE(SameMat(r.depth16, k.depth16));
    EXPECT_TRUE(SameMat(r.mask, k.mask));
    k.gt_T_W_B.reset();
    EXPECT_FALSE(rl::DecodeKeyframeIn(rl::EncodeKeyframeIn(k)).gt_T_W_B.has_value());
}

TEST(CollabLink, ReleaseAndRecordRoundTrip) {
    rt::CollabRelease c;
    c.id = 7, c.t_final_ns = 99, c.T_W_C = Pose(0.2, 0.7), c.odo_from = 6, c.odo_Z = Pose(0.01, 0.05);
    for (int i = 0; i < 36; ++i) c.odo_information[size_t(i)] = 1e3 + i;
    const rt::CollabRelease d = rl::DecodeRelease(rl::EncodeRelease(c));
    EXPECT_EQ(d.id, c.id);
    EXPECT_EQ(d.t_final_ns, c.t_final_ns);
    EXPECT_EQ(d.odo_from, c.odo_from);
    EXPECT_TRUE(Same(d.T_W_C, c.T_W_C));
    EXPECT_TRUE(Same(d.odo_Z, c.odo_Z));
    EXPECT_EQ(d.odo_information, c.odo_information);
    c.odo_from.reset();
    EXPECT_FALSE(rl::DecodeRelease(rl::EncodeRelease(c)).odo_from.has_value());

    rt::CollabStoreRecord s;
    s.id = 7, s.t_create_ns = 50, s.t_final_ns = 99, s.T_W_C = c.T_W_C, s.odo_from = 6, s.odo_Z = c.odo_Z;
    s.odo_information = c.odo_information;
    s.global.assign(256, 0.0f);
    for (int i = 0; i < 256; ++i) s.global[size_t(i)] = float(i) * 0.001f - 0.1f;
    s.camera = {600.f, 600.f, 640.f, 360.f, 0.12f};
    s.has_gt = true, s.gt_T_W_B = Pose(1.0, 2.0), s.T_B_C = Pose(-0.5, 0.0);
    uint32_t n = 0;
    const rt::CollabStoreRecord e = rl::DecodeRecord(rl::EncodeRecord(s, 1024), &n);
    EXPECT_EQ(n, 1024u);
    EXPECT_EQ(e.id, s.id);
    EXPECT_EQ(e.t_create_ns, s.t_create_ns);
    EXPECT_EQ(e.global, s.global);
    EXPECT_EQ(e.camera, s.camera);
    EXPECT_TRUE(e.has_gt);
    EXPECT_TRUE(Same(e.gt_T_W_B, s.gt_T_W_B));
    EXPECT_TRUE(Same(e.T_B_C, s.T_B_C));
    EXPECT_TRUE(e.keypoints.empty());   // #ckf carries no local features
}

TEST(CollabLink, FeatureReplyRequestWatermarkHello) {
    rl::CollabFeatureReply f;
    f.kf = 12;
    f.features = rt::CollabLocalFeatures{{1.f, 2.f}, std::vector<float>(128, 0.5f), {0.1f, NAN, 3.f}};
    const rl::CollabFeatureReply g = rl::DecodeFeatureReply(rl::EncodeFeatureReply(f));
    ASSERT_TRUE(g.features.has_value());
    EXPECT_EQ(g.features->keypoints, f.features->keypoints);
    EXPECT_EQ(g.features->descriptors, f.features->descriptors);
    EXPECT_EQ(std::memcmp(g.features->points3d.data(), f.features->points3d.data(), 3 * sizeof(float)), 0);
    f.features.reset();
    EXPECT_FALSE(rl::DecodeFeatureReply(rl::EncodeFeatureReply(f)).features.has_value());   // "gone"
    EXPECT_EQ(rl::DecodeFeatureRequest(rl::EncodeFeatureRequest(77)), 77u);
    EXPECT_EQ(rl::DecodeWatermark(rl::EncodeWatermark(-5)), -5);
    const rl::CollabHello h = rl::DecodeHello(rl::EncodeHello({"estimator", 2}));
    EXPECT_EQ(h.role, "estimator");
    EXPECT_EQ(h.agent, 2u);
}

TEST(CollabLink, DamagedMessagesAreRefused) {
    std::vector<uint8_t> b = rl::EncodeFeatureRequest(5);
    b.pop_back();
    EXPECT_THROW(rl::DecodeFeatureRequest(b), std::runtime_error);   // truncated
    b = rl::EncodeWatermark(1);
    b.push_back(0);
    EXPECT_THROW(rl::DecodeWatermark(b), std::runtime_error);        // trailing bytes
    rl::CollabFeatureReply f;
    f.kf = 1;
    f.features = rt::CollabLocalFeatures{{1.f}, {}, {}};
    b = rl::EncodeFeatureReply(f);
    b[9 + 7] = 0x7f;                                                  // keypoint count's top byte: implausible length
    EXPECT_THROW(rl::DecodeFeatureReply(b), std::runtime_error);
}
