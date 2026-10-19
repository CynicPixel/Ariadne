// 2.1 XFeatEngine (PERCEPTION_2.1.md §4.1): the §7.1 contract over the D50 network. The network itself is checked
// against upstream by test_xfeat_reference.cpp; these tests check what the engine adds: one forward per image,
// §7.3 selection with the mask applied first, input-size handling, and Describe() agreeing with the detector's own
// descriptor sampling.
//
// Needs $ARIADNE_XFEAT_REF (weights + input images), like the reference test: SKIPPED without it, FAILED without it
// under ARIADNE_REQUIRE_MODELS=1.
#include "perception/features/engine_factory.hpp"
#include "perception/features/xfeat_engine.hpp"
#include "perception/perception_config.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>

using namespace ariadne::perception;
namespace fs = std::filesystem;

namespace {

std::string Weights() {
    const char* d = std::getenv("ARIADNE_XFEAT_REF");
    return d ? (fs::path(d) / "xfeat_weights.atb").string() : std::string();
}

bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}

#define NEED_MODEL()                                                                              \
    do {                                                                                          \
        if (Weights().empty() || !fs::exists(Weights())) {                                        \
            if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no XFeat weights";             \
            GTEST_SKIP() << "no XFeat reference (set ARIADNE_XFEAT_REF)";                         \
        }                                                                                         \
    } while (0)

/// The EuRoC MH_01 frame of the reference export (736 x 480: a multiple of 32).
cv::Mat Euroc() {
    return cv::imread((fs::path(std::getenv("ARIADNE_XFEAT_REF")) / "input_euroc_mh01_f200.png").string(),
                      cv::IMREAD_GRAYSCALE);
}

FeatureParams Params() {
    FeatureParams p;
    p.engine = "xfeat";
    p.xfeat_weights = Weights();
    p.max_keypoints = 1000;
    return p;
}

}  // namespace

TEST(XFeatEngine, DescribeAtDetectedKeypointsIsTheDetectorsDescriptorExactly) {
    NEED_MODEL();
    const XFeatEngine e(Params());
    const cv::Mat img = Euroc();
    ASSERT_EQ(img.cols % 32, 0);
    const FeatureSet fs = e.DetectAndCompute(img, cv::Mat());
    ASSERT_GT(fs.keypoints.size(), 500u);
    ASSERT_EQ(fs.descriptors.rows, int(fs.keypoints.size()));
    ASSERT_EQ(fs.descriptors.type(), CV_32F);
    std::vector<cv::Point2f> pts;
    for (const auto& k : fs.keypoints) pts.push_back(k.pt);
    std::vector<uchar> ok;
    const cv::Mat d = e.Describe(img, pts, ok);
    // Native size: scale 1, the same sampler at the same pixel, so bit-identical.
    EXPECT_EQ(cv::norm(d, fs.descriptors, cv::NORM_INF), 0.0);
    for (uchar v : ok) EXPECT_EQ(v, 1);
    for (int r = 0; r < d.rows; ++r) EXPECT_NEAR(cv::norm(d.row(r)), 1.0, 1e-5);
}

TEST(XFeatEngine, OneForwardPerImageWhateverIsAsked) {
    NEED_MODEL();
    const XFeatEngine e(Params());
    cv::Mat img = Euroc().clone();
    const auto kps = e.Detect(img, cv::Mat(), 400);
    std::vector<cv::Point2f> pts{{100.f, 100.f}};
    std::vector<uchar> ok;
    e.Describe(img, pts, ok);
    e.DetectAndCompute(img, cv::Mat());
    EXPECT_EQ(e.forwards(), 1);
    EXPECT_EQ(e.cache_hits(), 2);
    // Same buffer, new content: a miss (the cache compares pixels, not pointers).
    img.at<uchar>(240, 368) ^= 0x55;
    e.Detect(img, cv::Mat(), 400);
    EXPECT_EQ(e.forwards(), 2);
}

TEST(XFeatEngine, MaskAppliesBeforeSelectionSoTheBudgetIsSpentWhereAllowed) {
    NEED_MODEL();
    const XFeatEngine e(Params());
    const cv::Mat img = Euroc();
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
    mask(cv::Rect(0, 0, img.cols / 2, img.rows)).setTo(255);
    const int budget = 300;
    const auto all = e.Detect(img, cv::Mat(), 4000);
    int left = 0;
    for (const auto& k : all) left += k.pt.x < img.cols / 2 - 1 ? 1 : 0;
    ASSERT_GT(left, budget) << "the test needs more candidates on the left than the budget";
    const auto kps = e.Detect(img, mask, budget);
    EXPECT_EQ(int(kps.size()), budget);
    for (const auto& k : kps) EXPECT_EQ(mask.at<uchar>(cvRound(k.pt.y), cvRound(k.pt.x)), 255);
}

TEST(XFeatEngine, PadModeNeverDetectsInThePad) {
    NEED_MODEL();
    FeatureParams p = Params();
    p.xfeat_input = "pad";
    const XFeatEngine e(p);
    const cv::Mat img = Euroc()(cv::Rect(0, 0, 720, 470)).clone();   // neither side a multiple of 32
    EXPECT_EQ(e.NetSize(img.size()), cv::Size(736, 480));
    const auto kps = e.Detect(img, cv::Mat(), 4000);
    ASSERT_GT(kps.size(), 500u);
    for (const auto& k : kps) {
        EXPECT_LT(k.pt.x, 720.f);
        EXPECT_LT(k.pt.y, 470.f);
    }
}

TEST(XFeatEngine, ResizeModeMapsBackToImagePixels) {
    NEED_MODEL();
    const XFeatEngine e(Params());
    // EuRoC's real size: 752 x 480 -> upstream's 736 x 480.
    cv::Mat img;
    cv::resize(Euroc(), img, cv::Size(752, 480), 0, 0, cv::INTER_LINEAR);
    EXPECT_EQ(e.NetSize(img.size()), cv::Size(736, 480));
    const FeatureSet fs = e.DetectAndCompute(img, cv::Mat());
    ASSERT_GT(fs.keypoints.size(), 500u);
    float max_x = 0;
    std::vector<cv::Point2f> pts;
    for (const auto& k : fs.keypoints) {
        max_x = std::max(max_x, k.pt.x);
        pts.push_back(k.pt);
    }
    EXPECT_GT(max_x, 735.f);   // the right edge is reached: keypoints were scaled by 752/736
    EXPECT_LE(max_x, 751.f);
    // Describe at image pixels samples at pt * 736/752: equal to the detector's up to that float round trip.
    std::vector<uchar> ok;
    const cv::Mat d = e.Describe(img, pts, ok);
    EXPECT_LT(cv::norm(d, fs.descriptors, cv::NORM_INF), 1e-4);
    EXPECT_EQ(e.ManifestFields()["xfeat_input"], "resize:736x480");
}

TEST(XFeatEngine, DescribeRefusesOnlyPointsOutsideTheImage) {
    NEED_MODEL();
    const XFeatEngine e(Params());
    const cv::Mat img = Euroc();
    std::vector<uchar> ok;
    const cv::Mat d = e.Describe(img, {{0.f, 0.f}, {735.f, 479.f}, {-0.5f, 10.f}, {10.f, 480.f}}, ok);
    EXPECT_EQ(ok, (std::vector<uchar>{1, 1, 0, 0}));
    EXPECT_EQ(cv::norm(d.row(2), cv::NORM_INF), 0.0);
}

TEST(XFeatEngine, FactoryKeepsOrbTheDefault) {
    FeatureParams p;
    EXPECT_EQ(MakeFeatureEngine(p)->Name(), "orb");
    EXPECT_TRUE(MakeFeatureEngine(p)->ManifestFields().empty());
    EXPECT_TRUE(MakeFeatureEngine(p)->ResultsYaml().empty());
}

TEST(XFeatEngine, ConfigRequiresWeightsAndValidSizes) {
    PerceptionConfig c;
    c.name = "t";
    EXPECT_NO_THROW(ValidatePerception(c));            // the defaults (orb) are valid: the throws below are the keys
    c.features.engine = "xfeat";
    EXPECT_THROW(ValidatePerception(c), ConfigError);   // no weights
    c.features.xfeat_weights = "w.atb";
    c.features.xfeat_input_width = 740;               // not a multiple of 32
    EXPECT_THROW(ValidatePerception(c), ConfigError);
    c.features.xfeat_input_width = 0;
    c.features.engine = "sift";
    EXPECT_THROW(ValidatePerception(c), ConfigError);
}

// The XFeat ablation config of D60 (row A4t-r, PERCEPTION_2.1.md §9.11) parses and carries the measured values.
TEST(XFeatEngine, ShippedAblationConfigIsTheMeasuredRow) {
    const PerceptionConfig c =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/euroc_v2_xfeat.yaml");
    EXPECT_EQ(c.features.engine, "xfeat");
    EXPECT_EQ(c.features.xfeat_input, "resize");
    EXPECT_EQ(c.tracker.replenish, "keyframe");
    EXPECT_EQ(c.tracker.replenish_min_tracks, 300);
    EXPECT_EQ(c.tracker.replenish_min_eig_quality, 0.0);
    EXPECT_TRUE(c.reassociation.enabled);
    EXPECT_DOUBLE_EQ(c.reassociation.ratio, 0.75);
    EXPECT_DOUBLE_EQ(c.reassociation.max_l2, 0.6);
}
