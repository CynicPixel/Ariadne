// 2.1 hard-pair path (PERCEPTION_2.1.md §12.3): HardPairMatcher's conventions. The model-backed tests use the
// ALIKED + LightGlue export in $ARIADNE_ALIKED_LG_REF (SKIPPED without it, FAILED under ARIADNE_REQUIRE_MODELS=1).
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/features/xfeat/atb.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>

namespace fs = std::filesystem;
using namespace ariadne::perception;
using ariadne::perception::xfeat::LoadAtb;
using ariadne::perception::xfeat::Tensor;

namespace {

std::string RefDir() {
    const char* d = std::getenv("ARIADNE_ALIKED_LG_REF");
    return d ? d : "";
}
bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}
#define REQUIRE_EXPORT(dir)                                                                         \
    if ((dir).empty() || !fs::exists(fs::path(dir) / "manifest.yaml")) {                           \
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no ALIKED/LightGlue export";        \
        GTEST_SKIP() << "no ALIKED/LightGlue export (set ARIADNE_ALIKED_LG_REF)";                  \
    }

HardPairParams Params(const std::string& dir, const std::string& norm) {
    HardPairParams p;
    p.aliked_onnx = (fs::path(dir) / "aliked_n16_k1024_752x480.onnx").string();
    p.lightglue_onnx = (fs::path(dir) / "lightglue_aliked.onnx").string();
    p.normalization = norm;
    return p;
}

/// HardPairFeatures from a reference pair file's upstream keypoints (stored normalised) and descriptors.
HardPairFeatures FromRef(const Tensor& kn, const Tensor& d) {
    HardPairFeatures f;
    f.size = cv::Size(752, 480);
    const int n = int(kn.shape[0]);
    for (int i = 0; i < n; ++i) {   // invert upstream's normalisation: px = kn * max(W,H)/2 + (W/2, H/2)
        f.keypoints.emplace_back(kn.f32[2 * size_t(i)] * 376.0f + 376.0f, kn.f32[2 * size_t(i) + 1] * 376.0f + 240.0f, 1.0f);
    }
    f.descriptors = cv::Mat(n, 128, CV_32F, const_cast<float*>(d.f32.data())).clone();
    return f;
}

}  // namespace

// Upstream normalize_keypoints with size (752, 480): shift (376, 240), ONE scale 376 for both axes.
TEST(HardPair, UpstreamNormalisationKeepsTheAspect) {
    const std::vector<cv::KeyPoint> kp = {{0, 0, 1}, {752, 480, 1}, {376, 240, 1}};
    const cv::Mat n = HardPairMatcher::NormalizeUpstream(kp, cv::Size(752, 480));
    EXPECT_FLOAT_EQ(n.at<float>(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(n.at<float>(0, 1), -240.0f / 376.0f);   // per-axis (OpenCV's rule) would give -1
    EXPECT_FLOAT_EQ(n.at<float>(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(n.at<float>(1, 1), 240.0f / 376.0f);
    EXPECT_FLOAT_EQ(n.at<float>(2, 0), 0.0f);
    EXPECT_FLOAT_EQ(n.at<float>(2, 1), 0.0f);
}

TEST(HardPair, RejectsBadParamsAndTheWrongImageSize) {
    HardPairParams bad;
    bad.normalization = "per-axis";
    EXPECT_THROW(HardPairMatcher{bad}, std::invalid_argument);
    const std::string dir = RefDir();
    REQUIRE_EXPORT(dir);
    const HardPairMatcher m(Params(dir, "upstream"));
    EXPECT_THROW(m.Extract(cv::Mat(720, 1280, CV_8UC1, cv::Scalar(0))), std::invalid_argument);
    EXPECT_THROW(m.Extract(cv::Mat(480, 752, CV_8UC3, cv::Scalar(0))), std::invalid_argument);
    EXPECT_EQ(m.Name(), "aliked_n16_k1024_752x480+lightglue_aliked/cv_dnn_new/norm_upstream");
}

// Through the wrapper, upstream mode reproduces upstream's match SET from upstream's features; OpenCV's per-axis
// mode runs on the same features and is reported (the §12.4 row-D question is measured on EuRoC, not here).
TEST(HardPair, UpstreamModeReproducesUpstreamMatches) {
    const std::string dir = RefDir();
    REQUIRE_EXPORT(dir);
    const HardPairMatcher up(Params(dir, "upstream")), cvm(Params(dir, "opencv"));
    for (const char* pair : {"mh01_f200_mh01_f205", "v101_a_v101_b"}) {
        const auto ref = LoadAtb((fs::path(dir) / ("ref_pair_" + std::string(pair) + "_k1024.atb")).string());
        const HardPairFeatures a = FromRef(ref.at("kpts0n"), ref.at("desc0")), b = FromRef(ref.at("kpts1n"), ref.at("desc1"));
        std::set<std::pair<int, int>> want;
        const auto& rm = ref.at("matches").i64;
        for (size_t i = 0; i + 1 < rm.size(); i += 2) want.insert({int(rm[i]), int(rm[i + 1])});
        std::set<std::pair<int, int>> got, got_cv;
        for (const auto& m : up.Match(a, b)) got.insert({m.queryIdx, m.trainIdx});
        for (const auto& m : cvm.Match(a, b)) got_cv.insert({m.queryIdx, m.trainIdx});
        int shared_cv = 0;
        for (const auto& m : got_cv) shared_cv += int(want.count(m));
        std::printf("[%s] upstream mode %zu (upstream %zu); opencv per-axis mode %zu, %d shared\n", pair, got.size(),
                    want.size(), got_cv.size(), shared_cv);
        EXPECT_EQ(got, want);
        EXPECT_GT(got_cv.size(), 0u);
    }
}
