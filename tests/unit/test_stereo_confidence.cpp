// 2.1 §9.3 confidence (perception.md §10): the DisparityWLSFilter map in StereoDepth, and the §9.4 validator's
// remaining refusals. Synthetic stereo with known geometry, so every expectation is exact by construction.
#include "core/random/philox.hpp"
#include "perception/perception_config.hpp"
#include "perception/stereo/sgbm_stereo.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>

using namespace ariadne;
using namespace ariadne::perception;

namespace {

constexpr int kW = 640, kH = 480;

RectifiedStereoRig SmallRig() {
    RectifiedStereoRig r;
    r.width = kW;
    r.height = kH;
    r.fx = r.fy = 400.0;
    r.cx = 319.5;
    r.cy = 239.5;
    r.baseline_m = 0.10;
    return r;
}

StereoParams Params() {
    StereoParams s;
    s.num_disparities = 96;          // Z_min = 400 * 0.1 / 95 = 0.42 m
    s.required_min_range_m = 0.5;
    return s;
}

/// Band-limited noise: textured everywhere, so SGBM matches everywhere.
cv::Mat Texture(uint64_t seed, int w = kW, int h = kH) {
    cv::Mat n(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
        core::PhiloxStream s(seed, 0, static_cast<uint64_t>(y));
        for (int x = 0; x < w; ++x) n.at<uchar>(y, x) = static_cast<uchar>(s.NextBits32() & 0xFF);
    }
    cv::Mat b;
    cv::GaussianBlur(n, b, cv::Size(5, 5), 1.0);
    cv::normalize(b, b, 0, 255, cv::NORM_MINMAX);
    return b;
}

/// A fronto-parallel foreground slab (columns [a, b) of the left image, integer disparity dF) in front of a
/// background plane (dB < dF). Left pixels x in [a - (dF - dB), a) see background that the slab hides from the right
/// camera: the occlusion band. x = a is a depth step of dF - dB px.
struct Layered {
    cv::Mat left, right;
    int a, b, dF, dB;
};
Layered SlabScene() {
    Layered s{cv::Mat(kH, kW, CV_8U), cv::Mat(kH, kW, CV_8U), 380, 560, 48, 24};
    const cv::Mat bg = Texture(11, kW + 64, kH), fg = Texture(12, kW + 64, kH);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            s.left.at<uchar>(y, x) = (x >= s.a && x < s.b) ? fg.at<uchar>(y, x) : bg.at<uchar>(y, x);
            const int xf = x + s.dF;   // the left column the right camera's ray at x hits on the slab
            s.right.at<uchar>(y, x) =
                (xf >= s.a && xf < s.b) ? fg.at<uchar>(y, xf) : bg.at<uchar>(y, x + s.dB);
        }
    return s;
}

double Fraction(const cv::Mat& m, const cv::Rect& r, const std::function<bool(uchar)>& pred) {
    int n = 0;
    for (int y = r.y; y < r.y + r.height; ++y)
        for (int x = r.x; x < r.x + r.width; ++x) n += pred(m.at<uchar>(y, x));
    return double(n) / r.area();
}

bool SameBytes(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    for (int y = 0; y < a.rows; ++y)
        if (std::memcmp(a.ptr(y), b.ptr(y), size_t(a.cols) * a.elemSize()) != 0) return false;
    return true;
}

}  // namespace

TEST(StereoConfidence, DefaultIsThePlaceholder) {
    const Layered s = SlabScene();
    const StereoDepth d = SgbmStereo(Params(), SmallRig()).Compute(s.left, s.right);
    EXPECT_TRUE(d.confidence_is_placeholder);
    EXPECT_EQ(cv::countNonZero(d.confidence), 0);
    EXPECT_EQ(d.confidence_ms, 0.0);
}

// Acceptance §10.7.1: confidence on, filtering off -> the depth outputs are byte-identical to confidence off.
TEST(StereoConfidence, WlsLeavesTheLeftDisparityUntouched) {
    const Layered s = SlabScene();
    StereoParams on = Params();
    on.confidence = "wls";
    const StereoDepth a = SgbmStereo(Params(), SmallRig()).Compute(s.left, s.right);
    const StereoDepth b = SgbmStereo(on, SmallRig()).Compute(s.left, s.right);
    EXPECT_TRUE(SameBytes(a.disparity, b.disparity));
    EXPECT_TRUE(SameBytes(a.Z, b.Z));
    EXPECT_TRUE(SameBytes(a.sigma_Z, b.sigma_Z));
    EXPECT_TRUE(SameBytes(a.valid, b.valid));
    EXPECT_FALSE(b.confidence_is_placeholder);
    EXPECT_GT(b.confidence_ms, 0.0);
    // No confidence without a depth.
    cv::Mat conf_where_invalid;
    b.confidence.copyTo(conf_where_invalid, b.valid == 0);
    EXPECT_EQ(cv::countNonZero(conf_where_invalid), 0);
}

// Smooth surfaces score high; the depth step at the slab's edge scores low (the variance term).
TEST(StereoConfidence, HighOnSmoothSurfacesLowAtTheDepthStep) {
    const Layered s = SlabScene();
    StereoParams p = Params();
    p.confidence = "wls";
    const StereoDepth d = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    const cv::Rect bg(200, 40, 120, 400);    // background, far from the slab
    const cv::Rect fg(420, 40, 100, 400);    // slab interior
    const cv::Rect step(s.a, 40, 3, 400);    // first columns of the slab: the 7x7 box straddles the 24 px step
    auto valid_high = [&](const cv::Rect& r) {
        int v = 0, hi = 0;
        for (int y = r.y; y < r.y + r.height; ++y)
            for (int x = r.x; x < r.x + r.width; ++x) {
                if (!d.valid.at<uchar>(y, x)) continue;
                ++v;
                hi += d.confidence.at<uchar>(y, x) >= 200;
            }
        return std::pair<int, int>(v, hi);
    };
    const auto [bv, bh] = valid_high(bg);
    const auto [fv, fh] = valid_high(fg);
    const auto [sv, sh] = valid_high(step);
    ASSERT_GT(bv, bg.area() * 8 / 10);
    ASSERT_GT(fv, fg.area() * 8 / 10);
    EXPECT_GT(bh, bv * 9 / 10);
    EXPECT_GT(fh, fv * 9 / 10);
    ASSERT_GT(sv, 0) << "no valid depth at the step: the test proves nothing";
    EXPECT_LT(sh, sv / 10);
}

// The left-right check proper. Our SGBM runs its own (disp12_max_diff 1, uniqueness, speckle), so it already drops
// most occluded pixels and the WLS check would be redundant there. With SGBM configured as OpenCV's WLS wiring
// intends (all three off), SGBM KEEPS the occlusion band, and only WLS's check can reject it.
TEST(StereoConfidence, LeftRightCheckRejectsTheOcclusionBand) {
    const Layered s = SlabScene();
    StereoParams p = Params();
    p.confidence = "wls";
    p.uniqueness_ratio = 0;
    // NOT -1: OpenCV 5.0.0 clamps a non-positive disp12MaxDiff to 1 (stereosgbm.cpp: `> 0 ? v : 1`), i.e. the
    // strictest check. 1e6 is how ximgproc's own wiring turns it off.
    p.disp12_max_diff = 1000000;
    p.speckle_window_size = 0;
    const StereoDepth d = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    const cv::Rect band(s.a - (s.dF - s.dB) + 3, 40, (s.dF - s.dB) - 6, 400);
    ASSERT_GT(Fraction(d.valid, band, [](uchar v) { return v != 0; }), 0.8)
        << "SGBM dropped the band itself: the WLS check is not what is being tested";
    EXPECT_GT(Fraction(d.confidence, band, [](uchar c) { return c == 0; }), 0.8);
    const cv::Rect bg(200, 40, 120, 400);
    EXPECT_GT(Fraction(d.confidence, bg, [](uchar c) { return c >= 200; }), 0.8);
}

TEST(StereoConfidence, FilteredDisparityIsOptInAndKeepsTheConfidence) {
    const Layered s = SlabScene();
    StereoParams p = Params();
    p.confidence = "wls";
    const StereoDepth raw = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    p.wls_filtered_disparity = true;
    const StereoDepth flt = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    EXPECT_EQ(flt.disparity.type(), CV_16S);
    EXPECT_FALSE(SameBytes(raw.disparity, flt.disparity));
    cv::Mat both;
    cv::bitwise_and(raw.valid, flt.valid, both);
    cv::Mat a, b;
    raw.confidence.copyTo(a, both);
    flt.confidence.copyTo(b, both);
    EXPECT_TRUE(SameBytes(a, b));   // the map comes from the raw pair either way
}

TEST(StereoConfidence, IndependentOfThreadCount) {
    const Layered s = SlabScene();
    StereoParams p = Params();
    p.confidence = "wls";
    const int saved = cv::getNumThreads();
    cv::setNumThreads(1);
    const StereoDepth one = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    cv::setNumThreads(8);
    const StereoDepth eight = SgbmStereo(p, SmallRig()).Compute(s.left, s.right);
    cv::setNumThreads(saved);
    EXPECT_TRUE(SameBytes(one.confidence, eight.confidence));
    EXPECT_TRUE(SameBytes(one.disparity, eight.disparity));
}

TEST(StereoConfidence, ConfigIsValidated) {
    PerceptionConfig c;
    c.name = "t";
    EXPECT_NO_THROW(ValidatePerception(c));
    c.stereo.confidence = "lrc";
    EXPECT_THROW(ValidatePerception(c), ConfigError);
    c.stereo.confidence = "none";
    c.stereo.wls_filtered_disparity = true;   // filtering needs the wls pass
    EXPECT_THROW(ValidatePerception(c), ConfigError);
    c.stereo.confidence = "wls";
    EXPECT_NO_THROW(ValidatePerception(c));
    c.stereo.wls_lrc_threshold = 0;
    EXPECT_THROW(ValidatePerception(c), ConfigError);
}

// §9.4's remaining refusals (the minimum-range one and the multiple of 16 are in test_perception_config.cpp).
TEST(StereoValidator, RangeCoveringTheImageWidthIsRefused) {
    RectifiedStereoRig r = SmallRig();
    StereoParams s = Params();
    s.num_disparities = 640;   // = the width
    s.required_min_range_m = 10.0;
    EXPECT_THROW(ValidateStereoAgainstRig(s, r), ConfigError);
}

TEST(StereoValidator, ZminExactlyAtTheRequirementIsAccepted) {
    RectifiedStereoRig r = SmallRig();
    StereoParams s = Params();
    s.required_min_range_m = MinObservableRange(s, r);   // Z_min == required: the boundary is allowed
    EXPECT_NO_THROW(ValidateStereoAgainstRig(s, r));
    s.required_min_range_m = std::nextafter(s.required_min_range_m, 0.0);
    EXPECT_THROW(ValidateStereoAgainstRig(s, r), ConfigError);
}

// §10.7.4 cost split on a EuRoC-sized pair (752 x 480, 96 disparities): left SGBM, right-matcher SGBM and
// DisparityWLSFilter::filter (which always runs the FGS smoothing). Printed, not gated: indicative on a shared host.
TEST(StereoConfidence, CostSplitIsReported) {
    const cv::Mat left = Texture(21, 752, 480), right = Texture(22, 752, 480);
    const cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 5, 200, 800, 1, 63, 15, 200, 2,
                                                                cv::StereoSGBM::MODE_SGBM_3WAY);
    const cv::Ptr<cv::StereoMatcher> rm = cv::ximgproc::createRightMatcher(sgbm);
    const cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls = cv::ximgproc::createDisparityWLSFilterGeneric(true);
    const auto ms = [](auto f) {
        double best = 1e9;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            f();
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        return best;
    };
    cv::Mat dl, dr, flt;
    const double t_left = ms([&] { sgbm->compute(left, right, dl); });
    const double t_right = ms([&] { rm->compute(right, left, dr); });
    const double t_filter = ms([&] { wls->filter(dl, left, flt, dr, cv::Rect(96, 0, 752 - 96, 480)); });
    std::printf("cost (best of 5, %d threads): left SGBM %.1f ms | right SGBM %.1f ms | WLS filter (FGS + map) %.1f ms\n",
                cv::getNumThreads(), t_left, t_right, t_filter);
    EXPECT_GT(t_filter, 0.0);
}
