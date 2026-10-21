// SMOKE §4s (PERCEPTION_2.1.md §12): ALIKED-n16 + LightGlue exported from upstream (cvg/LightGlue eb42fee) by
// python/model_tools/aliked_lightglue_reference.py and run through OpenCV's OWN classes, cv::ALIKED and
// cv::LightGlueMatcher (cv::dnn ENGINE_NEW), against upstream's unmodified PyTorch outputs on the same inputs.
//
// Needs the export in $ARIADNE_ALIKED_LG_REF: SKIPPED without it, FAILED without it under ARIADNE_REQUIRE_MODELS=1.
// The sim-rig export (1280x720 RGB, PERCEPTION_2.1 §14) runs the SAME checks from $ARIADNE_ALIKED_LG_SIM_REF.
#include "perception/features/xfeat/atb.hpp"

#include <gtest/gtest.h>
#include <opencv2/dnn.hpp>
#include <opencv2/features.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ariadne::perception::xfeat::LoadAtb;
using ariadne::perception::xfeat::Tensor;

namespace {

std::string RefDir(const char* var = "ARIADNE_ALIKED_LG_REF") {
    const char* d = std::getenv(var);
    return d ? d : "";
}
bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}
#define REQUIRE_EXPORT(dir)                                                                         \
    if ((dir).empty() || !fs::exists(fs::path(dir) / "manifest.yaml")) {                           \
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no ALIKED/LightGlue export";        \
        GTEST_SKIP() << "no ALIKED/LightGlue export";                                             \
    }

cv::Mat Rows(const Tensor& t) {   // [N, D] float -> N x D CV_32F
    return cv::Mat(int(t.shape[0]), int(t.numel() / t.shape[0]), CV_32F, const_cast<float*>(t.f32.data())).clone();
}

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// The sim-rig reference frames (models.sh aliked-lg-sim): agent 0 of the phase-1 ci_tiny recording.
const std::vector<std::string> kSimFrames = {"sim_a0_f200", "sim_a0_f205", "sim_a0_f700", "sim_a0_f710"};
const std::vector<std::string> kSimPairs = {"sim_a0_f200_sim_a0_f205", "sim_a0_f700_sim_a0_f710"};

// cv::ALIKED on the export: the network input equals the reference tensor; >= 99% of upstream's keypoints are found
// within 1e-3 px, with scores and descriptors within 1e-4 per element on those. The input PNG is read as BGR and
// cv::ALIKED swaps to RGB, so a colour input checks the channel order too (the reference input is upstream's RGB).
void CheckAliked(const std::string& dir, cv::Size size, const std::vector<int>& ks, const std::vector<std::string>& names) {
    int checked = 0;
    const std::string wh = std::to_string(size.width) + "x" + std::to_string(size.height);
    for (int k : ks) {
        cv::ALIKED::Params p;
        p.inputSize = size;
        p.engine = cv::dnn::ENGINE_NEW;
        const auto al = cv::ALIKED::create((fs::path(dir) / ("aliked_n16_k" + std::to_string(k) + "_" + wh + ".onnx")).string(), p);
        for (const std::string& name : names) {
            const auto ref = LoadAtb((fs::path(dir) / ("ref_img_" + std::string(name) + "_k" + std::to_string(k) + ".atb")).string());
            const cv::Mat img = cv::imread((fs::path(dir) / ("input_" + std::string(name) + ".png")).string(), cv::IMREAD_COLOR);
            ASSERT_FALSE(img.empty());
            const cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, p.inputSize, cv::Scalar(), true, false);
            const auto& in = ref.at("input").f32;
            ASSERT_EQ(blob.total(), in.size());
            double din = 0;
            for (size_t i = 0; i < in.size(); ++i) din = std::max(din, double(std::fabs(blob.ptr<float>()[i] - in[i])));
            std::vector<cv::KeyPoint> kp;
            cv::Mat desc;
            const auto t0 = std::chrono::steady_clock::now();
            al->detectAndCompute(img, cv::noArray(), kp, desc);
            const double ms = Ms(t0);
            const cv::Mat rk = Rows(ref.at("keypoints")), rd = Rows(ref.at("descriptors"));
            const auto& rs = ref.at("scores").f32;
            ASSERT_EQ(int(kp.size()), rk.rows);
            ASSERT_EQ(desc.rows, rd.rows);
            // Sets, not index order: float error of ~1e-6 in the score map flips near-tied NMS maxima and top-K
            // order (the exporter's own check, aliked_lightglue_reference.py set_compare). Each upstream keypoint is
            // paired with the nearest one here; matched = within 1e-3 px.
            int matched = 0, flipped = 0;
            double dk = 0, dd = 0, ds = 0, flip_int = 0, min_cos = 1;
            for (int i = 0; i < rk.rows; ++i) {
                const cv::Point2f r(rk.at<float>(i, 0), rk.at<float>(i, 1));
                size_t best = 0;
                double bd = 1e18;
                for (size_t j = 0; j < kp.size(); ++j) {
                    const double d = cv::norm(kp[j].pt - r);
                    if (d < bd) bd = d, best = j;
                }
                if (bd > 1e-3) continue;
                ++matched;
                dk = std::max(dk, bd);
                ds = std::max(ds, double(std::fabs(kp[best].response - rs[size_t(i)])));
                const double di = cv::norm(desc.row(int(best)), rd.row(i), cv::NORM_INF);
                if (di > 1e-4) {
                    // SDDH picks its offset patch from the TRUNCATED keypoint (upstream get_patches(kpts.long())):
                    // a keypoint within float noise of an integer can truncate the other way.
                    ++flipped;
                    flip_int = std::max({flip_int, double(std::fabs(r.x - std::round(r.x))), double(std::fabs(r.y - std::round(r.y)))});
                    min_cos = std::min(min_cos, desc.row(int(best)).dot(rd.row(i)));
                } else {
                    dd = std::max(dd, di);
                }
            }
            const double frac = double(matched) / rk.rows;
            std::printf("[%s k=%d] input |diff| %.1e; matched %.4f of %d, keypoints %.2e px, scores %.2e, "
                        "descriptors %.2e on %d; %d over 1e-4 (min cosine %.4f, max |px - round| %.1e); %.0f ms\n",
                        name.c_str(), k, din, frac, rk.rows, dk, ds, dd, matched - flipped, flipped, min_cos, flip_int, ms);
            EXPECT_LE(din, 1e-6);
            EXPECT_GE(frac, 0.99);
            EXPECT_LE(ds, 1e-4);   // a score is the map sampled at the sub-pixel keypoint: px error x a sharp peak
            EXPECT_LE(dd, 1e-4);
            EXPECT_LE(flipped, rk.rows / 100);
            if (flipped) EXPECT_LE(flip_int, 1e-3);   // every outlier is a truncation flip, nothing else
            ++checked;
        }
    }
    EXPECT_EQ(checked, int(ks.size() * names.size()));
}

// cv::LightGlueMatcher on the export, fed upstream's keypoints already normalised by upstream's normalize_keypoints
// and an empty Size() (no per-axis renormalisation): the same match set as upstream, scores within 1e-4.
void CheckLightGlue(const std::string& dir, const std::vector<int>& ks, const std::vector<std::string>& pairs) {
    const auto lg = cv::LightGlueMatcher::create((fs::path(dir) / "lightglue_aliked.onnx").string(), 0.0f);
    int checked = 0;
    for (int k : ks) {
        for (const std::string& pair : pairs) {
            const auto ref = LoadAtb((fs::path(dir) / ("ref_pair_" + std::string(pair) + "_k" + std::to_string(k) + ".atb")).string());
            const cv::Mat k0 = Rows(ref.at("kpts0n")), k1 = Rows(ref.at("kpts1n"));
            const cv::Mat d0 = Rows(ref.at("desc0")), d1 = Rows(ref.at("desc1"));
            const auto& rm = ref.at("matches");
            const auto& rsc = ref.at("mscores").f32;
            std::map<std::pair<int, int>, float> want;
            for (size_t i = 0; i < rsc.size(); ++i) want[{int(rm.i64[2 * i]), int(rm.i64[2 * i + 1])}] = rsc[i];
            lg->setPairInfo(k0, k1, cv::Size(), cv::Size());
            std::vector<cv::DMatch> got;
            const auto t0 = std::chrono::steady_clock::now();
            lg->match(d0, d1, got);
            const double ms = Ms(t0);
            int same = 0;
            double dsc = 0;
            for (const auto& m : got) {
                const auto it = want.find({m.queryIdx, m.trainIdx});
                if (it == want.end()) continue;
                ++same;
                dsc = std::max(dsc, double(std::fabs((1.0f - m.distance) - it->second)));
            }
            std::printf("[%s k=%d] matches %zu (upstream %zu, shared %d), |score| %.2e; %.0f ms\n", pair.c_str(), k, got.size(),
                        want.size(), same, dsc, ms);
            EXPECT_EQ(got.size(), want.size());
            EXPECT_EQ(same, int(want.size()));
            EXPECT_LE(dsc, 1e-4);   // 9 transformer layers of float error (the exporter checks the same bound)
            ++checked;
        }
    }
    EXPECT_EQ(checked, int(ks.size() * pairs.size()));
}

}  // namespace

TEST(AlikedLightGlueReference, AlikedMatchesUpstream) {
    const std::string dir = RefDir();
    REQUIRE_EXPORT(dir);
    CheckAliked(dir, {752, 480}, {1024, 2048}, {"mh01_f200", "mh01_f205", "v101_a", "v101_b"});
}

TEST(AlikedLightGlueReference, LightGlueMatchesUpstream) {
    const std::string dir = RefDir();
    REQUIRE_EXPORT(dir);
    CheckLightGlue(dir, {1024, 2048}, {"mh01_f200_mh01_f205", "v101_a_v101_b"});
}

// The sim rig (L3): 1280x720 RGB frames from the recorded bag, the same bars. The 752x480 validation of the export
// rewrites (int()-frozen dims, Floor before Cast, DKD/SDDH gathers) does not carry over by itself: this re-runs it.
TEST(AlikedLightGlueSimReference, AlikedMatchesUpstream) {
    const std::string dir = RefDir("ARIADNE_ALIKED_LG_SIM_REF");
    REQUIRE_EXPORT(dir);
    CheckAliked(dir, {1280, 720}, {1024}, kSimFrames);
}

TEST(AlikedLightGlueSimReference, LightGlueMatchesUpstream) {
    const std::string dir = RefDir("ARIADNE_ALIKED_LG_SIM_REF");
    REQUIRE_EXPORT(dir);
    CheckLightGlue(dir, {1024}, kSimPairs);
}
