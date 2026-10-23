// D171 / arch §15.7: local-descriptor wire encodings (collaboration/retrieval/local_descriptor_codec.hpp) and their
// effect on the inter-agent hard-pair matcher (ALIKED-n16 + LightGlue, D63) on the D90 sim reference frames.
// §15.7 asks for the int8 ablation "with matching precision/recall rather than as a byte saving alone": against the
// fp32 match set of the same pair, precision = |enc ∩ fp32| / |enc|, recall = |enc ∩ fp32| / |fp32|.
#include "collaboration/retrieval/local_descriptor_codec.hpp"
#include "perception/association/hard_pair_matcher.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>

namespace ac = ariadne::collaboration;
namespace ap = ariadne::perception;
namespace fs = std::filesystem;

TEST(LocalDescriptorCodec, RoundTripErrorsAreBoundedByTheEncoding) {
    std::vector<float> d;
    for (int i = 0; i < 128 * 7; ++i) d.push_back(std::sin(0.37f * float(i)) * 0.25f);
    const float s = 0.25f;
    for (auto e : {ac::LocalEncoding::kFp32, ac::LocalEncoding::kFp16, ac::LocalEncoding::kInt8}) {
        const auto b = ac::EncodeLocal(d, e);
        EXPECT_EQ(b.size(), d.size() * ac::BytesPerDim(e) + (e == ac::LocalEncoding::kInt8 ? 4u : 0u));
        const auto r = ac::DecodeLocal(b, e);
        ASSERT_EQ(r.size(), d.size());
        double worst = 0;
        for (size_t i = 0; i < d.size(); ++i) worst = std::max(worst, double(std::abs(r[i] - d[i])));
        const double bound = e == ac::LocalEncoding::kFp32 ? 0.0 : e == ac::LocalEncoding::kFp16 ? 2.5e-4 : s / 254.0 + 1e-7;
        EXPECT_LE(worst, bound) << int(e);
    }
}

TEST(LocalDescriptorCodec, LightGlueMatchSetUnderFp16AndInt8OnSimPairs) {
    const char* dir = std::getenv("ARIADNE_ALIKED_LG_SIM_REF");
    if (!dir || !fs::exists(fs::path(dir) / "manifest.yaml")) {
        const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
        if (r && std::string(r) == "1") FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no sim ALIKED/LightGlue export";
        GTEST_SKIP() << "no sim ALIKED/LightGlue export (ARIADNE_ALIKED_LG_SIM_REF)";
    }
    ap::HardPairParams p;
    p.aliked_onnx = (fs::path(dir) / "aliked_n16_k1024_1280x720.onnx").string();
    p.lightglue_onnx = (fs::path(dir) / "lightglue_aliked.onnx").string();
    p.width = 1280, p.height = 720;
    const ap::HardPairMatcher m(p);
    auto load = [&](const char* name) {
        cv::Mat bgr = cv::imread((fs::path(dir) / name).string(), cv::IMREAD_COLOR), rgb;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        return m.ExtractRgb(rgb);
    };
    auto requantize = [](ap::HardPairFeatures f, ac::LocalEncoding e) {
        std::vector<float> v(f.descriptors.begin<float>(), f.descriptors.end<float>());
        const auto r = ac::DecodeLocal(ac::EncodeLocal(v, e), e);
        cv::Mat(f.descriptors.rows, f.descriptors.cols, CV_32F, const_cast<float*>(r.data())).copyTo(f.descriptors);
        return f;
    };
    auto set_of = [](const std::vector<cv::DMatch>& ms) {
        std::set<std::pair<int, int>> s;
        for (const auto& x : ms) s.insert({x.queryIdx, x.trainIdx});
        return s;
    };
    for (auto [a, b] : {std::pair{"input_sim_a0_f200.png", "input_sim_a0_f205.png"},
                        std::pair{"input_sim_a0_f700.png", "input_sim_a0_f710.png"}}) {
        const ap::HardPairFeatures A = load(a), B = load(b);
        const auto ref = set_of(m.Match(A, B));
        ASSERT_GT(ref.size(), 50u);
        for (auto e : {ac::LocalEncoding::kFp16, ac::LocalEncoding::kInt8}) {
            const auto got = set_of(m.Match(requantize(A, e), requantize(B, e)));
            size_t both = 0;
            for (const auto& x : got) both += ref.count(x);
            const double prec = got.empty() ? 0 : double(both) / got.size(), rec = double(both) / ref.size();
            std::cout << "[ MEASURED ] " << a << " / " << b << " " << (e == ac::LocalEncoding::kFp16 ? "fp16" : "int8")
                      << ": fp32 " << ref.size() << " matches, encoded " << got.size() << ", precision " << prec
                      << ", recall " << rec << "\n";
            EXPECT_GT(rec, 0.5) << "sanity only: the decision is on the measured numbers (D171)";
        }
    }
}
