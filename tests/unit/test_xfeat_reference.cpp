// D15 acceptance test: the OpenCV-primitive XFeat port vs upstream's own PyTorch outputs on fixed inputs.
//
// Needs the reference export (python/model_tools/xfeat_reference.py) in $ARIADNE_XFEAT_REF. Without it the test is
// SKIPPED with a message, unless ARIADNE_REQUIRE_MODELS=1 (the pre-merge gate sets it), where a missing reference
// FAILS: a learned component whose acceptance test silently did not run is the failure mode §7.4 warns about.
#include "perception/features/xfeat/xfeat_net.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>

using namespace ariadne::perception::xfeat;
namespace fs = std::filesystem;

namespace {

std::string RefDir() {
    const char* d = std::getenv("ARIADNE_XFEAT_REF");
    return d ? d : "";
}

bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}

struct Diff {
    double max_abs = 0, ref_max = 0;
    double rel() const { return ref_max > 0 ? max_abs / ref_max : max_abs; }
};

Diff Compare(const FMap& ours, const Tensor& ref) {
    EXPECT_EQ(ref.shape.size(), 3u);
    EXPECT_EQ(ref.shape[0], ours.c);
    EXPECT_EQ(ref.shape[1], ours.h);
    EXPECT_EQ(ref.shape[2], ours.w);
    Diff d;
    const float* o = ours.data.ptr<float>(0);
    for (size_t i = 0; i < ref.f32.size(); ++i) {
        d.max_abs = std::max(d.max_abs, double(std::fabs(o[i] - ref.f32[i])));
        d.ref_max = std::max(d.ref_max, double(std::fabs(ref.f32[i])));
    }
    return d;
}

}  // namespace

class XFeatReference : public ::testing::TestWithParam<Engine> {};

TEST_P(XFeatReference, ForwardAndDetectMatchUpstream) {
    const std::string dir = RefDir();
    if (dir.empty() || !fs::exists(fs::path(dir) / "xfeat_weights.atb")) {
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no XFeat reference in ARIADNE_XFEAT_REF='" << dir << "'";
        GTEST_SKIP() << "no XFeat reference (set ARIADNE_XFEAT_REF to the xfeat_reference.py output)";
    }
    const XFeatNet net((fs::path(dir) / "xfeat_weights.atb").string(), GetParam());
    std::printf("engine %s, cv::getNumThreads %d\n", EngineName(net.engine()), cv::getNumThreads());
    int checked = 0;
    for (const auto& e : fs::directory_iterator(dir)) {
        const std::string fn = e.path().filename().string();
        if (fn.rfind("ref_", 0) != 0) continue;
        const std::string name = fn.substr(4, fn.size() - 8);
        const cv::Mat gray = cv::imread((fs::path(dir) / ("input_" + name + ".png")).string(), cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(gray.empty()) << name;
        const TensorMap ref = LoadAtb(e.path().string());

        net.Forward(gray);   // warm-up: the dnn engine builds its graph for this size on first use
        const auto t0 = std::chrono::steady_clock::now();
        const NetOutput o = net.Forward(gray);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const Diff f = Compare(o.feats, ref.at("feats")), k = Compare(o.keypoint_logits, ref.at("keypoint_logits")),
                   h = Compare(o.heatmap, ref.at("heatmap"));
        std::printf("[%s %dx%d] forward %.1f ms | max|diff| / max|ref|: feats %.2e  logits %.2e  heatmap %.2e\n",
                    name.c_str(), gray.cols, gray.rows, ms, f.rel(), k.rel(), h.rel());
        EXPECT_LT(f.rel(), 1e-4) << name;
        EXPECT_LT(k.rel(), 1e-4) << name;
        EXPECT_LT(h.rel(), 1e-4) << name;

        const Detection d = net.Detect(gray);
        const Tensor& rk = ref.at("keypoints");
        const Tensor& rs = ref.at("scores");
        const Tensor& rd = ref.at("descriptors");
        const size_t nr = size_t(rk.shape[0]);
        std::map<std::pair<int, int>, size_t> mine;
        for (size_t i = 0; i < d.keypoints.size(); ++i) mine[{int(d.keypoints[i].x), int(d.keypoints[i].y)}] = i;
        size_t matched = 0;
        double min_cos = 1, max_dscore = 0;
        for (size_t j = 0; j < nr; ++j) {
            const auto it = mine.find({int(rk.f32[2 * j]), int(rk.f32[2 * j + 1])});
            if (it == mine.end()) continue;
            ++matched;
            const float* a = d.descriptors.ptr<float>(int(it->second));
            double c = 0;
            for (int q = 0; q < 64; ++q) c += double(a[q]) * rd.f32[j * 64 + size_t(q)];
            min_cos = std::min(min_cos, c);
            max_dscore = std::max(max_dscore, double(std::fabs(d.scores[it->second] - rs.f32[j])));
        }
        std::printf("[%s] keypoints ours %zu ref %zu, same pixel %zu (%.3f); min descriptor cosine %.6f; "
                    "max |score diff| %.2e\n",
                    name.c_str(), d.keypoints.size(), nr, matched, double(matched) / double(nr), min_cos, max_dscore);
        EXPECT_GE(double(matched) / double(nr), 0.99) << name;
        EXPECT_GE(min_cos, 0.9999) << name;
        ++checked;
    }
    EXPECT_GE(checked, 1);
}

INSTANTIATE_TEST_SUITE_P(Engines, XFeatReference, ::testing::Values(Engine::kOpenCvDnn, Engine::kGemmReference),
                         [](const auto& i) { return std::string(EngineName(i.param)); });
