// SMOKE §4r (perception.md §11.4): EigenPlaces R50-256 exported to ONNX and run by cv::dnn, against upstream's
// own PyTorch descriptors on the same preprocessed input tensors.
//
// Needs the export (python/model_tools/eigenplaces_reference.py) in $ARIADNE_EIGENPLACES_REF: SKIPPED without it,
// FAILED without it under ARIADNE_REQUIRE_MODELS=1 (as test_xfeat_reference.cpp).
#include "perception/features/xfeat/atb.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ariadne::perception::xfeat::LoadAtb;
using ariadne::perception::xfeat::Tensor;

namespace {

std::string RefDir(const char* var = "ARIADNE_EIGENPLACES_REF") {
    const char* d = std::getenv(var);
    return d ? d : "";
}
bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}

cv::Mat Blob(const Tensor& t) {
    std::vector<int> sz(t.shape.begin(), t.shape.end());
    return cv::Mat(int(sz.size()), sz.data(), CV_32F, const_cast<float*>(t.f32.data())).clone();
}

double Cosine(const cv::Mat& a, const std::vector<float>& b) {
    const cv::Mat f = a.reshape(1, 1);
    double s = 0, na = 0, nb = 0;
    for (int i = 0; i < f.cols; ++i) {
        s += double(f.at<float>(0, i)) * b[size_t(i)];
        na += double(f.at<float>(0, i)) * f.at<float>(0, i);
        nb += double(b[size_t(i)]) * b[size_t(i)];
    }
    return s / std::sqrt(na * nb);
}

double ForwardMs(cv::dnn::Net& net, const cv::Mat& in, cv::Mat& out) {
    double best = 1e18;
    for (int i = 0; i < 3; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        net.setInput(in);
        out = net.forward();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

// Every ref_*_<size>.atb in dir through the <size> export: cosine >= 0.9999 against upstream's descriptor.
void CheckExport(const std::string& dir, int engine, const std::vector<std::string>& sizes, int expect) {
    const char* ename = engine == cv::dnn::ENGINE_NEW ? "ENGINE_NEW" : "ENGINE_CLASSIC";
    int checked = 0;
    for (const std::string& size : sizes) {
        const fs::path onnx = fs::path(dir) / (std::string("eigenplaces_r50_256_") + size + ".onnx");
        cv::dnn::Net net;
        try {
            net = cv::dnn::readNetFromONNX(onnx.string(), engine);
        } catch (const cv::Exception& e) {
            FAIL() << ename << " cannot import " << onnx.filename() << ": " << e.what();
        }
        for (const auto& e : fs::directory_iterator(dir)) {
            const std::string fn = e.path().filename().string();
            if (fn.rfind("ref_", 0) != 0 || fn.find(std::string("_") + size + ".atb") == std::string::npos) continue;
            const auto ref = LoadAtb(e.path().string());
            const cv::Mat in = Blob(ref.at("input"));
            cv::Mat out;
            const int saved = cv::getNumThreads();
            cv::setNumThreads(1);
            const double ms1 = ForwardMs(net, in, out);
            cv::setNumThreads(8);
            const double ms8 = ForwardMs(net, in, out);
            cv::setNumThreads(saved);
            const double cs = Cosine(out, ref.at("descriptor").f32);
            std::printf("[%s %s %s] cosine %.7f | |d| %.6f | forward %.0f ms (1 thread), %.0f ms (8)\n", ename, size.c_str(),
                        fn.c_str(), cs, cv::norm(out), ms1, ms8);
            EXPECT_EQ(int(out.total()), 256);
            EXPECT_GE(cs, 0.9999) << fn;
            ++checked;
        }
    }
    EXPECT_EQ(checked, expect);
}

}  // namespace

#define REQUIRE_EP_EXPORT(dir, var)                                                         \
    if ((dir).empty() || !fs::exists(fs::path(dir) / "manifest.yaml")) {                  \
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no EigenPlaces export in " var; \
        GTEST_SKIP() << "no EigenPlaces export (set " var ")";                             \
    }

class EigenPlacesReference : public ::testing::TestWithParam<int> {};

TEST_P(EigenPlacesReference, OnnxThroughCvDnnMatchesUpstream) {
    const std::string dir = RefDir();
    REQUIRE_EP_EXPORT(dir, "ARIADNE_EIGENPLACES_REF");
    CheckExport(dir, GetParam(), {"752x480", "376x240"}, 4);
}

// The sim rig (L3): 1280x720 RGB frames, the native size and the half-size default candidate.
TEST_P(EigenPlacesReference, SimRigOnnxThroughCvDnnMatchesUpstream) {
    const std::string dir = RefDir("ARIADNE_EIGENPLACES_SIM_REF");
    REQUIRE_EP_EXPORT(dir, "ARIADNE_EIGENPLACES_SIM_REF");
    CheckExport(dir, GetParam(), {"1280x720", "640x360"}, 4);
}

INSTANTIATE_TEST_SUITE_P(Engines, EigenPlacesReference,
                         ::testing::Values(int(cv::dnn::ENGINE_NEW), int(cv::dnn::ENGINE_CLASSIC)),
                         [](const auto& i) { return i.param == cv::dnn::ENGINE_NEW ? "ENGINE_NEW" : "ENGINE_CLASSIC"; });
