// The §7.4 threading contract (D91): dnn::PerThread, and every cv::dnn wrapper built on it. Two threads share ONE
// wrapper object, run CONCURRENTLY (released together), each on its own per-thread network, and both must produce
// outputs byte-identical to a single-thread reference computed first. Under the contract's second rule: OpenCV's
// own pool at 1 thread (dnn::kPoolOpenCvThreads), so the parallelism is the callers'. Measured without it
// (2026-10-22, D91): concurrent ENGINE_NEW ALIKED runs intermittently differ by up to 6e-5 px / 1.3e-6 per
// descriptor element (the shared pool partitions the work differently), with identical match indices. The model-backed tests use the exports in
// $ARIADNE_XFEAT_REF / $ARIADNE_EIGENPLACES_REF / $ARIADNE_ALIKED_LG_REF (SKIPPED without them, FAILED under
// ARIADNE_REQUIRE_MODELS=1).
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/dnn/per_thread.hpp"
#include "perception/features/xfeat/xfeat_net.hpp"
#include "perception/place/place_recognizer.hpp"

#include <gtest/gtest.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace ariadne::perception;

namespace {

std::string Env(const char* v) {
    const char* d = std::getenv(v);
    return d ? d : "";
}
bool Required() { return Env("ARIADNE_REQUIRE_MODELS") == "1"; }
#define REQUIRE_DIR(dir, var)                                                       \
    if ((dir).empty() || !fs::exists(fs::path(dir) / "manifest.yaml")) {          \
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no export in " var; \
        GTEST_SKIP() << "no export (set " var ")";                                 \
    }

using Bytes = std::vector<unsigned char>;
void Append(Bytes& b, const void* p, size_t n) {
    const auto* c = static_cast<const unsigned char*>(p);
    b.insert(b.end(), c, c + n);
}
void Append(Bytes& b, const cv::Mat& m) {
    ASSERT_TRUE(m.isContinuous());
    Append(b, m.data, m.total() * m.elemSize());
}

/// OpenCV's pool size for the test's scope (cv::setNumThreads is process-wide).
struct PoolThreads {
    explicit PoolThreads(int n) : saved(cv::getNumThreads()) { cv::setNumThreads(n); }
    ~PoolThreads() { cv::setNumThreads(saved); }
    int saved;
};

/// run() twice on each of two threads released together; every result must equal `ref` byte for byte.
void Concurrently(const std::function<Bytes()>& run, const Bytes& ref, int reps = 2) {
    std::atomic<int> ready{0};
    std::vector<Bytes> got[2];
    std::string err[2];
    auto body = [&](int t) {
        ++ready;
        while (ready.load() < 2) std::this_thread::yield();   // both threads in forward at once
        try {
            for (int r = 0; r < reps; ++r) got[t].push_back(run());
        } catch (const std::exception& e) {
            err[t] = e.what();
        }
    };
    std::thread a(body, 0), b(body, 1);
    a.join();
    b.join();
    for (int t = 0; t < 2; ++t) {
        EXPECT_TRUE(err[t].empty()) << err[t];
        ASSERT_EQ(int(got[t].size()), reps);
        for (const Bytes& g : got[t]) {
            ASSERT_EQ(g.size(), ref.size());
            EXPECT_EQ(std::memcmp(g.data(), ref.data(), ref.size()), 0) << "thread " << t << " differs from the reference";
        }
    }
    std::printf("2 threads x %d runs: byte-identical to the single-thread reference (%zu bytes)\n", reps, ref.size());
}

}  // namespace

TEST(PerThread, OneInstancePerThreadStableAcrossCalls) {
    std::atomic<int> made{0};
    const dnn::PerThread<int> pt([&] { return std::make_unique<int>(++made); });
    int& mine = pt.Local();
    EXPECT_EQ(&mine, &pt.Local());   // same thread, same instance
    int* other = nullptr;
    std::thread([&] { other = &pt.Local(); }).join();
    EXPECT_NE(other, &mine);
    EXPECT_EQ(made.load(), 2);
    EXPECT_EQ(pt.Instances(), 2u);
    EXPECT_THROW(dnn::PerThread<int>(nullptr), std::invalid_argument);
}

TEST(DnnContract, XFeatSharedAcrossTwoThreadsIsByteIdentical) {
    const PoolThreads pool(dnn::kPoolOpenCvThreads);
    const std::string dir = Env("ARIADNE_XFEAT_REF");
    REQUIRE_DIR(dir, "ARIADNE_XFEAT_REF");
    const xfeat::XFeatNet net((fs::path(dir) / "xfeat_weights.atb").string());
    const cv::Mat gray = cv::imread((fs::path(dir) / "input_euroc_mh01_f200.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());
    auto run = [&] {
        const xfeat::Detection d = net.Detect(gray);
        Bytes b;
        Append(b, d.keypoints.data(), d.keypoints.size() * sizeof(cv::Point2f));
        Append(b, d.scores.data(), d.scores.size() * sizeof(float));
        Append(b, d.descriptors);
        return b;
    };
    const Bytes ref = run();
    ASSERT_FALSE(ref.empty());
    Concurrently(run, ref);
}

TEST(DnnContract, PlaceRecognizerSharedAcrossTwoThreadsIsByteIdentical) {
    const PoolThreads pool(dnn::kPoolOpenCvThreads);
    const std::string dir = Env("ARIADNE_EIGENPLACES_REF");
    REQUIRE_DIR(dir, "ARIADNE_EIGENPLACES_REF");
    const PlaceRecognizer rec({(fs::path(dir) / "eigenplaces_r50_256_376x240.onnx").string(), 376, 240, "classic"});
    const cv::Mat gray = cv::imread((fs::path(dir) / "input_euroc_mh01_f200.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());
    auto run = [&] {
        const auto d = rec.Describe(gray);
        Bytes b;
        Append(b, d.data(), d.size() * sizeof(float));
        return b;
    };
    const Bytes ref = run();
    Concurrently(run, ref);
    EXPECT_EQ(rec.Record().engine, "cv_dnn_classic");
}

TEST(DnnContract, HardPairMatcherSharedAcrossTwoThreadsIsByteIdentical) {
    const PoolThreads pool(dnn::kPoolOpenCvThreads);
    const std::string dir = Env("ARIADNE_ALIKED_LG_REF");
    REQUIRE_DIR(dir, "ARIADNE_ALIKED_LG_REF");
    HardPairParams p;
    p.aliked_onnx = (fs::path(dir) / "aliked_n16_k1024_752x480.onnx").string();
    p.lightglue_onnx = (fs::path(dir) / "lightglue_aliked.onnx").string();
    const HardPairMatcher m(p);
    const cv::Mat a = cv::imread((fs::path(dir) / "input_mh01_f200.png").string(), cv::IMREAD_GRAYSCALE),
                  b = cv::imread((fs::path(dir) / "input_mh01_f205.png").string(), cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(a.empty() || b.empty());
    auto run = [&] {
        const HardPairFeatures fa = m.Extract(a), fb = m.Extract(b);
        const auto mt = m.Match(fa, fb);
        Bytes out;
        for (const auto& k : fa.keypoints) Append(out, &k, sizeof(k));
        Append(out, fa.descriptors);
        for (const auto& x : mt) Append(out, &x, sizeof(x));
        return out;
    };
    const Bytes ref = run();
    Concurrently(run, ref);
    ASSERT_EQ(m.Records().size(), 2u);
    EXPECT_EQ(m.Records()[0].engine, "cv_dnn_new");
}

