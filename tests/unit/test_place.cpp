// 2.1 retrieval (PERCEPTION_2.1.md §11): PlaceIndex (exact + cv::ANNIndex) and PlaceRecognizer end to end against
// the EigenPlaces reference export (SMOKE §4r). The reference tests SKIP without $ARIADNE_EIGENPLACES_REF and FAIL
// without it under ARIADNE_REQUIRE_MODELS=1.
#include "core/random/philox.hpp"
#include "perception/features/xfeat/atb.hpp"
#include "perception/place/place_index.hpp"
#include "perception/place/place_recognizer.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace ariadne;
using namespace ariadne::perception;
namespace fs = std::filesystem;

namespace {

/// A uniformly random unit vector; one Philox site per vector (a site holds 262144 words, D-philox).
std::vector<float> RandomUnit(uint64_t seed, uint64_t k, int dim) {
    core::PhiloxStream s(seed, k, 0);
    std::vector<float> v(static_cast<size_t>(dim));
    double n = 0;
    for (float& x : v) {
        // Box-Muller from two uniforms: an isotropic direction.
        const double u1 = ((s.NextBits32() >> 8) + 1) * (1.0 / 16777217.0), u2 = (s.NextBits32() >> 8) * (1.0 / 16777216.0);
        x = float(std::sqrt(-2 * std::log(u1)) * std::cos(2 * M_PI * u2));
        n += double(x) * x;
    }
    for (float& x : v) x = float(x / std::sqrt(n));
    return v;
}

std::vector<float> Perturb(const std::vector<float>& v, uint64_t seed, uint64_t k, double sigma) {
    std::vector<float> w = RandomUnit(seed, k, int(v.size()));
    double n = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        w[i] = float(v[i] + sigma * w[i]);
        n += double(w[i]) * w[i];
    }
    for (float& x : w) x = float(x / std::sqrt(n));
    return w;
}

std::string RefDir() {
    const char* d = std::getenv("ARIADNE_EIGENPLACES_REF");
    return d ? d : "";
}
bool Required() {
    const char* r = std::getenv("ARIADNE_REQUIRE_MODELS");
    return r && std::string(r) == "1";
}

}  // namespace

TEST(PlaceIndex, ExactFindsItselfAndHonoursTheFilter) {
    PlaceIndex idx(256);
    std::vector<std::vector<float>> v;
    for (uint64_t i = 0; i < 200; ++i) {
        v.push_back(RandomUnit(5, i, 256));
        idx.Add(1000 + i, v.back());
    }
    const auto hits = idx.Query(v[42], 3);
    ASSERT_EQ(hits.size(), 3u);
    EXPECT_EQ(hits[0].id, 1042u);
    EXPECT_NEAR(hits[0].cosine, 1.0f, 1e-5f);
    EXPECT_GE(hits[0].cosine, hits[1].cosine);
    const auto older = idx.Query(v[42], 3, [](uint64_t id) { return id < 1042; });
    for (const auto& h : older) EXPECT_LT(h.id, 1042u);
    EXPECT_TRUE(idx.Query(v[0], 5, [](uint64_t) { return false; }).empty());
}

// Annoy against exact on noisy queries: agreement, and repeatability within one process (same seed, same trees).
TEST(PlaceIndex, AnnoyAgreesWithExactAndRepeats) {
    PlaceIndex exact(256), ann(256, PlaceIndex::Backend::kAnnoy, 10, 7), ann2(256, PlaceIndex::Backend::kAnnoy, 10, 7);
    std::vector<std::vector<float>> v;
    for (uint64_t i = 0; i < 2000; ++i) {
        v.push_back(RandomUnit(6, i, 256));
        exact.Add(i, v.back());
        ann.Add(i, v.back());
        ann2.Add(i, v.back());
    }
    int agree = 0, n = 0, same = 0;
    for (size_t i = 0; i < v.size(); i += 10) {
        const auto q = Perturb(v[i], 60, i, 0.5);
        const auto e = exact.Query(q, 1), a = ann.Query(q, 1), b = ann2.Query(q, 1);
        ++n;
        agree += !a.empty() && a[0].id == e[0].id;
        same += !a.empty() && !b.empty() && a[0].id == b[0].id && a[0].cosine == b[0].cosine;
    }
    std::printf("annoy top-1 agreement with exact: %d / %d; repeat identical: %d / %d\n", agree, n, same, n);
    EXPECT_GE(agree, n * 9 / 10);
    EXPECT_EQ(same, n);
}

TEST(PlaceIndex, AddingAfterAQueryRebuilds) {
    PlaceIndex ann(256, PlaceIndex::Backend::kAnnoy, 4, 1);
    const auto a = RandomUnit(8, 0, 256), b = RandomUnit(8, 1, 256);
    ann.Add(1, a);
    EXPECT_EQ(ann.Query(a, 1).at(0).id, 1u);
    ann.Add(2, b);   // cv::ANNIndex cannot add after build: the index must rebuild from the full matrix
    EXPECT_EQ(ann.Query(b, 1).at(0).id, 2u);
    EXPECT_EQ(ann.Query(a, 1).at(0).id, 1u);   // and ids are not overwritten (addItems numbers from 0)
}

TEST(PlaceRecognizer, PreprocessingAndDescriptorMatchUpstream) {
    const std::string dir = RefDir();
    if (dir.empty() || !fs::exists(fs::path(dir) / "manifest.yaml")) {
        if (Required()) FAIL() << "ARIADNE_REQUIRE_MODELS=1 but no EigenPlaces export";
        GTEST_SKIP() << "no EigenPlaces export (set ARIADNE_EIGENPLACES_REF)";
    }
    int checked = 0;
    for (const auto& [w, h] : {std::pair<int, int>{376, 240}, std::pair<int, int>{752, 480}}) {
        const std::string size = std::to_string(w) + "x" + std::to_string(h);
        PlaceRecognizer rec({(fs::path(dir) / ("eigenplaces_r50_256_" + size + ".onnx")).string(), w, h, "classic"});
        for (const char* name : {"euroc_mh01_f200", "euroc_v101_f500"}) {
            const auto ref = xfeat::LoadAtb((fs::path(dir) / (std::string("ref_") + name + "_" + size + ".atb")).string());
            const cv::Mat gray = cv::imread((fs::path(dir) / (std::string("input_") + name + ".png")).string(),
                                            cv::IMREAD_GRAYSCALE);
            ASSERT_FALSE(gray.empty());
            const cv::Mat blob = PlaceRecognizer::Preprocess(gray, w, h);
            const auto& in = ref.at("input").f32;
            ASSERT_EQ(blob.total(), in.size());
            double maxd = 0;
            const float* b = blob.ptr<float>();
            for (size_t i = 0; i < in.size(); ++i) maxd = std::max(maxd, double(std::fabs(b[i] - in[i])));
            const auto d = rec.Describe(gray);
            double cs = 0;
            for (size_t i = 0; i < d.size(); ++i) cs += double(d[i]) * ref.at("descriptor").f32[i];
            std::printf("[%s %s] preprocessing max |diff| %.2e, end-to-end cosine %.7f\n", name, size.c_str(), maxd, cs);
            EXPECT_LE(maxd, 1e-5);
            EXPECT_GE(cs, 0.9999);
            ++checked;
        }
    }
    EXPECT_EQ(checked, 4);
}
