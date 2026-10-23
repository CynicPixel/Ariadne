// ---------------------------------------------------------------------------
// ariadne_perf_bench: the pipeline profile's DNN bench (A2, 2026-10-24; M's delegation; MEASUREMENT ONLY - it calls
// the shipped wrappers and the shipped verify path, it changes none of them). Two modes:
//
// TIMING (one process; its OpenCV pool is OPENCV_FOR_THREADS_NUM, read back with cv::getNumThreads and recorded):
//   ariadne_perf_bench timing --img-a PNG --img-b PNG --aliked WxH=ONNX [...] --lightglue ONNX
//       --place WxH=ONNX [...] [--ks 1024,512,256] [--reps 20] [--warmup 3] [--label L] --out DIR
//   DIR/timing.csv   label, stage, variant, engine, threads, p50_ms, p95_ms, mean_ms, cpu_mean_ms, n, out, digest1,
//                    digest2. cpu_mean_ms is the PROCESS CPU time per call (std::clock, all threads): cpu/wall is the
//                    pool's effective parallelism. digest1/digest2 = FNV-1a of the outputs of two calls (keypoints +
//                    descriptors, matches, or the place descriptor): equal = run-to-run bitwise stable at this count.
//   Stages: aliked_extract (per image, every --aliked size: cv::ALIKED, ENGINE_NEW), lightglue_match (per pair, the
//   1280x720 features truncated to each K by ALIKED score), eigenplaces_describe (per image, every --place, both
//   engines). DIR/engines.csv: every model under ENGINE_CLASSIC and ENGINE_NEW, load + one forward, ok or the error.
//   DIR/profile.csv: per-operator-type time of one forward (DNN_PROFILE_SUMMARY, ENGINE_NEW; the classic engine for
//   EigenPlaces too; LightGlue at K 1024 and 512 per op, DNN_PROFILE_DETAILED, so the per-layer cost can be summed from
//   the node names). DIR/build.txt: the CPU features line and the build information's ONNX Runtime lines.
//
// GATE (ariadne_perf_bench gate ...): externally made matches (scripts/lg_adaptive.py) through the same verify path.
//
// KNOBS (the speed knobs' accuracy cost on real verified-or-rejected pairs; the fleet's verify path, verbatim):
//   ariadne_perf_bench knobs --kf-dir DIR --pairs CSV --collab-config YAML --seed S --aliked WxH=ONNX [...]
//       --lightglue ONNX --settings 1280x720:1024,1280x720:512,640x360:1024 [--stores DIR] --out DIR
//   --kf-dir: ariadne_perf_kfdump's keyframes (RGB, masked depth, pinhole). --pairs: agent_a,kf_a,agent_b,kf_b plus
//   the production run's columns (passed through). Per keyframe and size: ALIKED on the RGB resized with INTER_AREA,
//   keypoints mapped back to 1280x720 pixel centres (x = s (x' + 0.5) - 0.5), sorted by score, each keypoint's 3D
//   point from the depth at its rounded pixel (CollabFeatureExtractor::ExtractLocal's rule). Per pair and setting
//   (size, K = the top-K by score): CollabFleet::Response (the configured local encoding) -> decode -> LightGlue ->
//   VerifyMatched (the fleet's per-pair seed) -> the D71 gate (GatedInterAgentEdge) -> the reason, as the fleet logs it.
//   --dump-features DIR: the 1280x720 K=1024 features of every keyframe as LightGlue sees them (DumpFeat), for GATE.
//   --stores: also the setting "store" - the pass-A store's own features (the production inputs) through the same
//   path, so knobs.csv shows how exactly this replay reproduces the production verdicts.
//   DIR/knobs.csv: the pair, its production columns, setting, n_a, n_b, matches, essential_inliers, pnp_inliers,
//   reason, ms_match_verify. DIR/fidelity.csv (with --stores): per keyframe, the 1280x720 K=1024 features recomputed
//   here vs the store's: count, matched within 1e-3 px, max px distance.
// ---------------------------------------------------------------------------
#include "collaboration/broker/edge_verifier.hpp"
#include "collaboration/retrieval/local_descriptor_codec.hpp"
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/place/place_recognizer.hpp"
#include "runtime/agent/collab_fleet.hpp"
#include "runtime/agent/collab_frontend.hpp"
#include "runtime/agent/collab_store.hpp"
#include "slam/robust/joint_robust.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ac = ariadne::collaboration;
using namespace ariadne;

namespace {

struct Stats {
    double p50 = 0, p95 = 0, mean = 0, cpu_mean = 0;
    int n = 0;
};

Stats Time(const std::function<void()>& f, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) f();
    std::vector<double> ms;
    double cpu = 0;
    for (int i = 0; i < reps; ++i) {
        const std::clock_t c0 = std::clock();
        const auto t0 = std::chrono::steady_clock::now();
        f();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        cpu += 1000.0 * double(std::clock() - c0) / CLOCKS_PER_SEC;
    }
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.n = reps;
    auto q = [&](double p) { return ms[std::min(ms.size() - 1, size_t(p * double(ms.size() - 1) + 0.5))]; };
    s.p50 = q(0.50), s.p95 = q(0.95);
    for (double v : ms) s.mean += v / reps;
    s.cpu_mean = cpu / reps;
    return s;
}

uint64_t Fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
}
uint64_t Digest(const perception::HardPairFeatures& f) {
    uint64_t h = 1469598103934665603ull;
    for (const auto& k : f.keypoints) h = Fnv(&k.pt, sizeof k.pt, h), h = Fnv(&k.response, sizeof k.response, h);
    for (int r = 0; r < f.descriptors.rows; ++r) h = Fnv(f.descriptors.ptr(r), f.descriptors.cols * 4, h);
    return h;
}
uint64_t Digest(const std::vector<cv::DMatch>& m) {
    uint64_t h = 1469598103934665603ull;
    for (const auto& d : m) h = Fnv(&d.queryIdx, 4, h), h = Fnv(&d.trainIdx, 4, h), h = Fnv(&d.distance, 4, h);
    return h;
}
uint64_t Digest(const std::vector<float>& v) { return Fnv(v.data(), v.size() * 4); }
std::string Hex(uint64_t h) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
    return b;
}

cv::Mat ReadRgb(const fs::path& p) {
    cv::Mat bgr = cv::imread(p.string(), cv::IMREAD_COLOR), rgb;
    if (bgr.empty()) throw std::runtime_error("cannot read " + p.string());
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

std::pair<int, int> ParseSize(const std::string& wxh) {
    int w = 0, h = 0;
    if (std::sscanf(wxh.c_str(), "%dx%d", &w, &h) != 2) throw std::invalid_argument("size WxH: " + wxh);
    return {w, h};
}

/// "WxH=PATH" -> (WxH, PATH)
std::pair<std::string, std::string> Spec(const std::string& s) {
    const auto e = s.find('=');
    if (e == std::string::npos) throw std::invalid_argument("expected WxH=PATH: " + s);
    return {s.substr(0, e), s.substr(e + 1)};
}

std::vector<std::string> Split(const std::string& s, char c) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, c);)
        if (!t.empty()) out.push_back(t);
    return out;
}

/// ALIKED at the export's size on the 1280x720 RGB: resized with INTER_AREA when smaller, keypoints mapped back to
/// 1280x720 pixel centres, sorted by score (descending, stable).
perception::HardPairFeatures Extract(const perception::HardPairMatcher& m, int w, int h, const cv::Mat& rgb) {
    cv::Mat in = rgb;
    if (rgb.cols != w || rgb.rows != h) cv::resize(rgb, in, cv::Size(w, h), 0, 0, cv::INTER_AREA);
    perception::HardPairFeatures f = m.ExtractRgb(in);
    const float sx = float(rgb.cols) / float(w), sy = float(rgb.rows) / float(h);
    for (auto& k : f.keypoints) k.pt = {sx * (k.pt.x + 0.5f) - 0.5f, sy * (k.pt.y + 0.5f) - 0.5f};
    f.size = rgb.size();
    return f;
}

perception::HardPairFeatures TopK(const perception::HardPairFeatures& f, int k) {
    std::vector<int> idx(f.keypoints.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return f.keypoints[size_t(a)].response > f.keypoints[size_t(b)].response; });
    if (int(idx.size()) > k) idx.resize(size_t(k));
    perception::HardPairFeatures o;
    o.size = f.size;
    o.descriptors.create(int(idx.size()), f.descriptors.cols, CV_32F);
    for (size_t i = 0; i < idx.size(); ++i) {
        o.keypoints.push_back(f.keypoints[size_t(idx[i])]);
        f.descriptors.row(idx[i]).copyTo(o.descriptors.row(int(i)));
    }
    return o;
}

// ---------------------------------------------------------------- timing

int Timing(std::map<std::string, std::vector<std::string>>& a) {
    const int reps = std::stoi(a["--reps"].back()), warmup = std::stoi(a["--warmup"].back());
    const std::string label = a["--label"].back();
    const fs::path out = a["--out"].back();
    fs::create_directories(out);
    const int threads = cv::getNumThreads();
    std::ofstream t(out / "timing.csv"), eng(out / "engines.csv"), prof(out / "profile.csv");
    t << "label,stage,variant,engine,threads,p50_ms,p95_ms,mean_ms,cpu_mean_ms,n,out,digest1,digest2\n";
    eng << "model,engine,ok,detail\n";
    prof << "label,model,engine,threads,op,time_ms,count\n";
    auto row = [&](const std::string& stage, const std::string& var, const std::string& engine, const Stats& s,
                   size_t n_out, uint64_t d1, uint64_t d2) {
        t << label << ',' << stage << ',' << var << ',' << engine << ',' << threads << ',' << s.p50 << ',' << s.p95
          << ',' << s.mean << ',' << s.cpu_mean << ',' << s.n << ',' << n_out << ',' << Hex(d1) << ',' << Hex(d2)
          << '\n';
        t.flush();
        std::printf("%-22s %-14s %-8s threads %2d  p50 %8.1f ms  cpu %8.1f ms  out %zu  stable %s\n", stage.c_str(),
                    var.c_str(), engine.c_str(), threads, s.p50, s.cpu_mean, n_out, d1 == d2 ? "yes" : "NO");
        std::fflush(stdout);
    };
    auto profile = [&](const std::string& model, const std::string& engine, cv::dnn::Net& net) {
        std::vector<std::string> names, ms, counts;
        net.getPerfProfile(names, ms, counts);
        for (size_t i = 0; i < names.size(); ++i)
            prof << label << ',' << model << ',' << engine << ',' << threads << ",\"" << names[i] << "\"," << ms[i]
                 << ',' << (i < counts.size() ? counts[i] : "") << '\n';
        if (names.empty()) prof << label << ',' << model << ',' << engine << ',' << threads << ",(empty profile),,\n";
    };
    {   // what this OpenCV build has (ORT is a build option, WITH_ONNXRUNTIME)
        std::ofstream b(out / "build.txt");
        b << "CPU_FEATURES_LINE=" << cv::getCPUFeaturesLine() << "\nthreads=" << threads << "\n";
        std::stringstream bi(cv::getBuildInformation());
        for (std::string l; std::getline(bi, l);)
            if (l.find("ONNX") != std::string::npos || l.find("Parallel framework") != std::string::npos ||
                l.find("Version control") != std::string::npos || l.find("HAL") != std::string::npos)
                b << l << "\n";
    }
    const cv::Mat ia = ReadRgb(a["--img-a"].back()), ib = ReadRgb(a["--img-b"].back());
    cv::Mat ia_bgr;
    cv::cvtColor(ia, ia_bgr, cv::COLOR_RGB2BGR);
    const std::string lg = a["--lightglue"].back();

    perception::HardPairFeatures fa_full, fb_full;
    std::unique_ptr<perception::HardPairMatcher> full;
    for (const auto& spec : a["--aliked"]) {
        const auto [size, onnx] = Spec(spec);
        const auto [w, h] = ParseSize(size);
        perception::HardPairParams p;
        p.aliked_onnx = onnx, p.lightglue_onnx = lg, p.width = w, p.height = h;
        auto m = std::make_unique<perception::HardPairMatcher>(p);
        perception::HardPairFeatures f1 = Extract(*m, w, h, ia);
        const Stats s = Time([&] { Extract(*m, w, h, ia); }, warmup, reps);
        const perception::HardPairFeatures f2 = Extract(*m, w, h, ia);
        row("aliked_extract", size, "cv_dnn_new", s, f1.keypoints.size(), Digest(f1), Digest(f2));
        {   // per-op profile of the raw graph, the wrapper's input (blobFromImage 1/255 swapRB at the export size)
            cv::Mat in;
            cv::resize(ia_bgr, in, cv::Size(w, h), 0, 0, cv::INTER_AREA);
            cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx, cv::dnn::ENGINE_NEW);
            net.setProfilingMode(cv::dnn::DNN_PROFILE_SUMMARY);
            net.setInput(cv::dnn::blobFromImage(in, 1.0 / 255, cv::Size(w, h), cv::Scalar(), true));
            std::vector<cv::Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());
            net.forward(outs, net.getUnconnectedOutLayersNames());   // the second forward: the first builds
            profile("aliked_" + size, "cv_dnn_new", net);
            // The bare forward under each engine (cv::ALIKED itself is pinned to ENGINE_NEW by HardPairMatcher):
            // out = the max |difference| of every output from ENGINE_NEW's, x1e6 (0 = identical), so a faster
            // engine is only a lever if it computes the same thing.
            const cv::Mat blob = cv::dnn::blobFromImage(in, 1.0 / 255, cv::Size(w, h), cv::Scalar(), true);
            for (const auto& [ename, e] : {std::pair{"cv_dnn_new", int(cv::dnn::ENGINE_NEW)},
                                           std::pair{"cv_dnn_classic", int(cv::dnn::ENGINE_CLASSIC)}}) {
                try {
                    cv::dnn::Net n2 = cv::dnn::readNetFromONNX(onnx, e);
                    std::vector<cv::Mat> o2;
                    auto fwd = [&] { n2.setInput(blob); n2.forward(o2, n2.getUnconnectedOutLayersNames()); };
                    fwd();
                    double diff = 0;
                    for (size_t i = 0; i < o2.size() && i < outs.size(); ++i)
                        if (o2[i].total() == outs[i].total())
                            diff = std::max(diff, cv::norm(o2[i].reshape(1, 1), outs[i].reshape(1, 1), cv::NORM_INF));
                        else
                            diff = 1e30;
                    uint64_t d1 = 1469598103934665603ull, d2 = d1;
                    for (const auto& o : o2) d1 = Fnv(o.data, o.total() * o.elemSize(), d1);
                    const Stats s = Time(fwd, warmup, reps);
                    for (const auto& o : o2) d2 = Fnv(o.data, o.total() * o.elemSize(), d2);
                    row("aliked_forward", size, ename, s, size_t(std::llround(std::min(diff, 1e12) * 1e6)), d1, d2);
                } catch (const cv::Exception& x) {
                    std::printf("aliked_forward %s %s: %s\n", size.c_str(), ename, x.what());
                }
            }
        }
        if (w == 1280 && h == 720) {
            fa_full = f1, fb_full = Extract(*m, w, h, ib);
            full = std::move(m);
        }
    }
    if (!full) throw std::invalid_argument("timing needs a 1280x720 --aliked (LightGlue runs on its features)");
    for (const auto& ks : Split(a["--ks"].back(), ',')) {
        const int k = std::stoi(ks);
        const perception::HardPairFeatures A = TopK(fa_full, k), B = TopK(fb_full, k);
        const std::vector<cv::DMatch> m1 = full->Match(A, B);
        const Stats s = Time([&] { full->Match(A, B); }, warmup, reps);
        const std::vector<cv::DMatch> m2 = full->Match(A, B);
        row("lightglue_match", "K" + ks, "cv_dnn_new", s, m1.size(), Digest(m1), Digest(m2));
        if (k == 1024 || k == 512) {   // per-op DETAILED: node names carry the layer (transformers.<i>), so a layer's
                                        // share of the forward is summed from them (the adaptive-depth payoff, M)
            cv::dnn::Net net = cv::dnn::readNetFromONNX(lg, cv::dnn::ENGINE_NEW);
            net.setProfilingMode(cv::dnn::DNN_PROFILE_DETAILED);
            const cv::Mat ka = perception::HardPairMatcher::NormalizeUpstream(A.keypoints, A.size).reshape(1, std::vector<int>{1, k, 2});
            const cv::Mat kb = perception::HardPairMatcher::NormalizeUpstream(B.keypoints, B.size).reshape(1, std::vector<int>{1, k, 2});
            net.setInput(ka, "kpts0"), net.setInput(kb, "kpts1");
            net.setInput(A.descriptors.reshape(1, std::vector<int>{1, k, 128}), "desc0");
            net.setInput(B.descriptors.reshape(1, std::vector<int>{1, k, 128}), "desc1");
            std::vector<cv::Mat> outs;
            net.forward(outs, net.getUnconnectedOutLayersNames());
            net.forward(outs, net.getUnconnectedOutLayersNames());
            profile("lightglue_K" + ks, "cv_dnn_new", net);
        }
    }
    for (const auto& spec : a["--place"]) {
        const auto [size, onnx] = Spec(spec);
        const auto [w, h] = ParseSize(size);
        for (const char* engine : {"classic", "new"}) {
            const perception::PlaceRecognizer rec({onnx, w, h, engine});
            const std::vector<float> d1 = rec.DescribeRgb(ia);
            const Stats s = Time([&] { rec.DescribeRgb(ia); }, warmup, reps);
            const std::vector<float> d2 = rec.DescribeRgb(ia);
            row("eigenplaces_describe", size, std::string("cv_dnn_") + engine, s, d1.size(), Digest(d1), Digest(d2));
            cv::Mat in;
            cv::resize(ia_bgr, in, cv::Size(w, h), 0, 0, cv::INTER_AREA);
            cv::dnn::Net net = cv::dnn::readNetFromONNX(
                onnx, std::string(engine) == "new" ? cv::dnn::ENGINE_NEW : cv::dnn::ENGINE_CLASSIC);
            net.setProfilingMode(cv::dnn::DNN_PROFILE_SUMMARY);
            net.setInput(cv::dnn::blobFromImage(in, 1.0 / 255, cv::Size(w, h), cv::Scalar(), true));
            net.forward();
            net.forward();
            profile("eigenplaces_" + size, std::string("cv_dnn_") + engine, net);
        }
    }
    // The engine matrix, verified rather than assumed (SMOKE §4s saw CLASSIC fail on opencv_extra's exports).
    std::vector<std::pair<std::string, std::string>> models;
    for (const auto& s : a["--aliked"]) models.push_back({"aliked_" + Spec(s).first, Spec(s).second});
    models.push_back({"lightglue", lg});
    for (const auto& s : a["--place"]) models.push_back({"eigenplaces_" + Spec(s).first, Spec(s).second});
    for (const auto& [name, onnx] : models) {
        for (const auto& [ename, e] : {std::pair{"cv_dnn_classic", int(cv::dnn::ENGINE_CLASSIC)},
                                       std::pair{"cv_dnn_new", int(cv::dnn::ENGINE_NEW)}}) {
            std::string detail = "load ok";
            bool ok = true;
            try {
                cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx, e);
                if (net.empty()) ok = false, detail = "empty net";
            } catch (const cv::Exception& x) {
                ok = false;
                detail = x.what();
            }
            std::replace(detail.begin(), detail.end(), '\n', ' ');
            std::replace(detail.begin(), detail.end(), '"', '\'');
            eng << name << ',' << ename << ',' << (ok ? 1 : 0) << ",\"" << detail.substr(0, 300) << "\"\n";
        }
    }
    return 0;
}

// ---------------------------------------------------------------- knobs

struct Kf {
    cv::Mat rgb, depth;   // RGB 1280x720; CV_32F metres, NaN = unusable
    std::array<float, 5> camera{};
};

Kf LoadKf(const fs::path& dir, int agent, uint64_t id, const std::map<std::pair<int, uint64_t>, std::array<float, 5>>& cams) {
    const std::string stem = "a" + std::to_string(agent) + "_kf" + std::to_string(id);
    Kf k;
    k.rgb = ReadRgb(dir / (stem + "_rgb.png"));
    std::ifstream f(dir / (stem + "_depth.f32"), std::ios::binary);
    int32_t rc[2] = {0, 0};
    f.read(reinterpret_cast<char*>(rc), sizeof rc);
    k.depth.create(rc[0], rc[1], CV_32F);
    for (int r = 0; r < rc[0]; ++r) f.read(reinterpret_cast<char*>(k.depth.ptr<float>(r)), rc[1] * 4);
    if (!f) throw std::runtime_error("short depth file " + stem);
    const auto c = cams.find({agent, id});
    if (c == cams.end()) throw std::runtime_error("no pinhole for " + stem);
    k.camera = c->second;
    return k;
}

/// The local features as CollabFeatureExtractor::ExtractLocal builds them (3D from the depth at the rounded pixel).
runtime::CollabLocalFeatures Local(const perception::HardPairFeatures& f, const Kf& k) {
    runtime::CollabLocalFeatures o;
    const float fx = k.camera[0], fy = k.camera[1], cx = k.camera[2], cy = k.camera[3];
    for (int i = 0; i < int(f.keypoints.size()); ++i) {
        const cv::Point2f& q = f.keypoints[size_t(i)].pt;
        o.keypoints.push_back(q.x);
        o.keypoints.push_back(q.y);
        const int u = int(std::lround(q.x)), v = int(std::lround(q.y));
        float z = NAN;
        if (u >= 0 && v >= 0 && u < k.depth.cols && v < k.depth.rows) z = k.depth.at<float>(v, u);
        const bool ok = std::isfinite(z) && z > 0.0f;
        o.points3d.push_back(ok ? float((q.x - cx) / fx * z) : NAN);
        o.points3d.push_back(ok ? float((q.y - cy) / fy * z) : NAN);
        o.points3d.push_back(ok ? z : NAN);
        const float* d = f.descriptors.ptr<float>(i);
        o.descriptors.insert(o.descriptors.end(), d, d + f.descriptors.cols);
    }
    return o;
}

/// The fleet's verify (CollabFleet h.verify), verbatim but for the logging: decode both responses, LightGlue,
/// VerifyMatched with the per-pair seed, then the D71 gate.
ac::VerifyResult Verify(const perception::HardPairMatcher& lg, const runtime::CollabConfig& cfg, uint64_t seed,
                        int agent_a, uint64_t kf_a, const core::FeatureResponseValue& fa, int agent_b, uint64_t kf_b,
                        const core::FeatureResponseValue& fb, ac::LocalEncoding enc) {
    auto hard = [&](const core::FeatureResponseValue& r) {
        perception::HardPairFeatures f;
        f.size = cv::Size(1280, 720);
        for (size_t i = 0; i + 1 < r.keypoints.size(); i += 2) f.keypoints.emplace_back(r.keypoints[i], r.keypoints[i + 1], 1.0f);
        const std::vector<float> d = ac::DecodeLocal(r.descriptors, enc);
        cv::Mat(int(d.size() / 128), 128, CV_32F, const_cast<float*>(d.data())).copyTo(f.descriptors);
        return f;
    };
    const perception::HardPairFeatures A = hard(fa), B = hard(fb);
    ac::VerifyResult v;
    if (A.keypoints.size() < 8 || B.keypoints.size() < 8) return v.reason = "too_few_features", v;
    const std::vector<cv::DMatch> m = lg.Match(A, B);
    std::vector<cv::Point2f> pa, pb;
    std::vector<int> ib;
    for (const auto& d : m) {
        pa.push_back(A.keypoints[size_t(d.queryIdx)].pt);
        pb.push_back(B.keypoints[size_t(d.trainIdx)].pt);
        ib.push_back(d.trainIdx);
    }
    const uint64_t vseed = seed * 1000003ull + uint64_t(agent_a) * 104729ull + kf_a * 7919ull + uint64_t(agent_b) * 31ull + kf_b;
    v = ac::VerifyMatched(pa, pb, ib, fb, cfg.verify, vseed);
    if (!v.ok) return v;
    slam::LoopGeometry geo;
    geo.rig.cam = {fb.camera[0], fb.camera[1], fb.camera[2], fb.camera[3]};
    geo.rig.baseline = fb.camera[4];
    for (const auto& q : v.inlier_points_B) geo.X_B.push_back(core::Vec3(q.x, q.y, q.z));
    for (const auto& q : v.inlier_pixels_A) geo.uv_A.push_back(core::Vec2(q.x, q.y));
    geo.T_A_B = v.T_A_B;
    if (!slam::GatedInterAgentEdge(0, 1, geo, cfg.gate)) v.ok = false, v.reason = "degenerate";
    return v;
}

/// One keyframe's features as LightGlue sees them in the fleet (descriptors AFTER the wire encoding's round trip):
/// int32 n, then float keypoints[2n] (1280x720 px), descriptors[128n], points3d[3n] (NaN = none), camera[5].
void DumpFeat(const fs::path& dir, int agent, uint64_t kf, const core::FeatureResponseValue& r, ac::LocalEncoding enc) {
    fs::create_directories(dir);
    const fs::path p = dir / ("a" + std::to_string(agent) + "_kf" + std::to_string(kf) + ".feat");
    if (fs::exists(p)) return;
    const std::vector<float> d = ac::DecodeLocal(r.descriptors, enc);
    const int32_t n = int32_t(r.keypoints.size() / 2);
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(&n), 4);
    f.write(reinterpret_cast<const char*>(r.keypoints.data()), r.keypoints.size() * 4);
    f.write(reinterpret_cast<const char*>(d.data()), d.size() * 4);
    f.write(reinterpret_cast<const char*>(r.points3d.data()), r.points3d.size() * 4);
    f.write(reinterpret_cast<const char*>(r.camera.data()), 5 * 4);
}

core::FeatureResponseValue ReadFeat(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    int32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    core::FeatureResponseValue r;
    std::vector<float> d(size_t(n) * 128);
    r.keypoints.resize(size_t(n) * 2), r.points3d.resize(size_t(n) * 3), r.camera.resize(5);
    f.read(reinterpret_cast<char*>(r.keypoints.data()), r.keypoints.size() * 4);
    f.read(reinterpret_cast<char*>(d.data()), d.size() * 4);
    f.read(reinterpret_cast<char*>(r.points3d.data()), r.points3d.size() * 4);
    f.read(reinterpret_cast<char*>(r.camera.data()), 5 * 4);
    if (!f) throw std::runtime_error("short feature file " + p.string());
    return r;
}

// ---------------------------------------------------------------- gate

/// GATE: matches made elsewhere (upstream LightGlue, adaptive or not: scripts/lg_adaptive.py) on the dumped features,
/// through the fleet's VerifyMatched (per-pair seed) + the D71 gate. --matches CSV: agent_a,kf_a,agent_b,kf_b,mode,
/// then any columns, the LAST one "i-j i-j ..." (A index - B index). DIR/gate.csv: the row without the match list,
/// + n_matches_here, essential_here, pnp_here, reason_here.
int Gate(std::map<std::string, std::vector<std::string>>& a) {
    const fs::path dir = a["--feat-dir"].back(), out = a["--out"].back();
    fs::create_directories(out);
    const runtime::CollabConfig cfg = runtime::LoadCollabConfig(a["--collab-config"].back());
    const uint64_t seed = std::stoull(a["--seed"].back());
    std::ifstream f(a["--matches"].back());
    std::ofstream g(out / "gate.csv");
    std::string line;
    std::getline(f, line);
    g << line.substr(0, line.rfind(',')) << ",n_matches_here,essential_here,pnp_here,reason_here\n";
    int rows = 0;
    while (std::getline(f, line)) {
        const auto c = Split(line, ',');
        const int aa = std::stoi(c[0]), ab = std::stoi(c[2]);
        const uint64_t ka = std::stoull(c[1]), kb = std::stoull(c[3]);
        const auto fa = ReadFeat(dir / ("a" + std::to_string(aa) + "_kf" + std::to_string(ka) + ".feat"));
        const auto fb = ReadFeat(dir / ("a" + std::to_string(ab) + "_kf" + std::to_string(kb) + ".feat"));
        std::vector<cv::Point2f> pa, pb;
        std::vector<int> ib;
        const std::string ml = line.substr(line.rfind(',') + 1);
        for (const auto& t : Split(ml, ' ')) {
            const auto d = t.find('-');
            const int i = std::stoi(t.substr(0, d)), j = std::stoi(t.substr(d + 1));
            pa.emplace_back(fa.keypoints[2 * size_t(i)], fa.keypoints[2 * size_t(i) + 1]);
            pb.emplace_back(fb.keypoints[2 * size_t(j)], fb.keypoints[2 * size_t(j) + 1]);
            ib.push_back(j);
        }
        const uint64_t vseed = seed * 1000003ull + uint64_t(aa) * 104729ull + ka * 7919ull + uint64_t(ab) * 31ull + kb;
        ac::VerifyResult v = ac::VerifyMatched(pa, pb, ib, fb, cfg.verify, vseed);
        if (v.ok) {
            slam::LoopGeometry geo;
            geo.rig.cam = {fb.camera[0], fb.camera[1], fb.camera[2], fb.camera[3]};
            geo.rig.baseline = fb.camera[4];
            for (const auto& q : v.inlier_points_B) geo.X_B.push_back(core::Vec3(q.x, q.y, q.z));
            for (const auto& q : v.inlier_pixels_A) geo.uv_A.push_back(core::Vec2(q.x, q.y));
            geo.T_A_B = v.T_A_B;
            if (!slam::GatedInterAgentEdge(0, 1, geo, cfg.gate)) v.ok = false, v.reason = "degenerate";
        }
        g << line.substr(0, line.rfind(',')) << ',' << v.matches << ',' << v.essential_inliers << ',' << v.pnp_inliers
          << ',' << v.reason << '\n';
        ++rows;
    }
    std::printf("gate: %d rows\n", rows);
    return 0;
}

int Knobs(std::map<std::string, std::vector<std::string>>& a) {
    const fs::path kfdir = a["--kf-dir"].back(), out = a["--out"].back();
    fs::create_directories(out);
    const runtime::CollabConfig cfg = runtime::LoadCollabConfig(a["--collab-config"].back());
    const ac::LocalEncoding enc = ac::ParseLocalEncoding(cfg.local_encoding);
    const uint64_t seed = std::stoull(a["--seed"].back());
    const std::string lgp = a["--lightglue"].back();

    std::map<std::pair<int, uint64_t>, std::array<float, 5>> cams;
    for (const auto& p : fs::directory_iterator(kfdir)) {
        const std::string n = p.path().filename().string();
        if (n.size() < 15 || n.substr(n.size() - 14) != "_keyframes.csv") continue;
        const int agent = std::stoi(n.substr(1));
        std::ifstream f(p.path());
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            const auto c = Split(line, ',');
            cams[{agent, std::stoull(c[0])}] = {std::stof(c[2]), std::stof(c[3]), std::stof(c[4]), std::stof(c[5]),
                                                std::stof(c[6])};
        }
    }
    struct Pair {
        int aa, ab;
        uint64_t ka, kb;
        std::string line;
    };
    std::vector<Pair> pairs;
    std::string header;
    {
        std::ifstream f(a["--pairs"].back());
        std::getline(f, header);
        for (std::string line; std::getline(f, line);) {
            const auto c = Split(line, ',');
            pairs.push_back({std::stoi(c[0]), std::stoi(c[2]), std::stoull(c[1]), std::stoull(c[3]), line});
        }
    }
    std::map<std::string, std::unique_ptr<perception::HardPairMatcher>> nets;   // size -> matcher
    std::map<std::string, std::pair<int, int>> dims;
    for (const auto& spec : a["--aliked"]) {
        const auto [size, onnx] = Spec(spec);
        const auto [w, h] = ParseSize(size);
        perception::HardPairParams p;
        p.aliked_onnx = onnx, p.lightglue_onnx = lgp, p.width = w, p.height = h;
        nets[size] = std::make_unique<perception::HardPairMatcher>(p);
        dims[size] = {w, h};
    }
    if (!nets.count("1280x720")) throw std::invalid_argument("knobs needs the 1280x720 --aliked (LightGlue + reference)");
    const perception::HardPairMatcher& lg = *nets["1280x720"];
    std::vector<std::pair<std::string, int>> settings;
    for (const auto& s : Split(a["--settings"].back(), ',')) {
        const auto c = s.find(':');
        settings.push_back({s.substr(0, c), std::stoi(s.substr(c + 1))});
        if (!nets.count(settings.back().first)) throw std::invalid_argument("setting without --aliked: " + s);
    }

    // Features per (agent, kf, size), computed once; ms per extraction recorded.
    std::map<std::tuple<int, uint64_t, std::string>, perception::HardPairFeatures> feats;
    std::map<std::pair<int, uint64_t>, Kf> kfs;
    std::ofstream ext(out / "extract.csv");
    ext << "agent,kf,size,threads,ms,n\n";
    for (const auto& p : pairs)
        for (const auto& [ag, id] : {std::pair{p.aa, p.ka}, std::pair{p.ab, p.kb}}) {
            if (kfs.count({ag, id})) continue;
            kfs[{ag, id}] = LoadKf(kfdir, ag, id, cams);
            for (const auto& [size, m] : nets) {
                const auto t0 = std::chrono::steady_clock::now();
                feats[{ag, id, size}] = Extract(*m, dims[size].first, dims[size].second, kfs[{ag, id}].rgb);
                ext << ag << ',' << id << ',' << size << ',' << cv::getNumThreads() << ','
                    << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << ','
                    << feats[{ag, id, size}].keypoints.size() << '\n';
            }
        }
    std::map<int, runtime::CollabStore> stores;
    if (a.count("--stores")) {
        std::ofstream fid(out / "fidelity.csv");
        fid << "agent,kf,n_store,n_here,matched_1e-3px,max_px\n";
        for (int ag : {0, 1}) stores[ag] = runtime::ReadCollabStore((fs::path(a["--stores"].back()) /
                                                                   ("store_agent" + std::to_string(ag) + ".bin")).string());
        for (const auto& [key, kf] : kfs) {
            const auto& st = stores[key.first];
            const auto it = std::find_if(st.records.begin(), st.records.end(), [&](const auto& r) { return r.id == key.second; });
            if (it == st.records.end()) { fid << key.first << ',' << key.second << ",missing,,,\n"; continue; }
            const auto& f = feats[{key.first, key.second, "1280x720"}];
            size_t matched = 0;
            double worst = 0;
            for (size_t i = 0; i + 1 < it->keypoints.size(); i += 2) {
                double best = 1e9;
                for (const auto& k : f.keypoints) best = std::min(best, double(std::hypot(k.pt.x - it->keypoints[i], k.pt.y - it->keypoints[i + 1])));
                matched += best < 1e-3;
                worst = std::max(worst, best);
            }
            fid << key.first << ',' << key.second << ',' << it->keypoints.size() / 2 << ',' << f.keypoints.size() << ','
                << matched << ',' << worst << '\n';
        }
    }
    std::ofstream k(out / "knobs.csv");
    k << header << ",setting,n_a,n_b,matches_here,essential_here,pnp_here,reason_here,ms_match_verify\n";
    auto put = [&](const Pair& p, const std::string& setting, const core::FeatureResponseValue& fa,
                   const core::FeatureResponseValue& fb) {
        const auto t0 = std::chrono::steady_clock::now();
        const ac::VerifyResult v = Verify(lg, cfg, seed, p.aa, p.ka, fa, p.ab, p.kb, fb, enc);
        k << p.line << ',' << setting << ',' << fa.keypoints.size() / 2 << ',' << fb.keypoints.size() / 2 << ','
          << v.matches << ',' << v.essential_inliers << ',' << v.pnp_inliers << ',' << v.reason << ','
          << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << '\n';
    };
    for (const auto& p : pairs) {
        if (!stores.empty()) {
            auto rec = [&](int ag, uint64_t id) -> const runtime::CollabStoreRecord& {
                const auto& st = stores[ag];
                const auto it = std::find_if(st.records.begin(), st.records.end(), [&](const auto& r) { return r.id == id; });
                if (it == st.records.end()) throw std::runtime_error("store has no kf " + std::to_string(id));
                return *it;
            };
            const auto& ra = rec(p.aa, p.ka);
            const auto& rb = rec(p.ab, p.kb);
            put(p, "store", runtime::CollabFleet::Response(p.ka, ra.keypoints, ra.descriptors, ra.points3d, ra.camera, enc),
                runtime::CollabFleet::Response(p.kb, rb.keypoints, rb.descriptors, rb.points3d, rb.camera, enc));
        }
        for (const auto& [size, kk] : settings) {
            const Kf& ka = kfs[{p.aa, p.ka}];
            const Kf& kb = kfs[{p.ab, p.kb}];
            const auto la = Local(TopK(feats[{p.aa, p.ka, size}], kk), ka);
            const auto lb = Local(TopK(feats[{p.ab, p.kb, size}], kk), kb);
            const auto ra = runtime::CollabFleet::Response(p.ka, la.keypoints, la.descriptors, la.points3d, ka.camera, enc);
            const auto rb = runtime::CollabFleet::Response(p.kb, lb.keypoints, lb.descriptors, lb.points3d, kb.camera, enc);
            put(p, size + ":" + std::to_string(kk), ra, rb);
            if (a.count("--dump-features") && size == "1280x720" && kk == 1024) {
                DumpFeat(fs::path(a["--dump-features"].back()), p.aa, p.ka, ra, enc);
                DumpFeat(fs::path(a["--dump-features"].back()), p.ab, p.kb, rb, enc);
            }
        }
        k.flush();
    }
    std::printf("knobs: %zu pairs x %zu settings%s, %zu keyframes, threads %d\n", pairs.size(), settings.size(),
                stores.empty() ? "" : " + store", kfs.size(), cv::getNumThreads());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s timing|knobs ... (see the header of ariadne_perf_bench.cpp)\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    std::map<std::string, std::vector<std::string>> a{
        {"--reps", {"20"}}, {"--warmup", {"3"}}, {"--ks", {"1024,512,256"}}, {"--label", {"run"}}};
    for (int i = 2; i + 1 < argc; i += 2) {
        auto& v = a[argv[i]];
        if (std::string(argv[i]) == "--aliked" || std::string(argv[i]) == "--place") v.push_back(argv[i + 1]);
        else v = {argv[i + 1]};
    }
    try {
        if (mode == "timing") {
            for (const char* k : {"--img-a", "--img-b", "--aliked", "--lightglue", "--place", "--out"})
                if (!a.count(k)) throw std::invalid_argument(std::string("timing needs ") + k);
            return Timing(a);
        }
        if (mode == "gate") {
            for (const char* k : {"--feat-dir", "--matches", "--collab-config", "--seed", "--out"})
                if (!a.count(k)) throw std::invalid_argument(std::string("gate needs ") + k);
            return Gate(a);
        }
        if (mode == "knobs") {
            for (const char* k : {"--kf-dir", "--pairs", "--collab-config", "--seed", "--aliked", "--lightglue",
                                  "--settings", "--out"})
                if (!a.count(k)) throw std::invalid_argument(std::string("knobs needs ") + k);
            return Knobs(a);
        }
        throw std::invalid_argument("mode timing|knobs|gate");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_perf_bench: %s\n", e.what());
        return 1;
    }
}
