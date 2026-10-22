// ariadne_dnn_timing (L5): single-process per-image latency of the learned front-end models at EuRoC and sim-rig
// sizes, p50 / p95 over --reps timed runs after --warmup untimed ones (the first forward builds the graph). One model
// at a time, one caller, OpenCV's pool pinned by --threads (cv::setNumThreads, recorded). Inputs are the reference
// frames in the exports' own directories, so every number is on a model the reference tests checked.
//
//   ariadne_dnn_timing --xfeat DIR --ep DIR --ep-sim DIR --al DIR --al-sim DIR [--threads 1] [--reps 30]
//                      [--warmup 3] --out timings.csv
//
// Rows: model, size, threads, p50_ms, p95_ms, mean_ms, n. XFeat needs sides that are multiples of 32, so its input
// is centre-cropped (752x480 -> 736x480, 1280x720 -> 1280x704; upstream's own rule is to resize to the multiple,
// the cost is the same). LightGlue is timed per PAIR, on ALIKED's own K=1024 features.
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/features/xfeat/xfeat_net.hpp"
#include "perception/place/place_recognizer.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace ariadne::perception;

namespace {

struct Stats {
    double p50 = 0, p95 = 0, mean = 0;
    int n = 0;
};

Stats Time(const std::function<void()>& f, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) f();
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.n = reps;
    auto q = [&](double p) { return ms[std::min(ms.size() - 1, size_t(p * double(ms.size() - 1) + 0.5))]; };
    s.p50 = q(0.50);
    s.p95 = q(0.95);
    for (double v : ms) s.mean += v / reps;
    return s;
}

cv::Mat Read(const fs::path& p, int flags) {
    cv::Mat m = cv::imread(p.string(), flags);
    if (m.empty()) throw std::runtime_error("cannot read " + p.string());
    return m;
}

cv::Mat Crop32(const cv::Mat& g) {
    const int w = g.cols / 32 * 32, h = g.rows / 32 * 32;
    return g(cv::Rect((g.cols - w) / 2, (g.rows - h) / 2, w, h)).clone();
}

}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a{{"--threads", "1"}, {"--reps", "30"}, {"--warmup", "3"}};
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    for (const char* k : {"--xfeat", "--ep", "--ep-sim", "--al", "--al-sim", "--out"}) {
        if (!a.count(k)) {
            std::fprintf(stderr, "usage: %s --xfeat DIR --ep DIR --ep-sim DIR --al DIR --al-sim DIR [--threads N] "
                                 "[--reps N] [--warmup N] --out CSV\n", argv[0]);
            return 2;
        }
    }
    try {
        const int threads = std::stoi(a["--threads"]), reps = std::stoi(a["--reps"]), warmup = std::stoi(a["--warmup"]);
        cv::setNumThreads(threads);
        std::ofstream out(a["--out"]);
        out << "model,size,threads,p50_ms,p95_ms,mean_ms,n\n";
        auto row = [&](const std::string& model, const std::string& size, const Stats& s) {
            out << model << ',' << size << ',' << cv::getNumThreads() << ',' << s.p50 << ',' << s.p95 << ',' << s.mean
                << ',' << s.n << '\n';
            out.flush();
            std::printf("%-34s %-9s threads %d  p50 %8.1f ms  p95 %8.1f ms  (n=%d)\n", model.c_str(), size.c_str(),
                        cv::getNumThreads(), s.p50, s.p95, s.n);
            std::fflush(stdout);
        };
        const fs::path xf = a["--xfeat"], ep = a["--ep"], eps = a["--ep-sim"], al = a["--al"], als = a["--al-sim"];
        const cv::Mat eu_gray = Read(ep / "input_euroc_mh01_f200.png", cv::IMREAD_GRAYSCALE);
        const cv::Mat sim_bgr = Read(eps / "input_sim_a0_f200.png", cv::IMREAD_COLOR);
        cv::Mat sim_rgb, sim_gray;
        cv::cvtColor(sim_bgr, sim_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(sim_rgb, sim_gray, cv::COLOR_RGB2GRAY);

        {   // XFeat (opencv_dnn_graph), Detect = forward + NMS + top-k + descriptor sampling
            const xfeat::XFeatNet net((xf / "xfeat_weights.atb").string());
            for (const cv::Mat& g : {Crop32(eu_gray), Crop32(sim_gray)}) {
                row("xfeat/opencv_dnn_graph", std::to_string(g.cols) + "x" + std::to_string(g.rows),
                    Time([&] { net.Detect(g); }, warmup, reps));
            }
        }
        for (const char* eng : {"classic", "new"}) {   // EigenPlaces, Describe = preprocessing + forward
            for (const auto& [dir, w, h, rgb] : {std::tuple{ep, 376, 240, false}, std::tuple{ep, 752, 480, false},
                                                 std::tuple{eps, 640, 360, true}, std::tuple{eps, 1280, 720, true}}) {
                const std::string size = std::to_string(w) + "x" + std::to_string(h);
                const PlaceRecognizer rec({(dir / ("eigenplaces_r50_256_" + size + ".onnx")).string(), w, h, eng});
                row(std::string("eigenplaces/cv_dnn_") + eng, size,
                    Time([&] { rgb ? rec.DescribeRgb(sim_rgb) : rec.Describe(eu_gray); }, warmup, reps));
            }
        }
        for (const auto& [dir, w, h, rgb, a0, b0] :
             {std::tuple{al, 752, 480, false, "input_mh01_f200.png", "input_mh01_f205.png"},
              std::tuple{als, 1280, 720, true, "input_sim_a0_f200.png", "input_sim_a0_f205.png"}}) {
            HardPairParams p;
            p.width = w;
            p.height = h;
            p.aliked_onnx = (dir / ("aliked_n16_k1024_" + std::to_string(w) + "x" + std::to_string(h) + ".onnx")).string();
            p.lightglue_onnx = (dir / "lightglue_aliked.onnx").string();
            const HardPairMatcher m(p);
            cv::Mat ia = Read(dir / a0, rgb ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE),
                    ib = Read(dir / b0, rgb ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
            if (rgb) {
                cv::cvtColor(ia, ia, cv::COLOR_BGR2RGB);
                cv::cvtColor(ib, ib, cv::COLOR_BGR2RGB);
            }
            auto ex = [&](const cv::Mat& im) { return rgb ? m.ExtractRgb(im) : m.Extract(im); };
            const std::string size = std::to_string(w) + "x" + std::to_string(h);
            row("aliked_n16_k1024/cv_dnn_new", size, Time([&] { ex(ia); }, warmup, reps));
            const HardPairFeatures fa = ex(ia), fb = ex(ib);
            row("lightglue_aliked/cv_dnn_new (pair)", size, Time([&] { m.Match(fa, fb); }, warmup, reps));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_dnn_timing: %s\n", e.what());
        return 1;
    }
    return 0;
}
