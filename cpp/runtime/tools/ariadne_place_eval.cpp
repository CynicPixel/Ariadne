// ---------------------------------------------------------------------------
// 2.1 retrieval evaluation dump (PERCEPTION_2.1.md §11.5). OFFLINE: no estimator, no runner hunk.
//
// For one EuRoC flight: one raw cam0 frame per --every images (20 = 1.0 s), kept only where ground truth exists
// within 10 ms. Per sample, three GLOBAL_DESCRIPTOR candidates (256-D unit vectors, cosine):
//   ep376  EigenPlaces R50-256 at 376x240 (PlaceRecognizer, cv::dnn ENGINE_CLASSIC)
//   ep752  EigenPlaces R50-256 at 752x480 (upstream's full-resolution protocol)
//   meanbit  the 1.6 placeholder: MeanBitDescriptor of OrbEngine::DetectAndCompute, as ariadne_sim_collab does
// and the GT camera centre and optical axis (T_W_B x T_B_C0). The recall is computed by
// python/analysis/place_recall_eval.py from these files; this tool also records, per method and query, the top-5 of
// the ANNOY backend under the same time filter (database = samples older than --exclude-s), so the script can score
// annoy against exact on identical queries.
//
// Writes: samples.csv, <method>.f32 (N x 256 float32, row i = sample i), annoy_<method>.csv, timing.yaml,
// manifest.yaml.
// ---------------------------------------------------------------------------
#include "collaboration/retrieval/mean_bit_descriptor.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/perception_config.hpp"
#include "perception/place/place_index.hpp"
#include "perception/place/place_recognizer.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <opencv2/core/utility.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace ariadne;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string sequence, perception_config, export_dir, out;
    int every = 20;
    double exclude_s = 30.0;
    int annoy_trees = 10;
};

Args Parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--sequence") a.sequence = next();
        else if (k == "--perception-config") a.perception_config = next();
        else if (k == "--export") a.export_dir = next();
        else if (k == "--out") a.out = next();
        else if (k == "--every") a.every = std::stoi(next());
        else if (k == "--exclude-s") a.exclude_s = std::stod(next());
        else if (k == "--annoy-trees") a.annoy_trees = std::stoi(next());
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.sequence.empty() || a.perception_config.empty() || a.export_dir.empty() || a.out.empty()) {
        throw std::runtime_error("usage: ariadne_place_eval --sequence DIR --perception-config YAML --export DIR "
                                 "--out DIR [--every 20] [--exclude-s 30] [--annoy-trees 10]");
    }
    return a;
}

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = Parse(argc, argv);
        runtime::EurocSource source(a.sequence);
        const perception::PerceptionConfig pc = perception::LoadPerception(a.perception_config);
        const perception::OrbEngine orb(pc.features);
        perception::PlaceRecognizer ep376({(fs::path(a.export_dir) / "eigenplaces_r50_256_376x240.onnx").string(),
                                           376, 240, "classic"});
        perception::PlaceRecognizer ep752({(fs::path(a.export_dir) / "eigenplaces_r50_256_752x480.onnx").string(),
                                           752, 480, "classic"});
        const std::vector<runtime::TimedPose> gt = source.GroundTruth();
        const core::SE3 T_B_C = source.T_B_C0();

        struct Sample {
            int64_t t_ns;
            core::Vec3 c, axis;
        };
        std::vector<Sample> samples;
        const std::vector<std::string> methods = {"ep376", "ep752", "meanbit"};
        std::map<std::string, std::vector<std::vector<float>>> desc;
        std::map<std::string, double> ms_sum;
        source.ForEachFrame(0, [&](const runtime::StereoFrame& f) {
            if (f.id % static_cast<uint64_t>(a.every) != 0) return;
            const auto it = std::lower_bound(gt.begin(), gt.end(), f.t_ns,
                                             [](const runtime::TimedPose& p, int64_t t) { return p.t_ns < t; });
            const runtime::TimedPose* best = nullptr;
            for (auto j = it == gt.begin() ? it : it - 1; j != gt.end() && j <= it; ++j) {
                if (!best || std::llabs(j->t_ns - f.t_ns) < std::llabs(best->t_ns - f.t_ns)) best = &*j;
            }
            if (!best || std::llabs(best->t_ns - f.t_ns) > 10000000LL) return;   // 10 ms
            const core::SE3 T_W_C = best->T * T_B_C;
            samples.push_back({f.t_ns, T_W_C.t, T_W_C.R * core::Vec3(0, 0, 1)});
            auto t0 = std::chrono::steady_clock::now();
            desc["ep376"].push_back(ep376.Describe(f.left));
            ms_sum["ep376"] += Ms(t0);
            t0 = std::chrono::steady_clock::now();
            desc["ep752"].push_back(ep752.Describe(f.left));
            ms_sum["ep752"] += Ms(t0);
            t0 = std::chrono::steady_clock::now();
            const perception::FeatureSet fs_orb = orb.DetectAndCompute(f.left, cv::Mat());
            std::vector<uint8_t> bytes(fs_orb.descriptors.datastart, fs_orb.descriptors.dataend);
            desc["meanbit"].push_back(collaboration::MeanBitDescriptor(bytes));
            ms_sum["meanbit"] += Ms(t0);
        });
        if (samples.empty()) throw std::runtime_error("no samples with ground truth");

        fs::create_directories(a.out);
        {
            std::ofstream s(fs::path(a.out) / "samples.csv");
            s << "index,t_ns,cx,cy,cz,ax,ay,az\n";
            s.precision(9);
            for (size_t i = 0; i < samples.size(); ++i) {
                const auto& x = samples[i];
                s << i << ',' << x.t_ns << ',' << x.c.x() << ',' << x.c.y() << ',' << x.c.z() << ',' << x.axis.x()
                  << ',' << x.axis.y() << ',' << x.axis.z() << '\n';
            }
        }
        for (const auto& m : methods) {
            std::ofstream f(fs::path(a.out) / (m + ".f32"), std::ios::binary);
            for (const auto& d : desc[m]) f.write(reinterpret_cast<const char*>(d.data()), std::streamsize(d.size() * 4));
            // annoy top-5 under the time filter, one row per query: index, then up to 5 sample indices.
            perception::PlaceIndex ann(256, perception::PlaceIndex::Backend::kAnnoy, a.annoy_trees, 1);
            for (size_t i = 0; i < desc[m].size(); ++i) ann.Add(i, desc[m][i]);
            std::ofstream o(fs::path(a.out) / ("annoy_" + m + ".csv"));
            o << "query,top1,top2,top3,top4,top5\n";
            const int64_t excl = static_cast<int64_t>(a.exclude_s * 1e9);
            for (size_t q = 0; q < desc[m].size(); ++q) {
                const auto hits = ann.Query(desc[m][q], 5, [&](uint64_t id) {
                    return samples[id].t_ns < samples[q].t_ns - excl;
                });
                o << q;
                for (const auto& h : hits) o << ',' << h.id;
                for (size_t k = hits.size(); k < 5; ++k) o << ",-1";
                o << '\n';
            }
        }
        {
            std::ofstream t(fs::path(a.out) / "timing.yaml");
            t << "# per-sample wall clock, ms (host shared: indicative only)\nsamples: " << samples.size()
              << "\nopencv_threads: " << cv::getNumThreads() << '\n';
            for (const auto& m : methods) t << m << "_ms_mean: " << ms_sum[m] / double(samples.size()) << '\n';
        }
        {
            YAML::Node em = YAML::LoadFile((fs::path(a.export_dir) / "manifest.yaml").string());
            std::ofstream mf(fs::path(a.out) / "manifest.yaml");
            std::string cmd;
            for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
            mf << "# ariadne_place_eval (PERCEPTION_2.1 §11.5). Generated.\n"
               << "tool: ariadne_place_eval\ncommand: \"" << cmd << "\"\n"
               << "scene: " << source.SceneName() << "\nbag_checksum: " << source.BagChecksum() << '\n'
               << "git_commit: " << (std::getenv("ARIADNE_GIT_COMMIT") ? std::getenv("ARIADNE_GIT_COMMIT") : "UNRECORDED")
               << "\ngit_dirty: " << (std::getenv("ARIADNE_GIT_DIRTY") ? std::getenv("ARIADNE_GIT_DIRTY") : "UNRECORDED")
               << "\nimage_digest: "
               << (std::getenv("ARIADNE_IMAGE_DIGEST") ? std::getenv("ARIADNE_IMAGE_DIGEST") : "UNRECORDED")
               << "\nperception_config: " << pc.name << "\nplace_recognition: [" << ep376.Name() << ", " << ep752.Name()
               << ", meanbit_orb_placeholder]\nplace_index: [exact (in place_recall_eval.py), annoy/trees"
               << a.annoy_trees << "/seed1]\n"
               << "eigenplaces_commit: " << em["commit"].as<std::string>()
               << "\neigenplaces_weights_sha256: " << em["weights_sha256"].as<std::string>()
               << "\nevery: " << a.every << "\nexclude_s: " << a.exclude_s << '\n';
        }
        std::printf("%s: %zu samples\n", source.RunName().c_str(), samples.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_place_eval: %s\n", e.what());
        return 1;
    }
}
