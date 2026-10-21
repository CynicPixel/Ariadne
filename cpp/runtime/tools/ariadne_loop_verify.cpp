// ---------------------------------------------------------------------------
// 2.1 hard-pair evaluation (PERCEPTION_2.1.md §12.4): loop-candidate verification on one EuRoC flight. OFFLINE: no
// estimator, no runner hunk.
//
// Pairs: the §11 place dump of this flight (samples.csv, ep376.f32 from ariadne_place_eval). For every query the
// EP-376 top-5 (exact cosine) among samples older than --exclude-s are its candidates: a = the older (database)
// sample, b = the query. Per pair and per row, the matches a -> b go through loop_verify_geometry.hpp (undistort,
// the 1.6 verifier's RANSAC essential matrix, recoverPose) and are scored against GT T_Cb_Ca:
//   A  ORB (perception config features) + BF Hamming, Lowe ratio 0.8   (the 1.6 verifier's stage 1: placeholder)
//   B  ALIKED-n16 K=1024 + mutual nearest neighbour (L2)               (isolates LightGlue)
//   C  ALIKED-n16 K=1024 + LightGlue, upstream normalisation            (candidate)
//   D  ALIKED-n16 K=1024 + LightGlue, OpenCV per-axis normalisation     (wrapper convention)
//   E  ALIKED-n16 K=2048 + LightGlue, upstream normalisation            (official K; only the --e-pairs subset)
// Features are computed once per sample per extractor and kept in memory.
//
// Writes: pairs.csv (pair, a, b, centre distance m, axis angle deg), verify_<row>.csv (pair, matches, inliers, ok,
// err_r, err_t, err in deg, match ms), timing.yaml (extraction ms per sample), manifest.yaml. Labels, AP, AUC are
// computed by python/analysis/loop_verify_eval.py.
// ---------------------------------------------------------------------------
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/features/orb_engine.hpp"
#include "perception/perception_config.hpp"
#include "perception/place/place_index.hpp"
#include "runtime/datasets/stereo_source.hpp"
#include "runtime/tools/loop_verify_geometry.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/features.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ariadne;
namespace fs = std::filesystem;

namespace {

struct Args {
    std::string sequence, place, perception_config, export_dir, out, rows = "ABCD", e_pairs;
    int top = 5;
    double exclude_s = 30.0;
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
        else if (k == "--place") a.place = next();
        else if (k == "--perception-config") a.perception_config = next();
        else if (k == "--export") a.export_dir = next();
        else if (k == "--out") a.out = next();
        else if (k == "--rows") a.rows = next();
        else if (k == "--e-pairs") a.e_pairs = next();
        else if (k == "--top") a.top = std::stoi(next());
        else if (k == "--exclude-s") a.exclude_s = std::stod(next());
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.sequence.empty() || a.place.empty() || a.perception_config.empty() || a.export_dir.empty() || a.out.empty()) {
        throw std::runtime_error("usage: ariadne_loop_verify --sequence DIR --place DIR --perception-config YAML "
                                 "--export DIR --out DIR [--rows ABCD] [--e-pairs CSV] [--top 5] [--exclude-s 30]");
    }
    if (a.rows.find('E') != std::string::npos && a.e_pairs.empty()) throw std::runtime_error("row E needs --e-pairs");
    return a;
}

double Ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Sample {
    int64_t t_ns = 0;
    core::SE3 T_W_C;
};

std::vector<Sample> LoadSamples(const fs::path& csv) {
    std::ifstream f(csv);
    if (!f) throw std::runtime_error("cannot read " + csv.string());
    std::string line;
    std::getline(f, line);   // header
    std::vector<Sample> s;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string idx, t;
        std::getline(ss, idx, ',');
        std::getline(ss, t, ',');
        s.push_back({std::stoll(t), {}});
    }
    return s;
}

std::vector<std::vector<float>> LoadF32(const fs::path& p, size_t n) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + p.string());
    std::vector<std::vector<float>> v(n, std::vector<float>(256));
    for (auto& r : v) f.read(reinterpret_cast<char*>(r.data()), 256 * 4);
    if (!f) throw std::runtime_error("short read " + p.string());
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = Parse(argc, argv);
        runtime::EurocSource source(a.sequence);
        const perception::PerceptionConfig pc = perception::LoadPerception(a.perception_config);
        const perception::PinholeRadtan cam = source.Calibration().left;
        std::vector<Sample> samples = LoadSamples(fs::path(a.place) / "samples.csv");
        const auto ep = LoadF32(fs::path(a.place) / "ep376.f32", samples.size());

        // GT camera pose per sample: nearest GT within 10 ms, exactly as ariadne_place_eval chose the samples.
        const auto gt = source.GroundTruth();
        const core::SE3 T_B_C = source.T_B_C0();
        for (auto& s : samples) {
            const auto it = std::lower_bound(gt.begin(), gt.end(), s.t_ns,
                                             [](const runtime::TimedPose& p, int64_t t) { return p.t_ns < t; });
            const runtime::TimedPose* best = nullptr;
            for (auto j = it == gt.begin() ? it : it - 1; j != gt.end() && j <= it; ++j) {
                if (!best || std::llabs(j->t_ns - s.t_ns) < std::llabs(best->t_ns - s.t_ns)) best = &*j;
            }
            if (!best || std::llabs(best->t_ns - s.t_ns) > 10000000LL) throw std::runtime_error("sample without GT");
            s.T_W_C = best->T * T_B_C;
        }

        // Candidate pairs (a = database, b = query): EP-376 top-k, exact, ties by the smaller index (§11).
        perception::PlaceIndex index(256);
        for (size_t i = 0; i < samples.size(); ++i) index.Add(i, ep[i]);
        const int64_t excl = static_cast<int64_t>(a.exclude_s * 1e9);
        std::vector<std::pair<int, int>> pairs;
        for (size_t q = 0; q < samples.size(); ++q) {
            for (const auto& h : index.Query(ep[q], a.top, [&](uint64_t id) { return samples[id].t_ns < samples[q].t_ns - excl; })) {
                pairs.push_back({int(h.id), int(q)});
            }
        }
        std::set<int> e_subset;   // pair indices for row E
        if (!a.e_pairs.empty()) {
            std::ifstream f(a.e_pairs);
            if (!f) throw std::runtime_error("cannot read " + a.e_pairs);
            std::string line;
            std::getline(f, line);
            while (std::getline(f, line)) e_subset.insert(std::stoi(line));
        }

        // Raw cam0 images of the samples.
        std::map<int64_t, int> by_t;
        for (size_t i = 0; i < samples.size(); ++i) by_t[samples[i].t_ns] = int(i);
        std::vector<cv::Mat> img(samples.size());
        source.ForEachFrame(0, [&](const runtime::StereoFrame& f) {
            const auto it = by_t.find(f.t_ns);
            if (it != by_t.end()) img[size_t(it->second)] = f.left.clone();
        });
        for (size_t i = 0; i < img.size(); ++i) if (img[i].empty()) throw std::runtime_error("missing image for sample");

        std::set<int> need_e;   // samples row E touches
        for (int p : e_subset) {
            need_e.insert(pairs.at(size_t(p)).first);
            need_e.insert(pairs.at(size_t(p)).second);
        }
        const bool any_lg = a.rows.find_first_of("BCD") != std::string::npos;
        auto hp = [&](const std::string& onnx, const std::string& norm) {
            perception::HardPairParams p;
            p.aliked_onnx = (fs::path(a.export_dir) / onnx).string();
            p.lightglue_onnx = (fs::path(a.export_dir) / "lightglue_aliked.onnx").string();
            p.normalization = norm;
            return p;
        };
        const perception::OrbEngine orb(pc.features);
        std::unique_ptr<perception::HardPairMatcher> m_up, m_cv, m_2k;
        if (any_lg) {
            m_up = std::make_unique<perception::HardPairMatcher>(hp("aliked_n16_k1024_752x480.onnx", "upstream"));
            m_cv = std::make_unique<perception::HardPairMatcher>(hp("aliked_n16_k1024_752x480.onnx", "opencv"));
        }
        if (a.rows.find('E') != std::string::npos) {
            m_2k = std::make_unique<perception::HardPairMatcher>(hp("aliked_n16_k2048_752x480.onnx", "upstream"));
        }

        std::vector<perception::FeatureSet> f_orb(samples.size());
        std::vector<perception::HardPairFeatures> f_1k(samples.size()), f_2k(samples.size());
        double ms_orb = 0, ms_1k = 0, ms_2k = 0;
        int n_2k = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            auto t0 = std::chrono::steady_clock::now();
            if (a.rows.find('A') != std::string::npos) f_orb[i] = orb.DetectAndCompute(img[i], cv::Mat());
            ms_orb += Ms(t0);
            t0 = std::chrono::steady_clock::now();
            if (any_lg) f_1k[i] = m_up->Extract(img[i]);
            ms_1k += Ms(t0);
            if (m_2k && need_e.count(int(i))) {
                t0 = std::chrono::steady_clock::now();
                f_2k[i] = m_2k->Extract(img[i]);
                ms_2k += Ms(t0);
                ++n_2k;
            }
        }

        fs::create_directories(a.out);
        {
            std::ofstream o(fs::path(a.out) / "pairs.csv");
            o << "pair,a,b,dist_m,axis_deg\n";
            o.precision(9);
            for (size_t p = 0; p < pairs.size(); ++p) {
                const auto& A = samples[size_t(pairs[p].first)].T_W_C;
                const auto& B = samples[size_t(pairs[p].second)].T_W_C;
                o << p << ',' << pairs[p].first << ',' << pairs[p].second << ',' << (A.t - B.t).norm() << ','
                  << runtime::AngleDeg(A.R * core::Vec3(0, 0, 1), B.R * core::Vec3(0, 0, 1)) << '\n';
            }
        }
        const cv::BFMatcher hamming(cv::NORM_HAMMING), l2_mutual(cv::NORM_L2, true);
        std::map<char, double> ms_match;
        std::map<char, int> n_match;
        for (const char row : a.rows) {
            std::ofstream o(fs::path(a.out) / (std::string("verify_") + row + ".csv"));
            o << "pair,matches,inliers,ok,err_r,err_t,err,ms\n";
            o.precision(9);
            for (size_t p = 0; p < pairs.size(); ++p) {
                if (row == 'E' && !e_subset.count(int(p))) continue;
                const int ia = pairs[p].first, ib = pairs[p].second;
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<cv::DMatch> m;
                const std::vector<cv::KeyPoint>* ka = nullptr;
                const std::vector<cv::KeyPoint>* kb = nullptr;
                if (row == 'A') {
                    ka = &f_orb[size_t(ia)].keypoints;
                    kb = &f_orb[size_t(ib)].keypoints;
                    if (!f_orb[size_t(ia)].descriptors.empty() && !f_orb[size_t(ib)].descriptors.empty()) {
                        std::vector<std::vector<cv::DMatch>> knn;
                        hamming.knnMatch(f_orb[size_t(ia)].descriptors, f_orb[size_t(ib)].descriptors, knn, 2);
                        for (const auto& k : knn) {
                            if (k.size() == 2 && k[0].distance < 0.8f * k[1].distance) m.push_back(k[0]);
                        }
                    }
                } else if (row == 'B') {
                    ka = &f_1k[size_t(ia)].keypoints;
                    kb = &f_1k[size_t(ib)].keypoints;
                    l2_mutual.match(f_1k[size_t(ia)].descriptors, f_1k[size_t(ib)].descriptors, m);
                } else if (row == 'C' || row == 'D') {
                    ka = &f_1k[size_t(ia)].keypoints;
                    kb = &f_1k[size_t(ib)].keypoints;
                    m = (row == 'C' ? m_up : m_cv)->Match(f_1k[size_t(ia)], f_1k[size_t(ib)]);
                } else if (row == 'E') {
                    ka = &f_2k[size_t(ia)].keypoints;
                    kb = &f_2k[size_t(ib)].keypoints;
                    m = m_2k->Match(f_2k[size_t(ia)], f_2k[size_t(ib)]);
                } else {
                    throw std::runtime_error(std::string("unknown row ") + row);
                }
                const double ms = Ms(t0);   // warm: features cached, matching only
                std::vector<cv::Point2f> pa, pb;
                for (const auto& d : m) {
                    pa.push_back((*ka)[size_t(d.queryIdx)].pt);
                    pb.push_back((*kb)[size_t(d.trainIdx)].pt);
                }
                const auto est = runtime::EstimateRelPose(pa, pb, cam);
                const core::SE3 T_Cb_Ca = samples[size_t(ib)].T_W_C.inverse() * samples[size_t(ia)].T_W_C;
                const auto err = runtime::PoseErrorDeg(est, T_Cb_Ca);
                o << p << ',' << est.matches << ',' << est.inliers << ',' << int(est.ok) << ',' << err.r_deg << ','
                  << err.t_deg << ',' << err.deg << ',' << ms << '\n';
                ms_match[row] += ms;
                ++n_match[row];
            }
        }
        {
            std::ofstream t(fs::path(a.out) / "timing.yaml");
            t << "# wall clock, ms (host shared: indicative only)\nsamples: " << samples.size()
              << "\npairs: " << pairs.size() << "\nopencv_threads: " << cv::getNumThreads()
              << "\nextract_ms_mean: {orb: " << ms_orb / double(samples.size())
              << ", aliked_k1024: " << ms_1k / double(samples.size())
              << ", aliked_k2048: " << (n_2k ? ms_2k / n_2k : 0.0) << "}\nmatch_ms_mean: {";
            bool first = true;
            for (const auto& [row, ms] : ms_match) {
                t << (first ? "" : ", ") << row << ": " << ms / std::max(1, n_match[row]);
                first = false;
            }
            t << "}\n";
        }
        {
            YAML::Node em = YAML::LoadFile((fs::path(a.export_dir) / "manifest.yaml").string());
            std::ofstream mf(fs::path(a.out) / "manifest.yaml");
            std::string cmd;
            for (int i = 0; i < argc; ++i) cmd += (i ? " " : "") + std::string(argv[i]);
            auto env = [](const char* k) { return std::getenv(k) ? std::getenv(k) : "UNRECORDED"; };
            mf << "# ariadne_loop_verify (PERCEPTION_2.1 §12.4). Generated.\n"
               << "tool: ariadne_loop_verify\ncommand: \"" << cmd << "\"\n"
               << "scene: " << source.SceneName() << "\nbag_checksum: " << source.BagChecksum()
               << "\ngit_commit: " << env("ARIADNE_GIT_COMMIT") << "\ngit_dirty: " << env("ARIADNE_GIT_DIRTY")
               << "\nimage_digest: " << env("ARIADNE_IMAGE_DIGEST") << "\nperception_config: " << pc.name
               << "\nrows: " << a.rows << "\nhard_pair: [" << (m_up ? m_up->Name() : "none") << ", "
               << (m_cv ? m_cv->Name() : "none") << ", " << (m_2k ? m_2k->Name() : "none") << "]"
               << "\nlightglue_commit: " << em["commit"].as<std::string>()
               << "\nestimator: findEssentialMat RANSAC 0.999 1.0px 1000 (classic, RNG (uint64)-1 per call) + recoverPose"
               << "\npose_error: glue-factory relative_pose_error (max of R and folded t angle)"
               << "\ntop: " << a.top << "\nexclude_s: " << a.exclude_s << "\ne_pairs: " << e_subset.size() << '\n';
        }
        std::printf("%s: %zu samples, %zu pairs, rows %s\n", source.RunName().c_str(), samples.size(), pairs.size(),
                    a.rows.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_loop_verify: %s\n", e.what());
        return 1;
    }
}
