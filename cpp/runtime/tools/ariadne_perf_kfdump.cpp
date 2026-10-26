// ---------------------------------------------------------------------------
// ariadne_perf_kfdump: the pipeline profile's keyframe inputs (2026-10-24; measurement only). Runs one agent's
// shipped front-end over a sim recording exactly as ariadne_collab_passa does (same StereoVoArgs, same seed and IMU
// noise operator, so the same keyframes at the same OpenCV thread count), and writes the CollabKeyframeInput of the
// chosen keyframes - what CollabFrontend::ExtractLocal consumes - so the speed knobs (ALIKED input size, LightGlue K)
// can be re-run offline on the REAL verified pairs:
//
//   ariadne_perf_kfdump --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML --agent K --seed S
//       --imu-noise YAML --ids 3,17,210 --out DIR
//
// Per id: DIR/a<K>_kf<id>_rgb.png (the rectified RGB, written BGR as cv::imwrite expects), DIR/a<K>_kf<id>_depth.f32
// (int32 rows, int32 cols, then rows*cols float32 metres, NaN = unusable: runtime::MaskedDepth32 of the bundle), and one
// line per id in DIR/a<K>_keyframes.csv (id, t_ns, fx, fy, cx, cy, baseline_m). DIR/a<K>_kfdump.yaml records the
// keyframe count (compare with pass A's passa_agent<K>.yaml: equal counts = the same VO run) and the thread count.
// ---------------------------------------------------------------------------
#include "runtime/agent/keyframe_records.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

using namespace ariadne;

int main(int argc, char** argv) {
    tools::StereoVoArgs a;
    a.tool = "ariadne_perf_kfdump";
    std::string rec, scene, imu_noise, ids, out;
    int agent = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); return 2; }
        const std::string v = argv[++i];
        if (k == "--rec") rec = v;
        else if (k == "--scene") scene = v;
        else if (k == "--agent") agent = std::stoi(v);
        else if (k == "--perception-config") a.perception_config = v;
        else if (k == "--slam-config") a.slam_config = v;
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--imu-noise") imu_noise = v;
        else if (k == "--ids") ids = v;
        else if (k == "--out") out = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || a.perception_config.empty() || a.slam_config.empty() || out.empty() ||
        ids.empty()) {
        std::fprintf(stderr, "usage: see the header of ariadne_perf_kfdump.cpp\n");
        return 2;
    }
    try {
        std::set<uint64_t> want;
        std::stringstream ss(ids);
        for (std::string t; std::getline(ss, t, ',');) want.insert(std::stoull(t));
        const std::filesystem::path dir(out);
        std::filesystem::create_directories(dir);
        a.out = (dir / ("vo_agent" + std::to_string(agent))).string();
        const std::string pre = "a" + std::to_string(agent) + "_kf";

        runtime::SimSource source(rec, scene, agent);
        if (!imu_noise.empty()) source.SetImuNoise(core::LoadImuNoiseOperator(imu_noise), a.seed);
        std::ofstream csv(dir / ("a" + std::to_string(agent) + "_keyframes.csv"));
        csv << "id,t_ns,fx,fy,cx,cy,baseline_m\n";
        csv.precision(17);
        uint64_t keyframes = 0, written = 0;
        a.on_keyframe = [&](const tools::KeyframeView& kv) {
            ++keyframes;
            const auto& kf = kv.keyframe;
            if (!want.count(kf.id)) return;
            const std::string stem = pre + std::to_string(kf.id);
            cv::Mat bgr;
            cv::cvtColor(kv.bundle.rgb, bgr, cv::COLOR_RGB2BGR);
            if (!cv::imwrite((dir / (stem + "_rgb.png")).string(), bgr))
                throw std::runtime_error("kfdump: cannot write " + stem + "_rgb.png");
            const cv::Mat z = runtime::MaskedDepth32(kv.bundle.depth, kv.bundle.mask);
            std::ofstream f(dir / (stem + "_depth.f32"), std::ios::binary);
            const int32_t rc[2] = {z.rows, z.cols};
            f.write(reinterpret_cast<const char*>(rc), sizeof rc);
            for (int r = 0; r < z.rows; ++r) f.write(reinterpret_cast<const char*>(z.ptr<float>(r)), z.cols * 4);
            const auto& K = kv.bundle.K;
            csv << kf.id << ',' << kf.t_ns << ',' << K.fx << ',' << K.fy << ',' << K.cx << ',' << K.cy << ','
                << kv.baseline_m << '\n';
            ++written;
        };
        const int rc = tools::RunStereoVo(source, a, argc, argv);
        std::ofstream y(dir / ("a" + std::to_string(agent) + "_kfdump.yaml"));
        y << "tool: ariadne_perf_kfdump\nagent: " << agent << "\nseed: " << a.seed << "\nkeyframes: " << keyframes
          << "\nrequested: " << want.size() << "\nwritten: " << written << "\nvo_exit: " << rc
          << "\nopencv_threads: " << cv::getNumThreads() << "\n";
        std::printf("kfdump agent %d: %llu keyframes, %llu of %zu written\n", agent, (unsigned long long)keyframes,
                    (unsigned long long)written, want.size());
        return rc != 0 ? rc : (written == want.size() ? 0 : 4);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_perf_kfdump: %s\n", e.what());
        return 1;
    }
}
