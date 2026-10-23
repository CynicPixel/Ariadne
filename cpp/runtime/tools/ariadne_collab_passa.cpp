// ---------------------------------------------------------------------------
// ariadne_collab_passa: pass A of the 3.2 sim integration (design SIMCOLLAB_3.2 §1-§2, D171). One agent's shipped
// front-end (ariadne_sim_vo's pipeline, unchanged: the hooks never change the run) over a sim recording, writing the
// keyframe store pass B replays (runtime/agent/collab_store.hpp).
//
//   ariadne_collab_passa --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML --out DIR
//       --agent K --seed S --imu-noise YAML --place-onnx ONNX --place-size WxH --aliked-onnx ONNX
//       --lightglue-onnx ONNX [--frames N]
//
// Per keyframe, at its creation (t_create): ALIKED-n16 on the rectified RGB (the D90 1280x720 sim export, K = 1024),
// its 3D points from the masked SGBM depth, and EigenPlaces R50-256 on the same RGB. Then the keyframe waits: a
// LoopBackend mirror with no loop candidates is synced at every keyframe, and a keyframe is RELEASED when the mirror
// takes it, i.e. its odometry edge became final (it left the BA window), at that frame's time (t_final). Keyframes
// before the mirror's anchor never become final (the D87 startup prefix) and are counted, never released.
//
// Output: DIR/store_agent<K>.bin, DIR/passa_agent<K>.yaml (counts, timings, the model records), plus everything the VO run
// writes (results.yaml, trajectories, manifest) in DIR/vo_agent<K>/.
// ---------------------------------------------------------------------------
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/place/place_recognizer.hpp"
#include "runtime/agent/collab_store.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"
#include "slam/loop_closure/loop_backend.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

using namespace ariadne;

namespace {

struct Pending {   // features computed at t_create, waiting for the keyframe's odometry to become final
    int64_t t_create_ns = 0;
    std::vector<float> global, keypoints, descriptors, points3d;
    std::array<float, 5> camera{};
    std::optional<core::SE3> gt_T_W_B;
    core::SE3 T_B_C;
};

/// Ground truth T_W_B nearest to t within 10 ms (evo's association rule, as the VO scoring).
std::optional<core::SE3> NearestGt(const std::vector<runtime::TimedPose>& gt, int64_t t) {
    auto it = std::lower_bound(gt.begin(), gt.end(), t,
                               [](const runtime::TimedPose& p, int64_t v) { return p.t_ns < v; });
    const runtime::TimedPose* best = nullptr;
    int64_t d = INT64_MAX;
    for (auto c : {it, it == gt.begin() ? it : it - 1})
        if (c != gt.end() && std::llabs(c->t_ns - t) < d) d = std::llabs(c->t_ns - t), best = &*c;
    if (!best || d > 10'000'000) return std::nullopt;
    return best->T;
}

}  // namespace

int main(int argc, char** argv) {
    tools::StereoVoArgs a;
    a.tool = "ariadne_collab_passa";
    std::string rec, scene, imu_noise, place_onnx, place_size = "640x360", aliked_onnx, lightglue_onnx, out;
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
        else if (k == "--out") out = v;
        else if (k == "--frames") a.frames = std::stoi(v);
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--imu-noise") imu_noise = v;
        else if (k == "--place-onnx") place_onnx = v;
        else if (k == "--place-size") place_size = v;
        else if (k == "--aliked-onnx") aliked_onnx = v;
        else if (k == "--lightglue-onnx") lightglue_onnx = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || a.perception_config.empty() || a.slam_config.empty() || out.empty() ||
        place_onnx.empty() || aliked_onnx.empty() || lightglue_onnx.empty()) {
        std::fprintf(stderr, "usage: see the header of ariadne_collab_passa.cpp\n");
        return 2;
    }
    try {
        std::filesystem::create_directories(out);
        a.out = (std::filesystem::path(out) / ("vo_agent" + std::to_string(agent))).string();

        perception::PlaceRecognizerParams pp;
        pp.onnx = place_onnx;
        if (std::sscanf(place_size.c_str(), "%dx%d", &pp.width, &pp.height) != 2)
            throw std::invalid_argument("--place-size WxH");
        const perception::PlaceRecognizer place(pp);
        perception::HardPairParams hp;
        hp.aliked_onnx = aliked_onnx;
        hp.lightglue_onnx = lightglue_onnx;
        hp.width = 1280, hp.height = 720;   // the D90 sim export
        const perception::HardPairMatcher aliked(hp);

        runtime::SimSource source(rec, scene, agent);
        if (!imu_noise.empty()) source.SetImuNoise(core::LoadImuNoiseOperator(imu_noise), a.seed);
        const std::vector<runtime::TimedPose> gt = source.GroundTruth();

        runtime::CollabStore store;
        store.agent = uint32_t(agent);
        std::map<uint64_t, Pending> pending;
        slam::LoopBackend mirror;   // no candidates ever: it only says which odometry is final
        std::optional<uint64_t> last_released;
        double ms_aliked = 0, ms_place = 0;
        uint64_t keyframes = 0;

        a.on_keyframe = [&](const tools::KeyframeView& kv) {
            ++keyframes;
            const auto& kf = kv.keyframe;
            if (kv.bundle.rgb.channels() != 3 || kv.bundle.rgb.cols != hp.width || kv.bundle.rgb.rows != hp.height)
                throw std::runtime_error("passa: the keyframe RGB is not the 1280x720 colour image the export needs");
            Pending p;
            p.t_create_ns = kf.t_ns;
            p.T_B_C = kv.T_B_C;
            p.gt_T_W_B = NearestGt(gt, kf.t_ns);
            const auto t0 = std::chrono::steady_clock::now();
            const perception::HardPairFeatures f = aliked.ExtractRgb(kv.bundle.rgb);
            const auto t1 = std::chrono::steady_clock::now();
            p.global = place.DescribeRgb(kv.bundle.rgb);
            const auto t2 = std::chrono::steady_clock::now();
            ms_aliked += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_place += std::chrono::duration<double, std::milli>(t2 - t1).count();
            const auto& cam = kv.bundle.K;   // the rectified pinhole
            const cv::Mat depth = runtime::MaskedDepth32(kv.bundle.depth, kv.bundle.mask);   // NaN where unusable
            p.camera = {float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy), float(kv.baseline_m)};
            for (int i = 0; i < int(f.keypoints.size()); ++i) {
                const cv::Point2f& q = f.keypoints[size_t(i)].pt;
                p.keypoints.push_back(q.x);
                p.keypoints.push_back(q.y);
                const int u = int(std::lround(q.x)), v = int(std::lround(q.y));
                float z = NAN;
                if (u >= 0 && v >= 0 && u < depth.cols && v < depth.rows) z = depth.at<float>(v, u);
                const bool ok = std::isfinite(z) && z > 0.0f;
                p.points3d.push_back(ok ? float((q.x - cam.cx) / cam.fx * z) : NAN);
                p.points3d.push_back(ok ? float((q.y - cam.cy) / cam.fy * z) : NAN);
                p.points3d.push_back(ok ? z : NAN);
                const float* d = f.descriptors.ptr<float>(i);
                p.descriptors.insert(p.descriptors.end(), d, d + f.descriptors.cols);
            }
            pending[kf.id] = std::move(p);

            // Release whatever the mirror now holds as final, in id order, at this frame's time.
            mirror.Sync(*kv.graph);
            const auto& nodes = mirror.graph().nodes();
            for (auto it = last_released ? nodes.upper_bound(*last_released) : nodes.begin(); it != nodes.end(); ++it) {
                const auto pit = pending.find(it->first);
                if (pit == pending.end()) throw std::logic_error("passa: a final keyframe without its features");
                runtime::CollabStoreRecord r;
                r.id = it->first;
                r.t_create_ns = pit->second.t_create_ns;
                r.t_final_ns = kf.t_ns;
                r.T_W_C = it->second.T_W_C;
                if (last_released) {
                    bool found = false;
                    for (const auto& e : mirror.graph().edges())
                        if (e.type == slam::EdgeType::kOdometry && e.from == *last_released && e.to == r.id) {
                            r.odo_from = e.from;
                            r.odo_Z = e.Z;
                            for (int q = 0; q < 36; ++q) r.odo_information[size_t(q)] = e.information(q / 6, q % 6);
                            found = true;
                        }
                    if (!found) throw std::logic_error("passa: no final odometry edge into a released keyframe");
                }
                r.global = std::move(pit->second.global);
                r.keypoints = std::move(pit->second.keypoints);
                r.descriptors = std::move(pit->second.descriptors);
                r.points3d = std::move(pit->second.points3d);
                r.camera = pit->second.camera;
                r.has_gt = pit->second.gt_T_W_B.has_value();
                if (r.has_gt) r.gt_T_W_B = *pit->second.gt_T_W_B;
                r.T_B_C = pit->second.T_B_C;
                store.records.push_back(std::move(r));
                pending.erase(pit);
                last_released = it->first;
            }
        };
        const int rc = tools::RunStereoVo(source, a, argc, argv);
        // Never final: the startup prefix (ids below the first released one) and the tail still in the window.
        uint64_t prefix = 0, tail = 0;
        for (const auto& [id, p] : pending) (store.records.empty() || id < store.records.front().id ? prefix : tail)++;
        store.prefix_unreleased = prefix;
        const std::string path = (std::filesystem::path(out) / ("store_agent" + std::to_string(agent) + ".bin")).string();
        runtime::WriteCollabStore(path, store);
        std::ofstream y(std::filesystem::path(out) / ("passa_agent" + std::to_string(agent) + ".yaml"));
        y << "tool: ariadne_collab_passa\nagent: " << agent << "\nseed: " << a.seed << "\nkeyframes: " << keyframes
          << "\nreleased: " << store.records.size() << "\nprefix_unreleased: " << prefix
          << "\ntail_unreleased: " << tail << "\nms_aliked_mean: " << (keyframes ? ms_aliked / keyframes : 0)
          << "\nms_place_mean: " << (keyframes ? ms_place / keyframes : 0) << "\nplace: {model: " << place.Name()
          << ", size: " << place_size << "}\naliked: {model: " << aliked.Name() << "}\nvo_exit: " << rc
          << "\nstore: " << path << "\n";
        std::printf("passa agent %d: %llu keyframes, %zu released, prefix %llu, tail %llu -> %s\n", agent,
                    (unsigned long long)keyframes, store.records.size(), (unsigned long long)prefix,
                    (unsigned long long)tail, path.c_str());
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_collab_passa: %s\n", e.what());
        return 1;
    }
}
