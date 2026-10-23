// ---------------------------------------------------------------------------
// ariadne_collab_passa: pass A of the 3.2 sim integration (design SIMCOLLAB_3.2 §1-§2, D171). One agent's shipped
// front-end (ariadne_sim_vo's pipeline, unchanged: the hooks never change the run) over a sim recording, writing the
// keyframe store pass B replays (runtime/agent/collab_store.hpp).
//
//   ariadne_collab_passa --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML --out DIR
//       --agent K --seed S --imu-noise YAML --place-onnx ONNX --place-size WxH --aliked-onnx ONNX
//       --lightglue-onnx ONNX [--frames N] [--check-lazy 1]
//
// Per keyframe, at its creation (t_create): ALIKED-n16 on the rectified RGB (the D90 1280x720 sim export, K = 1024),
// its 3D points from the masked SGBM depth, and EigenPlaces R50-256 on the same RGB. Then the keyframe waits: a
// LoopBackend mirror with no loop candidates is synced at every keyframe, and a keyframe is RELEASED when the mirror
// takes it, i.e. its odometry edge became final (it left the BA window), at that frame's time (t_final). Keyframes
// before the mirror's anchor never become final (the D87 startup prefix) and are counted, never released.
//
// The per-keyframe work is runtime/agent/collab_frontend.hpp (D174: the live estimator worker runs the same class).
// --check-lazy 1 runs a lazy twin of it on the same keyframes and checks, per released keyframe, that its on-request
// features are bitwise the eager ones (LIVE_COLLAB §2: lazy == eager); a mismatch exits 3.
//
// Output: DIR/store_agent<K>.bin, DIR/passa_agent<K>.yaml (counts, timings, the model records), plus everything the VO run
// writes (results.yaml, trajectories, manifest) in DIR/vo_agent<K>/.
// ---------------------------------------------------------------------------
#include "perception/association/hard_pair_matcher.hpp"
#include "perception/place/place_recognizer.hpp"
#include "runtime/agent/collab_frontend.hpp"
#include "runtime/agent/collab_store.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"
#include "slam/loop_closure/loop_backend.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

using namespace ariadne;

namespace {

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
    bool check_lazy = false;
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
        else if (k == "--check-lazy") check_lazy = v == "1";
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
        runtime::CollabFrontend front(place, aliked);   // eager: the store carries the local features
        runtime::CollabFrontend twin(place, aliked, {/*eager=*/false, /*keep_s=*/1e18});

        a.on_keyframe = [&](const tools::KeyframeView& kv) {
            const auto& kf = kv.keyframe;
            if (kv.bundle.rgb.channels() != 3 || kv.bundle.rgb.cols != hp.width || kv.bundle.rgb.rows != hp.height)
                throw std::runtime_error("passa: the keyframe RGB is not the 1280x720 colour image the export needs");
            runtime::CollabKeyframeInput in;
            in.id = kf.id;
            in.t_ns = kf.t_ns;
            in.rgb = kv.bundle.rgb;
            in.depth32 = runtime::MaskedDepth32(kv.bundle.depth, kv.bundle.mask);   // NaN where unusable
            const auto& cam = kv.bundle.K;   // the rectified pinhole
            in.fx = cam.fx, in.fy = cam.fy, in.cx = cam.cx, in.cy = cam.cy, in.baseline_m = kv.baseline_m;
            in.T_B_C = kv.T_B_C;
            in.gt_T_W_B = NearestGt(gt, kf.t_ns);
            in.graph = kv.graph;
            for (auto& r : front.OnKeyframe(in)) store.records.push_back(std::move(r));
            if (check_lazy) twin.OnKeyframe(in);
        };
        const int rc = tools::RunStereoVo(source, a, argc, argv);
        // Never final: the startup prefix (ids below the first released one) and the tail still in the window.
        const auto [prefix, tail] = front.Unreleased(store.records.empty() ? std::nullopt
                                                                           : std::optional<uint64_t>(store.records.front().id));
        store.prefix_unreleased = prefix;
        const uint64_t keyframes = front.stats().keyframes;
        const double ms_aliked = front.stats().ms_aliked, ms_place = front.stats().ms_place;
        uint64_t lazy_compared = 0, lazy_mismatch = 0;
        if (check_lazy) {
            auto same = [](const std::vector<float>& x, const std::vector<float>& y) {
                return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
            };
            for (const auto& r : store.records) {
                const auto f = twin.Features(r.id);
                ++lazy_compared;
                if (!f || !same(f->keypoints, r.keypoints) || !same(f->descriptors, r.descriptors) ||
                    !same(f->points3d, r.points3d))
                    ++lazy_mismatch;
            }
        }
        const std::string path = (std::filesystem::path(out) / ("store_agent" + std::to_string(agent) + ".bin")).string();
        runtime::WriteCollabStore(path, store);
        std::ofstream y(std::filesystem::path(out) / ("passa_agent" + std::to_string(agent) + ".yaml"));
        y << "tool: ariadne_collab_passa\nagent: " << agent << "\nseed: " << a.seed << "\nkeyframes: " << keyframes
          << "\nreleased: " << store.records.size() << "\nprefix_unreleased: " << prefix
          << "\ntail_unreleased: " << tail << "\nms_aliked_mean: " << (keyframes ? ms_aliked / keyframes : 0)
          << "\nms_place_mean: " << (keyframes ? ms_place / keyframes : 0) << "\nplace: {model: " << place.Name()
          << ", size: " << place_size << "}\naliked: {model: " << aliked.Name() << "}\nvo_exit: " << rc
          << "\nstore: " << path << "\nopencv_threads: " << cv::getNumThreads() << "\n";   // ALIKED's output depends on it (D174)
        if (check_lazy) y << "lazy_check: {compared: " << lazy_compared << ", mismatches: " << lazy_mismatch << "}\n";
        std::printf("passa agent %d: %llu keyframes, %zu released, prefix %llu, tail %llu -> %s\n", agent,
                    (unsigned long long)keyframes, store.records.size(), (unsigned long long)prefix,
                    (unsigned long long)tail, path.c_str());
        if (check_lazy) std::printf("lazy check: %llu compared, %llu mismatches\n", (unsigned long long)lazy_compared,
                                    (unsigned long long)lazy_mismatch);
        return lazy_mismatch ? 3 : rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_collab_passa: %s\n", e.what());
        return 1;
    }
}
