// ---------------------------------------------------------------------------
// ariadne_tum_vo: the stereo VO pipeline over one TUM RGB-D sequence, sensor depth through a virtual rig
// (2.6, D90; perception/rgbd/depth_stereo.hpp), scored against the mocap ground truth.
//
//   ariadne_tum_vo --sequence DIR --perception-config YAML --slam-config YAML
//                  --out DIR [--frames N] [--seed S] [--intrinsics calibrated|ros_default]
//
// Fetch first: infra/scripts/fetch_tum.py freiburg1_xyz. Pinned regression: configs/benchmark/tum_regression.yaml.
// ---------------------------------------------------------------------------
#include "runtime/datasets/tum.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    ariadne::tools::StereoVoArgs a;
    a.tool = "ariadne_tum_vo";
    std::string sequence;
    bool ros_default = false;   // --intrinsics ros_default: the TUM page's uncalibrated set (ablation)
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", k.c_str()); return 2; }
        const std::string v = argv[++i];
        if (k == "--sequence") sequence = v;
        else if (k == "--perception-config") a.perception_config = v;
        else if (k == "--slam-config") a.slam_config = v;
        else if (k == "--out") a.out = v;
        else if (k == "--frames") a.frames = std::stoi(v);
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--intrinsics" && (v == "calibrated" || v == "ros_default")) ros_default = v == "ros_default";
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (sequence.empty() || a.perception_config.empty() || a.slam_config.empty() || a.out.empty()) {
        std::fprintf(stderr, "usage: %s --sequence DIR --perception-config YAML --slam-config YAML --out DIR "
                             "[--frames N] [--seed S] [--intrinsics calibrated|ros_default]\n", argv[0]);
        return 2;
    }
    ariadne::runtime::TumSource source(sequence, ros_default);
    return ariadne::tools::RunStereoVo(source, a, argc, argv);
}
