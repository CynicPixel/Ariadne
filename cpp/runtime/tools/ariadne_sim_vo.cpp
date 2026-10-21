// ---------------------------------------------------------------------------
// ariadne_sim_vo: the same 1.4 + 1.5 pipeline as ariadne_euroc_vo, over one
// agent of a simulator recording (design SIM_1.1-1.3 §7, D41): frames from
// sensors/ through the read-time lens, calibration from rig.lens and the
// canonical /tf_static, ground truth from the canonical GT pose.
//
//   ariadne_sim_vo --rec RECDIR --scene SCENEDIR --perception-config YAML
//                  --slam-config YAML --out DIR [--agent 0] [--frames N] [--seed S]
//                  [--degradation YAML]   (2.2: apply D with this named config; default none)
// ---------------------------------------------------------------------------
#include "perception/degradation/degradation_config.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    ariadne::tools::StereoVoArgs a;
    a.tool = "ariadne_sim_vo";
    std::string rec, scene, degradation;
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
        else if (k == "--out") a.out = v;
        else if (k == "--frames") a.frames = std::stoi(v);
        else if (k == "--seed") a.seed = std::stoull(v);
        else if (k == "--degradation") degradation = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || a.perception_config.empty() || a.slam_config.empty() || a.out.empty()) {
        std::fprintf(stderr, "usage: %s --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML "
                             "--out DIR [--agent K] [--frames N] [--seed S] [--degradation YAML]\n", argv[0]);
        return 2;
    }
    try {
        ariadne::runtime::SimSource source(rec, scene, agent);
        if (!degradation.empty()) {
            source.SetDegradation(ariadne::perception::LoadDegradation(degradation), a.seed, cv::getNumThreads());
        }
        return ariadne::tools::RunStereoVo(source, a, argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_sim_vo: %s\n", e.what());
        return 1;
    }
}
