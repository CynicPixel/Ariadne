// ---------------------------------------------------------------------------
// ariadne_sim_vo: the same 1.4 + 1.5 pipeline as ariadne_euroc_vo, over one
// agent of a simulator recording (design SIM_1.1-1.3 §7, D41): frames from
// sensors/ through the read-time lens, calibration from rig.lens and the
// canonical /tf_static, ground truth from the canonical GT pose.
//
//   ariadne_sim_vo --rec RECDIR --scene SCENEDIR --perception-config YAML
//                  --slam-config YAML --out DIR [--agent 0] [--frames N] [--seed S]
//                  [--degradation YAML]   (2.2: apply D with this named config; default none)
//                  [--imu-noise YAML]     (L2, D101: expose the IMU through this seeded noise operator; default no IMU)
//                  [--lamp-calib YAML]    (5.2, D146: the rig's lamp calibration, for a perception falloff: section)
//                  [--radiance-event AT_S:STOPS]  (5.3, D147: Failure E -- the scene 2^STOPS brighter from AT_S on)
// ---------------------------------------------------------------------------
#include "perception/degradation/degradation_config.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    ariadne::tools::StereoVoArgs a;
    a.tool = "ariadne_sim_vo";
    std::string rec, scene, degradation, imu_noise, radiance_event;
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
        else if (k == "--imu-noise") imu_noise = v;
        else if (k == "--lamp-calib") a.lamp_calib = v;
        else if (k == "--radiance-event") radiance_event = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || a.perception_config.empty() || a.slam_config.empty() || a.out.empty()) {
        std::fprintf(stderr, "usage: %s --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML "
                             "--out DIR [--agent K] [--frames N] [--seed S] [--degradation YAML] [--imu-noise YAML] [--lamp-calib YAML] [--radiance-event AT_S:STOPS]\n", argv[0]);
        return 2;
    }
    try {
        ariadne::runtime::SimSource source(rec, scene, agent);
        if (!degradation.empty()) {
            source.SetDegradation(ariadne::perception::LoadDegradation(degradation), a.seed, cv::getNumThreads());
        }
        if (!imu_noise.empty()) source.SetImuNoise(ariadne::core::LoadImuNoiseOperator(imu_noise), a.seed);
        if (!radiance_event.empty()) {
            const size_t colon = radiance_event.find(':');
            if (colon == std::string::npos) throw std::invalid_argument("--radiance-event wants AT_S:STOPS");
            const ariadne::runtime::RadianceEvent e{std::stod(radiance_event.substr(0, colon)),
                                                    std::stod(radiance_event.substr(colon + 1))};
            source.SetRadianceEvent(e);
            a.scenario = "radiance_event at_s=" + std::to_string(e.at_s) + " stops=" + std::to_string(e.stops);
        }
        return ariadne::tools::RunStereoVo(source, a, argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_sim_vo: %s\n", e.what());
        return 1;
    }
}
