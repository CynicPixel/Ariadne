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
//                  [--right-camera NAME]  (5.5, D138: a sweep rig's right_b04/b08/b12/b20; default the main "right")
//                  [--depth-eval 1]       (5.5, D148: SGBM depth vs the recording's GT depth at keyframes, by range bin;
//                                 evaluation only: results.yaml depth_eval block)
//                  [--spool DIR --recon-config YAML]  (6.1, D195: write a reconstruction spool of this agent, schema 2
//                                 when a lamp calibration or a photometric state exists: the albedo proxy, the applied
//                                 exposure/lamp state per bundle, the lamp; never changes the run)
// ---------------------------------------------------------------------------
#include "perception/degradation/degradation_config.hpp"
#include "perception/radiometric/falloff.hpp"
#include "reconstruction/spool/keyframe_spool.hpp"
#include "runtime/agent/keyframe_records.hpp"
#include "runtime/tools/stereo_vo_runner.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

int main(int argc, char** argv) {
    ariadne::tools::StereoVoArgs a;
    a.tool = "ariadne_sim_vo";
    std::string rec, scene, degradation, imu_noise, radiance_event, right_camera = "right", spool_dir, recon_path;
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
        else if (k == "--right-camera") right_camera = v;
        else if (k == "--spool") spool_dir = v;
        else if (k == "--depth-eval") a.depth_eval = (v == "1");
        else if (k == "--recon-config") recon_path = v;
        else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
    }
    if (rec.empty() || scene.empty() || a.perception_config.empty() || a.slam_config.empty() || a.out.empty()) {
        std::fprintf(stderr, "usage: %s --rec RECDIR --scene SCENEDIR --perception-config YAML --slam-config YAML "
                             "--out DIR [--agent K] [--frames N] [--seed S] [--degradation YAML] [--imu-noise YAML] [--lamp-calib YAML] [--radiance-event AT_S:STOPS] [--right-camera NAME] [--depth-eval 1] [--spool DIR --recon-config YAML]\n", argv[0]);
        return 2;
    }
    try {
        ariadne::runtime::SimSource source(rec, scene, agent, "default", right_camera);
        if (!degradation.empty()) {
            source.SetDegradation(ariadne::perception::LoadDegradation(degradation), a.seed, cv::getNumThreads());
        }
        if (a.depth_eval) source.SetGtDepth(true);
        if (!imu_noise.empty()) source.SetImuNoise(ariadne::core::LoadImuNoiseOperator(imu_noise), a.seed);
        if (!radiance_event.empty()) {
            const size_t colon = radiance_event.find(':');
            if (colon == std::string::npos) throw std::invalid_argument("--radiance-event wants AT_S:STOPS");
            const ariadne::runtime::RadianceEvent e{std::stod(radiance_event.substr(0, colon)),
                                                    std::stod(radiance_event.substr(colon + 1))};
            source.SetRadianceEvent(e);
            a.scenario = "radiance_event at_s=" + std::to_string(e.at_s) + " stops=" + std::to_string(e.stops);
        }
        // 6.1 (D195): the spool hooks. They read the canonical records the runner builds and never change the run.
        namespace rc = ariadne::reconstruction;
        namespace fs = std::filesystem;
        if (spool_dir.empty() != recon_path.empty()) throw std::invalid_argument("--spool needs --recon-config");
        std::unique_ptr<rc::KeyframeSpool> spool;
        std::optional<rc::SpoolCamera> spool_cam;
        bool lamp_pending = false;
        if (!spool_dir.empty()) {
            if (fs::exists(spool_dir) && !fs::is_empty(spool_dir)) {
                throw std::invalid_argument("--spool " + spool_dir + " exists and is not empty");
            }
            const YAML::Node sp = YAML::LoadFile(recon_path)["spool"];
            rc::SpoolParams p;
            p.scale = sp["scale"].as<double>();
            p.keyframe_stride = sp["keyframe_stride"].as<int>();
            p.heldout_stride = sp["heldout_stride"].as<int>();
            spool = std::make_unique<rc::KeyframeSpool>(spool_dir, p);
            lamp_pending = !a.lamp_calib.empty() && source.Lamp().has_value();
        }
        auto photo_of = [](const std::optional<ariadne::runtime::FramePhoto>& f) -> std::optional<rc::SpoolPhoto> {
            if (!f) return std::nullopt;
            return rc::SpoolPhoto{f->ExposureGain(), f->LampScale(), f->decision_id};
        };
        if (spool) {
            a.on_keyframe = [&](const ariadne::tools::KeyframeView& kv) {
                const ariadne::runtime::KeyframeBundle& b = kv.bundle;
                if (lamp_pending) {   // the rectification is known from the first keyframe's rig
                    const ariadne::perception::radiometric::LampCalibration c =
                        ariadne::perception::radiometric::LoadLampCalibration(a.lamp_calib);
                    rc::SpoolLamp l;
                    l.calib_source = a.lamp_calib;
                    l.calib_name = c.name;
                    l.rig = source.Lamp()->rig;
                    l.intensity = source.Lamp()->intensity;
                    // X_rect = R1 X_raw and T_B_Crect.R X_rect = T_B_C0.R X_raw, so R1 = T_B_Crect.R^T T_B_C0.R.
                    const ariadne::core::Mat3 R1 = kv.T_B_C.R.transpose() * source.T_B_C0().R;
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j) l.R_rect_raw(i, j) = R1(i, j);
                    spool->SetLamp(l);
                    lamp_pending = false;
                }
                spool_cam = spool->Camera(b.K.fx, b.K.fy, b.K.cx, b.K.cy, b.rgb.cols, b.rgb.rows);
                spool->AddKeyframe(b.agent, kv.keyframe.frame_id, kv.keyframe.t_ns, b.frame_id_of_pose, b.T_WiC,
                                   *spool_cam, kv.baseline_m, b.rgb, ariadne::runtime::MaskedDepth32(b.depth, b.mask),
                                   ariadne::runtime::MaskedDepth32(b.depth_sigma, b.mask), b.albedo, b.albedo_valid,
                                   photo_of(kv.photo));
            };
            a.on_frame = [&](const ariadne::tools::FrameView& fv) {
                if (fv.is_keyframe || !spool_cam || !spool->WantHeldout(fv.frame_id)) return;
                const cv::Mat rgb = fv.left_rect_rgb();
                if (rgb.empty()) return;
                spool->AddHeldout(source.Agent(), fv.frame_id, fv.t_ns, fv.T_O_C, *spool_cam, rgb, photo_of(fv.photo));
            };
        }
        const int rc_run = ariadne::tools::RunStereoVo(source, a, argc, argv);
        if (spool) {
            const std::string scene_id = YAML::LoadFile(scene + "/scene.json")["scene_id"].as<std::string>();
            spool->Finish(source.RunName(), scene_id, false, ariadne::core::SE3(), "none (single-agent ariadne_sim_vo)");
            std::printf("spool: %zu bundles -> %s\n", spool->size(), spool_dir.c_str());
        }
        return rc_run;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ariadne_sim_vo: %s\n", e.what());
        return 1;
    }
}
