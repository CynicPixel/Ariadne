#pragma once
// ---------------------------------------------------------------------------
// Configuration of Stage 3 as wired into the runner (plan 5.2, D146): the
// perception config's optional `falloff:` section. Absent = mode off = the
// stage does not exist, and no output changes. The lamp calibration itself is
// not here: it describes the recording's rig, so the tool takes it on the
// command line next to the degradation config (--lamp-calib).
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/radiometric/falloff_params.hpp"

#include <string>

namespace ariadne::perception::radiometric {

struct FalloffStageParams {
    // off       the stage is absent (default)
    // keyframe  the albedo proxy of every keyframe, from that keyframe's SGBM depth (D146; the every-frame SGBM
    //           variant is a later ablation row). It goes into the keyframe bundle; tracking is untouched.
    std::string mode = "off";
    // raw    the tracker detects on the rectified gray, as without the stage (default)
    // proxy  at keyframes it detects on the albedo proxy (8 bits = 255 x reflectance, clipped), inside its
    //        feature_mask; LK still tracks the gray. Needs tracker replenish: keyframe, so detection happens only
    //        where the proxy exists; the row's baseline is that cadence with features: raw (D146, §31.5).
    std::string features = "raw";
    FalloffParams params;

    bool Enabled() const { return mode != "off"; }
};

inline void ValidateFalloff(const FalloffStageParams& f) {
    const auto fail = [](const std::string& m) { throw core::ConfigError("falloff: " + m); };
    const FalloffParams& p = f.params;
    if (f.mode != "off" && f.mode != "keyframe") fail("mode off|keyframe");
    if (f.features != "raw" && f.features != "proxy") fail("features raw|proxy");
    if (f.features == "proxy" && f.mode == "off") fail("features: proxy needs a mode that makes the proxy");
    if (p.incidence != "calibrated" && p.incidence != "off") fail("incidence calibrated|off");
    if (p.normals != "rgbd_fals" && p.normals != "rgbd_linemod" && p.normals != "rgbd_sri" && p.normals != "rgbd_cross") {
        fail("normals rgbd_fals|rgbd_linemod|rgbd_sri|rgbd_cross");
    }
    if (p.normal_window_px != 1 && p.normal_window_px != 3 && p.normal_window_px != 5 && p.normal_window_px != 7) {
        fail("normal_window_px 1|3|5|7 (cv::RgbdNormals)");
    }
    if (!(p.linemod_diff_threshold_m > 0.0) || p.feature_margin_px < 0 || !(p.max_gain >= 1.0) ||
        !(p.min_rid > 0.0 && p.min_rid < 1.0) || !(p.min_cos > 0.0 && p.min_cos < 1.0)) {
        fail("linemod_diff_threshold_m > 0, feature_margin_px >= 0, max_gain >= 1, 0 < min_rid < 1, 0 < min_cos < 1");
    }
}

inline FalloffStageParams ParseFalloff(core::StrictConfig& s) {
    FalloffStageParams f;
    FalloffParams& p = f.params;
    f.mode = s.Require<std::string>("mode");
    f.features = s.Optional<std::string>("features", f.features);
    p.incidence = s.Optional<std::string>("incidence", p.incidence);
    p.normals = s.Optional<std::string>("normals", p.normals);
    p.normal_window_px = s.Optional<int>("normal_window_px", p.normal_window_px);
    p.linemod_diff_threshold_m = s.Optional<double>("linemod_diff_threshold_m", p.linemod_diff_threshold_m);
    p.feature_margin_px = s.Optional<int>("feature_margin_px", p.feature_margin_px);
    p.max_gain = s.Optional<double>("max_gain", p.max_gain);
    p.min_rid = s.Optional<double>("min_rid", p.min_rid);
    p.min_cos = s.Optional<double>("min_cos", p.min_cos);
    s.RequireAllConsumed();
    ValidateFalloff(f);
    return f;
}

}  // namespace ariadne::perception::radiometric
