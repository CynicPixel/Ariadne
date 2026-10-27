#pragma once
// 2.1 (perception.md §3.4): the feature engine's and the selection caps' manifest entries, kept out of
// stereo_vo_runner.hpp so that shared file gains one call. A learned engine also records its weights' checksum, read
// from the export's provenance file
// (infra/scripts/models.sh writes manifest.yaml next to the weights, sha256 per file), the same way the dataset
// checksum is read from its provenance (runtime/datasets/stereo_source.hpp).
#include "perception/features/feature_engine.hpp"
#include "perception/perception_config.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::tools {

inline void RecordFeatureEngine(runtime::RunManifest& m, const perception::LocalFeatureEngine& engine,
                                const perception::PerceptionConfig& pc) {
    const perception::FeatureParams& p = pc.features;
    m.feature_engine = engine.Name();
    for (const auto& [k, v] : engine.ManifestFields()) m.extra[k] = v;
    // §7.3: the selection caps are manifest fields, not constants (perception.md §3.4).
    m.extra["max_keypoints"] = std::to_string(p.max_keypoints);
    m.extra["keypoint_grid"] = std::to_string(p.grid_cols) + "x" + std::to_string(p.grid_rows);
    m.extra["target_tracks"] = std::to_string(pc.tracker.target_tracks);
    m.extra["replenish_cadence"] = pc.tracker.replenish;   // §4.2
    if (pc.stereo.confidence != "none") {   // §10: what produced StereoDepth.confidence
        const perception::StereoParams& st = pc.stereo;
        m.extra["stereo_confidence"] = st.confidence + "/lrc" + std::to_string(st.wls_lrc_threshold) + "/radius" +
                                       std::to_string(st.wls_discontinuity_radius) + "/filtered_disparity:" +
                                       (st.wls_filtered_disparity ? "true" : "false");
    }
    if (p.engine != "xfeat") return;
    const std::filesystem::path w(p.xfeat_weights);
    const std::filesystem::path prov = w.parent_path() / "manifest.yaml";
    std::string sha = runtime::kUnknownField;
    if (std::filesystem::exists(prov)) {
        const YAML::Node n = YAML::LoadFile(prov.string());
        if (n["sha256"] && n["sha256"][w.filename().string()]) {
            sha = "sha256:" + n["sha256"][w.filename().string()].as<std::string>();
        }
        if (n["commit"]) m.extra["xfeat_upstream_commit"] = n["commit"].as<std::string>();
    }
    m.extra["xfeat_weights_checksum"] = sha;
}

/// results.yaml line for stereo.confidence = wls: cost of the right-matcher pass + filter per stereo pass (§10.7.4).
inline std::string StereoConfidenceResults(std::vector<double> ms) {
    std::sort(ms.begin(), ms.end());
    double sum = 0;
    for (double v : ms) sum += v;
    std::ostringstream o;
    o << "stereo_confidence: {mode: wls, stereo_passes: " << ms.size() << ", ms_mean: " << sum / double(ms.size())
      << ", ms_median: " << ms[ms.size() / 2] << ", ms_p90: " << ms[ms.size() * 9 / 10] << "}\n";
    return o.str();
}

}  // namespace ariadne::tools
