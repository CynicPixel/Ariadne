#pragma once
// 2.1 (PERCEPTION_2.1.md §3.4): the feature engine's manifest entries, kept out of stereo_vo_runner.hpp so that shared
// file gains one call. A learned engine also records its weights' checksum, read from the export's provenance file
// (infra/scripts/models.sh writes manifest.yaml next to the weights, sha256 per file), the same way the dataset
// checksum is read from its provenance (runtime/datasets/stereo_source.hpp).
#include "perception/features/feature_engine.hpp"
#include "perception/perception_config.hpp"
#include "runtime/manifest/run_manifest.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

namespace ariadne::tools {

inline void RecordFeatureEngine(runtime::RunManifest& m, const perception::LocalFeatureEngine& engine,
                                const perception::FeatureParams& p) {
    m.feature_engine = engine.Name();
    for (const auto& [k, v] : engine.ManifestFields()) m.extra[k] = v;
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

}  // namespace ariadne::tools
