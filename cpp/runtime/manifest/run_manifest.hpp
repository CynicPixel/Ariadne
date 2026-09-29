#pragma once
// ---------------------------------------------------------------------------
// Run manifest (arch §28.2, §36; CLAUDE.md "manifests or it didn't happen").
//
// Every run records what produced it. This is what makes a figure reproducible
// six weeks later and what lets a reviewer rebuild one result from the written
// instructions alone (the Phase 8 clean-machine test).
//
// Two rules the field set encodes:
//   - Container images are referenced BY DIGEST, never by tag. A tag is a
//     mutable pointer, so "ariadne-core:latest" identifies nothing.
//   - The OpenCV build is identified by a hash of cv::getBuildInformation()
//     (C7), not by a version string. Two builds of the same tag with different
//     module sets or SIMD settings are different software and can produce
//     different numbers.
//
// Emitted as YAML because §28.2 specifies it and because a human reads this
// far more often than a machine does.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::runtime {

/// A field that has not been filled in. Written to the manifest verbatim so an
/// unknown is visible rather than silently absent, which is the difference
/// between "we did not record this" and "this was empty".
inline constexpr const char* kUnknownField = "UNRECORDED";

struct RunManifest {
    // --- Provenance -------------------------------------------------------
    std::string run_id;
    std::string created_utc;
    std::string git_commit = kUnknownField;
    bool git_dirty = false;         // a dirty tree means the commit does not
                                    // identify the code that ran
    std::string image_digest = kUnknownField;

    // --- Software identity ------------------------------------------------
    std::string opencv_version = kUnknownField;
    std::string opencv_build_hash = kUnknownField;   // C7
    /// The CPU features OpenCV's dispatcher saw on this machine; the same
    /// build can select different SIMD kernels elsewhere (opencv_identity.hpp).
    std::string opencv_cpu_features = kUnknownField;
    std::string cool_version = kUnknownField;
    std::string cuda_version = kUnknownField;
    std::string compiler = kUnknownField;
    std::string architecture = kUnknownField;

    // --- Experiment -------------------------------------------------------
    std::string scene = kUnknownField;
    uint64_t seed = 0;
    // 0 is a legitimate seed, so the value alone cannot distinguish "seeded
    // with 0" from "nobody recorded the seed". The flag makes the difference
    // explicit; MissingRequiredFields treats an unset flag as unrecorded.
    bool seed_recorded = false;
    int agent_count = 0;
    /// Which regime produced this run. §5.9 requires every figure to declare
    /// it, because replayed runs are bit-deterministic and live closed-loop
    /// runs are not and must be reported over seeds.
    std::string execution_mode = "replay";           // "replay" | "live"
    std::string bag_checksum = kUnknownField;

    // --- Configuration selections (§28.2) ---------------------------------
    std::string camera_config = kUnknownField;
    std::string illumination_config = kUnknownField;
    std::string communication_config = kUnknownField;
    std::string planner_config = kUnknownField;
    std::string degradation_config = kUnknownField;  // e.g. dark2_dust1 (§5.6)
    /// §7.3: keypoint caps and SGBM range are manifest fields, not constants.
    std::string perception_config = kUnknownField;   // e.g. placeholder
    std::string slam_config = kUnknownField;         // configs/slam/*.yaml name

    // --- Backend selections -----------------------------------------------
    std::string feature_engine = kUnknownField;
    std::string place_recognition = kUnknownField;
    std::string loop_backend = kUnknownField;
    std::string reconstruction_backend = kUnknownField;
    /// §10.7: which inertial mode actually ran. Recorded so a fallback is an
    /// honest downgrade rather than a quiet one, and so the report's sensor
    /// claim can be checked against any individual figure.
    std::string imu_mode = "attitude_only";

    // --- Determinism digests ---------------------------------------------
    /// Committed digests from the determinism gate. Storing them per run means
    /// a result produced by drifted math is identifiable after the fact.
    std::string det_math_digest = kUnknownField;
    std::string philox_digest = kUnknownField;
    std::string degradation_digest = kUnknownField;
    /// Final state_hash of the event log (§41).
    std::string final_state_hash = kUnknownField;

    /// Anything not covered above. Kept ordered so the output is stable.
    std::map<std::string, std::string> extra;
};

namespace detail {

inline std::string YamlEscape(const std::string& in) {
    // Quote when the value could otherwise be parsed as something else.
    bool needs_quotes = in.empty();
    for (char c : in) {
        if (c == ':' || c == '#' || c == '\'' || c == '"' || c == '\n' ||
            c == '{' || c == '[' || c == '*' || c == '&') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) return in;
    std::string out = "\"";
    for (char c : in) {
        if (c == '"' || c == '\\') out += '\\';
        if (c == '\n') { out += "\\n"; continue; }
        out += c;
    }
    out += '"';
    return out;
}

inline void Field(std::ostream& out, const char* key, const std::string& value) {
    out << key << ": " << YamlEscape(value) << "\n";
}

}  // namespace detail

/// Serialize to YAML. Field order is fixed so two manifests can be diffed.
inline std::string ToYaml(const RunManifest& m) {
    std::ostringstream out;
    out << "# Ariadne run manifest (arch §28.2). Generated; do not hand-edit.\n";
    detail::Field(out, "run_id", m.run_id);
    detail::Field(out, "created_utc", m.created_utc);
    detail::Field(out, "git_commit", m.git_commit);
    out << "git_dirty: " << (m.git_dirty ? "true" : "false") << "\n";
    detail::Field(out, "image_digest", m.image_digest);

    out << "\n# Software identity\n";
    detail::Field(out, "opencv_version", m.opencv_version);
    detail::Field(out, "opencv_build_hash", m.opencv_build_hash);
    detail::Field(out, "opencv_cpu_features", m.opencv_cpu_features);
    detail::Field(out, "cool_version", m.cool_version);
    detail::Field(out, "cuda_version", m.cuda_version);
    detail::Field(out, "compiler", m.compiler);
    detail::Field(out, "architecture", m.architecture);

    out << "\n# Experiment\n";
    detail::Field(out, "scene", m.scene);
    out << "seed: " << m.seed << "\n";
    out << "agent_count: " << m.agent_count << "\n";
    detail::Field(out, "execution_mode", m.execution_mode);
    detail::Field(out, "bag_checksum", m.bag_checksum);

    out << "\n# Configuration\n";
    detail::Field(out, "camera_config", m.camera_config);
    detail::Field(out, "illumination_config", m.illumination_config);
    detail::Field(out, "communication_config", m.communication_config);
    detail::Field(out, "planner_config", m.planner_config);
    detail::Field(out, "degradation_config", m.degradation_config);
    detail::Field(out, "perception_config", m.perception_config);
    detail::Field(out, "slam_config", m.slam_config);

    out << "\n# Backends\n";
    detail::Field(out, "feature_engine", m.feature_engine);
    detail::Field(out, "place_recognition", m.place_recognition);
    detail::Field(out, "loop_backend", m.loop_backend);
    detail::Field(out, "reconstruction_backend", m.reconstruction_backend);
    detail::Field(out, "imu_mode", m.imu_mode);

    out << "\n# Determinism\n";
    detail::Field(out, "det_math_digest", m.det_math_digest);
    detail::Field(out, "philox_digest", m.philox_digest);
    detail::Field(out, "degradation_digest", m.degradation_digest);
    detail::Field(out, "final_state_hash", m.final_state_hash);

    if (!m.extra.empty()) {
        out << "\n# Additional\n";
        for (const auto& kv : m.extra) {
            detail::Field(out, kv.first.c_str(), kv.second);
        }
    }
    return out.str();
}

/// Fields that must be set before a run's results may be published.
/// Returns the names of any still unrecorded, so a caller can refuse to publish
/// rather than emit a figure nobody can reproduce.
inline std::vector<std::string> MissingRequiredFields(const RunManifest& m) {
    std::vector<std::string> missing;
    auto require = [&missing](const char* name, const std::string& value) {
        if (value.empty() || value == kUnknownField) missing.emplace_back(name);
    };
    require("run_id", m.run_id);
    require("git_commit", m.git_commit);
    require("scene", m.scene);
    require("feature_engine", m.feature_engine);
    require("loop_backend", m.loop_backend);

    // §5.6: degradation parameters are named configs REFERENCED BY MANIFESTS.
    // Without this field a run cannot be re-created, which is the entire point
    // of the manifest, so it is required rather than merely serialized.
    require("degradation_config", m.degradation_config);
    // NOT `require(...)`: execution_mode defaults to "replay", so a presence
    // check can never fire and would read as a guarantee while checking
    // nothing. §5.9 makes the DISTINCTION load-bearing -- replayed runs are
    // bit-deterministic, live ones are not and must be reported over seeds --
    // so what matters is that the value is one of the two known modes.
    if (m.execution_mode != "replay" && m.execution_mode != "live") {
        missing.emplace_back("execution_mode(must be 'replay' or 'live')");
    }
    if (!m.seed_recorded) missing.emplace_back("seed");

    // The determinism digests exist "so a result produced by drifted math is
    // identifiable after the fact". A manifest that omits them cannot support
    // that claim, so omitting them makes the run unpublishable.
    require("det_math_digest", m.det_math_digest);
    require("philox_digest", m.philox_digest);
    require("degradation_digest", m.degradation_digest);

    // A dirty tree means git_commit does not identify what actually ran.
    if (m.git_dirty) missing.emplace_back("git_dirty(tree was modified)");
    return missing;
}

}  // namespace ariadne::runtime
