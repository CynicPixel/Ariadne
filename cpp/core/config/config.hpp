#pragma once
// ---------------------------------------------------------------------------
// Strict configuration loading (Phase 0.1, arch §5.6).
//
// THE PROPERTY THAT MATTERS: an unknown key is an ERROR, never silently
// ignored. A permissive loader turns `radiance_scal: 0.02` into "use the
// default 1.0", so an entire dark-scene ablation matrix silently measures a
// fully lit scene and every number in it is wrong in a way no test would catch.
// Every key must be consumed, or loading fails and names the offender.
//
// The same argument applies to a missing required key: defaulting it hides the
// fact that a config is incomplete. Required keys are required.
//
// §5.6 requires degradation parameters to be named configs referenced by run
// manifests, never inline constants, which is what this exists to serve.
// ---------------------------------------------------------------------------
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::core {

/// Thrown for any configuration problem: missing file, bad syntax, missing
/// required key, wrong type, or an unrecognized key. Configuration errors are
/// startup errors, and a run that starts with the wrong parameters is worse
/// than one that does not start.
class ConfigError : public std::runtime_error {
public:
    explicit ConfigError(const std::string& what) : std::runtime_error(what) {}
};

/// A YAML mapping that tracks which keys have been read, so leftovers can be
/// reported. Construct, read every field, then call `RequireAllConsumed()`.
class StrictConfig {
public:
    static StrictConfig FromFile(const std::string& path) {
        YAML::Node node;
        try {
            node = YAML::LoadFile(path);
        } catch (const YAML::Exception& e) {
            throw ConfigError("failed to parse " + path + ": " + e.what());
        }
        if (!node.IsMap()) {
            throw ConfigError(path + ": top level must be a mapping");
        }
        return StrictConfig(std::move(node), path);
    }

    static StrictConfig FromString(const std::string& text,
                                   const std::string& origin = "<string>") {
        YAML::Node node;
        try {
            node = YAML::Load(text);
        } catch (const YAML::Exception& e) {
            throw ConfigError("failed to parse " + origin + ": " + e.what());
        }
        if (!node.IsMap()) {
            throw ConfigError(origin + ": top level must be a mapping");
        }
        return StrictConfig(std::move(node), origin);
    }

    /// Read a required key. Missing or wrong-typed is an error.
    template <typename T>
    T Require(const std::string& key) {
        consumed_.insert(key);
        const YAML::Node value = node_[key];
        if (!value || value.IsNull()) {
            throw ConfigError(origin_ + ": missing required key '" + key + "'");
        }
        try {
            return value.as<T>();
        } catch (const YAML::Exception&) {
            throw ConfigError(origin_ + ": key '" + key +
                              "' has the wrong type");
        }
    }

    /// Read an optional key, falling back to `fallback` when absent. A present
    /// but wrong-typed value is still an error: silence is a choice, a typo is
    /// not.
    template <typename T>
    T Optional(const std::string& key, T fallback) {
        consumed_.insert(key);
        const YAML::Node value = node_[key];
        if (!value || value.IsNull()) return fallback;
        try {
            return value.as<T>();
        } catch (const YAML::Exception&) {
            throw ConfigError(origin_ + ": key '" + key +
                              "' has the wrong type");
        }
    }

    bool Has(const std::string& key) const {
        const YAML::Node value = node_[key];
        return static_cast<bool>(value) && !value.IsNull();
    }

    /// A nested mapping, itself strict. The parent marks the key consumed; the
    /// child must be finished separately.
    StrictConfig Section(const std::string& key) {
        consumed_.insert(key);
        const YAML::Node value = node_[key];
        if (!value || !value.IsMap()) {
            throw ConfigError(origin_ + ": section '" + key +
                              "' is missing or is not a mapping");
        }
        return StrictConfig(value, origin_ + ":" + key);
    }

    /// Keys present in the file that nothing read. Sorted, so the message is
    /// stable across runs.
    std::vector<std::string> UnconsumedKeys() const {
        std::vector<std::string> extra;
        for (const auto& kv : node_) {
            const std::string key = kv.first.as<std::string>();
            if (consumed_.find(key) == consumed_.end()) extra.push_back(key);
        }
        std::sort(extra.begin(), extra.end());
        return extra;
    }

    /// Fail if anything was left over. This is the whole point of the class:
    /// it converts a silent typo into a startup error that names the key.
    void RequireAllConsumed() const {
        const std::vector<std::string> extra = UnconsumedKeys();
        if (extra.empty()) return;
        std::ostringstream msg;
        msg << origin_ << ": unrecognized key";
        if (extra.size() > 1) msg << "s";
        msg << " ";
        for (size_t i = 0; i < extra.size(); ++i) {
            if (i > 0) msg << ", ";
            msg << "'" << extra[i] << "'";
        }
        msg << ". A misspelled key would otherwise be silently ignored and the"
               " run would use defaults.";
        throw ConfigError(msg.str());
    }

    const std::string& Origin() const { return origin_; }

private:
    StrictConfig(YAML::Node node, std::string origin)
        : node_(std::move(node)), origin_(std::move(origin)) {}

    YAML::Node node_;
    std::string origin_;
    std::set<std::string> consumed_;
};

}  // namespace ariadne::core
