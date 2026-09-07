#pragma once
// ---------------------------------------------------------------------------
// Named degradation configs (arch §5.6).
//
// "Parameter vectors are named configs (dark2_dust1, ...) referenced by run
// manifests, never inline constants." This is the binding between a config file
// on disk and the DegradationParams the operator takes.
//
// The name travels into the run manifest, so a figure can be traced back to the
// exact parameter vector that produced it without anyone having to remember
// what "the dusty one" meant six weeks later.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/degradation/degradation.hpp"

#include <string>

namespace ariadne::perception {

using core::ConfigError;
using core::StrictConfig;

struct NamedDegradation {
    std::string name;
    DegradationParams params;
};

/// Range checks. These catch physically meaningless values that would still
/// parse: a negative photon scale, a saturation above full scale, a bit depth
/// that cannot be represented. Each would otherwise produce a plausible-looking
/// but meaningless image.
inline void ValidateDegradation(const NamedDegradation& d) {
    const DegradationParams& p = d.params;
    auto fail = [&d](const std::string& why) {
        throw ConfigError("degradation config '" + d.name + "': " + why);
    };
    if (d.name.empty()) fail("name must not be empty");
    if (p.radiance_scale < 0.0) fail("radiance_scale must be >= 0");
    if (p.photon_scale <= 0.0) fail("photon_scale must be > 0");
    if (p.read_noise_electrons < 0.0) fail("read_noise_electrons must be >= 0");
    if (p.vignette_strength < 0.0 || p.vignette_strength >= 1.0) {
        fail("vignette_strength must be in [0, 1)");
    }
    if (p.prnu_sigma < 0.0) fail("prnu_sigma must be >= 0");
    if (p.exposure <= 0.0) fail("exposure must be > 0");
    if (p.analog_gain <= 0.0 || p.digital_gain <= 0.0) {
        fail("gains must be > 0");
    }
    if (p.bit_depth < 1 || p.bit_depth > 16) fail("bit_depth must be in [1, 16]");
    if (p.saturation <= 0.0 || p.saturation > 1.0) {
        fail("saturation must be in (0, 1]");
    }
    if (p.dust_beta < 0.0) fail("dust_beta must be >= 0");
    if (p.dust_airlight < 0.0 || p.dust_airlight > 1.0) {
        fail("dust_airlight must be in [0, 1]");
    }
    if (p.dust_field_strength < 0.0 || p.dust_field_strength > 1.0) {
        fail("dust_field_strength must be in [0, 1]");
    }
    if (p.dust_field_octaves < 0 || p.dust_field_octaves > 16) {
        fail("dust_field_octaves must be in [0, 16]");
    }
    if (p.enable_dust && p.dust_beta <= 0.0) {
        fail("enable_dust is true but dust_beta is 0, which applies no dust");
    }
}

/// Parse a degradation config. Every key is read explicitly and the config is
/// then checked for leftovers, so a misspelled parameter fails here rather than
/// silently reverting to a default.
inline NamedDegradation ParseDegradation(StrictConfig& config) {
    NamedDegradation out;
    out.name = config.Require<std::string>("name");
    DegradationParams& p = out.params;

    p.radiance_scale = config.Require<double>("radiance_scale");

    p.enable_shot_noise = config.Optional<bool>("enable_shot_noise", true);
    p.photon_scale = config.Optional<double>("photon_scale", p.photon_scale);
    p.poisson_exact_max_lambda =
        config.Optional<double>("poisson_exact_max_lambda",
                                p.poisson_exact_max_lambda);
    p.read_noise_electrons =
        config.Optional<double>("read_noise_electrons", p.read_noise_electrons);

    p.vignette_strength =
        config.Optional<double>("vignette_strength", p.vignette_strength);
    p.prnu_sigma = config.Optional<double>("prnu_sigma", p.prnu_sigma);

    p.exposure = config.Optional<double>("exposure", p.exposure);
    p.analog_gain = config.Optional<double>("analog_gain", p.analog_gain);
    p.digital_gain = config.Optional<double>("digital_gain", p.digital_gain);

    p.bit_depth = config.Optional<int>("bit_depth", p.bit_depth);
    p.saturation = config.Optional<double>("saturation", p.saturation);

    p.enable_dust = config.Optional<bool>("enable_dust", false);
    p.dust_beta = config.Optional<double>("dust_beta", p.dust_beta);
    p.dust_airlight = config.Optional<double>("dust_airlight", p.dust_airlight);
    p.dust_field_frequency =
        config.Optional<double>("dust_field_frequency", p.dust_field_frequency);
    p.dust_field_octaves =
        config.Optional<int>("dust_field_octaves", p.dust_field_octaves);
    p.dust_field_strength =
        config.Optional<double>("dust_field_strength", p.dust_field_strength);

    config.RequireAllConsumed();
    ValidateDegradation(out);
    return out;
}

inline NamedDegradation LoadDegradation(const std::string& path) {
    StrictConfig config = StrictConfig::FromFile(path);
    return ParseDegradation(config);
}

}  // namespace ariadne::perception
