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
#include <vector>

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
    // A zero frequency maps every pixel onto lattice cell 0, so the dust field
    // becomes a spatially uniform constant while still LOOKING like a field in
    // the config. Silent uniformity is exactly the class of error the strict
    // loader exists to prevent.
    if (p.dust_field_frequency <= 0.0) {
        fail("dust_field_frequency must be > 0; zero collapses the field to a"
             " single lattice cell and makes the dust spatially uniform");
    }
    // SamplePoisson's exact branch is Knuth's, whose loop runs O(lambda) times
    // and is hard-capped at 10000 iterations. Raising this threshold past the
    // cap would make bright pixels return exactly 10000 photons -- a plausible
    // image built on fabricated statistics. The Gaussian approximation is
    // already accurate by lambda=30, so there is no reason to exceed 50 and
    // every reason to keep the cap unreachable.
    if (p.dust_field_tau_frames < 0) fail("dust_field_tau_frames must be >= 0 (0 = redrawn every frame)");
    if (!(p.lamp_power > 0.0)) fail("lamp_power must be > 0");
    if (p.dust_airlight_mode == DustAirlight::kLamp &&
        !(p.dust_lamp_focal_px > 0.0 && p.dust_lamp_cos_outer > 0.0 &&
          p.dust_lamp_cos_outer < p.dust_lamp_cos_inner && p.dust_lamp_cos_inner <= 1.0)) {
        fail("dust_airlight_mode lamp needs dust_lamp_focal_px > 0 and 0 < dust_lamp_cos_outer < "
             "dust_lamp_cos_inner <= 1");
    }
    if (p.dust_model == DustModel::kHeadlamp) {
        if (!p.enable_dust) fail("dust_model headlamp needs enable_dust");
        if (p.dust_order != DustOrder::kPreSensor) {
            fail("dust_model headlamp is light in the medium: set dust_order pre_sensor explicitly");
        }
        if (p.dust_backscatter < 0.0 || p.dust_beta_backscatter < 0.0 || !(p.dust_near_m > 0.0)) {
            fail("headlamp: dust_backscatter >= 0, dust_beta_backscatter >= 0, dust_near_m > 0");
        }
        if (!(p.dust_lamp_focal_px > 0.0 && p.dust_lamp_cos_outer > 0.0 &&
              p.dust_lamp_cos_outer < p.dust_lamp_cos_inner && p.dust_lamp_cos_inner <= 1.0)) {
            fail("headlamp needs dust_lamp_focal_px > 0 and 0 < dust_lamp_cos_outer < dust_lamp_cos_inner <= 1");
        }
    }
    if (p.poisson_exact_max_lambda <= 0.0 || p.poisson_exact_max_lambda > 50.0) {
        fail("poisson_exact_max_lambda must be in (0, 50]; beyond that Knuth's"
             " loop hits its iteration cap and silently truncates the count");
    }
}

/// Parse a degradation config. Every key is read explicitly and the config is
/// then checked for leftovers, so a misspelled parameter fails here rather than
/// silently reverting to a default.
inline NamedDegradation ParseDegradation(StrictConfig& config) {
    NamedDegradation out;
    out.name = config.Require<std::string>("name");
    DegradationParams& p = out.params;

    const std::string transfer = config.Optional<std::string>("input_transfer", "linear");
    if (transfer == "srgb") {
        p.input_transfer = InputTransfer::kSrgb;
    } else if (transfer != "linear") {
        throw ConfigError("degradation config '" + out.name + "': input_transfer must be linear or srgb");
    }
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

    // L9 / D135 fidelity options and the replay lamp actuator (all optional; absent = the §5.6 operator).
    const std::string order = config.Optional<std::string>("dust_order", "post_sensor");
    if (order == "pre_sensor") {
        p.dust_order = DustOrder::kPreSensor;
    } else if (order != "post_sensor") {
        throw ConfigError("degradation config '" + out.name + "': dust_order must be post_sensor or pre_sensor");
    }
    p.dust_field_tau_frames = config.Optional<int>("dust_field_tau_frames", 0);
    const auto drift = config.Optional<std::vector<double>>("dust_drift_px_per_frame", {0.0, 0.0});
    if (drift.size() != 2) {
        throw ConfigError("degradation config '" + out.name + "': dust_drift_px_per_frame is [x, y]");
    }
    p.dust_drift_x_px = drift[0];
    p.dust_drift_y_px = drift[1];
    const std::string air = config.Optional<std::string>("dust_airlight_mode", "constant");
    if (air == "lamp") {
        p.dust_airlight_mode = DustAirlight::kLamp;
    } else if (air != "constant") {
        throw ConfigError("degradation config '" + out.name + "': dust_airlight_mode must be constant or lamp");
    }
    p.dust_lamp_cos_inner = config.Optional<double>("dust_lamp_cos_inner", 0.0);
    p.dust_lamp_cos_outer = config.Optional<double>("dust_lamp_cos_outer", 0.0);
    p.dust_lamp_focal_px = config.Optional<double>("dust_lamp_focal_px", 0.0);
    p.lamp_power = config.Optional<double>("lamp_power", 1.0);
    const std::string model = config.Optional<std::string>("dust_model", "koschmieder");
    if (model == "headlamp") {
        p.dust_model = DustModel::kHeadlamp;
    } else if (model != "koschmieder") {
        throw ConfigError("degradation config '" + out.name + "': dust_model must be koschmieder or headlamp");
    }
    p.dust_beta_backscatter = config.Optional<double>("dust_beta_backscatter", 0.0);
    p.dust_backscatter = config.Optional<double>("dust_backscatter", 0.0);
    p.dust_near_m = config.Optional<double>("dust_near_m", p.dust_near_m);
    const auto lamp = config.Optional<std::vector<double>>("dust_lamp_in_camera_m", {0.0, 0.0, 0.0});
    if (lamp.size() != 3) {
        throw ConfigError("degradation config '" + out.name + "': dust_lamp_in_camera_m is [x, y, z] (optical frame)");
    }
    p.dust_lamp_x_m = lamp[0];
    p.dust_lamp_y_m = lamp[1];
    p.dust_lamp_z_m = lamp[2];
    config.RequireAllConsumed();
    ValidateDegradation(out);
    return out;
}

inline NamedDegradation LoadDegradation(const std::string& path) {
    StrictConfig config = StrictConfig::FromFile(path);
    return ParseDegradation(config);
}

}  // namespace ariadne::perception
