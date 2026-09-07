// Strict config loading (Phase 0.1) and the named degradation configs (§5.6).
#include "core/config/config.hpp"
#include "perception/degradation/degradation_config.hpp"

#include <gtest/gtest.h>

#include <set>
#include <string>

using namespace ariadne::core;
using namespace ariadne::perception;

namespace {

std::string ConfigDir() {
    // Tests run from the build directory; configs live in the source tree.
    return std::string(ARIADNE_SOURCE_DIR) + "/configs/degradation/";
}

}  // namespace

// --- The property the loader exists for ----------------------------------

// A misspelled key must be an error. If it were silently ignored, a dark-scene
// ablation could measure a fully lit scene and no test would notice.
TEST(StrictConfig, UnknownKeyIsAnError) {
    StrictConfig config = StrictConfig::FromString(
        "name: x\nradiance_scale: 0.5\nradiance_scal: 0.02\n");
    EXPECT_THROW(
        {
            try {
                ParseDegradation(config);
            } catch (const ConfigError& e) {
                // The message must name the offender, or the error is useless.
                EXPECT_NE(std::string(e.what()).find("radiance_scal"),
                          std::string::npos);
                throw;
            }
        },
        ConfigError);
}

TEST(StrictConfig, MissingRequiredKeyIsAnError) {
    StrictConfig config = StrictConfig::FromString("name: x\n");
    EXPECT_THROW(ParseDegradation(config), ConfigError);
}

TEST(StrictConfig, WrongTypeIsAnError) {
    StrictConfig config = StrictConfig::FromString(
        "name: x\nradiance_scale: not_a_number\n");
    EXPECT_THROW(ParseDegradation(config), ConfigError);
}

TEST(StrictConfig, MalformedYamlIsAnError) {
    EXPECT_THROW(StrictConfig::FromString("key: [unclosed\n"), ConfigError);
    EXPECT_THROW(StrictConfig::FromString("- just\n- a\n- list\n"), ConfigError);
}

TEST(StrictConfig, MissingFileIsAnError) {
    EXPECT_THROW(StrictConfig::FromFile("/nonexistent/path.yaml"), ConfigError);
}

TEST(StrictConfig, OptionalKeysFallBackButStillTypeCheck) {
    StrictConfig config =
        StrictConfig::FromString("present: 5\nbadtype: hello\n");
    EXPECT_EQ(config.Optional<int>("present", 0), 5);
    EXPECT_EQ(config.Optional<int>("absent", 42), 42);
    // A key that IS present must still parse: absence is a choice, a typo'd
    // value is not.
    EXPECT_THROW(config.Optional<int>("badtype", 0), ConfigError);
}

TEST(StrictConfig, ReportsEveryUnconsumedKeySorted) {
    StrictConfig config =
        StrictConfig::FromString("known: 1\nzebra: 2\nalpha: 3\n");
    config.Require<int>("known");
    const auto extra = config.UnconsumedKeys();
    ASSERT_EQ(extra.size(), 2u);
    EXPECT_EQ(extra[0], "alpha");   // sorted, so messages are stable
    EXPECT_EQ(extra[1], "zebra");
    EXPECT_THROW(config.RequireAllConsumed(), ConfigError);
}

// --- Validation -----------------------------------------------------------

// Values that parse but are physically meaningless must be rejected, since they
// would otherwise produce a plausible-looking but meaningless image.
TEST(DegradationConfig, RejectsPhysicallyMeaninglessValues) {
    auto parse = [](const std::string& body) {
        StrictConfig c = StrictConfig::FromString(body);
        return ParseDegradation(c);
    };
    EXPECT_THROW(parse("name: x\nradiance_scale: -1.0\n"), ConfigError);
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\nphoton_scale: 0\n"),
                 ConfigError);
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\nvignette_strength: 1.0\n"),
                 ConfigError);
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\nbit_depth: 0\n"),
                 ConfigError);
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\nsaturation: 1.5\n"),
                 ConfigError);
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\ndust_airlight: 2.0\n"),
                 ConfigError);
    // Dust enabled with zero extinction applies nothing, which is almost
    // certainly a mistake in the config rather than an intent.
    EXPECT_THROW(parse("name: x\nradiance_scale: 1.0\nenable_dust: true\n"
                       "dust_beta: 0.0\n"),
                 ConfigError);
}

TEST(DegradationConfig, AcceptsAMinimalValidConfig) {
    StrictConfig c = StrictConfig::FromString("name: minimal\nradiance_scale: 0.5\n");
    const NamedDegradation d = ParseDegradation(c);
    EXPECT_EQ(d.name, "minimal");
    EXPECT_EQ(d.params.radiance_scale, 0.5);
    EXPECT_FALSE(d.params.enable_dust);
}

// --- The 16 shipped configs (§5.6) ---------------------------------------

TEST(DegradationConfig, AllSixteenNamedConfigsLoadAndValidate) {
    int loaded = 0;
    for (int dark = 0; dark <= 3; ++dark) {
        for (int dust = 0; dust <= 3; ++dust) {
            const std::string name = "dark" + std::to_string(dark) + "_dust" +
                                     std::to_string(dust);
            const NamedDegradation d = LoadDegradation(ConfigDir() + name + ".yaml");
            EXPECT_EQ(d.name, name) << "name field must match the filename, or a"
                                       " manifest reference is ambiguous";
            ++loaded;
        }
    }
    EXPECT_EQ(loaded, 16);
}

// The ladder must be monotone, or the DVE matrix rows are not ordered by
// severity and the ablation plot means nothing.
TEST(DegradationConfig, DarknessAndDustLaddersAreMonotone) {
    double previous_radiance = 1e9;
    for (int dark = 0; dark <= 3; ++dark) {
        const NamedDegradation d =
            LoadDegradation(ConfigDir() + "dark" + std::to_string(dark) + "_dust0.yaml");
        EXPECT_LT(d.params.radiance_scale, previous_radiance)
            << "dark" << dark << " is not darker than dark" << (dark - 1);
        previous_radiance = d.params.radiance_scale;
    }

    double previous_beta = -1.0;
    for (int dust = 0; dust <= 3; ++dust) {
        const NamedDegradation d =
            LoadDegradation(ConfigDir() + "dark0_dust" + std::to_string(dust) + ".yaml");
        EXPECT_GT(d.params.dust_beta, previous_beta)
            << "dust" << dust << " is not thicker than dust" << (dust - 1);
        previous_beta = d.params.dust_beta;
    }
}

// The sensor must be identical across darkness levels: only scene radiance
// changes. If read noise fell with the light, low light would not get harder,
// and the whole DVE premise would be wrong.
TEST(DegradationConfig, SensorParametersAreConstantAcrossDarknessLevels) {
    const NamedDegradation base = LoadDegradation(ConfigDir() + "dark0_dust0.yaml");
    for (int dark = 1; dark <= 3; ++dark) {
        const NamedDegradation d =
            LoadDegradation(ConfigDir() + "dark" + std::to_string(dark) + "_dust0.yaml");
        EXPECT_EQ(d.params.read_noise_electrons, base.params.read_noise_electrons);
        EXPECT_EQ(d.params.photon_scale, base.params.photon_scale);
        EXPECT_EQ(d.params.vignette_strength, base.params.vignette_strength);
        EXPECT_EQ(d.params.prnu_sigma, base.params.prnu_sigma);
        EXPECT_LT(d.params.radiance_scale, base.params.radiance_scale);
    }
}

TEST(DegradationConfig, Dust0DisablesDustEverywhere) {
    for (int dark = 0; dark <= 3; ++dark) {
        const NamedDegradation d =
            LoadDegradation(ConfigDir() + "dark" + std::to_string(dark) + "_dust0.yaml");
        EXPECT_FALSE(d.params.enable_dust);
    }
}

// The configs must actually drive the operator, not merely parse. This is the
// end-to-end check that a named config produces a correspondingly degraded
// image.
TEST(DegradationConfig, LoadedConfigsProduceMonotonicallyDarkerImages) {
    ImageU8 input(48, 48, 1);
    std::fill(input.data.begin(), input.data.end(), static_cast<uint8_t>(200));

    double previous_mean = 1e9;
    for (int dark = 0; dark <= 3; ++dark) {
        const NamedDegradation d =
            LoadDegradation(ConfigDir() + "dark" + std::to_string(dark) + "_dust0.yaml");
        const ImageU8 out = Degrade(input, d.params, 11ull, 0ull);
        double sum = 0.0;
        for (uint8_t v : out.data) sum += v;
        const double mean = sum / static_cast<double>(out.data.size());
        EXPECT_LT(mean, previous_mean) << "dark" << dark << " was not darker";
        previous_mean = mean;
    }
}
