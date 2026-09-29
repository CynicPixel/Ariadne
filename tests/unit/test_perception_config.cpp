// Perception parameters and the §9.4 minimum-range validator. OpenCV-free, so
// these run in the host suite as well as in the image.
#include "perception/perception_config.hpp"

#include <gtest/gtest.h>

#include <string>

using namespace ariadne::perception;

namespace {

RectifiedStereoRig Rig(double baseline) {
    RectifiedStereoRig r;
    r.width = 1280;
    r.height = 720;
    r.fx = r.fy = 640.0;   // HFOV 90 deg at 1280 px (§9.4)
    r.cx = 639.5;
    r.cy = 359.5;
    r.baseline_m = baseline;
    return r;
}

}  // namespace

TEST(PerceptionConfig, ShippedPlaceholderLoadsAndValidates) {
    const PerceptionConfig c =
        LoadPerception(std::string(ARIADNE_SOURCE_DIR) + "/configs/perception/placeholder.yaml");
    EXPECT_EQ(c.name, "placeholder");
    EXPECT_EQ(c.stereo.num_disparities, 256);
    EXPECT_NO_THROW(ValidateStereoAgainstRig(c.stereo, Rig(0.10)));
}

// §9.4's worked numbers: f=640, B=0.10 m, 256 disparities gives Z_min ~0.25 m.
// The searched maximum is 255, not 256 (the range is [min, min + num - 1]).
TEST(PerceptionConfig, MinObservableRangeMatchesSection94) {
    StereoParams s;
    s.num_disparities = 256;
    EXPECT_EQ(MaxSearchedDisparity(s), 255);
    EXPECT_NEAR(MinObservableRange(s, Rig(0.10)), 64.0 / 255.0, 1e-12);
}

// §9.4: the B = 20 cm ablation must raise num_disparities or be refused.
TEST(PerceptionConfig, WideBaselineWithoutMoreDisparitiesIsRefused) {
    StereoParams s;
    s.num_disparities = 256;
    s.required_min_range_m = 0.30;
    EXPECT_THROW(ValidateStereoAgainstRig(s, Rig(0.20)), ConfigError);   // Z_min ~0.50 m
    s.num_disparities = 448;                                            // Z_min ~0.29 m
    EXPECT_NO_THROW(ValidateStereoAgainstRig(s, Rig(0.20)));
}

TEST(PerceptionConfig, SgbmContractViolationsAreRefused) {
    auto with = [](auto mutate) {
        PerceptionConfig c;
        c.name = "t";
        mutate(c);
        return c;
    };
    EXPECT_NO_THROW(ValidatePerception(with([](PerceptionConfig&) {})));
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.stereo.num_disparities = 250; })),
                 ConfigError);
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.stereo.block_size = 4; })),
                 ConfigError);
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.stereo.p2_coefficient = 8.0; })),
                 ConfigError);
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.stereo.mode = "3way"; })),
                 ConfigError);
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.tracker.win_size = 20; })),
                 ConfigError);
    EXPECT_THROW(ValidatePerception(with([](PerceptionConfig& c) { c.features.max_keypoints = 10; })),
                 ConfigError);
}

// The strict loader: a misspelled key inside a section fails, it does not
// silently fall back to the default.
TEST(PerceptionConfig, MisspelledSectionKeyFails) {
    StrictConfig config = StrictConfig::FromString(R"(
name: t
features: {grid_cols: 8, grid_rows: 6, max_keypoints: 1200}
tracker: {target_tracks: 1200, fb_treshold_px: 0.5}
stereo: {num_disparities: 256, block_size: 5, required_min_range_m: 0.3}
)", "inline");
    EXPECT_THROW(ParsePerception(config), ConfigError);
}
