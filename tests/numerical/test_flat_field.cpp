// Flat-field calibration (2.2, flat_field.hpp): the K measured from averaged D frames of a flat target
// agrees with D's analytic vignette x PRNU, and dividing by it restores one noise model for the frame.
#include "perception/radiometric/flat_field.hpp"

#include "perception/radiometric/noise_estimate.hpp"
#include <gtest/gtest.h>

#include <cmath>

using namespace ariadne::core;
using namespace ariadne::perception;
using namespace ariadne::perception::radiometric;

namespace {

/// The shared sensor of configs/degradation/*.yaml.
DegradationParams Sensor() {
    DegradationParams p;
    p.photon_scale = 10000.0;
    p.read_noise_electrons = 2.5;
    p.vignette_strength = 0.28;
    p.prnu_sigma = 0.012;
    return p;
}

double RmsRelative(const FlatField& a, const FlatField& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.k.size(); ++i) s += std::pow((a.k[i] - b.k[i]) / b.k[i], 2);
    return std::sqrt(s / a.k.size());
}

}  // namespace

// Acceptance (RADIOMETRIC_2.2.md §6): <= 0.5% RMS against the analytic K, RGB and mono. The PRNU alone
// is 1.2%, so a calibration that missed the fixed pattern (e.g. drew PRNU per frame) would fail this.
TEST(FlatField, CalibrationMatchesTheAnalyticResponse) {
    for (int channels : {3, 1}) {
        const FlatField cal = CalibrateFlatField(Sensor(), 4242, 320, 180, channels);
        const FlatField ana = AnalyticFlatField(Sensor(), 4242, 320, 180, channels);
        EXPECT_LT(RmsRelative(cal, ana), 0.005) << channels << " channels";
        EXPECT_NEAR(ana.KMin(), 0.72, 0.03);   // the 0.28 corner vignette
    }
}

// Negative control: another camera's seed has a different PRNU field, and the calibration notices.
TEST(FlatField, CalibrationIsSpecificToTheCamerasSilicon) {
    const FlatField cal = CalibrateFlatField(Sensor(), 4242, 320, 180, 3);
    const FlatField other = AnalyticFlatField(Sensor(), 4243, 320, 180, 3);
    EXPECT_GT(RmsRelative(cal, other), 0.008);
}

// What flat-fielding is for: D scales the noise by K, so the raw corner and centre follow different
// var(mean) lines; after z / K they share one. Measured temporally (per-pixel over frames).
TEST(FlatField, DividingByKRestoresOneNoiseModel) {
    DegradationParams p = Sensor();
    p.radiance_scale = 0.5;
    p.analog_gain = 4.0;
    const int w = 160, h = 90, n = 300;
    const FlatField k = AnalyticFlatField(p, 9, w, h, 1);
    ImageU8 flat(w, h, 1);
    std::fill(flat.data.begin(), flat.data.end(), uint8_t(120));
    // var/mean in the centre and in a corner, raw and flat-fielded
    double s[2][2] = {}, s2[2][2] = {};
    const int px[2][2] = {{w / 2, h / 2}, {1, 1}};
    for (int f = 0; f < n; ++f) {
        const ImageU8 d = Degrade(flat, p, 9, static_cast<uint64_t>(f));
        for (int r = 0; r < 2; ++r) {
            const double raw = d.At(px[r][0], px[r][1]);
            const double ff = raw / k.At(px[r][0], px[r][1]);
            s[r][0] += raw, s2[r][0] += raw * raw;
            s[r][1] += ff, s2[r][1] += ff * ff;
        }
    }
    double ratio[2];   // corner (var/mean) over centre (var/mean), raw and flat-fielded
    for (int m = 0; m < 2; ++m) {
        double vm[2];
        for (int r = 0; r < 2; ++r) {
            const double mean = s[r][m] / n;
            vm[r] = (s2[r][m] / n - mean * mean) / mean;
        }
        ratio[m] = vm[1] / vm[0];
    }
    EXPECT_LT(ratio[0], 0.85);            // raw: the corner's var/mean is ~K = 0.72 of the centre's
    EXPECT_NEAR(ratio[1], 1.0, 0.15);     // flat-fielded: one model
}
