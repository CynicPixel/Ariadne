// The radiometric stage end to end on D-degraded frames (2.2, radiometric_stage.hpp): pass-through when
// off, flat-field equalisation, the measured-noise gate's three outcomes, and the reason the exact
// unbiased inverse exists -- on a dim flat patch it keeps the mean where the algebraic inverse darkens it.
#include "perception/radiometric/radiometric_stage.hpp"

#include "perception/degradation/degradation.hpp"
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>

using namespace ariadne::core;
using namespace ariadne::perception;
using namespace ariadne::perception::radiometric;

namespace {

constexpr int kW = 320, kH = 240;

/// Left 60%: a ramp with step edges (what the noise fit needs: many levels, some structure).
/// Right 40%: a flat patch at `patch`.
ImageU8 Scene(uint8_t patch) {
    ImageU8 im(kW, kH, 1);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            int v = patch;
            if (x < kW * 6 / 10) {
                v = (x * 255) / (kW * 6 / 10);
                if ((x / 32) % 2) v = v * 3 / 4;
            }
            im.At(x, y) = static_cast<uint8_t>(v);
        }
    return im;
}

DegradationParams Sensor(double radiance, double gain, double vignette = 0.0) {
    DegradationParams p;
    p.radiance_scale = radiance;
    p.photon_scale = 10000.0;
    p.read_noise_electrons = 2.5;
    p.analog_gain = gain;
    p.vignette_strength = vignette;
    p.prnu_sigma = vignette > 0.0 ? 0.012 : 0.0;
    return p;
}

cv::Mat Mat(const ImageU8& im) { return cv::Mat(im.height, im.width, CV_8U, const_cast<uint8_t*>(im.data.data())).clone(); }

std::vector<cv::Mat> Frames(const DegradationParams& p, uint8_t patch, uint64_t seed, int n, int first = 0) {
    std::vector<cv::Mat> out;
    for (int i = 0; i < n; ++i) out.push_back(Mat(Degrade(Scene(patch), p, seed, static_cast<uint64_t>(first + i))));
    return out;
}

RadiometricParams Mode(const std::string& mode, const std::string& inverse = "exact", const std::string& denoiser = "nlm") {
    RadiometricParams r;
    r.mode = mode;
    r.inverse = inverse;
    r.denoiser = denoiser;
    r.gate_frames = 2;
    r.fit.stride = 1;   // these frames are 320 x 240; a 1280 x 720 frame at stride 2 has 3x the samples
    return r;
}

double PatchMean(const cv::Mat& m) { return cv::mean(m(cv::Rect(kW * 7 / 10, 8, kW * 3 / 10 - 8, kH - 16)))[0]; }

}  // namespace

TEST(RadiometricStage, OffIsAPassThrough) {
    RadiometricStage st(Mode("off"), {}, {});
    const cv::Mat in = Frames(Sensor(0.06, 8), 40, 1, 1)[0];
    const cv::Mat out = st.Apply(in, 0);
    EXPECT_EQ(out.data, in.data);   // the same buffer, not merely equal bytes
    EXPECT_EQ(st.Report().decision, "off");
}

// Stage 1 alone: with the camera's calibrated K the vignetted corner comes back to the centre's level.
TEST(RadiometricStage, FlatFieldEqualisesTheVignette) {
    const DegradationParams p = Sensor(1.0, 1.0, 0.28);
    ImageU8 flat(kW, kH, 1);
    std::fill(flat.data.begin(), flat.data.end(), uint8_t(150));
    const FlatField k = CalibrateFlatField(p, 5, kW, kH, 1);
    RadiometricStage st(Mode("flat_only"), k, k);
    const cv::Mat raw = Mat(Degrade(flat, p, 5, 0)), out = st.Apply(raw, 0);
    const auto corner = [](const cv::Mat& m) { return cv::mean(m(cv::Rect(0, 0, 16, 16)))[0]; };
    const auto centre = [](const cv::Mat& m) { return cv::mean(m(cv::Rect(kW / 2 - 8, kH / 2 - 8, 16, 16)))[0]; };
    EXPECT_LT(corner(raw) / centre(raw), 0.8);
    EXPECT_NEAR(corner(out) / centre(out), 1.0, 0.02);
}

// The gate's three outcomes. Gain-up darkness: on. Bright (dark0): off, by SNR. Unity-gain dark2 (posterised
// by the 8-bit output, preflight NLF probe): off, by the fit itself.
TEST(RadiometricStage, GateFollowsTheMeasuredNoise) {
    {
        RadiometricStage st(Mode("auto"), {}, {});
        st.Decide(Frames(Sensor(0.06, 8), 40, 1, 2), Frames(Sensor(0.06, 8), 40, 2, 2));
        EXPECT_EQ(st.Report().decision, "vst_on") << st.Report().reason;
        EXPECT_NEAR(st.Report().noise[0].sigma_z, 1.0, 0.1);
    }
    {
        RadiometricStage st(Mode("auto"), {}, {});
        st.Decide(Frames(Sensor(1.0, 1), 40, 1, 2), Frames(Sensor(1.0, 1), 40, 2, 2));
        EXPECT_EQ(st.Report().decision, "vst_off");
        EXPECT_NE(st.Report().reason.find("SNR"), std::string::npos) << st.Report().reason;
    }
    {
        RadiometricStage st(Mode("auto"), {}, {});
        st.Decide(Frames(Sensor(0.06, 1), 40, 1, 2), Frames(Sensor(0.06, 1), 40, 2, 2));
        EXPECT_EQ(st.Report().decision, "vst_off");
        EXPECT_NE(st.Report().reason.find("no valid noise fit"), std::string::npos) << st.Report().reason;
    }
}

TEST(RadiometricStage, ForcedOnRefusesWithoutANoiseModel) {
    RadiometricStage st(Mode("on"), {}, {});
    EXPECT_THROW(st.Decide(Frames(Sensor(0.06, 1), 40, 1, 2), Frames(Sensor(0.06, 1), 40, 2, 2)), std::runtime_error);
}

// Why the exact unbiased inverse (eq. 4.9): at ~15 photo-electrons per pixel and alpha ~ 0.8 DN/e-, the
// denoised dim patch keeps its mean with the exact inverse, while the algebraic inverse darkens it by about
// alpha/4 DN (thesis eq. 4.38 vs the algebraic form). Also: the denoiser does reduce the noise.
TEST(RadiometricStage, ExactInverseKeepsTheDimMeanAndTheAlgebraicDoesNot) {
    const DegradationParams p = Sensor(0.03, 32);   // alpha = 0.816 DN/e-; full scale 300 e- (245 DN); patch 13 -> 15.3 e- -> ~12.5 DN
    const auto gate_l = Frames(p, 13, 1, 2), gate_r = Frames(p, 13, 2, 2);
    const auto test = Frames(p, 13, 1, 6, 100);
    double raw_mean = 0, raw_sd = 0;
    for (const cv::Mat& m : test) raw_mean += PatchMean(m) / test.size();
    {
        cv::Scalar mu, sd;
        cv::meanStdDev(test[0](cv::Rect(kW * 7 / 10, 8, kW * 3 / 10 - 8, kH - 16)), mu, sd);
        raw_sd = sd[0];
    }
    double means[2] = {0, 0}, out_sd = 0;
    const char* inverses[2] = {"exact", "algebraic"};
    for (int k = 0; k < 2; ++k) {
        RadiometricStage st(Mode("on", inverses[k]), {}, {});
        st.Decide(gate_l, gate_r);
        for (const cv::Mat& m : test) {
            const cv::Mat o = st.ApplyFloat(m, 0);   // before the 8-bit rounding (see ApplyFloat)
            means[k] += PatchMean(o) / test.size();
            if (k == 0 && &m == &test[0]) {
                cv::Scalar mu, sd;
                cv::meanStdDev(o(cv::Rect(kW * 7 / 10, 8, kW * 3 / 10 - 8, kH - 16)), mu, sd);
                out_sd = sd[0];
            }
        }
    }
    const double alpha = 32 * 255.0 / 10000.0;
    EXPECT_NEAR(means[0], raw_mean, 0.25 * alpha * 0.4) << "exact " << means[0] << " raw " << raw_mean;
    EXPECT_LT(means[1], raw_mean - 0.25 * alpha * 0.6) << "algebraic " << means[1] << " raw " << raw_mean;
    EXPECT_LT(out_sd, 0.6 * raw_sd) << "denoised std " << out_sd << " raw " << raw_sd;
}
