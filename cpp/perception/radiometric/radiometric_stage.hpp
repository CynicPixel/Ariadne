#pragma once
// ---------------------------------------------------------------------------
// The radiometric front end as one per-camera stage (arch §6.2 stages 1-2 and
// 4, §6.4; 2.2). Raw 8-bit gray in, raw 8-bit gray out, BEFORE rectification:
// the rectifier's interpolation mixes neighbouring pixels' noise (the GAT's
// per-pixel model no longer holds after it), and K(x) lives in raw pixel
// coordinates.
//
//   [1] flat field     z = raw * (1/K)                (1/K stored fp16, one cv::multiply)
//   [2] VST denoise    f = GAT(z) -> denoise at sigma 1 -> exact unbiased inverse
//                      (only when the measured-noise gate is on)
//   [3] falloff        Phase 5 (needs lamp calibration and depth)
//   [4] exposure norm  inert: no source carries photometric state (t, g) until §6.5
//
// The gate and the noise models are decided once per run from the first
// frames (Decide); Apply then runs per frame. Same-architecture deterministic
// (OpenCV float paths, like the pgo digest), not bit-portable.
// ---------------------------------------------------------------------------
#include "perception/radiometric/flat_field.hpp"
#include "perception/radiometric/gat.hpp"
#include "perception/radiometric/radiometric_config.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace ariadne::perception::radiometric {

struct CameraNoiseReport {
    bool fit_valid = false;
    std::string why;   // why the fit is invalid, empty otherwise
    double a = 0.0, b = 0.0, sigma_z = 0.0, snr_at_median = 0.0, median_level = 0.0;
    size_t bins = 0;
    bool passes_gate = false;
};

struct RadiometricReport {
    std::string mode = "off";
    std::string decision = "off";   // off | flat_only | vst_on | vst_off
    std::string reason;             // why the gate decided as it did
    std::array<std::string, 2> flat_field{"none", "none"};
    std::array<double, 2> k_min{1.0, 1.0};
    std::array<CameraNoiseReport, 2> noise;
    std::string exposure_norm = "inert: no source carries photometric state (t, g); the §6.5 controller is Phase 5";
    long frames = 0;
    double ms_total = 0.0;
    double MsPerImage() const { return frames ? ms_total / frames : 0.0; }
};

class RadiometricStage {
public:
    /// `left`, `right`: the cameras' flat fields (empty = K 1, e.g. EuRoC, which publishes none).
    RadiometricStage(const RadiometricParams& params, FlatField left, FlatField right);

    bool Enabled() const { return p_.Enabled(); }
    /// How many leading stereo pairs Decide() needs (0: nothing to decide).
    int GateFrames() const { return p_.Vst() ? p_.gate_frames : 0; }
    /// Fit each camera's noise on the leading frames (raw 8-bit gray) and decide the gate. Throws in mode
    /// `on` if a fit is invalid. Must run before Apply when GateFrames() > 0.
    void Decide(const std::vector<cv::Mat>& left, const std::vector<cv::Mat>& right);
    /// One camera's raw 8-bit gray frame through the stage (cam 0 left, 1 right). Mode off returns `raw` itself.
    cv::Mat Apply(const cv::Mat& raw, int cam);
    /// The same before the final rounding to 8 bits (CV_32F, DN). The 8-bit hand-off re-quantizes a smooth
    /// denoised output without dither, which hides sub-DN mean differences; tests of the inverse look here.
    cv::Mat ApplyFloat(const cv::Mat& raw, int cam);

    bool VstActive() const { return vst_active_; }
    const RadiometricParams& Params() const { return p_; }
    const RadiometricReport& Report() const { return report_; }

private:
    struct Camera {
        cv::Mat inv_k;           // CV_32F holding fp16-rounded 1/K; empty = K 1
        double k_min = 1.0;      // the darkest relative response (sets the brightest flat-fielded value)
        PoissonGaussian model;
        double f_max = 0.0;      // GAT of the brightest possible flat-fielded value
        std::vector<float> lut;  // inverse over D in [0, d_max], reduced units y
        double d_max = 0.0;
    };
    cv::Mat FlatFielded(const cv::Mat& raw, int cam) const;
    void BuildInverse(Camera& c) const;

    RadiometricParams p_;
    std::array<Camera, 2> cam_;
    bool decided_ = false, vst_active_ = false;
    RadiometricReport report_;
};

}  // namespace ariadne::perception::radiometric
