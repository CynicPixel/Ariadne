#pragma once
// ---------------------------------------------------------------------------
// Parameters of the radiometric front end (arch §6.2, §6.4; 2.2). OpenCV-free,
// parsed as the optional `radiometric:` section of a perception config
// (perception_config.hpp). Absent section = mode off = the stage does not
// exist and every pixel passes through untouched.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "perception/radiometric/noise_estimate.hpp"

#include <string>

namespace ariadne::perception::radiometric {

struct RadiometricParams {
    // off        the stage is absent (default; byte-identical to a build without it)
    // flat_only  stage 1 only: divide by the source's flat field (K = 1 if it has none)
    // auto       flat field, then the VST denoise when the measured-noise gate says so (§6.4)
    // on         flat field and the VST denoise regardless of the gate (an ablation row; refuses
    //            to run if the noise fit itself is invalid -- there is no model to stabilise with)
    std::string mode = "off";
    // gaussian is the default: 0.8 ms per 1280x720 image, within the demo's per-frame budget. nlm
    // (cv::fastNlMeansDenoising, arch §6.2's suggestion) costs ~360 ms per image on this path: an ablation row
    // only (D67). none = the VST round trip alone.
    std::string denoiser = "gaussian";   // gaussian | nlm | none
    std::string inverse = "exact";     // exact (eq. 4.9 table) | closed_form (eq. 4.37) | algebraic (biased; ablation)
    double nlm_h = 1.0;                // NLM filter strength in units of the stabilised noise std (= 1)
    int nlm_template_px = 5;           // NLM patch size (odd)
    int nlm_search_px = 11;            // NLM search window (odd)
    double gaussian_sigma_px = 1.0;    // the cheap ablation denoiser, 5x5 kernel
    // The gate (§6.4: "the decision is driven by measured noise statistics"), decided once per run from
    // the first gate_frames stereo pairs, jointly for both cameras. ON iff, for each camera, the noise
    // fit is valid, the stabilised std is in [sigma_z_min, sigma_z_max] (the fit actually stabilises),
    // and the SNR at the median level is below snr_max (there is noise worth removing).
    // gate_frames: 8 pairs = 16 images, ~2.4M sampled pixels at stride 2 -- far above the fit's needs
    // (the draft's 30 bought nothing but warm-up latency).
    // gate_snr_max: the design set 20 between dark0 and the gain-up configs using SINGLE-CHANNEL D output;
    // the sim path is RGB -> D per channel -> BT.601 gray, which keeps the mean but scales the variance by
    // sum(w^2) = 0.447, so every SNR is 1/sqrt(0.447) = 1.50x higher. The same boundary in gray terms is
    // 20 x 1.5 = 30. Corrected 2026-10-21 BEFORE any VO measurement (D67; test GrayPathScalesTheModel).
    int gate_frames = 8;
    double gate_sigma_z_min = 0.8;
    double gate_sigma_z_max = 1.25;
    double gate_snr_max = 30.0;
    NoiseFitParams fit;                // the estimator's internals; not exposed in configs

    bool Enabled() const { return mode != "off"; }
    bool Vst() const { return mode == "auto" || mode == "on"; }
};

inline void ValidateRadiometric(const RadiometricParams& r) {
    const auto fail = [](const std::string& m) { throw core::ConfigError("radiometric: " + m); };
    if (r.mode != "off" && r.mode != "flat_only" && r.mode != "auto" && r.mode != "on") fail("mode off|flat_only|auto|on");
    if (r.denoiser != "nlm" && r.denoiser != "gaussian" && r.denoiser != "none") fail("denoiser nlm|gaussian|none");
    if (r.inverse != "exact" && r.inverse != "closed_form" && r.inverse != "algebraic") {
        fail("inverse exact|closed_form|algebraic");
    }
    if (!(r.nlm_h > 0.0) || r.nlm_template_px < 3 || r.nlm_template_px % 2 == 0 || r.nlm_search_px < r.nlm_template_px ||
        r.nlm_search_px % 2 == 0) {
        fail("nlm_h > 0, odd nlm_template_px >= 3, odd nlm_search_px >= nlm_template_px");
    }
    if (!(r.gaussian_sigma_px > 0.0)) fail("gaussian_sigma_px > 0");
    if (r.gate_frames < 1 || !(r.gate_sigma_z_min > 0.0 && r.gate_sigma_z_min < 1.0 && r.gate_sigma_z_max > 1.0) ||
        !(r.gate_snr_max > 0.0)) {
        fail("gate_frames >= 1, 0 < gate_sigma_z_min < 1 < gate_sigma_z_max, gate_snr_max > 0");
    }
}

inline RadiometricParams ParseRadiometric(core::StrictConfig& s) {
    RadiometricParams r;
    r.mode = s.Require<std::string>("mode");
    r.denoiser = s.Optional<std::string>("denoiser", r.denoiser);
    r.inverse = s.Optional<std::string>("inverse", r.inverse);
    r.nlm_h = s.Optional<double>("nlm_h", r.nlm_h);
    r.nlm_template_px = s.Optional<int>("nlm_template_px", r.nlm_template_px);
    r.nlm_search_px = s.Optional<int>("nlm_search_px", r.nlm_search_px);
    r.gaussian_sigma_px = s.Optional<double>("gaussian_sigma_px", r.gaussian_sigma_px);
    r.gate_frames = s.Optional<int>("gate_frames", r.gate_frames);
    r.gate_sigma_z_min = s.Optional<double>("gate_sigma_z_min", r.gate_sigma_z_min);
    r.gate_sigma_z_max = s.Optional<double>("gate_sigma_z_max", r.gate_sigma_z_max);
    r.gate_snr_max = s.Optional<double>("gate_snr_max", r.gate_snr_max);
    s.RequireAllConsumed();
    ValidateRadiometric(r);
    return r;
}

}  // namespace ariadne::perception::radiometric
