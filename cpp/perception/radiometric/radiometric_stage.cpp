#include "perception/radiometric/radiometric_stage.hpp"

#include "perception/radiometric/noise_estimate.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace ariadne::perception::radiometric {

namespace {

core::ImageU8 ToImageU8(const cv::Mat& m) {
    core::ImageU8 im(m.cols, m.rows, 1);
    for (int y = 0; y < m.rows; ++y) std::memcpy(&im.data[static_cast<size_t>(y) * m.cols], m.ptr<uint8_t>(y), m.cols);
    return im;
}

core::ImageF32 ToImageF32(const cv::Mat& m) {
    core::ImageF32 im(m.cols, m.rows);
    for (int y = 0; y < m.rows; ++y) std::memcpy(&im.data[static_cast<size_t>(y) * m.cols], m.ptr<float>(y), m.cols * 4);
    return im;
}

}  // namespace

RadiometricStage::RadiometricStage(const RadiometricParams& params, FlatField left, FlatField right) : p_(params) {
    ValidateRadiometric(p_);
    report_.mode = p_.mode;
    const FlatField* ff[2] = {&left, &right};
    for (int c = 0; c < 2; ++c) {
        report_.flat_field[c] = ff[c]->source;
        report_.k_min[c] = ff[c]->KMin();
        cam_[c].k_min = std::max(ff[c]->KMin(), kMinRelativeResponse);
        if (ff[c]->Empty()) continue;
        // §6.2: the reciprocal is precomputed and stored fp16; the multiply runs on the fp16-rounded values.
        cv::Mat k(ff[c]->height, ff[c]->width, CV_32F, const_cast<float*>(ff[c]->k.data())), inv, inv16;
        cv::divide(1.0, k, inv);
        inv.convertTo(inv16, CV_16F);
        inv16.convertTo(cam_[c].inv_k, CV_32F);
    }
    if (p_.mode == "flat_only") report_.decision = "flat_only", decided_ = true;
    if (!p_.Enabled()) decided_ = true;
}

cv::Mat RadiometricStage::FlatFielded(const cv::Mat& raw, int cam) const {
    cv::Mat z;
    raw.convertTo(z, CV_32F);
    if (!cam_[cam].inv_k.empty()) {
        if (cam_[cam].inv_k.size() != raw.size()) throw std::runtime_error("radiometric: flat field size != frame size");
        cv::multiply(z, cam_[cam].inv_k, z);
    }
    return z;
}

void RadiometricStage::Decide(const std::vector<cv::Mat>& left, const std::vector<cv::Mat>& right) {
    if (!p_.Vst()) {
        decided_ = true;
        return;
    }
    const std::vector<cv::Mat>* frames[2] = {&left, &right};
    std::ostringstream why;
    bool all_pass = true;
    for (int c = 0; c < 2; ++c) {
        std::vector<core::ImageU8> raw;
        std::vector<core::ImageF32> z;
        for (const cv::Mat& m : *frames[c]) {
            raw.push_back(ToImageU8(m));
            z.push_back(ToImageF32(FlatFielded(m, c)));
        }
        std::vector<const core::ImageU8*> rp;
        std::vector<const core::ImageF32*> zp;
        for (size_t i = 0; i < raw.size(); ++i) rp.push_back(&raw[i]), zp.push_back(&z[i]);
        const NoiseFit fit = FitNoise(zp, rp, p_.fit);
        CameraNoiseReport& r = report_.noise[c];
        r.fit_valid = fit.valid;
        r.why = fit.why;
        r.a = fit.a;
        r.b = fit.b;
        r.bins = fit.bins.size();
        r.median_level = fit.median_level;
        r.snr_at_median = fit.snr_at_median;
        const char* side = c == 0 ? "left" : "right";
        if (!fit.valid) {
            all_pass = false;
            why << side << ": no valid noise fit (" << fit.why << "); ";
            continue;
        }
        cam_[c].model = FromAffineFit(fit.a, fit.b);
        r.sigma_z = StabilisedSigma(zp, rp, cam_[c].model, p_.fit);
        const bool stabilises = r.sigma_z >= p_.gate_sigma_z_min && r.sigma_z <= p_.gate_sigma_z_max;
        const bool noisy = r.snr_at_median < p_.gate_snr_max;
        r.passes_gate = stabilises && noisy;
        if (!stabilises) why << side << ": sigma_z " << r.sigma_z << " outside [" << p_.gate_sigma_z_min << ", " << p_.gate_sigma_z_max << "]; ";
        if (!noisy) why << side << ": SNR " << r.snr_at_median << " at the median level >= " << p_.gate_snr_max << "; ";
        all_pass = all_pass && r.passes_gate;
    }
    if (p_.mode == "on") {
        for (int c = 0; c < 2; ++c) {
            if (!report_.noise[c].fit_valid) {
                throw std::runtime_error("radiometric: mode on, but camera " + std::to_string(c) +
                                         " has no valid noise fit: " + report_.noise[c].why);
            }
        }
        vst_active_ = true;
        report_.decision = "vst_on";
        report_.reason = all_pass ? "forced on (the gate would also pass)" : "forced on; the gate would refuse: " + why.str();
    } else {
        vst_active_ = all_pass;
        report_.decision = all_pass ? "vst_on" : "vst_off";
        report_.reason = all_pass ? "both cameras: valid fit, sigma_z in range, SNR below the threshold" : why.str();
    }
    if (vst_active_) {
        for (int c = 0; c < 2; ++c) BuildInverse(cam_[c]);
    }
    decided_ = true;
}

void RadiometricStage::BuildInverse(Camera& c) const {
    // The brightest flat-fielded value a frame can hold, and its GAT; the LUT spans D in [0, d_max].
    c.f_max = GatForward(255.0 / c.k_min, c.model);
    c.d_max = c.f_max + 2.0;
    const double s = ReducedSigma(c.model);
    std::unique_ptr<ExactInverse> exact;
    if (p_.inverse == "exact") exact = std::make_unique<ExactInverse>(s);
    const int n = 4096;
    c.lut.resize(n + 1);
    for (int i = 0; i <= n; ++i) {
        const double D = c.d_max * i / n;
        double y = p_.inverse == "exact" ? (*exact)(D) : p_.inverse == "closed_form" ? ClosedFormInverse(D, s)
                                                                                     : AlgebraicInverse(D, s);
        c.lut[i] = static_cast<float>(y);
    }
}

cv::Mat RadiometricStage::Apply(const cv::Mat& raw, int cam) {
    if (!p_.Enabled()) return raw;
    cv::Mat out;
    ApplyFloat(raw, cam).convertTo(out, CV_8U);   // rounds and saturates (a boosted corner may clip at 255)
    return out;
}

cv::Mat RadiometricStage::ApplyFloat(const cv::Mat& raw, int cam) {
    if (!p_.Enabled()) {
        cv::Mat z;
        raw.convertTo(z, CV_32F);
        return z;
    }
    if (!decided_) throw std::logic_error("radiometric: Apply before Decide");
    if (raw.type() != CV_8UC1) throw std::invalid_argument("radiometric: expects 8-bit gray");
    const auto t0 = std::chrono::steady_clock::now();
    const cv::Mat z = FlatFielded(raw, cam);
    cv::Mat out;
    if (!vst_active_) {
        out = z;
    } else {
        const Camera& c = cam_[cam];
        const PoissonGaussian& m = c.model;
        // [2a] f = (2/alpha) sqrt(max(0, alpha z + 3/8 alpha^2 + sigma^2 - alpha mu))   (eq. 4.3)
        cv::Mat f;
        z.convertTo(f, CV_32F, m.alpha, 0.375 * m.alpha * m.alpha + m.sigma * m.sigma - m.alpha * m.mu);
        cv::max(f, 0.0, f);
        cv::sqrt(f, f);
        f *= 2.0 / m.alpha;
        // [2b] denoise at noise std 1
        cv::Mat d;
        if (p_.denoiser == "nlm") {
            // fastNlMeansDenoising takes 8U (L2) or 16U (L1) only (SMOKE 2.2 preflight): map [0, f_max] to [0, 250].
            const double scale = 250.0 / c.f_max;
            cv::Mat f8, d8;
            f.convertTo(f8, CV_8U, scale);
            cv::fastNlMeansDenoising(f8, d8, static_cast<float>(p_.nlm_h * scale), p_.nlm_template_px, p_.nlm_search_px);
            d8.convertTo(d, CV_32F, 1.0 / scale);
        } else if (p_.denoiser == "gaussian") {
            cv::GaussianBlur(f, d, cv::Size(5, 5), p_.gaussian_sigma_px);
        } else {
            d = f;
        }
        // [2c] inverse D -> y (reduced) by the LUT, then back to DN: alpha y + mu
        cv::Mat y(d.size(), CV_32F);
        const float step = static_cast<float>(c.d_max / (c.lut.size() - 1));
        const int last = static_cast<int>(c.lut.size()) - 1;
        for (int r = 0; r < d.rows; ++r) {
            const float* dp = d.ptr<float>(r);
            float* yp = y.ptr<float>(r);
            for (int x = 0; x < d.cols; ++x) {
                const float u = std::max(0.0f, dp[x]) / step;
                const int i = std::min(static_cast<int>(u), last - 1);
                const float w = std::min(u - static_cast<float>(i), 1.0f);
                yp[x] = static_cast<float>(m.alpha) * (c.lut[i] + w * (c.lut[i + 1] - c.lut[i])) + static_cast<float>(m.mu);
            }
        }
        out = y;
    }
    report_.ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++report_.frames;
    return out;
}

}  // namespace ariadne::perception::radiometric
