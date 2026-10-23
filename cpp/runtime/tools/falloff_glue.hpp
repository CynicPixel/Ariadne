#pragma once
// ---------------------------------------------------------------------------
// 5.2 Stage 3 glue (arch §6.2 Stage 3, §19.1; D134, D140, D146): the albedo
// proxy of every keyframe, from that keyframe's SGBM depth. Owned by the
// runtime because it reads the source (lamp, photometric state), the rectifier
// (K, R1) and the keyframe's stereo. Kept out of stereo_vo_runner.hpp so that
// shared file only gains call sites.
//
//   in   the rectified left gray AFTER Stages 1-2 (CV_8U, linear in radiance
//        only when the frame carries `photo`), SGBM Z and its support mask,
//        the frame's photometric state
//   out  perception::radiometric::AlbedoProxy (into the keyframe bundle);
//        tracking is untouched, so the trajectory is the same as with it off
//
// Stage 4 happens here, just before the division: gray / 255 / (exposure x
// gain), so the proxy is at the calibration's reference exposure. The lamp
// power relative to the calibration is (render intensity / intensity_ref) x
// D's radiance scale x D's lamp_power.
//
// Refusals, each one a silently wrong division otherwise (D146): no lamp, or
// not the only light; a calibration of another rig or another lamp position;
// frames not known to be linear (no `photo`).
//
// Disabled (no `falloff:` section): nothing is called, nothing is written.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "perception/radiometric/falloff.hpp"
#include "perception/radiometric/falloff_config.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::tools {

class KeyframeFalloff {
public:
    /// The lamp position must agree with the calibration's to within this (the calibration's lamp is the rig's,
    /// sim truth, D133): a different position is a different rig.
    static constexpr double kLampPositionTolM = 1e-3;

    KeyframeFalloff(const perception::radiometric::FalloffStageParams& p, const std::string& calib_path,
                    const std::optional<runtime::LampRig>& lamp, const core::SE3& T_B_C0, const cv::Matx33d& K_rect,
                    const cv::Matx33d& R1)
        : p_(p), K_(K_rect), R1_(R1) {
        if (!p_.Enabled()) return;
        const auto refuse = [](const std::string& why) { throw std::runtime_error("falloff (Stage 3): " + why); };
        if (calib_path.empty()) refuse("mode " + p_.mode + " needs a lamp calibration (--lamp-calib)");
        calib_ = perception::radiometric::LoadLampCalibration(calib_path);
        calib_path_ = calib_path;
        if (!lamp) refuse("the source records no carried lamp");
        if (!lamp->lamp_only) refuse("the scene has other light (ambient > 0 or world lights): the proxy divides by the "
                                     "lamp alone");
        if (lamp->rig != calib_.rig) refuse("calibration " + calib_.name + " is of rig " + calib_.rig + ", the recording's "
                                            "is " + lamp->rig);
        if (calib_.render_transfer != "srgb") refuse("calibration render_transfer " + calib_.render_transfer +
                                                     ": only srgb (decoded, linear) is wired");
        const core::Vec3 in_cam = T_B_C0.inverse() * lamp->in_body_m;
        const cv::Vec3d& c = calib_.lamp_in_camera;
        const double d = (in_cam - core::Vec3(c[0], c[1], c[2])).norm();
        if (!(d <= kLampPositionTolM)) {
            refuse("the recording's lamp is " + std::to_string(d * 1e3) + " mm from calibration " + calib_.name + "'s");
        }
        intensity_rel_ = lamp->intensity / calib_.intensity_ref;
    }

    bool enabled() const { return p_.Enabled(); }
    bool detects_on_proxy() const { return p_.features == "proxy"; }

    /// The proxy as the feature engines' 8-bit input (D146): 255 x reflectance, clipped at 1. FalloffNormalize
    /// returns reflectance in the calibration's units, so no scale is chosen here. The fraction of feature-mask
    /// pixels clipped (reflectance > 1: specular, or a model/depth error) goes into results.
    cv::Mat DetectionImage(const perception::radiometric::AlbedoProxy& a) {
        cv::Mat det;
        a.albedo.convertTo(det, CV_8U, 255.0);   // saturate_cast: > 1 clips to 255, < 0 cannot occur
        const int in_mask = cv::countNonZero(a.feature_mask);
        const cv::Mat over = (a.albedo > 1.0f) & a.feature_mask;
        clipped_.push_back(in_mask > 0 ? double(cv::countNonZero(over)) / in_mask : 0.0);
        return det;
    }

    /// One keyframe. `gray`: rectified left, CV_8U. `Z`: SGBM depth (any float type, metres) on the same grid;
    /// `supported`: CV_8U, nonzero where the depth is usable (DenseSupportMask). Pixels without support get NaN, so
    /// they are invalid in the proxy rather than divided by an unsupported distance.
    perception::radiometric::AlbedoProxy Normalize(const cv::Mat& gray, const cv::Mat& Z, const cv::Mat& supported,
                                                   const std::optional<runtime::FramePhoto>& photo) {
        if (!photo) {
            throw std::runtime_error("falloff (Stage 3): frame without a photometric state, i.e. not known to be linear "
                                     "(sim needs D with input_transfer: srgb)");
        }
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat lin;
        gray.convertTo(lin, CV_32F, 1.0 / (255.0 * photo->ExposureGain()));   // Stage 4: to the reference exposure
        cv::Mat z;
        Z.convertTo(z, CV_32F);
        z.setTo(std::numeric_limits<float>::quiet_NaN(), supported == 0);
        const double lamp_rel = intensity_rel_ * photo->LampScale();
        perception::radiometric::AlbedoProxy a =
            perception::radiometric::FalloffNormalize(lin, z, K_, calib_, lamp_rel, p_.params, R1_);
        ms_.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        const double n = static_cast<double>(gray.total());
        valid_.push_back(cv::countNonZero(a.valid) / n);
        feature_.push_back(cv::countNonZero(a.feature_mask) / n);
        lamp_rel_ = lamp_rel;
        return a;
    }

    /// Manifest entry (runs with the stage on only).
    std::string Summary() const {
        return p_.mode + ": " + calib_.name + " (" + calib_path_ + "), incidence " + p_.params.incidence + ", normals " +
               p_.params.normals + " w" + std::to_string(p_.params.normal_window_px) + ", features " + p_.features;
    }

    /// results.yaml block; nothing when the stage is off.
    void WriteResults(std::ostream& out) const {
        if (!p_.Enabled()) return;
        const auto num = [](double v) { return std::isfinite(v) ? std::to_string(v) : std::string(".nan"); };
        out << "falloff: {mode: " << p_.mode << ", lamp_calib: " << calib_.name << ", keyframes: " << ms_.size()
            << ", lamp_rel: " << num(lamp_rel_) << ", valid_fraction_p50: " << num(Quantile(valid_, 0.5))
            << ", feature_mask_fraction_p50: " << num(Quantile(feature_, 0.5)) << ",\n"
            << "  ms_p50: " << num(Quantile(ms_, 0.5)) << ", ms_p95: " << num(Quantile(ms_, 0.95));
        if (detects_on_proxy()) out << ", features: proxy, clipped_fraction_p50: " << num(Quantile(clipped_, 0.5))
                                    << ", clipped_fraction_max: " << num(Quantile(clipped_, 1.0));
        out << "}\n";
    }

private:
    /// Nearest-rank quantile; NaN when empty (no keyframe normalised).
    static double Quantile(std::vector<double> v, double q) {
        if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::sort(v.begin(), v.end());
        const size_t k = static_cast<size_t>(std::ceil(q * double(v.size())));
        return v[std::clamp<size_t>(k, 1, v.size()) - 1];
    }

    perception::radiometric::FalloffStageParams p_;
    perception::radiometric::LampCalibration calib_;
    std::string calib_path_;
    cv::Matx33d K_, R1_;
    double intensity_rel_ = 1.0, lamp_rel_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> ms_, valid_, feature_, clipped_;
};

}  // namespace ariadne::tools
