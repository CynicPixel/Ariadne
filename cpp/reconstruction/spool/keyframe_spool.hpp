#pragma once
// ---------------------------------------------------------------------------
// The C++ -> Python reconstruction boundary (arch §18.2 "KeyframeBundle records
// over gRPC or spooled files"; design reconstruction.md#first-reconstruction-loop). One directory per run, written
// here, read only by python/reconstruction:
//
//   spool.yaml                      schema_version 1, run ids, merge record, index
//   agent{k}/kf_{frame}_rgb.png     rectified left RGB (8-bit, stored BGR as PNG does)
//   agent{k}/kf_{frame}_depth.png   depth Z, metres, fp16 bit patterns in a 16-bit PNG
//   agent{k}/kf_{frame}_sigma.png   SGBM sigma_Z, same encoding; NaN = no depth
//   agent{k}/ho_{frame}_rgb.png     held-out frame RGB (render target; no depth)
//   agent{k}/kf_{frame}_albedo.png  schema 2 (6.1, D195): the §19.1 albedo proxy (CV_32F reflectance in calibration
//                                   units, fp16 bits like depth; NaN where it is not defined)
//
// Schema 2 is written only when the run gives a lamp (SetLamp), an albedo or a photometric state; otherwise the
// output is schema 1, byte for byte (sim_collab's spools and their checksums are unchanged). Schema 2 adds, per
// bundle, rgb_kind albedo_proxy and the APPLIED photometric state (exposure x gain, lamp power, decision id), and at
// the top the lamp: its calibration file (copied in as lamp_calibration.yaml), render intensity, and R_rect_raw (rectified <- raw left camera), so
// the service can apply the calibrated light model at a held-out camera's lamp pose (§19.1).
//
// fp16-in-PNG: Z is converted to IEEE half, and the 16 raw bits are stored as a
// lossless 16-bit PNG: half the bytes of float32, NaN preserved, compressed.
// Every §18.2 field is written, explicit when not modeled (exposure, lamp, mask),
// and poses are the ones each keyframe had at insertion with the graph version
// current then (the same pose set the 1.6 merge used). GT never enters the spool.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::reconstruction {

struct SpoolParams {
    double scale = 0.5;          // image downscale (0.5: 1280x720 -> 640x360), exact factor
    int keyframe_stride = 1;     // keep every n-th keyframe per agent
    int heldout_stride = 200;    // a held-out frame every n frames (never a keyframe)
};

struct SpoolCamera {
    double fx = 0, fy = 0, cx = 0, cy = 0;   // rectified intrinsics at spool scale
    int width = 0, height = 0;
};

/// Schema 2: the photometric state a frame was ACTUALLY taken with (runtime::FramePhoto, D147).
struct SpoolPhoto {
    double exposure_gain = 1.0;   // exposure_rel x analog x digital gain
    double lamp_rel = 1.0;        // D's lamp power (x the render intensity relative to the calibration, below)
    uint64_t decision_id = 0;     // the controller decision applied (0 = none)
};

/// Schema 2: the lamp the albedo proxy was normalised with (KeyframeFalloff, D146).
struct SpoolLamp {
    std::string calib_source, calib_name, rig;   // the calibration file is COPIED into the spool (files alone)
    double intensity = 0.0;        // the render's lamp intensity; / the calibration's intensity_ref = its power
    cv::Matx33d R_rect_raw = cv::Matx33d::eye();   // rectified-left <- raw-left (the calibration's camera frame)
};

class KeyframeSpool {
public:
    KeyframeSpool(std::filesystem::path dir, SpoolParams p) : dir_(std::move(dir)), p_(p) {
        if (!(p_.scale > 0 && p_.scale <= 1)) throw std::invalid_argument("spool: scale must be in (0, 1]");
        if (p_.keyframe_stride < 1 || p_.heldout_stride < 1) throw std::invalid_argument("spool: strides >= 1");
        std::filesystem::create_directories(dir_);
    }

    /// Rectified intrinsics at spool scale. Pixel-centre convention: u' + 0.5 = s (u + 0.5).
    SpoolCamera Camera(double fx, double fy, double cx, double cy, int w, int h) const {
        const double s = p_.scale;
        return {fx * s, fy * s, s * (cx + 0.5) - 0.5, s * (cy + 0.5) - 0.5, int(std::lround(w * s)),
                int(std::lround(h * s))};
    }

    /// One keyframe (call for every keyframe in order; the stride is applied here).
    void AddKeyframe(uint32_t agent, uint64_t frame_id, int64_t t_ns, uint64_t graph_version, const core::SE3& T_O_C,
                     const SpoolCamera& cam, double baseline_m, const cv::Mat& rgb, const cv::Mat& depth_z,
                     const cv::Mat& sigma_z, const cv::Mat& albedo = cv::Mat(), const cv::Mat& albedo_valid = cv::Mat(),
                     const std::optional<SpoolPhoto>& photo = std::nullopt) {
        if (kf_seen_[agent % 2]++ % uint64_t(p_.keyframe_stride) != 0) return;
        if (rgb.empty() || rgb.type() != CV_8UC3) throw std::invalid_argument("spool: keyframe needs rectified RGB");
        if (depth_z.type() != CV_32F || sigma_z.type() != CV_32F) throw std::invalid_argument("spool: depth must be CV_32F");
        const std::string stem = Stem(agent, "kf", frame_id);
        WriteRgb(stem + "_rgb.png", rgb);
        WriteHalf(stem + "_depth.png", depth_z);
        WriteHalf(stem + "_sigma.png", sigma_z);
        Entry e{"keyframe", agent, frame_id, t_ns, graph_version, T_O_C, cam, baseline_m, stem};
        if (!albedo.empty()) {
            if (albedo.type() != CV_32F || albedo.size() != rgb.size() || albedo_valid.type() != CV_8U ||
                albedo_valid.size() != rgb.size()) {
                throw std::invalid_argument("spool: albedo must be CV_32F with a CV_8U valid mask, on the RGB grid");
            }
            cv::Mat a = albedo.clone();
            a.setTo(std::numeric_limits<float>::quiet_NaN(), albedo_valid == 0);
            WriteHalf(stem + "_albedo.png", a);
            e.albedo = true;
        }
        e.photo = photo;
        entries_.push_back(e);
    }

    /// One held-out frame (RGB + estimated pose only; a render target).
    void AddHeldout(uint32_t agent, uint64_t frame_id, int64_t t_ns, const core::SE3& T_O_C, const SpoolCamera& cam,
                    const cv::Mat& rgb, const std::optional<SpoolPhoto>& photo = std::nullopt) {
        if (rgb.empty() || rgb.type() != CV_8UC3) throw std::invalid_argument("spool: held-out frame needs RGB");
        const std::string stem = Stem(agent, "ho", frame_id);
        WriteRgb(stem + "_rgb.png", rgb);
        Entry e{"heldout", agent, frame_id, t_ns, 0, T_O_C, cam, 0.0, stem};
        e.photo = photo;
        entries_.push_back(e);
    }

    /// Schema 2: the lamp of the run (call once, before Finish); copies the calibration file into the spool.
    void SetLamp(const SpoolLamp& l) {
        std::filesystem::copy_file(l.calib_source, dir_ / kLampCalibFile,
                                   std::filesystem::copy_options::overwrite_existing);
        lamp_ = l;
    }
    static constexpr const char* kLampCalibFile = "lamp_calibration.yaml";

    bool WantHeldout(uint64_t frame_id) const {
        return frame_id % uint64_t(p_.heldout_stride) == uint64_t(p_.heldout_stride / 2);
    }

    /// Write spool.yaml. `merge` is the 1.6 record: T_O0_O1 maps agent 1's odometry frame into agent 0's.
    void Finish(const std::string& run_id, const std::string& scene_id, bool merge_ok, const core::SE3& T_O0_O1,
                const std::string& merge_edge) const {
        YAML::Emitter y;
        y.SetDoublePrecision(17);
        bool v2 = lamp_.has_value();
        for (const auto& e : entries_) v2 = v2 || e.albedo || e.photo.has_value();
        y << YAML::BeginMap;
        y << YAML::Key << "schema_version" << YAML::Value << (v2 ? 2 : 1);
        y << YAML::Key << "run_id" << YAML::Value << run_id;
        y << YAML::Key << "scene_id" << YAML::Value << scene_id;
        y << YAML::Key << "scale" << YAML::Value << p_.scale;
        y << YAML::Key << "keyframe_stride" << YAML::Value << p_.keyframe_stride;
        y << YAML::Key << "heldout_stride" << YAML::Value << p_.heldout_stride;
        y << YAML::Key << "image_encoding" << YAML::Value
          << "rgb: 8-bit PNG (BGR channel order on disk); depth/sigma: IEEE fp16 bits in 16-bit PNG, metres, NaN = none";
        y << YAML::Key << "frames" << YAML::Value << "T_O_C: rectified-left camera in agent k's odometry frame O_k";
        if (lamp_) {
            y << YAML::Key << "lamp" << YAML::Value << YAML::BeginMap;
            y << YAML::Key << "calibration" << YAML::Value << kLampCalibFile;
            y << YAML::Key << "calibration_source" << YAML::Value << lamp_->calib_source;
            y << YAML::Key << "calibration_name" << YAML::Value << lamp_->calib_name;
            y << YAML::Key << "rig" << YAML::Value << lamp_->rig;
            y << YAML::Key << "intensity" << YAML::Value << lamp_->intensity;
            std::vector<double> r;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) r.push_back(lamp_->R_rect_raw(i, j));
            y << YAML::Key << "R_rect_raw" << YAML::Value << YAML::Flow << r;
            y << YAML::EndMap;
        }
        y << YAML::Key << "merge" << YAML::Value << YAML::BeginMap;
        y << YAML::Key << "ok" << YAML::Value << merge_ok;
        y << YAML::Key << "T_O0_O1" << YAML::Value << Pose(T_O0_O1);
        y << YAML::Key << "edge" << YAML::Value << merge_edge;
        y << YAML::EndMap;
        y << YAML::Key << "bundles" << YAML::Value << YAML::BeginSeq;
        for (const auto& e : entries_) {
            y << YAML::Flow << YAML::BeginMap;
            y << YAML::Key << "role" << YAML::Value << e.role;
            y << YAML::Key << "agent" << YAML::Value << e.agent;
            y << YAML::Key << "kf_id" << YAML::Value << e.frame_id;
            y << YAML::Key << "t_ns" << YAML::Value << e.t_ns;
            y << YAML::Key << "frame_id_of_pose" << YAML::Value << e.graph_version;
            y << YAML::Key << "T_WiC" << YAML::Value << Pose(e.T);
            y << YAML::Key << "K" << YAML::Value << YAML::Flow << YAML::BeginSeq << e.cam.fx << e.cam.fy << e.cam.cx
              << e.cam.cy << YAML::EndSeq;
            y << YAML::Key << "width" << YAML::Value << e.cam.width;
            y << YAML::Key << "height" << YAML::Value << e.cam.height;
            y << YAML::Key << "baseline_m" << YAML::Value << e.baseline_m;
            y << YAML::Key << "files" << YAML::Value << e.stem;
            // rgb_kind names the TRAINING image: raw_radiance (the RGB file) or albedo_proxy (the _albedo file).
            y << YAML::Key << "rgb_kind" << YAML::Value << (e.albedo ? "albedo_proxy" : "raw_radiance");
            if (e.photo) {
                y << YAML::Key << "exposure_state" << YAML::Value << YAML::Flow << YAML::BeginMap << YAML::Key
                  << "exposure_gain" << YAML::Value << e.photo->exposure_gain << YAML::Key << "decision_id"
                  << YAML::Value << e.photo->decision_id << YAML::EndMap;
                y << YAML::Key << "lamp_state" << YAML::Value << YAML::Flow << YAML::BeginMap << YAML::Key
                  << "lamp_rel" << YAML::Value << e.photo->lamp_rel << YAML::EndMap;
            } else {
                y << YAML::Key << "exposure_state" << YAML::Value << "not_modeled";
                y << YAML::Key << "lamp_state" << YAML::Value << "inert";
            }
            y << YAML::Key << "mask" << YAML::Value << "none";
            y << YAML::EndMap;
        }
        y << YAML::EndSeq << YAML::EndMap;
        if (!y.good()) throw std::runtime_error("spool: yaml emitter: " + y.GetLastError());
        std::ofstream(dir_ / "spool.yaml") << "# Ariadne reconstruction spool (design reconstruction.md#first-reconstruction-loop). Generated.\n"
                                           << y.c_str() << "\n";
    }

    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        std::string role;
        uint32_t agent;
        uint64_t frame_id;
        int64_t t_ns;
        uint64_t graph_version;
        core::SE3 T;
        SpoolCamera cam;
        double baseline_m;
        std::string stem;
        bool albedo = false;
        std::optional<SpoolPhoto> photo;
    };

    static std::vector<double> Pose(const core::SE3& T) {   // row-major 3x4 [R | t]
        std::vector<double> v;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) v.push_back(T.R(i, j));
            v.push_back(T.t(i));
        }
        return v;
    }
    std::string Stem(uint32_t agent, const char* kind, uint64_t frame_id) const {
        const std::string sub = "agent" + std::to_string(agent);
        std::filesystem::create_directories(dir_ / sub);
        return sub + "/" + kind + "_" + std::to_string(frame_id);
    }
    cv::Mat Down(const cv::Mat& m, int interp) const {
        if (p_.scale == 1.0) return m;
        cv::Mat o;
        cv::resize(m, o, cv::Size(), p_.scale, p_.scale, interp);
        return o;
    }
    void WriteRgb(const std::string& rel, const cv::Mat& rgb) const {
        cv::Mat bgr;
        cv::cvtColor(Down(rgb, cv::INTER_AREA), bgr, cv::COLOR_RGB2BGR);
        if (!cv::imwrite((dir_ / rel).string(), bgr)) throw std::runtime_error("spool: cannot write " + rel);
    }
    // Nearest-neighbour downscale: averaging would blend depth across edges and NaN holes.
    // [ASSUMPTION] at scale 0.5 it samples the top-left source pixel of each 2x2 block, 0.5 source px from the
    // block centre the RGB average represents: ~1.5 mm lateral at 2 m (f 680), below SGBM noise.
    void WriteHalf(const std::string& rel, const cv::Mat& f32) const {
        cv::Mat h;
        Down(f32, cv::INTER_NEAREST).convertTo(h, CV_16F);
        const cv::Mat bits(h.size(), CV_16U, h.data, h.step);
        if (!cv::imwrite((dir_ / rel).string(), bits)) throw std::runtime_error("spool: cannot write " + rel);
    }

    std::filesystem::path dir_;
    SpoolParams p_;
    uint64_t kf_seen_[2] = {0, 0};
    std::vector<Entry> entries_;
    std::optional<SpoolLamp> lamp_;
};

}  // namespace ariadne::reconstruction
