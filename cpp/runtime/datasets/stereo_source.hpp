#pragma once
// ---------------------------------------------------------------------------
// One stereo input interface for the 1.4/1.5 pipeline, two implementations:
//
//   EurocSource  the EuRoC ASL folders (D35)
//   SimSource    a simulator recording (D39/D40) + its scene directory: frames
//                from sensors/, ground truth and static TF from canonical/,
//                every frame passed through the read-time LENS (D41) before
//                it is converted to gray. Calibration comes from rig.lens via
//                the mission manifest, never from the recorded camera_info
//                (whose D is zero and whose cx, cy are half a pixel off,
//                SMOKE §4n). ROS-free: RosbagReader + ros_cdr.
//
// A source hands out RAW (unrectified, still distorted) 8-bit gray stereo pairs
// in time order plus the raw calibration, exactly what EuRoC provides; the
// rectifier then exercises the undistort path on both.
//
// The sensor stages of Degrade() (2.2, D66): opt-in per run (SimSource::
// SetDegradation with a named config), applied to the RGB frame after the lens
// and before the gray conversion -- the order a real camera has (optics, then
// the sensor). Each (agent, camera) gets its own D seed, so the two cameras do
// not carry the same noise and PRNU fields. The dust stage gets GT depth for
// both cameras (D142: the right one warped from the left depth camera, then
// both through the lens like their images). Without SetDegradation the
// manifest records degradation_config = "none" and every pixel is as before.
// ---------------------------------------------------------------------------
#include "core/inertial/imu_noise_operator.hpp"
#include "core/random/seed.hpp"
#include "perception/degradation/degradation_config.hpp"
#include "perception/degradation/lens.hpp"
#include "perception/radiometric/flat_field.hpp"
#include "perception/rgbd/depth_stereo.hpp"
#include "perception/stereo/camera_model.hpp"
#include "runtime/datasets/sim_depth.hpp"
#include "runtime/datasets/euroc.hpp"
#include "runtime/manifest/run_manifest.hpp"
#include "runtime/metrics/trajectory_io.hpp"
#include "runtime/replay/ros_cdr.hpp"
#include "runtime/replay/rosbag_reader.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ariadne::runtime {

/// §4.1 `photo` (5.2, D146): the photometric state the frame was taken with, for Stage 4 (divide out exposure x
/// gain) and Stage 3 (the lamp power). Set only when the pixels are LINEAR in radiance: in sim, when D decodes the
/// sRGB render (input_transfer: srgb, D130). Absent means "not known to be linear": Stage 3 refuses such frames.
struct FramePhoto {
    double exposure_gain = 1.0;   // exposure x analog gain x digital gain (D's relative values)
    double lamp_scale = 1.0;      // what scales the lamp's radiance after the render: D radiance_scale x lamp_power
};

/// The carried lamp as the recording rendered it, and whether it was the only light (5.2, D146).
struct LampRig {
    std::string rig;               // the rig name, matched against the lamp calibration's
    double intensity = 0.0;        // the render's lamp intensity (gz <intensity>)
    core::Vec3 in_body_m = core::Vec3::Zero();
    bool lamp_only = false;        // scene ambient 0 and no world lights: every photon came from the lamp
};

/// §4.1 Frame, the subset Phase 1 produces. Not modeled yet, and left absent rather than defaulted:
///   imu_window    (§10; every Phase 1 run is imu_mode none)
///   illumination  (§6.5 lamp state beyond `photo`, Phase 5)
///   gt            ground truth never rides on the estimator's frame; sources expose it separately
///                 (GroundTruth()) for scoring only.
struct StereoFrame {
    uint32_t agent = 0;    // the source's agent (EuRoC: 0)
    uint64_t id = 0;       // 0-based index in this source's time-ordered stream
    int64_t t_ns = 0;
    cv::Mat left, right;   // raw 8-bit gray
    cv::Mat left_rgb;      // raw 8-bit RGB of the left camera (same pixels as `left`); empty if the source has none
    /// RGB-D sources only (2.6, D100): CV_32F metres registered to `left`'s pixels, 0 = no return. `right` is then
    /// empty. Empty for stereo sources.
    cv::Mat depth;
    std::optional<FramePhoto> photo;   // 5.2, D146: set only when the pixels are linear in radiance
};

class StereoSource {
public:
    virtual ~StereoSource() = default;
    virtual std::string RunName() const = 0;      // run directory stem, e.g. MH_01_easy
    virtual std::string SceneName() const = 0;    // manifest scene, e.g. EuRoC/MH_01_easy
    virtual std::string CameraConfig() const = 0;
    virtual std::string BagChecksum() const = 0;  // "sha256:..." or UNRECORDED
    virtual perception::RawStereoCalibration Calibration() const = 0;
    virtual core::SE3 T_B_C0() const = 0;         // body <- raw left camera
    virtual std::vector<TimedPose> GroundTruth() const = 0;   // T_W_B, time-sorted
    virtual size_t FrameCount() const = 0;
    virtual uint32_t Agent() const = 0;           // stamped on every StereoFrame
    /// Calls on_frame for the first max_frames pairs (all if 0), in time order.
    virtual size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) = 0;
    /// The body (IMU) stream, time-sorted, and its noise model (§10.2; plan 2.4). Empty when the source has no IMU.
    virtual std::vector<core::ImuMeasurement> Imu() const { return {}; }
    virtual core::ImuNoise ImuNoiseModel() const { return {}; }
    /// Ground-truth velocity and biases, time-sorted, for VALIDATING the inertial initialisation only (never fed to
    /// the estimator). Empty when the source has none.
    virtual std::vector<GroundTruthInertial> GroundTruthInertialStates() const { return {}; }
    /// The named degradation config applied to the frames (§5.6), "none" if none (2.2).
    virtual std::string DegradationConfig() const { return "none"; }
    /// Camera `cam`'s (0 left, 1 right) calibrated flat field for the radiometric stage; empty (K = 1) when the
    /// source has none -- EuRoC publishes none (2.2).
    virtual perception::radiometric::FlatField FlatFieldOf(int /*cam*/) const { return {}; }
    /// RGB-D sources (2.6, D100): the depth sensor's stated noise model and the virtual rig it is fed through.
    /// Present iff every frame carries `depth` and no `right`. Calibration() is then that virtual rig.
    virtual std::optional<perception::DepthSensorModel> DepthSensor() const { return std::nullopt; }
    /// The named IMU noise operator config applied to Imu() (L2, D101), "none" if none.
    virtual std::string ImuNoiseConfig() const { return "none"; }
    /// The carried lamp (5.2, D146); absent when the source has none or does not record it (EuRoC, Hilti).
    virtual std::optional<LampRig> Lamp() const { return std::nullopt; }
};

// --- EuRoC ---------------------------------------------------------------------

class EurocSource : public StereoSource {
public:
    explicit EurocSource(const std::string& sequence) : seq_(sequence), mav_(seq_ / "mav0") {
        cam0_ = LoadEurocCamera((mav_ / "cam0/sensor.yaml").string());
        cam1_ = LoadEurocCamera((mav_ / "cam1/sensor.yaml").string());
        pairs_ = PairByTimestamp(ParseEurocImageCsv(ReadTextFile((mav_ / "cam0/data.csv").string()), "cam0"),
                                 ParseEurocImageCsv(ReadTextFile((mav_ / "cam1/data.csv").string()), "cam1"));
        // Optional: an EuRoC-layout export of a dataset with only sparse GT (Hilti 2023, D141) has no
        // state_groundtruth_estimate0; the source then reports no GT and the runner scores nothing densely.
        if (std::filesystem::exists(GtCsv())) {
            for (const auto& g : ParseEurocGroundTruth(ReadTextFile(GtCsv()), "groundtruth")) {
                gt_.push_back({g.t_ns, g.T_W_B});
            }
        }
    }
    std::string RunName() const override { return seq_.filename().string(); }
    std::string SceneName() const override { return "EuRoC/" + seq_.filename().string(); }
    std::vector<core::ImuMeasurement> Imu() const override {
        return ParseEurocImu(ReadTextFile((mav_ / "imu0/data.csv").string()), "imu0");
    }
    core::ImuNoise ImuNoiseModel() const override {
        return ParseEurocImuNoise(ReadTextFile((mav_ / "imu0/sensor.yaml").string()), "imu0/sensor.yaml");
    }
    std::vector<GroundTruthInertial> GroundTruthInertialStates() const override {
        if (!std::filesystem::exists(GtCsv())) return {};
        return ParseEurocGroundTruthInertial(ReadTextFile(GtCsv()), "groundtruth");
    }
    std::string CameraConfig() const override { return "euroc_cam0_cam1"; }
    std::string BagChecksum() const override {
        const auto prov = seq_.parent_path() / (seq_.filename().string() + ".provenance.json");
        return std::filesystem::exists(prov) ? "sha256:" + YAML::LoadFile(prov.string())["sha256"].as<std::string>()
                                             : std::string(kUnknownField);
    }
    perception::RawStereoCalibration Calibration() const override { return EurocStereo(cam0_, cam1_); }
    core::SE3 T_B_C0() const override { return cam0_.T_B_C; }
    std::vector<TimedPose> GroundTruth() const override { return gt_; }
    size_t FrameCount() const override { return pairs_.t_ns.size(); }
    uint32_t Agent() const override { return 0; }
    size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) override {
        const size_t n = max_frames ? std::min(max_frames, FrameCount()) : FrameCount();
        for (size_t i = 0; i < n; ++i) {
            StereoFrame f;
            f.agent = 0;
            f.id = i;
            f.t_ns = pairs_.t_ns[i];
            f.left = cv::imread((mav_ / "cam0/data" / pairs_.left[i]).string(), cv::IMREAD_GRAYSCALE);
            f.right = cv::imread((mav_ / "cam1/data" / pairs_.right[i]).string(), cv::IMREAD_GRAYSCALE);
            if (f.left.empty() || f.right.empty()) throw std::runtime_error("failed to read frame " + std::to_string(i));
            on_frame(f);
        }
        return n;
    }

private:
    std::string GtCsv() const { return (mav_ / "state_groundtruth_estimate0/data.csv").string(); }
    std::filesystem::path seq_, mav_;
    EurocCamera cam0_, cam1_;
    StereoPairs pairs_;
    std::vector<TimedPose> gt_;
};

// --- Simulator recording ---------------------------------------------------------

class SimSource : public StereoSource {
public:
    SimSource(const std::string& rec, const std::string& scene, int agent, const std::string& mission = "default")
        : rec_(rec), agent_(agent) {
        const YAML::Node sc = YAML::LoadFile(scene + "/scene.json");
        const YAML::Node m = YAML::LoadFile(scene + "/mission_" + mission + ".json");
        scene_id_ = sc["scene_id"].as<std::string>();
        rig_ = m["rig"]["name"].as<std::string>();
        if (const YAML::Node lamp = m["rig"]["lamp"]) {   // 5.2 (D146): the lamp as rendered, and its scene's light
            LampRig l;
            l.rig = rig_;
            l.intensity = lamp["intensity"].as<double>();
            const auto b = lamp["in_body_m"].as<std::vector<double>>();
            l.in_body_m = core::Vec3(b.at(0), b.at(1), b.at(2));
            const YAML::Node lighting = sc["preset"]["values"]["lighting"];
            l.lamp_only = lighting && lighting["ambient"].as<double>() == 0.0 && sc["lights"] && sc["lights"].size() == 0;
            lamp_ = l;
        }
        YAML::Node ag;
        for (const auto& x : m["agents"]) {
            if (x["agent"].as<int>() == agent) ag.reset(x);
        }
        if (!ag) throw std::runtime_error("sim: agent " + std::to_string(agent) + " not in the mission");
        frame_ = ag["frame_id"].as<std::string>();
        t_left_ = ag["topics"]["left_image"].as<std::string>();
        t_right_ = ag["topics"]["right_image"].as<std::string>();
        if (m["rig"]["depth"]["enabled"].as<int>()) t_depth_ = ag["topics"]["depth_image"].as<std::string>();
        const std::string t_pose = ag["topics"]["gt_pose"].as<std::string>();
        t_imu_ = ag["topics"]["imu"].as<std::string>();
        if (!m["lens"]) throw std::runtime_error("sim: mission has no rig.lens (pre-D41 recording)");

        // The lens: the camera the estimator sees (D41).
        const auto L = m["lens"];
        const auto R = m["render_intrinsics_pixel_centre"];
        lens_.width = m["rig"]["camera"]["width"].as<int>();
        lens_.height = m["rig"]["camera"]["height"].as<int>();
        lens_.fx = L["fx"].as<double>();
        lens_.fy = L["fy"].as<double>();
        lens_.cx = L["cx"].as<double>();
        lens_.cy = L["cy"].as<double>();
        lens_.k1 = L["k1"].as<double>();
        lens_.k2 = L["k2"].as<double>();
        lens_.p1 = L["p1"].as<double>();
        lens_.p2 = L["p2"].as<double>();
        lens_.k3 = L["k3"].as<double>();
        lens_.render_fx = R["fx"].as<double>();
        lens_.render_fy = R["fy"].as<double>();
        lens_.render_cx = R["cx"].as<double>();
        lens_.render_cy = R["cy"].as<double>();
        if (lens_.k3 != 0.0) throw std::runtime_error("sim: PinholeRadtan carries k1 k2 p1 p2 only; rig.lens.k3 != 0");
        map_ = perception::BuildLensMap(lens_);

        // Canonical: GT body poses and the static camera extrinsics.
        RosbagReader bag;
        bag.Open(rec + "/canonical/canonical_0.mcap");
        bag.Replay([&](const RosMessage& msg) {
            if (msg.topic == t_pose) {
                const auto p = ros_cdr::DecodePoseStamped(msg.data, msg.size);
                TimedPose tp{p.header.stamp.ns(), {}};
                tp.T.R = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
                             .normalized()
                             .toRotationMatrix();
                tp.T.t = core::Vec3(p.position.x, p.position.y, p.position.z);
                gt_.push_back(tp);
            } else {
                for (const auto& t : ros_cdr::DecodeTfMessage(msg.data, msg.size)) {
                    if (t.header.frame_id != frame_) continue;
                    core::SE3 T;
                    T.R = Eigen::Quaterniond(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z)
                              .normalized()
                              .toRotationMatrix();
                    T.t = core::Vec3(t.translation.x, t.translation.y, t.translation.z);
                    const std::string base = frame_.substr(0, frame_.size() - 9);   // "agent_k/"
                    if (t.child_frame_id == base + "left_optical") T_B_left_ = T, ++found_;
                    if (t.child_frame_id == base + "right_optical") T_B_right_ = T, ++found_;
                }
            }
        }, {t_pose, "/tf_static"});
        if (gt_.empty() || found_ != 2) throw std::runtime_error("sim: canonical bag lacks GT pose or static TF");

        const YAML::Node recj = YAML::LoadFile(rec + "/recording.json");
        for (const auto& f : recj["streams"]["sensors"]["files"]) {
            const std::string name = f.first.as<std::string>();
            if (name.size() > 5 && name.substr(name.size() - 5) == ".mcap") {
                sensors_mcap_ = rec + "/sensors/" + name;
                checksum_ = "sha256:" + f.second["sha256"].as<std::string>();
            }
        }
        if (sensors_mcap_.empty()) throw std::runtime_error("sim: recording.json lists no sensors .mcap");
        frames_ = 0;
        for (const auto& t : recj["result"]["topics"]) {
            if (t.first.as<std::string>() == t_left_) frames_ = t.second["count"].as<size_t>();
        }
    }

    std::string RunName() const override { return scene_id_ + "-agent" + std::to_string(agent_); }
    std::string SceneName() const override { return "sim/" + scene_id_ + "/agent" + std::to_string(agent_); }
    std::string CameraConfig() const override { return rig_ + "+lens(D41)"; }
    std::string BagChecksum() const override { return checksum_; }
    perception::RawStereoCalibration Calibration() const override {
        perception::PinholeRadtan c;
        c.width = lens_.width;
        c.height = lens_.height;
        c.fu = lens_.fx;
        c.fv = lens_.fy;
        c.cu = lens_.cx;
        c.cv = lens_.cy;
        c.model = perception::DistortionModel::kRadtan;   // the D41 lens is Brown-Conrady
        c.dist = {lens_.k1, lens_.k2, lens_.p1, lens_.p2};
        perception::RawStereoCalibration s;
        s.left = s.right = c;   // identical lenses (rig_stereo_v1)
        s.T_right_left = T_B_right_.inverse() * T_B_left_;
        return s;
    }
    core::SE3 T_B_C0() const override { return T_B_left_; }
    std::vector<TimedPose> GroundTruth() const override { return gt_; }
    size_t FrameCount() const override { return frames_; }
    uint32_t Agent() const override { return uint32_t(agent_); }
    std::string DegradationConfig() const override { return degradation_ ? degradation_name_ : "none"; }
    perception::radiometric::FlatField FlatFieldOf(int cam) const override { return flat_[cam == 0 ? 0 : 1]; }
    /// The IMU only once SetImuNoise was called (L2, D101): without it the source exposes no IMU, so default sim
    /// runs (imu_mode none, the Phase-1 gate) see exactly what they saw before.
    std::vector<core::ImuMeasurement> Imu() const override { return imu_; }
    core::ImuNoise ImuNoiseModel() const override { return imu_noise_ ? imu_noise_->noise : core::ImuNoise{}; }
    std::string ImuNoiseConfig() const override { return imu_noise_ ? imu_noise_->name : "none"; }
    std::optional<LampRig> Lamp() const override { return lamp_; }

    /// Read the canonical (clean) IMU stream and apply the seeded noise operator `p` (L2, D101;
    /// core/inertial/imu_noise_operator.hpp). The seed derives from the run seed and the agent, as D's do. A config
    /// with all densities 0 exposes the clean stream. ImuNoiseModel() then reports exactly the injected model.
    void SetImuNoise(const core::ImuNoiseOperatorParams& p, uint64_t run_seed) {
        std::vector<core::ImuMeasurement> clean;
        RosbagReader bag;
        bag.Open(rec_ + "/canonical/canonical_0.mcap");
        bag.Replay([&](const RosMessage& msg) {
            const auto m = ros_cdr::DecodeImu(msg.data, msg.size);
            core::ImuMeasurement s;
            s.t_ns = m.header.stamp.ns();
            s.gyro = core::Vec3(m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z);
            s.accel = core::Vec3(m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z);
            clean.push_back(s);
        }, {t_imu_});
        if (clean.empty()) throw std::runtime_error("sim: canonical bag has no IMU on " + t_imu_);
        std::stable_sort(clean.begin(), clean.end(),
                         [](const core::ImuMeasurement& a, const core::ImuMeasurement& b) { return a.t_ns < b.t_ns; });
        const uint64_t seed = core::DeriveSeed(core::DeriveSeed(run_seed, core::seed_purpose::kImuNoise),
                                               ("agent" + std::to_string(agent_)).c_str());
        imu_ = core::ApplyImuNoise(clean, p, seed);
        imu_noise_ = p;
    }

    /// Apply D (named config `d`) to every frame from now on (2.2, D66). Camera seeds derive from the run seed,
    /// the agent and the side. Also calibrates each camera's flat field from D frames of a flat target under
    /// the same sensor and seed (radiometric/flat_field.hpp), as a lab would before the mission.
    void SetDegradation(const perception::NamedDegradation& d, uint64_t run_seed, int threads) {
        if (d.params.enable_dust && t_depth_.empty()) {
            throw std::runtime_error("sim: degradation " + d.name + " enables dust, but the mission records no GT "
                                     "depth (rig.depth.enabled 0); refused rather than run without it");
        }
        degradation_ = d.params;
        degradation_name_ = d.name;
        threads_ = threads;
        const uint64_t root = core::DeriveSeed(run_seed, core::seed_purpose::kDegradation);
        const std::string agent = "agent" + std::to_string(agent_);
        seed_[0] = core::DeriveSeed(root, (agent + "/left").c_str());
        seed_[1] = core::DeriveSeed(root, (agent + "/right").c_str());
        for (int c = 0; c < 2; ++c) {
            flat_[c] = perception::radiometric::CalibrateFlatField(d.params, seed_[c], lens_.width, lens_.height, 3,
                                                                   16, 200, threads_);
        }
    }

    size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) override {
        struct Stop {};
        // Stamps not yet complete: the clean images (and GT depth when D has dust, D142) wait here until every part
        // of the pair has arrived; the pair is then processed in one go. D's frame id is the pair index, exactly as
        // when each side was degraded on arrival (2.2), so dust-free runs are byte-identical to before.
        struct Raw {
            core::ImageU8 img[2];
            core::ImageF32 depth;
            bool have[3] = {false, false, false};
        };
        const bool dust = degradation_ && degradation_->enable_dust;
        std::map<int64_t, Raw> pending;
        int64_t last_emitted = -1;
        size_t n = 0;
        RosbagReader bag;
        bag.Open(sensors_mcap_);
        std::set<std::string> topics{t_left_, t_right_};
        if (dust) topics.insert(t_depth_);
        try {
            bag.Replay([&](const RosMessage& msg) {
                const auto im = ros_cdr::DecodeImage(msg.data, msg.size);
                ros_cdr::ValidateImage(im);
                const int64_t t = im.header.stamp.ns();
                Raw& r = pending[t];
                if (msg.topic == t_depth_) {
                    if (im.encoding != "32FC1") throw std::runtime_error("sim: expected 32FC1 depth, got " + im.encoding);
                    r.depth = core::ImageF32(int(im.width), int(im.height));
                    for (uint32_t y = 0; y < im.height; ++y) {
                        std::memcpy(&r.depth.data[size_t(y) * im.width], &im.data[size_t(y) * im.step], im.width * 4);
                    }
                    r.have[2] = true;
                } else {
                    if (im.encoding != "rgb8") throw std::runtime_error("sim: expected rgb8, got " + im.encoding);
                    const int c = msg.topic == t_left_ ? 0 : 1;
                    r.img[c] = core::ImageU8(int(im.width), int(im.height), 3);
                    for (uint32_t y = 0; y < im.height; ++y) {
                        std::memcpy(&r.img[c].data[size_t(y) * im.width * 3], &im.data[size_t(y) * im.step],
                                    im.width * 3);
                    }
                    r.have[c] = true;
                }
                if (!r.have[0] || !r.have[1] || (dust && !r.have[2])) return;
                // Complete pair: it must be the oldest pending stamp, or a frame went missing.
                if (pending.begin()->first != t || t <= last_emitted) {
                    throw std::runtime_error("sim: stereo pair at " + std::to_string(t) +
                                             " completed before an older stamp: a frame is missing");
                }
                last_emitted = t;
                StereoFrame done;
                done.t_ns = t;
                core::ImageF32 depth_seen[2];
                if (dust) {
                    // Each camera's depth through the same lens as its image; the right camera's by warping the
                    // left depth on the render grid first (runtime/datasets/sim_depth.hpp).
                    const double baseline = (T_B_left_.t - T_B_right_.t).norm();
                    depth_seen[0] = perception::ApplyLensDepth(r.depth, map_);
                    depth_seen[1] = perception::ApplyLensDepth(WarpDepthToRight(r.depth, lens_.render_fx, baseline), map_);
                }
                for (int c = 0; c < 2; ++c) {
                    core::ImageU8 seen = perception::ApplyLens(r.img[c], map_);
                    if (degradation_) {
                        seen = perception::DegradeParallel(seen, depth_seen[c], *degradation_, seed_[c], n, threads_);
                    }
                    const cv::Mat rgb(seen.height, seen.width, CV_8UC3, seen.data.data());
                    cv::Mat gray;
                    cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
                    (c == 0 ? done.left : done.right) = gray;
                    if (c == 0) done.left_rgb = rgb.clone();
                }
                pending.erase(t);
                // Linear pixels only with D's sRGB decode (D130): then the photometric state is D's (D146).
                if (degradation_ && degradation_->input_transfer == perception::InputTransfer::kSrgb) {
                    const perception::DegradationParams& d = *degradation_;
                    done.photo = FramePhoto{d.exposure * d.analog_gain * d.digital_gain, d.radiance_scale * d.lamp_power};
                }
                done.agent = uint32_t(agent_);
                done.id = n;
                on_frame(done);
                if (++n == max_frames) throw Stop{};
            }, topics);
        } catch (const Stop&) {
        }
        if (max_frames == 0 && !pending.empty()) throw std::runtime_error("sim: unpaired frames at the end");
        return n;
    }

private:
    std::string rec_, scene_id_, rig_, frame_, t_left_, t_right_, t_depth_, t_imu_, sensors_mcap_,
        checksum_ = kUnknownField;
    int agent_ = 0;
    std::optional<LampRig> lamp_;
    size_t frames_ = 0;
    int found_ = 0;
    perception::LensModel lens_;
    perception::LensMap map_;
    core::SE3 T_B_left_, T_B_right_;
    std::vector<TimedPose> gt_;
    std::optional<perception::DegradationParams> degradation_;   // 2.2: D, when SetDegradation was called
    std::string degradation_name_;
    uint64_t seed_[2] = {0, 0};
    int threads_ = 1;
    perception::radiometric::FlatField flat_[2];
    std::vector<core::ImuMeasurement> imu_;                  // L2: the noisy stream, when SetImuNoise was called
    std::optional<core::ImuNoiseOperatorParams> imu_noise_;
};

}  // namespace ariadne::runtime
