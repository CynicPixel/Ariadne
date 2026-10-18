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
// Not yet here (Phase 5): the sensor stages of Degrade() after the lens. They
// need GT depth (the dust stage) and a named degradation config; the manifest
// records degradation_config = "none" until then.
// ---------------------------------------------------------------------------
#include "perception/degradation/lens.hpp"
#include "perception/stereo/camera_model.hpp"
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
#include <filesystem>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ariadne::runtime {

/// §4.1 Frame, the subset Phase 1 produces. Not modeled yet, and left absent rather than defaulted:
///   imu_window    (§10; every Phase 1 run is imu_mode none)
///   photo, illumination  (§6.5 exposure / lamp state, Phase 5)
///   gt            ground truth never rides on the estimator's frame; sources expose it separately
///                 (GroundTruth()) for scoring only.
struct StereoFrame {
    uint32_t agent = 0;    // the source's agent (EuRoC: 0)
    uint64_t id = 0;       // 0-based index in this source's time-ordered stream
    int64_t t_ns = 0;
    cv::Mat left, right;   // raw 8-bit gray
    cv::Mat left_rgb;      // raw 8-bit RGB of the left camera (same pixels as `left`); empty if the source has none
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
};

// --- EuRoC ---------------------------------------------------------------------

class EurocSource : public StereoSource {
public:
    explicit EurocSource(const std::string& sequence) : seq_(sequence), mav_(seq_ / "mav0") {
        cam0_ = LoadEurocCamera((mav_ / "cam0/sensor.yaml").string());
        cam1_ = LoadEurocCamera((mav_ / "cam1/sensor.yaml").string());
        pairs_ = PairByTimestamp(ParseEurocImageCsv(ReadTextFile((mav_ / "cam0/data.csv").string()), "cam0"),
                                 ParseEurocImageCsv(ReadTextFile((mav_ / "cam1/data.csv").string()), "cam1"));
        for (const auto& g : ParseEurocGroundTruth(
                 ReadTextFile((mav_ / "state_groundtruth_estimate0/data.csv").string()), "groundtruth")) {
            gt_.push_back({g.t_ns, g.T_W_B});
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
        YAML::Node ag;
        for (const auto& x : m["agents"]) {
            if (x["agent"].as<int>() == agent) ag.reset(x);
        }
        if (!ag) throw std::runtime_error("sim: agent " + std::to_string(agent) + " not in the mission");
        frame_ = ag["frame_id"].as<std::string>();
        t_left_ = ag["topics"]["left_image"].as<std::string>();
        t_right_ = ag["topics"]["right_image"].as<std::string>();
        const std::string t_pose = ag["topics"]["gt_pose"].as<std::string>();
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

    size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) override {
        struct Stop {};
        std::map<int64_t, StereoFrame> pending;   // stamps with only one side so far
        int64_t last_emitted = -1;
        size_t n = 0;
        RosbagReader bag;
        bag.Open(sensors_mcap_);
        try {
            bag.Replay([&](const RosMessage& msg) {
                const bool left = msg.topic == t_left_;
                const auto im = ros_cdr::DecodeImage(msg.data, msg.size);
                ros_cdr::ValidateImage(im);
                if (im.encoding != "rgb8") throw std::runtime_error("sim: expected rgb8, got " + im.encoding);
                core::ImageU8 clean(int(im.width), int(im.height), 3);
                for (uint32_t r = 0; r < im.height; ++r) {
                    std::memcpy(&clean.data[size_t(r) * im.width * 3], &im.data[size_t(r) * im.step], im.width * 3);
                }
                core::ImageU8 seen = perception::ApplyLens(clean, map_);
                cv::Mat gray;
                cv::cvtColor(cv::Mat(int(im.height), int(im.width), CV_8UC3, seen.data.data()), gray, cv::COLOR_RGB2GRAY);
                const int64_t t = im.header.stamp.ns();
                StereoFrame& f = pending[t];
                f.t_ns = t;
                (left ? f.left : f.right) = gray;
                if (left) f.left_rgb = cv::Mat(int(im.height), int(im.width), CV_8UC3, seen.data.data()).clone();
                if (f.left.empty() || f.right.empty()) return;
                // Complete pair: it must be the oldest pending stamp, or a frame went missing.
                if (pending.begin()->first != t || t <= last_emitted) {
                    throw std::runtime_error("sim: stereo pair at " + std::to_string(t) +
                                             " completed before an older stamp: a frame is missing");
                }
                last_emitted = t;
                StereoFrame done = std::move(f);
                pending.erase(t);
                done.agent = uint32_t(agent_);
                done.id = n;
                on_frame(done);
                if (++n == max_frames) throw Stop{};
            }, {t_left_, t_right_});
        } catch (const Stop&) {
        }
        if (max_frames == 0 && !pending.empty()) throw std::runtime_error("sim: unpaired frames at the end");
        return n;
    }

private:
    std::string rec_, scene_id_, rig_, frame_, t_left_, t_right_, sensors_mcap_, checksum_ = kUnknownField;
    int agent_ = 0;
    size_t frames_ = 0;
    int found_ = 0;
    perception::LensModel lens_;
    perception::LensMap map_;
    core::SE3 T_B_left_, T_B_right_;
    std::vector<TimedPose> gt_;
};

}  // namespace ariadne::runtime
