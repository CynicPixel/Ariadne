#pragma once
// ---------------------------------------------------------------------------
// TUM RGB-D (Sturm et al., IROS 2012) as a StereoSource (2.6, D100).
//
// A sequence directory holds rgb/ and depth/ PNGs listed in rgb.txt / depth.txt ("timestamp filename", seconds
// since the epoch), and groundtruth.txt ("timestamp tx ty tz qx qy qz qw": the colour camera's optical centre in
// the mocap world, ~100 Hz). Facts below are from the dataset's file-format page, read 2026-10-21:
// - depth PNGs are 16-bit, metres = value / 5000, 0 = no return; already pre-scaled (fr1 x1.035, fr2 x1.031) and
//   pre-registered to the RGB image's (distorted) pixels;
// - RGB intrinsics with radtan k1 k2 p1 p2 k3 (table below). The page recommends the uncalibrated ROS default
//   (525, 525, 319.5, 239.5, no distortion) because undistorting registered depth "is not trivial". We default to
//   the calibrated set, as ORB-SLAM2's published TUM results do: the SAME undistortion map is applied to RGB
//   (bilinear) and depth (nearest), which is exact for depth registered to the distorted RGB pixels. The ROS
//   default is the `ros_default` option, the ablation.
// - Association: rgb <-> depth by nearest timestamp within 20 ms, one-to-one, greedily by |dt| (associate.py).
//
// The source undistorts itself (5 coefficients; our PinholeRadtan holds 4) and reports a distortion-free virtual
// stereo rig (right camera = left, translated by the DepthSensorModel's virtual baseline), so the runner's
// rectifier is the identity up to resampling. Timestamps are parsed from the decimal string, never via double.
// ---------------------------------------------------------------------------
#include "runtime/datasets/euroc.hpp"
#include "runtime/datasets/stereo_source.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ariadne::runtime {

/// "1305031102.175304" -> ns, exactly (no floating point). Up to 9 fractional digits.
inline int64_t ParseTumTimestampNs(const std::string& s) {
    const size_t dot = s.find('.');
    const std::string whole = s.substr(0, dot);
    std::string frac = dot == std::string::npos ? "" : s.substr(dot + 1);
    if (whole.empty() || frac.size() > 9 || whole.find_first_not_of("0123456789") != std::string::npos ||
        frac.find_first_not_of("0123456789") != std::string::npos) {
        throw ConfigError("TUM: bad timestamp '" + s + "'");
    }
    frac.resize(9, '0');
    return std::stoll(whole) * 1000000000LL + std::stoll(frac);
}

struct TumStamped {
    int64_t t_ns = 0;
    std::string file;
};

/// rgb.txt / depth.txt: "timestamp filename" lines, '#' comments.
inline std::vector<TumStamped> ParseTumList(const std::string& text, const std::string& what) {
    std::vector<TumStamped> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string t, f;
        if (!(ls >> t >> f)) throw ConfigError("TUM " + what + ": malformed line '" + line + "'");
        out.push_back({ParseTumTimestampNs(t), f});
    }
    return out;
}

/// groundtruth.txt -> T_W_C (colour camera), time-sorted.
inline std::vector<TimedPose> ParseTumGroundTruth(const std::string& text) {
    std::vector<TimedPose> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string t;
        std::array<double, 7> v{};
        if (!(ls >> t >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5] >> v[6])) {
            throw ConfigError("TUM groundtruth: malformed line '" + line + "'");
        }
        TimedPose p;
        p.t_ns = ParseTumTimestampNs(t);
        p.T.t = core::Vec3(v[0], v[1], v[2]);
        p.T.R = Eigen::Quaterniond(v[6], v[3], v[4], v[5]).normalized().toRotationMatrix();   // file: qx qy qz qw
        out.push_back(p);
    }
    std::sort(out.begin(), out.end(), [](const TimedPose& a, const TimedPose& b) { return a.t_ns < b.t_ns; });
    return out;
}

/// associate.py: every (rgb, depth) pair within max_dt, taken greedily by |dt|, one-to-one; sorted by rgb time.
inline std::vector<std::pair<size_t, size_t>> AssociateTum(const std::vector<TumStamped>& rgb,
                                                           const std::vector<TumStamped>& depth,
                                                           int64_t max_dt_ns = 20000000LL) {
    std::vector<std::pair<int64_t, std::pair<size_t, size_t>>> cand;
    size_t lo = 0;
    for (size_t i = 0; i < rgb.size(); ++i) {
        while (lo < depth.size() && depth[lo].t_ns < rgb[i].t_ns - max_dt_ns) ++lo;
        for (size_t j = lo; j < depth.size() && depth[j].t_ns <= rgb[i].t_ns + max_dt_ns; ++j) {
            cand.push_back({std::llabs(depth[j].t_ns - rgb[i].t_ns), {i, j}});
        }
    }
    std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<char> used_r(rgb.size(), 0), used_d(depth.size(), 0);
    std::vector<std::pair<size_t, size_t>> out;
    for (const auto& c : cand) {
        if (used_r[c.second.first] || used_d[c.second.second]) continue;
        used_r[c.second.first] = used_d[c.second.second] = 1;
        out.push_back(c.second);
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct TumIntrinsics {
    double fx, fy, cx, cy;
    std::array<double, 5> dist;   // k1 k2 p1 p2 k3 (the page's d0..d4)
};

/// The file-format page's RGB table; `ros_default` = the page's recommended uncalibrated set.
inline TumIntrinsics TumCameraFor(const std::string& sequence_dir_name, bool ros_default) {
    if (ros_default) return {525.0, 525.0, 319.5, 239.5, {0, 0, 0, 0, 0}};
    if (sequence_dir_name.find("freiburg1") != std::string::npos)
        return {517.3, 516.5, 318.6, 255.3, {0.2624, -0.9531, -0.0054, 0.0026, 1.1633}};
    if (sequence_dir_name.find("freiburg2") != std::string::npos)
        return {520.9, 521.0, 325.1, 249.7, {0.2312, -0.7849, -0.0033, -0.0001, 0.9172}};
    if (sequence_dir_name.find("freiburg3") != std::string::npos)
        return {535.4, 539.2, 320.1, 247.6, {0, 0, 0, 0, 0}};   // fr3 is distributed undistorted
    throw ConfigError("TUM: cannot tell the camera (freiburg1/2/3) from '" + sequence_dir_name + "'");
}

class TumSource : public StereoSource {
public:
    static constexpr int kWidth = 640, kHeight = 480;
    static constexpr double kDepthFactor = 5000.0;   // 16-bit PNG value per metre

    explicit TumSource(const std::string& sequence, bool ros_default_intrinsics = false,
                       perception::DepthSensorModel model = {})
        : seq_(sequence), cam_(TumCameraFor(seq_.filename().string(), ros_default_intrinsics)),
          ros_default_(ros_default_intrinsics), model_(model) {
        rgb_ = ParseTumList(ReadTextFile((seq_ / "rgb.txt").string()), "rgb.txt");
        depth_ = ParseTumList(ReadTextFile((seq_ / "depth.txt").string()), "depth.txt");
        pairs_ = AssociateTum(rgb_, depth_);
        gt_ = ParseTumGroundTruth(ReadTextFile((seq_ / "groundtruth.txt").string()));
        const cv::Mat K = (cv::Mat_<double>(3, 3) << cam_.fx, 0, cam_.cx, 0, cam_.fy, cam_.cy, 0, 0, 1);
        const cv::Mat D(1, 5, CV_64F, cam_.dist.data());
        // newCameraMatrix = K: the undistorted image keeps the calibrated pinhole.
        cv::initUndistortRectifyMap(K, D, cv::Mat(), K, {kWidth, kHeight}, CV_32FC1, map_[0], map_[1]);
    }
    std::string RunName() const override { return seq_.filename().string(); }
    std::string SceneName() const override { return "TUM/" + seq_.filename().string(); }
    std::string CameraConfig() const override {
        return std::string("tum_rgbd_") + (ros_default_ ? "ros_default" : "calibrated");
    }
    std::string BagChecksum() const override {
        const auto prov = seq_ / "provenance.json";
        return std::filesystem::exists(prov) ? "sha256:" + YAML::LoadFile(prov.string())["sha256"].as<std::string>()
                                             : std::string(kUnknownField);
    }
    /// The virtual rig: undistorted pinhole, right = left translated by -B along x (p_right = p_left - (B, 0, 0)).
    perception::RawStereoCalibration Calibration() const override {
        perception::PinholeRadtan c;
        c.width = kWidth;
        c.height = kHeight;
        c.fu = cam_.fx, c.fv = cam_.fy, c.cu = cam_.cx, c.cv = cam_.cy;
        perception::RawStereoCalibration r{c, c, core::SE3{}};
        r.T_right_left.t = core::Vec3(-model_.virtual_baseline_m, 0.0, 0.0);
        return r;
    }
    core::SE3 T_B_C0() const override { return core::SE3{}; }   // no IMU: the body IS the colour camera
    std::vector<TimedPose> GroundTruth() const override { return gt_; }
    size_t FrameCount() const override { return pairs_.size(); }
    uint32_t Agent() const override { return 0; }
    std::optional<perception::DepthSensorModel> DepthSensor() const override { return model_; }
    size_t ForEachFrame(size_t max_frames, const std::function<void(const StereoFrame&)>& on_frame) override {
        const size_t n = max_frames ? std::min(max_frames, FrameCount()) : FrameCount();
        for (size_t i = 0; i < n; ++i) {
            const auto [ri, di] = pairs_[i];
            const cv::Mat bgr = cv::imread((seq_ / rgb_[ri].file).string(), cv::IMREAD_COLOR);
            const cv::Mat d16 = cv::imread((seq_ / depth_[di].file).string(), cv::IMREAD_UNCHANGED);
            if (bgr.empty() || d16.type() != CV_16U) throw std::runtime_error("TUM: failed to read frame " + std::to_string(i));
            StereoFrame f;
            f.agent = 0;
            f.id = i;
            f.t_ns = rgb_[ri].t_ns;   // the pose is the colour camera's: stamp with the RGB time
            cv::Mat rgb, rgb_u, d32;
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
            cv::remap(rgb, rgb_u, map_[0], map_[1], cv::INTER_LINEAR, cv::BORDER_CONSTANT);
            d16.convertTo(d32, CV_32F, 1.0 / kDepthFactor);
            cv::remap(d32, f.depth, map_[0], map_[1], cv::INTER_NEAREST, cv::BORDER_CONSTANT);
            f.left_rgb = rgb_u;
            cv::cvtColor(rgb_u, f.left, cv::COLOR_RGB2GRAY);
            on_frame(f);
        }
        return n;
    }

private:
    std::filesystem::path seq_;
    TumIntrinsics cam_;
    bool ros_default_;
    perception::DepthSensorModel model_;
    std::vector<TumStamped> rgb_, depth_;
    std::vector<std::pair<size_t, size_t>> pairs_;
    std::vector<TimedPose> gt_;
    cv::Mat map_[2];
};

}  // namespace ariadne::runtime
