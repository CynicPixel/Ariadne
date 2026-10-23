#pragma once
// ---------------------------------------------------------------------------
// EuRoC MAV dataset reader, ASL format (Burri et al., IJRR 2016). The first
// real-data input (Invariant 5: datasets before simulator). ROS-free: reads
// the files, not the bags.
//
//   <seq>/mav0/cam0/sensor.yaml    intrinsics, distortion, T_BS
//   <seq>/mav0/cam0/data.csv       "#timestamp [ns],filename"
//   <seq>/mav0/cam0/data/<ns>.png  8-bit mono, 752x480, 20 Hz
//   (cam1 likewise; cam0 is the LEFT camera)
//
// T_BS is "body <- sensor": it maps camera coordinates into the body (IMU)
// frame, i.e. T_B_C in camera_model.hpp's convention.
//
// OpenCV-free: images are loaded by the OpenCV-side caller, which lets these
// parsers be tested on the host.
// ---------------------------------------------------------------------------
#include "core/config/config.hpp"
#include "core/inertial/imu.hpp"
#include "perception/stereo/camera_model.hpp"

#include <Eigen/Geometry>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ariadne::runtime {

using core::ConfigError;

struct EurocCamera {
    perception::PinholeRadtan intrinsics;
    core::SE3 T_B_C;   // body <- camera
    double rate_hz = 0.0;
};

/// Parse one camera's sensor.yaml. Strict: every key is read or explicitly
/// acknowledged, the model must be pinhole + radial-tangential or equidistant
/// (Kalibr's names; equidistant = cv::fisheye KB4, D143), and T_BS must be a
/// 4x4 rigid transform.
inline EurocCamera ParseEurocCamera(std::string text, const std::string& origin) {
    // EuRoC writes OpenCV's "%YAML:1.0" directive, which is not valid YAML
    // (the standard form is "%YAML 1.2") and which yaml-cpp rejects.
    if (text.rfind("%YAML:", 0) == 0) text = text.substr(text.find('\n') + 1);
    core::StrictConfig c = core::StrictConfig::FromString(text, origin);

    if (c.Require<std::string>("sensor_type") != "camera") {
        throw ConfigError(origin + ": sensor_type is not 'camera'");
    }
    (void)c.Optional<std::string>("comment", "");   // free text, acknowledged
    if (c.Require<std::string>("camera_model") != "pinhole") {
        throw ConfigError(origin + ": only camera_model 'pinhole' is supported");
    }
    const auto dm = c.Require<std::string>("distortion_model");
    if (dm != "radial-tangential" && dm != "equidistant") {
        throw ConfigError(origin + ": distortion_model must be 'radial-tangential' or 'equidistant'");
    }

    EurocCamera cam;
    cam.intrinsics.model =
        dm == "equidistant" ? perception::DistortionModel::kEquidistant : perception::DistortionModel::kRadtan;
    cam.rate_hz = c.Require<double>("rate_hz");
    const auto res = c.Require<std::vector<int>>("resolution");
    const auto k = c.Require<std::vector<double>>("intrinsics");
    const auto d = c.Require<std::vector<double>>("distortion_coefficients");
    if (res.size() != 2 || k.size() != 4 || d.size() != 4) {
        throw ConfigError(origin + ": resolution/intrinsics/distortion have the wrong length");
    }
    cam.intrinsics.width = res[0];
    cam.intrinsics.height = res[1];
    cam.intrinsics.fu = k[0];
    cam.intrinsics.fv = k[1];
    cam.intrinsics.cu = k[2];
    cam.intrinsics.cv = k[3];
    for (int i = 0; i < 4; ++i) cam.intrinsics.dist[static_cast<size_t>(i)] = d[static_cast<size_t>(i)];

    core::StrictConfig t = c.Section("T_BS");
    if (t.Require<int>("rows") != 4 || t.Require<int>("cols") != 4) {
        throw ConfigError(origin + ": T_BS must be 4x4");
    }
    const auto m = t.Require<std::vector<double>>("data");
    t.RequireAllConsumed();
    if (m.size() != 16) throw ConfigError(origin + ": T_BS data must have 16 entries");
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) cam.T_B_C.R(r, col) = m[static_cast<size_t>(r * 4 + col)];
        cam.T_B_C.t(r) = m[static_cast<size_t>(r * 4 + 3)];
    }
    // A calibration that is not a rotation would rectify into nonsense
    // without failing anywhere, so check it here.
    const double ortho = (cam.T_B_C.R.transpose() * cam.T_B_C.R - core::Mat3::Identity()).norm();
    if (ortho > 1e-6 || cam.T_B_C.R.determinant() < 0.0 || m[12] != 0.0 || m[13] != 0.0 ||
        m[14] != 0.0 || m[15] != 1.0) {
        throw ConfigError(origin + ": T_BS is not a rigid transform");
    }
    c.RequireAllConsumed();
    return cam;
}

inline std::string ReadTextFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw ConfigError("cannot open " + path);
    std::ostringstream s;
    s << in.rdbuf();
    return s.str();
}

inline EurocCamera LoadEurocCamera(const std::string& sensor_yaml) {
    return ParseEurocCamera(ReadTextFile(sensor_yaml), sensor_yaml);
}

/// cam0 = left, cam1 = right: the (R, T) stereoRectify takes.
inline perception::RawStereoCalibration EurocStereo(const EurocCamera& cam0, const EurocCamera& cam1) {
    perception::RawStereoCalibration s;
    s.left = cam0.intrinsics;
    s.right = cam1.intrinsics;
    s.T_right_left = cam1.T_B_C.inverse() * cam0.T_B_C;
    return s;
}

struct EurocImageIndex {
    std::vector<int64_t> t_ns;
    std::vector<std::string> files;   // relative to the camera's data/ directory
};

/// Parse data.csv ("#timestamp [ns],filename"). Timestamps must strictly
/// increase: replay order is time order, and a duplicate would make two frames
/// indistinguishable to everything keyed on time.
inline EurocImageIndex ParseEurocImageCsv(const std::string& text, const std::string& origin) {
    EurocImageIndex idx;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos) throw ConfigError(origin + ": malformed line '" + line + "'");
        const int64_t t = std::stoll(line.substr(0, comma));
        if (!idx.t_ns.empty() && t <= idx.t_ns.back()) {
            throw ConfigError(origin + ": timestamps not strictly increasing at " + std::to_string(t));
        }
        idx.t_ns.push_back(t);
        idx.files.push_back(line.substr(comma + 1));
    }
    return idx;
}

struct StereoPairs {
    std::vector<int64_t> t_ns;
    std::vector<std::string> left;
    std::vector<std::string> right;
    int unmatched_left = 0;
    int unmatched_right = 0;
};

/// Pair frames by EXACT timestamp (the VI-sensor triggers both cameras
/// together), never by row index: one dropped frame would otherwise shift
/// every later pair by 50 ms and SGBM would match unrelated images.
inline StereoPairs PairByTimestamp(const EurocImageIndex& l, const EurocImageIndex& r) {
    StereoPairs p;
    size_t i = 0, j = 0;
    while (i < l.t_ns.size() && j < r.t_ns.size()) {
        if (l.t_ns[i] == r.t_ns[j]) {
            p.t_ns.push_back(l.t_ns[i]);
            p.left.push_back(l.files[i++]);
            p.right.push_back(r.files[j++]);
        } else if (l.t_ns[i] < r.t_ns[j]) {
            ++i;
            ++p.unmatched_left;
        } else {
            ++j;
            ++p.unmatched_right;
        }
    }
    p.unmatched_left += static_cast<int>(l.t_ns.size() - i);
    p.unmatched_right += static_cast<int>(r.t_ns.size() - j);
    return p;
}

struct GroundTruthPose {
    int64_t t_ns = 0;
    core::SE3 T_W_B;   // body (IMU) -> Vicon/Leica world
};

/// state_groundtruth_estimate0/data.csv: "#timestamp, p_RS_R_x..z, q_RS_w, q_RS_x..z, v..., b_w..., b_a..."
/// p_RS_R and q_RS are the body pose in the reference frame R. The ground-truth
/// sensor's T_BS is identity (its sensor.yaml), so S is the body frame.
inline std::vector<GroundTruthPose> ParseEurocGroundTruth(const std::string& text, const std::string& origin) {
    std::vector<GroundTruthPose> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> v;
        int64_t t = 0;
        std::istringstream ls(line);
        std::string cell;
        for (int col = 0; std::getline(ls, cell, ','); ++col) {
            if (col == 0) t = std::stoll(cell);
            else v.push_back(std::stod(cell));
        }
        if (v.size() < 7) throw ConfigError(origin + ": ground-truth row has fewer than 8 columns");
        if (!out.empty() && t <= out.back().t_ns) throw ConfigError(origin + ": timestamps not increasing");
        GroundTruthPose g;
        g.t_ns = t;
        g.T_W_B.t = core::Vec3(v[0], v[1], v[2]);
        g.T_W_B.R = Eigen::Quaterniond(v[3], v[4], v[5], v[6]).normalized().toRotationMatrix();
        out.push_back(g);
    }
    return out;
}

/// imu0/data.csv: "#timestamp [ns], w_RS_S_x..z [rad/s], a_RS_S_x..z [m/s^2]" at 200 Hz. imu0's T_BS is identity,
/// so S is the body frame.
inline std::vector<core::ImuMeasurement> ParseEurocImu(const std::string& text, const std::string& origin) {
    std::vector<core::ImuMeasurement> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> v;
        int64_t t = 0;
        std::istringstream ls(line);
        std::string cell;
        for (int col = 0; std::getline(ls, cell, ','); ++col) {
            if (col == 0) t = std::stoll(cell);
            else v.push_back(std::stod(cell));
        }
        if (v.size() != 6) throw ConfigError(origin + ": IMU row must have 7 columns");
        if (!out.empty() && t <= out.back().t_ns) throw ConfigError(origin + ": IMU timestamps not increasing");
        out.push_back({t, core::Vec3(v[0], v[1], v[2]), core::Vec3(v[3], v[4], v[5])});
    }
    return out;
}

/// imu0/sensor.yaml noise densities (Kalibr convention) and rate; T_BS must be identity (body = IMU), else throws.
inline core::ImuNoise ParseEurocImuNoise(const std::string& text, const std::string& origin) {
    core::StrictConfig c = core::StrictConfig::FromString(text, origin);
    core::ImuNoise n;
    n.gyro_noise_density = c.Require<double>("gyroscope_noise_density");
    n.gyro_random_walk = c.Require<double>("gyroscope_random_walk");
    n.accel_noise_density = c.Require<double>("accelerometer_noise_density");
    n.accel_random_walk = c.Require<double>("accelerometer_random_walk");
    n.rate_hz = c.Require<double>("rate_hz");
    const std::vector<double> T = c.Section("T_BS").Require<std::vector<double>>("data");
    for (int i = 0; i < 16; ++i)
        if (T.size() != 16 || T[size_t(i)] != (i % 5 == 0 ? 1.0 : 0.0)) throw ConfigError(origin + ": imu0 T_BS is not identity");
    return n;
}

/// Ground-truth inertial state (state_groundtruth_estimate0 columns 8-16): velocity in R, gyro and accel biases in
/// the body frame. For VALIDATING the estimator's initialisation only; never fed to it.
struct GroundTruthInertial {
    int64_t t_ns = 0;
    core::Vec3 v_W, b_g, b_a;
};

inline std::vector<GroundTruthInertial> ParseEurocGroundTruthInertial(const std::string& text, const std::string& origin) {
    std::vector<GroundTruthInertial> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> v;
        int64_t t = 0;
        std::istringstream ls(line);
        std::string cell;
        for (int col = 0; std::getline(ls, cell, ','); ++col) {
            if (col == 0) t = std::stoll(cell);
            else v.push_back(std::stod(cell));
        }
        if (v.size() < 16) throw ConfigError(origin + ": ground-truth row has fewer than 17 columns");
        out.push_back({t, core::Vec3(v[7], v[8], v[9]), core::Vec3(v[10], v[11], v[12]), core::Vec3(v[13], v[14], v[15])});
    }
    return out;
}

/// T_B_Crect: the RECTIFIED left camera in the body frame. Rectification
/// rotates camera 0 by R1 (x_rect = R1 x_c0), so T_C0_Crect has rotation R1^T.
/// Leaving R1 out barely moves positions but adds a constant rotation that
/// trajectory alignment cannot remove (advisor, 1.5 design).
inline core::SE3 BodyFromRectifiedLeft(const core::SE3& T_B_C0, const core::Mat3& R1) {
    core::SE3 T_C0_Crect;
    T_C0_Crect.R = R1.transpose();
    return T_B_C0 * T_C0_Crect;
}

}  // namespace ariadne::runtime
