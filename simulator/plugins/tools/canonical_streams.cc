// ---------------------------------------------------------------------------
// ariadne_canonical_streams: writes the CANONICAL IMU and GT-pose streams of a
// mission as their own rosbag2 MCAP bag (DECISIONS D39, design SIM_1.1-1.3 §7).
//
//   ariadne_canonical_streams --world mission_default.sdf --end-ns N --out DIR
//
// Why: both streams are exact functions of the agent scripts (the mover computes
// them analytically; gz measures nothing), but carrying them through
// gz-transport -> ros_gz_bridge -> rosbag2 loses a sample now and then under load
// (SMOKE 4k). This tool evaluates the SAME script, with the SAME code
// (MoverScript.hh), at every stamp the plugin publishes, so the bag it writes is
// complete by construction and a pure function of (mission SDF, end time).
//
// Stamp grid per agent: t_k = k * imu_period_ns for k = 0 .. floor(N / period).
// The plugin publishes k >= 1 (gz's first PreUpdate is at one step); k = 0 is
// added so GT brackets the first camera frame (stamped before the first IMU tick).
// log_time = publish_time = header stamp, so log-time order == stamp order.
//
// Message content mirrors what ros_gz_bridge makes of the plugin's gz messages
// (identity IMU orientation, zero covariances, header.frame_id from the "frame_id"
// header data); sim_record.py checks every live sample field-by-field (bit
// patterns) against this bag, so any divergence fails the recording.
//
// Encoding: runtime/replay/ros_cdr_writer.hpp, NOT rclcpp serialization, whose
// CDR alignment padding is uninitialised memory (observed: two bytes "at"). With
// zeroed padding the bag is byte-reproducible; every message is decoded back
// with the runtime's own CDR decoder and compared bit-for-bit before writing.
// ---------------------------------------------------------------------------
#include "kinematic_mover/MoverScript.hh"
#include "runtime/replay/ros_cdr.hpp"
#include "runtime/replay/ros_cdr_writer.hpp"

#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sdf/Model.hh>
#include <sdf/Plugin.hh>
#include <sdf/Root.hh>
#include <sdf/World.hh>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ariadne::sim::MoverConfig;
using ariadne::sim::MoverSample;
namespace cdr = ariadne::runtime::ros_cdr;

constexpr char kPluginName[] = "ariadne::sim::KinematicMover";

struct Args {
    std::string world, out;
    int64_t end_ns = -1;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
        const std::string v = argv[++i];
        if (k == "--world") a.world = v;
        else if (k == "--out") a.out = v;
        else if (k == "--end-ns") a.end_ns = std::stoll(v);
        else throw std::invalid_argument("unknown argument " + k);
    }
    if (a.world.empty() || a.out.empty() || a.end_ns < 0) {
        throw std::invalid_argument("usage: ariadne_canonical_streams --world SDF --end-ns N --out DIR");
    }
    return a;
}

/// Every KinematicMover in the world, in model order, parsed exactly as gz-sim
/// hands the <plugin> to Configure (sdf::Plugin::ToElement()).
std::vector<MoverConfig> LoadMovers(const std::string& world_path) {
    sdf::Root root;
    const sdf::Errors errors = root.Load(world_path);
    if (!errors.empty()) {
        std::string msg = "cannot load " + world_path + ":";
        for (const auto& e : errors) msg += "\n  " + e.Message();
        throw std::runtime_error(msg);
    }
    if (root.WorldCount() != 1) throw std::runtime_error("expected exactly one <world>");
    const sdf::World* w = root.WorldByIndex(0);
    std::vector<MoverConfig> movers;
    for (uint64_t i = 0; i < w->ModelCount(); ++i) {
        for (const sdf::Plugin& p : w->ModelByIndex(i)->Plugins()) {
            if (p.Name() == kPluginName) movers.push_back(ariadne::sim::ParseMoverConfig(p.ToElement()));
        }
    }
    if (movers.empty()) throw std::runtime_error("no " + std::string(kPluginName) + " in " + world_path);
    for (size_t i = 1; i < movers.size(); ++i) {
        if (movers[i].period_ns != movers[0].period_ns) throw std::runtime_error("agents disagree on imu_period_ns");
    }
    return movers;
}

cdr::Header MakeHeader(const MoverConfig& c, int64_t ns) {
    cdr::Header h;
    h.stamp.sec = int32_t(ns / 1000000000);
    h.stamp.nanosec = uint32_t(ns % 1000000000);
    h.frame_id = c.frame_id;
    return h;
}

cdr::Imu ImuMsg(const MoverConfig& c, const MoverSample& m) {
    cdr::Imu msg;                        // covariances 0, as the bridge leaves them
    msg.header = MakeHeader(c, m.ns);
    msg.orientation = {0.0, 0.0, 0.0, 1.0};   // NOT measured (identity), exactly as the plugin publishes
    msg.angular_velocity = {m.imu.gyro[0], m.imu.gyro[1], m.imu.gyro[2]};
    msg.linear_acceleration = {m.imu.accel[0], m.imu.accel[1], m.imu.accel[2]};
    return msg;
}

cdr::PoseStamped PoseMsg(const MoverConfig& c, const MoverSample& m) {
    cdr::PoseStamped msg;
    msg.header = MakeHeader(c, m.ns);
    msg.position = {m.state.pose[0], m.state.pose[1], m.state.pose[2]};
    msg.orientation = {m.q[1], m.q[2], m.q[3], m.q[0]};   // (x, y, z, w) from (w, x, y, z)
    return msg;
}

// --- bit-exact round-trip self-check (a -0.0 must stay -0.0) -----------------
bool Same(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
bool Same(const cdr::Vector3& a, const cdr::Vector3& b) { return Same(a.x, b.x) && Same(a.y, b.y) && Same(a.z, b.z); }
bool Same(const cdr::Quaternion& a, const cdr::Quaternion& b) {
    return Same(a.x, b.x) && Same(a.y, b.y) && Same(a.z, b.z) && Same(a.w, b.w);
}
template <size_t N>
bool Same(const std::array<double, N>& a, const std::array<double, N>& b) {
    return std::memcmp(a.data(), b.data(), sizeof(double) * N) == 0;
}
bool Same(const cdr::Header& a, const cdr::Header& b) {
    return a.stamp.sec == b.stamp.sec && a.stamp.nanosec == b.stamp.nanosec && a.frame_id == b.frame_id;
}

std::vector<uint8_t> Encode(const cdr::Imu& m) {
    std::vector<uint8_t> b = cdr::EncodeImu(m);
    const cdr::Imu d = cdr::DecodeImu(b.data(), b.size());
    if (!(Same(d.header, m.header) && Same(d.orientation, m.orientation) &&
          Same(d.orientation_covariance, m.orientation_covariance) && Same(d.angular_velocity, m.angular_velocity) &&
          Same(d.angular_velocity_covariance, m.angular_velocity_covariance) &&
          Same(d.linear_acceleration, m.linear_acceleration) &&
          Same(d.linear_acceleration_covariance, m.linear_acceleration_covariance))) {
        throw std::runtime_error("IMU CDR round trip is not bit-exact");
    }
    return b;
}

std::vector<uint8_t> Encode(const cdr::PoseStamped& m) {
    std::vector<uint8_t> b = cdr::EncodePoseStamped(m);
    const cdr::PoseStamped d = cdr::DecodePoseStamped(b.data(), b.size());
    if (!(Same(d.header, m.header) && Same(d.position, m.position) && Same(d.orientation, m.orientation))) {
        throw std::runtime_error("PoseStamped CDR round trip is not bit-exact");
    }
    return b;
}

void Write(rosbag2_cpp::Writer& w, const std::vector<uint8_t>& bytes, const std::string& topic,
           const std::string& type, int64_t ns) {
    auto msg = std::make_shared<rclcpp::SerializedMessage>(bytes.size());
    auto& raw = msg->get_rcl_serialized_message();
    std::memcpy(raw.buffer, bytes.data(), bytes.size());
    raw.buffer_length = bytes.size();
    w.write(msg, topic, type, ns, ns);   // recv (MCAP logTime) = send (publishTime) = stamp
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = ParseArgs(argc, argv);
        if (std::filesystem::exists(a.out)) throw std::runtime_error(a.out + " exists; refusing to overwrite");
        const std::vector<MoverConfig> movers = LoadMovers(a.world);
        const int64_t period = movers[0].period_ns;

        rosbag2_storage::StorageOptions so;
        so.uri = a.out;
        so.storage_id = "mcap";
        so.storage_preset_profile = "zstd_fast";   // same container settings as the sensor bag
        rosbag2_cpp::Writer writer;
        writer.open(so, {"cdr", "cdr"});

        int64_t n = 0;
        for (int64_t k = 0; k * period <= a.end_ns; ++k) {
            const int64_t ns = k * period;
            for (const MoverConfig& c : movers) {   // agent order within a tick: fixed (model order)
                const MoverSample m = ariadne::sim::SampleAt(c, ns);
                Write(writer, Encode(ImuMsg(c, m)), c.imu_topic, "sensor_msgs/msg/Imu", ns);
                Write(writer, Encode(PoseMsg(c, m)), c.pose_topic, "geometry_msgs/msg/PoseStamped", ns);
            }
            ++n;
        }
        writer.close();
        std::printf("{\"agents\": %zu, \"period_ns\": %lld, \"samples_per_topic\": %lld, \"last_ns\": %lld}\n",
                    movers.size(), static_cast<long long>(period), static_cast<long long>(n),
                    static_cast<long long>((n - 1) * period));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ariadne_canonical_streams: " << e.what() << "\n";
        return 1;
    }
}
