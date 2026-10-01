// ---------------------------------------------------------------------------
// ariadne_canonical_streams: writes the CANONICAL IMU and GT-pose streams of a
// mission, and its /tf_static, as their own rosbag2 MCAP bag (DECISIONS D39, D40;
// design SIM_1.1-1.3 §7). The static TF comes from the camera sensor poses in the
// same SDF gz renders with (D40: no ROS node publishes it any more).
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
#include <rclcpp/qos.hpp>
#include <sdf/Link.hh>
#include <sdf/Model.hh>
#include <sdf/Plugin.hh>
#include <sdf/Root.hh>
#include <sdf/Sensor.hh>
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

struct Mission {
    std::vector<MoverConfig> movers;
    std::vector<cdr::TransformStamped> static_tf;   // body -> <cam>_optical, every agent
};

/// gz camera link frame (+x forward, image right = -y, down = -z) -> ROS optical
/// (x right, y down, z forward): R_B_optical columns (-y_B, -z_B, +x_B), i.e. the
/// quaternion (x, y, z, w) = (-0.5, 0.5, -0.5, 0.5). Valid because every camera
/// sensor's <pose> rotation is identity, which LoadMission asserts.
constexpr double kQOptical[4] = {-0.5, 0.5, -0.5, 0.5};

/// Every KinematicMover in the world, in model order, parsed exactly as gz-sim
/// hands the <plugin> to Configure (sdf::Plugin::ToElement()); and the static TF
/// of each mover's model, read from the camera sensor poses gz renders with.
Mission LoadMission(const std::string& world_path) {
    sdf::Root root;
    const sdf::Errors errors = root.Load(world_path);
    if (!errors.empty()) {
        std::string msg = "cannot load " + world_path + ":";
        for (const auto& e : errors) msg += "\n  " + e.Message();
        throw std::runtime_error(msg);
    }
    if (root.WorldCount() != 1) throw std::runtime_error("expected exactly one <world>");
    const sdf::World* w = root.WorldByIndex(0);
    Mission out;
    for (uint64_t i = 0; i < w->ModelCount(); ++i) {
        const sdf::Model* model = w->ModelByIndex(i);
        for (const sdf::Plugin& p : model->Plugins()) {
            if (p.Name() != kPluginName) continue;
            out.movers.push_back(ariadne::sim::ParseMoverConfig(p.ToElement()));
            const MoverConfig& c = out.movers.back();
            const sdf::Link* link = model->LinkByName("base_link");
            if (link == nullptr) throw std::runtime_error(model->Name() + ": no base_link");
            const std::string base = "base_link";
            if (c.frame_id.size() < base.size() || c.frame_id.compare(c.frame_id.size() - base.size(), base.size(), base)) {
                throw std::runtime_error("frame_id '" + c.frame_id + "' does not end in base_link");
            }
            for (const std::string cam : {"left", "right"}) {
                const sdf::Sensor* sensor = link->SensorByName(cam);
                if (sensor == nullptr) throw std::runtime_error(model->Name() + ": no '" + cam + "' sensor");
                if (!sensor->PoseRelativeTo().empty()) throw std::runtime_error(cam + ": pose not relative to its link");
                const auto pose = sensor->RawPose();
                const auto& r = pose.Rot();
                if (std::abs(r.W() - 1.0) > 1e-15 || r.X() != 0.0 || r.Y() != 0.0 || r.Z() != 0.0) {
                    throw std::runtime_error(model->Name() + "/" + cam + ": camera pose rotation is not identity");
                }
                cdr::TransformStamped t;          // stamp 0: static
                t.header.frame_id = c.frame_id;
                t.child_frame_id = c.frame_id.substr(0, c.frame_id.size() - base.size()) + cam + "_optical";
                t.translation = {pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()};
                t.rotation = {kQOptical[0], kQOptical[1], kQOptical[2], kQOptical[3]};
                out.static_tf.push_back(t);
            }
        }
    }
    if (out.movers.empty()) throw std::runtime_error("no " + std::string(kPluginName) + " in " + world_path);
    for (size_t i = 1; i < out.movers.size(); ++i) {
        if (out.movers[i].period_ns != out.movers[0].period_ns) throw std::runtime_error("agents disagree on imu_period_ns");
    }
    return out;
}

std::vector<uint8_t> Encode(const std::vector<cdr::TransformStamped>& v) {
    std::vector<uint8_t> b = cdr::EncodeTfMessage(v);
    const auto d = cdr::DecodeTfMessage(b.data(), b.size());
    bool same = d.size() == v.size();
    for (size_t i = 0; same && i < v.size(); ++i) {
        same = Same(d[i].header, v[i].header) && d[i].child_frame_id == v[i].child_frame_id &&
               Same(d[i].translation, v[i].translation) && Same(d[i].rotation, v[i].rotation);
    }
    if (!same) throw std::runtime_error("TFMessage CDR round trip is not bit-exact");
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args a = ParseArgs(argc, argv);
        if (std::filesystem::exists(a.out)) throw std::runtime_error(a.out + " exists; refusing to overwrite");
        const Mission mission = LoadMission(a.world);
        const std::vector<MoverConfig>& movers = mission.movers;
        const int64_t period = movers[0].period_ns;

        rosbag2_storage::StorageOptions so;
        so.uri = a.out;
        so.storage_id = "mcap";
        so.storage_preset_profile = "zstd_fast";   // same container settings as the sensor bag
        rosbag2_cpp::Writer writer;
        writer.open(so, {"cdr", "cdr"});

        // /tf_static first, at stamp 0, offered as latched (reliable, transient-local,
        // depth 1) so `ros2 bag play` republishes it the way tf2 expects.
        rosbag2_storage::TopicMetadata tf_topic;
        tf_topic.name = "/tf_static";
        tf_topic.type = "tf2_msgs/msg/TFMessage";
        tf_topic.serialization_format = "cdr";
        tf_topic.offered_qos_profiles = {rclcpp::QoS(1).reliable().transient_local()};
        writer.create_topic(tf_topic);
        Write(writer, Encode(mission.static_tf), tf_topic.name, tf_topic.type, 0);

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
        std::printf("{\"agents\": %zu, \"period_ns\": %lld, \"samples_per_topic\": %lld, \"last_ns\": %lld, "
                    "\"static_transforms\": %zu}\n",
                    movers.size(), static_cast<long long>(period), static_cast<long long>(n),
                    static_cast<long long>((n - 1) * period), mission.static_tf.size());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ariadne_canonical_streams: " << e.what() << "\n";
        return 1;
    }
}
