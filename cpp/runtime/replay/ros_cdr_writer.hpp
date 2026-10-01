#pragma once
// ---------------------------------------------------------------------------
// ROS-free ENCODING of the message types Ariadne writes itself into the canonical
// bag of a simulated recording: IMU, GT pose (D39) and the static TF (D40). The inverse
// of ros_cdr.hpp's decoders, same XCDR1 layout (little-endian encapsulation
// 0x00 0x01 0x00 0x00, primitives aligned to their own size from the first byte
// after the header, strings as uint32 length-with-NUL + bytes + NUL).
//
// Why not rclcpp::Serialization: Fast-CDR SKIPS alignment padding without
// writing it, so the pad bytes are whatever the buffer held (observed: "at" from
// a previous frame_id), which makes the output non-reproducible. Here every pad
// byte is 0, so equal messages encode to equal bytes, always. Padding carries no
// content: the decoder ignores it and so does any conforming reader.
//
// Header-only and std-only so the sim-side writer tool can use it without
// linking the runtime library. Tested against rclpy's bytes in
// tests/unit/test_ros_cdr.cpp.
// ---------------------------------------------------------------------------
#include "runtime/replay/ros_cdr.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace ariadne::runtime::ros_cdr {

class CdrWriter {
public:
    CdrWriter() { buf_ = {0x00, 0x01, 0x00, 0x00}; }   // CDR little-endian, options 0

    template <typename T>
    void Write(T v) {
        static_assert(std::is_arithmetic_v<T>, "primitive types only");
        Align(sizeof(T));
        uint8_t b[sizeof(T)];
        std::memcpy(b, &v, sizeof(T));                 // host order; checked little-endian below
        buf_.insert(buf_.end(), b, b + sizeof(T));
    }
    void WriteString(const std::string& s) {
        if (s.find('\0') != std::string::npos) throw CdrError("CDR: string with embedded NUL");
        Write<uint32_t>(uint32_t(s.size() + 1));
        buf_.insert(buf_.end(), s.begin(), s.end());
        buf_.push_back(0);
    }
    template <size_t N>
    void WriteFloat64Array(const std::array<double, N>& a) {
        for (double v : a) Write<double>(v);
    }
    const std::vector<uint8_t>& bytes() const { return buf_; }

private:
    void Align(size_t n) {
        const size_t pos = buf_.size() - 4;             // alignment origin: after the header
        buf_.insert(buf_.end(), (n - pos % n) % n, uint8_t{0});
    }
    static_assert(sizeof(double) == 8 && sizeof(float) == 4, "IEEE-754 sizes");
    std::vector<uint8_t> buf_;
};

namespace detail {

inline bool HostIsLittleEndian() {
    const uint16_t one = 1;
    uint8_t b;
    std::memcpy(&b, &one, 1);
    return b == 1;
}

inline void WriteHeader(CdrWriter& w, const Header& h) {
    w.Write<int32_t>(h.stamp.sec);
    w.Write<uint32_t>(h.stamp.nanosec);
    w.WriteString(h.frame_id);
}

inline void WriteVector3(CdrWriter& w, const Vector3& v) {
    w.Write<double>(v.x);
    w.Write<double>(v.y);
    w.Write<double>(v.z);
}

inline void WriteQuaternion(CdrWriter& w, const Quaternion& q) {   // ROS field order x, y, z, w
    w.Write<double>(q.x);
    w.Write<double>(q.y);
    w.Write<double>(q.z);
    w.Write<double>(q.w);
}

inline void RequireLittleEndianHost() {
    if (!HostIsLittleEndian()) throw CdrError("CDR writer: big-endian host unsupported");
}

}  // namespace detail

/// sensor_msgs/msg/Imu
inline std::vector<uint8_t> EncodeImu(const Imu& m) {
    detail::RequireLittleEndianHost();
    CdrWriter w;
    detail::WriteHeader(w, m.header);
    detail::WriteQuaternion(w, m.orientation);
    w.WriteFloat64Array(m.orientation_covariance);
    detail::WriteVector3(w, m.angular_velocity);
    w.WriteFloat64Array(m.angular_velocity_covariance);
    detail::WriteVector3(w, m.linear_acceleration);
    w.WriteFloat64Array(m.linear_acceleration_covariance);
    return w.bytes();
}

/// geometry_msgs/msg/PoseStamped
inline std::vector<uint8_t> EncodePoseStamped(const PoseStamped& m) {
    detail::RequireLittleEndianHost();
    CdrWriter w;
    detail::WriteHeader(w, m.header);
    detail::WriteVector3(w, m.position);          // geometry_msgs/Point has the same layout
    detail::WriteQuaternion(w, m.orientation);
    return w.bytes();
}

/// tf2_msgs/msg/TFMessage (a sequence of geometry_msgs/TransformStamped)
inline std::vector<uint8_t> EncodeTfMessage(const std::vector<TransformStamped>& v) {
    detail::RequireLittleEndianHost();
    CdrWriter w;
    w.Write<uint32_t>(uint32_t(v.size()));
    for (const auto& t : v) {
        detail::WriteHeader(w, t.header);
        w.WriteString(t.child_frame_id);
        detail::WriteVector3(w, t.translation);   // geometry_msgs/Transform: translation, rotation
        detail::WriteQuaternion(w, t.rotation);
    }
    return w.bytes();
}

}  // namespace ariadne::runtime::ros_cdr
