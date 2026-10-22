#pragma once
// ---------------------------------------------------------------------------
// ROS-free decoding of the ROS 2 messages the simulator records (D25's gap,
// design SIM_1.1-1.3 §7). rosbag2 stores each message as its RMW serialization:
// XCDR1 ("plain CDR"), which is
//
//   4-byte encapsulation header: 0x00, kind, options(2); kind 0x00 = CDR big-
//     endian, 0x01 = CDR little-endian. rmw_fastrtps writes 0x01.
//   fields in declaration order, each primitive ALIGNED TO ITS OWN SIZE counted
//     from the first byte after the header (so a float64 after a 1-byte bool
//     skips up to 7 padding bytes);
//   string    = uint32 length INCLUDING the terminating NUL, the bytes, the NUL;
//   sequence  = uint32 element count, then the elements;
//   fixed array (float64[9]) = the elements only.
//
// Only the message types Ariadne consumes are decoded, into Ariadne's own
// structs: no ROS header or type ever enters the codebase (arch §3 dependency
// rule), and replay works from the bag file alone.
//
// Every read is bounds-checked and malformed input throws CdrError; a bag that
// decodes "mostly" is worse than one that fails, because the result would look
// plausible (same rule as McapRecordReader's problem handler).
// Verified against bytes produced by rclpy.serialize_message
// (tests/fixtures/ros_cdr/, tests/unit/test_ros_cdr.cpp).
// ---------------------------------------------------------------------------
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ariadne::runtime::ros_cdr {

struct CdrError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class CdrReader {
public:
    /// `data` must outlive the reader. Parses and checks the encapsulation header.
    CdrReader(const uint8_t* data, size_t size);

    template <typename T>
    T Read() {
        static_assert(std::is_arithmetic_v<T>, "primitive types only");
        Align(sizeof(T));
        Need(sizeof(T));
        T v;
        std::memcpy(&v, p_ + pos_, sizeof(T));
        if (swap_) v = ByteSwap(v);
        pos_ += sizeof(T);
        return v;
    }

    bool ReadBool();
    std::string ReadString();
    /// uint8[] sequence as one block (image payloads): no per-element loop.
    std::vector<uint8_t> ReadByteSequence();
    std::vector<double> ReadFloat64Sequence();
    template <size_t N>
    std::array<double, N> ReadFloat64Array() {
        std::array<double, N> a{};
        for (auto& v : a) v = Read<double>();
        return a;
    }
    uint32_t ReadSequenceLength(size_t min_element_bytes);

    size_t position() const { return pos_; }
    size_t remaining() const { return size_ - kHeader - pos_; }
    /// Everything consumed except trailing alignment padding (< 8 bytes).
    void ExpectEnd() const;

private:
    static constexpr size_t kHeader = 4;
    void Align(size_t n) {
        const size_t pad = (n - pos_ % n) % n;
        Need(pad);
        pos_ += pad;
    }
    void Need(size_t n) const {
        if (n > remaining()) throw CdrError("CDR: read past the end of the message");
    }
    template <typename T>
    static T ByteSwap(T v) {
        uint8_t b[sizeof(T)];
        std::memcpy(b, &v, sizeof(T));
        for (size_t i = 0; i < sizeof(T) / 2; ++i) std::swap(b[i], b[sizeof(T) - 1 - i]);
        std::memcpy(&v, b, sizeof(T));
        return v;
    }

    const uint8_t* p_;     // first payload byte (after the encapsulation header)
    size_t size_;          // total size including the header
    size_t pos_ = 0;       // offset from p_: the alignment origin
    bool swap_ = false;
};

// --- Ariadne-side mirrors of the recorded ROS message types ------------------

struct Time {
    int32_t sec = 0;
    uint32_t nanosec = 0;
    int64_t ns() const { return int64_t(sec) * 1000000000 + nanosec; }
};
struct Header {
    Time stamp;
    std::string frame_id;
};
struct Vector3 {
    double x = 0, y = 0, z = 0;
};
struct Quaternion {
    double x = 0, y = 0, z = 0, w = 1;
};

struct Image {                       // sensor_msgs/msg/Image
    Header header;
    uint32_t height = 0, width = 0;
    std::string encoding;            // "rgb8", "mono8", "32FC1", ...
    uint8_t is_bigendian = 0;
    uint32_t step = 0;               // bytes per row
    std::vector<uint8_t> data;
};

struct RegionOfInterest {
    uint32_t x_offset = 0, y_offset = 0, height = 0, width = 0;
    bool do_rectify = false;
};
struct CameraInfo {                  // sensor_msgs/msg/CameraInfo
    Header header;
    uint32_t height = 0, width = 0;
    std::string distortion_model;
    std::vector<double> d;
    std::array<double, 9> k{}, r{};
    std::array<double, 12> p{};
    uint32_t binning_x = 0, binning_y = 0;
    RegionOfInterest roi;
};

struct Imu {                         // sensor_msgs/msg/Imu
    Header header;
    Quaternion orientation;
    std::array<double, 9> orientation_covariance{};
    Vector3 angular_velocity;
    std::array<double, 9> angular_velocity_covariance{};
    Vector3 linear_acceleration;
    std::array<double, 9> linear_acceleration_covariance{};
};

struct PoseStamped {                 // geometry_msgs/msg/PoseStamped
    Header header;
    Vector3 position;
    Quaternion orientation;
};

struct Pose {                        // geometry_msgs/msg/Pose (Point has Vector3's layout)
    Vector3 position;
    Quaternion orientation;
};
struct PoseArray {                   // geometry_msgs/msg/PoseArray: the live command (design LIVE_RUNTIME_L1 D151)
    Header header;
    std::vector<Pose> poses;
};

struct TransformStamped {            // geometry_msgs/msg/TransformStamped
    Header header;
    std::string child_frame_id;
    Vector3 translation;
    Quaternion rotation;
};

Image DecodeImage(const uint8_t* data, size_t size);
CameraInfo DecodeCameraInfo(const uint8_t* data, size_t size);
Imu DecodeImu(const uint8_t* data, size_t size);
PoseStamped DecodePoseStamped(const uint8_t* data, size_t size);
std::vector<TransformStamped> DecodeTfMessage(const uint8_t* data, size_t size);   // tf2_msgs/TFMessage
Time DecodeClock(const uint8_t* data, size_t size);                                 // rosgraph_msgs/Clock
PoseArray DecodePoseArray(const uint8_t* data, size_t size);                        // geometry_msgs/PoseArray
std::string DecodeString(const uint8_t* data, size_t size);                         // std_msgs/String

/// Checks an Image's internal consistency (step >= width*channels, data == step*height).
void ValidateImage(const Image& img);
/// Bytes per pixel for the encodings Ariadne records; throws for any other.
size_t BytesPerPixel(const std::string& encoding);

}  // namespace ariadne::runtime::ros_cdr
