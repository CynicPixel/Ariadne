// The ROS-free CDR decoder against bytes produced by ROS 2 itself
// (rclpy.serialize_message; tests/fixtures/ros_cdr/gen_fixtures.py).
#include "runtime/replay/ros_cdr.hpp"

#include "../fixtures/ros_cdr/ros_cdr_fixtures.inc"

#include <gtest/gtest.h>

#include <functional>

using namespace ariadne::runtime::ros_cdr;
namespace fx = ros_cdr_fixtures;

namespace {

template <typename F>
auto Decode(F f, const std::vector<uint8_t>& b) {
    return f(b.data(), b.size());
}

/// Every strict prefix of a valid message must throw CdrError, never crash or
/// silently succeed (the message's last field always needs its last byte).
template <typename F>
void EveryTruncationThrows(F f, const std::vector<uint8_t>& b) {
    for (size_t n = 0; n + 1 < b.size(); ++n) {
        std::vector<uint8_t> cut(b.begin(), b.begin() + n);
        EXPECT_THROW(f(cut.data(), cut.size()), CdrError) << "prefix length " << n << " of " << b.size();
    }
}

/// A big-endian (encapsulation kind 0x0000) rosgraph_msgs/Clock built by hand, so
/// the byte-swap path is checked against bytes the decoder did not produce.
std::vector<uint8_t> BigEndianClock(int32_t sec, uint32_t nsec) {
    std::vector<uint8_t> b = {0x00, 0x00, 0x00, 0x00};
    for (int i = 3; i >= 0; --i) b.push_back(uint8_t(uint32_t(sec) >> (8 * i)));
    for (int i = 3; i >= 0; --i) b.push_back(uint8_t(nsec >> (8 * i)));
    return b;
}

}  // namespace

TEST(RosCdr, MonoImage) {
    const Image m = Decode(DecodeImage, fx::kImageMono);
    EXPECT_EQ(m.header.stamp.sec, 12);
    EXPECT_EQ(m.header.stamp.nanosec, 345678901u);
    EXPECT_EQ(m.header.stamp.ns(), 12345678901LL);
    EXPECT_EQ(m.header.frame_id, "a");
    EXPECT_EQ(m.height, 3u);
    EXPECT_EQ(m.width, 5u);
    EXPECT_EQ(m.encoding, "mono8");
    EXPECT_EQ(m.step, 5u);
    ASSERT_EQ(m.data.size(), 15u);
    for (size_t i = 0; i < 15; ++i) EXPECT_EQ(m.data[i], i);
}

TEST(RosCdr, RgbImageWithLongFrameId) {
    const Image m = Decode(DecodeImage, fx::kImageRgb);
    EXPECT_EQ(m.header.stamp.ns(), 50000000LL);
    EXPECT_EQ(m.header.frame_id, "ariadne/agent_1/left");
    EXPECT_EQ(m.encoding, "rgb8");
    EXPECT_EQ(m.step, 6u);
    const std::vector<uint8_t> want = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    EXPECT_EQ(m.data, want);
}

TEST(RosCdr, CameraInfoAlignmentAndArrays) {
    const CameraInfo m = Decode(DecodeCameraInfo, fx::kCameraInfo);
    EXPECT_EQ(m.header.frame_id, "left_optical");
    EXPECT_EQ(m.height, 720u);
    EXPECT_EQ(m.width, 1280u);
    EXPECT_EQ(m.distortion_model, "plumb_bob");
    const std::vector<double> d = {-0.05, 0.01, 0.0005, -0.0002, 0.0};
    EXPECT_EQ(m.d, d);
    EXPECT_EQ(m.k[0], 640.0);
    EXPECT_EQ(m.k[2], 639.5);
    EXPECT_EQ(m.k[5], 359.5);
    EXPECT_EQ(m.k[8], 1.0);
    EXPECT_EQ(m.r[4], 1.0);
    EXPECT_EQ(m.p[3], -64.0);
    EXPECT_EQ(m.p[10], 1.0);
    EXPECT_EQ(m.binning_x, 1u);
    EXPECT_EQ(m.binning_y, 2u);
    EXPECT_EQ(m.roi.x_offset, 3u);
    EXPECT_EQ(m.roi.y_offset, 4u);
    EXPECT_EQ(m.roi.height, 5u);
    EXPECT_EQ(m.roi.width, 6u);
    EXPECT_TRUE(m.roi.do_rectify);
}

TEST(RosCdr, Imu) {
    const Imu m = Decode(DecodeImu, fx::kImu);
    EXPECT_EQ(m.header.stamp.sec, 99);
    EXPECT_EQ(m.header.stamp.nanosec, 999999999u);
    EXPECT_EQ(m.orientation.x, 0.1);
    EXPECT_EQ(m.orientation.w, 0.9);
    EXPECT_EQ(m.orientation_covariance[0], -1.0);
    EXPECT_EQ(m.angular_velocity.z, 0.44);
    EXPECT_EQ(m.angular_velocity_covariance[8], 8.0);
    EXPECT_EQ(m.linear_acceleration.z, 9.80665);
    EXPECT_EQ(m.linear_acceleration_covariance[3], 1.5);
}

TEST(RosCdr, PoseStamped) {
    const PoseStamped m = Decode(DecodePoseStamped, fx::kPoseStamped);
    EXPECT_EQ(m.header.frame_id, "map");
    EXPECT_EQ(m.position.x, 1.5);
    EXPECT_EQ(m.position.y, -2.25);
    EXPECT_EQ(m.orientation.z, 0.70710678);
}

TEST(RosCdr, TfMessageWithTwoTransforms) {
    const auto v = Decode(DecodeTfMessage, fx::kTf);
    ASSERT_EQ(v.size(), 2u);
    EXPECT_EQ(v[0].header.stamp.sec, 3);
    EXPECT_EQ(v[0].header.frame_id, "base");
    EXPECT_EQ(v[0].child_frame_id, "left_optical");
    EXPECT_EQ(v[0].translation.y, 0.05);
    EXPECT_EQ(v[1].child_frame_id, "rgt");     // 4 bytes incl. NUL: padding before the doubles
    EXPECT_EQ(v[1].translation.y, -0.05);
    EXPECT_EQ(v[1].rotation.w, 0.5);
    EXPECT_EQ(v[1].rotation.x, -0.5);
}

TEST(RosCdr, Clock) {
    const Time t = Decode(DecodeClock, fx::kClock);
    EXPECT_EQ(t.sec, 42);
    EXPECT_EQ(t.nanosec, 7u);
}

TEST(RosCdr, BigEndianEncapsulationIsHonoured) {
    const auto be = BigEndianClock(0x01020304, 999);
    const Time t = DecodeClock(be.data(), be.size());
    EXPECT_EQ(t.sec, 0x01020304);
    EXPECT_EQ(t.nanosec, 999u);
}

TEST(RosCdr, EveryTruncationThrows) {
    EveryTruncationThrows(DecodeImage, fx::kImageMono);
    EveryTruncationThrows(DecodeCameraInfo, fx::kCameraInfo);
    EveryTruncationThrows(DecodeImu, fx::kImu);
    EveryTruncationThrows(DecodePoseStamped, fx::kPoseStamped);
    EveryTruncationThrows(DecodeTfMessage, fx::kTf);
    EveryTruncationThrows(DecodeClock, fx::kClock);
}

TEST(RosCdr, CorruptionIsRejected) {
    // Unknown encapsulation kind (0x0002 = PL_CDR_BE).
    auto b = fx::kClock;
    b[1] = 0x02;
    EXPECT_THROW(DecodeClock(b.data(), b.size()), CdrError);
    // An image data length far beyond the message must not allocate, just throw.
    auto img = fx::kImageMono;
    const size_t len_at = img.size() - 15 - 4;         // the uint32 before the 15 data bytes
    img[len_at + 3] = 0x7f;
    EXPECT_THROW(DecodeImage(img.data(), img.size()), CdrError);
    // do_rectify is the CameraInfo's last byte: 2 is not a boolean.
    auto ci = fx::kCameraInfo;
    ci.back() = 2;
    EXPECT_THROW(DecodeCameraInfo(ci.data(), ci.size()), CdrError);
    // Trailing garbage (>= 8 bytes) is not silently ignored.
    auto clk = fx::kClock;
    clk.insert(clk.end(), 8, 0);
    EXPECT_THROW(DecodeClock(clk.data(), clk.size()), CdrError);
    // Inconsistent image geometry: step * height != data size.
    auto bad = fx::kImageMono;
    bad[4 + 8 + 4 + 2 + 2 + 0] = 4;                   // height 3 -> 4 (after stamp, "a\0", pad)
    EXPECT_THROW(DecodeImage(bad.data(), bad.size()), CdrError);
}

TEST(RosCdr, BytesPerPixel) {
    EXPECT_EQ(BytesPerPixel("rgb8"), 3u);
    EXPECT_EQ(BytesPerPixel("32FC1"), 4u);
    EXPECT_THROW(BytesPerPixel("yuv422"), CdrError);
}

// --- the encoder (ros_cdr_writer.hpp, D39): exact bytes, zero padding ----------
#include "runtime/replay/ros_cdr_writer.hpp"

#include "../fixtures/ros_cdr/ros_cdr_live_samples.inc"

#include <cmath>

namespace live = ros_cdr_live;

TEST(RosCdrWriter, ReproducesRclpyBytesExactly) {
    // gen_fixtures.py frame ids ("imu", "map") need no padding: byte-exact vs rclpy.
    EXPECT_EQ(EncodeImu(Decode(DecodeImu, fx::kImu)), fx::kImu);
    EXPECT_EQ(EncodePoseStamped(Decode(DecodePoseStamped, fx::kPoseStamped)), fx::kPoseStamped);
}

TEST(RosCdrWriter, ReproducesLiveBridgeBytesIncludingPadding) {
    // Captured from a recording: "agent_0/base_link" + NUL = 18 bytes ends at
    // payload offset 30, so bytes 34..35 of the message are alignment padding.
    for (const auto* b : {&live::kLiveImuFirst, &live::kLiveImuMoving}) {
        const Imu m = Decode(DecodeImu, *b);
        EXPECT_EQ(m.header.frame_id, "agent_0/base_link");
        ASSERT_EQ(b->size(), 332u);
        EXPECT_EQ((*b)[34], 0);
        EXPECT_EQ((*b)[35], 0);
        EXPECT_EQ(EncodeImu(m), *b);
    }
    const PoseStamped p = Decode(DecodePoseStamped, live::kLivePoseMoving);
    ASSERT_EQ(live::kLivePoseMoving.size(), 92u);
    EXPECT_EQ(EncodePoseStamped(p), live::kLivePoseMoving);
    // The first IMU sample at rest has gyro x = -hd*sin(pitch) = -0.0: the sign bit
    // survives decode and encode (bytes equal above), and it is really negative zero.
    const Imu first = Decode(DecodeImu, live::kLiveImuFirst);
    EXPECT_EQ(first.angular_velocity.x, 0.0);
    EXPECT_TRUE(std::signbit(first.angular_velocity.x));
    EXPECT_EQ(first.linear_acceleration.z, 9.80665);
}

TEST(RosCdrWriter, PaddingIsZeroForEveryFrameIdLengthAndRoundTripsBitExact) {
    for (size_t len = 0; len <= 17; ++len) {
        Imu m;
        m.header.stamp = {123, 456789};
        m.header.frame_id = std::string(len, 'f');
        m.orientation = {0.0, -0.0, 1e-300, 1.0};
        m.angular_velocity = {-0.0, 0.1, -2.5};
        m.linear_acceleration = {std::nextafter(9.80665, 10.0), 0.0, -1e-17};
        m.angular_velocity_covariance[4] = -0.0;
        const auto b = EncodeImu(m);
        const size_t str_end = 4 + 8 + 4 + len + 1;                     // absolute offset after the NUL
        const size_t pad = (8 - (str_end - 4) % 8) % 8;
        ASSERT_EQ(b.size(), str_end + pad + 37 * 8) << "len " << len;
        for (size_t i = str_end; i < str_end + pad; ++i) EXPECT_EQ(b[i], 0) << "pad byte " << i << " len " << len;
        EXPECT_EQ(EncodeImu(m), b);                                      // deterministic
        const Imu d = DecodeImu(b.data(), b.size());
        EXPECT_EQ(EncodeImu(d), b);                                      // decode->encode is the identity
        EXPECT_EQ(std::memcmp(&d.angular_velocity.x, &m.angular_velocity.x, 8), 0);   // -0.0 kept
        EXPECT_EQ(d.linear_acceleration.x, m.linear_acceleration.x);

        PoseStamped p;
        p.header = m.header;
        p.position = {1.0 / 3.0, -0.0, 7.0};
        p.orientation = {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)};
        const auto pb = EncodePoseStamped(p);
        ASSERT_EQ(pb.size(), str_end + pad + 7 * 8);
        for (size_t i = str_end; i < str_end + pad; ++i) EXPECT_EQ(pb[i], 0);
        EXPECT_EQ(EncodePoseStamped(DecodePoseStamped(pb.data(), pb.size())), pb);
    }
}

TEST(RosCdrWriter, RejectsEmbeddedNul) {
    Imu m;
    m.header.frame_id = std::string("a\0b", 3);
    EXPECT_THROW(EncodeImu(m), CdrError);
}

// --- a real rosbag2 MCAP (zstd chunks), written by rosbag2_py ----------------
#include "runtime/replay/rosbag_reader.hpp"

#include <map>

TEST(RosbagReader, ReadsEveryTopicInLogTimeOrderAndDecodes) {
    ariadne::runtime::RosbagReader bag;
    bag.Open(std::string(ARIADNE_SOURCE_DIR) + "/tests/fixtures/ros_cdr/tiny_bag/tiny_bag_0.mcap");
    const auto& topics = bag.topics();
    ASSERT_EQ(topics.size(), 6u);
    EXPECT_EQ(topics.at("/ariadne/agent_0/left/image_raw"), "sensor_msgs/msg/Image");
    EXPECT_EQ(topics.at("/tf_static"), "tf2_msgs/msg/TFMessage");
    EXPECT_EQ(bag.message_count(), 18u);

    std::map<std::string, int> seen;
    uint64_t last = 0;
    const size_t n = bag.Replay([&](const ariadne::runtime::RosMessage& m) {
        EXPECT_GE(m.log_time_ns, last);
        last = m.log_time_ns;
        ++seen[std::string(m.type)];
        if (m.type == "sensor_msgs/msg/Image") {
            EXPECT_EQ(DecodeImage(m.data, m.size).encoding, "mono8");
        } else if (m.type == "sensor_msgs/msg/CameraInfo") {
            EXPECT_EQ(DecodeCameraInfo(m.data, m.size).width, 1280u);
        } else if (m.type == "sensor_msgs/msg/Imu") {
            EXPECT_EQ(DecodeImu(m.data, m.size).linear_acceleration.z, 9.80665);
        } else if (m.type == "geometry_msgs/msg/PoseStamped") {
            EXPECT_EQ(DecodePoseStamped(m.data, m.size).position.x, 1.5);
        } else if (m.type == "tf2_msgs/msg/TFMessage") {
            EXPECT_EQ(DecodeTfMessage(m.data, m.size).size(), 2u);
        } else if (m.type == "rosgraph_msgs/msg/Clock") {
            EXPECT_EQ(DecodeClock(m.data, m.size).sec, 42);
        } else {
            ADD_FAILURE() << "unexpected type " << m.type;
        }
    });
    EXPECT_EQ(n, 18u);
    for (const auto& [type, count] : seen) EXPECT_EQ(count, 3) << type;
}

TEST(RosbagReader, TopicFilterAndMissingFile) {
    ariadne::runtime::RosbagReader bag;
    bag.Open(std::string(ARIADNE_SOURCE_DIR) + "/tests/fixtures/ros_cdr/tiny_bag/tiny_bag_0.mcap");
    const size_t n = bag.Replay([](const ariadne::runtime::RosMessage& m) { EXPECT_EQ(m.topic, "/clock"); },
                                {"/clock"});
    EXPECT_EQ(n, 3u);
    ariadne::runtime::RosbagReader missing;
    EXPECT_THROW(missing.Open("/nonexistent.mcap"), std::runtime_error);
}

// --- a canonical-stream bag (D39), written by ariadne_canonical_streams ---------
// tests/fixtures/ros_cdr/canonical_1s/: ci_tiny seed 0, v1 rig, 1 s. Regenerate with
//   build/sim-plugins/ariadne_canonical_streams --world <scene>/mission_default.sdf \
//     --end-ns 1000000000 --out tests/fixtures/ros_cdr/canonical_1s/canonical
// (sim image, ROS sourced). The values are pinned by python/tests/sim; this test
// pins what the ROS-free replay path sees: topics, grid, log time == stamp, and
// stored bytes that re-encode identically (so their CDR padding is zero).
TEST(RosbagReader, ReadsTheCanonicalStreamBagRosFree) {
    ariadne::runtime::RosbagReader bag;
    bag.Open(std::string(ARIADNE_SOURCE_DIR) + "/tests/fixtures/ros_cdr/canonical_1s/canonical/canonical_0.mcap");
    const std::map<std::string, std::string> want = {
        {"/ariadne/agent_0/imu", "sensor_msgs/msg/Imu"},
        {"/ariadne/agent_0/gt/pose", "geometry_msgs/msg/PoseStamped"},
        {"/ariadne/agent_1/imu", "sensor_msgs/msg/Imu"},
        {"/ariadne/agent_1/gt/pose", "geometry_msgs/msg/PoseStamped"}};
    EXPECT_EQ(bag.topics(), want);
    constexpr int64_t kPeriod = 5000000, kSamples = 201;   // k = 0..200 over 1 s
    EXPECT_EQ(bag.message_count(), uint64_t(4 * kSamples));

    std::map<std::string, int64_t> next;                   // next expected stamp per topic
    uint64_t last_log = 0;
    const size_t n = bag.Replay([&](const ariadne::runtime::RosMessage& m) {
        EXPECT_GE(m.log_time_ns, last_log);
        last_log = m.log_time_ns;
        const std::string topic(m.topic);
        const std::string frame = topic.substr(9, 7) + "/base_link";   // "/ariadne/agent_k/..."
        const std::vector<uint8_t> bytes(m.data, m.data + m.size);
        Header h;
        if (m.type == "sensor_msgs/msg/Imu") {
            const Imu d = DecodeImu(m.data, m.size);
            h = d.header;
            EXPECT_EQ(EncodeImu(d), bytes);
            EXPECT_EQ(d.orientation.w, 1.0);                // identity: NOT measured
        } else {
            const PoseStamped d = DecodePoseStamped(m.data, m.size);
            h = d.header;
            EXPECT_EQ(EncodePoseStamped(d), bytes);
            const double qn = d.orientation.w * d.orientation.w + d.orientation.x * d.orientation.x +
                              d.orientation.y * d.orientation.y + d.orientation.z * d.orientation.z;
            EXPECT_NEAR(qn, 1.0, 1e-15);
        }
        EXPECT_EQ(h.frame_id, frame);
        EXPECT_EQ(h.stamp.ns(), next[topic]) << topic;
        EXPECT_EQ(uint64_t(h.stamp.ns()), m.log_time_ns);
        EXPECT_EQ(m.publish_time_ns, m.log_time_ns);
        next[topic] += kPeriod;
    });
    EXPECT_EQ(n, size_t(4 * kSamples));
    for (const auto& [topic, stamp] : next) EXPECT_EQ(stamp, kSamples * kPeriod) << topic;
}
