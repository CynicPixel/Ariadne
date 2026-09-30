#include "runtime/replay/ros_cdr.hpp"

namespace ariadne::runtime::ros_cdr {

CdrReader::CdrReader(const uint8_t* data, size_t size) : p_(data + kHeader), size_(size) {
    if (data == nullptr || size < kHeader) throw CdrError("CDR: message shorter than its header");
    if (data[0] != 0x00 || data[1] > 0x01) {
        // 0x0000/0x0001 = plain CDR BE/LE. Other kinds (PL_CDR, XCDR2) are never
        // written by rmw_fastrtps for these types; refusing is safer than guessing.
        throw CdrError("CDR: unsupported encapsulation kind");
    }
    const bool little = data[1] == 0x01;
    uint16_t probe = 1;
    uint8_t first;
    std::memcpy(&first, &probe, 1);
    const bool host_little = first == 1;
    swap_ = little != host_little;
}

bool CdrReader::ReadBool() {
    const uint8_t b = Read<uint8_t>();
    if (b > 1) throw CdrError("CDR: boolean byte is neither 0 nor 1");
    return b == 1;
}

uint32_t CdrReader::ReadSequenceLength(size_t min_element_bytes) {
    const uint32_t n = Read<uint32_t>();
    // Reject counts that cannot fit in what is left: a corrupt length must not
    // turn into a multi-gigabyte allocation.
    if (min_element_bytes > 0 && uint64_t(n) * min_element_bytes > remaining()) {
        throw CdrError("CDR: sequence length exceeds the message");
    }
    return n;
}

std::string CdrReader::ReadString() {
    const uint32_t n = ReadSequenceLength(1);
    if (n == 0) throw CdrError("CDR: string length 0 (must count the NUL)");
    const char* s = reinterpret_cast<const char*>(p_ + pos_);
    if (s[n - 1] != '\0') throw CdrError("CDR: string is not NUL-terminated");
    std::string out(s, n - 1);
    pos_ += n;
    return out;
}

std::vector<uint8_t> CdrReader::ReadByteSequence() {
    const uint32_t n = ReadSequenceLength(1);
    std::vector<uint8_t> out(p_ + pos_, p_ + pos_ + n);
    pos_ += n;
    return out;
}

std::vector<double> CdrReader::ReadFloat64Sequence() {
    const uint32_t n = ReadSequenceLength(8);
    std::vector<double> out(n);
    for (auto& v : out) v = Read<double>();
    return out;
}

void CdrReader::ExpectEnd() const {
    if (remaining() >= 8) throw CdrError("CDR: trailing bytes after the message");
}

namespace {

Time ReadTime(CdrReader& r) {
    Time t;
    t.sec = r.Read<int32_t>();
    t.nanosec = r.Read<uint32_t>();
    if (t.nanosec >= 1000000000u) throw CdrError("CDR: nanosec >= 1e9");
    return t;
}

Header ReadHeader(CdrReader& r) {
    Header h;
    h.stamp = ReadTime(r);
    h.frame_id = r.ReadString();
    return h;
}

Vector3 ReadVector3(CdrReader& r) {
    Vector3 v;
    v.x = r.Read<double>();
    v.y = r.Read<double>();
    v.z = r.Read<double>();
    return v;
}

Quaternion ReadQuaternion(CdrReader& r) {
    Quaternion q;
    q.x = r.Read<double>();
    q.y = r.Read<double>();
    q.z = r.Read<double>();
    q.w = r.Read<double>();
    return q;
}

}  // namespace

size_t BytesPerPixel(const std::string& e) {
    if (e == "mono8" || e == "8UC1") return 1;
    if (e == "rgb8" || e == "bgr8" || e == "8UC3") return 3;
    if (e == "mono16" || e == "16UC1") return 2;
    if (e == "32FC1") return 4;
    throw CdrError("image encoding not supported by Ariadne: " + e);
}

void ValidateImage(const Image& img) {
    const size_t bpp = BytesPerPixel(img.encoding);
    if (img.is_bigendian != 0 && bpp > 1) throw CdrError("big-endian multi-byte image data");
    if (uint64_t(img.step) < uint64_t(img.width) * bpp) throw CdrError("image step < width * bytes per pixel");
    if (uint64_t(img.data.size()) != uint64_t(img.step) * img.height) {
        throw CdrError("image data size != step * height");
    }
}

Image DecodeImage(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    Image m;
    m.header = ReadHeader(r);
    m.height = r.Read<uint32_t>();
    m.width = r.Read<uint32_t>();
    m.encoding = r.ReadString();
    m.is_bigendian = r.Read<uint8_t>();
    m.step = r.Read<uint32_t>();
    m.data = r.ReadByteSequence();
    r.ExpectEnd();
    ValidateImage(m);
    return m;
}

CameraInfo DecodeCameraInfo(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    CameraInfo m;
    m.header = ReadHeader(r);
    m.height = r.Read<uint32_t>();
    m.width = r.Read<uint32_t>();
    m.distortion_model = r.ReadString();
    m.d = r.ReadFloat64Sequence();
    m.k = r.ReadFloat64Array<9>();
    m.r = r.ReadFloat64Array<9>();
    m.p = r.ReadFloat64Array<12>();
    m.binning_x = r.Read<uint32_t>();
    m.binning_y = r.Read<uint32_t>();
    m.roi.x_offset = r.Read<uint32_t>();
    m.roi.y_offset = r.Read<uint32_t>();
    m.roi.height = r.Read<uint32_t>();
    m.roi.width = r.Read<uint32_t>();
    m.roi.do_rectify = r.ReadBool();
    r.ExpectEnd();
    return m;
}

Imu DecodeImu(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    Imu m;
    m.header = ReadHeader(r);
    m.orientation = ReadQuaternion(r);
    m.orientation_covariance = r.ReadFloat64Array<9>();
    m.angular_velocity = ReadVector3(r);
    m.angular_velocity_covariance = r.ReadFloat64Array<9>();
    m.linear_acceleration = ReadVector3(r);
    m.linear_acceleration_covariance = r.ReadFloat64Array<9>();
    r.ExpectEnd();
    return m;
}

PoseStamped DecodePoseStamped(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    PoseStamped m;
    m.header = ReadHeader(r);
    m.position = ReadVector3(r);
    m.orientation = ReadQuaternion(r);
    r.ExpectEnd();
    return m;
}

std::vector<TransformStamped> DecodeTfMessage(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    // Smallest TransformStamped: 8 (time) + 5 (empty string) + 5 + 56 doubles > 64 bytes.
    const uint32_t n = r.ReadSequenceLength(64);
    std::vector<TransformStamped> out(n);
    for (auto& t : out) {
        t.header = ReadHeader(r);
        t.child_frame_id = r.ReadString();
        t.translation = ReadVector3(r);
        t.rotation = ReadQuaternion(r);
    }
    r.ExpectEnd();
    return out;
}

Time DecodeClock(const uint8_t* data, size_t size) {
    CdrReader r(data, size);
    const Time t = ReadTime(r);
    r.ExpectEnd();
    return t;
}

}  // namespace ariadne::runtime::ros_cdr
