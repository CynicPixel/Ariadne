#pragma once
// ---------------------------------------------------------------------------
// The pass-A keyframe store of the 3.2 sim integration (design collaboration.md#simulated-collaboration §1, D171).
//
// Pass A runs one agent's shipped front-end over its recording and writes one record per keyframe whose odometry
// became FINAL (it left the BA window: the LoopBackend mirror took it). Pass B releases each record whole at
// `t_final_ns` into CollabNode::AddKeyframe, so nothing reaches the protocol before the agent would know it.
//
// Per record (data in -> data out):
//   id, t_create (the keyframe's frame time), t_final (the frame time its odometry became final)
//   T_W_C        rectified-left camera in W_agent at t_final (the mirror's pose: final odometry composed)
//   odometry     the final edge into it from the previous released keyframe (none for the first), Z + information
//   global       EigenPlaces R50-256 (float; pass B encodes it int8 + per-vector scale on the wire, D93)
//   aliked       ALIKED-n16 keypoints (x y), descriptors (K x 128 float, unit rows), 3D points in the camera frame
//                (NaN = no usable depth), from the keyframe's rectified RGB and masked SGBM depth (D63, D90)
//   camera       fx fy cx cy baseline of the rectified rig
//   gt_T_W_B     ground-truth body pose at t_create (EVALUATION ONLY: pass B scores with it, never decides)
//   T_B_C        body <- rectified-left camera (to score body positions)
//
// Format: little-endian binary, magic "ARCS", version, record count, then fixed-order fields; vectors are
// length-prefixed. The file's sha256 is the pass-A cache key recorded in every pass-B manifest.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::runtime {

struct CollabStoreRecord {
    uint64_t id = 0;
    int64_t t_create_ns = 0, t_final_ns = 0;
    core::SE3 T_W_C;
    std::optional<uint64_t> odo_from;       // previous released keyframe
    core::SE3 odo_Z;                        // T_from_to (left-multiplicative convention, as OdometryEdge)
    std::array<double, 36> odo_information{};
    std::vector<float> global;              // 256
    std::vector<float> keypoints;           // x y per feature
    std::vector<float> descriptors;         // 128 per feature
    std::vector<float> points3d;            // x y z per feature, NaN = none
    std::array<float, 5> camera{};          // fx fy cx cy baseline
    core::SE3 gt_T_W_B;
    bool has_gt = false;
    core::SE3 T_B_C;
};

struct CollabStore {
    uint32_t agent = 0;
    uint64_t prefix_unreleased = 0;   // VO keyframes before the mirror's anchor: never released (D87 carry)
    std::vector<CollabStoreRecord> records;
};

namespace detail {
template <class T> void Put(std::ofstream& f, const T& v) { f.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> void Get(std::ifstream& f, T& v) {
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("collab store: truncated");
}
inline void PutSE3(std::ofstream& f, const core::SE3& T) {   // R row-major then t: exact round trip
    double v[12];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) v[3 * r + c] = T.R(r, c);
    for (int i = 0; i < 3; ++i) v[9 + i] = T.t(i);
    f.write(reinterpret_cast<const char*>(v), sizeof(v));
}
inline core::SE3 GetSE3(std::ifstream& f) {
    double v[12];
    f.read(reinterpret_cast<char*>(v), sizeof(v));
    if (!f) throw std::runtime_error("collab store: truncated");
    core::SE3 T;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) T.R(r, c) = v[3 * r + c];
    T.t = core::Vec3(v[9], v[10], v[11]);
    return T;
}
inline void PutVec(std::ofstream& f, const std::vector<float>& v) {
    Put(f, uint64_t(v.size()));
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}
inline std::vector<float> GetVec(std::ifstream& f) {
    uint64_t n = 0;
    Get(f, n);
    if (n > (uint64_t(1) << 28)) throw std::runtime_error("collab store: implausible vector length");
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(float)));
    if (!f) throw std::runtime_error("collab store: truncated");
    return v;
}
constexpr char kMagic[4] = {'A', 'R', 'C', 'S'};
constexpr uint32_t kVersion = 1;
}  // namespace detail

inline void WriteCollabStore(const std::string& path, const CollabStore& s) {
    using namespace detail;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("collab store: cannot write " + path);
    f.write(kMagic, 4);
    Put(f, kVersion);
    Put(f, s.agent);
    Put(f, s.prefix_unreleased);
    Put(f, uint64_t(s.records.size()));
    for (const auto& r : s.records) {
        Put(f, r.id);
        Put(f, r.t_create_ns);
        Put(f, r.t_final_ns);
        PutSE3(f, r.T_W_C);
        Put(f, uint8_t(r.odo_from.has_value()));
        Put(f, r.odo_from.value_or(0));
        PutSE3(f, r.odo_Z);
        f.write(reinterpret_cast<const char*>(r.odo_information.data()), sizeof(r.odo_information));
        PutVec(f, r.global);
        PutVec(f, r.keypoints);
        PutVec(f, r.descriptors);
        PutVec(f, r.points3d);
        f.write(reinterpret_cast<const char*>(r.camera.data()), sizeof(r.camera));
        Put(f, uint8_t(r.has_gt));
        PutSE3(f, r.gt_T_W_B);
        PutSE3(f, r.T_B_C);
    }
    if (!f) throw std::runtime_error("collab store: write failed " + path);
}

inline CollabStore ReadCollabStore(const std::string& path) {
    using namespace detail;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("collab store: cannot read " + path);
    char m[4];
    f.read(m, 4);
    uint32_t ver = 0;
    Get(f, ver);
    if (!f || std::memcmp(m, kMagic, 4) != 0 || ver != kVersion) throw std::runtime_error("collab store: bad header");
    CollabStore s;
    Get(f, s.agent);
    Get(f, s.prefix_unreleased);
    uint64_t n = 0;
    Get(f, n);
    s.records.resize(n);
    for (auto& r : s.records) {
        Get(f, r.id);
        Get(f, r.t_create_ns);
        Get(f, r.t_final_ns);
        r.T_W_C = GetSE3(f);
        uint8_t has = 0;
        uint64_t from = 0;
        Get(f, has);
        Get(f, from);
        if (has) r.odo_from = from;
        r.odo_Z = GetSE3(f);
        f.read(reinterpret_cast<char*>(r.odo_information.data()), sizeof(r.odo_information));
        r.global = GetVec(f);
        r.keypoints = GetVec(f);
        r.descriptors = GetVec(f);
        r.points3d = GetVec(f);
        f.read(reinterpret_cast<char*>(r.camera.data()), sizeof(r.camera));
        uint8_t g = 0;
        Get(f, g);
        r.has_gt = g != 0;
        r.gt_T_W_B = GetSE3(f);
        r.T_B_C = GetSE3(f);
    }
    return s;
}

}  // namespace ariadne::runtime
