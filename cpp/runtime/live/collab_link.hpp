#pragma once
// ---------------------------------------------------------------------------
// The live collaboration's messages (design live-collaboration.md §2, D174) between three processes per agent k:
//   estimator worker k  -> ariadne_collab_worker k   #hello, #ckin (a keyframe at creation), #crel (its release)
//   ariadne_collab_worker k -> parent                #ckf (a released keyframe's record), #fresp, #cwm (watermark)
//   parent -> ariadne_collab_worker k                #hello, #freq (local features of a keyframe, please)
// Framing is LinkFrame (runtime/live/link_frame.hpp); the payloads below are little-endian PODs, cv::Mats as
// (rows, cols, type, bytes). Worker and parent are the same build, so there is no versioning beyond the topic.
//
// #ckin carries the bundle's CV_16F depth and CV_8U mask, not MaskedDepth32: half the bytes, and the collab worker
// computes MaskedDepth32 itself, exactly as pass A does from the same bundle.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "runtime/agent/collab_frontend.hpp"
#include "runtime/agent/collab_store.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::runtime::live {

struct CollabHello {
    std::string role;   // "estimator" | "parent"
    uint32_t agent = 0;
};

struct CollabKeyframeIn {   // #ckin
    uint64_t id = 0;
    int64_t t_ns = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0, baseline_m = 0;
    core::SE3 T_B_C;
    std::optional<core::SE3> gt_T_W_B;   // EVALUATION ONLY
    cv::Mat rgb, depth16, mask;
};

struct CollabFeatureReply {   // #fresp
    uint64_t kf = 0;
    std::optional<CollabLocalFeatures> features;   // nullopt = gone (expired or never released)
};

namespace collab_wire {

class Writer {
public:
    template <typename T> void Pod(const T& v) {
        const auto* q = reinterpret_cast<const uint8_t*>(&v);
        b.insert(b.end(), q, q + sizeof v);
    }
    void Se3(const core::SE3& T) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) Pod(double(T.R(i, j)));
        for (int i = 0; i < 3; ++i) Pod(double(T.t(i)));
    }
    void Floats(const std::vector<float>& v) {
        Pod(uint64_t(v.size()));
        const auto* q = reinterpret_cast<const uint8_t*>(v.data());
        b.insert(b.end(), q, q + v.size() * sizeof(float));
    }
    void Mat(const cv::Mat& m0) {
        const cv::Mat m = m0.isContinuous() ? m0 : m0.clone();
        Pod(int32_t(m.rows)), Pod(int32_t(m.cols)), Pod(int32_t(m.type()));
        const auto* q = m.ptr<uint8_t>();
        b.insert(b.end(), q, q + m.total() * m.elemSize());
    }
    void Str(const std::string& s) {
        Pod(uint32_t(s.size()));
        b.insert(b.end(), s.begin(), s.end());
    }
    std::vector<uint8_t> b;
};

class Reader {
public:
    Reader(const std::vector<uint8_t>& v) : p(v.data()), e(v.data() + v.size()) {}
    template <typename T> T Pod() {
        Need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof v);
        p += sizeof v;
        return v;
    }
    core::SE3 Se3() {
        core::SE3 T;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) T.R(i, j) = Pod<double>();
        for (int i = 0; i < 3; ++i) T.t(i) = Pod<double>();
        return T;
    }
    std::vector<float> Floats() {
        const uint64_t n = Pod<uint64_t>();
        if (n > (uint64_t(1) << 26)) throw std::runtime_error("collab link: implausible vector length");
        Need(n * sizeof(float));
        std::vector<float> v(n);
        std::memcpy(v.data(), p, n * sizeof(float));
        p += n * sizeof(float);
        return v;
    }
    cv::Mat Mat() {
        const int rows = Pod<int32_t>(), cols = Pod<int32_t>(), type = Pod<int32_t>();
        if (rows < 0 || cols < 0 || rows > 8192 || cols > 8192) throw std::runtime_error("collab link: bad Mat shape");
        cv::Mat m(rows, cols, type);
        Need(m.total() * m.elemSize());
        if (m.total()) std::memcpy(m.ptr<uint8_t>(), p, m.total() * m.elemSize());
        p += m.total() * m.elemSize();
        return m;
    }
    std::string Str() {
        const uint32_t n = Pod<uint32_t>();
        Need(n);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    void End() const {
        if (p != e) throw std::runtime_error("collab link: trailing bytes");
    }

private:
    void Need(size_t n) const {
        if (size_t(e - p) < n) throw std::runtime_error("collab link: truncated message");
    }
    const uint8_t* p;
    const uint8_t* e;
};

}  // namespace collab_wire

// --- encoders / decoders, one pair per message -------------------------------------------------------------------

inline std::vector<uint8_t> EncodeHello(const CollabHello& h) {
    collab_wire::Writer w;
    w.Str(h.role), w.Pod(h.agent);
    return w.b;
}
inline CollabHello DecodeHello(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    CollabHello h;
    h.role = r.Str(), h.agent = r.Pod<uint32_t>();
    r.End();
    return h;
}

inline std::vector<uint8_t> EncodeKeyframeIn(const CollabKeyframeIn& k) {
    collab_wire::Writer w;
    w.Pod(k.id), w.Pod(k.t_ns), w.Pod(k.fx), w.Pod(k.fy), w.Pod(k.cx), w.Pod(k.cy), w.Pod(k.baseline_m);
    w.Se3(k.T_B_C);
    w.Pod(uint8_t(k.gt_T_W_B.has_value()));
    w.Se3(k.gt_T_W_B.value_or(core::SE3()));
    w.Mat(k.rgb), w.Mat(k.depth16), w.Mat(k.mask);
    return w.b;
}
inline CollabKeyframeIn DecodeKeyframeIn(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    CollabKeyframeIn k;
    k.id = r.Pod<uint64_t>(), k.t_ns = r.Pod<int64_t>();
    k.fx = r.Pod<double>(), k.fy = r.Pod<double>(), k.cx = r.Pod<double>(), k.cy = r.Pod<double>();
    k.baseline_m = r.Pod<double>();
    k.T_B_C = r.Se3();
    const bool has_gt = r.Pod<uint8_t>() != 0;
    const core::SE3 gt = r.Se3();
    if (has_gt) k.gt_T_W_B = gt;
    k.rgb = r.Mat(), k.depth16 = r.Mat(), k.mask = r.Mat();
    r.End();
    return k;
}

inline std::vector<uint8_t> EncodeRelease(const CollabRelease& c) {
    collab_wire::Writer w;
    w.Pod(c.id), w.Pod(c.t_final_ns);
    w.Se3(c.T_W_C);
    w.Pod(uint8_t(c.odo_from.has_value())), w.Pod(c.odo_from.value_or(0));
    w.Se3(c.odo_Z);
    for (double v : c.odo_information) w.Pod(v);
    return w.b;
}
inline CollabRelease DecodeRelease(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    CollabRelease c;
    c.id = r.Pod<uint64_t>(), c.t_final_ns = r.Pod<int64_t>();
    c.T_W_C = r.Se3();
    const bool has = r.Pod<uint8_t>() != 0;
    const uint64_t from = r.Pod<uint64_t>();
    if (has) c.odo_from = from;
    c.odo_Z = r.Se3();
    for (double& v : c.odo_information) v = r.Pod<double>();
    r.End();
    return c;
}

/// #ckf: a released keyframe's record WITHOUT local features, plus the feature count it advertises.
inline std::vector<uint8_t> EncodeRecord(const CollabStoreRecord& s, uint32_t n_features) {
    collab_wire::Writer w;
    w.Pod(s.id), w.Pod(s.t_create_ns), w.Pod(s.t_final_ns);
    w.Se3(s.T_W_C);
    w.Pod(uint8_t(s.odo_from.has_value())), w.Pod(s.odo_from.value_or(0));
    w.Se3(s.odo_Z);
    for (double v : s.odo_information) w.Pod(v);
    w.Floats(s.global);
    for (float v : s.camera) w.Pod(v);
    w.Pod(uint8_t(s.has_gt));
    w.Se3(s.gt_T_W_B);
    w.Se3(s.T_B_C);
    w.Pod(n_features);
    return w.b;
}
inline CollabStoreRecord DecodeRecord(const std::vector<uint8_t>& b, uint32_t* n_features) {
    collab_wire::Reader r(b);
    CollabStoreRecord s;
    s.id = r.Pod<uint64_t>(), s.t_create_ns = r.Pod<int64_t>(), s.t_final_ns = r.Pod<int64_t>();
    s.T_W_C = r.Se3();
    const bool has = r.Pod<uint8_t>() != 0;
    const uint64_t from = r.Pod<uint64_t>();
    if (has) s.odo_from = from;
    s.odo_Z = r.Se3();
    for (double& v : s.odo_information) v = r.Pod<double>();
    s.global = r.Floats();
    for (float& v : s.camera) v = r.Pod<float>();
    s.has_gt = r.Pod<uint8_t>() != 0;
    s.gt_T_W_B = r.Se3();
    s.T_B_C = r.Se3();
    const uint32_t n = r.Pod<uint32_t>();
    if (n_features) *n_features = n;
    r.End();
    return s;
}

/// #freq: the keyframe and the sim time of the asking tick (-1 = untimed: the legacy presence rule).
struct CollabFeatureRequest {
    uint64_t kf = 0;
    int64_t t_req_ns = -1;
};
inline std::vector<uint8_t> EncodeFeatureRequest(uint64_t kf, int64_t t_req_ns = -1) {
    collab_wire::Writer w;
    w.Pod(kf), w.Pod(t_req_ns);
    return w.b;
}
inline CollabFeatureRequest DecodeFeatureRequest(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    CollabFeatureRequest q;
    q.kf = r.Pod<uint64_t>(), q.t_req_ns = r.Pod<int64_t>();
    r.End();
    return q;
}

inline std::vector<uint8_t> EncodeFeatureReply(const CollabFeatureReply& f) {
    collab_wire::Writer w;
    w.Pod(f.kf), w.Pod(uint8_t(f.features.has_value()));
    if (f.features) w.Floats(f.features->keypoints), w.Floats(f.features->descriptors), w.Floats(f.features->points3d);
    return w.b;
}
inline CollabFeatureReply DecodeFeatureReply(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    CollabFeatureReply f;
    f.kf = r.Pod<uint64_t>();
    if (r.Pod<uint8_t>()) {
        CollabLocalFeatures l;
        l.keypoints = r.Floats(), l.descriptors = r.Floats(), l.points3d = r.Floats();
        f.features = std::move(l);
    }
    r.End();
    return f;
}

inline std::vector<uint8_t> EncodeWatermark(int64_t t_ns) {
    collab_wire::Writer w;
    w.Pod(t_ns);
    return w.b;
}
inline int64_t DecodeWatermark(const std::vector<uint8_t>& b) {
    collab_wire::Reader r(b);
    const int64_t t = r.Pod<int64_t>();
    r.End();
    return t;
}

}  // namespace ariadne::runtime::live
