#pragma once
// ---------------------------------------------------------------------------
// Local-descriptor encodings for the feature response (arch §15.7, D171): what a keyframe's descriptors become on
// the wire. OpenCV-free.
//
//   fp32   4 B per dimension (the extractor's own floats; reference only)
//   fp16   2 B per dimension (IEEE half; arch §15.7's ablation row)
//   int8   1 B per dimension plus ONE float32 scale per keyframe, prepended: q = round(127 * d / s), s = max |d| over
//          the keyframe (arch §15.7's default: "int8 with per-keyframe scale (ablated vs fp16)"). ALIKED descriptors
//          are unit rows, so s <= 1 and the step is s / 127.
//
// Encode takes K x dims row-major floats and returns the descriptor bytes of the FeatureResponseValue; Decode
// inverts it (lossy for fp16 / int8). The per-feature byte count the broker prices (`feature_bytes`) excludes the
// 4-byte int8 scale header, which the codec's exact payload pricing includes.
// ---------------------------------------------------------------------------
#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::collaboration {

enum class LocalEncoding { kFp32, kFp16, kInt8 };

inline LocalEncoding ParseLocalEncoding(const std::string& s) {
    if (s == "fp32") return LocalEncoding::kFp32;
    if (s == "fp16") return LocalEncoding::kFp16;
    if (s == "int8") return LocalEncoding::kInt8;
    throw std::invalid_argument("local descriptor encoding: fp32 | fp16 | int8, got " + s);
}
inline uint16_t BytesPerDim(LocalEncoding e) { return e == LocalEncoding::kFp32 ? 4 : e == LocalEncoding::kFp16 ? 2 : 1; }

inline std::vector<uint8_t> EncodeLocal(const std::vector<float>& d, LocalEncoding e) {
    std::vector<uint8_t> out;
    switch (e) {
        case LocalEncoding::kFp32:
            out.resize(d.size() * 4);
            std::memcpy(out.data(), d.data(), out.size());
            break;
        case LocalEncoding::kFp16:
            out.resize(d.size() * 2);
            for (size_t i = 0; i < d.size(); ++i) {
                const uint16_t b = Eigen::numext::bit_cast<uint16_t>(Eigen::half(d[i]));
                std::memcpy(&out[2 * i], &b, 2);
            }
            break;
        case LocalEncoding::kInt8: {
            float s = 0.0f;
            for (float x : d) s = std::max(s, std::abs(x));
            if (!(s > 0.0f)) s = 1.0f;
            out.resize(4 + d.size());
            std::memcpy(out.data(), &s, 4);
            for (size_t i = 0; i < d.size(); ++i)
                out[4 + i] = uint8_t(int8_t(std::clamp(std::lround(127.0f * d[i] / s), -127L, 127L)));
            break;
        }
    }
    return out;
}

inline std::vector<float> DecodeLocal(const std::vector<uint8_t>& b, LocalEncoding e) {
    std::vector<float> d;
    switch (e) {
        case LocalEncoding::kFp32:
            d.resize(b.size() / 4);
            std::memcpy(d.data(), b.data(), d.size() * 4);
            break;
        case LocalEncoding::kFp16:
            d.resize(b.size() / 2);
            for (size_t i = 0; i < d.size(); ++i) {
                uint16_t h;
                std::memcpy(&h, &b[2 * i], 2);
                d[i] = float(Eigen::numext::bit_cast<Eigen::half>(h));
            }
            break;
        case LocalEncoding::kInt8: {
            if (b.size() < 4) throw std::invalid_argument("DecodeLocal: int8 without its scale");
            float s;
            std::memcpy(&s, b.data(), 4);
            d.resize(b.size() - 4);
            for (size_t i = 0; i < d.size(); ++i) d[i] = s * float(int8_t(b[4 + i])) / 127.0f;
            break;
        }
    }
    return d;
}

}  // namespace ariadne::collaboration
