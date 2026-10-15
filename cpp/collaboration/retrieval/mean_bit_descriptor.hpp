#pragma once
// ---------------------------------------------------------------------------
// 1.6 placeholder global descriptor (plan 1.6: "mean-pooled patch vector"): the
// per-bit mean of a keyframe's 256-bit ORB descriptors, centred at 0.5 and
// L2-normalized, compared by cosine similarity. Cheap and weak by design; the
// real retrieval (EigenPlaces, §13) replaces it in Phase 3 behind the same
// GLOBAL_DESCRIPTOR message. OpenCV-free: takes the raw 32-byte descriptors.
// ---------------------------------------------------------------------------
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ariadne::collaboration {

inline constexpr size_t kOrbBytes = 32;
inline constexpr size_t kMeanBitDims = 256;

inline std::vector<float> MeanBitDescriptor(const std::vector<uint8_t>& descriptors) {
    if (descriptors.size() % kOrbBytes) throw std::invalid_argument("mean-bit: descriptor bytes not a multiple of 32");
    const size_t n = descriptors.size() / kOrbBytes;
    std::vector<double> acc(kMeanBitDims, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t b = 0; b < kOrbBytes; ++b) {
            const uint8_t v = descriptors[i * kOrbBytes + b];
            for (int k = 0; k < 8; ++k) acc[b * 8 + size_t(k)] += (v >> k) & 1;
        }
    }
    std::vector<float> out(kMeanBitDims, 0.0f);
    if (n == 0) return out;
    double norm = 0.0;
    for (size_t d = 0; d < kMeanBitDims; ++d) {
        acc[d] = acc[d] / double(n) - 0.5;   // centre: a bit set half the time carries no information
        norm += acc[d] * acc[d];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (size_t d = 0; d < kMeanBitDims; ++d) out[d] = float(acc[d] / norm);
    }
    return out;
}

inline double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("cosine: size mismatch");
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += double(a[i]) * double(b[i]);
    return s;   // both unit vectors
}

}  // namespace ariadne::collaboration
