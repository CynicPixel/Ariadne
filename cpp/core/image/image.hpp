#pragma once
// ---------------------------------------------------------------------------
// Minimal raw image buffers.
//
// Deliberately not cv::Mat. The degradation operator is the determinism
// substrate everything downstream is compared against, so it must build and be
// testable without OpenCV present (arch §3 dependency rule). A thin cv::Mat
// adapter lives at the perception boundary, not here.
//
// No strides, no ROIs, no reference counting: tightly packed row-major buffers
// only. Every one of those features is a way for two runs to disagree about
// what "the same image" means.
// ---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ariadne::core {

/// 8-bit image, row-major, tightly packed, interleaved channels.
struct ImageU8 {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<uint8_t> data;

    ImageU8() = default;
    ImageU8(int w, int h, int c)
        : width(w), height(h), channels(c),
          data(static_cast<size_t>(w) * h * c, 0) {}

    size_t PixelCount() const {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    size_t Size() const { return PixelCount() * static_cast<size_t>(channels); }
    bool Empty() const { return width <= 0 || height <= 0 || data.empty(); }

    /// Index of channel `c` at (x, y). No bounds checking on the hot path;
    /// callers iterate over known extents.
    size_t Index(int x, int y, int c = 0) const {
        return (static_cast<size_t>(y) * width + x) * channels + c;
    }

    uint8_t& At(int x, int y, int c = 0) { return data[Index(x, y, c)]; }
    uint8_t At(int x, int y, int c = 0) const { return data[Index(x, y, c)]; }
};

/// Single-channel float image. Used for ground-truth depth in metres, which is
/// what the dust stage of §5.6 composites against.
struct ImageF32 {
    int width = 0;
    int height = 0;
    std::vector<float> data;

    ImageF32() = default;
    ImageF32(int w, int h)
        : width(w), height(h), data(static_cast<size_t>(w) * h, 0.0f) {}

    size_t PixelCount() const {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }
    bool Empty() const { return width <= 0 || height <= 0 || data.empty(); }

    size_t Index(int x, int y) const {
        return static_cast<size_t>(y) * width + x;
    }
    float& At(int x, int y) { return data[Index(x, y)]; }
    float At(int x, int y) const { return data[Index(x, y)]; }
};

/// True when `depth` can be composited against `image` pixel for pixel.
inline bool SameExtent(const ImageU8& image, const ImageF32& depth) {
    return image.width == depth.width && image.height == depth.height;
}

}  // namespace ariadne::core
