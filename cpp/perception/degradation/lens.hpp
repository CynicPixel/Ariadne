#pragma once
// ---------------------------------------------------------------------------
// The read-time LENS stage (DECISIONS D41, design SIM_1.1-1.3 §5): turns the
// simulator's clean pinhole render into the image a Brown-Conrady camera would
// have captured, before the sensor stages of Degrade() (optics, then sensor).
//
// Why at read time: gz Harmonic's ogre2 renderer ignores SDF lens distortion
// (SMOKE §4n), and degradation-at-read is the substrate rule anyway (D37): the
// recording stays a clean pinhole, every consumer sees identical distorted pixels,
// and the coefficients are known exactly, so the undistort path is tested
// against truth.
//
// THE MAPPING. Output pixel (u, v) of the real camera, intrinsics (fx, fy, cx, cy)
// in the pixel-centre convention (pixel i covers [i-0.5, i+0.5]), sees the
// normalized distorted point x_d = ((u-cx)/fx, (v-cy)/fy). Brown-Conrady maps an
// undistorted ray (x, y), r^2 = x^2 + y^2, to
//   x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
//   y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y
// (OpenCV's model and coefficient order k1 k2 p1 p2 k3). We invert it per pixel by
// the fixed-point iteration OpenCV's undistortPoints uses, then sample the clean
// render at (render_fx x + render_cx, render_fy y + render_cy), bilinearly.
// With barrel distortion (k1 < 0) the inverted ray lies FURTHER out than x_d, so
// the output focal length must exceed the render's for every output pixel to
// land inside the render: BuildLensMap checks that and throws otherwise (no
// black corners, no silent clamping).
//
// DETERMINISM: only +, -, *, / and floor in IEEE double with -ffp-contract=off
// (top-level CMake), a fixed iteration count, and round-half-up to 8 bits: the
// same bytes on every platform, like the rest of this directory. No OpenCV here
// (it is a test oracle in tests/unit/test_lens.cpp).
// ---------------------------------------------------------------------------
#include "core/image/image.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ariadne::perception {

/// The camera the estimator sees, and the pinhole render it is synthesized from.
struct LensModel {
    int width = 0, height = 0;                       // output = render size
    double fx = 0, fy = 0, cx = 0, cy = 0;           // output intrinsics, pixel-centre convention
    double k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0;   // Brown-Conrady, OpenCV order
    double render_fx = 0, render_fy = 0, render_cx = 0, render_cy = 0;   // clean render, pixel-centre
};

/// Per output pixel, where to sample the clean render (row-major).
struct LensMap {
    int width = 0, height = 0;
    std::vector<double> src_u, src_v;
    double max_inverse_residual = 0;   // worst |distort(undistort(x_d)) - x_d|, normalized units
    double min_margin_px = 0;          // closest approach of a sample point to the render's border
};

inline constexpr int kLensInverseIterations = 30;
inline constexpr double kLensInverseTolerance = 1e-12;

/// Forward Brown-Conrady on a normalized undistorted point.
inline void LensDistort(const LensModel& m, double x, double y, double& xd, double& yd) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + r2 * (m.k1 + r2 * (m.k2 + r2 * m.k3));
    xd = x * radial + 2.0 * m.p1 * x * y + m.p2 * (r2 + 2.0 * x * x);
    yd = y * radial + m.p1 * (r2 + 2.0 * y * y) + 2.0 * m.p2 * x * y;
}

/// Inverse by fixed-point iteration (OpenCV undistortPoints' scheme), fixed count.
inline void LensUndistort(const LensModel& m, double xd, double yd, double& x, double& y) {
    x = xd;
    y = yd;
    for (int i = 0; i < kLensInverseIterations; ++i) {
        const double r2 = x * x + y * y;
        const double radial = 1.0 + r2 * (m.k1 + r2 * (m.k2 + r2 * m.k3));
        const double dx = 2.0 * m.p1 * x * y + m.p2 * (r2 + 2.0 * x * x);
        const double dy = m.p1 * (r2 + 2.0 * y * y) + 2.0 * m.p2 * x * y;
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }
}

/// Builds the sampling map; throws if the model is malformed, the inverse did not
/// converge, or any output pixel would sample outside the render.
inline LensMap BuildLensMap(const LensModel& m) {
    if (m.width <= 1 || m.height <= 1) throw std::invalid_argument("lens: bad image size");
    if (!(m.fx > 0 && m.fy > 0 && m.render_fx > 0 && m.render_fy > 0)) {
        throw std::invalid_argument("lens: focal lengths must be > 0");
    }
    LensMap map;
    map.width = m.width;
    map.height = m.height;
    const size_t n = size_t(m.width) * size_t(m.height);
    map.src_u.resize(n);
    map.src_v.resize(n);
    map.min_margin_px = 1e300;
    const double umax = m.width - 1, vmax = m.height - 1;
    for (int v = 0; v < m.height; ++v) {
        for (int u = 0; u < m.width; ++u) {
            const double xd = (u - m.cx) / m.fx, yd = (v - m.cy) / m.fy;
            double x, y, xr, yr;
            LensUndistort(m, xd, yd, x, y);
            LensDistort(m, x, y, xr, yr);
            const double res = std::max(std::abs(xr - xd), std::abs(yr - yd));
            if (!(res <= kLensInverseTolerance)) {   // also catches NaN
                throw std::runtime_error("lens: distortion inverse did not converge at pixel (" + std::to_string(u) +
                                         ", " + std::to_string(v) + "), residual " + std::to_string(res));
            }
            map.max_inverse_residual = std::max(map.max_inverse_residual, res);
            const double su = m.render_fx * x + m.render_cx, sv = m.render_fy * y + m.render_cy;
            const double margin = std::min(std::min(su, umax - su), std::min(sv, vmax - sv));
            if (margin < 0) {
                throw std::runtime_error("lens: output pixel (" + std::to_string(u) + ", " + std::to_string(v) +
                                         ") samples outside the render at (" + std::to_string(su) + ", " +
                                         std::to_string(sv) + "); raise the output focal length");
            }
            map.min_margin_px = std::min(map.min_margin_px, margin);
            const size_t i = size_t(v) * size_t(m.width) + size_t(u);
            map.src_u[i] = su;
            map.src_v[i] = sv;
        }
    }
    return map;
}

/// Bilinear resampling of every channel; round half up to 8 bits.
/// Rows [v0, v1) of the resample: pure scalar arithmetic, no OpenCV (this header is the determinism substrate
/// and the manifest/determinism include path - arch §3 dependency rule; the Kaggle/AWS test_unit build broke when
/// D226 put cv::parallel_for_ here). lens_parallel.hpp wraps this span under cv::parallel_for_ for OpenCV-linked
/// consumers; every output pixel is an independent pure function, so any row partition is bit-identical.
inline void ApplyLensRowSpan(const core::ImageU8& clean, const LensMap& map, int v0, int v1, core::ImageU8& out) {
    const int C = clean.channels;
    for (int v = v0; v < v1; ++v) {
        for (int u = 0; u < map.width; ++u) {
            const size_t i = size_t(v) * size_t(map.width) + size_t(u);
            const double su = map.src_u[i], sv = map.src_v[i];
            int x0 = int(std::floor(su)), y0 = int(std::floor(sv));
            double ax = su - x0, ay = sv - y0;
            // A sample exactly on the last row/column: step back one so x0+1 exists
            // (weight 1 on the far neighbour reproduces the same value exactly).
            if (x0 >= map.width - 1) {
                x0 = map.width - 2;
                ax = 1.0;
            }
            if (y0 >= map.height - 1) {
                y0 = map.height - 2;
                ay = 1.0;
            }
            for (int c = 0; c < C; ++c) {
                const double p00 = clean.data[clean.Index(x0, y0, c)], p10 = clean.data[clean.Index(x0 + 1, y0, c)];
                const double p01 = clean.data[clean.Index(x0, y0 + 1, c)];
                const double p11 = clean.data[clean.Index(x0 + 1, y0 + 1, c)];
                const double top = p00 + ax * (p10 - p00), bot = p01 + ax * (p11 - p01);
                const double val = top + ay * (bot - top);
                const double r = std::floor(val + 0.5);
                out.data[out.Index(u, v, c)] = uint8_t(r < 0 ? 0 : (r > 255 ? 255 : r));
            }
        }
    }
}

inline core::ImageU8 ApplyLens(const core::ImageU8& clean, const LensMap& map) {
    if (clean.width != map.width || clean.height != map.height) {
        throw std::invalid_argument("lens: image size does not match the map");
    }
    core::ImageU8 out(map.width, map.height, clean.channels);
    ApplyLensRowSpan(clean, map, 0, map.height, out);
    return out;
}

/// Nearest-neighbour resampling for depth (GT only): blending depths across an
/// occlusion edge would invent surfaces that do not exist.
inline core::ImageF32 ApplyLensDepth(const core::ImageF32& depth, const LensMap& map) {
    if (depth.width != map.width || depth.height != map.height) {
        throw std::invalid_argument("lens: depth size does not match the map");
    }
    core::ImageF32 out(map.width, map.height);
    for (size_t i = 0; i < map.src_u.size(); ++i) {
        const int x = int(std::floor(map.src_u[i] + 0.5)), y = int(std::floor(map.src_v[i] + 0.5));
        out.data[i] = depth.data[size_t(y) * size_t(depth.width) + size_t(x)];
    }
    return out;
}

}  // namespace ariadne::perception
