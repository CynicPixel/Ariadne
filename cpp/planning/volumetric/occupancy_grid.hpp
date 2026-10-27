#pragma once
// ---------------------------------------------------------------------------
// 1.8 placeholder occupancy (arch §20.2; design mission.md): a dense 3D voxel
// grid with OctoMap's sensor model (Hornung et al., Autonomous Robots 2013):
// per-voxel log-odds, a hit adds logit(0.7), a miss adds logit(0.4), clamped to
// [logit(0.12), logit(0.97)]. Stored as fixed-point integers (x1000) so the
// result is bit-identical on x86 and Arm. Unknown = never updated (a separate
// flag: log-odds can return to 0 after updates and that is NOT unknown).
//
// Rays are walked with Amanatides & Woo's exact voxel traversal ("A Fast Voxel
// Traversal Algorithm for Ray Tracing", Eurographics 1987), the walk OctoMap's
// ray casting uses: every voxel the segment passes through, each exactly once.
//
// The 2D slab collapses a z band (a standing body) into one tri-state per column:
// OCCUPIED if any voxel in the band is occupied, FREE if at least one is observed
// free and none occupied, else UNKNOWN.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ariadne::planning {

enum : uint8_t { kUnknown = 0, kFree = 1, kOccupied = 2 };

struct GridSpec {
    double origin_x = 0, origin_y = 0, origin_z = 0;   // world metres of the grid's min corner
    double cell = 0.05;                                // voxel edge, metres
    int nx = 0, ny = 0, nz = 0;                        // x = columns (east), y = rows (north), z = layers (up)
};

/// OctoMap sensor-model constants in fixed point (log-odds x 1000).
struct LogOddsModel {
    static int32_t Fixed(double p) { return int32_t(std::lround(1000.0 * std::log(p / (1.0 - p)))); }
    int32_t hit = Fixed(0.7), miss = Fixed(0.4), min = Fixed(0.12), max = Fixed(0.97);
    int32_t occupied_above = 0;   // p > 0.5
};

/// Amanatides & Woo: calls f(ix, iy, iz) for every voxel the segment a->b passes through,
/// in order, starting with a's voxel and ending with b's. Coordinates in voxel units.
template <class F>
void TraverseVoxels(const core::Vec3& a, const core::Vec3& b, F&& f) {
    int ix = int(std::floor(a.x())), iy = int(std::floor(a.y())), iz = int(std::floor(a.z()));
    const int ex = int(std::floor(b.x())), ey = int(std::floor(b.y())), ez = int(std::floor(b.z()));
    const core::Vec3 d = b - a;
    const int sx = d.x() > 0 ? 1 : (d.x() < 0 ? -1 : 0), sy = d.y() > 0 ? 1 : (d.y() < 0 ? -1 : 0),
              sz = d.z() > 0 ? 1 : (d.z() < 0 ? -1 : 0);
    const double inf = std::numeric_limits<double>::infinity();
    // t at which the segment crosses the next boundary on each axis, and the t per whole voxel.
    auto first = [&](double p, int i, int s, double dd) {
        if (s == 0) return inf;
        return ((s > 0 ? i + 1 : i) - p) / dd;
    };
    double tx = first(a.x(), ix, sx, d.x()), ty = first(a.y(), iy, sy, d.y()), tz = first(a.z(), iz, sz, d.z());
    const double dtx = sx ? std::abs(1.0 / d.x()) : inf, dty = sy ? std::abs(1.0 / d.y()) : inf,
                 dtz = sz ? std::abs(1.0 / d.z()) : inf;
    const int max_steps = std::abs(ex - ix) + std::abs(ey - iy) + std::abs(ez - iz);
    f(ix, iy, iz);
    for (int k = 0; k < max_steps; ++k) {
        if (tx <= ty && tx <= tz) {
            ix += sx;
            tx += dtx;
        } else if (ty <= tz) {
            iy += sy;
            ty += dty;
        } else {
            iz += sz;
            tz += dtz;
        }
        f(ix, iy, iz);
    }
}

class OccupancyGrid {
public:
    explicit OccupancyGrid(const GridSpec& g, LogOddsModel m = {}) : g_(g), m_(m) {
        if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0 || !(g.cell > 0)) throw std::invalid_argument("occupancy: bad grid");
        const size_t n = size_t(g.nx) * size_t(g.ny) * size_t(g.nz);
        L_.assign(n, 0);
        seen_.assign(n, 0);
    }

    const GridSpec& spec() const { return g_; }
    core::Vec3 ToVoxel(const core::Vec3& w) const {
        return {(w.x() - g_.origin_x) / g_.cell, (w.y() - g_.origin_y) / g_.cell, (w.z() - g_.origin_z) / g_.cell};
    }

    /// One depth measurement: camera centre `c` and surface point `p` (world). Voxels from c up to
    /// `free_until` (a point on c->p, short of p by the surface margin) get a miss; p's voxel a hit.
    void IntegrateRay(const core::Vec3& c, const core::Vec3& free_until, const core::Vec3& p) {
        TraverseVoxels(ToVoxel(c), ToVoxel(free_until), [&](int x, int y, int z) { Update(x, y, z, m_.miss); });
        const core::Vec3 v = ToVoxel(p);
        Update(int(std::floor(v.x())), int(std::floor(v.y())), int(std::floor(v.z())), m_.hit);
    }

    uint8_t State(int x, int y, int z) const {
        const size_t i = Index(x, y, z);
        if (!seen_[i]) return kUnknown;
        return L_[i] > m_.occupied_above ? kOccupied : kFree;
    }

    /// Tri-state per (row = y, col = x) over voxel layers [z0, z1). CV_8U, row 0 = south (y index 0).
    cv::Mat Slab(int z0, int z1) const {
        z0 = std::max(z0, 0);
        z1 = std::min(z1, g_.nz);
        cv::Mat s(g_.ny, g_.nx, CV_8U, cv::Scalar(kUnknown));
        for (int y = 0; y < g_.ny; ++y) {
            uint8_t* row = s.ptr<uint8_t>(y);
            for (int x = 0; x < g_.nx; ++x) {
                bool any_free = false, any_occ = false;
                for (int z = z0; z < z1 && !any_occ; ++z) {
                    const uint8_t st = State(x, y, z);
                    any_occ = st == kOccupied;
                    any_free = any_free || st == kFree;
                }
                row[x] = any_occ ? kOccupied : (any_free ? kFree : kUnknown);
            }
        }
        return s;
    }

    /// Route support per column (§22.2, [PROXY] for TSDF weight): for FREE columns, the strongest free
    /// evidence in the band, -L / -L_min in [0, 1] (1 = clamped free, ~5 consistent misses); else 0. CV_32F.
    cv::Mat SlabSupport(int z0, int z1, const cv::Mat& slab) const {
        z0 = std::max(z0, 0);
        z1 = std::min(z1, g_.nz);
        cv::Mat s(g_.ny, g_.nx, CV_32F, cv::Scalar(0.0f));
        for (int y = 0; y < g_.ny; ++y) {
            for (int x = 0; x < g_.nx; ++x) {
                if (slab.at<uint8_t>(y, x) != kFree) continue;
                int32_t best = 0;
                for (int z = z0; z < z1; ++z) {
                    const size_t i = Index(x, y, z);
                    if (seen_[i]) best = std::max(best, -L_[i]);
                }
                s.at<float>(y, x) = float(std::min(1.0, double(best) / double(-m_.min)));
            }
        }
        return s;
    }

    uint64_t updates() const { return updates_; }

private:
    size_t Index(int x, int y, int z) const { return (size_t(z) * size_t(g_.ny) + size_t(y)) * size_t(g_.nx) + size_t(x); }
    void Update(int x, int y, int z, int32_t delta) {
        if (x < 0 || y < 0 || z < 0 || x >= g_.nx || y >= g_.ny || z >= g_.nz) return;   // outside the mapped volume
        const size_t i = Index(x, y, z);
        L_[i] = std::clamp(L_[i] + delta, m_.min, m_.max);
        seen_[i] = 1;
        ++updates_;
    }

    GridSpec g_;
    LogOddsModel m_;
    std::vector<int32_t> L_;
    std::vector<uint8_t> seen_;
    uint64_t updates_ = 0;
};

}  // namespace ariadne::planning
