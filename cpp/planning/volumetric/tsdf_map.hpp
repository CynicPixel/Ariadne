#pragma once
// ---------------------------------------------------------------------------
// 4.1 voxel-hash TSDF (arch §20.1-§20.2; design VOLUMETRIC_4.1 §3.1-§3.2, D110).
//
// Why from scratch: OpenCV 5's cv::Volume(HashTSDF) integrates and raycasts but
// exposes no voxel value or weight and takes no per-pixel weight (SMOKE §4u), so
// the tri-state, the ESDF and the frontiers cannot be read from it. It stays the
// correctness oracle (tests/unit/test_volumetric_cv.cpp).
//
// Layout (Voxblox, Oleynikova et al. IROS 2017): 8^3-voxel blocks in a hash map
// keyed by the packed block index; a voxel is (signed distance m, weight). A
// voxel never observed has weight 0 = unknown.
//
// Integration, per depth sample on a pixel grid: the ray camera -> point is
// walked (Amanatides & Woo, the 1.8 walk) to range + tau; each voxel takes the
// signed distance ALONG the ray, s = range - t_voxel, clamped to [-tau, tau], so
// voxels well in front of the surface are carved free (s = tau). Within one
// keyframe the samples hitting a voxel are averaged first, then fused once
// (D <- (W D + w s)/(W + w), W <- min(W + w, W_max)): evidence, not ray density.
// Weights w = (sigma_ref / sigma_Z)^2 (arch §20.1's 1/sigma_Z^2 up to a constant).
// A ray at a depth discontinuity (4-neighbour depth differs by > tau) integrates
// only s >= 0: its behind-surface band would otherwise land in the free space
// beyond the silhouette (measured: 112 floating occupied voxels in the test room).
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "planning/volumetric/occupancy_grid.hpp"   // kUnknown/kFree/kOccupied, TraverseVoxels

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ariadne::planning {

struct Idx3 {
    int x = 0, y = 0, z = 0;
    bool operator==(const Idx3& o) const { return x == o.x && y == o.y && z == o.z; }
    bool operator<(const Idx3& o) const { return z != o.z ? z < o.z : (y != o.y ? y < o.y : x < o.x); }
};

/// 21 bits per axis (offset 2^20): +-1M voxels, +-52 km at 5 cm.
inline int64_t Pack(const Idx3& i) {
    constexpr int64_t o = int64_t(1) << 20, m = (int64_t(1) << 21) - 1;
    return ((int64_t(i.x) + o) & m) | (((int64_t(i.y) + o) & m) << 21) | (((int64_t(i.z) + o) & m) << 42);
}

inline int FloorDiv(int a, int b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

struct TsdfParams {
    double voxel = 0.05;           // m (D110: 5 cm, measured against the rho_responder lattice)
    int truncation_voxels = 3;     // tau = 3 voxels
    double max_weight = 1e4;       // W_max
    double w_min = 0.0;            // unknown iff W <= w_min; 0 = arch §20.2 exactly ("unknown = weight 0")
    double sigma_ref = 0.01;       // m; w = (sigma_ref / sigma_Z)^2
    bool unit_weights = false;     // w = 1 (the cv::Volume oracle comparison)
    int pixel_stride = 4;          // sample every stride-th pixel in u and v
    double max_depth = 6.0;        // m (1.8 gate)
    double max_sigma = 0.06;       // m (1.8 gate, D45)
};

struct TsdfVoxel {
    float sdf = 0.f;   // m, along-ray signed distance, + in front of the surface
    float w = 0.f;     // 0 = never observed
};

constexpr int kBlockSide = 8;
constexpr int kBlockVoxels = kBlockSide * kBlockSide * kBlockSide;

struct TsdfBlock {
    std::array<TsdfVoxel, kBlockVoxels> v{};
};

/// One keyframe's depth, as MissionInput carries it (replay: the 1.7 spool; live: L1).
struct DepthFrame {
    core::SE3 T_W_C;                 // world <- camera
    double fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
    const float* depth = nullptr;    // row-major, metres (<= 0 or non-finite = no measurement)
    const float* sigma = nullptr;    // row-major, metres; may be null when unit_weights
    size_t row_stride = 0;           // floats per row (0 = width)
};

class TsdfMap {
public:
    explicit TsdfMap(const TsdfParams& p) : p_(p) {}

    /// Integrates one keyframe. Returns the blocks it touched (sorted, unique): the dirty set of the ESDF update.
    std::vector<Idx3> Integrate(const DepthFrame& f);

    const TsdfVoxel* Find(const Idx3& v) const;
    /// Writes one voxel directly (tests with exact SDFs, offline import); allocates its block. Not a fusion.
    void SetVoxel(const Idx3& v, const TsdfVoxel& x);
    uint8_t State(const Idx3& v) const;   // kUnknown / kFree / kOccupied (§20.2)
    double Voxel() const { return p_.voxel; }
    const TsdfParams& Params() const { return p_; }
    Idx3 ToVoxel(const core::Vec3& p) const;
    core::Vec3 Centre(const Idx3& v) const;
    size_t Blocks() const { return blocks_.size(); }
    /// Voxel-index bounds of all allocated blocks, inclusive. False if the map is empty.
    bool Bounds(Idx3* lo, Idx3* hi) const;
    /// Visits every allocated block (unspecified order) with its block index.
    void ForEachBlock(const std::function<void(const Idx3&, const TsdfBlock&)>& f) const;

private:
    TsdfParams p_;
    std::unordered_map<int64_t, TsdfBlock> blocks_;
    std::unordered_map<int64_t, Idx3> block_index_;
};

}  // namespace ariadne::planning
