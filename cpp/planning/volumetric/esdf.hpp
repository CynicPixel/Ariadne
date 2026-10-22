#pragma once
// ---------------------------------------------------------------------------
// 4.1 ESDF for clearance (arch §20.3; design VOLUMETRIC_4.1 §3.3, D110).
//
// Exact 3D Euclidean distance transform by three separable passes of
// Felzenszwalb & Huttenlocher's 1D lower-envelope transform ("Distance
// Transforms of Sampled Functions", Theory of Computing 8, 2012, Algorithm 1):
// the squared distance to the nearest site is the lower envelope of parabolas
// (q - p)^2 + f(p), one rooted at each sample p; computing it along x, then y,
// then z gives the 3D squared EDT exactly (squared distance is separable).
// Sites are blocked voxels (f = 0), others +inf; +inf samples are skipped when
// the envelope is built (the paper's intersection formula is undefined for them).
//
// Clearance = distance from a voxel centre to the nearest blocked voxel's EDGE
// (edt * voxel - voxel / 2; the GT's and D45's convention), capped at d_max.
// Two fields: egress (blocked = occupied or unknown: "observed free", D45) and
// agent (blocked = occupied).
//
// Incremental, never global per keyframe: with a capped field a change at a
// voxel moves only voxels within d_max of it, so an update recomputes the box
// dirty (+) d_max reading sites from dirty (+) 2 d_max and writes back only the
// inner box. Exact for the capped field (bit-equal to a rebuild; tested).
// Stored per voxel as min(squared EDT, 255) in voxel units (uint8): exact for any
// d_max <= (sqrt(255) - 0.5) voxel; recomputed in x-y tiles, each reading sites
// from its own k-voxel margin, so memory stays at the tile scale.
// ---------------------------------------------------------------------------
#include "planning/volumetric/tsdf_map.hpp"

#include <cstdint>
#include <vector>

namespace ariadne::planning {

/// FH Algorithm 1 on n samples of f (stride apart), squared Euclidean, into out (same stride). +inf allowed.
/// v and z are scratch of size >= n and n + 1.
void DistanceTransform1D(const double* f, int n, size_t stride, double* out, int* v, double* z);

/// Squared EDT (voxel units) of a dense nx*ny*nz grid (x fastest); +inf where there is no site at all.
std::vector<double> SquaredEdt3D(const std::vector<uint8_t>& blocked, int nx, int ny, int nz);

class Esdf {
public:
    /// d_max: clearance cap (m); pad_m: extra margin kept around the map so growth rarely forces a rebuild.
    Esdf(double voxel, double d_max, double pad_m = 2.0);

    /// Global recompute over the map's bounds (+ margin). Used at start, on box growth, and by the tests.
    void Rebuild(const TsdfMap& map);
    /// Incremental update after Integrate returned dirty_blocks. Rebuilds instead when the map outgrew the box.
    /// Returns true if it was incremental.
    bool Update(const TsdfMap& map, const std::vector<Idx3>& dirty_blocks);

    /// Clearance (m) at a voxel, in [0, d_max]. Outside the box: egress 0 (unknown), agent d_max (nothing seen).
    double Egress(const Idx3& v) const;
    double Agent(const Idx3& v) const;
    double DMax() const { return d_max_; }
    bool Empty() const { return egress_.empty(); }
    Idx3 Lo() const { return lo_; }
    Idx3 Dims() const { return dims_; }
    size_t LastUpdatedVoxels() const { return last_updated_; }

private:
    void Recompute(const TsdfMap& map, Idx3 out_lo, Idx3 out_hi);   // tiled; sites from each tile (+) k
    void RecomputeTile(const TsdfMap& map, Idx3 site_lo, Idx3 site_hi, Idx3 out_lo, Idx3 out_hi);
    double Clear(uint8_t d2) const;
    size_t At(const Idx3& v) const {
        return size_t(v.x - lo_.x) + size_t(dims_.x) * (size_t(v.y - lo_.y) + size_t(dims_.y) * size_t(v.z - lo_.z));
    }
    bool Inside(const Idx3& v) const {
        return v.x >= lo_.x && v.y >= lo_.y && v.z >= lo_.z && v.x < lo_.x + dims_.x && v.y < lo_.y + dims_.y &&
               v.z < lo_.z + dims_.z;
    }
    double vox_, d_max_, pad_m_;
    int k_;   // ceil(d_max / voxel) + 1: the reach of a capped change, in voxels
    Idx3 lo_, dims_;
    std::vector<uint8_t> egress_, agent_;   // min(squared EDT, 255), voxel units
    size_t last_updated_ = 0;
};

}  // namespace ariadne::planning
