#pragma once
// ---------------------------------------------------------------------------
// 4.1 frontiers and the coarse global layer (arch §20.4, §20.6; design
// volumetric.md §3.4-§3.5, D110).
//
// Frontier voxel: FREE, inside the motion band [z_lo, z_hi], with an UNKNOWN
// 6-neighbour (Yamauchi 1997, in 3D). Clustered by 26-connectivity; clusters
// below min_voxels are dropped (counted). (FrontierCluster3D: the 1.8 header owns
// planning::FrontierCluster, a different type in the same test binary.) Per cluster: centroid, normal (the
// normalised sum of unit offsets towards unknown 6-neighbours) and a visible-
// unknown estimate (unknown voxels hit by a fixed 64-ray fan in a 60-degree
// half-angle cone about the normal, 2 m, each ray stopping at the first
// occupied voxel): candidate generator A of §21.2 and its Delta-coverage seed.
//
// Incremental (FUEL, Zhou et al. RA-L 2021): Update() drops the clusters that
// touch the updated box and re-detects inside it (plus a one-cluster-wide
// flood beyond it, so a cluster crossing the box edge is re-assembled whole);
// kept clusters are re-summarised, because their fan reads states 2 m away.
//
// Coarse layer: 0.5 m cells (10^3 voxels at 5 cm): known fraction, occupied
// fraction, mean weight; state unknown (< 0.5 known) / occupied (any occupied)
// / free. RLE codec (state, run) as varints, z-y-x order: the class-5 submap
// summary's payload (the wire field belongs to 3.2).
// ---------------------------------------------------------------------------
#include "planning/volumetric/tsdf_map.hpp"

#include <cstdint>
#include <vector>

namespace ariadne::planning {

struct FrontierParams {
    double z_lo = 0.6, z_hi = 1.6;   // motion band (m, world z)
    int min_voxels = 20;
    int fan_rays = 64;
    double fan_half_angle_deg = 60.0;
    double fan_range_m = 2.0;
};

struct FrontierCluster3D {
    std::vector<Idx3> voxels;     // sorted
    core::Vec3 centroid = core::Vec3::Zero();
    core::Vec3 normal = core::Vec3::Zero();   // towards the unknown side; zero if undefined
    int visible_unknown = 0;      // voxels, by the fan
    Idx3 lo, hi;                  // voxel bounding box
};

/// Is v a frontier voxel of map in band [zv_lo, zv_hi] (voxel z)?
bool IsFrontier(const TsdfMap& map, const Idx3& v, int zv_lo, int zv_hi);

class FrontierSet3D {
public:
    explicit FrontierSet3D(const FrontierParams& p) : p_(p) {}
    /// Full detection over the map.
    void Detect(const TsdfMap& map);
    /// Incremental: re-detect inside [lo, hi] (voxel box), keep clusters entirely outside it.
    void Update(const TsdfMap& map, const Idx3& lo, const Idx3& hi);
    /// Clusters with >= min_voxels, sorted by their first voxel (deterministic; equal to a full Detect).
    const std::vector<FrontierCluster3D>& Clusters() const { return clusters_; }
    int RejectedSmall() const { return rejected_small_; }

private:
    void Grow(const TsdfMap& map, std::vector<Idx3> seeds);   // flood from seeds, append clusters
    void Summarise(const TsdfMap& map, FrontierCluster3D& c) const;
    void Canonicalise();
    void Filter();   // all_ -> clusters_ (size filter) and rejected_small_
    FrontierParams p_;
    std::vector<FrontierCluster3D> all_, clusters_;   // all_ includes the small ones (needed by Update)
    int rejected_small_ = 0;
};

/// The coarse 0.5 m layer (§20.6).
struct CoarseCell {
    Idx3 c;                    // coarse index
    float known = 0, occupied = 0, mean_w = 0;   // fractions of the cell's voxels; mean weight of known ones
    uint8_t state = kUnknown;
};

struct CoarseLayer {
    double cell_m = 0.5;
    Idx3 lo, dims;             // coarse box
    std::vector<CoarseCell> cells;   // z-y-x order, dims.x fastest
};

CoarseLayer BuildCoarse(const TsdfMap& map, double cell_m = 0.5);
/// RLE of the coarse states: varint(state) varint(run) pairs, after a header (lo, dims, cell_mm) of varints.
std::vector<uint8_t> EncodeCoarseRle(const CoarseLayer& c);
CoarseLayer DecodeCoarseRle(const std::vector<uint8_t>& bytes);   // states only (known/occupied/mean_w = 0)

}  // namespace ariadne::planning
