#include "planning/volumetric/tsdf_map.hpp"

#include <algorithm>
#include <cmath>

namespace ariadne::planning {

namespace {

Idx3 BlockOf(const Idx3& v) {
    return {FloorDiv(v.x, kBlockSide), FloorDiv(v.y, kBlockSide), FloorDiv(v.z, kBlockSide)};
}

int Offset(const Idx3& v, const Idx3& b) {
    return (v.x - b.x * kBlockSide) + kBlockSide * ((v.y - b.y * kBlockSide) + kBlockSide * (v.z - b.z * kBlockSide));
}

}  // namespace

Idx3 TsdfMap::ToVoxel(const core::Vec3& p) const {
    return {int(std::floor(p.x() / p_.voxel)), int(std::floor(p.y() / p_.voxel)), int(std::floor(p.z() / p_.voxel))};
}

core::Vec3 TsdfMap::Centre(const Idx3& v) const {
    return core::Vec3((v.x + 0.5) * p_.voxel, (v.y + 0.5) * p_.voxel, (v.z + 0.5) * p_.voxel);
}

const TsdfVoxel* TsdfMap::Find(const Idx3& v) const {
    const Idx3 b = BlockOf(v);
    const auto it = blocks_.find(Pack(b));
    return it == blocks_.end() ? nullptr : &it->second.v[size_t(Offset(v, b))];
}

uint8_t TsdfMap::State(const Idx3& v) const {
    const TsdfVoxel* x = Find(v);
    if (!x || x->w <= p_.w_min) return kUnknown;
    return x->sdf < 0.5 * p_.voxel ? kOccupied : kFree;
}

bool TsdfMap::Bounds(Idx3* lo, Idx3* hi) const {
    if (block_index_.empty()) return false;
    Idx3 a{1 << 30, 1 << 30, 1 << 30}, b{-(1 << 30), -(1 << 30), -(1 << 30)};
    for (const auto& [k, i] : block_index_) {
        a = {std::min(a.x, i.x), std::min(a.y, i.y), std::min(a.z, i.z)};
        b = {std::max(b.x, i.x), std::max(b.y, i.y), std::max(b.z, i.z)};
    }
    *lo = {a.x * kBlockSide, a.y * kBlockSide, a.z * kBlockSide};
    *hi = {(b.x + 1) * kBlockSide - 1, (b.y + 1) * kBlockSide - 1, (b.z + 1) * kBlockSide - 1};
    return true;
}

void TsdfMap::ForEachBlock(const std::function<void(const Idx3&, const TsdfBlock&)>& f) const {
    for (const auto& [k, blk] : blocks_) f(block_index_.at(k), blk);
}

std::vector<Idx3> TsdfMap::Integrate(const DepthFrame& f) {
    struct Acc {
        Idx3 v;
        double ws = 0, w = 0;
        int n = 0;
    };
    std::unordered_map<int64_t, Acc> frame;   // this keyframe's samples per voxel
    std::vector<int64_t> order;                // first-touch order (deterministic: the ray order)
    const double vox = p_.voxel, tau = p_.truncation_voxels * vox;
    const size_t stride = f.row_stride ? f.row_stride : size_t(f.width);
    const core::Vec3 c = f.T_W_C.t;
    for (int v = 0; v < f.height; v += p_.pixel_stride) {
        for (int u = 0; u < f.width; u += p_.pixel_stride) {
            const float z = f.depth[size_t(v) * stride + size_t(u)];
            const float s = f.sigma ? f.sigma[size_t(v) * stride + size_t(u)] : 0.f;
            if (!std::isfinite(z) || z <= 0 || z > p_.max_depth) continue;
            if (!p_.unit_weights && (!std::isfinite(s) || s <= 0 || s > p_.max_sigma)) continue;
            const double w = p_.unit_weights ? 1.0 : (p_.sigma_ref / s) * (p_.sigma_ref / s);
            // Depth discontinuity (a 4-neighbour pixel's depth differs by > tau, or is missing): the ray grazes a
            // silhouette, and its behind-surface band would stamp occupied voxels into the free space beyond the
            // edge (narrowing doorways seen obliquely). Such a ray integrates only its in-front part (s >= 0).
            bool edge = false;
            for (const auto [du, dv] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                const int uu = u + du, vv = v + dv;
                if (uu < 0 || vv < 0 || uu >= f.width || vv >= f.height) continue;
                const float zn = f.depth[size_t(vv) * stride + size_t(uu)];
                if (!std::isfinite(zn) || zn <= 0 || std::abs(zn - z) > tau) edge = true;
            }
            const core::Vec3 X = f.T_W_C * core::Vec3((u - f.cx) / f.fx * z, (v - f.cy) / f.fy * z, z);
            const double range = (X - c).norm();
            if (!(range > 0)) continue;
            const core::Vec3 dir = (X - c) / range;
            const core::Vec3 end = c + dir * (range + tau);
            TraverseVoxels(c / vox, end / vox, [&](int ix, int iy, int iz) {
                const Idx3 i{ix, iy, iz};
                const double t = (Centre(i) - c).dot(dir);
                const double sd = std::clamp(range - t, -tau, tau);
                if (range - t < -tau) return;   // the walk's last voxels can overshoot the band
                if (edge && sd < 0) return;     // silhouette ray: no behind-surface band
                const int64_t key = Pack(i);
                auto [it, fresh] = frame.try_emplace(key);
                if (fresh) {
                    it->second.v = i;
                    order.push_back(key);
                }
                it->second.ws += w * sd;
                it->second.w += w;
                ++it->second.n;
            });
        }
    }
    std::vector<Idx3> dirty;
    for (const int64_t key : order) {
        const Acc& a = frame.at(key);
        const double s_obs = a.ws / a.w, w_obs = a.w / a.n;
        const Idx3 b = BlockOf(a.v);
        const int64_t bk = Pack(b);
        auto [bit, fresh] = blocks_.try_emplace(bk);
        if (fresh) block_index_.emplace(bk, b);
        TsdfVoxel& x = bit->second.v[size_t(Offset(a.v, b))];
        const double W = x.w;
        x.sdf = float((W * x.sdf + w_obs * s_obs) / (W + w_obs));
        x.w = float(std::min(W + w_obs, p_.max_weight));
        dirty.push_back(b);
    }
    std::sort(dirty.begin(), dirty.end());
    dirty.erase(std::unique(dirty.begin(), dirty.end()), dirty.end());
    return dirty;
}

}  // namespace ariadne::planning
