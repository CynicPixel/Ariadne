#include "planning/volumetric/frontiers3d.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace ariadne::planning {

namespace {

constexpr int kSix[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

int Band(double z, double vox) { return int(std::floor(z / vox)); }

bool InBox(const Idx3& v, const Idx3& lo, const Idx3& hi) {
    return v.x >= lo.x && v.y >= lo.y && v.z >= lo.z && v.x <= hi.x && v.y <= hi.y && v.z <= hi.z;
}

bool BoxesTouch(const Idx3& alo, const Idx3& ahi, const Idx3& blo, const Idx3& bhi) {
    return alo.x <= bhi.x && blo.x <= ahi.x && alo.y <= bhi.y && blo.y <= ahi.y && alo.z <= bhi.z && blo.z <= ahi.z;
}

}  // namespace

bool IsFrontier(const TsdfMap& map, const Idx3& v, int zv_lo, int zv_hi) {
    if (v.z < zv_lo || v.z > zv_hi || map.State(v) != kFree) return false;
    for (const auto& d : kSix) {
        if (map.State({v.x + d[0], v.y + d[1], v.z + d[2]}) == kUnknown) return true;
    }
    return false;
}

void FrontierSet3D::Grow(const TsdfMap& map, std::vector<Idx3> seeds) {
    const int zl = Band(p_.z_lo, map.Voxel()), zh = Band(p_.z_hi, map.Voxel());
    std::sort(seeds.begin(), seeds.end());
    std::unordered_set<int64_t> seen;
    for (const Idx3& s : seeds) {
        if (seen.count(Pack(s)) || !IsFrontier(map, s, zl, zh)) continue;
        FrontierCluster3D c;
        std::vector<Idx3> stack{s};
        seen.insert(Pack(s));
        while (!stack.empty()) {
            const Idx3 v = stack.back();
            stack.pop_back();
            c.voxels.push_back(v);
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const Idx3 n{v.x + dx, v.y + dy, v.z + dz};
                        if (seen.count(Pack(n)) || !IsFrontier(map, n, zl, zh)) continue;
                        seen.insert(Pack(n));
                        stack.push_back(n);
                    }
                }
            }
        }
        std::sort(c.voxels.begin(), c.voxels.end());
        Summarise(map, c);
        clusters_.push_back(std::move(c));
    }
}

void FrontierSet3D::Summarise(const TsdfMap& map, FrontierCluster3D& c) const {
    c.lo = c.hi = c.voxels.front();
    core::Vec3 sum = core::Vec3::Zero(), nsum = core::Vec3::Zero();
    for (const Idx3& v : c.voxels) {
        sum += map.Centre(v);
        c.lo = {std::min(c.lo.x, v.x), std::min(c.lo.y, v.y), std::min(c.lo.z, v.z)};
        c.hi = {std::max(c.hi.x, v.x), std::max(c.hi.y, v.y), std::max(c.hi.z, v.z)};
        for (const auto& d : kSix) {
            if (map.State({v.x + d[0], v.y + d[1], v.z + d[2]}) == kUnknown) nsum += core::Vec3(d[0], d[1], d[2]);
        }
    }
    c.centroid = sum / double(c.voxels.size());
    c.normal = nsum.norm() > 1e-12 ? core::Vec3(nsum / nsum.norm()) : core::Vec3::Zero();
    c.visible_unknown = 0;
    if (c.normal.isZero()) return;
    // Fan: Fibonacci directions on the spherical cap of half-angle a about the normal (deterministic).
    const core::Vec3 n = c.normal;
    const core::Vec3 t1 = (std::abs(n.z()) < 0.9 ? n.cross(core::Vec3::UnitZ()) : n.cross(core::Vec3::UnitX())).normalized();
    const core::Vec3 t2 = n.cross(t1);
    const double cos_a = std::cos(p_.fan_half_angle_deg * M_PI / 180.0), golden = M_PI * (3.0 - std::sqrt(5.0));
    std::unordered_set<int64_t> hit;
    const double vox = map.Voxel();
    for (int i = 0; i < p_.fan_rays; ++i) {
        const double cz = 1.0 - (1.0 - cos_a) * (i + 0.5) / p_.fan_rays, r = std::sqrt(std::max(0.0, 1.0 - cz * cz));
        const double phi = golden * i;
        const core::Vec3 dir = n * cz + t1 * (r * std::cos(phi)) + t2 * (r * std::sin(phi));
        bool stop = false;
        TraverseVoxels(c.centroid / vox, (c.centroid + dir * p_.fan_range_m) / vox, [&](int x, int y, int z) {
            if (stop) return;
            const uint8_t s = map.State({x, y, z});
            if (s == kOccupied) stop = true;
            else if (s == kUnknown) hit.insert(Pack({x, y, z}));
        });
    }
    c.visible_unknown = int(hit.size());
}

void FrontierSet3D::Canonicalise() {
    std::sort(clusters_.begin(), clusters_.end(),
              [](const FrontierCluster3D& a, const FrontierCluster3D& b) { return a.voxels.front() < b.voxels.front(); });
}

void FrontierSet3D::Detect(const TsdfMap& map) {
    clusters_.clear();
    const int zl = Band(p_.z_lo, map.Voxel()), zh = Band(p_.z_hi, map.Voxel());
    std::vector<Idx3> seeds;
    map.ForEachBlock([&](const Idx3& b, const TsdfBlock&) {
        for (int z = 0; z < kBlockSide; ++z) {
            const int vz = b.z * kBlockSide + z;
            if (vz < zl || vz > zh) continue;
            for (int y = 0; y < kBlockSide; ++y) {
                for (int x = 0; x < kBlockSide; ++x) {
                    const Idx3 v{b.x * kBlockSide + x, b.y * kBlockSide + y, vz};
                    if (IsFrontier(map, v, zl, zh)) seeds.push_back(v);
                }
            }
        }
    });
    Grow(map, std::move(seeds));
    Canonicalise();
    all_ = clusters_;
    Filter();
}

void FrontierSet3D::Update(const TsdfMap& map, const Idx3& lo, const Idx3& hi) {
    // Frontier status can change within lo-1..hi+1 (a voxel's status reads its 6-neighbours); a kept cluster must
    // not even touch that region, so the test box is lo-2..hi+2.
    const Idx3 flo{lo.x - 1, lo.y - 1, lo.z - 1}, fhi{hi.x + 1, hi.y + 1, hi.z + 1};
    const Idx3 tlo{lo.x - 2, lo.y - 2, lo.z - 2}, thi{hi.x + 2, hi.y + 2, hi.z + 2};
    std::vector<FrontierCluster3D> kept;
    std::vector<Idx3> seeds;
    for (auto& c : all_) {
        if (BoxesTouch(c.lo, c.hi, tlo, thi)) seeds.insert(seeds.end(), c.voxels.begin(), c.voxels.end());
        else kept.push_back(std::move(c));
    }
    const int zl = Band(p_.z_lo, map.Voxel()), zh = Band(p_.z_hi, map.Voxel());
    for (int z = std::max(flo.z, zl); z <= std::min(fhi.z, zh); ++z) {
        for (int y = flo.y; y <= fhi.y; ++y) {
            for (int x = flo.x; x <= fhi.x; ++x) seeds.push_back({x, y, z});
        }
    }
    // A kept cluster's normal and fan read states up to fan_range away: re-summarise them (a few rays each).
    for (auto& c : kept) Summarise(map, c);
    clusters_ = std::move(kept);
    Grow(map, std::move(seeds));
    Canonicalise();
    all_ = clusters_;
    Filter();
}

void FrontierSet3D::Filter() {
    clusters_.clear();
    rejected_small_ = 0;
    for (const auto& c : all_) {
        if (int(c.voxels.size()) >= p_.min_voxels) clusters_.push_back(c);
        else ++rejected_small_;
    }
}

// --- coarse layer ---------------------------------------------------------------------------------------------

CoarseLayer BuildCoarse(const TsdfMap& map, double cell_m) {
    CoarseLayer out;
    out.cell_m = cell_m;
    const int r = int(std::lround(cell_m / map.Voxel()));
    if (r < 1 || std::abs(r * map.Voxel() - cell_m) > 1e-9) throw std::invalid_argument("coarse cell must be a voxel multiple");
    Idx3 lo, hi;
    if (!map.Bounds(&lo, &hi)) return out;
    out.lo = {FloorDiv(lo.x, r), FloorDiv(lo.y, r), FloorDiv(lo.z, r)};
    const Idx3 chi{FloorDiv(hi.x, r), FloorDiv(hi.y, r), FloorDiv(hi.z, r)};
    out.dims = {chi.x - out.lo.x + 1, chi.y - out.lo.y + 1, chi.z - out.lo.z + 1};
    const size_t n = size_t(out.dims.x) * size_t(out.dims.y) * size_t(out.dims.z);
    std::vector<int> known(n, 0), occ(n, 0);
    std::vector<double> wsum(n, 0);
    const double w_min = map.Params().w_min;
    map.ForEachBlock([&](const Idx3& b, const TsdfBlock& blk) {
        for (int i = 0; i < kBlockVoxels; ++i) {
            const TsdfVoxel& x = blk.v[size_t(i)];
            if (x.w <= w_min) continue;
            const Idx3 v{b.x * kBlockSide + i % kBlockSide, b.y * kBlockSide + (i / kBlockSide) % kBlockSide,
                         b.z * kBlockSide + i / (kBlockSide * kBlockSide)};
            const Idx3 c{FloorDiv(v.x, r) - out.lo.x, FloorDiv(v.y, r) - out.lo.y, FloorDiv(v.z, r) - out.lo.z};
            const size_t k = size_t(c.x) + size_t(out.dims.x) * (size_t(c.y) + size_t(out.dims.y) * size_t(c.z));
            ++known[k];
            wsum[k] += x.w;
            if (x.sdf < 0.5 * map.Voxel()) ++occ[k];
        }
    });
    const double per = double(r) * r * r;
    out.cells.resize(n);
    for (int z = 0; z < out.dims.z; ++z) {
        for (int y = 0; y < out.dims.y; ++y) {
            for (int x = 0; x < out.dims.x; ++x) {
                const size_t k = size_t(x) + size_t(out.dims.x) * (size_t(y) + size_t(out.dims.y) * size_t(z));
                CoarseCell& c = out.cells[k];
                c.c = {out.lo.x + x, out.lo.y + y, out.lo.z + z};
                c.known = float(known[k] / per);
                c.occupied = float(occ[k] / per);
                c.mean_w = known[k] ? float(wsum[k] / known[k]) : 0.f;
                c.state = occ[k] > 0 ? kOccupied : (c.known < 0.5f ? kUnknown : kFree);
            }
        }
    }
    return out;
}

namespace {

void PutVarint(std::vector<uint8_t>& b, uint64_t v) {
    while (v >= 0x80) b.push_back(uint8_t(v | 0x80)), v >>= 7;
    b.push_back(uint8_t(v));
}
uint64_t GetVarint(const std::vector<uint8_t>& b, size_t& i) {
    uint64_t v = 0;
    for (int s = 0; s < 64; s += 7) {
        if (i >= b.size()) throw std::runtime_error("coarse rle: truncated");
        const uint8_t x = b[i++];
        v |= uint64_t(x & 0x7f) << s;
        if (!(x & 0x80)) return v;
    }
    throw std::runtime_error("coarse rle: bad varint");
}
uint64_t Zig(int64_t v) { return (uint64_t(v) << 1) ^ uint64_t(v >> 63); }
int64_t Unzig(uint64_t v) { return int64_t(v >> 1) ^ -int64_t(v & 1); }

}  // namespace

std::vector<uint8_t> EncodeCoarseRle(const CoarseLayer& c) {
    std::vector<uint8_t> b;
    for (int v : {c.lo.x, c.lo.y, c.lo.z}) PutVarint(b, Zig(v));
    for (int v : {c.dims.x, c.dims.y, c.dims.z}) PutVarint(b, uint64_t(v));
    PutVarint(b, uint64_t(std::llround(c.cell_m * 1000)));
    for (size_t i = 0; i < c.cells.size();) {
        size_t j = i;
        while (j < c.cells.size() && c.cells[j].state == c.cells[i].state) ++j;
        PutVarint(b, c.cells[i].state);
        PutVarint(b, j - i);
        i = j;
    }
    return b;
}

CoarseLayer DecodeCoarseRle(const std::vector<uint8_t>& b) {
    CoarseLayer c;
    size_t i = 0;
    c.lo.x = int(Unzig(GetVarint(b, i))), c.lo.y = int(Unzig(GetVarint(b, i))), c.lo.z = int(Unzig(GetVarint(b, i)));
    c.dims.x = int(GetVarint(b, i)), c.dims.y = int(GetVarint(b, i)), c.dims.z = int(GetVarint(b, i));
    c.cell_m = double(GetVarint(b, i)) / 1000.0;
    const size_t n = size_t(c.dims.x) * size_t(c.dims.y) * size_t(c.dims.z);
    while (c.cells.size() < n) {
        const uint8_t s = uint8_t(GetVarint(b, i));
        const uint64_t run = GetVarint(b, i);
        if (run == 0 || c.cells.size() + run > n) throw std::runtime_error("coarse rle: bad run");
        for (uint64_t k = 0; k < run; ++k) {
            const size_t q = c.cells.size();
            CoarseCell cc;
            cc.c = {c.lo.x + int(q % size_t(c.dims.x)), c.lo.y + int((q / size_t(c.dims.x)) % size_t(c.dims.y)),
                    c.lo.z + int(q / (size_t(c.dims.x) * size_t(c.dims.y)))};
            cc.state = s;
            c.cells.push_back(cc);
        }
    }
    if (i != b.size()) throw std::runtime_error("coarse rle: trailing bytes");
    return c;
}

}  // namespace ariadne::planning
