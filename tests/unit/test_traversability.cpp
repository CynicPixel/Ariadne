// 4.2 traversability and the structural-risk proxy (design traversability-risk.md §4, D111), on exact SDFs of known
// geometry: the capsule against brute force at both rho, a doorway and a beam by hand, the risk proxy's zero, its
// response to a cantilever and its monotonicity, and egress-implies-agent.
#include "planning/volumetric/esdf.hpp"
#include "planning/volumetric/traversability.hpp"
#include "planning/volumetric/tsdf_map.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace ariadne;
using namespace ariadne::planning;

namespace {

struct Solid {
    core::Vec3 lo, hi;
};

double BoxSdf(const core::Vec3& p, const Solid& b) {   // exact signed distance to an axis-aligned box
    const core::Vec3 c = (b.lo + b.hi) / 2, h = (b.hi - b.lo) / 2;
    const core::Vec3 q = (p - c).cwiseAbs() - h;
    return q.cwiseMax(0.0).norm() + std::min(q.maxCoeff(), 0.0);
}

// A 4 x 3 m room (floor z < 0, ceiling z >= 2.6), split at x = 2 by a 0.1 m wall with a doorway of `door_w`
// centred at y = 1.5; optional extra solids. Every voxel in the box is observed (weight 1), SDF clamped to tau.
TsdfMap Scene(double door_w, const std::vector<Solid>& extra) {
    TsdfParams tp;
    TsdfMap map(tp);
    std::vector<Solid> s = {{{-1, -1, -1}, {5, 4, 0}},  {{-1, -1, 2.6}, {5, 4, 3.6}},   // floor, ceiling
                            {{-1, -1, -1}, {0, 4, 3.6}}, {{4, -1, -1}, {5, 4, 3.6}},    // west, east
                            {{-1, -1, -1}, {5, 0, 3.6}}, {{-1, 3, -1}, {5, 4, 3.6}},    // south, north
                            {{1.95, -1, -1}, {2.05, 1.5 - door_w / 2, 3.6}}, {{1.95, 1.5 + door_w / 2, -1}, {2.05, 4, 3.6}}};
    s.insert(s.end(), extra.begin(), extra.end());
    const double vox = tp.voxel, tau = tp.truncation_voxels * vox;
    for (int z = -4; z < 56; ++z)
        for (int y = -4; y < 64; ++y)
            for (int x = -4; x < 84; ++x) {
                const core::Vec3 c = map.Centre({x, y, z});
                double d = std::numeric_limits<double>::infinity();
                for (const auto& b : s) d = std::min(d, BoxSdf(c, b));
                map.SetVoxel({x, y, z}, {float(std::clamp(d, -tau, tau)), 1.f});
            }
    return map;
}

GridFrame Frame() {
    GridFrame g;
    g.ox = -0.1, g.oy = -0.1, g.cell = 0.05, g.cols = 84, g.rows = 64;
    return g;
}

TraversabilityMaps Trav(const TsdfMap& map, const TravParams& p = {}) {
    Esdf e(map.Voxel(), 0.50);
    e.Rebuild(map);
    return ComputeTraversability(map, e, Frame(), p);
}

uint8_t At(const cv::Mat& m, double x, double y) {
    const GridFrame g = Frame();
    return m.at<uint8_t>(int(std::floor((y - g.oy) / g.cell)), int(std::floor((x - g.ox) / g.cell)));
}

}  // namespace

TEST(Traversability, CapsuleMatchesBruteForceAtBothRho) {
    const TsdfMap map = Scene(1.05, {{{2.9, 0.6, 1.5}, {3.3, 1.0, 1.7}}});   // a doorway and a hanging block
    const TravParams p;
    const TraversabilityMaps t = Trav(map, p);
    const GridFrame g = Frame();
    const double vox = map.Voxel(), d_max = 0.50;
    std::vector<Idx3> blocked_r, blocked_a;   // egress: not free; agent: occupied
    Idx3 lo, hi;
    map.Bounds(&lo, &hi);
    for (int z = lo.z; z <= hi.z; ++z)
        for (int y = lo.y; y <= hi.y; ++y)
            for (int x = lo.x; x <= hi.x; ++x) {
                const uint8_t s = map.State({x, y, z});
                if (s != kFree) blocked_r.push_back({x, y, z});
                if (s == kOccupied) blocked_a.push_back({x, y, z});
            }
    // Independent oracle: centre (x, y, z) to the nearest point of each blocked cube, by clamping; capped.
    auto Clear = [&](const std::vector<Idx3>& B, int x, int y, int z) {
        double m = d_max;
        for (const auto& b : B) {
            if (std::abs(b.x - x) > 12 || std::abs(b.y - y) > 12 || std::abs(b.z - z) > 12) continue;
            const double px = std::clamp(double(x), b.x - 0.5, b.x + 0.5), py = std::clamp(double(y), b.y - 0.5, b.y + 0.5),
                         pz = std::clamp(double(z), b.z - 0.5, b.z + 0.5);
            m = std::min(m, std::sqrt((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz)) * vox);
        }
        return m;
    };
    // 1.8's column rule over the body band, brute force, and the non-free columns for the 2D clearance.
    const int sl = int(std::floor(p.slab_z_lo / vox)), sh = int(std::floor(p.slab_z_hi / vox));
    std::vector<std::pair<int, int>> nonfree;
    cv::Mat slab(g.rows, g.cols, CV_8U);
    for (int r = 0; r < g.rows; ++r)
        for (int c = 0; c < g.cols; ++c) {
            bool f = false, o = false;
            for (int z = sl; z <= sh; ++z) {
                const uint8_t st = map.State({c - 2, r - 2, z});
                f |= st == kFree, o |= st == kOccupied;
            }
            slab.at<uint8_t>(r, c) = o ? kOccupied : (f ? kFree : kUnknown);
            if (slab.at<uint8_t>(r, c) != kFree) nonfree.push_back({r, c});
        }
    int checked = 0;
    for (int r = 4; r < g.rows - 4; r += 3)
        for (int c = 4; c < g.cols - 4; c += 3, ++checked) {
            const int x = c - 2, y = r - 2;   // ox = oy = -0.1 m = -2 voxels
            bool strict = true, cap_occ = true, ba = true;
            for (int z = int(std::floor(p.responder.axis_lo / vox)); z <= int(std::floor(p.responder.axis_hi / vox)); ++z) {
                strict = strict && Clear(blocked_r, x, y, z) >= p.responder.rho - 1e-9;
                cap_occ = cap_occ && Clear(blocked_a, x, y, z) >= p.responder.rho - 1e-9;
            }
            for (int z = int(std::floor(p.agent.axis_lo / vox)); z <= int(std::floor(p.agent.axis_hi / vox)); ++z)
                ba = ba && Clear(blocked_a, x, y, z) >= p.agent.rho - 1e-9;
            double c2 = d_max;   // to the nearest point of each non-free column's square, by clamping
            for (const auto& [rr, cc] : nonfree) {
                const double pr = std::clamp(double(r), rr - 0.5, rr + 0.5), pc = std::clamp(double(c), cc - 0.5, cc + 0.5);
                c2 = std::min(c2, std::sqrt((r - pr) * (r - pr) + (c - pc) * (c - pc)) * vox);
            }
            const bool hybrid = slab.at<uint8_t>(r, c) == kFree && c2 >= p.responder.rho - 1e-9 && cap_occ;
            ASSERT_EQ(t.slab.at<uint8_t>(r, c), slab.at<uint8_t>(r, c)) << r << "," << c;
            ASSERT_EQ(t.trav_r_strict.at<uint8_t>(r, c), strict) << r << "," << c;
            ASSERT_EQ(t.trav_r.at<uint8_t>(r, c), hybrid) << r << "," << c;
            ASSERT_EQ(t.trav_a.at<uint8_t>(r, c), ba) << r << "," << c;
        }
    EXPECT_GT(checked, 400);
}

TEST(Traversability, DoorwayAndBeamByHand) {
    TravParams p;
    // A 1.05 m doorway: open at rho_responder 0.40 m (needs 0.80 m), closed at 0.55 m (needs 1.10 m).
    EXPECT_EQ(At(Trav(Scene(1.05, {}), p).trav_r, 2.0, 1.5), 1);
    TravParams wide = p;
    wide.responder.rho = 0.55;   // d_max 0.50 caps the field: use a field cap above rho for this check
    {
        const TsdfMap map = Scene(1.05, {});
        Esdf e(map.Voxel(), 0.65);
        e.Rebuild(map);
        EXPECT_EQ(At(ComputeTraversability(map, e, Frame(), wide).trav_r, 2.0, 1.5), 0);
    }
    // A beam across the doorway with its underside at 1.5 m: closes a standing responder (axis to 1.4 m: 0.1 m from
    // the beam), not the rig flying at 0.8-1.2 m (0.3 m >= rho_agent 0.2 m).
    const TraversabilityMaps t = Trav(Scene(1.05, {{{1.9, 0.9, 1.5}, {2.1, 2.1, 1.6}}}), p);
    EXPECT_EQ(At(t.trav_r, 2.0, 1.5), 0);
    EXPECT_EQ(At(t.trav_a, 2.0, 1.5), 1);
}

TEST(Traversability, RiskProxyIsZeroInABareRoomAndRespondsToACantilever) {
    const TraversabilityMaps bare = Trav(Scene(1.05, {}));
    double mx;
    cv::minMaxLoc(bare.risk, nullptr, &mx);
    EXPECT_LT(mx, 0.05) << "a bare room under a ceiling at its design height must read ~0";
    std::vector<double> peak;
    for (double len : {0.5, 1.0, 1.5}) {   // a slab cantilevered from the east wall, underside at 2.0 m
        const TraversabilityMaps t = Trav(Scene(1.05, {{{4.0 - len, 1.0, 2.0}, {4.0, 2.0, 2.15}}}));
        cv::minMaxLoc(t.risk, nullptr, &mx);
        peak.push_back(mx);
        const GridFrame g = Frame();
        const float under = t.risk.at<float>(int((1.5 - g.oy) / g.cell), int((4.0 - len / 2 - g.ox) / g.cell));
        EXPECT_GT(under, 0.5) << "under a " << len << " m cantilever";
    }
    EXPECT_LT(peak[0], peak[2] + 1e-6);
    EXPECT_LE(peak[0], peak[1] + 1e-6);
}

TEST(Traversability, UnobservedColumnsCloseEgressButNotTheAgent) {
    // Partially observed: drop the observation of a strip (weight 0) so egress (unknown = blocked) loses it.
    TsdfMap map = Scene(1.05, {});
    for (int z = -4; z < 56; ++z)
        for (int y = 20; y < 30; ++y)
            for (int x = 50; x < 70; ++x) map.SetVoxel({x, y, z}, {0.f, 0.f});
    const TraversabilityMaps t = Trav(map);
    int lost = 0;
    for (int r = 0; r < t.trav_r.rows; ++r)
        for (int c = 0; c < t.trav_r.cols; ++c) {
            if (t.trav_r.at<uint8_t>(r, c)) ASSERT_EQ(t.slab.at<uint8_t>(r, c), kFree);   // egress needs an observed column
            lost += t.trav_a.at<uint8_t>(r, c) && !t.trav_r.at<uint8_t>(r, c);
        }
    EXPECT_GT(lost, 50) << "the unobserved strip must close responder columns the agent still may use";
}

namespace {

// Sphere-traced depth of the Scene geometry (union of boxes) from a pinhole at T (optical frame x right, y down).
cv::Mat RenderScene(const std::vector<Solid>& solids, const core::SE3& T, int w, int h, double f) {
    cv::Mat Z(h, w, CV_32F);
    for (int v = 0; v < h; ++v)
        for (int u = 0; u < w; ++u) {
            const core::Vec3 r = core::Vec3((u - (w - 1) / 2.0) / f, (v - (h - 1) / 2.0) / f, 1.0);
            const core::Vec3 d = T.R * r.normalized();
            double t = 0;
            for (int it = 0; it < 400 && t < 8; ++it) {
                double m = 1e9;
                for (const auto& b : solids) m = std::min(m, BoxSdf(T.t + d * t, b));
                if (m < 1e-4) break;
                t += m;
            }
            Z.at<float>(v, u) = float(t * r.normalized().z());   // depth along the optical axis
        }
    return Z;
}

core::SE3 LookAt(const core::Vec3& c, double yaw) {
    const core::Vec3 z(std::cos(yaw), std::sin(yaw), 0), x = z.cross(core::Vec3::UnitZ()), y = z.cross(x);
    core::SE3 T;
    T.R.col(0) = x, T.R.col(1) = y, T.R.col(2) = z;
    T.t = c;
    return T;
}

}  // namespace

TEST(Traversability, AHorizontalCameraCannotObserveTheStandingBodyVolumeButTheHybridRuleConnects) {
    // The rig's geometry: altitude 1.0 m, a 90-degree hfov at 16:9 (vertical half-FOV ~29 deg), depth gate 2.84 m.
    const double door = 1.05;
    const std::vector<Solid> solids = {{{-1, -1, -1}, {5, 4, 0}},  {{-1, -1, 2.6}, {5, 4, 3.6}},  {{-1, -1, -1}, {0, 4, 3.6}},
                                       {{4, -1, -1}, {5, 4, 3.6}}, {{-1, -1, -1}, {5, 0, 3.6}},   {{-1, 3, -1}, {5, 4, 3.6}},
                                       {{1.95, -1, -1}, {2.05, 1.5 - door / 2, 3.6}}, {{1.95, 1.5 + door / 2, -1}, {2.05, 4, 3.6}}};
    TsdfParams tp;
    tp.pixel_stride = 2, tp.max_depth = 2.84, tp.unit_weights = true;
    TsdfMap map(tp);
    const int w = 160, h = 90;
    const double f = 80;
    for (const core::Vec3& c : {core::Vec3(0.8, 1.5, 1.0), core::Vec3(1.4, 0.8, 1.0), core::Vec3(1.4, 2.2, 1.0),
                                core::Vec3(2.6, 1.5, 1.0), core::Vec3(3.3, 0.8, 1.0), core::Vec3(3.3, 2.2, 1.0)})
        for (int k = 0; k < 8; ++k) {
            const core::SE3 T = LookAt(c, k * M_PI / 4);
            const cv::Mat Z = RenderScene(solids, T, w, h, f);
            DepthFrame fr;
            fr.T_W_C = T, fr.fx = f, fr.fy = f, fr.cx = (w - 1) / 2.0, fr.cy = (h - 1) / 2.0, fr.width = w, fr.height = h;
            fr.depth = Z.ptr<float>();
            map.Integrate(fr);
        }
    const TraversabilityMaps t = Trav(map), truth = Trav(Scene(door, {}));
    const int strict = cv::countNonZero(t.trav_r_strict), hybrid = cv::countNonZero(t.trav_r);
    int false_claims = 0;
    for (int r = 0; r < t.trav_r.rows; ++r)
        for (int c = 0; c < t.trav_r.cols; ++c) false_claims += t.trav_r.at<uint8_t>(r, c) && !truth.trav_r.at<uint8_t>(r, c);
    std::printf("[ PARTIAL  ] responder columns: strict 3D %d, hybrid %d, fully observed %d; hybrid false claims %d\n",
                strict, hybrid, cv::countNonZero(truth.trav_r), false_claims);
    EXPECT_EQ(false_claims, 0);
    // Even with 48 noise-free views the strict rule loses columns to the body volume a horizontal camera at 1.0 m
    // never sees (measured here: 1535 vs 1949; on the evidence spool, with real gating and fewer views, the loss
    // closes the target: traversability-risk.md §6).
    EXPECT_LT(strict, hybrid);
    // The hybrid connects the two rooms through the doorway (8-connected flood from west to east).
    cv::Mat lab;
    cv::connectedComponents(t.trav_r, lab, 8, CV_32S);
    const GridFrame g = Frame();
    auto L = [&](double x, double y) { return lab.at<int>(int((y - g.oy) / g.cell), int((x - g.ox) / g.cell)); };
    ASSERT_NE(L(1.0, 1.5), 0);
    EXPECT_EQ(L(1.0, 1.5), L(3.0, 1.5));
}
