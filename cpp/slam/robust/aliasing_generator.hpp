#pragma once
// ---------------------------------------------------------------------------
// Perceptual-aliasing outlier generator (D75; design ROBUST_BACKEND_2.5.md §6). Test and evaluation support:
// header-only, never linked into a runtime target. Reused by 3.5 for inter-agent groups (JointId nodes, chains =
// one per agent, type = kInterAgentLoop).
//
// WHY. Random SE(3) outliers are the easy case: no two agree, so PCM's maximum clique and GNC's convex start both
// discard them (D73: no breakpoint up to 90 %). Real false loops come from PERCEPTUAL ALIASING: two different
// places look alike (two identical corridors), so a whole run of consecutive keyframes on segment B matches the
// corresponding keyframes on segment A. Those false loops agree with each other and with the odometry. This is the
// "grouped" outlier policy of Sünderhauf & Protzel (ICRA 2013, the Vertigo benchmark: random / local /
// random-grouped / local-grouped), and the failure mode of consistency-based rejection that DC-GM (Lajoie, Hu,
// Beltrame, Carlone, RA-L 2019) models explicitly.
//
// CONSTRUCTION. Segments A = (a_0 .. a_{m-1}) and B = (b_0 .. b_{m-1}) are runs of consecutive nodes along chains.
// One spurious transform Delta = T_{a_0} J T_{b_0}^-1 ("b_0 IS a_0, up to a small offset J") gives every member
//     Z_k = Exp(noise) * T_{a_k}^-1 Delta T_{b_k}        (T = ground truth),
// so any two members close a cycle through the TRUE odometry exactly (up to loop noise):
//     (T_{a_k}^-1 T_{a_l}) Z_l (T_{b_l}^-1 T_{b_k}) Z_k^-1 = I.
// PCM therefore accepts a group as one clique (asserted in the generator's test); only the global geometry (the
// odometry between A and B, or the true loops) contradicts it. Different groups use different Delta, so groups
// disagree with each other and with the true loops.
//
// OpenCV-free.
// ---------------------------------------------------------------------------
#include "core/math/lie.hpp"
#include "slam/pose_graph/pose_graph.hpp"

#include <Eigen/Cholesky>

#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace ariadne::slam {

struct AliasingGroupSpec {
    int size = 10;                         // members (consecutive keyframes on each segment)
    Mat6 loop_cov = Mat6::Identity();      // left covariance of each member's measurement; also its information
    EdgeType type = EdgeType::kLocalLoop;  // kInterAgentLoop for 3.5
    double min_separation_m = 3.0;         // the segments' true start positions are at least this far apart
    double offset_t_m = 0.3, offset_r_rad = 0.05;   // J: the apparent revisit's offset (uniform per axis, +-)
    bool cross_chain_only = false;         // 3.5: A and B on different chains (agents)
    int max_tries = 1000;                  // rejection sampling of the segment pair
};

struct PlantedGroup {
    size_t chain_a = 0, chain_b = 0, start_a = 0, start_b = 0;   // segment starts (indices into the chains)
    core::SE3 Delta;
    std::vector<size_t> edges;                                  // indices into the graph's edges
};

/// Left-noise sample with covariance `cov`.
template <class Rng>
Vec6 DrawLeftNoise(Rng& rng, const Mat6& cov) {
    std::normal_distribution<double> n(0.0, 1.0);
    Vec6 z;
    for (int i = 0; i < 6; ++i) z(i) = n(rng);
    return Eigen::LLT<Mat6>(cov).matrixL() * z;
}

/// One true loop a -> b: the ground-truth relative pose with left noise `cov`.
template <class Rng>
size_t AddTrueLoop(PoseGraph& g, const std::map<uint64_t, core::SE3>& truth, uint64_t a, uint64_t b, const Mat6& cov,
                   EdgeType type, Rng& rng) {
    const core::SE3 Z = core::SE3Exp(DrawLeftNoise(rng, cov)) * (truth.at(a).inverse() * truth.at(b));
    g.AddEdge(OdometryEdge(a, b, Z, cov, type));
    return g.edges().size() - 1;
}

/// Plants one aliasing group (see header) on two segments drawn from `chains` (ordered node ids; every id must be
/// in `truth` and in `g`). Segments do not overlap and their starts are >= min_separation_m apart in the truth.
/// Returns nullopt if no admissible pair was found in max_tries draws (the graph is then unchanged).
template <class Rng>
std::optional<PlantedGroup> PlantAliasingGroup(PoseGraph& g, const std::map<uint64_t, core::SE3>& truth,
                                               const std::vector<std::vector<uint64_t>>& chains,
                                               const AliasingGroupSpec& s, Rng& rng) {
    if (s.size < 1 || chains.empty()) throw std::invalid_argument("PlantAliasingGroup: empty group or no chains");
    std::uniform_int_distribution<size_t> pick_chain(0, chains.size() - 1);
    for (int attempt = 0; attempt < s.max_tries; ++attempt) {
        PlantedGroup p;
        p.chain_a = pick_chain(rng);
        p.chain_b = pick_chain(rng);
        if (s.cross_chain_only && p.chain_a == p.chain_b) continue;
        const auto &A = chains[p.chain_a], &B = chains[p.chain_b];
        if (A.size() < size_t(s.size) || B.size() < size_t(s.size)) continue;
        p.start_a = std::uniform_int_distribution<size_t>(0, A.size() - size_t(s.size))(rng);
        p.start_b = std::uniform_int_distribution<size_t>(0, B.size() - size_t(s.size))(rng);
        if (p.chain_a == p.chain_b) {   // same chain: the two runs must not overlap
            const size_t lo = std::min(p.start_a, p.start_b), hi = std::max(p.start_a, p.start_b);
            if (hi < lo + size_t(s.size)) continue;
        }
        const core::SE3 &Ta0 = truth.at(A[p.start_a]), &Tb0 = truth.at(B[p.start_b]);
        if ((Ta0.t - Tb0.t).norm() < s.min_separation_m) continue;
        std::uniform_real_distribution<double> ut(-s.offset_t_m, s.offset_t_m), ur(-s.offset_r_rad, s.offset_r_rad);
        Vec6 j;
        j << ut(rng), ut(rng), ut(rng), ur(rng), ur(rng), ur(rng);
        p.Delta = Ta0 * core::SE3Exp(j) * Tb0.inverse();
        for (int k = 0; k < s.size; ++k) {
            const uint64_t a = A[p.start_a + size_t(k)], b = B[p.start_b + size_t(k)];
            const core::SE3 Z = core::SE3Exp(DrawLeftNoise(rng, s.loop_cov)) * (truth.at(a).inverse() * p.Delta * truth.at(b));
            g.AddEdge(OdometryEdge(a, b, Z, s.loop_cov, s.type));
            p.edges.push_back(g.edges().size() - 1);
        }
        return p;
    }
    return std::nullopt;
}

}  // namespace ariadne::slam
