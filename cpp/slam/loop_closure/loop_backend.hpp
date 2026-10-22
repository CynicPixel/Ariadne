#pragma once
// ---------------------------------------------------------------------------
// The intra-agent loop backend (D73 part 2b; design ROBUST_BACKEND_2.5.md §8): degeneracy gate (D71) -> PCM (D72)
// -> GNC-TLS with weight reuse (D73 parts 1, 2a), in that inviolable order (arch §14.5), on a MIRROR of the VO's
// pose graph. Plain slam types only (the runtime adapter does features, retrieval storage and verification).
//
// THE MIRROR. The VO's graph is never written (read-only: VO outputs stay byte-identical with loops on or off). The
// backend keeps its own graph: Sync() copies the VO's odometry edges as they become final and places each new node
// at X_mirror(from) * Z, i.e. dead-reckoned on the CORRECTED trajectory. That adds zero odometry residual, so the
// previous event's optimum stays the optimum without the new loops, which part 2a's innovation test relies on.
// In window mode an odometry edge appears when its keyframe LEAVES the BA window, so a keyframe enters the mirror
// about one window after it was made: loops wait for it (Ready()), and that wait is part of their latency.
//
// EDGE CONVENTION. A = the keyframe whose PIXELS matched, B = the one whose 3D points did (p_A = T_A_B p_B). The
// loop edge goes A -> B with Z = the gate's refined T_A_B and the gate's information (residual tangent).
// ---------------------------------------------------------------------------
#include "slam/pose_graph/pose_graph.hpp"
#include "slam/robust/certificate.hpp"
#include "slam/robust/degeneracy.hpp"
#include "slam/robust/gnc.hpp"
#include "slam/robust/pcm.hpp"

#include <string>
#include <vector>

namespace ariadne::slam {

struct LoopCandidate {
    uint64_t a = 0, b = 0;   // A: pixels, B: 3D points
    LoopGeometry geometry;
    double retrieval_score = 0;
};

struct LoopBackendParams {
    DegeneracyParams gate;
    PcmParams pcm;
    GncParams gnc;
    EdgeType loop_type = EdgeType::kLocalLoop;
    CertificateOptions certificate;   // D74, off by default: run after each solved event, read-only
};

/// One loop event (all candidates of one query keyframe), for the decision trace.
struct LoopEventRecord {
    int64_t t_ns = 0;
    struct Candidate {
        uint64_t a = 0, b = 0;
        double retrieval_score = 0;
        bool degenerate = true;
        std::vector<std::string> gate_reasons;
        double sigma_t = 0, sigma_r = 0;
        SE3 T_A_B;                // the gate's refined edge and its left covariance (for ground-truth labelling)
        Mat6 cov_left = Mat6::Zero();
        long edge = -1;           // edge index in the mirror if added
        bool pcm_accepted = false;
        bool gnc_inlier = false;
        double gnc_weight = 0;
    };
    std::vector<Candidate> candidates;
    int loops_total = 0, pcm_accepted = 0, gnc_inliers = 0;
    bool solved = false, fast_path = false;
    std::string gnc_stop;
    double innovation = 0, innovation_bound = 0;
    double ms_gate = 0, ms_pcm = 0, ms_gnc = 0;
    bool certificate_ran = false, certified = false;   // D74 (certificate.enabled)
    std::string certificate_verdict;
    double ms_certificate = 0;
};

class LoopBackend {
public:
    explicit LoopBackend(const LoopBackendParams& p = {}) : p_(p) {}

    /// Copy the VO graph's new final odometry edges (append-only), dead-reckoning new nodes on the corrected
    /// trajectory. Throws if an edge's from-node is unknown to the mirror (a broken chain is a bug, never skipped).
    /// With gravity priors on (gnc.lm.gravity.enabled, D73 part 2c) the VO's GRAVITY_PRIOR edges are copied too and
    /// the mirror is re-solved (current loop weights, no GNC) whenever new ones land: the priors act between and
    /// without loops, and the next loop event again starts from the optimum without its new loops (part 2a).
    void Sync(const PoseGraph& vo);

    /// Both keyframes are in the mirror (their odometry is final).
    bool Ready(uint64_t a, uint64_t b) const { return g_.nodes().count(a) && g_.nodes().count(b); }

    /// The keyframe can never be mirrored: the mirror grows forward from its anchor (the first final odometry edge)
    /// and a gap after the anchor throws in Sync, so an unmirrored keyframe before the anchor never will be. The VO
    /// leaves such a prefix when its window solve is skipped before inertial init (D87).
    bool Unreachable(uint64_t k) const { return seeded_ && k < anchor_ && !g_.nodes().count(k); }

    /// Gate every candidate, add the survivors, then PCM + GNC once for the batch. Candidates not Ready() throw.
    LoopEventRecord OnCandidates(const std::vector<LoopCandidate>& cands, int64_t t_ns);

    const PoseGraph& graph() const { return g_; }
    uint64_t anchor() const { return anchor_; }

private:
    LoopBackendParams p_;
    PoseGraph g_;
    GncState state_;
    size_t vo_edges_seen_ = 0;
    size_t vo_priors_seen_ = 0;
    std::vector<GravityPrior> pending_priors_;   // keyframe not mirrored yet: queued, never dropped
    bool seeded_ = false;
    uint64_t anchor_ = 0;
};

}  // namespace ariadne::slam
