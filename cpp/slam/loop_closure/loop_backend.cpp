// The intra-agent loop backend (D73 part 2b): see loop_backend.hpp for what and why.
#include "slam/loop_closure/loop_backend.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace ariadne::slam {

namespace {

bool IsOdometry(EdgeType t) { return t == EdgeType::kOdometry || t == EdgeType::kStereoOdometry; }

double MsSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

void LoopBackend::Sync(const PoseGraph& vo) {
    if (vo.edges().size() < vo_edges_seen_) throw std::logic_error("LoopBackend::Sync: the VO graph lost edges");
    const Mat6 unknown = Mat6::Constant(std::numeric_limits<double>::quiet_NaN());
    for (size_t i = vo_edges_seen_; i < vo.edges().size(); ++i) {
        const PoseEdge& e = vo.edges()[i];
        if (!IsOdometry(e.type)) continue;
        if (!seeded_) {   // the gauge: the first node at its VO pose
            PoseNode n = vo.nodes().at(e.from);
            n.covariance = Mat6::Zero();
            g_.AddNode(n);
            anchor_ = e.from;
            seeded_ = true;
        }
        const bool has_from = g_.nodes().count(e.from), has_to = g_.nodes().count(e.to);
        if (!has_from && !has_to) {
            throw std::logic_error("LoopBackend::Sync: odometry edge " + std::to_string(e.from) + "->" +
                                   std::to_string(e.to) + " touches no mirrored node (broken chain)");
        }
        if (!has_to || !has_from) {   // dead-reckon the new node on the corrected trajectory: zero odometry residual
            PoseNode n = vo.nodes().at(has_from ? e.to : e.from);
            n.T_W_C = has_from ? g_.nodes().at(e.from).T_W_C * e.Z : g_.nodes().at(e.to).T_W_C * e.Z.inverse();
            n.covariance = unknown;
            g_.AddNode(n);
        }
        PoseEdge copy = e;
        copy.gnc_weight = 1.0;
        g_.AddEdge(copy);
    }
    vo_edges_seen_ = vo.edges().size();
    if (!p_.gnc.lm.gravity.enabled) return;
    for (size_t i = vo_priors_seen_; i < vo.gravity_priors().size(); ++i) pending_priors_.push_back(vo.gravity_priors()[i]);
    vo_priors_seen_ = vo.gravity_priors().size();
    bool added = false;
    for (auto it = pending_priors_.begin(); it != pending_priors_.end();) {
        if (g_.nodes().count(it->keyframe)) {
            g_.AddGravityPrior(*it);
            it = pending_priors_.erase(it);
            added = true;
        } else {
            ++it;
        }
    }
    if (added && !g_.edges().empty()) {
        PgoSettings s = p_.gnc.lm;
        s.recompute_covariance = false;
        SolvePoseGraph(g_, anchor_, s);
    }
}

LoopEventRecord LoopBackend::OnCandidates(const std::vector<LoopCandidate>& cands, int64_t t_ns) {
    LoopEventRecord rec;
    rec.t_ns = t_ns;
    auto t0 = std::chrono::steady_clock::now();
    for (const LoopCandidate& c : cands) {
        if (!Ready(c.a, c.b)) throw std::logic_error("LoopBackend: candidate keyframes are not in the mirror yet");
        LoopEventRecord::Candidate rc;
        rc.a = c.a;
        rc.b = c.b;
        rc.retrieval_score = c.retrieval_score;
        const DegeneracyReport d = AssessLoop(c.geometry, p_.gate);
        rc.degenerate = d.degenerate;
        rc.gate_reasons = d.reasons;
        rc.sigma_t = d.sigma_t_worst;
        rc.sigma_r = d.sigma_r_worst;
        rc.T_A_B = d.T_A_B;
        rc.cov_left = d.cov_left;
        if (!d.degenerate) {
            PoseEdge e;
            e.type = p_.loop_type;
            e.from = c.a;
            e.to = c.b;
            e.Z = d.T_A_B;
            e.information = d.information;
            g_.AddEdge(e);
            rc.edge = long(g_.edges().size() - 1);
        }
        rec.candidates.push_back(rc);
    }
    rec.ms_gate = MsSince(t0);
    bool added = false;
    for (const auto& rc : rec.candidates) added = added || rc.edge >= 0;
    if (!added) return rec;

    t0 = std::chrono::steady_clock::now();
    const PcmReport pcm = Pcm(g_, p_.pcm);
    std::vector<bool> active(g_.edges().size(), false);
    for (size_t l : pcm.accepted) active[l] = true;
    rec.ms_pcm = MsSince(t0);
    t0 = std::chrono::steady_clock::now();
    const GncReport gnc = SolveGnc(g_, anchor_, &active, p_.gnc, &state_);
    rec.ms_gnc = MsSince(t0);

    rec.solved = true;
    rec.fast_path = gnc.fast_path;
    rec.gnc_stop = gnc.stop;
    rec.innovation = gnc.innovation;
    rec.innovation_bound = gnc.innovation_bound;
    rec.loops_total = int(gnc.loops.size());
    rec.pcm_accepted = int(pcm.accepted.size());
    rec.gnc_inliers = int(gnc.inliers.size());
    for (auto& rc : rec.candidates) {
        if (rc.edge < 0) continue;
        const size_t e = size_t(rc.edge);
        rc.pcm_accepted = active[e];
        rc.gnc_inlier = std::find(gnc.inliers.begin(), gnc.inliers.end(), e) != gnc.inliers.end();
        rc.gnc_weight = g_.edges()[e].gnc_weight;
    }
    if (p_.certificate.enabled) {   // D74: read-only on the mirror, so the flag never changes the trajectory
        t0 = std::chrono::steady_clock::now();
        rec.certificate_ran = true;
        try {
            const CertificateReport c = CertifyPoseGraph(g_, anchor_, p_.certificate);
            rec.certified = c.certified;
            rec.certificate_verdict = c.verdict;
        } catch (const std::exception& ex) {   // e.g. a split active graph, a singular Laplacian: nothing certified
            rec.certificate_verdict = std::string("not run: ") + ex.what();
        }
        rec.ms_certificate = MsSince(t0);
    }
    return rec;
}

}  // namespace ariadne::slam
