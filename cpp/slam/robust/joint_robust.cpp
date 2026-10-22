// The robust stage on a joint multi-agent graph (D76): see joint_robust.hpp.
#include "slam/robust/joint_robust.hpp"

#include "slam/pose_graph/chordal_init.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace ariadne::slam {

std::optional<PoseEdge> GatedInterAgentEdge(uint64_t from, uint64_t to, const LoopGeometry& geometry,
                                            const DegeneracyParams& gate, DegeneracyReport* report) {
    const DegeneracyReport d = AssessLoop(geometry, gate);
    if (report) *report = d;
    if (d.degenerate) return std::nullopt;
    PoseEdge e;   // as LoopBackend::OnCandidates
    e.type = EdgeType::kInterAgentLoop;
    e.from = from;
    e.to = to;
    e.Z = d.T_A_B;
    e.information = d.information;
    return e;
}

JointRobustReport RobustJointSolve(PoseGraph& g, uint64_t anchor, const JointRobustParams& p, GncState* state) {
    JointRobustReport rep;
    rep.pcm = Pcm(g, p.pcm);
    std::vector<bool> active(g.edges().size(), false);
    for (size_t l : rep.pcm.accepted) active[l] = true;
    for (size_t k = 0; k < g.edges().size(); ++k)
        if (g.edges()[k].type == EdgeType::kInterAgentLoop) {
            rep.inter_agent.push_back(k);
            rep.inter_pcm_accepted += active[k];
        }

    if (p.reinit_chordal) {
        // The anchor's component over odometry + PCM-accepted loops; only it is re-initialised.
        std::map<uint64_t, uint64_t> parent;
        for (const auto& [id, n] : g.nodes()) parent[id] = id;
        auto find = [&parent](uint64_t x) {
            while (parent[x] != x) x = parent[x] = parent[parent[x]];
            return x;
        };
        auto kept = [&](size_t k) { return !IsLoopType(g.edges()[k].type) || active[k]; };
        for (size_t k = 0; k < g.edges().size(); ++k)
            if (kept(k)) parent[find(g.edges()[k].from)] = find(g.edges()[k].to);
        const uint64_t root = find(anchor);
        PoseGraph sub;
        for (const auto& [id, n] : g.nodes())
            if (find(id) == root) sub.AddNode(n);
        for (size_t k = 0; k < g.edges().size(); ++k)
            if (kept(k) && find(g.edges()[k].from) == root) {
                PoseEdge e = g.edges()[k];
                if (IsLoopType(e.type)) e.gnc_weight = 1.0;   // the init sees accepted loops at full weight
                sub.AddEdge(e);
            }
        if (!sub.edges().empty()) {
            const Mat6 unknown = Mat6::Constant(std::numeric_limits<double>::quiet_NaN());
            for (const auto& [id, T] : InitialPoses(sub, anchor, PgoInit::kChordal)) g.UpdateNode(id, T, unknown);
            rep.reinitialised = true;
            rep.reinit_nodes = int(sub.nodes().size());
        }
    }

    rep.gnc = SolveGnc(g, anchor, &active, p.gnc, state);
    for (size_t k : rep.inter_agent)
        rep.inter_inliers += std::find(rep.gnc.inliers.begin(), rep.gnc.inliers.end(), k) != rep.gnc.inliers.end();
    return rep;
}

}  // namespace ariadne::slam
