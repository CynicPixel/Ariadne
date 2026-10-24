// The broker's multi-agent solve: see multi_agent.hpp.
#include "slam/pose_graph/multi_agent.hpp"

#include <deque>
#include <set>
#include <stdexcept>

namespace ariadne::slam {

SE3 FrameFit(const std::vector<SE3>& T_W_C, const std::vector<SE3>& T_G_C) {
    if (T_W_C.empty() || T_W_C.size() != T_G_C.size()) throw std::invalid_argument("FrameFit: need matched, non-empty sets");
    core::Mat3 M = core::Mat3::Zero();
    for (size_t i = 0; i < T_W_C.size(); ++i) M += T_G_C[i].R * T_W_C[i].R.transpose();
    SE3 T;
    T.R = ProjectToSO3(M);
    core::Vec3 t = core::Vec3::Zero();
    for (size_t i = 0; i < T_W_C.size(); ++i) t += T_G_C[i].t - T.R * T_W_C[i].t;
    T.t = t / double(T_W_C.size());
    return T;
}

MultiAgentResult SolveMultiAgent(const MultiAgentInput& in) {
    MultiAgentResult out;
    std::map<uint32_t, const AgentSubgraph*> by_agent;
    std::map<uint32_t, std::map<uint64_t, const PoseNode*>> node_of;
    for (const auto& a : in.agents) {
        by_agent[a.agent] = &a;
        for (const auto& n : a.nodes) node_of[a.agent][n.keyframe] = &n;
    }
    if (!by_agent.count(in.gauge_agent) || by_agent.at(in.gauge_agent)->nodes.empty())
        throw std::invalid_argument("SolveMultiAgent: the gauge agent has no nodes");
    auto node = [&](uint32_t agent, uint64_t kf) -> const PoseNode& {
        const auto a = node_of.find(agent);
        if (a == node_of.end() || !a->second.count(kf))
            throw std::invalid_argument("SolveMultiAgent: inter-agent edge names an unknown node");
        return *a->second.at(kf);
    };
    for (const auto& e : in.edges) {
        node(e.agent_a, e.keyframe_a);
        node(e.agent_b, e.keyframe_b);
    }

    // 1-2. Placement and the starting T_G_W: spanning tree over agents along inter-agent edges, in input order.
    std::map<uint32_t, SE3> T_G_W0{{in.gauge_agent, in.gauge_T_G_W}};
    std::deque<uint32_t> q{in.gauge_agent};
    while (!q.empty()) {
        const uint32_t a = q.front();
        q.pop_front();
        for (const auto& e : in.edges)
            for (int dir = 0; dir < 2; ++dir) {
                const uint32_t from = dir ? e.agent_b : e.agent_a, to = dir ? e.agent_a : e.agent_b;
                if (from != a || T_G_W0.count(to)) continue;
                const SE3& Pf = node(from, dir ? e.keyframe_b : e.keyframe_a).T_W_C;
                const SE3& Pt = node(to, dir ? e.keyframe_a : e.keyframe_b).T_W_C;
                const SE3 Z = dir ? e.Z.inverse() : e.Z;
                T_G_W0[to] = T_G_W0.at(from) * Pf * Z * Pt.inverse();
                q.push_back(to);
            }
    }
    for (const auto& [a, sub] : by_agent)
        if (!T_G_W0.count(a)) out.unplaced.push_back(a);
    for (auto& [a, T] : T_G_W0)   // D178: a known-start prior replaces the spanning-tree start (not the gauge's)
        if (a != in.gauge_agent && in.prior_T_G_W.count(a)) T = in.prior_T_G_W.at(a);

    // 3. The joint graph over placed agents.
    PoseGraph g;
    for (const auto& [a, T] : T_G_W0) {
        for (const auto& n : by_agent.at(a)->nodes) {
            PoseNode j = n;
            j.agent = a;
            j.keyframe = JointId(a, n.keyframe);
            j.T_W_C = T * n.T_W_C;
            g.AddNode(j);
        }
        for (PoseEdge e : by_agent.at(a)->edges) {
            e.from = JointId(a, e.from);
            e.to = JointId(a, e.to);
            g.AddEdge(e);
        }
    }
    for (const auto& e : in.edges) {
        if (!T_G_W0.count(e.agent_a) || !T_G_W0.count(e.agent_b)) continue;
        PoseEdge pe;
        pe.type = EdgeType::kInterAgentLoop;
        pe.from = JointId(e.agent_a, e.keyframe_a);
        pe.to = JointId(e.agent_b, e.keyframe_b);
        pe.Z = e.Z;
        pe.information = e.information;
        g.AddEdge(pe);
    }
    uint64_t first_kf = by_agent.at(in.gauge_agent)->nodes.front().keyframe;
    for (const auto& n : by_agent.at(in.gauge_agent)->nodes) first_kf = std::min(first_kf, n.keyframe);
    const uint64_t anchor = JointId(in.gauge_agent, first_kf);

    // Nothing to optimise without an edge (e.g. a lone broker with one keyframe: zero free variables).
    out.initial_cost = out.final_cost = g.Cost();
    std::set<uint32_t> kept;
    for (const auto& [a, T] : T_G_W0) kept.insert(a);
    if (!g.edges().empty()) {
        if (in.solver) {
            kept = in.solver(g, anchor);
            kept.insert(in.gauge_agent);
        } else {
            if (in.init != PgoInit::kCurrent) ApplyPoses(g, InitialPoses(g, anchor, in.init));
            out.initial_cost = g.Cost();
            out.report = SolvePoseGraph(g, anchor, in.settings);
        }
        out.final_cost = g.Cost();
    }

    // 4. Per-agent outputs.
    for (const auto& [a, T0] : T_G_W0) {
        if (!kept.count(a)) {
            out.unplaced.push_back(a);
            continue;
        }
        std::vector<SE3> in_w, in_g;
        auto& kfs = out.T_G_C[a];
        for (const auto& n : by_agent.at(a)->nodes) {
            const SE3& T_G_C = g.nodes().at(JointId(a, n.keyframe)).T_W_C;
            kfs[n.keyframe] = T_G_C;
            in_w.push_back(n.T_W_C);
            in_g.push_back(T_G_C);
        }
        out.T_G_W[a] = a == in.gauge_agent ? in.gauge_T_G_W : FrameFit(in_w, in_g);
    }
    return out;
}

}  // namespace ariadne::slam
