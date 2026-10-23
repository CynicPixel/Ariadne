#pragma once
// A synthetic multi-agent scenario for the 3.2 protocol tests (COLLAB_3.2 §8):
// agents circle a shared loop with phase offsets, so their keyframes revisit
// each other's places. Ground truth drives the injected hooks: retrieval
// returns other agents' keyframes within `overlap_m`, verification returns the
// exact relative pose, and the solve chains agent frames along edges (a
// spanning-tree alignment; the production solve is the multi-agent PGO).
#include "collaboration/protocol/collab_node.hpp"
#include "collaboration/protocol/outcome.hpp"

#include <cmath>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace ariadne::collaboration::testing {

inline core::Timestamp At(double s) { return core::Timestamp{int64_t(std::llround(s * 1e9))}; }

inline const core::WireCodec& FragCodec() {
    static const core::WireCodec c(core::Framing::kFragmented);
    return c;
}

inline core::SE3 Pose(double yaw, double x, double y, double z = 0.0) {
    core::SE3 T;
    T.R = Eigen::AngleAxisd(yaw, core::Vec3::UnitZ()).toRotationMatrix();
    T.t = core::Vec3(x, y, z);
    return T;
}

struct Scenario {
    size_t agents = 4;
    double keyframe_dt_s = 1.0;
    double keyframes_until_s = 60.0;
    double radius_m = 8.0;
    double omega = 0.25;          // rad/s along the loop
    double overlap_m = 0.8;
    size_t keypoints = 200;       // feature response size: ~6 datagrams
    bool advertise_features = false;   // 3.4: descriptors carry n_features / feature_bytes
    uint64_t first_keyframe = 0;  // ids below it are never released (a VO startup prefix, SIMCOLLAB_3.2 §1)

    /// Agent a's camera in the world at time t.
    core::SE3 T_World_C(size_t a, double t) const {
        const double th = omega * t + 1.7 * double(a);
        return Pose(th + M_PI / 2, radius_m * std::cos(th), radius_m * std::sin(th), 0.1 * double(a));
    }
    /// W_a = agent a's start pose (its odometry frame).
    core::SE3 T_World_W(size_t a) const { return T_World_C(a, 0.0); }
    core::SE3 T_W_C(size_t a, double t) const { return T_World_W(a).inverse() * T_World_C(a, t); }
    double KfTime(uint64_t kf) const { return double(kf) * keyframe_dt_s; }
    /// 1 cm / ~0.6 deg standard deviations: [rho; phi] information.
    static core::Mat6 Information() {
        core::Mat6 I = core::Mat6::Zero();
        I.diagonal() << 1e4, 1e4, 1e4, 1e4, 1e4, 1e4;
        return I;
    }

    NodeHooks Hooks(size_t self) const {
        NodeHooks h;
        const Scenario s = *this;
        h.retrieve = [s](const KeyframeRef& q, const std::map<KeyframeRef, core::GlobalDescriptorValue>& db) {
            std::vector<KeyframeRef> out;
            const core::Vec3 pq = s.T_World_C(q.agent, s.KfTime(q.keyframe)).t;
            for (const auto& [k, g] : db) {
                if (k.agent == q.agent) continue;
                if ((s.T_World_C(k.agent, s.KfTime(k.keyframe)).t - pq).norm() < s.overlap_m) out.push_back(k);
            }
            return out;
        };
        h.features = [s, self](uint64_t kf) {
            core::FeatureResponseValue f;
            f.keyframe_id = kf;
            for (size_t i = 0; i < s.keypoints; ++i) {
                f.keypoints.insert(f.keypoints.end(), {float(i % 640), float((i * 37 + kf) % 480)});
                for (int b = 0; b < 32; ++b) f.descriptors.push_back(uint8_t((i * 131 + b * 17 + kf * 7 + self) & 0xFF));
                f.points3d.insert(f.points3d.end(), {float(i) * 0.01f, 0.5f, 2.0f});
            }
            f.camera = {400, 400, 320, 240, 0.1f};
            return f;
        };
        h.verify = [s](const KeyframeRef& a, const core::FeatureResponseValue&, const KeyframeRef& b,
                       const core::FeatureResponseValue&) -> std::optional<LoopEdgeValue> {
            LoopEdgeValue e;
            e.agent_a = a.agent;
            e.keyframe_a = a.keyframe;
            e.agent_b = b.agent;
            e.keyframe_b = b.keyframe;
            e.measurement = ToPoseValue(s.T_World_C(a.agent, s.KfTime(a.keyframe)).inverse() *
                                        s.T_World_C(b.agent, s.KfTime(b.keyframe)));
            e.frame_a = {a.agent, 0};
            e.frame_b = {b.agent, 0};
            e.status = ariadne::wire::EdgeStatus::GEOMETRY_VERIFIED;
            e.inlier_count = 100;
            e.information = PackInformation(Information());
            return e;
        };
        h.solve = &AlignSolve;
        return h;
    }

    /// Spanning-tree frame alignment: the gauge agent fixed, other members
    /// reached along inter-agent edges (lowest edge key first, deterministic);
    /// unreached members already in G(anchor) keep their reported T_G_W.
    static SolveOutput AlignSolve(const SolveInput& in) {
        SolveOutput out;
        out[in.gauge_agent] = in.gauge_T_G_W;
        auto node_pose = [&](AgentId a, uint64_t kf) -> std::optional<core::SE3> {
            const auto it = in.graphs.find(a);
            if (it == in.graphs.end()) return std::nullopt;
            for (const auto& n : it->second.nodes)
                if (n.keyframe == kf) return FromPoseValue(n.pose);
            return std::nullopt;
        };
        std::deque<AgentId> q{in.gauge_agent};
        while (!q.empty()) {
            const AgentId a = q.front();
            q.pop_front();
            for (const auto& e : in.edges) {
                for (int dir = 0; dir < 2; ++dir) {
                    const AgentId from = dir ? e.agent_b : e.agent_a, to = dir ? e.agent_a : e.agent_b;
                    if (from != a || out.count(to)) continue;
                    const auto Pf = node_pose(from, dir ? e.keyframe_b : e.keyframe_a);
                    const auto Pt = node_pose(to, dir ? e.keyframe_a : e.keyframe_b);
                    if (!Pf || !Pt) continue;
                    core::SE3 Z = FromPoseValue(e.measurement);   // T_Ca_Cb
                    if (dir) Z = Z.inverse();
                    out[to] = out[from] * (*Pf) * Z * Pt->inverse();
                    q.push_back(to);
                }
            }
        }
        for (const auto& [a, g] : in.graphs)
            if (!out.count(a) && !g.agent_frames.empty() && g.anchor.agent == in.anchor)
                out[a] = FromPoseValue(g.agent_frames.front().T_G_W);
        return out;
    }
};

/// N collaboration nodes over one network, stepped on the sim clock.
struct Swarm {
    Scenario sc;
    std::unique_ptr<LinkModel> base;
    std::unique_ptr<FaultPlan> plan;
    std::unique_ptr<Network> net;
    core::ByteLedger ledger;
    std::vector<std::unique_ptr<CollabNode>> nodes;
    uint64_t next_kf = 0;

    Swarm(Scenario s, LinkState link, uint64_t seed, DatagramFaults df = {}, NodeParams np = {},
          const std::function<void(NodeHooks&)>& customize = {})
        : sc(s) {
        base = std::make_unique<ScriptedLinkModel>(link);
        plan = std::make_unique<FaultPlan>(*base);
        net = std::make_unique<Network>(*plan, seed, df);
        std::vector<AgentId> fleet;
        for (size_t a = 0; a < sc.agents; ++a) fleet.push_back(AgentId(a));
        for (size_t a = 0; a < sc.agents; ++a)
        {
            NodeHooks h = sc.Hooks(a);
            if (customize) customize(h);
            nodes.push_back(std::make_unique<CollabNode>(AgentId(a), fleet, FragCodec(), ledger, std::move(h), np));
        }
    }

    void Step(core::Timestamp t) {
        for (const auto& d : net->DeliverUntil(t)) nodes[d.to]->OnDatagram(d, t);
        const double ts = t.toSeconds();
        while (sc.KfTime(next_kf) <= ts + 1e-9 && sc.KfTime(next_kf) <= sc.keyframes_until_s) {
            for (size_t a = 0; a < sc.agents && next_kf >= sc.first_keyframe; ++a) {
                core::GlobalDescriptorValue g;
                g.keyframe_id = next_kf;
                g.descriptor.assign(256, 0.0f);
                if (sc.advertise_features) g.n_features = uint32_t(sc.keypoints), g.feature_bytes = 32;
                std::optional<OdometryEdge> odo;
                if (next_kf > sc.first_keyframe) {
                    odo = OdometryEdge{next_kf - 1,
                                       sc.T_W_C(a, sc.KfTime(next_kf - 1)).inverse() * sc.T_W_C(a, sc.KfTime(next_kf)),
                                       Scenario::Information()};
                }
                nodes[a]->AddKeyframe(next_kf, int64_t(sc.KfTime(next_kf) * 1e9), sc.T_W_C(a, sc.KfTime(next_kf)), odo,
                                      g, t);
            }
            ++next_kf;
        }
        for (auto& n : nodes) n->Tick(t, *net);
    }
    void RunUntil(double t0, double t1, double dt = 0.05) {
        for (double t = t0; t <= t1 + 1e-9; t += dt) Step(At(t));
        now = At(t1);
    }
    core::Timestamp now;
    OutcomeReport Outcome() const {
        std::vector<const CollabNode*> v;
        for (const auto& n : nodes) v.push_back(n.get());
        return EvaluateOutcome(v, now);
    }
};

}  // namespace ariadne::collaboration::testing
