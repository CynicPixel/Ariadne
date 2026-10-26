#pragma once
// ---------------------------------------------------------------------------
// D175 item 4 glue: the planning library's plain structs (planning::multi, no comms type by design) to and from
// the wire's value structs (core::*Value, serializer.hpp). the wire tables mirror the planning structs field for field,
// so every function here is a mechanical copy - nothing is decided, rounded or reordered. The runtime
// (ariadne_live) is the only caller: planner_k's exports go out through CollabNode::SendPatch/SendBid/
// SendConsensus, and what NodeHooks::coordination delivers comes back through the From* functions.
// ---------------------------------------------------------------------------
#include "core/serialization/serializer.hpp"
#include "planning/mission/multiagent.hpp"

#include <utility>
#include <vector>

namespace ariadne::runtime::live {

namespace multi = planning::multi;

inline core::SubmapPatchValue ToWire(const multi::SubmapPatch& p, core::FrameTag frame) {
    core::SubmapPatchValue v;
    v.sender = p.sender, v.epoch = p.epoch, v.version = p.version, v.t_ns = p.t_ns;
    v.frame = frame;
    v.blocks.reserve(p.blocks.size());
    for (const auto& b : p.blocks) v.blocks.push_back({b.bx, b.by, b.cells});
    return v;
}

inline multi::SubmapPatch FromWire(const core::SubmapPatchValue& v) {
    multi::SubmapPatch p;
    p.sender = v.sender, p.epoch = v.epoch, p.version = v.version, p.t_ns = v.t_ns;
    p.blocks.reserve(v.blocks.size());
    for (const auto& b : v.blocks) {
        multi::SlabBlock sb;
        sb.bx = b.bx, sb.by = b.by, sb.cells = b.cells;
        p.blocks.push_back(std::move(sb));
    }
    return p;
}

inline core::TaskBidValue ToWire(const multi::TaskBid& b, core::FrameTag frame) {
    core::TaskBidValue v;
    v.sender = b.sender, v.epoch = b.epoch, v.seq = b.seq, v.t_ns = b.t_ns, v.lambda_path = b.lambda_path;
    v.frame = frame;
    v.options.reserve(b.options.size());
    for (const auto& o : b.options) {
        core::BidOptionValue ov;
        ov.task_key = o.task_key, ov.goal_x = o.goal_x, ov.goal_y = o.goal_y;
        for (int i = 0; i < 5; ++i) ov.ig_raw[size_t(i)] = o.ig_raw[i];
        ov.path_m = o.path_m, ov.cost_bandwidth = o.cost_bandwidth, ov.cost_risk = o.cost_risk;
        ov.risk = o.risk, ov.source = o.source;
        for (const auto& g : o.footprint) ov.footprint.push_back({g.bx, g.by, g.w});
        v.options.push_back(std::move(ov));
    }
    return v;
}

inline multi::TaskBid FromWire(const core::TaskBidValue& v) {
    multi::TaskBid b;
    b.sender = v.sender, b.epoch = v.epoch, b.seq = v.seq, b.t_ns = v.t_ns, b.lambda_path = v.lambda_path;
    b.options.reserve(v.options.size());
    for (const auto& ov : v.options) {
        multi::BidOption o;
        o.task_key = ov.task_key, o.goal_x = ov.goal_x, o.goal_y = ov.goal_y;
        for (int i = 0; i < 5; ++i) o.ig_raw[i] = ov.ig_raw[size_t(i)];
        o.path_m = ov.path_m, o.cost_bandwidth = ov.cost_bandwidth, o.cost_risk = ov.cost_risk;
        o.risk = ov.risk, o.source = ov.source;
        for (const auto& g : ov.footprint) o.footprint.push_back({g.bx, g.by, g.w});
        b.options.push_back(std::move(o));
    }
    return b;
}

inline core::TaskConsensusValue ToWire(const multi::TaskConsensus& c, core::FrameTag frame) {
    core::TaskConsensusValue v;
    v.broker = c.broker, v.epoch = c.epoch, v.seq = c.seq, v.t_ns = c.t_ns, v.bids_t_ns = c.bids_t_ns;
    v.members = c.members;
    v.frame = frame;
    v.assignments.reserve(c.assignments.size());
    for (const auto& a : c.assignments) v.assignments.push_back({a.agent, a.task_key, a.goal_x, a.goal_y});
    return v;
}

inline multi::TaskConsensus FromWire(const core::TaskConsensusValue& v) {
    multi::TaskConsensus c;
    c.broker = v.broker, c.epoch = v.epoch, c.seq = v.seq, c.t_ns = v.t_ns, c.bids_t_ns = v.bids_t_ns;
    c.members = v.members;
    c.assignments.reserve(v.assignments.size());
    for (const auto& a : v.assignments) c.assignments.push_back({a.agent, a.task_key, a.goal_x, a.goal_y});
    return c;
}

}  // namespace ariadne::runtime::live
