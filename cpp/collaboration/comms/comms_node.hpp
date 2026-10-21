#pragma once
// ---------------------------------------------------------------------------
// 3.3 (D92): the comms broker node -- the accounting/policy layer of arch §5.7.
//
// It OWNS, for one mission:
//   - the link stack: a base LinkModel (ideal, or RFComms-derived with or without wall attenuation;
//     rf_link_model.hpp) under a FaultPlan for scripted partitions and disconnects;
//   - the Network that every datagram crosses (COLLAB_3.2 §2), which enforces that link state;
//   - the ONE ByteLedger every Endpoint records into (§15.6, COLLAB_3.2 §2).
// Ledger rule: only Endpoints write the ledger, at their serialization boundary (first sends, retransmissions,
// heartbeats, acks). The node forwards and drops datagrams through the Network, which has no ledger at all, so a
// forwarded datagram can never be counted twice (tested: draining the network adds no rows).
//
// Scenario config (configs/comms/*.yaml):
//   name, mode: ideal | range_limited | attenuated
//   rf: {RfParams keys}              (range_limited ignores wall_loss_db)
//   partitions:  [{t0_s, t1_s, groups: [[agents], ...]}]   (arch §5.7 "partitioned")
//   disconnects: [{agent, t0_s, t1_s}]                    ("intermittent")
// Scenario times are seconds from `origin` (the mission's first sim-clock stamp).
// ---------------------------------------------------------------------------
#include "collaboration/comms/rf_link_model.hpp"
#include "collaboration/protocol/network.hpp"
#include "core/config/config.hpp"
#include "core/random/seed.hpp"
#include "core/serialization/serializer.hpp"

#include <yaml-cpp/yaml.h>

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace ariadne::collaboration {

struct CommsScenario {
    struct Partition { double t0_s, t1_s; std::vector<std::vector<AgentId>> groups; };
    struct Disconnect { AgentId agent; double t0_s, t1_s; };
    std::string name = "ideal";
    std::string mode = "ideal";   // ideal | range_limited | attenuated
    RfParams rf;
    std::vector<Partition> partitions;
    std::vector<Disconnect> disconnects;
};

inline CommsScenario LoadCommsScenario(const std::string& path) {
    const YAML::Node n = YAML::LoadFile(path);
    const auto fail = [&](const std::string& why) { throw core::ConfigError(path + ": " + why); };
    for (const auto& kv : n) {
        const std::string k = kv.first.as<std::string>();
        if (k != "name" && k != "mode" && k != "rf" && k != "partitions" && k != "disconnects")
            fail("unrecognized key '" + k + "'");
    }
    CommsScenario s;
    if (!n["name"] || !n["mode"]) fail("name and mode are required");
    s.name = n["name"].as<std::string>();
    s.mode = n["mode"].as<std::string>();
    if (s.mode != "ideal" && s.mode != "range_limited" && s.mode != "attenuated") fail("mode: ideal|range_limited|attenuated");
    if (n["rf"]) {
        const std::set<std::string> keys{"max_range_m", "fading_exponent", "l0_db", "sigma_db", "capacity_bps",
                                         "tx_power_dbm", "noise_floor_dbm", "wall_loss_db", "latency_s",
                                         "datagram_bytes"};
        for (const auto& kv : n["rf"]) {
            if (!keys.count(kv.first.as<std::string>())) fail("rf: unrecognized key '" + kv.first.as<std::string>() + "'");
        }
        const YAML::Node r = n["rf"];
        const auto get = [&](const char* k, double& v) { if (r[k]) v = r[k].as<double>(); };
        get("max_range_m", s.rf.max_range_m);
        get("fading_exponent", s.rf.fading_exponent);
        get("l0_db", s.rf.l0_db);
        get("sigma_db", s.rf.sigma_db);
        get("capacity_bps", s.rf.capacity_bps);
        get("tx_power_dbm", s.rf.tx_power_dbm);
        get("noise_floor_dbm", s.rf.noise_floor_dbm);
        get("wall_loss_db", s.rf.wall_loss_db);
        get("latency_s", s.rf.latency_s);
        get("datagram_bytes", s.rf.datagram_bytes);
        if (!(s.rf.sigma_db >= 0.0) || !(s.rf.capacity_bps > 0.0) || !(s.rf.datagram_bytes >= 1.0) ||
            !(s.rf.wall_loss_db >= 0.0) || !(s.rf.latency_s >= 0.0)) {
            fail("rf: sigma, wall loss, latency >= 0; capacity > 0; datagram_bytes >= 1");
        }
    }
    for (const auto& p : n["partitions"]) {
        CommsScenario::Partition q{p["t0_s"].as<double>(), p["t1_s"].as<double>(), {}};
        for (const auto& g : p["groups"]) q.groups.push_back(g.as<std::vector<AgentId>>());
        s.partitions.push_back(q);
    }
    for (const auto& d : n["disconnects"]) {
        s.disconnects.push_back({d["agent"].as<AgentId>(), d["t0_s"].as<double>(), d["t1_s"].as<double>()});
    }
    return s;
}

/// The base link model for a scenario. `tracks` and `walls` come from the sim (runtime/datasets/scene_walls.hpp).
inline std::unique_ptr<LinkModel> MakeLinkModel(const CommsScenario& s, std::map<AgentId, AgentTrack> tracks,
                                                std::vector<WallTriangle> walls) {
    if (s.mode == "ideal") return std::make_unique<ScriptedLinkModel>(LinkState{true, s.rf.latency_s, 0.0, 0.0});
    RfParams p = s.rf;
    if (s.mode == "range_limited") {
        p.wall_loss_db = 0.0;
        walls.clear();
    }
    return std::make_unique<RfLinkModel>(p, std::move(tracks), std::move(walls));
}

class CommsNode {
public:
    /// `base` must outlive the node. `seed` is the run seed (the loss draws use its comms_loss purpose).
    CommsNode(const LinkModel& base, uint64_t seed, Timestamp origin = {}, DatagramFaults faults = {})
        : origin_(origin), faults_(base),
          net_(faults_, core::DeriveSeed(seed, core::seed_purpose::kCommsLoss), faults) {}

    void Apply(const CommsScenario& s) {
        for (const auto& p : s.partitions) faults_.Partition(At(p.t0_s), At(p.t1_s), p.groups);
        for (const auto& d : s.disconnects) faults_.Disconnect(d.agent, At(d.t0_s), At(d.t1_s));
    }

    core::ByteLedger& ledger() { return ledger_; }   // hand this to every Endpoint; nothing else writes it
    const core::ByteLedger& ledger() const { return ledger_; }
    Network& network() { return net_; }
    const LinkModel& links() const { return faults_; }   // the effective link state (base + faults)

private:
    Timestamp At(double s) const { return origin_ + core::Duration::SecondsFromDouble(s); }

    Timestamp origin_;
    FaultPlan faults_;
    Network net_;
    core::ByteLedger ledger_;
};

}  // namespace ariadne::collaboration
