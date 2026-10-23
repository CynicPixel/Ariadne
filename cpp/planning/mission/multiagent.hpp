#pragma once
// ---------------------------------------------------------------------------
// D175 (MULTIAGENT_PIPELINE §1-§3): the planning side of agents that coordinate through the radio. Each agent runs
// its own Phase4Planner on its OWN keyframes (Phase4Planner::PlanOwn); what it knows of its peers it RECEIVED:
//   SubmapPatch   (SUBMAP_PATCH, class 8)  the sender's own slab + rho_responder bit, changed 32x32 blocks, 2 bits/cell
//   TaskBid       (TASK_BID, class 9)      the sender's top-M options with BuildAuction's raw terms
//   TaskConsensus (TASK_CONSENSUS, 10)     the component broker's sequential-greedy assignment
// Plain structs, no comms type: M's wire tables mirror them field for field and CommsNode carries them, lossy,
// budgeted and partitionable; every byte is counted at the serializer (invariant 6), not here.
//   SlabExporter     own maps -> versioned blocks -> SubmapPatch (only blocks changed after the receiver's ack)
//   ReceivedLayers   peers' patches placed at T_G_W(j) -> the known mask (NBV gain, PeerKnown) and the merged layer
//                    (the operator's map; its egress at rho_responder, invariant 4)
//   BuildBid / BrokerAssign / ApplyConsensus   bids -> SGA over what arrived -> the member's pick, or "uncoordinated"
// Task identity across agents is spatial (r_same; frontier ids are per-agent hashes): the stated D175 deviation from
// CBBA's shared task list. Deterministic: sim time only, no wall clock.
// ---------------------------------------------------------------------------
#include "planning/mission/allocation.hpp"
#include "planning/mission/nbv.hpp"
#include "planning/volumetric/traversability.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ariadne::planning::multi {

constexpr int kBlock = 32;   // cells per block side (1.6 m on the 0.05 m briefing grid)

/// One cell of the exchanged layer (2 bits on the wire): the slab tri-state plus the rho_responder bit.
enum CellCode : uint8_t { kCellUnknown = 0, kCellOccupied = 1, kCellFree = 2, kCellFreeResponder = 3 };

struct SlabBlock {
    int32_t bx = 0, by = 0;        // cells [by*32, by*32+32) x [bx*32, bx*32+32) of the briefing grid
    std::vector<uint8_t> cells;    // kBlock * kBlock CellCodes, row-major (unpacked; the codec packs 2 bits/cell)
};

struct SubmapPatch {
    int32_t sender = 0;
    uint32_t epoch = 0;            // the sender's frame epoch
    uint64_t version = 0;          // the sender's map version at export
    int64_t t_ns = 0;              // sim time of the export
    std::vector<SlabBlock> blocks; // in the sender's frame W_j (the briefing grid)
};

struct BidOption {
    int64_t task_key = -1;         // the sender's task_id
    double goal_x = 0, goal_y = 0; // in G
    float ig_raw[5] = {0, 0, 0, 0, 0};
    float path_m = 0;
    float cost_bandwidth = 0, cost_risk = 0;   // D114 cost[1], cost[4]
    float risk = 0;
    uint8_t source = 0;            // TaskSource char
};

struct TaskBid {
    int32_t sender = 0;
    uint32_t epoch = 0;
    uint64_t seq = 0;
    int64_t t_ns = 0;
    float lambda_path = 1;
    std::vector<BidOption> options;   // descending utility, at most M
};

struct Assignment {
    int32_t agent = 0;
    int64_t task_key = -1;         // the assignee's own key
    double goal_x = 0, goal_y = 0;
};

struct TaskConsensus {
    int32_t broker = 0;
    uint32_t epoch = 0;
    uint64_t seq = 0;
    int64_t t_ns = 0;              // computed at
    int64_t bids_t_ns = 0;         // the replan the bids were from (the newest bid)
    std::vector<int32_t> members;
    std::vector<Assignment> assignments;
};

struct ComponentInfo {             // CommsNode's ComponentOf(agent): local API, not wire
    int32_t broker = 0;
    std::vector<int32_t> members;
    uint32_t epoch = 0;
};

struct PeerPose {                  // from HEARTBEAT / #ckf, in G: local API
    int32_t agent = 0;
    int64_t kf_id = 0;
    double x = 0, y = 0, z = 0;
    int64_t t_ns = 0;
};

struct Rigid2 {                    // the 2D part of T_G_W(j): planar maps under known starts (identity until a loop)
    double x = 0, y = 0, yaw = 0;
    bool Identity() const { return x == 0 && y == 0 && yaw == 0; }
};

/// The exchanged code of each grid cell from an agent's own maps (slab tri-state, rho_responder bit from trav_r).
cv::Mat EncodeCells(const TraversabilityMaps& m);

/// An agent's own layer, versioned per block.
class SlabExporter {
public:
    SlabExporter(int sender, const GridFrame& g);
    /// Re-encode from this agent's latest maps; every block whose codes changed gets the new version. Returns it.
    uint64_t Update(const TraversabilityMaps& m);
    /// The blocks changed after `since` (a receiver's acked version; 0 = everything ever observed).
    SubmapPatch Export(uint64_t since, uint32_t epoch, int64_t t_ns) const;
    uint64_t Version() const { return version_; }

private:
    int sender_;
    GridFrame g_;
    int bw_, bh_;
    cv::Mat codes_;                     // CV_8U, the grid
    std::vector<uint64_t> changed_;     // per block: the version it last changed at (0 = never observed)
    uint64_t version_ = 0;
};

/// What one agent has received from its peers, placed in G.
class ReceivedLayers {
public:
    ReceivedLayers(int self, int n_agents, const GridFrame& g);
    /// Place a peer's patch (in its frame W_j) at T_G_W(j), nearest cell; a newer block overwrites the older one.
    void Integrate(const SubmapPatch& p, const Rigid2& T_G_Wj = {});
    /// CV_8U, nonzero where some peer observed the column (free or occupied): PeerKnown's mask.
    const cv::Mat& KnownMask() const { return known_; }
    /// CellCode per cell over every peer (and `own`, if given): occupied if anyone says occupied, else free with the
    /// responder bit if anyone has it, else free, else unknown [DESIGN: conservative for the operator's egress].
    cv::Mat Merged(const cv::Mat* own = nullptr) const;
    uint64_t AckedVersion(int sender) const { return acked_.at(size_t(sender)); }

private:
    int self_;
    GridFrame g_;
    std::vector<cv::Mat> layer_;        // per peer: CellCodes in G
    std::vector<uint64_t> acked_;       // per peer: the newest version integrated
    cv::Mat known_;
};

/// The sender's top-M options as a bid (return / hold are outside the auction, as in BuildAuction), goals in G.
TaskBid BuildBid(int sender, uint32_t epoch, uint64_t seq, int64_t t_ns, const NbvPlan& nbv, int max_options,
                 const Rigid2& T_G_W = {});

/// The component broker's sequential greedy (arch §21.7's sanctioned shortcut) over the bids that ARRIVED from its
/// members: common scales = max over bidders of each raw term's p90 (BuildAuction's rule over the received options),
/// tasks = options within r_same of one another (greedy, in bid order), value = D114 G on the common scales net of
/// bandwidth and risk, L_t = 1. Bids from non-members or a different epoch are ignored; one bid per member (the newest).
TaskConsensus BrokerAssign(const std::vector<TaskBid>& bids, const ComponentInfo& comp, double r_same, int64_t t_ns,
                           uint64_t seq);

struct Pick {
    int option = -1;               // index into the agent's options, -1 none
    bool coordinated = false;
    std::string why;               // "coordinated" | "uncoordinated: <reason>"
};

/// The member's pick from the newest consensus it RECEIVED: its assigned task by key, else the option within r_same
/// of the assigned goal; otherwise (none received, older than timeout_s, another epoch, no assignment, the task gone)
/// its own best option, logged "uncoordinated".
Pick ApplyConsensus(const TaskConsensus* c, int self, uint32_t epoch, int64_t now_ns, double timeout_s,
                    const std::vector<TaskCandidate>& options, double r_same);

}  // namespace ariadne::planning::multi
