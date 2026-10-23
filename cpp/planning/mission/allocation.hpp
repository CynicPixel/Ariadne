#pragma once
// ---------------------------------------------------------------------------
// 4.4 allocation (arch §21.7; design ALLOCATION_4.4, D115): CBBA (Choi, Brunet & How, T-RO 2009; ALGORITHMS
// §22.3.1: Algorithm 3, eqs. 3-6, Table I) and the broker's sequential greedy (SGA, the paper's Theorem 1
// comparator), over one abstract instance, with exact optima for grading.
//
//   score     S_i(p) = sum_{j in p} cbar_ij exp(-rate_i d_ij(p)), d_ij(p) the distance along p (agent -> p_0 -> ...).
//             DMG by the paper's eqs. 11-12 when the legs satisfy the triangle inequality (geodesics do).
//   compare   one comparator everywhere: bid, then the lower agent id (§III-B's lexicographic tie-break); BuildBundle's
//             argmax over tasks breaks ties by the lower task index.
//   rounds    synchronous: every agent builds its bundle, then every agent receives its neighbours' snapshots
//             (Table I per task, eq. 5 timestamps), then releases (eq. 6). Timestamps are round indices (sim time live).
// Planning-level and deterministic; the wire codec is a function object (bytes at the serializer, invariant 6).
//
// Measured (D115 correction, test_allocation.cpp): CBBA's assignment equals SGA's exactly at L_t = 1. At L_t = 2 it
// can differ: a second-slot bid placed on top of a first task the bidder later loses (eq. 6 releases it) can outbid
// a peer, whose bundle then fills with a lower task and never re-enters BuildBundle (Algorithm 3 adds only while
// |b_i| < L_t). Every such divergence carries an eq. 6 tail release; the regression instance pins the mechanism.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ariadne::planning {

struct AuctionInstance {
    int n_agents = 0;
    std::vector<int64_t> task_ids;                        // N_t stable ids (exactly-once identity)
    std::vector<std::vector<double>> value;               // [agent][task] cbar >= 0; 0 = the agent cannot do it
    std::vector<std::vector<double>> d0;                  // [agent][task] distance agent -> task goal
    std::vector<std::vector<std::vector<double>>> leg;    // [agent][task][task] goal -> goal distance
    std::vector<double> rate;                             // [agent] discount per distance unit (lambda_path / P_path)
    int max_bundle = 2;                                   // L_t
    int Tasks() const { return int(task_ids.size()); }
};

/// S_i(p) for agent a's path p (task indices, in visiting order).
double PathScore(const AuctionInstance& in, int a, const std::vector<int>& path);

/// The marginal of adding task j to path p at its best position (eq. 3), and that position; j must not be in p.
double Marginal(const AuctionInstance& in, int a, const std::vector<int>& path, int j, int* best_pos = nullptr);

/// The single comparator: (bid_a, agent_a) beats (bid_b, agent_b). agent -1 = no winner, beaten by any positive bid.
bool Outbids(double bid_a, int agent_a, double bid_b, int agent_b);

struct AgentAssignment {
    std::vector<int> bundle, path;   // task indices
    std::vector<double> bids;        // y at the bundle's tasks, in bundle order
};

struct Allocation {
    std::vector<AgentAssignment> agents;
    int rounds = 0;
    size_t messages = 0, bytes = 0;
    bool converged = false;
    double total = 0;                // sum over agents of PathScore(path)
    std::vector<int> winner;         // per task: the agent, -1 none
    size_t tail_releases = 0;        // CBBA: tasks reset by eq. 6 at n > n_bar (bids placed on a lost earlier task)
};

/// Broker SGA: repeatedly the (agent, task) with the largest marginal (comparator above), inserted at its best
/// position, until bundles fill or no positive marginal remains.
Allocation SequentialGreedy(const AuctionInstance& in);

/// One CBBA message: a snapshot of the sender's (y, z, s) (and, in the full system, the auction scales).
struct CbbaMessage {
    int sender = 0;
    std::vector<int64_t> task_ids;
    std::vector<double> y;
    std::vector<int> z;
    std::vector<int64_t> s;   // per agent
};

/// Table I's three actions.
enum class CbbaAction { kLeave, kUpdate, kReset };

/// Table I, receiver i, sender k, task j: z_kj / z_ij in {agent id, -1 = none}, y and the timestamps s (indexed by
/// agent). Pure, so tested row by row.
CbbaAction TableI(int i, int k, int z_kj, int z_ij, double y_kj, double y_ij, const std::vector<int64_t>& s_k,
                  const std::vector<int64_t>& s_i);

struct CbbaParams {
    int max_rounds = 1000;
    std::function<size_t(const CbbaMessage&)> bytes;   // the codec's size; null = count messages only
    std::function<CbbaMessage(const CbbaMessage&)> transport;   // what the receiver gets; null = the snapshot itself
};

/// Synchronous CBBA from an empty state. graph(t) is the round-t adjacency (symmetric; self loops implied).
Allocation RunCbba(const AuctionInstance& in, const std::function<std::vector<std::vector<int>>(int)>& graph,
                   const CbbaParams& p);

/// Exact optimum by brute force (every task to one agent or none, bundles <= L_t, each at its best order). Small
/// instances only (tests).
double BruteForceOptimum(const AuctionInstance& in);

/// Exact optimum for 2 agents (any L_t <= 2) by the sorted disjoint scan (ALLOCATION_4.4 §6).
double TwoAgentOptimum(const AuctionInstance& in);

/// The exact test codec of ALLOCATION_4.4 §5: per task 8 (id) + 4 (y) + 1 (z), per agent 8 (s), 16 framing.
size_t TestCodecBytes(const CbbaMessage& m);

}  // namespace ariadne::planning
