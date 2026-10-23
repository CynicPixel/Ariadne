# MULTIAGENT_PIPELINE: agents that coordinate through the radio (D175)

**Owner:** M (overall design, comms classes, live collab fleet). **A1:** per-agent planning, coverage summaries, allocation
over messages. **B:** the live runtime wiring and the console.
**Status:** design. No new algorithms. This wires what exists so the thesis holds in the live system.

## 0. The gap this closes

Thesis: agents coordinate through limited communication to map the whole space with minimal exchange. Each
subsystem exists and is tested, but the live loop is centralised end to end:
- **One planner sees every agent's keyframes.** One shared TSDF/ESDF; the peers list uses every agent's full track.
- **Allocation is computed in one process.** CBBA runs inside `Phase4Planner::Allocate`. It counts the bytes its
  messages would cost, but never sends them.
- **Alignment is the briefing's constant, with `linked = true`.** No SLAM collaboration runs live.

The architecture already specifies the decentralised design, so this document implements it:
- **§21.7:** each agent scores its own candidates; bids; consensus resolves duplicate ownership; event-triggered;
  consensus messages ride class 2 of §15.3. Broker-connected components may shortcut to a sequential greedy auction
  at the broker, logged identically.
- **§15.3:** a per-peer outbox, strict priorities, and `SUBMAP_SUMMARY` / `SUBMAP_PATCH` (map sharing, class 5).
- **wire.fbs:** classes 7–10 (`SUBMAP_SUMMARY`, `SUBMAP_PATCH`, `TASK_BID`, `TASK_CONSENSUS`) are reserved in the
  enum but have no tables.

## 1. The rule: an agent knows only what it observed or received

Per agent k, onboard (in the sim, k's worker plus a planner instance owned by k):

```text
own sensors ──> VO + CollabFrontend (worker k) ──> own keyframes
                                                     │
              ┌──────────────────────────────────────┼─────────────────────────────┐
              v                                      v                             v
   planner_k.Integrate(own kf, at T_G_W(k))   CollabNode k (SLAM collab,   SUBMAP_SUMMARY out (own
                                              LIVE_COLLAB, D174)           coverage, 0.5 m, §20.6)
   planner_k.IntegrateSummary(peer j sum,    T_G_W(j) for every j it      TASK_BID out to k's broker
              at T_G_W(j) as k knows it)      has an estimate of
   peers_k = heartbeat poses of the agents k hears
              │
              v
   planner_k.Plan(t, own + received)  ──>  own options  ──>  TASK_BID  ──>  broker of k's component
                                                                             │ sequential greedy (SGA)
   planner_k executes its assignment  <──  TASK_CONSENSUS  <─────────────────┘
   (no assignment within timeout: its own best option, logged "uncoordinated")
```

Nothing reaches planner_k except through CommsNode, the same lossy, budgeted, partitionable link the SLAM
collaboration uses. Every byte is counted at the serializer and tagged by class (invariant 6).

**Peer alignment.** Entry poses are known (maintainer, 2026-10-23: drones inserted through windows), so every agent
holds an entry-pose T_G_W(j) for each peer from t = 0, and peer summaries are placeable before any SLAM collaboration.
Once LIVE_COLLAB (D174) runs, adopted broker estimates refine those values. The POSE_GRAPH estimate already carries
every member's frame (`agent_frames`); CollabNode is changed to keep the peers' values as well as its own, with no
wire change.

**The operator's map.** The operator station is a comms node at the building entrance. It has no sensors and no vote
in allocation, and receives the agents' coverage summaries over the same link model. The console's top-view map and splat
therefore show what the operator has RECEIVED, and agents out of radio range stop updating it, which is the honest
behaviour. A separate view, labelled "ground truth / evaluation", shows the GT coverage for comparison and is never
presented as the system's map.

## 2. Messages (M adds the tables, codec, ledger classes and CommsNode routing)

| Class | Content | Cadence | Priority (§15.3) |
|---|---|---|---|
| `HEARTBEAT` (exists) | pose summary in G (already in §15.2), plus the keyframe watermark | 1 s | 1 |
| `SUBMAP_SUMMARY` (7) | arch §20.6: the sender's OWN coverage on the 0.5 m global coarse layer (occupancy / uncertainty states), the cells changed since the last summary each receiver acked, run-length coded + zstd, in the sender's frame W_j, tagged with frame and epoch. Never raw maps (§15.1). `SUBMAP_PATCH` (8) stays reserved | event: ≥ N new coarse cells or ≥ 5 s | 5 (the first class cancelled under budget, §20.6) |
| `TASK_BID` (9) | the sender's top-M options: goal (x, y) in G, task key, utility terms on the common scales | per replan (5 s) or on event | 2 |
| `TASK_CONSENSUS` (10) | the broker's assignment for its component: per member, a task key and goal, plus the component member set and epoch | after each SGA round | 2 |

- **Task identity across agents.** Tasks are frontier clusters, and each agent's clusters come from its own map, so
  ids differ between agents. The broker treats two bids whose goals lie within `r_same` as the same task when it
  enforces exactly-once. `r_same` = the planner's `min_goal_m` (1.0 m in live_v1): goals closer than that are already
  "the same place" to the planner. This approximates CBBA's shared task list, and the deviation is stated.
- **Byte budget.** One link budget per agent (D170's token bucket), split into two shares, each with its own bucket:
  - SLAM collaboration (classes 1, 3, 4) gets 1 - s_x, and the spectral policy runs inside this share as today;
  - exploration (classes 7–10) gets s_x. Under it, SUBMAP_SUMMARY is cancellable, oldest first, and bids are not.
  - Heartbeats, pose edges and pose-graph batches stay outside both shares (small, "the product", §15.3).
  - s_x = 0.3 [ASSUMPTION], fixed for every row and reported. Per-class bytes are reported so the split can be
    judged.
- **Frames.** Patches stay in the sender's frame W_j. Receivers place them with their current T_G_W(j). When an
  adopted estimate moves T_G_W(j) by more than (`reanchor_m`, `reanchor_deg`), the receiver re-integrates that
  sender's stored summaries. Known entry poses make this rare: loops correct drift, not merge frames.

## 3. Allocation over messages: broker SGA first, full CBBA as the ablation

Broker SGA per connected component (§21.7's sanctioned shortcut):
- Members send `TASK_BID`.
- The broker runs `JointPick`'s exactly-once greedy over the bids it received, with spatial dedup.
- It returns `TASK_CONSENSUS`.
- Under a partition, each component's broker allocates its own members. Duplicates across components are possible
  and are the honest cost of the partition; they are counted.
- `cbba_distributed` (each agent runs its own CBBA state over delivered TASK_CONSENSUS messages) is the second
  allocation row. It needs the same messages plus per-agent CBBA state; built only if time allows.
- The current centralised Phase4Planner stays as the **oracle row**: full information, zero bytes. It is the upper
  bound the comms-limited system is compared against.

## 4. Experiment rows (the thesis evidence)

Same scene, seeds and starts. Lockstep, so the comparison is deterministic.

**Pre-registered required set** (written before any data):
- oracle, independent and ariadne, at one budget level;
- the level is D171's derived 25 %, applied to the total, with the share split above;
- 2 scenes × 3 seeds = 18 jobs, about 2 h on two lanes;
- the partition row and the budget sweep are extras, run only if time allows.

**Pre-registered pass criterion:** on time-to-90 % coverage (mean over the 6 runs), ariadne closes at least 50 % of
the gap between independent and oracle. Also reported: bytes, redundancy and duplicate tasks. Failing the criterion
is reported, not tuned away.

| Row | Map shared | Allocation | What it shows |
|---|---|---|---|
| oracle | everything (centralised) | centralised CBBA | upper bound |
| independent | nothing | each agent its own best | lower bound: no communication |
| **ariadne** | `SUBMAP_SUMMARY` (§20.6) under budget | broker SGA over messages | the thesis |
| ariadne, partition | as ariadne, `collapsed_partition_v1` | per-component brokers | graceful degradation |
| ariadne, budget sweep | budget at {10, 25, 50, 100} % | as ariadne | coverage vs bytes |

Metrics, all computed by the existing mission eval:
- coverage of the GT free space vs time;
- time to 90 % coverage;
- redundant observation ratio (cells seen by ≥ 2 agents);
- duplicate-task count;
- total bytes by class;
- merged ATE (SLAM collab).

The claim is made only if "ariadne" sits between independent and oracle, closer to the oracle, at a stated byte cost.
Otherwise the result is reported as measured (invariant 7).

## 5. Determinism and replay

- **Sim clock everywhere.** Patch and bid cadences are in sim time, and CommsNode delivery is sim-timed and seeded.
- **Lockstep barrier.** At the 5 s barrier, every agent's planner runs after the comms tick for T, so what each one
  received is a function of the seed. The LIVE_COLLAB §3 watermark rules apply to keyframe and feature data.
- **Acceptance.** Three replays equal the live run; the `collab: false` digest is unchanged (D168).

## 6. Ownership and order (final submission, Oct 26)

| # | Who | What | Depends on |
|---|---|---|---|
| 1 | M | wire tables and codec for classes 7–10, CommsNode routing to per-agent mailboxes, delivery events, ledger tags; protocol tests | — |
| 2 | M | LIVE_COLLAB fleet (D174): CollabFrontend in the worker, CollabFleet in the parent, T_G_W per agent | — |
| 3 | A1 | per-agent `Phase4Planner` instances, `ExportSummary` / `IntegrateSummary` on the §20.6 0.5 m coarse layer (planner_k plans on own fine map + peers' coarse coverage), the task key, `TASK_BID` building, the broker SGA over received bids, the uncoordinated fallback; the identity check that the oracle row is unchanged | 1 (message structs) |
| 4 | B | ariadne_live: one planner per agent fed only through its mailbox; heartbeat poses as peers; the operator node and its received map; the console shows bytes by class, assignments and "uncoordinated" | 1, 3 (NOT 2: entry poses place the summaries) |
| 5 | M | §4 required rows in lockstep on AWS (CPU), then evidence and plots | 1, 3, 4 |

Item 2 (LIVE_COLLAB) runs in parallel. It adds drift correction and the SLAM-collaboration panel, and it is not on
the thesis rows' critical path.

**Compute.** N planners at ~3–6 s CPU per 5 s replan is ~0.6–1.2 core per agent, on top of LIVE_COLLAB §6. With
3 agents, the judge box carries:

| Component | Cores |
|---|---|
| gz | ~2–4 |
| estimators | ~3 |
| collab frontends | ~2.5 |
| planners | ~3 |
| **total** | **~11–13** |

That fits one g4dn.4xlarge (16 vCPU, the quota being requested).

Fallback order if the time runs out: rows oracle / independent / ariadne first, then partition, then the budget sweep.
`cbba_distributed` is last.

## 7. §49

- **catalogue:** no new algorithm (CBBA / SGA per arch §21.7; coarse coverage summaries per §20.6 and §15.3).
- **arch:** an as-built note for §21.7 and §15.3: classes 7–10 implemented, the task key deviation.
- **ablation:** the §4 rows.
- **benchmark config:** `configs/benchmark/multiagent_v1.yaml`.
- **manifest:** row, budget, scenario, per-class bytes, task-key radius.

## 8. Planner-side decisions (A1 and M, 2026-10-23)

1. **Received layer.** planner_k's TSDF is built from its OWN keyframes only. Each peer's coverage summary lands in a
   separate per-peer received layer: tri-state, on the §20.6 0.5 m grid. That layer:
   - suppresses frontier clusters and candidates whose target lies in peer-known space;
   - zeroes the IG of voxels whose column a peer already knows.

   Paths and risk use only the agent's own map: no routing through space only a peer saw (the conservative safety
   choice).
2. **Egress (invariant 4).** Each summary cell also carries the sender's ρ_responder traversable/support bit,
   computed by the sender from its own TSDF: 2 bits per cell. The operator node runs D* Lite egress on its received
   layers, so the §22 claim holds at the operator.
3. **Coordination lag (pre-registered as part of what the ariadne row measures).** At barrier T each agent executes
   the TASK_CONSENSUS it RECEIVED by T, which was computed from the T−5 s bids.
   - It takes the assigned goal if that task key is still present among its own options.
   - Otherwise, or with nothing received within `consensus_timeout_s`, it takes its own best option, logged
     "uncoordinated".
   - CommsNode's latency and capacity apply. Zero latency is never assumed.
4. **Broker identity per barrier:** `ComponentOf(agent) -> {broker, members, epoch}` from the comms layer (§15.4: the
   lowest id in the component, with an epoch). TASK_CONSENSUS from a stale epoch is ignored.
5. **Task key:** spatial only. Frontier ids are per-agent hashes. The broker dedups bids greedily by utility; a bid
   within `r_same = min_goal_m` of an already-assigned goal counts as the same task.

**Planning-library interfaces (no comms types; the wire tables mirror these structs):**
- `PlanAgent(t, own inputs, received layers, peer poses)` → options
- `ExportSummary(since_version)` → changed coarse cells
- `IntegrateSummary(sender, cells, T_G_W(sender))`
- `BuildBid(options, M)` → TASK_BID payload
- `BrokerAssign(bids, members, r_same)` → consensus
- `ApplyConsensus(consensus, options, timeout)` → a pick, or "uncoordinated"

**Oracle row:** the current Phase4Planner code path, untouched; proven identical by A1's replay/eval check.

**Deviation from §20.6, decided 2026-10-23 (A1 and M).** Coverage travels as `SUBMAP_PATCH` (8) at the PLANNER grid
(0.05 m, 32 × 32 blocks, 2-bit codes: unknown / occupied / free / free-and-ρ_responder-passable), not as §20.6's
0.5 m coarse summary. Reason: egress at ρ_responder (0.40 m) on 0.5 m cells would erase doorways, breaking invariant 4
at the operator. Sizing:
- **Wire:** 256 B packed per block plus ≤ 64 B FlatBuffers overhead, zstd-compressed (protocol test
  `CoordinationWire.*`).
- **Estimate:** 0.5–3 KB per 5 s patch [ESTIMATE; A1 measures on a real run].

`SUBMAP_SUMMARY` (7) stays reserved.
