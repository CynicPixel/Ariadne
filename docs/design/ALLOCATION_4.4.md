# 4.4 Allocation: event-triggered CBBA (bundles 1–2) and broker sequential greedy, logged identically (A1)

Status: **reviewed** (advisor 2026-10-23, §9), before any code. The wire table waits on the master session (§5). Decision: D115. Arch §21.7; ALGORITHMS §22.3/§22.3.1
(CBBA, transcribed from the paper: Algorithm 3, eqs. 3–6, Table I, Theorems 1–2). Consumes D114's per-agent
`TaskCandidate` sets. Replaces `Phase4Planner`'s greedy joint pick, which stays as the ablation's simplest row.

## 1. What the paper gives us, and what it means here

- **Theorem 1:** under a DMG score, with synchronized conflict resolution, **CBBA's assignment equals sequential
  greedy's (SGA)**. It converges within N_min·D iterations, with N_min = min{N_t, N_u·L_t}.
  **Theorem 2:** under DMG, M_OPT ≤ 2·CBBA.
- **Consequence for the arch's "auction vs broker-greedy" row:** with a DMG score the two give the **same
  assignment**. The row measures what decentralization costs in **messages, bytes and rounds** (and robustness when
  the broker is lost), not solution quality.
  - "Logged identically" becomes an exact test: same (agent, task, bid) set, every round.
  - A difference in assignment is a bug, or a score that is not DMG.
- **N_u = 2 in our missions.** Consensus is one hop when the pair is connected, so CBBA's interesting regimes are
  disconnection, reconnection and stale timestamps. The unit tests run N_u up to 8 on random graphs so that Table
  I's m/n rows are exercised.

## 2. Tasks (§21.7 steps 1–2)

- **A task is a 4.3 target.** Its `task_id` is stable across replans.
- **§21.7's spatial clustering is 4.3's targets.** Frontier clusters split along their PCA axis, and coarse 0.5 m
  cells, are already spatial clusters.
- **No second clustering pass** (advisor review):
  - `cv::partition` is a transitive closure, so it would chain together the pieces that generator A's PCA split
    just separated;
  - a "smallest member" id changes whenever membership changes, which breaks exactly-once across replans.
  - If adjacent bottom-decile B cells should merge, that belongs inside generator B, with a stable anchor.
- **Per agent, per task:** the agent's best view of it (the highest D114 utility). Each agent has its own view,
  because the goal depends on the agent's reachable space.

## 3. The score must be DMG (eq. 7): time-discounted, per eq. 11

- **D114's U = G − λ_path·C_path − C_b − C_r is not DMG once bundles hold two tasks.**
  - With an additive path cost, the marginal of task j is G_j minus the cheapest insertion detour.
  - Adding a task k next to j **shrinks** j's detour and raises its marginal.
  - That breaks eq. 7 and, with it, Theorem 1's equality.
- **Adopted:** the paper's DMG form (eq. 11), with our terms:

  `S_i(p) = Σ_{j ∈ p} c̄_ij · exp(−λ_path · d_ij(p) / P_path)`

  - c̄_ij = max(0, G_ij − C_b,ij/P_b − C_r,ij/P_r) > 0: the task's value net of its non-path costs;
  - **G_ij is recomputed on the auction's common scales P:** Σ_k raw_k / P_k, with P_k the max over agents of each
    agent's p90 scale for term k. The scales are exchanged in the first message of an allocation event and frozen
    for its rounds (advisor review).
    - With per-agent p90s, an agent with a sparse candidate set would bid on an inflated scale. Every Table I
      y-comparison would then be meaningless, and CBBA = SGA would still hold, on wrong inputs.
    - The max is a max-consensus: idempotent and order-free, so it settles within D hops, like CBBA itself.
  - d_ij(p) = the agent's ρ_agent path distance to j **along p**: from its position through the earlier tasks'
    goals;
  - the discount's rate is D114's λ_path over the same p90 path scale, so no new constant enters.
- **This is DMG**, by the paper's own argument (eqs. 11–12): an insertion can only lengthen the distance to every
  later task (triangle inequality on geodesics), and c̄ > 0.
- **With L_t = 1** the bid is c̄·exp(−λ·d/p90), monotone in the same direction as D114's linear U. The ranking can
  differ, which is stated; D114's ranking is what the runtime shows the operator.
- **Leg distances:**
  - agent → goal: 4.3's geodesic;
  - goal → goal: a Dijkstra from each candidate goal on the agent's plan space. That's N_t Dijkstras per agent per
    event. Its cost is **measured on the spool** (§8) before it is relied on. The fallback, if too slow, is to
    compute legs only for tasks that can enter a bundle (h_ij = 1 against the current y), stated if used.
- **One comparator** everywhere: Table I's y-comparisons, BuildBundle's argmax, SGA's argmax and the oracle.
  Compare by bid, then lexicographic (agent id, task id) (§III-B's tie-break). Otherwise ties would break test 2's
  equality for spurious reasons.
- **Timestamps s:** round indices in replay and tests, sim time live. Never the wall clock (B's determinism
  requirement).

## 4. Two allocators, one log format

- **CBBA:** each agent runs BuildBundle (Algorithm 3, L_t = 2) and exchanges (y, z, s). The receiver applies Table I
  task by task, then eq. 6's release. Iterate until no change.
  - Event-triggered (§21.7): a round starts only on a new task cluster, a completed task, or a topology change. A
    replan where none occurred reuses the last assignment and sends nothing.
- **Broker SGA:** the broker holds every agent's c̄ and distances, and repeats "pick the (agent, task) with the
  largest marginal bid, insert it at its best position" until bundles fill or bids run out. That is the paper's SGA
  (Theorem 1's comparator), computed in one place.
- **Logged identically**, one record per allocation event:
  - trigger, t_s, rounds (CBBA iterations; 1 for SGA);
  - per agent: bundle, path, bids;
  - bytes, per message and in total, counted at the serializer (invariant 6).
- **Ablation rows:** greedy-joint (`Phase4Planner`), SGA, CBBA. Each reports assignment, total score, the exact
  optimum, rounds and bytes.

## 5. Bytes, at the serializer (invariant 6)

- **The wire already has the classes:** `wire.fbs` defines `MessageClass` TASK_BID (9) and TASK_CONSENSUS (10).
  `WireCodec` and `ByteLedger` are on this branch.
- **A CBBA message needs one table and one field:**
  - `TaskConsensus { tasks:[ulong]; y:[float]; z:[byte]; s:[long]; scales:[float]; }`
  - one field appended to `Envelope` (`task_consensus:TaskConsensus`), plus `WireCodec::EncodeTaskConsensus` and
    `DecodeTaskConsensus`.
- **Coordination:** the master session has uncommitted 3.4 edits to `wire.fbs` and `serializer.hpp`, at exactly
  that append point. I've asked it how it wants the table added. The CBBA core takes encode/decode as two function
  objects, so the codec plugs in without the core depending on FlatBuffers.
- **Until the table lands:**
  - unit tests use an exact test codec;
  - any byte column is labelled "test codec" and is never a claim.
  - Invariant 6's bytes are counted only once the FlatBuffers path exists.

## 6. Exact optimum oracle

- **L_t = 1:** the assignment LP is integral (total unimodularity), so `cv::solveLP` gives the optimum (ALGORITHMS
  §22.3).
- **L_t = 2, N_u = 2, the missions:** a sorted scan, exact without enumerating every combination (advisor review):
  - each agent's bundles are ≤ 2 tasks, scored at their best order, about N_t²/2 per agent;
  - sort agent 1's bundles by score;
  - for each agent-0 bundle, take the first disjoint agent-1 bundle. Only bundles sharing one of agent 0's ≤ 2
    tasks are skipped, so the scan is O(B log B).
  - Exact by construction, and run on every logged round.
  - About 170 targets per replan (4.3 log) gives about 14.5k bundles per agent.
- **L_t = 2, N_u ≤ 4 (tests):** brute force over small instances (N_t ≤ 8).

## 7. Tests

1. **Table I, row by row:** the 17 rows (one cell per (z_kj, z_ij) pair, including n ∉ {i,k,m}), each with the
   condition true and false. Expected actions are written from §22.3.1, not from the code.
2. **CBBA = SGA under DMG:** random DMG instances, N_u ∈ {2..8}, N_t ≤ 12, L_t ∈ {1, 2}, on random connected graphs
   with synchronous rounds. The assignments are equal, and the rounds are ≤ N_min·D.
   - That's Theorem 1(2), unmodified for DMG scores. The paper's "slight modification", Lemma 4's
     c_ij(t) = min(c°_ij(t), c_ij(t−1)), is for scores that are not DMG, and is checked as not needed here.
3. **Conflict-free** at convergence on dynamic graphs whose ρ-windows are jointly connected. The rounds are
   ≤ ρ·N_min.
4. **50 % bound:** total score ≥ ½ of the brute-force optimum on every instance (Theorem 2). The empirical ratio is
   reported.
5. **The DMG check itself:** eq. 7 holds numerically on the time-discounted score over random paths. The additive
   score fails it on a constructed counterexample (the reason for §3).
6. **Event triggering:** with no trigger, no message is sent and the assignment is unchanged.
7. **Release (eq. 6):** an outbid first task releases the second one too.
8. **Common scales:** an agent with extra weak candidates bids exactly what a peer with the same view of a shared
   task bids. With per-agent p90s they would differ, and this test would catch it.

## 8. Evaluation

- **Replay:** run on the 4.3 decision log's per-agent candidate sets at each replan. The two agents are connected
  after the merge and disconnected before it, which is the spool's real topology.
  - **What replay can't show** (advisor review): agent 1's keyframes are held until the merge, so it has no
    candidates before it. Replay exercises only the connected N_u = 2 case, 1–2 rounds.
  - The interesting regimes (disconnection, reconnection, stale timestamps) are covered by the synthetic tests
    (2, 3) and by live L1. They're never claimed from replay.
  - Rows: greedy-joint, SGA, CBBA, plus the exact optimum.
  - Columns: score, optimum ratio, rounds, messages, bytes (test codec).
- **Live (lockstep L1):** the gate ablation uses Ariadne with CBBA. The broker-SGA row reuses the same runs'
  logged inputs, so it needs no extra live runs.

## 9. Review (advisor, 2026-10-23)

Blocking points, all adopted:
1. Common auction scales: the max of the agents' p90s, frozen per event. Test 8 (§3).
2. No `cv::partition` second pass: 4.3's targets are the spatial clusters, and their ids stay stable (§2).
3. An exact, feasible oracle: the sorted disjoint scan; the cost of the leg distances measured before it is
   relied on (§3, §6).

Sharpening points, all adopted:
- one comparator; timestamps from rounds or sim time;
- Theorem 1(2) needs no modification for DMG (Lemma 4 is for scores that are not DMG);
- replay's limits stated;
- the wire table coordinated with the master session.

The remaining choice is the time-discounted DMG score itself (§3). It is the paper's own form, with D114's λ_path
and the common scales: no new constant.

## 10. Correction (2026-10-23, measured before any commit of code)

- **§1's "CBBA = SGA" holds exactly at L_t = 1 only.** At L_t = 2 a stale second-slot bid can make them differ
  (ALGORITHMS §22.3.1, correction). The first test sweep found it on 1 of 60 instances. The advisor's three
  separating checks then confirmed the mechanism:
  - L_t = 1: 2000 / 2000 equal, no tail release;
  - L_t = 2: 5 / 1000 diverge, all with eq. 6 tail releases;
  - a hand-built B–A–C instance reproduces the worked outcome.
- **Default L_t = 1** (inside arch §21.7's 1–2):
  - CBBA then equals SGA exactly, so the "auction vs broker-greedy" row measures communication only;
  - Theorems 1–2 apply as stated.
- **L_t = 2 is a reported row:** divergence rate, and score against SGA and against the optimum.
- **Tests split by what is provable:**
  - L_t = 1: exact equality and the N_min·D bound;
  - L_t = 2: converged, conflict-free, divergence only with tail releases, and ≥ 50 % of brute force, labelled
    empirical.

## 11. Wire (2026-10-23, master session's option (b))

- **Schema:** `TaskConsensus { tasks:[ulong]; y:[float]; z:[byte]; s:[long]; scales:[float] }` in `wire.fbs`,
  appended last in `Envelope`.
- **Encoding:** `WireCodec::EncodeTaskConsensus` goes through `FinishEnvelope`, so the `ByteLedger` counts class
  TASK_CONSENSUS (10) like every class. It is **not compressed** (§15.7 compresses classes 4–5 only).
- **Unequal per-task vectors decode as malformed.**
- **Existing bytes unchanged, proven:** `Serialization.PreTaskConsensusBytesUnchanged` pins the size and FNV-1a of
  a fixed heartbeat, loop edge, pose graph and ack, recorded with the pre-change tree (= main's). All 59 protocol
  tests pass.
- **CBBA counts real wire bytes:** `allocation_wire.hpp` measures each message as its envelope's `wire_bytes`.
  - A 6-task, 2-agent message is 274 B on the wire, against 110 B from the test codec, which is a 2.5×
    undercount. That's why the test codec was never a claim.
  - If CBBA bytes ever feed a predicted-vs-measured claim, COLLAB_3.2 §7's rules apply: first transmission only,
    overhead as separate rows.
