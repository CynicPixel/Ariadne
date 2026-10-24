# 4.3 Next-best-view (I8): candidates A–E, raycast IG, self-normalisation, λ scheduling, the energy-reserve rule (A1)

Status: **reviewed** (advisor 2026-10-23: points 1–6 addressed below; the gate regime decided by the maintainer: lockstep L1, §9), before any code. Decision: D114. Arch §21.1–§21.4; ALGORITHMS §22.1–§22.2.
Consumes D110 (map, frontiers, coarse layer), D111 (trav_a, risk), D113 (the egress route and its support). Feeds
4.4 allocation and B's `MissionPlanner::Plan` (through `TaskCandidate`, commit e2e538d). The 1.8 nearest-frontier
policy stays as the ablation baseline.

## 0. What the gate asks, and the regime it needs

- **Plan 4.3 gate:** a planner ablation plot of reconstruction coverage vs mission budget, for random,
  coverage-greedy, uncertainty-only and Ariadne, on ≥ 2 scenes × 3 seeds.
- **Arch §5.9:** closed-loop active-perception experiments must run live, because the trajectory depends on the
  estimate. They are reported over seeds with variance, and every figure declares its regime.
- **So replay cannot produce the gate's plot.** A replay spool's trajectory is fixed, so a planner can only be
  scored counterfactually there ("what it would have chosen").
- **The planner is built behind the seam and run in two regimes:**
  1. **replay decision log** (deterministic, now): every replan on the evidence spool logs the full candidate set,
     with terms, costs and utility. This tests the machinery, not the policy's value;
  2. **closed loop** (the gate): the choice of loop is the one open question (§9).

## 1. Interface (planning-level; no runtime, ROS or AWS type)

```text
NbvInputs  { const TsdfMap&, const Esdf&, const FrontierSet3D&, const CoarseLayer&, const TraversabilityMaps&,
             const EgressReplan* route, const cv::Mat* route_support,
             per agent: pose (world), energy used, peers' keyframe positions;
             mission clock t, budget T; ConnectivityOracle*; AliasSource* }
NbvPlanner::Plan(inputs, agent) -> std::vector<TaskCandidate>   (sorted by utility, ties by task_id)
```

- `ConnectivityOracle` returns the §16.2 spectral gain Δλ₂ of the inter-agent edge that a view near a peer's
  keyframe would enable. Its producer is 3.4's selector (D170), whose API was read from `feat/3.4-spectral`:
  - `SelectionProblem` has `n` nodes, `fixed` `GraphEdge{a, b, kappa}`, `candidates` `CandidateEdge{id, a, b,
    kappa}`, `node_bytes` and a budget;
  - `SolveFiedler` / `SelectionReport.lambda2` give λ₂.
- **The oracle is a planning-side interface over plain numbers:** `double Gain(int agent, int64_t peer_kf_id)`.
  As built, the adapter maps (agent, keyframe) to the broker's graph nodes.
  - It holds no collaboration type, so planning keeps its core-only dependency rule.
  - A runtime adapter implements it as λ₂(G ∪ {e}) − λ₂(G) with 3.4's `SolveFiedler`, on the broker's pose graph
    (`fixed`), with kappa the predicted retrieval similarity.
- `AliasSource` returns §17.3's alias-flagged place pairs.
- **Conflict, surfaced:** arch §21.3 says every term has a producer. On `feat/ph4-mission` (from main `f8c7030`):
  - **3.4 is not merged:** it's on `feat/3.4-spectral`, the main session's;
  - **§17.3 alias flags have no producer at all.**
- Until each lands, its term is **0 with the reason in the trace**, never a stand-in value. Generators C and D still
  produce candidates, so their geometry and costs are tested. Their gain is reported as missing.

## 2. Candidate generators (§21.2)

Every generator yields **targets** (a point plus a size). One shared **viewpoint sampler** turns a target into
views. That is FUEL's per-cluster cylindrical sampling (ALGORITHMS §22.2):
- **Sample grid:** radii {1.0, 1.5, 2.0} m (inside the 2.84 m depth gate), 16 azimuths, z = 1.0 m (the motion
  altitude), yaw toward the target.
- **A view is kept** if its cell is in `trav_a` and a voxel ray from the view to the target hits no occupied voxel
  (line of sight).
- **Per target:** the best `k = 2` views by Δcoverage (FUEL keeps the best per cluster; two leave a fallback).
- No RNG: the grid is fixed, so plans are deterministic.

The targets per generator:

| Gen | Target | Source |
|---|---|---|
| A frontier | each kept `FrontierCluster3D`: its centroid, and its normal when defined. A cluster whose AABB extent is > 2 m is split along its first principal axis (FUEL's PCA split) | D110 |
| B uncertainty | coarse 0.5 m cells holding surface (occupied > 0) whose mean weight is in the **bottom decile** of such cells (rank-based, so no weight threshold) | D110 coarse layer |
| C connectivity | peers' keyframe positions within 4 m of this agent's map that this agent's keyframes have not come within 2 m of | spool / L1 poses |
| D alias | each alias-flagged place pair: both places | `AliasSource` (none yet) |
| E egress | connected runs of current-route cells with support < 1, centred on their lowest-support cell | D113 |

`task_id` = a hash of (generator, the target's anchor voxel), stable across replans (`TaskCandidate`).

## 3. Information gain (§21.3)

- **One bundle per view:** 256 rays (inside the arch's 200–500), on a 16 × 16 grid over the rig's frustum. The
  frustum is the spool's left intrinsics at 640 × 360. Rays run to 2.84 m and stop at the first occupied voxel.
- **Voxel walk:** 1.8's Amanatides–Woo walk (`occupancy_grid.hpp`). Voxels are de-duplicated across the bundle.
- **The five terms:**
  - **Δcoverage** = unknown voxels reached.
  - **Δentropy** = Σ H(p) over reached voxels, the expected entropy removed if the view resolves them.
    - Unknown: p = 0.5 (1 bit).
    - Free: p_occ = ½·e^(−w). Occupied: p_occ = 1 − ½·e^(−w).
    - Here w is the TSDF weight, whose unit is one observation at σ_ref (D110). That unit is the only scale; there
      is no fitted constant.
  - **Δegress_confidence** = reached voxels that are unknown and inside the current route's body envelope (the
    support definition of D113).
  - **Δpose_connectivity** = ConnectivityOracle gain (C only; 0 otherwise).
  - **Δloop_disambiguation** = AliasSource value (D only; 0 otherwise).
- **Saturation check (advisor point 2; evidence-spool map, no GT; `ariadne_volumetric_eval` `known_weights`):**

  | | known voxels | w p10 / p50 / p90 | mean H | H > 0.9 bit |
  |---|---|---|---|---|
  | free | 775,710 | 0.110 / 0.743 / 4.87 | 0.660 bit | 34.4 % |
  | occupied | 196,132 | 0.112 / 0.832 / 9.78 | 0.611 bit | 33.4 % |

  - H runs from about 1 bit (w ≈ 0.1) to about 0.04 bit (w ≈ 5). Two-thirds of known voxels are below 0.9 bit.
  - So Δentropy separates re-observing weakly seen surfaces from re-observing well-seen ones. It is not
    "reachable voxels".
  - **The mapping is adopted.**
- **Self-normalisation:** each term is divided by its 90th percentile over the current candidate set. If the p90 is
  0 (≥ 90 % zeros), divide by the max; if that's 0 too, the term is 0.
- **G = Σ normalised terms**, with α = β = γ = δ = 1 (the arch fixes no others; normalisation makes 1 the neutral
  choice).

## 4. Costs, λ scheduling and the hard rule (§21.4)

- **C_motion** = path length at ρ_agent: `GridDistances` on `trav_a` from the agent's cell (1.8, exact geodesic).
- **C_time** = path / v, with v = worldgen's motion speed.
- **C_energy** = e_per_m · path.
  - In the kinematic rig all three are ∝ path. After per-term p90 normalisation they become the **same number for
    every candidate** (advisor, review point 1).
  - Kept as three separate terms, with λ⁰ = 1 each, the path weight would be 3 at t = 0. That's against a G where
    usually only 2–3 of the 5 terms are nonzero, so "λ⁰ = 1 is neutral" would be false.
  - **Proposed:** count the proportional costs once:
    `C_path = path/p90(path)`, `λ_path(t) = (λ_d⁰ + λ_e(t) + λ_t(t)) / 3`.
    That's 1 at t = 0 and E_used = 0, and 5/3 when both the energy and the clock are exhausted. The cubic schedules
    act as one multiplier, and the effective weight is stated in D114.
  - Rigs where time or energy is not ∝ path (hover, comms waits) get the terms back, separately.
- **C_bandwidth** = the §15.7 byte-table cost of the exchange a view triggers: generator C views only (a
  keyframe-summary exchange); 0 otherwise.
- **C_risk** = the candidate's risk, as `TaskCandidate`:
  - max of the geometric bottleneck term and the structural [PROXY];
  - the proxy enters only if D112's calibration passes.
- **Cost normalisation: an extension, flagged.**
  - The arch normalises the IG terms only, which leaves λ⁰ unit-bearing: metres vs bytes vs risk.
  - Proposed: each cost is also divided by its candidate-set p90, so λ⁰ = 1 is unit-free. Without this, every λ⁰ is
    an unfalsifiable tuning choice, which is exactly what §21.3 rejects for gains.
- **Schedules, as the arch gives them:**
  - λ_e(t) = λ_e⁰ (1 + (E_used/E_total)³), and λ_t the same against t/T;
  - λ_d, λ_b, λ_r constant at 1.
- **U(a) = G − λ_path(t)·C_path − λ_b·C_b/p90_b − λ_r·C_r/p90_r**, with λ_b = λ_r = 1.
- **Hard rule, outside the utility:** if E_remaining < 1.3 · E(return path), the candidate set becomes
  {return_to_safe_zone, hold}.
  - The return path is the ρ_agent geodesic to the nearest briefed entrance.
  - It's logged as a supervisor event (§24).
  - It's tested as unreachable from the optimizer: no gain and no weight can re-admit a candidate.
- **New `mission_v2` keys:** E_total, e_per_m, v (from worldgen motion), budget T. In the kinematic rig E is a
  distance budget, stated as such.

## 5. Baselines for the ablation (plan 4.3)

**Gate axes, pre-registered (2026-10-23, before any closed-loop run):**
- **Coverage (y):** 4.1's map metric, the fraction of GT-free cells observed free in the 1.8 slab over the
  [0.15, 1.90] m band (`volumetric_eval.py` "coverage"; the same scorer as the D110 rows).
  - Secondary: the fraction of GT structure cells adjacent to free space observed occupied (surface coverage).
- **Budget (x):** metres flown, summed over agents, at 10 fixed fractions of the run's distance budget. In the
  kinematic rig that's the energy budget E_total.
- **Scenes:** `ci_tiny` and `maze_short` (existing presets), seeds 0, 1, 2. With hazards, `ci_tiny_hazard`
  joins as a third if its spools exist.
- **Report:** per policy and budget, the mean over the 3 seeds, with min and max (§5.9: over seeds with variance,
  never a single run). Scalar: area under the coverage-budget curve. The regime is declared on the figure.
- **The claim is "Ariadne ≥ every baseline at every budget fraction".** It's tested per scene. A loss is reported
  as a loss.
- **Amendment (2026-10-23, before any gate run and before any coverage from the pilot was looked at).** The reason
  is the adapter pilot's flight speed, not a result: B's 30 s lockstep pilot of A1Planner (`feat/l1-a1planner`
  ea88599) flew 4.7 and 4.2 m, about 0.15–0.2 m/s against the 0.6 m/s cap.
  - Why it's slow: the mover turns in place to each goal's final yaw, goals are 1.4–2.3 m away, and every 5 s
    barrier picks a new goal before the last one is reached.
  - The consequence: in 200 s an agent flies about 30 m, a quarter of the 120 m budget, so most of the fixed
    budget fractions would never be reached.
  - **Budget (x), primary: mission time.** 10 fractions of the 200 s mission, T = 20, 40, …, 200 s. Each falls on a
    5 s lockstep barrier, so the barrier slab snapshots are scored exactly. No dump hook, no interpolation.
  - **Secondary view:** coverage against summed metres flown (`flown_m` from the trace), at the actual barrier
    points. No fixed fractions, and no final coverage carried into fractions a run never reached.
  - **`e_total_m` stays 120 m** (the physical cap, not tuned to the pilot). At ≈ 0.15 m/s neither the reserve rule
    nor λ_e engages in the gate (λ_e's factor ≈ 1.02 at 30 m). Their evidence is the unit tests only, stated.
  - **Every row is scored from the same Phase 4 map:** the slab of a shadow Phase 4 mapper fed the row's keyframes,
    the 1.8 placeholder row included. Otherwise a row's coverage partly measures its mapper (D110: OccupancyGrid vs
    TSDF coverage 0.933 vs 0.942 on one spool).
  - **The claim becomes "Ariadne ≥ every baseline at every time fraction",** per scene, losses reported.
  - **Pilot fact, not fixed before the gate:** the chosen task changes at every barrier, because target ids churn
    as the map grows and there is no goal persistence. That's the same for every row, so it is stated, not
    corrected.
- **Added 2026-10-23 (EXPLORATION_4.6 review), before any run:** the §31.6 row **TARE-inspired** (D116). That makes
  6 policies × 2 scenes × 3 seeds = 36 lockstep runs, ≈ 14 h on llvmpipe (≈ 22 min each).
  - On ci_tiny (about one 6 m subspace) the TARE row is nearly local-only, so a null result there is not over-read.

All four share the same generators, sampler and costs, so the comparison isolates the scoring:
- **random:** a uniform choice among reachable candidates (seeded Philox stream);
- **coverage-greedy:** G = Δcoverage only, cost-free; nearest wins ties;
- **uncertainty-only:** G = Δentropy only;
- **Ariadne:** the full U(a).

1.8's nearest-frontier policy is a fifth row (the D45 baseline).

## 6. Tests

1. The IG raycast against a brute-force voxel count on a synthetic map: an unknown box behind a wall with a hole.
   Exact counts.
2. Self-normalisation is scale-invariant: multiplying a term by k leaves the normalised terms, G and the ranking
   unchanged.
3. The λ schedules at E_used = 0 / ½ / E_total, by hand.
4. The hard rule:
   - below the reserve, only {return, hold} are emitted, whatever the gains (a 10⁶ gain candidate is still dropped);
   - above it, the normal set.
5. The sampler: every emitted view is in `trav_a`, with line of sight to its target. Deterministic: two calls give
   identical output.
6. Generator E: a route with a low-support run yields a target at its minimum.

## 7. Replay decision log (now)

At every 5 s replan on the evidence spool, per agent: the full candidate set (generator, goal, terms raw and
normalised, costs, U), the choice, and the timings. This is evidence that the machinery runs at replan rate. It is
**not** the policy's value, and it is never plotted as coverage vs budget.

## 8. Preflight record

- **Literature:** ALGORITHMS §22.1–22.2:
  - read in primary text: TARE, FUEL, SEE, RACER;
  - secondary only: Isler and Delmerico. Delmerico's result (unobserved-voxel counting did best) backs Δcoverage
    as a term separate from Δentropy.
- **OpenCV-native:** none for raycast IG. `cv::Volume::raycast` renders depth only and has no voxel read (SMOKE
  §4u). PCA uses `cv::PCA`. Line of sight uses the voxel walk.
- **B's seam:** `TaskCandidate` → `TaskOption` (8ff170a), with risk and its reason per option.

## 9. Open question for the maintainer: the closed loop for the gate

**L1 facts (B, 2026-10-23):**
- **Closed loop works today.** `ariadne_live` (feat/ph7-agentic) calls `MissionPlanner::Plan` at every
  sim-time barrier T = k·5 s. The pick is validated, then `Route` → mover `PoseArray`; the mover checks against the
  ESDF and holds if refused. gz renders whatever is executed. 4.3 plugs in by implementing `MissionPlanner`.
- **Cost:** no GPU is needed. llvmpipe RTF is 0.13–0.15, about 22–25 min wall per 200 s mission, one run at a time
  on the dev host (memory), so 24 runs ≈ 9–10 h. NVIDIA EGL on g4dn is unknown until SMOKE §4v (the D112 run).
- **Not reproducible live:** gz keeps running while the planner thinks. Commands apply about 1.7 s of sim time
  after their stamp, depending on wall latency, so runs N and N+1 differ. Replaying a recording is bit-exact
  (D152a).
- **Lockstep** (pause at each barrier via gz WorldControl, resume after the commands) is not built; about half a
  day for B. It removes the latency confound and makes classical/NBV runs reproducible.

**The options:**
- **(a) L1 as is:** 24+ runs, ≈ 10 h on the dev host. Variance includes wall-latency noise, which differs by
  policy (the costlier planners lag more), confounding the comparison.
- **(b) L1 with lockstep (B, ½ day):** the same 24+ runs, reproducible per seed. The policies differ only in their
  choices. On g4dn, if §4v passes, faster.
- **(c) A GT-raycast kinematic loop now, as a development harness, then (a) or (b) for the gate figure:**
  CPU-only, seconds per run, a different regime, labelled.

**Decision (maintainer, 2026-10-23): (b), lockstep L1.** B is asked to schedule lockstep. The gate runs go through
A1's `MissionPlanner` on `ariadne_live`, with the axes of §5. The library and the replay decision log (§7) are built
meanwhile.

## 10. Review (advisor, 2026-10-23)

1. Path-proportional costs counted once (λ_path); the effective weight stated (§4).
2. Δentropy saturation measured before adoption: not saturated (§3).
3. Gate axes pre-registered (§5).
4. ConnectivityOracle shaped to 3.4's real API, read-only from its branch (§1).
5. L1's closed-loop facts got from B before asking the maintainer (§9).
6. AWS provenance: the D112 payload binaries were built at `df38a12` (pinned by `inputs.SHA256SUMS`); run commit
   `c8a3262`. Both go into the D112 run record.

## 11. As built (2026-10-23)

Code: `cpp/planning/mission/nbv.{hpp,cpp}` (D114). Tests: `tests/unit/test_nbv.cpp`, 6 tests, all pass with the 7
egress tests:
- the raycast on a wall: exactly 0 coverage behind a solid wall; the entropy identity with and without a hole;
  egress counts only envelope voxels; clipping to bounds;
- p90 scale invariance and its fallbacks;
- λ_path by hand;
- the reserve rule;
- the sampler contract;
- generator E's target at the route's lowest support.

**Replay decision log on the evidence spool** (`evaluation/runs/nbv_4.3_replay/`, 52 decisions: agent 0 at all 40
replans, agent 1 at the 12 after the merge):
- **First run: 5.9 s per agent per plan (p95 8.0 s).** The cost was the sampler: every target's 48 sampled views
  got the full 256-ray bundle.
  - Fixed: views are ranked by coverage on a 64-ray bundle, and only the 2 kept views get the full 256-ray terms.
    The voxel state now takes one hash lookup.
  - IG is now also clipped to the briefing's extent (grid × [0, `z_max`]), so rays leaving the building are not gain.
- **After the fix: mean 1.98 s, p95 3.1 s.**
  - That's fine under lockstep L1 (the world is paused while planning).
  - Unpaused, it would exceed the 5 s period's budget on the dev host. Not a target; stated.
- **Generator mix by the end:** A 32–36 targets, B 109–118, C 20 (agent 0), E 2. D: 0 (no producer).
- **The top choice:** A 27, B 24, C 1. None empty.
- Both missing producers are named in every line's notes.
- The early views' Δcoverage (≈ 15k voxels) is real: at t = 5 s the building's interior is unknown and inside the
  extent.

## 12. Amendment: plan space and the risk rule (2026-10-23, before any gate run)

**Found by B on a replay of the A1Planner pilot:** risk 1.0 on 27/28 → 162/172 options. Two causes, fixed in order:
1. **The arrival rule** (`30ac351`). I read D158's bottleneck at the goal cell alone. B's D158 takes the max over
   the goal's escape_m disc, the widest way to arrive near the goal. That's now `ArrivalBottleneck`.
   - It barely helped: 154/172 were still at 1.0.
2. **The plan space.**
   - D111's `trav_a` (the agent capsule on the occupied-only field) counts unknown space as passable. So views and
     paths ran through unobserved columns, where the egress clearance `clear_r` is 0, and every maximin was 0.
   - **Now:** the plan space and every view goal are restricted to `trav_a ∧ clear_r ≥ ρ_agent`: known-free
     columns with ρ_agent clearance to anything not observed free (`KnownFreePlan`).
     - That is B's D153b standoff rule, adopted live after the mover refused 10/10 goals next to unseen walls.

**Literature check** (asked by the maintainer, done before this was committed):
- **TARE** (Cao et al., RSS 2021), read in primary text:
  - §III defines a viewpoint position p_v ∈ Q_trav, the traversable subspace;
  - the local horizon's candidates come from H_trav, "the traversable subspace identified by considering collision
    and connectivity";
  - the surfaces to cover are the boundary between free and non-free space, the latter "includes both occupied and
    unknown spaces".
  - So TARE also keeps viewpoints in traversable, connected space and treats unknown as non-free. The amendment
    follows it.
- **FUEL** (Zhou et al., RA-L 2021): `[UNVERIFIED]` for this point. arXiv served an HTML page instead of the PDF on
  two attempts, so its viewpoint constraint is not cited from memory.
- **FAR** (Yang et al., IROS 2022): routes "attemptably" through unknown space, but for goal-directed navigation,
  not exploration viewpoints. It stays a §31.6 comparator; `Route` doesn't use it.

**Effect:** to be measured on B's replay of the same recording after the rebuild (risk histogram per replan), before
the gate starts. The gate waits on it (YIELD set by A1).

## 13. Literature review of every post-design fix, and the return space (2026-10-23, before any gate run)

The maintainer asked for a primary-text review of every fix made after a measurement. The full reports, with quotes,
pages and the URLs read, are in `docs/design/lit/phase4_fixes_nbv.md` and `phase4_fixes_route_cbba.md`. The
findings that change this item:

**Known-free plan space (§12): supported.**
- **FUEL:** the paper (§IV-C) says only that viewpoints lie "within the free space". The released code (@662dd23c7b52)
  is explicit:
  - viewpoints with an unknown voxel within 0.21 m are rejected;
  - A\* rejects unknown cells;
  - the exploration config disables optimistic mode, so the ESDF treats unknown as an obstacle. That is our `clear_r`.
  - This replaces §12's `[UNVERIFIED]`.
- **NBVP** (Auton. Robots 2018, §3): "Paths are only planned through known free volume V_free".
- **GBPlanner2** (§IV-A): "samples vertices in the known free space".
- **Optimistic planning through unknown space** exists only for navigation to a given goal with constant replanning:
  D\* Lite's free-space assumption, FAR's attemptable mode, Nav2's defaults, FUEL's point-goal launch files.

**The goal's risk: D158's max over the escape disc is dropped; the score is taken at the commanded cell.**
- Maximin clearance along a route is an established route-quality measure: Voronoi maximum-clearance paths (Geraerts &
  Overmars, IJRR 2007, §1.2), and the maximum spanning tree of Agarwal, Fox & Salzman (arXiv 1706.02939).
- No paper read scores a goal by the best cell of a neighbourhood. The practice (FUEL, the NBVP frontier baseline,
  GBPlanner2) is to set the goal back from the frontier and require the body clear at the pose actually commanded.
- The disc compensated for goals in unknown space. Since §12, every commanded goal is known-free with ρ_agent clearance,
  so the risk is read at that cell (`arrive = 0`). It can no longer score a goal by a cell the agent is never sent to.
  - `ArrivalBottleneck` stays (rad 0).
  - The effect on the risk spread is measured on B's replay before the gate.

**The return distance for §21.4's hard rule: the swept body along the flown track (`FlownReturnSpace`).**
- **Finding (B, a1p replay of ci_tiny_0, 2026-10-23):** after §12, from t = 10 s every replan has
  `E_remaining 118.83 < 1.30 × E(return) inf`.
  - The forward rig never observes the way in, so the entrance is not known-free.
  - The rule fired on every replan, and the unreachable return left only {hold}.
- **Literature:**
  - GBPlanner builds its homing graph "alongside the actual robot odometry" (FSR 2019, §5.2); GBPlanner2's code adds
    each executed path.
  - PLGRIM keeps breadcrumb nodes from the pose graph.
  - CERBERUS's Recovery Homing backtracks the traversed path (§3.3.3).
  - The closest precedent for a never-seen start: GBPlanner2 marks the robot's own box free for its first 5 triggers.
  - Nav2's `footprint_clearing_enabled` clears only the current footprint.
- **Now:** the return space is the known-free plan, plus every cell within ρ_agent of the polyline from the nearest
  briefing entrance through the agent's track, minus every column the slab now marks occupied.
  - The track is the integrated frame positions. Entrances are the scene's `start_xy_m`, the deployment points.
  - The swept band comes from an exact L2 distance transform of the 1-px centre line, so it never claims a column the
    body did not cover.
  - It feeds only d_ret and the return option's risk. That risk is computed on the same space, with clearance
    max(clear_r, ρ_agent) on swept columns, because the body passed there.
  - Views, the exploration plan and every egress claim stay as before. The swept band is certified at agent width
    only, so it never enters a ρ_responder claim (invariant 4).
- **A lost route is its own event.** If no entrance is reachable even over the track (a collapse across it), the planner
  emits the supervisor event "return route lost" and does not restrict the options.
  - Holding forever does not prevent stranding; the agent is already cut off from the known way back.
  - No source read has a deliberate policy here. GBPlanner2 logs a warning on a `@TODO` path and keeps exploring.
    This choice is `[DESIGN]`.
- **The ×1.3 factor is the architecture's (§21.4), `[DESIGN]`.** GBPlanner2 uses an additive margin (time home >
  min(budget, battery) − 20 s); CERBERUS looks one exploration step ahead.
- **Test:** `Nbv.ReturnOverTheFlownTrackWhenTheWayInWasNeverObserved`, with the way in unobserved:
  - no track gives "return route lost", no restriction and more than one option;
  - the track gives no rule and normal options;
  - low energy gives {return, hold}, with the return 0.70 m long and its risk below 1;
  - a slab wall across the track gives "return route lost";
  - the band is pinned at 4 cells.

**64-ray view ranking and the gate's time axis:** being checked (`lit_gate_ig.md`); result below when it lands.

## 14. Gate result (2026-10-23): the claim fails on both scenes

**Verdict: "Ariadne ≥ every baseline at every time fraction" is false on `ci_tiny` and on `maze_short`.** This is a
loss, reported as one (invariant 7).

The set is 24 lockstep live runs: 4 policies × 2 scenes × 3 seeds, 200 s each, every job exit 0. They ran on AWS
c7i.4xlarge with llvmpipe, at RTF 0 with D168, from a1p `d59ee3d`. Evidence: `evaluation/runs/nbv_4.3_gate/` (summary,
rows, run ids, command, manifest, both diagnostic scripts and their outputs).

**Primary axis: coverage at T, mean of 3 seeds.**

`ci_tiny`:

| policy | T = 20 s | 40 s | 60 s | 80 s | 100 s | 200 s | AUC |
|---|---|---|---|---|---|---|---|
| ariadne | 0.387 | 0.699 | 0.846 | **0.938** | **0.952** | **0.962** | 0.884 |
| coverage_greedy | **0.468** | 0.805 | 0.876 | 0.900 | 0.930 | 0.955 | **0.892** |
| uncertainty_only | **0.468** | **0.817** | 0.875 | 0.898 | 0.922 | 0.938 | 0.883 |
| random | 0.458 | 0.760 | **0.878** | 0.910 | 0.922 | 0.933 | 0.876 |

- Losses: Ariadne is below all three baselines at T = 20, 40 and 60 s.
- From T = 80 s it is best at every point.

`maze_short`:

| policy | T = 20 s | 60 s | 100 s | 120 s | 160 s | 180 s | 200 s | AUC |
|---|---|---|---|---|---|---|---|---|
| ariadne | 0.082 | 0.302 | 0.403 | 0.455 | 0.591 | **0.696** | **0.757** | 0.440 |
| coverage_greedy | **0.112** | **0.322** | **0.497** | **0.593** | **0.668** | 0.691 | 0.713 | **0.495** |
| uncertainty_only | **0.112** | 0.310 | 0.429 | 0.444 | 0.489 | 0.518 | 0.532 | 0.399 |
| random | 0.078 | 0.274 | 0.430 | 0.446 | 0.468 | 0.468 | 0.473 | 0.374 |

- Losses: Ariadne is below coverage_greedy at T = 20–160 s, the largest gap 0.455 vs 0.593 at 120 s.
- It is below uncertainty_only at T = 20–100 s, and below random at T = 80–100 s.
- It is best at T = 180 and 200 s.

The full per-T list of losses is `gate_summary.yaml` → `losses`.

**Secondary view (pre-registered §5): coverage against summed metres flown.** Per seed, coverage is linearly
interpolated between barrier points, only within the metres that run reached; the table gives the mean over seeds.

| metres | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| ci_tiny ariadne / best baseline | 0.440 / 0.417 | 0.837 / 0.797 | 0.938 / 0.898 | 0.960 / 0.913 | 0.963 / 0.923 | 0.962 / 0.936 |
| maze_short ariadne / best baseline | 0.187 / 0.167 | 0.349 / 0.288 | 0.422 / 0.392 | 0.485 / 0.455 | 0.556 / 0.531 | 0.641 / 0.591 |

- Per metre, Ariadne is best of these four policies at every point from 5 to 30 m on both scenes, all 3 seeds in
  every mean. With the two 4.6 rows it is not best everywhere (§14.1).
- At 40 m only one Ariadne seed got that far, so the 40 m point is not compared.
- This view does not rescue the claim. The claim is on the time axis, by the §5 amendment, made before any run.

**Diagnostic (a read of the traces, not a causal ablation).**
- Ariadne chooses the nearest goals. The median path of the chosen candidate over the first 60 s is 0.84–1.70 m
  across its 6 runs, against 1.77–2.41 m for every baseline run.
- In the same time it flies the least. Summed metres by T = 60 s:
  - ci_tiny: 8.7–12.0 m, against greedy 14.4–17.5 m;
  - maze_short by T = 160 s: 26.1–29.4 m, against greedy 39.6–46.3 m.
- With a new goal every 5 s barrier and an in-place turn to each goal's yaw (the §5 pilot facts), short goals turn
  into less distance per second. The gate's time axis then charges for exactly that.
- **Reading [UNVERIFIED]:** U(a)'s path term (PathScore, λ_path(t)) is too strong for a time-axis objective in this
  rig. Ariadne's coverage per metre is the best, but its metres per second are the lowest.
- Not tested: the fix would be a time-based path cost, or goal persistence across barriers. By the design-preflight
  rule this is a design change made through §49, not a knob tuned after the failed run. It is open, after the D175 POC.

**Not in this set:** the TARE row and the 1.8 nearest-frontier row ("-"). They are batches b09–b12, the 4.6 record.

### 14.1 With the 4.6 rows: all 36 runs (2026-10-24)

The TARE row and the 1.8 nearest-frontier row ("-") ran in batches b09–b12: same rig, same commit, every job exit 0.
The four rows above are unchanged, since each run is scored on its own. The claim stays false on both scenes.

| scene | ariadne | coverage_greedy | uncertainty_only | random | tare | nearest_frontier_1.8 |
|---|---|---|---|---|---|---|
| ci_tiny AUC | 0.884 | 0.892 | 0.883 | 0.876 | **0.895** | 0.789 |
| maze_short AUC | 0.440 | **0.495** | 0.399 | 0.374 | 0.443 | 0.237 |

**New losses on the time axis.**
- ci_tiny: Ariadne is below TARE at T = 20–60 s (TARE 0.901 at 60 s vs 0.846), and below the 1.8 row at T = 20 and
  40 s (0.522 vs 0.387 at 20 s).
- maze_short: Ariadne is below TARE at T = 20 s and T = 80–160 s, and below the 1.8 row at T = 20 s (0.087 vs 0.082).
- Ariadne is above TARE at T ≥ 80 s on ci_tiny and at 180–200 s on maze_short (0.757 vs 0.681 at 200 s).

**Correction to the secondary view.** Ariadne is not best per metre at every point:
- TARE is above it at 10 m on ci_tiny (0.849 vs 0.837) and at 15 m on maze_short (0.428 vs 0.422).
- The 1.8 row is above it at 5 m on ci_tiny (0.482 vs 0.440).
- At every other point from 5 to 30 m Ariadne is best (`coverage_per_metre.txt`).

**The diagnostic reading is weaker than stated in §14.** TARE also chooses near goals: the median chosen path over
the first 60 s on ci_tiny is 1.01–1.46 m. By T = 60 s it has flown about as far as Ariadne (10.4–11.5 m, against
8.7–12.0 m), yet its coverage at 60 s is 0.901 against 0.846. So "near goals, fewer metres per second" does not
explain the whole early gap on ci_tiny. The cause stays [UNVERIFIED]; the time-axis fix waits until after the D175 POC.

## 15. §49 design preflight for the time-axis loss (2026-10-24, before any code or run): D117–D120

**Status: design only.** Nothing here is built or run. It is ready to implement once the D175 POC is up (M's order).
Every number below is from the 36 gate runs already on disk, through three committed scripts (§15.1). No new run was
made.

### 15.1 What the existing runs show (the cause, measured)

The §14.1 reading ("near goals, fewer metres per second") was [UNVERIFIED], and TARE weakened it. The GT body
trajectories (200 Hz, every run, both agents) answer the question directly: where does each policy's time go?
`time_budget.py` (10 Hz; move = planar speed > 0.03 m/s, turn = in place with |yaw rate| > 0.03 rad/s):

| scene, T ≤ 200 s | move | turn in place | still | yaw per metre |
|---|---|---|---|---|
| ci_tiny ariadne | 0.31 | **0.46** | 0.23 | **1.72 rad** |
| ci_tiny coverage_greedy | 0.44 | 0.42 | 0.14 | 1.10 rad |
| ci_tiny uncertainty_only | 0.47 | 0.38 | 0.15 | 0.88 rad |
| ci_tiny tare | 0.31 | 0.48 | 0.21 | 1.78 rad |
| maze_short ariadne | 0.32 | **0.45** | 0.23 | **1.60 rad** |
| maze_short coverage_greedy | 0.48 | 0.31 | 0.22 | 0.74 rad |
| maze_short tare | 0.34 | 0.41 | 0.25 | 1.48 rad |

(Full table, T ≤ 60 s too, in `time_budget.txt`.)

- **Ariadne turns in place for 45–46 % of the mission and moves for 31–32 %.** The mover turns at ω_max = 0.44 rad/s
  (α_max 0.6 rad/s², `command_planner.hpp` MotionLimits). A 90° turn takes ≈ 4.3 s, nearly a whole 5 s barrier.
  One radian of turning costs the time of ≈ 1.4 m of travel at the 0.6 m/s cap.
- **U(a) charges metres only.** C_path is the ρ_agent geodesic; turning is free in the utility. Every row shares the
  mover, so the row that picks the most turn-heavy goals per metre loses on the time axis. That row is Ariadne,
  together with TARE, which ranks by the same utility.
- **The turns are mostly wasted.** `turn_back.py` finds turn → move → opposite-turn triples, the "face the waypoint,
  drive, turn back to the view's yaw" pattern:
  - over 200 s, 47 % (maze_short) and 50 % (ci_tiny) of Ariadne's turning time is in such pairs;
  - in maze_short's first 60 s, where the loss is largest, the share is 67 %: a median 435° spent for 231° net;
  - over 200 s, the median pair spends 179–215° of rotation for a net heading change of 57–67°.
- **No turn goes the long way round.** `turn_episodes.py`: no continuous in-place turn exceeds 180° in any run (max
  179.8°), since `PlanCommand` wraps each turn. The waste is the pairing, not a wrap error. (The maintainer reported
  turns of "more than 180 degree … turn all the way, not back". No single turn is over 180°; this pairing is what
  the runs show.)
- **Where the pairs happen.** `still_decomp.py` classifies each pair by when its second turn starts:
  - 17–22 of Ariadne's 22 pairs per scene over 200 s start inside a command: the final-yaw turn at the end of the
    path;
  - 0–5 start at a barrier, i.e. a reversal by the next command.
  So the final yaw is the lever, and goal churn is a minor one.
- **Why the pairing happens (code).**
  - The view's yaw always faces its target: `v.yaw = atan2(target − eye)` (`nbv.cpp` SampleViews).
  - The eye is sampled FUEL-style around the target, so the travel direction into the eye is often away from the
    target.
  - The live loop sends `follow(path)` with the last pose at `face_yaw` (`face_frontier: true`, ariadne_live.cpp).
  - So the rig turns to each polyline vertex, drives, then turns back to face the target.
- **A shared command defect, found on the way.** `Phase4Planner::Route` cuts the polyline at `max_length_m` = 3.0 m
  but still attaches `goal_yaw`. On a cut command the rig turns to the view's yaw mid-route, at a point that is not
  the view, then turns back on the next command.
  - This hits the baselines more than Ariadne: the chosen path exceeds 3.0 m in 64 / 52 / 49 % of coverage-greedy /
    uncertainty-only / random decisions on ci_tiny, and in 15 % of Ariadne's.
  - So it does not explain Ariadne's loss. But it is a defect in every row.
- **Stillness, decomposed** (`still_decomp.py`, per 5 s interval and agent).
  - Two parts are the same for every row: the pauses between segments inside a command (0.4–0.6 s per interval;
    every segment starts and ends at rest), and the startup intervals with no motion (6/72 on ci_tiny, 22/72 on
    maze_short, identical across rows).
  - **Ariadne alone has 5 extra idle intervals on ci_tiny in the first 60 s (11/72 against 6/72).** That is why its
    "still" share is 0.28 against 0.21. All 5 come from one livelock, `livelock_ci_tiny_2.txt`:
    - in ci_tiny seed 2, agent 0 chose the same view for six barriers in a row (t = 20–45 s);
    - the view's goal is 0.02 m from the agent (path_m 0);
    - `Phase4Planner::Route` returns an empty path when the goal is the agent's own cell, so the loop sends `hold`;
    - the turn to the view's yaw is never commanded, the view's gain is never collected, and the view keeps winning.
  - The Phase 4 rows have no `min_goal_m` filter (only the 1.8 placeholder has one, `planner.hpp`), so at-cell views
    are admissible but cannot be executed. TARE shows the same, with 4 hold decisions on ci_tiny.
  - **This is a defect (D125), and D117 would make it worse,** since a turn-only view is cheap under an executed-time
    cost. So D125 is a prerequisite of D117.
- **Per seed, the livelock carries most of the ci_tiny loss at 40–60 s** (`gate_rows.jsonl`):

  | ci_tiny coverage, seeds 0 / 1 / 2 | 20 s | 40 s | 60 s |
  |---|---|---|---|
  | ariadne | 0.453 / 0.358 / 0.351 | 0.806 / 0.762 / **0.531** | **0.909 / 0.900** / **0.728** |
  | tare | 0.475 / 0.441 / 0.406 | 0.760 / 0.712 / 0.772 | 0.875 / 0.898 / 0.929 |
  | coverage_greedy | 0.474 / 0.530 / 0.400 | 0.793 / 0.780 / 0.841 | 0.872 / 0.886 / 0.870 |

  - On seeds 0 and 1 Ariadne is level at 40 s and best of every row at 60 s.
  - Seed 2 is the livelock (t = 20–45 s), and it pulls the 3-seed mean below every baseline. §14.1's "TARE 0.901 vs
    0.846 at 60 s" is mostly that one defect, not view placement as first read. That reading is withdrawn.
  - The loss at T = 20 s is on every seed and predates the livelock: the turning and short-command cost.
  - maze_short has no holds (0/240). Its loss to coverage-greedy at 100–160 s is on every seed at 120 s, and on
    seeds 0 and 2 at 160 s (seed 1: 0.534 against 0.492). That is the turning cost and view placement.
  - What sets Ariadne apart from the cost-free baselines is yaw per metre: many short commands, each with its own
    turns. D117 prices exactly that.
  - The five IG terms per candidate are not logged, so "Δentropy dilutes Δcoverage" is still untested. D120 adds the
    log.

### 15.2 SOTA (primary text; `docs/design/lit/early_coverage.md`)

This was read in the primary texts and pinned code, 2026-10-24. Quotes and line numbers are in the lit file.

- **Time, not distance, is the cost the field uses.**
  - FUEL (RA-L 2021, Eq. 1) charges `t_lb = max(L/v_max, Δyaw/ψ̇_max)` between views. The code (`662dd23`,
    `ViewNode::computeCost`) uses yd = 60°/s.
  - Schmid et al. (RA-L 2020, Eq. 6) use "the expected execution time", from "a constant acceleration model for
    position and yaw". That is D117's form, here exact because the mover's profile is known.
  - FUEL's `max()` holds only because a quadrotor yaws while it translates. TARE (RSS 2021, Eq. 4) is written for
    a vehicle that "can turn in one place" and adds a penalty per stop. For an in-place turner the times add.
- **Turning on the spot is a named failure.** Schmid §VII-A, on RH-NBVP: "spending a lot of time turning on the
  spot, and frequent changes of the exploration site … travels only 210 m, covering 60 %".
- **Back-and-forth between replans is a named failure.** FUEL §V-A: "back-and-forth maneuvers may be generated in
  successive planning steps and slow down the progress". FUEL's remedy is a heading-consistency term on the first
  leg (w_c = 1.5 s/rad).
  - For an in-place turner, D117 charges the actual turn from the current heading, which is the physical version
    of that term. So no w_c is added.
- **View yaw.** FUEL's views face their target, as ours do (`sampleViewpoints`: the mean bearing to the cluster's
  cells).
  - FUEL's answer to the final turn: up to 15 views per cluster with ≥ 0.8× the best coverage, and the time term
    picks among them (`getViewpointsInfo`).
  - No surveyed planner lets a view keep its arrival heading, so D118's twin is [DESIGN].
  - FUEL's multi-view set is the cited alternative. It would add FUEL's 0.8 threshold; D118 has none.
- **Gain–cost form.** NBVP and AEP use `g·exp(−λc)`; our U is the linear `g − αc`.
  - Schmid Table V (coverage at 30 min): linear 76.4 ± 6.5 %, exponential 96.6 ± 5.3 %, accumulated gain over
    accumulated time along a branch 99.8 ± 0.1 %. Their reason: "both linear and exponential penalties favor
    suboptimal viewpoints that have low cost".
  - This is not an early-rate result, and their remedy needs a horizon: a per-candidate g/t rescored every 5 s
    would favour near goals even more.
  - So it is **reserved as D121** (efficiency over the TARE local tour), not part of this change. It is the next
    candidate if §15.5 fails.
- **Refinement terms.** Schmid's single gain (Eq. 5) scores an unknown voxel 1 and a known one less, falling as it is
  re-observed. It "initially focuses on global surface exploration, before focusing on areas of low map weight" and
  was fastest to 95 % (Table IV).
  - FUEL and TARE keep gain out of the objective entirely.
  - No surveyed planner normalises each term by its own p90, which gives Δentropy the same weight as Δcoverage
    whatever their magnitudes.
  - It is a candidate cause of the maze_short gap that D117 leaves. It is **reserved as D122** (one per-voxel
    scale), decided on D120's term log, not before.
- **Early-rate metric.** Cieslewski et al. (IROS 2017, Eq. 9) define t_exp, the mean voxel discovery time, which
  is the area above the coverage–time curve. Their frontier-in-FOV rule wins most on it. Our AUC is the same
  quantity up to sign and scale.
- **Our TARE row is weaker than TARE.** TARE's code (`44500592`, `GetLookAheadPoint`, `PublishWaypoint`) does three
  things the §31.6 row omits:
  - it keeps the previous goal while it is ahead (cos > 0) and more than half the lookahead away;
  - otherwise it continues in the previous direction;
  - it extends near waypoints (8.0 / 3.5 m) so the vehicle never stops.

  The TARE row stays labelled "TARE-inspired". Any comparison to it is to that row, not to TARE.

### 15.3 OpenCV-native check

- The new cost is the duration of a trapezoidal velocity profile per turn and per segment, already implemented in
  `core/trajectory/command_planner.hpp` (`PlanProfile`, `PlanCommand`). That is the mover's own profile code.
  - The speed input differs: the mover's clearance comes from the GT world, the planner's from its own map (unknown
    counts as occupied). So C_time is an estimate [PROXY], not the executed time.
  - §15.5 measures the gap.
- No OpenCV primitive computes a kinematic profile's duration. `cv::` is not the right home and is not used.
- The extra views (D118) reuse `RaycastIg` and the existing `cv::parallel_for_` over targets (20de9e8), so their
  cost lands inside the parallel stage.
- Nothing hand-rolls what OpenCV already provides.

### 15.4 Decisions (proposed)

**D117: the motion cost is the executed time, by the mover's own model.**
- C_time(a) = Σ durations of `PlanCommand(from = agent pose and heading now, path = Route(a), final yaw = a's yaw,
  MotionLimits, clearance)`.
  - Route(a) is the string-pulled polyline the loop will actually send.
  - This covers every vertex turn, every move at its clearance-limited speed, and the final turn.
- §4's "time and energy ∝ path, so count once" premise is false in this rig (measured: turning is 45 % of the
  time). By §4's own rule, the terms come back separately:
  `U = G − [(λ_d⁰ + λ_e(t))/3]·C_path/p90 − [λ_t(t)/3]·C_time/p90 − C_b/p90 − C_r/p90`.
  - At t = 0 the weights sum to 1, exactly as before. No new constant: v, ω, α are the worldgen Motion preset.
- Cost: one `PlanCommand` per kept view (O(polyline vertices)). It needs `Route` per kept view; today it runs only
  for the chosen one (A* + string-pull per view; measured at implementation, reported).
- Heading at plan time: the yaw of the latest camera pose in the agent's map frame (the live loop already reads it
  for D169).
- **Clearance and speed.** `PlanCommand`'s speed is `min(v_cap, v_lk · clearance)`.
  - The planner evaluates clearance on its own ESDF (unknown = occupied); the mover uses the GT world. The planner's
    clearance is never larger, so C_time over-estimates in open space.
  - It is still the mover's model on the planner's information, with no new constant.
- **Horizon (preemption).** A barrier's new command replaces the running one, so the tail of a route may never
  execute.
  - C_time prices the whole command the loop sends: at most `max_length_m` = 3 m, plus its turns. That is the planner's
    commitment horizon.
  - Pricing only the part that fits before the next barrier was rejected: it would make every far goal look as cheap
    as a near one.

**D118: each view is also offered at its arrival heading.**
- For every kept view the sampler adds a twin at yaw = the heading of Route's last segment, with its own 256-ray IG
  and its own C_time (no final turn).
- U picks between the twins. No threshold, no "turn only if gain > x" knob: if facing the target pays for its
  turn, the facing view wins.
- Route returns "no final yaw" for an arrival twin. The loop already supports a pose without yaw (zero quaternion,
  D151).
- This is the "arrive and keep the travel heading, don't turn back" behaviour, chosen by the utility rather than
  hard-coded. It reads the maintainer's "come straight … not back" one way. Their answer to the open question
  (keep the heading vs drive in reverse) is pending; reversing would need a rear view the rig does not have (§1
  sensor rig).

**D119: no final yaw on a cut command (defect fix, every row).**
- When `Route` cuts the polyline at `max_length_m`, the command carries no final yaw. The view's yaw applies only
  when the command ends at the view.
- It changes every row that routes through `Phase4Planner::Route`: all four 4.3 policies and TARE. Those rows
  re-run (§15.5).
- The 1.8 row has its own `Route` (`planner.hpp`), whose yaw faces the centroid from wherever the cut leaves it. It
  is untouched, so its gated runs are reused, provided a replay identity check shows its commands are byte-identical
  at the new commit.

**D125: an at-cell view runs as an in-place turn (defect fix, every Phase 4 row).**
- When the view's goal is the agent's own cell, `Route` returns that one pose with the view's yaw instead of an empty
  path. `PlanCommand` then produces a pure turn (no move, `turn_to(yaw)`), and the gain is collected.
- No `min_goal_m` filter is added. An at-cell view is legitimate (turn to look), as long as it is executed.
- It is a prerequisite of D117 (§15.1). It is always on, with no key.

**D120: the trace logs each candidate's five normalised IG terms and its C_path, C_time, and turn time.**
- This is what makes the remaining early gap on ci_tiny (TARE vs Ariadne at equal time budgets) testable, instead
  of read from utilities.

**Goal commitment: not proposed as a decision yet.**
- D117 already favours the current goal, because its cost is computed from the current heading: a goal the rig
  already faces carries no turn.
- FUEL has no explicit hysteresis. It replans every ≤ 1.5 s, from the state 0.2 s ahead, and gets continuity from
  the consistency term plus its tour objective.
- TARE has one: it keeps the goal while the goal is ahead and > L/2 away. That adds L, TARE's lookahead distance
  (8 m indoors), which we would have to re-derive for our rig.
- Proposed: none until D117's soft commitment has been measured. §15.5 reports the goal-change rate per row (from
  `current_task.since_t_s`). TARE's rule is the cited candidate if churn stays high.
- The data supports waiting: 0–5 of 22 turn-back pairs per scene are next-command reversals (§15.1).
- **Reserved D126:** TARE's commitment rule plus waypoint extension (`GetLookAheadPoint`, `kExtendWayPoint`). It is
  the next candidate after D121 and D122. It needs its L re-derived for this rig first.

### 15.5 Pre-registered ablation (before any run; the claim and the reporting are fixed here)

- **Rig: as the gate.** Lockstep live, RTF 0 + D168, c7i.4xlarge llvmpipe, 200 s, ci_tiny and maze_short × seeds
  0–2, the same Phase 4 shadow mapper and scorer for every row.
- **Commit:** the gate tree (a1p d59ee3d) + D117–D120 and D125 only. D117, D118 and D120 sit behind mission_v2 keys,
  so one binary runs every row. The two defect fixes, D119 and D125, are always on.
- **Rows** (each 2 scenes × 3 seeds = 6 runs):

| row | D119 + D125 shared fixes | D117 time cost | D118 arrival twins | purpose |
|---|---|---|---|---|
| random, coverage_greedy, uncertainty_only, tare | ✓ | — | — | the baselines, re-run with the shared fixes |
| nearest_frontier_1.8 | n/a | — | — | gated runs reused (neither fix touches its Route), if the replay identity check holds |
| ariadne_v0 | ✓ | — | — | Ariadne as gated + the shared fixes; the reference for D117/D118 |
| ariadne_t | ✓ | ✓ | — | D117 alone |
| ariadne_ta | ✓ | ✓ | ✓ | D117 + D118 (the proposed policy) |

  That is 7 rows × 6 = **42 runs**, ≈ 22 min each on llvmpipe, the same rig and batching as the 36-run gate. It runs
  through slotq in band 30+, with the user's OK for the spend first.
- **Before any run:**
  - a replay identity check that, with every new key off, decisions and commands are byte-identical to d59ee3d
    (as for 20de9e8);
  - unit tests:
    - an arrival twin carries no final yaw;
    - a cut command carries no yaw;
    - an at-cell view yields a one-pose turn command, not a hold;
    - C_time is monotone in the turn angle and the path.
- **The C_time estimate is measured, not assumed.** Per row, it reports the chosen candidate's predicted C_time
  against the realised GT time to the goal, or to the next barrier if the command is preempted. It gives the median
  and p90 of predicted/realised.
  - A unit test that feeds `PlanCommand` and the mover the same clearance would be tautological, so none is claimed.
- **Also reported per row:** the goal-change rate (from `current_task.since_t_s`) and plan_ms. D117/D118 add
  `Route` and IG work per kept view, and that cost is stated.
- **Claim (unchanged from §5):** the proposed policy (ariadne_ta) ≥ every re-run baseline at every T = 20…200 s, per
  scene. Every loss is listed per T.
- **Also reported, not claimed:**
  - AUC per row;
  - coverage per metre;
  - the §15.1 time budget per row (turn share, turn-back share, yaw per metre);
  - ariadne_v0 against the gated ariadne row, which is the run-to-run plus D119 change.
- **Pre-registered non-regression:** ariadne_ta's coverage at 200 s ≥ ariadne_v0's minus the seed spread (max − min)
  of ariadne_v0 at 200 s. The late advantage must survive.
- **No tuning after the runs.** If the claim fails, the result is reported as a loss, and the next change goes
  through §49 again (D120's term log is the input).
- **TARE with the new utility** (tare_ta) is a separate row, not a replacement for the baseline TARE. It is added
  only if the maintainer wants the §31.6 "same utility, different scheduling" comparison redone (+6 runs).

**Amendments (2026-10-24, before any run; DECISIONS "D125/D119 implemented").**
- (a) The replay identity check is: identical with every key off, except the enumerated cut and at-cell records,
  where the fixes fire. The base is feat/ph4-nagent `9efbdba` (main 4a12b50 merged; D227 lens-off is on B's branch only, NOT in it), not d59ee3d.
- (b) The 1.8 row is re-run, not reused: **48 runs** (`configs/mission/ablation_4.3/runs_48.jobs`). These are
  development runs.
  - A run is valid only with every barrier answered and no reorder timeouts.
  - Off-AWS was tried first and declined (test-compute-session: no docker or gz on its kernels today). So they run on
    the gate's own rig (AWS c7i, llvmpipe, slotq band 30+) once the maintainer approves the spend.
  - Measured gate wall time: median 13.1 min per 200 s run (2.8–26.8, 34 logged jobs), not the ≈ 22 min estimated
    above. That puts 48 runs at ≈ 11–12 c7i.4xlarge spot-hours.
- (c) The gated `ariadne` row is context only against ariadne_v0: the base commit and the fixes
  differ.
- (d) **Predicted vs realised C_time** is measured on ariadne_t and ariadne_ta only: the rows whose trace says
  `terms.time_model: exact_command`. Every other row logs path / speed under `time_s`, and is not mixed in.
- (e) **Plan cost.** The trace records no plan time. The proxy is `lockstep.jsonl` `wall_paused_s` per barrier, which
  also includes the command round-trip to the mover. It is reported per row as such.
- (f) **First run = the abort check.** `ci_tiny 2 200 ariadne_v0` (the livelock seed) runs first. The other jobs go
  only if all of these hold:
  - every `kind:"turn"` trace record has its `mover_state` answer `accepted: true` with `segments: 1`;
  - `answer_timeouts` = 0;
  - the agent's GT shows no hold at a won at-cell view.
  The live-loop turn-here branch has never executed (no recording had an at-cell view).
- (g) **D119 evidence** is in the mover logs, not the trace (the trace carries no final yaw): no final-turn segment
  on a command whose polyline is 3.0 m.
- (h) **One commit for all 48 runs.** No merge between batches.

### 15.6 The §49 ritual, at implementation

1. ALGORITHMS.md, the NBV entry: the time-cost form (FUEL-style yaw-aware time, exact profile here) and the yaw
   twins, with the citations of §15.2.
2. Arch §21.4: C_time is the executed-command duration; the ∝-path collapse applies only to rigs that translate and
   yaw together.
3. The ablation matrix: the rows of §15.5.
4. The benchmark config: mission_v2 keys `cost_time: exact_command` and `view_yaw: target | target+arrival`, plus
   the D119 and D125 fixes (always on, no key: they are defects).
5. The run manifest: the planner block records `cost_time` and `view_yaw`, and the trace adds the D120 per-candidate
   terms.
