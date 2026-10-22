# 4.5 Egress (I12) — ρ_responder lattice, D* Lite replanning, 5th-percentile route confidence (A1)

Status: design reviewed (advisor, 2026-10-23; §8), before any code. Decision: D113. Arch §22.1–§22.3, invariant 4. Consumes D110
(the map) and D111 (egress traversability, clearance, risk). The 1.8 egress (A* on the slab, D45) stays as the
ablation baseline.

## 0. Seam (B's L1 / 7.x contract, 2026-10-23, maintainer's option B)

Phase 4 implements `runtime/live/planner.hpp`'s `MissionPlanner` (`Integrate`, `Plan`, `Route`), on
`feat/ph7-agentic`. B's requirements:
- `Plan` is deterministic given its inputs (replay = live), works in sim time, with no wall-clock work;
- every option carries a risk in [0, 1] and a reason (the §23 gate reads it);
- operator directives are honoured: keep_out drops, assign / explore restrict, priority_region weights;
- `Route` returns world-xy waypoints.

The egress route goes into `PlanOutput.maps` (MapSummary's egress cells). Until B's header is on a branch A1 can build
on, 4.5 is a library plus a replay evaluator. The adapter is one class.

## 1. Preflight

- **D* Lite** (Koenig & Likhachev, AAAI 2002), **Figure 3, transcribed from the rendered page** (its text layer does
  not extract):
  - CalculateKey(s) = [min(g, rhs) + h(s_start, s) + k_m; min(g, rhs)];
  - UpdateVertex: `rhs(u) = min over Succ(u) of c(u, s') + g(s')` unless u is the goal, then re-queue if
    inconsistent;
  - ComputeShortestPath: pop while TopKey <· Key(s_start) or rhs(s_start) ≠ g(s_start). An overconsistent vertex
    sets g = rhs and updates its predecessors; otherwise g = ∞ and it updates its predecessors and itself;
  - Main: on changed edge costs, `k_m += h(s_last, s_start)`, then UpdateVertex on each changed edge's tail, then
    recompute.
  - It searches **from the goal**, so g is cost-to-goal and the path is read greedily from s_start. The heuristic
    must be consistent.
- **Our case has a fixed start.** The egress route is target → exit for a responder; nobody moves along it while
  the agents map. So k_m stays 0, and D* Lite reduces to its incremental core (LPA* with the search direction
  reversed). It is still the right tool (§22.1): the *map* changes under the plan, and each replan touches only the
  vertices whose costs changed.
- **Many exits:** every traversable exit is seeded with `rhs = 0`, a multi-source start of the backward search. The
  heuristic is `h(s) = octile(target, s)`, the distance from the search's *start*, the target cell. It is consistent
  because every edge cost is ≥ the octile length. (Review point c: a virtual goal with a 0-cost edge would need care
  to keep `h` consistent.)
- **FAR planner** (Yang et al., IROS 2022) is the exploration-time visibility-graph router. It is the §31.6
  comparator and lives in 4.6, not here.
- **OpenCV-native:** none; graph search over our lattice. `planning::AStar` (1.8) is the oracle's baseline.

## 2. The lattice and the edge cost (§22.1, as built in D111)

- **Nodes:** the egress-traversable columns of D111 (observed column ∧ occupied-field capsule, at ρ_responder).
  8-connected, with 1.8's corner rule: a diagonal needs both orthogonal neighbours traversable.
- **Edge cost** (arch §22.1, transcribed):
  `c(x→y) = ‖x−y‖ · (1 + w_u·U(x,y) + w_r·Risk(x,y) + w_c·max(0, ρ_safe − ESDF)/ρ_safe)`.
  - **U = 1 − support**, the mean over the edge's two endpoints. **support(column)** = the fraction of the
    standing-body envelope (voxels within ρ of the capsule axis) that is **observed**, i.e. known.
    - This is the quantity whose absence made strict 3D unusable (D111). Hybrid traversability admits a column on
      the observed-column rule; support says how much of the body's volume was actually seen.
    - It reads the TSDF weights, so §22.1's "U derived from observation count and TSDF weight" is literal.
  - **Risk:** D111's proxy, **only if it passes D112's bar** (evaluation AUC ≥ 0.75). Otherwise `w_r = 0`, and the
    manifest says so.
  - **Clearance term:** ESDF = D111's per-column egress clearance; ρ_safe = d_max = ρ + 2 voxels (0.50 m).
  - **Weights w_u = w_c = w_r = 1, fixed a priori.** Every multiplier is ≥ 1, so the octile distance is a
    consistent heuristic. The ablation rows are shortest (all weights 0) vs weighted.
- **Never an egress edge into a non-traversable column**, whatever the costs: invariant 4 is a constraint, not a
  term.

## 3. Route confidence (§22.2)

`confidence(R) = 5th percentile of support over the route's cells` (arch: "the 5th percentile of per-voxel support
along the route"). On the 1.8 baseline the same statistic used its slab support (D45).

## 4. Replanning (§22.3)

- Every replan period (5 s of mission time, `mission_v1`), and on each keyframe once the live seam exists:
  - recompute D111's maps (incremental ESDF already);
  - diff the per-column (traversable, cost) against the previous replan;
  - for every changed column, call UpdateVertex on its **whole 3×3 block** (review point a). The corner rule makes a
    diagonal between two neighbours depend on the changed column, and a cost change alters all 8 edges at it;
  - ComputeShortestPath.
- Triggers logged, per arch §22.3: a new obstacle (a column that left traversable), confidence fell, risk changed.
  Agent and comms events are L1's; their hooks are left in the interface.
- **Invalidate-and-reroute (plan 4.5):** a route cell that becomes non-traversable makes its edges ∞. D* Lite
  re-routes, and the trace logs `reroute: obstacle at (r, c)`.

## 5. Metrics (§22.2, the Mission block)

1. **Time-to-first-valid-route:** the first replan whose route exists with every cell traversable under D111, i.e.
   observed free at ρ_responder.
2. **Route validity vs GT:** the fraction of route cells in GT `trav_responder`, and whether all are.
3. **Confidence calibration:** every replan's route is one sample (predicted confidence, realized validity),
   pooled over runs. The plot emits even if validity is all ones; a degenerate plot is reported as such.

## 6. Tests

1. **D* Lite = Dijkstra, driven through the evaluator's own cell-flip diff** on the corner-ruled lattice (review point
   a). Random grids with random per-cell costs; 30 rounds of random **cell** flips (traversable ↔ not, cost up or
   down). Each round goes through the same diff → 3×3 UpdateVertex path the evaluator uses. g(start) and the
   extracted path cost equal a fresh Dijkstra, exactly. It expands fewer vertices than the fresh searches, summed
   over the rounds.
2. **Many exits:** on a map where the exit nearest by octile is **not** the cheapest, the route takes the cheapest,
   as Dijkstra from each exit does.
3. **Corner rule:** no diagonal through a blocked corner.
4. **Invariant 4:** no route cell is non-traversable, ever; a test sweeps random maps.
5. **Obstacle-after-route scenario** (exit gate): a route is planned, then a doorway on it closes. The next replan
   reroutes through the other exit or reports none, and equals a fresh search.

## 7. Evaluation (replay)

- **Before scoring validity, check support's range (review point b).** Compute support along the evidence spool's
  final hybrid route, support only, no GT. If it is flat along the route (spread < 0.1, because the body envelope
  above 1.3 m is never observed, D111 §6), switch support to the band the rig can observe (the capsule axis inside
  the camera's vertical field) and record the switch.
  - Pre-registered: the support definition used for the calibration rows is the one fixed by this range check, made
    before any validity is computed.
  - Implementation: per z-layer known-masks, convolved with the capsule disc (`cv::filter2D`), not a per-column loop.
- **Exit-gate reroute (review point d):** every reroute in the replan loop is logged with its cause. If none occurs on
  the spools, the gate's "obstacle discovered after route reroutes" stays **open**. Test 5 proves the mechanism and
  does not stand in for a run.

- On the evidence spool, the replan loop at 5 s gives, per replan: the route, its length, confidence and validity
  against GT, plus the time-to-first-valid-route. Side by side with 1.8's 140 s on the same spool.
- The same on the three hazard seeds, once their spools exist: the risk-aware rows.
- The live obstacle-after-route scenario is an L1 run (a prop appears behind the agents). It is listed for 8.x;
  replay covers test 5.

## 8. Review (advisor, 2026-10-23)

1. **3×3 invalidation**, with the D* Lite = Dijkstra test driven through the cell-flip diff: adopted (§4, §6).
2. **Support's range checked before it becomes the confidence**, with an observable-band fallback pre-registered:
   adopted (§7).
3. **A consistent heuristic with many exits:** seeded rhs and the octile from the target (§1); a test with a
   non-nearest cheapest exit (§6).
4. **The gate's reroute must be a run-level instance**, not the unit test: adopted (§7).
