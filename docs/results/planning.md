# Final report — Phase 4: volumetric mapping, traversability, exploration, allocation, egress (DRAFT)

Draft for plan item 8.4, written 2026-10-24. Every number cites a run directory or a decision. Negative
results are reported as measured. Status words follow the completion register: done, partial, FAIL.

## 4.0 What Phase 4 is

Phase 4 replaces the Phase 1 placeholder planner (1.8: a 2D occupancy grid and "go to the nearest frontier") with
the full-spec planning stack behind the runtime's `MissionPlanner` seam:

| item | what it does | built from |
|---|---|---|
| 4.1 | voxel-hash TSDF at 5 cm, incremental ESDF, 3D frontiers, coarse layer | from scratch; `cv::Volume` is the oracle (it exposes no voxel read, SMOKE §4u) |
| 4.2 | traversability at two body sizes (agent ρ_a, responder ρ_r = 0.40 m), a structural-risk proxy | D111, D112 |
| 4.3 | next-best-view: five target generators, a 256-ray information-gain raycast, self-normalised utility, λ scheduling, an energy-reserve hard rule | D114 |
| 4.4 | multi-agent task allocation: CBBA (from the paper) and broker sequential-greedy, logged identically | D115 |
| 4.5 | the responder egress route: D* Lite on the ρ_responder lattice, a 5th-percentile confidence | D113 |
| 4.6 | TARE-style scheduling and a FAR-style visibility graph, as comparators | D116 |

Everything is planning-level C++: no ROS, gz or AWS type. Replay (from a recorded spool) and the live closed loop
drive the same object.

## 4.1 What was built, and the replay evidence

All replay rows use the evidence spool `aadffb8df5790ca0-20261017T072237Z` (ci_tiny, 2 agents, 199 s).

- **4.1 volumetric** (`evaluation/runs/mapping/volumetric/`, `f3eb6bb`).
  - The incremental ESDF equals brute force exactly, and incremental frontiers equal a full recompute.
  - Coverage 0.942 against 1.8's 0.933; occupied precision 0.865 against 0.707.
  - The ESDF diagonal convention was corrected 2026-10-23 (it was optimistic; docs/design/next-best-view.md).
  - Figure: `slab.png`.
- **4.2 traversability** (`evaluation/runs/mapping/traversability/`).
  - Responder space: precision 1.000, recall 0.900 (1.8: 1.000 / 0.898). Both starts reach the target.
  - The strict-3D variant is reported, not used: recall 0.354, and the target is closed.
  - **Risk proxy calibrated (D112)** against a pre-registered bar on 3 hazard seeds rendered on AWS g4dn
    (`evaluation/runs/mapping/risk_calibration/`): AUC 0.831 ≥ 0.75, control FPR 0. It enters planning.
    - It saturates: at threshold 1.0 the FPR is 0.34.
  - Figures: `trav_responder.png`, `trav_agent.png`, `trav_responder_strict3d.png`.
- **4.3 NBV** replay decision log (`evaluation/runs/planning/nbv_replay/`): 52 decisions, mean 1.98 s per agent per
  plan.
  - Two of the five gain terms have no producer on this path (pose connectivity needs 3.4 merged; loop
    disambiguation has no §17.3 alias source). They are 0, with a note in every trace record.
- **4.4 allocation** (`evaluation/runs/planning/allocation_replay/`).
  - CBBA was transcribed from the paper, with Table I tested row by row. CBBA = SGA on 40/40 replans; CBBA's score
    is 0.994 of the exact optimum on average.
  - A measured correction: CBBA = SGA exactly only at bundle length 1 (2000/2000), not at 2 (5/1000 differ). So
    L_t = 1 is the default.
  - Sparse TASK_CONSENSUS: 226 B per message, counted at the serializer.
- **4.5 egress** (`evaluation/runs/egress/routes/`).
  - D* Lite equals a fresh Dijkstra exactly on 180 random map-change rounds, with 7.6× fewer expansions.
  - 12/12 routes are valid against GT. The final route is GT-optimal: 4.441 m, the same as 1.8.
  - **The confidence calibration is degenerate**: every route is valid, and the confidences are 0.900–0.902.
    It is not evidence for the 5th-percentile confidence.
  - Figure: `egress_support.png`.
- **4.6 comparators** (`evaluation/runs/planning/route_compare/`). D* Lite is shortest; FAR and A* agree on all 12
  routes (FAR 4.403 m = the string-pulled lattice).
  - **Open bug:** on hazard seed 0, FAR gives 11.115 m against 4.424 m. Gate rows are unaffected, since the live
    `Route` uses A*.
  - The "as FAR joins start and goal" claim was wrong and is now `[DESIGN]` (D116 review).
  - **"FAR ≤ string-pulled lattice" is FALSE in general** (D116 amended, D123). On the unit test's random maps,
    once their draws are sequenced, one instance routes 37.40 cells against 33.24 (+12.5 %). The test is quarantined
    and OPEN. This report makes no FAR ≤ lattice claim.

## 4.2 The 4.3 gate: FAIL on both scenes

**Pre-registered claim** (next-best-view.md §5, fixed before any run): Ariadne's coverage ≥ every baseline at every
T = 20…200 s, per scene.

**Rig.**
- 36 lockstep live runs: 6 policies × {ci_tiny, maze_short} × seeds 0–2.
- 2 agents, 200 s of sim time.
- AWS c7i.4xlarge spot, llvmpipe rendering, RTF 0 plus D168, so the world pauses at every 5 s replan barrier.
- One binary (a1p `d59ee3d`). Every policy is scored by the same shadow mapper.
- Run ids: `evaluation/runs/planning/nbv_gate/runs.txt`; command: `command.txt`.
- Scores: `gate_rows.jsonl`, `gate_summary.yaml`, made by `python/analysis/gate_coverage.py`.

**Figure 4.1** (`evaluation/runs/planning/nbv_gate/coverage_time.png`, by `plot_gate.py`): coverage against sim time.
Each line is the mean over 3 seeds; the band is min–max.

**Figure 4.2** (`coverage_metre.png`, same script): coverage against metres flown, summed over both agents, per
run. This is the secondary view.

**Table 4.1.** Coverage, mean of 3 seeds (`gate_summary.yaml`). AUC is over 0–200 s.

| scene | policy | 20 s | 40 s | 60 s | 100 s | 160 s | 200 s | AUC |
|---|---|---|---|---|---|---|---|---|
| ci_tiny | **Ariadne** | 0.387 | 0.699 | 0.846 | 0.952 | 0.962 | 0.962 | 0.884 |
| ci_tiny | TARE-inspired | 0.441 | 0.748 | 0.901 | 0.946 | 0.959 | 0.960 | **0.895** |
| ci_tiny | coverage-greedy | 0.468 | 0.805 | 0.876 | 0.930 | 0.953 | 0.955 | 0.892 |
| ci_tiny | uncertainty-only | 0.468 | 0.817 | 0.875 | 0.922 | 0.937 | 0.938 | 0.883 |
| ci_tiny | random | 0.458 | 0.760 | 0.878 | 0.922 | 0.930 | 0.933 | 0.876 |
| ci_tiny | nearest frontier (1.8) | 0.522 | 0.725 | 0.802 | 0.812 | 0.821 | 0.823 | 0.789 |
| maze_short | **Ariadne** | 0.082 | 0.197 | 0.302 | 0.403 | 0.591 | **0.757** | 0.440 |
| maze_short | TARE-inspired | 0.092 | 0.182 | 0.299 | 0.446 | 0.602 | 0.681 | 0.443 |
| maze_short | coverage-greedy | 0.112 | 0.204 | 0.322 | 0.497 | 0.668 | 0.713 | **0.495** |
| maze_short | uncertainty-only | 0.112 | 0.208 | 0.310 | 0.429 | 0.489 | 0.532 | 0.399 |
| maze_short | random | 0.078 | 0.159 | 0.274 | 0.430 | 0.468 | 0.473 | 0.374 |
| maze_short | nearest frontier (1.8) | 0.087 | 0.127 | 0.201 | 0.253 | 0.291 | 0.307 | 0.237 |

**Result: the claim is false on both scenes, on all 36 runs.**
- ci_tiny: Ariadne is below every Phase 4 baseline (TARE-inspired, coverage-greedy, uncertainty-only, random) at
  T ≤ 60 s, and below the 1.8 placeholder at 20 and 40 s. It is best of all six rows at every T from 80 s.
- maze_short: Ariadne is below coverage-greedy at every T ≤ 160 s. It is best only at 180–200 s.
- On AUC, Ariadne is third of six on both scenes (ci_tiny 0.884 behind 0.895 / 0.892; maze_short 0.440 behind
  0.495 / 0.443).
- What Ariadne does win:
  - final coverage on maze_short (0.757 against 0.713);
  - coverage per metre flown, best at every point from 5 to 30 m on both scenes (next-best-view.md §14, `coverage_per_metre.txt`).
- Neither win rescues the pre-registered time-axis claim, and neither is presented as if it did.
- The Phase 1 placeholder (1.8) is last on AUC on both scenes, though it leads ci_tiny at 20 s. Phase 4 beats the placeholder; it does not beat the
  cost-free baselines on time.

**On the TARE row.** Our TARE row is weaker than TARE as published: it has no goal commitment and no waypoint
extension (next-best-view.md §15.2; TARE code `44500592`). It is labelled "TARE-inspired", and any comparison is to that row
only.

## 4.3 Diagnosis: why the time axis was lost (measured from the gate runs' GT; no new run)

From the GT body trajectories of the 36 runs (`evaluation/runs/planning/nbv_gate/{time_budget,turn_back,turn_episodes,still_decomp}.txt`).
next-best-view.md §15.1 has the full analysis.

1. **Ariadne spends its time turning in place.**
   - It turns for 45–46 % of the mission and moves for 31–32 %, at 1.6–1.7 rad of yaw per metre.
   - Coverage-greedy moves 44–48 % of the time at 0.7–1.1 rad/m.
   - One radian at the mover's ω_max (0.44 rad/s) costs as much time as 1.4 m of driving. Yet Ariadne's utility
     charges metres only.
2. **About half of that turning (47–50 %) is turn → drive → turn back.**
   - Every view faces its target, and views are sampled around the target. So the rig turns to the path, drives,
     then turns back to face the target.
   - 17–22 of 22 such pairs per scene are the final-yaw turn inside one command, not a reversal by the next command.
3. **One livelock carries most of ci_tiny's 40–60 s loss.**
   - In seed 2, agent 0 chose a view whose eye was 2 cm away, in its own grid cell, for six barriers in a row
     (t = 20–45 s).
   - The router returned an empty path for an at-cell goal, so the loop sent `hold`. The turn to the view's yaw was
     never commanded, the view's gain was never collected, and it kept winning (`livelock_ci_tiny_2.txt`).
   - On seeds 0 and 1, Ariadne is best of every row at 60 s (0.909 / 0.900 against TARE-inspired 0.875 / 0.898).
   - The loss at 20 s is on every seed and predates the livelock: that is the turning cost.
4. **A shared defect:** a command cut at 3 m still carried the view's final yaw. This hits the baselines more than
   Ariadne: their chosen path exceeds 3 m in 49–64 % of decisions, Ariadne's in 15 %.

## 4.4 The fix, through the §49 ritual (implemented 2026-10-24; not yet run)

Before any code: a primary-source literature pass (the literature notes `early_coverage` (archive/dev-history branch): FUEL, TARE, Schmid et al.,
AEP, NBVP, Cieslewski et al.), the OpenCV-native check, and pre-registration (next-best-view.md §15). The decisions:

- **D125 (defect, every row): an at-cell view runs as an in-place turn.**
  - The command protocol gains an explicit "turn here" (a NaN position), not a tolerance.
  - The mover plans from its true pose, a few cm from the planner's estimate. A finite waypoint there would first
    turn to face that offset.
  - Unit tests: from 8 nearby poses, the command is exactly one turn segment of the mover's profile. Integrating
    what that turn sees collapses the view's gain more than 5×, so the view stops winning.
- **D119 (defect, every row):** a cut command carries no final yaw.
- **D117: the motion cost is the executed time.** C_time is the duration of the mover's own command planner
  (`PlanCommand`: per-turn and per-move trapezoidal profiles, clearance-limited speed) over the exact command the
  loop will send, from the current heading.
  - The time schedule's third of the path weight moves onto C_time. At t = 0 the weights still sum to 1, with no new
    constant.
  - [PROXY]: the planner's clearance comes from its own map, the mover's from GT. Predicted against realised time
    is measured per row.
- **D118: arrival-heading twins.** Each view is also offered at the heading of its last move, with its own IG and
  no final turn. The utility picks between them, with no threshold. [DESIGN]: no surveyed planner does this; FUEL's
  multi-view set is the cited alternative.
- **D120:** the trace logs every candidate's terms, so "Δentropy dilutes Δcoverage" becomes testable rather than
  read off utilities.
- **Reserved, not built:** D121 (efficiency over a tour, Schmid's best form), D122 (one per-voxel gain scale),
  D126 (TARE's goal commitment).

**Pre-registered evaluation (next-best-view.md §15.5 with its amendments).**
- 8 rows × 2 scenes × 3 seeds = 48 lockstep runs: the four baselines and the 1.8 row re-run with the fixes,
  ariadne_v0, ariadne_t (D117) and ariadne_ta (D117 + D118).
- Off-AWS **development evidence**. The gate evidence comes later, on AWS.
- The claim is unchanged.
- Non-regression: ariadne_ta's coverage at 200 s ≥ ariadne_v0's minus its seed spread.
- No tuning after the runs. A loss is reported as a loss.

**Abort check (job 1 of 48), done 2026-10-24: GO** (`evaluation/runs/planning/nbv_ablation/abort_check/`, run
`20261024T055409Z-cpux86-8d7693`).
- On the livelock seed (ci_tiny seed 2, ariadne_v0), the at-cell view ran as one accepted 7.84 s turn.
- Every barrier was answered, and no view was held.
- Development observation, one run: coverage at 60 s 0.878, against the gated 0.728 on the same seed. The 20 s value
  is unchanged (0.351).

**[RESULT PENDING].** The other 47 runs wait on spend approval. This section gets the 48-run table
and its losses per T, whatever they are.

## 4.5 Limitations (stated, not mitigated)

- **4.3** is not shown better than cost-free baselines on time. Its advantages are late coverage and per-metre
  coverage.
- **4.5**'s confidence is uncalibrated: the calibration is degenerate.
- **4.2**'s risk proxy saturates (FPR 0.34 at threshold 1).
- **4.6:** the TARE row is TARE-inspired. FAR has open routing defects: hazard seed 0, and a random-map instance
  where it is 12.5 % longer than the lattice. So "FAR ≤ lattice" is not claimed.
- **Two of the five gain terms are 0** in every run (no producer).
- **The gate rig is llvmpipe on CPU.** Every row consumed the same rig, but the absolute times are not
  real-robot times.
