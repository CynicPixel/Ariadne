# Design — Phase 1.8: placeholder mission layer (occupancy, nearest frontier, egress route)

Plan 1.8: "occupancy from raw depth into a plain voxel grid, nearest-frontier policy, A* route to exit at ρ_agent
(§20–§22 interfaces, minimal internals)". Runs in replay from the 1.7 spool (keyframe depth + σ + poses + the 1.6
merge), deterministically, in C++ (`cpp/planning/`).

**ρ discipline (invariant 4 wins over the plan's wording).** The plan says "A* route to exit at ρ_agent". Arch §20.5/
§22.1 and invariant 4 say every egress claim is computed at ρ_responder. So 1.8 computes BOTH and labels them:
the **egress route** (the product, every egress metric) at ρ_responder = 0.40 m, and the **agent route** at
ρ_agent = 0.20 m (what an agent could drive). Values from `scene.json` `clearance`, mirrored in the config.

## Pipeline (placeholder internals behind §20–§22 interfaces)

```text
spool keyframes (time order, both agents)          mission briefing (scene.json, NOT GT layers)
  depth + σ, T_O_C, t_ns, agent                       entrances (start pose per agent) = exits
        |                                             target_xy_m (where the casualty was reported)
        v
3D occupancy voxels (§20.2 tri-state: unknown / free / occupied), world frame, 0.05 m
  ray casting: cells along each depth ray → free; the end cell → occupied (log-odds counts)
        v
2D traversability slab (z band of a standing body): occupied if any occupied voxel in the band,
  free if observed free in the band and not occupied, else unknown
        v
clearance = cv::distanceTransform(DIST_L2, DIST_MASK_PRECISE) over occupied (exact Euclidean in 2D, §20.3 proxy)
        v
frontiers (§20.4): free cells 4-adjacent to unknown, cv::connectedComponentsWithStats → clusters
nearest-frontier policy (§21 placeholder): per agent, the cluster with the shortest ρ_agent A* path
egress route (§22.1): A* (8-connected, octile) target → nearest exit over OBSERVED-free cells with
  clearance ≥ ρ_responder (unknown is not free: "observed, not merely not contradicted", §22.2)
```

Cadence: keyframes are integrated in timestamp order; every `replan_period_s` (5 s mission time) the planner
recomputes frontiers, the policy decision per agent, and both routes, and appends one JSONL record per decision to
the **decision trace** (the start of the agentic evidence, invariant 3: map → decision → (would-be) action).

## Literature basis (placeholders are the field's standard baselines, not inventions)

- **Occupancy (§20.2 placeholder): OctoMap's sensor model** (Hornung et al., *Autonomous Robots* 2013): per-voxel
  log-odds, hit 0.7 / miss 0.4, clamped to [0.12, 0.97], occupied above 0.5, unknown = never updated (the
  free/occupied/unknown tri-state of §20.2). Library defaults (OctoMap, MRPT) `[VERIFIED web 2026-10-15]`; the
  paper's own parameter section not read. Stored as fixed-point integer log-odds (×1000) so x86 and Arm agree.
- **Frontiers + policy (§20.4/§21 placeholder): Yamauchi 1997** frontier-based exploration — frontier = free cell
  bordering unknown; go to the closest frontier by path length. It is the greedy baseline that the modern planners
  are measured against: FUEL (Zhou et al., RA-L 2021) reports 3–8× faster exploration than greedy/closest-frontier
  strategies via an incremental frontier structure and hierarchical planning; TARE (RSS 2021) and FAR Planner
  (IROS 2022) are the SubT-proven, sensor-agnostic planners. The full survey (algorithms.md Domain 3) is still
  owed and is the Phase 4 input; nearest-frontier stays as the ablation baseline (placeholders are never deleted).
- **Route (§22.1 placeholder): A\*** on the 8-connected grid with an octile heuristic (admissible), worldgen's
  corner rule (a diagonal step needs both orthogonal neighbours traversable). D\* Lite replaces it in Phase 4.

## Clearance, ray carving and the 2D slab (fixed before the first measurement)

- **Distance convention = GT's** (`worldgen/raster.py`): distance from a cell centre to the nearest EDGE of a
  blocked square (per axis max(|d| − 0.5, 0) cells), traversable iff ≥ ρ − 1e-9. `cv::distanceTransform` is
  centre-to-centre and would disagree by up to half a cell along every wall, so the C++ port is exact and tested
  against it and against a hand-computed doorway.
- **Two clearances, deliberately asymmetric.** Egress route (ρ_responder): distance to the nearest cell that is NOT
  observed free (occupied OR unknown) — the route may not pass near unobserved space. Agent route and frontier
  policy (ρ_agent): distance to occupied only — an agent may drive toward the unknown, that is exploration.
- **Ray traversal: Amanatides & Woo**, "A Fast Voxel Traversal Algorithm for Ray Tracing" (Eurographics 1987) —
  the exact voxel walk OctoMap's ray casting uses; fixed-step sampling skips corner voxels and double-counts others,
  exactly at the wall edges that decide doorway clearance. Tested against brute force on axis-aligned and diagonal
  rays.
- **Ray carving:** every `pixel_stride`-th depth pixel with Z ≤ `max_depth_m` and σ_Z ≤ `max_sigma_m`; cells from
  the camera to Z − k·σ_Z get a miss, the cell at Z a hit (no free carving within k·σ of the surface, so noisy far
  depth cannot erode walls). σ_Z is the spool's SGBM sigma (Z²σ_d/(f·B) at full resolution); `max_sigma_m` and `k`
  are set from that model, not from trial runs.
- **Frontier clusters smaller than `min_frontier_cells`** (≈ 2ρ_agent/g) are ignored and counted in the trace, so the
  policy does not chase pinholes left by the pixel stride. `[UNVERIFIED]` that Yamauchi 1997 sizes frontier regions
  by the robot; the threshold is ours, fixed before measuring.
- **Slab rule (a standing body, `z_band_m`):** a column is OCCUPIED if any voxel in the band is occupied, FREE if at
  least one voxel in the band is observed free and none occupied, else UNKNOWN. ("Whole band observed" would mark
  the agents' own corridor unknown: from ~1 m height with ±28° vertical FOV the top of the band is never seen
  nearby.)

## Frames and what the planner may know

- The planner works in the **world frame W** of the mission briefing. `T_W_O0 = T_W_B0(entry) · T_B_C`: agent 0's
  entry pose (entrance position + yaw at the briefed start height) and the rig's body-to-rectified-camera
  extrinsic. `[ASSUMPTION]` a responder team knows where it entered; this is the only world anchor, and it is NOT
  ground truth (no GT trajectory, GT map or GT traversability is read by the planner). The spool gains `T_B_C` per
  agent (additive field; schema stays 1; the Python loader ignores unknown keys).
- **Only agent 0's entry pose anchors W.** The briefing also lists agent 1's entry; using it would make the 1.6
  loop redundant for fusion. Agent 1 is placed in W solely through the verified inter-agent loop (`T_O0_O1`), which
  is what keeps the collaboration claim meaningful.
- **Cross-check (evaluator assertion):** agent 0's estimated body poses over the first 10 s, mapped through the
  briefing T_W_O0, must match GT to ≤ 1 cm (VO drift is mm there); a wrong T_B_C direction or an optical/rectified
  mix-up gives cm–dm. (A global Umeyama would absorb 200 s of drift, so it is the wrong oracle.) O0 is the
  rectified-left camera at frame 0 (`stereo_vo.cpp`: "the first frame defines the world"). The entry height comes
  from the route script's start (1.0 m), `[ASSUMPTION]` with the entry pose.
  T_B_C comes from `mission_default.json` (`t_B_camera_m`, `R_B_optical_rows`; rectification is ~identity on the
  ideal sim rig) — the check is what licenses that.
- **Agent 1 joins the map only after the merge** (`T_O0_O1`, 1.6), at `t_merge` = the later of the two matched
  keyframes' stamps. Before that its keyframes are held back, and the trace records agent 1's policy step as `"no decision: not merged"`.
  `[ASSUMPTION]` the 1.6 merge edge was picked with
  hindsight (best-ranked of all candidates); 1.8 uses its timestamp as if it had been found then.
- GT is read only by the evaluator: `scene_gt.npz` (`free`, `trav_agent`, `trav_responder`, GT routes and geodesic
  lengths), on the identical 0.05 m grid (origin (−3.5, −0.5), row 0 south), so comparisons are cell for cell.

Determinism: keyframes integrated in (t_ns, agent) order, integer log-odds, A\* ties broken by (f, g, cell
index); two runs must give byte-identical trace and results.

## Measured outputs (§30.2 mission + §22.2 egress metrics)

- **Time-to-first-valid-route** (§22.2): the first replan time whose egress route exists with every cell observed
  free at ρ_responder; and whether that route is valid against GT `trav_responder`.
- **Route validity vs GT**: fraction of route cells inside GT `trav_responder` (and `trav_agent` for the agent
  route); route length vs the GT geodesic length from the same exit.
- **Route confidence** (§22.2): 5th percentile of per-cell support along the route; support = observed-free count
  saturating at `support_full_hits` (`[PROXY]` for TSDF weight until Phase 4).
- **Coverage**: fraction of GT `free` cells the map has observed free; map accuracy: fraction of map-free cells that
  are GT free and map-occupied cells that are GT non-free.
- Frontier count over time, policy decisions (JSONL, every record `mode: replay, executed: false` — in replay the
  agents follow their scripts, so these are logged decisions, never closed-loop agentic evidence), runtime per
  replan.
- **Collaboration comparison (three rows):** merged; agent 0 alone; agent 1 alone anchored by its own entry pose
  (`--agents merged|0|1`; the merged run still never uses agent 1's entry pose). Each exit needs an
  observed-free 0.4 m disk, and agents enter facing inward: agent 0 faces its exit only on the exit leg (~185 s),
  agent 1 faced its exit on its return leg (~44–56 s) but joins W only at t_merge (~138 s). The difference in
  time-to-first-valid-route between the two runs is a measured collaboration effect, reported whichever way it goes.
  Endpoints are never snapped to nearby cells.
- **Map-vs-GT image** (first dev output): GT grid with map free/occupied/unknown, both routes and the GT route; the
  egress route also goes onto the viewer page.

## Code layout

`cpp/planning/volumetric/occupancy_grid.hpp` (3D tri-state + ray integration + 2D slab),
`cpp/planning/frontier/frontiers.hpp`, `cpp/planning/route/astar.hpp` (grid A* with clearance gate),
tool `cpp/runtime/tools/ariadne_mission.cpp` (reads the spool + briefing, writes results/trace/routes), config
`configs/mission/mission_v1.yaml`, `core.sh mission`. The viewer page gains the egress route.

## Not in 1.8

TSDF/HashTSDF (§20.1, Phase 4), ESDF in 3D and incremental (§20.3), D* Lite (§22.1), NBV utility, TARE, CBBA,
structural risk. Closed-loop exploration (the agents follow the policy) needs live runs; in replay the policy's
choices are logged, not executed.
