# 4.2 risk calibration — worldgen hazard geometry, GT risk zones, the calibration protocol (A1)

Status: design fixed before any hazard scene exists (2026-10-23). Decision: D112. Calibrates D111's structural-risk
`[PROXY]` (`TRAVERSABILITY_4.2.md` §3, arch §20.5: "calibrated against the GT-labeled zones in the simulated
scenes"). Renders on AWS g4dn, as the maintainer approved on 2026-10-23.

## 1. Preflight

**What a rescuer calls a structural hazard.** These are secondary sources: fire-service training summaries of the
US&R collapse patterns, not FEMA's own text, which the search did not return
([Firehouse, "Building Collapse: Maximum Survivability Within the Debris"](https://firehouse.com/21259455),
[CNN on Surfside](https://edition.cnn.com/2021/06/25/us/pancake-collapse-explainer-trnd/index.html)). The patterns
are lean-to, V, pancake, **cantilever** ("the most dangerous type of collapse to operate in", needing shoring before
operations) and A-frame. Survivable voids are most likely in V, lean-to and cantilever collapses.

**What v1 models, and why:**
- **Cantilever slab:** a slab held by one wall, projecting over the floor above head height.
- **Hung mass:** a debris block suspended over walkable floor.

Both stay **above a standing responder** (underside ≥ 1.95 m, the body is 1.8 m). So GT traversability does not
change: a hazard *threatens* a route without *blocking* it, which is exactly what a risk term must detect and an
occupancy test cannot.

Lean-to and V slabs reach the floor. They would need GT traversability computed from a 3D occupancy, and the GT is a
2.5D raster, so they are out of v1 and stated as such.

**No published TSDF-based collapse-risk proxy was found** (`TRAVERSABILITY_4.2.md` §1). The calibration is ours,
against our labels.

**OpenCV-native:** the scenes are worldgen's (meshes and SDF). Nothing here is image analysis.

## 2. Worldgen changes (opt-in; existing scenes byte-identical)

- **New preset section `hazards`** (`params.Hazards`): counts `cantilevers` / `hung` (default `(0, 0)`), sizes,
  underside height, and `zone_margin_m`. **When it is at its default, it is left out of `Preset.to_canonical()`.** So
  every existing preset digest, scene id (`aadffb8df5790ca0`, …) and golden scene-directory digest is unchanged.
  `test_golden.py` and a new digest test prove it.
- **New module `worldgen/hazards.py`:** placement from its own Philox stream (`S_HAZARDS`), so no other stream's
  draws move.
  - **Cantilever:** in a room, a random wall side and position along it. Projection 0.8–1.6 m, width along the wall
    1.0–2.0 m, thickness 0.15–0.25 m, underside 1.95–2.25 m.
  - **Hung mass:** a box 0.4–0.8 m on a side, 0.3–0.5 m tall, underside 1.95–2.20 m. It hangs from the ceiling by an
    implicit cable (not modelled: a cable is sub-voxel).
  - **Keep-outs:** a 1 m square at the room centre (the target point and the ceiling light; the props' 2.6 m square
    would leave less than a cantilever's minimum projection in ci_tiny's 4 m rooms), and other hazards plus a 0.3 m
    gap. No hazard in a passage or a vestibule.
- **Geometry:** a new surface kind `K_HAZARD`, with six faces per box, **bottom included**. The underside is what the
  overhang term looks for.
- **GT (`scene_gt.npz`, only when the scene has hazards):**
  - `risk_zone`: the union of hazard footprints dilated by `zone_margin_m` = 0.5 m. A debris-fall margin, labelled
    `[ASSUMPTION]`.
  - `hazard_footprint`: the footprints alone.
  - `scene.json` gains a `hazards` list (kind, room, rect, z range).
  - `free`, `trav_*`, `esdf_m` and the routes are unchanged by construction; a test asserts it.
- **New preset `ci_tiny_hazard`:** `ci_tiny` plus `hazards: {cantilevers: [2, 3], hung: [1, 2]}`. Seeds 0, 1, 2.

## 3. Calibration protocol (fixed now)

- **Data:** per seed, render the default mission (AWS g4dn, NVIDIA EGL), then run the Phase 1 estimator chain to a
  spool, then build the 4.1 map (`ariadne_volumetric_eval`), the same pipeline as the evidence spool.
- **Score per GT cell** in the free walkable area (GT `trav_responder`), the population a route can use:
  - the label is `risk_zone`;
  - the score is the proxy `risk`;
  - only **observed** cells count (the TSDF slab is free there). An unobserved hazard is a coverage question for 4.3,
    not a calibration one.
- **Metrics:** ROC AUC, and the threshold at **recall 0.9** (a rescuer would rather be warned too often), with its
  false-positive rate.
- **Split:** the threshold is fixed on **seed 0**. AUC, recall and FPR at that threshold are reported on **seeds 1
  and 2** only.
- **Ablations, the same split:** each term alone (overhang, unsupported) against the noisy-OR. A term with AUC ≤ 0.6
  on seed 0 is dropped and reported as such.
- **Hazard-free control:** the evidence spool (ci_tiny, no hazards). The FPR at the seed-0 threshold is reported.
- **Bar for using the proxy in planning** (4.3 C_risk, 4.5 route cost): evaluation-seed AUC ≥ 0.75. Below it, the
  proxy stays out of planning and the negative result is reported (invariant 7).

## 4. Tests

1. **Golden:** every existing preset and seed gives the same scene-directory digest (`test_golden.py`, unchanged),
   and the same `Preset.digest()` as recorded before the change.
2. **Hazards change no GT traversability:** `ci_tiny_hazard` seed 0's `free` / `trav_*` equal those of the same
   layout without hazards.
3. **Placement:** every hazard is inside a room, clear of the centre keep-out and of other hazards, with its
   underside ≥ 1.95 m and its top ≤ wall height − 0.1 m.
4. **Geometry:** six outward faces per hazard; the bottom face's normal is −z.

## 5. Cost

- 3 renders at about 14 min of llvmpipe wall time each locally. On a T4 with EGL this is unmeasured: SMOKE first.
- Then 3 estimator runs. Everything runs on one spot g4dn.xlarge through `spinup.sh --profile gpu`, under its three
  brakes.

## Result (2026-10-23; `evaluation/runs/risk_calibration_4.2/`, protocol §3 as fixed before the data)

- **Data:** three ci_tiny_hazard seeds, rendered, run through the estimator chain and mapped on AWS g4dn (one
  spot run, ~$0.21 compute). The control is the evidence spool's hazard-free map.
- **Population per seed:** 11–12k cells (GT trav_responder ∩ map-observed free), 37–43 % of them in GT risk zones.

| score | seed 0 AUC | threshold (recall ≥ 0.9 on seed 0) | seeds 1–2 pooled: AUC / recall / FPR | control FPR |
|---|---|---|---|---|
| **proxy** (noisy-OR) | 0.841 | 1.0 (saturated) | **0.831** / 1.00 / 0.338 | 0.000 |
| overhang alone | 0.921 | 1.0 | 0.922 / 0.999 / 0.156 | 0.000 |
| unsupported alone | 0.841 | 1.0 | 0.831 / 1.00 / 0.338 | 0.000 |

- **Bar passed:** evaluation AUC 0.831 ≥ 0.75. No term is dropped (both seed-0 AUCs > 0.6). **The proxy enters
  planning** (w_r = 1 in the egress cost; `risk_calibrated: true`), as pre-registered.
- **Reported with it, not hidden:**
  - **The proxy saturates.** The recall-0.9 threshold lands on the clamp value 1.0, so its operating point is
    effectively binary. At it, 34 % of in-population cells outside the risk zones also read 1.0 (FPR). The
    N_ref = 100 saturation is reached too easily.
  - **The overhang term alone is clearly better:** AUC 0.92, FPR 0.16.
    - Switching to it now would be a choice made from this data.
    - It's recorded as the next pre-registered candidate: overhang-only, threshold fixed on seed 0, its own fresh
      seeds. It is not applied here.
  - **The unsupported term carries the proxy's ranking:** proxy and unsupported AUCs are equal to 3 decimals.
- **What it means for routing:** with w_r = 1, about a third of the non-hazard cells a route may use get the
  maximal risk penalty. The hazard-seed egress rows (4.5) measure what that does to route length and validity.
