# 4.2 Traversability and structural risk at ρ_agent and ρ_responder (A1)

Status: 2026-10-23. Decision: D111. **Changed after measuring; each change is in §6, and the preflight gaps are in
`docs/design/next-best-view.md`.** Traversability is done in replay; the risk proxy is built but **uncalibrated**. Arch §20.5, §22.1 (the lattice), §44 (honesty
list). Consumes 4.1 (D110: TSDF, egress and agent ESDFs). Consumed by 4.3 (C_risk), 4.5 (the egress lattice and its
Risk term).

## 1. Preflight

**Literature (2026-10-23):**
- **STEP** (Fan et al., JPL/Caltech, DARPA SubT, [arXiv:2303.01614](https://arxiv.org/abs/2303.01614)): terrain
  hazards scored with CVaR into a risk map for planning. That justifies a *risk map consumed as a cost*, not a
  binary gate. Ariadne's terrain is flat floors, so STEP's slope and step terms do not apply; its framing does.
- **GBPlanner2** (Kulkarni et al. 2022, CERBERUS): each graph edge checks that traversable ground lies beneath it.
  The capsule test below is the same idea for a standing body. Observed floor is part of "observed free".
- No published proxy for *structural collapse risk* from a TSDF was found (search 2026-10-23). The closest, MIT LL's
  RubbleSim ([arXiv:2510.20529](https://arxiv.org/abs/2510.20529)), notes that collapse voids lack evaluation
  data. So the proxy here is a heuristic, labelled `[PROXY]` everywhere it appears (§20.5, §44), and calibrated only
  against our own GT-labelled zones.

**OpenCV-native:** nothing applies. The clearance comes from 4.1's ESDF, and the normals are central differences of
the fused SDF. `cv::Volume::fetchNormals` exists, but it reads OpenCV's volume, not ours (SMOKE §4u).

## 2. Traversability

**As built (the egress product, at ρ_responder):** the intersection of two tests (§6 explains why).

1. **Observed:** D45's accepted column rule on 4.1's TSDF slab. Some voxel in the body band [0.15, 1.90] m is
   observed free and none is occupied, and 1.8's own `EdgeClearance` to the non-free columns is ≥ ρ.
2. **Capsule:** the vertical capsule below fits the **occupied-only** 3D field, so a low overhang still closes the
   column.

The capsule on the egress field (unknown = blocked) is the **strict 3D** rule, reported and not the product.

### 2.1 The capsule (first design, kept as test 2 and as the strict row)

A body is a vertical capsule of radius ρ, with its axis from `z_floor + ρ + voxel` to `h_body − ρ`. Column (x, y)
is traversable at ρ if every voxel centre on that axis has ESDF ≥ ρ:

```text
trav_ρ(x, y)  <=>  for all z in axis(ρ):  ESDF(x, y, z) >= ρ          (§20.5's ESDF(x) >= ρ, for a body)
```

- **Egress** (ρ_responder = 0.40 m, h = 1.80 m: a standing adult) uses the **egress** field (unknown = blocked), so
  the route's clearance, floor included, must be observed (invariant 4, D45).
- **Agent** (ρ_agent = 0.20 m) uses the **agent** field (occupied only). The agents are kinematic movers that fly
  at 1.0 ± 0.2 m (worldgen `motion.altitude_m` ± jitter), so their capsule's axis is that altitude band, 0.8–1.2 m,
  not a ground capsule.
- **The axis starts one voxel above `z_floor + ρ`:** the floor's own voxels are occupied (their centres sit half a
  voxel up), and the edge convention puts the floor's edge one voxel above z = 0.
- **Why not §22.1's 26-connected 3D lattice:** a responder walks on a floor, and every worldgen floor is flat at
  z = 0. A 3D lattice would route through the air above a prop. The 2.5D lattice with a 3D capsule test keeps what
  3D buys, a low overhang cuts clearance, without the flying routes. Ramps and stairs are not modelled.
- **Output:** two 2D grids on the briefing's cells (the GT layer's layout), `trav_responder` and `trav_agent`, plus
  the per-column egress clearance (min over the axis), which 4.5's cost term reads.

## 3. Structural risk `[PROXY]`

Two terms per column, each in [0, 1], over a horizontal disc of radius `r_risk = 1.0 m` and the **risk band**
1.8–2.4 m: above a standing responder (h = 1.8 m), and 0.2 m under the lowest ceiling (worldgen ceilings sit at the
wall height, 2.6–3.0 m), so a ceiling at its design height never counts. Both terms saturate at
`N_ref = 100` voxels, the face of a 0.5 m × 0.5 m slab at 5 cm. Raw fractions of the nearby occupied voxels would be
diluted by any wall in the disc.

- **Overhang:** `min(1, n_down / N_ref)`, where `n_down` counts the occupied voxels in the disc and band whose SDF
  normal points down (`n_z < −cos 45°`): an underside facing the floor. A cantilever or a sagging slab exposes
  undersides. The normal is the central difference of the fused SDF, defined only where all 6 neighbours are known.
- **Unsupported mass:** `min(1, n_hang / N_ref)`, where `n_hang` counts the occupied voxels in the disc and band
  whose column is open beneath the band: no occupied voxel from one voxel above the floor up to 1.8 m. That is mass
  hanging over walking space. A wall column is occupied all the way down, so it counts 0.
  *Fixed after the first spool run, before any calibration data existed (§6):* "open beneath" now also requires no
  occupied voxel within ±2 voxels of the column below the band, since a thin wall's voxels jitter by one between
  heights. It also requires the column itself to be observed free somewhere below the band: unobserved is not
  open.
- Disc sums are `cv::filter2D` with a disc kernel over the per-column counts (OpenCV-native).
- **risk = 1 − (1 − overhang)(1 − unsupported)**, a noisy-OR of the two. It is 0 in an empty room under a ceiling
  above the band.

**Calibration (§20.5, register "risk proxy calibrated against GT-labelled zones"):** worldgen gains hazard geometry
(cantilevered slabs, hung masses) and a GT layer `risk_zone` (each hazard's footprint dilated by `r_risk`). The
proxy is then scored per column against `risk_zone`: ROC AUC and the threshold at a fixed 0.9 recall, on an
evaluation seed set that does not include the calibration seed. The threshold is fixed on seed 0 and reported on
seeds 1–2. Until that lands, the proxy is reported as **uncalibrated**.

## 4. Tests

1. **Capsule vs brute force:** on a synthetic TSDF (a room with a doorway of known width and a low beam),
   `trav_ρ` equals a brute-force capsule fit (every voxel within ρ of the axis is free), at both ρ.
2. **Doorway by hand:** a 1.05 m doorway is traversable at ρ = 0.40 m and closed at 0.55 m; a beam at 1.5 m closes
   the responder (h = 1.8) and not the agent (h = 1.3).
3. **Risk:** 0 in the bare room; > 0.5 under the beam's underside; monotone in the beam's overhang length.
4. **Egress ≤ agent:** egress traversability implies agent traversability (unknown ⊂ blocked); checked on the
   spool's final map.

## 5. Evaluation on the evidence spool (reported)

`trav_responder` from the final map against GT `trav_responder` (2D, from the scene's structure): precision of
traversable cells (claimed-traversable that are truly traversable). This is the number invariant 4 protects. Also
recall, side by side with 1.8's.

## 6. Results and the changes made after measuring (2026-10-23)

**Change 1: the strict 3D capsule cannot be the product.** On the evidence spool's final map the strict rule (the
capsule on the egress field) has recall 0.374 and closes the target cell for both starts. 91 % of its misses (7066 of
7724) are closed by **unobserved** voxels, not occupied ones. The unknown fraction over GT-traversable columns is
about 2 % at 0–0.8 m, scattered pinholes that a 0.4 m sphere of about 2000 voxels cannot avoid, rising to
11–39 % above 1.3 m. The rig flies at 1.0 m with a 28° vertical half-FOV, so a standing body's volume is not
observable from it. So the egress product is the hybrid of §2, on the advisor's review. It keeps D45's accepted
definition of "observed" and adds the 3D obstacle test. Strict 3D is kept and reported as a finding.

**Change 2: the clearance convention** (see `docs/design/next-best-view.md` §5). The first hybrid made 4 false claims (GT
clearance 0.395 m), caused by 4.1's optimistic `√edt·v − v/2`. It was fixed in the ESDF, in the 2D part (now 1.8's
`EdgeClearance`) and in independent brute-force tests.

**Change 3: the support test** of the unsupported term. On the hazard-free spool it fired over 6415 cells (max
0.62). A thin wall's voxels jitter by one between heights, and unobserved space read as "open". Fixed as in §3: max
0.23, 1061 cells above 0.2. **Still not zero, so the proxy is uncalibrated and not used.** This change was made
before any calibration data existed.

**Tests:** 5, all passing.
1. Capsule, strict and hybrid against independent brute force (nearest point of cube or square, by clamping).
2. Doorway and beam by hand.
3. Risk: zero in a bare room; responds to a cantilever, monotone in its length.
4. Unobserved columns close egress but not the agent.
5. A horizontal camera at 1.0 m (48 noise-free views): strict 1452 columns, hybrid 1923, fully observed 2144. The
   hybrid makes 0 false claims and connects the two rooms through the doorway.

**Evidence spool** (`evaluation/runs/mapping/traversability/`; GT `trav_responder`, ρ = 0.40 m):

| rule | claimed cells | precision | recall | false claims | start 0 / start 1 → target |
|---|---|---|---|---|---|
| **hybrid (the product)** | 11100 | **1.000** | **0.900** | 0 | yes / yes |
| strict 3D (capsule on the egress field) | 4367 | 1.000 | 0.354 | 0 | no / no |
| 1.8 placeholder (its own rule) | 11072 | 1.000 | 0.898 | 0 | yes / yes |

**Open:**
- **Risk calibration** needs worldgen hazard geometry with GT `risk_zone` labels, plus rendered recordings (GPU
  time: to be asked). T2 (the plane-based overhang term via `cv::RgbdNormals` / `cv::findPlanes`) is decided there.
- **P1** (behind-surface drop-off) and **P2** (half-voxel inflation, Voxblox's rule) are decided separately
  (`docs/design/next-best-view.md` §3, §5).

---

<a id="risk-zones"></a>

# Risk zones

Status: design fixed before any hazard scene exists (2026-10-23). Decision: D112. Calibrates D111's structural-risk
`[PROXY]` (`traversability-risk.md` §3, arch §20.5: "calibrated against the GT-labeled zones in the simulated
scenes"). Renders on AWS g4dn, as the maintainer approved on 2026-10-23.

### 1. Preflight

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

**No published TSDF-based collapse-risk proxy was found** (`traversability-risk.md` §1). The calibration is ours,
against our labels.

**OpenCV-native:** the scenes are worldgen's (meshes and SDF). Nothing here is image analysis.

### 2. Worldgen changes (opt-in; existing scenes byte-identical)

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

### 3. Calibration protocol (fixed now)

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

### 4. Tests

1. **Golden:** every existing preset and seed gives the same scene-directory digest (`test_golden.py`, unchanged),
   and the same `Preset.digest()` as recorded before the change.
2. **Hazards change no GT traversability:** `ci_tiny_hazard` seed 0's `free` / `trav_*` equal those of the same
   layout without hazards.
3. **Placement:** every hazard is inside a room, clear of the centre keep-out and of other hazards, with its
   underside ≥ 1.95 m and its top ≤ wall height − 0.1 m.
4. **Geometry:** six outward faces per hazard; the bottom face's normal is −z.

### 5. Cost

- 3 renders at about 14 min of llvmpipe wall time each locally. On a T4 with EGL this is unmeasured: SMOKE first.
- Then 3 estimator runs. Everything runs on one spot g4dn.xlarge through `spinup.sh --profile gpu`, under its three
  brakes.

### Result (2026-10-23; `evaluation/runs/mapping/risk_calibration/`, protocol §3 as fixed before the data)

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
