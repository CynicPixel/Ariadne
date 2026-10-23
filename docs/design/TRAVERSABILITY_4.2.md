# 4.2 Traversability and structural risk at ρ_agent and ρ_responder (A1)

Status: 2026-10-23. Decision: D111. **Changed after measuring; each change is in §6, and the preflight gaps are in
`PHASE4_PREFLIGHT.md`.** Traversability is done in replay; the risk proxy is built but **uncalibrated**. Arch §20.5, §22.1 (the lattice), §44 (honesty
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

**Change 2: the clearance convention** (see `PHASE4_PREFLIGHT.md` §5). The first hybrid made 4 false claims (GT
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

**Evidence spool** (`evaluation/runs/traversability_4.2/`; GT `trav_responder`, ρ = 0.40 m):

| rule | claimed cells | precision | recall | false claims | start 0 / start 1 → target |
|---|---|---|---|---|---|
| **hybrid (the product)** | 11100 | **1.000** | **0.900** | 0 | yes / yes |
| strict 3D (capsule on the egress field) | 4367 | 1.000 | 0.354 | 0 | no / no |
| 1.8 placeholder (its own rule) | 11072 | 1.000 | 0.898 | 0 | yes / yes |

**Open:**
- **Risk calibration** needs worldgen hazard geometry with GT `risk_zone` labels, plus rendered recordings (GPU
  time: to be asked). T2 (the plane-based overhang term via `cv::RgbdNormals` / `cv::findPlanes`) is decided there.
- **P1** (behind-surface drop-off) and **P2** (half-voxel inflation, Voxblox's rule) are decided separately
  (`PHASE4_PREFLIGHT.md` §3, §5).
