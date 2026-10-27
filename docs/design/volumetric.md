# Volumetric mapping: voxel-hash TSDF, tri-state occupancy, incremental ESDF, frontiers

Status: design fixed before measuring (2026-10-22). Decision: D110. Arch §20.1–§20.4, §20.6. Consumers: 4.2
(traversability), 4.3 (NBV candidates A, IG raycasts), 4.5 (egress lattice), 4.6 (coarse layer). The 1.8
placeholder (`cpp/planning/volumetric/occupancy_grid.hpp`, `clearance.hpp`, `frontier/frontiers.hpp`, D45) is not
modified: it stays the ablation baseline.

## 1. Preflight

**OpenCV-native (SMOKE §4u).** `cv::Volume(HashTSDF)` integrates and raycasts. It has no voxel or weight read and no
per-pixel weight, so tri-state, ESDF and frontiers cannot be built on it. That is §20.1's fallback case: a
from-scratch voxel-hash TSDF, with `cv::Volume` HashTSDF as the **oracle** (§5). `cv::distanceTransform` is 2D only
and measures centre to centre, so the 3D ESDF is from scratch too. It is checked against a brute-force
nearest-point-of-cube oracle, and 2D slices of it against 1.8's `planning::EdgeClearance`.

> **This preflight was incomplete; see `docs/design/next-best-view.md`.** Primary sources were read after the first commit.
> They changed nothing in the TSDF itself, but the ESDF's clearance convention was wrong as first committed (§3.3).

**Literature (2026-10-22).**
- **Voxblox** (Oleynikova et al., IROS 2017, [arXiv:1611.03631](https://arxiv.org/abs/1611.03631)): TSDF in a
  block hash, with an ESDF derived incrementally from it. It reports TSDF integration faster than OctoMap and ESDF
  more accurate from a TSDF than from occupancy. Its quasi-Euclidean wavefront is what FIESTA and nvblox later
  replaced with full-Euclidean updates. The block hash and the weighted fusion are taken from Voxblox; the ESDF is
  not.
- **Felzenszwalb & Huttenlocher**, *Distance Transforms of Sampled Functions* (Theory of Computing 8:415–428, 2012,
  [PDF](https://theoryofcomputing.org/articles/v008a019/v008a019.pdf)), Algorithm 1: the exact 1D squared-Euclidean
  lower-envelope transform, applied once per axis. Transcribed from the paper (§3.3).
- **FUEL** (Zhou et al., RA-L 2021, [arXiv:2010.11561](https://arxiv.org/abs/2010.11561)): the frontier structure is
  updated only where space was newly observed. §3.4 takes that incremental rule.
- **OctoMap / 1.8 (D45):** the ray-walk (Amanatides & Woo 1987) and the σ depth gates are reused unchanged.

**Measured before choosing the voxel size** (GT of `ci_tiny_0` and `maze_medium_0`, rasterised conservatively; the
ρ_responder lattice and its start→target connectivity; **re-run 2026-10-23 with worldgen's own `_edt_to` and the
GT's `free` mask** (props included), which reproduces GT `trav_responder` exactly at 5 cm. The first run used a wrong
clearance formula and walls only; the table did not change):

| voxel | no bleed: start0 / start1 | one-cell wall bleed (D45's measured artefact): start0 / start1 |
|---|---|---|
| 0.05 m | ok / ok (both scenes) | ok / ok (both scenes) |
| 0.08 m | ok / ok | ok / **lost** (both scenes) |
| 0.10 m | ok / ok | ok / **lost** (both scenes) |

At arch §20.1's 8–10 cm, the one-cell bleed that D45 measured (93 % of falsely occupied cells sit one cell from a
wall) closes agent 1's 1.05 m entrance at ρ_responder = 0.40 m in both scenes.
**So the voxel is 0.05 m (D110), superseding the arch's 8–10 cm.** That is the resolution D45's egress ran at and
the GT grid's. The cost is memory and time, measured in §6.

## 2. Data in → data out

```text
in  (per keyframe, from MissionInput; replay = the 1.7 spool, live = L1 later):
      agent, t_ns, T_W_C (world <- camera), K, depth Z (m), sigma_Z (m)
out (queries):
      TSDF(x), W(x)         -> State(x) in {unknown, free, occupied}
      Clearance_egress(x)   (unknown counts as blocked)  |  Clearance_agent(x)  (occupied only)
      FrontierClusters      {centroid, normal, voxels, visible-unknown estimate}
      CoarseLayer 0.5 m     {known, occupied fraction, mean weight}  + RLE bytes (class-5 summary)
```

## 3. Components

### 3.1 Voxel-hash TSDF (`cpp/planning/volumetric/tsdf_map.{hpp,cpp}`)
- **Blocks:** 8³ voxels (0.4 m at 0.05 m), in an `unordered_map` keyed by the packed block index. Each voxel holds a
  `float` SDF (m) and a `float` weight. A voxel that was never touched has weight 0, which means unknown.
- **Integration**, per depth sample on a `pixel_stride` grid:
  - Gate exactly as 1.8 does (D45): finite, `0 < Z ≤ max_depth`, `σ ≤ max_sigma`.
  - Walk the ray from the camera centre to `range + τ` (τ = truncation = 3 voxels) with Amanatides–Woo.
  - **Silhouette guard** (added after test 4 measured it): a ray whose pixel has a 4-neighbour with depth differing
    by more than τ, or missing, grazes a depth discontinuity. It integrates only its in-front part (s ≥ 0).
    Otherwise its behind-surface band lands in the free space beyond the edge, the artefact that narrows a doorway
    seen obliquely. In the test room it cut floating occupied voxels from 112 to 0 (34 one-voxel bleed remain).
  - For each voxel, take the signed distance **along the ray**, `s = range − t_vox`, where `t_vox` is the voxel
    centre's projection on the ray. Clamp to `[−τ, τ]`.
  - **Carving:** voxels well in front of the surface get `s = τ`. That makes "observed free" explicit.
- **One observation per voxel per frame.** Rays near the camera cross the same voxel many times. Within one
  keyframe, the samples per voxel are averaged (SDF and weight), then fused once:
  `D ← (W·D + w·s)/(W + w)`, `W ← min(W + w, W_max)`. Without this, near-camera voxels would be over-weighted by
  ray density, not by evidence.
- **Weights:** `w = (σ_ref / σ_Z)²`, where `σ_ref = 0.01 m` makes the weights O(1). This is §20.1's `1/σ_Z²` up to a
  constant, which cancels in the fused mean.
- **Order:** the frame's voxels are merged in sorted key order, so a run is bit-identical however the hash iterates.

### 3.2 Tri-state (§20.2)
- `unknown` if `W < w_min`;
- `occupied` if `W ≥ w_min` and `D < 0.5·voxel` (the surface passes within half a voxel of the centre, or the voxel
  is behind it);
- `free` otherwise.

Unknown is `W = 0` exactly (arch §20.2). *Correction after the first spool run (§7):* the first version used
`w_min = 0.5`, half of one σ_ref-quality observation. With weights `(σ_ref/σ)²`, a σ = 0.06 m sample weighs 0.028, so
far voxels needed about 18 observations to count as known, and coverage fell to 0.861. That contradicted the arch's
definition; it was not a tuned threshold, and both runs are reported.

### 3.3 ESDF (§20.3), incremental per updated block
- **Clearance** is the distance from a voxel centre to the **nearest point of a blocked voxel's cube**: per axis an
  offset of Δ voxels contributes `max(|Δ| − ½, 0)`, and d² is the sum of the squares. This is the GT's and D45's
  convention (worldgen `raster._edt_to`, `planning::EdgeClearance`). It is capped at `d_max = ρ_responder + 2·voxel`
  (0.50 m), since nothing downstream reads more.
- **Correction (2026-10-23).** As first committed (`f3eb6bb`), clearance was `edt·voxel − voxel/2` on a
  centre-to-centre FH transform, and this design called that the GT's convention. It is only along an axis; on
  diagonals it is optimistic. It produced 4 false egress claims at GT clearance 0.395 m. The brute-force test had
  copied the same formula, so it passed. Found by the design review, `docs/design/next-best-view.md` §5.
- **Transform:** FH's lower envelope is exact for centre-to-centre distance only. The cube metric is `(2|Δ| − 1)²/4`
  per axis, still separable. Each axis pass (x, then y, then z) is an exact **windowed min-convolution** over the
  capped reach `k = ⌈d_max/v⌉ + 1 = 11`: a true distance ≤ d_max has every axis offset ≤ `d_max/v + ½`. The passes
  are stored as `4d²`, an integer, in uint16. A deviation from arch §20.3's "FH three-pass", for exactness in the
  GT's convention.
- **Two fields:** egress (blocked = occupied ∪ unknown: "observed, not merely not contradicted", D45) and agent
  (blocked = occupied).
- **Incremental:** after a keyframe, the dirty set is the blocks it touched. With a capped field, a change at a voxel
  affects only voxels within `d_max`. So the update recomputes the box `dirty ⊕ d_max`, reading sites from
  `dirty ⊕ 2·d_max`, and writes back only `dirty ⊕ d_max`. That is exact for the capped field, and a test checks it
  bit for bit against a global recompute.
- The ESDF lives in a dense array over the map's bounding box. The box only grows, and growing it reallocates and
  copies.

### 3.4 Frontiers (§20.4)
- Free voxels with a 6-neighbour that is unknown, inside the configured **motion band** `z ∈ [z_lo, z_hi]`. The
  agents are ground movers at a fixed body height; without the band, floor and ceiling unknowns would dominate. This
  is the 3D form of 1.8's z band.
- Clustered by 26-connectivity; clusters below `min_voxels` are dropped and counted.
- Per cluster:
  - **centroid**;
  - **normal**: the normalised sum, over its voxels, of the unit offsets to unknown 6-neighbours;
  - **visible-unknown estimate**: unknown voxels hit by a fixed 64-ray fan (60° half-angle cone about the normal,
    2.0 m), cast from the centroid and stopping at the first occupied voxel. This feeds Δcoverage in 4.3.
- **Incremental (FUEL's rule):** clusters touching the ESDF's update region are dropped and re-detected inside it.
  Clusters elsewhere are kept as they are. A test checks this against a full re-detection.

### 3.5 Coarse global layer (§20.6)
- 0.5 m cells, 10³ voxels each. Per cell: known fraction, occupied fraction, mean weight. A cell is `unknown` if its
  known fraction is below 0.5, `occupied` if any voxel in it is occupied, and `free` otherwise.
- RLE: (state, run) varint pairs in z-y-x order. The byte count is reported.
- **Not on the wire here:** `wire.fbs` and the byte table belong to 3.2. This provides the codec and
  its byte count, for message class 5 when 3.2 adds it.

## 4. Interface (replay now, live later)

`planning::MissionInput`: the per-keyframe record of §2, plus agent states (pose, energy used and total, link
up/down). `ariadne_mission --layer phase4` fills it from the spool exactly as 1.8 does (same anchor and merge
timing, D45). The L1 live runtime fills the same struct from its estimator. The layer never reads GT.
`--layer placeholder` (the default) is 1.8, unchanged.

## 5. Tests (in the commit)

1. **1D min-convolution:** against brute force (the distance to each sample's segment, by clamping) on random inputs,
   including empty runs.
2. **ESDF vs brute-force oracle (§37.6, exit gate):** 50 random 3D grids with obstacle densities from 1 % to 30 %.
   The oracle finds the nearest point of each cube by clamping, independently of the implementation's formula.
   Exact (4d² is an integer). One 2D slice is also checked against 1.8's `EdgeClearance` (≤ 1e-12).
3. **Incremental ESDF:** after 30 random block edits, equal bit for bit to a global recompute.
4. **TSDF on a known scene:** a box room rendered to synthetic depth from several poses.
   - The occupied voxels lie within one voxel of the true walls.
   - Carved voxels are free.
   - Unobserved space stays unknown.
   - `cv::Volume` HashTSDF on the same frames (unit weights, same voxel size and τ) gives a raycast surface within
     **two** voxels of ours on ≥ 90 % of rays. *Corrected from one voxel:* the two maps' errors add, and the oracle's
     own p95 error against truth is 1.35 voxels (ours 0.74).
5. **Frontiers vs the GT unknown boundary:** on a synthetic map with known free and unknown, they equal brute
   force, and incremental equals full re-detection.
6. **Determinism:** two integrations of the same frames, in different hash insertion orders, are byte-identical.

## 6. Evaluation on the 1.7 evidence spool (reported, no success bar)

- Map vs GT in the z band, projected to the GT's 0.05 m 2D grid as D45 did: coverage, free precision, occupied
  precision, side by side with 1.8's 0.93 / 0.97 / 0.71.
- Time per keyframe for integration and for the incremental ESDF and frontier update, and memory (blocks), on this
  host (indicative).

## 7. Results (2026-10-23)

**Tests:** 8 new in `tests/unit/test_volumetric.cpp`, all passing; 1.8's 6 planning tests unchanged.

| test | measured |
|---|---|
| 1D min-convolution vs brute force (500 random inputs with empty runs) | exact |
| ESDF vs brute-force nearest point of cube, 50 random grids, 1–30 % sites (§37.6) | exact |
| 2D slice vs 1.8's `EdgeClearance` | ≤ 1e-12 |
| Incremental ESDF vs rebuild (12 keyframes, ≥ 6 incremental) | bit-identical |
| Known room: occupied floating > 1.5 voxels in open space | 0 (34 one-voxel bleed, 0.06 %) |
| Known room: free voxels > half a voxel inside structure | 0 |
| Known room: free voxels that contain a surface (grazing) | 2.17 % (≤ 3 % asserted) |
| HashTSDF oracle, held-out raycast | agreement within 2 voxels on 93.1 % of rays; ours vs truth p50 0.014 / p95 0.037 m; OpenCV vs truth 0.028 / 0.067 m |
| Frontiers vs brute force; incremental vs full | identical voxel sets (5749), identical clusters and fans |
| Determinism | two runs bit-identical, same coarse RLE bytes |

**Known biases, measured:**
- *Optimistic clearance at grazing incidence.* Along-ray SDF ≥ perpendicular distance, so 2.2 % of free voxels contain
  a surface. Clearance can be overestimated by at most one voxel (5 cm) there. 4.5 carries it as a route-confidence
  input, not as a hidden margin.
- *Pessimistic one-voxel bleed* (D45's artefact, in its TSDF form): 0.06 % of occupied voxels. The 5 cm voxel was
  chosen to survive it (§1).

**On the 1.7 evidence spool** (`evaluation/runs/mapping/volumetric/`; same keyframes, anchor and merge as 1.8's merged
run; scored by `volumetric_eval.py`, which reproduces `mission_eval.py`'s 1.8 numbers exactly):

| map | coverage | free precision | occupied precision | false-occupied cells (1 / 2 / 3 cells from GT structure) |
|---|---|---|---|---|
| **Phase 4 TSDF (D110)** | **0.942** | **0.979** | **0.865** | 620 (588 / 24 / 8) |
| 1.8 placeholder (D45) | 0.933 | 0.961 | 0.707 | 793 (756 / 36 / 1) |
| *first run, `w_min = 0.5` (superseded)* | *0.861* | *0.995* | *0.855* | *527* |

- **Cost** (dev host, shared; 571 keyframes; corrected ESDF): integrate 22.6 ms mean (p95 54.7), incremental ESDF
  80.6 (144.0), frontier update 59.8 (96.7), about 163 ms per keyframe in all. The two agents produce about 2.9
  keyframes per second together, so the budget is about 340 ms. The first run, with the wrong FH metric, measured
  14.8 / 43.9 / 45.8 ms at lower host load: the windowed transform costs more, and the host was shared. ESDF: 565
  incremental updates, 6 rebuilds (box growth). Map: 3067 blocks, 12.6 MB.
- **The map metrics in the table above do not depend on the ESDF** (they read the TSDF slab), so the correction
  leaves them unchanged.
- **Coarse layer:** 4320 cells at 0.5 m, 1047 RLE bytes for the whole merged map.
- **Reading:** occupied precision rises from 0.71 to 0.87 because surfaces come from the SDF zero crossing, not from
  "the voxel a ray ended in". The false-occupied cells are still almost all one cell from GT structure: the thin-wall
  convention effect D45 measured.
