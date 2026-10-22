# 4.1 Volumetric stack — voxel-hash TSDF, tri-state occupancy, incremental ESDF, frontiers, coarse layer (A1)

Status: design fixed before measuring (2026-10-22). Decision: D110. Arch §20.1–§20.4, §20.6. Consumers: 4.2
(traversability), 4.3 (NBV candidates A, IG raycasts), 4.5 (egress lattice), 4.6 (coarse layer). The 1.8
placeholder (`cpp/planning/volumetric/occupancy_grid.hpp`, `clearance.hpp`, `frontier/frontiers.hpp`, D45) is not
modified: it stays the ablation baseline.

## 1. Preflight

**OpenCV-native (SMOKE §4u).** `cv::Volume(HashTSDF)` integrates and raycasts. It has no voxel or weight read and no
per-pixel weight, so tri-state, ESDF and frontiers cannot be built on it. That is §20.1's fallback case: a
from-scratch voxel-hash TSDF, with `cv::Volume` HashTSDF as the **oracle** (§5). `cv::distanceTransform` is 2D only,
so the 3D ESDF is from scratch too. It is checked against a brute-force oracle, and 2D slices of it against
`cv::distanceTransform(DIST_L2, DIST_MASK_PRECISE)`.

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
ρ_responder lattice and its start→target connectivity, edge-clearance convention):

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

`w_min = 0.5`: half of one σ_ref-quality observation. This is fixed here and not tuned.

### 3.3 ESDF (§20.3), incremental per updated block
- FH Algorithm 1 along x, then y, then z, on squared distances in voxel units. Sites are blocked voxels (f = 0);
  others are +∞. The paper's `(f(q)+q²) − (f(v)+v²)` is undefined for +∞, so +∞ samples are skipped when the
  envelope is built (an all-+∞ line stays +∞).
- **Clearance** is the distance from a voxel centre to the nearest blocked voxel's **edge**: `edt·voxel − voxel/2`.
  This is the GT's and D45's convention. It is capped at `d_max = ρ_responder + 2·voxel` (0.50 m), since nothing
  downstream reads more.
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
- **Not on the wire here:** `wire.fbs` and the byte table belong to 3.2 (main session). This provides the codec and
  its byte count, for message class 5 when 3.2 adds it.

## 4. Interface (replay now, live later)

`planning::MissionInput`: the per-keyframe record of §2, plus agent states (pose, energy used and total, link
up/down). `ariadne_mission --layer phase4` fills it from the spool exactly as 1.8 does (same anchor and merge
timing, D45). The L1 live runtime fills the same struct from its estimator. The layer never reads GT.
`--layer placeholder` (the default) is 1.8, unchanged.

## 5. Tests (in the commit)

1. **FH 1D:** against brute force on random sampled functions, including `+∞` runs.
2. **ESDF vs brute-force oracle (§37.6, exit gate):** 50 random 3D grids with obstacle densities from 1 % to 30 %.
   Exact in voxel units (squared distances are integers); one 2D slice also against `cv::distanceTransform`
   (≤ 1e-4).
3. **Incremental ESDF:** after 30 random block edits, equal bit for bit to a global recompute.
4. **TSDF on a known scene:** a box room rendered to synthetic depth from several poses.
   - The occupied voxels lie within one voxel of the true walls.
   - Carved voxels are free.
   - Unobserved space stays unknown.
   - `cv::Volume` HashTSDF on the same frames (unit weights, same voxel size and τ) gives a raycast surface within
     one voxel of ours, the tolerance SMOKE §4u justifies.
5. **Frontiers vs the GT unknown boundary:** on a synthetic map with known free and unknown, they equal brute
   force, and incremental equals full re-detection.
6. **Determinism:** two integrations of the same frames, in different hash insertion orders, are byte-identical.

## 6. Evaluation on the 1.7 evidence spool (reported, no success bar)

- Map vs GT in the z band, projected to the GT's 0.05 m 2D grid as D45 did: coverage, free precision, occupied
  precision, side by side with 1.8's 0.93 / 0.97 / 0.71.
- Time per keyframe for integration and for the incremental ESDF and frontier update, and memory (blocks), on this
  host (indicative).
