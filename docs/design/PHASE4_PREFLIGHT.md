# Phase 4 preflight — the record, and what it changes in 4.1 / 4.2 (A1)

Status: 2026-10-23; review in §5. Written after the maintainer asked whether the 4.1/4.2 designs had a real preflight. **They did not
have a complete one.** This file records what was done, what was missing, what reading the primary sources changes,
and what is still owed before 4.3–4.6. Nothing below is implemented until the advisor review in §5.

## 1. What the first pass was, honestly

| preflight part | done before 4.1/4.2 | gap |
|---|---|---|
| Literature | one batch of web searches for all of Phase 4. Primary text read only for FH 2012 (Algorithm 1 transcribed) | Voxblox, nvblox, FIESTA, FUEL, TARE, FAR, RACER, GBPlanner, STEP and the NBV information-gain papers known from abstracts or secondary summaries only. D* Lite's pseudocode did not extract from its PDF; CBBA's consensus table was not found |
| `ALGORITHMS.md` | grepped for Domain 3 (owed) | it has **no** volumetric / ESDF / traversability / frontier entry; arch §49's catalogue step was skipped |
| OpenCV-native | `cv::Volume` (SMOKE §4u), `cv::distanceTransform`, `filter2D` | `ptcloud/depth.hpp` (`RgbdNormals`, `findPlanes`) and `triangleRasterize*` were not looked at |
| Advisor | once, mid-implementation, when 4.2's egress rule broke | not before designing 4.1 or 4.2 |

So five design changes were made *after* measuring, each disclosed (D110, `VOLUMETRIC_4.1.md`, `TRAVERSABILITY_4.2.md`):

1. unknown = weight 0;
2. the silhouette guard;
3. the oracle tolerance of two voxels;
4. the hybrid egress rule;
5. the "open beneath" support test.

Three of them rediscover known TSDF artefacts, which the literature below already handles.

## 2. Primary sources read now (2026-10-23)

- **Voxblox** (Oleynikova et al., IROS 2017, [arXiv:1611.03631](https://arxiv.org/abs/1611.03631)), §IV–V:
  - **Weighting, eq. 5:** `w = 1/z²` in front of the surface (d > −ε), then a **linear drop-off behind it** to 0 at
    −δ, with δ = 4v and ε = v. This is after Bylow et al., who found a linear drop-off "often yielded the best
    results". The paper says it matters most "in the presence of thin surfaces that are observed from multiple
    viewpoints, as this reduces the influence of voxels that have actually not been directly observed (those behind
    the surface)". That is the artefact 4.1's silhouette guard patched.
  - **Merging:** grouped raycasting, one raycast per end voxel on the weighted mean of its points. Projection mapping
    is faster "but leads to strong aliasing effects for larger voxels".
  - **Projective-distance error:** `r_p(θ) = d sin θ − d`, an expected −0.30·d for one observation; with 3 or more
    observations it is bounded by 0.5δ at p = 0.95. The stated rule: *"If only a single voxel of the surface frontier
    is used, then it is safe to increase the safety distance by half of one voxel."* That is 4.1's measured grazing
    bias (2.2 % of free voxels contain a surface) and the cause of 4.2's 4 false claims (GT clearance 0.395 m vs
    0.40 m).
  - **ESDF:** quasi-Euclidean wavefront, with a fixed band of TSDF-derived distances near the surface.
- **nvblox** (Millane et al. 2023, [arXiv:2311.00626](https://arxiv.org/abs/2311.00626)), §III:
  - **Integration:** projection mapping (voxel → depth image, nearest sample). Weights from a sensor model, summed to
    a cap. No interpolation across foreground–background boundaries.
  - **ESDF:** *sites* are voxels within ε of the TSDF zero crossing, so only the surface layer. Full Euclidean
    distances by a parallel-banding sweep, block-incremental, with explicit invalidation when a site is freed. It
    runs every 4 frames: the ESDF may lag the sensor rate.
- **FIESTA** (Han et al., IROS 2019, [arXiv:1903.02144](https://arxiv.org/abs/1903.02144)), abstract and method:
  - occupancy-based (not TSDF);
  - incremental BFS with separate insert and delete queues;
  - "near-optimal" distances, not exact.
- **Felzenszwalb & Huttenlocher 2012:** exact 1D lower envelope (already transcribed; 4.1's ESDF).

## 3. What this changes (proposals; to be fixed by §5's review before any code)

### 4.1 (D110)
- **P1, behind-surface weight drop-off (Voxblox eq. 5):** the weight falls linearly from the surface to 0 at −δ,
  instead of the band at full weight.
  - The question is whether it *replaces* the silhouette guard or complements it. A floating voxel seen only from a
    silhouette ray keeps a small weight and stays occupied under the drop-off alone. nvblox's answer, no
    interpolation across depth edges, is the guard's.
  - Proposal: keep the guard, add the drop-off. Pre-registered check: the test room's floating and bleed counts
    must not rise, and the spool's occupied precision must not fall.
- **P2, half-voxel safety inflation (Voxblox's rule)** for every clearance consumer. The egress lattice requires
  `clearance ≥ ρ + v/2`. This is a published compensation for a measured bias, not a margin tuned on the 4 cells it
  happens to remove. Its cost is recall; to be reported.
- **P3, weight model:** keep the arch's `1/σ_Z²`, i.e. `1/Z⁴` for stereo, where σ_Z ∝ Z². Voxblox's `1/z²` comes
  from an RGB-D model, so the difference is a sensor fact, not a choice. Stated in D110.
- **P4, ESDF:** keep the exact FH recompute per dirty box. It is exact, bit-equal to a rebuild, simple, and measured
  at 44 ms mean. It is a deliberate deviation from the wavefront/banding family (Voxblox, FIESTA, nvblox), which
  trades exactness for speed. nvblox's rate decimation is the documented fallback if the per-keyframe cost matters
  live. Also stated: "sites" = everything with SDF < v/2 gives the same distances outside surfaces as nvblox's
  surface-only sites.
- **P5, integration:** keep raycasting with per-frame, per-voxel averaging, closest to Voxblox's merged raycasting.
  Projection mapping (nvblox) is the noted alternative, not taken: it aliases at our voxel/pixel ratio, per Voxblox.

### 4.2 (D111, not committed yet)
- **T1:** the hybrid egress rule (observed column ∧ occupied-field capsule), plus P2's inflation.
- **T2, the overhang term, OpenCV-native alternative:** `cv::RgbdNormals` on each keyframe's depth, then
  `cv::findPlanes`. Planes whose world normal points down (n_z < −cos 45°) inside the 1.8–2.4 m band are accumulated
  per column. A fitted plane is far less noisy than central differences of the fused SDF. The candidates are
  compared on the calibration scenes, chosen by AUC on seed 0, and reported on seeds 1–2.
- **T3, the unsupported term:** on the hazard-free spool it still reads up to 0.23 (1061 cells above 0.2) after the
  support fix. So the proxy is not usable uncalibrated. Its first evidence must be the GT-zone calibration, and the
  term may be dropped if it has no discriminative power there. That is reported either way.
- **T4, test geometry:** render worldgen's own meshes to depth with `cv::triangleRasterizeDepth`, giving GT-geometry
  depth without Gazebo, instead of the tests' hand-written sphere tracer.

## 4. Still owed before 4.3–4.6

- **`ALGORITHMS.md`:**
  - a volumetric-mapping catalogue entry (the sources above), added before D110/D111 are final;
  - the **Domain 3 survey** (owed since D45): NBV and information gain (Isler 2016, Delmerico 2018, the SEE family),
    exploration (FUEL, TARE, GBPlanner, RACER), allocation (CBBA, its successors, grouping auctions) and routing
    (FAR, D* Lite), from primary text.
- **Primary pseudocode:** D* Lite (Koenig & Likhachev 2002, Fig. 3/4) from a readable copy, and CBBA's consensus
  table (Choi, Brunet & How 2009, Table I).
- **An advisor review** of each item's design before its code.

## 5. Review (advisor, 2026-10-23) and what it found

**A clearance-convention bug, not quantisation.** The 4 false egress claims all sat at GT clearance 0.395 m. That is
the point-to-square distance at offsets (8, 3) and (7, 5): `√(7.5² + 2.5²)·0.05 = √(6.5² + 4.5²)·0.05 = 0.3953`.
4.1's formula `√edt·v − v/2` gives 0.402 and 0.405 there, so both cells were claimed.

The GT and 1.8 measure to the nearest point of the blocked square: per axis `max(|Δ| − ½, 0)`. The wrong formula was
in five places:
- `Esdf::Clear`;
- the hybrid's 2D part;
- both brute-force tests, which copied it, so they validated the bug against itself;
- the scorer's "1.8" row;
- the voxel-size script.

**Fixed:**
- **The 3D ESDF:** exact windowed min-convolutions in the cube metric.
- **The hybrid's 2D part:** calls 1.8's `EdgeClearance` directly.
- **The scorer:** imports worldgen's `_edt_to`.
- **The brute forces:** now clamp the query point into each cube, independently of the implementation.
- **The voxel-size check:** re-run as `python/analysis/voxel_size_check.py` with `_edt_to` and the GT's `free`
  mask. It reproduces GT `trav_responder` exactly, and **the 5 cm conclusion stands** (8 and 10 cm with a one-cell
  bleed still close agent 1's entrance in both scenes).

**Result after the fix**, evidence spool, at ρ_responder:

| rule | precision | recall | false claims | starts → target |
|---|---|---|---|---|
| hybrid (the product) | 1.000 | 0.900 | 0 | both |
| strict 3D | 1.000 | 0.354 | 0 | neither |
| 1.8 (its true rule) | 1.000 | 0.898 | 0 | both |

**Decisions from the review:**
- **P2** (half-voxel inflation) is deferred and decided on principle, separately. It addresses projective error, a
  different source; applying it first would have hidden the convention bug.
- **P1** (drop-off) stands, with its pre-registered criteria, as secondary.
- **T2/T3** wait for calibration scenes, which need rendered hazard worlds (GPU time: the maintainer is asked first).
  They are sequenced after 4.5.
- **D* Lite / CBBA primary text:** render the PDF page (`pdftoppm`) and read the figure. Search the CBBA DOI
  (10.1109/TRO.2009.2022423) and the papers reproducing its Table I; no guessed URLs.
