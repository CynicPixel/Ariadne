# Design — Phase 2.5: robust backend

Plan 2.5 (PHASE2_PLAN §1, phased plan Phase 2): replace the Phase-1 pose-graph placeholders with arch §13–§14 as
specified — chordal initialisation, degeneracy gate → PCM → GNC-TLS (the §14.5 order is inviolable), the SE-Sync-style
certificate behind a flag, and a real intra-agent loop-candidate source. Decisions D70–D74. Status: **design, as of
2026-10-18**; the "As built" notes are appended per section when each lands.

## 0. Preflight (done before this design; evidence in the session handover)

- **Literature, checked at the sources.** GNC (Yang et al., RA-L 2020, arXiv 1909.08605: Prop. 4, Remark 5, §VI-B);
  SE-Sync (Rosen et al., IJRR 2019, arXiv 1612.07386: eqs. 12–18, Thm 7, eq. 119, Table 1; reference code read for
  conventions only — LGPL-3, nothing transcribed); PCM (Mangelson et al., ICRA 2018); TACO (arXiv 2606.29851);
  EGNC-PGO (2310.06765); AGNC (2308.11444); group-k consistency (2308.02674); COSMO-Bench (2508.16731). None changes the
  §14 pipeline or its order; all are ablation candidates only.
- **OpenCV-native check (installed 5.0.0).** Present and used: `cv::SVDecomp`, `cv::buildMST`, `cv::eigen` (dense, test
  oracle only), `cv::detail::LevMarqBase`. Absent: sparse eigensolver / Lanczos, max-clique, robust PGO
  (`cv::detail::PoseGraph` rejected in D51).
- **Oracles (ariadne-oracle:1, unchanged).** GTSAM Python exposes `InitializePose3`, `GncLMOptimizer`, `GncLMParams`,
  `GncLossType`, `GncScheduler`. Oracles only, never runtime (CLAUDE.md §1).

## 1. Placement (arch §3 conflict — surfaced, D70 records the resolution)

Arch §3 puts `degeneracy/` under `cpp/perception/`, `pcm/` and `gnc/` under `cpp/collaboration/`. The 2.5 brief says
`cpp/slam/robust/`. Resolution proposed: **`cpp/slam/robust/`** (degeneracy, pcm, gnc, certificate) because every one of
them operates on pose-graph edges and must also run intra-agent (no broker involved); `cpp/slam/pose_graph/` gains
`chordal_init`; `cpp/slam/loop_closure/` holds the intra-agent candidate source. The arch §3 tree is updated in the same
change (§49: the arch doc must not disagree with the code).

## 2. Step 0 — LM damping diagnosis (open item (b), D70)

**What and why.** Every GNC outer iteration is a full LM solve, so an LM that needs 1000 iterations where GTSAM needs 8
multiplies into the whole robust pipeline. OpenCV's `LevMarqBase` damps with `λ·diag(H)` and, with `clampDiagonal`,
clamps each damping entry to an **absolute** `[1e-6, 1e32]` (levmarq.cpp). Garage's information spans 3.7e-10 in
eigenvalue ratio (POSE_GRAPH_2.md §5), so the hypothesis is that the absolute floor dominates the weak directions.

**Protocol (fixed before measuring).** `PgoOracle.DampingAblation` (tests/unit/test_pgo_oracle.cpp, env-gated): same
dead-reckoning start, same tolerances as `SeSyncBenchmarkGraphs`, one setting changed at a time — baseline / clamp off /
Jacobi on / both. Report iterations, linearisations, failed factorisations, final energy vs GTSAM, max|grad|. Fix = a
config change in `configs/slam/pgo_v1.yaml` (→ `pgo_v2`) only if it wins on garage **and does not regress** the other
SE-Sync graphs and the 100-case oracle. Honest `λI` damping is not available: the driver predicts the cost change from
its own `lmDiag`, so faking it in the backend would corrupt the step-quality test (recorded, not attempted).

**Findings (2026-10-18, garage from the file's vertices):**

| setting | test tolerances (1e-14) | pgo_v1 tolerances (1e-6) |
|---|---|---|
| baseline: clamp on, jacobi off | 1000 its (cap), at the minimum, max\|g\| 1.2e-9 never under 1e-9 | **95 its, stops 3.7e-5 above the minimum** |
| clamp off | 23 its, max\|g\| 2e-10 | **15 its, at the minimum** |
| jacobi on | 1000 its, 2.4e-4 above | — |
| clamp off + jacobi on | 18 its | — |

- The hypothesis "the floor binds at the start" was wrong (0 of 9960 entries). It binds **late**: λ shrinks by up to 3× per
  good step, and once `λ·diag < 1e-6` the floor turns the last Gauss–Newton steps into heavily damped ones (energy at
  50 its: clamp on 2.3e-4 relative above the minimum, clamp off already there at 20). Production is affected, not only
  the test.
- **Literature/reference check.** Ceres (`levenberg_marquardt_strategy.cc`) clamps the *column norms* `diag(JᵀJ)` to
  `[min_lm_diagonal = 1e-6, max_lm_diagonal = 1e32]` **before** dividing by the trust radius, so its damping still goes
  to zero as λ → 0. OpenCV (`levmarq.cpp:192–194`) clamps `λ·diag`, an absolute floor on the damping itself. OpenCV's
  placement is the deviation from the source it follows; `clampDiagonal = false` is the closer match to Ceres.
- **Not "different minima" — an oracle bug.** GTSAM's `Rot3.Quaternion` does not normalise. The SE-Sync g2o files print
  ~7 digits (garage: `| |q|−1 |` up to 6.8e-7), so `pgo_gtsam.py` gave GTSAM non-orthogonal "rotations"
  (det 1 + 2.7e-6) — a different problem from the one the C++ loader (which normalises) solves. Per edge the two costs
  agree to 1e-14 on identical poses; our cost at GTSAM's old optimum was 0.63419231 with max|g| 3.1e-2 (not
  stationary). With the quaternion normalised, GTSAM reaches 0.634192400 (7 its) vs ours 0.634192268: same minimum
  (2.1e-7 relative). `SeSyncBenchmarkGraphs` now asserts equal initial energies (the check whose absence hid this);
  the SE-Sync oracle is regenerated under `.cache/feat-2.5-robust-backend/oracle/pgo_sesync_v2/` (the shared
  `.cache/oracle/pgo_sesync` is left untouched; the maintainer re-runs `models.sh pgo-sesync` after merge).
- **Guard before adopting clamp-off on the GNC path.** The 1e-6 floor is currently the only regulariser if a block of H
  loses rank — e.g. GNC zeroes the only loop joining two odometry segments (D56 makes segmented graphs real). The GNC
  wrapper therefore anchors every connected component of the *active* graph explicitly, and a disconnection test is in
  the no-regression set. The OpenCV floor is a hard-coded constant, so it cannot be made relative instead.

**As built (2026-10-19).** There is no `configs/slam/pgo_v1.yaml`: the protocol above assumed one, but pgo_v1 has only
ever been the `PgoSettings` defaults, and nothing at runtime calls `SolvePoseGraph` yet. pgo_v2 is therefore the preset
`PgoV2()` (`pgo_levmarq.hpp`) and the default argument of `SolvePoseGraph`. The struct defaults stay pgo_v1, because the
windowed BA (`stereo_vo.cpp`) builds its settings from them: changing them would have silently changed odometry, which
2.5 does not own. The config file, the manifest fields and the `pgo_init` selection land with the first runtime caller
(the D73 loop source). Regressions under pgo_v2: the 100-case oracle 100/100 (worst relative energy 1.48e-12); the
SE-Sync graphs from the files' vertices reach the same minimum as GTSAM on all six (§9); ctest 300/300 (8 env-gated skips) with BA
back on pgo_v1. Open item (b) is reduced, not closed at parity: garage takes 18 iterations from the file's vertices at the
SE-Sync tolerances and 15 from dead reckoning at the production tolerances, against GTSAM's 7.

## 3. Chordal initialisation (arch §13.1, D70)

**What.** A global initial guess that does not accumulate drift. Each edge says `R_j ≈ R_i R̃_ij`; that is *linear* in
the 9 entries of every rotation. Stack all edges into one sparse least-squares problem (anchor rotation fixed), solve
once with three right-hand sides (the three columns), then project each 3×3 block back onto SO(3):
`R = U · diag(1, 1, det(U Vᵀ)) · Vᵀ` from `cv::SVDecomp` — the determinant fix is mandatory (without it a reflection
can come out). Then translations by a second linear least squares with rotations fixed, then the existing LM.

**Weights.** Rotation rows weighted by κ, translation rows by τ — the isotropic SE-Sync weights from the edge
information (§7 below), so the same conversion serves init and certificate.

**Baselines (ablation, §49).** Dead reckoning (odometry chain, the current behaviour), `cv::buildMST` spanning tree
(composition along the tree of highest-κ edges), chordal. Config `pgo_init: {odometry, mst, chordal}`.

**Tests.** Noise-free graphs recovered exactly (to 1e-12); a hand-built case where the unprojected block has det < 0
(the fix must fire); agreement with GTSAM `InitializePose3` on the 100-case oracle (rotation-only stage, then after LM
the same minimum).

**As built (2026-10-18).** `cpp/slam/pose_graph/chordal_init.{hpp,cpp}`: `InitialPoses(graph, anchor,
{kCurrent, kMst, kChordal})`, `IsotropicFromInformation` (κ/τ, shared with D74), `ProjectToSO3` (`cv::SVDecomp`, det
fix), MST via `cv::buildMST` (Prim, weight −κ). Tests (`tests/unit/test_chordal_init.cpp`, all pass): projection is the
nearest rotation and the determinant fix fires on `R·diag(2, 1.5, −0.5)` (returns R, not the reflection); κ/τ exact on a
hand case; noise-free random graphs (40 nodes, large rotations, full anisotropic information, anchor not node 0)
recovered to 1e-9 by both chordal and MST; a node not connected to the anchor is refused. Oracle
(`PgoInit.ChordalRotationsAgreeWithGtsam`): 50 isotropic cases agree with GTSAM's `computeOrientationsChordal` to
**2.6e-10 rad**; the 50 correlated cases differ (up to 0.32 rad) because GTSAM weights an edge by one axis's √precision
— reported, not gated.

**GTSAM's chordal has a soft-anchor bias (found while building the oracle).** Its linear orientation graph pins the
anchor with a hard-coded unit precision while edges weigh `whiten(e₁)[0]` (≈15 on these graphs), so the relaxed
solution drifts by a non-rotational right factor that the projection does not undo: 6.5e-3 rad on random_016 against
the hard-anchored relaxation (ours; a dense numpy solve agrees with ours). A numpy replica of GTSAM's exact system
reproduces GTSAM to 6e-11 rad and moves onto ours once the anchor weight is raised (1e4, 1e8). The oracle therefore
scales every edge's information by 2⁻⁵⁴ (exact in floating point; relative edge weights unchanged) so that GTSAM's unit
anchor is effectively hard. This is an oracle setting, not a change to our method; GTSAM's `InitializePose3` as a
*reference start* (§9) keeps its default behaviour.

**Open item (a) — diagnosed 2026-10-18** (probe over the 100 cached cases, GTSAM 4.3 in ariadne-oracle:1, source of
`InitializePose3.cpp` / `InitializePose.h` read):
- **Without a `PriorFactorPose3`, it throws on every case.** `buildPoseGraph` turns only `PriorFactor`s into links to
  its internal anchor key; the oracle fixes the gauge with `NonlinearEqualityPose3`, which is dropped, leaving the
  anchor disconnected. The original call was never committed, so "case 2" itself is not reproduced [UNVERIFIED], but
  this mechanism throws on the first case it meets.
- **With a unit prior, 3 / 100 throw** (random_073, 079, 081, all full-correlated-information cases). The chordal
  rotation stage succeeds on all 100; the throw is in the single Gauss–Newton translation step (`computePoses`),
  multifrontal Cholesky only: on the identical linearisation `MULTIFRONTAL_QR` and `SEQUENTIAL_CHOLESKY` succeed, and
  H is positive definite (λ_min ≥ 3e-3). Conditioning does not explain it (random_071 passes at cond 9.2e10, the
  failures are ~4e10): a numerical false positive of that elimination, not an indeterminate problem.
- **Consequence for the oracle test.** GTSAM weights each rotation row by `whiten(e₁)[0]` (√ of one axis's precision,
  not κ), so agreement with our κ-weighted chordal is tested on the isotropic (even) cases, at the
  `computeOrientationsChordal` stage, with a unit `PriorFactorPose3`; the three multifrontal throws are excluded by
  name with this reason.

## 4. Degeneracy gate (arch §14.4, D71) — runs BEFORE PCM

**What.** A loop closure whose geometry cannot constrain all six degrees of freedom produces a confident wrong edge.
Per verified loop (PnP inliers from `edge_verifier`), compute on the PnP Gauss–Newton Hessian at σ = 0.46 px
[ASSUMPTION: the 2.3 reprojection noise]:

- rotation block and translation block conditioning, tested **separately** (metres and radians never mix in one
  eigenvalue ratio);
- parallax distribution of the inliers;
- 2D inlier spread (near-collinear test: smallest/largest eigenvalue of the image-point covariance);
- 3D planarity and a corridor score (dominant-direction ratio of the 3D inliers).

**Information.** The same Hessian, mapped into the edge's residual tangent, replaces the `[PROXY]` diagonal on loop
edges. Monte-Carlo test of the conversion; tagged [ASSUMPTION] because stereo depth noise of the 3D points is ignored.

**Tests.** One synthetic scene per criterion that trips **only** that criterion, plus a well-conditioned control that
passes all of them.

**As built (2026-10-19, D71).** `cpp/slam/robust/degeneracy.{hpp,cpp}`, `AssessLoop(LoopGeometry)`. Corrections to
the plan above:
- **Conditioning uses marginals, not raw blocks.** The raw rotation and translation blocks of H drop the coupling
  where visual degeneracy lives. The gate bounds the worst-direction 1σ of the translation and rotation **marginals**
  of the edge covariance in the pose-graph residual tangent, the matrix the PGO consumes. The bounds are absolute
  (Zhang 2016's precedent), and the within-block ratios are reported (§49 deviation from arch §14.4, recorded there).
- **Pose and information.** The pose is refined by `cv::solvePnPRefineLM` followed by Gauss–Newton weighted with
  per-point pixel covariances that carry B's stereo noise in (u, v, disparity) (Vakhitov 2021). This replaces the
  [ASSUMPTION] above that depth noise is ignored. The information uses `OdometryEdge`'s conversion exactly (tested).
- **Covariance inflation.** The covariance is inflated by max(1, σ̂²).
- **Parallax** is the count of inliers with a stereo triangulation angle ≥ θ_min. **Collinearity** is
  λ_mid/λ_max of the 3D inliers' covariance (a 3D line). **Pixel spread, planarity and corridor** are reported only.
- **"Only that criterion" is dropped.** The criteria are physically coupled, so each scene must trip its own
  criterion and have it named. A scene may trip more than one.

Tests (`tests/unit/test_degeneracy.cpp`, 8, all pass): a finite-difference check of the stereo back-projection
Jacobian, the information conversion, Monte Carlo NEES, the control and four scenes.

| scene | σ_t worst | σ_r worst | near points | 3D line spread | pixel spread | verdict |
|---|---|---|---|---|---|---|
| control (120 pts, full image, 1.5–6 m) | 3.9 mm | 0.85 mrad | 116 | 0.66 | 0.65 | ok |
| pixels on one row (planar fan, 1.5–6 m) | 6.4 mm | 1.35 mrad | 115 | 0.60 | 8e-6 | **ok** (must not be refused) |
| 3D line (rail, 1.5–9 m) | 53 mm | 8.3 mrad | 95 | 9e-4 | 6e-5 | collinear |
| far-only (25–40 m) | 27 mm | 0.66 mrad | 0 | 0.18 | 0.42 | parallax |
| small object (25 pts, 20 px patch, 3.5 m) | 137 mm | 32 mrad | 25 | 0.09 | 0.88 | conditioning_translation, conditioning_rotation |

- **Monte Carlo NEES.** Over 500 trials with pixels and disparities both noisy (1.5–12 m), mean NEES is **5.95**
  against χ²₆'s mean of 6.
- **The far-only scene shows why parallax is its own gate.** The linearised covariance says 2.7 cm, but mean NEES
  at 25–40 m is **54.7** (500 trials): about 9× overconfident at 2.8 px of disparity against 0.65 px of noise.
- **The collinearity gate was moved from pixels to 3D after measurement.** A pixel-spread gate refused the planar
  fan, which is well conditioned. Conditioning missed the rail (8.3 mrad reported), because stereo noise scattering
  the points off the line fabricates rotation information. Whether that 8.3 mrad is overconfident has not been
  measured [UNVERIFIED].
- **Scene correction.** 120 points on a 40 px patch are not degenerate (2.7 cm and 0.4° measured), so the scene was
  corrected, not the threshold.
- **Noise model on the small-object scene.** Residuals run about 3× the model (σ̂² 3.2), so near the thresholds the
  covariance is honest only because of the inflation. NEES was validated at 1.5–12 m on full-image scenes only.

**Thresholds, all [ASSUMPTION] until calibrated:** σ_t ≤ 0.10 m, σ_r ≤ 0.02 rad, ≥ 10 inliers at ≥ 1°, 3D line
spread ≥ 0.01; σ_px 0.46 (EuRoC, carried to the sim) and σ_d = √2·σ_px.
**Calibration protocol (run in D73, fixed now).** Real loop candidates from the intra-agent source on the sim
missions, with ground truth. A candidate is labelled bad when its PnP edge error exceeds 3σ of its own covariance or
0.2 m / 2°. Each threshold goes to the midpoint of the margin between the good and bad groups (Zhang 2016). The
thresholds are never tuned on the synthetic scenes.
**Not wired.** `VerifyResult` does not export its inlier correspondences, and replacing `ProxyInformation` in the 1.6
broker would change the Phase-1 sim_collab outputs. Both wait for D73, pending a maintainer decision.

## 5. PCM (arch §14.5, D72)

**What.** Two loops are *pairwise consistent* if going round the cycle they form with the odometry comes back to the
identity within noise: for loops a→b (Z₁) and c→d (Z₂),
`e = Log( Z₁⁻¹ · odo(a→c) · Z₂ · odo(d→b) )`, accept if `eᵀ Σₑ⁻¹ e ≤ χ²₆(0.95) = 12.59` (config). The odometry
relative poses and their covariance are composed **from the odometry-only chain by adjoints** — never from the solved
graph, which already contains the loops under test (circular). The largest mutually consistent set is a maximum clique
of the consistency graph: greedy maximal clique (ties broken by edge id, deterministic) by default; exact Bron–Kerbosch
behind a flag for ≤ 25 candidates.

**Known blind spot (kept as a test).** Pairwise consistency does not imply joint consistency, and a self-consistent
cluster of wrong loops passes PCM (catalogue §17.2). The test asserts the current behaviour and is labelled.

## 6. GNC-TLS (arch §14.6, D73)

**What.** Graduated non-convexity: start from a convex surrogate of the truncated-least-squares cost, solve, then make
it gradually less convex, re-solving each time; edges whose residual stays large get weight → 0. Per Yang et al.
Prop. 4 (verified in the paper), with `r̂² = rᵀΩr` (full, whitened, **no ½**) and `c̄² = χ²₆(0.99) = 16.81`:

- `w = 0` if `r̂² ≥ ((μ+1)/μ)c̄²`; `w = 1` if `r̂² ≤ (μ/(μ+1))c̄²`; else `w = (c̄/r̂)√(μ(μ+1)) − μ`;
- `μ₀ = c̄² / (2 r̂²max − c̄²)` after a first solve with all `w = 1`; `μ ← 1.4 μ`;
- stop when `Σ wᵢ r̂ᵢ²` changes < 1e-5 relative, or all weights are within 1e-4 of {0, 1}, or 100 outer iterations;
  if `2 r̂²max ≤ c̄²` everything is an inlier and GNC stops at once (GTSAM's rule).

GTSAM uses `½χ²` as its threshold because its error carries the ½; our energy also carries ½ but the TLS test is on the
full `rᵀΩr` — the factor of 2 is stated here once and asserted by the oracle test.

**Scope.** Only loop-type edges are weighted; odometry stays at `w = 1` (`PoseGraph::AddEdge` already enforces this).
Exposed per run: initial and final residual per edge, the full weight trajectory (edge × outer iteration), accept /
reject sets, μ schedule, LM iterations per outer step. If rejections disconnect a segment that only loops connected,
that is a reported condition (the segment keeps its odometry-only pose), not a crash.

**Oracle.** GTSAM `GncLMOptimizer` (TLS, odometry as `knownInliers`, same c̄² convention) on synthetic graphs: same
inlier set, same final energy to 1e-6 relative.

**Acceptance, fixed now (before any measurement).** Synthetic graphs (grid and sphere topologies), outliers = random
node pairs with uniformly random SE(3) measurements (the paper's protocol), 20 seeds per rate:

- at ≤ 40 % outliers: **precision ≥ 0.99 and recall ≥ 0.95** on loop-edge classification, every seed;
- the 10 → 90 % ramp: success probability, ATE, runtime, and the **breakpoint** reported (no gate);
- one self-consistent wrong cluster per graph: result reported whatever it is (GNC is not expected to beat it).

## 7. Certificate (arch §14.8, D74) — behind a flag

**What is certified, stated plainly.** SE-Sync certifies global optimality of **its** cost, the isotropic chordal cost
`Σ κ‖R_j − R_i R̃_ij‖²_F + τ‖t_j − t_i − R_i t̃_ij‖²` (Rosen eq. 12), not our runtime anisotropic Log cost. So the
certificate path is: inlier graph (after GNC) → κ/τ per edge → refine to a stationary point of the chordal cost (LM on a
chordal-residual backend; the distance moved from the runtime solution is reported) → certificate. Reports say
"certified for the isotropic chordal surrogate".

**κ/τ from an information matrix** (SE-Sync's g2o reader convention, from the diagonal blocks):
`τ = 3 / tr(I_t⁻¹)`, `κ = 3 / (2 tr(I_R⁻¹))`. This also settles open item (c): cubicle and rim, refused by the
anisotropic path for indefinite information, are valid inputs here when the diagonal blocks are positive definite.

**The matrix.** Translation-explicit sparse data matrix `M` (Rosen eqs. 13–18; size `(d+1)n`), never the dense reduced
`Q̃`. With `Λ` the symmetric block-diagonal of `M Yᵀ Y` on the rotation blocks only (eq. 119 / Thm 7), the certificate
matrix is `S = M − Λ`, and the point is certified iff `S ⪰ 0` up to tolerance. Sign: SE-Sync's (`S = Q − Λ`); arch
§14.8 writes `Λ − Q̃` — corrected per PHASE2_PLAN §4, and a noise-free test fixes the sign empirically.

**The test (§49 substitution, recorded in D74).** Arch §14.8 says shifted Lanczos. Adopted instead (SE-Sync's own
practice): attempt a Cholesky (our `GuardedLdlt`) of `S + ηI`; success ⇒ `λ_min(S) ≥ −η` ⇒ certified. On failure, our
own Lanczos estimates `λ_min` for the report. `η` is **relative** (`η = ε‖M‖`, ε in config) because SE-Sync's absolute
1e-3 is meaningless at garage's information scale; the absolute-η verdict is reported alongside. Gauge: `S` has a null
space by construction (global rotation and translation symmetries); its dimension is derived for the explicit form and
asserted on a small graph.

**Tests.** 5-node noise-free graph: dense `cv::eigen` oracle, null-space dimension, certified. SE-Sync graphs: our
refinement reproduces the Table 1 objectives (sphere 1.687e3, torus 2.423e4, grid 8.432e4, garage 1.263, cubicle
7.171e2, rim 5.461e3) and certifies. A perturbed point fails. A **stationary non-global** point (max|g| ≈ 0, higher
cost) fails — the case the certificate exists for.

## 8. Intra-agent loop-candidate source (D73)

`cpp/slam/loop_closure/`: reads the keyframe records and `StereoVo::graph()` from outside (the odometry files are not
touched). Retrieval: 1.6's mean-bit descriptor with a temporal exclusion window — labelled placeholder until 2.1's
EigenPlaces lands. Verification: `edge_verifier` unchanged. Output flows gate → PCM → GNC. The Phase-1 1.6 false loop
accept becomes a regression case that must now be rejected.

## 9. Init acceptance gate — "the SE-Sync graphs that failed from dead reckoning" (maintainer decision 2026-10-18)

The maintainer asked for the advisor's recommendation: a physically logical gate, measured rather than assumed, without a
rabbit hole. Premise corrected: the SE-Sync oracle always started from the files' own vertices, so which graphs fail
from dead reckoning was never measured. It is measured first, then gated. Protocol, fixed before running:

- **Graphs:** tinyGrid3D, smallGrid3D, sphere2500, torus3D, parking-garage, grid3D. Cubicle and rim are refused by the
  anisotropic path and belong only to the D74 certificate gate (κ/τ form, against SE-Sync Table 1, whose objectives
  are in the isotropic chordal cost and are never compared with anisotropic energies).
- **Dead-reckoning start:** node i+1 = node i · Z of the first i→i+1 edge (`dead_reckoning()` in `pgo_gtsam.py`).
- **Reference minimum:** GTSAM LM from GTSAM `InitializePose3` on the same anisotropic cost (unit `PriorFactorPose3`
  on node 0; if multifrontal Cholesky throws, §3: `computeOrientationsChordal` + zero translations, then LM). GTSAM LM
  from dead reckoning is recorded as a free ablation row. `pgo_gtsam.py --init {file,dr,chordal}`.
- **Failure set S:** graphs where our LM from dead reckoning ends stationary but > 1e-6 relative above the reference.
- **Gate (every graph, hence S):** chordal init + LM lands within 1e-6 relative of the reference, stationary, in
  ≤ 50 iterations under pgo_v2.
- **Ablation reported:** dead reckoning / MST / chordal start — start energy, final energy, iterations.
- **If S is empty:** said so in D70; the failure set becomes 20 seeds of the random generator at 6σ noise
  (`--noise-scale`, the only recorded dead-reckoning failure mode). The old generator is not reconstructed.
- **Literature anchor:** Carlone, Tron, Daniilidis, Dellaert, ICRA 2015, "Initialization techniques for 3D SLAM" ran
  the odometry-vs-chordal experiment on these benchmarks; cited in the ALGORITHMS entry (one fetch, its numbers checked
  at the source).
- **Out of scope here:** global optimality of chordal + LM (the D74 certificate decides that later; it does not block
  this gate); the original "case 2".

**Result (2026-10-19, `PgoInit.SeSyncDeadReckoningGate`, 2 h 35 min; GATE PASSED).** Energies are the anisotropic cost
(½Σ rᵀΩr). The signed number is (final − reference) / reference at the production stop; every run was stationary after
the polish.

| graph | dead reckoning → LM | GTSAM LM from DR | MST → LM | chordal → LM | reference |
|---|---|---|---|---|---|
| tinyGrid3D | 9.313909434 (5 its, +9.3e-9) | same (10) | 5 its, +9.9e-10 | 4 its, +7.0e-9 | 9.313909434 |
| smallGrid3D | 517.9253324 (7, +6.5e-9) | same (11) | 6, +7.5e-10 | 5, +1.9e-10 | 517.9253324 |
| sphere2500 | 675.7009629 (10, +2.5e-10) | same (7) | 9, +1.4e-10 | 8, +6.2e-10 | 675.7009629 |
| parking-garage | 0.6341923996 (15, +3.8e-10) | same (7) | 15, +1.6e-10 | 15, +1.4e-10 | 0.6341923996 |
| **torus3D** | **29950.00596** (89, +1.5) | **29950.00596** (353) | 12117.63688 (7, +1.0e-9) | 12117.63688 (6, +2.3e-9) | 12117.63688 |
| **grid3D** | **253372.1013** (71, +5.0) | **253372.1013** (135) | 42179.97894 (9, +1.9e-11) | 42179.97894 (8, +2.4e-10) | 42179.97894 |

The energies shown are after the polish; the iteration counts and signed gaps come from the production solve. For
example, torus3D from dead reckoning stopped at 29950.38657 and polished to 29950.00596; grid3D stopped at 253372.7759.

**Reproducing it** (oracle image `ariadne-oracle:1`; output under the branch cache, never the shared `.cache/oracle`):
`python -m oracle.pgo_gtsam --random 0 --init dr --out <cache>/pgo_sesync_dr --g2o NAME=PATH ...` (one `--g2o` per
graph, the files `models.sh pgo-sesync` fetches into `.cache/datasets/pgo/se_sync`), the same with
`--init chordal --out <cache>/pgo_sesync_chordal`, then `ARIADNE_PGO_SESYNC_DR=<cache>/pgo_sesync_dr
ARIADNE_PGO_SESYNC_CHORDAL=<cache>/pgo_sesync_chordal test_opencv --gtest_filter='PgoInit.SeSyncDeadReckoningGate'`.

- **S = {torus3D, grid3D}.** On both, our LM from dead reckoning stops in the same local minimum as GTSAM's LM from
  the same start (all printed digits agree after the polish), so the failure is the start, not our solver.
- **Gate:** chordal + LM within 7.0e-9 relative of the reference on every graph (worst: tinyGrid3D), stationary, in
  4–15 iterations (≤ 50). MST also rescues both members of S (9 and 7 iterations); it stays as the ablation baseline.
- **Against Carlone et al. 2015, Table I,** on the graphs both studies use (their sphere n = 2500 = sphere2500, torus
  n = 5000 = torus3D, cube n = 8000 = grid3D, garage n = 1661 = parking-garage): they report Gauss–Newton from
  odometry failing on torus and cube and fine on sphere and garage, which is exactly our S. Their sphere-a, cubicle and
  rim are not in this gate (sphere-a is not in our set; cubicle and rim belong to D74). Their costs use the angular
  distance (their eq. 2), so only the verdicts are compared, not the numbers.
- The 6σ fallback was not needed (S is not empty).
- **Evidence settings.** The gate, the SE-Sync check and the 100-case oracle ran while pgo_v2 was the struct default;
  `PgoV2()` sets the same two fields (name, `clamp_diagonal`), so the results stand for the pose graph as built.
- **Cost.** grid3D takes most of the 2.6 h. This gate is an evidence run (env-gated, never in `--premerge`), re-run only
  when the solver or the initialisation changes.

## 10. Decisions and the §49 ritual

| ID | Content |
|---|---|
| D70 | chordal init + baselines; LM damping fix; open item (a); placement (arch §3 update) |
| D71 | degeneracy gate; PnP-Hessian loop information replacing the [PROXY] diagonal |
| D72 | PCM (odometry-only composition; greedy + exact flag) |
| D73 | GNC-TLS + intra-agent loop source |
| D74 | certificate (chordal surrogate, Cholesky of S+ηI substitution, relative η); κ/τ for cubicle/rim |

Each adopted piece: ALGORITHMS.md entry (append-only hunk) → arch decision → ablation matrix → benchmark config
(`configs/slam/*`, isolated hunks) → manifest fields (`cpp/runtime/manifest/*`, isolated hunks). Every hunk to a shared
file is listed in the handover.
