# Robust back end: degeneracy gate, PCM, GNC-TLS, optimality certificate

Plan 2.5 (local-slam.md#estimator-plan §1, phased plan Phase 2): replace the Phase-1 pose-graph placeholders with arch §13–§14 as
specified — chordal initialisation, degeneracy gate → PCM → GNC-TLS (the §14.5 order is inviolable), the SE-Sync-style
certificate behind a flag, and a real intra-agent loop-candidate source. Decisions D70–D76.

**Status (2026-10-22): implemented through D76.**
- D70 chordal init, D71 degeneracy gate, D72 PCM.
- D73: GNC-TLS, weight reuse, the intra-agent loop source, the calibrated loop noise model and gravity priors.
- D74: the certificate (§7), behind `loop_closure.certificate`.
- D75: the aliasing outlier ramp and its breakpoint (§6).
- D76: the joint-graph entry point for the broker, plus shared gravity (§11).

The per-section "As built" notes record what was measured. **Still open:**
- **SE-Sync certificate run on the large graphs** (sphere, torus, grid3D, cubicle, rim): pending approval
  (heavy). garage, tinyGrid3D and smallGrid3D certify.
- **The planted-outlier gate wording.** "Not certified on planted-outlier graphs" holds for 4/10; 6/10 certify. This is
  reported, not tuned (local-slam.md#estimator-plan §3; D74).

**Defaults.** Intra-agent loop closure and gravity priors in the pose graph are **off by default**. Only
`configs/slam/euroc_v2_loops.yaml` (loops on, `gravity_priors: false`) and `euroc_v2_loops_gravity.yaml` carry the
`loop_closure:` section; every other shipped config omits it, so the runner never constructs the loop adapter. The
run manifest records the slam config by name (`slam_config:`), which is how a run shows which mode it used. There is
no separate manifest field. Evidence: D73 part 2b, `evaluation/runs/robust-backend/loop_closure/`.

**Reproducibility of the oracles.** The oracles are env-gated and skip without their variables:
- **`infra/scripts/models.sh pgo-oracle` → `ARIADNE_PGO_ORACLE`:** the 100 random cases, which take seconds and are
  in ctest when set.
- **`models.sh pgo-sesync` → `ARIADNE_PGO_SESYNC`:** the SE-Sync graphs, about 2 h (grid3D and rim dominate).
- **`ARIADNE_PGO_SESYNC_DR` / `ARIADNE_PGO_SESYNC_CHORDAL`:** the §9 init gate, 2.6 h, from `pgo_gtsam.py --init
  dr|chordal` (§9 gives the commands).
`--premerge` sets only `ARIADNE_PGO_ORACLE`. The SE-Sync suites are hour-long evidence runs that the gate never
enables.

## 0. Preflight (done before this design)

- **Literature, checked at the sources.** GNC (Yang et al., RA-L 2020, arXiv 1909.08605: Prop. 4, Remark 5, §VI-B);
  SE-Sync (Rosen et al., IJRR 2019, arXiv 1612.07386: eqs. 12–18, Thm 7, eq. 119, Table 1; reference code read for
  conventions only — LGPL-3, nothing transcribed); PCM (Mangelson et al., ICRA 2018); TACO (arXiv 2606.29851);
  EGNC-PGO (2310.06765); AGNC (2308.11444); group-k consistency (2308.02674); COSMO-Bench (2508.16731). None changes the
  §14 pipeline or its order; all are ablation candidates only.
- **OpenCV-native check (installed 5.0.0).** Present and used: `cv::SVDecomp`, `cv::buildMST`, `cv::eigen` (dense, test
  oracle only), `cv::detail::LevMarqBase`. Absent: sparse eigensolver / Lanczos, max-clique, robust PGO
  (`cv::detail::PoseGraph` rejected in D51).
- **Oracles (ariadne-oracle:1, unchanged).** GTSAM Python exposes `InitializePose3`, `GncLMOptimizer`, `GncLMParams`,
  `GncLossType`, `GncScheduler`. Oracles only, never runtime (arch §0.2).

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
eigenvalue ratio (pose-graph.md §5), so the hypothesis is that the absolute floor dominates the weak directions.

**Protocol (fixed before measuring).** `PgoOracle.DampingAblation` (tests/unit/test_pgo_oracle.cpp, env-gated): same
dead-reckoning start, same tolerances as `SeSyncBenchmarkGraphs`, one setting changed at a time — baseline / clamp off /
Jacobi on / both. Report iterations, linearisations, failed factorisations, final energy vs GTSAM, max|grad|. Fix = a
settings change from pgo_v1 to `pgo_v2` only if it wins on garage **and does not regress** the other
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
  `.cache/oracle/pgo_sesync` is left untouched; `models.sh pgo-sesync` is re-run after merge).
- **Guard before adopting clamp-off on the GNC path.** The 1e-6 floor is currently the only regulariser if a block of H
  loses rank — e.g. GNC zeroes the only loop joining two odometry segments (D56 makes segmented graphs real). The GNC
  wrapper therefore anchors every connected component of the *active* graph explicitly, and a disconnection test is in
  the no-regression set. The OpenCV floor is a hard-coded constant, so it cannot be made relative instead.

**As built (2026-10-19).** There is no pgo_v1 config file: the protocol above assumed one, but pgo_v1 has only
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
broker would change the Phase-1 sim_collab outputs. Both wait for D73, pending a decision.

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

**Acceptance (fixed before the first measured run).** Synthetic single chain of 300 keyframes, 30 true loops, outliers
are random node pairs with uniformly random SE(3) measurements, 20 seeds per rate. Recall of true loops must be ≥ 0.95
on every seed at ≤ 40 % outliers: a true loop PCM drops is lost for good, while an outlier that passes is GNC's job.
Precision, greedy against exact clique size, and time per event at about 300 loops are reported, not gated.

**As built (2026-10-19, D72).** `cpp/slam/robust/pcm.{hpp,cpp}`, `Pcm(PoseGraph)`, a pure library with no wiring.
- **Odometry legs** come from the odometry-only chain (`kOdometry` and `kStereoOdometry` edges) by prefix-composed
  adjoints, so each leg is O(1). Components must be chains; anything else throws. Parallel odometry edges between one
  node pair are counted, and the first in edge order is used.
- **Shared odometry edges.** When both legs run along one chain they can share edges: crossing loops 10→100 and
  90→20 share [20, 90). The chain is split at the four endpoints, and each piece carries the summed coefficient
  σ₁M₁ + σ₂M₂, so shared edges count once, with signs (legs in opposite directions cancel).
- **Groups:** the unordered pair of odometry components. A group with 1 loop is accepted and tagged "untested". A
  group of 2 or more whose maximum clique is 1 accepts nothing ("no_consensus").
- **Clique:** greedy CLIQUEHEU (Pattabiraman et al., arXiv 1209.5818) by default. Exact Bron–Kerbosch with Tomita
  pivoting sits behind a flag for groups of ≤ 25. Loops are ordered by (from, to), so the result does not depend on
  edge insertion order (tested over 5 shuffles).
- **Threshold, a §49 deviation from arch §14.5's fixed χ²₆(0.95).** A maximum clique keeps a true loop only if all
  of its n − 1 pairs pass, so a 5 % per-pair rejection drops true loops.
  - **Measured at the fixed quantile:** recall mean 0.84–0.86, worst 0.60–0.70.
  - **The default controls false rejection per loop instead:** α_pair = α_loop/(n − 1), with the threshold from the
    closed-form χ²₆ quantile.
  - **α_loop = 0.001 is derived from the requirement, not fitted.** About n·α_loop/2 true loops are lost per group,
    and recall ≥ 0.95 on 30 loops allows one loss. P(≥ 2 lost) is about 1e-4 per group at 0.001, and about 0.75
    loops are lost on average at 0.05.
  - The fixed quantile is kept as the ablation baseline.

Tests (`tests/unit/test_pcm.cpp`, 12/12; ctest 355/355 on the merged tree):

| check | result |
|---|---|
| odometry leg Monte Carlo NEES, forward / backward, forward and mixed storage | 6.01 / 6.01, 6.24 / 6.24 |
| pair statistic mean over 400 trials (χ²₆: 6): disjoint, nested, crossing, backward, shared end | 5.90, 6.15, 5.95, 5.93, 5.75 |
| pair statistic, two agents, loop 2 stored forward / reversed | 6.07 / 5.96 |
| greedy clique size = exact, planted-clique graphs of 8–24 vertices | 200 / 200 |
| random outliers 10–40 %, default: recall (mean / worst), precision | **1.000 / 1.000**, 1.000 at every rate |
| random outliers 10–40 %, fixed χ²₆(0.95): recall (mean / worst) | 0.84–0.86 / 0.60–0.70 |
| near-miss outliers 0.2 m / 1°, default vs fixed 0.95: precision (recall) | 0.75 (1.00) vs 0.79 (0.83) |
| near-miss outliers 0.5 m / 5° | 0.94 (0.99) vs 1.00 (0.85) |
| near-miss outliers 1.0 m / 10° | 1.00 (1.00) vs 1.00 (0.85) |
| blind spot: 5 loops aliasing one rigidly displaced region against 3 true loops | the 5 wrong loops accepted (labelled) |
| 3000 nodes, 300 loops (60 random outliers) | 186 ms per event, all 240 true loops accepted |

- **The trade, stated plainly.** The default lets through near misses close to the loop noise (0.2 m / 1° is a 4σ
  translation error at σ_t = 5 cm, and about all 10 pass) and some at 0.5 m / 5°. The fixed quantile rejects them, but it drops
  15–17 % of true loops at every error size. GNC (D73) is the stage that must reject the near misses, so its test
  includes them.
- **D56 bridges.** A lost stretch is bridged by an edge with identity covariance, so pairs spanning it pass
  trivially there. That is conservative, and it blunts PCM across the bridge. Not measured on real data.
- **The bound loosens as a group grows.** χ² bound at α_loop = 0.001: 27.6 at 10 loops, 30.3 at 30, 33.1 at 100,
  35.6 at 300, against the fixed 12.59. A mission's single intra-agent group becomes more permissive over time.
- **The blind spot is unchanged by this choice.** A self-consistent wrong cluster passes under either threshold, and
  the wider bound makes a small-offset cluster slightly more likely to merge with the true loops' clique. The defence
  remains the degeneracy gate and Iteration 7.

**Decision (2026-10-19), following the design review: the per-loop threshold is the default.**
The fixed quantile's failure (dropping true loops) cannot be recovered downstream. The default's failure (near misses
passing) is GNC's job. A 0.5 m / 5° near miss has rᵀΩr ≈ (0.5/0.05)² + (0.087/0.01)² ≈ 176, against GNC's
c̄² = 16.8. **Condition, carried into §6:** if GNC fails its near-miss gate, the first remedy is a tighter α_loop
(0.01), not a return to the fixed quantile.

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
- **Near misses after PCM (added 2026-10-19 with the D72 decision).** GNC's acceptance runs on the loops PCM
  accepted, with 0.5 m / 5° near misses mixed in, and requires **precision ≥ 0.99 on them**. Near misses at
  0.2 m / 1° (about 4σ) are scored by their effect on ATE, not by classification. If the 0.5 m gate fails, the first
  remedy is α_loop = 0.01 in PCM (§5).

**Definitions (fixed before the first measured run, 2026-10-20).** Inlier means final w > 0.5, and precision and
recall are counted over loop edges only. Every run starts from the odometry-composed poses, never from a chordal
init that includes the outliers. The near-miss precision is TP/(TP + accepted near misses), pooled over 20 seeds.

**D73 is split into three parts** (D59's pattern; IDs are limited to D70–D74):
1. **Part 1:** the GNC core plus component anchoring, a pure library.
2. **Part 2:** the intra-agent loop source, wired gate → PCM → GNC, with config and manifest fields, covariances
   off on the event path, and the D71 threshold calibration.
3. **Part 3:** the `VerifyResult` inlier export and the `ProxyInformation` swap, with `--premerge` and a re-pin.

**As built, part 1 (2026-10-20).** `cpp/slam/robust/gnc.{hpp,cpp}`, `SolveGnc(graph, anchor, active mask)`.
- **Runs on a working copy.** The input graph gets the final poses and weights once (`PoseGraph::SetGncWeight`,
  loop-type edges only, w ∈ [0, 1]).
- **Edge indices are stable.** The mask comes from PCM's accepted set, so the gate, PCM and GNC reports join on the
  same index. Inner solves never recompute marginals.
- **Conventions checked against GTSAM 4.3.0's `GncOptimizer.h`:** barcSq = 0.5·χ²inv; TLS λ₀ = barcSq/(2e − barcSq)
  over factors, taking the smallest positive value; stop on relative cost, weight convergence or λ. These match ours.
  The GTSAM oracle comparison is moved to part 2 [UNVERIFIED until then].
- **`SolvePoseGraph` solves each connected component of the active graph** (loops at w ≤ 1e-4 are inactive) with
  its own anchor: the given one, else the lowest id. Covariances outside the anchor's component are NaN. A single
  component takes the unchanged path, and the existing PGO tests stay green (numerical 10/10; `PgoLevMarq`, chordal
  and the 100-case GTSAM oracle 9/9).

| test (`tests/unit/test_gnc.cpp`, 10/10) | result |
|---|---|
| Prop. 4 weight is the argmin of the Black–Rangarajan dual w r² + μc̄²(1−w)/(μ+w) | to 1e-4 on a 2·10⁵ grid, all (μ, r²) |
| no outliers | stops (weights), energy equals plain LM to 1e-9 |
| **grid (6×6×3), 10–40 % random outliers, 20 seeds: precision / recall worst** | **1.000 / 0.967** (gate ≥ 0.99 / ≥ 0.95) |
| **sphere (8 rings × 20), same** | **1.000 / 0.967** |
| ATE (grid / sphere) | 0.081–0.092 m / 0.128–0.148 m |
| ramp 50–90 %, grid, 10 seeds | 10/10 successes at every rate (random outliers are easy; a real breakpoint needs harder outliers) |
| **near misses 0.5 m / 5° through PCM → GNC** | PCM removed 200/200, so pooled precision is 1.000. **GNC alone:** 0/200 accepted, 1186/1200 true kept |
| near misses 0.2 m / 1° through PCM → GNC | 199/200 pass PCM, 143 accepted by GNC; ATE 0.102 m against 0.091 m without them (+11 %) |
| self-consistent wrong cluster (8 loops, one rigid offset, 30 true loops) | 0/8 accepted (reported) |
| two agents linked only by 3 random loops | GNC alone keeps 1 (a lone link is unverifiable); PCM → no_consensus → GNC 0/3, 2 components, B held at its anchor |
| bit-identical twice | yes |
| **timing, maze graphs with 10 % outliers** | **1.6 s at 1,000 nodes, 4.75 s at 3,000** (about 27 outer iterations, 100–120 LM iterations) |

- **The 0.5 m near-miss gate is met, but trivially.** On the grid topology PCM rejects every 0.5 m near miss, so
  the gate does not exercise GNC. GNC on its own rejects all of them as well, measured separately.
- **0.2 m / 1° near misses** are 4σ translation errors, and neither stage separates them. They cost 11 % ATE. That
  is the price of the D72 decision, now measured.
- **Timing replaces the earlier [ASSUMPTION]** given for demo latency (0.3–0.8 s at 1,000 nodes).
  Measured: 1.6 s at 1,000 and 4.75 s at 3,000 per loop event, run in the background. The cost is about 27
  warm-started LM solves. The obvious lever, warm-starting the weights from the previous event so that only new
  loops start at w = 1, is not implemented. It is part 2's first optimisation if the demo needs it.

**Part 2a — weight reuse across loop events (decision: "weight reuse then wire up", 2026-10-20).**
- **State.** The caller carries a `GncState` (per edge: −1 not active last event, 0 outlier, 1 inlier). Edges are
  append-only, so indices are stable keys. A loop that was PCM-active last event and is still active starts at its
  previous decision; every other active loop starts at 1. The PCM mask always wins.
- **Fast path.** One solve, accepted only if the split is a TLS fixed point:
  - every loop held at 1 has r² ≤ c̄²;
  - every active loop held at 0 has r² > c̄² (otherwise TLS would include it).
  This is looser than Remark 5's 2r²max ≤ c̄², deliberately: converged inliers sit just under c̄², so the c̄²/2
  rule would almost never fire.
- **Fallback.** Otherwise the fast-path solve is discarded, and cold GNC runs from the event's input poses with every
  active loop a candidate again.
- **Path dependence:** the result depends on the event history. It stays deterministic under replay of the recorded
  sequence.
- **Protocol (fixed before the first run).**
  - **Setup:** a maze graph of 1,000 nodes, loops arriving in trajectory order with one event per loop. The mix is
    true revisits, about 10 % random outliers and three 0.5 m / 5° near misses. An early outlier is paired with the
    first true loop, so PCM rejects that loop and accepts it later.
  - **Per event:** PCM, warm GNC, and cold GNC on a copy of the same input.
  - **Gate:** the warm inlier set equals cold's at every event, or is no worse in precision and recall against the
    truth. Final ATE within 1 % of cold's, and the PCM rejected-then-accepted case must occur.
  - **Reported:** the fast-path hit rate, fallback reasons, warm against cold time, and the warm path alone at
    3,000 nodes.
  - If the hit rate is low, the next lever (cold GNC started at a high μ from the previous weights) is measured
    first, not built.
- **As built (2026-10-20), and what measurement changed.** Three fast-path rules were tried in turn, the protocol
  unchanged throughout:

  | fast-path rule | fast path | warm ≠ cold | final precision / ATE vs cold | gate |
  |---|---|---|---|---|
  | TLS fixed point only | 82/83 | 25 events | 1.000 at the end, 0.88–0.94 during | fails: a wrong loop bent the odometry and kept its own r² small |
  | + innovation 2ΔE ≤ χ²₆ₙ(0.99) | 79/83 | 45 events | 0.986, ATE +3.6 % | fails: an early 0.5 m near miss is statistically consistent with large drift, TLS keeps it, cold GNC rejects it |
  | + innovation ≤ ½χ²₆ₙ(0.99) (Remark 5's region) | **60/83 (72 %)** | **0 events** | **1.000, ATE identical (0.1127 m)** | **passes** |

  - **The adopted rule.** The fixed point, plus the innovation of the new loops (one extra warm solve with them held
    at 0) within half the χ²₆ₙ(0.99) bound. The fast path takes only what GNC's convex start would accept; the
    ambiguous band goes to cold GNC.
  - **Measured cost.** 1,000 nodes: 82 ms per event warm against 105 ms cold (−22 %). 3,000 nodes, warm only:
    83 % fast path, 227 ms mean per event, 655 ms worst. Bit-identical over a fixed event sequence.
  - **Correction to the part 1 timing story.** The 1.6 s / 4.75 s of part 1 was GNC *without* PCM in front.
    Wired gate → PCM → GNC, PCM removes the random outliers, and even cold GNC costs about one LM solve per event.
    Weight reuse is a 20 % saving there, not the order of magnitude part 1's numbers suggested.

**Outlier ramp with perceptual-aliasing groups (D75, 2026-10-22).**
- **Why a new protocol.** Random outliers gave no breakpoint, so false loops now come in mutually consistent groups.
  The generator is `cpp/slam/robust/aliasing_generator.hpp`: one spurious Δ per group, the Sünderhauf & Protzel
  grouped policy. PCM accepts a planted group whole, which its test asserts.
- **Protocol:** in `tests/unit/test_aliasing_ramp.cpp`.
- **Result** (`evaluation/runs/robust-backend/outlier_ramp/`):
  - PCM → GNC breaks at **90 %** with groups of 10 and **70 %** with two growing groups. In both cases the largest
    false group outnumbers the true loops, PCM's maximum-clique limit.
  - GNC alone has no breakpoint (worst 5/10 at 70 %), at ~10× the runtime.

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
§14.8 writes `Λ − Q̃` — corrected per local-slam.md#estimator-plan §4, and a noise-free test fixes the sign empirically.

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

**As built (D74, 2026-10-22).** `cpp/slam/robust/certificate.{hpp,cpp}`, as specified above, with one correction
from review.
- **The shift.** The Cholesky test grounds the anchor's translation and shifts only the rotation diagonal. By Schur
  this is exactly the reduced test; a full S + ηI over-certifies, passing at 0.61× the reduced η on a measured graph.
- **η.** η = 1e-10·‖M‖∞. A certified point carries f − f* ≤ |gap| + 3nη.
- **Results.**
  - Unit tests 8/8.
  - garage f = 1.26252 (Table 1: 1.263), certified, with bound 0.024 % of f. tinyGrid3D and smallGrid3D certify.
  - Planted outliers kept in the cost: 6/10 certify; after GNC 10/10.
  - A stationary non-global point and a perturbed point fail.
- **Flag.** `loop_closure.certificate` (optional key, off): read-only on the mirror, bit-identical tested.

## 8. Intra-agent loop-candidate source (D73)

`cpp/slam/loop_closure/`: reads the keyframe records and `StereoVo::graph()` from outside (the odometry files are not
touched). Retrieval: 1.6's mean-bit descriptor with a temporal exclusion window — labelled placeholder until 2.1's
EigenPlaces lands. Verification: `edge_verifier` unchanged. Output flows gate → PCM → GNC. The Phase-1 1.6 false loop
accept becomes a regression case that must now be rejected.

**As built, part 2b backend (2026-10-20, slam-only; the runtime adapter follows the main-branch merge).**
- **Pieces.** `cpp/slam/loop_closure/retrieval.hpp` takes plain float descriptors: cosine similarity, a temporal
  exclusion window, top-k. `loop_backend.{hpp,cpp}` runs gate → PCM → GNC (with `GncState`) once per query keyframe
  (candidates batched) on a mirror of the VO graph.
- **The mirror.** `Sync` copies the VO's final odometry edges and dead-reckons new nodes on the corrected
  trajectory, which adds zero energy (tested), so part 2a's innovation premise holds. A broken chain throws.
- **Read-only towards the VO:** VO outputs are unchanged with loops on or off.
- **Latency.** In window mode a keyframe enters the mirror when it leaves the BA window, about one window (10
  keyframes) late. Its loops wait (`Ready()`), and the wait is part of their latency, recorded per event. In
  two-frame-stub mode the delay is zero.
- **Edge:** A (pixels) → B (3D points), Z = the gate's refined T_A_B, pinned by a zero-residual test.
- **Synthetic result** (`tests/unit/test_loop_backend.cpp`, 4/4): two laps of a 6 m square, 10 loop events, one
  self-consistent wrong loop that passes the gate. 10/10 true loops are GNC inliers, the wrong loop is rejected, and
  ATE goes from 0.304 m (odometry) to 0.086 m (mirror), with 7/10 events on the fast path.

**EuRoC evaluation definitions (fixed 2026-10-20, before the adapter exists).**
- **Funnel per sequence:** keyframes → retrieved candidates → verified (`edge_verifier`) → passed the gate →
  accepted by PCM → GNC inliers.
- **Retrieval recall** against ground-truth revisits, meaning keyframe pairs outside the exclusion window within
  1 m and 30° of each other. The mean-bit descriptor is a placeholder, so low recall is reported as is.
- **ATE:** the mirror's keyframes against the VO's keyframes, the same keyframe set, with the existing scorer and
  alignment. Keyframes still in the window at the end of a run are scored at their VO-relative dead-reckoned pose.
- **Latency per loop event:** window wait plus gate, PCM and GNC.
- **D71 calibration** (design §4 protocol). Each verified candidate is labelled bad if its PnP edge error against
  the ground-truth relative pose exceeds 3σ of its own covariance, or 0.2 m / 2°. Each threshold goes to the
  midpoint of the good/bad margin.
- **Memory:** features kept per keyframe are capped in the adapter (3,000 keyframes at demo scale).

**First real-data run, V1_01_easy (2026-10-21; one flight, so it is not yet D73's result).** Keyframe-refined ATE
0.0563 m → 0.0392 m with loops. Funnel, labelled by ground truth at 0.2 m / 2° absolute:

| stage | n | good | precision |
|---|---|---|---|
| verified (edge_verifier) | 464 | 144 | 31 % |
| passed the degeneracy gate | 154 | 97 | 63 % |
| refused by the gate | 310 | 47 | 15 % |
| PCM accepted | 42 | 28 | 67 % |
| GNC inlier | 26 | 21 | 81 % |

- **The gate discriminates.** It refuses 82 % of the bad candidates and keeps 67 % of the good ones (σ_t median
  0.030 m for good, 0.096 m for bad).
- **The loop covariance is ~100–1000× overconfident on real data** (NEES p10 1.1e3, p50 1.1e4), against 5.95 on
  synthetic data. PCM and GNC test against χ² bounds that assume honest covariances, so they reject good loops and
  filter poorly.
- **Label rule.** The protocol's "3σ of its own covariance" label prong is dropped (it made every candidate bad);
  labels are absolute.
- **Known mismodelling, stated [ASSUMPTION]:**
  - ORB keypoint σ grows ×1.2 per octave, but the gate uses 0.46 px for all;
  - SGBM depth is read at the rounded pixel with stereo-LK's disparity σ, and its bias is invisible to residuals;
  - ground truth is the nearest sample within 10 ms, not interpolated, so part of the "error" is timing.

**Noise-model fix, criterion fixed before fitting (2026-10-21).**
- **Model.** A model-error covariance per loop, Σ = Σ_PnP + Σ_model, in T_A_B's left tangent. Two candidates:
  1. an additive floor diag(σ_t² I₃, σ_r² I₃);
  2. k·Σ_PnP + floor.
- **Fit** on Machine Hall's ground-truth-good loops, translation and rotation separately and robustly:
  σ² = median(err²) / 2.366 (the χ²₃ median).
- **Validate on V1_01, held out:** mean NEES of the good loops within [5, 7], and ≤ 3 % above χ²₆(0.99). Both models
  reported; the simpler one is adopted if both pass.
- **Then,** in this order:
  - recompute σ_t / σ_r with the floor, and rerun the D71 midpoint calibration of the gate thresholds offline on
    the same CSVs (the 0.10 m / 0.02 rad values are only right for the overconfident covariance);
  - check PCM's odometry input, the pair statistic over ground-truth-good loop pairs (should be ~6), before and
    after; if it stays far above 6, a scale on the odometry legs inside PCM (never in the VO), decided only on
    those numbers;
  - one rerun per flight.
- **Sim:** the EuRoC floor does not transfer (different camera, rendered images). The demo needs its own
  calibration from one sim mission run with ground truth. Every calibrated value in config carries its provenance.

**Result with the noise model (2026-10-21; the model accepted with its tail miss stated).**
- **Fits.** The MH_01-only fit failed held-out V1_01 (mean NEES 9.53, 11.8 % above χ²₆(0.99)). Refit jointly on
  MH_01 + V1_01: k_t 1.33, floor_t 0.014 m, k_r 0.178, floor_r 0.0095 rad.
- **Held-out V1_02:** mean NEES **5.24** (pass) but **4.4 %** above χ²₆(0.99) (fail, against ≤ 3 %). The median of
  3.61 shows a slightly conservative bulk under a heavy tail that one Gaussian cannot match. GNC handles the tail:
  it kept 142 of the 145 good loops PCM accepted.
- **Ground truth** for labels is interpolated between samples. The fit used nearest-sample labels, so the floor is
  slightly conservative.

| flight | keyframe-refined ATE, VO → loops | retrieval recall | good / all: verified · gate · PCM · GNC | good-loop NEES (mean, > χ²₆(0.99)) | good-pair PCM stat (median) | latency mean / max |
|---|---|---|---|---|---|---|
| MH_01 (train) | 0.0676 → **0.0207 m (−69 %)** | 152/245 | 156/302 · 118/199 · 118/178 · 118/177 | 4.76, 0.0 % | 2.1 | 2.4 / 7.3 s |
| V1_01 (train) | 0.0563 → **0.0382 m (−32 %)** | 109/212 | 144/464 · 98/154 · 90/117 · 86/95 | 6.96, 4.9 % | 6.7 | 2.4 / 5.3 s |
| V1_02 (held out) | 0.0517 → **0.0254 m (−51 %)** | 129/194 | 180/341 · 168/209 · 145/165 · 142/146 | 5.24, 4.4 % | 7.8 | 1.3 / 4.4 s |

- **Precision rises at every stage.** On V1_02: verified 0.53 → gate 0.80 → PCM 0.88 → GNC 0.97. Before the model
  (V1_01) PCM did not improve on the gate.
- **MH_01's "bad" GNC inliers (59).** None is gross (none is over 1 m or 10°). Translation error has a median of
  2 cm, but rotation error a median of 3.8°, and these loops are mutually consistent: a systematic rotation offset,
  not random wrong loops. A ground-truth camera–body rotation error showing up on loops between very different
  headings is one candidate cause [UNVERIFIED]. ATE still improves by 69 %.
- **PCM good-pair statistic.** MH_01 2.1 (conservative), V1 6.7–7.8 (about honest). The odometry legs need no
  scale.
- **On MH_01 the model traded precision for recall, not ATE.** Without it GNC kept 51 good loops of 55 accepted
  (0.93); with it, 118 of 177 (0.67). ATE was 0.0205 m against 0.0207 m. On V1_01 it raised both recall (21 → 86
  good) and precision (0.81 → 0.91).
- **Evidence:** `evaluation/runs/robust-backend/loop_closure/` (README, `analyze_loops.py` reproduces these tables).
- **Latency** is dominated by waiting for the BA window to release the keyframe (about 10 keyframes).
- **Retrieval recall** is 51–66 % with the mean-bit placeholder; EigenPlaces (2.1) is expected to raise it.
- **Not done.** The sim needs its own calibration. ORB keypoint σ per octave is still a flat 0.46 px. The gate's
  σ thresholds are left at 0.10 m / 0.02 rad: the Youden fit (0.075 / 0.0145) moved them little and was not adopted.

**Part 2c — gravity priors in the pose graph (2026-10-21; the D59 handoff, items 1–3; item 4 is Phase 3).**
- **Solver** (`PgoProblem`, `GravityOptions`, off by default). One 2-DoF gravity-direction variable per agent with
  priors, initialised at their common g_ref_W (a mismatch throws) and retracted on the sphere along
  GravityTangentBasis(ĝ). Each prior costs ½ρ(s), with s = rᵀ(0.6 Λ)r and a Huber kernel at k² = χ²₂(0.99) = 9.21
  (Mahony's p99 ≈ 18). The priors are not in GNC (§12.1).
- **The 0.6 information scale** is an effective-N correction: the lag-1 autocorrelation of Mahony's emitted-prior
  errors is 0.12–0.36 and lag 2 is ≈ 0 (MH_01, V1_01, V1_02 traces). The pose-derived candidates reach 0.82, which
  confirms D59's choice of Mahony.
- **Mirror.** `Sync` copies the VO's priors, queues those whose keyframe is not mirrored yet, and re-solves when new
  ones land (current loop weights, no GNC), so the priors act between and without loops. Each loop event therefore
  again starts from the optimum without its new loops: the part 2a premise, which zero-energy sync alone no longer
  guarantees once priors exist.
- **Config:** `loop_closure.gravity_priors` (false in `euroc_v2_loops`, so its evidence still reproduces; true in
  `euroc_v2_loops_gravity`).
- **Tests** (`tests/unit/test_gravity_pgo.cpp`, 4/4):
  - the analytic gradient matches finite differences of the energy, Huber regime included;
  - disabled, the solve is bit-identical to a graph without priors;
  - a 1.50° reference-gravity error is estimated to 0.62°, and tilt RMS drops from 3.78° to 2.01°.
    **Corrected 2026-10-24 (D123–D124):** that is one seed under x86's argument-evaluation order; arm64 gets 2.71°.
    The fixture's own ĝ σ is ≈ 2.0° per axis, so < 1° is not a property it can show. The test becomes a 50-seed NEES
    consistency check (mean 1.7–2.5, inside the χ² 99 % band);
  - a 30° outlier prior moves its keyframe by 1.85°.
  `PgoLevMarq` and the numerical `PgoProblem`/`PoseGraph` tests are unchanged and pass.
- **SOTA comparison, recorded, not built.** The common inertial loop-closure design keeps roll and pitch from the
  VIO and optimises a 4-DoF pose graph (VINS-Mono; ORB-SLAM3's inertial essential graph) [UNVERIFIED at the
  source]. Soft 2-DoF priors plus an estimated gravity direction is the alternative used here; the 4-DoF graph is a
  future ablation.
- **Not run on EuRoC yet** (decision: the sim calibration and part 3 come first). A null result at σ 2.3° is
  plausible and would be recorded as such.

**Maintainer decisions (2026-10-19, after D71).**
1. **D73 may add an isolated inlier-export hunk to `VerifyResult`** (`edge_verifier.hpp`) so the degeneracy gate gets
   the PnP correspondences.
2. **The inter-agent `ProxyInformation` is replaced in 2.5** (D73) by the gate's covariance. The answer was "yes" to
   a "2.5 or Phase 3?" question; it is read as "in 2.5". This changes the Phase-1 sim_collab outputs that
   `--premerge` pins, so D73 re-pins them and lists the change.

## 9. Init acceptance gate — "the SE-Sync graphs that failed from dead reckoning" (decision 2026-10-18)

The design review's recommendation was requested: a physically logical gate, measured rather than assumed, without a
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
  the odometry-vs-chordal experiment on these benchmarks; cited in the algorithms.md entry (one fetch, its numbers checked
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
| D74 | certificate (chordal surrogate, grounded Cholesky with rotation-only η shift, relative η); κ/τ for cubicle/rim |
| D75 | aliasing outlier generator + ramp; breakpoint 90 % (size10) / 70 % (count2) for PCM → GNC |
| D76 | joint-graph robust entry point (PCM → chordal re-init → GNC); `GravityOptions::shared` |

Each adopted piece: algorithms.md entry (append-only hunk) → arch decision → ablation matrix → benchmark config
(`configs/slam/*`, isolated hunks) → manifest fields (`cpp/runtime/manifest/*`, isolated hunks). Every hunk to a shared
file is listed in the handover.

## 11. The robust stage on a joint multi-agent graph (D76) — handoff to main (3.2 / 3.5)

**Entry point.** `cpp/slam/robust/joint_robust.hpp`: `RobustJointSolve(PoseGraph&, anchor, JointRobustParams,
GncState*)` on a graph with JointId(agent, kf) nodes and `kInterAgentLoop` edges.
- **Order.** PCM, then a chordal re-init over the anchor's PCM-accepted component, then GNC-TLS.
- **Gating.** `GatedInterAgentEdge` runs the D71 gate when the broker has the geometry. Otherwise the gate stays at
  the verifying agent (collaboration.md §11).

**Wiring in `SolveMultiAgent`.** Call it in place of `SolvePoseGraph` after step 3 (assembly). Placement along
unfiltered edges stays harmless because of the re-init.

**Preconditions.**
- Edge information includes the loop noise model.
- `GncState` only for an append-only joint graph.
- GNC re-decides every loop, local loops included.
- PCM is per agent pair.

**Gravity.**
- Per-agent gravity variables (the default) decouple the agents' tilt.
- For a joint graph in G, set `GravityOptions::shared` (one physical gravity, g_ref in G). That is the yaw-only gauge,
  up to a global rotation; never fix gravity and the anchor's tilt together.
- `AgentSubgraph` does not carry gravity priors yet.

**Measured** (`tests/unit/test_joint_robust.cpp`).
- A 6-loop inter-agent aliasing group, placed first, is rejected on 5/5 seeds, with 10/10 true loops kept.
- Shared gravity cuts a 2° inter-agent tilt bias by 10 % (0.01 rad edges) to 68 % (0.05 rad edges). Per-agent
  gravity cuts none.
