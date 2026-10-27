# Local SLAM: windowed bundle adjustment, keyframe policy, landmark lifecycle, window exit

Plan 2.3 (I3): "keyframe policy (§8.3), landmark lifecycle (§11.2), analytic-Jacobian Schur LM windowed BA
(§11.3–11.6), window-exit policy (§11.7), covariance recovery, local pose graph." It replaces the 1.5 two-frame BA
stub inside `StereoVo`. The stub stays selectable as the ablation baseline (arch §0.2).

**Preflight (project rule):** literature (§7), OpenCV-native check (§6) and design review were done before
this document. Every threshold below cites its source.

## 1. Data in → data out

- **in, per frame:** LK tracks (ids + pixels) from 1.4, and at keyframes the left↔right matches and SGBM depth with
  σ_Z (§9.6).
- **out, per frame:** `T_W_C` and its covariance.
- **out, per keyframe:** the §4.2 Keyframe, now carrying `pose_cov` from the window solve and landmark ids.
- **out, at window exit:** an ODOMETRY edge (Z, joint covariance) into the local pose graph, which is optimised by the
  2.0 PGO solver (`pose-graph.md`).

## 2. State and parameterisation (one convention, derived here, tested)

- Window: the last **N = 10** keyframes. Config, ablated at 7 and 15. DSO uses N_f = 7 and reports "more than 7 …
  only marginal impact"; the catalogue suggests 10–15.
- Camera block: `T_W_C`, **left** perturbation `T ← exp(δ) T`, δ = [ρ; φ]. This is the PGO/`PoseNode` convention, so
  window exit needs no conversion.
- Landmark block: a Euclidean world point X (3). Inverse depth was considered; it is better for far points under
  monocular parallax, but our landmarks are stereo-triangulated close points (ORB-SLAM3 ThDepth: ≤ 60 baselines,
  1.5) and XYZ is what ORB-SLAM uses.
- **Gauge:** the oldest in-window keyframe is fixed (§11.3), and so is not a variable.

**Residual** for an observation of X in the left (or right) rectified camera of keyframe k:

```text
X_c = R_kᵀ (X − t_k)            (− (B, 0, 0) for the right camera)
r   = π(X_c) − z,   π(X) = (fx·x/z + cx, fy·y/z + cy)
```

**Jacobians** (left perturbation of `T_W_C`): `T⁻¹ ← T⁻¹ exp(−δ)`, so `X_c' ≈ Rᵀ(X − ρ − φ×X − t)`:

```text
∂X_c/∂ρ = −Rᵀ      ∂X_c/∂φ = Rᵀ [X]×      ∂X_c/∂X = Rᵀ
∂π/∂X_c = [[fx/z, 0, −fx·x/z²], [0, fy/z, −fy·y/z²]]
```

**Null-space test:** a common left δ on every pose, with every landmark moved by `X ← exp(δ) X`, leaves every
residual unchanged. The unanchored H therefore annihilates N, whose camera rows are I and landmark rows are
`[I, −[X]×]`. This is asserted, together with finite-difference tests at near-zero depth (z → Z_min) and wide-FOV
rays (|x/z| up to tan 60°), which is an exit-gate item.

## 3. The solve: OpenCV 5's LevMarqBase, Schur inside our backend (D51)

Same driver as the PGO. The backend keeps three things:

- `H_cc`: dense 6N_free × 6N_free (≤ 84 × 84);
- `H_cl`: 6×3 blocks;
- `H_ll`: block-diagonal 3×3.

The driver damps the **full** diagonal, cameras and landmarks alike, through `setDiag`. The damping is therefore
applied before elimination:

- `decompose()`: invert each damped `H_ll` block (pivot-guarded), form `S = H_cc − Σ H_cl H_ll⁻¹ H_lc`, then a dense
  LDLT with the positive-pivot guard;
- `solveDecomposed(right)`: solve `S δ_c = −r_c + Σ H_cl H_ll⁻¹ r_l`, back-substitute
  `δ_l = H_ll⁻¹(−r_l − H_lc δ_c)`, and return the **full** step. The driver's predicted-decrease and step-norm tests
  use the whole vector.

**Robust kernel:** Huber on the whitened residual s = ‖r‖/σ, δ = 2.0 (§11.3: 1.5–2.5). The probe energy is the
**robust** cost: `½s²` for s ≤ δ, otherwise `δ(s − ½δ)`. H and g are IRLS-weighted at the linearisation point, with
w = 1 or δ/s, so the gradient is exactly the robust cost's. σ is the per-observation pixel sigma (config).
Loop-closure robustness stays in the pose graph (§11.3, §14), not here.

**Budget:** `max_iterations` 10 per keyframe solve (§11.6 says 15; real-time setting, ablated). Wall time per solve
is recorded in results and the manifest.

## 4. Covariance and window exit (§11.6, §11.7 option (b), no FEJ marginalisation)

- `pose_cov` of the newest keyframe: the 6×6 block of S⁻¹ from `S X = E` (never an explicit inverse). This is the
  absolute covariance relative to the window's fixed oldest keyframe. It feeds §4.2, §8.3 and later §21.3.
- **Exit edge (corrected by design review).** The exiting keyframe k is the gauge, so its covariance is zero and no
  joint (k, k+1) block exists. With k fixed, the marginal Σ of k+1 *is* the uncertainty of the relative pose
  `Z = X_k⁻¹ X_{k+1}`. Transported to Z's left tangent: `Σ_Z = Ad(X_k⁻¹) Σ Ad(X_k⁻¹)ᵀ`. Then
  `OdometryEdge(k, k+1, Z, Σ_Z)` converts it to the residual tangent (1.5's tested path).
- **Stated cost of option (b):** consecutive exit edges share landmark information, and chaining them ignores those
  correlations, so the chain is overconfident. OKVIS2 makes the same criticism of ORB-SLAM3's fixing of old states
  ("not a conservative approximation"). It is measured, not assumed: the NEES of exit edges against ground truth
  should average ≈ 6, and is reported either way.
- Landmarks with fewer than 2 in-window observations after the exit are retired from the optimisation (§11.7).

## 5. Keyframe policy (§8.3) and landmark lifecycle (§11.2)

**Keyframe score** (§8.3, the DSO form `w_f f + w_ft f_t + w_a a > T_kf`, T_kf = 1):

```text
S = w_t·‖Δt‖/τ_t + w_r·‖Δφ‖/τ_r + w_p·parallax/τ_p + w_f·(1 − tracked/tracked_at_kf) + w_u·tr(Σ)/tr(Σ_kf) > 1
```

Hard triggers are kept from 1.5 (ORB-SLAM3 stereo `thRefRatio` 0.75; ORB-SLAM "≥ 50 tracked points"; a 1 s cap).
Every weight and τ is config.

**Landmarks:**

- **Creation:** §11.2's rejections, as in 1.5: parallax, depth interval, reprojection, a 3σ SGBM gate, relative σ.
- **Observation:** an LK track that survives across keyframes adds one observation at each of them, so a landmark is
  no longer two-view only.
- **Post-solve rejection:** an observation whose whitened residual exceeds χ²₂(0.95) = 5.991 is dropped, as ORB-SLAM
  drops outliers "at the middle and at the end of the optimisation".
- **Culling:** ORB-SLAM's test, adapted to stereo:
  - A new landmark must be found in > 25% of the frames where it is predicted visible.
  - Once 2 keyframes have passed, it must be observed from ≥ 2 keyframes. ORB-SLAM requires 3 *monocular*
    keyframes; a stereo landmark already has two rays at birth.
- **Re-association of lost landmarks: DEFERRED to 2.1**, where XFeat keyframe descriptors give a projection-and-
  descriptor search. 2.3 measures what LK-continuous multi-keyframe observation alone gives, so the 2.1 gain is
  attributable.

## 6. OpenCV-native and adjacent (checked against the installed build)

- **Used:**
  - `cv::detail::LevMarqBase` (the solve);
  - `cv::solvePnPRansac` (USAC) and `cv::triangulatePoints` (kept from 1.5);
  - `cv::solvePnPRefineLM` as the oracle for a one-free-pose window;
  - `cv::projectPoints` as a Jacobian cross-check. Its Jacobian is with respect to `rvec`/`tvec`, not our Lie
    perturbation, so the point Jacobian is compared directly and the pose Jacobian through the Rodrigues Jacobian or
    by finite differences.
- **Not available:** a BA routine. Contrib `sfm` is not in our build, and it wraps Ceres anyway. There is also no
  sparse solver; dense LDLT of S suffices at ≤ 84×84.

## 7. Literature (primary sources, read 2026-10-16)

- **ORB-SLAM** (T-RO 2015) §V-E keyframe decision, §VI-B map-point culling, §VI-E keyframe culling (90% seen by ≥ 3
  others).
- **ORB-SLAM3** Table II (stereo and stereo-inertial rows).
- **DSO** (arXiv 1607.02565) §3.1 window N_f = 7, keyframe criteria and marginalisation rules.
- **OKVIS2** (arXiv 2202.09199): the fixing-vs-marginalising critique.
- **Considered and not needed at ≤ 15 keyframes in double precision:**
  - √BA and √-marginalisation (Demmel et al., CVPR/ICCV 2021): algebraically equivalent to Schur; the gain is at
    large scale or in single precision.
  - Power BA (Weber et al., CVPR 2023): series inversion of S for very large problems.
- **MAC-VO** (2025): covariance-aware keypoint selection and weighting. It supports σ-weighted residuals now and the
  §7.3 selection in 2.1.

## 8. Target, fixed before measuring

2.3 is stereo VO without IMU (IMU arrives in 2.4).

**Reference class: ORB-SLAM3 Table II stereo rows, RMS ATE in metres.**

| system | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ORB-SLAM2 (SLAM) | 0.035 | 0.018 | 0.028 | 0.119 | 0.060 | 0.035 | 0.020 | 0.048 | 0.037 | 0.035 | – | 0.044* |
| VINS-Fusion stereo | 0.540 | 0.460 | 0.330 | 0.780 | 0.500 | 0.550 | 0.230 | – | 0.230 | 0.200 | – | 0.424* |
| SVO | 0.040 | 0.070 | 0.270 | 0.170 | 0.120 | 0.040 | 0.040 | 0.070 | 0.050 | 0.090 | 0.790 | 0.159 |
| ORB-SLAM3 (SLAM) | 0.029 | 0.019 | 0.024 | 0.085 | 0.052 | 0.035 | 0.025 | 0.061 | 0.041 | 0.028 | 0.521 | 0.084 |

**2.3 passes when:**

- (1) on every available sequence, its ATE is **better than the 1.5 placeholder's**. The placeholder is measured
  first on all 11 sequences.
- (2) its per-sequence ATE is ≤ the worst of this class.

The protocol is that of `local-slam.md#estimator-plan` §3: full trajectory, SE(3) alignment, processed ground truth, evo as the
oracle.

**Stated with the numbers:** ORB-SLAM2/3 are full SLAM systems with loop closure, while 2.3 is VO. The stereo-inertial
target of the exit gate belongs to 2.4.

## 9. Config, manifest, contracts, gate

- `configs/slam/*_v2.yaml` select `backend: window`. The `_v1` configs keep `backend: two_frame_stub`, so the
  pre-merge gate's placeholder half stays byte-identical.
- The manifest gains `slam_backend`, `window_size`, `ba_max_iterations` and `huber_delta`.
- `contracts.yaml`'s unset list loses `uncertainty`, as the keyframe uncertainty summary gets filled. The gate check
  is updated in the same change.

## 10. As built and measured (2026-10-16/17; D53, D54)

**What the window path does** (`StereoVo::AddKeyframeWindow`, configs `*_v2.yaml`):

- **Solve.** The last 10 keyframes on OpenCV's `LevMarqBase`, with the oldest fixed.
- **Dead-track landmarks** stay while an in-window keyframe observes them.
- **Observability guards** (`window_ba.hpp`; added after crashes on MH_02, MH_05, MH_04, V2_0x):
  - A landmark enters only if its own unweighted information gives a worst-direction std ≤ `max_relative_sigma` ×
    range: the §9.5 triangulation test, generalised to N rays.
  - Every free keyframe must reach a fixed one through ≥ `pnp.min_correspondences` shared landmarks (a BFS over
    covisibility). The oldest keyframe of an unreached group is held.
  - Both tests run again at the solution before the marginals are taken.
  - A remaining **joint** degeneracy gives an uninformative marginal and is counted (`marginal_degenerate`). Example:
    V1_03, whose null vector is a common scale of two keyframes' translations.
  - Factor failures are never swallowed: `WindowBaCovariances` still throws.
- **Hygiene.**
  - χ²₂(0.95) post-solve rejection.
  - Culling, ORB-SLAM's two tests: a landmark whose LK track is alive in this frame is never culled (test 1, "being
    found"); otherwise it needs ≥ 2 in-window keyframes once 2 keyframes have passed. Without the exemption, V2_02
    doubled its lost frames.
- **σ = 0.46 px, measured, not tuned.** It is the dof-corrected inlier σ, Σ‖r‖²/(m − p) over χ² inliers, from MH_01 at
  σ = 1. Re-measured after D54: 0.453. A median of post-fit residuals would read low, because of the lost degrees of
  freedom.
- **Covariance.** The newest node gets its anchor's absolute covariance plus its marginal. The exit edge uses window[1]'s
  marginal only when window[0] is its anchor; otherwise it is uninformative.

**Stated deviations:**

- Inner-node covariances are not recomputed after each re-solve.
- The final N−1 keyframes get no exit edges at run end.
- Solves that hit the 10-iteration cap (`found = false`) keep their progress, because only a factor failure counts
  as a failure. The share is 23% at σ = 1 before D54 and **55.6% at σ = 0.46 after D54** (4630 of 8332 solves, 11
  flights). §11.6 specifies 15 iterations: the cap is an open item.
- `sim_v2.yaml` keeps σ = 1.0: σ is calibrated on EuRoC only, not yet on the sim.
- Row refinement is applied even when the estimate is not significant (MH_01: a = 0.17 px). A significance threshold
  is a follow-up.
- The scored trajectory is the causal per-frame one; ORB-SLAM re-composes frames on refined keyframes. **Closed by
  D57:** the keyframe-refined score (`results.yaml` `keyframe_refined`) differs by ≤ 1 mm on all 11 flights.

**Result against §8** (after D54; ATE in m):

| | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 (1.5 placeholder) | 0.112 | 0.128 | 0.151 | 1.60 | 0.257 | 0.162 | 0.226 | 1.74 | 0.105 | 0.213 | 1.65 |
| v2 (2.3) | **0.075** | **0.075** | **0.093** | 2.78 | 0.560 | **0.078** | **0.113** | 0.439 | **0.083** | **0.124** | 2.12 |
| worst of the class | 0.54 | 0.46 | 0.33 | 0.78 | 0.50 | 0.55 | 0.23 | 0.07 | 0.23 | 0.20 | 0.79 |
| lost frames (v2) | 0 | 16 | 0 | 278 | 107 | 4 | 14 | 146 | 20 | 17 | 183 |

- (1) **8/11 pass**: all except MH_04, MH_05 and V2_03.
- (2) **7/11 pass**: all except MH_04, MH_05, V1_03 and V2_03.
- **Every failure loses ≥ 100 frames.** There, 2.3 is worse than v1, because lost stretches put landmarks at coasted
  poses and the window carries them across 10 keyframes (`local-slam.md#window-audit` §4). The fix is upstream of 2.3: lost-frame
  handling next, then 2.1 / 2.4.
- **After D56** (lost-frame handling, `local-slam.md#lost-frames`): v2 reads MH04 0.364, MH05 0.349, V103 0.562, V203 1.266,
  and the rest within ±0.03 except V102 0.137. (1) **10/11**; (2) **9/11**.
- **Before D54** the same table read 4/11 on both criteria. The frontend defects, not the window, were most of the
  gap (AUDIT §1).

## 11. Status (2026-10-21, on `main` 3cf5f0c; premerge PASS)

Every item of the plan 2.3 line is built:
- keyframe policy (§5);
- landmark lifecycle (§5);
- the analytic-Jacobian Schur LM window BA on `LevMarqBase` (§3, D51/D52);
- window exit by drop plus odometry edge, no FEJ marginalisation (§4);
- covariance recovery (§4);
- the local pose graph.

The audit (`local-slam.md#window-audit`, D53/D54) fixed the map-point defects, and the re-baseline is pinned (D55). Later EuRoC
accuracy work moved to 2.4 (lost-frame handling D56, VIO D58/D59) and 2.1 (re-association D60). No 2.3-specific item
is open.

---

<a id="window-audit"></a>

# Window bundle adjustment audit

**Question:** audit 2.3 against the state of the art. Can we do better with the current features, or must
feature detection improve first? And is frame dropping benign, or adversarial further down the line?

**Answer, in one paragraph.** The hard EuRoC flights are not limited by feature quality or by the 2.3 backend. They
are limited by **three defects in how tracked points become 3-D map points**. Each was measured and attributed, and
each is fixable with the current features:

1. A stereo row gate too strict for a calibration error on the V2 recordings. Widening it alone takes v1 on V2_01
   from 764 lost frames and 2.36 m to 16 and **0.082 m**, and V2_02 from 914 and 2.47 m to 16 and **0.176 m**.
   *Proven by intervention.*
2. A far-point cap that discards what ORB-SLAM keeps. Raising it from 60 to 90 baselines takes v1 on MH_04 from 411
   lost frames and 3.93 m to 30 and **1.04 m**. *Proven by intervention.*
3. A brightness-constancy stereo match that the V2_03 exposure mismatch breaks. *Measured (57% of stereo matches lost,
   right camera 36% darker), not yet tested by intervention.*

Learned features are worth ≈ 5% on EuRoC (one 2026 preprint, unverified by us). They matter later, for Ariadne's dark
domain, not now.

Lost frames are **adversarial**, not benign. While tracking is lost, the system keeps inserting keyframes and creating
map points at a pose it is only guessing. When tracking resumes, it locks onto those misplaced points, so the error
becomes permanent and looks confident.

Every defect is in the Phase-1 frontend (1.4/1.5). So fixing them changes the v1 numbers, and the 2.3 verdict must be
re-measured afterwards.

---

**State of the tree for this audit:** the pre-merge gate passes on this working tree, 54/54 with ctest 286/286
(`build/premerge/aadffb8df5790ca0-20261016T173330Z`). That covers the USAC redraw, the frozen coasting velocity and
the PSD check, all byte-identical against Phase 1.

### 1. What was measured (all from this tree, `build/diag`, 2026-10-16)

The 11 flights, v1 (the 1.5 placeholder) and v2 (2.3), as in `local-slam.md`:

| flight | v1 ATE | v2 ATE | lost frames (v1) | stereo matches accepted | dominant cause, measured |
|---|---|---|---|---|---|
| MH_01, MH_02, MH_03 | 0.08–0.14 | 0.07–0.15 | 0–13 | 66% | none |
| MH_04, MH_05 | 0.8–3.9 | 1.0–3.1 | 370–411 | ≈ 50% | far points discarded (36–38% of candidates); MH_04 proven by intervention, MH_05 by the same signature |
| V1_01, V1_02 | 0.15–0.22 | 0.11–0.14 | 3–14 | ≈ 75% | none |
| V1_03 | 1.18 | 0.39 | 171 | 44% | fast motion: more LK and forward-backward rejects (11% / 26%) |
| V2_01, V2_02 | 2.4–2.5 | 3.5–4.3 | 764–914 | **3–4%** | **stereo row gate vs a 1.7 px vertical calibration error** |
| V2_03 | 3.7 | 5.4 | 1615 | 2% | row gate, **plus** left/right exposure mismatch (right 36% darker), **plus 414 dropped input frames** |

ATE values are in metres.

Measurements behind the "dominant cause" column:

- **Stereo reject breakdown.** A replica of `stereo_lk_matcher.cpp` (same OpenCV calls and parameters, Python,
  pinned OpenCV 5) on 30 rectified pairs per flight gives:

  | flight | accepted | wrong row | LK lost | forward-backward |
  |---|---|---|---|---|
  | MH_01 | 71% | 3% | 8% | 18% |
  | V2_01 | 2% | **76%** | 6% | 17% |
  | V2_02 | 3% | **64%** | 8% | 25% |
  | V2_03 | 1% | 20% | **57%** | 22% |

- **Rectification residual.** The row offset of matches in rectified images is 0.21 px (MH_01) and 0.14 px (V1_01),
  but **1.70 px on V2_01**. The frontend tool's own `rect_check_lk` agrees. A fit
  `dy ≈ −1.1…−1.4 px − (1.0…1.25 px)·(y / H)` has a residual spread of only 0.16 px. So this is a clean geometric
  error that grows from the top of the image to the bottom (a small relative pitch, or a vertical-scale mismatch),
  not noise. The calibration files are byte-identical across all 11 flights. The V2 recordings simply don't match
  them as well. *Which physical cause — mounting shift, or a calibration taken for an earlier session — is
  [UNVERIFIED].*
- **The experiments that prove it** (each changes one setting, single runs):

  | run | lost frames | ATE | stereo matches accepted |
  |---|---|---|---|
  | v1 V2_01, row tolerance 1.0 → 2.5 px | 764 → 16 | 2.364 → **0.082 m** | 3% → 45% |
  | v1 V2_02, row tolerance 1.0 → 2.5 px | 914 → 16 | 2.467 → **0.176 m** | 4% → 61% |
  | v1 MH_04, far-point cap 60 → 90 baselines | 411 → 30 | 3.927 → **1.036 m** | — |
  | v2 V2_01, row tolerance 1.0 → 2.5 px | 819 → 16 | 3.498 → **0.101 m** (v1: 0.082) | 3% → 46% |

  The last row is why loosening is not the fix for v2: see §2, fix 1.
- **The replica and the VO disagree on MH_04's stereo rate** (74% against the VO's ≈ 50%). The VO's count is
  dominated by keyframes forced during lost stretches. The far-point intervention removes those stretches, after which
  the VO's accepted rate rises to 46% of a much smaller candidate set.
- **Brightness.** The cam1/cam0 ratio of mean intensity, median per flight, is 0.99–1.03 everywhere except
  **V2_03 = 0.64** (p10 0.39). EuRoC's own page lists this as a known issue: *"automatic exposure control
  independently per camera, leading to varying brightness between the stereo pair."*
- **Dropped frames.** All flights run at a steady 50 ms except **V2_03**, which has 414 gaps of 100 ms (≈ 18% of its
  frames missing).

### 2. Can we do better with the current features? Yes — and that comes first

- **The comparator does it with the same kind of features.** VINS-Fusion stereo, plain KLT (`feature_tracker.cpp`,
  read 2026-10-16), gets 0.23 / 0.20 m on V2_01 / V2_02. Its left→right match is forward LK + backward check
  (≤ 0.5 px) + image border, **with no row constraint**. That is why the V2 calibration error doesn't break it.
  Our row gate was a good outlier filter on MH_01 (D35), but it is brittle against a 1–2 px rectification error.
- **ORB-SLAM2 keeps far points** (§III-A): points beyond 40 baselines are "far", give accurate rotation, and are
  triangulated "when they are supported by multiple views". Our `max_depth_baselines: 60` *discards* them, and
  its config comment cites ORB-SLAM's ThDepth. That is a misreading: ThDepth classifies points, it doesn't reject
  them. In the machine hall it cut 36–38% of candidate points.
- **Learned features help little on EuRoC.** DL-VINS-Factory (arXiv 2607.01757, July 2026, a preprint whose numbers
  we have not reproduced) reports learned front-ends "not universally superior to classical tracking": ALIKED +
  LightGlue −5% ATE against GFTT + LK, and XFeat + LightGlue poor in stereo-inertial mode. It also calls EuRoC
  "largely solved under well-lit conditions".
- **Caveat that keeps 2.1 alive.** EuRoC is not Ariadne's domain. In dark, dusty, lamp-lit scenes, learned and
  illumination-robust frontends do matter (AirSLAM, arXiv 2408.03520; Light-SLAM). XFeat (2.1) is the right
  investment *after* the association defects are fixed, judged on the sim + DVE degradations rather than on
  EuRoC.

**Fixes, all with current features, in order of measured payoff:**

1. **Stereo association that tolerates a small rectification error** — two candidates:
   (a) VINS-style: drop the row gate and keep forward-backward + positive disparity;
   (b) estimate the row offset online from the matches (it is a clean affine function of y) and gate around it.
   The offset also biases **every right-camera residual inside BA**, and the experiment shows it. With only the gate
   widened, v2 on V2_01 fails 8.2% of residuals on the χ² test, against 1.5–2.1% on clean flights at the same
   σ = 0.46 px, whose gate (≈ 1.13 px) is below the 1.7 px bias. And v2 ends worse than v1: 0.101 against 0.082 m.
   SGBM depth also assumes aligned rows, and loosening the gate does nothing for it. So (b), which corrects the
   geometry, is needed. (a), which only stops rejecting, is not enough for v2.
   - *Disclosure:* (b) estimates a correction from the evaluated flight itself. It must be reported as online
     calibration refinement, with the estimated values in the manifest.
2. **Keep far points:** monocular observations now, multi-view triangulation later (ORB-SLAM2 §III-A); remove the
   discard.
3. **Exposure-robust stereo match:** per-pair gain/bias compensation before LK, as RTAB-Map does (RTAB-Map EuRoC
   evaluation) and Trifo-VIO does with histogram matching. OpenCV-native: `cv::equalizeHist` / CLAHE, or a 2-number
   gain/bias fit.
4. **Make the motion model and metrics use time, not frame count:** coasting velocity per second;
   `max_frames_between`; RPE (`ComputeRPE` indexes by sample, so its "1 s" window spans up to 2 s on V2_03, ≈ 1.2 s on
   average).

### 3. The 2.3 backend against the literature

| aspect | 2.3 as built | literature | verdict |
|---|---|---|---|
| window | last 10 keyframes, oldest fixed | VINS 10 + marginalisation; DSO 7; ORB-SLAM local BA over the **covisibility** set, with keyframes that see the points held fixed | fine for VO; covisibility windows are the upgrade for revisits (2.1 re-association) |
| window exit | drop + odometry edge, no marginalisation (locked) | OKVIS2: fixing is "not a conservative approximation" | accepted cost; NEES check still owed |
| robust cost | Huber + χ²₂(0.95) gate, σ measured (0.46 px) | ORB-SLAM: Huber + χ² at the middle and end of the optimisation | matches |
| map hygiene | χ² rejection + "seen by ≥ 2 keyframes" culling; live LK tracks exempt | ORB-SLAM: "found in > 25% of the frames where predicted" + 3 keyframes | matches since the exemption |
| map-point creation | stereo, close points only, at keyframes | ORB-SLAM2: close **and** far, plus multi-view triangulation | **gap (§2 fix 2)** |
| behaviour when lost | keyframe every lost frame, new points at the coasted pose | ORB-SLAM3 Atlas: stop, relocalise or start a **new map**, merge later; VINS: IMU bridges | **gap (§4)** |
| reported trajectory | causal per-frame poses | ORB-SLAM: frames stored relative to their keyframe and re-composed after refinement | protocol caveat on criterion (2): our numbers are handicapped against Table II |

2.3 itself is sound. Where both run cleanly it beats v1 by 10–35% (MH_01, MH_02, V1_01, V1_02). The failures sit
upstream of it.

### 4. Frame dropping: benign or adversarial?

**Adversarial**, in the specific way the current code handles loss. Two kinds of "dropped frame" exist:

- **Input frames that never arrive.** These exist in V2_03 (414 gaps) and will exist under Phase 5 degradation and
  transport loss. They are benign if the system reasons in time rather than frame counts (fix 4). Today the
  coasting velocity, the 20-frame keyframe cap and RPE all count frames.
- **Frames the frontend can't localise (lost).** These are the dangerous ones. While lost:
  - **Map poisoning.** Every lost frame is forced to become a keyframe (V2_02 v1: 2325 keyframes for 2348 frames),
    and it triangulates new points at the *coasted* pose, which is wrong. When tracking resumes, PnP locks onto
    those wrong points: the error is frozen into the map and looks confident. The uncertainty attached to the
    lost keyframe *is* uninformative, but its map points carry no such mark.
  - **Downstream.** The keyframe flood reaches the Phase 3 inter-agent exchange: bytes are counted, and these
    keyframes are worth nothing. Phase 4 trains splats on mis-posed images. Phase 6 plans routes through a
    corrupted map, which is the safety-relevant one. Downstream code currently **cannot tell** a lost-segment
    keyframe from a good one, except by its covariance. Checked: `runtime/agent/keyframe_records.hpp` has no
    tracking-status field.
  - **In Ariadne's domain this is the central failure mode, not an edge case.** Darkness, dust and lamp failure
    cause loss, and they cluster in exactly the places that matter.

**Mitigations from the literature:**

- Do not insert keyframes or points while lost.
- On loss, start a new map segment (ORB-SLAM3 Atlas), joined later by place recognition. That is Phase 3's loop
  machinery.
- Tag lost-segment keyframes in the keyframe record so that collaboration, reconstruction and planning can weight
  or exclude them.
- Bridge short gaps with the IMU (2.4: preintegration gives a metric prediction over seconds, where constant
  velocity diverges).

### 5. Decision needed

The defects are in the 1.x frontend, so this is a sequencing question.

- **(A) Fix now, re-baseline.** Land fixes 1–4 as documented Phase-1 fixes: v1 numbers change, and the 2.3 verdict
  is re-measured. Then the lost-handling of §4, then 2.1. *Recommended:* the payoff is measured (V2_01
  2.36 → 0.08 m, V2_02 2.47 → 0.18 m, MH_04 3.93 → 1.04 m), and the 2.3 table without it measures the frontend,
  not 2.3.
  - **Gate consequence:** the sim's Phase-1 output changes **by design**, since stereo matches the row gate used to
    reject get accepted. So the pre-merge byte-identity check fails on purpose and needs a D-entry and new pinned
    evidence, from a full gate run with a fresh render.
- **(B) Fix behind v2 configs only.** v1 stays the frozen Phase-1 baseline, the fixes ship as part of 2.3, and the
  gate stays byte-identical. Cleaner gate history, but the "beats v1" criterion then credits frontend fixes to the
  backend.

Either way: none of the fixes needs new features, and the frame-loss handling (§4) belongs in the same change set,
because it is what turns a bad stretch into a permanent error.

### Sources

- EuRoC MAV dataset page, "Known issues": <https://projects.asl.ethz.ch/datasets/euroc-mav/>
- Mur-Artal & Tardós, ORB-SLAM2 (T-RO 2017), §III-A and §III-E: <https://arxiv.org/abs/1610.06475>
- VINS-Fusion `feature_tracker.cpp` (read 2026-10-16):
  <https://github.com/HKUST-Aerial-Robotics/VINS-Fusion/blob/master/vins_estimator/src/featureTracker/feature_tracker.cpp>
- Lim et al., DL-VINS-Factory (arXiv 2607.01757, preprint): <https://arxiv.org/abs/2607.01757>
- Campos et al., ORB-SLAM3 (multi-map Atlas): <https://arxiv.org/abs/2007.11898>; Elvira et al., ORBSLAM-Atlas:
  <https://arxiv.org/abs/1908.11585>
- RTAB-Map (exposure compensation on EuRoC): <https://arxiv.org/abs/2403.06341>
- Trifo-VIO (histogram matching, V2_03): <https://arxiv.org/abs/1803.02403>
- DS-VIO (KLT stereo failure on V2_03): <https://arxiv.org/abs/1905.00684>
- AirSLAM (illumination-robust learned frontend): <https://arxiv.org/abs/2408.03520>

### 6. Outcome (2026-10-17, (A) chosen; D54)

All four fixes landed in the Phase-1 frontend:

- row refinement, with samples pooled across keyframes;
- far points kept;
- exposure match, which landed on evidence: V2_03 v1 lost 1026 → 192;
- time-based prediction, keyframe cap and RPE.

Re-baselined v1 / v2 ATE (m):

| flight | v1 | v2 |
|---|---|---|
| MH01 | 0.112 | 0.075 |
| MH02 | 0.128 | 0.075 |
| MH03 | 0.151 | 0.093 |
| MH04 | 1.60 | 2.78 |
| MH05 | 0.257 | 0.560 |
| V101 | 0.162 | 0.078 |
| V102 | 0.226 | 0.113 |
| V103 | 1.74 | 0.439 |
| V201 | 0.105 | 0.083 |
| V202 | 0.213 | 0.124 |
| V203 | 1.65 | 2.12 |

- **2.3 criterion (1): 8/11, criterion (2): 7/11.** Before D54 it was 6/11 on (1), 5/11 on (2) and 4/11 on both.
  Criterion (1) now compares against a *different* v1, so the before/after counts on (1) are not like for like.
- **D54 made v1 worse on 5 of the 6 flights that were already clean:**
  - clearly: MH_02 0.081 → 0.128, V1_03 1.18 → 1.74;
  - slightly: MH_03, V1_01 and V1_02.
  - **[UNATTRIBUTED]** The likely cause is that keeping far points feeds higher-σ landmarks to the two-frame stub,
    which has no window to average them. Not tested.
- The remaining failures all still lose ≥ 100 frames, and §4 is now the binding constraint.
- **Row-refinement acceptance:** the residual after refinement, re-estimated on a later keyframe, is ≤ 0.38 px on
  10 flights (V1_03 0.38, the rest ≤ 0.25) and 0.62 px on V2_02, against 1.46–1.77 px before on the V2 flights.

---

<a id="lost-frames"></a>

# Lost-frame handling

`local-slam.md#window-audit` §4 found lost frames **adversarial**: a lost stretch puts keyframes and landmarks at a predicted
pose, and downstream code cannot tell. It also found that every remaining EuRoC failure loses at least 100 frames.
This design changes what the VO does while PnP cannot localise. It does not change how often that happens: 2.1
(re-association) and 2.4 (IMU) attack the loss rate.

**Preflight (project rule):**
- Literature (§2).
- OpenCV-native check against the installed 5.0.0 headers: `findHomography` / `findEssentialMat` `UsacParams`
  overloads, and `recoverPose` with `distanceThresh`, all in `opencv2/geometry/3d.hpp`.
- Design review before the design and again before the sweep.

### 1. What a lost stretch looks like (measured, `vo_trace.csv`)

`vo_trace.csv` is a new per-frame trace written next to every run's `results.yaml`. On the four hard flights it
shows two regimes:

| regime | example | true motion | 2-D tracks | stereo matches | D54 coasting error |
|---|---|---|---|---|---|
| dark hover | MH_04 frames 214–420, 10.4 s | 0.09 m, 7.5° | ~364 | ~23 per keyframe | 7.56 m, 117° |
| blur burst | V2_03, 15 stretches of 0.3–1.3 s | 0.3–1.2 m, 17–38° | 35–180 | 7–28 | up to 1.4 m, 75° |

- **The 2-D tracks survive.** Only the landmarks fail: the right image is too dark to match, or LK loses the
  landmark tracks under blur.
- **Coasting at a frozen constant velocity** extrapolates a hover 7.6 m away. On short bursts it is often worse than
  holding still: in V2_03 frames 1200–1208 it was off by 72° against a true 27°.
- **What is lost first is rotation**, and rotation is observable from 2-D tracks alone.

### 2. Literature

| source | what it does while lost |
|---|---|
| ORB-SLAM3, `Tracking.cc` (stereo, no IMU) | RECENTLY_LOST: relocalise with DBoW2 for 3 s and **make no keyframes**. Then LOST: a new map in the Atlas, or a reset below 10 keyframes. Maps merge on revisit (Atlas, Elvira et al. IROS 2019). |
| ORB-SLAM3 (with IMU) | RECENTLY_LOST: predict from the IMU for 5 s; optionally insert keyframes. |
| VINS-Mono (T-RO 2018) | Failure detection (few features; jumps in position, rotation or bias) → re-initialise → a **new pose-graph segment**, joined by loop detection. |
| ORB-SLAM initialiser (T-RO 2015 §IV-B) | Chooses H or F by `R_H = S_H / (S_H + S_F) > 0.45`, from χ²-truncated transfer-error scores. |
| ORB-SLAM `TrackReferenceKeyFrame` | Tracks against a reference keyframe rather than chaining frame to frame. |
| SP-VIO (arXiv 2411.07551) | Smooths the gap once tracking returns (RTS backtracking). Needs an inertial model: a 2.4 option. |

Ariadne has no relocalisation (2.1) and no IMU (2.4) yet. What remains measurable is the rotation from the tracks.

### 3. Design

#### Pose while lost (`lost.motion`)

**`bridge` (v2)** combines two parts:
- **Rotation, measured** from the 2-D tracks. `EstimateFrameRotation` (`rotation_bridge.{hpp,cpp}`) runs
  seeded USAC H and E on normalised coordinates.
  - It selects between them by ORB-SLAM's published score rule.
  - The two views are the **anchor** and the current frame. The anchor is the last measured frame, or, once fewer
    than 20 tracks reach back to it, the previous frame.
  - The error at any frame is therefore ONE two-view estimate whose baseline grows, not a chain.
- **Translation, predicted:** the last measured velocity, clamped to 3 m/s, decaying as `exp(−t/τ)` with
  τ = 0.5 s. A hover settles instead of flying off.
- If no model is plausible, the same decaying angular velocity takes over.

**`coast`** is the 1.5 behaviour (frozen constant velocity). It stays as the baseline, and v1 uses it.

#### Keyframes while lost (`lost.reanchor`)

- **`every_frame` (v2, and 1.5):** every lost frame is a keyframe.
- **`gated`:** a keyframe only if tracked plus to-be-triangulated landmarks ≥ 30, which is 2 × the PnP minimum.
  Otherwise none, except at the time cap. It follows ORB-SLAM3's no-keyframes-while-RECENTLY_LOST.
  - **Rejected on evidence (§5):** it raises lost frames.

#### Provenance

- `PoseSource {measured, bridged, coasted}` and a **segment** counter (incremented at every loss onset) are on every
  `VoTrackResult`.
- Both are carried into `runtime::Keyframe` as `pose_source` and `segment` (AUDIT §4: downstream must be able to tell
  a predicted keyframe from a measured one).
- They also appear in `vo_trace.csv`, in `results.yaml` `lost:` (only when a `lost:` section is configured, so v1
  bytes are unchanged), and in the manifest field `lost_handling`.

**The seam for 2.4:** the rotation source while lost is one call. The preintegrated gyro replaces it, and the
translation prior becomes the preintegrated velocity.

### 4. Model selection: what was tried and why it changed

Each step was driven by a failing synthetic test or a measurement.

1. **Prototype** (Python, fresh corners per pair, the 15 gaps; **in-sample**):
   - E alone flips by ~180° in a hover (MH_04 128.9°, MH_05 179.9°).
   - H alone is robust but biased under translation.
2. **ORB-SLAM's ratio on inlier COUNTS, chained frame to frame:**
   - On a synthetic forward motion of 1.75 cm per frame, H explained 295/300 tracks within 1.5 px and won.
   - Its rotation was biased by 0.1° per frame, 40% of the true rotation; E's error was 0.05° and unbiased.
   - Counts cannot see sub-pixel parallax. Over 1.5 s: 2.7°.
3. **"E unless it disagrees with H by > 5°":** fixed the per-frame case.
4. **Anchor instead of chaining:**
   - The baseline grows, so H's bias grows (20°), and rule 3 then handed frames to the wrong model: 20° jumps.
   - `recoverPose`'s default overload also counted only points within 50 baselines, i.e. none at 1–2 cm per frame.
     E was silently dropped. Fixed with `distanceThresh`.
   - ORB-SLAM's ReconstructF cheirality test (≥ 90% in front) cannot be applied through `recoverPose` for that
     reason.
5. **Final: anchor + ORB-SLAM's score rule as published** (S_H, S_F; Γ = T_H = 5.991, T_F = 3.841, σ = 1 px;
   R_H > 0.45 → H).
   - At small parallax H wins, and its bias is small in absolute terms and no longer accumulates.
   - At real parallax E wins.
   - Synthetic stretch: 0.01–0.06° per frame, against the anchor.
   - All 16 VO and bridge tests pass.

### 5. Results (all 11 EuRoC flights; ATE in m; RMS of the causal per-frame trajectory)

**Held-out per-pair accuracy.**
- The frame-to-frame bridge was run on every TRACKED frame, against ground truth, on all 11 flights. No lost
  frame is included.
- Over 25,606 pairs:

  | model | share | median | p90 | p99 | > 5° |
  |---|---|---|---|---|---|
  | H | 69% | 0.21° | 0.53° | 1.2° | 0% |
  | E | 31% | 0.036° | 0.17° | 1.1° | 0.01% |

- Robust σ per camera axis: 0.09°, 0.09°, 0.04° per pair. **This is the FRAME-TO-FRAME estimator**; lost frames use
  the anchor estimate, validated only on the synthetic test and the 15 in-sample gaps. An edge covariance needs the
  anchor mode measured; edges stay uninformative for now.

**Ablation matrix.** v1_D54 / v2_D54 = the D54 sweep. Every other column has a `lost:` section.

| flight | v1 D54 | v1 bridge | v2 D54 | v2 coast·gated | v2 bridge·gated | **v2 bridge** | v2 bridge, no decay | v2 bridge, τ 0.2 | class worst |
|---|---|---|---|---|---|---|---|---|---|
| MH01 | 0.112 | 0.112 | 0.075 | 0.075 | 0.075 | **0.075** | 0.075 | 0.075 | 0.54 |
| MH02 | 0.128 | 0.089 | 0.075 | 0.295 | 0.244 | **0.070** | 0.080 | 0.090 | 0.46 |
| MH03 | 0.151 | 0.151 | 0.093 | 0.093 | 0.093 | **0.093** | 0.093 | 0.093 | 0.33 |
| MH04 | 1.598 | 0.826 | 2.777 | 2.768 | 1.074 | **0.364** | 2.343 | 0.403 | 0.78 |
| MH05 | 0.257 | 0.226 | 0.560 | 3.176 | 0.399 | **0.349** | 0.340 | 0.353 | 0.50 |
| V101 | 0.162 | 0.160 | 0.078 | 0.078 | 0.070 | **0.070** | 0.078 | 0.071 | 0.55 |
| V102 | 0.226 | 0.219 | 0.113 | 0.109 | 0.138 | **0.137** | 0.143 | 0.135 | 0.23 |
| V103 | 1.735 | 1.667 | 0.439 | 0.442 | 0.491 | **0.562** | 0.463 | 0.700 | 0.07 |
| V201 | 0.105 | 0.097 | 0.083 | 0.196 | 0.072 | **0.071** | 0.078 | 0.068 | 0.23 |
| V202 | 0.213 | 0.211 | 0.124 | 0.116 | 0.117 | **0.120** | 0.117 | 0.120 | 0.20 |
| V203 | 1.650 | 1.126 | 2.118 | 1.918 | 1.104 | **1.266** | 1.350 | 1.106 | 0.79 |

"v2 bridge" is `bridge` / `every_frame` with τ = 0.5 s, the v2 configuration.

- **`coast` / `every_frame` is the 1.5 behaviour exactly.** Its 11 `results.yaml` match D54 except the wall-clock
  `timing_ms` line.
- **The bridge is the win.** MH_04 2.78 → 0.36 (rotation RMSE 55° → 1.8°). V2_03 2.12 → 1.27. MH_05 0.56 → 0.35.
  - Over the 15 D54 gaps (in-sample), the summed error is 431° / 15.9 m → 111° / 4.4 m.
- **Both halves matter.**
  - Rotation alone (no decay) fixes MH_05 and V2_03 but leaves MH_04 at 2.34.
  - The decay fixes the hover.
- **Costs, reported:**
  - The decay hurts V1_03, a fast flight with many short losses: 0.46 with no decay → 0.56 (τ 0.5) → 0.70 (τ 0.2).
  - V1_02 0.113 → 0.137 under every bridge variant.
  - τ = 0.5 (and the 3 m/s clamp) were set after the in-sample 15-gap prototype but before this sweep, and are kept; τ is not tuned on the sweep.
- **`gated` is rejected:** lost frames rise (V2_01: 20 → 80 coasting, 120 bridged; MH_02: 16 → 38), and ATE with
  them (MH_05 3.18 with coasting).
  - Without landmarks from forced keyframes, recovery waits, and nothing re-associates (2.1).
  - [INFERENCE, from gated raising the losses; not measured directly] once the bridge makes predicted poses good,
    the keyframe "flood" is how tracking recovers, not what poisons the map. It is now tagged (`pose_source = bridged`).
- **Lost-frame counts barely move** (MH_04 278 → 264). This design limits the damage of a loss; 2.1 and 2.4 must
  reduce the losses.

**Against `local-slam.md` §8:**
- (2) within the worst of the published stereo class: **9/11**, up from 7/11. Still out: V1_03 0.56 vs 0.07, V2_03
  1.27 vs 0.79.
- (1) better than the 1.5 placeholder: **10/11** (MH_05 0.35 vs 0.26).
- With the same lost handling given to v1, v2 beats it on 9/11 (MH_05, V2_03 lose).

### 6. Open

- Bridged odometry edges keep the uninformative covariance (1 m, 1 rad). The measured per-pair σ (§5) is the input
  for an informative one, which needs a NEES check first.
- The three worst blur gaps (V2_03 frames 1200, 1241, 1775: 23–26°) have too few tracks back to any anchor. That
  needs 2.4's gyro.
- The bridge is computed on every frame for the held-out check (two USAC fits per frame): VO tracking 9.3 → 12.4 ms
  per frame, averaged over the 11 flights. Its timing goes in the
  COOL budget review; it can be limited to lost frames once it is validated.

---

<a id="estimator-plan"></a>

# Estimator build plan

Plan, Phase 2 (I2 + I3 + I3.5 + I4): perception swap-in, radiometric stages, real local SLAM, the inertial layer, the
robust single-agent backend, and the dataset program. This document fixes the **order**, the **conventions**, the
**acceptance targets** (from primary sources, before anything is measured) and the **gate mechanics**. Per-item
designs (`perception.md`, `local-slam.md`, …) follow as each item opens.

### 1. Order (critical path first, de-risk first)

| # | item | why here |
|---|---|---|
| 0 | **XFeat smoke test** (D15) — done, SMOKE §4p | the riskiest unverified commitment of the phase; exact vs upstream, 12–15 ms |
| 1 | **Sparse SE(3) pose-graph solver** (§11.6, §12) + gauge null-space test + FD Jacobians + GTSAM oracle on 100 random graphs | underpins 2.3 (local pose graph, window exit) and 2.5; retires one exit-gate item |
| 2 | **2.3 windowed BA** (§11): keyframe policy, landmark lifecycle, analytic-Jacobian Schur LM with a sparse reduced camera system, window exit → ODOMETRY edge with the *joint* covariance, covariance recovery | the estimator itself |
| 3 | **2.4 inertial** (§10): `attitude_only` (Mahony) **first**, then Forster preintegration in BA, stationary/dynamic init, GRAVITY_PRIOR, LK motion seeding | time budget below; fallback exists before the risky part starts |
| 4 | **2.5 robust backend** (§13–§14): chordal init → degeneracy gate → PCM → GNC-TLS → certificate, in that order (§14.5 ordering is inviolable); a real intra-agent loop-candidate source | needs 1 and real loops from 2/3 |
| 5 | **2.1 perception** (§7–§9): XFeat engine at keyframes, spatial-bin selection, DNN contract, SGBM + WLS confidence, §9.4 validator, ANNIndex retrieval, ALIKED + LightGlue hard path | parallel-safe (∥A) |
| 6 | **2.2 radiometric** (§6.2, §6.4): flat-field, exposure normalisation, GAT VST + exact unbiased inverse, gated on measured noise | parallel-safe (∥A); needs the noise stages wired into `SimSource` |
| 7 | **2.6 datasets** (§34.1): all 11 EuRoC sequences (stereo + IMU), a TUM RGB-D subset (sensor depth replaces SGBM) | downloads started at Phase 2 open; regression values pinned per sequence |

Items marked ∥A/∥B may run in parallel; the unmarked items are the critical
path.

### 2. Conventions (fixed before any Jacobian is written)

- **One perturbation convention per parameterization, derived, not copied.** The project convention is the LEFT
  perturbation, tangent `[ρ; φ]` (D1). The existing BA perturbs `T_C_W` on the left; `PoseNode` stores `T_W_C` with
  a left-tangent covariance. The gauge null space is derived for *our* parameterization and asserted with
  `‖H·N‖ ≈ 0` (the catalogue's §7.3 basis is written for the right convention and is not copied). For left
  perturbations of `T_W_C` and the residual `Log(Z⁻¹ Xᵢ⁻¹ Xⱼ)`, a common left perturbation of every pose leaves
  every residual unchanged, so the six null vectors are "the same ξ on every node". The test proves it.
- **Inertial factors.** Forster et al. (T-RO 2017) perturb R on the RIGHT. They are re-derived in left form, or
  converted at one boundary with adjoints; every block is finite-difference tested. Never transcribed as-is.
- **Determinism.** Solver loops stay single-threaded with ordered containers. A fixed small BA problem and a fixed
  PGO problem join the determinism digests, so cross-architecture CI covers the new numerics.
- **Manifest fields, not constants:** keypoint caps (§7.3), `imu_mode`, the DNN engine that ran, the feature
  engine, the window size, the GNC/PCM thresholds.

### 3. Acceptance targets, fixed from primary sources BEFORE measuring

**EuRoC (exit gate: "stereo-VIO ATE within the published error class of reference stereo-inertial systems").**
Source: ORB-SLAM3 (Campos et al., T-RO 2021, arXiv 2007.11898) Table II, stereo-inertial rows, RMS ATE in metres.
Other systems' values are as reported by their authors, except VINS-Fusion (run by the ORB-SLAM3 authors at default
settings) and BASALT (keyframe trajectory).

| system | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VINS-Fusion | 0.166 | 0.152 | 0.125 | 0.280 | 0.284 | 0.076 | 0.069 | 0.114 | 0.066 | 0.091 | 0.096 | 0.138 |
| BASALT | 0.080 | 0.060 | 0.050 | 0.100 | 0.080 | 0.040 | 0.020 | 0.030 | 0.030 | 0.020 | – | 0.051* |
| Kimera | 0.080 | 0.090 | 0.110 | 0.150 | 0.240 | 0.050 | 0.110 | 0.120 | 0.070 | 0.100 | 0.190 | 0.119 |
| ORB-SLAM3 | 0.036 | 0.033 | 0.035 | 0.051 | 0.082 | 0.038 | 0.014 | 0.024 | 0.032 | 0.014 | 0.024 | 0.035 |

**The class, defined now:** on every sequence our RMS ATE is ≤ the **worst** of these four on that sequence, and
our average over all 11 is ≤ **0.138 m** (the class's upper end, VINS-Fusion). Two qualifiers are reported with the
number, never hidden: ORB-SLAM3 is a full SLAM system with loop closing, and BASALT reports keyframe trajectories.

**Protocol:** the full trajectory (every frame), SE(3) alignment only, no scale (Ariadne is metric; Sim(3) would hide
scale error), the processed ground truth `state_groundtruth_estimate0`, evo APE as the oracle, a single run (ours is
deterministic; the reference values are medians of 10 runs).

**TUM RGB-D:** a pinned regression, not a claim. Values are recorded per sequence on first pass and must not regress
by more than a stated tolerance.

**PGO oracle:** Ariadne PGO vs GTSAM (pip wheel, version pinned from the index) on 100 seeded random synthetic graphs
(g2o files): final cost within 1e-6 relative, poses within 1e-6. **Certificate:** `certified` on noise-free and
low-noise graphs. On planted-outlier graphs the *fraction* certified is reported: a certificate proves global
optimality of the given cost, not correctness, so a certified outlier-corrupted optimum is correct behaviour and is
not tuned away. **Outlier ramp:** 10→90% false loop closures; success probability, ATE and runtime plotted, and the
breakpoint reported.

### 4. Two contradictions in the docs, resolved against the papers

- **GNC μ schedule.** Arch §14.6 says μ ×1.4 and the catalogue says μ/1.4. Yang et al. (RA-L 2020, arXiv
  1909.08605), Remark 5: for **TLS**, start near μ → 0 (convex surrogate) and **increase** μ ← 1.4 μ until the
  weighted residual sum converges, with μ₀ = c̄² / (2 r²_max − c̄²). The "÷1.4" is **Geman-McClure**'s schedule, which
  starts large and decreases. The architecture is right; the weight formula in §14.6 is checked against the paper and
  GTSAM's `GncOptimizer` when 2.5 opens.
- **Certificate sign.** Arch §14.8 writes `S = Λ − Q̃`; the catalogue (following SE-Sync) writes `S = Q̃ − Λ` with a
  different product order inside Λ. The formulation transcribed will be SE-Sync's (Rosen et al., IJRR 2019, eq. for
  the certificate matrix). The unit test fixes the sign: on a noise-free graph the correct S has exactly d zero
  eigenvalues and all others ≥ 0. The test is `λ_min ≥ −ε·‖Q̃‖` after deflating the gauge null space, never `> 0`.

### 5. Gate mechanics in Phase 2 (D49 applied)

- **Placeholders stay selectable** (arch §0.2: demoted to ablation baselines, never deleted). ORB + LK + the
  two-frame BA stub remain reachable by config.
- **The pre-merge gate runs twice per change.** (a) The Phase 1 chain on the *placeholder* configs must stay
  byte-identical to the Phase 1 evidence; this proves the baseline survives and shared code did not regress. (b) The
  new pipeline runs with the I1, contract and stage checks only, until Phase 2 evidence exists; after that, against
  Phase 2's own evidence. If a shared-code change must move placeholder outputs, that is a recorded re-baseline
  decision, never a loosened exclusion.
- **"Regression suite pinned in CI" (plan 2.6)** means: per-sequence values pinned in a committed file and checked by
  the pre-merge gate (D49). GitHub `ci.yml` keeps build, tests and cross-architecture determinism, plus the small
  BA/PGO digests.
- `contracts.yaml`'s unset list shrinks as fields are filled (keyframe uncertainty in 2.3, inertial state in 2.4,
  LoopEdge pcm/gnc/degeneracy in 2.5). The gate checks are updated in the same change, or the file starts lying.

### 6. Inertial time budget and inputs

- **Budget:** `attitude_only` (Mahony) about half a day, then full preintegration about two days. If the full mode is
  not at the EuRoC class by then, ship `attitude_only`, record it in the manifest and proceed (plan fallback). The
  phase does not stall.
- **Initialization:** the gyro/accel variance over the first 2 s of each sequence decides whether it is stationary.
  EuRoC sequences are not all at rest at their start. Non-stationary starts initialise from stereo poses plus
  preintegration; stereo already provides scale, so no monocular VI alignment is needed (§10.3).
- **Sim IMU noise:** the D39 canonical IMU stream is analytic and noise-free by design (noise, bias and jitter are a
  seeded consumption-time operator, D37). 2.4 implements that operator, with the densities recorded once in the
  manifest and shared by simulator and estimator (§10.2).

### 7. Sources

ORB-SLAM3 (arXiv 2007.11898, Table II); Forster et al. T-RO 2017 (arXiv 1512.02363); Yang et al. RA-L 2020 (arXiv
1909.08605); Mangelson et al. ICRA 2018 (PCM); Carlone et al. ICRA 2015 and Martinec & Pajdla CVPR 2007 (chordal
initialisation); Rosen et al. IJRR 2019 (SE-Sync, arXiv 1612.07386); Potje et al. CVPR 2024 (XFeat); Mäkitalo & Foi
IEEE TIP 2011/2013 (exact unbiased GAT inverse); `docs/algorithms.md` §1.3, §1.7, §7.3–§7.11 for the catalogue's
gotchas.
