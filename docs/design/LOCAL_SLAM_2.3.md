# Design — 2.3: real local SLAM (windowed BA, keyframe policy, landmark lifecycle, window exit)

Plan 2.3 (I3): "keyframe policy (§8.3), landmark lifecycle (§11.2), analytic-Jacobian Schur LM windowed BA
(§11.3–11.6), window-exit policy (§11.7), covariance recovery, local pose graph." It replaces the 1.5 two-frame BA
stub inside `StereoVo`. The stub stays selectable as the ablation baseline (CLAUDE.md §4).

**Preflight (maintainer's rule):** literature (§7), OpenCV-native check (§6) and advisor review were done before
this document. Every threshold below cites its source.

## 1. Data in → data out

- **in, per frame:** LK tracks (ids + pixels) from 1.4, and at keyframes the left↔right matches and SGBM depth with
  σ_Z (§9.6).
- **out, per frame:** `T_W_C` and its covariance.
- **out, per keyframe:** the §4.2 Keyframe, now carrying `pose_cov` from the window solve and landmark ids.
- **out, at window exit:** an ODOMETRY edge (Z, joint covariance) into the local pose graph, which is optimised by the
  2.0 PGO solver (`POSE_GRAPH_2.md`).

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
- **Exit edge (corrected by advisor review).** The exiting keyframe k is the gauge, so its covariance is zero and no
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

The protocol is that of `PHASE2_PLAN.md` §3: full trajectory, SE(3) alignment, processed ground truth, evo as the
oracle.

**Stated with the numbers:** ORB-SLAM2/3 are full SLAM systems with loop closure, while 2.3 is VO. The stereo-inertial
target of the exit gate belongs to 2.4.

## 9. Config, manifest, contracts, gate

- `configs/slam/*_v2.yaml` select `backend: window`. The `_v1` configs keep `backend: two_frame_stub`, so the
  pre-merge gate's placeholder half stays byte-identical.
- The manifest gains `slam_backend`, `window_size`, `ba_max_iterations` and `huber_delta`.
- `contracts.yaml`'s unset list loses `uncertainty`, as the keyframe uncertainty summary gets filled. The gate check
  is updated in the same change.
