# Design — Phase 2: the single-agent estimator to full spec (plan items 2.1–2.6)

Plan, Phase 2 (I2 + I3 + I3.5 + I4): perception swap-in, radiometric stages, real local SLAM, the inertial layer, the
robust single-agent backend, and the dataset program. This document fixes the **order**, the **conventions**, the
**acceptance targets** (from primary sources, before anything is measured) and the **gate mechanics**. Per-item
designs (`PERCEPTION_2.1.md`, `LOCAL_SLAM_2.3.md`, …) follow as each item opens.

## 1. Order (critical path first, de-risk first)

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

Items marked ∥A/∥B may run in parallel worktree sessions the maintainer launches; this session takes the critical
path.

## 2. Conventions (fixed before any Jacobian is written)

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

## 3. Acceptance targets, fixed from primary sources BEFORE measuring

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

## 4. Two contradictions in the docs, resolved against the papers

- **GNC μ schedule.** Arch §14.6 says μ ×1.4 and the catalogue says μ/1.4. Yang et al. (RA-L 2020, arXiv
  1909.08605), Remark 5: for **TLS**, start near μ → 0 (convex surrogate) and **increase** μ ← 1.4 μ until the
  weighted residual sum converges, with μ₀ = c̄² / (2 r²_max − c̄²). The "÷1.4" is **Geman-McClure**'s schedule, which
  starts large and decreases. The architecture is right; the weight formula in §14.6 is checked against the paper and
  GTSAM's `GncOptimizer` when 2.5 opens.
- **Certificate sign.** Arch §14.8 writes `S = Λ − Q̃`; the catalogue (following SE-Sync) writes `S = Q̃ − Λ` with a
  different product order inside Λ. The formulation transcribed will be SE-Sync's (Rosen et al., IJRR 2019, eq. for
  the certificate matrix). The unit test fixes the sign: on a noise-free graph the correct S has exactly d zero
  eigenvalues and all others ≥ 0. The test is `λ_min ≥ −ε·‖Q̃‖` after deflating the gauge null space, never `> 0`.

## 5. Gate mechanics in Phase 2 (D49 applied)

- **Placeholders stay selectable** (CLAUDE.md §4: demoted to ablation baselines, never deleted). ORB + LK + the
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

## 6. Inertial time budget and inputs

- **Budget:** `attitude_only` (Mahony) about half a day, then full preintegration about two days. If the full mode is
  not at the EuRoC class by then, ship `attitude_only`, record it in the manifest and proceed (plan fallback). The
  phase does not stall.
- **Initialization:** the gyro/accel variance over the first 2 s of each sequence decides whether it is stationary.
  EuRoC sequences are not all at rest at their start. Non-stationary starts initialise from stereo poses plus
  preintegration; stereo already provides scale, so no monocular VI alignment is needed (§10.3).
- **Sim IMU noise:** the D39 canonical IMU stream is analytic and noise-free by design (noise, bias and jitter are a
  seeded consumption-time operator, D37). 2.4 implements that operator, with the densities recorded once in the
  manifest and shared by simulator and estimator (§10.2).

## 7. Sources

ORB-SLAM3 (arXiv 2007.11898, Table II); Forster et al. T-RO 2017 (arXiv 1512.02363); Yang et al. RA-L 2020 (arXiv
1909.08605); Mangelson et al. ICRA 2018 (PCM); Carlone et al. ICRA 2015 and Martinec & Pajdla CVPR 2007 (chordal
initialisation); Rosen et al. IJRR 2019 (SE-Sync, arXiv 1612.07386); Potje et al. CVPR 2024 (XFeat); Mäkitalo & Foi
IEEE TIP 2011/2013 (exact unbiased GAT inverse); `docs/ALGORITHMS.md` §1.3, §1.7, §7.3–§7.11 for the catalogue's
gotchas.
