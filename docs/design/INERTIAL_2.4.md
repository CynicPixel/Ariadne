# Design — 2.4 inertial layer: attitude_only (built, D58), then full preintegration

Plan 2.4 (arch §10). Order fixed by PHASE2_PLAN §1: `attitude_only` first (§10.7 fallback, about half a day), then
Forster preintegration in the window BA (about two days). If `full` is not in the stereo-inertial class by then, ship
`attitude_only` with the manifest recording it.

**Preflight:**
- Literature (`ALGORITHMS.md` §1.3b): Forster 2017, Mahony 2008, Hwangbo 2009, Campos 2020.
- OpenCV-native check: OpenCV 5 has no inertial primitives; the LM driver and SE(3) maths are already ours.
- Advisor review.

## 1. Validated before use (step 2)

The gyro integral (C++, `core::IntegrateGyro`, zero-order hold, intervals clipped) was rotated into the rectified
left camera through `T_B_Crect` and compared with ground truth. All 11 flights, 26,530 tracked pairs:

| | median | p90 | p99 |
|---|---|---|---|
| raw gyro | 0.23° | 0.25° | 0.47° |
| ground-truth bias removed | **0.019°** | 0.043° | 0.11° |
| time shift −10 / −5 / 0 / +5 / +10 ms | 0.069 / 0.045 / **0.018** / 0.019 / 0.045° | | |

Frame, sign and extrinsic are right, and timing is aligned within a sample. The bias dominates the raw error.

## 2. attitude_only (built)

`slam/inertial/attitude.hpp`; runtime glue in `stereo_vo_runner.hpp`; slam config `imu:`.

- **Gyro bias, VO-aided.** Every pair of consecutive tracked frames gives a sample `(Log R_gyro,raw − Log R_vo)/dt`;
  the estimate is the per-axis median of the last 200. Before any sample: the mean gyro if the first 2 s pass the
  stationary test (std < 3× white noise), else zero.
  - Mahony's own bias cannot be used: it cannot see the gravity-axis component.
- **Rotation while lost:** `R_W_C = R_W_C(prev) · R_C(prev)C(cur)` from the bias-corrected gyro. Translation is the
  D56 decaying velocity. `PoseSource::kGyro`. The visual bridge remains the fallback without IMU coverage.
- **Gravity:** Mahony et al. 2008, Theorem 5.1, kP = 1, kI = 0.3 rad/s (the paper's §VI values), initialised from
  the first accelerometer sample (yaw 0, arch §10.3). It is carried on every keyframe record as `gravity_dir_C`.
- **LK seeding (§10.5):** `KltTracker::Process(..., const cv::Matx33d* predicted)` takes `H = K Rᵀ K⁻¹`, the pure
  rotation homography, as the initial flow. Default off (§3).

## 3. Results (EuRoC v2, ATE in m)

| flight | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 D56 (visual bridge) | 0.075 | 0.070 | 0.093 | 0.364 | 0.349 | 0.070 | 0.137 | 0.562 | 0.071 | 0.120 | 1.266 | 0.289 |
| **v2 + gyro while lost (shipped)** | 0.075 | 0.069 | 0.093 | 0.394 | 0.353 | 0.087 | **0.099** | **0.421** | 0.070 | 0.116 | **0.680** | **0.223** |
| + gyro-seeded LK | 0.080 | 0.071 | 0.095 | 0.313 | 0.342 | 0.082 | 0.148 | 0.820 | 0.070 | 0.114 | 0.609 | 0.250 |
| stereo-inertial class worst (target) | 0.166 | 0.152 | 0.125 | 0.280 | 0.284 | 0.076 | 0.110 | 0.120 | 0.070 | 0.100 | 0.190 | ≤ 0.138 |

**Stereo class (`LOCAL_SLAM_2.3.md` §8): 10/11** (was 9/11). V2_03 enters it. Its rotation RMSE went from 48° to 9°
with seeding; without seeding the ATE drops to 0.68.

**The gyro beats the visual bridge on the hard rotations** (V2_03, V1_03, V1_02). It costs V1_01 (+17 mm) and MH_04
(+30 mm).

**LK seeding:** lost frames 743 → 720 (−3%); MH_04 and V2_03 better, V1_03 much worse. Off by default. This
confirms D56: tracks are not what fails. Landmark creation in the dark is.

**Gravity vs ground truth, at keyframes:** median 1.3–4.2° per flight, p90 2.3–8.3°, max 15°.
- The MAV's dynamic accelerations leak through Mahony's 1 rad/s crossover.
- A GRAVITY_PRIOR from this filter must carry σ ≈ several degrees.
- `full` estimates gravity properly.

**Not yet in the stereo-inertial class:** average 0.223 vs ≤ 0.138; per flight, MH_04, MH_05, V1_01, V1_03 and
V2_03 are out. As planned, `attitude_only` cannot reach it.

## 4. Full preintegration (D59): design, fixed before code

**Timebox: end of 2026-10-20.** If `full` is not inside the stereo-inertial class by then, `attitude_only` stays
shipped (plan fallback) and `full` stays opt-in.

### 4.1 Validated before BA: preintegration against EuRoC ground truth

Between instants 0.5 s apart, Forster's preintegrated ΔR, Δv, Δp (eqs. 35–38) were compared with ground-truth pose,
velocity and biases (residuals of eq. 45, g = (0, 0, −9.81)). They were whitened by the propagated Σᵢⱼ (eqs. 62–63,
discrete noise density²/Δt).

- **Physical residuals are small:** median |r_v| 1.8–5.6 cm/s, |r_p| 5–15 mm over 0.5 s. Frames, gravity sign and
  accelerometer convention are right; an error there gives metres.
- **χ²/dof is 33–350 with the datasheet densities.** χ² stays flat when the interval goes from 0.1 to 2 s (MH_01:
  42, 43, 65), so the excess grows like IMU noise, not like a constant ground-truth error.
- **The Vicon rooms read 5–20× higher than the Leica Machine Hall for the same IMU.** Their ground truth (orientation
  and velocity) is noisier, so they cannot calibrate the IMU.
- **Measured inflation** (Machine Hall, 1,271 intervals, median per-block χ², re-measured with the mid-interval
  integration of §4.2): gyro √9.75 = **3.1**, accelerometer √74–81 = **8.6–9.0**.
  - The first-order integration had given 3 and ~9.5, so the excess is sensor noise (rotor vibration on a real MAV),
    not integration error.
  - Config `imu.noise_inflation_gyro: 3`, `imu.noise_inflation_accel: 9`, applied to the white-noise and random-walk
    densities alike.
  - **EuRoC only:** the simulator's IMU noise is exactly its configured model (factor 1).
  - Ablations: 1× and the pooled 5.4 / 12.4. Never tuned on ATE.
  - Published VIO configs inflate EuRoC's densities by similar or larger factors.

### 4.2 State, factors, conventions

- Per keyframe: `T_W_C` (left perturbation, as now), `v_W`, `b_g`, `b_a`: a 15-dimensional block.
  - **As built (advisor review, 2026-10-18; replaces "every 6 becomes the block size"):** the parameter vector is
    `[poses 6·n_c | inertial 9·n_i | landmarks 3·n_l]`. Landmarks never touch v or b, so `Hcl`, the Schur
    elimination, `CameraCovariance` and the exit edge are unchanged; only the reduced system S grows. A pose's
    fixedness and its velocity/bias's are separate (window[0]: pose fixed, v and b free). With no inertial blocks every
    number is bit-identical to before.
- Inertial factor between consecutive window keyframes: Forster's 9-dimensional residual (eq. 45) with first-order
  bias correction (eq. 44); repropagate when |δb| exceeds a threshold.
- **Integration, as built** (`slam/inertial/preintegration.hpp`):
  - The accelerometer sample is rotated by the MID-interval orientation, as VINS-Mono does. The paper's
    start-of-interval rotation left 1.8 mm/s and 0.45 mm of residual at the truth of a closed-form test trajectory (0.5 s,
    1 kHz).
  - The covariance and bias Jacobians were re-derived for that mean.
  - Tests: bias Jacobians vs re-integration, every residual block vs central differences (20 random states, large
    rotations), residual ≈ 0 at truth, Σ vs a 4,000-run Monte Carlo. All pass.
  - Re-derived for our left perturbation of `T_W_C` through the fixed `T_B_C`; every block finite-difference tested,
    including large rotations.
- **Bias random walk** between consecutive keyframes: `b_j − b_i`, covariance random-walk density² × Δt.
- **Energy ½ rᵀΩr** (the reprojection term's Huber quadratic is s²/2): the IMU-vs-vision weighting is exact. No
  robust kernel on inertial factors.
- **Window[0]** (maintainer, 2026-10-18; NOT ORB-SLAM3's choice, which also fixes velocity):
  - its pose is the gauge, fixed;
  - its velocity is free;
  - its biases get a loose Gaussian prior from the previous solve, with covariance × `imu.bias_prior_inflation`
    (config);
  - the window[0]→window[1] inertial factor stays.
- **Observability guards (corrected while building).** An inertial factor does NOT by itself tie consecutive keyframes:
  the IMU observes relative rotation and, given the velocity, relative position, but not the velocity. Adding c to
  every v_k and c(t_k − t_0) to every p_k leaves every inertial residual unchanged (a Galilean null space; test
  `InertialOnlyChainHasTheGalileanNullSpace`: |Hn|/|n| 2.5e-10 vs 6e6 for a real direction). So `AnchorWindow`
  follows inertial factors only out of a group that holds two pose-determined keyframes (fixed, or reached through
  covisibility); two positions at different times fix c.
- **Dark windows: v_0 held (maintainer, 2026-10-18).** When window[0] (the gauge) has no covisible partner, the rule
  above would hold every lost keyframe at its predicted pose and the stiff inertial factors would pull the biases
  toward the prediction's error. Instead window[0]'s velocity is held at the previous solve's estimate
  (`BaInertial::hold_v0`, set by `AnchorWindow` only in that case): window[0]'s group then ties inertially and the
  lost keyframes are IMU-determined. Normal windows keep v_0 free (bias prior only). Tests: the held window's H is
  PD (min eigenvalue 1.6) and the released one is singular (−2.8e-8); a dark window is dead-reckoned from perturbed
  starts to the truth.
- **Solve gate.** An inertial window is solved even with fewer landmarks than `pnp.min_correspondences`: a dark
  stretch is exactly where the IMU must act.
- **Bias prior bookkeeping.** window[1]'s bias marginal comes from the same factorization as the pose marginals.
  When it is degenerate, or no solve ran, the previous prior is carried forward by the random walk
  (Σ_b += diag(σ_rw² Δt)) instead of being dropped; `results.yaml imu_full.bias_prior` counts both branches.
- Intervals are re-integrated between solves only (never inside one, so the energy LM compares never changes), when
  the start keyframe's bias moved more than 0.01 rad/s or 0.1 m/s² from the interval's linearisation point
  (VINS-Mono's thresholds).

### 4.3 Initialisation and gravity

- **Gravity is estimated once, in the VO world, then held fixed.** The world is not rotated: graph, keyframe records
  and the gate depend on it.
- **Init after N consecutive TRACKED keyframes** (MH_04 and MH_05 start dark: lost poses never feed it).
  - **b_g (changed while building, stated here per §49):** from the init keyframes' own window-refined rotations,
    Gauss–Newton on the rotation residuals weighted by the propagated Σ_R (VINS-Mono `solveGyroscopeBias`), NOT from
    the attitude tracker. It uses exactly the poses the rest of the initialisation uses and keeps StereoVo independent
    of the runner's tracker. The intervals are then re-integrated at that b_g.
  - v_i, g and b_a come from the linear MAP on Δv, Δp weighted by the preintegration information, with a zero-mean
    b_a prior (σ 0.2 m/s²): ORB-SLAM3's inertial-only MAP with the scale known (stereo), hence linear. g is then
    re-solved on the |g| = 9.81 sphere (VINS-Mono `RefineGravity`).
  - **Two refusal checks.** The unconstrained |g| within 10 %, and the fit's χ²/dof. A scale or frame error moves |g|
    only by (motion acceleration / 9.81): at 1.5× scale |g| moved 8 % (accepted) while χ²/dof rose ×1000
    (`WrongScaleIsVisibleInTheFit`). The χ² threshold (`imu.full.init_max_chi2_per_dof`) is `[ASSUMPTION]` until set
    from EuRoC initialisations checked against ground truth.
  - **b_a is weakly observable (measured on synthetic data):** over ~2 s of mild rotation it trades off against
    gravity's tilt (a tilt θ looks like a bias of 9.81 θ: b_a error 0.076 = 0.44° × 9.81). The prior decides that
    direction; the window BA keeps refining b_a. With a flat prior the same data recovers everything to 1e-6.
  - Checked against ground truth in each keyframe's own body frame (immune to VO drift): gravity angle, velocity RMS,
    b_g and b_a errors (`results.yaml imu_full.init_vs_gt`, `imu_trace.csv` per keyframe).
- Before init: `attitude_only` behaviour.

### 4.4 While lost

`full` can propagate the last keyframe's v and biases instead of the D56 decaying velocity. Which wins, and up to what
outage length, is MEASURED on held-out fake outages of 0.5–10 s cut into tracked stretches. It is not guessed.

### 4.5 Tests that land with the code

- FD Jacobians of every inertial block.
- Monte Carlo: the sample covariance of the preintegration error matches Σᵢⱼ.
- Synthetic end-to-end with analytic IMU: residuals ≈ 0 at the truth, and VIO beats VO across a dark stretch.
- Determinism with `imu.mode: full`.
- On EuRoC: inertial χ²/dof at the solutions, reported.

### 4.6 Defaults

`imu.mode: full` is opt-in. `euroc_v2` switches only after it beats `attitude_only`; `sim_v2` stays `none`.

### 4.7 Results, part 2 (2026-10-19, `euroc_v2_full`, refinement + init gate)

| flight | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attitude_only (D58) | 0.075 | 0.069 | 0.093 | 0.394 | 0.353 | 0.087 | 0.099 | 0.421 | 0.070 | 0.116 | 0.680 | 0.223 |
| full, init only | 0.077 | 0.036 | 0.099 | 0.267 | 0.204 | 0.063 | 0.095 | 0.252 | 0.080 | 0.104 | 0.304 | 0.144 |
| **full + refinement + gate** | 0.077 | 0.042 | 0.093 | 0.242 | 0.195 | 0.061 | 0.081 | 0.254 | 0.077 | 0.107 | 0.281 | **0.137** |
| stereo-inertial class worst | 0.166 | 0.152 | 0.125 | 0.280 | 0.284 | 0.076 | 0.110 | 0.120 | 0.070 | 0.100 | 0.190 | ≤ 0.138 |

- Average inside the target by 1 mm; per flight 7/11 (V1_03, V2_01, V2_02, V2_03 out).
- Inertial χ²/dof at the solutions 0.38–1.23. Init vs GT (body frame): gravity 0.09–1.65° after the gate (MH_04
  was 8.4° before it), b_g error ≤ 0.003 rad/s, b_a error 0.03–0.57 m/s² (the tilt trade-off).
- **Gravity over the run degrades on the hard flights** (body-frame median 3.5°, final 5.6° on V2_03): the 6-DoF gauge
  on window[0] lets roll/pitch drift that gravity could correct. Part 3: gravity direction estimated in the window
  (carried prior), then the window rotated back to the fixed g_W about window[0] (a 4-DoF gauge in effect).
- §4.4 outages (decay vs propagate) re-measured after part 3; the first measurement is in DECISIONS D59.

## 5. Sim

The IMU noise operator is 2.2's degradation wiring (coordinate first). `sim_v2` stays `imu: none` until then.
