# Design — 2.4 inertial layer: attitude_only (D58) and full preintegration (D59), both built

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

- **2026-10-19 (maintainer): `euroc_v2` is `imu.mode: full`** (avg 0.121 vs 0.223 m, 9/11 vs 4/11 in the
  stereo-inertial class; costs MH_03 +6 mm, V1_03 still out). The attitude_only config is kept as
  `euroc_v2_attitude` (the D58 baseline and ablation). The opt-in `euroc_v2_full` it replaces was identical apart
  from its name; the §4.7–4.9 evidence directories still carry that name.
- `sim_v2` stays `none` (§5).

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

### 4.8 Gravity in the window (part 3): built, NEGATIVE, kept as an ablation (`imu.full.estimate_gravity`)

**Idea.** window[0]'s 6-DoF pose is the gauge, so the accelerometer cannot pull the trajectory's roll/pitch back to
gravity. Gravity's direction became 2 tilt parameters of every window (carried prior), and after each solve the whole
solution was rotated about window[0]'s position, with no yaw, so gravity is the reference again: a 4-DoF gauge in
effect. Every cached world-frame state in StereoVo got the same rotation.

**Tested correct:** FD gradient and Hessian with the tilt block, Schur = dense, energy invariant under the alignment,
no yaw, and a window tilted 2° was found at 2.0000° and rotated back onto the truth.

**Result on EuRoC: much worse.** Average ATE 0.137 → 0.566 m, 0/11 in class; rotate-back corrections summed to
620–4,760° per flight (single steps 8–20°). The MH_01 per-keyframe trace shows a feedback loop:
- a 10-keyframe window cannot separate tilt from b_a (a tilt θ looks like a bias of 9.81 θ), so the window's tilt
  marginal is mostly its own prior;
- the prior was carried with the bias-prior bookkeeping (marginal × 10), which loosens an unobserved direction tenfold
  per window: prior σ 3.6° → 12° within a few keyframes, tilt moves of 1–3° per window, b_a swinging 0.1 → 1.4 m/s².

**Why part 2 stands.** ORB-SLAM3's local visual-inertial BA anchors on fixed keyframes too: §V-C optimises a sliding
window "keeping [covisible keyframes'] pose fixed", and its VI welding BA has "the keyframe immediately before the
local window … included but fixed". Long-term tilt is held there by map reuse (old points seen from fixed covisible
keyframes) and loop closure. Here those are 2.1 (re-association) and Phase 3. The per-keyframe gravity for the pose
graph is arch §10.4's GRAVITY_PRIOR: produced here (plan 2.4, "gravity-prior edges"; next), consumed by the pose
graph (2.5). So the residual tilt drift on hard flights is not a 2.4 gauge defect.

**Lesson applied to the bias prior.** The same mechanism exists there: ×10 on the whole marginal loses the prior's own
information on any direction a window does not observe (b_a in a dark window). Measured in the fake outages:
V2_01 / propagate / 10 s, |b_a error| 0.15 → 2.0 m/s². Since part 3 the next prior keeps the carried prior's
information and deflates only the data's addition: Λ_next = Λ_carried + PSD(Λ_marg − Λ_carried) / κ.

### 4.9 Results, part 3 (2026-10-19): bias prior never looser; §4.4 decided

| flight | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attitude_only (D58, shipped) | 0.075 | 0.069 | 0.093 | 0.394 | 0.353 | 0.087 | 0.099 | 0.421 | 0.070 | 0.116 | 0.680 | 0.223 |
| full, part 2 (×10 bias prior) | 0.077 | 0.042 | 0.093 | 0.242 | 0.195 | 0.061 | 0.081 | 0.254 | 0.077 | 0.107 | 0.281 | 0.137 |
| **full, part 3** | 0.068 | 0.039 | 0.099 | 0.222 | 0.128 | 0.056 | 0.052 | 0.357 | 0.039 | 0.101 | 0.168 | **0.121** |
| stereo-inertial class worst | 0.166 | 0.152 | 0.125 | 0.280 | 0.284 | 0.076 | 0.110 | 0.120 | 0.070 | 0.100 | 0.190 | ≤ 0.138 |

- **Average 0.121 (target ≤ 0.138 met by 17 mm); per flight 9/11** (V1_03 out at 0.357; V2_02 out by 1 mm).
- The states against ground truth improved on every hard flight, V1_03 included (b_a median 0.245 → 0.192 m/s²,
  b_g 0.0076 → 0.0034 rad/s, v 0.046 → 0.040 m/s); V1_03's ATE regression (0.254 → 0.357) is not a worse IMU
  estimate. V1_03 has flipped on small changes before (D58: 0.42 ↔ 0.82 with LK seeding). Reported as measured.
- Body-frame gravity error, median over the run: V2_03 3.51 → 1.37°, MH_04 1.27 → 0.40°.
- **§4.4 decided (fake outages, corrected metric: drift 2.5 s after the outage ends):** under `lost.reanchor:
  every_frame` (euroc_v2's), every lost frame is a keyframe and the inertial window solves its pose, so `decay` and
  `propagate` give identical results to the millimetre (354 propagated frames on MH_01, same poses). `lost_motion`
  only sets the window's starting point there; it stays `decay`. What the outages measure is full vs attitude_only:
  0.5 s 0.042 vs 0.102 m, 2 s 0.18 vs 0.61, 5 s 0.64 vs 1.29, **10 s 4.6 vs 3.1 (full worse)**: ten seconds of IMU
  dead reckoning on a MAV diverges (V2_01 9.1 m). ORB-SLAM3 bounds IMU-only tracking at 5 s before starting a new
  map; the analogue here belongs with map segments and re-association (2.1 / Phase 3) and is recorded, not tuned.

### 4.10 Gravity-prior edges (arch §10.4, §12.1): design, fixed before code

**Scope.** 2.4 produces the edges (struct, residual, Jacobian, emission, validation). The pose-graph solver (2.5)
consumes them: `PoseGraph::Cost()` and `pgo_levmarq` are not touched; the edges live in their own container with
their own cost, and a handoff note goes to 2.5. Inter-agent use (broker PGO, §15) and a wire field are Phase 3.

**Factor (literature: GTSAM `AttitudeFactor<Pose3>`, `navigation/AttitudeFactor.h`, [VERIFIED web 2026-10-19]:
`error = nRef.errorVector(nRb * bMeasured)`).** Unary on a keyframe's T_W_C:
- measurement: g_C, the unit "down" direction in the keyframe's rectified left camera;
- reference: ĝ_W, the agent's gravity direction in its world;
- residual r = B(ĝ_W)ᵀ (R_WC g_C) ∈ R², B = `GravityTangentBasis` (a basis of the plane ⟂ ĝ_W): zero when the
  camera's tilt agrees, blind to yaw;
- Jacobian for the left perturbation [ρ; φ] of T_W_C: ∂r/∂ρ = 0, ∂r/∂φ = −B(ĝ_W)ᵀ [R_WC g_C]×;
- information: 2×2 in that basis. OpenCV has no equivalent (OpenCV-native check: none).
- Our basis differs from GTSAM's `Unit3` basis by a 2×2 rotation: a GTSAM comparison is on costs, with isotropic
  information.

**The measurement and its σ (decided by NEES, not argued).** A prior that copies the node's own tilt (g_C = R_WCᵀ ĝ_W)
is zero at emission and its error is the accumulated tilt drift; it would resist loop closures that correct that
drift unless σ covers it. Two candidates are logged per emission and compared:
- **A (independent):** gravity's direction re-estimated from the solved window's own data (the §4.8 tilt variables,
  as a one-shot query that never feeds back), σ from that estimate's marginal. Caveat: the carried b_a prior encodes
  the fixed-gravity history, so a query that keeps it is not independent along tilt ↔ b_a; the query variants with and
  without it are both measured.
- **B (pose-derived):** g_C = R_WCᵀ ĝ_W, σ² = the initialisation's gravity variance + the node's absolute rotation
  variance on the two tilt axes (it grows with the drift).
- **Criterion:** per flight, the 2-dof NEES of the emitted g_C against the ground-truth camera-frame gravity, mean ≈ 2,
  tail reported. Never tuned on ATE.
- **Correlation:** consecutive windows share data, so an edge per keyframe would count the same information many
  times. Emission stride = `ba.window_size` keyframes (non-overlapping windows); the NEES is measured at that stride.

**attitude_only producer.** Mahony's g_C on the same stride, σ from config. Its error is heavy-tailed (D58: median
1.3–4.2°, worst 15°) and §12.1 robustifies only loop edges: the NEES tail is reported and flagged to the consumer.

**No IMU (sim_v2: none): no edges.** The sim gate must not change.

**Tests:** FD Jacobian (large rotations; translation block exactly zero); yaw invariance about ĝ_W; zero residual
at emission; cost against GTSAM on isotropic information when the oracle image is used.

### 4.11 Gravity-prior edges: measured and decided (2026-10-20)

Every emission (stride 10 keyframes) logged four candidates; 22 EuRoC runs (11 flights, full and attitude_only).
2-dof NEES against the ground-truth camera-frame gravity (ideal mean 2, p90 4.6), and median angle error:

| flight | pose | query | query_free_ba | Mahony (σ 5°) |
|---|---|---|---|---|
| MH_01 | 3.5 / 0.44° | 6.3 / 0.46° | 3.2 / 2.7° | 0.20 / 1.3° |
| MH_03 | 25.8 / 1.77° | 73.0 / 1.75° | 7.2 / 5.6° | 1.08 / 4.1° |
| V1_01 | 366 / 2.41° | 145 / 2.38° | 10.5 / 2.6° | 0.51 / 3.2° |
| V2_03 | 0.0 / 1.35° | 76.2 / 1.45° | 20.0 / 8.6° | 0.60 / 2.8° |
| (all 11) | 0–366 | 4.7–145 | 2.0–20 | 0.09–1.08 |

- **VIO-derived candidates are accurate but not honest.** `pose` and `query` err 0.4–2.4° (median), but every one
  inherits the fixed reference gravity's own error, a constant per-flight offset no window marginal contains
  (V1_01: 2.4° median, its initialisation was 2.7° off). Overconfident by a flight-dependent 2–70×; `pose`'s σ
  also becomes uninformative after lost stretches (NEES 0). Without the b_a prior (`query_free_ba`) the estimate is
  near-honest but carries 2–9° errors.
- **Mahony is independent of g_W and consistent across flights: chosen, σ = 2.3°.** Calibrated on Machine Hall (the
  accurate Leica ground truth, as in §4.1; pooled NEES 2.00), validated on the held-out Vicon flights: pooled NEES
  3.0, p90 7.3, p99 18.9 (part of the excess is the Vicon rooms' noisier orientation). Heavy tail: the consumer
  should robustify these edges (§12.1 robustifies only loops by default).
- **Reference direction.** g_ref_W is the producer's estimate (full: the VIO gravity; attitude_only: the first emitted
  keyframe's Mahony direction). Its own error is a constant tilt shared by every edge: the solver should estimate the
  agent's gravity direction as a 2-DoF variable initialised at g_ref_W.
- Emitting edges changes no trajectory (all 11 full-mode runs byte-identical to D59 part 3); `graph_version` in
  results.yaml changes.

## 5. Sim

The IMU noise operator was planned as part of 2.2's degradation wiring. **2.2 did not build it:** D65–D69 contain no
IMU work. So `sim_v2` stays `imu: none`. The sim recordings carry clean IMU streams, but no seeded IMU noise or bias
operator consumes them. The item is open and unowned (§6).

> **Built (L2, D101, 2026-10-22).** `core/inertial/imu_noise_operator.hpp`, through `ariadne_sim_vo --imu-noise
> configs/sim/imu_noise_v1.yaml`. `sim_v2` is now `imu.mode full`; the old `sim_v2` is `sim_v2_vo`.

## 6. Status (2026-10-21, on `main` 3cf5f0c; premerge PASS)

The plan 2.4 line, item by item:

| plan item | state | where |
|---|---|---|
| preintegration with bias states | built: Forster on-manifold, 9-dof residual, bias random walk, carried bias prior | D59 parts 2–3, §4 |
| stationary initialization | built: gyro bias from a stationary test (`attitude_only`); the full mode initialises b_g, v, g, b_a from keyframes (inertial-only MAP) and refines at +5 s / +15 s | D58 §2, D59 §4 |
| gravity-prior edges | built and emitted: Mahony source, σ 2.3°; consumed by 2.5's pose graph (D73 part 2c, off by default) | D59 gravity priors, §4.10–4.11 |
| motion-predicted LK seeding | built, **off by default**: lost frames −3%, but V1_03 worse | D58 §3 |
| `attitude_only` fallback, selectable | built: `euroc_v2_attitude` | D58, D59 switch |

**Measured:** `euroc_v2` (`imu.mode full`) average ATE 0.121 m, 9/11 flights in the stereo-inertial class. The two
out of class are V1_03 (0.357 m) and V2_02 (+1 mm over).

**Open, stated rather than hidden:**
- **The combined EuRoC run with 2.1's re-association: done (D85).** Average 0.120, still 9/11 in class (V1_03 and
  V2_02 out). The effect is neutral overall, with mixed ±10 mm per flight.
- **Long outages:** 10 s without vision diverges (`--outages` evidence, D59). With re-association enabled, the runner
  now refuses `--outages` (2.1's guard).
- **Gravity estimated inside the window:** a negative result, kept off (`estimate_gravity: false`, D59 part 3).
- **The sim IMU noise operator** is not built (§5).

