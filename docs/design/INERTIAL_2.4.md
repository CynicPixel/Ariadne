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
  - Every hard-coded 6 in `BaLayout`, `BaSchur`, `BaOplus`, the marginals and the exit edge becomes the per-keyframe
    block size; the exit edge takes the 6×6 pose sub-block.
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
- **Observability guards:** an inertial factor TIES consecutive keyframes. `AnchorWindow` must not hold keyframes made
  while lost when an inertial chain connects them to the gauge, and the degeneracy fallback counts inertial
  information the same way.

### 4.3 Initialisation and gravity

- **Gravity is estimated once, in the VO world, then held fixed.** The world is not rotated: graph, keyframe records
  and the gate depend on it.
- **Init after N consecutive TRACKED keyframes** (MH_04 and MH_05 start dark: lost poses never feed it).
  - b_g comes from the attitude tracker (VO-aided).
  - v_i, g and b_a come from linear least squares on Δv, Δp (VINS-style, scale known: stereo); |g| is projected to
    9.81.
  - b_a gets a loose prior (weakly observable, arch §10.3).
  - Checked against ground-truth v, b_a and the gravity direction.
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

## 5. Sim

The IMU noise operator is 2.2's degradation wiring (coordinate first). `sim_v2` stays `imu: none` until then.
