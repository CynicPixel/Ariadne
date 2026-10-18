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

## 4. Next: full preintegration (decisions taken)

- **State per keyframe:** T_WB, v_W, b_g, b_a (15). Inertial factors touch only camera blocks; the Schur structure
  is unchanged.
- **Factors:** Forster's three-block residual plus the bias random walk, re-derived in left perturbation, every block
  FD-tested.
- **Window exit (maintainer, 2026-10-18): bias prior only.** The exiting keyframe's bias estimate becomes a loose
  Gaussian prior on its successor's bias (inflated covariance). Poses and velocity still drop and emit an edge. This
  is a recorded departure, not FEJ marginalisation.
- **Initialisation:** stationary when detected; otherwise from stereo poses plus preintegration (stereo gives scale;
  Campos 2020's MAP formulation without the scale variable).
- **Sim:** the IMU noise operator is 2.2's degradation wiring (coordinate first). `sim_v2` stays `imu: none` until
  then.
