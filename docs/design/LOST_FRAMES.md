# Design — lost-frame handling: the visual rotation bridge, decaying velocity, segments (D56)

`AUDIT_2.3.md` §4 found lost frames **adversarial**: a lost stretch puts keyframes and landmarks at a predicted
pose, and downstream code cannot tell. It also found that every remaining EuRoC failure loses at least 100 frames.
This design changes what the VO does while PnP cannot localise. It does not change how often that happens: 2.1
(re-association) and 2.4 (IMU) attack the loss rate.

**Preflight (maintainer's rule):**
- Literature (§2).
- OpenCV-native check against the installed 5.0.0 headers: `findHomography` / `findEssentialMat` `UsacParams`
  overloads, and `recoverPose` with `distanceThresh`, all in `opencv2/geometry/3d.hpp`.
- Advisor review before the design and again before the sweep.

## 1. What a lost stretch looks like (measured, `vo_trace.csv`)

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

## 2. Literature

| source | what it does while lost |
|---|---|
| ORB-SLAM3, `Tracking.cc` (stereo, no IMU) | RECENTLY_LOST: relocalise with DBoW2 for 3 s and **make no keyframes**. Then LOST: a new map in the Atlas, or a reset below 10 keyframes. Maps merge on revisit (Atlas, Elvira et al. IROS 2019). |
| ORB-SLAM3 (with IMU) | RECENTLY_LOST: predict from the IMU for 5 s; optionally insert keyframes. |
| VINS-Mono (T-RO 2018) | Failure detection (few features; jumps in position, rotation or bias) → re-initialise → a **new pose-graph segment**, joined by loop detection. |
| ORB-SLAM initialiser (T-RO 2015 §IV-B) | Chooses H or F by `R_H = S_H / (S_H + S_F) > 0.45`, from χ²-truncated transfer-error scores. |
| ORB-SLAM `TrackReferenceKeyFrame` | Tracks against a reference keyframe rather than chaining frame to frame. |
| SP-VIO (arXiv 2411.07551) | Smooths the gap once tracking returns (RTS backtracking). Needs an inertial model: a 2.4 option. |

Ariadne has no relocalisation (2.1) and no IMU (2.4) yet. What remains measurable is the rotation from the tracks.

## 3. Design

### Pose while lost (`lost.motion`)

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

### Keyframes while lost (`lost.reanchor`)

- **`every_frame` (v2, and 1.5):** every lost frame is a keyframe.
- **`gated`:** a keyframe only if tracked plus to-be-triangulated landmarks ≥ 30, which is 2 × the PnP minimum.
  Otherwise none, except at the time cap. It follows ORB-SLAM3's no-keyframes-while-RECENTLY_LOST.
  - **Rejected on evidence (§5):** it raises lost frames.

### Provenance

- `PoseSource {measured, bridged, coasted}` and a **segment** counter (incremented at every loss onset) are on every
  `VoTrackResult`.
- Both are carried into `runtime::Keyframe` as `pose_source` and `segment` (AUDIT §4: downstream must be able to tell
  a predicted keyframe from a measured one).
- They also appear in `vo_trace.csv`, in `results.yaml` `lost:` (only when a `lost:` section is configured, so v1
  bytes are unchanged), and in the manifest field `lost_handling`.

**The seam for 2.4:** the rotation source while lost is one call. The preintegrated gyro replaces it, and the
translation prior becomes the preintegrated velocity.

## 4. Model selection: what was tried and why it changed

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

## 5. Results (all 11 EuRoC flights; ATE in m; RMS of the causal per-frame trajectory)

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

**Against `LOCAL_SLAM_2.3.md` §8:**
- (2) within the worst of the published stereo class: **9/11**, up from 7/11. Still out: V1_03 0.56 vs 0.07, V2_03
  1.27 vs 0.79.
- (1) better than the 1.5 placeholder: **10/11** (MH_05 0.35 vs 0.26).
- With the same lost handling given to v1, v2 beats it on 9/11 (MH_05, V2_03 lose).

## 6. Open

- Bridged odometry edges keep the uninformative covariance (1 m, 1 rad). The measured per-pair σ (§5) is the input
  for an informative one, which needs a NEES check first.
- The three worst blur gaps (V2_03 frames 1200, 1241, 1775: 23–26°) have too few tracks back to any anchor. That
  needs 2.4's gyro.
- The bridge is computed on every frame for the held-out check (two USAC fits per frame): VO tracking 9.3 → 12.4 ms
  per frame, averaged over the 11 flights. Its timing goes in the
  COOL budget review; it can be limited to lost frames once it is validated.
