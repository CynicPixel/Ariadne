# Audit — 2.3 windowed BA against the literature, and where the lost frames come from (2026-10-16)

**Question (maintainer):** audit 2.3 against the state of the art. Can we do better with the current features, or must
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

## 1. What was measured (all from this tree, `build/diag`, 2026-10-16)

The 11 flights, v1 (the 1.5 placeholder) and v2 (2.3), as in `LOCAL_SLAM_2.3.md`:

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

## 2. Can we do better with the current features? Yes — and that comes first

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

## 3. The 2.3 backend against the literature

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

## 4. Frame dropping: benign or adversarial?

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

## 5. Decision for the maintainer

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

## Sources

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

## 6. Outcome (2026-10-17, maintainer chose (A); D54)

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
