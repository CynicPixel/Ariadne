# Design — 2.1: perception swap-in (XFeat keyframes + LK, re-association, SGBM/WLS, retrieval, hard path)

DRAFT 2026-10-17, rev 2.
- **rev 1** was written in the scratchpad before the `feat/2.1-perception` worktree existed. It was recovered verbatim
  after a session crash.
- **rev 2** adds:
  - D56 (lost-frame handling, landed on `feat/phase2-foundations` by the critical-path session) to the baseline;
  - the §4.0 attribution, measured from D56's `vo_trace.csv`;
  - the window-expiry finding that reshapes §4.3.

Every number in §3 is fixed BEFORE measuring any 2.1 code. The §4.0 numbers are measurements of the existing D56 v2
runs, not of anything new.

**Preflight:** literature (§8), OpenCV-native check against the installed build (§7), advisor review. The advisor
reviewed the design twice on 2026-10-17, and its corrections are folded in.

**Ownership (resolved by D56):** the critical-path session owns lost-frame handling. This means:
- the visual rotation bridge, the decaying velocity, `PoseSource`, segments, and `lost.reanchor`;
- 2.1 is measured on top of D56;
- 2.1 changes no lost-handling code. It only adds configs, and the lost-mode re-association call in §4.3.

## 1. Data in → data out

- **In, per frame:**
  - a rectified stereo pair;
  - the LK tracks;
  - the StereoVo window (landmark ids, X_w, birth keyframe, and that keyframe's `PoseSource`).
- **In, at keyframes:**
  - XFeat keypoints and the dense 64-D descriptor map;
  - SGBM disparity, plus the right-matcher pass for WLS.
- **Out, per frame:** VoFrame entries for re-associated landmarks, i.e. existing landmark ids attached to pixels in
  this frame. LK then carries them on.
- **Out, at keyframes:**
  - a descriptor per observed landmark, held in a runtime-side table keyed by landmark id;
  - StereoDepth with a real confidence (§9.3/§9.6);
  - later, a 256-D global descriptor into an ANNIndex.

## 2. Pinned baselines (and a D54 label correction)

### 2.1 D54 v2 (`build/euroc_d53/v2`, D54 tree, v2 configs, seed 1)

| | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ATE v2 (m) | 0.0749 | 0.0751 | 0.0933 | 2.777 | 0.5598 | 0.0777 | 0.1133 | 0.4389 | 0.0825 | 0.1238 | 2.118 |
| lost v2 | 0 | 16 | 0 | 278 | 107 | 4 | 14 | 146 | 20 | 17 | 183 |
| keyframes v2 | 719 | 716 | 767 | 910 | 944 | 556 | 672 | 1159 | 512 | 872 | 1132 |

D54's "MH_04 266, V2_03 181, V1_03 139, MH_05 100" are **v1's** lost frames, taken from the v1 rows of the same
sweep. This is recorded in a new D6x entry; D54 itself is not edited (append-only).

### 2.2 D56 v2 = `bridge` / `every_frame`, τ 0.5 s (`build/euroc_lost/v2_bridge_every_frame`, seed 1)

This is the tree 2.1 is built on, once `feat/phase2-foundations` is merged in.

| | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ATE (m) | 0.0749 | 0.0705 | 0.0933 | 0.3642 | 0.3490 | 0.0704 | 0.1369 | 0.5619 | 0.0706 | 0.1195 | 1.2662 |
| lost | 0 | 16 | 0 | 264 | 98 | 4 | 12 | 137 | 18 | 16 | 184 |
| segments | 0 | 2 | 0 | 4 | 4 | 4 | 12 | 110 | 3 | 11 | 93 |
| vo_track_mean (ms) | 13.3 | 13.2 | 13.5 | 12.4 | 12.6 | 12.8 | 11.9 | 10.7 | 12.7 | 12.3 | 10.5 |

**D56 moved several D54 numbers on its own, before 2.1 adds anything:**
- V1_02 ATE 0.113 → 0.137 (+21%, far outside any plausible seed spread).
- V1_03 ATE 0.439 → 0.562. D56 attributes this to the velocity decay.
- V2_03 lost frames 183 → 184.

## 3. Acceptance, fixed before measuring

**CONFIRMED by the maintainer 2026-10-17** (rev 2 as written; V1_03 held to D54's 0.4389). The brief says "beats the
D54 table"; read literally, the merged tree fails that before 2.1 adds anything (V1_02 above). So:

1. **Failing four** (MH_04, MH_05, V1_03, V2_03), under the 2.1 v2 config: lost frames AND ATE both strictly below
   **min(D54 v2, D56 v2)** per flight. That gives:

   | | MH04 | MH05 | V103 | V203 |
   |---|---|---|---|---|
   | lost < | 264 | 98 | 137 | 183 |
   | ATE < (m) | 0.3642 | 0.3490 | 0.4389 | 1.2662 |

   - V1_03's ATE bar (0.4389, from D54) sits below D56's own 0.562, a regression 2.1 does not control.
   - If the maintainer prefers, V1_03 is held to the D56 row instead, and that choice is stated in the results.
2. **Passing seven:** no regression against the **D56 v2** row (§2.2). The tolerance comes from a measured spread, not
   a guess:
   - Before any new-config run, D56 v2 is re-run on the 7 passing flights with seeds 2 and 3. The seed perturbs USAC
     sampling in PnP, row refinement, and D56's bridge, which brings its own seeded USAC H/E. So the spread has to be
     measured on the D56 config, not D54's.
   - Tolerance per flight = max(|ATE_seed − ATE_seed1|) over the seeds, floored at 2% relative.
   - Lost frames ≤ D56 + the same measured spread.
   - Each flight must also stay within the LOCAL_SLAM §8 class.
   - The tolerance values get written into this section before the first 2.1 run.
   - **Measured 2026-10-17** (`build/a1/D56seed/s{2,3}` against D56 seed 1; local dev box, same binary as the
     §9.1 identity check). Written before any 2.1 run on these flights:

     | flight | ATE s1 / s2 / s3 (m) | max abs. diff | tolerance | ATE must be ≤ | lost s1/s2/s3 | lost must be ≤ |
     |---|---|---|---|---|---|---|
     | MH_01 | 0.0749 / 0.0664 / 0.0808 | 0.0085 (11%) | 0.0085 | 0.0835 | 0/0/0 | 0 |
     | MH_02 | 0.0705 / 0.0897 / 0.0804 | 0.0192 (27%) | 0.0192 | 0.0897 | 16/19/18 | 19 |
     | MH_03 | 0.0933 / 0.1130 / 0.0925 | 0.0196 (21%) | 0.0196 | 0.1130 | 0/0/0 | 0 |
     | V1_01 | 0.0704 / 0.0948 / 0.1172 | 0.0467 (66%) | 0.0467 | 0.1172 | 4/4/4 | 4 |
     | V1_02 | 0.1369 / 0.1597 / 0.1356 | 0.0227 (17%) | 0.0227 | 0.1597 | 12/13/13 | 13 |
     | V2_01 | 0.0706 / 0.0682 / 0.0708 | 0.0025 (3.5%) | 0.0025 | 0.0731 | 18/19/18 | 19 |
     | V2_02 | 0.1195 / 0.1280 / 0.0998 | 0.0198 (17%) | 0.0198 | 0.1393 | 16/17/17 | 17 |

     ⚠️ **The seed only changes USAC sampling, yet ATE moves by up to 66%.** The VO is sensitive to tiny
     perturbations. Two consequences:
     - single-seed verdicts on the failing four (§3.1) are weak evidence;
     - every 2.1 row is re-run on seeds 2 and 3 before it is claimed.
3. **XFeat timing:**
   - The budget is fixed by the size decision (§4.1), scaled from §4p by pixel count, measured on the same host
     (i5-1135G7) with 8 threads and `opencv_dnn_graph`.
   - At 768×480: 15.3 ms × 768/736 = **≤ 16.0 ms median**.
   - Also reported:
     - the 1-thread figure;
     - the amortised per-frame cost. Keyframes are 20–54% of frames on D54 v2, and on D56 every lost frame is also a
       keyframe.
4. **Reporting:**
   - Every result goes in `results.yaml`, and every result is reported against both the D54 and the D56 table.
   - The manifest records `feature_engine`, `dnn_engine` (per model), `xfeat_input_size`, `replenish_cadence`,
     `reassociation` (mode, descriptor policy, store), and the keypoint caps. §7.3 makes the caps manifest fields,
     not constants.

## 4. Components, in build order

### 4.0 Attribution first (no behaviour change)

Re-association helps only when tracks are alive but have no landmark. It does nothing when darkness or blur leaves
nothing to track; that case belongs to 2.4 (IMU) and 2.2 (radiometry). The claim that "the remaining failures are
darkness/blur" was asserted, not measured, after D54.

**Measured (2026-10-17, from D56 v2's `vo_trace.csv`, read-only).** Lost = `status == 2`. The PnP gates are
`min_correspondences` 15 and `min_inliers` 12. Each lost frame is classified as:
- **(a)** fewer than 30 live tracks;
- **(b)** at least 30 tracks but fewer than 15 landmark correspondences;
- **(c)** at least 15 correspondences, but PnP rejects the frame.

| flight | lost | tracks median [min] | corr median | (a) | (b) | (c) | stereo_valid at lost KFs, median |
|---|---|---|---|---|---|---|---|
| MH_04 | 264 | 369 [60] | 13 | 0 | 263 | 1 | 20 |
| MH_05 | 98 | 347 [53] | 2 | 0 | 97 | 1 | 7 |
| V1_03 | 137 | 100 [36] | 10 | 0 | 120 | 17 | 56 |
| V2_03 | 184 | 128.5 [30] | 9 | 0 | 153 | 31 | 22.5 |

- **No lost frame is (a).** The 2-D tracks survive, as D56 found. 633 of 683 lost frames are (b). This is the case
  re-association targets.
- **(c) is reported separately** (V1_03 17, V2_03 31). Those frames have enough correspondences but fail PnP's
  inliers. Re-association does not fix them, and they do not count toward its ceiling.
- **Low `stereo_valid` at lost keyframes** (MH_05 median 7) is why new tracks get no landmark: the right image does
  not match, so the forced keyframes triangulate almost nothing.

**Deficit (15 − corr) over the (b) frames.** This is how many re-associated landmarks a frame would need:

| flight | deficit median | p25 | p75 | frames with deficit ≤ 3 |
|---|---|---|---|---|
| MH_04 | 2 | 2 | 2 | 215 |
| MH_05 | 13 | 13 | 13 | 7 |
| V1_03 | 6 | 3 | 11 | 32 |
| V2_03 | 7 | 4 | 10 | 32 |

- **MH_04** sits two correspondences under the gate for most of its 10 s hover. A small re-association yield moves
  it.
- **MH_05** needs about 13 per frame.

**Window expiry, the finding that reshapes §4.3.**

What D56 does while lost:
- Under D56 `every_frame` reanchoring, **every lost frame is a keyframe** (264/264, 98/98, 137/137, 184/184).
- StereoVo keeps a dead-track landmark only while one of the last `window_size − 1` = 9 keyframes observes it
  (`stereo_vo.cpp`, the retirement loop in `AddKeyframe`).
- So about 9 lost frames into a stretch, every landmark seen before the loss has been erased from `landmarks_`.

The share of lost frames that fall after the 9th lost keyframe of their stretch:

| flight | stretches | length median / max | lost frames after pre-loss landmarks expired |
|---|---|---|---|
| MH_04 | 4 | 27.5 / 201 | 232 (88%) |
| MH_05 | 4 | 4.5 / 85 | 77 (79%) |
| V1_03 | 110 | 1 / 10 | 2 (1%) |
| V2_03 | 93 | 1 / 26 | 23 (12%) |

- **The MH flights are long stretches.** A window-only, healthy-born-only lost mode has an empty candidate set for
  most of their lost frames, and `Adopt(point, old_id)` would fail because `landmarks_.find` no longer finds the
  id.
- **The V flights are many one-frame blips.** Their pre-loss landmarks are still in the window, so the window-only
  lost mode fits.

**Still to measure, once code exists (the ceiling):**
- At the next recovered pose (and at GT for an upper bound), count the healthy-born landmarks that:
  - are not currently tracked;
  - project in-image;
  - have a descriptor-consistent keypoint within r px.
- Count them twice: in the window, and in the out-of-window store (§4.3).
- Report the share of (b) frames where that count ≥ the deficit.
- **If a flight's ceiling is small, it is reported before building**, and that flight's criterion moves to 2.4. That
  move is the maintainer's call, not mine.

### 4.1 XFeatEngine (§7.1, §7.4)

- `LocalFeatureEngine` over the existing `XFeatNet` (D50).
  - `Detect` uses keypoints only.
  - `DetectAndCompute` uses keypoints plus 64-D descriptors.
  - New: `Describe(points)` samples the dense map at arbitrary pixels, which is how upstream samples descriptors.
    It gives LK-tracked points a descriptor at keyframes without detecting them.
- **Input size is a D6x decision with a measurement.**
  - §7.4 says "640×640 first, measure localisation error vs full-res before changing". EuRoC's 752 and the sim's 720
    are not multiples of 32.
  - Candidates: resize to 640×640; pad to 768×480; centre-crop to 736.
  - Measure keypoint localisation error against full resolution (repeatability on EuRoC pairs with GT), then re-time
    at the chosen size.
- **Threading (§7.4):** one `XFeatNet` per thread. Its net cache (`mutable std::map<pair<int,int>, cv::dnn::Net>`)
  is mutable behind a const API, so sharing one instance across threads is a data race.

### 4.2 Track cadence (§7.2, C8)

- `tracker.replenish: every_frame | keyframe`. With `keyframe`, replenishment runs only at keyframes and in
  `DEGRADED` (the recovery path).
- v1 keeps ORB every frame (byte-identical).
- Cadence changes track supply on its own, so it gets its own ablation row (§5).
- Interaction with D56: every lost frame is a keyframe, so under `keyframe` cadence a lost stretch still replenishes
  every frame.

### 4.3 Re-association (PRIORITY; deferred here by LOCAL_SLAM_2.3 §5)

**Mechanism:**
- Re-seed an LK track **with the old landmark id** (`KltTracker::Adopt(point, id)`).
- StereoVo already keys landmarks, observations, the culling exemption and retirement by track id, so the slam core
  needs no new association state for landmarks still in the window.
- StereoVo gets one const accessor: the window's landmarks, each with id, X_w, `born_kf`, and that keyframe's
  `PoseSource`.
- The descriptor table stays runtime-side.

**Poisoning guard (AUDIT §4).**
- While lost, the window holds landmarks triangulated at bridged or coasted poses, and re-associating to them locks
  the error in. Only landmarks whose birth keyframe has `PoseSource::kMeasured` are candidates.
- **D56 already stores the provenance** (`KeyframeState.source`, plus `Landmark.born_kf`). So the guard needs only
  the accessor, with no new provenance hunk.
- `PoseSource::kBridged` landmarks are an ablation (§5). The bridge is good on MH_04, with 1.8° rotation RMSE, so
  excluding them may cost yield.

**Tracking mode (at keyframes).** ORB-SLAM's Track Local Map (§V-D), adapted:
1. Project each in-window, healthy-born landmark with no live track through the current PnP pose.
2. Keep in-image projections with positive depth.
3. Search candidate keypoints within r = max(r₀, 3σ), with σ the projected pose uncertainty.
4. Take the best descriptor distance, then apply Lowe's ratio test, and skip pixels already held by a live track.
5. Accept → Adopt.

**Lost mode (every lost frame, before that frame's PnP).** ORB-SLAM §V-C relocalisation, restricted to the healthy
map. When lost, `covariance_given_map` is the 1 m / 1 rad placeholder, so a projection search is meaningless.
1. Match descriptors only, mutual NN plus ratio test, against the candidate landmarks' descriptors.
2. Run USAC `solvePnPRansac`.
3. Accept → the correspondences enter VoFrame under their landmark ids, and the frame tracks.

**This is the mode that can move the failing flights.**

**Candidate set for lost mode, made necessary by the window-expiry finding (§4.0).** Window-only works for the V
flights, but it is empty for 79–88% of the MH flights' lost frames.

Options:
- **(i) Out-of-window healthy store.**
  - A runtime-side store of (id, X_w, descriptor, birth segment) for healthy-born landmarks. It outlives the window
    and is bounded by count and age.
  - On a lost-mode PnP success, the matched landmarks are put back into StereoVo under their old ids.
  - Reinsertion is a new StereoVo hunk (e.g. `ReinstateLandmarks(ids, X_w)`). StereoVo is a shared file, so this
    is coordinated with the critical-path session.
  - **Risk:** a stale X_w. A landmark that left the window was last BA-refined there and is now fixed. That is
    acceptable for relocalisation, and the BA refines it again once it is back in the window.
- **(ii) `lost.reanchor: gated` (config only, no code).**
  - Gated makes no keyframes while lost, so the pre-loss window stays intact and window-only lost mode has
    candidates throughout.
  - D56 rejected gated because "without landmarks from forced keyframes, recovery waits, and nothing re-associates
    (2.1)". That reason may no longer hold once 2.1 re-associates. It is measured as an ablation row (§5) and touches
    no lost-handling code.
- Plan: build window-only first. It is the smallest hunk, and it is enough for V1_03 / V2_03. Then measure (ii) as a
  config row before building (i). Build (i) only if (ii) does not recover MH_04 / MH_05.

**Landmark descriptor.** The descriptor at the landmark's latest keyframe observation. ORB-SLAM's representative
(medoid) descriptor is an ablation.

**Built descriptor-agnostic, on ORB first**, so the gain is attributable to association, not to XFeat (§5).

### 4.4 SGBM + WLS confidence (§9.3), §9.4 validator, §9.6 contract

- Use `ximgproc::createRightMatcher` + `createDisparityWLSFilter`, and take `getConfidenceMap()` as the u8
  confidence. That sets `confidence_is_placeholder = false`.
- **Disparity is left unfiltered unless a config says otherwise.** The WLS-filtered disparity changes Z, so it changes
  the SGBM gate and every v1 output.
- The right-matcher pass roughly doubles SGBM cost per keyframe. It is measured. Under D56 every lost frame is a
  keyframe, so the cost is reported per frame too.
- **§9.4:** the `SgbmStereo` constructor already refuses a blind zone past `required_min_range_m`. It is checked
  against the spec's full (f, B, numDisparities, Z_min) form, with a unit test per refusal.
- **§9.6:** StereoDepth {disparity CV_16S, Z/σ_Z CV_16F with σ_Z = Z²/(fB)·σ_d (§9.5), confidence CV_8U, valid}.
  Confidence becomes real; its use as a σ_d modulator is a later decision, not 2.1.

### 4.5 ANNIndex retrieval; ALIKED + LightGlue hard path

- **Facts that shape this:**
  - The installed build has no ONNX Runtime, so every ONNX model runs on `ENGINE_NEW` or not at all. Rebuilding
    OpenCV is ruled out: build hash and COOL baseline.
  - `cv::XFeat` is NOT in the pinned 5.0.0 build, although ALGORITHMS §1.0 claims it is. An append-only note goes
    there.
- **Each model gets a timeboxed smoke test first** (SMOKE §4r onwards), with its results written there before
  dependent code:
  - EigenPlaces R50-256 → ONNX → `cv::dnn`;
  - `cv::ALIKED`;
  - `cv::LightGlueMatcher`.

  The export images need `infra/docker/*` changes. Those are a maintainer ask, or a session image
  `ARIADNE_CORE_DEV_IMAGE=ariadne/core-dev:feat-2.1-perception`.
- **LightGlue pairs with ALIKED only.** In DL-VINS-Factory, XF+LG gave no valid monocular trajectory on any dataset and
  0.377 m stereo-inertial on EuRoC, while XF+LK was competitive.
- **ANNIndex** holds the keyframe global descriptors. It is the intra-agent loop-candidate source 2.5 needs.

## 5. Ablation matrix (arch ablation matrix, §49 step 3)

All rows run on the merged D56 tree (bridge, τ 0.5) unless stated.

| row | features | cadence | re-association | reanchor | purpose |
|---|---|---|---|---|---|
| D56 v2 | ORB | every frame | off | every_frame | baseline |
| A1 | ORB | every frame | on, window-only | every_frame | association gain, features fixed |
| A1g | ORB | every frame | on, window-only | gated | does re-association rehabilitate `gated`? (§4.3 ii) |
| A1s | ORB | every frame | on, + out-of-window store | every_frame | only if A1g fails on MH (§4.3 i) |
| A1b | ORB | every frame | on, admits `kBridged`-born | best of above | poisoning-guard cost/benefit |
| A2 | XFeat | keyframe | off | every_frame | feature + cadence gain |
| A3 | XFeat | every frame | off | every_frame | cadence isolated (timing-only row; violates C8, not a candidate) |
| A3r | XFeat | every frame | on, window-only (A1's), XFeat matcher | every_frame | engine gain with A1's association held fixed (added 2026-10-18 after the §9.6 smoke run; violates C8 like A3, so not a candidate config, only evidence of what the engine buys) |
| A1k | ORB | keyframe (`replenish_min_tracks` 30) | on, window-only | every_frame | diagnostic for §9.9: cadence isolated with ORB (added 2026-10-18, before running) |
| A4 | XFeat | keyframe | on (best A1 variant) | best | the 2.1 config |
| A4t | XFeat | keyframe, `replenish_min_tracks` 300 (supply threshold, 75% of target) | on | every_frame | §9.10 cadence fix (added before running) |
| A4g | XFeat | keyframe (30) + Shi-Tomasi gate `replenish_min_eig_quality` 0.01 | on | every_frame | §9.10 engine fix: arch §7.2's image-gradient quality step |
| A4gt | XFeat | keyframe (300) + gate 0.01 | on | every_frame | §9.10 both fixes: the expected candidate |
| A4t-off | XFeat | keyframe (300) | **off** | every_frame | §9.11: is V1_02's A4t failure re-association's? (added before running) |
| A4t-r | XFeat | keyframe (300) | on, XFeat matcher + Lowe ratio 0.75 | every_frame | §9.11: does a ratio test remove it? (added before running) |
| A5 | XFeat | keyframe | on, medoid descriptor | best | descriptor policy |

## 6. Shared-file hunks (for mechanical merges)

| file | hunk |
|---|---|
| `cpp/slam/odometry/stereo_vo.{hpp,cpp}` | const window-landmark accessor (id, X_w, `born_kf`, birth `PoseSource`); only if A1s is built: `ReinstateLandmarks` (coordinated) |
| `cpp/runtime/tools/stereo_vo_runner.hpp` | engine selection; lost-mode call before `vo.Track`; keyframe-mode call after `AddKeyframe`; results/manifest lines. D57's scoring hunk sits near the results writer, so keep these hunks apart from it |
| `cpp/perception/tracking/klt_tracker.{hpp,cpp}` | `Adopt(point, id)` (perception-owned) |
| `configs/perception/*` | new keys, with v1 values that preserve current behaviour |
| `configs/slam/*_v2.yaml` | new keys, v2 only |
| `docs/ALGORITHMS.md` | insert-only: the XFeat/OpenCV 5.0.0 note; a re-association catalogue entry (ORB-SLAM §V-C/§V-D, OKVIS2 §VI-A) |

Not touched: `bundle_adjustment/`, `pose_graph/`, `robust/`, the 2.4 inertial files, `stereo_lk_matcher.*`, D56's
lost-handling code (`rotation_bridge.*`, the lost branches in `stereo_vo.cpp`).

Handover note: `.cache` in this worktree is a symlink. The `.gitignore` pattern `.cache/` matches only directories,
so it shows as `?? .cache`. Do not commit it.

## 7. OpenCV-native check (installed `ariadne/core-dev:dev`, `/opt/opencv5`, 2026-10-17)

| item | status |
|---|---|
| `cv::XFeat` | **absent** (0 hits); our `XFeatNet` (D50) is the path |
| `cv::ALIKED` | present: `create(modelPath, Params{inputSize 640², normalizeDescriptors, engine, backend, target})` |
| `cv::LightGlueMatcher` | present: `DescriptorMatcher` subclass; `setPairInfo(qKpts, tKpts, qSize, tSize)` before `match()` |
| `cv::ANNIndex` | present: `create(dim, DIST_EUCLIDEAN/ANGULAR/…)`, `addItems`, `build(trees)`, `knnSearch(…, search_k)` |
| `ximgproc::DisparityWLSFilter` | present: `createRightMatcher`, `createDisparityWLSFilter`, `getConfidenceMap`, `setLambda/SigmaColor/LRCthresh` |
| dnn engines | `ENGINE_CLASSIC/NEW/AUTO/ORT`; ORT needs `WITH_ONNXRUNTIME`, and `cvconfig.h` has no ORT symbol → **[UNVERIFIED pending `getBuildInformation`]**, treated as absent |
| modules | core dnn features flann geometry imgproc optflow photo ptcloud quality stereo video ximgproc xphoto … (no sfm, no line_descriptor) |
| USAC H/E, `recoverPose(distanceThresh)` | present; D56 uses them (`opencv2/geometry/3d.hpp`). Lost-mode PnP reuses the seeded USAC path |

## 8. Literature (primary sources, read 2026-10-17)

- **ORB-SLAM** (T-RO 2015, arXiv 1502.00956):
  - §V-B: a guided search from the motion model, widened when that fails;
  - §V-C: relocalisation by BoW candidates + PnP-RANSAC + guided search;
  - §V-D: Track Local Map, projecting covisible map points, with a viewing-angle < 60° test, a scale-range test and
    a representative descriptor;
  - §III-C: the representative descriptor is the medoid.
- **ORB-SLAM3** `Tracking.cc` (via D56 / LOST_FRAMES.md §2):
  - RECENTLY_LOST relocalises for 3 s and makes no keyframes. This is the `gated` analogue, and the A1g row tests it
    with re-association present.
- **OKVIS2** (arXiv 2202.09199) §VI-A:
  - matches keypoints to existing 3-D landmarks using both descriptor distance and reprojected distance;
  - DL-VINS-Factory measures it as the best stereo odometry on EuRoC (0.060 m) among the systems it ran.
- **DL-VINS-Factory** (arXiv 2607.01757, preprint, not reproduced):
  - ALIKED+LG −5% EuRoC ATE (mono), −7% (stereo + loop closure) against GFTT+LK;
  - XF+LG fails or is poor; XF+LK is best on the SubT-MRS over-exposure sequence (0.638 m);
  - in darkness, the illumination-trained LET-NET leads; learned features are not universally better;
  - "EuRoC … largely solved under well-lit conditions".
  - ⇒ the EuRoC gain from features alone is expected to be small. The gain should come from association. §4.0 now
    supports this with measurements: 633/683 lost frames have tracks but no landmarks.
- **AirVO** (arXiv 2212.07595) §IV-A:
  - on OIVIO (onboard lamp, tunnels and mines), VINS-Fusion and other KLT/ZNCC/LBD trackers lost track on many
    sequences, while SuperPoint+SuperGlue matching to the last keyframe held;
  - it is the most domain-matched evidence for learned keyframe matching under a carried lamp.
- **MID** (Song et al., ICCV 2021): in very dark scenes, matching benefits from RAW and enhancement → 2.2 / Phase 5,
  not here.
- **VINS-Mono** `feature_tracker.cpp` + `euroc_config.yaml`:
  - CLAHE (3.0, 8×8) before KLT, `equalize: 1` on EuRoC ("if image is too dark");
  - a 2.2 radiometric candidate: listed, not implemented here.
  - Relevant to MH_05's stereo_valid median of 7 at lost keyframes: the dark right image is a radiometric problem.

## 9. Results log (measured; dev box i5-1135G7, shared CPU — timings indicative only)

### 9.1 Identity: re-association code present, section absent (2026-10-17)

`build/a1/identity/s1`, `configs/perception/euroc.yaml` + `configs/slam/euroc_v2.yaml`, seed 1.
- MH_01 and MH_04: `results.yaml` is identical to D56 v2 except the `timing_ms` line, and `estimate_body.tum` is
  identical byte for byte.
- So the 2.1 code changes nothing until a config enables it.

### 9.2 A1 (ORB, lost mode, window-only, measured-born candidates), failing four, seed 1

`build/a1/A1/s1`, `configs/perception/euroc_v2.yaml` + `configs/slam/euroc_v2.yaml`.

| flight | ATE A1 (m) | D56 | bar (§3) | lost A1 | D56 | bar | runs / accepted / relabeled | runs with no candidate | verdict |
|---|---|---|---|---|---|---|---|---|---|
| MH_04 | 0.3670 | 0.3642 | < 0.3642 | 266 | 264 | < 264 | 376 / 33 / 37 | 274 (73%) | **fail** (+0.8%, +2) |
| MH_05 | 0.3119 | 0.3490 | < 0.3490 | 97 | 98 | < 98 | 317 / 16 / 28 | 255 (80%) | pass (lost by 1: within noise until §3's spread is measured) |
| V1_03 | **0.3929** | 0.5619 | < 0.4389 | **91** | 137 | < 137 | 271 / 121 / 1434 | 7 | **pass** (also beats D54's 0.439) |
| V2_03 | **0.9323** | 1.2662 | < 1.2662 | **135** | 184 | < 183 | 287 / 101 / 825 | 22 | **pass** |

- **The split is the one §4.0 predicted from window expiry.**
  - The short-blip V flights have candidates almost every time: V1_03 −35% ATE and −34% lost frames; V2_03 −26% ATE
    and −27% lost frames.
  - The long-stretch MH flights have no healthy candidate in 73–80% of runs, so window-only re-association cannot
    move them. That is the case for §4.3 (ii) `gated` → row A1g, then (i) the store.
- Cost (shared CPU, indicative): 2.9–5.0 ms per triggered frame; 0.8–1.2 ms of Describe per keyframe.
- Seed 1 only: the failing-four verdicts get the same seed-spread treatment as the passing seven before any claim.

### 9.3 A1g (re-association + `lost.reanchor: gated`), MH flights, seed 1 — §4.3 (ii) rejected

| flight | A1g ATE / lost | D56 bridge·gated, no re-association | A1 (every_frame) |
|---|---|---|---|
| MH_04 | 1.0743 / 357 | 1.0742 / 357 | 0.3670 / 266 |
| MH_05 | 0.3566 / 125 | 0.3995 / 125 | 0.3119 / 97 |

- Gated still raises lost frames, with re-association or without. D56's rejection stands.
- **Why MH_04 cannot be re-associated (`reassoc_trace.csv`, A1g):**
  - when a run has candidates, the median is **21** healthy untracked landmarks against **335** orphan tracks and 14
    anchors;
  - the median match count is 5, and the dominant outcome is "no match is an inlier".
- The pre-loss window holds a SPARSE map, because the dark right image barely stereo-matches (§4.0: about 20 valid
  stereo matches per lost keyframe).
- **MH_04's re-association ceiling is low by measurement.** Its loss is a stereo-triangulation problem in the dark: 2.2
  radiometry, or 2.4's IMU through the hover. Per §4.0 this goes to the maintainer before (i) the out-of-window store
  is built, because the store cannot hold landmarks that were never triangulated.

### 9.4 Three seeds: A1 against D56 v2 (local dev box); a retraction

**Failing four** (ATE m s1/s2/s3 | lost s1/s2/s3; median in brackets):

| flight | D56 v2 | A1 | per-seed A1 < D56 (ATE; lost) | §3 bar | median vs bar |
|---|---|---|---|---|---|
| MH_04 | 0.364/0.334/0.427 \| 264/262/266 [0.364/264] | 0.367/0.332/0.393 \| 266/**68/71** [0.367/71] | ✗✓✓ ; ✗✓✓ | 0.3642 / 264 | lost ✓, ATE ✗ (+0.8%) |
| MH_05 | 0.349/0.307/0.327 \| 98/98/101 [0.327/98] | 0.312/0.322/0.514 \| 97/98/102 [0.322/98] | ✓✗✗ ; ✓✗✗ | 0.3490 / 98 | ✗ (no effect) |
| V1_03 | 0.562/0.900/0.739 \| 137/149/143 [0.739/143] | 0.393/0.518/0.635 \| 91/90/91 [0.518/91] | ✓✓✓ ; ✓✓✓ | 0.4389 / 137 | lost ✓, ATE ✗ against D54's single-seed 0.4389 |
| V2_03 | 1.266/0.787/1.014 \| 184/171/186 [1.014/184] | 0.932/0.771/1.040 \| 135/130/132 [0.932/132] | ✓✓✗ ; ✓✓✓ | 1.2662 / 183 | ✓ |

**Passing seven, A1 seed 1 against the §3 tolerances: no regression.**

| flight | MH_01 | MH_02 | MH_03 | V1_01 | V1_02 | V2_01 | V2_02 |
|---|---|---|---|---|---|---|---|
| A1 ATE / lost | 0.0749/0 | 0.0686/19 | 0.0933/0 | 0.0565/1 | 0.1047/3 | 0.0706/17 | 0.1214/7 |
| limit | 0.0835/0 | 0.0897/19 | 0.1130/0 | 0.1172/4 | 0.1597/13 | 0.0731/19 | 0.1393/17 |

V1_02 lost 12 → 3, V2_02 16 → 7, V1_01 4 → 1. MH_02 sits at its limit (19).

**What is robust across seeds:**
- V1_03's lost frames −36% on every seed (137–149 → 90–91), with ATE better on every seed.
- V2_03's lost frames −28% on every seed.
- Fewer lost frames on the passing V flights.

ATE gains are seed-dependent everywhere except V1_03.

**⚠️ Retraction of §9.3's "MH_04's re-association ceiling is low".** That was inferred from seed 1 and A1g. Under A1,
seeds 2 and 3 do not lose the dark hover (frames 214–414) at all: lost 68 and 71, against 266 at seed 1. The
re-association work is near-identical on every seed (31–33 accepted, 36–39 relabels; the first acceptances are at
frames 209, 210 and 215).

What this shows: the hover sits at about 13 correspondences against PnP's 15, and whether it is lost is a knife edge,
which a few re-found landmarks tip on 2 of 3 seeds. MH_04's ATE barely moves either way, because D56's bridge
already handles the hover well. The pre-loss window IS sparse (that measurement stands); the inference that
re-association therefore cannot matter was wrong.

### 9.5 XFeatEngine as built (§4.1), 2026-10-18

What exists:
- `cpp/perception/features/xfeat_engine.{hpp,cpp}`: the §7.1 contract over D50's `XFeatNet`.
  - **One forward per image.** The last image's dense maps are cached. A hit needs identical pixels (`memcmp`
    per row), not the same pointer, so a reused buffer with new content is a miss. On a keyframe, the tracker's
    `Detect`, re-association's `Describe` and the keyframe description share one forward.
  - **Selection (§7.3).** Every NMS survivor with score > 0 that the mask allows is a candidate. The mask applies
    BEFORE selection. Then `GridSelect`, exactly as the ORB engine does. Upstream's global top-k is not used.
  - **`Describe(points)`** samples the dense map with upstream's own bicubic sampler at any sub-pixel position.
    `described = 0` only outside the image.
  - **Input:** `resize` (upstream `preprocess_tensor`: floor each side to 32, bilinear on the float image, keypoints
    × W/Wn as upstream does, ≤ 0.011 px from the half-pixel-exact mapping at 752 → 736) or `pad` (replicate to the
    next multiple of 32, keypoints in the pad dropped). The choice is still the §4.1 measured decision.
  - Not thread-safe (§7.4): one engine per thread.
- `XFeatNet::Detect` is split into `Candidates` (NMS + scores) and `SampleDescriptors` (any pixels), and
  `Forward` also accepts a float image. `Detect` = Candidates + top-k + SampleDescriptors, same arithmetic.
- `perception/features/engine_factory.hpp` (`features.engine: orb | xfeat`, default orb).
- `runtime/tools/feature_engine_glue.hpp`: manifest entries `xfeat_dnn_engine`, `xfeat_input` (e.g.
  `resize:736x480`), `xfeat_weights`, `xfeat_weights_checksum` (sha256 from the export's provenance file, as the
  dataset checksum is read), `xfeat_upstream_commit`. results.yaml gains an `xfeat:` line (forwards, cache hits,
  forward ms median/p90, OpenCV threads). Nothing is written for ORB.
- `configs/perception/euroc_v2_xfeat.yaml`: euroc_v2 + `engine: xfeat`; re-association uses upstream's matcher,
  `XFeat.match` (e92685f): mutual NN, `min_cossim = 0.82` → `max_l2 = √(2 − 2·0.82) = 0.6`, no ratio test
  (`ratio: 1.0` rejects only exact ties). It replenishes every frame: plumbing, not a §5 row until §4.2 lands.

Verified (dev container, shared host):
- **The split changes nothing against upstream.** D50's reference test, both engines: 4096/4096 keypoints at the
  same pixel, min descriptor cosine 1.000000, maps ≤ 8e-6 relative (unchanged from D50).
- **`resize` reproduces upstream's preprocessing.** pip `cv2` 5.0.0 `cv::resize(float, INTER_LINEAR)` vs torch
  2.5.1 `F.interpolate(bilinear, align_corners=False)`, in `ariadne-xfeat-ref:1`, real EuRoC frames 752×480 → 736×480:
  max |diff| 4.5e-3 and 1.5e-3 on a 0–255 scale. That is float rounding, not a convention offset, which would show
  as whole grey levels.
- `tests/unit/test_xfeat_engine.cpp`, 8 tests, all pass:
  - `Describe` at detected keypoints = the detector's descriptors bit for bit at native size, ≤ 1e-4 through the
    752 → 736 round trip;
  - one forward for Detect + Describe + DetectAndCompute, and a miss after one changed pixel;
  - a half mask still gets the full budget, all inside the mask;
  - pad mode never detects in the pad;
  - resize keypoints reach x > 735 in a 752-wide image;
  - Describe refuses only points outside the image;
  - ORB stays the default with no extra manifest/results output;
  - config validation.
- The D50 timing (12–15 ms, 8 threads) was NOT re-measured: the host ran at load 18 with other sessions' containers
  (the reference test printed 61 ms). §3.3 timing is measured on a quiet host only.

### 9.6 First XFeat run end to end: MH_04, seed 1 (smoke config, not a §5 row), 2026-10-18

`euroc_v2_xfeat.yaml` + `slam/euroc_v2.yaml`, `build/a1/A2smoke/s1`. XFeat replenishes every frame and re-association
uses XFeat descriptors, so this combines A3's cadence with re-association on. The point was "does it run".

| | ATE (m) | lost | keyframes | segments |
|---|---|---|---|---|
| D56 v2, s1/s2/s3 | 0.364 / 0.334 / 0.427 | 264 / 262 / 266 | | |
| A1 (ORB + re-assoc), s1/s2/s3 | 0.367 / 0.332 / 0.393 | 266 / 68 / 71 | | |
| **XFeat smoke, s1** | **0.3507** | **49** | 972 / 2032 | 3 |
| §3 bar | < 0.3642 | < 264 | | |

- One seed, on a flight shown to be bimodal (§9.4): **not evidence of a pass**. It earns a 3-seed run of the
  failing four.
- Re-association did little here: 324 runs, 25 accepted, 50 relabels, and 259 runs with no healthy candidate. So the
  gain comes from the engine (track supply in the dark hover), not from re-finding landmarks. That needs checking in
  vo_trace before it is claimed.
- Cost: 2032 forwards (one per frame, as configured), 1290 cache hits; forward 28.3 ms median / 51.5 ms p90 at load
  ~18 (indicative only). 48% of frames are keyframes, against 20–54% on D54 v2.
- The manifest carries `feature_engine: xfeat`, `xfeat_input: resize:736x480`, the weights' sha256 and the upstream
  commit.

### 9.7 A3r (XFeat every frame + A1's re-association with XFeat's matcher), failing four, seeds 1–3, 2026-10-18

`build/a1/A3r/s{1,2,3}`; `euroc_v2_xfeat.yaml` + `slam/euroc_v2.yaml`. **A3r violates C8 (XFeat at frame rate), so
it is evidence of what the engine buys, not a candidate config.** Bars: §3 (rev 2, min(D54, D56) seed 1).

| flight | ATE s1/s2/s3 (m) [median] | lost s1/s2/s3 [median] | bar ATE / lost | per seed ATE ; lost | median verdict |
|---|---|---|---|---|---|
| MH_04 | 0.351/0.365/0.386 [0.365] | 49/45/46 [46] | 0.3642 / 264 | ✓✗✗ ; ✓✓✓ | lost ✓, ATE ✗ by 0.25% |
| MH_05 | 0.219/0.162/0.243 [0.219] | 91/90/92 [91] | 0.3490 / 98 | ✓✓✓ ; ✓✓✓ | ✓ |
| V1_03 | 0.474/0.327/0.403 [0.403] | 32/33/28 [32] | 0.4389 / 137 | ✗✓✓ ; ✓✓✓ | ✓ |
| V2_03 | 1.058/0.597/0.754 [0.754] | 74/76/77 [76] | 1.2662 / 183 | ✓✓✓ ; ✓✓✓ | ✓ |

Against A1 (ORB + the same re-association, §9.4), median lost frames: MH_04 71 → 46; MH_05 98 → 91;
V1_03 91 → 32; V2_03 132 → 76.

- **Lost frames beat the bar on all 12 runs.** MH_05, which ORB re-association did not move at all, improves on
  both metrics on every seed (ATE −33% at the median).
- **MH_04's ATE does not improve.** The hover loss is gone (46 lost frames against D56's 264), but the
  trajectory error stays at D56's level. The remaining error is not in the lost stretch.
- Re-association does real work where landmarks exist: V1_03 150–164 acceptances and V2_03 132–137, against 1–5 on
  MH_05 and 25–28 on MH_04. Without A2 (re-association off), how the gain splits between engine and association is
  not known.
- Keyframes: 958–1331 per flight, about 48% of frames on MH, well above D54's 20–54%. Each lost frame is a keyframe.
- Forward 48–78 ms median at 2 parallel runs on a loaded host. **Not a timing result** (§3.3: quiet host only).
- Passing seven: not run on A3r (not a candidate). The no-regression check runs on the candidate row (A4).

### 9.8 A4 (candidate: XFeat at keyframes + re-association) and A2 (re-association off), failing four, seeds 1–3, 2026-10-18

`build/a1/{A4,A2}/s{1,2,3}`; `build/a1/cfg/euroc_v2_xfeat_{a4,a2}.yaml` (= `euroc_v2_xfeat.yaml` + `replenish:
keyframe`, `replenish_min_tracks: 30`; A2 drops the `reassociation:` section). Same binary as the identity check:
ORB `euroc_v2` on MH_01 and V1_03 is again byte-identical to A1 (results excluding timing, trajectory, vo_trace,
reassoc_trace) after the cadence change.

| flight | A4 ATE s1/s2/s3 [med] | A4 lost s1/s2/s3 [med] | A2 ATE [med] | A2 lost [med] | bar ATE / lost | A4 per seed ATE ; lost | A4 median |
|---|---|---|---|---|---|---|---|
| MH_04 | 0.422/0.321/0.357 [0.357] | 50/46/46 [46] | 0.476/0.432/0.478 [0.476] | 51/50/50 [50] | 0.3642 / 264 | ✗✓✓ ; ✓✓✓ | ✓ |
| MH_05 | 0.209/0.218/0.229 [0.218] | 95/94/95 [95] | 0.239/0.241/0.196 [0.239] | 94/95/94 [94] | 0.3490 / 98 | ✓✓✓ ; ✓✓✓ | ✓ |
| V1_03 | 0.378/0.386/0.649 [0.386] | 60/56/66 [60] | 0.461/0.256/0.420 [0.420] | 127/123/132 [127] | 0.4389 / 137 | ✓✓✗ ; ✓✓✓ | ✓ |
| V2_03 | 0.831/0.810/1.060 [0.831] | 152/148/156 [152] | 0.947/1.123/1.406 [1.123] | 198/184/184 [184] | 1.2662 / 183 | ✓✓✓ ; ✓✓✓ | ✓ |

- **Under 3-seed medians, A4 passes both bars on all four.** Under seed 1 alone (rev 2 as confirmed), MH_04's ATE
  fails: 0.422 against 0.3642. Both verdicts stand until the maintainer rules on rev 3.
- **Engine and association do different jobs** (A2 against A4):
  - XFeat alone removes the MH hover losses (MH_04 lost 50 with or without re-association).
  - Re-association is what cuts the V flights: V1_03 127 → 60, V2_03 184 → 152.
  - Without re-association, V2_03 fails its lost bar (184 against 183) and MH_04's ATE is worse (0.476 against 0.357).
- **Cadence costs on V2_03:** lost 76 (A3r, every frame) → 152 (A4). Fast motion thins tracks between keyframes. It
  still passes; `replenish_min_tracks` (30, = degraded threshold) is the knob, an ablation for later.
- Forwards: 36% (MH_04, MH_05) to 58% (V2_03) of frames, about one per keyframe plus a few off-cadence.
  Amortised cost ≈ 0.36–0.58 × the forward time. Keyframes run 36–58% of frames (lost frames are keyframes).
- Next: A4 on the passing seven (3 seeds) against the §3 tolerances; timing on a quiet host.

### 9.9 A4 on the passing seven: a regression; the engine × cadence diagnostic (written BEFORE its runs)

A4, seeds 1–3, against the §3 tolerances (`build/a1/A4/s{1,2,3}`):

| flight | ATE s1/s2/s3 [med] | limit | lost s1/s2/s3 | limit | verdict |
|---|---|---|---|---|---|
| MH_01 | 0.0921/0.0973/0.0980 [0.0973] | 0.0835 | 0/0/0 | 0 | ATE ✗ on every seed |
| MH_02 | 0.1073/0.0925/0.1139 [0.1073] | 0.0897 | 29/29/28 | 19 | ✗ ✗ |
| MH_03 | 0.1330/0.1061/0.1229 [0.1229] | 0.1130 | 0/0/0 | 0 | ATE ✗ |
| V1_01 | 0.0632/0.0726/0.0566 [0.0632] | 0.1172 | 1/0/1 | 4 | ✓ |
| V1_02 | 0.2580/0.2101/0.1713 [0.2101] | 0.1597 | 3/2/0 | 13 | ATE ✗ |
| V2_01 | 0.1407/0.0678/0.0859 [0.0859] | 0.0731 | 13/12/13 | 19 | ATE ✗ |
| V2_02 | 0.0835/0.1515/0.2331 [0.1515] | 0.1393 | 3/5/4 | 17 | ATE ✗ |

**A4 fails acceptance (§3.2) under any seed rule.** MH_01, MH_03 and V1_02 sit far outside their measured spread.
Rev 3 cannot rescue it.

Free checks on the existing outputs (seed 1):
- **Re-association is not the cause on MH.** MH_01 had 3 acceptances, MH_03 had 0. V1_02 is the exception, with
  30 acceptances and 517 relabels, so it stays a suspect there.
- **Cadence thins tracks moderately**, A1 → A4 on MH_01: median tracks 398 → 336, per-keyframe-interval minimum
  379 → 302, frames under 200 tracks 1.1% → 3.2%. MH_03 behaves the same way.
- BA σ rises 0.303 → 0.315 px (MH_01), 0.311 → 0.323 (MH_03), 0.330 → 0.337 (V2_01, where A4 holds MORE tracks than
  A1). That is consistent with worse points for LK/stereo, but not proof: older tracks under keyframe cadence would
  also raise it.

**Diagnostic 2×2**, re-association on throughout. MH_01, MH_03 and V1_02, seeds 1–3, 18 runs. MH_01 and MH_03 have
no lost frames, so lost handling cannot confound them.

| | every frame | keyframe |
|---|---|---|
| ORB | A1 (§9.4) | **A1k (new)** |
| XFeat | **A3r** | A4 |

Reading, fixed now:
- **A1k regresses, A3r does not → cadence.** Fix in-spec: `replenish_min_tracks` is a DEGRADED trigger (30), not a
  supply threshold. Sweep it, reporting forwards per frame next to each value.
- **A3r regresses, A1k does not → engine.** Fix in-spec: §7.2's "image-gradient quality" step does not exist for new
  tracks (`min_eig_threshold` is LK's own conditioning cutoff). Add a Shi-Tomasi gate (`cornerMinEigenVal` at the
  keypoint) on XFeat replenishment.
- **Both → both fixes.**
- **Neither** (each alone holds, the combination regresses) → an interaction; look at track age and stereo-valid
  fraction before guessing.
- Out of bounds without a §49 decision by the maintainer: "FAST corners for LK, XFeat for descriptors only". It
  contradicts "XFeat primary" and §7.2's "tracked XFeat keypoints".

**2×2 result** (ATE median of 3 seeds; A1 = seed 1, with the D56 3-seed median in brackets):

| ATE (m) | MH_01 (≤ 0.0835) | MH_03 (≤ 0.1130) | V1_02 (≤ 0.1597) |
|---|---|---|---|
| A1: ORB, every frame | 0.0749 (0.0749) | 0.0933 (0.0933) | 0.1047 (0.1369) |
| A1k: ORB, keyframe | 0.0775/0.0939/0.0657 [0.0775] ✓ | 0.1357/0.1533/0.1178 [0.1357] ✗ | 0.1556/0.1774/0.1371 [0.1556] ✓ (by 2.5%) |
| A3r: XFeat, every frame | 0.0663/0.0646/0.0846 [0.0663] ✓ | 0.1539/0.1512/0.1133 [0.1512] ✗ | 0.1547/0.1213/0.1707 [0.1547] ✓ (by 3%) |
| A4: XFeat, keyframe | [0.0973] ✗ | [0.1229] ✗ | [0.2101] ✗ |

Reading (the rule fixed above): **both.**
- Cadence alone regresses MH_03 badly and V1_02 almost to its limit, even with ORB.
- XFeat at frame rate does the same.
- Each alone holds MH_01. The combination fails all three.
- Lost frames are 0–5 everywhere, so this is accuracy, not loss.

So both in-spec fixes, rows A4t / A4g / A4gt.

### 9.10 The two fixes (rows registered in §5 before running), 2026-10-18

- **Supply threshold** (`replenish_min_tracks: 300`, 75% of `target_tracks` 400). Off-cadence replenishment
  whenever tracks fall below 300, not only on collapse (30). The value is [ASSUMPTION], not sourced. Reported with
  forwards per frame, which is its cost.
- **Shi-Tomasi gate** (`replenish_min_eig_quality: 0.01`): `KltTracker::AddTracks` accepts a new track only if
  `cornerMinEigenVal(blockSize 3, ksize 3)` at its pixel is > 0.01 × the image maximum.
  - This is OpenCV 5.0.0 `goodFeaturesToTrack`'s qualityLevel rule (`modules/features/src/featureselect.cpp`:
    `minMaxLoc` then `threshold(eig, maxVal·qualityLevel, THRESH_TOZERO)`), except that the max is over the whole
    image, not the spacing mask.
  - 0.01 is the qualityLevel VINS-Fusion's feature tracker passes, from memory; its source is not in the cache
    [UNVERIFIED].
  - Off (0) for ORB, so ORB rows are unchanged.
  - Test `TrackerCadence.QualityGateKeepsNewTracksOffWeakGradients`: texture at 12% contrast is detected with the
    gate off and rejected at 0.1, while the strong half still seeds.

**Pass condition, fixed before running:** a fix row is worth taking to the full 11-flight × 3-seed sweep only if its
median ATE is within the §3 tolerance on all three of MH_01, MH_03 and V1_02. Otherwise it is reported as a
negative result, and the next step goes to the maintainer.

**Results** (`build/a1/{A4T,A4G,A4GT}/s{1,2,3}`; ORB `euroc_v2` on MH_01 again byte-identical to A1 with the gate
code present). ATE s1/s2/s3 [median] against the §3 tolerance; forwards per frame (median of seeds) = the XFeat
cost:

| row | MH_01 (≤ 0.0835) | MH_03 (≤ 0.1130) | V1_02 (≤ 0.1597) | forwards/frame (MH_01 / MH_03 / V1_02) |
|---|---|---|---|---|
| A4 | 0.0921/0.0973/0.0980 [0.0973] ✗ | 0.1330/0.1061/0.1229 [0.1229] ✗ | 0.2580/0.2101/0.1713 [0.2101] ✗ | 0.17 / 0.26 / 0.34 |
| A4t (supply 300) | 0.0599/0.0627/0.0749 [0.0627] ✓ | 0.1062/0.1273/0.1007 [0.1062] ✓ | 0.1743/0.2150/0.1268 [0.1743] ✗ | 0.25 / 0.37 / 0.67 |
| A4g (gate 0.01) | 0.0596/0.0568/0.0926 [0.0596] ✓ | 0.1398/0.1947/0.1309 [0.1398] ✗ | 0.2160/0.1703/0.2313 [0.2160] ✗ | 0.16 / 0.26 / 0.53 |
| A4gt (both) | 0.0864/0.0828/0.0845 [0.0845] ✗ (+1.2%) | 0.1221/0.1000/0.1124 [0.1124] ✓ | 0.1497/0.1466/0.1231 [0.1466] ✓ | **0.98 / 0.96 / 1.00** |

Lost frames: 0 on MH_01 and MH_03 in every row. On V1_02: A4t 0/0/0, A4g 7/8/8, A4gt 2/2/3.

**Verdict under the pass condition fixed above: no row passes.** Negative result, reported as such.

- **A4gt is not keyframe cadence.** The gate rejects enough detections that tracks never reach 300, so off-cadence
  replenishment fires on 96–100% of frames. It is A3r plus a gate, and it violates C8 like A3r. Its near-pass says
  nothing about a keyframe-cadence config.
- **A4t (the supply threshold alone) fixes both MH flights** at 25–37% forwards per frame (A4: 17–26%) and fails
  only V1_02 (+9%). On V1_02 its forwards rise to 67%.
- **The gate alone helps MH_01 and hurts MH_03/V1_02** (more lost frames on V1_02). It is not a fix on its own.
- **V1_02 suspect, re-association with XFeat's matcher** (not proven):
  - A4t's lost-mode re-association is accepted on every run (29/29, 31/31, 30/30), relabelling 670–738 tracks per
    flight, about 23 per acceptance.
  - ORB's A1 relabels 443 on the same flight with ATE 0.105, so heavy relabelling is not harmful in itself.
  - But the XFeat matcher has no ratio test (upstream's `match`), and a 100% acceptance rate is what aliasing would
    look like.
  - The diagnostic that would decide it, A4t with re-association off or with ratio 0.75 on V1_02, was NOT run: the
    pass condition sends the next step to the maintainer.

### 9.11 V1_02 on A4t: is re-association the cause? (maintainer chose this step, 2026-10-18; reading fixed before running)

V1_02, seeds 1–3: A4t-off (re-association off) and A4t-r (ratio 0.75 on top of XFeat's mutual-NN + cos > 0.82;
ratio 0.75 is ORB-SLAM2 relocalisation's value, already in A1). The tolerance is median ATE ≤ 0.1597 and lost ≤ 13.

- **A4t-r passes V1_02** → A4t-r is the candidate: register it and run the full 11 flights × 3 seeds. It keeps
  re-association, which the V flights of the failing four need (§9.8).
- **Only A4t-off passes** → re-association with XFeat descriptors hurts V1_02 in a way a ratio test does not fix.
  Back to the maintainer, because the failing four need re-association.
- **Neither passes** → V1_02's failure is not re-association's (the engine/cadence again). Back to the maintainer.
