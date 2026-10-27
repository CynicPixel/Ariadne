# Visual front end: ORB/FAST + LK tracking (shipped), XFeat (cv::dnn, ablation), re-association, SGBM/WLS, retrieval, hard-pair path

DRAFT 2026-10-17, rev 2.
- **rev 1** was written as a draft before the 2.1 branch existed. It was recovered verbatim
  after a crash.
- **rev 2** adds:
  - D56 (lost-frame handling, landed on the Phase 2 foundations branch) to the baseline;
  - the §4.0 attribution, measured from D56's `vo_trace.csv`;
  - the window-expiry finding that reshapes §4.3.

Every number in §3 is fixed BEFORE measuring any 2.1 code. The §4.0 numbers are measurements of the existing D56 v2
runs, not of anything new.

**Preflight:** literature (§8), OpenCV-native check against the installed build (§7), design review. The design review
reviewed the design twice on 2026-10-17, and its corrections are folded in.

**Ownership (resolved by D56):** the Phase 2 foundations work owns lost-frame handling. This means:
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

This is the tree 2.1 is built on, once the Phase 2 foundations branch is merged in.

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

**CONFIRMED 2026-10-17** (rev 2 as written; V1_03 held to D54's 0.4389). The brief says "beats the
D54 table"; read literally, the merged tree fails that before 2.1 adds anything (V1_02 above). So:

1. **Failing four** (MH_04, MH_05, V1_03, V2_03), under the 2.1 v2 config: lost frames AND ATE both strictly below
   **min(D54 v2, D56 v2)** per flight. That gives:

   | | MH04 | MH05 | V103 | V203 |
   |---|---|---|---|---|
   | lost < | 264 | 98 | 137 | 183 |
   | ATE < (m) | 0.3642 | 0.3490 | 0.4389 | 1.2662 |

   - V1_03's ATE bar (0.4389, from D54) sits below D56's own 0.562, a regression 2.1 does not control.
   - If preferred, V1_03 is held to the D56 row instead, and that choice is stated in the results.
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

### 3 rev 3 — re-baselined on D58, 3-seed medians on both sides (decision, 2026-10-18; written BEFORE any D58-tree run)

Why: the D58 merge (`dbbf8d5`) turns on `imu: attitude_only` in `slam/euroc_v2.yaml`, so every 2.1 row now runs
with the gyro while lost. Against D54/D56 bars, D58's gain would be credited to 2.1. Every rev-2 row (§9.2–9.11)
stays as measured, on the D56 tree. Rev 3 supersedes rev 2 for the 2.1 claim.

Baselines, on the merged tree:
- **D58 v2** = `perception/euroc.yaml` + `slam/euroc_v2.yaml`, seeds 1–3, all 11 flights.
  - Seed 1 must reproduce `build/euroc_d58/v2` (D58's own run): results excluding timing, and trajectory.
- **D54-equivalent v2** = `perception/euroc.yaml` + `slam/euroc_v2.yaml` with the `lost:` and `imu:` sections
  removed (absent = coast / every_frame, imu none; D56 and D58 each state that this is unchanged), seeds 1–3, the
  failing four.
  - Seed 1 must equal the D54 table (§2.1). Labelled "D54-equivalent (byte-identical at seed 1)" if it does; if it
    does not, the D54 table's single seed is used and that is stated.

Bars (median = median of seeds 1–3):
1. **Failing four:** the 2.1 row's median lost frames AND median ATE are each strictly below
   min(D54-equivalent median, D58 median) per flight.
2. **Passing seven:** the 2.1 row's median ATE ≤ the worst D58 seed, and at least D58 median × 1.02; its median
   lost frames ≤ the worst D58 seed. Same idea as rev 2 (the measured seed spread is the tolerance), on medians.
3. **Reporting:**
   - every seed is shown next to each median;
   - the seed-1 verdict is reported next to the median verdict;
   - a flight where the medians pass but a seed fails is stated as such (MH_04 is bimodal, §9.4).
4. The timing (§3.3) and reporting (§3.4) rules are unchanged.
5. The bar values are written below when the baselines finish, before the 2.1 candidate (A4t-r, §9.11) runs on this
   tree.

**Rev 3 amendment (2026-10-18, time-gated):** the passing seven run the 2.1 candidate at **seed 1
only**; the failing four keep 3 seeds. The passing-seven limit is still the worst D58 seed (floored at D58 median ×
1.02), now applied to the candidate's single seed. This is weaker evidence than a 3-seed median, and the results say
so.

**Baselines measured** (merged tree `dbbf8d5`; `build/a1/{D58,D54EQ}/s{1,2,3}`):
- D58 seed 1 reproduces `build/euroc_d58/v2` on all 11 flights (results excluding timing, trajectory byte-identical).
- D54-equivalent seed 1 reproduces the D54 table on all four (same ATE and lost frames, trajectory byte-identical),
  so it is labelled "D54-equivalent (byte-identical at seed 1)".

| failing four | D58 ATE s1/s2/s3 [med] | D58 lost [med] | D54-eq ATE s1/s2/s3 [med] | D54-eq lost [med] | **bar: ATE < / lost <** |
|---|---|---|---|---|---|
| MH_04 | 0.3939/0.3266/0.3728 [0.3728] | 265/262/264 [264] | 2.7770/17.2225/1.4703 [2.7770] | 278/317/281 [281] | **0.3728 / 264** |
| MH_05 | 0.3528/0.3189/0.3429 [0.3429] | 101/99/102 [101] | 0.5598/0.2165/0.2274 [0.2274] | 107/105/105 [105] | **0.2274 / 101** |
| V1_03 | 0.4206/0.6274/0.6156 [0.6156] | 141/144/137 [141] | 0.4389/0.5600/0.5529 [0.5529] | 146/147/143 [146] | **0.5529 / 141** |
| V2_03 | 0.6797/0.4898/0.5950 [0.5950] | 171/170/178 [171] | 2.1179/2.0946/1.9482 [2.0946] | 183/181/186 [183] | **0.5950 / 171** |

Note MH_05: D54-equivalent seeds 2 and 3 (0.216, 0.227) beat D58 by far. D54's seed 1 (0.560) was an unlucky draw,
so MH_05's ATE bar is now the D54-equivalent median, 0.2274, tighter than rev 2's 0.3490.

| passing seven (seed 1 of the candidate) | D58 ATE s1/s2/s3 [med] | D58 lost | **limit ATE ≤ / lost ≤** |
|---|---|---|---|
| MH_01 | 0.0749/0.0664/0.0808 [0.0749] | 0/0/0 | **0.0808 / 0** |
| MH_02 | 0.0686/0.0763/0.0705 [0.0705] | 16/19/17 | **0.0763 / 19** |
| MH_03 | 0.0933/0.1130/0.0925 [0.0933] | 0/0/0 | **0.1130 / 0** |
| V1_01 | 0.0867/0.1066/0.1192 [0.1066] | 4/4/4 | **0.1192 / 4** |
| V1_02 | 0.0987/0.1489/0.1388 [0.1388] | 13/13/13 | **0.1489 / 13** |
| V2_01 | 0.0701/0.0653/0.0710 [0.0701] | 18/19/18 | **0.0715 / 19** (median × 1.02 floor) |
| V2_02 | 0.1165/0.1058/0.1182 [0.1165] | 14/16/16 | **0.1182 / 16** |

Candidate: **A4t-r** (§9.11), `build/a1/cfg/euroc_v2_xfeat_a4t_r.yaml` + `slam/euroc_v2.yaml` (D58), launched
2026-10-18 with these bars written. V1_03's ATE bar was filled in when D54-eq s3 finished, before any A4t-r result was read.

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
  move is a project decision, not this design's.

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

### 4.3 Re-association (PRIORITY; deferred here by local-slam.md §5)

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
    is coordinated with the Phase 2 foundations work.
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
  - `cv::XFeat` is NOT in the pinned 5.0.0 build, although algorithms.md §1.0 claims it is. An append-only note goes
    there.
- **Each model gets a timeboxed smoke test first** (SMOKE §4r onwards), with its results written there before
  dependent code:
  - EigenPlaces R50-256 → ONNX → `cv::dnn`;
  - `cv::ALIKED`;
  - `cv::LightGlueMatcher`.

  The export images need `infra/docker/*` changes. Those need approval, or a local image
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
| A4t-off·D58, A1·D58 | XFeat / ORB | keyframe (300) / every frame | off / on | every_frame, D58 gyro | §9.13: which component regresses V2 + MH_02 once the gyro is on (added before running) |
| A5 | XFeat | keyframe | on, medoid descriptor | best | descriptor policy |

## 6. Shared-file hunks (for mechanical merges) — AS BUILT (rewritten 2026-10-19; the planned list is superseded)

| file | hunk |
|---|---|
| `cpp/runtime/tools/stereo_vo_runner.hpp` | include `engine_factory.hpp` (replacing `orb_engine.hpp`), `feature_engine_glue.hpp`, `reassociation_glue.hpp`; `MakeFeatureEngine` replaces `OrbEngine orb` (tracker + re-associator take `*features`); `reassoc.BeforeTrack` before `vo.Track`; `reassoc.AfterKeyframe` + `tracker.Replenish` after `AddKeyframe`; `reassoc_trace.csv` if enabled; `RecordFeatureEngine(m, *features, pcfg)` replaces `m.feature_engine = orb.Name()`; manifest `reassociation` + results lines if enabled; `res << features->ResultsYaml()` (empty for ORB); §10: `std::vector<double> t_conf` after the re-associator, `t_conf.push_back(depth.confidence_ms)` after `sgbm.Compute` when the map is real, `res << StereoConfidenceResults(t_conf)` when non-empty |
| `cpp/slam/odometry/stereo_vo.hpp` | `LandmarkInfo` + const `Landmarks()` accessor (id, X_w, born_kf, birth `PoseSource`) |
| `cpp/perception/tracking/klt_tracker.{hpp,cpp}` | `Relabel(from, to)`; `Replenish(gray)`; step 4 factored into `AddTracks`; the `replenish_min_eig_quality` gate; `<map>`/`<set>` includes. D58's `predicted` seam is untouched |
| `cpp/perception/features/feature_engine.hpp` | pure virtual `Describe(gray, points, described)`; virtual `ManifestFields()` / `ResultsYaml()` (empty defaults) |
| `cpp/perception/perception_config.hpp` | `ReassociationParams` + optional `reassociation:` section; `FeatureParams` `engine`/`xfeat_*`; `TrackerParams` `replenish`, `replenish_min_tracks`, `replenish_min_eig_quality`; §10 `StereoParams` `confidence`, `wls_*` + their validation; all optional, defaults = old behaviour |
| `configs/perception/` | new `euroc_v2.yaml` (the D60 v2 config) and `euroc_v2_xfeat.yaml` (the measured XFeat ablation). No existing config edited |
| `docs/algorithms.md` | insert-only: the ⚠️ cv::XFeat-absent note after the §1.0 class table; the ⚠️ WLS-factory / disp12MaxDiff note after the OpenCV-5 module inventory (D61); §1.3c re-association entry (renumbered from 1.3b at the D58 merge); §1.3c measured result |
| `docs/architecture.md` | "As built" blockquotes: end of §7.2 (D60); after §28.2's manifest field list (D60); end of §9.3 (D61) |
| `docs/decisions.md`, `the development log (archive/dev-history branch)` | appended D60 (approved) and D61; two new top STATUS entries |

New, not shared: `perception/association/landmark_reassociation.*`, `perception/features/{xfeat_engine.*,
engine_factory.hpp}`, `runtime/tools/{reassociation_glue,feature_engine_glue}.hpp`, `xfeat/xfeat_net.*` (D50's, split
into `Candidates`/`SampleDescriptors`, reference test unchanged), `tests/unit/test_{reassociation,xfeat_engine}.cpp`.
Philox site of the re-association PnP: `kMaxSite-16`.

Not touched: `bundle_adjustment/`, `pose_graph/`, `robust/`, the 2.4 inertial files, `stereo_lk_matcher.*`, D56's
lost-handling code (`rotation_bridge.*`, the lost branches in `stereo_vo.cpp`).

Merge of the Phase 2 foundations branch (D57 + D58) into this branch, 2026-10-18 (`dbbf8d5`):
- `stereo_vo_runner.hpp`: include block only; both sides kept.
- `docs/algorithms.md`: D58 and 2.1 both inserted a "§1.3b". D58's (inertial) keeps 1.3b because it is already on
  the shared branch. **The 2.1 re-association entry is now §1.3c.**
- `klt_tracker.*` auto-merged: D58's optional `predicted` homography on `Process` and 2.1's
  `Replenish`/`AddTracks`/quality gate touch different steps.
- The merged tree passes ctest 332/332 (2 fixture tests skipped by env, as before).

Note: `.cache` in a worktree checkout is a symlink. The `.gitignore` pattern `.cache/` matches only directories,
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
- **ORB-SLAM3** `Tracking.cc` (via D56 / local-slam.md#lost-frames §2):
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
  radiometry, or 2.4's IMU through the hover. Per §4.0 this goes to review before (i) the out-of-window store
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
- The D50 timing (12–15 ms, 8 threads) was NOT re-measured: the host ran at load 18 with other containers
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
  fails: 0.422 against 0.3642. Both verdicts stand until rev 3 is ruled on.
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
- Out of bounds without a §49 decision: "FAST corners for LK, XFeat for descriptors only". It
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
negative result, and the next step goes to review.

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
    pass condition sends the next step to review.

### 9.11 V1_02 on A4t: is re-association the cause? (step chosen 2026-10-18; reading fixed before running)

V1_02, seeds 1–3: A4t-off (re-association off) and A4t-r (ratio 0.75 on top of XFeat's mutual-NN + cos > 0.82;
ratio 0.75 is ORB-SLAM2 relocalisation's value, already in A1). The tolerance is median ATE ≤ 0.1597 and lost ≤ 13.

- **A4t-r passes V1_02** → A4t-r is the candidate: register it and run the full 11 flights × 3 seeds. It keeps
  re-association, which the V flights of the failing four need (§9.8).
- **Only A4t-off passes** → re-association with XFeat descriptors hurts V1_02 in a way a ratio test does not fix.
  Back to review, because the failing four need re-association.
- **Neither passes** → V1_02's failure is not re-association's (the engine/cadence again). Back to review.

**Results** (D56 tree, pinned binary md5 `ce812f7d…`, `slam/euroc_v2.yaml` as of D56; `build/a1/A4T_R`, `build/a1/A4T_OFF`):

| row | ATE s1/s2/s3 [median] | lost | re-association runs / accepted / relabeled (s1, s2, s3) | verdict (≤ 0.1597, ≤ 13) |
|---|---|---|---|---|
| A4t | 0.1743/0.2150/0.1268 [0.1743] | 0/0/0 | 29/29/670, 31/31/738, 30/30/692 | ✗ |
| **A4t-r** | 0.1483/0.1714/0.1348 [**0.1483**] | 0/0/0 | 30/30/548, 31/31/564, 30/30/521 | **✓** |
| A4t-off | 0.1577/0.1685/0.1769 [0.1685] | 10/7/8 | off | ✗ (ATE) |

- **A4t-r passes V1_02**, so by the rule above it is the candidate for the full 11 × 3 sweep.
- The ratio test removes 20–25% of the relabels (670–738 → 521–564) without changing how often lost mode is
  accepted. It does not stop the 100% acceptance rate. So the ratio test thins the matches inside each accepted set,
  which is consistent with aliasing; seed 2 still sits above the limit.
- Process note: A4t-off first failed. `sweep.sh` was edited in place while bash was still reading it (bash reads
  scripts incrementally), and the D58 merge then put an `imu:` key into `slam/euroc_v2.yaml`, which the D56 binary's
  strict parser refused. It was re-run on the D56 slam config (`build/a1/cfg/slam_euroc_v2_d56.yaml`, = `53a3c0e`).
- **Re-association is not V1_02's problem; it helps.** Off, ATE gets worse (median 0.1685) and lost frames rise
  0 → 7–10. With XFeat descriptors it needs the ratio test: plain XFeat mutual NN (A4t, 0.1743) is worse than no
  re-association, and the ratio version (A4t-r, 0.1483) is better than both.
- **So the candidate is A4t-r:** XFeat at keyframes, supply threshold 300, XFeat re-association with ratio 0.75.
  The engine/cadence diagnosis (§9.9–9.10) and this one were both measured on the D56 tree. The full sweep has to run
  on the merged tree (D58 `imu: attitude_only` in `slam/euroc_v2.yaml`), against a D58 baseline.

### 9.12 A4t-r on the merged tree (D58 gyro on) against §3 rev 3: FAILS, 2026-10-19

`build/a1/A4TR_D58/s{1,2,3}`; `build/a1/cfg/euroc_v2_xfeat_a4t_r.yaml` + `slam/euroc_v2.yaml` (D58). Failing four
on 3 seeds, passing seven on seed 1 (rev 3 amendment). The host was suspended overnight mid-sweep; the runs resumed
and completed. The progress log was lost in the crash; results were read from the run directories.

| failing four | ATE s1/s2/s3 [med] | bar | lost s1/s2/s3 [med] | bar | per seed ATE ; lost | verdict | forwards/frame |
|---|---|---|---|---|---|---|---|
| MH_04 | 0.3858/0.2829/0.2582 [0.2829] | < 0.3728 | 47/46/48 [47] | < 264 | ✗✓✓ ; ✓✓✓ | ✓ | 0.55 |
| MH_05 | 0.2474/0.2876/0.2183 [0.2474] | < 0.2274 | 93/92/92 [92] | < 101 | ✗✗✓ ; ✓✓✓ | ATE ✗ | 0.53 |
| V1_03 | 0.3619/0.3491/0.2983 [0.3491] | < 0.5529 | 36/35/39 [36] | < 141 | ✓✓✓ ; ✓✓✓ | ✓ | 0.78 |
| V2_03 | 0.8269/0.7120/0.7805 [0.7805] | < 0.5950 | 89/83/82 [83] | < 171 | ✗✗✗ ; ✓✓✓ | ATE ✗ | 0.88 |

| passing seven, seed 1 | ATE | limit | lost | limit | verdict | forwards/frame |
|---|---|---|---|---|---|---|
| MH_01 | 0.0599 | 0.0808 | 0 | 0 | ✓ | 0.24 |
| MH_02 | 0.0964 | 0.0763 | 28 | 19 | ✗ ✗ | 0.30 |
| MH_03 | 0.1062 | 0.1130 | 0 | 0 | ✓ | 0.37 |
| V1_01 | 0.0705 | 0.1192 | 0 | 4 | ✓ | 0.31 |
| V1_02 | 0.1483 | 0.1489 | 0 | 13 | ✓ (by 0.4%) | 0.68 |
| V2_01 | 0.1118 | 0.0715 | 7 | 19 | ATE ✗ (+56%) | 0.53 |
| V2_02 | 0.1362 | 0.1182 | 2 | 16 | ATE ✗ | 0.79 |

**Verdict: A4t-r fails §3 rev 3.** 2 of 4 failing flights pass both bars; 4 of 7 passing flights hold.

What the numbers say:
- **Lost frames improve everywhere, on every seed.** MH_04 264 → 47, V1_03 141 → 36, V2_03 171 → 83, against
  D58's medians. The lost-frame half of the brief is met on all four.
- **ATE splits by environment:**
  - MH_04, V1_03, MH_01, MH_03 and V1_01 improve.
  - All three V2 flights get WORSE than D58: V2_01 0.070 → 0.112, V2_02 0.117 → 0.136, V2_03 0.595 → 0.781 (every
    seed), though V2_03 loses half as many frames.
  - With D58's gyro, a lost frame is bridged well. A2.1 frame that is no longer "lost" is now a PnP pose that can be
    worse than the gyro bridge it replaced. On the D56 tree (visual bridge) A4t-r's V-flight gains were real.
    That is the leading hypothesis, NOT a measurement.
- **MH_05:** A4t-r beats D58 (0.343 → 0.247) but not the D54-equivalent median (0.227). D54 coast scores 0.216 /
  0.227 on seeds 2 and 3: D56/D58's lost handling made MH_05's ATE worse than plain coasting, a regression that
  predates 2.1.
- V1_02, MH_01 and MH_03 at seed 1 are identical to the D56-tree A4t-r runs (§9.11, §9.10). Those flights have no
  lost frames, so the gyro never acts.
- XFeat cost: 0.24–0.88 forwards per frame. The V2 flights sit near frame rate (0.79–0.88), which is close to C8's
  limit in practice.

### 9.13 Which component regresses under the gyro? (chosen 2026-10-19; reading fixed BEFORE running)

Seed 1, merged tree, `slam/euroc_v2.yaml` (D58), on V2_01, V2_03 and MH_02:
- **A4t-off·D58**: XFeat + cadence 300, re-association OFF (`build/a1/cfg/euroc_v2_xfeat_a4t_off.yaml`).
- **A1·D58**: ORB every frame + re-association ON (`configs/perception/euroc_v2.yaml`).

"Holds" = within the rev-3 limit: V2_01 ATE ≤ 0.0715, MH_02 ATE ≤ 0.0763 and lost ≤ 19, V2_03 ATE < 0.5950.

- **A4t-off holds, A1 regresses** → re-association is what regresses once the gyro bridges lost frames. Fix
  direction: re-associate only when it beats the bridge (e.g. stricter verification, or not while the gyro prior is
  available).
- **A1 holds, A4t-off regresses** → the XFeat engine/cadence is the regressor on V2 (texture/motion), independent
  of association.
- **Both regress** → both. **Neither** → an interaction of XFeat with re-association (they only regress together).

One seed: this ranks hypotheses, it does not accept anything.

**Results** (`build/a1/{A4TOFF_D58,A1_D58}/s1`; D58 and A4t-r rows from §3 rev 3 and §9.12):

| seed 1 | V2_01 ATE / lost | V2_03 ATE / lost | MH_02 ATE / lost |
|---|---|---|---|
| limit | ≤ 0.0715 / ≤ 19 | < 0.5950 / < 171 | ≤ 0.0763 / ≤ 19 |
| D58 (ORB, re-assoc off) | 0.0701 / 18 | 0.6797 / 171 | 0.0686 / 16 |
| A4t-r (XFeat + re-assoc) | 0.1118 ✗ / 7 | 0.8269 ✗ / 89 | 0.0964 ✗ / 28 ✗ |
| A4t-off (XFeat, re-assoc off) | 0.0621 ✓ / 8 | 0.7227 ✗ / 144 | 0.1066 ✗ / 27 ✗ |
| A1 (ORB + re-assoc) | 0.0703 ✓ / 17 | 0.5694 ✓ / 130 | 0.0728 ✓ / 16 |

Reading (rule fixed above):
- **V2_03 and MH_02: "A1 holds, A4t-off regresses" → the XFeat engine/cadence is the regressor.**
- **V2_01: "neither" → an interaction** (each alone holds; together 0.112).
- **The §9.12 hypothesis is refuted.** Re-association does not become harmful under the gyro. With ORB it improves
  V2_03 on D58: ATE 0.680 → 0.569, lost 171 → 130 (accepted 103, relabeled 878), and it is neutral on V2_01 and
  MH_02.
- So on the D58 tree, at one seed, **ORB + re-association (A1) is the only configuration that holds on all three**.
  XFeat cuts lost frames hardest (V2_01 18 → 8) but costs accuracy on V2_03 and MH_02 even without re-association.
- One seed. A1·D58 has not been run on the failing four or on the other passing flights.

### 9.14 A1·D58 (ORB + lost-mode re-association, D58 gyro) as the 2.1 candidate (decision, 2026-10-19)

Judged against §3 rev 3 exactly as written, the same bars A4t-r failed, with no change: the failing four on 3 seeds,
the passing seven on seed 1. The seed-1 runs of V2_01, V2_03 and MH_02 come from §9.13 (deterministic, not re-run).

If it passes, it deviates from the locked "XFeat primary features" (arch §0.2, arch §7.2), so it needs a §49
arch decision (D60–D64). XFeat stays config-selectable as the measured ablation (§9.6–9.13). If it
fails, it is reported like A4t-r.

**Results** (`build/a1/A1_D58/s{1,2,3}`), against §3 rev 3 unchanged:

| failing four | ATE s1/s2/s3 [med] | bar | lost s1/s2/s3 [med] | bar | per seed ATE ; lost | verdict |
|---|---|---|---|---|---|---|
| MH_04 | 0.3650/0.3294/0.3720 [0.3650] | < 0.3728 | 266/66/67 [67] | < 264 | ✓✓✓ ; ✗✓✓ | ✓ (bimodal lost, as on D56 §9.4) |
| MH_05 | 0.3142/0.3415/0.3128 [0.3142] | < 0.2274 | 101/99/97 [99] | < 101 | ✗✗✗ ; ✗✓✓ | ATE ✗ |
| V1_03 | 0.3910/0.5986/0.4341 [0.4341] | < 0.5529 | 89/92/95 [92] | < 141 | ✓✗✓ ; ✓✓✓ | ✓ |
| V2_03 | 0.5694/0.5291/0.5345 [0.5345] | < 0.5950 | 130/121/122 [122] | < 171 | ✓✓✓ ; ✓✓✓ | ✓ |

| passing seven, seed 1 | MH_01 | MH_02 | MH_03 | V1_01 | V1_02 | V2_01 | V2_02 |
|---|---|---|---|---|---|---|---|
| ATE / lost | 0.0749/0 | 0.0728/16 | 0.0933/0 | 0.0566/1 | 0.1054/3 | 0.0703/17 | **0.1262**/8 |
| limit | 0.0808/0 | 0.0763/19 | 0.1130/0 | 0.1192/4 | 0.1489/13 | 0.0715/19 | 0.1182/16 |
| verdict | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ATE ✗ (+6.8%) |

**Verdict: A1·D58 fails §3 rev 3 on two items**: MH_05 ATE and V2_02 ATE (seed 1). Everything else holds.
Lost frames fall on 10 of 11 flights; MH_01 and MH_03 stay at 0.

| | failing four passing both bars | passing seven held | misses |
|---|---|---|---|
| A4t-r·D58 (XFeat) | 2 / 4 | 4 / 7 | MH_05 ATE, V2_03 ATE, MH_02, V2_01, V2_02 |
| **A1·D58 (ORB)** | **3 / 4** | **6 / 7** | MH_05 ATE, V2_02 ATE |

About the two misses (facts, not excuses; the verdict stands):
- **MH_05's ATE bar (0.2274) is beaten by no D58-tree config.** D58 alone scores 0.343, A4t-r 0.247, A1 0.314. The
  bar comes from D54-equivalent seeds 2 and 3 (0.216, 0.227), i.e. plain coasting. On MH_05, D56/D58's lost handling
  is worse than coasting, a regression that predates 2.1. Only the D56-tree XFeat rows ever went below it (A4 0.218,
  A3r 0.219).
- **V2_02 is one seed.** D58's own spread there is 0.106–0.118. A1 is 7% over the worst of it. Seeds 2–3 were not
  run (rev 3 amendment), so it is not known whether this is noise.

**Post-hoc check (chosen after the seed-1 verdict): V2_02 seeds 2 and 3.**

| V2_02 | ATE s1/s2/s3 [med] | lost s1/s2/s3 [med] |
|---|---|---|
| D58 | 0.1165/0.1058/0.1182 [0.1165] | 14/16/16 [16] |
| A1·D58 | 0.1262/0.1169/0.1054 [0.1169] | 8/9/4 [8] |

On 3 seeds, A1's V2_02 ATE equals D58's (median 0.1169 against 0.1165, within the 0.1182 limit by 1%), with half the
lost frames. The seed-1 miss is consistent with seed noise. **This check was added after the seed-1 result was
known.** The rev-3 verdict (seed 1: ✗) is reported unchanged next to it. Remaining unexplained miss: MH_05 ATE.

### 9.15 Why MH_05's ATE bar (0.2274, D54-equivalent coasting) is not beaten: a compensating error (2026-10-19)

This investigation was requested before closing out. It is read-only on existing outputs; scripts are in
a scratch directory (`mh05*.py`, numpy, Umeyama SE(3) as in the runner's ATE). The lost-handling code belongs to
the Phase 2 foundations work (D56/D58), so the finding goes there.

1. **Where MH_05 loses frames.**
   - Frames 1–85 (start): frame 0's keyframe has 6 stereo-valid points and creates 2 landmarks.
   - Frames 1242–1249 (D54-eq, 8 frames) or 1242–1245 (D58, 4 frames).
   - The last ~12 frames.
   - There are no short blips.
2. **Start (frames 29–86, the first ground truth).** D58's visual bridge gets rotation right (0.23° error, against
   coasting's 2.0°); both miss translation by 4.4 cm. Not the cause.
3. **The frame-1242 loss, frame before → frame after.**
   - D58's gyro: 0.019–0.021 m, 0.14° relative error.
   - Coasting: 0.31–0.51 m, 4.3–4.5°.
   - D58 is about 25× better.
4. **Local drift, 2 s RPE summed over the flight.** D58 8.39 m / 54.9°, D54-eq 10.48 m / 78.5° (seed 2). Equal
   in every 200-frame bin except the start and the 1242 loss, where D58 is far better.
5. **Each half aligned on its own scores about 0.08 m in every configuration:**

   | | before 1242 | after 1250 | whole | second half under the first half's alignment |
   |---|---|---|---|---|
   | D54-eq s1 / s2 / s3 | 0.081 / 0.081 / 0.137 | 0.081 / 0.086 / 0.084 | 0.560 / 0.216 / 0.227 | 1.073 / 0.403 / 0.381 |
   | D58 s1 / s2 / s3 | 0.081 / 0.083 / 0.078 | 0.081 / 0.086 / 0.084 | 0.353 / 0.319 / 0.343 | 0.674 / 0.607 / 0.655 |
   | A1·D58 s1 / s2 / s3 | 0.067 / 0.084 / 0.078 | 0.079 / 0.086 / 0.081 | 0.314 / 0.341 / 0.313 | 0.603 / 0.650 / 0.599 |
   | A4t-r·D58 s1 / s2 / s3 | 0.082 / 0.082 / 0.080 | 0.059 / 0.063 / 0.056 | 0.247 / 0.288 / 0.218 | 0.469 / 0.549 / 0.404 |

**Conclusion.**
- MH_05's whole-flight ATE is decided by how the two halves are registered across the 1242 loss: the drift
  accumulated before it, plus the error made crossing it.
- Coasting's 0.5 m / 4.3° crossing error happens to oppose the accumulated drift on seeds 2 and 3, which gives 0.216
  and 0.227. On seed 1 the same coasting error compounds it (1.07 m, ATE 0.560: D54's own table value).
- **The 0.2274 bar therefore rewards a compensating error, not better lost handling.** By every local measure
  (crossing error, RPE) D58 is better than coasting.
- Nothing is wrong with D56/D58's lost handling here. That is the message to the Phase 2 foundations work.
- The A1·D58 MH_05 "miss" is a comparison against seed-luck in the baseline.
- The verdict as written stands (rev 3: ✗). Whether to judge MH_05 against D58 (0.3429, which A1 beats on all three
  seeds) or by RPE is a project decision. Changing the bar after the result would be post-hoc, and must be
  reported as such.

## 10. SGBM + WLS confidence (§9.3), §9.4 validator, §9.6 contract — design (2026-10-19, BEFORE code)

### 10.1 What and why

`StereoDepth.confidence` is a placeholder today (all zero, `confidence_is_placeholder = true`). §9.3 requires the
confidence map from `cv::ximgproc::DisparityWLSFilter` and forbids inventing a heuristic. §9.6 names its consumers:
TSDF weighting (§20.1) and the §11.2 stereo cross-check.

- **Data in → data out:** rectified left/right CV_8UC1 → `StereoDepth` with `confidence` CV_8U, 0–255, made real.
- The left disparity is **unchanged** (byte-identical) unless a config asks for the WLS-filtered disparity.
- Consuming the confidence, as a σ_d modulator or a gate, is NOT 2.1 (§4.4).

### 10.2 Literature (primary sources, checked 2026-10-19)

- **WLS smoothing:** Min, Choi, Lu, Ham, Sohn, Do, "Fast Global Image Smoothing Based on Weighted Least Squares",
  IEEE TIP 23(12):5638–5653, 2014. It is the `fastGlobalSmootherFilter` that `DisparityWLSFilter::filter` runs.
  It shapes only the *filtered* disparity, not the confidence.
- **Confidence evaluation protocol:** Hu & Mordohai, "A quantitative evaluation of confidence measures for stereo
  vision", TPAMI 34(11):2121–2133, 2012 [13 in Poggi]. The protocol, quoted from Poggi, Tosi, Mattoccia, ICCV 2017
  pp. 5228–5237, §4:
  - pixels are taken in decreasing-confidence order (5% steps, ties included together);
  - the error rate (|d − d_gt| > τ) of each growing subset is plotted;
  - **AUC** is the area under that curve, with optimum **AUC_opt = ε + (1−ε) ln(1−ε)** (eq. 1), where ε is the
    overall error rate. A random confidence scores AUC ≈ ε.
- **Where WLS's measure sits:** a left-right consistency measure (category 3.3) gated by local disparity variance.
  Poggi Table 1(e), **SGM** on KITTI 2012 (ε = 16.78%):

  | measure | AUC | AUC / ε |
  |---|---|---|
  | best of category 3.3 (UCC) | 0.0735 | ≈ 0.44 |
  | CCNN (learned) | 0.0358 | ≈ 0.21 |
  | optimal | 0.0227 | |

  Learned confidences need a CNN per pixel, outside §9.3. So 2.1 measures where OpenCV's map sits on our data; it
  does not chase the learned SOTA. (Corrected 2026-10-19: an earlier draft quoted sub-table (c), which is MC-CNN.)

### 10.3 OpenCV-native check (pinned contrib source, installed `libopencv_ximgproc.so.5.0.0`)

`modules/ximgproc/src/disparity_filters.cpp`:
- `createDisparityWLSFilter(matcher_left)` **mutates the left matcher**: `setDisp12MaxDiff(1000000)`,
  `setSpeckleWindowSize(0)`, `setUniquenessRatio(0)`. Using it on our `SgbmStereo` would silently change the left
  disparity, the SGBM gate and every v1 output. **Not used.**
- Used instead:
  - `createDisparityWLSFilterGeneric(true)`, with the ROI the native SGBM wiring would set, `Rect(min+num, 0,
    W − (min+num) − max(0, −min), H)`, and `setDepthDiscontinuityRadius(ceil(0.5·blockSize))`;
  - `createRightMatcher(sgbm)`, which builds a NEW SGBM (min disparity −(min+num)+1, uniqueness 0, same P1/P2/mode)
    and does not touch ours.
- **The confidence, as computed:**
  - per pixel, the left-right check: |d_L + d_R(x − d_L)| < `LRC_thresh` (24 in 1/16 px = 1.5 px), else 0;
  - when it passes: min(c_L, c_R) with c = max(1 − 0.001·var₁₆(d), 0), the disparity variance in a box of the
    discontinuity radius;
  - ×255 → **CV_32F**, which we convert to CV_8U;
  - 0 outside the ROI.
- **Deviation:** natively, the confidence is computed from a uniqueness-0 left disparity. Ours comes from our real
  left disparity (uniqueness 15, speckle 200). This is stated in the code; the AUC measures the result either way.
- Invalid SGBM pixels (value −16) are masked to 0 confidence AND-ed with `valid`, never trusted from the LRC.
- `lambda` (default 8000) and `sigma_color` (1.0) affect only the filtered disparity.
- Thread split: `num_stripes = getNumThreads()` at creation. Thread-count independence is a test.

### 10.4 Config (`stereo:` section; all optional, defaults = today)

- `confidence: none | wls` (default `none`: placeholder as now, byte-identical, no right-matcher cost).
- `wls_filtered_disparity: false`: if true, `disparity` is the WLS output. **v2 keeps it false** (it changes Z).
- `wls_lrc_threshold: 24`, `wls_discontinuity_radius: 0` (0 = ceil(0.5·blockSize), the native SGBM rule),
  `wls_lambda: 8000`, `wls_sigma_color: 1.0`. All OpenCV defaults, recorded in the manifest.

### 10.5 §9.4 validator

`ValidateStereoAgainstRig` already refuses a blind zone past `required_min_range_m`. Check it against the spec's
full form, Z_min = f·B / d_max with d_max = min_disparity + num_disparities − 1. One unit test per refusal:
- Z_min > required;
- the disparity range covers the image width;
- num_disparities not a multiple of 16 (an OpenCV requirement);
- and the boundary case (Z_min == required, accepted).

### 10.6 Evaluation on real ground truth: EuRoC Vicon rooms

V1_* and V2_* ship Leica MS50 scans (`mav0/pointcloud0/data.ply`, 3.2 M points, `T_WR` = identity, the Vicon world
frame of the GT poses).
- **Ground-truth disparity at a keyframe:** the scan is put into the rectified left camera with the GT body pose,
  `T_B_C0` and the rectification rotation R1, then z-buffered (nearest point per pixel, 1-px splat). Then
  d_gt = f·B / Z_gt.
- **Frames:** every 40th frame of V1_01, V1_02, V2_01 and V2_02, the flights where GT and the room are clean. The
  difficult flights are reported, not judged.
- **Per frame, on SGBM-valid pixels that have GT:** ε at τ = 1 px (Middlebury-style; EuRoC's disparities are
  small, ≤ 96 px); AUC of the WLS confidence; AUC_opt; AUC of a random order (= ε); AUC of D34's support density
  (the existing in-house measure, as a baseline).
- **Sanity, before any AUC is read:** the median |d_sgbm − d_gt| on SGBM-valid pixels ≤ 0.5 px. Otherwise the GT
  rendering or the alignment is wrong, and AUCs are meaningless.
- **Tools:**
  - a C++ dump tool writes per frame: rectified disparity (CV_16S), confidence and support density, plus intrinsics,
    R1 and timestamp, as raw binaries;
  - a Python script (numpy only) renders GT and computes the curves. [DESIGN] This evaluation was specified but not run; no result is claimed.
  - A manifest goes with the figure directory (§36).

### 10.6b Design review (2026-10-19), folded in before any code

- **Our SGBM already runs a left-right check** (`disp12_max_diff: 1`, uniqueness 15, speckle 200). That is why the
  native wiring turns all three off. So on SGBM-valid pixels, WLS's own check (1.5 px) is mostly redundant, and its
  confidence reduces to the variance term.
  - The unit test asserts on the RAW `getConfidenceMap()`, on pixels SGBM keeps and WLS rejects, not on masked output.
  - The evaluation adds a second row: an SGBM configured as OpenCV intends (uniqueness 0, disp12 1e6, speckle 0),
    scored on the same GT, to show what our deviation costs.
  - A close result against support density is expected.
- **Holes in the GT render:** 3.2 M points leave gaps at short range, through which the z-buffer would show the wall
  behind. Keep a GT pixel only if its depth is within 2% of the minimum depth in its 5×5 neighbourhood. A
  side-by-side image (GT vs SGBM disparity) of one frame is checked before any AUC is read.
- **The disparity evaluated is the pipeline's:** row refinement and exposure matching included. It is dumped from
  inside the runner behind a flag, with `T_B_Crect` exported from C++, so Python never recomposes R1.
- **Cost:** `filter()` always runs the FGS smoothing. If that dominates, re-implement `computeConfidenceMap` exactly,
  with a bit-equality test against `getConfidenceMap()` (the XFeat-reference pattern). Shipped configs stay
  `confidence: none` until a consumer exists.
- §9.4: `num_disparities % 16` is already refused (`perception_config.hpp`); no new hunk for it.
- **Found while testing:** OpenCV 5.0.0 SGBM clamps a non-positive `disp12MaxDiff` to 1 (`stereo/src/stereosgbm.cpp`
  lines 498, 1231, 1640: `params.disp12MaxDiff > 0 ? params.disp12MaxDiff : 1`). "Non-positive disables the check"
  is false in this build: it is the strictest setting. A large value (ximgproc uses 1e6) disables it. Our configs
  use 1, unaffected.

### 10.7 Acceptance (fixed now, before measuring)

1. **Identity:**
   - `confidence: none` → `StereoDepth` byte-identical to today, and all v1/v2 EuRoC results unchanged (unit test
     on disparity/Z/valid, plus one EuRoC run);
   - `confidence: wls` with `wls_filtered_disparity: false` → disparity/Z/valid byte-identical to `none` (unit test).
2. **Correctness:**
   - unit tests on synthetic stereo: confidence is 0 where `valid` is 0, high on a textured plane, low at a
     disparity step and where the left-right check fails (an occlusion band);
   - thread-count independence (1 vs 8 threads, byte-identical).
3. **Quality on EuRoC (median over the judged frames):** the WLS confidence AUC is < 0.75·ε (clearly better than
   random) AND < the support-density AUC. Reported with AUC_opt and AUC/ε next to the literature (category 3.3 on
   SGM ≈ 0.44, CCNN ≈ 0.21, Poggi Table 1(e)). Failing this is a reported
   negative result about OpenCV's map on this data, not a reason to invent a heuristic (§9.3).
4. **Cost:** right-matcher + filter time per keyframe, reported. No gate in 2.1 because nothing consumes it yet.
5. **§9.4:** one passing test per refusal and for the boundary case.

### 10.8 Results (2026-10-19)

Against §10.7:
1. **Identity ✓.**
   - Unit: confidence on, filter off → disparity/Z/σ_Z/valid byte-identical.
   - Pipeline: `build/a1/cfg/euroc_v2_wlsconf.yaml` (v2 + `confidence: wls`) on MH_01 and V2_01 (`build/a1/WLSCONF`)
     → results (excluding timing and the new line) and trajectories byte-identical to A1·D58. The manifest records
     `stereo_confidence: wls/lrc24/radius0/filtered_disparity:false`.
2. **Correctness ✓** (`tests/unit/test_stereo_confidence.cpp`):
   - smooth background and slab interior: > 90% of valid pixels ≥ 200;
   - the first 3 slab columns at the 24 px step: < 10%;
   - occlusion band, SGBM's own checks off (disp12 = 1e6): > 80% valid, yet > 80% confidence 0, which is WLS's
     left-right check at work;
   - 1 vs 8 threads byte-identical.
3. **Quality: UNMEASURED** (decision: deferred). The §10.6 evaluation is designed, not built.
4. **Cost:** right SGBM 23.3 ms + filter 13.5 ms (left SGBM 34.4 ms for scale), best of 5, 8 threads, load 3.8, at
   752×480 / 96 disparities. Pipeline: median 39.7 ms (MH_01, 719 passes), 52.2 ms (V2_01, 512 passes) under
   2-parallel load.
   - The right-matcher pass dominates; it is the left-right check's price.
   - Re-implementing `computeConfidenceMap` to skip the FGS smoothing would save ≤ 14 ms: not done.
5. **§9.4 ✓:** width-coverage refusal and the Z_min == required boundary added. Min-range, wide-baseline and
   multiple-of-16 were already tested.

Shipped configs keep `confidence: none`.


## 11. Place retrieval — EigenPlaces R50-256 + an index (§13, §7.4) — preflight and design (2026-10-19, BEFORE code)

### 11.1 Primary sources (checked 2026-10-19)

- **EigenPlaces:** Berton, Trivigno, Caputo, Masone, ICCV 2023 (arXiv 2308.10832). Official code
  `github.com/gmberton/EigenPlaces`, MIT licence, HEAD `a2969f7` (2026-03-15). Model, from upstream source:
  - `hubconf.get_trained_model("ResNet50", 256)`: weights from
    `github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_256_eigenplaces.pth`;
  - network: torchvision ResNet-50 minus avgpool/fc (`children()[:-2]`), then `L2Norm(dim=1) → GeM(p learned,
    init 3, eps 1e-6: avg_pool(clamp(x, eps)^p)^(1/p)) → Flatten → Linear(2048, 256) → L2Norm`.
  - Upstream evaluates positives within 25 m (outdoor). Test-time resolution and normalisation are to be read from
    upstream's test dataset code at export, NOT assumed. The arch §7.4 table says "224×224 or the repo's inference
    resolution — record it".
- **Index:** `cv::ANNIndex` (OpenCV 5 `features`, "features_annoy") is Spotify's Annoy (random-projection forest),
  vendored in `modules/features/3rdparty/annoy`.

### 11.2 OpenCV-native check (pinned source), and three findings about `cv::ANNIndex`

1. **`addItems` always numbers from 0** (`features/src/annoy.cpp`: `add_item(0…n−1)`). A second call overwrites the
   first batch's ids, so incremental keyframe insertion through this API corrupts the index. It must be rebuilt
   from the full matrix (Annoy is build-once anyway: no adds after `build()`).
2. **(Read from source, not tested: `hardware_concurrency` cannot be varied in this container.) The build is
   multithreaded over `hardware_concurrency()` threads** (`build(trees, -1)`, not settable through
   the wrapper), and **every thread seeds its RNG identically** (`thread_build`: `Random _random(_seed)`). Two
   consequences:
   - with `trees` explicit, threads build DUPLICATE trees, so diversity, and recall, depend on the core count;
   - with `trees = -1`, threads race to a shared `2·n_items` node budget, so the tree count itself is scheduling-
     dependent.
   Replay determinism across machines is not guaranteed. A test decides it.
3. `DIST_ANGULAR` / `DIST_DOTPRODUCT` exist. On unit vectors, cosine ranking equals Euclidean ranking.

**ONNX Runtime is absent from the build** (§4.5), so EigenPlaces runs through `cv::dnn::readNetFromONNX` or not
at all. ResNet-50 uses only Conv/BN/ReLU/MaxPool/Add, and the head adds Clip, Pow, ReduceMean (or AveragePool),
MatMul/Gemm and the ReduceL2/Div of `F.normalize`. All are common importer ops; the smoke test decides.

### 11.3 Proposed design

- **`PlaceRecognizer` (perception):** EigenPlaces R50-256 via `cv::dnn`, one Net per thread (§7.4).
  - Gray EuRoC images are replicated to 3 channels, resized to the recorded inference size, and normalised as
    upstream does.
  - Output: 256-D L2-normalised float.
  - Engine and model sha in the manifest (`place_recognition` is already a schema field).
- **`PlaceIndex` (perception or collaboration/retrieval):** add(keyframe id, descriptor), query(descriptor, k,
  exclude ids) → (ids, cosine). Two backends:
  - **`exact`**: brute force via one `cv::gemm` on unit vectors. Deterministic, optimal recall, ~N×256 FLOPs (a few
    µs at thousands of keyframes).
  - **`annoy`**: `cv::ANNIndex` with explicit trees and seed, rebuilt from the full matrix when stale.
  - **Default `exact`** while N is in the thousands per agent. `annoy` is the persistence/scale ablation the arch
    names. Which one ran is logged.
  - This keeps the arch's index behind the interface. Choosing `exact` as the default is a deviation from "Use
    cv::ANNIndex", so it needs the §49 ritual (catalogue note, decision D62) if adopted.
- **2.1 scope:** descriptors at keyframes plus the index, and the loop-candidate query 2.5 needs: top-k excluding a
  temporal window. No PGO consumer in 2.1.

### 11.4 Smoke test (SMOKE §4r) before dependent code

1. Export in a **local image** (`ariadne-eigenplaces-ref:feat-2.1`, FROM `ariadne-xfeat-ref:1` plus pinned
   torchvision and onnx). Shared `infra/docker/*` is not touched; moving the Dockerfile into the repo needs
   approval.
   - The weights are downloaded once and their sha256 recorded.
   - The export runs `--network none` after that.
   - Outputs: an ONNX file (opset 17) and reference descriptors from upstream's own model on fixed inputs (a EuRoC
     frame and a synthetic one).
2. `cv::dnn` load + forward. **Pass: cosine ≥ 0.9999 against upstream's descriptor on every reference input**
   (D50's acceptance style). Forward time at the inference size: 1 and 8 threads.
3. If the importer fails, the result is recorded as such. Options then: D50's pattern (our graph from weights) or
   ORT (absent → a build change, not allowed).

### 11.5 Evaluation and acceptance (FIXED 2026-10-19 after the design review and SMOKE §4r, before any run)

Design-review points folded in:
- the existing `GLOBAL_DESCRIPTOR` contract (256-D unit vector, cosine) is the interface;
- mean-bit ORB is the baseline;
- the shapes are fixed;
- cost is measured first (SMOKE §4r);
- an offline tool, with no runner hunk, because nothing consumes retrieval in 2.1.

**Samples.** Every EuRoC flight, one cam0 frame per 1.0 s (every 20th image), raw (unrectified) cam0, only frames
with GT within 10 ms. The spacing equals the deployment plan (descriptors ≥ 1 s apart, §4r), so the evaluation
measures what would run.

**Descriptors per sample:**
- **EigenPlaces at 376×240**, the primary (cost), via our C++ `PlaceRecognizer` (cv::dnn ENGINE_CLASSIC, exact per
  §4r);
- **EigenPlaces at 752×480**, the upstream protocol, reported;
- **mean-bit ORB**, the placeholder: `MeanBitDescriptor` of `OrbEngine::DetectAndCompute` with `perception/euroc.yaml`
  features, exactly what `ariadne_sim_collab` computes;
- **random**: the analytic expectation.

**Queries and database (within a flight).** For a query at time t_q, the database is the samples with t < t_q − 30 s.
That is loop closure against the past; 30 s excludes temporal neighbours, slow drift and the MH_04 hover (~10 s).
Stationary stretches therefore never pair with themselves.

**Positive:** camera centres within **1.0 m** AND optical axes within **30°** (GT body pose × `T_B_C0` from
`cam0/sensor.yaml`). Position alone over-counts rooms revisited facing the other way. Only queries with ≥ 1 positive
count.

**Metrics:** recall@1 and @5 (a hit = ≥ 1 positive in the top k by cosine), per flight and pooled over all queries of
the 11 flights.

**Acceptance:**
1. Pooled R@1 of EigenPlaces-376 ≥ mean-bit R@1 + 0.10 (absolute) AND ≥ 2× random R@1.
2. EigenPlaces-376 is within 0.05 of EigenPlaces-752 pooled R@1, or the gap is reported and the size decision is
   revisited.

Distribution shift, stated rather than hidden: the model is trained on outdoor RGB street views (SF-XL) and gets
indoor gray replicated to RGB. CLAHE is the arch's ablation (§1473), not run here.

**Index:** `exact` (one gemm over unit vectors) gives the reference top-k. `annoy` (`cv::ANNIndex`, DIST_ANGULAR,
explicit trees = 10, seed fixed) is measured for:
- (a) recall@k agreement with exact on the same queries;
- (b) repeatability: build twice in one process → identical results.

Cross-machine determinism (core count) is a stated risk read from source, not testable here.

### 11.6 Results (measured 2026-10-19/20; `ariadne_place_eval` on all 11 EuRoC flights, `place_recall_eval.py`)

Run at `abe704b` + the uncommitted retrieval files (dirty tree, recorded in each flight's `manifest.yaml`), image
`sha256:2f3bc8cb…`, EigenPlaces `a2969f7`, weights sha256 `2923c102…`. Raw outputs: `build/a1/place/<flight>/`;
table: `build/a1/place/recall.yaml`. 1325 samples, 297 queries with ≥ 1 positive.

| flight | queries | EP-376 R@1 | EP-376 R@5 | EP-752 R@1 | EP-752 R@5 | mean-bit R@1 | mean-bit R@5 | random R@1 |
|---|---|---|---|---|---|---|---|---|
| MH_01 | 43 | 0.721 | 0.977 | 0.721 | 0.907 | 0.674 | 0.814 | 0.457 |
| MH_02 | 23 | 0.913 | 1.000 | 0.913 | 1.000 | 0.783 | 0.913 | 0.565 |
| MH_03 | 37 | 0.486 | 0.892 | 0.595 | 0.919 | 0.514 | 0.892 | 0.069 |
| MH_04 | 13 | 0.769 | 0.923 | 0.846 | 0.923 | 0.692 | 0.923 | 0.204 |
| MH_05 | 15 | 0.867 | 0.867 | 0.800 | 0.867 | 0.733 | 0.867 | 0.092 |
| V1_01 | 50 | 0.540 | 0.780 | 0.660 | 0.920 | 0.220 | 0.420 | 0.058 |
| V1_02 | 15 | 0.600 | 0.867 | 0.733 | 1.000 | 0.333 | 0.667 | 0.097 |
| V1_03 | 46 | 0.478 | 0.804 | 0.543 | 0.848 | 0.283 | 0.565 | 0.054 |
| V2_01 | 11 | 0.455 | 0.909 | 0.545 | 1.000 | 0.364 | 0.909 | 0.037 |
| V2_02 | 20 | 0.750 | 0.900 | 0.600 | 0.800 | 0.200 | 0.400 | 0.050 |
| V2_03 | 24 | 0.625 | 0.833 | 0.708 | 0.833 | 0.375 | 0.708 | 0.083 |
| **pooled** | **297** | **0.626** | **0.875** | **0.677** | **0.902** | **0.444** | **0.694** | **0.167** |

**Acceptance (§11.5):**
1. **PASS.** EP-376 0.626 ≥ mean-bit 0.444 + 0.10 = 0.544, and ≥ 2 × random 0.167 = 0.334.
2. **MISS, by less than one query.** EP-752 − EP-376 = 201/297 − 186/297 = 15/297 = 0.0505 > 0.05 (the bar allows
   14.85). Paired top-1 outcomes: both 173, only-752 28, only-376 13, neither 83; exact McNemar two-sided
   p = 0.028. That p assumes independent queries; ours are not (samples 1 s apart share a revisit and tend to hit or
   miss together), so it is optimistic. The direction is consistent (752 ahead on 7 flights, behind on 2). Per the
   bar, the size decision is revisited in §11.7, not rounded away.

**Reading the table.**
- The Machine Hall rows have inflated random baselines (MH_01 0.457, MH_02 0.565). Those flights start and end
  hovering over the same take-off area, so a query has many positives. There, mean-bit is close to EigenPlaces.
- The Vicon-room rows are where the learned descriptor earns its cost: V1_01 0.22 → 0.54/0.66, V2_02 0.20 → 0.75/0.60.
- 752 beats 376 on 7 flights, ties on 2 and loses on 2 (MH_05, V2_02).

**Cost** (mean per descriptor; host shared, 2 dumps in parallel, cv threads 8; indicative only):

| | EP-376 | EP-752 | mean-bit |
|---|---|---|---|
| range over flights, ms | 147–205 | 644–853 | 22–44 |

EP-752 costs ≈ 4.3× EP-376 for +0.051 R@1.

**Index (annoy vs exact).** Pooled R@1 identical for all three descriptors, and annoy's top-1 equals exact's on
297/297 queries. N ≤ 181 items per flight is not a stress test of the approximation: this shows the wrapper is
correct (ids, filter, rescoring), not that annoy is needed or that it holds at scale. Bar (b), repeatability, is
covered by the unit test only (`PlaceIndex.AnnoyAgreesWithExactAndRepeats`, 200/200 on random vectors); the EuRoC
dumps built each index once. With N in the hundreds-to-thousands per agent, the exact
gemm is a few µs and is the default proposed in D62.

**Not measured here:** CLAHE (arch ablation); a size between 376 and 752; cross-machine annoy determinism (read
from source as a risk, §11.2).

### 11.7 Size decision, revisited (bar 2 missed) — DECIDED 2026-10-20: (a) 376×240

- **(a) Keep 376×240 as the default.** It is ≈ 4.3× cheaper (≈ 150–200 ms against 650–850 ms here), passes bar 1
  with margin, and misses bar 2 by half a query. 752 stays a config option.
- **(b) Make 752×480 the default.** It follows the upstream protocol and gains +0.051 pooled R@1 (+0.03 R@5). At one
  descriptor per ≥ 1 s it takes most of a core.
- **(c) Measure one size in between first** (for example 560×360: a new export plus one more dump, ≈ 30 min
  wall-clock).

**Scope.** The decision is EuRoC-scoped. The rig is 1280×720 RGB. The exports are fixed at EuRoC's 752:480 aspect,
and `PlaceRecognizer::Preprocess` accepts CV_8UC1 only. Sim needs its own export (the aspect ratio) and an RGB path.
This is a stated limitation and does not block 2.1.

## 12. Hard-pair path — ALIKED + LightGlue (§7.2 "hard pairs only") — preflight, design and acceptance (2026-10-21, BEFORE the evaluation code)

### 12.1 Primary sources

- **ALIKED.** Zhao, Wu, Zhang, Wu, Li, Liu, IEEE T-IM 2023, arXiv:2304.03608, `Shiaoming/ALIKED` (BSD-3). The SDDH
  head samples descriptors at learned deformable positions around each keypoint; the backbone has DCN v1 blocks.
- **LightGlue.** Lindenberger, Sarlin, Pollefeys, ICCV 2023, arXiv:2306.13643, `cvg/LightGlue` (Apache-2.0,
  `eb42fee`). A transformer matcher over two keypoint sets (self- and cross-attention, rotary positional encoding of
  normalised keypoints, a dual-softmax assignment with a matchability score). Adaptive depth and width are speed
  mechanisms.
- **Evaluation protocol.** glue-factory (`cvg/glue-factory@2d17e3b`, the code behind LightGlue's results):
  - pose error = max(rotation error, translation-direction error), the latter taken as min(e, 180 − e) because E
    fixes t only up to sign (`geometry/epipolar.py relative_pose_error`);
  - an estimator failure counts as ∞;
  - AUC@{5, 10, 20}° by `utils/tools.py cal_error_auc` (trapezoid of recall vs error, normalised by the
    threshold).

  Official ALIKED config: `configs/aliked+lightglue-official.yaml`.
- **Locked.** "ALIKED+LightGlue hard pairs only" (arch §0.2, arch C8). Alternatives are noted, not adopted:
  - XFeat's LighterGlue (XFeat repo, CVPR 2024): the frame-rate tier's matcher, not the hard path;
  - dense matchers (LoFTR, RoMa): too heavy for CPU;
  - `raco-aliked` in upstream's feature table: no paper found, not in the pinned ecosystem `[UNVERIFIED]`.

### 12.2 OpenCV-native check and smoke test — SMOKE §4s / §4s-2

- `cv::ALIKED` and `cv::LightGlueMatcher` exist in 5.0.0, run on ENGINE_NEW only, and ship no weights.
- opencv_extra's files: unlicensed export, n16rot, 640×640 square.
- Our export from upstream reproduces it through OpenCV's own classes:
  - ALIKED keypoints 100% within 9.3e-5 px, descriptors within 4e-6;
  - LightGlue match sets identical, scores within 6.4e-5.
- Cost: ALIKED ≈ 0.5–0.6 s per image; LightGlue ≈ 0.6 s per pair at K = 1024, ≈ 1.9 s at K = 2048.

### 12.3 Design

- **`perception::HardPairMatcher`** (`cpp/perception/association/hard_pair_matcher.{hpp,cpp}`). Owns one
  `cv::ALIKED` and one `cv::LightGlueMatcher`. **One instance per thread** (§7.4: `clone()` shares the Net).
  - **`Extract(gray)`** → keypoints (px), descriptors K×128. Gray is replicated to BGR. The image size must equal
    the export size; anything else throws, since sim at 1280×720 needs its own export.
  - **`Match(a, b)`** → DMatch list. Keypoints are normalised as upstream (shift W/2, H/2; scale max(W,H)/2) and
    passed with an empty `Size()`. `normalization: opencv` passes pixels and real sizes, which uses OpenCV's
    per-axis rule (the ablation).
  - **`Name()`** for the manifest, e.g. `aliked_n16_k1024@752x480+lightglue/cv_dnn_new/norm_upstream`.
- **No runtime consumer in 2.1.** 2.5's loop closure and 3.x's inter-agent edges call it. The 1.6 verifier
  (`collaboration/broker/edge_verifier.hpp`) keeps ORB as the ablation baseline.

### 12.4 Evaluation — loop-candidate verification on EuRoC (offline tool `ariadne_loop_verify`)

- **Pairs.** The §11 place samples (11 flights, 1 per second), so the hard path is evaluated on exactly the
  candidates retrieval hands it.
  - For every query with a non-empty database (samples older than 30 s), take its **EP-376 top-5** by exact cosine.
    The false retrievals are the hard negatives.
  - Labels from GT camera poses:
    - **positive:** centres ≤ 1 m AND optical axes ≤ 30° (§11.5's definition);
    - **negative:** centres > 3 m OR axes > 60°;
    - **in between:** excluded from the verification metrics (counted and reported). Overlapping views may
      verify legitimately.
- **Geometry, identical for every row: the 1.6 verifier's own call.** Raw cam0 pixels → `cv::undistortPoints`
  (EuRoC radtan, `cam0/sensor.yaml`, P = K) → `findEssentialMat(pa, pb, K, cv::RANSAC, 0.999, 1.0, 1000, mask)` →
  `recoverPose` (amended after the design review; an earlier draft said USAC_MAGSAC, whose threshold is a sigma
  bound rather than a crisp inlier cut).
  - Read from the pinned `geometry/src/five-point.cpp`: the `RANSAC` flag runs classic OpenCV RANSAC, not USAC.
    The threshold is divided by the mean focal length and is a crisp cut.
  - The registrator seeds its own `RNG((uint64)-1)` on every call, so each pair is deterministic and independent
    of processing order (2 processes) and of `cv::setRNGSeed`.
  - 1.0 px is the verifier's value, fixed a priori, not tuned on the scored set.
  - Score = essential inliers (0 if fewer than 5 matches or no model).
  - The 1.6 PnP stage needs stereo depth and is out of scope here (monocular pairs), stated.
- **Rows (ablation matrix):**

  | row | features | matcher | role |
  |---|---|---|---|
  | A | ORB, `perception/euroc.yaml` features | BF Hamming, ratio 0.8 (the 1.6 verifier's stage 1) | placeholder baseline |
  | B | ALIKED-n16, K = 1024 | mutual NN (L2) | isolates LightGlue |
  | C | ALIKED-n16, K = 1024 | LightGlue, upstream normalisation | **candidate** |
  | D | ALIKED-n16, K = 1024 | LightGlue, OpenCV per-axis normalisation | wrapper convention |
  | E | ALIKED-n16, K = 2048 | LightGlue, upstream normalisation | the official K (subset, below) |

- **Size and cost** (counted from the §11 dumps before any run): **4815 pairs**, of them 676 positive, 2537 negative
  and 1602 in between.
  - ALIKED features are computed once per sample per K and cached in memory: 1325 samples × 0.55 s × 2 K.
  - Rows C and D: about 48 min each at 2 processes. Row E on all pairs would be about 2.5 h, so **E runs on a
    subset declared now:** all positives plus the 500 non-positive pairs with the highest row-C scores.
  - One process per flight, 2 at a time. The tool writes raw per-pair rows (matches, inliers, R, t, ms) and
    Python scores them, as `place_eval` does.

- **Metrics:**
  1. **Verification, judged by geometry** (design review). Every pair has a GT relative pose, so a wrong accept
     is defined by what breaks a pose graph, not by the distance labels. A pair is:
     - **relevant:** positive-labelled and its estimated pose error ≤ 10°;
     - **false:** pose error > 10° or an estimator failure, whatever its label;
     - **neutral:** not positive-labelled but correct (≤ 10°). Overlap that a better matcher rightly verifies is
       excluded rather than penalised.

     Ranked by inliers:
     - **AP_geom** (the bar);
     - recall at ≤ 2 false accepts;
     - recall at zero false accepts (reported only: one pair decides it);
     - the label-only AP (positives vs negatives, in-between excluded), reported.

     Deployable threshold: τ = the smallest inlier count with zero false accepts on the **MH flights**, applied to
     the **V flights** (precision and recall reported).
  2. **Pose** (positive-labelled pairs): glue-factory's error (an estimator failure = ∞) and AUC@5/10/20°, pooled
     and per flight, against GT T_Cb_Ca (p_b = T_Cb_Ca p_a, from T_W_B · T_B_C0).
     - Rotation-only AUC is reported too: EuRoC positives include small baselines, where the translation direction
       is noise for every method.
     - Absolute AUC is **not** compared with the paper's MegaDepth numbers (larger baselines). Rows are compared
       with each other only, on identical pairs.
  3. **Cost**, indicative on a shared CPU, two figures:
     - **warm:** per pair with both keyframes' features cached (LightGlue only), the deployment case;
     - **cold:** two extractions plus the match.
- **Convention test before EuRoC:** a synthetic pair with known T_Cb_Ca (random 3D points, both projected with the
  EuRoC K and distortion) through the exact pipeline (undistort → E → recoverPose → error) must give an error of
  about 0° (< 0.1°). Same for the swapped pair, where the error must be large unless the inverse is used. This
  catches an inverted R or a swapped point order.

### 12.5 Acceptance (FIXED 2026-10-21 after the design review, before any EuRoC run)

1. **C vs A (the placeholder):**
   - pooled pose AUC@10°(C) ≥ AUC@10°(A) + 0.05;
   - AND C ≥ A in per-flight AUC@10° on at least 8 of the 11 flights;
   - AND AP_geom(C) ≥ AP_geom(A).
2. **C vs B (does LightGlue earn its cost?):** if AUC@10°(C) − AUC@10°(B) < 0.02, LightGlue's ≈ 0.6 s per pair is
   not earned on EuRoC. That is reported and goes to review; it is not decided in the doc.
3. **Reported, no bar:**
   - D vs C, which decides the wrapper's default normalisation by measurement;
   - E vs C on E's subset, which decides K;
   - recall at zero and at ≤ 2 false accepts;
   - τ from MH applied to V;
   - label-only AP;
   - rotation-only AUC;
   - warm and cold cost.

### 12.6 Results (measured 2026-10-21; `ariadne_loop_verify` on all 11 EuRoC flights, `loop_verify_eval.py`)

Run at `c91c5cf` + uncommitted 2.1 files (dirty, recorded per flight), image `sha256:2f523507…`, our export (SMOKE
§4s-2). Raw rows: `build/a1/verify/<flight>/`; table: `build/a1/verify/loop_verify.yaml`. 4815 candidate pairs (676
positive, 2537 negative, 1602 in between); row E on its declared subset of 1176 (all positives + the 500
highest-C non-positives, selected before E ran).

| row | AP_geom | AUC@5/10/20° (pose) | rotation AUC@5/10/20° | label-only AP | relevant / false items |
|---|---|---|---|---|---|
| A ORB + ratio (placeholder) | 0.578 | 0.129 / 0.284 / 0.443 | 0.521 / 0.694 / 0.811 | 0.943 | 354 / 3780 |
| B ALIKED + mutual NN | 0.891 | 0.322 / 0.565 / 0.742 | 0.713 / 0.838 / 0.907 | 0.891 | 592 / 2434 |
| **C ALIKED + LightGlue (upstream norm)** | **0.891** | **0.361 / 0.607 / 0.773** | **0.728 / 0.857 / 0.927** | 0.865 | 612 / 1914 |
| D ALIKED + LightGlue (OpenCV per-axis) | 0.912 | 0.354 / 0.602 / 0.773 | 0.729 / 0.854 / 0.924 | 0.867 | 616 / 1918 |
| E K = 2048 (subset) | 0.926 (C on the subset: 0.906) | 0.354 / 0.598 / 0.767 (C: 0.607) | 0.721 / 0.849 / 0.922 | 0.965 | 610 / 77 |

**AP_geom denominator, stated (found in the design review's post-run review).** The scorer as first written counts recall
over each row's OWN correct positives (354 for A, 612 for C). A positive the row gets wrong is then a false item,
not missed recall, so the AP measures how well a row ranks its own successes. The fixed-denominator version (all 676
positive-labelled pairs, the same for every row) is the like-for-like one:

| row | A | B | C | D | E (subset) |
|---|---|---|---|---|---|
| AP_geom, fixed denominator | **0.303** | 0.781 | **0.807** | 0.831 | 0.836 |
| recall at ≤ 2 false accepts, fixed | 0.000 | 0.013 | 0.016 | 0.025 | 0.015 |

**Acceptance (§12.5):**
1. **PASS under both denominators.** AUC@10° C − A = 0.607 − 0.284 = **+0.322** (bar +0.05); C ≥ A on **11/11**
   flights (bar 8); AP_geom C ≥ A: per-row 0.891 ≥ 0.578, fixed **0.807 ≥ 0.303**.
2. **LightGlue earns its cost:** AUC@10° C − B = **+0.042** (bar 0.02). B is close on AP_geom (0.891 both) but
   recovers worse poses.

**Reported, no bar:**
- **D vs C (normalisation): a wash.** AP_geom D 0.912 vs C 0.891, but AUC@10° 0.602 vs 0.607. The default stays
  **upstream**: it is the convention the weights were trained with and the one that reproduces upstream exactly
  (§4s-2). OpenCV's per-axis rule is kept as the ablation.
- **E vs C (K):** on E's subset AUC@10° 0.598 vs 0.607, AP_geom 0.926 vs 0.906, at ≈ 2.8× the match cost.
  - The AP comparison is **biased against C by construction**: the subset's non-positives are the 500 pairs where C
    scored highest, which concentrates C's false accepts.
  - The K decision rests on the unbiased evidence instead: pose AUC on all 676 positives (E 0.598 vs C 0.607) and
    cost. **K = 1024 stays the default.**
- **Zero-false-accept metrics as fixed: ≈ 0 for every row,** and τ_MH = ∞ for A and C. Diagnosed rather than
  re-scored: the highest-scoring "false" pairs are positive revisits with **6–13 cm baselines and < 1° rotation
  error**, for example 671 inliers at 0.13 m (err_t 31.5°) and 668 at 0.06 m (err_t 46.1°). The matches are right;
  an essential matrix cannot fix the direction of a near-zero baseline. The fixed definition (max of R and t) counts
  them false. 44 of the 63 C pairs with ≥ 200 inliers and error > 10° have rotation within 5°.
- **Diagnostic only (defined after seeing the data, NOT a bar):** false = rotation error > 10° or failure. The
  recalls below use the per-row denominator.

  | row | R@FA0 | R@FA2 | τ_MH | V recall at τ_MH | V false accepts |
  |---|---|---|---|---|---|
  | A | 0.072 | 0.160 | 53 | 0.656 | 27 |
  | B | 0.017 | 0.372 | 234 | 0.522 | 13 |
  | C | 0.381 | 0.454 | 314 | 0.485 | 4 |
  | D | 0.278 | 0.416 | 119 | 0.911 | 34 |
  | E | 0.297 | 0.385 | 588 | 0.503 | 7 |

  C is the most precise at zero false accepts. The τ transfer from MH to V is not stable across rows (D's τ = 119
  admits 34 false accepts). An inlier count alone is not a deployable loop-closure gate.
- **Genuine wrong verifications with high confidence: 4 C pairs** (V1_01 #140 and #156, V1_03 #245, V2_02 #303):
  210–264 inliers, rotation error 37–59°. Their cause (repeated or planar structure in the Vicon rooms) is not
  diagnosed. The essential-matrix gate alone does not stop them. The 1.6 verifier's PnP stage with stereo depth
  and the 2.5 degeneracy gate (D71) are the next gates downstream; neither is exercised here.
- **Label-only AP favours A** (0.943 vs 0.865). Observed cause: C correctly verifies many "negative" pairs, those
  3–5 m apart or 60–80° apart that still overlap, with rotation error < 3° and 200–440 inliers. Counted: of the 2537
  negatives, 151 reach ≥ 200 C inliers, and 124 of those have rotation error < 3°. Label-only AP counts every one of them as a false accept. ORB rarely
  verifies them, so it is not penalised. This is the overlap the geometric definition (§12.4) was written to treat
  as neutral.

**Cost** (median over flights; load average 17–20 from other jobs, about 2× the quiet smoke-test figures;
indicative only):

| | ORB | ALIKED K = 1024 | ALIKED K = 2048 |
|---|---|---|---|
| extraction per image | 31 ms | 955 ms | 1122 ms |
| match per pair (warm) | A: 6 ms | B: 20 ms; C: 1216 ms; D: 1220 ms | E: 3352 ms |

Cold pair (two extractions + match) for C ≈ 3.1 s under this load, against ≈ 1.7 s in SMOKE §4s-2. That is a
hard-pair cost: rare pairs only (C8).

## 13. Regenerating the reference exports (fresh checkout; phase1.sh requires both)

`infra/scripts/phase1.sh` fails without `.cache/models/eigenplaces_ref/manifest.yaml` and
`.cache/models/aliked_lg_ref/manifest.yaml`. They were made in this work item (outputs under
`.cache/feat-2.1-perception/models/`, copied to `.cache/models/`).

**One command each (L4, 2026-10-21):** `infra/scripts/models.sh eigenplaces` and `infra/scripts/models.sh aliked-lg`
do steps 1–3 below and then check every output against sha256s **pinned in the script** (not against the freshly
written manifest). `ARIADNE_UPSTREAM` / `ARIADNE_MODELS` redirect the clones and the output. Verified: a full
regeneration into `.cache/feat-2.1-perception/models_regen/` reproduced all 11 EigenPlaces and all 19
ALIKED+LightGlue files **bit-exactly**, the ONNX files included (18 s and 2 min). The steps by hand:

**1. Image** (FROM `ariadne-xfeat-ref:1`, which `infra/scripts/models.sh xfeat` builds). Not in `infra/docker/`
(shared); moving it there needs approval.
```dockerfile
FROM ariadne-xfeat-ref:1
RUN pip install --no-cache-dir --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu \
      torchvision==0.20.1+cpu \
 && pip install --no-cache-dir --root-user-action=ignore onnx==1.17.0 \
 && python -c "import torch, torchvision; assert torch.__version__ == '2.5.1+cpu', torch.__version__; assert torchvision.__version__ == '0.20.1+cpu', torchvision.__version__"
```
`docker build -t ariadne-eigenplaces-ref:feat-2.1 -f <that file> .` (built here as `sha256:115b188c…`).

**2. Upstream sources and weights** (the network is used only here; check every sha256):
```sh
U=.cache/upstream; mkdir -p $U/weights_ep $U/weights_aliked
git clone https://github.com/gmberton/EigenPlaces $U/eigenplaces_src && git -C $U/eigenplaces_src checkout a2969f71d5ea31017443af490b15273ca4c50af1
curl -L -o $U/weights_ep/ResNet50_256_eigenplaces.pth https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_256_eigenplaces.pth
#   sha256 2923c10295563ea15efa1571b499c67a7aab4d1ee6e19eba5e8b92756dc9d7e7
git clone https://github.com/cvg/LightGlue $U/lightglue_src && git -C $U/lightglue_src checkout eb42fee2d71449efb0aa5c10549752b5d75384d8
curl -L -o $U/weights_aliked/aliked-n16.pth https://github.com/Shiaoming/ALIKED/raw/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth
#   sha256 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2
curl -L -o $U/weights_aliked/aliked_lightglue.pth https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth
#   sha256 d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
```

**3. Exports** (offline, `--network none`; each script refuses to write unless its own upstream check passes):
```sh
E=.cache/datasets/euroc; C=$PWD/.cache; IMG=ariadne-eigenplaces-ref:feat-2.1
docker run --rm --network none -u $(id -u):$(id -g) -v $PWD/python/model_tools:/tools:ro \
  -v $C/upstream/eigenplaces_src:/ep:ro -v $C/upstream/weights_ep:/w:ro -v $C/datasets/euroc:/euroc:ro \
  -v $C/models/eigenplaces_ref:/out $IMG python -I /tools/eigenplaces_reference.py --upstream /ep \
  --weights /w/ResNet50_256_eigenplaces.pth --out /out --size 752x480 --size 376x240 \
  --image euroc_mh01_f200=/euroc/MH_01_easy/mav0/cam0/data/1403636589713555456.png \
  --image euroc_v101_f500=/euroc/V1_01_easy/mav0/cam0/data/1403715298212143104.png \
  --upstream-commit a2969f71d5ea31017443af490b15273ca4c50af1
docker run --rm --network none -u $(id -u):$(id -g) -v $PWD/python/model_tools:/tools:ro \
  -v $C/upstream/lightglue_src:/lg:ro -v $C/upstream/weights_aliked:/w:ro -v $C/datasets/euroc:/euroc:ro \
  -v $C/models/aliked_lg_ref:/out $IMG python -I /tools/aliked_lightglue_reference.py --upstream /lg --weights /w \
  --out /out --size 752x480 --k 1024 --k 2048 \
  --image mh01_f200=/euroc/MH_01_easy/mav0/cam0/data/1403636589713555456.png \
  --image mh01_f205=/euroc/MH_01_easy/mav0/cam0/data/1403636589963555584.png \
  --image v101_a=/euroc/V1_01_easy/mav0/cam0/data/1403715282262142976.png \
  --image v101_b=/euroc/V1_01_easy/mav0/cam0/data/1403715384262142976.png \
  --pair mh01_f200,mh01_f205 --pair v101_a,v101_b \
  --upstream-commit eb42fee2d71449efb0aa5c10549752b5d75384d8 --aliked-commit 683d7c65197395c0b3f01ebe76e1084a27e73a65
```

**4. Check:** `ARIADNE_EIGENPLACES_REF=… ARIADNE_ALIKED_LG_REF=… ARIADNE_REQUIRE_MODELS=1 test_opencv
--gtest_filter='*EigenPlacesReference*:PlaceRecognizer*:AlikedLightGlueReference.*:HardPair.*'` must pass.
Expected output hashes are in each `manifest.yaml` and in SMOKE §4r / §4s-2. A torch-side re-run should reproduce
them bit-exactly: 1 thread, deterministic algorithms.

## 14. Sim-rig exports (L3, D90) — 1280×720 RGB

The rig is 1280×720 RGB (arch §1). The D62/D63 exports are fixed at EuRoC's 752×480 gray, so the sim needs its own.
The method is the same as D62/D63: our export from the pinned upstream, exact against upstream PyTorch on recorded
frames.

**Frames.** Four `left_rgb` frames of the phase-1 recording, agent 0, frames 200/205/700/710, as the pipeline sees
them (lens applied, no degradation):
```sh
docker run --rm -u $(id -u):$(id -g) -v $PWD:/work -v <REC>/raw:/rec:ro -v <OUT>:/out ariadne/core-dev:dev \
  /work/build/docker/cpp/runtime/ariadne_frame_dump --rec /rec/rec --scene /rec/scene --agent 0 \
  --frames 200,205,700,710 --out /out
```
`<REC>` = `evaluation/runs/gates/phase1_thin_e2e/sim_record/ci_tiny_0-20261001T133914Z`. Create `<OUT>` first; a directory docker
creates is root-owned. PNGs are written in the BGR file convention, so the reader converts to RGB explicitly.
`models.sh` sha256-checks the four files, then:
```sh
ARIADNE_SIM_FRAMES=<OUT> infra/scripts/models.sh eigenplaces-sim   # -> $MODELS/eigenplaces_sim_ref
ARIADNE_SIM_FRAMES=<OUT> infra/scripts/models.sh aliked-lg-sim     # -> $MODELS/aliked_lg_sim_ref
```
**Check:** `ARIADNE_EIGENPLACES_SIM_REF=… ARIADNE_ALIKED_LG_SIM_REF=… ARIADNE_REQUIRE_MODELS=1 test_opencv
--gtest_filter='*SimRig*:*SimRgb*:AlikedLightGlueSimReference.*'`.

**Channel order** is where this can silently go wrong. Upstream EigenPlaces and ALIKED take RGB, `cv2.imread` returns
BGR, and `cv::ALIKED` swaps (blobFromImage swapRB). Each wrapper therefore has an RGB-named entry point taking RGB
order. The tests feed the BGR-ordered frame on purpose and assert the result changes: the input moves by up to 0.77
and the cosine drops to 0.996–0.998, and ALIKED keeps 0/1024 keypoints exactly.

**Results:** SMOKE §4t, exact on every check. **Open:**
- EigenPlaces' sim default size: 640×360 is proposed, retrieval on sim views is unmeasured (3.1);
- one scene, weakly coloured;
- quiet-host cost (L5).
