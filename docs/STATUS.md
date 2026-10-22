# Status

Dated entries. Newest first. Phase tracking checkboxes live in `ARIADNE_PHASED_PLAN.md`.

---

## 2026-10-23 — Phase 5 (A2): GPU dark-render session running; 5.5 worlds + sweep rig; TUM fr1 target sourced

- **GPU session (D137), maintainer-approved:** g4dn.xlarge spot at $0.0858/h, instance `i-09d36054dbc53dab8`, cap
  150 min. It renders `ci_tiny_dark` (lamp) and `ci_tiny_dark_fixed` (fixed lights). The NVIDIA EGL smoke runs first.
- **5.5 (D138):** `confined_v1` (20/20 accepted) and one-pass baseline-sweep cameras; v1 output byte-identical.
- **TUM (D86's condition):** the fr1/xyz target is sourced from Kerl et al. IROS 2013 Table III: 0.014, was 0.017
  [UNVERIFIED]. On `feat/2.6-tum` at `90d5fc8`; verdict unchanged.
- **TUM fr1 (D136):** the scoring-side diagnosis rules out a GT clock offset, a depth scale bias and rotation drift. The
  RGB/depth-offset rerun follows the 2.6 merge.

## 2026-10-22 — Phase 5 (A2): render probe, lamp calibration, Stage 3, controller, dust options (D130–D135)

- **Probe (SMOKE §4u):**
  - the sim RGB is sRGB-encoded (D130: D gets an opt-in decode);
  - the lamp falls off as 1/(c′ + r²), and the shipped rig was nearly flat;
  - SDF spot angles are full-cone;
  - the lobe is flat-topped;
  - shadows work on llvmpipe (D131).
- **Built:**
  - `ci_tiny_dark` + `rig_stereo_v2` (v1 worldgen byte-identical);
  - lamp calibration executed (D133);
  - Stage 3 albedo proxy (D134);
  - exposure + lamp co-controller, Stage 4, replay `lamp_power` (D132);
  - L9 dust options (D135).
- **Tests:** test_numerical Degradation 25/25 (golden digest unchanged), test_unit Config/Degrad 20/20, test_opencv
  ExposureControl/Falloff/DegradationSrgb 11/11.
- **Not done:**
  - runner wiring: waits on the 2.6/L2 merge, which touches the same seams;
  - dark renders, which need GPU approval;
  - 5.5;
  - 5.6, which is blocked: OIVIO is offline.
- **Caveat (D130):** the 2.2 radiometric rows ran D on sRGB-encoded sim values. EuRoC's encoding (the 3.1 rows) is
  `[UNVERIFIED]`.

## 2026-10-22 — Phase 5 (DVE) opens on `feat/ph5-dve`: session A2 claims D130–D149

- **Decision IDs D130–D149 are claimed by session A2** for Phase 5 (5.1–5.6 and the exit gate), the 3.1 sim dark
  retrieval row and the TUM fr1 time-offset follow-up. D100–D109 stay A2's for 2.6/L2/3.3 (D100–D102 used).
- Branch `feat/ph5-dve` from `main` (f8c7030). Design preflight first (literature, OpenCV-native check, review).

## 2026-10-22 — 2.5 close-out (session B): certificate (D74), aliasing ramp breakpoint (D75), joint-graph entry (D76)

- **Decision IDs D75–D79 are claimed by session B.** D75 and D76 are used; D77–D79 are free again after this merge.
- **D74 certificate, behind `loop_closure.certificate` (off).**
  - Certified: low-noise graphs 10/10; garage (f 1.26252 vs SE-Sync Table 1's 1.263, f − f* ≤ 0.024 %); tinyGrid3D;
    smallGrid3D.
  - Not certified: a stationary non-global point, and a perturbed point.
  - **Open:** the large SE-Sync graphs (sphere, torus, grid3D, cubicle, rim) are not run yet: heavy, the maintainer is
    asked first.
  - **Gate wording:** with planted outliers kept in the cost, 6/10 certify, so "not certified on planted-outlier
    graphs" holds for only 4/10. This is reported, not tuned (PHASE2_PLAN §3). After GNC, 10/10 certify.
- **D75 outlier ramp with perceptual-aliasing groups.**
  - PCM → GNC breakpoint: **90 %** (groups of 10), **70 %** (two growing groups). That is PCM's maximum-clique limit.
  - GNC alone: no breakpoint, at ~10× the runtime.
  - Evidence in `evaluation/runs/2.5/outlier_ramp/`. The generator is reusable for 3.5.
- **D76 for main.** `RobustJointSolve` (`cpp/slam/robust/joint_robust.hpp`) runs PCM → chordal re-init → GNC on a
  JointId graph; `GravityOptions::shared` gives one gravity for all agents (the yaw-only gauge). Handoff:
  `ROBUST_BACKEND_2.5.md` §11.

---

## 2026-10-21 — A1 (perception) opens its register close-out; D90–D95 claimed

- **Scope (COMPLETION_REGISTER, owner A1), in order:** D62 approval, L4 `models.sh eigenplaces|aliked-lg`, L3
  sim-rig exports (1280×720 RGB), L6 per-thread `Net` contract, L5 timings, then 3.1 (CLAHE retrieval split) and 3.6
  (descriptor compression rows).
- **Decision IDs D90–D95 are claimed by A1** for L3, L6, 3.1 and 3.6.
- **D62 approved** by the maintainer (DECISIONS 2026-10-21): exact index default, `cv::ANNIndex` the scale ablation.
- **L4 done:** `infra/scripts/models.sh eigenplaces | aliked-lg`. Each clones and fetches pinned sources,
  sha256-checks the weights, exports offline, then checks every output against sha256s pinned in the script. The
  export image is built inline from a heredoc (`ariadne-model-ref:1`, FROM `ariadne-xfeat-ref:1`), so
  `infra/docker/` is untouched. A full regeneration into `.cache/feat-2.1-perception/models_regen/` was **bit-exact**:
  11/11 and 19/19 files, ONNX included. That regeneration ran with the L3 RGB additions already in the scripts; the
  gray path is unchanged by them. `models.sh test` now builds with `ARIADNE_BUILD_JOBS` (default 2), not 8.
- **L3 done (D90, draft on the EigenPlaces sim default size, 640×360 proposed):** sim-rig exports at 1280×720 RGB,
  exact against upstream on four recorded `left_rgb` frames; regeneration bit-exact; RGB entry points with a
  channel-swap guard (SMOKE §4t, PERCEPTION_2.1 §14).
- **L6 done (D91):** `dnn::PerThread<T>` in every cv::dnn wrapper. The two-thread test found that concurrent
  ENGINE_NEW forwards sharing OpenCV's default pool drift at the 1e-5 level, so the contract's second rule is OpenCV
  pool = 1 when several callers share a wrapper. Under it the test is byte-identical, 5/5 repeats.
- **3.1, EuRoC part (D92):** CLAHE on the retrieval path is built behind `enhance` (default `none`). On EuRoC it has no
  measurable effect (ΔR@1 ≤ 0.013, n.s.) under clean and two darkened-proxy conditions. The darkened proxy is
  *easier* than clean (p = 0.023, cause unverified), so the dark question stays with the §31.5 sim row, which is open.
- **L5 done:** quiet-host timings, single process, p50/p95 over 30 reps after 3 warm-ups. Measured on a fresh
  **g4dn.xlarge** (Xeon Platinum 8259CL 2.5 GHz: 4 vCPU = 2 cores × 2 hyper-threads, so `threads 4` includes HT),
  nothing else running. This is **not the dev box and not m8g**: the demo-machine budget stays open unless the demo
  runs on this class. Evidence: `evaluation/runs/l5_timing/20261021T211117Z-gpu-e2b3e3/` (CSVs, lscpu, load,
  manifest). Cost $0.035. ECR image, S3 payload and run prefix deleted after; ECR repo empty again.

  | model (engine) | size | p50 / p95 ms, 1 thread | p50 / p95 ms, 4 threads |
  |---|---|---|---|
  | XFeat detect (opencv_dnn_graph) | 736×480 | 60.5 / 62.0 | 41.5 / 42.8 |
  | XFeat detect | 1280×704 | 153.5 / 155.9 | 101.5 / 102.8 |
  | EigenPlaces (classic) | 376×240 | 229.8 / 235.3 | 126.3 / 129.0 |
  | EigenPlaces (classic) | 752×480 | 914.2 / 924.5 | 467.6 / 477.8 |
  | EigenPlaces (classic) | 640×360 | 602.2 / 605.4 | 311.8 / 315.9 |
  | EigenPlaces (classic) | 1280×720 | 2472.8 / 2509.2 | 1279.6 / 1302.3 |
  | EigenPlaces (new) | 376×240 / 640×360 | 314.1 / 880.3 (p50) | 149.7 / 391.8 (p50) |
  | ALIKED-n16 K=1024 (new) | 752×480 | 1245.0 / 1266.9 | 811.5 / 817.0 |
  | ALIKED-n16 K=1024 (new) | 1280×720 | 3078.8 / 3106.4 | 1998.6 / 2050.6 |
  | LightGlue, per pair (new) | 752×480 / 1280×720 | 1593.9 / 1642.9 (p50) | 912.6 / 921.0 (p50) |

  **Reading:**
  - p95 is within 3 % of p50 everywhere, as a quiet host should give.
  - ENGINE_CLASSIC is 1.2–1.6× faster than ENGINE_NEW for EigenPlaces, so classic stays the default (D62).
  - XFeat at sim size (~100 ms) fits keyframe rate, not 20 Hz per frame.
  - EigenPlaces at the proposed sim default 640×360 costs ~0.3 s per described keyframe.
  - A sim hard pair (2 × ALIKED + LightGlue) costs ~5 s on 4 vCPU, so it stays "hard pairs only".
 offline on the D62 dumps. fp16 and int8 (258 B) are lossless on EuRoC,
  and PCA is not worth its shared basis. The wire candidate is int8; the `wire.fbs` change is 3.2's. The XFeat local
  row is a stated gap.
- **Tests:** `test_opencv` targeted (2.1 reference, wrapper and contract suites), all pass, with
  `ARIADNE_REQUIRE_MODELS=1`. No premerge was run: phase 1 runs ORB, and none of these wrappers is on its path.

## 2026-10-21 — Combined EuRoC run done (D85); completion register created

- **2.4 full VIO + 2.1 re-association:** average 0.120 (0.121 for 2.4 alone), 9/11 in class. V1_03 and V2_02 are
  still out. Neutral overall. The projection-guided re-association trigger fires for V1_03 only: the maintainer
  decides.
- **`docs/COMPLETION_REGISTER.md`:** every plan line, gate, invariant and locked decision, with owners and carries.
- **Unowned gaps it found:**
  - L1, the live closed-loop runtime (`ros2/` is empty);
  - 0.2/0.6, no arm64 OpenCV build or image (this blocks COOL);
  - 0.3, the COOL trial not started;
  - L2, the sim IMU noise operator;
  - L3, the sim-rig retrieval and hard-pair exports;
  - TUM.

## 2026-10-21 — 2.5 closed for bookkeeping: implemented through D73 part 2c; two exit-gate items open

- **Open: the D74 certificate**, not started (plus κ/τ for cubicle and rim).
- **Open: the outlier-ramp breakpoint**, not found: random outliers succeed up to 90 %, so a harder protocol is
  needed.
- **Defaults:** loop closure and gravity priors are off unless the slam config has a `loop_closure:` section
  (`euroc_v2_loops*.yaml`, D73 evidence). Oracle reproducibility is in `ROBUST_BACKEND_2.5.md`'s header.

## 2026-10-21 — 2.2 closed: complete except exposure normalisation (Stage 4), deferred to Phase 5

- Delivered: Stage 1 (flat field) and Stage 2 (GAT + exact unbiased inverse, gated on measured noise), D in
  SimSource (D65–D69). Opt-in; measured-neutral on the one segment run.
- **Not delivered:** the plan line's exposure normalisation. Its real version is §6.2 Stage 4, which needs the
  Phase 5 lamp/exposure model; 2.2 has no exposure stage (the manifest's `exposure_norm` is inert text). The 2.2
  checkbox means "complete except this deferral".

## 2026-10-22 — 2.1 close-out: plan line 2.1 against what shipped

| 2.1 item | state | D-ID |
|---|---|---|
| XFeat keyframes + LK tracking | **done, as a measured ablation**: engine, keyframe cadence and LK built and exact against upstream; ORB + re-association ships for EuRoC v2 (XFeat regressed ATE) | D50, D60 |
| spatial-bin keypoint selection | **done** (grid selection, `keypoint_grid` in the manifest) | D50, D60 |
| DNN execution contract (per-thread Nets) | **partial**: one Net per instance in every wrapper (XFeat, PlaceRecognizer, HardPairMatcher); no shared contract object | D60–D63 |
| SGBM + WLS confidence (§9.3) | **done, off by default**; left disparity byte-identical; confidence AUC **not measured** | D61 |
| §9.4 confined-space validator | **done** (unit tests) | D61 |
| §9.6 depth-uncertainty contract | **done** (σ_Z/valid contract tested; byte-identical with confidence on) | D61 |
| ANNIndex retrieval | **done**: EigenPlaces R50-256 via cv::dnn; `cv::ANNIndex` kept as the ablation, exact index the default (**D62 draft**) | D62 |
| ALIKED + LightGlue hard path | **done**: our own upstream export, exact; EuRoC pose AUC@10 0.607 vs ORB 0.284 | D63 |
| projection + descriptor re-association (PRIORITY) | descriptor half **done**; projection-guided half **deferred** (trigger: lost stretches still dominate V1_03/V2_02 in the combined EuRoC run) | D60 |
| ORB placeholder kept as baseline | **done** | D60 |

Still owed:
- quiet-host timings (XFeat, EigenPlaces, ALIKED+LightGlue);
- WLS AUC;
- the cause of cv::dnn's rounding behind the SDDH Floor fix (`[UNVERIFIED]`).

Reference-export regeneration: PERCEPTION_2.1 §13.

## 2026-10-21 — 2.3 / 2.4 closed out (bookkeeping)

- **2.3:** every item of the plan line is built (`LOCAL_SLAM_2.3.md` §11). Nothing open.
- **2.4:** every item of the plan line is built (`INERTIAL_2.4.md` §6). LK seeding is off by default.
  `euroc_v2` = full VIO, 0.121 m and 9/11 in class.
- **Open, stated:**
  - The combined EuRoC run with 2.1's re-association has not been done; the numbers above are 2.4-only.
  - 10 s vision outages diverge.
  - The sim IMU noise operator is **not built and has no owner**. This corrects the 2026-10-20 entry, which said it was
    2.2's; D65–D69 contain no IMU work.
- `main` 3cf5f0c integrates 2.1 (through D63), 2.2, 2.3, 2.4, 2.5 (through D73 part 2c) and 3.2. Its premerge passes:
  495/495 ctest, 54/54 gate checks, `bytes.json` identical. That is premerge evidence, not exit-gate evidence.

## 2026-10-21 — 2.1 hard-pair path: ALIKED + LightGlue, our own upstream export, through OpenCV's classes (D63, approved)

## 2026-10-21 — AWS unblocked (D84)

- **Bedrock works:** the September verification block has cleared. The orchestrator model is **Nova 2 Lite**
  (`eu.amazon.nova-2-lite-v1:0`), with DeepSeek V3.2 as fallback; both pass a tool-use smoke test. Claude is not needed.
- **GPU:** spot G/VT, 4 vCPU in eu-north-1, approved 2026-10-19. On-demand is still pending.
- Phase 7's AWS prerequisites are no longer blocked.

## 2026-10-21 — 3.2 protocol built and tested; production broker solve (D82, D83)

- `cpp/collaboration/protocol/`:
  - a seeded datagram network behind the `LinkModel` seam for 3.3;
  - a fragment-level ARQ transport with per-class delivery;
  - membership with hysteresis, Swarm-SLAM's election, epochs;
  - the collaboration node (two-stage exchange, broker duties, frames);
  - the I5 outcome audit.
- **I5 campaign (oracle hooks):** 54/54 runs consistent, 0 corrupt.
  - Split-brain merges under a new epoch.
  - The classifier is shown to flag corruption and incompleteness.
  - Deterministic per seed.
- **Production broker solve:** a multi-agent PGO (node ATE 0.228 → 0.019 m on a synthetic swarm). The protocol with
  this solve ends consistent under a partition.
- **Next:** 3.4 spectral prioritization (the policy that decides which feature requests to send), then the sim
  integration (`protocol: v2`). The robust stage waits on 2.5.

## 2026-10-21 — Phase 3 opened (D80); 3.2 protocol designed (D81)

- Phase 3 opened before the Phase 2 gate, on maintainer instruction. This session owns 3.2 → 3.4 → 3.5.
  3.1/3.6 (∥A) and 3.3 (∥B) are free for parallel sessions. Decision IDs D80–D89 are claimed for Phase 3.
- 3.2 design: `docs/design/COLLAB_3.2.md`.
  - Per-class delivery, fragment ARQ.
  - Swarm-SLAM's one-hop (anchor, id) election with epochs, and a split-brain merge rule.
  - The I6 byte-gate definition, fixed before any numbers exist.
  - The `LinkModel` seam for 3.3.
- **Next:** `Network`/`FaultPlan` → Inbox v2 → Outbox → ARQ → election → frames → outcome harness → campaign. These are
  seeded discrete-event unit tests; no premerge run is needed.

## 2026-10-21 — 2.2: radiometric front end, opt-in with a measured-noise gate (D65–D69)

- **Built:**
  - GAT with our own exact unbiased inverse, which reproduces the paper's accuracy figures;
  - a single-image noise fit with the σ_z ≈ 1 check;
  - a flat field calibrated to ≤ 0.5% RMS;
  - D applied in SimSource (opt-in, per-camera seeds, byte-identical parallel D, digest unchanged);
  - two gain-up degradation configs.
- **Gate:** ON at the sim gain-up configs; OFF at dark0, at posterised dark2, and on all 11 EuRoC flights.
- **Effect** (dark3_ag16, 1000 frames): no measurable benefit. Keyframes +37%, landmarks −17%. Default stays off.
  Denoiser defaults to Gaussian (14 ms per image for the stage); NLM is an ablation only (≈ 360 ms per image).
- **Tests run:** only the targeted ones (test_numerical Gat/NoiseEstimate/FlatField/Degradation 39/39; test_unit
  Config/PerceptionConfig 6/6; the stage test standalone 5/5). Pre-merge not run: default configs are a call-through.

## 2026-10-21 — 2.5: gravity priors consumed by the pose graph (D73 part 2c)

- **Per-agent 2-DoF gravity variable plus Huber priors** in `PgoProblem`, off by default, with the information
  scaled 0.6 (measured lag-1 correlation). The mirror syncs the priors and re-solves when they land.
- **Synthetic:** a 1.5° reference-gravity error is estimated to 0.62° and tilt RMS drops 3.78° → 2.01°; a 30° outlier
  prior is contained. Tests 4/4; existing PGO tests unchanged.
- **Not yet on EuRoC:** the sim calibration and part 3 come first (maintainer).

## 2026-10-21 — 2.1 hard-pair path: ALIKED + LightGlue, our own upstream export, through OpenCV's classes (D63 draft)

- **Built:**
  - Our export from upstream (`cvg/LightGlue@eb42fee`, ALIKED-n16, native 752×480), exact through cv::ALIKED /
    cv::LightGlueMatcher (SMOKE §4s-2).
  - `perception::HardPairMatcher`.
  - The offline `ariadne_loop_verify` with `loop_verify_eval.py`.
  - 7 new tests pass (only the new ones were run).
- **EuRoC loop-candidate verification** (4815 retrieval candidates, the 1.6 verifier's essential matrix): pose
  AUC@10° 0.607 vs the ORB placeholder's 0.284, better on 11/11 flights; AP_geom 0.807 vs 0.303 (all 676
  positives as the denominator). LightGlue adds
  +0.042 over mutual NN. Both bars pass (PERCEPTION_2.1 §12.6).
- **Stated:**
  - 4 confident wrong verifications in the Vicon rooms, so the downstream PnP and degeneracy gates stay.
  - Zero-false-accept recall is ≈ 0 by the fixed definition, because tiny baselines break the translation term.
  - Cost ≈ 1.2 s per pair warm under load; hard pairs only.

## 2026-10-20 — 2.1 retrieval: EigenPlaces R50-256 via cv::dnn + PlaceIndex (D62 draft)

- **Built:** `PlaceRecognizer`, exact against upstream (SMOKE §4r), and `PlaceIndex` (exact default; `cv::ANNIndex`
  ablation). The offline tool `ariadne_place_eval` plus `python/analysis/place_recall_eval.py`; no runner hunk.
- **EuRoC loop-closure recall** (11 flights, 297 queries, 1 m / 30°, database older than 30 s): pooled R@1
  EP-376 0.626, EP-752 0.677, mean-bit ORB 0.444, random 0.167.
  - Bar 1 (beat the placeholder by 0.10 and twice random) passes.
  - Bar 2 (376 within 0.05 of 752) is missed by 0.0005. The maintainer chose 376×240 as the default
    (2026-10-20, PERCEPTION_2.1 §11.7); 752 stays selectable.
- **Cost:** EP-376 ≈ 150–200 ms, EP-752 ≈ 650–850 ms per descriptor (shared CPU). Annoy = exact on 297/297 top-1.

## 2026-10-19 — 2.1 §9.3: WLS stereo confidence wired, tested, off by default (D61)

- **Built:** `stereo.confidence: wls` fills `StereoDepth.confidence` from OpenCV's `DisparityWLSFilter` without
  touching our SGBM (the documented factory mutates it). The left disparity is byte-identical, verified in unit tests
  and on EuRoC MH_01 and V2_01.
- **Cost** about 37 ms per stereo pass at 752×480 (right SGBM 23 + filter 14). Default `none` until a consumer exists.
- **Quality (AUC against EuRoC laser scans): unmeasured** (deferred). §9.4 validator tests completed. ctest 343/343.
- **Found:** OpenCV 5.0.0 SGBM treats `disp12MaxDiff <= 0` as 1 (strictest), not "disabled".

## 2026-10-19 — 2.1 perception: ORB + lost-mode re-association for EuRoC v2; XFeat measured, not shipped (D60)

- **Ships (D60, approved as scoped):** `perception/euroc_v2.yaml` = `euroc.yaml` + lost-mode re-association, the
  ORB-SLAM relocalisation idea limited to the BA window and verified by seeded PnP.
  - **EuRoC v2 sweeps must switch to `--perception-config configs/perception/euroc_v2.yaml`**; no repo file defines
    the pair.
  - With the section absent, runs are byte-identical on EuRoC v2 (verified), and on v1/sim by construction.
- **EuRoC v2 on the merged D58 tree:**
  - failing-four median lost frames 264/101/141/171 → 67/99/92/122;
  - ATE passes on MH_04, V1_03 and V2_03;
  - 11-flight seed-1 average ATE 0.223 → 0.204 m;
  - against the D54 table, it beats all four failing flights on both metrics, and V2_02's ATE is 1.9% worse.
- **One stated miss: MH_05 ATE.** Its bar is a coasting run whose large error happened to cancel earlier drift
  (PERCEPTION_2.1 §9.15). The baseline's lost handling is fine.
- **XFeat is built** (engine, keyframe cadence, Shi-Tomasi gate, exact against upstream) **and measured**: it cuts
  lost frames hardest but regresses accuracy on V2 and MH_02 even without re-association. It stays
  config-selectable.
- Bars were re-derived on D58 with 3-seed medians (rev 3) before measuring. ctest 332/332 on the merged tree.
  Pre-merge e2e not run (maintainer: only when needed; unit + targeted EuRoC sweeps are the evidence).
- **Unmet:** XFeat timing (unmeasured); projection-guided re-association (not built).
- **Next in 2.1:** WLS / §9.4 / §9.6, ANNIndex, ALIKED+LightGlue.

## 2026-10-21 — 2.5: intra-agent loop closure live on EuRoC; loop noise model calibrated (D73 part 2b)

- **Wired:** retrieval → verifier → degeneracy gate → PCM → GNC on a mirror of the VO graph, behind a `loop_closure:`
  config section (`euroc_v2_loops`). Off by default and read-only towards the VO.
- **Keyframe-refined ATE:** MH_01 0.0676 → 0.0207 m (−69 %), V1_01 0.0563 → 0.0382 m (−32 %), held-out V1_02
  0.0517 → 0.0254 m (−51 %).
- **The PnP loop covariance is about 1000× overconfident on real data.** A calibrated model, k σ²_PnP + floor², gives
  held-out mean NEES 5.24, with a 4.4 % tail (criterion ≤ 3 %: missed, stated). Precision rises at every stage, to
  0.97 at GNC.
- **Open:**
  - sim calibration for the demo;
  - MH_01's systematic ~3.8° rotation offset on accepted loops [UNVERIFIED cause];
  - retrieval recall 51–66 % with the placeholder descriptor.
- **Next:** part 2c (gravity priors in the PGO), then part 3 (proxy swap + re-pin).

## 2026-10-20 — 2.5: GNC weight reuse across loop events (D73 part 2a)

- **`GncState`** carries each loop's decision from one event to the next. The fast path is one solve, plus one more
  for the innovation test.
- **Measured safe rule:** the TLS fixed point plus an innovation ≤ ½χ²₆ₙ(0.99). Two looser rules failed the
  warm-equals-cold gate (a wrong loop hid in bent odometry; an early near miss was kept).
- **1,000 nodes, 84 events:** warm = cold at every event, ATE identical, 72 % fast path, −22 % time.
- **3,000 nodes:** 83 % fast path, 227 ms mean per event, 655 ms worst.
- **Correction:** part 1's 1.6–4.75 s was GNC without PCM. Behind PCM a loop event costs about 0.1–0.2 s, which is
  inside the demo's "a little worse than real time".
- **Next:** part 2b, the wiring (loop source → gate → PCM → GNC, config and manifest fields, calibration).

## 2026-10-20 — 2.5: GNC-TLS core + per-component anchoring (D73 part 1)

- **`cpp/slam/robust/gnc`:** Yang et al. Prop. 4, with conventions checked against GTSAM 4.3.0's source. It runs on
  a working copy and keeps edge indices stable from the gate through GNC. `SolvePoseGraph` now anchors every
  connected component.
- **Results, random outliers 10–40 %, grid and sphere:** precision 1.000, recall worst 0.967 (gate met). 0.5 m / 5°
  near misses: all rejected. 0.2 m / 1° near misses: 71 % pass, +11 % ATE (the price of the D72 decision).
- **Measured limitation:** a lone inter-agent link cannot be verified by GNC. PCM's no-consensus rule handles
  several disagreeing links.
- **Timing:** 1.6 s at 1,000 nodes and 4.75 s at 3,000 per loop event. This corrects the earlier 0.3–0.8 s estimate.
- **Tests:** GNC 10/10; existing PGO tests unchanged.
- **Next:** part 2 (loop source, wiring, calibration, GTSAM GNC oracle), then part 3 (verifier export, proxy swap,
  re-pin).

## 2026-10-20 — 2.4: gravity-prior edges produced (D59)

- One GRAVITY_PRIOR edge per 10 keyframes on `euroc_v2` and `euroc_v2_attitude`, from Mahony's gravity (σ 2.3°,
  NEES-calibrated on MH, held-out V NEES 3.0). VIO-derived alternatives measured and rejected (overconfident 2–70×).
- Trajectories unchanged. Handoff to 2.5: consume them in the PGO (robustified, gravity direction as a variable).
- 2.4 plan items left: none in the plan list; the sim IMU noise operator is 2.2's.

## 2026-10-19 — 2.5: PCM with a per-loop threshold; recall 1.000 at 10–40 % outliers (D72)

- **`cpp/slam/robust/pcm`:** cycle consistency against the odometry-only chain. Every leg configuration on one chain
  and both inter-agent orientations give a mean statistic of 5.75–6.24 (χ²₆ mean 6).
- **Arch's fixed χ²₆(0.95) drops 15 % of true loops** (worst seed 40 %), because a maximum clique needs every pair
  of a true loop to pass. The default is a per-loop Bonferroni threshold (α_loop = 0.001, derived from the
  requirement).
- **Results:** recall 1.000 and precision 1.000 on random outliers at 10–40 %. Near misses at 0.2–0.5 m pass more
  often (precision 0.75–0.94); they are GNC's job (D73). 186 ms for 300 loops on 3000 nodes.
- **The blind spot** (a self-consistent wrong cluster passes) is asserted and labelled.
- **Tests:** 12/12; ctest 355/355 on the merged tree (D55–D59 included).
- **Next:** GNC-TLS + the intra-agent loop source + the verifier inlier export + the proxy swap (D73).

## 2026-10-19 — EuRoC v2 now runs the full inertial layer (D59 switch)

- `euroc_v2` = `imu.mode full` (avg ATE 0.121 m, 9/11 in the stereo-inertial class); attitude_only kept as
  `euroc_v2_attitude`. Both verified byte-identical to their previous runs.
- **Next (2.4):** gravity-prior edges per keyframe for the pose graph (arch §10.4), then the combined sweep after the
  2.1 merge.

## 2026-10-19 — 2.4 full, part 3: bias prior fixed; 9/11 in the stereo-inertial class (D59)

- **EuRoC `euroc_v2_full`:** average ATE **0.121 m** (target ≤ 0.138); 9/11 flights in class (V1_03 0.357, V2_02
  +1 mm out). attitude_only (shipped) is 0.223.
- **Bias prior:** keeps the carried prior's information; only the window's data is deflated (fixed a b_a runaway).
- **Gravity estimated per window:** negative (0.566 m), off, kept as an ablation.
- **Lost motion:** decay = propagate under every-frame re-anchoring (the window solves lost keyframes); stays decay.
  Long (10 s) IMU-only outages diverge more than attitude_only: recorded for 2.1 / Phase 3.
- ctest 328/328; `euroc_v2` byte-identical to D58. Pre-merge not run (maintainer's instruction).

## 2026-10-19 — 2.4 full, part 2: inertial window BA + initialisation, opt-in (D59)

- **Built:** velocity/bias states in the window BA, IMU initialisation with refinement at +5/+15 s, StereoVo and
  runner wiring, `euroc_v2_full` (opt-in), per-flight checks against ground truth (`imu_full`, `imu_trace.csv`).
- **EuRoC:** average ATE 0.223 → **0.137 m** (target ≤ 0.138 on the average); 7/11 flights in the stereo-inertial
  class. `euroc_v2` unchanged (byte-identical to D58).
- **Synthetic:** a 2 s dark stretch drifts 2 cm with the IMU vs 52 cm with the visual bridge.
- **Open:** gravity cannot correct roll/pitch drift (6-DoF gauge; part 3); `lost_motion` decision re-measured after it.
- Pre-merge not run (maintainer's instruction).

## 2026-10-19 — 2.5: loop degeneracy gate with honest loop covariance (D71)

- **`cpp/slam/robust/degeneracy`:** each PnP loop gets a covariance with B's stereo depth noise propagated
  (NEES 5.95 against 6 at 1.5–12 m; 54.7 at 25–40 m, hence a separate parallax gate).
- **Gates:** marginal σ_t / σ_r (metres and radians kept apart), parallax, and 3D collinearity. Pixel collinearity
  was measured to refuse a well-conditioned planar fan, so it is reported only.
- **Tests:** 8/8; ctest 308/308. The thresholds are [ASSUMPTION], to be calibrated in D73 on real candidates.
- **Not wired:** exporting the verifier's inliers and swapping out the proxy information need a decision.
- **Next:** PCM (D72).

## 2026-10-19 — 2.5 opened: chordal init passes the SE-Sync gate; pgo_v2 damping; oracle fixed (D70)

- **Chordal initialisation** (`cpp/slam/pose_graph/chordal_init`): `cv::SVDecomp` projection with the determinant
  fix, plus a `cv::buildMST` baseline.
- **Measured failure set from dead reckoning: torus3D and grid3D.** GTSAM's LM fails the same way from the same start.
  Chordal + LM reaches GTSAM's minimum on all six graphs (within 7.0e-9 relative, 4–15 its). **Gate PASSED.**
- **pgo_v2:** OpenCV's absolute damping floor is turned off for the pose graph. Garage drops from 95 to 15 iterations
  at production tolerances. The windowed BA stays on pgo_v1.
- **Oracle bugs fixed:** unnormalised quaternions (garage's "different minimum" was the oracle's) and GTSAM's chordal
  soft anchor. Open item (a) is diagnosed (GTSAM multifrontal false positive). (b) is reduced, not at parity.
  (c) moves to D74.
- **Placement:** `cpp/slam/robust/` (arch §3 updated).
- **Tests:** ctest 300/300 (8 env-gated skips); 100-case oracle 100/100; SE-Sync no-regression on the corrected oracle.
- **Next:** the degeneracy gate (D71), then PCM (D72), GNC-TLS and the loop source (D73), then the certificate (D74).

## 2026-10-18 — 2.4 step 1: attitude_only ships for EuRoC v2 (D58)

- **Gyro validated against ground truth:** 0.019° per pair with the bias removed.
- **Bias** is estimated from VO rotations.
- **While lost, the gyro replaces the visual bridge:** average ATE 0.289 → 0.223 m, V2_03 1.27 → 0.68. Stereo class
  10/11.
- **Gyro-seeded LK:** built, tested, off (losses −3%, V1_03 worse).
- **Gravity on keyframes:** median 1.3–4.2° error (MAV dynamics).
- ctest 309/309; pre-merge PASS.
- **Next:** full preintegration in the window BA (stereo-inertial target average ≤ 0.138 m).

## 2026-10-17 — Two cheap levers closed: scoring protocol and window iterations (D57)

- The committed D56 code reproduces the D56 sweep exactly. The D56 pre-merge passed.
- ORB-SLAM's keyframe-refined trajectory scoring is now reported: it differs by ≤ 1 mm, so scoring is not the gap.
- 15 window iterations: no net gain (V1_03 0.56 → 0.96). Kept at 10.
- The remaining gap is data association (2.1), loop closure (Phase 3), and the IMU (2.4, next here).

## 2026-10-17 — Lost-frame handling: visual rotation bridge + decaying velocity + tagged segments (D56)

- **Diagnosis:** while lost, the 2-D tracks survive; the landmarks don't. Coasting turned MH_04's 9 cm dark hover
  into 7.6 m / 117°.
- **Built:** rotation measured from the tracks against an anchor frame (seeded USAC H/E, ORB-SLAM's score rule), a
  velocity decaying with τ = 0.5 s, and `pose_source` + `segment` on every frame and keyframe record. Also a per-frame
  `vo_trace.csv`.
- **EuRoC (v2):** MH_04 2.78 → 0.36 m, V2_03 2.12 → 1.27, MH_05 0.56 → 0.35.
  - Costs: V1_03 0.44 → 0.56 (the decay), V1_02 0.11 → 0.14.
  - §8 criterion (2): 9/11 (was 7/11); criterion (1): 10/11.
- **Held-out:** per-pair bridge rotation on 25,606 tracked pairs: median 0.21° (H) / 0.04° (E).
- **Rejected:** no-keyframes-while-lost raised the losses.
- **Tests:** ctest 300/300. v1 / legacy mode is byte-identical to D54 except timing.
- **Next:** lost-frame COUNTS are unchanged, so 2.1 re-association and 2.4 IMU.

## 2026-10-17 — Gate re-pinned to the post-D54 clean run; sim change attributed (D55)

- **Full gate from clean 150ce28** (fresh Kaggle render): 22/22 non-determinism checks PASS. The 32 determinism
  diffs are all VO-derived, by design; digests unchanged.
- **Re-pinned** `phase1_gate.yaml` to that run. A pre-merge against it: **54/54 PASS**. Parallel sessions can now
  pre-merge.
- **Sim "regression" attributed:** merged ATE 9.8 → 14.4 mm is ONE loop edge (the 1.6 merge uses the best-ranked
  edge, and keyframe timing changed which one ranks first).
  - Per-agent translation improved: 9.9 → 8.4 mm and 4.4 → 4.3 mm.
  - Per-agent rotation rose 3.2 → 4.9 and 2.6 → 4.6 mrad: partly the far points, otherwise **[UNATTRIBUTED]**.
    EuRoC rotation did not rise.
- **Row refinement:** no gate added. V1_01's −0.30 px correction is real; the sim's 0.2 px is harmless.
- **Next:** lost-frame handling (D56, in progress).

## 2026-10-17 — 2.3 windowed BA in StereoVo; first full EuRoC sweep; frontend re-baselined (D53, D54)

- **2.3 integrated:** a 10-keyframe window on OpenCV's LM with observability guards, measured σ = 0.46 px, χ²
  rejection, and culling that spares live tracks. Exit edges come from the window marginals.
- **The first 11-flight EuRoC sweep** (Phase 1 had run MH_01 only) found four 1.x defects, fixed byte-identically
  (D53): an OpenCV 5.0.0 USAC throw (SMOKE §4q), unstable coasting that went NaN, a silent NaN score, and indefinite
  edge information accepted.
- **Audit** (`AUDIT_2.3.md`): lost frames come from point→map creation, not from features or BA.
  - V2's rows are 1.7 px off their published calibration.
  - The far-point cap misread ORB-SLAM.
  - V2_03 has an exposure mismatch and 414 dropped frames.
  - Each was proven by intervention.
- **Fixed by re-baselining (D54, option A):** online row refinement, far points kept, exposure match, time-based
  motion and RPE.
- **2.3 against its target:** beats v1 on 8/11 and within the published stereo class on 7/11, up from 4/11. The four
  failures all lose ≥ 100 frames.
- **Lost frames are adversarial** (map poisoning; downstream can't tell) → next.
- **Tests:** ctest 293/293 with oracles and the USAC fixture required. The gate needs re-pinning from a clean-commit
  full run, because the Phase-1 sim outputs change by design.
- **Next:** commit; full gate (fresh Kaggle render) → new reference; then lost-frame handling (AUDIT §4), then
  2.1 / 2.4.

## 2026-10-16 — Phase 2 solver cores on OpenCV 5's LM driver: PGO = GTSAM 100/100; windowed BA = OpenCV PnP refine (D51, D52)

- **PGO** (`POSE_GRAPH_2.md`): our residuals, exact Jacobians and sparse LDLT behind `cv::detail::LevMarqBase`. GTSAM
  4.3 oracle **100/100** (half the graphs with full correlated information; energy ≤ 1.5e-12 relative); gauge exactly
  6; driver contract pinned; the solve is bit-identical twice; `pgo_digest` is in every manifest. Pre-merge gate now
  requires the oracle (`models.sh pgo-oracle`). SE-Sync benchmark graphs run as a separate, slow test.
- **Windowed BA core** (`LOCAL_SLAM_2.3.md`, preflight done: literature, OpenCV, advisor): finite-difference
  Jacobians including near-zero depth and wide-FOV rays; gauge exactly 6; Schur equals dense; one free pose matches
  `cv::solvePnPRefineLM` to 1e-9.
- 2.3 target fixed before measuring: better than the 1.5 placeholder on every sequence, and within the ORB-SLAM3
  Table II stereo class.
- EuRoC: 8/11 sequences present; MH_04, V1_02 and V2_02 timed out at ETH (to retry).
- Next: BA into `StereoVo` (window, §8.3 keyframe score, landmark lifecycle, exit edges), then the EuRoC/sim runs.

## 2026-10-16 — Phase 2 opened: plan fixed, XFeat de-risked (D50)

- `docs/design/PHASE2_PLAN.md`: order (XFeat smoke → sparse PGO → 2.3 windowed BA → 2.4 inertial → 2.5 robust backend
  → 2.1/2.2 → 2.6), conventions, the EuRoC target fixed from ORB-SLAM3 Table II **before** measuring (per-sequence ≤ the
  worst of VINS-Fusion/BASALT/Kimera/ORB-SLAM3; average ≤ 0.138 m; full trajectory, SE(3) alignment), and the two docs
  contradictions settled against the papers (GNC-TLS increases μ; the certificate sign is fixed by a noise-free test).
- **XFeat on OpenCV 5 (D15/D50, SMOKE §4p):** our forward pass on upstream's weights, exact to ≤ 8e-6 with identical
  keypoints and descriptors. As a `cv::dnn` graph we assemble ourselves, without ONNX: 12–15 ms per frame.
- EuRoC: all 11 sequences downloading (ETH, ~1.2 MB/s).
- Next: the sparse SE(3) pose-graph solver + gauge null-space test + GTSAM oracle.

## 2026-10-16 — Phase 1 exit gate PASSED (evidence); pre-merge gate replaces nightly CI (D48, D49)

- **Evidence, clean commit e10dd8e** (`evaluation/runs/phase1/exit_gate/aadffb8df5790ca0-20261016T052930Z`):
  **VERDICT PASS, 54/54 checks**, `inputs_clean: true`. Every I1 condition, every §4 contract counted per hop, ctest
  257/257 + protocol tests, fault identity, §30.2 report check, console check. Every compared file matches the
  committed 1.6–1.9 evidence, with only run-identity and timing lines excluded. The fresh Kaggle render is sha256-identical to the
  1.7 render (the third session to give `cc3a2498…`). Wall time 36 min.
- D49: no nightly CI. `phase1.sh --premerge` (~20 min, working tree, render reused only for a byte-identical package)
  runs before a pipeline change is handed over and before a plan item is checked off. Tested: PASS 54/54, labelled
  "NOT exit-gate evidence". CLAUDE.md §4 updated.
- **Phase 1 is complete.** Carried into Phase 2+: the 1.6 probe false accept (Phase 3 PCM/GNC), the inert retrieval
  threshold, the overconfident proxy information, the single late loop (85 s), the Domain 3 literature survey, the
  §37.7 items beyond the gate (Phase 3), and the AWS GPU quota for the live demo.
- Next: Phase 2 (single-agent estimator to full spec).

## 2026-10-16 — Phase 1 exit gate: one replay command, §4 contracts wired + counted (D48)

- `infra/scripts/phase1.sh` runs the whole thin chain from the bag in one go (about 40 min; Kaggle render included),
  then `analysis.phase1_gate` → `gate.yaml` + `VERDICT`. It checks: the I1 condition, every §4 contract counted per hop
  (`contracts.yaml`), ctest + protocol tests + fault identity + §30.2 + console, and determinism against the committed
  1.6–1.9 evidence (named exclusions only).
- Correction: §4.2 `Keyframe` and §18.2 `KeyframeBundle` were not used by any run; they are now built by the VO hook
  and consumed by the exchange and the spool, with outputs byte-identical (EuRoC regression included).
- Dev run: everything PASS except `inputs_clean` (dirty tree, by design). The fresh Kaggle render is sha256-identical to
  the 1.7 render.
- Next: the evidence run from a clean commit. Open: nightly CI of the thin run (where and which subset).

## 2026-10-16 — 1.10 operator console stub: replayed telemetry, top-down map, byte gauges, incident timeline (D47)

- `infra/scripts/console.sh serve CHAIN` → http://127.0.0.1:8080: WebSocket telemetry at 5 Hz from a committed run chain
  (estimated agents, merge, map snapshots, egress/agent routes, cumulative bytes + rate, incident timeline);
  `?at=<s>` gives a static frame. `console.sh check`: ledger = results.yaml, merge event = t_merge (137.95 s), first
  route = 140 s, causal order, deterministic frames — PASS.
- New producer outputs (`ledger.csv` with causal send times; per-replan map snapshots) leave every committed 1.6–1.8
  output byte-identical, including the render package (the 1.7 Kaggle render stays valid).
- Next: the Phase 1 exit gate (one clean replay chain end to end + protocol tests).

## 2026-10-15 — 1.9 evaluation harness v1: all 27 §30.2 metrics, gate PASS, evo agrees (D46)

- `infra/scripts/report.sh` over a run chain → `report.yaml` (27 metrics + definitions), `oracle.yaml`, `inputs.yaml`
  (sha256 of every input), `report_check.txt` (the C++ D31 gate), `report.html` (figures, definitions, provenance).
- Dev on the committed 1.6–1.8 evidence: **gate PASS** (0 missing, 0 unknown); per-agent results, the 1.6 merged ATE
  and **evo 1.31.1 APE + RPE** all agree (≤ 1.5e-11 relative; the first RPE oracle). ATE 9.76 mm, RPE 0.86 mm / 1 s,
  closure precision 0.962, false closure rate 0, completeness 0.66, coverage 0.93, 1.27 MB, egress 140 s.
- **Evidence, clean commit 4f0e187** (`evaluation/runs/phase1/report/aadffb8df5790ca0-20261015T114028Z`): gate PASS,
  oracles agree, identical to the dev run. Report page published privately.
- Next: 1.10 (console stub), then the Phase 1 exit gate.

## 2026-10-15 — 1.8 placeholder mission layer: valid egress routes; the merge is 85 s late (D45)

- `ariadne_mission` over the 1.7 spool: OctoMap-model voxel occupancy (exact Amanatides–Woo rays), 2D standing-body
  slab, Yamauchi nearest-frontier decisions (JSONL, replay / not executed), A* egress route at ρ_responder 0.40 m
  and agent route at ρ_agent 0.20 m, GT-convention clearance. Evaluator `python/analysis/mission_eval.py`.
- Dev, ci_tiny seed 0: frame anchor 5 mm / 0.07°; coverage 0.93, free precision 0.97, occupied precision 0.71.
  First valid egress route: agent 0 alone 195 s (west, 11.26 m), **agent 1 alone 55 s** (north, 4.44 m), **merged
  140 s** (north, 4.44 m = GT geodesic; agent 1 joins at 138 s). All routes valid on every cell vs GT. The merge's
  value here is the joint map (coverage 0.93 vs 0.89 / 0.43); its single late loop costs 85 s (Phase 3 case).
  Agent 1 placed through the loop: 3.0 cm / 0.47°. Occupied precision 0.71 = wall-plane convention (93 % of false
  cells exactly 1 cell from GT structure). Deterministic; 257/257 C++ + 7 Python tests.
- Viewer page now draws the egress route. Literature basis recorded in the design (OctoMap, Amanatides–Woo,
  Yamauchi; FUEL/TARE/FAR as the Phase 4 path; the Domain 3 survey is still owed).
- **Evidence, clean commit 60763bf** (`evaluation/runs/phase1/mission/`, three runs + `eval.yaml` + `map_vs_gt.png`):
  byte-identical to the dev runs; the manifest's spool sha256 matches the 1.7 evidence checksum.
- Next: 1.9 (evaluation harness v1).

## 2026-10-15 — 1.7 placeholder reconstruction: spool → fused cloud → gsplat init → Kaggle T4 render (D44)

- **GPU:** every AWS GPU quota is 0 (EC2 G/VT in 5 regions, SageMaker); G/VT re-filed (cases 179205877700888 spot,
  179205877900026 on-demand, escalated by AWS). Maintainer decision D44: gsplat renders on Kaggle until then; the
  realtime demo still needs a live GPU. gsplat cannot rasterize on CPU at all (SMOKE §4o).
- **Boundary (§18.2):** `ariadne_sim_collab --spool` writes the KeyframeBundle spool (rectified RGB, fp16 depth + SGBM
  σ, insertion-time poses + graph version, explicit not-modeled fields); Python service in a pinned CPU image:
  validate → fuse → init (transcribed from gsplat v1.4.0 simple_trainer) → geometry metrics → 15 MB render package →
  `kaggle.sh render` (private dataset + T4 kernel) → local PSNR/SSIM + remote LPIPS.
- **Dev run, ci_tiny seed 0:** 577 keyframes + 39 held-out frames (215 MB spool in `build/`); 1.08 M voxels at 2 cm,
  capped to 300k Gaussians. Fused cloud vs GT meshes: accuracy **21 mm** mean / 18 mm median, completeness **0.66**
  (≤ 5 cm, whole maze), Chamfer 0.126 m. Held-out renders of the UNTRAINED initialization: PSNR **17.5 dB**,
  SSIM **0.718**, LPIPS **0.68**; visually pose-aligned (doorway, board, floor marks in place), blurry (the 300k cap
  inflates kNN scales), ceiling black (no keyframe depth on it). Local stages byte-identical across runs.
- **Kaggle path:** T4, torch 2.11.0+cu128, gsplat 1.4.0 from PyPI source, JIT CUDA compile ~12.5 min per fresh
  session (no prebuilt wheel for pt211/cu128). Found on the way: CLI 2.x needs the accelerator named; an account
  without phone verification silently runs GPU kernels on CPU.
- Regressions: EuRoC `22d3a40d…`, 1.6 CSVs and both `sim_vo` estimates byte-identical; 251/251 C++ + 6 Python tests.
- **Evidence, clean commit c9f17c1** (`evaluation/runs/phase1/recon/aadffb8df5790ca0-20261015T114028Z`, spool's
  1.6 run in `sim_collab/` beside it): every manifest `git_dirty: false`; fused cloud, render package, the Kaggle T4
  renders and every score reproduce the dev run byte for byte. Large outputs stay in `build/`, sha256 recorded.
- **Viewer:** `recon.sh viewer` writes a self-contained WebGL page (embedded `.splat`, both agents' estimated paths,
  world frame; gaussian-splats-3d 0.4.7 + three 0.172.0). Published privately; the maintainer confirmed both
  paths match the scripted routes.
- Next: 1.8 (placeholder mission layer).

## 2026-10-15 — 1.6 placeholder collaboration: 20/20 exchanged edges correct; 1 false accept in the negative probe (D43)

- `ariadne_sim_collab` (`core.sh sim-collab`): broker = agent 0; mean-bit ORB descriptors → cosine top-1 →
  MATCH_REQUEST/RESPONSE → `findEssentialMat` gate + PnP SE(3) → POSE_EDGE, all through `WireCodec` +
  `ByteLedger` + idempotent inbox. 248/248 core tests (4 protocol + 2 verifier tests new); EuRoC byte-identical.
- Dev runs, ci_tiny seed 0: 20 candidates (cap), 20 verified, 20/20 within 5 cm / 1° of GT (max 3.3 cm / 0.57°),
  17 in the 30° opportunity cone; merged two-agent ATE 9.8 mm; 1.27 MB total, MATCH_RESPONSE 76 %.
  Fault injection (duplicates + held-back reorders, exercised on both links): outputs identical.
- Negative probe (20 hardest non-opportunity pairs, outside the exchange): 14 rejected (all 12 different-place
  pairs among them), 5 correct overlaps, **1 false accept** (near pair, 1.10 m / 51.6° error). Planarity and
  h/E diagnostics do NOT separate it from the 25 correct edges (maze walls make correct edges near-planar too);
  cause `[UNVERIFIED]`. Not gated in 1.6: Phase 3 PCM/GNC; the pair is a Phase 3 regression case.
- Evidence, clean commit 2ed0a49 (`evaluation/runs/phase1/sim_collab/`): no-fault + fault runs, identity check
  VERDICT PASS; identical to the dev runs.
- Found and fixed before any evidence run: 19 of 20 POSE_EDGEs were delivered and silently dropped by a receive loop
  that only looked for match requests; now every delivery is dispatched or fails the run.
- Open (D43): retrieval threshold inert (top-1 cosine 0.88–1.0), proxy information ≈ 5σ overconfident, candidate
  clustering (agent 1 parks ~70 % of its run, ~57.5 s to 199 s — explains its constant VO track count).
- Next: 1.7.

## 2026-10-15 — 1.4/1.5 run from the sim recording (D42)

- `StereoSource` (EuRoC, sim + read-time lens) and a shared `stereo_vo_runner.hpp`; EuRoC MH_01 reproduces
  byte-identically through the refactor. Sim configs `sim_v1` (240 disparities: Z_min 0.281 m, §9.4).
- Evidence, clean commit 869b946 (`evaluation/runs/phase1/sim_vo/`), ci_tiny seed 0: agent 0 ATE 9.9 mm over
  31.9 m (0.03 %, calibration spins/tilts included, 0 lost); agent 1 ATE 4.4 mm over 10.7 m (0.04 %, 1 lost
  frame, not yet investigated). Identical to the dev runs. Sensor degradation after the lens: Phase 5.
- Next: 1.6 placeholder collaboration (`docs/design/COLLAB_1.6.md`).

## 2026-10-01 — 1.1–1.3: mover, recording, canonical IMU/GT, direct gz recorder (D38–D40); full missions clean

- ROS-free CDR decoder + rosbag reader (D38); kinematic mover plugin running `scripted.hpp`; mission composer
  and rig; `sim_record` with strict verification.
- Deep queues did not make the 200 Hz path lossless (one IMU sample lost in a 199 s mission), so IMU and GT
  pose now come from `ariadne_canonical_streams` (D39): same parser and sampler as the plugin, own CDR encoder
  (rclcpp leaves padding uninitialised), byte-reproducible, content digest pinned by `worldgen.sh simtest`.
  Live streams are a bit-exact cross-check: 159,199 + 159,200 samples over two full runs, 0 mismatches.
- Three logged full runs put every loss on the bridge → `ros2 bag record` DDS link (residual 0 elsewhere,
  queues 1000 deep not enough). D40: `ariadne_gz_recorder` reads gz-transport directly, converts with the
  bridge's own functions, writes synchronously, fails loudly on any gap/duplicate/late start/incomplete end/
  backlog overflow; world starts paused until every publisher is discovered; `/tf_static` now canonical.
- Two full `ci_tiny` missions PASSED strict verification: 0 recorder errors, every frame and `/clock` step
  present, live IMU/GT byte-identical to canonical. Open: render determinism across runs (files differ;
  content not yet compared). All runs from a dirty tree: re-run from a clean commit for evidence.
- 1.3 evidence recording from clean commit 9cdb7eb PASSED (`evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T042105Z`).
- §6 calibration recovery (SMOKE §4n): renders are an exact pinhole to ~0.06 px; lens distortion is NOT rendered
  under ogre2 while camera_info claims it; camera_info cx/cy are +0.5 px off the pixel-centre convention;
  render/pose time offset ≈ 0. gz warnings are now a recording gate.
- D41 (option A): pinhole render + read-time Brown lens (`lens.hpp`, fx 680). §6 with the lens passes thresholds
  derived from the same dev run (circular until the evidence run; GT reprojection 0.07-0.09 px, k1 recovered
  -0.0506 vs -0.05) and fails a 5 mm board error. D41-rig 1.3 evidence recording PASSED from clean commit
  0f7b0e4. §6 on it reproduced the dev run to every digit (renders reproducible); the independent §6 test,
  ci_tiny seed 1, PASSED with margin on every check (clean commit 5de3f70). 1.1 acceptance met. Lens digest
  added to manifests and the determinism gate. Next: 1.4/1.5 consume the sim recording (sim adapter).
- Tests: core image 242/242, worldgen 104/104, simtest 19/19.

## 2026-09-30 — 1.1–1.3: worldgen implemented and frozen

Procedural room-maze worlds (D37, `docs/design/SIM_1.1-1.3_WORLDGEN.md`): floorplan, exact 2D GT, textured quad
geometry, props, ChArUco boards, lights, S-curve yaw+pitch scripts with an analytic IMU, closure-opportunity and
separation gates, four presets, 100 tests green in the sim image, golden digests pinned. Smoke results §4g–§4j:
kinematic pose writes work physics-free, render time = pose time (depth camera), gz IMU is blind under
kinematic motion, OBJ UV convention verified, relative `<include>` fails, two full rigs in a real scene run at
RTF ~0.17 and ~84 MB/sim-s. Next: ROS-free CDR decoder, mover plugin + synthesized IMU + agent SDF + mission
composer, first real bag, §6 calibration-recovery test. To re-run from a clean commit: census + accept sweeps.

## 2026-09-30 (latest) — 1.5 placeholder stereo VO: MH_01 ATE 0.126 m (0.16% of path), evo-verified

- Conventions pinned by tests: FD Jacobians (near-zero depth, wide FOV, right camera); Monte Carlo NEES 5.93 over 2000 trials (~6);
  pose-graph edge information checked by Monte Carlo.
- One LM (Huber, Schur, §11.6 damping, S X = E covariance) for PnP refinement and the two-frame BA; agrees with
  `solvePnPRefineLM` to 1e-6.
- Stereo VO: seeded USAC PnP, ORB-SLAM3 keyframe rule, §11.2 triangulation with an SGBM cross-check, pose graph with
  a version counter; `KeyframeBundle` (§18.2) and `Keyframe` (§4.2) records.
- Synthetic CI test: 4.6 mm RMSE over 3.5 m, robust to 10% corrupted tracks, deterministic across thread counts.
- MH_01 (3682 frames): ATE 0.126 m RMSE, Sim(3) scale 1.004, 0 lost frames, rotation drift 4.9 deg/182 s (no loop
  closure yet). evo reproduces the ATE translation and rotation numbers exactly.
- Tests: image 210/210, host 185/185. Evidence run: `core.sh euroc-vo` from a clean tree after committing.

---

## 2026-09-30 — 1.4 complete on EuRoC MH_01_easy

- EuRoC fetched by a ranged single-member download (1.57 GB of a 12.7 GB archive), CRC and sha256 verified.
  Licence InC-NC: no EuRoC pixels in submitted material (D35).
- Rectifier (alpha 0, fixed-point maps) verified synthetically (1e-3 px) and on real data: median row error
  0.135 px vs 13.1 px raw and 8.1 px for a 1-degree error.
- Full sequence, 3682 frames: 377 tracks/frame, 0 DEGRADED, 22.9 ms/frame tracking, 31.6 ms/keyframe SGBM.
- Near-range aliasing on real texture is episodic (worst keyframe 26%); the density gate catches 87%. 1.8 must pick
  "no stereo free-space carving" or a near-field pass.
- Tests: image 186/186, host 166/166. Evidence run: re-run `core.sh euroc` from a clean tree after committing.

---

## 2026-09-29 — 1.4 placeholder perception: components built, EuRoC run pending

- ORB engine with §7.3 grid selection, pyramidal-LK tracker with forward-backward and epipolar rejection, SGBM
  depth to the §9.6 contract, §9.4 range validator. D34.
- 178/178 tests in the image (13 new OpenCV perception tests); host 161/161 (5 new config/validator tests).
- Deterministic across OpenCV thread counts and global RNG state (verified bit-for-bit).
- **Hazard found and bounded:** SGBM reports phantom FAR depths (0.3 m to >3 m) for surfaces nearer than Z_min.
  Speckle window 200 cuts them ~10-30x; a measured dense-support gate (`DenseSupportMask`) rejects all survivors.
  Rule for 1.8: carve free space only through supported depth. Synthetic numbers; on EuRoC the gate catches 87%
  (D35).
- Open for 1.4: EuRoC adapter + rectification + a real-sequence run (Invariant 5).

---

## 2026-09-29 — `ariadne-core` image; Phase 0 gate 4 closed on amd64

- `infra/docker/core.Dockerfile` (`dev` + `ci` stages) FROM the pinned `opencv-base`; `infra/scripts/core.sh
  image|gate|dev`; CI job `core-image` via `docker-bake.hcl` (`[UNVERIFIED]` on GitHub). D33.
- Inside the image: **160/160 tests pass**, including the first OpenCV-linked code (`ariadne_runtime_opencv`,
  `ariadne_identity`, `test_opencv`). The host build without OpenCV still passes 156/156.
- **Phase 0 gate 4 PASS (amd64):** the `cv::getBuildInformation()` hash recorded in a run manifest at run time equals
  the one baked in at build time (`fnv1a64:cf8ea084bb984703`). The manifest also records `opencv_cpu_features`.
- The COOL loader switch is protected: no RPATH/RUNPATH on anything we build (enforced at build time); OpenCV is
  resolved through `ld.so.conf.d`.
- Still open: arm64 image (with 0.2), ECR push by digest, COOL soname check (0.3).

---

## 2026-09-29 — AWS live: tests-only spin-ups; determinism proven on real Graviton4

- Account on the PAID plan with $200 of credits kept (D26, D27). Region **eu-north-1**, the cheapest region
  meeting every need (D28). Minimal network: default VPC plus one egress-only security group (D29).
- Terraform applied: bootstrap (state bucket, state migrated to S3) and main (23 resources + the
  adopted budget, raised to $200; converged).
- `infra/scripts/spinup.sh`: spot by default, cheapest AZ, three independent brakes, results pulled back,
  cost ledger. First real run: **determinism PASS on Graviton4 with gcc 11.5 / clang 15**, identical digests
  to x86-64 gcc 13 / clang 18, for **$0.0003**. See SMOKE_TEST_RESULTS §4e.
- Pending with AWS: GPU quota (G/VT 4 vCPU, spot + on-demand) in eu-north-1. Bedrock is blocked until
  post-upgrade account verification completes.
- Still open: 0.2 arm64 OpenCV 5 build; 0.3 COOL trial (deliberately not started: it lasts 7 days and
  can't be paused, so start it when the benchmark exists).

---

## 2026-09-29 — `ariadne-sim` image builds and renders headlessly

Parallel track to Phase 1, started on the maintainer's go-ahead. `infra/docker/sim.Dockerfile`
(ROS 2 Jazzy + Gazebo Harmonic, gz-sim8 8.15.0), run by `infra/scripts/sim.sh`, acceptance test
`infra/scripts/sim_smoke.py`. Full evidence in `SMOKE_TEST_RESULTS.md` §4c.

- **PASS on Mesa llvmpipe** (this box has no GPU): headless camera frames rendered, bridged to
  ROS 2, and recorded to MCAP. The test judges pixel statistics, and each of its three failure
  branches has been made to fire.
- **Not verified: NVIDIA EGL on g5/g6.** Needs a GPU instance; first task there.
- **No OpenCV in the image** (D25): the `ros-jazzy-ros-gz` metapackage would have pulled Ubuntu's
  OpenCV 4.6, so the image uses the bridge/sim subset and the build fails if OpenCV appears.
- ROS packages pinned via the dated snapshot `jazzy/2026-09-11` (D25).

Honest record of the bring-up: the first background run was reported green by its log wrapper
while the test itself had failed (`SMOKE_EXIT=1`). The cause was the test's colour detector, not the
renderer. Two further test/tooling mistakes were caught by looking at frames rather than trusting
exit codes: `<sky>false</sky>` enables a sky in SDF, and `sim.sh` did not forward the timeout,
which made a normal 5 Hz camera look like 15 Hz.

**Not done by this:** item 1.1 (the two-room corridor scene) is still open; the smoke world is an
acceptance fixture, not a Phase 1 scene. **Design gap for 1.3:** rosbag2 writes CDR images, the
replay path reads Ariadne FlatBuffers (D25).

---

## 2026-09-08 — Phase 0 end-to-end audit: six claim-vs-reality gaps found and fixed

A full re-read of Phase 0 with the advisor, hunting for claims the code does not
support. Every finding below is that same shape: **not a crash, but a comment,
test name or job name asserting a guarantee the code did not provide.** None of
them produced a wrong published result, because nothing has been published.

| # | Claim made | Reality | Status |
|---|---|---|---|
| 1 | `DetExp` accurate "to a few ulp" over its whole declared domain | returned `inf` for x in [709.44, 709.78] and `0` for x in [-745.13, -745.0], both INSIDE its own guard | fixed |
| 2 | Philox block advance "without ever colliding with a neighbouring site's stream" | `(site=0, block=1)` and `(site=2^32, block=0)` produced byte-identical output | fixed |
| 3 | CI determinism gate enforces bit-identity | `script \| tee file` under `bash -e` without `pipefail` exited with `tee`'s status; a `DETERMINISM=FAIL` went green | fixed |
| 4 | `poisson_exact_max_lambda` is a safe config knob | above ~50 Knuth's loop hits its 10000-iteration cap and silently truncates every bright pixel | fixed |
| 5 | manifest digests exist "so drifted math is identifiable after the fact" | `MissingRequiredFields` passed a manifest with all three digests `UNRECORDED` | fixed |
| 6 | `CoeffC` cancellation is "~1e-9 relative" at t=1e-2 | actual is `eps*t/(t - sin t)` ~ 1.3e-11, off by ~100x | fixed |
| 7 | `EventLog` chain localizes where two runs diverged | `state_hash_ != 0 ? ... : basis` used zero as a sentinel, so a zero chain value would silently restart the chain and let two histories converge | fixed |
| 8 | "28 B emulated UDP/IP header per datagram" (§15.7) | charged once per MESSAGE; a 4266 B pose graph is 3 datagrams, so large classes were under-billed | fixed |
| 9 | D17 (written in this same audit): Philox bounds "are asserted, so the guarantee is structural" | `assert` is compiled out under `NDEBUG`, and this project builds `Release`; in the shipped binary `site=2^48` aliased onto `site=0` and block 65536 wrapped onto block 0 | fixed |
| 10 | `require("execution_mode", ...)` | defaults to `"replay"`, so the check could never fire; it read as a guarantee and verified nothing | fixed |

Row 9 is the sharpest instance of the pattern: it is a false guarantee **introduced
by this audit**, in the commit that fixed the original one. The bounds are now
enforced by code that survives `NDEBUG` -- `site` is masked (documented 48-bit
truncation, with `SiteFromHash64()` for callers holding 64-bit ids), and block
exhaustion throws rather than silently wrapping onto a stream it already used.
The determinism gate could not have caught it: `check_determinism.sh` compiles
with bare `-O0/-O2/-O3` and no `-DNDEBUG`, so asserts are live there and dead in
the binary that would run on Graviton.

### The finding behind the findings

Re-running the digests after fixing 1 and 2, **two of the three were unchanged**:

- `det_math_digest` swept only x in [-500, 500] — the `DetExp` bug lived entirely outside it.
- `philox_digest` used sites < 2^32 drawing 2 words each, so `site >> 32 == 0` and
  `block == 0` — it never varied the two counter fields that aliased.

Both golden digests were byte-identical across genuinely broken code. A digest only
guards the inputs it evaluates, and these evaluated the safe interior of both
functions. The sweeps now cover the overflow/underflow boundaries and the
high site bits / multi-block draws respectively, and all three digests were
re-recorded. This is the durable lesson from the audit, more than any individual bug.

### Re-verification after the fixes

```
x86-64   PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   6/6 bit-identical
aarch64  PASS g++ -O0/-O2/-O3 (QEMU)                  3/3, identical to x86-64
det_math_digest = 11806531094783593156
philox_digest   = 14182439601419266695
degrade_digest  = 8535765251477677907
```

Control still fires: on aarch64 with `-ffp-contract=fast` at -O2/-O3 the det_math digest
becomes `1750406000263991614`, so `-ffp-contract=off` remains load-bearing and the
extended sweep is still a sensitive canary.

`DetExp` is now within **1 ulp** of libm across both previously-broken bands.

**146 tests green** (was 137; +9 regression tests pinning the defects). Note the
suite builds `Release`, so the two new Philox bound tests exercise exactly the
configuration where the old asserts were absent.

Byte accounting moved with the per-datagram fix: the compressed 200n/40e pose graph
goes 4294 -> 4350 wire bytes (ratio 4.93x unchanged, since that is payload over
transmitted), and the Phase 0 gate 8647 -> 8759. MCAP still reproduces the gate
byte-for-byte.

Phase 0 gate re-run: `poses=200 messages=4 wire_bytes=8647`, ATE rmse 0.150672 m,
RPE trans rmse 0.0175348 m, rot rmse 0.0033766 rad.

The ATE moved from 0.0633 m because trajectory noise draws from Philox and the
stream changed. **Measured, not assumed:** ATE rmse across 8 seeds spans
0.0717 - 0.1507 m (median 0.1139), a 2.1x spread, so this metric is dominated by
which noise realization it drew. Stated precisely: the new value sits at the top
of that sampled spread and the old one at or below its bottom, so neither is
anomalous but neither is central either. ATE is a single accumulated random-walk
statistic and is high-variance by construction.

The load-bearing check is RPE, which is local and averaged over 199 steps:
trans rmse 0.0175 m still matches the model's sqrt(3) x 0.01 = 0.0173. That is
what validates the noise model; ATE is reported, not gated.

### Correction to an earlier claim in this file

`cross-arch-determinism` was labelled "THE JOB THAT MATTERS". It is not: each
architecture already asserts against the committed constants, so if both per-arch
steps pass they necessarily agree with each other and the job cannot fail. It has
been downgraded to a second, independent comparison that does not route through
those constants, and given `if: always()` so it runs when a per-arch gate failed —
which is the only case where it is informative.

### Still open from the audit

- **§30.2 "Core metrics" (arch line 4201) has never been read.** ATE/RPE were
  implemented from §30.1 plus judgement. Phase 1.9 requires every §30.2 metric to
  emit a number, so this is a **Phase 1 prerequisite**, not a Phase 0 defect. Two
  things to check specifically: whether rotational ATE is wanted (only translation
  is computed today, which is `evo`'s default but not universal), and whether RPE
  deltas should be keyframe-, time-, or distance-based (only keyframe exists).
- `IsCompressedClass` was checked and is sound: the policy is stated once, asserted
  by a test, and D14 documents the POSE_GRAPH extension to §15.7's classes 4-5.
  `ChainHash` and the framing arithmetic were both defective and are now fixed
  (rows 7 and 8 above).
- The MTU model is an `[ASSUMPTION]`: 1500 B Ethernet, IPv4 headers, no jumbo
  frames. It should be revisited against the real link model in Phase 3.

---

## 2026-09-06 — Phase 0 exit gate built and passing

### The gate

`tests/end_to_end/test_phase0_gate.cpp` runs the loop the plan names: a synthetic
trajectory is generated, serialized to a record file, replayed back, and scored against
ground truth. Two independent runs are compared with exact equality, not a tolerance,
because a tolerance would hide precisely the drift the gate exists to detect.

```
poses=200  messages=4  wire_bytes=12632
ATE  rmse=0.0633155 m   max=0.130391 m
RPE  trans rmse=0.0174459 m   rot rmse=0.00357854 rad
```

The RPE figure validates the noise model rather than merely reporting it: per-step sigma is
0.01 m on three independent axes, and sqrt(3) * 0.01 = 0.0173 against a measured 0.01745.

**gcc 13.3.0 and clang 18.1.3 produce identical metrics**, which is stronger than the gate
requires (it asks only for reproducibility twice within a run).

### What the gate covers beyond the headline

- Replay reconstructs the recorded trajectory to float32 exactness.
- A different seed changes the metrics, so the gate can fail.
- Umeyama alignment recovers a known rigid transform, and recovers a known scale in the
  Sim(3) variant used by the monocular ablation.
- **Reflections are rejected.** An unconstrained SVD can return an improper transform that
  fits mirrored points better than any rotation; the determinant fix is asserted directly.
- A truncated record file is reported as truncated, not passed off as a clean end of file.

### Pieces added

- `core/trajectory/synthetic.hpp` -- Lissajous ground truth (self-intersecting, so it
  produces genuine revisits for later loop-closure work) plus a noisy estimate built by
  corrupting relative motions and re-integrating, so error compounds into realistic drift.
- `runtime/replay/record_file.hpp` -- length-delimited framed messages written through the
  SAME `WireCodec` the network uses, so replay exercises the production encode path and a
  wire-format bug cannot hide behind a separate file format.
- `runtime/metrics/trajectory_metrics.hpp` -- Umeyama alignment, ATE, RPE.

**101 tests green.**

### Honest limits of this gate

- The record file is **not MCAP**. §5.9 specifies MCAP as the simulator recording container;
  that remains open. This format is the deterministic substrate the gate needs, and MCAP will
  wrap the same framed messages when it lands.
- Cross-ARCHITECTURE identity of the METRICS is not claimed. The dominant reason is that
  `GenerateGroundTruth` calls `std::sin`/`std::cos`/`std::atan2`, which libm is not required
  to round correctly; Eigen's SVD in Umeyama is a second, smaller contributor. (An earlier
  version of this file named only the SVD, which understated the cause.) This is harmless:
  the gate asserts equality WITHIN a run, and CI never compares gate metrics across
  architectures. Cross-architecture bit-identity IS proven for the degradation operator and
  the deterministic math, which is what the plan's gate wording actually requires.

### 0.5 done — observability and seed plumbing

- `core/random/seed.hpp` — no global RNG anywhere. Each subsystem derives its seed from the
  run's root seed plus a NAME-hashed purpose id, so adding a subsystem later cannot shift an
  existing one's stream and silently move every ablation baseline. splitmix64 finalizer, so
  sweep seeds 1..N decorrelate rather than producing structured sub-seeds.
- `runtime/logging/event_log.hpp` — §41's 20 event types, JSONL, with a chained `state_hash`.
  The chain is the diagnostic: comparing two runs gives the exact `sequence_id` where they
  first diverged, turning a determinism failure in a long mission into a bisection.
- `runtime/manifest/run_manifest.hpp` — §28.2 fields. Images by digest not tag; OpenCV by
  build hash not version string (C7); `execution_mode` recorded because replay is
  bit-deterministic and live is not (§5.9); `imu_mode` recorded so §10.7's fallback is an
  honest downgrade. `MissingRequiredFields` refuses a run whose tree was dirty, since the
  commit then does not identify the code that ran.

### 0.6 partial — CI and infrastructure as code

- `.github/workflows/ci.yml` — build and test on amd64 and arm64 hosted runners, ccache and
  dependency caches keyed on the pins file, the determinism gate on each arch, and then the
  job that matters: **`cross-arch-determinism`**, which compares the two architectures'
  digests against EACH OTHER. Each arch already checks itself against recorded values; this
  proves they agree, which is what lets a Graviton result be compared with a dev-box result.
  The digest-extraction logic was verified against real script output rather than assumed.
  `[UNVERIFIED]` the workflow has not executed on GitHub. arm64 hosted runners are free for
  public repos; if unavailable to this account, switch to a self-hosted Graviton label per
  §27.2 and serialize benchmark jobs behind a lock.
- `infra/aws/terraform/` — network, storage, registry modules (§27.3's stated 0.6 scope).
  `terraform validate` passes. **Not applied**: no AWS credentials here, and applying is a
  billable action that is the maintainer's to take.
  - network: one VPC, one public subnet, no NAT gateway (~$32/month standing for nothing
    this design needs). Worker SG is egress-only with SSM access rather than an open SSH
    port. Console SG is closed until `allowed_console_cidrs` is set deliberately.
  - storage: one bucket with §28.1 prefixes; versioning ON because run artifacts are
    evidence and an overwritten metrics file with no history is indistinguishable from a
    fabricated one; `runs/` to IA at 30 d, noncurrent versions expired at 90 d, incomplete
    multipart uploads aborted at 7 d so they cannot bill invisibly.
  - registry: five repos per §27.1, scan-on-push, untagged images expired at 14 d. Tags stay
    mutable only because manifests pin digests (§27.2).

### 0.1 done — config loader, named configs, licences, Python env

- `core/config/config.hpp` — strict loading. **An unknown key is an error**, which is the
  whole point: a permissive loader turns `radiance_scal:` into "use the default", so a
  dark-scene ablation silently measures a lit scene and no test notices. Missing required
  keys and wrong types fail the same way, and the message names the offending key.
- `configs/degradation/dark{0..3}_dust{0..3}.yaml` — the 16 named configs §5.6 requires,
  referenced by name from run manifests instead of inline constants. The sensor is identical
  across darkness levels and only scene radiance changes, so read noise stays constant while
  signal falls. That is what makes low light hard rather than merely dim, and a test asserts
  it rather than trusting the files.
- `LICENSES/THIRD_PARTY.md` — every dependency's licence verified from the fetched source,
  not from memory. Two findings: zstd is dual BSD-OR-GPLv2 and we take BSD; Eigen ships some
  LGPL code, so the build now defines **`EIGEN_MPL2_ONLY`**, making inclusion of any LGPL
  Eigen header a compile error. Permissive licensing is therefore proven by compiling rather
  than asserted. Ariadne's own licence is still `[UNDECIDED — maintainer]`; the competition
  does not mandate one.
- `pyproject.toml` — analysis environment with bounded pins, and `evo` as the trajectory
  oracle (§30.1) so our ATE/RPE is cross-checked rather than self-certified.

### 0.7 done — MCAP container

`runtime/replay/mcap_recorder.hpp` over the standalone Foxglove SDK (D3), so replay works
from files alone with no ROS present while bags stay rosbag2-readable.

Bags store the FRAMED message, so replay exercises the production decode path rather than a
parallel on-disk format. Verified:
- Replaying through MCAP reproduces the record-file gate **exactly**: ATE 0.0633155, RPE
  0.0174459, 8626 wire bytes. Adopting the container costs no determinism.
- Frames survive the container byte for byte.
- Messages replay in log-time order, and log time comes from the message stamp rather than a
  wall clock, which would otherwise make every bag unique.
- **Bags written twice from identical inputs are byte-identical**, which is stronger than the
  gate requires and is what makes a bag checksum in a §28.2 manifest meaningful.
- A missing or non-MCAP file is rejected rather than silently replayed as empty.

### Remaining in Phase 0

0.5 logging/event schema, run manifest, seed plumbing; 0.6 CI and Terraform; 0.7 MCAP
container; 0.1 config schema loader, `pyproject.toml`, `LICENSES/`; 0.2 arm64 OpenCV build;
0.3 COOL day-1.

---

## 2026-09-06 (later) — degradation operator and serialization boundary

### 0.7 — the operator `D` (§5.6)

`cpp/perception/degradation/degradation.hpp` implements the full sensor chain:
radiance scale, Poisson shot noise, read noise, PRNU and vignette, exposure and gain,
quantization and clipping, dust compositing. Signature is
`D(image, depth_gt, params, seed, frame_id)`: the plan's four-argument form has nowhere
for GT depth to enter, which §5.6's dust stage requires.

Physical properties that are asserted, not assumed:
- PRNU is keyed frame-invariantly, so fixed-pattern noise stays fixed. Drawing it per
  frame would make it removable by frame averaging and inflate the apparent value of
  §6.2's flat-field stage.
- Each stage draws from its own stream, so disabling one stage in a Phase 5 ablation does
  not reshuffle the noise every later stage sees.
- Shot-noise variance scales with signal, read noise does not; both verified.
- Vignette follows cos^4 and matches the requested corner response.
- Dust matches the closed form `I = J*t + A*(1-t)`, `t = exp(-beta*d)`.
- Locality: perturbing one input pixel changes exactly one output pixel, which is what
  makes the operator safe to parallelize over any partition.

### 0.4 — serialization boundary (§4.4, §15.7)

`wire.fbs` plus `serializer.hpp`. FlatBuffers headers are generated by `flatc` at build
time and never committed, so the schema cannot drift from the code. zstd is applied to
message classes 4 and 5 (MATCH_RESPONSE, POSE_EDGE) per §15.7. Byte accounting is computed
in exactly one place. §15.5 frame tagging is enforced: untagged poses are rejected rather
than defaulted, since a pose with an unknown frame cannot be composed with anything.

### Determinism gate

Now covers `D` end to end, which is what the Phase 0 exit criterion names. Six
configurations (gcc and clang, -O0/-O2/-O3) are bit-identical. `-ffast-math` still changes
the det_math digest, confirming the control has teeth.

**90 tests green.**

### Not started

0.5 logging/manifest/seed plumbing, 0.6 CI + Terraform, 0.7 MCAP replay skeleton,
the thin end-to-end trajectory loop that the Phase 0 gate names, 0.3 COOL day-1.

---

## 2026-09-06 — Phase 0: OpenCV verified, determinism substrate landed

### 0.2 smoke tests — now `[BUILD-VERIFIED]` on amd64

`ariadne/opencv-base:5.0.0` built and verified. `infra/scripts/smoke_test.cpp` compiles,
links and runs inside it; **the compile-and-link succeeding is the evidence**. Full results
in `SMOKE_TEST_RESULTS.md` §4b. Highlights:

- **HashTSDF constructs** → §20.1 satisfied, the from-scratch TSDF fallback is not needed.
- `ALIKED`/`DISK`/`LightGlueMatcher`/`ANNIndex` present; `ALIKED::create` resolves in the
  installed `.so`. **XFeat still absent** (issue I1 below).
- `DisparityWLSFilter` and the `quality` module link → C7 contrib path works.
- `build_info_hash = 9127192543017384188` for the §28.2 manifest.
- Headers install to `include/opencv5/`, **not** `opencv4/`.

Two build corrections came out of this (D7, D8): `xfeatures2d` removed from `BUILD_LIST`
(it stalled the first build on configure-time blob downloads and nothing in the
architecture uses it), and manifests must pin the **peeled commit**, not the tag-object
SHA that `git ls-remote --refs` returns for an annotated tag.

### 0.4/0.7 — deterministic primitives, the substrate C6 rests on

- `core/random/philox.hpp` — Philox4x32-10 counter-based RNG. **Passes all three
  authoritative known-answer vectors** from the Random123 reference implementation, so our
  stream matches the published algorithm rather than merely being self-consistent.
  Stateless by construction: value = f(key, counter), so thread count and iteration order
  cannot change a single bit (both asserted under real concurrency).
- `core/math/det_math.hpp` — `DetExp`/`DetLog` built from correctly-rounded operations
  only. libm's exp/log are *not* specified to be correctly rounded and legitimately differ
  between glibc versions and between x86-64 and AArch64; calling them inside `D` would
  break cross-machine bit-identity silently, one ulp at a time. libm is used as the
  **accuracy oracle in tests**, never as the runtime path.

### Determinism gate — measured, not asserted

`infra/scripts/check_determinism.sh` (repeatable; belongs in CI):

```
PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   →  6/6 bit-identical
det_math_digest = 8409701359450404215
philox_digest   = 215287054852263674
```

**The control proves the test has teeth:** `-ffast-math` *changes* the det_math digest
(`8409…` → `12633…`), which is exactly the failure D5's `-ffp-contract=off` guards against.
The Philox digest is unchanged even under `-ffast-math`, as an integer-only algorithm should be.

Precise caveat: on this host the *default* (no `-ffp-contract=off`) also matched, because
the default `-march` has no FMA. **On Arm, FMA is baseline and the default would contract**,
so the flag is load-bearing for the arm64 arm, not decorative.

**Tests: 48/48 green.**

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seed plumbing,
0.6 CI + Terraform, 0.7 MCAP replay and the `D` operator itself (its primitives now exist),
0.3 COOL day-1.

---

## 2026-08-31 — Phase 0 opened


### Done and verified

- **0.1 (partial)** Monorepo tree per arch §3 (106 directories); CMake superbuild with
  hermetic `FetchContent` pins; `.gitignore` / `.dockerignore`; this doc plus
  `DECISIONS.md` and `SMOKE_TEST_RESULTS.md`.
- **0.4 (partial)** `cpp/core` — Eigen-only, no OpenCV, per the §3 dependency rule:
  - `core/math/lie.hpp` — SO(3)/SE(3) exp/log/adjoint, left-multiplicative convention,
    `[rho; phi]` ordering (D1), stable small-angle and near-pi branches.
  - `core/time/timestamp.hpp` — integer-nanosecond `Timestamp`/`Duration`, §10.6
    time-offset contract carried explicitly as a stated simplification.
  - `core/uncertainty/covariance.hpp` — SE(3) covariance transport, composition and
    inverse rules derived for the left convention.
  - **22/22 tests green** (`ctest`). Covariance composition and inverse are checked
    against 200k-sample Monte Carlo, not against a re-derivation; a control test
    confirms the naive no-adjoint rule is rejected by the same check.
- **0.2 (partial)** Smoke tests at `[SOURCE-VERIFIED]` level — see `SMOKE_TEST_RESULTS.md`.
  C3's module-naming discrepancy is **resolved**; HashTSDF **confirmed present**, so the
  from-scratch TSDF fallback is not needed.

### In flight

- `ariadne/opencv-base:5.0.0` amd64 image building locally (`-j4`). Promotes the
  smoke-test rows from `[SOURCE-VERIFIED]` to `[BUILD-VERIFIED]`.

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seeds,
0.6 CI + Terraform, 0.7 MCAP replay + degradation operator `D`, 0.3 COOL day-1.

---

## Open issues

### I1 `[RESOLVED 2026-09-06]` — XFeat will be implemented on OpenCV 5 primitives

`cv::XFeat` does not exist in OpenCV 5.0.0. Maintainer decision (D15): implement the forward
pass directly on OpenCV 5 primitives rather than routing upstream weights through ONNX, since
substantive OpenCV 5 implementation scores better and it removes the `cv::dnn` ONNX-coverage
risk. Inference is ours, weights are upstream (Apache-2.0) and credited. Phase 2.1 work, with
validation against upstream PyTorch outputs as the acceptance test.

Original finding retained below for provenance.

### I1-orig `[HISTORICAL]` — XFeat is not in OpenCV 5

CLAUDE.md §1 locks XFeat as the primary feature extractor; it has no implementation in
OpenCV 5.0.0 or contrib. Evidence in `SMOKE_TEST_RESULTS.md` §2, decision record D6.
Needs the §49 update ritual. **Does not block Phase 0.**

### I2 `[SCHEDULE]` — the phase durations do not sum to the calendar

Phase durations total **62 days** (`4+6+10+8+7+7+5+6+9`) plus Phase 8's stated buffer,
against **57 days** to the Oct 26 platform deadline and **55** to the Oct 24 internal
target. The plan's own mitigations are gates-not-dates and the `∥A/∥B/∥C` parallel
workstreams; the compressible slack is Phase 2 (10 d) and Phase 8 (9 d), and §10.7
`attitude_only` is the named fallback for a Phase 2 overrun. Flagged now so the decision
is taken deliberately at the Phase 1 exit gate rather than discovered in October.

### I3 `[RESOLVED for determinism]` — arm64 bit-identity proven under emulation

QEMU aarch64 (gcc 13.4.0) reproduces all three digests exactly, at -O0/-O2/-O3. Bit-identity
is not a timing property, so emulation is valid evidence here. The control fired: without
`-ffp-contract=off`, Arm's baseline FMA changes `det_math_digest`, which x86-64 could not
have revealed. See DECISIONS.md D5.

Still open on real Arm hardware, for reasons unrelated to determinism: the OpenCV 5 arm64
BUILD, and every COOL timing number (§26.3 is Arm-vs-Arm and emulated timings are worthless).

### I3b `[OPEN]` — arm64 has no local host for builds and timing

Cross-**toolchain** determinism is now proven (gcc + clang, 3 optimization levels each);
cross-**architecture** is not, and is the half that actually matters for Graviton. The
digests above are committed so the arm64 check is a one-line comparison when a box exists.

Phase 0.2/0.6 require an arm64 arm. This box is amd64-only; QEMU timings would be
meaningless and the COOL benchmark must run on a real `m8g` regardless. Deferred to a
Graviton box or GitHub Arm runner. Recorded as `[PENDING]`, not as passing.

### I5 `[PARTIALLY RESOLVED]` — §15.7 byte predictions vs measurement

Resolved for the aggregate: POSE_GRAPH is now compressed (D14), taking a 200n/40e batch from
21.0 KB to 4.3 KB, under §15.7's 12 KB prediction rather than 75% over it.

Still open: the per-element figures. Node measures 65.8 B against a predicted 28 B, edge
193 B against 150 B. The doc amendment (D12) is unmade, so the §15.7 table still states
numbers this schema does not produce.

Original finding below.

### I5-orig `[HISTORICAL]` — §15.7 byte predictions are low by 29-135%

Measured: pose-graph node 65.8 B against a predicted 28 B; pose edge 193 B against 150 B.
The node figure is unreachable, not merely optimistic: §15.5 requires frame-tagged poses,
and 28 B is exactly a bare pose with no identity, timestamp or tag. Phase 3 gates
predicted-vs-measured at 15% and §16's policy divides by predicted cost, so this needs
resolving before spectral selection is tuned. Detail and options in `DECISIONS.md` D12.

### I6 `[OPEN]` — dust model fidelity, grouped for one decision

D9 (dust composited after the sensor, so it cannot affect shot-noise statistics) and D11
(the dust field is temporally white, so frame averaging removes it). Both make Phase 5's
restoration ablations easier than reality. Neither blocks Phase 0.

### I4 `[COSMETIC]` — doc filename drift

Arch §3's tree names `docs/ARCHITECTURE.md`; the real file is
`docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md`. Source of truth not renamed.
