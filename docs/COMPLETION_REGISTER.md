# Completion register

One row per plan line, exit-gate condition, CLAUDE.md invariant, locked decision and competition hard gate. Each row
has an owner, a state and the evidence. The register exists so nothing is lost at an ownership boundary: the sim IMU
noise operator was dropped exactly that way.

**Rules:**
- **Deferred** always names the plan item where the work now lands. That item's row lists the work under "carries".
- A row changes only together with the evidence (D-ID, STATUS entry or run directory).
- The plan's checkboxes live in the maintainer's `ARIADNE_PHASED_PLAN.md`. This file is the committed mirror; the
  maintainer copies "carries" into the plan.

**State vocabulary:** `done` · `partial` (what is missing is named) · `deferred → X` · `not started` · `blocked (by)`.

**Owners:**

| owner | session / role |
|---|---|
| M | main session (critical path) |
| A1 | perception (2.1) |
| A2 | radiometric (2.2) |
| B | robust back end (2.5) |
| — | unowned (a gap; must be assigned) |

As of 2026-10-21, `main` 62441c4. The integrated premerge passes: 495/495 ctest, 54/54 gate checks, `bytes.json`
identical.

---

## Competition hard gates and CLAUDE.md invariants

| # | item | state | evidence / missing |
|---|---|---|---|
| HG1 | substantive OpenCV 5 image/video analysis | partial | Built: rectify, SGBM/WLS, LK, ORB, `cv::dnn` XFeat/EigenPlaces/ALIKED/LightGlue, findEssentialMat/PnP, LevMarq (D50–D63). The submission's module table is 8.4. |
| HG2 | meaningful AWS component | **partial, at risk** | Today only tests-only spin-ups (D27) and Bedrock, which works but has no consumer (D84). The real component lands in 7.1 (Bedrock orchestrator), 7.3 (COOL on m8g), 7.4 (SQS sweeps) and GPU gsplat (6.x): all not started. |
| INV1 | both hard gates | see HG1, HG2 | |
| INV2 | COOL Arm-vs-Arm, same binary, loader switch | **blocked by 0.2 / 0.3** | There is no arm64 OpenCV 5 build (I3b OPEN). The COOL trial has not started (7 days, can't be paused). |
| INV3 | the agentic claim rests only on the runtime loop | not started | 7.1. Also needs the **live closed-loop runtime (L1)** to act. |
| INV4 | egress at ρ_responder | done (placeholder) | 1.8 uses ρ_responder 0.40 m (D45). Full spec is 4.5. |
| INV5 | datasets before simulator | partial | EuRoC done (D53–D60). TUM done: pinned regression, fr1 not in class (D100). OIVIO is 5.6. |
| INV6 | bytes counted at the serializer | done | D14, D82. The ≤15% byte-table gate is a Phase 3 exit item. Carries **I5**: §15.7 per-element table (node 65.8 B vs 28 B) never amended (D12). |
| INV7 | honesty: no "optimal" without a certificate | done (mechanism) | D74: `CertifyPoseGraph`, behind `loop_closure.certificate`. A claim must cite a `certified` verdict. The verdict is for the isotropic chordal surrogate and is one-directional. |
| INV8 | tagged facts, `[SMOKE-TEST]` first | done (practice) | |

## Locked decisions (CLAUDE.md §1)

| area | state | evidence / missing |
|---|---|---|
| Sensor rig: stereo RGB 1280×720 + IMU + lamp | partial | SDF rig built (1.1). Lamp lit in `rig_stereo_v2` (D131, 5.1); v1 stays inert. Sim IMU noise operator built (L2, D101). |
| Sim stack: Jazzy + Harmonic | done | `ariadne-sim` image (D2x). |
| Perception | partial | XFeat primary: D60 overrides it for the EuRoC v2 VO only (approved), and XFeat is built and exact (D50). EigenPlaces via `cv::dnn`: D62, approved 2026-10-21 (exact index default, `cv::ANNIndex` the scale ablation). ALIKED+LightGlue hard pairs: D63. Sim-rig exports (1280×720 RGB): D90 (SMOKE §4t); EigenPlaces sim default size awaits approval. |
| SLAM | partial | Built: stereo-VIO (D59), Schur-LM (D51–D53), drop + odometry edge (2.3), chordal (D70), gate → PCM → GNC (D71–D73), certificate behind a flag (D74). |
| Edges SE(3); FlatBuffers | done | D14, D81. |
| Collaboration | partial | Built: two-stage exchange, elected broker with epochs (D81–D83). Not started: spectral λ₂ and vertex cover (3.4). |
| Reconstruction (gsplat service, albedo proxy, affine exposure, chunk pruning) | partial | Placeholder only (D44). Full spec is 6.1–6.3. |
| Determinism: record/replay substrate | done | D17, D55. The premerge determinism comparison passes. |
| AWS (m8g COOL, g5/g6 Gazebo + gsplat, SQS + ASG, Bedrock, Terraform) | partial | Terraform main stack applied (D29). Bedrock works (D84). GPU spot quota 4 vCPU (D84). COOL, SQS and GPU use: not started. |

## Cross-cutting items no plan line names

| id | item | state | owner |
|---|---|---|---|
| L1 | **Live closed-loop runtime:** `ros2/ariadne_{agent,bridge,operator,sim}` hold no source. Gazebo live → bridge → estimator → mission → mover commands. Needed by 4.3/4.4 (NBV acting), 7.1 (agentic actions), 8.1/8.5 (missions, demo). | **not started** | — (must be assigned; arch §5.4, §5.10, D38–D40 note "the bridge stays for live runs") |
| L2 | Sim IMU noise operator (seeded, consumption-time, like D) | done | A2: D101. `--imu-noise`, prefix-replay bit-identical; sim_v2 imu full initialises (ATE 3.06 mm, 1000 frames). Not wired into `ariadne_sim_collab`. |
| L3 | Sim-rig retrieval / hard-pair exports (1280×720 RGB) for EigenPlaces and ALIKED+LightGlue | done (D90 draft on the EigenPlaces default size) | A1. `models.sh eigenplaces-sim` / `aliked-lg-sim`, RGB path `DescribeRgb` / `ExtractRgb`, exact vs upstream on recorded sim frames, regeneration bit-exact (SMOKE §4t, PERCEPTION_2.1 §14). Sim retrieval quality → 3.1 |
| L4 | Reference-export regeneration as one command (`models.sh eigenplaces`, `aliked-lg`) | done | A1. Both subcommands check against sha256s pinned in the script; a full regeneration was bit-exact, 11/11 and 19/19 files (PERCEPTION_2.1 §13, STATUS 2026-10-21 A1) |
| L5 | Quiet-host timings: XFeat, EigenPlaces, ALIKED+LightGlue (near-realtime demo budget) | done on g4dn.xlarge (x86, 2C/4T); demo-machine and m8g numbers open | A1. `ariadne_dnn_timing`, `evaluation/runs/l5_timing/20261021T211117Z-gpu-e2b3e3/` (STATUS 2026-10-21 A1 table). The m8g/COOL timings stay with 7.3 |
| L6 | DNN per-thread `Net` contract object (§7.4) | done (D91) | A1. `dnn::PerThread<T>` in XFeat / PlaceRecognizer / HardPairMatcher; two concurrent threads byte-identical to single-thread under the pool rule (OpenCV pool = 1 with several callers; measured 1e-5-level drift without it). `test_dnn_contract` 5/5 repeats. Runtime pool + manifest field → Phase 7 / 7.3 |
| L7 | Runtime `verify` hook must gate on PnP + D71 + PCM, never essential-matrix inliers alone | **open, known risk** | M (3.2; D63 found 4 confident wrong verifications at 210–264 inliers) |
| L8 | MTU / IPv4 framing model revisited against the real link model | **revisited; recommendation open** | A2: D102 / COMMS_3.3 §4. Keep 1500 B under RFComms as modelled (≤3 % edge cost). Under bit-wise PER + slow fading, 1472 B loses 3–10× goodput at the edge (64–256 B wins): the 3.2 owner decides. Seam limit: `loss_p` for one size; heartbeats 1.3–2.3× over-lossy near the edge. |
| L9 | I6 dust model fidelity (D9, D11) | **model built; decision open** (→ 5.4) | A2: D135 options + D139 headlamp model (ALGORITHMS §10.2: round-trip attenuation, near-field backscatter, lamp offset; tests 31/31, golden unchanged). Measurement on the D137 dark recording pending |
| L10 | Stale branches `feat/phase2-foundations` and `feat/3.2-protocol` | done | Both were fully contained in `main` and were deleted locally and on origin, 2026-10-22. `main` is the integration branch. |

---

## Phase 0

| item | state | evidence / missing | owner |
|---|---|---|---|
| 0.1 monorepo, config loader, licences | done | STATUS 2026-09-06 | |
| 0.2 OpenCV 5 + contrib on amd64 **and arm64**; smoke list | **partial** | amd64 `[BUILD-VERIFIED]` (SMOKE §4b). **arm64 build never done (I3b).** The ORT fallback is not exercised. | — |
| 0.3 COOL day-1 on the free trial | **partial: ready to run, awaiting the subscription** | **Publication terms verified (2026-10-22):** the EULA is the AWS Marketplace Standard Contract (2023-11-07) with no benchmark or publication restriction; confidentiality covers licensor disclosures, not our measurements. COOL v3.1 is built on OpenCV 5.0, Ubuntu 24.04; C++ SDK at `/opt/cool/cpp_sdk/lib`; the vendor documents the `LD_LIBRARY_PATH` switch; software fee $0.04/h on m8g.4xlarge; trial 7 days, auto-converts. Kit: `infra/cool/day1.sh` + `probe.cpp`, `spinup.sh --profile cool`. Quota: Standard vCPU 32 (spot and on-demand). | M (doing C here) |
| 0.4 core math, FlatBuffers, ByteRecord | done | | |
| 0.5 logging, manifest, seeds | done | | |
| 0.6 CI amd64 + arm64, ECR by digest, Terraform | **partial** | Terraform applied. CI workflow `[UNVERIFIED]` (never ran on GitHub); D49 replaced nightly CI with the local premerge. No arm64 image, no ECR push. | — |
| 0.7 MCAP replay + operator D | done | | |
| Gate: unit + numerical | done | | |
| Gate: determinism twice, D across machines | done | Graviton4 digests match x86 (SMOKE §4e). | |
| Gate: CI green on both arches; images print the build hash | **partial** | amd64 hash gate PASS (D33). arm64 image missing. | — |

## Phase 1: exit gate PASSED (D48)

| item | state | evidence |
|---|---|---|
| 1.1 world + SDF rig + bridge YAML | done | D38–D40 (bridge YAML exists; **live use is L1**) |
| 1.2 kinematic mover, 2 agents | done | D38 |
| 1.3 MCAP recording; "from the bag **as well as live**" | partial | Bag path done. **Live: L1.** |
| 1.4–1.10 | done | D42–D47 |
| Gate (I1, §4 contracts, protocol idempotency) | done | D48. Premerge 2026-10-21 PASS. |

## Phase 2

| item | state | evidence / missing | owner |
|---|---|---|---|
| 2.1 perception | done except stated gaps | STATUS 2026-10-22 table. D62 approved 2026-10-21. Projection-guided re-association deferred; its trigger fires for V1_03 only (D85). D86: V1_03 is not won't-fix; it is carried to 5.5 as the degraded-vision regression case, with this as the candidate fix. L4 (models.sh), L5 (timings) and L6 (D91) done. CLAHE retrieval split done for EuRoC (D92); the sim dark row is carried to 5.1. | A1 |
| 2.2 radiometric 1–2 + VST | done except stage 4 | Stage 4 exposure normalisation deferred → **5.3**. Unrun matrix rows marked NOT RUN. | A2 |
| 2.3 local SLAM | done | `LOCAL_SLAM_2.3.md` §11 | M |
| 2.4 inertial | done | `INERTIAL_2.4.md` §6. L2 built (D101). **Carry (D87):** the VO graph has a disconnected prefix when the window solve is skipped before inertial init (MH_04: 33 keyframes, MH_05: 77). Loop closure now drops candidates into it; the fix on the VO side is open because an uninformative bridge edge would weaken PCM. The same chain feeds the 3.2 broker through CollabNode. | M |
| 2.5 robust back end | done except the planted-outlier gate wording (maintainer) | D70–D76 (`ROBUST_BACKEND_2.5.md` header). Large SE-Sync certificate run done 2026-10-22: 4/5 certified, rim not (D74). The broker wiring of D76 is M (3.2/3.5). | B |
| 2.6 datasets: EuRoC + **TUM RGB-D**; regression pinned | done | EuRoC done (premerge, D49). TUM is a **separate pinned check, not in `--premerge`** (maintainer, 2026-10-21): `configs/benchmark/tum_regression.yaml` checked by `infra/scripts/tum_regression.py`. `TumSource` + RGB-D depth path, `tum_regression.yaml` pinned (fr1/xyz 0.0496, fr1/desk 0.0581, fr2/xyz 0.0139 m; evo-checked). **fr1 not in class**, unattributed (D100). | A2 |
| Gate: FD Jacobian suite | done | | |
| Gate: oracle 100/100; certificate certified / not certified | **partial** | Oracle done (D52). Certified (D74): low-noise 10/10; garage (f 1.26252 vs Table 1 1.263), tinyGrid3D, smallGrid3D; sphere2500, torus3D, cubicle, grid3D (every f = Table 1 to 4 s.f.). **rim: not certified** (λ_min −0.233 at η 1e-10‖M‖; f matches Table 1; reported, not re-tuned). Evidence: `evaluation/runs/2.5/certificate_sesync/`. Planted outliers kept in the cost: **4/10 not certified, 6/10 certified**. That is reported per PHASE2_PLAN §3, not tuned. Gate wording accepted as "certified fraction reported" (D86); the large graphs have landed, so D86's block is cleared. rim: f equals Table 1, so the optimum is reached; the certificate is one-directional, so "not certified" means unknown, not wrong. A stationary non-global point and a perturbed point: not certified. | B |
| Gate: outlier ramp with breakpoint | done | D75, `evaluation/runs/2.5/outlier_ramp/`: aliasing groups. PCM → GNC breakpoint 90 % (groups of 10) / 70 % (two growing groups). GNC only: none. Success, ATE and runtime are plotted. | B |
| Gate: EuRoC VIO in class; TUM regression pinned | EuRoC done | VIO 0.120, 9/11 (D85, reproduced on AWS, D87). **Full system with loops on (D87): 0.066, 10/11 in class; at or better than ORB-SLAM3 on 7/11; V1_03 0.288 is the outlier.** Loop FIFO deadlock fixed (D87). TUM: pinned (D100, D86); fr1/xyz target sourced (Kerl et al. IROS 2013, 0.014), fr1 still not in class (stated). | M / A2 |
| Exit run (full, fresh render, clean tree) | not started | after the items above | M |

## Phase 3 (opened early, D80)

| item | state | evidence / missing; carries | owner |
|---|---|---|---|
| 3.1 EigenPlaces retrieval + CLAHE split | partial | Core D62 (approved). CLAHE split built (D92, `enhance`, default none); EuRoC rows: no effect, darkened proxy turned out easier than clean (`evaluation/runs/retrieval_3.1/`). **Open: the §31.5 sim row (dark + co-moving light), needing a sim retrieval eval with revisit GT, which also settles D90's sim size.** L3 done (D90). | A1 |
| 3.2 full protocol | partial | D81–D83: unit-level I5 54/54. Missing: robust stage in the broker solve; sim integration (`protocol: v2`); per-keyframe corrections on the wire (carries to 6.2); L7. | M |
| 3.3 comms broker node (RFComms + attenuation + partitions) | **partial** | D102: `RfLinkModel` + wall ray-cast + `CommsNode` (one ledger, tested) + scenario configs + `ariadne_comms_probe` on ci_tiny. **Waits on** 3.2's sim integration (`protocol: v2`) to carry protocol traffic. | A2 |
| 3.4 spectral prioritization | not started | literature done. Carries I5 (the cost model must use measured per-element bytes). | M |
| 3.5 aliasing laboratory | not started | can reuse B's synthetic aliasing-outlier protocol | M |
| 3.6 descriptor compression rows | done for global descriptors (D93); XFeat local row = stated gap | `evaluation/runs/retrieval_3.6/20261021T212015Z`: fp16/int8 lossless on EuRoC, PCA not worth it; int8 258 B is the wire candidate, **needs a wire.fbs change → 3.2 (M)**. XFeat fp16/int8 row → the first XFeat-on-the-wire config (D60 ships ORB) | A1 |
| Gate: fault injection + split brain | partial | unit level done (D82). Sim-level after 3.3. | M |
| Gate: byte table ≤15% | not started | definition fixed (COLLAB_3.2 §7) | M |
| Gate: ATE vs bytes, ≥2 scenes × 3 seeds | not started | needs 3.3, 3.4, L3 | M |
| Gate: aliasing campaign | not started | 3.5 | M |

## Phases 4–8: not started (rows carry what lands there)

| item | carries |
|---|---|
| 4.1 volumetric (HashTSDF) | HashTSDF API confirmed (SMOKE §4b) |
| 4.2 traversability / risk at ρ_agent and ρ_responder | |
| 4.3 NBV (I8) | needs L1 for closed loop |
| 4.4 allocation (CBBA + broker-greedy) | needs L1 |
| 4.5 egress (D* Lite, 5th-percentile confidence) | |
| 4.6 TARE / FAR scheduling | Domain 3 survey owed (D45) |
| 5.1 dark worlds, lamp calibration | **partial** (A2): lamp probe SMOKE §4u; D130 sRGB, D131 lamp. `ci_tiny_dark` + `rig_stereo_v2` (v1 worldgen byte-identical). Calibration done (D133, `lamp_calib_v2.yaml`; cos-incidence in, held-out 0.048 vs 0.167). **Dark renders done** (D137: `ci_tiny_dark_0` lamp + `ci_tiny_dark_fixed_0`, T4 spot, $0.1241). **Lighting rows measured** (D142, AWS `380c900`): dark+lamp ATE 15.2 mm / 219 KF vs dark+fixed light 2.5 mm / 78 KF vs lit 3 mm, sensor D +2–3 mm; preliminary, brightness/coverage not matched (D145). Dust rows + Stage 3 row to do |
| 5.2 falloff normalisation (Stage 3) | **partial** (A2): D134/D140 `falloff.{hpp,cpp}`. **Wired at keyframes (D146):** `falloff:` perception section + `--lamp-calib`; `runtime/tools/falloff_glue.hpp` does Stage 4 (÷ exposure·gain), lamp power from the render and D, and refuses non-linear frames, non-lamp-only scenes, another rig or lamp position, or `radiometric: off`. The proxy goes into `KeyframeBundle.albedo*`, not into tracking. Tests: test_opencv Falloff 6 + FalloffGlue 4 (median albedo error 0.25 % through exposure ×2 and lamp 0.4), test_unit PerceptionConfig 6. Open: the trajectory-equality run and the §31.5 rows on dark renders (next AWS session); features-on-proxy row (second commit, D146); the spool writing the proxy is 6.1 |
| 5.3 active exposure + lamp co-control | **partial** (A2): D132 controller + Stage 4 + replay `lamp_power`; tests 7/7 incl. closed loop (fixed a blind-metric stall and a limit cycle). Not wired; live `light_config` loop and the overexposure scenario to do. Carries 2.2's Stage 4. Real-data evidence (D143): Hilti OAK-D auto-exposure pinned at its ~30 ms cap, only ISO moves (930–1549) -- the controller's exposure-then-gain order |
| 5.4 degradation configs frozen | **open** (A2): L9 options built (D135: pre-sensor dust, correlated field, lamp airlight; tests 4/4, golden digest unchanged). Decision + freeze wait on one measured pair on a dark recording |
| 5.5 confined-space worlds, baseline sweep | **partial** (A2): D138 `confined_v1` (20/20 seeds accepted, narrow passages 0.45–0.60 m) + one-pass sweep cameras `rig_stereo_sweep_v1` (right_b04/08/12/20, recorded and verified; worldgen 106/106, v1 byte-identical). Not rendered; SimSource camera selection after the 2.6/L2 merge. Carries EuRoC V1_03 as the dataset regression case (main D86; candidate fix projection-guided re-association) |
| 5.6 real-data validation (OIVIO → **Hilti 2023 site2_robot_3**, D141; supersedes D88/D89) | **measured, failing** (A2): extracted on AWS (D142; 3592 stereo pairs, 200 Hz IMU, per-frame exposure, 5 LiDAR-measured GCPs, sha256-checked). Equidistant cameras via `cv::fisheye` (D143). Matrix `configs/benchmark/hilti_5.6.yaml`, fixed before measuring. **D144: all 3 rows score 0/100** (GCP rms 10–229 m; 14–22 % frames lost). Not a scoring artefact (rigid-invariant distances checked). Losses come while the robot turns with a dark far field (174–207/300 lost) and vanish while it drives straight; keyframes 200–300/300 whenever it moves. Static ring edges measured and rejected as the mechanism (≤ 10 % static tracks where it fails). Stereo FB rejection dominates (50 %), but EuRoC MH_04 with the same config is lower still: cause open, diagnosis timeboxed. Fix candidates await preflight. Evidence `evaluation/runs/ph5/hilti_5.6/` |
| 6.1 albedo-proxy training | |
| 6.2 submap lifecycle, PGO re-anchoring | **carries per-keyframe T_G_C on the wire (from 3.2/D83)** |
| 6.3 merge pruning, affine exposure | |
| 6.4 console full spec | |
| 7.1 MCP + Bedrock orchestrator | Bedrock model chosen (D84: Nova 2 Lite, DeepSeek V3.2). Needs L1. |
| 7.2 classical-planner ablation | |
| 7.3 COOL benchmark | **blocked by 0.2 arm64 + 0.3 trial**. Carries L5, L6. D86: the thread setting is decided explicitly and recorded in the manifest; D91's pool=1 is never process-wide in the timed runtime. |
| 7.4 SQS / ASG sweeps, S3, budget alarms | |
| 8.1–8.6 missions, failure campaign, sweeps, report, video, submission | need L1 (live demo), 7.x |
