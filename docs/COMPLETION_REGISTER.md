# Completion register

One row per plan line, exit-gate condition, CLAUDE.md invariant, locked decision and competition hard gate. Each row
has an owner, a state and the evidence. The register exists so nothing is lost at an ownership boundary: the sim IMU
noise operator was dropped exactly that way.

**Rules:**
- **Deferred** always names the plan item where the work now lands. That item's row lists the work under "carries".
- A row changes only together with the evidence (D-ID, STATUS entry or run directory).
- The plan's checkboxes live in the maintainer's `ARIADNE_PHASED_PLAN.md`. This file is the committed mirror; the
  maintainer copies "carries" into the plan.
- A D-ID with no entry in `main`'s `DECISIONS.md` is labelled: "AGENTIC_7 D15x" (recorded only in a design doc),
  "cited, no entry" (named in STATUS or a design doc only), or "branch `feat/…`, not on main" (proposed there).

**State vocabulary:** `done` · `partial` (what is missing is named) · `deferred → X` · `not started` · `blocked (by)` ·
`FAIL` / `NEGATIVE` (a measured result against the claim, kept as measured) · `proposed (branch)`.

**Owners:**

| owner | session / role |
|---|---|
| M | master session: 2.x SLAM, 3.x collaboration, coordination, live-collab design (LIVE_COLLAB D174, MULTIAGENT_PIPELINE D175) |
| A1 | Phase 4 planning (4.1–4.6) + 2.1 perception |
| A2 | Phase 5 DVE + Phase 6 reconstruction (incl. the splat service); earlier 2.2, 2.6, L2, 3.3 |
| B | live runtime L1 + Phase 7 agentic + console; earlier 2.5 (D74–D76); Phase 8 demo mission (D213, D216, D218) |
| — | unowned (a gap; must be assigned) |

As of 2026-10-24, `main` 4a12b50. Latest pre-merge (STATUS 2026-10-24 03:30 IST, `a66dac6`): AWS
`20261023T212435Z-cpux86-083c02` on B's `584415e`, **54/54**, ctest **695/695**, VERDICT PASS (render reused, working
tree; not exit-gate evidence). Last full gate: Phase 2, D172.

---

## Competition hard gates and CLAUDE.md invariants

| # | item | state | evidence / missing |
|---|---|---|---|
| HG1 | substantive OpenCV 5 image/video analysis | partial | Built: rectify, SGBM/WLS, LK, ORB, `cv::dnn` XFeat/EigenPlaces/ALIKED/LightGlue, findEssentialMat/PnP, LevMarq (D50–D63); `cv::fisheye` (D143); `cv::ALIKED` + `cv::LightGlueMatcher` on the inter-agent path (D171); findContours/approxPolyDP (D116 in `EXPLORATION_4.6.md`). The submission's module table is 8.4 (not started). |
| HG2 | meaningful AWS component | partial | Real use now: AWS pre-merges on c7i spot (8 runs, `evaluation/runs/premerge_aws/`); Phase 2 full gate (D172); g4dn GPU renders (D112, D137) with NVIDIA EGL PASS (SMOKE §4v); Bedrock Nova 2 Lite in a live run (`evaluation/runs/L1/llm_live_run5_maze_short_90s`); 4.3 gate on c7i (24 runs); arm64 build on c8g (`evaluation/runs/arm64/20261023T193621Z-cpuarm-7f8c96`); recon-gpu image built on AWS (`evaluation/runs/ph6/recon_gpu_build_20261023T194812Z`). **Missing:** COOL on m8g (0.3/7.3), SQS/ASG sweeps (7.4; `slotq.sh` is a host-side spot dispatcher, not SQS), gsplat on AWS GPU (T4 numbers are Kaggle, D196). |
| INV1 | both hard gates | see HG1, HG2 | |
| INV2 | COOL Arm-vs-Arm, same binary, loader switch | **blocked by 0.3** | arm64 OpenCV 5 now builds natively (c8g, 0.2). Found there: stock arm64 OpenCV already enables KleidiCV 26.03 + carotene HAL, so "stock vs COOL" is KleidiCV-stock vs COOL. COOL trial not started; m8g not used. |
| INV3 | the agentic claim rests only on the runtime loop | partial | Live Bedrock run 5 (Nova 2 Lite, maze_short, 90 s): 36/36 decisions answered, 0 fallbacks, replay 15/15 checks PASS, `trace.jsonl` (`evaluation/runs/L1/llm_live_run5_maze_short_90s`). **Missing** (AGENTIC_7 §3): a fallback or approval exercised for real (gate never crossed); the classical comparison (7.2). Run 5 was on a working tree, commit pending (its README). |
| INV4 | egress at ρ_responder | done | 4.5 full spec on D111's ρ_responder lattice (D113), invariant-4 test in `tests/unit/test_egress.cpp`; agent-width return band never enters a ρ_responder claim (D115 review). |
| INV5 | datasets before simulator | partial | EuRoC done (D53–D60, D87). TUM pinned, fr1 not in class (D100, D86). OIVIO replaced by Hilti 2023 (D141): **FAIL 0/100** (D144). |
| INV6 | bytes counted at the serializer | done | D14, D82; per-message predictions at the ledger (D171). The ≤15% byte-table gate is a Phase 3 exit item (open). Carries **I5**: §15.7 per-element table (node 65.8 B vs 28 B) never amended (D12). |
| INV7 | honesty: no "optimal" without a certificate | done (mechanism) | D74: `CertifyPoseGraph`, behind `loop_closure.certificate`. Claims cite a `certified` verdict; rim not certified (D172 stated deviation). |
| INV8 | tagged facts, `[SMOKE-TEST]` first | done (practice) | SMOKE §4u, §4v (2026-10-22/23). |

## Locked decisions (CLAUDE.md §1)

| area | state | evidence / missing |
|---|---|---|
| Sensor rig: stereo RGB 1280×720 + IMU + lamp | partial | SDF rig built (1.1). Lamp lit in `rig_stereo_v2` (D131, calibrated D133). Sim IMU noise operator (L2, D101). Live IMU not consumed (L1). Lens off in live: D227, branch `feat/l1-a1p-llm`, not on main. |
| Sim stack: Jazzy + Harmonic | done | `ariadne-sim` image (D2x). NVIDIA EGL headless on g4dn PASS (SMOKE §4v). |
| Perception | partial | XFeat primary: D60 overrides it for the EuRoC v2 VO (approved); built and exact (D50). EigenPlaces via `cv::dnn`: D62 approved; sim default 640×360 (D90). ALIKED+LightGlue hard pairs (D63) and inter-agent verification (D171). |
| SLAM | partial | Built: stereo-VIO (D59), Schur-LM (D51–D53), drop + odometry edge (2.3), chordal (D70), gate → PCM → GNC (D71–D73), certificate (D74). Phase 2 closed (D172). **GravityPgo test FAILS on arm64** (`evaluation/runs/arm64/20261023T193621Z-cpuarm-7f8c96`); fixture-defect diagnosis D123/D124 proposed, branch `feat/a1-gpgo-arm64`, not on main. |
| Edges SE(3); FlatBuffers | done | D14, D81; appended v2 fields keep v1 golden bytes (D171). |
| Collaboration | partial | Two-stage exchange, broker with epochs (D81–D83); spectral λ₂ (D170, merged); robust broker solve (D76 via D171). Vertex cover: deferred, D181 branch `feat/live-collab-live`, not on main. Async live collab D179: same branch, not on main. |
| Reconstruction (gsplat service, albedo proxy, affine exposure, chunk pruning) | partial | Service with live `--listen` ingest and per-agent placement (D230, D231, merged); RTG lifecycle (D191); albedo proxy default, RGB on lit demo (D195, D199). Affine exposure likely never switches on live (D192). Overlap pruning not built (6.3). |
| Determinism: record/replay substrate | done | D17, D55; live recordings replay = live (L1, D152a); D175 POC double replay PASS (D218). Estimator thread-count invariance: D232, branch `feat/ph5-dve`, not on main. |
| AWS (m8g COOL, g5/g6 Gazebo + gsplat, SQS + ASG, Bedrock, Terraform) | partial | Terraform (D29). Bedrock live (INV3). g4dn renders + EGL (D112, D137, SMOKE §4v). CPU spot via slotq (STATUS 2026-10-24). **Not started:** COOL on m8g, SQS/ASG, gsplat on AWS GPU. |

## Cross-cutting items no plan line names

| id | item | state | owner |
|---|---|---|---|
| L1 | **Live closed-loop runtime:** Gazebo live → `ros_gz_bridge` → estimator → mission → mover commands. Needed by 4.3/4.4, 7.1, 8.1/8.5. | **done (local llvmpipe + AWS c7i llvmpipe lockstep); AWS GPU live run not evidenced** | B. D150–D155, D152a, D153b; `docs/design/LIVE_RUNTIME_L1.md`. Acceptance `evaluation/runs/L1/local_llvmpipe_45s` (`1bb27f9`): 13 commands accepted, 0 refused; replay ×2 = live; RTF ≈ 0.14. A1 planner in lockstep, replay PASS (`evaluation/runs/L1/a1planner_lockstep_ci_tiny_30s`, one run, not a gate). 24 lockstep gate runs on AWS c7i llvmpipe (`evaluation/runs/nbv_4.3_gate/`). NVIDIA EGL PASS (SMOKE §4v), so the GPU path is unblocked, but lockstep is verified on llvmpipe only. Multi-agent: D175 POC merged (D218: 3 agents over seeded CommsNode, double replay PASS, oracle A/B digest identical; build/live, 20 s, ideal link); D174 steps 1–2 merged. `RobustJointSolve` now wired in the broker solve (D171). Not evidenced: live IMU, gravity priors in agent subgraphs. Not on main: D178–D180 (`feat/live-collab-live`), D225/D227 (`feat/l1-a1p-llm`). |
| L2 | Sim IMU noise operator (seeded, consumption-time, like D) | done | A2: D101. `--imu-noise`, prefix-replay bit-identical; sim_v2 imu full initialises (ATE 3.06 mm, 1000 frames). Pass A of the 3.2 sim run seeds IMU noise (D171 item 7). |
| L3 | Sim-rig retrieval / hard-pair exports (1280×720 RGB) for EigenPlaces and ALIKED+LightGlue | done (D90 settled: 640×360) | A1. `models.sh eigenplaces-sim` / `aliked-lg-sim`, exact vs upstream, regeneration bit-exact (SMOKE §4t, PERCEPTION_2.1 §14); `evaluation/runs/retrieval_sim_d90/`. |
| L4 | Reference-export regeneration as one command (`models.sh eigenplaces`, `aliked-lg`) | done | A1. sha256s pinned in the script; regeneration bit-exact, 11/11 and 19/19 files (PERCEPTION_2.1 §13, STATUS 2026-10-21 A1). |
| L5 | Quiet-host timings: XFeat, EigenPlaces, ALIKED+LightGlue | done on g4dn.xlarge; demo-machine and m8g numbers open | A1. `evaluation/runs/l5_timing/20261021T211117Z-gpu-e2b3e3/`. m8g/COOL timings stay with 7.3. |
| L6 | DNN per-thread `Net` contract object (§7.4) | done (D91) | A1. `dnn::PerThread<T>`; pool rule → 7.3 (D172 carry). ALIKED output depends on the thread count (LIVE_COLLAB §7, STATUS 2026-10-24 02:20). |
| L7 | Runtime `verify` hook must gate on PnP + D71 + PCM, never essential-matrix inliers alone | **done (wiring); residual risk OPEN** | M. D171: verify chain = COLLAB_3.2 §11 (E → PnP → D71), then PCM → GNC (D76). Residual: a self-consistent aliasing group with no competitor passes PCM (D171 smoke, merged ATE 3.55 m on `collapsed_v1`); the fixed sweep FAILS geometry on scene 0 under the byte budget (STATUS 2026-10-24 02:20; D176 cited, entry only on branch `feat/live-collab-live`). |
| L8 | MTU / IPv4 framing model revisited against the real link model | done | A2/M: D102 analysis; D171 item 4: datagram size on the link seam, per-datagram loss by RFComms `ExpectedPer`, MTU stays 1500. |
| L9 | I6 dust model fidelity (D9, D11) | **open, STRETCH; paused** (→ 5.4) | A2: D135 options + D139 headlamp model. Dust is stretch (D199, D149) and paused in every session until the D175 POC runs live (maintainer, STATUS 2026-10-24). |
| L10 | Stale branches `feat/phase2-foundations` and `feat/3.2-protocol` | done | Deleted locally and on origin, 2026-10-22. |

---

## Phase 0

| item | state | evidence / missing | owner |
|---|---|---|---|
| 0.1 monorepo, config loader, licences | done | STATUS 2026-09-06 | |
| 0.2 OpenCV 5 + contrib on amd64 **and arm64**; smoke list | **partial** | amd64 `[BUILD-VERIFIED]` (SMOKE §4b). **arm64 built natively** on c8g.4xlarge spot (Graviton, Neoverse-V2), OpenCV 5.0.0 + contrib, 244 s, smoke PASS, ~$0.013 (`evaluation/runs/arm64/20261023T193621Z-cpuarm-7f8c96`; no STATUS entry on main). ctest **595/596: GravityPgo.RecoversTheReferenceGravityErrorAndReducesTilt FAILS** (2.71° vs x86 0.62°), OPEN on main. Proposed cause = test-fixture RNG draw order (D123) + redesign as NEES test (D124, validated on x86): branch `feat/a1-gpgo-arm64`, not on main; blocked on permission to edit 2.x test files. ORT fallback not exercised. | — (A1 proposing D123/D124; M signs off) |
| 0.3 COOL day-1 on the free trial | **not started** | Trial not subscribed (STATUS 2026-10-21: deliberately not started, 7 days, can't pause). Kit present: `infra/cool/day1.sh` + `probe.cpp`, `spinup.sh --profile cool`. The EULA/publication-terms check and the $0.04/h fee carried here before are **[UNSOURCED on main]**: no STATUS/DECISIONS/SMOKE entry found. | M |
| 0.4 core math, FlatBuffers, ByteRecord | done | | |
| 0.5 logging, manifest, seeds | done | | |
| 0.6 CI amd64 + arm64, ECR by digest, Terraform | **partial** | Terraform applied. CI workflow `[UNVERIFIED]` (never ran on GitHub); D49 replaced nightly CI with the local / AWS premerge. Sim/core images in ECR for `live_aws.sh` (STATUS 2026-10-23 L1). arm64 core dev image built on c8g (run above); its ECR push by digest is not evidenced. | — |
| 0.7 MCAP replay + operator D | done | | |
| Gate: unit + numerical | done | | |
| Gate: determinism twice, D across machines | done | Graviton4 digests match x86 (SMOKE §4e). | |
| Gate: CI green on both arches; images print the build hash | **partial** | amd64 hash gate PASS (D33). arm64 ctest 595/596 (GravityPgo FAIL, above). | — |

## Phase 1: exit gate PASSED (D48)

| item | state | evidence |
|---|---|---|
| 1.1 world + SDF rig + bridge YAML | done | D38–D40; live use is L1 (D150) |
| 1.2 kinematic mover, 2 agents | done | D38; N agents live in the D175 POC (D218) |
| 1.3 MCAP recording; "from the bag **as well as live**" | done | Bag path done. Live: L1 (D150–D152) records what it consumed (`inputs.mcap`) and its replay reproduces the run (`evaluation/runs/L1`). |
| 1.4–1.10 | done | D42–D47 |
| Gate (I1, §4 contracts, protocol idempotency) | done | D48. Latest premerge 2026-10-24 PASS 54/54 (as-of line). |

## Phase 2: exit gate PASSED (D172, Phase 2 CLOSED)

| item | state | evidence / missing | owner |
|---|---|---|---|
| 2.1 perception | done except stated gaps | STATUS 2026-10-22 table. D62 approved. Projection-guided re-association deferred (D85); V1_03 carried to 5.5 (D86). L4/L5/L6 done. CLAHE split done for EuRoC (D92); the sim dark row is carried to 5.1. | A1 |
| 2.2 radiometric 1–2 + VST | done except stage 4 | Stage 4 deferred → **5.3** (done in replay, D147). Unrun matrix rows marked NOT RUN. Caveat D130: 2.2 rows ran D on sRGB-encoded sim values. | A2 |
| 2.3 local SLAM | done | `LOCAL_SLAM_2.3.md` §11 | M |
| 2.4 inertial | done | `INERTIAL_2.4.md` §6. L2 built (D101). **Carry (D87, D172):** VO-side disconnected startup prefix (MH_04: 33 KF, MH_05: 77) open; 3.x never releases it (D171 item 2). | M |
| 2.5 robust back end | done (gate wording accepted, D86) | D70–D76 (`ROBUST_BACKEND_2.5.md`). Large SE-Sync run: sphere/torus/cubicle/grid3D certified, **rim not certified** (D74, stated deviation in D172). Broker wiring of D76 done (D171). GravityPgo (D73 part 2c) test fails on arm64: see 0.2 (D123/D124 proposed, branch `feat/a1-gpgo-arm64`). | B |
| 2.6 datasets: EuRoC + **TUM RGB-D**; regression pinned | done | EuRoC in premerge (D49). TUM a separate pinned check (`configs/benchmark/tum_regression.yaml`, `infra/scripts/tum_regression.py`): fr1/xyz 0.0496, fr1/desk 0.0581, fr2/xyz 0.0139 m. **fr1 not in class** (D100); fr1/xyz target re-sourced 0.014 (Kerl 2013); scoring-side causes ruled out (D136), RGB/depth-offset rerun open. | A2 |
| Gate: FD Jacobian suite | done | D172 table. | |
| Gate: oracle 100/100; certificate certified / not certified | done, one stated deviation | Oracle (D52). Certified (D74): low-noise 10/10; garage, tinyGrid3D, smallGrid3D, sphere2500, torus3D, cubicle, grid3D (f = Table 1). **rim: not certified** ("unknown, not wrong", not re-tuned). Planted outliers in the cost: 4/10 not certified, 6/10 certified, reported (D86 wording). `evaluation/runs/2.5/certificate_sesync/`. | B |
| Gate: outlier ramp with breakpoint | done | D75, `evaluation/runs/2.5/outlier_ramp/`: PCM → GNC breakpoint 90 % (groups of 10) / 70 % (two growing groups). | B |
| Gate: EuRoC VIO in class; TUM regression pinned | done (accepted) | VIO 0.120, 9/11 (D85, D87). Full system with loops 0.066, 10/11 in class, ≥ ORB-SLAM3 on 7/11; V1_03 0.288 the outlier (D87). TUM pinned, fr1 not in class (D100, D86). | M / A2 |
| Exit run (full, fresh render, clean tree) | done | D172: `phase1.sh` full on b59a77d, AWS, fresh Kaggle render; 53/54, the 54th = D55 reference's own dirty flag, `report.yaml` re-pinned and confirmed 54/54 at the next premerge. `evaluation/runs/phase2/exit_gate/aadffb8df5790ca0-20261023T050501Z/`. | M |

## Phase 3 (opened early, D80)

| item | state | evidence / missing; carries | owner |
|---|---|---|---|
| 3.1 EigenPlaces retrieval + CLAHE split | partial | Core D62. CLAHE split (D92): no effect on EuRoC (`evaluation/runs/retrieval_3.1/`). Sim eval with revisit GT: R@1 0.705 at both sizes, D90 settled (`evaluation/runs/retrieval_sim_d90/`). **The §31.5 dark + co-moving-lamp row is carried to 5.1** (paused, STATUS 2026-10-24). | A1 |
| 3.2 full protocol | partial | Merged: sim integration (D171, 3c5b8ae): two-pass causal replay over CommsNode, ALIKED+LightGlue verify, D76 robust solve, per-message byte predictions; D173 false-closure injection + schema-3 spool (f772dad); D174 steps 1–2 (2cb160f). Sweep fixed (`eee456`): **outcomes PASS; geometry FAIL on scene 0 under the byte budget** (aliasing beats PCM, merged ATE 3.8 m vs 0.03 m unbudgeted; STATUS 2026-10-24 02:20) [conflict: STATUS cites D176; its entry exists only on branch `feat/live-collab-live`]. Not on main: D176 place-grouped PCM, D177/D178 known-entry gate, D179 async collab, D180 re-send accounting (`feat/live-collab-live`). Per-keyframe corrections on the wire carry to 6.2. | M |
| 3.3 comms broker node (RFComms + attenuation + partitions) | done | D102 (`RfLinkModel`, wall ray-cast, `CommsNode`); carries protocol traffic in pass B with scenarios, partitions and disconnects (D171). | A2 |
| 3.4 spectral prioritization | done except vertex cover | D170 merged (c720e77): Fiedler space, greedy + 1-swap, bound valid 60/60; premerge 54/54. Budget defaults are placeholders until D171's derived levels. Vertex cover: measured, deferred (D181, branch `feat/live-collab-live`, not on main). | M |
| 3.5 aliasing laboratory | not started | Group-level defence is 3.5's (D171 smoke). Place-grouped PCM pre-registered: D176, branch `feat/live-collab-live`, not on main. Reuses D75's generator. | M |
| 3.6 descriptor compression rows | done (global D93; local D171) | Global: int8 258 B lossless on EuRoC, PCA not worth it (`evaluation/runs/retrieval_3.6/20261021T212015Z`); wire carries fp16 global (`descriptor_f16`, D171), int8 global stays a row. Local ALIKED int8: LightGlue match set identical to fp32 on near-temporal sim pairs (D171; inter-agent pairs harder). XFeat local row: stated gap. | A1 |
| Gate: fault injection + split brain | partial | Unit level (D82). Sim-level false-closure injection (D173): kept 1 solve, then rejected (smoke, not gate data). Composer split-brain test (D173). Sim-level split-brain run not evidenced. | M |
| Gate: byte table ≤15% | partial | Unit, synthetic: response prediction error 7.2 %, request 0.29 % on a disjoint scene (D170). Sim pairing per message + overhead rows built (D171). Sim sweep's ≤15 % verdict: not stated on main. | M |
| Gate: ATE vs bytes, ≥2 scenes × 3 seeds | **FAIL on scene 0 (budgeted)** | Sweep trimmed to 25/50 % budgets (D171 note). Scene 0 merged ATE 3.8 m under budget vs 0.03 m unbudgeted (STATUS 2026-10-24 02:20). Defence being redesigned through preflight. | M |
| Gate: aliasing campaign | not started | 3.5 | M |

## Phases 4–8

| item | state | evidence / missing; carries | owner |
|---|---|---|---|
| 4.1 volumetric (HashTSDF) | done in replay | D110: from-scratch voxel-hash TSDF at 5 cm (`cv::Volume` the oracle, SMOKE §4u), incremental FH ESDF, 3D frontiers, RLE coarse layer. Coverage 0.942 / 0.933 vs 1.8, occupied precision 0.865 / 0.707 (`evaluation/runs/volumetric_4.1/`). ESDF clearance corrected 2026-10-23 (`PHASE4_PREFLIGHT.md`). In the live loop via A1Planner (L1). `ALGORITHMS.md` entry owed (D110 §49). | A1 |
| 4.2 traversability / risk at ρ_agent and ρ_responder | partial | D111: precision 1.000, recall 0.900 (`evaluation/runs/traversability_4.2/`). Risk proxy D112: AUC 0.831 ≥ 0.75, control FPR 0, **enters planning**; saturates (FPR 0.34); overhang alone AUC 0.92 = next pre-registered candidate (`evaluation/runs/risk_calibration_4.2/`). Open: Voxblox half-voxel inflation (P2). | A1 |
| 4.3 NBV (generators A–E, raycast IG, self-normalisation, λ scheduling) | **partial; gate FAIL on both scenes** | D114 library + replay log (`evaluation/runs/nbv_4.3_replay/`). **Gate (NBV_4.3 §14, `evaluation/runs/nbv_4.3_gate/`): FAIL** on ci_tiny and maze_short: 24 lockstep runs (4 policies × 2 scenes × 3 seeds, AWS c7i llvmpipe). Below every baseline at T ≤ 60 s (ci_tiny), below coverage_greedy at T ≤ 160 s (maze_short); AUC 0.884 vs 0.892, 0.440 vs 0.495. Per metre best at 5–30 m (secondary view, does not rescue the claim). 24 runs on main; 36 with the TARE and 1.8 rows on `feat/ph4-nagent`. Diagnosis (NBV_4.3 §15.1, branch): turning is 45 % of Ariadne's mission, half of it turn-backs; ci_tiny's 40–60 s loss is mostly one 25 s hold livelock (seed 2). Fix through §49: D117–D122 + D125 (time-axis cost, arrival-heading twins, cut-command yaw, at-cell in-place turn) and D126 (reserved), **proposed, branch `feat/ph4-nagent`, not on main**; Implementation on that branch: D119 + D125 (`00ab21e`; the ci_tiny seed-2 hold livelock fixed in the planner and the live loop), D117/D118/D120 behind mission_v2 keys (2026-10-24, unit tests only, no run yet). Ablation pre-registered as **48 off-AWS development runs** (8 rows; NBV_4.3 §15.5 amendments) — not gate evidence. The TARE + 1.8 rows (36 runs in all) are on that branch (`18761ea`). | A1 |
| 4.4 allocation (CBBA + broker-greedy, logged identically) | partial | D115: CBBA from the paper, L_t = 1 default; replay CBBA = SGA 40/40, CBBA/opt 0.994; sparse TASK_CONSENSUS 226 B/msg (D115a) (`evaluation/runs/allocation_4.4_replay/`). Live: CBBA record per decision in lockstep (`evaluation/runs/L1/a1planner_lockstep_ci_tiny_30s`); auction picks over the radio in the D175 POC (D218). **Open:** gate live rows; disconnected regimes live. | A1 |
| 4.5 egress (D* Lite, 5th-percentile confidence) | partial | D113: = fresh Dijkstra on 180 flip rounds; 12/12 routes GT-valid on the evidence spool (`evaluation/runs/egress_4.5/`). Hazard seeds (EGRESS_4.5 §10, `evaluation/runs/egress_4.5_hazard/`): all routes GT-valid; w_r = 1 moves seed 1 out of the risk zone (0.340 → 0.000) at equal length; seed 2 +4.4 % for nothing (proxy saturation). Confidence calibration **degenerate**. **Open:** exit-gate reroute (no reroute occurred; needs a live instance). | A1 |
| 4.6 TARE / FAR | partial | D116 (record in `EXPLORATION_4.6.md`; on main only inside D115's review): FAR-style visibility graph + `NbvPolicy::kTare`. Comparator (`evaluation/runs/route_compare_4.6/`): FAR 4.403 m = string-pulled lattice. Hazard seeds: **open FAR bug** on seed 0 (11.115 m vs 4.424 m; gate rows unaffected, `Route` uses A*). D116 FAR claim corrected to `[DESIGN]`. **TARE and 1.8 gate rows run** (12 more runs, 36 in all; `18761ea` on `feat/ph4-nagent`, not yet merged to main): TARE level with Ariadne on AUC; the 4.3 claim stays false on all 36 runs. Our TARE row is weaker than TARE's code (no goal commitment, no waypoint extension; NBV_4.3 §15.2), so it stays "TARE-inspired". | A1 |
| Phase 4 exit gate | **FAIL / open** | 4.3 claim FAIL (above); TARE row, 4.4 live rows, 4.5 reroute open. | A1 |
| 5.1 dark worlds, lamp calibration | done (D149) | D130, D131, D133 (held-out 0.048 vs 0.167), D137 renders. Lighting rows (D142): dark+lamp 15.2 mm vs fixed light 2.5 mm vs lit 3 mm; preliminary, brightness not matched (D145). **Carries 3.1's §31.5 sim row** (dark + co-moving lamp, CLAHE on/off). Dark work paused until the D175 POC runs live (STATUS 2026-10-24). | A2 |
| 5.2 falloff normalisation (Stage 3) | done (D149) | D146 at keyframes; AWS (D145–D147 Results, `649f156`, `evaluation/runs/ph5/d145_rows_20261023T113916Z`): trajectory byte-identical to flat-only; **features-on-proxy NEGATIVE** (+2.9 mm), default stays raw. Every-frame SGBM row open. | A2 |
| 5.3 active exposure + lamp co-control | done in replay | **Carries 2.2's Stage 4.** D147 + Results: trace checks 171/171, 168/168; Failure E recovered in 2.35 s (16.1 vs 19.1 mm). Live `light_config` sink is B's; live-vs-replay **blocked: stretch** (D149, D199). | A2 |
| 5.4 degradation configs frozen | **open, STRETCH; paused** | Carries L9. D135/D139 built; freeze waits a measured dark+dust pair (D149, D199). | A2 |
| 5.5 confined-space worlds, baseline sweep | partial | D138 worlds; right-camera, sweep configs, depth_eval, corridor gate pre-registered (D148). Sweep render FAILED twice (TF bug fixed `82ffa00`; one-pass writer not recordable on g4dn); redesign open (D149). Carries V1_03 regression case (D86). | A2 |
| 5.6 real-data validation (Hilti 2023 site2_robot_3, D141) | **FAIL (0/100)** | D144: all 3 rows 0/100 (GCP rms 10–229 m; 14–22 % lost while turning with a dark far field); not a scoring artefact; cause open, timeboxed; fix candidates await preflight (`evaluation/runs/ph5/hilti_5.6/`). | A2 |
| Phase 5 exit gate | **NOT closed** (D149) | DVE rows partial; live-vs-replay stretch; Hilti FAIL; corridor bullet blocked on a recording. | A2 |
| 6.1 albedo-proxy training | partial | D190–D199: spool schema 2/3; albedo proxy default (D195), demo trains RGB on lit worlds (D199); T4 measured on Kaggle (D196). Held-out light-model evaluation not built. | A2 |
| 6.2 submap lifecycle, PGO re-anchoring | partial | Re-anchoring = one 4×4 (D190), RTG lifecycle (D191), Gaussian-SLAM recipe, T4 smokes (D198). Merged: per-agent placement from briefing/merge (D230), live `#kf` / `#t_w_o` tee contract + `--listen` (D231), recon tests 38/38; recon-gpu image (`evaluation/runs/ph6/recon_gpu_build_20261023T194812Z`). Live re-anchoring (`pose_version`) not exercised. Carries 3.2's per-keyframe corrections. | A2 |
| 6.3 merge pruning, affine exposure | not started | Overlap pruning unchanged, per-agent submaps separate (D230). Affine exposure likely never switches on live (D192). | A2 |
| 6.4 console full spec | partial | `/splat` router + Spark (D197), not yet browser-checked. Comms panel D222 and splat tee D220 [conflict: STATUS 2026-10-24 03:30 says D218–D222 landed; main's DECISIONS has only D218]. | B (A2 splat panel) |
| 7.1 MCP + Bedrock orchestrator | partial | AGENTIC_7 D156–D164 (design-doc records, no DECISIONS entries). Built: `MissionPlanner` seam, directives, LLM chooser with C++ validation, MCP server, orchestrator image + `live.sh` wiring (AGENTIC_7 D163), Stubber test 3/3. **Live Bedrock run 5 PASS** (`evaluation/runs/L1/llm_live_run5_maze_short_90s`: Nova 2 Lite, 36/36 answered, 0 fallbacks, replay 15/15; runs 1–4 failed, kept in its README). **Missing:** fallback or approval exercised for real (AGENTIC_7 §3 item 1); a committed-tree run; demo mode (AGENTIC_7 D169) acceptance on g4dn at RTF 1. | B |
| 7.2 classical-planner ablation | not started (runs) | Config switch exists (`chooser: classical`); the §D162 comparison table (AGENTIC_7 §3 item 3) not run. | B |
| 7.3 COOL benchmark | **blocked by 0.3** | 0.2 arm64 now built (c8g; stock includes KleidiCV). Carries L5 (m8g timings), L6/D91 pool rule (D172 carry), D86 thread setting in the manifest. Thread-count invariance D232: branch `feat/ph5-dve`, not on main. | M |
| 7.4 SQS / ASG sweeps, S3, budget alarms | not started | S3 in use; CPU spot through `infra/scripts/slotq.sh` (STATUS 2026-10-24), not SQS/ASG. | — |
| 8.1 missions (demo mission) | partial | B's D213 (cited on main, no entry): demo mission; D216 known starts (cited, no entry). D175 POC merged (D218, 20 s, 3 agents, lit maze, ideal link). Not on main: D178–D180 (`feat/live-collab-live`); D225 lockstep N-gate guard, D227 lens off (`feat/l1-a1p-llm`). D210–D217, D219, D221, D223, D224, D226, D228, D229: no record found on main. | B |
| 8.2–8.6 failure campaign, sweeps, report, video, submission | not started | Needs 7.x, live demo at real time; 8.4 carries HG1's module table. Internal submit Oct 24 / build ends Oct 26 (CLAUDE.md). | — |
