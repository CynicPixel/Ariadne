# Post-demo backlog: compute now, implement after the demo

Maintainer, 2026-10-24 12:00 IST: schedule the open items in priority order on spare compute and record the results
here, so the post-demo implementation starts from measurements rather than guesses.

The feature freeze still holds. Jobs here run only **code that already exists** (configs, flags, existing tools). An
item that needs new code gets **no** run: it gets a row with what is known and the inputs to its preflight
(design-preflight rule).

**Slot bands** (`infra/scripts/slotq.sh` dispatches in name order, so lower numbers go first):

| band | used for |
|---|---|
| 10–29 | demo-critical: consolidation pre-merge, B's D232 (a2) pipelining acceptance, demo recordings |
| 30–79 | in-flight owner work (D123/D124 re-baselines, A2 perf) |
| **80–99** | **this backlog**, in the priority order below; fills idle slots, never delays the demo |

The Kaggle queue (`build/ci_kaggle_q`) takes CPU-only items that need no gz (dev evidence only, not judge evidence).

Each result is recorded with its run id, cost, and outcome: PASS, FAIL or NEGATIVE, kept as measured.

## Queue (priority order)

| P | item | owner | compute | runnable with existing code? | state | result / run id | post-demo implementation |
|---|---|---|---|---|---|---|---|
| 1 | **COOL day 1** (0.3 → 7.3, INV2): `/opt/cool` inspect, stock build on the same box, loader-switch probe, per-op checksums and timings | M | AWS c8g.4xlarge, `--profile cool` | yes (`infra/cool/day1.sh`) | **blocked: the maintainer must subscribe to the COOL Marketplace trial** (7 days, cannot pause) | | 7.3 benchmark on the always-on pipeline |
| 2 | **7.2 classical vs LLM chooser**: the AGENTIC_7 §D162 comparison table | B | AWS c7i lockstep + Bedrock | yes (`chooser: classical`) | classical leg queued (81-B-d162-classical: live_v1, maze_demo_0, 90 s); LLM leg waits for Bedrock on the runner role (terraform planned, apply pending the maintainer) | | report table |
| 3 | **Agentic fallback / approval fired for real** (AGENTIC_7 §3 item 1) | B | AWS c7i + Bedrock | yes: `risk_gate.autonomous_threshold`, `llm.timeout_s`, operator directives | MECHANISM test only, queued 82-B-gate-mech (`live_gate_mech_test.yaml`: threshold 0.0, timeout 0.5 s; B 40d7b5a). Not agentic evidence: live_llm_v1 forbids lowering thresholds to make it fire | | genuine firing needs a risky row (dark/dust) or the P9 campaign |
| 4 | **4.3 ablation**: 47 runs after the job-1 abort check (NBV_4.3 §15.5, about $1.6) | A1 | AWS c7i lockstep | yes (`feat/ph4-nagent`) | running: 12/47 done and valid (b00 94ff13, b01 29066c, b02 bc1129; ≈ $0.12 per batch; answer_timeouts 0, reorder_released_on_timeout 0); b03 running, b04 requeued (yielded its slot to the demo recording, ~$0.01 lost), b05–b11 queued; no numbers until all 48 complete (pre-registered) | job 1 GO: 20261024T055409Z-cpux86-8d7693 (`feat/ph4-nagent` e73b8ba): turn accepted (1 segment, 7.84 s), 441/441 barriers, 0 timeouts; dev: ci_tiny seed 2 coverage at 60 s 0.878 vs gated 0.728 | whichever of D117–D125 the ablation supports |
| 5 | **4.4 live allocation rows; 4.5 live reroute** | A1 | AWS c7i lockstep | A1 checks | A1 to answer | | |
| 9 | **Failure campaign seed** (8.2) using the existing operator actions (cut/restore/inject comms) | B | AWS c7i | yes (D221/D222) | after the consolidation merge | | 8.2 |

## Needs code (no run now; preflight inputs recorded)

| item | owner | known so far | preflight inputs |
|---|---|---|---|
| 4.6 FAR route defect: hazard seed 0 (11.115 vs 4.424 m); random map #9 (37.40 vs 33.24 cells, D123) | A1 | the claim "FAR ≤ string-pulled lattice" is false; the gate rows use A* | the two failing instances |
| 3.5 aliasing laboratory and campaign | M | D176 negative (place grouping); D178 entry gate PASS | D75 generator, the D178 sweep |
| D183 match_response byte predictor | M | the per-message price is accurate (±0.4 %); the miss is selected requests never answered (0–5 per run) | `evaluation/runs/collab_d180_bytes/` |
| live re-anchoring (`pose_version`) | A2 | not exercised live | |
| D181 vertex cover, 3-agent re-measure (build if saving ≥ 20 %) | M | 0.0 % saving on 24/24 two-agent rows; pass A is 2 agents per scene (`collab_aws.sh`) | a 3-agent pass-A set: D212 worldgen scenes or the ma200 live recordings via D214 |
| 3.6 local-feature wire coding, re-test (D233 → D234) | A2 | D234 (62eeae3, Kaggle 20261024T053045Z-kaggle-a2-bytes36): nothing adopted; the fp32 control fails bar 1 (1 flip per scene vs int8), and lossless z32 flips 1 pair, so a 0-flip bar is below the 1-ulp floor. Post hoc only: int4 (79.4 kB/KF) and bin1 (33.6 kB/KF, −75 %) match fp32 verdicts 120/120, while production int8 differs from fp32 on 2 | new pre-registration: fp32 as the reference, tolerance tied to the 1-ulp floor, int4/bin1 candidates; `evaluation/runs/ph3/bytes_3.6_20261024/` |
| Perf leads (profile README `evaluation/runs/perf/pipeline_profile_20261024/`, $0.59) | A2 | not measured as speed-ups: upstream adaptive LightGlue 2.5–2.9× in torch (loses 1/63), but the cv::dnn export has no early exit; ORT runs EigenPlaces 1.9× faster than cv::dnn at 1 thread, LightGlue 1.4× slower; the replay reader (ConsumedRecorder + ZSTD ≈ 36 ms/step) is the next single thread after lane pipelining; two c7i instances differ by ~30 % (cause unknown) | early-exit export; engine mix with an end-to-end identity check |
| TUM fr1 RGB/depth-offset rerun (D100, D136), was P7 | A2 | `ariadne_tum_vo` has no `--depth-max-dt-ms`: `Associate()` max_dt is fixed at 20 ms (`cpp/runtime/datasets/tum.hpp:102`); no depth-to-RGB-stamp warp | one CLI flag into `Associate(max_dt_ns)` (then the 5 ms row runs on Kaggle CPU) plus a velocity warp |
| Dark + co-moving lamp rows, CLAHE on/off (3.1 §31.5 carried to 5.1), was P8 | A2 | `ariadne_place_eval`: `--sim-d90` and `--retrieval-3.1` are exclusive; the sim path has only RGB ep640/ep1280, no CLAHE methods. D137 recordings exist only in S3 (`runs/ph5-dark/631b43b…`), so a run needs AWS (egress or in-region) and an approval | CLAHE methods on the sim RGB path |
| 5.4 dark0..3 × dust0..3 frozen | A2 | dust model in D (D139 headlamp airlight, replacing D135); D135 options built, unmeasured; stretch (D199) | ALGORITHMS §10.2, D135/D139, the GT-depth compositing contract (§5.6), one measured dark+dust pair on the D137 renders |
| 5.5 confined worlds + baseline sweep 4/8/12/20 cm | A2 | confined_v1 built (D138, 20/20 accepted); the one-pass sweep rig is not recordable on g4dn (writer backlog); corridor gate pre-registered (D148), unmeasured | a 2-camera-per-render sweep redesign (D138 correction), D148 criteria, the GPU render budget |
| 5.6 real-data validation (Hilti) | A2 | NEGATIVE, measured (D144): site2_robot_3 0/100 on 3 rows; losses in dark-far-field turns (FC-5.6-1); cause open | the stereo forward-backward rejection analysis (dominant loss), MH_04 comparison; no re-tuning without a new pre-registration |
| 6.3 merge: pruning, delayed 3×4 exposure, NFL residual, specular down-weighting | A2 | the per-KF affine switches on only after 5000 submap steps, which the live budget likely never reaches (D192); no overlap pruning on the live path | RECON_6 §18.7/§19.3–19.4, D192, the batch-path evidence from the 626abe Kaggle training (pending B spools) |
| arm64 `Serialization.PreTaskConsensusBytesUnchanged` (#185) fails: latent cross-architecture issue, first run on arm64 | M (A1 diagnosing) | arm64 f85185 vs x86 goldens: heartbeat 238 B both, FNV 0xc8b28c6f9ba11f89 vs 0x5d85433fd5441b89; pose-graph **419 vs 417 B**, 0xdc8dcbc2eeaebf27 vs 0xb6f2ab38c40f0161; ack 110 B both, 0x052a83640e07845d vs 0x011ede3f1e7c8e79; loop-edge identical. Decode round-trips pass. Ruled out: char signedness (Fnv iterates uint8_t), FlatBuffers version (v25.12.19 both), zstd for ack, uninitialised fixture fields. Per-message byte figures are therefore not bit-portable across architectures (relevant to INV6 and COOL) | hex dump of the 4 messages, arm64 next to x86 (~$0.02, approved) |
| RTF ≥ 1: per-frame track/keyframe compute or keyframe work off the lockstep path | B (A2 measured) | D229 lane pipelining NEGATIVE (A2 run 4d412c, $0.027, one instance): pipelined 0.833 vs kill switch 0.839, both byte-identical to the lens-off reference; ingest is only a few ms with the lens off; the lane is track + keyframe ≈ 52 ms/frame including barrier wait, and 30 % of steps carry a > 80 ms keyframe (~126 ms) on some agent | [INFERENCE] the lever is FAST/SGBM/LK per frame or asynchronous keyframe work; needs its own preflight (determinism of an off-lockstep keyframe) |
| Live runtime on the full local SLAM: `infra/scripts/live.sh:174,266` hardcode `perception/sim_v1` + `slam/sim_v1` (two-frame BA stub, no IMU, no lost handling); the demo recordings ran that. Wire `sim_v2` (10-KF Schur-LM window, Forster IMU, D56 bridge) and re-record (maintainer, 2026-10-24) | B | `sim_v2` needs the seeded IMU noise operator (D101), which `ariadne_live` does not take today; the RTF cost of the window + IMU is unmeasured live | D101, LOCAL_SLAM_2.3, INERTIAL_2.4, the D229 lane timings |
| 7.4 SQS/ASG sweeps | M | slotq is a host-side dispatcher | |
| 8.3 sweeps | B | after 8.2 | |
