# Design — Phase 1 exit gate: one deterministic replay, the §4 contracts, protocol idempotency

Plan, "Phase 1 testing / exit gate":

1. The I1 exit condition verbatim: two agents explore, build local maps, merge via at least one verified
   inter-agent closure, render a splat, and produce an exit route — **in one deterministic replay run**.
2. Every §4 data contract exercised end to end; every metric in §30.2 emits a number (however bad).
3. Protocol tests: duplicate/out-of-order message injection is idempotent (§15.3, §37.7).

Items 1.6–1.10 each produced evidence from their own clean commit, stage by stage. The gate replaces that chain of
hand-run stages with **one command** (`infra/scripts/phase1.sh`) from one commit, and one verdict
(`python/analysis/phase1_gate.py`, `configs/benchmark/phase1_gate.yaml`).

## The run

`phase1.sh` replays the recorded bag `ci_tiny_0-20261001T133914Z` through every stage, in order, into
`evaluation/runs/phase1/exit_gate/<scene>-<UTC>/`:

| step | stage | output in the gate dir |
|---|---|---|
| 0 | build + full ctest; protocol tests `CollabProtocol.*`, `Serialization.*` on their own | `ctest.txt`, `protocol_tests.txt` |
| 1 | 1.6 `ariadne_sim_collab` twice, concurrently, from the same bag: clean (+ the 1.7 spool) and `--inject-faults 1`; then `check_collab_fault_identity.sh` | `sim_collab/<run>/`, `sim_collab/<run>-faults/fault_identity.txt` |
| 2 | 1.7 `recon.sh prepare` → `kaggle.sh render` (remote T4, D44) → `recon.sh score` | `recon/<run>/` (small files + `checksums.sha256`) |
| 3 | 1.8 `ariadne_mission` merged / agent 0 alone / agent 1 alone + `mission_eval` | `mission/<3 runs>/` |
| 4 | 1.9 `report.sh` (§30.2 registry check, oracles) | `report/<run>/` |
| 5 | 1.10 `console.sh check` | `console_check.txt` |
| 6 | the verdict | `gate.yaml`, `gate.txt`, `command.txt`, `gate.log` |

The large outputs (spool 215 MB, fused cloud, render package, renders) stay in `build/phase1_gate/<id>/`; their
sha256 are committed in `recon/<run>/checksums.sha256`. Nothing is copied by hand: the curation of the recon
directory is part of the script.

**The render is part of the run.** Reusing the 1.7 Kaggle render would make the gate a chain of two runs. The gate
re-renders on a fresh Kaggle session; its renders compare by sha256 with the committed 1.7 renders, which is also
cross-session GPU determinism evidence.

## Determinism: compared with the committed evidence, with named exclusions only

"Deterministic" is checked against **independent** runs: the committed 1.6/1.7/1.8/1.9 evidence, produced by separate
processes from four different clean commits. `configs/benchmark/phase1_gate.yaml` lists every compared file. A file
passes if it is byte-identical, or identical once the lines matching its **named** exclusions are removed. Every
exclusion carries one of two reasons, and the checker refuses an exclusion without one:

- `run_identity`: run id, timestamps, commit, image, the paths of this run;
- `timing`: wall-clock measurements.

Manifests are not compared (they are identity by definition). The spool itself (215 MB of PNGs) has no committed
per-file reference: the 1.7 digest folded in `spool.yaml`, which embeds the run id. Its equality is therefore shown
**indirectly**: the render package npz and the fused cloud built from it match by sha256, and every mission output
read from it is byte-identical. The gate's `spool_files_except_yaml` digest (every spool file except `spool.yaml`) is
the baseline for that comparison from the evidence run on. Files that postdate the evidence (`ledger.csv`,
`contracts.yaml`, `snapshots/`) are listed as "new, not compared". The large outputs compare by sha256 against the
1.7 `checksums.sha256`.

## The §4 contracts: what "exercised end to end" means

A struct that only compiles is not exercised. Before the gate, the runtime bypassed two of the contracts: `runtime::Keyframe`
(§4.2) and `runtime::KeyframeBundle` (§18.2) existed, but no run built them. The VO hook passed loose fields, and
`ariadne_mission` declared a second `struct Keyframe`. The gate's first change wires them in. **Every Phase 1 output
stays byte-identical** (proved by a dev rerun and by the gate's own determinism check):

| contract | producer | crosses | consumer | counted in `contracts.yaml` |
|---|---|---|---|---|
| §4.1 Frame (`StereoFrame`, + `agent`, `id`) | `SimSource` / `EurocSource` | in process | `RunStereoVo` (asserts identity in sequence) | frames delivered per agent |
| §4.2 Keyframe (`runtime::Keyframe`) | `RunStereoVo` hook: graph node pose + absolute covariance, SGBM depth, observed landmarks (`StereoVo::KeyframeLandmarks`) | in process | 1.6 exchange: completes it with ORB keypoints/descriptors + mean-bit global; wire features derive from it | records, and records with every Phase-1 field filled |
| §18.2 KeyframeBundle | `BuildKeyframeBundle` at the graph's current version | spool on disk (fp16 depth/σ, RGB PNG, pose + version) | Python recon service; mission reads the spool (`SpooledKeyframe`) | built, spooled (after the stride) |
| §4.3 LoopEdge (`LoopEdgeValue`) | broker verifier | POSE_EDGE through `WireCodec` | agent 1: every decoded edge compared **field for field** with the encoded one; the merge's edge must be among those received (see below) | encoded, decoded identical, merge edge received |
| §4.4 ByteRecord / MessageClass | `ByteLedger` at the serializer | `ledger.csv` | results.yaml totals, report, console | ledger rows, classes used, total wire bytes |

**Where §4.3 stops in Phase 1.** The merge (T_O0_O1, used by the merged ATE, the spool and the mission) is computed
by the broker from its own double-precision verifier output, not from agent 1's decoded copy. The broker owns the
global estimate (the Swarm-SLAM pattern, no distributed PGO). Agent 1 receives and validates every edge, but does not
yet integrate it into its own graph: that is Phase 3. Merging from the decoded copy would change results, because the
wire pose is float32 (§15.7). So the gate proves producer → wire → consumer with exact field identity. It does not
prove that a decoded edge drives an estimate.

Fields with no Phase 1 producer are **declared unset**, in the record headers and in `contracts.yaml`, never
defaulted: Frame `imu_window`, `photo`, `illumination` (`gt` never rides on the estimator's frame); Keyframe inertial
state, `local_cloud`, `motion/energy_cost`, `timestamp_age`, `uncertainty`; Bundle `exposure_state`, `lamp_state`;
LoopEdge `pcm_score`, `gnc_weight`, `degeneracy_flag` (at defaults until Phase 3); message classes with no Phase 1
producer (HEARTBEAT, LOCAL_DESCRIPTOR, POSE_GRAPH, SUBMAP_*, TASK_*, HUMAN_REQUEST, ACK).

## Protocol: what the gate covers, and what §37.7 asks beyond it

Covered: duplicate and out-of-order delivery give identical results (the fault run against the clean run: both CSVs
byte-identical, results identical apart from the protocol counters, both VO estimates by sha256, faults proven exercised
on both links). The unit tests `CollabProtocol.InboxIsIdempotentAndRestoresOrder` and the serializer suite also run.

§37.7 also lists items that are **not** Phase 1 gate items. They are reported, not hidden:

- FlatBuffers round trip for *every* message class: tested for the classes that have payloads today (pose graph, loop
  edge, global descriptor, match request, feature response). The remaining classes have no schema yet.
- `wire_bytes` predicted vs measured per class: the ≤15% gate is Phase 3 (Iteration 6).
- Stale-epoch rejection (§15.4): the Phase 1 broker is fixed (agent 0), with no election and no epochs. Phase 3.

## The verdict

`phase1_gate.py` writes `gate.yaml` (every check with its numbers, the determinism table, the chain) and prints one
line per check and `VERDICT PASS|FAIL`; `phase1.sh` exits non-zero on FAIL. Checks:

- **i1**: keyframes for both agents; each agent's map alone has non-zero coverage of the GT free space; merge ok with
  ≥1 verified edge, all edges received, ≥1 edge correct vs GT, merged ATE finite; renders present and scored (views
  > 0, PSNR finite); first valid egress route exists and the final route is GT-valid at ρ_responder; every stage's
  manifest records the same commit.
- **contracts**: every count above > 0 and consistent with results.yaml.
- **stages**: full ctest, protocol tests, fault identity, §30.2 report check, console check.
- **determinism**: every compared file identical (or identical except named exclusions), the three large sha256.

## Not in the gate

Nightly CI of the thin run (CLAUDE.md §4) needs the 12 GB recording and a GPU render, so it can't run on hosted
runners as built. Open: which subset runs nightly, and where. The single-loop 85 s cost, the 1.6 probe false accept,
and the inert retrieval threshold are Phase 1 *results*. They are reported in the 1.6–1.9 evidence and handed to
Phases 2–3; they are not gate failures.
