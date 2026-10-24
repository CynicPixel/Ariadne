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
| 2 | **7.2 classical vs LLM chooser**: the AGENTIC_7 §D162 comparison table | B | AWS c7i lockstep + Bedrock | yes (`chooser: classical`) | to queue (band 81) | | report table |
| 3 | **Agentic fallback / approval fired for real** (AGENTIC_7 §3 item 1) | B | AWS c7i + Bedrock | B checks whether an existing knob forces it (timeout, gate threshold, operator action) | B to answer | | if no knob exists: the knob |
| 4 | **4.3 ablation**: 47 runs after the job-1 abort check (NBV_4.3 §15.5, about $1.6) | A1 | AWS c7i lockstep | yes (`feat/ph4-nagent`) | to queue (band 83–84) | job 1: 20261024T055409Z-cpux86-8d7693 rc 0 | whichever of D117–D125 the ablation supports |
| 5 | **4.4 live allocation rows; 4.5 live reroute** | A1 | AWS c7i lockstep | A1 checks | A1 to answer | | |
| 7 | **TUM fr1 RGB/depth-offset rerun** (D100, D136) | A2 | Kaggle CPU | A2 checks | | | |
| 8 | **Dark + co-moving lamp rows** (3.1 §31.5 carried to 5.1) on existing D137 renders | A2 | Kaggle / AWS CPU | A2 checks (D217 pause lifted for compute only) | | | dark demo toggle (D217) |
| 9 | **Failure campaign seed** (8.2) using the existing operator actions (cut/restore/inject comms) | B | AWS c7i | yes (D221/D222) | after the consolidation merge | | 8.2 |

## Needs code (no run now; preflight inputs recorded)

| item | owner | known so far | preflight inputs |
|---|---|---|---|
| 4.6 FAR route defect: hazard seed 0 (11.115 vs 4.424 m); random map #9 (37.40 vs 33.24 cells, D123) | A1 | the claim "FAR ≤ string-pulled lattice" is false; the gate rows use A* | the two failing instances |
| 3.5 aliasing laboratory and campaign | M | D176 negative (place grouping); D178 entry gate PASS | D75 generator, the D178 sweep |
| D183 match_response byte predictor | M | the per-message price is accurate (±0.4 %); the miss is selected requests never answered (0–5 per run) | `evaluation/runs/collab_d180_bytes/` |
| 6.3 overlap pruning, affine exposure; live re-anchoring (`pose_version`) | A2 | D192: the affine exposure likely never switches on live | |
| 5.4 dust freeze, 5.5 sweep render redesign, 5.6 Hilti fix | A2 | Hilti 0/100 (D144); the 5.5 one-pass writer is not recordable on g4dn | |
| D181 vertex cover, 3-agent re-measure (build if saving ≥ 20 %) | M | 0.0 % saving on 24/24 two-agent rows; pass A is 2 agents per scene (`collab_aws.sh`) | a 3-agent pass-A set: D212 worldgen scenes or the ma200 live recordings via D214 |
| 7.4 SQS/ASG sweeps | M | slotq is a host-side dispatcher | |
| 8.3 sweeps | B | after 8.2 | |
