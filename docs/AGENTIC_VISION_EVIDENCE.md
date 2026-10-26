# Ariadne: Agentic Vision evidence

The typeset version is `docs/Latex/AGENTIC_VISION_EVIDENCE.pdf`, built from `AGENTIC_VISION_EVIDENCE.tex`. The design is in `docs/design/AGENTIC_7.md`.

## Workflow: perception → orchestration → action
```
Perceive (OpenCV 5)              Interpret                       Decide                          Act
calcOpticalFlowPyrLK, FAST/ORB,  stereo-inertial VIO (10-KF       Bedrock Nova 2 Lite via MCP:    C++ validation (candidate set,
findFundamentalMat USAC,         Schur-LM window, IMU, loops) →   list_candidates, observe_map,   reachability, risk, uniqueness)
StereoSGBM, solvePnPRansac,      TSDF/ESDF map, frontier          observe_image (OpenCV-rendered  → goto / follow / turn / hold
cv::dnn EigenPlaces,             candidates, risk                 map + keyframes),               → collision-checked mover
ALIKED + LightGlue               (clearance + structural proxy)   query_uncertainty, query_route,   (accept / refuse) → new frames
                                                                  assign, request_human_approval
        ▲─────────────────────── new camera frames and command outcomes ─────────────────────────────┘
```
- **When the model fails:** the chooser falls back to the in-region DeepSeek, but only on a Bedrock error or throttling, never on a bad answer.
- **Risk gate:** candidates above risk 0.5 need an operator approval packet.
- **Replays never call the model:** recorded decisions are re-applied (D161).

## Runs

| run | outcome | evidence |
|---|---|---|
| **Final integrated stack** (AWS, 2 agents, 200 s, `maze_short_6`) | 40 joint replans; **80/80 Nova 2 Lite decisions validated**, 0 fallbacks, 0 answer timeouts. 80 commands (70 goto, 8 follow, 2 turn). The mover accepted 79 of 82 command states and refused 3 at the 0.20 m ESDF collision envelope. 4,001 frames and 335/342 keyframes per agent; VIO = `sim_v2_loops` with seeded IMU noise; GPU splat from 677 keyframes; replay PASS. Source: `c143db14` plus a recorded working-tree diff (live wiring of the full VIO configuration; `source_status.txt`) | `Ariadne-e2e-final/evaluation/runs/final_stack_e2e/20261026T000432Z/` (trace.jsonl, decisions/, snapshots/, mover_state.jsonl, results.yaml) |
| First Bedrock loop (90 s) | 18 joint replans, 36/36 assignments, 0 fallbacks, 15/15 replay checks | `evaluation/runs/L1/llm_live_run5_maze_short_90s/` |
| Matched chooser pair (90 s) at `c143db14` | Bedrock differs from classical on 6/18 replans; rank-0 picks 31/36 vs 26/36; 34/36 commands accepted per arm, none refused; replay PASS | `evaluation/runs/agentic_planning_pair_20261025/` |
| Matched pair (500 s) | 100 joint replans, 200 commands and 10,001 frames per agent in each arm; two replays per arm identical | `evaluation/runs/agentic_planning_pair_500s_20261025/` |
| Live loop, local | 45 s, 13 commands accepted, 0 refused; replay identical twice | `evaluation/runs/L1/local_llvmpipe_45s/` |
| Three-agent coordination | 20 s: 6 coordinated assignments, 3 no-consensus and 3 no-assignment events (pre-registered); double replay PASS | D218, `build/live/ma_smoke3` |
| Allocation oracle | CBBA = centralised greedy on 40/40 replans; 0.994 of the exact optimum; 11.8 kB sparse vs 123.4 kB dense | D115, `evaluation/runs/allocation_4.4_replay/` |

## Final-stack decision statistics
Source: `trace.jsonl`, 80 agent records.
- **Tool calls:** `list_candidates`, `query_route` and `assign` 40 each; `observe_map` and `query_uncertainty` 74 each; `observe_image` 31. That is 196 Bedrock calls in total.
- **Latency:** median 3.43 s, p90 9.40 s, maximum 13.22 s.
- **Choices:**
  - The model picked the rank-0 candidate in 73 of 80 records and departed from the classical choice in 6.
  - **Every record offered a `needs_approval` candidate, and the model chose one in 0 of 80.**

## OpenCV output changing the next action
Example: agent 0 in the final-stack run.

| t (s) | mapped free cells | candidates | validated `goto` x (m) | model's stated reason |
|---|---|---|---|---|
| 10 | 0 | 16 | 0.83 | best-ranked candidate, low risk, reveals new space |
| 15 | 1,986 | 40 | 1.38 | same |
| 20 | 3,205 | 48 | 2.88 | same |
| 25 | 6,628 | 63 | 2.28 | same |
| 30 | 8,537 | 79 | 2.93 | same |

1. Stereo keyframes are integrated into the TSDF.
2. The map grows, and the candidate set is regenerated from it.
3. The model queries the map, uncertainty and route, then assigns a newly revealed candidate.
4. The validated waypoint is accepted and executed by the mover.
5. The mover's new frames feed the next replan.

The run's decision map and keyframes are in `docs/Latex/figures/fig_decision.png`.

## Failure handling, observability, human control
- **Validation:** C++ rejects any answer that is a timeout, malformed, outside the candidate set, above the risk threshold without an approval, or a duplicate assignment (`cpp/runtime/tools/ariadne_live.cpp`).
- **Refusal feedback:**
  - The mover refuses goals outside the collision envelope. It did so 3 times in the final run (ESDF clearance 0.175–0.190 m against a 0.20 m limit).
  - Refused goals are excluded from the next replans (D153b).
  - Earlier, an early frontier-follow configuration drew 10 refusals. Goals were then constrained to standoffs in observed space, and 9 recomputed paths passed the ground-truth ESDF clearance check.
- **Observability:**
  - per-decision JSONL holding the vision summary, candidates, classical choice, chosen task, tool calls, latency, tokens and command;
  - per-decision map and keyframe PNGs;
  - `mover_state.jsonl`;
  - live-vs-replay digests;
  - the hosted console (https://www.ariadne.cynicpixel.xyz/), with camera feature overlays, trajectories, task assignments, link traffic, responder routes, and operator actions (link cut/restore, false-edge injection).
- **Human authority:**
  - The operator sends directives through the console.
  - Any candidate above risk 0.5 needs an approval packet.
  - The responder route is computed from observed 3-D clearance at ρ = 0.40 m, independent of the model.
  - The approval routing is exercised by scripted and stubbed tests. The live Bedrock arms never chose an approval-gated candidate.
