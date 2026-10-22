# L1 — live closed-loop runtime: acceptance evidence

Design: `docs/design/LIVE_RUNTIME_L1.md`. Decisions: D150–D155, D152a, D153b (`docs/DECISIONS.md`).
Collected by `python/analysis/live_evidence.py RUN OUT` from `infra/scripts/live.sh run` + `live.sh replay`.

**Regime, stated:** local dev host, no GPU, **Mesa llvmpipe** rendering, real-time factor ≈ 0.14 (sim time runs
slower than wall time; every decision and stamp is in sim time). Live closed-loop runs are not bit-deterministic
across runs (arch §5.9); each run's RECORDING is, and that is what the replay check verifies. The AWS g4dn path
(`infra/scripts/live_aws.sh`) is built but was not run (spend not yet approved).

Scene `ci_tiny` seed 0 (`aadffb8df5790ca0`), two agents, each starting at its entrance with only the stationary start
hold scripted: all motion after 2 s is commanded by the runtime.

## `local_llvmpipe_45s/` — the acceptance run (`1bb27f9`, clean tree)

| | |
|---|---|
| decisions | 18 (every 5 s of sim time, both agents) — 16 `go`, 2 `hold` (no candidate) |
| mover answers | 13 accepted, **0 refused**, 3 unanswered (commands at 40/45 s: their answers arrived after the 45 s input cut, so they were not consumed) |
| motion (GT) | agent 0 drove 2.82 m, agent 1 5.08 m, under command only |
| decide → move → observe | e.g. t = 5 s: goto accepted; each agent's ESTIMATED distance to its goal 2.30 → 0.63 / 0.66 m within 5 s (`loop.json`) |
| replay | `inputs.mcap` (2.36 GB, sha256 `6bd62ab0…`, not committed) replayed twice: **PASS** — consumed digest, both agents' `estimate_body.tum` (sha256), the decision trace and the mover answers identical to the live run (`replay_check.json`) |
| console | 1729 frames served live over 350 s of wall time, sim time advancing to 45 s (`console_frames.jsonl`; served to `live.sh`'s probe WebSocket client, not a human) |

Limitation observed: every command is preempted by the next decision 5 s later (receding horizon, by design). When
consecutive commands share a goal but start with a turn, the agent can spend a replan turning without translating
(agent 0 held near (0.42, 3.08) from 15 to 25 s). Planner policy is Phase 4's (4.3 NBV); not tuned here.

## `local_llvmpipe_30s_first/` — the first acceptance run (`943a558`), kept because it changed the design

The loop closed at t = 5 s (both gotos accepted; estimated distance to goal 2.50 → 0.85 / 0.88 m), then the mover's
GT-envelope check **refused 10/10** follows: the goal was the frontier cell itself, next to an unseen wall (D153b).
Its console served 1012 frames all at sim time 0 (wrong directory, D154 fix). Replay check: PASS (determinism holds
even when the behaviour is wrong).

## Not exercised

IMU: bridged, relayed and recorded, not consumed (`slam/sim_v1` has no IMU mode, as in replay); D39's live IMU-gap
bridging is untouched. Collaboration: each agent is anchored by its own briefed entry pose; no inter-agent merge live.
