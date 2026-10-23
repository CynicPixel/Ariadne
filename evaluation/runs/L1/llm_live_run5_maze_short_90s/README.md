# 7.1 LLM live run 5 — A1 planner + Nova orchestrator, maze_short_6, 90 s (PASS)

Run live-20261023T084158Z-64df, 2026-10-23 08:42–08:55 UTC, continuous (no lockstep),
chooser llm (live_llm_v1: gate 0.5, timeout 120 s), planner mission_v2 (A1 Phase4Planner,
policy ariadne), scene maze_short_6, mesa-llvmpipe. Branch feat/l1-a1p-llm (working tree:
run-3 + run-4 fixes, commit pending).

## Result
- Runtime exit 0, full 90 s mission: 18 replans, 36/36 decisions answered by the
  orchestrator (Nova 2 Lite), 0 errors, 0 fallbacks, 0 crashes. 86 481 messages consumed
  (digest e4d957e844fa7211), 1801 frames, 130 keyframes per agent-0 track.
- Transport: 1797/1800 agent-0 left images and 17 971/18 000 IMU messages delivered at the
  bridge; the runtime consumed everything delivered (producer-paced; 90 s sim ≈ 787 s wall,
  8.7x, llvmpipe).
- Replay determinism: replay_check.json verdict PASS, all 15 checks (two replays from
  inputs.mcap alone, trace/mover_state/directives sha256 identical to live and to each other).
- Picks: Nova took the planner's rank-0 candidate 27/36 times; 17 picks under 1 m of which
  11 WERE the planner's own top rank (maze_short frontier clusters are dense — planner-side,
  not a chooser fault); 6 picks took a tiny goal against instruction. No option crossed the
  0.5 risk gate, so no approval requests and the reserve-return exemption never triggered.

## The three failed predecessors (why runs 1–4 are not evidence)
- run 3 live-20261023T064414Z-e8b3 t=40 s: one-time env-file credential export expired
  (~15 min); botocore raised RuntimeError (uncaught class), every later replan waited out
  its 120 s timeout. Fix: host re-exports session keys every 240 s into a mode-700 dir read
  via credential_process; orchestrator answers on ANY exception.
- run 4 live-20261023T073340Z-e971 t=55 s: the decision request grew past asyncio's default
  64 KiB line limit; LimitOverrunError killed the handler outside the per-line guard; the
  runtime threw on the dead socket and std::terminate'd (exit 139). Fix: 16 MiB LINE_LIMIT +
  guarded stream loop + handler closes its writer; runtime degrades a dead chooser link to
  the classical fallback. Regression test: 300 KiB request line answered (orchestrator 5/5).
- Nova's tiny-step goals (0.05–0.9 m, run 3): candidates now carry rank + planner utility,
  best-first, and the prompt prefers the top rank and discourages sub-1 m goals.

## Files
inputs.mcap stays in the run dir (5.5 GB); its sha256 is inputs.mcap.sha256. decisions/ has
every request/choice pair; trace.jsonl is the agentic evidence trail (vision -> decision ->
action, D156/D161); replay_check.json is the determinism verdict.
