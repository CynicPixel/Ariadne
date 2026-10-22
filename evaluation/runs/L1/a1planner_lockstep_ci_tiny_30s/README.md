# A1Planner in the live loop, lockstep (test branch feat/l1-a1planner)

A1's Phase4Planner (feat/ph4-mission def61de, policy ariadne, CBBA L_t = 1) behind the runtime's MissionPlanner seam
(`cpp/runtime/live/a1_planner.hpp`), ci_tiny seed 0, 30 s, lockstep, classical chooser (= the planner's own pick),
Mesa llvmpipe:

    ARIADNE_LOCKSTEP=1 ARIADNE_PLANNER_CONFIG=configs/mission/mission_v2.yaml \
      infra/scripts/live.sh run build/live/scenes/ci_tiny_0 build/live/runs_a1p 30
    infra/scripts/live.sh replay RUN build/live/scenes/ci_tiny_0

- 6 barriers; every command applied AT its barrier; no mover refusal; no reorder or answer timeout.
- World paused 2.6 → 5.5 s of wall time per barrier while the planner ran (grows with the map).
- 28 → 172 options per agent; CBBA allocation record per decision in `trace.jsonl` (`planner.allocation`).
- flown 4.7 / 4.2 m in 30 s (`planner.flown_m`: ground-truth path length to T; `linked` constant true, stated).
- Replay ×2 from the recording = live (`replay_check.json`: PASS): the Phase 4 planner is deterministic in the loop.
One run, one scene, one renderer: not a gate result.
