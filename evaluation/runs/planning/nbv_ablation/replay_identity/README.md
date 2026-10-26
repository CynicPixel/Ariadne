# 4.3 ablation: replay identity check (next-best-view.md §15.5 amendment (a), before any run)

**What it checks.** With every new key off, the new binary reproduces the gate binary's decisions, except where the
always-on fixes fire (D119: a cut command; D125: an at-cell view).

**Input.** The local 20 s lockstep recording of ci_tiny seed 0, live on a1p `d59ee3d` with the gate `ariadne`
planner config (`/data/code/Ariadne-a1p/build/gate/local/repro_a/live-20261023T063623Z-8782`; its `inputs.mcap`
stays there, 1.1 GB). Its live trace is copied here as `live_trace_d59ee3d.jsonl`.

**Binary.** `ariadne_live` built from feat/ph4-nagent: `9efbdba` plus the D117/D118/D120 + live-loop D125
changes, committed right after as the commit that adds this directory. The manifests say
`git_commit: UNRECORDED` because the replay ran in a plain `docker run`, without the dev wrapper's
`ARIADNE_GIT_COMMIT`.

**Command.** One per row. `R` = the recording dir; `RUN` = its `aadffb8df5790ca0-live-*` subdir.

```
docker run --rm --memory=4g --memory-swap=4g -u 1000:1000 -v /data/code/Ariadne-2.1:/work \
  -v /data/code/Ariadne-a1p:/a1p:ro -w /work ariadne/core-dev:dev build/docker/cpp/runtime/ariadne_live \
  --scene /a1p/build/live/scenes/ci_tiny_0 --perception-config configs/perception/sim_v1.yaml \
  --slam-config configs/slam/sim_v1.yaml --mission-config configs/mission/mission_v1.yaml \
  --live-config $R/live_config.yaml --out build/replay_check/<row> --replay $R/$RUN/inputs.mcap \
  --directives $R/$RUN/directives.jsonl --shadow-map-config $R/shadow_map_config.yaml \
  --planner-config {$R/planner_config.yaml | configs/mission/ablation_4.3/mission_v2_ariadne_ta.yaml}
python3 -I trace_diff.py live_trace_d59ee3d.jsonl keys_off/trace.jsonl > keys_off_vs_live.txt
```

## Results

**keys_off** (the recorded gate config): all 8 decisions (4 replans × 2 agents) have the same choice and the same
command (kind and waypoints). The differences are:
- trace fields that only exist in the new schema (`candidates[].rank`, `utility`; from main's merge);
- `planner.allocation.bytes` 920 vs 904. This is the CBBA TASK_CONSENSUS wire size, which grew with main's appended
  D175/D171 wire fields (`aa972dc`, `950fbe3`). It is not this change.

**Limitation.** This recording has no cut command and no at-cell view, so it shows the fixes change nothing
elsewhere. It does not exercise the fixes themselves; `tests/unit/test_nbv_time.cpp` does that.

**keys_ta** (D117 + D118 + D120 on):
- the run completes and every trace record parses as JSON (the D120 `terms` block included);
- 0 unroutable options;
- every routed, uncut view has its arrival twin (n/2 twins per decision);
- the arrival twin was chosen in 7 of 8 decisions.

These are development observations, not a result. The ablation (48 runs) is the result.

**Twin sanity check (keys_ta, from the D120 term log).** Each chosen view was compared with its sibling: same task,
same eye, the other yaw.
- The chosen twins face 27–179° away from the target-facing view.
- Their coverage term is 0.92–1.04 (p90-normalised), against the sibling's 0.16–0.96.
- Same eye height (`SamplerParams.z`) and position as the sibling.
- So the higher gain is the forward-facing bundle seeing still-unknown space early in a 20 s mission, not a geometry
  mismatch. The one target-facing pick (t = 10 s, agent 0) had the higher gain (1.32 vs 0.85) despite 10 s of
  turning.

This is an observation on 8 decisions, not a result.
