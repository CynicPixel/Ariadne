#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# D225 regression guard: the lockstep barrier must wait on EVERY lane (the pre-D225 wait was hardcoded to
# lanes 0 and 1; agent 2's keyframes slid past barrier cuts under load - the ma200 trace divergence).
# Method: replay a 3-agent recording TWICE, once with lane 2's parent-side delivery wall-delayed per frame
# (ARIADNE_TEST_LANE_DELAY - timing only, never data). A correct barrier gives byte-identical traces; the
# broken one diverges. PASS = identical.
#   test_barrier_n3.sh RUN_INNER_DIR SCENE   (defaults: the local ma_smoke3 run, maze_demo_0)
# ---------------------------------------------------------------------------
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:-$(ls -d "$REPO_ROOT"/build/live/ma_smoke3/live-*/8*-live-* 2>/dev/null | head -1)}"
SCENE="${2:-build/live/scenes/maze_demo_0}"
[ -f "$RUN/inputs.mcap" ] || { echo "no recording at $RUN"; exit 2; }
RUNREL="${RUN#"$REPO_ROOT"/}"
OUT="$REPO_ROOT/build/live/barrier_n3"
rm -rf "$OUT"; mkdir -p "$OUT"
LC="$(dirname "$RUNREL")/live_config.yaml"; [ -f "$REPO_ROOT/$LC" ] || LC="$RUNREL/../live_config.yaml"
for v in plain delayed; do
  env_delay=(); [ "$v" = delayed ] && env_delay=(-e ARIADNE_TEST_LANE_DELAY=2:40)
  docker run --rm --memory=4g --memory-swap=4g -u "$(id -u):$(id -g)" "${env_delay[@]}" \
    -v "$REPO_ROOT":/work -w /work ariadne/core-dev:dev build/docker/cpp/runtime/ariadne_live \
    --scene "$SCENE" --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v1.yaml \
    --mission-config configs/mission/mission_v1.yaml --live-config "$LC" \
    --planner-config configs/mission/mission_v2.yaml \
    --out "build/live/barrier_n3/$v" --replay "$RUNREL/inputs.mcap" \
    --directives "$RUNREL/directives.jsonl" > "$OUT/$v.log" 2>&1 || { echo "replay $v failed"; tail -3 "$OUT/$v.log"; exit 1; }
done
T1="$(ls -d "$OUT"/plain/8*)/trace.jsonl"; T2="$(ls -d "$OUT"/delayed/8*)/trace.jsonl"
if cmp -s "$T1" "$T2"; then echo "PASS: traces identical with lane-2 delay (the barrier waits on every lane)"; else
  echo "FAIL: lane-2 delay changed the trace (the barrier does not wait on every lane)"; exit 1; fi
