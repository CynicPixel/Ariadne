#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The live closed-loop run (design docs/design/LIVE_RUNTIME_L1.md, D150-D155).
#
#   infra/scripts/live.sh build                    sim side: the mover plugin + ros2/ (colcon) in ariadne/sim
#   infra/scripts/live.sh compose PRESET SEED RTF  scene + mission_live.* -> build/live/scenes/<preset>_<seed>
#   infra/scripts/live.sh run SCENE OUT [END_S]    one live run -> OUT/<run>/  (+ sim/console logs)
#   infra/scripts/live.sh replay RUN_DIR SCENE     the recording, replayed twice, compared (replay_check.json)
#
# Three containers per run, named ariadne-live-<id>-{rt,sim,console}. Only those are ever stopped, by the exact
# container ids `docker run -d` printed for them (other sessions share the docker daemon):
#   rt       ariadne/core-dev  ariadne_live --link <run>/link.sock  (listens first)
#   sim      ariadne/sim       ros2 launch ariadne_sim live_world.launch.py: gz sim PAUSED + RosGzBridge +
#                              RuntimeLink in one component container; unpaused here once <run>/link.ready exists
#   console  ariadne-console   ARIADNE_LIVE=<run>:<scene> uvicorn, 127.0.0.1:${ARIADNE_CONSOLE_PORT:-8080};
#                              a probe client keeps a WebSocket open so frames are served (and logged) during the run
# Render: NVIDIA EGL when the docker NVIDIA runtime exists (the AWS g4dn host), else Mesa llvmpipe (slow; smoke only).
# Parallelism: the builds use 2 jobs (the dev host crashed above that).
# ---------------------------------------------------------------------------
set -Eeuo pipefail
trap 'echo "[live] FAILED (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM_IMAGE="${ARIADNE_SIM_IMAGE:-ariadne/sim:dev}"
CORE_IMAGE="${ARIADNE_CORE_IMAGE:-ariadne/core-dev:dev}"
CONSOLE_IMAGE="${ARIADNE_CONSOLE_IMAGE:-ariadne-console:1}"
JOBS="${ARIADNE_BUILD_JOBS:-2}"
U="$(id -u):$(id -g)"
log() { printf '[live %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

render_args() {
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "--gpus all -e ARIADNE_RENDER_MODE=nvidia-egl"
  else
    echo "-e ARIADNE_RENDER_MODE=mesa-llvmpipe -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe"
  fi
}

sim_run() {   # one-shot command in the sim image with ROS sourced, repo at /work
  docker run --rm -u "$U" -v "$REPO_ROOT":/work -w /work "$SIM_IMAGE" bash -c "source /opt/ros/jazzy/setup.bash && $*"
}

cmd_build() {
  log "sim plugins (mover live mode) -> build/sim-plugins"
  sim_run "cmake -S simulator/plugins -B build/sim-plugins -DCMAKE_BUILD_TYPE=Release >/dev/null && \
           cmake --build build/sim-plugins --parallel $JOBS"
  log "ros2 packages -> build/ros2/install"
  sim_run "colcon --log-base build/ros2/log build --base-paths ros2 --build-base build/ros2/build \
           --install-base build/ros2/install --parallel-workers $JOBS --event-handlers console_direct- \
           --cmake-args -DCMAKE_BUILD_TYPE=Release"
}

cmd_compose() {
  local preset="$1" seed="$2" rtf="$3" out="build/live/scenes/${1}_${2}"
  log "compose $preset seed $seed (real-time factor $rtf) -> $out"
  sim_run "cd python && python3 -m dataset_tools.worldgen generate --preset ../configs/worldgen/$preset.yaml \
           --seed $seed --out ../$out --rig ../configs/sim/rig_stereo_v1.yaml --live-rtf $rtf"
}

CONTAINERS=()
stop_mine() {
  local c
  for c in "${CONTAINERS[@]}"; do docker stop -t 10 "$c" >/dev/null 2>&1 || true; done
}

cmd_run() {
  local scene="$1" out="$2" end_s="${3:-}"
  [ -f "$REPO_ROOT/$scene/mission_live.sdf" ] || { log "no $scene/mission_live.sdf (live.sh compose)"; exit 2; }
  local id; id="$(date -u +%Y%m%dT%H%M%SZ)-$(head -c2 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  local run="$out/live-$id"
  mkdir -p "$REPO_ROOT/$run"
  local live_cfg=configs/mission/live_v1.yaml
  if [ -n "$end_s" ]; then
    sed "s/^end_s: .*/end_s: $end_s/" "$REPO_ROOT/$live_cfg" > "$REPO_ROOT/$run/live_config.yaml"
    live_cfg="$run/live_config.yaml"
  fi
  trap 'stop_mine' EXIT
  local world; world="$(grep -o '<world name="[^"]*"' "$REPO_ROOT/$scene/mission_live.sdf" | head -1 | cut -d'"' -f2)"
  log "run $id: scene $scene, world '$world', end ${end_s:-config} s, render: $(render_args | cut -d' ' -f1-2)"

  # 1. The runtime listens first.
  CONTAINERS+=("$(docker run -d --name "ariadne-live-$id-rt" -u "$U" -v "$REPO_ROOT":/work -w /work \
    -e ARIADNE_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    -e ARIADNE_GIT_DIRTY="$([ -n "$(git -C "$REPO_ROOT" status --porcelain -- cpp configs ros2 simulator)" ] && echo true || echo false)" \
    -e ARIADNE_IMAGE_DIGEST="$(docker image inspect "$CORE_IMAGE" --format '{{.Id}}')" \
    "$CORE_IMAGE" build/docker/cpp/runtime/ariadne_live --scene "$scene" \
      --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v1.yaml \
      --mission-config configs/mission/mission_v1.yaml --live-config "$live_cfg" \
      --out "$run" --link "$run/link.sock")")

  # 2. gz (paused) + bridge + RuntimeLink.
  # shellcheck disable=SC2046
  CONTAINERS+=("$(docker run -d --name "ariadne-live-$id-sim" -u "$U" $(render_args) -v "$REPO_ROOT":/work -w /work \
    -e GZ_SIM_SYSTEM_PLUGIN_PATH=/work/build/sim-plugins -e HOME=/tmp \
    -e GZ_TRANSPORT_RCVHWM=0 -e GZ_TRANSPORT_SNDHWM=0 \
    "$SIM_IMAGE" bash -c "source /opt/ros/jazzy/setup.bash && source build/ros2/install/setup.bash && \
      ros2 launch ariadne_sim live_world.launch.py scene_dir:=/work/$scene run_dir:=/work/$run")")

  # 3. Unpause once every topic is matched end to end.
  local t0; t0=$(date +%s)
  until [ -f "$REPO_ROOT/$run/link.ready" ]; do
    [ $(( $(date +%s) - t0 )) -lt 300 ] || { log "link not ready after 300 s"; docker logs "${CONTAINERS[1]}" 2>&1 | tail -40 >&2; exit 1; }
    docker inspect -f '{{.State.Running}}' "${CONTAINERS[1]}" | grep -q true || { log "sim container exited"; docker logs "${CONTAINERS[1]}" 2>&1 | tail -40 >&2; exit 1; }
    sleep 2
  done
  log "link ready: $(cat "$REPO_ROOT/$run/link.ready")"
  docker exec "${CONTAINERS[1]}" bash -c "source /opt/ros/jazzy/setup.bash && gz service -s /world/$world/control \
    --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 10000 --req 'pause: false'" >&2
  log "world running"

  # 4. Console, fed live, with a probe client so frames are served during the run.
  local port="${ARIADNE_CONSOLE_PORT:-8080}"
  CONTAINERS+=("$(docker run -d --name "ariadne-live-$id-console" -u "$U" -v "$REPO_ROOT":/work -w /work/python \
    -p "127.0.0.1:$port:8080" -e ARIADNE_LIVE="/work/$run:/work/$scene" "$CONSOLE_IMAGE" \
    uvicorn operator_console.server:app --host 0.0.0.0 --port 8080 --log-level warning)")
  docker exec -d "${CONTAINERS[2]}" python3 -c "
import asyncio, websockets
async def main():
    await asyncio.sleep(3)
    async with websockets.connect('ws://127.0.0.1:8080/ws/telemetry') as ws:
        async for _ in ws: pass
asyncio.run(main())" || log "console probe client not started (websockets missing?)"

  # 5. Wait for the runtime (it ends at end_s of sim time), then stop the rest.
  local rc; rc="$(docker wait "${CONTAINERS[0]}")"
  docker logs "${CONTAINERS[0]}" > "$REPO_ROOT/$run/runtime.log" 2>&1 || true
  docker logs "${CONTAINERS[1]}" > "$REPO_ROOT/$run/sim.log" 2>&1 || true
  sleep 3   # the console's last poll sees results.yaml and ends its stream
  docker logs "${CONTAINERS[2]}" > "$REPO_ROOT/$run/console.log" 2>&1 || true
  printf '%s\n' "${CONTAINERS[@]}" > "$REPO_ROOT/$run/container_ids.txt"
  stop_mine
  for c in "${CONTAINERS[@]}"; do docker rm "$c" >/dev/null 2>&1 || true; done
  CONTAINERS=()
  log "runtime exit $rc; run dir $run"
  [ "$rc" = 0 ]
}

cmd_replay() {
  local run="$1" scene="$2" i
  for i in 1 2; do
    docker run --rm -u "$U" -v "$REPO_ROOT":/work -w /work "$CORE_IMAGE" build/docker/cpp/runtime/ariadne_live \
      --scene "$scene" --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v1.yaml \
      --mission-config configs/mission/mission_v1.yaml \
      --live-config "$( [ -f "$REPO_ROOT/$run/live_config.yaml" ] && echo "$run/live_config.yaml" || echo configs/mission/live_v1.yaml)" \
      --out "$run/replay$i" --replay "$run/$(cd "$REPO_ROOT/$run" && ls -d */inputs.mcap | head -1)" \
      > "$REPO_ROOT/$run/replay$i.log" 2>&1
  done
  python3 "$REPO_ROOT/python/analysis/live_replay_check.py" "$REPO_ROOT/$run"
}

case "${1:-}" in
  build) shift; cmd_build "$@" ;;
  compose) shift; cmd_compose "$@" ;;
  run) shift; cmd_run "$@" ;;
  replay) shift; cmd_replay "$@" ;;
  *) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
