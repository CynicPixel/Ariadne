#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the ariadne-sim image with the right render path for this host.
#
#   infra/scripts/sim.sh                  # acceptance test (the image's CMD)
#   infra/scripts/sim.sh gz sim -s -r --headless-rendering <world.sdf>
#
# The same image runs on a GPU-less dev box and on a g5/g6 instance; only the
# docker flags differ, and this script is the one place that decides them:
#   - NVIDIA runtime present  -> --gpus all; the toolkit injects driver EGL.
#   - otherwise               -> Mesa llvmpipe software EGL. Slow but correct,
#                                good enough to validate worlds and the bridge.
# Nothing is installed on the host: Docker is the only dependency.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ARIADNE_SIM_IMAGE:-ariadne/sim:dev}"
OUT_DIR="${ARIADNE_SMOKE_OUT_HOST:-$REPO_ROOT/build/sim-out}"
mkdir -p "$OUT_DIR"

run_args=(--rm --init -v "$OUT_DIR":/out -e ARIADNE_SMOKE_OUT=/out)

# Forward the acceptance-test knobs when set. Without this, a host-side
# ARIADNE_SMOKE_TIMEOUT silently never reached the container, and a 180 s run
# was misread as a 60 s one (3x the real frame rate).
for var in ARIADNE_SMOKE_TIMEOUT ARIADNE_SMOKE_WORLD; do
  [ -n "${!var:-}" ] && run_args+=(-e "$var=${!var}")
done

if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  run_args+=(--gpus all -e ARIADNE_RENDER_MODE=nvidia-egl)
else
  # llvmpipe is Mesa's multithreaded software rasteriser; LIBGL_ALWAYS_SOFTWARE
  # keeps Mesa from probing for a hardware DRI device that does not exist.
  run_args+=(-e ARIADNE_RENDER_MODE=mesa-llvmpipe
             -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe)
fi

exec docker run "${run_args[@]}" "$IMAGE" "$@"
