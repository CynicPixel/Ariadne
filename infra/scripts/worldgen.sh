#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Procedural world generation (arch §5.5) in the pinned ariadne-sim image.
#
#   infra/scripts/worldgen.sh test
#       worldgen unit tests (pytest inside the image: the numpy/pyyaml that
#       generate the worlds are the ones tested)
#   infra/scripts/worldgen.sh simtest
#       sim-side tests with ROS sourced: builds the plugin + canonical-stream
#       tool and pins the canonical IMU/GT streams of ci_tiny seed 0 (D39)
#   infra/scripts/worldgen.sh census PRESET [SEEDS=0:1000]
#   infra/scripts/worldgen.sh accept PRESET [SEEDS=0:1000]
#       seed sweeps -> evaluation/runs/phase1/worldgen_<cmd>/<preset>-<UTC>/
#   infra/scripts/worldgen.sh preview PRESET SEED [OUT=build/worldgen/<preset>_<seed>.png]
#   infra/scripts/worldgen.sh generate PRESET SEED [OUT=build/worldgen/scenes/<preset>_<seed>]
#       full scene directory (model.sdf, meshes, textures, world.sdf, scene_gt.npz,
#       scene.json, preview.png, manifest.json)
#
# The image is the authority on generated worlds: numpy 1.26.4 / pyyaml 6.0.1
# come from the dated ROS snapshot, so a world is reproducible from
# (GENERATOR_VERSION, seed, preset) plus this image id, all recorded.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ARIADNE_SIM_IMAGE:-ariadne/sim:dev}"

# Same exclusions as core.sh: paths that cannot change what runs here.
git_dirty() {
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out' ':!evaluation/runs')" ]; then
    echo true
  else
    echo false
  fi
}

run() {
  docker run --rm --init -v "$REPO_ROOT":/work -w /work \
    -e PYTHONPATH=/work/python \
    -e ARIADNE_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" \
    -e ARIADNE_GIT_DIRTY="$(git_dirty)" \
    -e ARIADNE_IMAGE_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}')" \
    --entrypoint python3 "$IMAGE" "$@"
}

preset_path() {
  local p="configs/worldgen/$1.yaml"
  [ -f "$REPO_ROOT/$p" ] || { echo "no preset $p" >&2; exit 1; }
  echo "$p"
}

case "${1:-}" in
  test)
    run -m pytest -q -p no:cacheprovider python/tests/worldgen
    ;;
  simtest)
    docker run --rm --init -v "$REPO_ROOT":/work -w /work -e PYTHONPATH=/work/python \
      --entrypoint bash "$IMAGE" -lc \
      "source /opt/ros/jazzy/setup.bash && python3 -m pytest -q -p no:cacheprovider python/tests/sim"
    ;;
  census|accept)
    cmd="$1"; preset="${2:?preset name}"; seeds="${3:-0:1000}"
    path="$(preset_path "$preset")"      # assignment: a failed lookup stops the script
    out="evaluation/runs/phase1/worldgen_${cmd}/${preset}-$(date -u +%Y%m%dT%H%M%SZ)"
    run -m dataset_tools.worldgen "$cmd" --preset "$path" --seeds "$seeds" --out "$out"
    echo "wrote $out"
    ;;
  preview)
    preset="${2:?preset name}"; seed="${3:?seed}"
    out="${4:-build/worldgen/${preset}_${seed}.png}"
    path="$(preset_path "$preset")"
    mkdir -p "$REPO_ROOT/$(dirname "$out")"
    run -m dataset_tools.worldgen preview --preset "$path" --seed "$seed" --out "$out"
    ;;
  generate)
    preset="${2:?preset name}"; seed="${3:?seed}"
    out="${4:-build/worldgen/scenes/${preset}_${seed}}"
    path="$(preset_path "$preset")"
    run -m dataset_tools.worldgen generate --preset "$path" --seed "$seed" --out "$out" \
      --rig "${ARIADNE_RIG:-configs/sim/rig_stereo_v1.yaml}"
    echo "wrote $out"
    ;;
  *)
    sed -n '2,20p' "$0"; exit 2
    ;;
esac
