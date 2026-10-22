#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Record a generated mission to MCAP and verify it (plan 1.3). Host wrapper
# around infra/scripts/sim_record.py in the ariadne-sim image.
#
#   infra/scripts/sim_record.sh SCENE_DIR OUT_REC_DIR [--seconds S] [--mission NAME] [--verify-only]
#
# SCENE_DIR comes from `infra/scripts/worldgen.sh generate PRESET SEED` (it holds
# mission_default.sdf/json and bridge_default.yaml). OUT_REC_DIR gets canonical/
# (IMU + GT pose written offline from the scripts, D39), sensors/ (the rendered
# bag with the live IMU/GT cross-check) and recording.json: strict checks, live
# gap counts, sha256, versions, git commit/dirty and image id. Exit 0 only if
# the recording verified.
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ARIADNE_SIM_IMAGE:-ariadne/sim:dev}"
scene="${1:?scene dir}"; out="${2:?output recording dir}"; shift 2
# A copy of the tree without .git (spinup.sh on AWS) passes ARIADNE_GIT_COMMIT / ARIADNE_GIT_DIRTY instead.
commit="${ARIADNE_GIT_COMMIT:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
dirty="${ARIADNE_GIT_DIRTY:-false}"
[ -z "${ARIADNE_GIT_DIRTY:-}" ] && [ -n "$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out' ':!evaluation/runs')" ] && dirty=true
render=(-e ARIADNE_RENDER_MODE=mesa-llvmpipe -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe)
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  render=(--gpus all -e ARIADNE_RENDER_MODE=nvidia-egl)
fi
exec docker run --rm --init "${render[@]}" -v "$REPO_ROOT":/work -w /work \
  -e ARIADNE_GIT_COMMIT="$commit" -e ARIADNE_GIT_DIRTY="$dirty" \
  -e ARIADNE_IMAGE_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}')" \
  --entrypoint bash "$IMAGE" -lc "source /opt/ros/jazzy/setup.bash && python3 infra/scripts/sim_record.py --scene $scene --out $out $*"
