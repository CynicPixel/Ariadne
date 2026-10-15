#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The reconstruction service's LOCAL (CPU) stages in the pinned ariadne-recon
# image (infra/docker/recon.Dockerfile; design RECON_1.7). Repo at /work.
#
#   infra/scripts/recon.sh test                         python unit tests
#   infra/scripts/recon.sh prepare SPOOL SCENE EVALREC OUT   fuse + init + geometry + render package
#   infra/scripts/recon.sh score OUT REMOTE             PSNR/SSIM (+ remote LPIPS) from remote renders
#   infra/scripts/recon.sh viewer OUT RUN SPOOL PAGE [MISSION SCENE]  WebGL splat page (+ 1.8 egress route)
#   infra/scripts/recon.sh mission-eval RUN SCENE COLLAB_RUN [SOLO_RUN...]  1.8 mission run vs scene GT
#
# Paths are repo-relative. The remote render between prepare and score:
#   infra/scripts/kaggle.sh render OUT/package REMOTE
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=ariadne-recon:1
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build -q -t "$IMAGE" -f "$REPO_ROOT/infra/docker/recon.Dockerfile" "$REPO_ROOT/infra/docker" >/dev/null
dirty="$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out' ':!evaluation/runs' \
  | grep -q . && echo true || echo false)"
run() {
  docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work" \
    -e ARIADNE_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" -e ARIADNE_GIT_DIRTY="$dirty" \
    -e ARIADNE_IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')" "$IMAGE" "$@"
}

case "${1:-}" in
  test) run python -m pytest -q -p no:cacheprovider python/reconstruction/tests ;;
  prepare)
    run python -m ariadne_recon.cli prepare --spool "/work/${2:?spool}" --scene "/work/${3:?scene}" \
      --eval-record "/work/${4:?recon_eval.yaml}" --config /work/configs/recon/recon_v1.yaml --out "/work/${5:?out}" ;;
  score) run python -m ariadne_recon.cli score --out "/work/${2:?out}" --remote "/work/${3:?remote dir}" ;;
  viewer) run python -m ariadne_recon.cli viewer --out "/work/${2:?out}" --run "/work/${3:?sim_collab run}" \
      --spool "/work/${4:?spool}" --page "/work/${5:?page.html}" ${6:+--mission "/work/$6" --scene "/work/${7:?scene for the route grid}"} ;;
  mission-eval)
    run_dir="${2:?mission run}" scene="${3:?scene}" collab="${4:?sim_collab run}"; shift 4
    compare=(); for c in "$@"; do compare+=("/work/$c"); done
    docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work" -w /work -e PYTHONPATH=/work/python \
      -e ARIADNE_GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" -e ARIADNE_GIT_DIRTY="$dirty" "$IMAGE" \
      python -m analysis.mission_eval --run "/work/$run_dir" --scene "/work/$scene" --collab-run "/work/$collab" \
      --config /work/configs/mission/mission_v1.yaml ${compare:+--compare "${compare[@]}"} ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
