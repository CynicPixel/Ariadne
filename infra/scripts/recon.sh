#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The reconstruction service's LOCAL (CPU) stages in the pinned ariadne-recon
# image (infra/docker/recon.Dockerfile; design RECON_1.7). Repo at /work.
#
#   infra/scripts/recon.sh test                         python unit tests
#   infra/scripts/recon.sh prepare SPOOL SCENE EVALREC OUT   fuse + init + geometry + render package
#   infra/scripts/recon.sh score OUT REMOTE             PSNR/SSIM (+ remote LPIPS) from remote renders
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
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
