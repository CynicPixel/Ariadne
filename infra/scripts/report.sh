#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The 1.9 evaluation harness over one Phase 1 run chain (design HARNESS_1.9):
#   1. dump the §30.2 registry (C++, the single source of truth)
#   2. compute every metric + gates + oracles (analysis image: numpy, evo 1.31.1)
#   3. ariadne_report_check: MissingMetrics / UnknownMetrics on the report (the gate)
#   4. render figures + report.html
#
#   infra/scripts/report.sh COLLAB_RUN RECON_RUN MISSION_RUN SCENE OUT [SOLO_RUN...]
#
# Paths repo-relative. Exits non-zero if any gate, oracle or the §30.2 check fails.
# DIRTY_OK=1 allows dirty inputs (dev only; evidence must come from clean commits).
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
collab="${1:?collab run}" recon="${2:?recon run}" mission="${3:?mission run}" scene="${4:?scene}" out="${5:?out}"
shift 5
IMAGE=ariadne-analysis:1
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build -q -t "$IMAGE" -f "$REPO_ROOT/infra/docker/analysis.Dockerfile" "$REPO_ROOT/infra/docker" >/dev/null
dirty="$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out' ':!evaluation/runs' \
  | grep -q . && echo true || echo false)"
commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
mkdir -p "$REPO_ROOT/$out"

"$REPO_ROOT/infra/scripts/core.sh" dev bash -c "cmake --build /work/build/docker --target ariadne_report_check \
  --parallel 8 >/dev/null && /work/build/docker/cpp/runtime/ariadne_report_check --dump-registry /work/$out/registry.in.yaml"

compare=()
for c in "$@"; do compare+=("$c"); done
docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work" -w /work \
  -e ARIADNE_GIT_COMMIT="$commit" -e ARIADNE_GIT_DIRTY="$dirty" \
  -e ARIADNE_IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')" "$IMAGE" \
  python -m analysis.report compute --collab-run "$collab" --recon-run "$recon" --mission-run "$mission" \
  --scene "$scene" --registry "$out/registry.in.yaml" --out "$out" ${compare:+--mission-compare "${compare[@]}"} \
  ${DIRTY_OK:+--allow-dirty}
rm -f "$REPO_ROOT/$out/registry.in.yaml"

set +e
"$REPO_ROOT/infra/scripts/core.sh" dev bash -c \
  "/work/build/docker/cpp/runtime/ariadne_report_check --check /work/$out/report.yaml" > "$REPO_ROOT/$out/report_check.txt"
check=$?
set -e
cat "$REPO_ROOT/$out/report_check.txt"

docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work" -w /work "$IMAGE" python -m analysis.report render --out "$out"
exit "$check"
