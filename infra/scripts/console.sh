#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The operator console stub (arch §29, design live-runtime.md#operator-console) over a committed run chain.
#
#   infra/scripts/console.sh check COLLAB_RUN MISSION_RUN SCENE        chain checks (exit 1 on failure)
#   infra/scripts/console.sh serve COLLAB_RUN MISSION_RUN SCENE [PORT] http://127.0.0.1:PORT (default 8080)
#
# Binds 127.0.0.1 only. From another machine: ssh -L 8080:localhost:8080 <this host>, then open
# http://localhost:8080. (A claude.ai artifact page cannot reach a localhost WebSocket.)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE=ariadne-console:1
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build -q -t "$IMAGE" -f "$REPO_ROOT/infra/docker/console.Dockerfile" "$REPO_ROOT/infra/docker" >/dev/null
cmd="${1:-}"; shift || true
collab="${1:?collab run}" mission="${2:?mission run}" scene="${3:?scene}"
case "$cmd" in
  check)
    docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work:ro" -w /work "$IMAGE" \
      python -m operator_console.check "$collab" "$mission" "$scene" ;;
  serve)
    port="${4:-8080}"
    echo "Ariadne console: http://127.0.0.1:$port  (Ctrl-C to stop)"
    docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT:/work:ro" -w /work -p "127.0.0.1:$port:8080" \
      -e ARIADNE_CHAIN="$collab:$mission:$scene" "$IMAGE" \
      uvicorn operator_console.server:app --host 0.0.0.0 --port 8080 --log-level warning ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
