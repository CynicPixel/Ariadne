#!/usr/bin/env bash
# One host-side controller, backed by the existing prewarmed Docker images.
# Usage: infra/scripts/demo.sh [PORT]. Build browser assets once: npm ci && npm run build
# in python/operator_console. No cloud provisioning or image pulls occur here.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[ -f python/operator_console/static/dist/main.js ] || {
  echo 'Build frontend first: cd python/operator_console && npm ci && npm run build' >&2
  exit 1
}
export PYTHONPATH="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export ARIADNE_DEMO=1
exec "${ARIADNE_DEMO_PYTHON:-python3}" -m uvicorn operator_console.server:app \
  --host "${ARIADNE_DEMO_BIND:-127.0.0.1}" --port "${1:-8080}" --workers 1
