#!/usr/bin/env bash
# live_to_rec.sh RUN_INNER_DIR SCENE OUT  (P8 D214): a live recording as a sim_record-layout recording, so
# SimSource-based tools (3.2 pass A, ariadne_sim_vo --spool) read it unchanged. Canonical bag via
# ariadne_live_canonical; sensors = a hard link to inputs.mcap; recording.json with sha256 + per-topic counts.
set -euo pipefail
RUN="$1" SCENE="$2" OUT="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$OUT/sensors"
LOG="$(mktemp)"
docker run --rm -u "$(id -u):$(id -g)" -v "$REPO_ROOT":/work -w /work "${ARIADNE_CORE_IMAGE:-ariadne/core:dev}" \
  build/docker/cpp/runtime/ariadne_live_canonical --run "$RUN" --scene "$SCENE" --out "$OUT" | tee "$LOG"
ln -f "$RUN/inputs.mcap" "$OUT/sensors/inputs.mcap" 2>/dev/null || cp "$RUN/inputs.mcap" "$OUT/sensors/inputs.mcap"
SHA=$(sha256sum "$OUT/sensors/inputs.mcap" | cut -d' ' -f1)
{
  echo "streams:"
  echo "  sensors:"
  echo "    files:"
  echo "      inputs.mcap: {sha256: $SHA}"
  echo "result:"
  echo "  topics:"
  grep '^count ' "$LOG" | while read -r _ topic n; do echo "    $topic: {count: $n}"; done
} > "$OUT/recording.json"
rm -f "$LOG"
echo "live_to_rec: $OUT ready"
