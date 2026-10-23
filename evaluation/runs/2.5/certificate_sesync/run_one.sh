#!/usr/bin/env bash
# One SE-Sync benchmark graph through Certificate.SeSyncBenchmarkGraphs (D74), in its own container with a hard
# memory cap (no swap), so an over-budget graph is OOM-killed inside its cgroup instead of taking the host down.
# Peak memory = the container cgroup's memory.peak (cgroup v2), read after the test exits.
#   usage (repo root): evaluation/runs/2.5/certificate_sesync/run_one.sh <stem> [cap, default 6g]
set -u
STEM=$1; CAP=${2:-6g}
OUT=evaluation/runs/2.5/certificate_sesync; C=/data/code/Ariadne/.cache
docker run --rm --name "ariadne-cert-$STEM" --memory="$CAP" --memory-swap="$CAP" \
  -v "$PWD":/work -v $C:$C:ro -w /work \
  -e ARIADNE_PGO_SESYNC=$C/oracle/pgo_sesync -e ARIADNE_CERT_GRAPHS="$STEM" \
  ariadne/core-dev:dev bash -c '
    s=$(date +%s%N)
    build/docker/tests/test_opencv --gtest_filter=Certificate.SeSyncBenchmarkGraphs; rc=$?
    echo "exit $rc wall_ms $(( ($(date +%s%N) - s) / 1000000 ))"
    echo "memory.peak_bytes $(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo NA)"
    grep -E "^oom_kill " /sys/fs/cgroup/memory.events 2>/dev/null' > "$OUT/$STEM.log" 2>&1
echo "docker exit $?" >> "$OUT/$STEM.log"
