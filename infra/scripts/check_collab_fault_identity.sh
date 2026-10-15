#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# The 1.6 protocol check (design COLLAB_1.6, D43): a sim-collab run with
# --inject-faults 1 must produce the same results as one without. Compares
#   - candidates.csv and negative_probe.csv byte for byte,
#   - results.yaml with only the inject_faults / protocol lines removed,
#   - each agent's raw estimate_body.tum by sha256,
# and requires the fault run to have actually exercised the inbox (duplicates
# and buffered reorders on both links). Writes the verdict into the fault run's
# directory (fault_identity.txt) and exits non-zero on FAIL.
#
#   infra/scripts/check_collab_fault_identity.sh CLEAN_RUN_DIR FAULT_RUN_DIR
# ---------------------------------------------------------------------------
set -euo pipefail

clean="${1:?clean run dir}" fault="${2:?fault run dir}"
report="$fault/fault_identity.txt"
fail=0
{
  echo "# 1.6 fault identity: $(basename "$clean") vs $(basename "$fault")"
  for f in candidates.csv negative_probe.csv; do
    if cmp -s "$clean/$f" "$fault/$f"; then echo "PASS $f byte-identical"; else echo "FAIL $f differs"; fail=1; fi
  done
  if diff <(grep -v -E '^(inject_faults|protocol):' "$clean/results.yaml") \
          <(grep -v -E '^(inject_faults|protocol):' "$fault/results.yaml") >/dev/null; then
    echo "PASS results.yaml identical except inject_faults/protocol"
  else
    echo "FAIL results.yaml differs"; fail=1
  fi
  grep -q '^inject_faults: 0' "$clean/results.yaml" || { echo "FAIL clean run has faults on"; fail=1; }
  grep -q '^inject_faults: 1' "$fault/results.yaml" || { echo "FAIL fault run has faults off"; fail=1; }
  proto="$(grep '^protocol:' "$fault/results.yaml")"
  echo "fault run $proto"
  for k in broker_duplicates broker_buffered agent1_duplicates agent1_buffered; do
    n="$(sed -E "s/.*$k: ([0-9]+).*/\1/" <<<"$proto")"
    [ "$n" -gt 0 ] || { echo "FAIL $k = 0 (fault not exercised)"; fail=1; }
  done
  for agent in agent0 agent1; do
    a="$(sha256sum "$clean"/raw/*-"$agent"-*/estimate_body.tum | cut -d' ' -f1)"
    b="$(sha256sum "$fault"/raw/*-"$agent"-*/estimate_body.tum | cut -d' ' -f1)"
    if [ -n "$a" ] && [ "$a" = "$b" ]; then echo "PASS $agent estimate_body.tum $a"; else echo "FAIL $agent estimate differs"; fail=1; fi
  done
  [ "$fail" -eq 0 ] && echo "VERDICT PASS" || echo "VERDICT FAIL"
} > "$report"   # not a pipe: `fail` must survive the group (a pipe would run it in a subshell)
cat "$report"
exit "$fail"
