#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared AWS slot queue: keeps the CPU spot quota busy. Sessions drop job files
# into the queue; one dispatcher launches the next job, in name order, as soon
# as the quota has room for it. A freed slot waits at most POLL_S seconds.
#
#   infra/scripts/slotq.sh dispatch          # the one dispatcher (M runs it)
#   infra/scripts/slotq.sh status            # queue, running, done, free vCPU
#
# A job is $Q/queue/<NN>-<name>.job, priority = name order (10-, 20-, ...):
#   OWNER=A1                       # session to tell when it starts/ends
#   DIR=/data/code/Ariadne-a1p     # working tree the command runs in
#   VCPU=16                        # vCPU it needs from the CPU spot quota
#   CMD='infra/eval/foo_launch.sh run'   # ONE launch attempt, foreground until the run ends
# Write it as <name>.tmp and rename it to .job, so the dispatcher never reads half a file.
#
# The command must be a single attempt: if AWS refuses it for spot quota or
# capacity, the dispatcher puts it back at the head of the queue and retries
# when room appears. Any other failure moves it to failed/. Events (launch,
# end with rc, failure) go to $Q/events.log, one line each, UTC plus IST.
# Only CPU spot is counted; GPU (G/VT) instances use their own quota.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS="$REPO_ROOT/infra/scripts/aws.sh"
Q="${SLOTQ:-/data/code/Ariadne/build/slotq}"
QUOTA="${SLOTQ_QUOTA_VCPU:-32}"
POLL_S="${SLOTQ_POLL_S:-30}"
mkdir -p "$Q"/{queue,running,done,failed}

ev() { printf '%s (%s IST) %s\n' "$(date -u +%FT%TZ)" "$(TZ=Asia/Kolkata date +%H:%M)" "$*" | tee -a "$Q/events.log"; }

# vCPU held by live non-GPU ariadne instances; empty on an API error (never read as 0).
used_vcpu() {
  local out
  out=$("$AWS" aws ec2 describe-instances --region eu-north-1 \
        --filters Name=tag:Project,Values=ariadne Name=instance-state-name,Values=pending,running,shutting-down,stopping \
        --query 'Reservations[].Instances[].[InstanceType,CpuOptions.CoreCount,CpuOptions.ThreadsPerCore]' \
        --output text 2>/dev/null) || return 1
  printf '%s\n' "${out//$'\r'/}" | awk '$1 !~ /^(g|p|inf|trn|vt)/ && NF==3 {s += $2*$3} END {print s+0}'
}

reap() {   # finished runs -> done/ (or failed/ if the launch itself never happened)
  local f pid rc
  for f in "$Q"/running/*.job; do
    [ -e "$f" ] || continue
    pid=$(cat "$f.pid"); kill -0 "$pid" 2>/dev/null && continue
    rc=$(cat "$f.rc" 2>/dev/null || echo "?")
    ev "END $(basename "$f" .job) owner=$(. "$f"; echo "$OWNER") rc=$rc"
    mv "$f" "$f.pid" "$f.rc" "${f%.job}.log" "$Q/done/" 2>/dev/null
  done
}

launch() {   # $1 = queued job file; returns 0 launched, 2 refused for quota/capacity, 1 other failure
  local f=$1 name; name=$(basename "$f" .job)
  ( . "$f"; : "${OWNER:?}" "${DIR:?}" "${CMD:?}" ) || { ev "FAILED $name: bad job file"; mv "$f" "$Q/failed/"; return 1; }
  local OWNER DIR CMD LOG
  eval "$(. "$f"; declare -p OWNER DIR CMD 2>/dev/null)"
  LOG="$Q/running/$name.log"
  mv "$f" "$Q/running/$name.job"; f="$Q/running/$name.job"
  ( cd "$DIR" && bash -c "$CMD"; echo $? > "$f.rc" ) > "$LOG" 2>&1 < /dev/null &
  local pid=$!; echo "$pid" > "$f.pid"
  local t=0
  while kill -0 "$pid" 2>/dev/null && ! grep -q 'launched i-' "$LOG"; do sleep 3; t=$((t+3)); [ $t -ge 600 ] && break; done
  if grep -q 'launched i-' "$LOG"; then
    ev "START $name owner=$OWNER $(grep -o 'run [0-9TZ]*-[a-z0-9]*-[0-9a-f]*' "$LOG" | head -1) $(grep -o 'launched i-[0-9a-f]*' "$LOG" | head -1) log=$LOG"
    return 0
  fi
  wait "$pid" 2>/dev/null; local rc=$?
  if grep -qE 'MaxSpotInstanceCountExceeded|InsufficientInstanceCapacity|VcpuLimitExceeded' "$LOG"; then
    mv "$f" "$Q/queue/$name.job"; rm -f "$f.pid" "$f.rc"; cp "$LOG" "$Q/queue/$name.lastrefusal.log"; return 2
  fi
  ev "FAILED $name owner=$OWNER rc=$rc (launch never happened; log $Q/failed/$name.log)"
  mv "$f" "$LOG" "$Q/failed/"; rm -f "$f.pid" "$f.rc"; return 1
}

dispatch() {
  exec 9> "$Q/.dispatcher.lock"; flock -n 9 || { echo "another dispatcher is running" >&2; exit 1; }
  ev "DISPATCHER up pid=$$ quota=${QUOTA} poll=${POLL_S}s"
  local refused_until=0
  while true; do
    reap
    local head; head=$(ls "$Q"/queue/*.job 2>/dev/null | sort | head -1)
    if [ -n "$head" ] && [ "$(date +%s)" -ge "$refused_until" ]; then
      local used need; need=$(. "$head"; echo "${VCPU:-16}")
      if used=$(used_vcpu) && [ -n "$used" ] && [ $((QUOTA - used)) -ge "$need" ]; then
        launch "$head"; [ $? -eq 2 ] && refused_until=$(( $(date +%s) + POLL_S ))
        continue
      fi
    fi
    sleep "$POLL_S"
  done
}

status() {
  echo "free vCPU: $(( QUOTA - $(used_vcpu || echo "$QUOTA") )) of $QUOTA"
  for d in running queue; do echo "$d:"; ls "$Q/$d" 2>/dev/null | grep '\.job$' | sed 's/^/  /'; done
  echo "recent events:"; tail -8 "$Q/events.log" 2>/dev/null | sed 's/^/  /'
}

case "${1:-}" in
  dispatch) dispatch ;;
  status) status ;;
  *) sed -n '2,24p' "$0"; exit 2 ;;
esac
