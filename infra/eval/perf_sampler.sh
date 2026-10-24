#!/bin/bash
# ---------------------------------------------------------------------------
# Host-side CPU sampler for the pipeline profile (A2, 2026-10-24; measurement only). B's live.sh starts it beside a
# run when ARIADNE_SAMPLER=<this script> (the D179+ profiling flag, B's code) and stops it with SIGTERM when the run's
# containers stop; it never touches the run itself. Runs as root on the instance:
#
#   infra/eval/perf_sampler.sh OUTDIR           (foreground until SIGTERM/SIGINT)
#
# OUTDIR gets (1 s cadence unless noted; every file is plain text with UTC stamps):
#   mpstat.txt         mpstat -P ALL 1: per-core user/sys/iowait/idle -> core utilisation, saturation
#   pidstat.txt        pidstat -t -u -w -h 1: per-THREAD CPU % and voluntary/involuntary context switches; thread
#                      names (OpenCV's pool threads, the fleet thread, gz, llvmpipe) show who competes for cores
#   psi_cpu.txt        /proc/pressure/cpu once a second: the share of wall time runnable tasks waited for a core
#                      (oversubscription measured directly, not inferred)
#   docker_stats.txt   docker stats --no-stream every 5 s: CPU % and memory per container (= per process role)
#   cgroups.txt        per container, cpu.stat (usage_usec, nr_throttled) at start and at stop: exact CPU-seconds per
#                      container over the run
#   perf.data + perf_report.txt  best effort: perf record -a -g -F 99 (needs linux-tools for this kernel; skipped,
#                      and said so in sampler.log, if apt cannot supply it)
#   sampler.log        what started, what was skipped, and why
# Needs sysstat (apt, installed here if missing). Overhead: pidstat/mpstat at 1 Hz are ~1 % of one core; perf at 99 Hz
# is ~1-2 % - recorded in sampler.log so a reader can discount it.
# ---------------------------------------------------------------------------
set -u
OUT="${1:?usage: perf_sampler.sh OUTDIR}"
mkdir -p "$OUT"; rm -f "$OUT/.stop"
log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/sampler.log"; }
PIDS=()
log "start on $(hostname), $(nproc) vCPU, kernel $(uname -r)"
command -v pidstat >/dev/null || { apt-get -o DPkg::Lock::Timeout=120 install -y -qq sysstat >/dev/null 2>&1 \
  && log "installed sysstat" || log "sysstat unavailable: no mpstat/pidstat"; }
lscpu > "$OUT/lscpu.txt" 2>&1

cgroup_snapshot() {   # $1 = label: cpu.stat of every running container's cgroup (v2 path, then v1)
  for id in $(docker ps -q --no-trunc 2>/dev/null); do
    name=$(docker inspect --format '{{.Name}}' "$id" 2>/dev/null)
    for f in /sys/fs/cgroup/system.slice/docker-$id.scope/cpu.stat /sys/fs/cgroup/docker/$id/cpu.stat; do
      [ -r "$f" ] && { echo "$1 $(date -u +%FT%TZ) $name $id"; cat "$f"; break; }
    done
  done >> "$OUT/cgroups.txt"
}

stop() {
  log "stop"
  cgroup_snapshot stop
  touch "$OUT/.stop"                     # the shell loops poll it (a SIGINT to a looping bash is deferred, not fatal)
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null; done
  for _ in $(seq 15); do                 # explicit pids, bounded: never a bare wait (job 44's first sampler hung on it)
    alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
    [ "$alive" = 0 ] && break; sleep 1
  done
  for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && { kill -9 "$p" 2>/dev/null; log "killed $p after 15 s"; }; done
  if [ -s "$OUT/perf.data" ]; then
    timeout 300 perf report -i "$OUT/perf.data" --stdio --sort comm,dso --no-children 2>/dev/null \
      | head -200 > "$OUT/perf_report.txt"
  fi
  log "done"
  exit 0
}
trap stop TERM INT

cgroup_snapshot start
if command -v mpstat >/dev/null; then
  S_TIME_FORMAT=ISO mpstat -P ALL 1 > "$OUT/mpstat.txt" 2>&1 & PIDS+=($!)
  S_TIME_FORMAT=ISO pidstat -t -u -w -h 1 > "$OUT/pidstat.txt" 2>&1 & PIDS+=($!)
fi
( while [ ! -e "$OUT/.stop" ]; do echo "$(date -u +%FT%TZ) $(tr '\n' ' ' < /proc/pressure/cpu)"; sleep 1; done ) > "$OUT/psi_cpu.txt" 2>&1 &
PIDS+=($!)
( while [ ! -e "$OUT/.stop" ]; do echo "== $(date -u +%FT%TZ)"; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'
  sleep 5; done ) > "$OUT/docker_stats.txt" 2>&1 & PIDS+=($!)
if ! command -v perf >/dev/null; then
  apt-get -o DPkg::Lock::Timeout=120 install -y -qq "linux-tools-$(uname -r)" >/dev/null 2>&1 \
    && log "installed linux-tools-$(uname -r)" || log "perf unavailable for $(uname -r): skipped"
fi
if command -v perf >/dev/null && perf record -a -g -F 99 -o "$OUT/perf.data" -- sleep 0.1 >/dev/null 2>&1; then
  perf record -a -g -F 99 -o "$OUT/perf.data" > "$OUT/perf_record.log" 2>&1 & PIDS+=($!)
  log "perf record -a -g -F 99 running"
else
  log "perf record not started"
fi
while :; do sleep 1; done
