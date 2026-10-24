#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Kaggle CPU slot queue: keeps the account's 5 batch CPU sessions busy with ci_kaggle_launch.sh jobs, as
# infra/scripts/slotq.sh keeps the AWS spot quota busy. Sessions drop job files; one dispatcher stages and submits
# the next job, in name order, whenever Kaggle has a free session (a push refused for "session count" waits), and
# pulls each finished kernel's results to build/runs-offaws/<id>/.
#
#   infra/eval/ci_kaggle_queue.sh dispatch     the one dispatcher (test-compute-session runs it)
#   infra/eval/ci_kaggle_queue.sh status       queue, running, done
#
# A job is $Q/queue/<NN>-<name>.job (write <name>.tmp, rename to .job), sourced by bash:
#   OWNER=A2                        session to tell when it ends
#   NAME=a2-knobs-s0                ci_kaggle_launch.sh NAME ([a-z0-9-], <= 36 chars)
#   COMMIT=13cad72                  the commit to build (any ref in the shared object DB)
#   INPUTS_ROOT=/data/code/Ariadne-2.2-radiometric   optional: the tree INPUTS are relative to
#   INPUTS=".cache/models/x build/y" optional: unset = the default model+oracle set, "" = none
#   TESTS=none                      ctest -R regex; empty = all; none = no ctest
#   EXTRA_FILE=/abs/path/extra.sh   optional: its contents are EXTRA (no quoting games)
#   MAX_MINUTES=300                 optional: the kernel's own timeout
# Events (submit, refusal, end with rc and result dir) go to $Q/events.log, UTC plus IST.
# Only the host-side dispatcher runs here (CLI calls in a container, then sleep): no heavy local work.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/../.."
L=infra/eval/ci_kaggle_launch.sh
Q="${KQ:-$PWD/build/ci_kaggle_q}"
POLL_S="${KQ_POLL_S:-60}"
mkdir -p "$Q"/{queue,running,done,failed}
ev() { printf '%s (%s IST) %s\n' "$(date -u +%FT%TZ)" "$(TZ=Asia/Kolkata date +%H:%M)" "$*" | tee -a "$Q/events.log"; }

jobenv() {   # $1 = job file; runs the rest of the args with the job's variables exported
  local f=$1; shift
  ( set -a; . "$f"; set +a
    [ -n "${EXTRA_FILE:-}" ] && export EXTRA="$(cat "$EXTRA_FILE")"
    "$@" )
}

reap() {
  local f name st rc
  for f in "$Q"/running/*.job; do
    [ -e "$f" ] || continue
    name=$(. "$f"; echo "$NAME")
    st="$("$L" state "$name" 2>&1 | tail -1)"
    case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) ;; *) continue ;; esac
    jobenv "$f" "$L" wait "$name" > "${f%.job}.log" 2>&1; rc=$?
    ev "END $(basename "$f" .job) owner=$(. "$f"; echo "$OWNER") rc=$rc dir=build/runs-offaws/$(cat "build/ci_kaggle/$name/run_id" 2>/dev/null)"
    mv "$f" "${f%.job}.log" "$Q/done/" 2>/dev/null
  done
}

fill() {
  local f name rc
  for f in "$Q"/queue/*.job; do
    [ -e "$f" ] || return 0
    name=$(. "$f"; echo "$NAME")
    if [ ! -e "$f.staged" ]; then
      jobenv "$f" bash -c '[ "${INPUTS+set}" = set ] || unset INPUTS; exec "$0" stage "$NAME" "$COMMIT"' "$L" \
        > "$f.stage.log" 2>&1 || { ev "FAILED $(basename "$f" .job): stage ($(tail -1 "$f.stage.log"))"; mv "$f" "$f.stage.log" "$Q/failed/"; continue; }
      touch "$f.staged"
    fi
    jobenv "$f" "$L" submit "$name" > "$f.submit.log" 2>&1; rc=$?
    case $rc in
      0)  ev "SUBMIT $(basename "$f" .job) owner=$(. "$f"; echo "$OWNER") $(tail -1 "$f.submit.log" | sed 's/.*pushed //')"
          rm -f "$f.staged" "$f.stage.log" "$f.submit.log"; mv "$f" "$Q/running/" ;;
      75) return 0 ;;                       # all 5 CPU sessions busy: try again next poll, same order
      *)  ev "FAILED $(basename "$f" .job): submit ($(tail -1 "$f.submit.log"))"; mv "$f" "$f.submit.log" "$Q/failed/" ;;
    esac
  done
}

case "${1:-}" in
  dispatch) ev "dispatcher up (pid $$, poll ${POLL_S}s)"; while true; do reap; fill; sleep "$POLL_S"; done ;;
  status) for d in queue running done failed; do echo "[$d]"; ls "$Q/$d" | grep -E '\.job$' || true; done
          echo "[events]"; tail -8 "$Q/events.log" 2>/dev/null || true ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
