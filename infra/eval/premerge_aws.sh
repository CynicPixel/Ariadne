#!/bin/bash
# The D49 pre-merge gate (infra/scripts/phase1.sh --premerge) on one fresh AWS instance. Runs as root ON THE INSTANCE:
#   infra/scripts/spinup.sh run --profile cpux86 --max-minutes 120 -- \
#     'STAGE=s3://<bucket>/staging/<id> DATA=s3://<bucket>/datasets/<rec> COMMIT=<sha> REG=<ecr registry> bash infra/eval/premerge_aws.sh'
# Inputs, all staged under $STAGE by the launching host (nothing here is built or rendered remotely):
#   repo.bundle      git bundle of COMMIT (phase1.sh needs a real checkout: commit id, dirty check)
#   inputs.tar       the ignored inputs the gate reads, with SHA256SUMS: .cache/models/{xfeat_ref,eigenplaces_ref,
#                    aliked_lg_ref}, .cache/oracle/pgo, build/render_cache
#   $DATA            the Phase 1 sim recording (evaluation/runs/gates/phase1_thin_e2e/sim_record/<id>/raw): a persistent S3
#                    dataset, not staged per run
#   images.txt       "<ecr ref> <local name> <image id>" per line: pulled, retagged to the names the scripts use,
#                    and their ids checked. core.sh is told not to rebuild the dev image (ARIADNE_SKIP_DEV_BUILD).
# Each step is logged to $ARIADNE_RESULTS/steps.txt as it happens. The gate directory, the ctest log and every stage
# log come back. JOBS defaults to 8 (16 vCPU host; the 2-job cap is the maintainer's workstation's).
# MODE=full runs the FULL gate (phase1.sh without --premerge) into evaluation/runs/gates/phase2_exit/exit_gate: its fresh Kaggle
# render is relayed to the launching host through $STAGE/relay (infra/eval/render_relay.sh), whose token stays there.
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${STAGE:?}" "${DATA:?}" "${COMMIT:?}" "${REG:?}"
JOBS="${JOBS:-8}"
MODE="${MODE:-premerge}"
case "$MODE" in premerge) args=--premerge out=build/premerge ;; full) args=evaluation/runs/gates/phase2_exit/exit_gate out=evaluation/runs/gates/phase2_exit/exit_gate ;;
  *) echo "MODE: premerge | full" >&2; exit 1 ;; esac
REPO=/opt/ariadne/repo
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }

step "0 host"
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"
printf 'commit %s\nstage %s\njobs %s\nmode %s\n' "$COMMIT" "$STAGE" "$JOBS" "$MODE" > "$R/run_params.txt"

step "1 docker + images"
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io > "$R/apt.txt" 2>&1; }
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || fail "ecr login"
aws s3 cp "$STAGE/images.txt" "$R/images.txt" --only-show-errors || fail "images.txt"
while read -r ref name want; do
  docker pull -q "$ref" > /dev/null || fail "pull $ref"
  docker tag "$ref" "$name"
  got="$(docker image inspect "$name" --format '{{.Id}}')"
  echo "$name $ref $got (local $want)" >> "$R/image_ids.txt"
done < "$R/images.txt"

step "2 checkout"
aws s3 cp "$STAGE/repo.bundle" /opt/ariadne/repo.bundle --only-show-errors || fail "bundle"
git clone -q /opt/ariadne/repo.bundle "$REPO" && git -C "$REPO" checkout -q "$COMMIT" || fail "checkout $COMMIT"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse "$COMMIT")" ] || fail "HEAD is not $COMMIT"
git -C "$REPO" log -1 --format='%H %s' > "$R/head.txt"

step "3 inputs"
aws s3 cp "$STAGE/inputs.tar" - | tar x -C "$REPO" || fail "inputs.tar"
(cd "$REPO" && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
rec=$(cat "$REPO/inputs.recording_path")
mkdir -p "$REPO/$rec"
# File by file from the checksum list: GetObject only (the runner role has no ListBucket, so no `s3 sync`).
(cd "$REPO/$rec" && sed -E 's/^[0-9a-f]{64}  \.\///' "$REPO/recording.SHA256SUMS" \
  | xargs -P 4 -I{} sh -c 'mkdir -p "$(dirname "{}")" && aws s3 cp "'"$DATA"'/{}" "{}" --only-show-errors') \
  || fail "recording fetch"
(cd "$REPO/$rec" && sha256sum -c --quiet "$REPO/recording.SHA256SUMS") > "$R/recording_check.txt" 2>&1 \
  || fail "recording sha256"
rm -f "$REPO/inputs.SHA256SUMS" "$REPO/recording.SHA256SUMS" "$REPO/inputs.recording_path"
df -h / > "$R/df.txt"
# The dev image runs as its own non-root user; root-run containers (recon, analysis, console) write anyway.
uid="$(docker run --rm ariadne/core-dev:dev id -u)"; gid="$(docker run --rm ariadne/core-dev:dev id -g)"
chown -R "$uid:$gid" "$REPO"
# user-data runs as root without $HOME, so --global never applies: system-wide, and give git a HOME anyway.
export HOME="${HOME:-/root}"
git config --system --add safe.directory "$REPO"
git -C "$REPO" status --porcelain > "$R/git_status.txt"

step "4 $MODE gate (JOBS=$JOBS)"
t0=$(date +%s); touch /opt/ariadne/gate_start
# Run the gate as the image's uid, like on the workstation: phase1.sh creates the gate directories on the host and
# the dev container (uid 1000) writes into them, so as root they would be root-owned and unwritable (run 222331Z).
U="$(getent passwd "$uid" | cut -d: -f1)"
[ -n "$U" ] || { useradd -m -u "$uid" gate; U=gate; }
usermod -aG docker "$U"
runuser -u "$U" -- env HOME="$(getent passwd "$U" | cut -d: -f6)" ARIADNE_SKIP_DEV_BUILD=1 ARIADNE_BUILD_JOBS="$JOBS" \
  RELAY="$STAGE/relay" ARIADNE_RENDER_CMD="$REPO/infra/eval/render_relay.sh" \
  bash -c "cd '$REPO' && git config --global --add safe.directory '$REPO' && infra/scripts/phase1.sh $args" \
  > "$R/phase1.log" 2>&1
rc=$?
echo "gate_exit $rc seconds $(( $(date +%s) - t0 ))" >> "$R/steps.txt"

step "5 collect"
g=$(ls -d "$REPO/$out"/*/ 2>/dev/null | tail -1)
[ -n "$g" ] && cp -r "$g" "$R/gate"
b=$(ls -d "$REPO"/build/phase1_gate/*/ 2>/dev/null | tail -1)
[ -n "$b" ] && { mkdir -p "$R/big_logs"; cp "$b"/*.log "$R/big_logs/" 2>/dev/null; }
# A full run's fresh render enters the render cache; it comes back so later pre-merges can reuse it.
# Files, not directories: a re-render of a cached package overwrites files and leaves the directory's mtime alone.
[ "$MODE" = full ] && (cd "$REPO/build" && find render_cache -type f -newer /opt/ariadne/gate_start -print0 \
  | tar cf - --null -T -) | tar xf - -C "$R"
[ -f "$R/gate/gate.txt" ] && tail -40 "$R/gate/gate.txt"
step "done ($MODE exit $rc)"
exit "$rc"
