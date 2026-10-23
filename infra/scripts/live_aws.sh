#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Live runs on AWS GPU (design LIVE_RUNTIME_L1 D155): one g4dn.xlarge SPOT (T4, 4 vCPU: the whole D84 quota) through
# infra/scripts/spinup.sh, with its three brakes (shutdown timer on the instance, trap here, sweep).
#
#   infra/scripts/live_aws.sh push                 local images -> ECR, tag live-<image id 12>; prints the refs
#   infra/scripts/live_aws.sh payload SCENE        build outputs + the live scene -> S3 (SHA256SUMS); prints the URI
#   infra/scripts/live_aws.sh smoke                [SMOKE-TEST] NVIDIA EGL in ariadne/sim on g4dn (~15 min)
#   infra/scripts/live_aws.sh run SCENE END_S      payload + images -> live.sh run + replay x2 on the instance
#   infra/scripts/live_aws.sh batch JOBS MAX_MIN   lockstep jobs, one after another on one instance (D165; 4.3 gate):
#       JOBS: one job per line "PRESET SEED END_S PLANNER_CONFIG|-" (- = the 1.8 placeholder row); every job runs
#       with ARIADNE_LOCKSTEP=1 and ARIADNE_SHADOW_MAP_CONFIG=configs/mission/mission_v2.yaml. Scenes must be composed
#       locally first (live.sh compose PRESET SEED 0.15). FIRST, an EGL reproducibility check: ci_tiny seed 0, 20 s,
#       twice, lockstep_repro_check.py; the batch STOPS if it fails. Each job's slim result (everything but
#       inputs.mcap, ~10 GB per 200 s run, and sockets) goes to S3 as soon as it ends, so a spot interruption loses
#       one job only. MAX_MIN <= 240 (spinup's cap): split long gates into several batches.
#
# The build outputs are made locally (the 2-job dev container) and shipped, not rebuilt on 4 vCPUs:
#   build/docker/cpp/runtime/ariadne_live, build/sim-plugins/*.so, build/ros2/install/, SCENE/.
# The instance gets the working tree from spinup.sh (tracked + untracked-not-ignored) and the payload on top, so
# every relative path in live.sh is the same as on the dev host. Results come back to build/runs/<run-id>/output/.
# ---------------------------------------------------------------------------
set -Eeuo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AWS="$REPO_ROOT/infra/scripts/aws.sh"
REGION=eu-north-1
log() { printf '[live_aws %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

account() { "$AWS" aws sts get-caller-identity --query Account --output text | tr -d '\r'; }
bucket() { "$AWS" tf main output -json runner | python3 -c 'import json,sys; print(json.load(sys.stdin)["artifact_bucket"])'; }

# Profile/cap knobs (D219): LIVE_AWS_PROFILE=cpux86 runs the same flow on the c7i CPU profile (llvmpipe render,
# proven deterministic across machines at D168 M1); LIVE_AWS_MAX_MIN raises cmd_run's instance cap for long runs.
PROFILE="${LIVE_AWS_PROFILE:-gpu}"
RUN_MAX_MIN="${LIVE_AWS_MAX_MIN:-60}"

# local image -> ECR repo
declare -A REPO=([ariadne/sim:dev]=ariadne/ariadne-sim [ariadne/core-dev:dev]=ariadne/ariadne-core [ariadne-console:1]=ariadne/ariadne-console [ariadne-orchestrator:1]=ariadne/ariadne-orchestrator)

ref_of() {   # ECR reference of a local image, tagged by its own image id (content-addressed, not "latest")
  local img="$1" reg="$2" id
  id="$(docker image inspect "$img" --format '{{.Id}}' | cut -d: -f2 | cut -c1-12)"
  echo "$reg/${REPO[$img]}:live-$id"
}

cmd_push() {
  local reg; reg="$(account).dkr.ecr.$REGION.amazonaws.com"
  "$AWS" aws ecr get-login-password --region "$REGION" | tr -d '\r' | docker login --username AWS --password-stdin "$reg" >/dev/null
  local img
  for img in "${!REPO[@]}"; do
    local ref; ref="$(ref_of "$img" "$reg")"
    docker tag "$img" "$ref"
    log "push $img -> $ref"
    docker push -q "$ref" >/dev/null
    echo "$img $ref $(docker image inspect "$img" --format '{{.Id}}')"
  done
}

cmd_payload() {   # SCENE... (one or more scene directories)
  local id stage
  id="$(date -u +%Y%m%dT%H%M%SZ)"
  stage="$REPO_ROOT/build/live_payload/$id"
  mkdir -p "$stage/p"
  (cd "$REPO_ROOT" && tar cf - build/docker/cpp/runtime/ariadne_live build/sim-plugins/libariadne_kinematic_mover.so \
     build/ros2/install "$@") | tar xf - -C "$stage/p"
  (cd "$stage/p" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  tar czf "$stage/payload.tar.gz" -C "$stage" p
  local uri="s3://$(bucket)/runs/live-payload-$id/payload.tar.gz"
  "$AWS" aws s3 cp "build/live_payload/$id/payload.tar.gz" "$uri" --only-show-errors
  log "payload $(du -h "$stage/payload.tar.gz" | cut -f1) -> $uri"
  echo "$uri"
}

# The instance-side prologue: docker + NVIDIA runtime present, images pulled and tagged with their LOCAL names, so
# live.sh runs unchanged.
prologue() {   # $1 = "sim core console" refs file contents (img ref id per line)
  cat <<EOF
set -euo pipefail
R=\$ARIADNE_RESULTS
REG=$(account).dkr.ecr.$REGION.amazonaws.com
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io; }
nvidia-smi > \$R/nvidia-smi.txt 2>&1 || true
docker info --format '{{json .Runtimes}}' > \$R/docker_runtimes.json
lscpu > \$R/lscpu.txt; free -m > \$R/free.txt
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin \$REG >/dev/null
while read -r img ref id; do
  docker pull -q "\$ref" >/dev/null && docker tag "\$ref" "\$img"
  got=\$(docker image inspect "\$img" --format '{{.Id}}'); [ "\$got" = "\$id" ] || { echo "digest mismatch \$img"; exit 3; }
  echo "\$img \$ref \$got" >> \$R/images.txt
done <<'REFS'
$1
REFS
EOF
}

cmd_smoke() {
  local refs; refs="$(cmd_push)"
  local script="$REPO_ROOT/build/live_payload/smoke-$(date -u +%Y%m%dT%H%M%SZ).sh"
  mkdir -p "$(dirname "$script")"
  { prologue "$refs"; cat <<'EOF'
# [SMOKE-TEST] the sim image's headless EGL under the NVIDIA runtime (never run before: llvmpipe only, SMOKE §4).
ARIADNE_SMOKE_OUT_HOST=$R/sim-smoke ARIADNE_SMOKE_TIMEOUT=120 infra/scripts/sim.sh > $R/sim_smoke.log 2>&1; echo "sim_smoke_exit=$?" >> $R/sim_smoke.log
docker run --rm --gpus all ariadne/sim:dev bash -c 'ls -la /usr/share/glvnd/egl_vendor.d/; ldconfig -p | grep -iE "libEGL_nvidia|libnvidia-egl" | head' > $R/egl_vendor.txt 2>&1 || true
EOF
  } > "$script"
  "$REPO_ROOT/infra/scripts/spinup.sh" run --profile "$PROFILE" --max-minutes 25 -- "$(cat "$script")"
}

cmd_run() {
  local scene="$1" end_s="$2"
  local refs; refs="$(cmd_push)"
  local uri; uri="$(cmd_payload "$scene")"
  local script="$REPO_ROOT/build/live_payload/run-$(date -u +%Y%m%dT%H%M%SZ).sh"
  { prologue "$refs"; cat <<EOF
aws s3 cp $uri - | tar xz -C /tmp
(cd /tmp/p && sha256sum -c --quiet SHA256SUMS) && echo "payload sha256 OK" > \$R/payload_check.txt
cp -a /tmp/p/. . && cp /tmp/p/SHA256SUMS \$R/
mkdir -p build/live/runs
ARIADNE_LOCKSTEP=${ARIADNE_LOCKSTEP:-} ARIADNE_LIVE_CONFIG=${ARIADNE_LIVE_CONFIG:-} ARIADNE_PLANNER_CONFIG=${ARIADNE_PLANNER_CONFIG:-} ARIADNE_RT_MEM=${ARIADNE_RT_MEM:-6g} \
  infra/scripts/live.sh run $scene build/live/runs $end_s > \$R/live_run.log 2>&1 || echo "live_run_failed" >> \$R/live_run.log
run=\$(ls -d build/live/runs/live-* | tail -1)
infra/scripts/live.sh replay \$run $scene > \$R/replay.log 2>&1 || echo "replay_check_failed" >> \$R/replay.log
tar czf \$R/run.tar.gz --exclude='*.sock' \$run
EOF
  } > "$script"
  "$REPO_ROOT/infra/scripts/spinup.sh" run --profile "$PROFILE" --max-minutes "$RUN_MAX_MIN" -- "$(cat "$script")"
}

cmd_batch() {
  local jobs="$1" max_min="$2"
  [ -f "$REPO_ROOT/$jobs" ] || { log "no job file $jobs"; exit 2; }
  local scenes=(build/live/scenes/ci_tiny_0) preset seed end_s cfg
  while read -r preset seed end_s cfg; do
    { [ -z "$preset" ] || [ "${preset:0:1}" = "#" ]; } && continue
    [ -f "$REPO_ROOT/build/live/scenes/${preset}_${seed}/mission_live.sdf" ] || { log "compose ${preset} ${seed} first"; exit 2; }
    [ "$cfg" = "-" ] || [ -f "$REPO_ROOT/$cfg" ] || { log "no planner config $cfg"; exit 2; }
    [ $(( ${end_s%.*} % 5 )) = 0 ] || { log "END_S $end_s: lockstep needs a multiple of the 5 s replan"; exit 2; }
    scenes+=("build/live/scenes/${preset}_${seed}")
  done < "$REPO_ROOT/$jobs"
  local refs; refs="$(cmd_push)"
  local uri; uri="$(cmd_payload $(printf '%s\n' "${scenes[@]}" | sort -u))"
  local script="$REPO_ROOT/build/live_payload/batch-$(date -u +%Y%m%dT%H%M%SZ).sh"
  { prologue "$refs"; cat <<BATCH
aws s3 cp $uri - | tar xz -C /tmp
(cd /tmp/p && sha256sum -c --quiet SHA256SUMS) && echo "payload sha256 OK" > \$R/payload_check.txt
cp -a /tmp/p/. . && cp /tmp/p/SHA256SUMS \$R/
cp $jobs \$R/jobs.txt
BATCH
  cat <<'BATCH'
export ARIADNE_LOCKSTEP=1 ARIADNE_SHADOW_MAP_CONFIG=configs/mission/mission_v2.yaml
slim() {   # a run directory without its recording -> $R/$2.tar.gz, uploaded at once
  tar czf $R/$2.tar.gz --exclude='*.sock' --exclude='inputs.mcap' "$1"
  if [ -n "${ARIADNE_RESULTS_URI:-}" ]; then aws s3 cp $R/$2.tar.gz $ARIADNE_RESULTS_URI/$2.tar.gz --only-show-errors || true; fi
}
# 1. EGL reproducibility (lockstep is verified on llvmpipe only): the same seed twice, compared; stop if it differs.
for i in 1 2; do infra/scripts/live.sh run build/live/scenes/ci_tiny_0 build/live/repro 20 > $R/repro_run$i.log 2>&1 || true; done
if ! python3 python/analysis/lockstep_repro_check.py $(ls -d build/live/repro/live-*/ | head -1) \
     $(ls -d build/live/repro/live-*/ | tail -1) $R/repro_check.json > $R/repro_check.txt 2>&1; then
  echo "EGL repro check FAILED: batch stopped" | tee -a $R/batch.log
  slim build/live/repro repro
  exit 4
fi
echo "EGL repro check PASS" | tee -a $R/batch.log
slim build/live/repro repro
# 2. The jobs, in file order.
n=0
while read -r preset seed end_s cfg; do
  { [ -z "$preset" ] || [ "${preset:0:1}" = "#" ]; } && continue
  n=$((n + 1)); name=$(printf 'job%02d_%s_%s_%s' $n "$preset" "$seed" "$(basename "${cfg%.yaml}")")
  t0=$(date +%s)
  if [ "$cfg" = "-" ]; then unset ARIADNE_PLANNER_CONFIG; else export ARIADNE_PLANNER_CONFIG="$cfg"; fi
  rc=0
  infra/scripts/live.sh run build/live/scenes/${preset}_${seed} build/live/batch/$name $end_s > $R/$name.log 2>&1 || rc=$?
  echo "$name exit $rc wall_s $(( $(date +%s) - t0 ))" | tee -a $R/batch.log
  slim build/live/batch/$name $name
  if [ -n "${ARIADNE_RESULTS_URI:-}" ]; then aws s3 cp $R/batch.log $ARIADNE_RESULTS_URI/batch.log --only-show-errors || true; fi
done < $R/jobs.txt
BATCH
  } > "$script"
  "$REPO_ROOT/infra/scripts/spinup.sh" run --profile "$PROFILE" --max-minutes "$max_min" -- "$(cat "$script")"
}

case "${1:-}" in
  push) shift; cmd_push "$@" ;;
  payload) shift; cmd_payload "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  run) shift; cmd_run "$@" ;;
  batch) shift; cmd_batch "$@" ;;
  *) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
