#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Live runs on AWS GPU (design LIVE_RUNTIME_L1 D155): one g4dn.xlarge SPOT (T4, 4 vCPU: the whole D84 quota) through
# infra/scripts/spinup.sh, with its three brakes (shutdown timer on the instance, trap here, sweep).
#
#   infra/scripts/live_aws.sh push                 local images -> ECR, tag live-<image id 12>; prints the refs
#   infra/scripts/live_aws.sh payload SCENE        build outputs + the live scene -> S3 (SHA256SUMS); prints the URI
#   infra/scripts/live_aws.sh smoke                [SMOKE-TEST] NVIDIA EGL in ariadne/sim on g4dn (~15 min)
#   infra/scripts/live_aws.sh run SCENE END_S      payload + images -> live.sh run + replay x2 on the instance
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

# local image -> ECR repo
declare -A REPO=([ariadne/sim:dev]=ariadne/ariadne-sim [ariadne/core-dev:dev]=ariadne/ariadne-core [ariadne-console:1]=ariadne/ariadne-console)

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

cmd_payload() {
  local scene="$1" id stage
  id="$(date -u +%Y%m%dT%H%M%SZ)"
  stage="$REPO_ROOT/build/live_payload/$id"
  mkdir -p "$stage/p"
  (cd "$REPO_ROOT" && tar cf - build/docker/cpp/runtime/ariadne_live build/sim-plugins/libariadne_kinematic_mover.so \
     build/ros2/install "$scene") | tar xf - -C "$stage/p"
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
  "$REPO_ROOT/infra/scripts/spinup.sh" run --profile gpu --max-minutes 25 -- "$(cat "$script")"
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
infra/scripts/live.sh run $scene build/live/runs $end_s > \$R/live_run.log 2>&1 || echo "live_run_failed" >> \$R/live_run.log
run=\$(ls -d build/live/runs/live-* | tail -1)
infra/scripts/live.sh replay \$run $scene > \$R/replay.log 2>&1 || echo "replay_check_failed" >> \$R/replay.log
tar czf \$R/run.tar.gz --exclude='*.sock' \$run
EOF
  } > "$script"
  "$REPO_ROOT/infra/scripts/spinup.sh" run --profile gpu --max-minutes 60 -- "$(cat "$script")"
}

case "${1:-}" in
  push) shift; cmd_push "$@" ;;
  payload) shift; cmd_payload "$@" ;;
  smoke) shift; cmd_smoke "$@" ;;
  run) shift; cmd_run "$@" ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
