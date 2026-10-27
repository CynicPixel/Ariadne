#!/bin/bash
# Host side of the 4.2 risk-calibration renders (design traversability-risk.md#risk-zones §3, D112): stage the inputs, run
# infra/eval/risk_calib_aws.sh on one g4dn.xlarge through spinup.sh, then tear down everything this run staged.
# Pattern of main's infra/eval/premerge_aws_launch.sh. Runs from the repo root on a clean HEAD.
#   infra/eval/risk_calib_aws_launch.sh stage     inputs.tar (scenes + payload, SHA256SUMS), images to ECR (temp tags)
#   infra/eval/risk_calib_aws_launch.sh run       spinup.sh (gpu, 240 min cap) -> build/runs/<id>/
#   infra/eval/risk_calib_aws_launch.sh teardown  delete this run's temp ECR tags and staged inputs; sweep strays
# Nothing persistent is left in AWS: the recordings are deleted on the instance after their spool is built.
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
ACCT=836505524290 REGION=eu-north-1
REG=$ACCT.dkr.ecr.$REGION.amazonaws.com REPO_ECR=ariadne/ariadne-core
BUCKET=s3://ariadne-$ACCT-eun1
STAGE=$BUCKET/runs/risk_calib-staging   # under runs/: the instance role reads runs/* only (terraform compute module)
W=build/risk_calib_aws
SEEDS="${SEEDS:-0 1 2}"
IMAGES=("ariadne/core-dev:dev ariadne/core-dev:dev" "ariadne/sim:dev ariadne/sim:dev")
BINS=(ariadne_sim_collab ariadne_volumetric_eval)
log() { echo "[launch $(date -u +%H:%M:%S)] $*" >&2; }
stage() {
  [ -z "$(git status --porcelain -- . ':!.cache')" ] || { log "dirty tree"; exit 1; }
  rm -rf "$W" payload; mkdir -p "$W" payload
  git rev-parse HEAD > "$W/commit"
  for b in "${BINS[@]}"; do cp "build/docker/cpp/runtime/$b" payload/; done
  local ins=(payload)
  for s in $SEEDS; do ins+=("build/worldgen/scenes/ci_tiny_hazard_$s"); done
  find "${ins[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$W/inputs.SHA256SUMS"
  tar cf "$W/inputs.tar" "${ins[@]}" -C "$W" inputs.SHA256SUMS
  rm -rf payload
  "$AWS" aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG >/dev/null
  : > "$W/images.txt"; : > "$W/ecr_tags.txt"
  for pair in "${IMAGES[@]}"; do
    read -r src name <<< "$pair"
    local id; id=$(docker image inspect "$src" --format '{{.Id}}')
    local tag="rc-$(echo "${name%%:*}" | tr '/' '-')-${id#sha256:}"; tag=${tag:0:40}
    docker tag "$src" "$REG/$REPO_ECR:$tag"
    docker push -q "$REG/$REPO_ECR:$tag" >/dev/null
    echo "$REG/$REPO_ECR:$tag $name $id" >> "$W/images.txt"
    echo "$tag" >> "$W/ecr_tags.txt"
  done
  "$AWS" aws s3 cp "$W/inputs.tar" "$STAGE/inputs.tar" --only-show-errors
  "$AWS" aws s3 cp "$W/images.txt" "$STAGE/images.txt" --only-show-errors
  log "staged $(cat "$W/commit"): $(du -sh "$W/inputs.tar" | cut -f1) inputs, ${#IMAGES[@]} images"
}
run() {
  infra/scripts/spinup.sh run --profile gpu --max-minutes 240 -- \
    "STAGE=$STAGE COMMIT=$(cat "$W/commit") REG=$REG SEEDS='$SEEDS' bash infra/eval/risk_calib_aws.sh"
}
teardown() {
  if [ -s "$W/ecr_tags.txt" ]; then
    local ids=(); while read -r t; do ids+=("imageTag=$t"); done < "$W/ecr_tags.txt"
    "$AWS" aws ecr batch-delete-image --repository-name $REPO_ECR --image-ids "${ids[@]}" \
      --query 'imageIds[].imageTag' --output text
    while read -r t; do docker rmi "$REG/$REPO_ECR:$t" >/dev/null 2>&1 || true; done < "$W/ecr_tags.txt"
  fi
  "$AWS" aws s3 rm --recursive "$STAGE/" --only-show-errors
  infra/scripts/spinup.sh sweep
  docker logout $REG >/dev/null 2>&1 || true
  log "teardown done"
}
"${1:?stage | run | teardown}"
