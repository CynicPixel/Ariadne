#!/bin/bash
# Host side of the AWS pre-merge gate: stage the inputs, launch infra/eval/premerge_aws.sh through spinup.sh, then
# tear down what this run staged. Runs from the repo root on a clean HEAD.
#   infra/eval/premerge_aws_launch.sh stage     git bundle, inputs.tar, recording sync, images to ECR (temp tags)
#   infra/eval/premerge_aws_launch.sh run       spinup.sh (cpux86, 120 min cap) -> build/runs/<id>/
#   infra/eval/premerge_aws_launch.sh teardown  delete this run's temp ECR tags and staged inputs; sweep strays
# The recording (12 GB) is a PERSISTENT dataset under $DATA (maintainer, 2026-10-23: re-uploading per run costs
# too much time). `stage` uploads it once: `s3 sync` skips unchanged files, and its SHA256SUMS is computed once and
# kept next to it. Teardown never deletes it; remove it by hand when the recording is retired.
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
ACCT=836505524290 REGION=eu-north-1
REG=$ACCT.dkr.ecr.$REGION.amazonaws.com REPO_ECR=ariadne/ariadne-core
BUCKET=s3://ariadne-$ACCT-eun1
# Both under runs/: the runner role may read and write runs/* only and "cannot touch datasets/" (compute module,
# least privilege; maintainer kept it, 2026-10-23). runs/ has no expiry (IA transition only), so the copy persists.
STAGE=$BUCKET/runs/_staging/premerge
DATA=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_0-20261001T133914Z/raw
REC=evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T133914Z/raw
W=build/premerge_aws
IMAGES=("ariadne/core-dev:m-premerge ariadne/core-dev:dev" "ariadne-recon:1 ariadne-recon:1"
        "ariadne-analysis:1 ariadne-analysis:1" "ariadne-console:1 ariadne-console:1")
log() { echo "[launch $(date -u +%H:%M:%S)] $*" >&2; }

stage() {
  [ -z "$(git status --porcelain -- . ':!IDEA.md' ':!docs/ARIADNE_PHASED_PLAN.md')" ] || { log "dirty tree"; exit 1; }
  rm -rf "$W"; mkdir -p "$W"
  local commit; commit=$(git rev-parse HEAD); echo "$commit" > "$W/commit"
  git bundle create "$W/repo.bundle" HEAD 2>/dev/null
  # The dev image from this tree's Dockerfile, under a private tag (the shared :dev tag moves with other sessions).
  DOCKER_BUILDKIT=1 docker build -q -f infra/docker/core.Dockerfile --target dev -t ariadne/core-dev:m-premerge . >/dev/null
  local ins=(.cache/models/xfeat_ref .cache/models/eigenplaces_ref .cache/models/aliked_lg_ref .cache/oracle/pgo
             build/render_cache)
  find "${ins[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$W/inputs.SHA256SUMS"
  local sums=build/recording_ci_tiny_0.SHA256SUMS   # once per recording, locally cached
  [ -s "$sums" ] || (cd "$REC" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$sums"
  cp "$sums" "$W/recording.SHA256SUMS"
  echo "$REC" > "$W/inputs.recording_path"
  tar cf "$W/inputs.tar" "${ins[@]}" -C "$W" inputs.SHA256SUMS recording.SHA256SUMS inputs.recording_path
  "$AWS" aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG >/dev/null
  : > "$W/images.txt"
  for pair in "${IMAGES[@]}"; do
    read -r src name <<< "$pair"
    local id; id=$(docker image inspect "$src" --format '{{.Id}}'); id=${id#sha256:}
    local tag="pm-$(echo "${name%%:*}" | tr '/' '-')-${id:0:12}"
    docker tag "$src" "$REG/$REPO_ECR:$tag"
    docker push -q "$REG/$REPO_ECR:$tag" >/dev/null
    echo "$REG/$REPO_ECR:$tag $name sha256:$id" >> "$W/images.txt"
    echo "$tag" >> "$W/ecr_tags.txt"
  done
  "$AWS" aws s3 cp "$W/repo.bundle" "$STAGE/repo.bundle" --only-show-errors
  "$AWS" aws s3 cp "$W/inputs.tar" "$STAGE/inputs.tar" --only-show-errors
  "$AWS" aws s3 cp "$W/images.txt" "$STAGE/images.txt" --only-show-errors
  "$AWS" aws s3 sync "$REC/" "$DATA/" --only-show-errors   # persistent: uploads only what is missing or changed
  log "staged $commit: $(du -sh "$W/inputs.tar" | cut -f1) inputs, $(du -sh "$REC" | cut -f1) recording, ${#IMAGES[@]} images"
}

run() {
  local commit; commit=$(cat "$W/commit")
  infra/scripts/spinup.sh run --profile cpux86 --max-minutes 120 -- \
    "STAGE=$STAGE DATA=$DATA COMMIT=$commit REG=$REG bash infra/eval/premerge_aws.sh"
}

teardown() {
  if [ -f "$W/ecr_tags.txt" ]; then
    local ids=(); while read -r t; do ids+=("imageTag=$t"); done < "$W/ecr_tags.txt"
    "$AWS" aws ecr batch-delete-image --repository-name $REPO_ECR --image-ids "${ids[@]}" \
      --query 'imageIds[].imageTag' --output text
  fi
  "$AWS" aws s3 rm --recursive "$STAGE/" --only-show-errors
  infra/scripts/spinup.sh sweep
  [ -f "$W/ecr_tags.txt" ] && while read -r t; do docker rmi "$REG/$REPO_ECR:$t" >/dev/null 2>&1 || true; done < "$W/ecr_tags.txt"
  log "teardown done"
}

"${1:?stage | run | teardown}"
