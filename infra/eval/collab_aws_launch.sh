#!/bin/bash
# Host side of the 3.2/3.4 gate sweep (infra/eval/collab_aws.sh; design SIMCOLLAB_3.2, D171): stage the inputs,
# launch through spinup.sh, tear down what this run staged. Runs from the repo root on a clean HEAD.
#   infra/eval/collab_aws_launch.sh stage      git bundle, inputs.tar, both recordings (persistent), image to ECR
#   infra/eval/collab_aws_launch.sh run [STAGES]   spinup.sh (cpux86, 300 min cap) -> build/runs/<id>/
#   infra/eval/collab_aws_launch.sh teardown   delete this run's temp ECR tag and staged inputs; sweep strays
# The recordings are PERSISTENT datasets under runs/_datasets/ (maintainer, 2026-10-23: re-uploading per run costs
# too much time): `stage` syncs each once (s3 sync skips unchanged files); teardown never deletes them.
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
ACCT=836505524290 REGION=eu-north-1
REG=$ACCT.dkr.ecr.$REGION.amazonaws.com REPO_ECR=ariadne/ariadne-core
BUCKET=s3://ariadne-$ACCT-eun1
STAGE=$BUCKET/runs/_staging/collab
REC0=evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T133914Z/raw
REC1=evaluation/runs/phase1/sim_record/ci_tiny_1-20261001T143351Z/raw
DATA0=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_0-20261001T133914Z/raw
DATA1=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_1-20261001T143351Z/raw
W=build/collab_aws
log() { echo "[collab $(date -u +%H:%M:%S)] $*" >&2; }

sums() {   # $1 = recording dir, $2 = cached checksum list (computed once per recording)
  [ -s "$2" ] || (cd "$1" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$2"
}

stage() {
  [ -z "$(git status --porcelain -- . ':!IDEA.md' ':!docs/ARIADNE_PHASED_PLAN.md')" ] || { log "dirty tree"; exit 1; }
  rm -rf "$W"; mkdir -p "$W"
  local commit; commit=$(git rev-parse HEAD); echo "$commit" > "$W/commit"
  git bundle create "$W/repo.bundle" HEAD 2>/dev/null
  DOCKER_BUILDKIT=1 docker build -q -f infra/docker/core.Dockerfile --target dev -t ariadne/core-dev:m-collab . >/dev/null
  sums "$REC0" build/recording_ci_tiny_0.SHA256SUMS
  sums "$REC1" build/recording_ci_tiny_1.SHA256SUMS
  cp build/recording_ci_tiny_0.SHA256SUMS "$W/rec0.SHA256SUMS"
  cp build/recording_ci_tiny_1.SHA256SUMS "$W/rec1.SHA256SUMS"
  echo "$REC0" > "$W/inputs.rec0"; echo "$REC1" > "$W/inputs.rec1"
  local ins=(.cache/models/aliked_lg_sim_ref .cache/models/eigenplaces_sim_ref)
  find "${ins[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$W/inputs.SHA256SUMS"
  tar cf "$W/inputs.tar" "${ins[@]}" -C "$W" inputs.SHA256SUMS rec0.SHA256SUMS rec1.SHA256SUMS inputs.rec0 inputs.rec1
  "$AWS" aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG >/dev/null
  local id; id=$(docker image inspect ariadne/core-dev:m-collab --format '{{.Id}}'); id=${id#sha256:}
  local tag="cl-core-dev-${id:0:12}"
  docker tag ariadne/core-dev:m-collab "$REG/$REPO_ECR:$tag"
  docker push -q "$REG/$REPO_ECR:$tag" >/dev/null
  echo "$REG/$REPO_ECR:$tag ariadne/core-dev:dev sha256:$id" > "$W/images.txt"
  echo "$tag" > "$W/ecr_tags.txt"
  "$AWS" aws s3 cp "$W/repo.bundle" "$STAGE/repo.bundle" --only-show-errors
  "$AWS" aws s3 cp "$W/inputs.tar" "$STAGE/inputs.tar" --only-show-errors
  "$AWS" aws s3 cp "$W/images.txt" "$STAGE/images.txt" --only-show-errors
  # The AWS CLI runs in a container that mounts this checkout only: a recording symlinked in from another checkout
  # (a worktree) cannot be synced from here, so it is skipped and the persistent copy is used (the instance still
  # checks every file against rec<K>.SHA256SUMS).
  local r d
  for r in "$REC0:$DATA0" "$REC1:$DATA1"; do
    d=${r#*:}; r=${r%%:*}
    if [ -L "$r" ]; then log "$r is a symlink: sync skipped, persistent copy used"
    else "$AWS" aws s3 sync "$r/" "$d/" --only-show-errors; fi
  done
  log "staged $commit: $(du -sh "$W/inputs.tar" | cut -f1) inputs, image $tag"
}

run() {
  local commit; commit=$(cat "$W/commit")
  local stages="${1:-A B1 B2 C}"
  infra/scripts/spinup.sh run --profile cpux86 --max-minutes 300 -- \
    "STAGE=$STAGE DATA0=$DATA0 DATA1=$DATA1 COMMIT=$commit REG=$REG STAGES='$stages' bash infra/eval/collab_aws.sh"
}

teardown() {
  if [ -f "$W/ecr_tags.txt" ]; then
    "$AWS" aws ecr batch-delete-image --repository-name $REPO_ECR --image-ids "imageTag=$(cat "$W/ecr_tags.txt")" \
      --query 'imageIds[].imageTag' --output text
    docker rmi "$REG/$REPO_ECR:$(cat "$W/ecr_tags.txt")" >/dev/null 2>&1 || true
  fi
  "$AWS" aws s3 rm --recursive "$STAGE/" --only-show-errors
  infra/scripts/spinup.sh sweep
  log "teardown done"
}

"${1:?stage | run [STAGES] | teardown}" "${@:2}"
