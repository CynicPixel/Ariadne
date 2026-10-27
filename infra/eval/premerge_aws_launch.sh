#!/bin/bash
# Host side of the AWS pre-merge gate: stage the inputs, launch infra/eval/premerge_aws.sh through spinup.sh, then
# tear down what this run staged. Runs from the repo root on a clean HEAD.
#   infra/eval/premerge_aws_launch.sh stage [full]  git bundle, inputs.tar, recording sync, images to ECR (temp tags)
#   infra/eval/premerge_aws_launch.sh run       spinup.sh (cpux86, 120 min cap; 150 full) -> build/runs/<id>/
#   infra/eval/premerge_aws_launch.sh teardown  delete this run's temp ECR tags and staged inputs; sweep strays
# The recording (12 GB) is a PERSISTENT dataset under $DATA (2026-10-23: re-uploading per run costs
# too much time). `stage` uploads it once: `s3 sync` skips unchanged files, and its SHA256SUMS is computed once and
# kept next to it. Teardown never deletes it; remove it by hand when the recording is retired.
# `stage full` stages the FULL gate (phase1.sh without --premerge; phase gates): its fresh Kaggle render (D44) is
# relayed. The instance hands the render package to $STAGE/relay, `run` renders it here with this host's
# kaggle.sh (the Kaggle token never leaves this host) and hands renders.npz + remote.json back
# (infra/eval/render_relay.sh is the instance half).
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
ACCT=836505524290 REGION=eu-north-1
REG=$ACCT.dkr.ecr.$REGION.amazonaws.com REPO_ECR=ariadne/ariadne-core
BUCKET=s3://ariadne-$ACCT-eun1
# Both under runs/: the runner role may read and write runs/* only and "cannot touch datasets/" (compute module,
# least privilege; kept, 2026-10-23). runs/ has no expiry (IA transition only), so the copy persists.
STAGE=$BUCKET/runs/_staging/premerge
DATA=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_0-20261001T133914Z/raw
REC=evaluation/runs/gates/phase1_thin_e2e/sim_record/ci_tiny_0-20261001T133914Z/raw
W=build/premerge_aws
IMAGES=("ariadne/core-dev:m-premerge ariadne/core-dev:dev" "ariadne-recon:1 ariadne-recon:1"
        "ariadne-analysis:1 ariadne-analysis:1" "ariadne-console:1 ariadne-console:1")
log() { echo "[launch $(date -u +%H:%M:%S)] $*" >&2; }

stage() {
  [ -z "$(git status --porcelain -- .)" ] || { log "dirty tree"; exit 1; }
  local mode="${1:-premerge}"; [ "$mode" = premerge ] || [ "$mode" = full ] || { log "mode: premerge | full"; exit 1; }
  rm -rf "$W"; mkdir -p "$W"
  local commit; commit=$(git rev-parse HEAD); echo "$commit" > "$W/commit"; echo "$mode" > "$W/mode"
  git bundle create "$W/repo.bundle" HEAD 2>/dev/null
  # The dev image from this tree's Dockerfile, under a private tag (the shared :dev tag moves with other builds).
  DOCKER_BUILDKIT=1 docker build -q -f infra/docker/core.Dockerfile --target dev -t ariadne/core-dev:m-premerge . >/dev/null
  local ins=(.cache/models/xfeat_ref .cache/models/eigenplaces_ref .cache/models/aliked_lg_ref
             .cache/models/eigenplaces_sim_ref .cache/models/aliked_lg_sim_ref .cache/oracle/pgo build/render_cache)
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
  "$AWS" aws s3 rm --recursive "$STAGE/relay/" --only-show-errors   # a stale package.done would start the relay early
  "$AWS" aws s3 cp "$W/repo.bundle" "$STAGE/repo.bundle" --only-show-errors
  "$AWS" aws s3 cp "$W/inputs.tar" "$STAGE/inputs.tar" --only-show-errors
  "$AWS" aws s3 cp "$W/images.txt" "$STAGE/images.txt" --only-show-errors
  # Persistent: uploads only what is missing or changed. The AWS CLI runs in a container that mounts this checkout
  # only, so a recording symlinked in from another checkout (a worktree) cannot be synced from here: it is skipped,
  # the persistent copy is used, and the instance still checks every file against recording.SHA256SUMS.
  if [ -L "$REC" ]; then log "recording is a symlink: sync skipped, persistent copy used"
  else "$AWS" aws s3 sync "$REC/" "$DATA/" --only-show-errors; fi
  log "staged $commit ($mode): $(du -sh "$W/inputs.tar" | cut -f1) inputs, $(du -sh "$REC" | cut -f1) recording, ${#IMAGES[@]} images"
}

# The host half of a full gate's render: wait for the instance's package, render it on Kaggle, hand it back.
# Every key is a fresh object under $STAGE/relay (teardown removes it); a failure is reported to the instance.
relay() {
  local r="$W/relay" k="$STAGE/relay"; rm -rf "$r"; mkdir -p "$r/package" "$r/render"
  # Any failure still reaches the instance (which would otherwise wait out RELAY_TIMEOUT_MIN).
  trap 'echo "host relay error (line $LINENO)" > "$r/render.failed"
        "$AWS" aws s3 cp "$r/render.failed" "$k/render.failed" --only-show-errors; log "relay: FAILED"' ERR
  until "$AWS" aws s3 cp "$k/package.done" "$r/package.done" --only-show-errors >/dev/null 2>&1; do sleep 30; done
  "$AWS" aws s3 cp "$k/package.tar" "$r/package.tar" --only-show-errors
  "$AWS" aws s3 cp "$k/package.sha256" "$r/package.sha256" --only-show-errors
  tar xf "$r/package.tar" -C "$r/package"
  local want got; want=$(cat "$r/package.sha256"); got=$(sha256sum "$r/package/render_package.npz" | cut -d' ' -f1)
  if [ "$want" != "$got" ]; then
    echo "package sha256 $got, instance said $want" > "$r/render.failed"
  elif infra/scripts/kaggle.sh render "$r/package" "$r/render" > "$r/kaggle.log" 2>&1; then
    tar cf "$r/render.tar" -C "$r/render" renders.npz remote.json
    "$AWS" aws s3 cp "$r/render.tar" "$k/render.tar" --only-show-errors
    echo "$want" > "$r/render.done"; "$AWS" aws s3 cp "$r/render.done" "$k/render.done" --only-show-errors
    log "relay: rendered $want on Kaggle, handed back"; return
  else
    tail -30 "$r/kaggle.log" > "$r/render.failed"
  fi
  "$AWS" aws s3 cp "$r/render.failed" "$k/render.failed" --only-show-errors; log "relay: FAILED ($r/render.failed)"
}

run() {
  local commit mode; commit=$(cat "$W/commit"); mode=$(cat "$W/mode")
  local rpid=""; if [ "$mode" = full ]; then relay < /dev/null & rpid=$!; fi
  local rc=0
  infra/scripts/spinup.sh run --profile cpux86 --max-minutes "$([ "$mode" = full ] && echo 150 || echo 120)" -- \
    "STAGE=$STAGE DATA=$DATA COMMIT=$commit REG=$REG MODE=$mode bash infra/eval/premerge_aws.sh" || rc=$?
  if [ -n "$rpid" ]; then kill "$rpid" 2>/dev/null || true; wait "$rpid" 2>/dev/null || true; fi
  return $rc
}

teardown() {
  if [ -f "$W/ecr_tags.txt" ]; then
    local ids=(); while read -r t; do ids+=("imageTag=$t"); done < "$W/ecr_tags.txt"
    "$AWS" aws ecr batch-delete-image --repository-name $REPO_ECR --image-ids "${ids[@]}" \
      --query 'imageIds[].imageTag' --output text
  fi
  "$AWS" aws s3 rm --recursive "$STAGE/" --only-show-errors   # includes relay/
  infra/scripts/spinup.sh sweep
  [ -f "$W/ecr_tags.txt" ] && while read -r t; do docker rmi "$REG/$REPO_ECR:$t" >/dev/null 2>&1 || true; done < "$W/ecr_tags.txt"
  log "teardown done"
}

"${1:?stage [full] | run | teardown}" "${@:2}"
