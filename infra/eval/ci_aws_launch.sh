#!/bin/bash
# Host side of the generic "build + selected tests on AWS" job (infra/eval/ci_aws.sh): stage a clean HEAD, launch
# it through spinup.sh, tear down what this run staged. Moves compiles and ctest runs off the maintainer's
# workstation (hard cap: 2 heavy local jobs) onto one 16-vCPU spot instance. Runs from the repo root of WHATEVER
# checkout or worktree it lives in; NAME keys every run, so sessions never collide.
#   infra/eval/ci_aws_launch.sh stage NAME     git bundle, inputs.tar, dev image to ECR (temp tag) -> per-run prefix
#   infra/eval/ci_aws_launch.sh run NAME       ONE spinup.sh attempt (cpux86, MAX_MINUTES, default 90) -> build/runs/<id>/
#   infra/eval/ci_aws_launch.sh teardown NAME  delete this run's ECR temp tag and staging prefix; sweep strays
# NAME: [a-z0-9][a-z0-9._-]* (it ends up in an S3 key and an ECR tag), e.g. d174-proto.
#
# Environment:
#   stage  INPUTS       the ignored inputs the tests read, as repo-relative dirs. Default: the pre-merge list minus
#                       the recording and build/render_cache (only phase1.sh's render step reads that; no ctest):
#                       the five model reference exports and the PGO oracle (~1 GB). INPUTS="" stages none; the
#                       model tests then SKIP (the manifest says so).
#   run    TESTS        a ctest -R regex; empty = every test; "none" = no ctest (build + EXTRA only)
#          EXTRA        optional shell command, run after the build in the dev image from the repo root (/work),
#                       e.g. a smoke binary: EXTRA='build/docker/cpp/runtime/ariadne_identity --manifest /tmp/m.yaml'
#          MAX_MINUTES  spinup's cap (brake 1), default 90
#          JOBS / CTEST_JOBS  build and ctest parallelism on the instance, default 16 / 8
# TESTS and EXTRA travel base64-encoded (a regex like '^(CollabProtocol|Serialization)\.' survives no hand quoting).
#
# Usage from a session (times in IST; the dispatcher logs both):
#   1. Locally, by the owner, on a clean HEAD:  infra/eval/ci_aws_launch.sh stage d174-proto
#      Light except the dev image's docker build, which is cached unless infra/docker/ changed.
#   2. A queue job (/data/code/Ariadne/infra/scripts/slotq.sh; write <name>.tmp, rename to .job):
#        OWNER=M
#        DIR=/data/code/Ariadne-3x
#        VCPU=16
#        CMD='TESTS="^(CollabProtocol|Serialization)\." EXTRA="ctest --test-dir build/docker -N | tail -1" infra/eval/ci_aws_launch.sh run d174-proto'
#      `run` is one spinup attempt whose stdout/stderr pass through untouched: the dispatcher looks for
#      "launched i-" and requeues on MaxSpotInstanceCountExceeded, so a refused run must leave the staging intact.
#      Inside CMD='...' neither TESTS nor EXTRA can hold a single quote. A worktree has no Terraform state:
#      build/aws-json/runner.json is used as ARIADNE_RUNNER_JSON when it exists (else set it in CMD).
#   3. After the END event, by the owner: results in build/runs/<id>/output/results/, then
#      infra/eval/ci_aws_launch.sh teardown d174-proto
# The instance script itself comes from the working tree at `run` time (spinup.sh ships that snapshot); the code
# under test is the bundled commit from `stage`.
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
ACCT=836505524290 REGION=eu-north-1
REG=$ACCT.dkr.ecr.$REGION.amazonaws.com REPO_ECR=ariadne/ariadne-core
BUCKET=s3://ariadne-$ACCT-eun1
# Under runs/: the runner role may Get/Put runs/* only (no ListBucket, so the instance fetches named keys only).
# One prefix PER RUN, not premerge's shared one: concurrent jobs never overwrite each other's inputs.
STAGE_ROOT=$BUCKET/runs/_staging/ci
DEFAULT_INPUTS=".cache/models/xfeat_ref .cache/models/eigenplaces_ref .cache/models/aliked_lg_ref
  .cache/models/eigenplaces_sim_ref .cache/models/aliked_lg_sim_ref .cache/oracle/pgo"
log() { echo "[ci $(date -u +%H:%M:%S)] $*" >&2; }

case "${1:-}" in stage|run|teardown) ;; *) sed -n '2,9p' "$0"; exit 2 ;; esac
NAME="${2:-}"
[[ "$NAME" =~ ^[a-z0-9][a-z0-9._-]{0,47}$ ]] || { log "NAME: [a-z0-9][a-z0-9._-]*, at most 48 chars"; exit 1; }
W=build/ci_aws/$NAME

stage() {
  [ -z "$(git status --porcelain -- .)" ] || { log "dirty tree"; exit 1; }
  # A second stage under the same NAME would orphan the first one's ECR tag and prefix (and could pull them from
  # under a queued run): refuse.
  [ ! -e "$W/ecr_tags.txt" ] || { log "$W is staged: teardown $NAME first, or pick another NAME"; exit 1; }
  rm -rf "$W"; mkdir -p "$W"
  local commit short; commit=$(git rev-parse HEAD); short=$(git rev-parse --short=12 HEAD)
  local stage="$STAGE_ROOT/$NAME-$short"
  echo "$commit" > "$W/commit"; echo "$stage" > "$W/stage"
  git bundle create "$W/repo.bundle" HEAD 2>/dev/null
  # The dev image from this tree's Dockerfile, under a private tag (the shared :dev tag moves with other sessions;
  # docker tags are host-global, so the tag carries NAME).
  local img="ariadne/core-dev:ci-$NAME"
  DOCKER_BUILDKIT=1 docker build -q -f infra/docker/core.Dockerfile --target dev -t "$img" . >/dev/null
  echo "$img" > "$W/local_image"
  local ins=(); read -r -a ins <<< "$(echo ${INPUTS-$DEFAULT_INPUTS})"
  local p; for p in "${ins[@]}"; do [ -d "$p" ] || { log "missing input $p (set INPUTS)"; exit 1; }; done
  # Never `find` an empty list: it would scan . and checksum the whole repo.
  # In a worktree .cache is a symlink into the main checkout (and an input dir may be one too): find -L and tar -h
  # follow them, so the tar holds real files and the instance (and its dev container) needs no second mount.
  if [ ${#ins[@]} -gt 0 ]; then find -L "${ins[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$W/inputs.SHA256SUMS"
  else : > "$W/inputs.SHA256SUMS"; fi
  : > "$W/inputs.list"; [ ${#ins[@]} -eq 0 ] || printf '%s\n' "${ins[@]}" > "$W/inputs.list"
  tar chf "$W/inputs.tar" "${ins[@]}" -C "$W" inputs.SHA256SUMS inputs.list
  "$AWS" aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REG >/dev/null
  local id; id=$(docker image inspect "$img" --format '{{.Id}}'); id=${id#sha256:}
  local tag="ci-$NAME-$short-${id:0:12}"
  docker tag "$img" "$REG/$REPO_ECR:$tag"
  docker push -q "$REG/$REPO_ECR:$tag" >/dev/null
  echo "$REG/$REPO_ECR:$tag ariadne/core-dev:dev sha256:$id" > "$W/images.txt"
  echo "$tag" > "$W/ecr_tags.txt"
  "$AWS" aws s3 cp "$W/repo.bundle" "$stage/repo.bundle" --only-show-errors
  "$AWS" aws s3 cp "$W/inputs.tar" "$stage/inputs.tar" --only-show-errors
  "$AWS" aws s3 cp "$W/images.txt" "$stage/images.txt" --only-show-errors
  log "staged $commit as $NAME: $(du -sh "$W/inputs.tar" | cut -f1) inputs (${#ins[@]} dirs), image $tag -> $stage"
}

run() {
  [ -f "$W/ecr_tags.txt" ] || { log "$NAME is not staged (stage $NAME first)"; exit 1; }
  local commit stage; commit=$(cat "$W/commit"); stage=$(cat "$W/stage")
  if [ -z "${ARIADNE_RUNNER_JSON:-}" ] && [ -f build/aws-json/runner.json ]; then
    export ARIADNE_RUNNER_JSON="$PWD/build/aws-json/runner.json"
  fi
  local tests_b64 extra_b64
  tests_b64=$(printf '%s' "${TESTS:-}" | base64 -w0); extra_b64=$(printf '%s' "${EXTRA:-}" | base64 -w0)
  # One spinup call, nothing piped: its stderr ("launched i-", a quota refusal) is what the queue reads, and its
  # rc is this script's rc. No teardown here: a refused attempt is retried by the queue against the same staging.
  infra/scripts/spinup.sh run --profile cpux86 --max-minutes "${MAX_MINUTES:-90}" -- \
    "STAGE=$stage COMMIT=$commit REG=$REG NAME=$NAME TESTS_B64=$tests_b64 EXTRA_B64=$extra_b64 \
JOBS=${JOBS:-16} CTEST_JOBS=${CTEST_JOBS:-8} bash infra/eval/ci_aws.sh"
}

teardown() {
  [ -f "$W/stage" ] || { log "$W has no staging record"; exit 1; }
  local stage; stage=$(cat "$W/stage")
  if [ -f "$W/ecr_tags.txt" ]; then
    local ids=(); while read -r t; do ids+=("imageTag=$t"); done < "$W/ecr_tags.txt"
    "$AWS" aws ecr batch-delete-image --repository-name $REPO_ECR --image-ids "${ids[@]}" \
      --query 'imageIds[].imageTag' --output text
    while read -r t; do docker rmi "$REG/$REPO_ECR:$t" >/dev/null 2>&1 || true; done < "$W/ecr_tags.txt"
  fi
  [ -f "$W/local_image" ] && { docker rmi "$(cat "$W/local_image")" >/dev/null 2>&1 || true; }
  "$AWS" aws s3 rm --recursive "$stage/" --only-show-errors
  infra/scripts/spinup.sh sweep
  rm -f "$W/ecr_tags.txt"   # this NAME may be staged again
  log "teardown $NAME done"
}

"$1"
