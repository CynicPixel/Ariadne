#!/bin/bash
# Host side of the pipeline profile's AWS replay (infra/eval/perf_aws.sh; A2, 2026-10-24). From the repo root, clean HEAD:
#   infra/eval/perf_aws_launch.sh stage            inputs.tar (models incl. the 960x540/640x360 exports) -> S3 staging
#   infra/eval/perf_aws_launch.sh run PROFILE [STAGES]   ONE foreground spinup.sh attempt (cpux86 | cpuarm), for slotq
#   infra/eval/perf_aws_launch.sh replay           ONE spinup.sh attempt (cpux86): perf_replay_aws.sh, B's ma200 run replayed
#                                                  from its inputs.mcap with its own code/images, 3 OpenCV pool sizes
#   infra/eval/perf_aws_launch.sh teardown         delete the staged inputs
# The recordings are the persistent datasets collab_aws_launch.sh keeps (never re-uploaded, never deleted here); the
# pass-A stores are read from run 0345a5's results (STORES_FROM). Nothing is pushed to ECR: the instance builds the
# images from source on both arches (perf_aws.sh step 2).
set -euo pipefail
cd "$(dirname "$0")/../.."
AWS=infra/scripts/aws.sh
BUCKET=s3://ariadne-836505524290-eun1
STAGE=$BUCKET/runs/_staging/perf-a2
DATA0=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_0-20261001T133914Z/raw
DATA1=$BUCKET/runs/_datasets/phase1_sim_record/ci_tiny_1-20261001T143351Z/raw
STORES_FROM=$BUCKET/runs/20261023T163202Z-cpux86-0345a5/output/results
W=build/perf_aws
log() { echo "[perf $(date -u +%H:%M:%S)] $*" >&2; }
INPUTS=(
  .cache/models/aliked_lg_sim_ref
  .cache/models/aliked_lg_sim_960x540
  .cache/models/aliked_lg_sim_640x360
  .cache/models/aliked_lg_sim_l2conv
  .cache/models/eigenplaces_sim_ref/eigenplaces_r50_256_640x360.onnx
  .cache/models/eigenplaces_sim_ref/ref_sim_a0_f200_640x360.atb
  .cache/models/eigenplaces_sim_ref/input_sim_a0_f200.png
  .cache/models/eigenplaces_sim_ref/manifest.yaml
)
stage() {
  [ -z "$(git status --porcelain -- . ':!IDEA.md' ':!docs/ARIADNE_PHASED_PLAN.md')" ] || { log "dirty tree"; exit 1; }
  rm -rf "$W"; mkdir -p "$W"
  git rev-parse HEAD > "$W/commit"
  find "${INPUTS[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > "$W/inputs.SHA256SUMS"
  tar cf "$W/inputs.tar" "${INPUTS[@]}" -C "$W" inputs.SHA256SUMS
  "$AWS" aws s3 cp "$W/inputs.tar" "$STAGE/inputs.tar" --only-show-errors
  log "staged $(cat "$W/commit"): $(du -sh "$W/inputs.tar" | cut -f1) inputs"
}
run() {   # $1 = cpux86 | cpuarm, $2 = stages
  local commit; commit=$(cat "$W/commit")
  [ "$(git rev-parse HEAD)" = "$commit" ] || { log "HEAD moved since stage"; exit 1; }
  local stages="${2:-timing knobs adaptive}"
  [ "$1" = cpuarm ] && stages="${2:-timing}"
  # MAXMIN: set from the measured step times (M, 2026-10-24: the instance's own brake must bound a hang).
  # SINGLE_THREADS / C3_THREADS / ORT_RUNS: a rerun skips what an earlier run already measured.
  exec infra/scripts/spinup.sh run --profile "$1" --max-minutes "${MAXMIN:-120}" -- \
    "COMMIT=$commit STAGE=$STAGE DATA0=$DATA0 DATA1=$DATA1 STORES_FROM=$STORES_FROM STAGES='$stages' \
     SINGLE_THREADS='${SINGLE_THREADS-1 2 4 8 16}' C3_THREADS='${C3_THREADS-2 4 5}' ORT_RUNS=${ORT_RUNS-yes} \
     bash infra/eval/perf_aws.sh"
}
replay() {   # needs no staging: every input is B's run's own S3 objects (snapshot, payload, run.tar.gz) + its ECR image
  exec infra/scripts/spinup.sh run --profile cpux86 --max-minutes "${MAXMIN:-70}" -- \
    "RUN_ID=20261023T200447Z-cpux86-70a514 PAYLOAD=$BUCKET/runs/live-payload-20261023T200347Z/payload.tar.gz BUCKET=$BUCKET \
     VARIANTS='${VARIANTS:-asis t5 t2}' ${CODE_SNAPSHOT:+CODE_SNAPSHOT=$CODE_SNAPSHOT} ${CODE_PAYLOAD:+PAYLOAD=$CODE_PAYLOAD} \
     ${CORE_REF:+CORE_REF=$CORE_REF} ${CORE_ID:+CORE_ID=$CORE_ID} REPLAY_ARGS='${REPLAY_ARGS:-}' REPLAY_ENV='${REPLAY_ENV:-}' VARIANT_ENV='${VARIANT_ENV:-}' \
     bash infra/eval/perf_replay_aws.sh"
}
teardown() { "$AWS" aws s3 rm --recursive "$STAGE/" --only-show-errors; log "teardown done"; }
"${1:?stage | run PROFILE [STAGES] | replay | teardown}" "${@:2}"
