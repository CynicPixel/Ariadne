#!/usr/bin/env bash
# Build and run the adopted live stack in one reproducible AWS campaign.
# Artifacts are retained by spinup.sh in S3 and copied locally under evaluation/runs.
set -Eeuo pipefail
trap 'echo "[final-e2e] FAILED (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE="evaluation/runs/live-agentic/final_stack_e2e/$STAMP"
mkdir -p "$EVIDENCE"
exec > >(tee -a "$EVIDENCE/campaign.log") 2>&1

PERCEPTION="configs/perception/sim_v1_rad_flat.yaml"
SLAM="configs/slam/sim_v2_loops.yaml"
IMU_NOISE="configs/sim/imu_noise_v1.yaml"
MISSION="configs/mission/mission_v1.yaml"
LIVE="configs/mission/live_final_e2e.yaml"
PLANNER="configs/mission/mission_v2.yaml"
SCENE="build/live/scenes/maze_short_6"
END_S="${ARIADNE_FINAL_END_S:-200}"
MODEL_SOURCE="${ARIADNE_MODEL_SOURCE:-/data/code/Ariadne/.cache/models}"

printf '[final-e2e] source commit: '; git rev-parse HEAD
git status --short > "$EVIDENCE/source_status.txt"
printf 'perception=%s\nslam=%s\nimu_noise=%s\nmission=%s\nlive=%s\nplanner=%s\nscene=%s\nend_s=%s\n' \
  "$PERCEPTION" "$SLAM" "$IMU_NOISE" "$MISSION" "$LIVE" "$PLANNER" "$SCENE" "$END_S" | tee "$EVIDENCE/selection.txt"

for f in "$PERCEPTION" "$SLAM" "$MISSION" "$LIVE" "$PLANNER"; do test -f "$f"; done
test -f configs/collab/collab_final_e2e.yaml
mkdir -p .cache/models
for model_dir in eigenplaces_sim_ref aliked_lg_sim_ref; do
  if [ ! -e ".cache/models/$model_dir" ]; then
    test -d "$MODEL_SOURCE/$model_dir" || { echo "missing model directory: $MODEL_SOURCE/$model_dir" >&2; exit 2; }
    ln -s "$MODEL_SOURCE/$model_dir" ".cache/models/$model_dir"
  fi
done
test -f .cache/models/eigenplaces_sim_ref/eigenplaces_r50_256_640x360.onnx
test -f .cache/models/aliked_lg_sim_ref/aliked_n16_k1024_1280x720.onnx
test -f .cache/models/aliked_lg_sim_ref/lightglue_aliked.onnx

echo '[final-e2e] build current source and run complete CTest suite'
if [ -n "${ARIADNE_FINAL_VALIDATED_PAYLOAD:-}" ]; then
  # Packaging-only retries reuse the exact executables that passed validation.
  grep -q '^100% tests passed, 0 tests failed' "$ARIADNE_FINAL_VALIDATION_LOG"
  for bin in ariadne_live ariadne_collab_worker; do
    cmp "$ARIADNE_FINAL_VALIDATED_PAYLOAD/build/docker/cpp/runtime/$bin" "build/docker/cpp/runtime/$bin"
  done
  echo "[final-e2e] reuse byte-identical validated executables; CTest evidence: $ARIADNE_FINAL_VALIDATION_LOG"
elif [ -n "${ARIADNE_FINAL_BUILD_CONTAINER:-}" ]; then
  # Continue the already-running validation if its interactive client was
  # interrupted. The container's actual exit code still gates the AWS launch.
  docker logs -f "$ARIADNE_FINAL_BUILD_CONTAINER" &
  build_log_pid=$!
  build_rc="$(docker wait "$ARIADNE_FINAL_BUILD_CONTAINER")" || build_rc=125
  wait "$build_log_pid" || true
  [ "$build_rc" = 0 ] || { echo "build/CTest container exit $build_rc"; exit 1; }
else
  ARIADNE_BUILD_JOBS="${ARIADNE_BUILD_JOBS:-2}" infra/scripts/core.sh dev
fi
echo '[final-e2e] build simulation plugins and ROS bridge'
ARIADNE_BUILD_JOBS="${ARIADNE_BUILD_JOBS:-2}" infra/scripts/live.sh build

echo '[final-e2e] compose deterministic two-agent maze scene'
infra/scripts/live.sh compose maze_short 6 0.15

export LIVE_AWS_PROFILE="${LIVE_AWS_PROFILE:-gpu}"
export LIVE_AWS_MAX_MIN="${LIVE_AWS_MAX_MIN:-180}"
export ARIADNE_LOCKSTEP=1
export ARIADNE_LIVE_CONFIG="$LIVE"
export ARIADNE_PERCEPTION_CONFIG="$PERCEPTION"
export ARIADNE_SLAM_CONFIG="$SLAM"
export ARIADNE_IMU_NOISE_CONFIG="$IMU_NOISE"
export ARIADNE_MISSION_CONFIG="$MISSION"
export ARIADNE_PLANNER_CONFIG="$PLANNER"
export ARIADNE_SHADOW_MAP_CONFIG="$PLANNER"
export ARIADNE_SPLAT=1
export ARIADNE_RECON_BACKEND="${ARIADNE_RECON_BACKEND:-gpu}"
export ARIADNE_RECON_IMAGE="${ARIADNE_RECON_IMAGE:-ariadne-recon-gpu:1}"
export ARIADNE_RECON_GPU_URI="${ARIADNE_RECON_GPU_URI:-s3://ariadne-836505524290-eun1/runs/images/ariadne-recon-gpu-a827de5514fe.tar.zst}"
export ARIADNE_RECON_ITERS="${ARIADNE_RECON_ITERS:-100}"
export ARIADNE_ARCHIVE_RECORDING=separate
export ARIADNE_SNAPSHOT_EXCLUDE='evaluation/runs/live-agentic/final_stack_e2e/*'
export ARIADNE_ROOT_GB=120
export ARIADNE_RT_MEM="${ARIADNE_RT_MEM:-10g}"

# A linked worktree has no independent Terraform backend cache. Read the
# already-applied runner outputs from the primary checkout and pass the JSON
# file to spinup.sh; no Terraform init/apply is performed here.
if [ -z "${ARIADNE_RUNNER_JSON:-}" ]; then
  STATE_ROOT="$(git worktree list --porcelain | sed -n 's/^worktree //p' | head -1)"
  test -x "$STATE_ROOT/infra/scripts/aws.sh"
  mkdir -p build/live_payload
  "$STATE_ROOT/infra/scripts/aws.sh" tf main output -json runner > build/live_payload/runner.json
  export ARIADNE_RUNNER_JSON="$ROOT/build/live_payload/runner.json"
fi

printf '[final-e2e] launch all-in-one recorded mission at %ss\n' "$END_S"
launch_rc=0
if infra/scripts/live_aws.sh run "$SCENE" "$END_S"; then launch_rc=0; else launch_rc=$?; fi

latest="$(find build/runs -mindepth 1 -maxdepth 1 -type d -name '*-gpu-*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$latest"
printf '%s\n' "$latest" | tee "$EVIDENCE/aws_run_dir.txt"
cp -a "$latest/output/." "$EVIDENCE/aws_output/"
mkdir -p "$EVIDENCE/aws_run"
accept_rc=0
if [ "$launch_rc" -ne 0 ]; then accept_rc="$launch_rc"; fi
if tar xzf "$latest/output/results/run.tar.gz" -C "$EVIDENCE/aws_run"; then
  python3 infra/eval/validate_final_stack_e2e.py "$EVIDENCE/aws_run" "$EVIDENCE/aws_output/results" "$ARIADNE_RECON_BACKEND" || accept_rc=$?
else
  echo '{"verdict":"FAIL","failures":["AWS mission archive missing or invalid"]}' \
    > "$EVIDENCE/aws_output/results/final_stack_acceptance.json"
  accept_rc=1
fi
find "$EVIDENCE" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$EVIDENCE/SHA256SUMS"
echo "[final-e2e] evidence: $EVIDENCE"
exit "$accept_rc"
