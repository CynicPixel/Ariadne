#!/bin/bash
# Host side of the Phase 0.2/0.6 arm64 build (infra/eval/arm64_aws.sh runs on the instance).
#   infra/eval/arm64_aws_launch.sh run     ONE foreground spinup.sh attempt (profile cpuarm = c8g.4xlarge Ubuntu
#                                          24.04 arm64, 150 min cap) -> build/runs/<run-id>/output/results/
#   infra/eval/arm64_aws_launch.sh print   the instance command `run` would send, without launching
# Nothing is staged: spinup ships this working tree (tracked + untracked-not-ignored), and the instance builds
# everything from source (no ECR: the runner role cannot push, so the images come back as a zstd tarball).
# COMMIT/DIRTY are read at launch time (a queued job may start after later commits); DIRTY uses core.sh's pathspec.
# spinup's output is passed through untouched, so the slot queue sees "launched i-" and any spot-quota refusal.
set -euo pipefail
cd "$(dirname "$0")/../.."

instance_cmd() {
  local commit dirty tag
  commit=$(git rev-parse HEAD)
  if [ -n "$(git status --porcelain -- . ':!docs' ':!graphify-out' ':!evaluation/runs')" ]; then
    dirty=true; else dirty=false; fi
  # The one OpenCV pin lives in opencv-base.Dockerfile; read it rather than repeat it.
  tag=$(sed -n 's/^ARG OPENCV_TAG=//p' infra/docker/opencv-base.Dockerfile)
  [ -n "$tag" ] || { echo "no OPENCV_TAG in opencv-base.Dockerfile" >&2; exit 1; }
  echo "COMMIT=$commit DIRTY=$dirty OPENCV_TAG=$tag bash infra/eval/arm64_aws.sh"
}

run() {
  local cmd; cmd=$(instance_cmd)
  echo "[arm64 launch $(date -u +%H:%M:%S)Z] $cmd" >&2
  exec infra/scripts/spinup.sh run --profile cpuarm --max-minutes 150 -- "$cmd"
}

case "${1:-}" in
  run) run ;;
  print) instance_cmd ;;
  *) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
