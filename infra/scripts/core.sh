#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build, gate and use the ariadne-core image (arch §27.1).
#
#   infra/scripts/core.sh image        build the ci stage: compile + full ctest
#                                      against OpenCV 5; fails if any test fails
#   infra/scripts/core.sh gate         Phase 0 gate 4: the OpenCV build hash a
#                                      manifest records at run time must equal
#                                      the one baked in at image build time
#   infra/scripts/core.sh dev [cmd]    dev stage, repo bind-mounted at /work;
#                                      default cmd configures + builds + tests
#                                      into build/docker/
#   infra/scripts/core.sh euroc [SEQ]  run the 1.4 front-end over a fetched
#                                      EuRoC sequence (default MH_01_easy) and
#                                      write manifest + results under
#                                      evaluation/runs/phase1/euroc_frontend/
#                                      (fetch first: infra/scripts/fetch_euroc.py)
#   infra/scripts/core.sh sim-calib REC SCENE  the 1.1 acceptance test (design §6)
#                                      over a sim recording; results + manifest into
#                                      evaluation/runs/phase1/sim_calib/; exit 1 on FAIL
#   infra/scripts/core.sh sim-vo REC SCENE [AGENT]  the same 1.4+1.5 pipeline over one
#                                      agent of a sim recording (configs sim_v1), into
#                                      evaluation/runs/phase1/sim_vo/
#   infra/scripts/core.sh sim-collab REC SCENE [FAULTS]  1.6 placeholder collaboration
#                                      (both agents, broker = agent 0, collab_v1) into
#                                      evaluation/runs/phase1/sim_collab/; FAULTS=1
#                                      duplicates/reorders messages (results must match)
#   infra/scripts/core.sh euroc-vo [SEQ]  run the 1.5 stereo VO over a fetched
#                                      sequence, scored against ground truth,
#                                      into evaluation/runs/phase1/euroc_vo/
#
# Docker is the only host dependency.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ARIADNE_CORE_IMAGE:-ariadne/core:dev}"
DEV_IMAGE="${ARIADNE_CORE_DEV_IMAGE:-ariadne/core-dev:dev}"
JOBS="${ARIADNE_BUILD_JOBS:-4}"

# Untracked files count: they are copied into the build context too. Paths
# that .dockerignore keeps out of the context (docs/, IDEA.md, graphify-out/,
# evaluation/runs/) cannot change the image, so they are excluded here; keep the
# two lists in step. evaluation/runs/ holds run OUTPUTS: an evidence run that
# copies its manifests there must not mark the next run's tree dirty.
git_dirty() {
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out' ':!evaluation/runs')" ]; then
    echo true
  else
    echo false
  fi
}

cmd_image() {
  local commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  DOCKER_BUILDKIT=1 docker build -f "$REPO_ROOT/infra/docker/core.Dockerfile" \
    --target ci --build-arg ARIADNE_GIT_COMMIT="$commit" \
    --build-arg ARIADNE_GIT_DIRTY="$dirty" --build-arg BUILD_JOBS="$JOBS" \
    --progress=plain -t "$IMAGE" "$REPO_ROOT"
  echo "IMAGE=$IMAGE ID=$(docker image inspect "$IMAGE" --format '{{.Id}}') GIT_DIRTY=$dirty"
}

cmd_gate() {
  local image_id
  image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  # One container, so both values come from the same filesystem and loader.
  # The manifest is written by the binary; the baked file was written by the
  # same binary at build time. They can only differ if the loader resolved a
  # different libopencv_core than the image was built with.
  docker run --rm -e ARIADNE_IMAGE_DIGEST="$image_id" "$IMAGE" bash -c '
    set -euo pipefail
    ariadne_identity --manifest /tmp/manifest.yaml
    baked=$(sed -n "s/^OPENCV_BUILD_HASH=//p" /opt/ariadne/opencv_identity.env)
    recorded=$(sed -n "s/^opencv_build_hash: \"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" /tmp/manifest.yaml)
    echo "--- manifest written at run time ---"; cat /tmp/manifest.yaml; echo
    echo "BAKED_OPENCV_BUILD_HASH=$baked"
    echo "MANIFEST_OPENCV_BUILD_HASH=$recorded"
    if [ -n "$baked" ] && [ "$baked" = "$recorded" ]; then
      echo "CORE_IMAGE_GATE=PASS"
    else
      echo "CORE_IMAGE_GATE=FAIL"; exit 1
    fi'
}

cmd_dev() {
  DOCKER_BUILDKIT=1 docker build -q -f "$REPO_ROOT/infra/docker/core.Dockerfile" \
    --target dev -t "$DEV_IMAGE" "$REPO_ROOT" >/dev/null
  local tty=()
  [ -t 0 ] && tty=(-it)
  if [ $# -eq 0 ]; then
    set -- bash -c "cmake -S /work -B /work/build/docker -G Ninja -DARIADNE_REQUIRE_OPENCV=ON \
      -DCMAKE_SKIP_RPATH=ON -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-docker \
      && cmake --build /work/build/docker --parallel $JOBS \
      && ctest --test-dir /work/build/docker --output-on-failure --parallel $JOBS"
  fi
  # ccache lives under the repo's gitignored .cache/ so it survives containers.
  # The image id reaches every run manifest (it was UNRECORDED before 2026-10-01).
  exec docker run --rm "${tty[@]}" -v "$REPO_ROOT":/work \
    -e CCACHE_DIR=/work/.cache/ccache-docker \
    -e ARIADNE_IMAGE_DIGEST="$(docker image inspect "$DEV_IMAGE" --format '{{.Id}}')" "$DEV_IMAGE" "$@"
}

cmd_euroc() {
  local seq="${1:-MH_01_easy}" commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  [ -d "$REPO_ROOT/.cache/datasets/euroc/$seq/mav0" ] \
    || { echo "missing $seq; run infra/scripts/fetch_euroc.py $seq" >&2; exit 1; }
  cmd_dev bash -c "cmake --build /work/build/docker --target ariadne_euroc_frontend --parallel $JOBS >/dev/null \
    && ARIADNE_GIT_COMMIT=$commit ARIADNE_GIT_DIRTY=$dirty \
       /work/build/docker/cpp/runtime/ariadne_euroc_frontend \
       --sequence /work/.cache/datasets/euroc/$seq \
       --config /work/configs/perception/euroc.yaml \
       --out /work/evaluation/runs/phase1/euroc_frontend --stereo-every 10"
}

cmd_euroc_vo() {
  local seq="${1:-MH_01_easy}" commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  [ -d "$REPO_ROOT/.cache/datasets/euroc/$seq/mav0" ] \
    || { echo "missing $seq; run infra/scripts/fetch_euroc.py $seq" >&2; exit 1; }
  cmd_dev bash -c "cmake --build /work/build/docker --target ariadne_euroc_vo --parallel $JOBS >/dev/null \
    && ARIADNE_GIT_COMMIT=$commit ARIADNE_GIT_DIRTY=$dirty \
       /work/build/docker/cpp/runtime/ariadne_euroc_vo \
       --sequence /work/.cache/datasets/euroc/$seq \
       --perception-config /work/configs/perception/euroc.yaml \
       --slam-config /work/configs/slam/euroc.yaml \
       --out /work/evaluation/runs/phase1/euroc_vo"
}

cmd_sim_calib() {
  local rec="${1:?recording dir}" scene="${2:?scene dir}" commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  cmd_dev bash -c "cmake --build /work/build/docker --target ariadne_sim_calib --parallel $JOBS >/dev/null \
    && ARIADNE_GIT_COMMIT=$commit ARIADNE_GIT_DIRTY=$dirty ARIADNE_SOURCE_DIR=/work \
       /work/build/docker/cpp/runtime/ariadne_sim_calib --rec /work/$rec --scene /work/$scene \
       --out /work/evaluation/runs/phase1/sim_calib"
}

cmd_sim_vo() {
  local rec="${1:?recording dir}" scene="${2:?scene dir}" agent="${3:-0}" commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  cmd_dev bash -c "cmake --build /work/build/docker --target ariadne_sim_vo --parallel $JOBS >/dev/null \
    && ARIADNE_GIT_COMMIT=$commit ARIADNE_GIT_DIRTY=$dirty \
       /work/build/docker/cpp/runtime/ariadne_sim_vo --rec /work/$rec --scene /work/$scene --agent $agent \
       --perception-config /work/configs/perception/sim_v1.yaml --slam-config /work/configs/slam/sim_v1.yaml \
       --out /work/evaluation/runs/phase1/sim_vo"
}

cmd_sim_collab() {
  local rec="${1:?recording dir}" scene="${2:?scene dir}" faults="${3:-0}" commit dirty
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  dirty="$(git_dirty)"
  cmd_dev bash -c "cmake --build /work/build/docker --target ariadne_sim_collab --parallel $JOBS >/dev/null \
    && ARIADNE_GIT_COMMIT=$commit ARIADNE_GIT_DIRTY=$dirty \
       /work/build/docker/cpp/runtime/ariadne_sim_collab --rec /work/$rec --scene /work/$scene \
       --perception-config /work/configs/perception/sim_v1.yaml --slam-config /work/configs/slam/sim_v1.yaml \
       --collab-config /work/configs/collab/collab_v1.yaml --inject-faults $faults \
       --out /work/evaluation/runs/phase1/sim_collab"
}

case "${1:-}" in
  image) shift; cmd_image "$@" ;;
  gate)  shift; cmd_gate "$@" ;;
  dev)   shift; cmd_dev "$@" ;;
  euroc) shift; cmd_euroc "$@" ;;
  euroc-vo) shift; cmd_euroc_vo "$@" ;;
  sim-calib) shift; cmd_sim_calib "$@" ;;
  sim-vo) shift; cmd_sim_vo "$@" ;;
  sim-collab) shift; cmd_sim_collab "$@" ;;
  *) sed -n '2,33p' "$0"; exit 2 ;;
esac
