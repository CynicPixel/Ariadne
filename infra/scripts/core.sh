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
#
# Docker is the only host dependency.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${ARIADNE_CORE_IMAGE:-ariadne/core:dev}"
DEV_IMAGE="${ARIADNE_CORE_DEV_IMAGE:-ariadne/core-dev:dev}"
JOBS="${ARIADNE_BUILD_JOBS:-4}"

cmd_image() {
  local commit dirty=false
  commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
  # Untracked files count: they are copied into the build context too. Paths
  # that .dockerignore keeps out of the context (docs/, IDEA.md, graphify-out/) cannot change
  # the image, so they are excluded here; keep the two lists in step.
  [ -n "$(git -C "$REPO_ROOT" status --porcelain -- . ':!docs' ':!IDEA.md' ':!graphify-out')" ] && dirty=true
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
  exec docker run --rm "${tty[@]}" -v "$REPO_ROOT":/work \
    -e CCACHE_DIR=/work/.cache/ccache-docker "$DEV_IMAGE" "$@"
}

case "${1:-}" in
  image) shift; cmd_image "$@" ;;
  gate)  shift; cmd_gate "$@" ;;
  dev)   shift; cmd_dev "$@" ;;
  *) sed -n '2,16p' "$0"; exit 2 ;;
esac
