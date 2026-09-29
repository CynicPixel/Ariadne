# ---------------------------------------------------------------------------
# Image build graph for CI (arch §27.2). `core` builds FROM `opencv-base`
# through a bake `target:` context, so one buildx invocation builds both in
# dependency order without the base ever needing a registry. Each target keeps
# its own layer cache, so the multi-hour OpenCV compile is paid once and then
# reused until OPENCV_TAG or its Dockerfile changes.
#
#   docker buildx bake -f infra/docker/docker-bake.hcl core
#
# CI overrides the cache settings (GitHub Actions cache); local builds use the
# plain docker build in infra/scripts/core.sh instead.
# ---------------------------------------------------------------------------
variable "OPENCV_TAG" { default = "5.0.0" }
variable "ARIADNE_GIT_COMMIT" { default = "UNRECORDED" }
variable "ARIADNE_GIT_DIRTY" { default = "true" }

target "opencv-base" {
  context    = "infra/docker"
  dockerfile = "opencv-base.Dockerfile"
  args       = { OPENCV_TAG = OPENCV_TAG }
  tags       = ["ariadne/opencv-base:${OPENCV_TAG}"]
}

target "core" {
  context    = "."
  dockerfile = "infra/docker/core.Dockerfile"
  target     = "ci"
  # Satisfies core.Dockerfile's default `FROM ariadne/opencv-base:5.0.0` with
  # the opencv-base target built above, not with an image pulled from anywhere.
  contexts = { "ariadne/opencv-base:${OPENCV_TAG}" = "target:opencv-base" }
  args = {
    OPENCV_BASE        = "ariadne/opencv-base:${OPENCV_TAG}"
    ARIADNE_GIT_COMMIT = ARIADNE_GIT_COMMIT
    ARIADNE_GIT_DIRTY  = ARIADNE_GIT_DIRTY
  }
  tags = ["ariadne/core:ci"]
}
