#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 0.2/0.6 arm64 arm: native OpenCV 5 + ariadne-core build and test on a Graviton host. Runs as root ON THE
# INSTANCE, launched by infra/eval/arm64_aws_launch.sh through spinup.sh (profile cpuarm, Ubuntu 24.04 arm64):
#   COMMIT=<sha> DIRTY=true|false OPENCV_TAG=5.0.0 bash infra/eval/arm64_aws.sh
# The working tree arrives as spinup's snapshot (no .git), so the launcher passes COMMIT/DIRTY.
#
# Steps (each logged to $ARIADNE_RESULTS/steps.txt with UTC time and seconds; each heavy step has its own timeout
# inside one overall budget, so brake 1 (shutdown at --max-minutes) never cuts off the result upload):
#   1 docker.io + docker-buildx (BuildKit: core.Dockerfile uses RUN --mount), zstd, 8 GB swap
#   2 opencv-base for linux/arm64, native, BUILD_JOBS=nproc        -> ariadne/opencv-base:<tag>-arm64
#   3 0.2 smoke test (infra/scripts/smoke_test.cpp, validation.md §4b recipe) + cv::getBuildInformation()
#     straight from the base image, so the OpenCV evidence survives a later core/test failure
#   4 core `dev` stage, then the repo built and the FULL ctest run inside it (Release, OpenCV required, no RPATH),
#     with a JUnit report: the per-test arm64 evidence, and it does not stop at the first failing test
#   5 core `ci` stage (compile + full ctest again + RPATH check + identity bake) -> ariadne/core:arm64,
#     only when step 4 passed (a failing ctest would fail it identically)
#   6 Phase 0 gate 4 on arm64 (core.sh gate) + the run manifest from ariadne_identity (§28.2 fields measurable
#     without a run, incl. the determinism digests MEASURED here)
#   7 docker save of the arm64 images, one zstd tarball (shared layers once) if it is < 3 GB
# Tags carry -arm64/:arm64 so a `docker load` on the amd64 workstation never overwrites its own images.
# Exit 0 only when every step passed.
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${COMMIT:?}" "${DIRTY:?}" "${OPENCV_TAG:?}"
SRC="$(pwd)"                                   # spinup runs this from the snapshot's root
JOBS="${JOBS:-$(nproc)}"
BUDGET_MIN="${BUDGET_MIN:-135}"                # spinup's --max-minutes is 150: 15 min left for upload + shutdown
T0=$(date +%s); DEADLINE=$(( T0 + BUDGET_MIN * 60 ))
BASE_IMG="ariadne/opencv-base:${OPENCV_TAG}-arm64"
DEV_IMG="ariadne/core-dev:arm64"
CORE_IMG="ariadne/core:arm64"
export HOME="${HOME:-/root}" DOCKER_BUILDKIT=1
RC=0

step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; RC=1; }
# timeout in seconds for a step: its own cap, but never past the overall deadline (minus 8 min for step 7).
cap() { local left=$(( DEADLINE - 8 * 60 - $(date +%s) )); [ "$left" -lt 60 ] && left=60
        [ $(( $1 * 60 )) -lt "$left" ] && echo $(( $1 * 60 )) || echo "$left"; }
timed() { local name=$1; shift; local t=$(date +%s); "$@"; local rc=$?
          echo "$name,$(( $(date +%s) - t )),$rc" >> "$R/timing.csv"; return $rc; }
kv() { printf '%s: %s\n' "$1" "$2" >> "$R/run_manifest.yaml"; }
echo "step,seconds,exit_code" > "$R/timing.csv"

step "0 host"
[ "$(uname -m)" = aarch64 ] || { fail "not an arm64 host ($(uname -m))"; exit 1; }
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"; uname -a > "$R/uname.txt"
cat /etc/os-release > "$R/os-release.txt"
# The snapshot's identity: sha256 over every shipped file (sorted), since there is no .git to name the dirty part.
SRC_HASH=$(cd "$SRC" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
{
  echo "# arm64 arm of Phase 0.2/0.6 (infra/eval/arm64_aws.sh). Times UTC."
  kv run_kind arm64_opencv_core_build
  kv created_utc "$(date -u +%FT%TZ)"
  kv git_commit "$COMMIT"; kv git_dirty "$DIRTY"; kv source_snapshot_sha256 "$SRC_HASH"
  kv opencv_tag "$OPENCV_TAG"
  kv instance_type "$(imds instance-type)"; kv ami_id "$(imds ami-id)"; kv availability_zone "$(imds placement/availability-zone)"
  kv kernel "$(uname -r)"; kv nproc "$(nproc)"; kv build_jobs "$JOBS"
  kv cpu_model "$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
} 2>/dev/null

step "1 docker + buildx + zstd + swap"
fallocate -l 8G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile \
  || echo "swap not added (continuing)" >> "$R/steps.txt"
apt-get -o DPkg::Lock::Timeout=600 update -qq > "$R/apt.txt" 2>&1
apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io docker-buildx zstd >> "$R/apt.txt" 2>&1 \
  || apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io zstd >> "$R/apt.txt" 2>&1 \
  || { fail "apt install docker.io"; exit 1; }
systemctl start docker 2>/dev/null || true
if ! docker buildx version >/dev/null 2>&1; then
  # [UNVERIFIED] fallback: noble ships docker-buildx in universe; if apt did not deliver it, the pinned upstream binary.
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-buildx \
    https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-arm64 \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
fi
docker buildx version > "$R/buildx_version.txt" 2>&1 || { fail "docker buildx missing (BuildKit is required)"; exit 1; }
docker version > "$R/docker_version.txt" 2>&1
kv docker_version "$(docker version --format '{{.Server.Version}}' 2>/dev/null)"
kv buildx_version "\"$(head -1 "$R/buildx_version.txt")\""
df -h / > "$R/df_start.txt"

step "2 opencv-base (OPENCV_TAG=$OPENCV_TAG, BUILD_JOBS=$JOBS)"
timed opencv_base timeout "$(cap 75)" docker build --platform linux/arm64 --progress=plain \
  -f infra/docker/opencv-base.Dockerfile --build-arg OPENCV_TAG="$OPENCV_TAG" --build-arg BUILD_JOBS="$JOBS" \
  -t "$BASE_IMG" infra/docker > "$R/opencv_base_build.log" 2>&1 \
  || { fail "opencv-base build (log opencv_base_build.log)"; tail -60 "$R/opencv_base_build.log"; exit 1; }
kv base_image_id "$(docker image inspect "$BASE_IMG" --format '{{.Id}}')"
kv base_image_arch "$(docker image inspect "$BASE_IMG" --format '{{.Os}}/{{.Architecture}}')"
kv base_image_bytes "$(docker image inspect "$BASE_IMG" --format '{{.Size}}')"
# FROM ubuntu:24.04 is unpinned (D33: pinned at this rebuild); record which one this build actually used.
kv ubuntu_base_repo_digest "$(docker image inspect ubuntu:24.04 --format '{{index .RepoDigests 0}}' 2>/dev/null)"
[ "$(docker image inspect "$BASE_IMG" --format '{{.Architecture}}')" = arm64 ] || fail "base image is not arm64"

step "3 smoke test + build information"
docker run --rm "$BASE_IMG" sh -c 'cat /opt/opencv5/opt-opencv-core.sha /opt/opencv5/opt-opencv-contrib.sha' \
  > "$R/opencv_commits.txt" 2>&1
kv opencv_core_commit "$(sed -n 1p "$R/opencv_commits.txt")"
kv opencv_contrib_commit "$(sed -n 2p "$R/opencv_commits.txt")"
# Peeled 5.0.0 commits from the opencv-base.Dockerfile comment (amd64 build): the arm64 checkout must match.
if [ "$OPENCV_TAG" = 5.0.0 ]; then
  printf '40738fb16ceddb5fb3fea747585f7ce6abb0605b\n755e50675d97db9b7d449d8bd6b09888646f6c6e\n' \
    | cmp -s - "$R/opencv_commits.txt" && kv opencv_commits_match_dockerfile true \
    || { kv opencv_commits_match_dockerfile false; fail "OpenCV commits differ from the pinned 5.0.0 peeled commits"; }
fi
cat > /opt/ariadne/build_info.cpp <<'CPP'
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <cstdio>
int main() {
    std::printf("CPU_FEATURES_LINE=%s\n", cv::getCPUFeaturesLine().c_str());
    std::fputs(cv::getBuildInformation().c_str(), stdout);
    return 0;
}
CPP
timed smoke_test docker run --rm -v "$SRC/infra/scripts:/w:ro" -v /opt/ariadne/build_info.cpp:/b/build_info.cpp:ro \
  "$BASE_IMG" sh -c '
  set -e
  libs=$(ls /opt/opencv5/lib/libopencv_*.so | sed "s|.*/lib|-l|;s|\.so$||")
  g++ -std=c++17 -O2 /b/build_info.cpp -o /tmp/bi -I/opt/opencv5/include/opencv5 -L/opt/opencv5/lib -lopencv_core
  LD_LIBRARY_PATH=/opt/opencv5/lib /tmp/bi > /tmp/build_info.txt
  g++ -std=c++17 -O2 /w/smoke_test.cpp -o /tmp/smoke -I/opt/opencv5/include/opencv5 -L/opt/opencv5/lib $libs
  echo "=== build info"; cat /tmp/build_info.txt
  echo "=== smoke"; LD_LIBRARY_PATH=/opt/opencv5/lib /tmp/smoke' > "$R/smoke_and_build_info.txt" 2>&1 \
  || fail "smoke test (smoke_and_build_info.txt)"
sed -n '/^=== build info/,/^=== smoke/p' "$R/smoke_and_build_info.txt" | sed '1d;$d' > "$R/opencv_build_info.txt"
sed -n '/^=== smoke/,$p' "$R/smoke_and_build_info.txt" | sed '1d' > "$R/smoke_test.txt"
# What "stock OpenCV" means on arm64 for the COOL Arm-vs-Arm benchmark: HALs and SIMD dispatch, verbatim.
grep -E 'CPU_FEATURES_LINE|HAL|Baseline|Dispatched|requested|KleidiCV|carotene|Parallel framework' \
  "$R/opencv_build_info.txt" > "$R/opencv_hal_simd.txt" 2>/dev/null || true

step "4 core dev image + full ctest (JUnit)"
timed core_dev timeout "$(cap 10)" docker build --platform linux/arm64 --progress=plain -f infra/docker/core.Dockerfile \
  --target dev --build-arg OPENCV_BASE="$BASE_IMG" -t "$DEV_IMG" . > "$R/core_dev_build.log" 2>&1 \
  || { fail "core dev image (core_dev_build.log)"; DEV_FAILED=1; }
CTEST_OK=0
if [ -z "${DEV_FAILED:-}" ]; then
  kv dev_image_id "$(docker image inspect "$DEV_IMG" --format '{{.Id}}')"
  chown -R 1000:1000 "$SRC"                    # the dev image runs as uid 1000 (ariadne)
  timed ctest_dev timeout "$(cap 40)" docker run --rm -v "$SRC":/work \
    -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY="$DIRTY" "$DEV_IMG" bash -c "
    set -o pipefail
    mkdir -p /work/build/arm64 /work/.cache   # the shipped tree has no build/ (run 63e43d failed here)
    cmake -S /work -B /work/build/arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DARIADNE_REQUIRE_OPENCV=ON \
      -DCMAKE_SKIP_RPATH=ON -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-arm64 > /work/build/configure.log 2>&1 \
      || { tail -50 /work/build/configure.log; exit 2; }
    cmake --build /work/build/arm64 --parallel $JOBS > /work/build/compile.log 2>&1 \
      || { grep -m40 -B2 -A8 'error' /work/build/compile.log; exit 3; }
    gcc --version | head -1 > /work/build/gcc_version.txt
    ctest --test-dir /work/build/arm64 --output-on-failure --parallel $JOBS \
      --output-junit /work/build/ctest_junit.xml" > "$R/ctest.log" 2>&1
  rc=$?
  for f in configure.log compile.log gcc_version.txt ctest_junit.xml; do cp "$SRC/build/$f" "$R/" 2>/dev/null; done
  kv gcc_version "\"$(cat "$R/gcc_version.txt" 2>/dev/null)\""
  {
    echo "exit_code: $rc   (2 configure, 3 compile, 124 timeout, else ctest)"
    grep -E 'tests passed|tests failed out of|Total Test time' "$R/ctest.log"
    sed -n '/The following tests FAILED/,$p' "$R/ctest.log"
    echo "skipped: $(grep -c 'Skipped' "$R/ctest.log")"
  } > "$R/ctest_summary.txt"
  kv ctest_result "\"$(grep -E 'tests passed' "$R/ctest.log" | tail -1)\""
  [ "$rc" = 0 ] && CTEST_OK=1 || fail "ctest in the dev image (rc $rc; ctest_summary.txt)"
fi

step "5 core ci image"
if [ "$CTEST_OK" = 1 ]; then
  timed core_ci timeout "$(cap 35)" docker build --platform linux/arm64 --progress=plain -f infra/docker/core.Dockerfile \
    --target ci --build-arg OPENCV_BASE="$BASE_IMG" --build-arg ARIADNE_GIT_COMMIT="$COMMIT" \
    --build-arg ARIADNE_GIT_DIRTY="$DIRTY" --build-arg BUILD_JOBS="$JOBS" -t "$CORE_IMG" . > "$R/core_ci_build.log" 2>&1 \
    || fail "core ci image (core_ci_build.log)"
  grep -E 'tests passed|tests failed out of' "$R/core_ci_build.log" | sed 's/^#[0-9]* [0-9.]* //' > "$R/ctest_ci_stage.txt"
else
  echo "skipped: dev-image ctest did not pass" | tee -a "$R/steps.txt"
fi

step "6 gate 4 + manifest"
if docker image inspect "$CORE_IMG" >/dev/null 2>&1; then
  CORE_ID=$(docker image inspect "$CORE_IMG" --format '{{.Id}}')
  kv core_image_id "$CORE_ID"
  kv core_image_arch "$(docker image inspect "$CORE_IMG" --format '{{.Os}}/{{.Architecture}}')"
  ARIADNE_CORE_IMAGE="$CORE_IMG" timed gate4 bash infra/scripts/core.sh gate > "$R/core_gate.txt" 2>&1 \
    || fail "core.sh gate (core_gate.txt)"
  kv core_image_gate "$(sed -n 's/^CORE_IMAGE_GATE=//p' "$R/core_gate.txt")"
  docker run --rm -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY="$DIRTY" -e ARIADNE_IMAGE_DIGEST="$CORE_ID" \
    "$CORE_IMG" bash -c 'ariadne_identity --manifest /tmp/m.yaml && cat /tmp/m.yaml' > "$R/manifest.yaml" 2>&1 \
    || fail "ariadne_identity --manifest"
  docker run --rm "$CORE_IMG" cat /opt/ariadne/opencv_identity.env > "$R/opencv_identity.env" 2>&1
else
  echo "skipped: no core image" | tee -a "$R/steps.txt"
fi

step "7 save images"
imgs=()
for i in "$BASE_IMG" "$DEV_IMG" "$CORE_IMG"; do docker image inspect "$i" >/dev/null 2>&1 && imgs+=("$i"); done
if [ ${#imgs[@]} -gt 0 ]; then
  out=/opt/ariadne/images-arm64.tar.zst
  timed save timeout "$(( $(( DEADLINE - $(date +%s) )) > 120 ? DEADLINE - $(date +%s) : 120 ))" \
    bash -c "docker save ${imgs[*]} | zstd -T0 -6 -q -o $out" || fail "docker save"
  sz=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$sz" -gt 0 ] && [ "$sz" -lt 3000000000 ]; then
    mv "$out" "$R/"; sha256sum "$R/images-arm64.tar.zst" | sed "s|$R/||" > "$R/images-arm64.tar.zst.sha256"
    printf '%s\n' "${imgs[@]}" > "$R/images-arm64.txt"
    kv images_tarball "images-arm64.tar.zst ($sz bytes: ${imgs[*]})"
  else
    kv images_tarball "not kept ($sz bytes)"
  fi
fi

df -h / > "$R/df_end.txt"
kv total_seconds "$(( $(date +%s) - T0 ))"
kv exit_code "$RC"
step "done (exit $RC)"
cat "$R/ctest_summary.txt" 2>/dev/null; cat "$R/run_manifest.yaml"
exit "$RC"
