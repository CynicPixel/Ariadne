#!/bin/bash
# ---------------------------------------------------------------------------
# DIAGNOSTIC (A1, not a gate): where does GravityPgo.RecoversTheReferenceGravityErrorAndReducesTilt diverge on arm64
# (x86 0.62 deg, Graviton 2.71 deg, run 20261023T193621Z-cpuarm-7f8c96)? Runs as root ON a cpuarm instance via
# spinup.sh. Steps 1-2 and the dev image are infra/eval/arm64_aws.sh's own recipe (same Dockerfiles, same pin); then
# only the slam library + test_opencv are built, and infra/eval/gpgo_diag.cpp (input digests, a per-call LM trace,
# a scalar re-implementation of the LM driver, the driver's cv::Mat ops against scalar loops, a seed sweep) runs
# with Eigen vectorized and with -DEIGEN_DONT_VECTORIZE.
#   COMMIT=<sha> DIRTY=true|false OPENCV_TAG=5.0.0 bash infra/eval/gpgo_arm64.sh
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${COMMIT:?}" "${DIRTY:?}" "${OPENCV_TAG:?}"
SRC="$(pwd)"
JOBS="$(nproc)"
BASE_IMG="ariadne/opencv-base:${OPENCV_TAG}-arm64"
DEV_IMG="ariadne/core-dev:arm64"
export HOME="${HOME:-/root}" DOCKER_BUILDKIT=1
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
step "0 host $(uname -m) commit $COMMIT dirty $DIRTY"
[ "$(uname -m)" = aarch64 ] || { step "FAILED: not arm64"; exit 1; }
lscpu > "$R/lscpu.txt"

step "1 docker + buildx"
apt-get -o DPkg::Lock::Timeout=600 update -qq > "$R/apt.txt" 2>&1
apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io docker-buildx >> "$R/apt.txt" 2>&1 || { step "FAILED: apt"; exit 1; }
systemctl start docker 2>/dev/null || true

step "2 opencv-base"
timeout 4500 docker build --platform linux/arm64 --progress=plain -f infra/docker/opencv-base.Dockerfile \
  --build-arg OPENCV_TAG="$OPENCV_TAG" --build-arg BUILD_JOBS="$JOBS" -t "$BASE_IMG" infra/docker \
  > "$R/opencv_base_build.log" 2>&1 || { step "FAILED: opencv-base"; tail -40 "$R/opencv_base_build.log"; exit 1; }

step "3 core dev image"
timeout 900 docker build --platform linux/arm64 --progress=plain -f infra/docker/core.Dockerfile --target dev \
  --build-arg OPENCV_BASE="$BASE_IMG" -t "$DEV_IMG" . > "$R/core_dev_build.log" 2>&1 \
  || { step "FAILED: core dev"; tail -40 "$R/core_dev_build.log"; exit 1; }
chown -R 1000:1000 "$SRC" "$R"               # the dev image runs as uid 1000 and writes /out = $R

step "4 build test_opencv + diag, run"
timeout 2400 docker run --rm -v "$SRC":/work -v "$R":/out "$DEV_IMG" bash -c "
  set -o pipefail
  mkdir -p /work/build/arm64 /work/.cache
  cmake -S /work -B /work/build/arm64 -G Ninja -DCMAKE_BUILD_TYPE=Release -DARIADNE_REQUIRE_OPENCV=ON \
    -DCMAKE_SKIP_RPATH=ON -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-arm64 > /out/configure.log 2>&1 || exit 2
  cmake --build /work/build/arm64 --target test_opencv --parallel $JOBS > /out/compile.log 2>&1 \
    || { grep -m40 -B2 -A8 error /out/compile.log; exit 3; }
  gcc --version | head -1 > /out/gcc_version.txt
  grep -E 'FLAGS = ' /work/build/arm64/build.ninja | sort | uniq -c | sort -rn | head -5 > /out/compile_flags.txt
  E=/work/.cache/fetchcontent-arm64/eigen-src
  for v in vec novec; do
    X=''; [ \$v = novec ] && X='-DEIGEN_DONT_VECTORIZE'
    g++ -O3 -DNDEBUG -std=c++17 -ffp-contract=off \$X -DEIGEN_MPL2_ONLY -I/work/cpp -I/work/build/arm64/cpp/core/generated \
      -isystem \$E -isystem /opt/opencv5/include/opencv5 /work/infra/eval/gpgo_diag.cpp -o /work/build/gpgo_diag_\$v \
      /work/build/arm64/cpp/slam/libariadne_slam_opencv.a /opt/opencv5/lib/libopencv_geometry.so /opt/opencv5/lib/libopencv_core.so \
      > /out/diag_build_\$v.log 2>&1 || { echo diag build \$v failed; cat /out/diag_build_\$v.log | head -30; }
  done
  export LD_LIBRARY_PATH=/opt/opencv5/lib
  D=/work/build/gpgo_diag_vec; N=/work/build/gpgo_diag_novec
  \$D trace > /out/trace_vec.txt 2>&1; \$N trace > /out/trace_novec.txt 2>&1
  \$D pure > /out/pure_vec.txt 2>&1; \$N pure > /out/pure_novec.txt 2>&1
  \$D halcheck > /out/halcheck.txt 2>&1
  \$D tight > /out/tight.txt 2>&1
  \$D perturb 8 1e-12 > /out/perturb.txt 2>&1
  \$D seeds 100 > /out/seeds.txt 2>&1
  /work/build/arm64/tests/test_opencv --gtest_filter='GravityPgo.*:PgoLevMarq.*:JointRobust.SharedGravity*:Gnc.NearMisses*' \
    > /out/gtest_targeted.txt 2>&1
  echo done
" > "$R/diag_run.log" 2>&1
rc=$?
step "done rc=$rc"
exit $rc
