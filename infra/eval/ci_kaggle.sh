#!/bin/bash
# The Kaggle CPU twin of infra/eval/ci_aws.sh: the same "build + selected tests (+ EXTRA)" job, run inside a Kaggle
# batch kernel instead of an AWS spot instance. Launched by infra/eval/ci_kaggle_launch.sh run NAME, which embeds this
# file (from the working tree, as spinup ships ci_aws.sh) in the kernel's kernel.py; never run by hand.
#
# DEVELOPMENT EVIDENCE ONLY (platform kaggle-cpu): phase-gate evidence, COOL Arm-vs-Arm and judge-box timing stay on
# AWS. The Kaggle box is a shared 4-vCPU Xeon @2.20GHz with AVX2 and no AVX-512, so OpenCV dispatches its AVX2 kernels
# where c7i takes AVX-512: floats can differ in the last bits. cpu_flags in the manifest records which path ran.
#
# Kaggle has no docker, but its image is Ubuntu 24.04 / glibc 2.39 / gcc 13 like ariadne/opencv-base, so the job
# recreates the dev stage natively instead (probe 2026-10-24, build/offaws/kaggle/probe/out/probe.json):
#   /opt/opencv5            unpacked from the opencv dataset (a tar of ariadne/opencv-base:5.0.0's /opt/opencv5),
#                           resolved through ld.so.conf, as the dev stage does (no RPATH, CMAKE_SKIP_RPATH=ON)
#   /work                   the repo at COMMIT (git archive from the run dataset) plus inputs.tar.keep, as the bind mount
#   user ariadne (uid 1000) builds and tests, as in the dev image: root would pass permission-denied tests
#   PATH / LD_LIBRARY_PATH  reset to the system's: Kaggle prepends conda/CUDA dirs and its own python 3.13 + cmake
# Inputs (env, set by kernel.py): NAME COMMIT TESTS EXTRA JOBS CTEST_JOBS; datasets mounted under /kaggle/input.
# Results: /kaggle/working/results/ (manifest.yaml, ctest.log, ctest.junit.xml, build_tail.txt, build.log.gz,
# extra.log, extra_out/, host facts); the launcher pulls them to build/runs-offaws/<id>/.
set -uo pipefail
R=/kaggle/working/results
mkdir -p "$R"
NAME="${NAME:-ci}" JOBS="${JOBS:-4}" CTEST_JOBS="${CTEST_JOBS:-4}" TESTS="${TESTS:-}" EXTRA="${EXTRA:-}"
: "${COMMIT:?}"
C=/work/build/ci
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
T0=$(date +%s) BUILD_RC=not_run CTEST_RC=not_run EXTRA_RC=not_run MODEL_ENV="" REQUIRE_MODELS=0 OCV_HASH=""
q() { python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"; }   # a JSON string is a YAML scalar
find_one() { find -L /kaggle/input -name "$1" -type f 2>/dev/null | head -1; }   # mount depth varies (render_kernel)

finish() {   # $1 = exit code; collects whatever exists, on every path
  step "collect"
  [ -f "$C/build.log" ] && { tail -200 "$C/build.log" > "$R/build_tail.txt"; gzip -c "$C/build.log" > "$R/build.log.gz"; }
  cp "$C/ctest.log" "$C/ctest.junit.xml" "$C/extra.log" "$R/" 2>/dev/null
  [ -d "$C/out" ] && [ -n "$(ls -A "$C/out")" ] && cp -r "$C/out" "$R/extra_out"
  [ -f "$C/ctest.log" ] && grep -E "tests passed|tests failed|No tests were found" "$C/ctest.log" > "$R/ctest_summary.txt"
  local skipped=0; [ -f "$C/ctest.log" ] && skipped=$(grep -c "\*\*\*Skipped\|Not Run" "$C/ctest.log")
  {
    echo "name: $(q "$NAME")"
    echo "platform: kaggle-cpu"
    echo "evidence: development (off-AWS; not phase-gate, COOL or judge-box timing evidence)"
    echo "commit: $(q "$COMMIT")"
    echo "kaggle_image: $(q "${KAGGLE_DOCKER_IMAGE:-}")"
    echo "kaggle_zone: $(q "${KAGGLE_GCP_ZONE:-}")"
    echo "opencv_dataset: $(q "$(cat /opt/opencv5/kaggle_provenance.txt 2>/dev/null | paste -sd' ')")"
    echo "opencv_build_hash: $(q "$OCV_HASH")"
    echo "tests: $(q "${TESTS:-<all>}")"
    echo "extra: $(q "$EXTRA")"
    echo "inputs: [$(paste -sd, "$R/inputs.list" 2>/dev/null)]"
    echo "model_env: [${MODEL_ENV// /, }]"
    echo "require_models: $REQUIRE_MODELS"
    echo "jobs: $JOBS"
    echo "ctest_jobs: $CTEST_JOBS"
    echo "host: $(q "$(nproc) vCPU, $(lscpu | sed -n 's/^Model name: *//p'), $(free -g | awk '/Mem/{print $2}') GiB")"
    echo "cpu_flags: $(q "$(grep -m1 -o -w -E 'avx2|avx512f|avx512bw|avx512_vnni|amx_tile' /proc/cpuinfo | sort -u | paste -sd' ')")"
    echo "build_rc: $BUILD_RC"
    echo "ctest_rc: $CTEST_RC"
    echo "ctest_summary: $(q "$(head -1 "$R/ctest_summary.txt" 2>/dev/null)")"
    echo "ctest_skipped_or_not_run: $skipped"
    echo "extra_rc: $EXTRA_RC"
    echo "exit: $1"
    echo "seconds: $(( $(date +%s) - T0 ))"
  } > "$R/manifest.yaml"
  cat "$R/manifest.yaml"
  step "done ($NAME exit $1)"
  exit "$1"
}
fail() { step "FAILED: $*"; finish 1; }

step "0 host"
lscpu > "$R/lscpu.txt"; free -m > "$R/free.txt"; cat /etc/os-release > "$R/os_release.txt"
printf 'kaggle LD_LIBRARY_PATH=%s\nkaggle PATH=%s\n' "${LD_LIBRARY_PATH:-}" "$PATH" > "$R/kaggle_env_paths.txt"
# The system's paths only, /usr/bin before /usr/local/bin: the dev image's cmake 3.28, ninja and python 3.12 are apt's.
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset LD_LIBRARY_PATH PYTHONPATH PYTHONHOME CONDA_PREFIX

step "1 apt (opencv-base's build list + ccache binutils of the dev stage)"
apt-get update -qq > "$R/apt.txt" 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  build-essential g++ cmake ninja-build git ca-certificates pkg-config \
  libzstd-dev zlib1g-dev libpng-dev libjpeg-turbo8-dev libtiff-dev python3 python3-dev ccache binutils \
  >> "$R/apt.txt" 2>&1 || fail "apt (apt.txt)"
# Kaggle replaced /usr/bin/python3 with 3.13; apt's 3.12 (the image's) stays at /usr/bin/python3.12: a shim first on
# PATH makes `python3` in EXTRA mean it, as in the dev image. ctest itself runs no python.
mkdir -p /opt/pyshim && [ -x /usr/bin/python3.12 ] && ln -sf /usr/bin/python3.12 /opt/pyshim/python3 \
  && ln -sf /usr/bin/python3.12 /opt/pyshim/python
export PATH=/opt/pyshim:$PATH
{ cmake --version | head -1; ninja --version; gcc --version | head -1; python3 --version; } > "$R/toolchain.txt" 2>&1

step "2 opencv"
ocv="$(find_one opencv5.tar.keep)"; [ -n "$ocv" ] || fail "no opencv5.tar.keep under /kaggle/input"
tar xf "$ocv" -C /opt || fail "untar $ocv"
echo /opt/opencv5/lib > /etc/ld.so.conf.d/opencv5.conf && ldconfig
missing="$(ldd /opt/opencv5/lib/*.so 2>&1 | grep 'not found' | sort -u)"
[ -z "$missing" ] || { echo "$missing" > "$R/ldd_missing.txt"; fail "opencv libs miss deps (ldd_missing.txt)"; }
# Kaggle's /usr/local/lib is in the loader cache: a dependency resolving there is not the image's (recorded, not fatal).
ldd /opt/opencv5/lib/*.so 2>/dev/null | awk '/=>/{print $1, $3}' | sort -u > "$R/opencv_ldd.txt"
grep -E ' /usr/local/| /opt/conda' "$R/opencv_ldd.txt" > "$R/opencv_ldd_nonsystem.txt" || true

step "3 checkout + inputs"
repo="$(find_one repo.tar.keep)"; [ -n "$repo" ] || fail "no repo.tar.keep under /kaggle/input"
# uid 1000 is taken in Kaggle's image: -o adds `ariadne` as a second name for it (files stay uid 1000, as in the image).
id -u ariadne >/dev/null 2>&1 || useradd -m -o -u 1000 -s /bin/bash ariadne || fail "useradd"
mkdir -p /work && tar xf "$repo" -C /work || fail "untar repo"
[ "$(cat /work/.kaggle_commit 2>/dev/null)" = "$COMMIT" ] || fail "repo.tar.keep is not $COMMIT"
: > "$R/inputs.list"
inp="$(find_one inputs.tar.keep)"
if [ -n "$inp" ]; then
  tar xf "$inp" -C /work || fail "untar inputs"
  if [ -s /work/inputs.SHA256SUMS ]; then
    (cd /work && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
  fi
  cp /work/inputs.list "$R/inputs.list"; rm -f /work/inputs.SHA256SUMS /work/inputs.list
fi
# The six inputs the tests read through the environment (ci_aws.sh's list). Only the ones staged are set.
for kv in xfeat_ref:XFEAT_REF eigenplaces_ref:EIGENPLACES_REF aliked_lg_ref:ALIKED_LG_REF \
          eigenplaces_sim_ref:EIGENPLACES_SIM_REF aliked_lg_sim_ref:ALIKED_LG_SIM_REF; do
  [ -d "/work/.cache/models/${kv%%:*}" ] && MODEL_ENV+="ARIADNE_${kv#*:}=/work/.cache/models/${kv%%:*} "
done
[ -d /work/.cache/oracle/pgo ] && MODEL_ENV+="ARIADNE_PGO_ORACLE=/work/.cache/oracle/pgo "
MODEL_ENV="${MODEL_ENV% }"
[ "$(wc -w <<< "$MODEL_ENV")" -eq 6 ] && REQUIRE_MODELS=1
df -h / /kaggle/working > "$R/df.txt"

mkdir -p "$C/out"
{
  for e in $MODEL_ENV; do echo "export $e"; done
  [ "$REQUIRE_MODELS" = 1 ] && echo "export ARIADNE_REQUIRE_MODELS=1"
  printf 'export ARIADNE_GIT_COMMIT=%q ARIADNE_GIT_DIRTY=false ARIADNE_SOURCE_DIR=/work CI_OUT=/work/build/ci/out\n' \
    "$COMMIT"
  printf 'export PATH=/opt/pyshim:/usr/sbin:/usr/bin:/sbin:/bin HOME=/home/ariadne CCACHE_DIR=/home/ariadne/.ccache\n'
  printf 'export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache\n'
  printf 'JOBS=%q CTEST_JOBS=%q\n' "$JOBS" "$CTEST_JOBS"
} > "$C/env.sh"
printf '%s' "$TESTS" > "$C/tests.regex"
printf '%s\n' "$EXTRA" > "$C/extra.sh"
# build.sh and ctest.sh: ci_aws.sh's, verbatim (same flags, same build dir), so a result reads the same on both.
cat > "$C/build.sh" <<'EOF'
. /work/build/ci/env.sh
cmake -S /work -B /work/build/docker -G Ninja -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON \
  -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-docker && cmake --build /work/build/docker --parallel "$JOBS"
EOF
cat > "$C/ctest.sh" <<'EOF'
. /work/build/ci/env.sh
args=(); [ -s /work/build/ci/tests.regex ] && args=(-R "$(cat /work/build/ci/tests.regex)")
ctest --test-dir /work/build/docker --output-on-failure --no-tests=error --parallel "$CTEST_JOBS" \
  --output-junit /work/build/ci/ctest.junit.xml "${args[@]}"
EOF
chown -R ariadne:ariadne /work
as_dev() { runuser -u ariadne -- env -i bash -c "cd /work && $1"; }   # env -i: nothing of Kaggle's leaks in

step "4 build (JOBS=$JOBS)"
t=$(date +%s)
as_dev 'bash /work/build/ci/build.sh' > "$C/build.log" 2>&1
BUILD_RC=$?
echo "build_exit $BUILD_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
grep -m3 -E 'Found OpenCV|OpenCV_DIR|OpenCV [0-9]' "$C/build.log" > "$R/opencv_found.txt" 2>/dev/null
[ "$BUILD_RC" = 0 ] || { tail -30 "$C/build.log"; step "FAILED: build (build_tail.txt)"; finish "$BUILD_RC"; }
# The build hash a run's manifest records (Phase 0 gate 4), from the binary just built against /opt/opencv5.
idb="$(find /work/build/docker -name ariadne_identity -type f -perm -u+x | head -1)"
[ -n "$idb" ] && OCV_HASH="$(as_dev "$idb" 2>/dev/null | sed -n 's/^OPENCV_BUILD_HASH=//p' | head -1)"

if [ "$TESTS" = none ]; then
  step "5 ctest skipped (TESTS=none)"
else
  step "5 ctest -R '${TESTS:-<all>}' (CTEST_JOBS=$CTEST_JOBS, require_models=$REQUIRE_MODELS)"
  t=$(date +%s)
  as_dev 'bash /work/build/ci/ctest.sh' > "$C/ctest.log" 2>&1
  CTEST_RC=$?
  echo "ctest_exit $CTEST_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
  tail -15 "$C/ctest.log"
fi

if [ -n "$EXTRA" ]; then
  step "6 extra"
  t=$(date +%s)
  as_dev '. /work/build/ci/env.sh && bash /work/build/ci/extra.sh' > "$C/extra.log" 2>&1
  EXTRA_RC=$?
  echo "extra_exit $EXTRA_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
  tail -15 "$C/extra.log"
fi

rc=0
if [ "$CTEST_RC" != not_run ] && [ "$CTEST_RC" != 0 ]; then rc=$CTEST_RC
elif [ "$EXTRA_RC" != not_run ] && [ "$EXTRA_RC" != 0 ]; then rc=$EXTRA_RC; fi
finish "$rc"
