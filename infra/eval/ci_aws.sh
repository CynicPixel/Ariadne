#!/bin/bash
# The generic "build + selected tests" job on one fresh AWS instance, as root, launched by
# infra/eval/ci_aws_launch.sh run NAME through spinup.sh:
#   infra/scripts/spinup.sh run --profile cpux86 --max-minutes 90 -- \
#     'STAGE=s3://<bucket>/runs/_staging/ci/<NAME>-<sha12> COMMIT=<sha> REG=<ecr registry> NAME=<NAME> \
#      TESTS_B64=<base64> EXTRA_B64=<base64> JOBS=16 CTEST_JOBS=8 bash infra/eval/ci_aws.sh'
# Inputs, all staged under $STAGE by the launching host (as for the pre-merge, premerge_aws.sh):
#   repo.bundle   git bundle of COMMIT: the code under test (this script comes from spinup's working-tree copy)
#   inputs.tar    the ignored inputs the tests read (INPUTS on the host; may be none), with inputs.SHA256SUMS and
#                 inputs.list. Each model export present sets its ARIADNE_*_REF variable; ARIADNE_REQUIRE_MODELS=1
#                 only when all six are present (with a subset it would turn the others' SKIPs into FAILs).
#   images.txt    the dev image: pulled, retagged ariadne/core-dev:dev, its id recorded next to the staged one.
#                 core.sh is told not to rebuild it (ARIADNE_SKIP_DEV_BUILD).
# TESTS (ctest -R regex; empty = all; "none" = no ctest) and EXTRA (shell, run in the dev image from /work after
# the build) arrive base64-encoded and are written to files the container reads: no quoting survives
# spinup -> bash -> core.sh -> docker -> bash otherwise. EXTRA sees the same environment as ctest, plus
# CI_OUT=/work/build/ci/out: whatever it writes there comes back as extra_out/.
# Build and tests run through infra/scripts/core.sh dev, the workstation's path, from the BUNDLE checkout (core.sh
# finds its repo from its own location). JOBS=16 builds on all 16 vCPU; ctest at 8, the parallelism the pre-merge
# proved on this instance type (some tests load ONNX models).
# Each step is logged to $ARIADNE_RESULTS/steps.txt. Every exit path collects: manifest.yaml, the build log (tail +
# gzip), ctest.log, ctest.junit.xml, extra.log, extra_out/. Exit code: the build's if it failed, else ctest's if
# it failed, else EXTRA's. A SKIP is a ctest pass: model_env / require_models in manifest.yaml are the coverage
# evidence; ctest_skipped_or_not_run is a lower bound [UNVERIFIED: it relies on gtest_discover_tests marking a
# GTEST_SKIP as Skipped in ctest's output].
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${STAGE:?}" "${COMMIT:?}" "${REG:?}"
NAME="${NAME:-ci}" JOBS="${JOBS:-16}" CTEST_JOBS="${CTEST_JOBS:-8}"
TESTS="$(printf '%s' "${TESTS_B64:-}" | base64 -d)" EXTRA="$(printf '%s' "${EXTRA_B64:-}" | base64 -d)"
REPO=/opt/ariadne/repo
C=$REPO/build/ci            # /work/build/ci in the container; build/ is gitignored, so the tree stays clean
CORE=$REPO/infra/scripts/core.sh
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
T0=$(date +%s) BUILD_RC=not_run CTEST_RC=not_run EXTRA_RC=not_run MODEL_ENV="" REQUIRE_MODELS=0 IMAGE_ID=""
q() { python3 -c 'import json, sys; print(json.dumps(sys.argv[1]))' "$1"; }   # a JSON string is a YAML scalar

finish() {   # $1 = exit code; collects whatever exists, on every path
  step "collect"
  [ -f "$C/build.log" ] && { tail -200 "$C/build.log" > "$R/build_tail.txt"; gzip -c "$C/build.log" > "$R/build.log.gz"; }
  cp "$C/ctest.log" "$C/ctest.junit.xml" "$C/extra.log" "$R/" 2>/dev/null
  [ -d "$C/out" ] && [ -n "$(ls -A "$C/out")" ] && cp -r "$C/out" "$R/extra_out"
  [ -f "$C/ctest.log" ] && grep -E "tests passed|tests failed|No tests were found" "$C/ctest.log" > "$R/ctest_summary.txt"
  local skipped=0; [ -f "$C/ctest.log" ] && skipped=$(grep -c "\*\*\*Skipped\|Not Run" "$C/ctest.log")
  {
    echo "name: $(q "$NAME")"
    echo "commit: $(q "$COMMIT")"
    echo "head: $(q "$(git -C "$REPO" log -1 --format='%H %s' 2>/dev/null)")"
    echo "stage: $STAGE"
    echo "image: $(q "$(head -1 "$R/image_ids.txt" 2>/dev/null)")"
    echo "image_id: $(q "$IMAGE_ID")"
    echo "tests: $(q "${TESTS:-<all>}")"
    echo "extra: $(q "$EXTRA")"
    echo "inputs: [$(paste -sd, "$R/inputs.list" 2>/dev/null)]"
    echo "model_env: [${MODEL_ENV// /, }]"
    echo "require_models: $REQUIRE_MODELS"
    echo "jobs: $JOBS"
    echo "ctest_jobs: $CTEST_JOBS"
    echo "host: $(q "$(nproc) vCPU, $(lscpu | sed -n 's/^Model name: *//p')")"
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
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"
printf 'name %s\ncommit %s\nstage %s\njobs %s\nctest_jobs %s\n' "$NAME" "$COMMIT" "$STAGE" "$JOBS" "$CTEST_JOBS" \
  > "$R/run_params.txt"

step "1 docker + image"
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io > "$R/apt.txt" 2>&1; }
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || fail "ecr login"
aws s3 cp "$STAGE/images.txt" "$R/images.txt" --only-show-errors || fail "images.txt"
while read -r ref name want; do
  docker pull -q "$ref" > /dev/null || fail "pull $ref"
  docker tag "$ref" "$name"
  got="$(docker image inspect "$name" --format '{{.Id}}')"
  echo "$name $ref $got (local $want)" >> "$R/image_ids.txt"
  IMAGE_ID=$got
done < "$R/images.txt"

step "2 checkout"
aws s3 cp "$STAGE/repo.bundle" /opt/ariadne/repo.bundle --only-show-errors || fail "bundle"
git clone -q /opt/ariadne/repo.bundle "$REPO" && git -C "$REPO" checkout -q "$COMMIT" || fail "checkout $COMMIT"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$(git -C "$REPO" rev-parse "$COMMIT")" ] || fail "HEAD is not $COMMIT"
git -C "$REPO" log -1 --format='%H %s' > "$R/head.txt"

step "3 inputs"
aws s3 cp "$STAGE/inputs.tar" - | tar x -C "$REPO" || fail "inputs.tar"
if [ -s "$REPO/inputs.SHA256SUMS" ]; then
  (cd "$REPO" && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
fi
cp "$REPO/inputs.list" "$R/inputs.list"
rm -f "$REPO/inputs.SHA256SUMS" "$REPO/inputs.list"
# The six inputs the tests read through the environment (phase1.sh's list). Only the ones staged are set.
for kv in xfeat_ref:XFEAT_REF eigenplaces_ref:EIGENPLACES_REF aliked_lg_ref:ALIKED_LG_REF \
          eigenplaces_sim_ref:EIGENPLACES_SIM_REF aliked_lg_sim_ref:ALIKED_LG_SIM_REF; do
  [ -d "$REPO/.cache/models/${kv%%:*}" ] && MODEL_ENV+="ARIADNE_${kv#*:}=/work/.cache/models/${kv%%:*} "
done
[ -d "$REPO/.cache/oracle/pgo" ] && MODEL_ENV+="ARIADNE_PGO_ORACLE=/work/.cache/oracle/pgo "
MODEL_ENV="${MODEL_ENV% }"
[ "$(wc -w <<< "$MODEL_ENV")" -eq 6 ] && REQUIRE_MODELS=1
df -h / > "$R/df.txt"

# What the container reads: values through printf %q (one quoting layer, done once), commands as fixed text.
mkdir -p "$C/out"
{
  for e in $MODEL_ENV; do echo "export $e"; done
  [ "$REQUIRE_MODELS" = 1 ] && echo "export ARIADNE_REQUIRE_MODELS=1"
  printf 'export ARIADNE_GIT_COMMIT=%q ARIADNE_GIT_DIRTY=false ARIADNE_SOURCE_DIR=/work CI_OUT=/work/build/ci/out\n' \
    "$COMMIT"
  printf 'JOBS=%q CTEST_JOBS=%q\n' "$JOBS" "$CTEST_JOBS"
} > "$C/env.sh"
printf '%s' "$TESTS" > "$C/tests.regex"
printf '%s\n' "$EXTRA" > "$C/extra.sh"
cat > "$C/build.sh" <<'EOF'
. /work/build/ci/env.sh
cmake -S /work -B /work/build/docker -G Ninja -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON \
  -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-docker && cmake --build /work/build/docker --parallel "$JOBS"
EOF
# --no-tests=error: a mistyped regex fails instead of passing with "No tests were found".
# --output-junit takes an absolute path (a relative one lands under the test dir).
cat > "$C/ctest.sh" <<'EOF'
. /work/build/ci/env.sh
args=(); [ -s /work/build/ci/tests.regex ] && args=(-R "$(cat /work/build/ci/tests.regex)")
ctest --test-dir /work/build/docker --output-on-failure --no-tests=error --parallel "$CTEST_JOBS" \
  --output-junit /work/build/ci/ctest.junit.xml "${args[@]}"
EOF
# The dev image runs as its own non-root user: the checkout must be its.
uid="$(docker run --rm ariadne/core-dev:dev id -u)"; gid="$(docker run --rm ariadne/core-dev:dev id -g)"
chown -R "$uid:$gid" "$REPO"
# user-data runs as root without $HOME, so --global never applies: system-wide, and give git a HOME anyway.
export HOME="${HOME:-/root}"
git config --system --add safe.directory "$REPO"
git -C "$REPO" status --porcelain > "$R/git_status.txt"
dev() { ARIADNE_SKIP_DEV_BUILD=1 ARIADNE_BUILD_JOBS="$JOBS" "$CORE" dev "$@"; }

step "4 build (JOBS=$JOBS)"
t=$(date +%s)
dev bash /work/build/ci/build.sh > "$C/build.log" 2>&1
BUILD_RC=$?
echo "build_exit $BUILD_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
[ "$BUILD_RC" = 0 ] || { tail -30 "$C/build.log"; step "FAILED: build (build_tail.txt)"; finish "$BUILD_RC"; }

if [ "$TESTS" = none ]; then
  step "5 ctest skipped (TESTS=none)"
else
  step "5 ctest -R '${TESTS:-<all>}' (CTEST_JOBS=$CTEST_JOBS, require_models=$REQUIRE_MODELS)"
  t=$(date +%s)
  dev bash /work/build/ci/ctest.sh > "$C/ctest.log" 2>&1
  CTEST_RC=$?
  echo "ctest_exit $CTEST_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
  tail -15 "$C/ctest.log"
fi

if [ -n "$EXTRA" ]; then
  step "6 extra"
  t=$(date +%s)
  dev bash -c 'cd /work && . /work/build/ci/env.sh && bash /work/build/ci/extra.sh' > "$C/extra.log" 2>&1
  EXTRA_RC=$?
  echo "extra_exit $EXTRA_RC seconds $(( $(date +%s) - t ))" >> "$R/steps.txt"
  tail -15 "$C/extra.log"
fi

rc=0
if [ "$CTEST_RC" != not_run ] && [ "$CTEST_RC" != 0 ]; then rc=$CTEST_RC
elif [ "$EXTRA_RC" != not_run ] && [ "$EXTRA_RC" != 0 ]; then rc=$EXTRA_RC; fi
finish "$rc"
