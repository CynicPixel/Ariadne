#!/bin/bash
# ---------------------------------------------------------------------------
# The pipeline profile's AWS replay (A2, 2026-10-24; M's delegation, MEASUREMENT ONLY): the dnn matrix and the speed
# knobs' accuracy cost, on recorded data, on one instance. Runs as root ON THE INSTANCE, launched by
# infra/eval/perf_aws_launch.sh through spinup.sh (cpux86 = c7i.4xlarge, cpuarm = c8g.4xlarge):
#   COMMIT=<sha> STAGE=<s3 prefix> DATA0=<s3> DATA1=<s3> STORES_FROM=<s3 results of 0345a5> STAGES="timing knobs" \
#     bash infra/eval/perf_aws.sh
# The working tree arrives as spinup's snapshot; the inputs (models incl. the 960x540/640x360 exports, the pass-A
# stores of run 0345a5, the sampled pairs) come from $STAGE/inputs.tar, checked against inputs.SHA256SUMS; the two
# recordings (knobs only) from the persistent datasets, checked file by file.
#
# Both arches build the same images from source (opencv-base at the pinned tag, then core `dev`), so the x86-vs-Arm
# rows differ only in the machine. Steps (UTC + seconds in steps.txt):
#   1 docker + buildx, sysstat;  2 opencv-base + core dev;  3 inputs;  4 build ariadne_perf_bench/_kfdump (Release)
#   T timing: ONE process at OPENCV_FOR_THREADS_NUM = 1 2 4 8 16, then THREE concurrent processes (the N=3 judge-box
#     shape) at 2 4 5 threads each, every one with mpstat beside it; ORT (pinned Python onnxruntime) at 1 4 16 and
#     three concurrent at 5
#   K knobs (needs the recordings): ariadne_perf_kfdump for (scene 0|1, agent 0|1) at seed 1 and 4 OpenCV threads
#     (= pass A's OPENCV_FOR_THREADS_NUM, so the keyframes and their ALIKED are pass A's), then ariadne_perf_bench
#     knobs per scene (4 threads: fidelity vs the stores; every setting) and a store-only replay at 2 threads (= pass
#     B's count: how exactly the replay reproduces the production verdicts)
# Results: $ARIADNE_RESULTS/{steps.txt,run_manifest.yaml,lscpu.txt,timing/,knobs/,ort/,mpstat/}
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${COMMIT:?}" "${STAGE:?}"
STAGES="${STAGES:-timing knobs}" OPENCV_TAG="${OPENCV_TAG:-5.0.0}" ORT_VERSION=1.24.4
SRC="$(pwd)" W=/opt/ariadne/work
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }
kv() { printf '%s: %s\n' "$1" "$2" >> "$R/run_manifest.yaml"; }
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
ARCH=$(uname -m)
mkdir -p "$W" "$R"/{timing,knobs,ort,mpstat}
export HOME="${HOME:-/root}" DOCKER_BUILDKIT=1

step "0 host"
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"
{
  echo "# pipeline profile replay (infra/eval/perf_aws.sh). Times UTC."
  kv run_kind pipeline_profile_replay; kv created_utc "$(date -u +%FT%TZ)"; kv git_commit "$COMMIT"
  kv stages "\"$STAGES\""; kv arch "$ARCH"; kv instance_type "$(imds instance-type)"
  kv availability_zone "$(imds placement/availability-zone)"; kv nproc "$(nproc)"
  kv cpu_model "\"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\""
  kv threads_per_core "$(lscpu | sed -n 's/^Thread(s) per core: *//p')"
  kv opencv_tag "$OPENCV_TAG"; kv ort_version "$ORT_VERSION"
}

step "1 docker + buildx + sysstat"
apt-get -o DPkg::Lock::Timeout=600 update -qq > "$R/apt.txt" 2>&1
apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io docker-buildx zstd sysstat python3-venv >> "$R/apt.txt" 2>&1 \
  || fail "apt"
systemctl start docker 2>/dev/null || true
docker buildx version > /dev/null 2>&1 || fail "docker buildx"
mpstat_on() { S_TIME_FORMAT=ISO mpstat -P ALL 1 > "$R/mpstat/$1.txt" 2>&1 & MP=$!; }
mpstat_off() { kill -INT "$MP" 2>/dev/null; wait "$MP" 2>/dev/null; }

step "2 opencv-base + core dev ($ARCH, from source)"
BASE=ariadne/opencv-base:$OPENCV_TAG DEV=ariadne/core-dev:perf
docker build --progress=plain -f infra/docker/opencv-base.Dockerfile --build-arg OPENCV_TAG="$OPENCV_TAG" \
  --build-arg BUILD_JOBS="$(nproc)" -t "$BASE" infra/docker > "$R/opencv_base_build.log" 2>&1 || fail "opencv-base"
docker build --progress=plain -f infra/docker/core.Dockerfile --target dev --build-arg OPENCV_BASE="$BASE" -t "$DEV" . \
  > "$R/core_dev_build.log" 2>&1 || fail "core dev"
kv base_image_id "$(docker image inspect "$BASE" --format '{{.Id}}')"
kv dev_image_id "$(docker image inspect "$DEV" --format '{{.Id}}')"
kv ubuntu_base_repo_digest "$(docker image inspect ubuntu:24.04 --format '{{index .RepoDigests 0}}' 2>/dev/null)"
docker run --rm "$DEV" sh -c 'cat /opt/opencv5/opt-opencv-core.sha 2>/dev/null' > "$R/opencv_commit.txt" 2>&1

step "3 inputs"
aws s3 cp "$STAGE/inputs.tar" - | tar x -C "$SRC" || fail "inputs.tar"
(cd "$SRC" && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
chown -R 1000:1000 "$SRC" "$W"
dev() { docker run --rm -u 1000:1000 -v "$SRC:/work" -v "$W:/w" -w /work "$@"; }

step "4 build"
dev "$DEV" bash -c "cmake -S /work -B /work/build/perf -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON -DFETCHCONTENT_BASE_DIR=/work/build/fc > /w/configure.log 2>&1 && \
  cmake --build /work/build/perf --target ariadne_perf_bench ariadne_perf_kfdump --parallel $(nproc) > /w/compile.log 2>&1" \
  || { cp "$W"/configure.log "$W"/compile.log "$R/" 2>/dev/null; fail "build"; }
BIN=/work/build/perf/cpp/runtime
M=/work/.cache/models
AL="--aliked 1280x720=$M/aliked_lg_sim_ref/aliked_n16_k1024_1280x720.onnx \
    --aliked 960x540=$M/aliked_lg_sim_960x540/aliked_n16_k1024_960x540.onnx \
    --aliked 640x360=$M/aliked_lg_sim_640x360/aliked_n16_k1024_640x360.onnx"
LG="--lightglue $M/aliked_lg_sim_ref/lightglue_aliked.onnx"
PL="--place 640x360=$M/eigenplaces_sim_ref/eigenplaces_r50_256_640x360.onnx"
IMGS="--img-a $M/aliked_lg_sim_ref/input_sim_a0_f200.png --img-b $M/aliked_lg_sim_ref/input_sim_a0_f205.png"

if [[ " $STAGES " == *" timing "* ]]; then
  bench() {   # $1 = label, $2 = threads, $3 = reps
    dev -e OPENCV_FOR_THREADS_NUM="$2" "$DEV" $BIN/ariadne_perf_bench timing $IMGS $AL $LG $PL --reps "$3" \
      --warmup 3 --label "$1" --out "/w/timing/$1" > "$W/timing_$1.log" 2>&1
  }
  for t in 1 2 4 8 16; do
    step "T single process, $t threads"
    mpstat_on "single_t$t"; bench "single_t$t" "$t" $([ "$t" = 1 ] && echo 10 || echo 20); mpstat_off
  done
  for t in 2 4 5; do
    step "T three concurrent processes x $t threads"
    mpstat_on "c3_t$t"
    for p in 0 1 2; do bench "c3_t${t}_p$p" "$t" 12 & done; wait
    mpstat_off
  done
  step "T ORT $ORT_VERSION"
  python3 -m venv /opt/ortvenv && /opt/ortvenv/bin/pip install -q "onnxruntime==$ORT_VERSION" > "$R/ort/pip.log" 2>&1 \
    && /opt/ortvenv/bin/pip freeze > "$R/ort/freeze.txt"
  ORTB() { /opt/ortvenv/bin/python -I "$SRC/evaluation/runs/perf/pipeline_profile_20261024/scripts/ort_bench.py" \
           --models-dir "$SRC/.cache/models" --threads "$1" --reps "$2" --label "$3" --out "$R/ort/$3.csv" >> "$R/ort/ort.log" 2>&1; }
  if [ -x /opt/ortvenv/bin/python ]; then
    for t in 1 4 16; do mpstat_on "ort_t$t"; ORTB "$t" 10 "ort_single_t$t"; mpstat_off; done
    mpstat_on ort_c3_t5; for p in 0 1 2; do ORTB 5 10 "ort_c3_t5_p$p" & done; wait; mpstat_off
  else
    echo "ORT venv/pip failed (ort/pip.log)" | tee -a "$R/steps.txt"
  fi
  cp -r "$W/timing/." "$R/timing/"; cp "$W"/timing_*.log "$R/timing/" 2>/dev/null
fi

if [[ " $STAGES " == *" knobs "* ]]; then
  : "${DATA0:?}" "${DATA1:?}"
  step "K recordings"
  fetch() {   # $1 = S3 prefix, $2 = local dir, $3 = checksum list (GetObject only, as collab_aws.sh)
    mkdir -p "$2"
    (cd "$2" && sed -E 's/^[0-9a-f]{64}  \.\///' "$3" \
      | xargs -P 8 -I{} sh -c 'mkdir -p "$(dirname "{}")" && aws s3 cp "'"$1"'/{}" "{}" --only-show-errors') || return 1
    (cd "$2" && sha256sum -c --quiet "$3")
  }
  PI=evaluation/runs/perf/pipeline_profile_20261024/inputs
  REC0=$(cat "$SRC/$PI/rec0.txt") REC1=$(cat "$SRC/$PI/rec1.txt")
  fetch "$DATA0" "$SRC/$REC0" "$SRC/$PI/rec0.SHA256SUMS" > "$R/knobs/rec0_check.txt" 2>&1 || fail "recording 0"
  fetch "$DATA1" "$SRC/$REC1" "$SRC/$PI/rec1.SHA256SUMS" > "$R/knobs/rec1_check.txt" 2>&1 || fail "recording 1"
  # The pass-A stores (seed 1, both scenes) straight from run 0345a5's results: the production features (fidelity).
  fetch "${STORES_FROM:?}/passa" "$SRC/$PI/stores" "$SRC/$PI/stores_seed1.SHA256SUMS" > "$R/knobs/stores_check.txt" 2>&1 \
    || fail "pass-A stores"
  chown -R 1000:1000 "$SRC"
  step "K kfdump (4 jobs x 4 threads, seed 1)"
  mpstat_on kfdump
  for sc in 0 1; do for ag in 0 1; do
    rec=$([ $sc = 0 ] && echo "$REC0" || echo "$REC1")
    dev -e OPENCV_FOR_THREADS_NUM=4 "$DEV" $BIN/ariadne_perf_kfdump --rec "/work/$rec/rec" --scene "/work/$rec/scene" \
      --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v2.yaml \
      --imu-noise configs/sim/imu_noise_v1.yaml --agent $ag --seed 1 --ids "$(cat "$SRC/$PI/ids_s${sc}_a${ag}.txt")" \
      --out "/w/kf/s$sc" > "$W/kfdump_s${sc}_a$ag.log" 2>&1 &
  done; done; wait
  mpstat_off
  for f in "$W"/kfdump_*.log; do tail -2 "$f" > "$R/knobs/$(basename "$f")"; done
  cp "$W"/kf/s*/a*_kfdump.yaml "$W"/kf/s*/a*_keyframes.csv "$R/knobs/" 2>/dev/null
  for sc in 0 1; do mkdir -p "$R/knobs/s$sc"; cp "$W"/kf/s$sc/a*_kfdump.yaml "$W"/kf/s$sc/a*_keyframes.csv "$R/knobs/s$sc/" 2>/dev/null; done
  step "K knobs (2 scenes x 4 threads) + store replay (2 threads)"
  mpstat_on knobs
  for sc in 0 1; do
    dev -e OPENCV_FOR_THREADS_NUM=4 "$DEV" $BIN/ariadne_perf_bench knobs --kf-dir "/w/kf/s$sc" \
      --pairs "/work/$PI/pairs_s$sc.csv" --collab-config configs/collab/collab_v2.yaml --seed 1 $AL $LG \
      --settings 1280x720:1024,1280x720:512,1280x720:256,960x540:1024,960x540:512,640x360:1024,640x360:512 \
      --stores "/work/$PI/stores/s$sc/seed1" --out "/w/knobs/s$sc" > "$W/knobs_s$sc.log" 2>&1 &
  done; wait
  for sc in 0 1; do
    dev -e OPENCV_FOR_THREADS_NUM=2 "$DEV" $BIN/ariadne_perf_bench knobs --kf-dir "/w/kf/s$sc" \
      --pairs "/work/$PI/pairs_s$sc.csv" --collab-config configs/collab/collab_v2.yaml --seed 1 \
      --aliked "1280x720=$M/aliked_lg_sim_ref/aliked_n16_k1024_1280x720.onnx" $LG --settings 1280x720:1024 \
      --stores "/work/$PI/stores/s$sc/seed1" --out "/w/knobs_t2/s$sc" > "$W/knobs_t2_s$sc.log" 2>&1 &
  done; wait
  mpstat_off
  for sc in 0 1; do
    mkdir -p "$R/knobs/s$sc" "$R/knobs/t2_s$sc"
    cp "$W"/knobs/s$sc/*.csv "$R/knobs/s$sc/" 2>/dev/null; cp "$W"/knobs_t2/s$sc/*.csv "$R/knobs/t2_s$sc/" 2>/dev/null
  done
  cp "$W"/knobs*.log "$R/knobs/" 2>/dev/null
fi
kv total_seconds "$(( $(date +%s) - T0 ))"
step "done"
