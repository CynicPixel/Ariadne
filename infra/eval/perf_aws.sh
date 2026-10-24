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
#   A adaptive (after knobs): upstream LightGlue (torch) all9 vs depth/width-adaptive on the same pairs' features, every
#     match list through the D71 gate (scripts/lg_adaptive.py + ariadne_perf_bench gate): exit layer, pruning, flips
# Results: $ARIADNE_RESULTS/{steps.txt,run_manifest.yaml,lscpu.txt,timing/,knobs/,ort/,mpstat/}
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${COMMIT:?}" "${STAGE:?}"
STAGES="${STAGES:-timing knobs}" SINGLE_THREADS="${SINGLE_THREADS-1 2 4 8 16}" C3_THREADS="${C3_THREADS-2 4 5}"
ORT_RUNS="${ORT_RUNS-yes}" OPENCV_TAG="${OPENCV_TAG:-5.0.0}" ORT_VERSION=1.24.4
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
  kv stages "\"$STAGES\""; kv single_threads "\"$SINGLE_THREADS\""; kv c3_threads "\"$C3_THREADS\""; kv arch "$ARCH"; kv instance_type "$(imds instance-type)"
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
# Every fan-out waits on ITS OWN pids: a bare `wait` also waits on the mpstat sampler, which never exits (runs 42/43,
# 20261024T001208Z-cpux86-6ba75f / 20261024T003403Z-cpuarm-46e7b8, hung at the first concurrent step).
PIDS=(); spawn() { "$@" & PIDS+=($!); }; join() { wait "${PIDS[@]}"; PIDS=(); }

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
  for t in $SINGLE_THREADS; do
    step "T single process, $t threads"
    mpstat_on "single_t$t"; bench "single_t$t" "$t" $([ "$t" = 1 ] && echo 10 || echo 20); mpstat_off
  done
  for t in $C3_THREADS; do
    step "T three concurrent processes x $t threads"
    mpstat_on "c3_t$t"
    for p in 0 1 2; do spawn bench "c3_t${t}_p$p" "$t" 12; done; join
    mpstat_off
  done
  [ "$ORT_RUNS" = yes ] && {
  step "T ORT $ORT_VERSION"
  python3 -m venv /opt/ortvenv && /opt/ortvenv/bin/pip install -q "onnxruntime==$ORT_VERSION" > "$R/ort/pip.log" 2>&1 \
    && /opt/ortvenv/bin/pip freeze > "$R/ort/freeze.txt"
  ORTB() { /opt/ortvenv/bin/python -I "$SRC/evaluation/runs/perf/pipeline_profile_20261024/scripts/ort_bench.py" \
           --models-dir "$SRC/.cache/models" --threads "$1" --reps "$2" --label "$3" --out "$R/ort/$3.csv" >> "$R/ort/ort.log" 2>&1; }
  if [ -x /opt/ortvenv/bin/python ]; then
    for t in 1 4 16; do mpstat_on "ort_t$t"; ORTB "$t" 10 "ort_single_t$t"; mpstat_off; done
    mpstat_on ort_c3_t5; for p in 0 1 2; do spawn ORTB 5 10 "ort_c3_t5_p$p"; done; join; mpstat_off
  else
    echo "ORT venv/pip failed (ort/pip.log)" | tee -a "$R/steps.txt"
  fi
  }
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
    spawn dev -e OPENCV_FOR_THREADS_NUM=4 "$DEV" $BIN/ariadne_perf_kfdump --rec "/work/$rec/rec" --scene "/work/$rec/scene" \
      --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v2.yaml \
      --imu-noise configs/sim/imu_noise_v1.yaml --agent $ag --seed 1 --ids "$(cat "$SRC/$PI/ids_s${sc}_a${ag}.txt")" \
      --out "/w/kf/s$sc" > "$W/kfdump_s${sc}_a$ag.log" 2>&1
  done; done; join
  mpstat_off
  for f in "$W"/kfdump_*.log; do tail -2 "$f" > "$R/knobs/$(basename "$f")"; done
  cp "$W"/kf/s*/a*_kfdump.yaml "$W"/kf/s*/a*_keyframes.csv "$R/knobs/" 2>/dev/null
  for sc in 0 1; do mkdir -p "$R/knobs/s$sc"; cp "$W"/kf/s$sc/a*_kfdump.yaml "$W"/kf/s$sc/a*_keyframes.csv "$R/knobs/s$sc/" 2>/dev/null; done
  step "K knobs (2 scenes x 4 threads) + store replay (2 threads)"
  mpstat_on knobs
  for sc in 0 1; do
    spawn dev -e OPENCV_FOR_THREADS_NUM=4 "$DEV" $BIN/ariadne_perf_bench knobs --kf-dir "/w/kf/s$sc" \
      --pairs "/work/$PI/pairs_s$sc.csv" --collab-config configs/collab/collab_v2.yaml --seed 1 $AL $LG \
      --settings 1280x720:1024,1280x720:512,1280x720:256,960x540:1024,960x540:512,640x360:1024,640x360:512 \
      --stores "/work/$PI/stores/s$sc/seed1" --dump-features "/w/feat/s$sc" --out "/w/knobs/s$sc" \
      > "$W/knobs_s$sc.log" 2>&1
  done; join
  for sc in 0 1; do
    spawn dev -e OPENCV_FOR_THREADS_NUM=2 "$DEV" $BIN/ariadne_perf_bench knobs --kf-dir "/w/kf/s$sc" \
      --pairs "/work/$PI/pairs_s$sc.csv" --collab-config configs/collab/collab_v2.yaml --seed 1 \
      --aliked "1280x720=$M/aliked_lg_sim_ref/aliked_n16_k1024_1280x720.onnx" $LG --settings 1280x720:1024 \
      --stores "/work/$PI/stores/s$sc/seed1" --out "/w/knobs_t2/s$sc" > "$W/knobs_t2_s$sc.log" 2>&1
  done; join
  mpstat_off
  for sc in 0 1; do
    mkdir -p "$R/knobs/s$sc" "$R/knobs/t2_s$sc"
    cp "$W"/knobs/s$sc/*.csv "$R/knobs/s$sc/" 2>/dev/null; cp "$W"/knobs_t2/s$sc/*.csv "$R/knobs/t2_s$sc/" 2>/dev/null
  done
  cp "$W"/knobs*.log "$R/knobs/" 2>/dev/null
fi
if [[ " $STAGES " == *" adaptive "* ]]; then
  # M (2026-10-24): size the LightGlue adaptive depth/width payoff before any design. Upstream LightGlue in torch (the
  # exporter's pins and loader) on the knobs run's dumped features: all9 (= our export, the control) and upstream's
  # defaults; then every match list through the fleet's VerifyMatched + D71 gate (bench gate).
  [ -d "$W/feat/s0" ] || fail "adaptive needs the knobs stage's --dump-features"
  step "A model-ref image + upstream LightGlue"
  LG_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8 AL_COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
  docker build -q -t ariadne-xfeat-ref:1 -f infra/docker/xfeat_ref.Dockerfile infra/docker > "$R/adaptive_image.log" 2>&1 \
    || fail "xfeat-ref image"
  docker build -q -t ariadne-model-ref:1 - >> "$R/adaptive_image.log" 2>&1 <<'DOCKERFILE' || fail "model-ref image"
FROM ariadne-xfeat-ref:1
RUN pip install --no-cache-dir --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu \
      torchvision==0.20.1+cpu \
 && pip install --no-cache-dir --root-user-action=ignore onnx==1.17.0 \
 && python -c "import torch, torchvision; assert torch.__version__ == '2.5.1+cpu', torch.__version__"
DOCKERFILE
  kv model_ref_image_id "$(docker image inspect ariadne-model-ref:1 --format '{{.Id}}')"
  UPL=/opt/ariadne/up; mkdir -p "$UPL/weights"
  git clone -q https://github.com/cvg/LightGlue.git "$UPL/lg" && git -C "$UPL/lg" checkout -q "$LG_COMMIT" || fail "LightGlue clone"
  curl -sfL -o "$UPL/weights/aliked_lightglue.pth" https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth
  echo "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503  $UPL/weights/aliked_lightglue.pth" \
    | sha256sum -c --quiet || fail "aliked_lightglue.pth sha256"
  chmod -R a+rX "$UPL"
  step "A upstream LightGlue all9 + adaptive (torch, 16 threads), then the D71 gate"
  mpstat_on adaptive
  for sc in 0 1; do
    docker run --rm --network none -u 1000:1000 -v "$SRC:/work:ro" -v "$W:/w" -v "$UPL:/up:ro" ariadne-model-ref:1 \
      python -I /work/evaluation/runs/perf/pipeline_profile_20261024/scripts/lg_adaptive.py --upstream /up/lg \
      --weights /up/weights --exporter /work/python/model_tools --feat-dir "/w/feat/s$sc" \
      --pairs "/work/$PI/pairs_s$sc.csv" --threads 16 --out "/w/adaptive_s$sc.csv" > "$W/adaptive_s$sc.log" 2>&1 \
      || { cp "$W/adaptive_s$sc.log" "$R/knobs/"; fail "lg_adaptive scene $sc"; }
    dev -e OPENCV_FOR_THREADS_NUM=4 "$DEV" $BIN/ariadne_perf_bench gate --feat-dir "/w/feat/s$sc" \
      --matches "/w/adaptive_s$sc.csv" --collab-config configs/collab/collab_v2.yaml --seed 1 \
      --out "/w/gate/s$sc" > "$W/gate_s$sc.log" 2>&1 || fail "gate scene $sc"
    mkdir -p "$R/knobs/adaptive_s$sc"
    cp "$W/gate/s$sc/gate.csv" "$W/adaptive_s$sc.log" "$W/gate_s$sc.log" "$R/knobs/adaptive_s$sc/"
    cut -d, -f1-10 "$W/adaptive_s$sc.csv" > "$R/knobs/adaptive_s$sc/adaptive_nomatches.csv"
  done
  mpstat_off
fi
kv total_seconds "$(( $(date +%s) - T0 ))"
step "done"
