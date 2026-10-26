#!/bin/bash
# ---------------------------------------------------------------------------
# Phase 6 splat training in-region on an AWS GPU (A2, 2026-10-24; M-approved freeze exception, <= $0.30). It also
# validates the judge-box image ariadne-recon-gpu on a real GPU, and closes HG2's "gsplat on AWS GPU" gap (the T4
# numbers so far are Kaggle's). Runs as root on the instance (spinup.sh --profile gpu: the NVIDIA-driver DLAMI), from
# this repo's snapshot:
#
#   IMAGE_TAR=s3://.../ariadne-recon-gpu-<id>.tar.zst IMAGE_ID=sha256:... SPOOLS=s3://.../spools.tar.gz \
#   SCENE_ID=<expected> [BUDGETS='{"smoke": [15, 30], "live": 15, "batch": 100}'] bash infra/eval/recon_gpu_aws.sh
#
# Steps:
#   0  host and GPU facts
#   1  the image from S3 (zstd), its id checked
#   2  B's spools.tar.gz: build/spools/agent<k>/ + build/live/scenes/maze_demo_0/; scene_id checked against every spool
#   3  the D196 micro-bench (python/reconstruction/kaggle/bench_kernel.py): gsplat ms/iteration, comparable to Kaggle
#   4  the training (spool_train_kernel.py), one budget after another: a smoke budget first, and the first failing
#      budget stops the rest
# A GPU utilisation log runs beside steps 3-4. Results ($ARIADNE_RESULTS): steps.txt, run_manifest.yaml,
# nvidia_smi.txt, gpu_util.csv, bench/remote.json, train/{remote.json, splat_<budget>.tar.gz}. The splat tarballs are
# left in S3 (SPINUP_FETCH_EXCLUDE='*.tar.gz' on the launching host).
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${IMAGE_TAR:?}" "${IMAGE_ID:?}"
# RENDER mode (SPLAT_TAR=s3://.../splat_<budget>.tar.gz): the demo-video renders of a trained splat instead of training
# (python/reconstruction/render/render_gsplat.py; RENDER_ARGS e.g. "--seconds 9 --fps 30"). Output: $R/render/.
# WORLDGEN mode (WORLDGEN_TAR=s3://.../data.tar.gz, maintainer 2026-10-27 for the film's beat 10): trains the generated
# maze's splat with the tarball's own train_worldgen_gsplat.py (python/video/tools), run byte-for-byte; WORLDGEN_STEPS
# (default 3000). Output: $R/worldgen/{trained_worldgen.splat, manifest.json, train.log, heldout/target PNGs}.
[ -n "${SPLAT_TAR:-}${WORLDGEN_TAR:-}" ] || : "${SPOOLS:?}" "${SCENE_ID:?}"
BUDGETS="${BUDGETS:-}"; [ -n "$BUDGETS" ] || BUDGETS='{"smoke": [15, 30], "live": 15, "batch": 100}'   # smoke first
SRC="$(pwd)" IN=/opt/in/data
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }
kv() { printf '%s: %s\n' "$1" "$2" >> "$R/run_manifest.yaml"; }
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
export HOME="${HOME:-/root}"

step "0 host + GPU"
kv run_kind recon_gpu_training; kv created_utc "$(date -u +%FT%TZ)"; kv instance_type "$(imds instance-type)"
nvidia-smi > "$R/nvidia_smi.txt" 2>&1 || fail "nvidia-smi"
kv gpu "\"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)\""
command -v docker >/dev/null || fail "no docker on this AMI"
docker info 2>/dev/null | grep -qi nvidia || fail "docker has no nvidia runtime"
command -v zstd >/dev/null || apt-get -o DPkg::Lock::Timeout=600 install -y -qq zstd > "$R/apt.txt" 2>&1 || fail "zstd"

step "1 image"
aws s3 cp "$IMAGE_TAR" - | zstd -dc | docker load > "$R/docker_load.txt" 2>&1 || fail "image load"
tag=$(sed -n 's/^Loaded image: //p' "$R/docker_load.txt" | head -1)
[ "$(docker image inspect "$tag" --format '{{.Id}}')" = "$IMAGE_ID" ] || fail "image id differs from $IMAGE_ID"
kv image "$tag"; kv image_id "$IMAGE_ID"; kv image_tar "$IMAGE_TAR"

if [ -n "${SPLAT_TAR:-}" ]; then
  step "2 render ${SPLAT_TAR##*/}"
  mkdir -p /opt/splat "$R/render"
  aws s3 cp "$SPLAT_TAR" - | tar xz -C /opt/splat || fail "splat"
  d=$(dirname "$(find /opt/splat -name table.json | head -1)")
  kv splat_tar "$SPLAT_TAR"; kv splat_table_sha256 "$(sha256sum "$d/table.json" | cut -d' ' -f1)"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$R/gpu_util.csv" 2>&1 & GU=$!
  docker run --rm --gpus all -v "$SRC":/src:ro -v "$d":/splat:ro -v "$R/render":/out "$tag" \
    python3 -I /src/python/reconstruction/render/render_gsplat.py --splat /splat --out /out ${RENDER_ARGS:-} \
    > "$R/render/log.txt" 2>&1
  rc=$?
  kill "$GU" 2>/dev/null
  step "done rc=$rc"
  exit "$rc"
fi

if [ -n "${WORLDGEN_TAR:-}" ]; then
  step "2 worldgen ${WORLDGEN_TAR##*/}"
  mkdir -p /opt/wg "$R/worldgen"
  aws s3 cp "$WORLDGEN_TAR" - | tar xz -C /opt/wg || fail "worldgen data"
  d=$(dirname "$(find /opt/wg -name train_worldgen_gsplat.py | head -1)")
  kv worldgen_tar "$WORLDGEN_TAR"; kv worldgen_script_sha256 "$(sha256sum "$d/train_worldgen_gsplat.py" | cut -d' ' -f1)"
  # The image has opencv but (no evidence either way) maybe no Pillow, which the script uses only to save its
  # held-out preview PNGs; the image cannot pip at run time, so a missing Pillow gets a cv2-backed Image.fromarray.
  cat > "$R/worldgen/run.py" <<'EOF'
import runpy, sys, types
try:
    import PIL.Image
    print("PIL", PIL.__version__, flush=True)
except ImportError:
    import cv2
    class _Img:
        def __init__(self, a): self.a = a
        def save(self, p): cv2.imwrite(str(p), cv2.cvtColor(self.a, cv2.COLOR_RGB2BGR))
    m = types.ModuleType("PIL"); m.Image = types.SimpleNamespace(fromarray=_Img)
    sys.modules["PIL"], sys.modules["PIL.Image"] = m, m.Image
    print("PIL absent: cv2-backed Image.fromarray(...).save", flush=True)
sys.argv = ["train_worldgen_gsplat.py", "train", "--data", "/data", "--out", "/out", "--steps", sys.argv[1]]
runpy.run_path("/data/train_worldgen_gsplat.py", run_name="__main__")
EOF
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$R/gpu_util.csv" 2>&1 & GU=$!
  docker run --rm --gpus all -v "$d":/data:ro -v "$R/worldgen":/out "$tag" \
    python3 /out/run.py "${WORLDGEN_STEPS:-3000}" > "$R/worldgen/train.log" 2>&1
  rc=$?
  kill "$GU" 2>/dev/null
  step "done rc=$rc"
  exit "$rc"
fi

step "2 spools"
mkdir -p /opt/spools "$IN"
aws s3 cp "$SPOOLS" - | tar xz -C /opt/spools || fail "spools"
mv /opt/spools/build/spools "$IN/spools" && mv /opt/spools/build/live/scenes/maze_demo_0 "$IN/scene" || fail "spool layout"
scene=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['scene_id'])" "$IN/scene/scene.json")
[ "$scene" = "$SCENE_ID" ] || fail "scene_id $scene != $SCENE_ID"
for y in "$IN"/spools/*/spool.yaml; do
  grep -q "scene_id: *['\"]*$SCENE_ID" "$y" || fail "$y: scene_id is not $SCENE_ID"
done
kv spools "$SPOOLS"; kv scene_id "$SCENE_ID"; kv spool_dirs "\"$(cd "$IN/spools" && ls | tr '\n' ' ')\""
du -sh "$IN/spools"/* > "$R/spools_du.txt"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$R/gpu_util.csv" 2>&1 & GU=$!
run() {   # $1 = output dir under $R; the rest: the container's command
  local o="$R/$1"; shift
  mkdir -p "$o"
  docker run --rm --gpus all -v "$SRC":/src:ro -v "$IN":/in:ro -v "$o":/out -e ARIADNE_WORK=/out -e ARIADNE_INPUT=/in \
    -e ARIADNE_PKG=/src/python/reconstruction -e ARIADNE_BUDGETS="$BUDGETS" "$tag" "$@" > "$o/log.txt" 2>&1
}
step "3 D196 micro-bench"
run bench python3 -I /src/python/reconstruction/kaggle/bench_kernel.py || step "bench rc=$? (training still runs)"
step "4 training $BUDGETS"
run train python3 -I /src/python/reconstruction/kaggle/spool_train_kernel.py
rc=$?
kill "$GU" 2>/dev/null
step "done rc=$rc"
exit "$rc"
