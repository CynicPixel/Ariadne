#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build ariadne-recon-gpu (infra/docker/recon-gpu.Dockerfile) on an AWS CPU spot instance and ship it to S3 as a
# tarball (the instance role cannot push to ECR; the g4dn demo instance loads it from S3, as remote_dark_render.sh
# loads the sim image). Run BY spinup.sh on the instance, from the repo root:
#
#   infra/scripts/spinup.sh run --profile cpux86 --type c7i.4xlarge --max-minutes 90 -- \
#     'BUCKET=<artifact bucket> COMMIT=<git HEAD> bash infra/scripts/remote_recon_gpu_build.sh'
#
# Results (also uploaded to s3://$BUCKET/runs/images/ as they appear):
#   build.log, image_id.txt, csrc_proof.txt (`from gsplat import csrc` in the built image, no GPU, no compile),
#   ariadne-recon-gpu-<short id>.tar.zst + .sha256, provenance.txt
# No GPU is needed: nvcc compiles for TORCH_CUDA_ARCH_LIST without a device.
# ---------------------------------------------------------------------------
set -euo pipefail
: "${BUCKET:?}" "${COMMIT:?}" "${ARIADNE_RESULTS:?}"
R="$ARIADNE_RESULTS"
DEST="s3://$BUCKET/runs/images"
log() { echo "[remote $(date -u +%T)] $*" | tee -a "$R/progress.log"; }
up() { aws s3 cp "$1" "$DEST/recon-gpu-build-$COMMIT/$(basename "$1")" --only-show-errors || true; }
printf 'commit %s\ndockerfile infra/docker/recon-gpu.Dockerfile sha256 %s\n' "$COMMIT" \
  "$(sha256sum infra/docker/recon-gpu.Dockerfile | cut -d' ' -f1)" > "$R/provenance.txt"
log "host: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GiB, disk $(df -h / | awk 'NR==2{print $4}') free"
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io zstd; }
command -v zstd >/dev/null || apt-get install -y -qq zstd

log "build"
mkdir -p /tmp/ctx                       # the Dockerfile copies nothing from the context
t0=$(date +%s)
if ! docker build -f infra/docker/recon-gpu.Dockerfile -t ariadne/recon-gpu:build /tmp/ctx \
     > "$R/build.log" 2>&1; then
  log "build FAILED after $(( $(date +%s) - t0 )) s (build.log)"; up "$R/build.log"; up "$R/progress.log"; exit 2
fi
log "build PASSED in $(( $(date +%s) - t0 )) s"
id="$(docker image inspect ariadne/recon-gpu:build --format '{{.Id}}')"
echo "$id" > "$R/image_id.txt"
short="${id#sha256:}"; short="${short:0:12}"
docker image inspect ariadne/recon-gpu:build --format '{{.Size}}' > "$R/image_bytes.txt"
grep -E "gsplat.csrc AOT:|torch .* gsplat .* csrc" "$R/build.log" > "$R/csrc_proof.txt" || true
docker run --rm --network none ariadne/recon-gpu:build python -c \
  "import torch, gsplat; from gsplat import csrc; print('run-time import, no GPU:', torch.__version__, gsplat.__version__, csrc.__file__, 'cuda available', torch.cuda.is_available())" \
  >> "$R/csrc_proof.txt" 2>&1
for f in provenance.txt build.log image_id.txt image_bytes.txt csrc_proof.txt; do up "$R/$f"; done

log "save $short"
tar="ariadne-recon-gpu-$short.tar.zst"
docker save ariadne/recon-gpu:build | zstd -T0 -10 -q -o "/tmp/$tar"
sha256sum "/tmp/$tar" | sed "s|/tmp/||" > "$R/$tar.sha256"
log "upload $(du -h "/tmp/$tar" | cut -f1)"
aws s3 cp "/tmp/$tar" "$DEST/$tar" --only-show-errors
aws s3 cp "$R/$tar.sha256" "$DEST/$tar.sha256" --only-show-errors
rm -f "/tmp/$tar"
log "done: $DEST/$tar ($id)"
up "$R/progress.log"
