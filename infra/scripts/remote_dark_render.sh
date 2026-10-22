#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 5 dark renders on a GPU spot instance (D137). Run BY spinup.sh on the instance, from the repo root:
#
#   infra/scripts/spinup.sh run --profile gpu --max-minutes 150 -- \
#     'BUCKET=<artifact bucket> IMAGE_ID=<sha256:...> COMMIT=<git HEAD> bash infra/scripts/remote_dark_render.sh'
#
# 1. Loads the EXACT local sim image (uploaded to s3://$BUCKET/runs/images/, the only prefix the instance may read)
#    and checks its id: the recordings must come from the image every local replay is pinned to.
# 2. GPU smoke first (sim.sh's acceptance test): the NVIDIA EGL path has never run (SMOKE §4). If it does not print
#    "render mode: nvidia-egl" and SIM_SMOKE=PASS, stop: llvmpipe on 4 vCPU cannot finish a mission in the budget.
# 3. Generates and records, full mission, in priority order:
#      ci_tiny_dark seed 0 + rig_stereo_v2 (lamp)   -- the dark world
#      ci_tiny_dark_fixed seed 0 + rig_stereo_v1      -- the fixed-light twin (§31.5)
#    Each recording is uploaded to s3://$BUCKET/runs/ph5-dark/<COMMIT>/ as soon as it finishes: a spot reclaim kills
#    the instance without running spinup's end-of-run upload, so incremental upload is what keeps finished work.
#    The local copy is then dropped, so spinup's own upload does not send the same ~11 GB again.
# The shipped tree has no .git: the commit comes in as COMMIT and is written to provenance.txt (recording.json's
# git fields will be empty on this host).
# ---------------------------------------------------------------------------
set -euo pipefail
: "${BUCKET:?}" "${IMAGE_ID:?}" "${COMMIT:?}" "${ARIADNE_RESULTS:?}"
R="$ARIADNE_RESULTS"
DEST="s3://$BUCKET/runs/ph5-dark/$COMMIT"
log() { echo "[remote $(date -u +%T)] $*" | tee -a "$R/progress.log"; aws s3 cp "$R/progress.log" "$DEST/progress.log" --only-show-errors || true; }
printf 'commit %s\nimage %s\n' "$COMMIT" "$IMAGE_ID" > "$R/provenance.txt"

log "host: $(nproc) vCPU, $(free -g | awk '/Mem/{print $2}') GiB"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$R/nvidia.csv"
command -v zstd >/dev/null || { apt-get update -qq && apt-get install -y -qq zstd; }
if ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
  log "configuring the NVIDIA container runtime"
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi
docker info --format '{{json .Runtimes}}' > "$R/docker_runtimes.json"

log "loading image"
# The object is named by the 12-hex short id (what `docker images` prints). A missing key answers 403, not 404:
# the instance role has no s3:ListBucket (run 20261022T193443Z-gpu-5bdc2c failed exactly so, asking for the full id).
short="${IMAGE_ID#sha256:}"; short="${short:0:12}"
aws s3 cp "s3://$BUCKET/runs/images/ariadne-sim-$short.tar.zst" - --only-show-errors \
  | zstd -dc | docker load
got="$(docker image inspect ariadne/sim:dev --format '{{.Id}}')"
[ "$got" = "$IMAGE_ID" ] || { log "image id $got != $IMAGE_ID"; exit 2; }

# The image runs as its non-root user (ariadne), but spinup unpacks the tree as root: every container that writes into
# /work (smoke PNGs, plugin builds, recordings) failed with EACCES (run 20261022T195315Z-gpu-820b83; main's D86 hit
# the same). Hand the tree to the image's uid:gid.
uid="$(docker run --rm --entrypoint id ariadne/sim:dev -u)"; gid="$(docker run --rm --entrypoint id ariadne/sim:dev -g)"
mkdir -p build/sim-out build/rec build/out
chown -R "$uid:$gid" "$PWD"
log "tree owned by image uid $uid:$gid"

log "GPU smoke"
ARIADNE_SMOKE_OUT_HOST="$PWD/build/sim-out" infra/scripts/sim.sh > "$R/smoke.log" 2>&1 || true
grep -q 'render mode: nvidia-egl' "$R/smoke.log" && grep -q 'SIM_SMOKE=PASS' "$R/smoke.log" \
  || { log "GPU smoke FAILED (see smoke.log)"; exit 3; }
log "GPU smoke PASS"

record() {   # preset rig name
  local preset="$1" rig="$2" name="$3"
  log "$name: generate ($preset, $rig)"
  ARIADNE_RIG="$rig" infra/scripts/worldgen.sh generate "$preset" 0 "build/worldgen/scenes/$name" > "$R/$name.worldgen.log" 2>&1
  log "$name: record"
  local t0; t0=$(date +%s)
  if infra/scripts/sim_record.sh "build/worldgen/scenes/$name" "build/rec/$name" > "$R/$name.record.log" 2>&1; then
    log "$name: PASSED in $(( $(date +%s) - t0 )) s"
  else
    log "$name: record FAILED in $(( $(date +%s) - t0 )) s"
  fi
  mkdir -p "build/out/$name"
  cp -r "build/worldgen/scenes/$name" "build/out/$name/scene"
  mv "build/rec/$name" "build/out/$name/rec" 2>/dev/null || true
  cp "$R/$name.worldgen.log" "$R/$name.record.log" "build/out/$name/" 2>/dev/null || true
  log "$name: uploading $(du -sh "build/out/$name" | cut -f1)"
  aws s3 cp --recursive "build/out/$name" "$DEST/$name" --only-show-errors && rm -rf "build/out/$name"
  log "$name: uploaded"
}
record ci_tiny_dark configs/sim/rig_stereo_v2.yaml ci_tiny_dark_0
record ci_tiny_dark_fixed configs/sim/rig_stereo_v1.yaml ci_tiny_dark_fixed_0
log "done"
