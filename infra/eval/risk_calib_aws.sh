#!/bin/bash
# 4.2 risk-proxy calibration renders (design RISK_ZONES_4.2 §3, D112) on ONE fresh g4dn.xlarge. Runs as root ON THE
# INSTANCE, from spinup.sh's copy of the launching working tree:
#   infra/scripts/spinup.sh run --profile gpu --max-minutes 240 -- \
#     'STAGE=s3://<bucket>/runs/risk_calib-staging COMMIT=<sha> REG=<ecr registry> SEEDS="0 1 2" bash infra/eval/risk_calib_aws.sh'
# Inputs staged under $STAGE by infra/eval/risk_calib_aws_launch.sh (nothing is built remotely):
#   images.txt     "<ecr ref> <local name> <image id>": ariadne/sim:dev (render) and ariadne/core-dev:dev (estimator)
#   inputs.tar     the generated scenes build/worldgen/scenes/ci_tiny_hazard_<seed> and the payload binaries
#                  (ariadne_sim_collab, ariadne_volumetric_eval, built in that core-dev image), with SHA256SUMS
# Steps, each logged to $ARIADNE_RESULTS/steps.txt:
#   1. images by id; 2. inputs by sha256;
#   3. [SMOKE-TEST] the NVIDIA EGL render path (infra/scripts/sim.sh acceptance): must report nvidia-egl and PASS,
#      or the run stops (sim_record.sh would otherwise fall back to llvmpipe without saying so);
#   4. per seed: render the default mission (sim_record.sh), the Phase 1 estimator chain to a spool
#      (ariadne_sim_collab, clean, as phase1.sh stage 1), the 4.1/4.2 map (ariadne_volumetric_eval); then the
#      recording is deleted (12 GB; it is re-renderable) and the spool + small outputs come back.
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${STAGE:?}" "${COMMIT:?}" "${REG:?}"
SEEDS="${SEEDS:-0 1 2}"
REPO="$(pwd)"
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }

step "0 host"
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"; nvidia-smi > "$R/nvidia_smi.txt" 2>&1
docker info --format '{{json .Runtimes}}' > "$R/docker_runtimes.txt" 2>&1
printf 'commit %s\nstage %s\nseeds %s\n' "$COMMIT" "$STAGE" "$SEEDS" > "$R/run_params.txt"

step "1 images"
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || fail "ecr login"
aws s3 cp "$STAGE/images.txt" "$R/images.txt" --only-show-errors || fail "images.txt"
while read -r ref name want; do
  docker pull -q "$ref" > /dev/null || fail "pull $ref"
  docker tag "$ref" "$name"
  got="$(docker image inspect "$name" --format '{{.Id}}')"
  [ "$got" = "$want" ] || fail "image $name is $got, staged $want"
  echo "$name $ref $got" >> "$R/image_ids.txt"
done < "$R/images.txt"

step "2 inputs"
aws s3 cp "$STAGE/inputs.tar" - | tar x -C "$REPO" || fail "inputs.tar"
(cd "$REPO" && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
cp "$REPO/inputs.SHA256SUMS" "$R/"
uid="$(docker run --rm ariadne/core-dev:dev id -u)"; gid="$(docker run --rm ariadne/core-dev:dev id -g)"
df -h / > "$R/df_start.txt"

step "3 [SMOKE-TEST] NVIDIA EGL render path"
mkdir -p "$R/egl_smoke"
ARIADNE_SMOKE_OUT_HOST="$R/egl_smoke" infra/scripts/sim.sh > "$R/egl_smoke/sim_smoke.log" 2>&1
smoke_rc=$?
grep -h -i 'render mode\|SIM_SMOKE' "$R/egl_smoke/sim_smoke.log" | tail -5 > "$R/egl_smoke/summary.txt"
[ $smoke_rc -eq 0 ] && grep -q 'nvidia-egl' "$R/egl_smoke/summary.txt" && grep -q 'SIM_SMOKE=PASS' "$R/egl_smoke/summary.txt" \
  || fail "EGL smoke (rc $smoke_rc): $(tr '\n' ' ' < "$R/egl_smoke/summary.txt")"

for s in $SEEDS; do
  SC=build/worldgen/scenes/ci_tiny_hazard_$s REC=build/risk_calib/rec_$s OUT=build/risk_calib/seed_$s
  mkdir -p "$R/seed_$s" "$OUT"
  step "4.$s render seed $s"
  t0=$(date +%s)
  ARIADNE_GIT_COMMIT="$COMMIT" ARIADNE_GIT_DIRTY=false \
    infra/scripts/sim_record.sh "$SC" "$REC" > "$R/seed_$s/sim_record.log" 2>&1 || fail "sim_record seed $s"
  echo "render_wall_s $(( $(date +%s) - t0 ))" >> "$R/seed_$s/timing.txt"
  cp "$REC/recording.json" "$R/seed_$s/" 2>/dev/null
  cp "$REC/logs/gz_recorder.json" "$R/seed_$s/" 2>/dev/null
  chown -R "$uid:$gid" "$REPO/build" "$REPO/evaluation" 2>/dev/null
  step "4.$s estimator chain seed $s"
  t0=$(date +%s)
  docker run --rm -v "$REPO":/work -v "$REPO/payload":/p:ro -w /work \
    -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY=false \
    -e ARIADNE_IMAGE_DIGEST="$(docker image inspect ariadne/core-dev:dev --format '{{.Id}}')" \
    ariadne/core-dev:dev /p/ariadne_sim_collab --rec /work/$REC --scene /work/$SC \
      --perception-config /work/configs/perception/sim_v1.yaml --slam-config /work/configs/slam/sim_v1.yaml \
      --collab-config /work/configs/collab/collab_v1.yaml --inject-faults 0 \
      --spool /work/$OUT/spool --recon-config /work/configs/recon/recon_v1.yaml --out /work/$OUT/sim_collab \
    > "$R/seed_$s/collab.log" 2>&1 || fail "sim_collab seed $s"
  echo "estimator_wall_s $(( $(date +%s) - t0 ))" >> "$R/seed_$s/timing.txt"
  spool="$(ls -d "$OUT"/spool/*/ | head -1)"; spool="${spool%/}"
  [ -f "$spool/spool.yaml" ] || fail "no spool for seed $s"
  step "4.$s map seed $s"
  docker run --rm -v "$REPO":/work -v "$REPO/payload":/p:ro -w /work \
    -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY=false \
    ariadne/core-dev:dev /p/ariadne_volumetric_eval --spool /work/$spool --scene /work/$SC \
      --config /work/configs/mission/mission_v1.yaml --out /work/$OUT/map > "$R/seed_$s/map.log" 2>&1 || fail "map seed $s"
  cp -r "$OUT/map" "$R/seed_$s/map"
  find "$OUT/sim_collab" -maxdepth 2 -name '*.yaml' -size -1M -exec cp {} "$R/seed_$s/" \;
  tar czf "$R/seed_$s/spool.tar.gz" -C "$(dirname "$spool")" "$(basename "$spool")"
  (cd "$REC" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$R/seed_$s/recording.SHA256SUMS"
  rm -rf "$REC"   # 12 GB; re-renderable from the scene + image id
  df -h / > "$R/seed_$s/df.txt"
done
step "done"
