#!/bin/bash
# Phase 5 CPU session (D142): Hilti 2023 extraction + the no-dust lighting replays, next to S3 and Hugging Face.
# Runs as root ON THE INSTANCE, launched by
#   infra/scripts/spinup.sh run --profile cpux86 --type c7i.2xlarge --max-minutes 150 -- \
#     'IMAGE=<ecr>/ariadne/ariadne-core:<tag> GIT_COMMIT=<sha> BUCKET=<bucket> DARK_PREFIX=<s3 prefix> \
#      bash infra/eval/ph5_aws.sh'
# Each step logs to $ARIADNE_RESULTS/steps.txt; big outputs go to s3://$BUCKET/runs/ph5-aws/$GIT_COMMIT/ as soon as
# they exist (a spot reclaim skips spinup's end-of-run upload).
#   1. host tools: docker, python venv (rosbags 0.10.4, numpy, opencv-python-headless 4.10.0.84), the ECR image
#   2. Hilti site2_robot_3 (pinned HF revision, sha256 checked against the LFS oid) + calibration.zip + ground truth
#      -> topic list -> one stereo pair + every IMU + exposure + control points in EuRoC layout -> tar -> S3.
#      Calibration files go to results. HILTI=extras|skip for reruns (below).
#   3. build ariadne_sim_vo from the shipped tree in the image
#   4. the dark recordings from S3 (keys in ph5_dark_manifest.txt) -> 4 replay rows (dark lamp / dark fixed x no D / sensor D), agent 0, 1000 frames,
#      sim_v1 perception + sim_v2 slam: the configs of D69's lit rows on the same trajectories (ci_tiny seed 0)
#      ROWS=d145 instead runs the §31.5 rows of D145-D147 (below): Stage 3 equality, features on the proxy, active
#      exposure, Failure E, the lamp sweep and the brightness-matched fixed-light row. ROWS=lighting (default) = D142's.
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${IMAGE:?}" "${GIT_COMMIT:?}" "${BUCKET:?}" "${DARK_PREFIX:?}"
SRC=/opt/ariadne/src WORK=/opt/work OUT=/opt/ariadne/out
DEST="s3://$BUCKET/runs/ph5-aws/$GIT_COMMIT"
HF=https://huggingface.co/datasets/Hilti-Research/hilti-slam-challenge-2023
REV=8e2a7a2de63670730a322d74090c1a5ea4b641eb
mkdir -p "$WORK" "$OUT"
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; aws s3 cp "$R/steps.txt" "$DEST/steps.txt" --only-show-errors || true; }

step "0 host"
nproc > "$R/nproc.txt"; free -m > "$R/free.txt"; df -h / > "$R/df0.txt"
printf 'image %s\ngit_commit %s\nhf_revision %s\n' "$IMAGE" "$GIT_COMMIT" "$REV" > "$R/run_params.txt"

step "1 tools"
apt-get update -qq && apt-get install -y -qq docker.io python3-venv zstd > "$R/apt.txt" 2>&1
python3 -m venv /opt/venv && /opt/venv/bin/pip install -q rosbags==0.10.4 numpy opencv-python-headless==4.10.0.84 \
  > "$R/pip.txt" 2>&1 || { step "pip FAILED"; exit 1; }
/opt/venv/bin/pip freeze > "$R/pip_freeze.txt"
REG="${IMAGE%%/*}"
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || { step "ecr login FAILED"; exit 1; }
docker pull -q "$IMAGE" > /dev/null || { step "pull FAILED"; exit 1; }
docker image inspect "$IMAGE" --format '{{.Id}}' | tee "$R/image_id.txt"

# HILTI=full (default): fetch + export images, IMU and side files. extras: fetch, then only the side files (exposure
# + control points) -- a rerun after a full export that predates them. skip: an earlier session already did it all.
HILTI="${HILTI:-full}"
case "$HILTI" in full|extras|skip) ;; *) step "HILTI must be full|extras|skip"; exit 1 ;; esac
if [ "$HILTI" = skip ]; then
  step "2 hilti skipped (HILTI=skip)"
else
step "2 hilti fetch"
H="$WORK/hilti"; mkdir -p "$H"
oid() { curl -s "https://huggingface.co/api/datasets/Hilti-Research/hilti-slam-challenge-2023/paths-info/$REV" \
          -H 'Content-Type: application/json' -d "{\"paths\":[\"$1\"]}" | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["lfs"]["oid"])'; }
for f in rosbags/site2_robot_3.bag calibration/calibration.zip; do
  curl -sL --retry 5 -o "$H/$(basename $f)" "$HF/resolve/$REV/$f" || { step "fetch $f FAILED"; exit 1; }
  want="$(oid $f)"; got="$(sha256sum "$H/$(basename $f)" | cut -d' ' -f1)"
  echo "$f sha256 $got lfs_oid $want" | tee -a "$R/hilti_sha256.txt"
  [ "$got" = "$want" ] || { step "sha256 MISMATCH $f"; exit 1; }
done
curl -sL -o "$R/site2_robot_3_gt.txt" "$HF/resolve/$REV/ground_truth/site2_robot_3.txt"
mkdir -p "$R/hilti_calibration" && (cd "$R/hilti_calibration" && python3 -m zipfile -e "$H/calibration.zip" .) \
  && find "$R/hilti_calibration" -type f -size +5M -delete   # keep yaml/txt, drop any images or bags
step "2b hilti topics"
cd "$SRC/python" && /opt/venv/bin/python -m dataset_tools.hilti.to_asl list "$H/site2_robot_3.bag" > "$R/hilti_topics.txt" 2>&1
cd "$SRC"
# One stereo pair: HILTI_LEFT/HILTI_RIGHT if given, else the first image topic containing "left" whose "right" twin
# exists (sorted). The choice and the full topic list both go to results, so a wrong pick is visible and cheap to redo.
read -r LEFT RIGHT <<< "$(python3 - "$R/hilti_topics.txt" "${HILTI_LEFT:-}" "${HILTI_RIGHT:-}" <<'PY'
import sys
rows = [l.split("\t") for l in open(sys.argv[1]) if "\t" in l]
imgs = sorted(t for t, ty, n in rows if ty.endswith("Image"))
if sys.argv[2] and sys.argv[3]:
    print(sys.argv[2], sys.argv[3]); raise SystemExit
for t in imgs:
    if "left" in t and t.replace("left", "right") in imgs:
        print(t, t.replace("left", "right")); break
else:
    print("NONE NONE")
PY
)"
echo "left $LEFT right $RIGHT" | tee "$R/hilti_pair.txt"
if [ "$LEFT" != NONE ] && [ "$HILTI" = extras ]; then
  step "2c hilti side files"
  X="$WORK/site2_robot_3_extras"
  (cd "$SRC/python" && /opt/venv/bin/python -m dataset_tools.hilti.to_asl export "$H/site2_robot_3.bag" "$X" \
     --left "$LEFT" --right "$RIGHT" --no-images) > "$R/hilti_extras.txt" 2>&1 || step "hilti side files FAILED"
  mkdir -p "$R/hilti_extras/cam0" "$R/hilti_extras/cam1"
  cp "$X/gcp.csv" "$R/hilti_extras/" && cp "$X/mav0/cam0/exposure.csv" "$R/hilti_extras/cam0/" \
    && cp "$X/mav0/cam1/exposure.csv" "$R/hilti_extras/cam1/" && cp "$R/hilti_pair.txt" "$R/hilti_extras/" \
    && tar -C "$R" -czf "$WORK/site2_robot_3_extras.tar.gz" hilti_extras \
    && aws s3 cp "$WORK/site2_robot_3_extras.tar.gz" "$DEST/site2_robot_3_extras.tar.gz" --only-show-errors \
    && step "hilti side files uploaded"
  rm -rf "$X"
elif [ "$LEFT" != NONE ]; then
  step "2c hilti export"
  (cd "$SRC/python" && /opt/venv/bin/python -m dataset_tools.hilti.to_asl export "$H/site2_robot_3.bag" \
     "$WORK/site2_robot_3_asl" --left "$LEFT" --right "$RIGHT") > "$R/hilti_export.txt" 2>&1 \
    || step "hilti export FAILED"
  cp -r "$R/hilti_calibration" "$WORK/site2_robot_3_asl/" 2>/dev/null
  cp "$R/site2_robot_3_gt.txt" "$R/hilti_pair.txt" "$WORK/site2_robot_3_asl/"
  tar -C "$WORK" -cf - site2_robot_3_asl | zstd -T0 -3 -q -o "$WORK/site2_robot_3_asl.tar.zst"
  ls -l "$WORK/site2_robot_3_asl.tar.zst" | tee -a "$R/steps.txt"
  aws s3 cp "$WORK/site2_robot_3_asl.tar.zst" "$DEST/site2_robot_3_asl.tar.zst" --only-show-errors \
    && step "hilti uploaded"
else
  step "hilti: no left/right image pair found; see hilti_topics.txt"
fi
rm -rf "$H" "$WORK/site2_robot_3_asl"   # free the disk for the recordings
df -h / > "$R/df_after_hilti.txt"
fi

step "3 build"
# Not under /work/.cache: in a worktree .cache is a symlink to the main checkout's, shipped dangling (2026-10-22).
chown -R "$(docker run --rm "$IMAGE" id -u):$(docker run --rm "$IMAGE" id -g)" "$SRC" "$OUT"
t0=$(date +%s)
docker run --rm --network host -v "$SRC":/work -w /work "$IMAGE" bash -c "cmake -S /work -B /work/build/aws -G Ninja \
    -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON -DCMAKE_BUILD_TYPE=Release \
    -DFETCHCONTENT_BASE_DIR=/work/build/fetchcontent >/dev/null \
  && cmake --build /work/build/aws --target ariadne_sim_vo --parallel $(nproc)" > "$R/build.log" 2>&1 \
  || { step "build FAILED"; tail -40 "$R/build.log"; exit 1; }
echo "build_seconds $(( $(date +%s) - t0 ))" >> "$R/steps.txt"

step "4 recordings"
# By key from the committed manifest: the instance role may GetObject but not ListBucket, so --recursive fails.
grep -v '^#' "$SRC/infra/eval/ph5_dark_manifest.txt" | awk '{print $2}' \
  | xargs -P 8 -I{} aws s3 cp "$DARK_PREFIX/{}" "$WORK/{}" --only-show-errors || { step "fetch recordings FAILED"; exit 1; }
grep -v '^#' "$SRC/infra/eval/ph5_dark_manifest.txt" | while read -r bytes key; do
  [ "$(stat -c %s "$WORK/$key" 2>/dev/null)" = "$bytes" ] || echo "size mismatch $key"
done > "$R/fetch_check.txt"
[ -s "$R/fetch_check.txt" ] && { step "fetch recordings: size mismatch (fetch_check.txt)"; exit 1; }
chown -R "$(docker run --rm "$IMAGE" id -u):$(docker run --rm "$IMAGE" id -g)" "$WORK"
step "4b replays"
N=$(nproc); PER=$(( N / 2 ))
run() {   # name rec degradation cpus [perception config] [extra tool args...]
  local name=$1 rec=$2 deg=$3 cpus=$4 pcfg=${5:-sim_v1} extra=() s0; s0=$(date +%s)
  shift 4; [ $# -gt 0 ] && shift
  [ "$deg" != none ] && extra=(--degradation "/work/configs/degradation/$deg.yaml")
  extra+=("$@")
  if docker run --rm --network none --cpuset-cpus "$cpus" -v "$SRC":/work -v "$WORK":/rec:ro -v "$OUT":/out \
       -e ARIADNE_GIT_COMMIT="$GIT_COMMIT" -e ARIADNE_GIT_DIRTY=false -e ARIADNE_IMAGE_DIGEST="$(cat "$R/image_id.txt")" \
       "$IMAGE" /work/build/aws/cpp/runtime/ariadne_sim_vo --rec "/rec/$rec/rec" --scene "/rec/$rec/scene" --agent 0 \
       --frames 1000 --seed 1 --perception-config "/work/configs/perception/$pcfg.yaml" \
       --slam-config /work/configs/slam/sim_v2.yaml "${extra[@]}" --out "/out/$name" > "$R/run_$name.log" 2>&1; then
    echo "RUN_OK $name $(( $(date +%s) - s0 ))s"
  else
    echo "RUN_FAILED $name"
  fi
}
if [ "${ROWS:-lighting}" = d145 ]; then
  # §31.5 rows (D145-D147), two lanes. L = the lamp-lit recording, sensor D with the sRGB decode (Stage 3 and the
  # controller need linear frames). Row 2 must reproduce row 1's trajectory (D146); 3 vs 4 is the features row; 1 vs 5
  # fixed vs active exposure; 6 vs 7 Failure E with and without the controller (scene +2 stops at 20 s).
  L=ci_tiny_dark_0 F=ci_tiny_dark_fixed_0 S=lampdark_sensor_v0 CAL=(--lamp-calib /work/configs/sim/lamp_calib_v2.yaml)
  A="0-$((PER-1))" B="$PER-$((N-1))"
  { run r01_flat "$L" "$S" "$A" sim_v1_rad_flat
    run r02_falloff "$L" "$S" "$A" sim_v1_ph5_falloff "${CAL[@]}"
    run r03_kfcad "$L" "$S" "$A" sim_v1_ph5_kfcad
    run r04_kfcad_proxy "$L" "$S" "$A" sim_v1_ph5_kfcad_proxy "${CAL[@]}"
    run r05_expo "$L" "$S" "$A" sim_v1_ph5_expo; } | tee -a "$R/runs.txt" &
  { run r06_failE_expo "$L" "$S" "$B" sim_v1_ph5_expo --radiance-event 20:2
    run r07_failE_fixed "$L" "$S" "$B" sim_v1_rad_flat --radiance-event 20:2
    run r08_lamp50 "$L" lampdark_lamp50_v0 "$B"
    run r09_lamp15 "$L" lampdark_lamp15_v0 "$B"
    run r10_fixed_matched "$F" lampdark_fixed_matched_v0 "$B"; } | tee -a "$R/runs.txt" &
  wait
else
{ run lamp_noD ci_tiny_dark_0 none "0-$((PER-1))"; run lamp_sensor ci_tiny_dark_0 lampdark_sensor_v0 "0-$((PER-1))"; } \
  | tee -a "$R/runs.txt" &
{ run fixed_noD ci_tiny_dark_fixed_0 none "$PER-$((N-1))"; run fixed_sensor ci_tiny_dark_fixed_0 lampdark_sensor_v0 "$PER-$((N-1))"; } \
  | tee -a "$R/runs.txt" &
wait
fi

step "5 collect"
for d in "$OUT"/*/; do
  name=$(basename "$d"); r=$(ls -d "$d"*/ 2>/dev/null | tail -1); [ -n "$r" ] || continue
  mkdir -p "$R/replays/$name"
  cp "$r"results.yaml "$r"manifest.yaml "$r"estimate_body.tum "$r"vo_trace.csv "$r"exposure_trace.csv \
    "$R/replays/$name/" 2>/dev/null
done
step "done"
