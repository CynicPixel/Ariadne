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
#      -> topic list -> one stereo pair + every IMU in EuRoC layout -> tar -> S3. Calibration files go to results.
#   3. build ariadne_sim_vo from the shipped tree in the image
#   4. the dark recordings from S3 -> 4 replay rows (dark lamp / dark fixed x no D / sensor D), agent 0, 1000 frames,
#      sim_v1 perception + sim_v2 slam: the configs of D69's lit rows on the same trajectories (ci_tiny seed 0)
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

if [ "${SKIP_HILTI:-0}" = 1 ]; then   # already extracted by an earlier session (its tar is under that commit's prefix)
  step "2 hilti skipped (SKIP_HILTI=1)"
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
if [ "$LEFT" != NONE ]; then
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
for n in ci_tiny_dark_0 ci_tiny_dark_fixed_0; do
  aws s3 cp --recursive "$DARK_PREFIX/$n" "$WORK/$n" --only-show-errors || { step "fetch $n FAILED"; exit 1; }
done
chown -R "$(docker run --rm "$IMAGE" id -u):$(docker run --rm "$IMAGE" id -g)" "$WORK"
step "4b replays"
N=$(nproc); PER=$(( N / 2 ))
run() {   # name rec degradation cpus
  local name=$1 rec=$2 deg=$3 cpus=$4 extra=() s0; s0=$(date +%s)
  [ "$deg" != none ] && extra=(--degradation "/work/configs/degradation/$deg.yaml")
  if docker run --rm --network none --cpuset-cpus "$cpus" -v "$SRC":/work -v "$WORK":/rec:ro -v "$OUT":/out \
       -e ARIADNE_GIT_COMMIT="$GIT_COMMIT" -e ARIADNE_GIT_DIRTY=false -e ARIADNE_IMAGE_DIGEST="$(cat "$R/image_id.txt")" \
       "$IMAGE" /work/build/aws/cpp/runtime/ariadne_sim_vo --rec "/rec/$rec/rec" --scene "/rec/$rec/scene" --agent 0 \
       --frames 1000 --seed 1 --perception-config /work/configs/perception/sim_v1.yaml \
       --slam-config /work/configs/slam/sim_v2.yaml "${extra[@]}" --out "/out/$name" > "$R/run_$name.log" 2>&1; then
    echo "RUN_OK $name $(( $(date +%s) - s0 ))s"
  else
    echo "RUN_FAILED $name"
  fi
}
{ run lamp_noD ci_tiny_dark_0 none "0-$((PER-1))"; run lamp_sensor ci_tiny_dark_0 lampdark_sensor_v0 "0-$((PER-1))"; } \
  | tee -a "$R/runs.txt" &
{ run fixed_noD ci_tiny_dark_fixed_0 none "$PER-$((N-1))"; run fixed_sensor ci_tiny_dark_fixed_0 lampdark_sensor_v0 "$PER-$((N-1))"; } \
  | tee -a "$R/runs.txt" &
wait

step "5 collect"
for d in "$OUT"/*/; do
  name=$(basename "$d"); r=$(ls -d "$d"*/ 2>/dev/null | tail -1); [ -n "$r" ] || continue
  mkdir -p "$R/replays/$name"
  cp "$r"results.yaml "$r"manifest.yaml "$r"estimate_body.tum "$r"vo_trace.csv "$R/replays/$name/" 2>/dev/null
done
step "done"
