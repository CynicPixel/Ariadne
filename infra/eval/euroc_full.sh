#!/bin/bash
# All 11 EuRoC flights on one fresh AWS instance (D86: the full-system run). Runs as root ON THE INSTANCE, launched by
#   infra/scripts/spinup.sh run --profile cpux86 --max-minutes 120 -- \
#     'IMAGE=<ecr>/ariadne/ariadne-core:<tag> GIT_COMMIT=<sha> bash infra/eval/euroc_full.sh'
# Steps, each logged to $ARIADNE_RESULTS/steps.txt as it happens so a later failure keeps the earlier evidence:
#   1. docker + the pinned dev image from ECR (the image the local build used; its id is recorded)
#   2. EuRoC from ETH via fetch_euroc.py (ranged GET, CRC-32 checked, sha256 recorded; compared to the local
#      provenance files after the flight), PAR_FETCH sequences at a time to stay polite to ETH's rate limiter
#   3. a clean build of ariadne_euroc_vo from the shipped tree, inside the image, at all instance cores
#   4. the flights, PAR flights at a time; each flight gets nproc/PAR OpenCV threads so nothing is oversubscribed.
#      Loop closure runs on a worker in wall-clock time, so contention would change its results.
#   5. one summary table (VIO keyframe-refined ATE next to the loop-closed one) + every results.yaml / manifest
# Env: IMAGE (required), GIT_COMMIT (required: the tree ships without .git), SLAM_CONFIG (euroc_v2_loops),
#      PERCEPTION_CONFIG (euroc_v2), PAR (2), PAR_FETCH (2), SEQUENCES (all 11).
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${IMAGE:?}" "${GIT_COMMIT:?}"
SLAM_CONFIG="${SLAM_CONFIG:-euroc_v2_loops}"
PERCEPTION_CONFIG="${PERCEPTION_CONFIG:-euroc_v2}"
PAR="${PAR:-2}"; PAR_FETCH="${PAR_FETCH:-2}"
SEQUENCES="${SEQUENCES:-MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult}"
SRC=/opt/ariadne/src DATA=/opt/euroc OUT=/opt/ariadne/out
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }

step "0 host"
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"; free -m > "$R/free.txt"; cat /etc/os-release > "$R/os-release.txt"
printf 'image %s\ngit_commit %s\nslam %s\nperception %s\npar %s\n' "$IMAGE" "$GIT_COMMIT" "$SLAM_CONFIG" \
  "$PERCEPTION_CONFIG" "$PAR" > "$R/run_params.txt"

step "1 docker + image"
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io > "$R/apt.txt" 2>&1; }
REG="${IMAGE%%/*}"
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || { step "ecr login FAILED"; exit 1; }
docker pull -q "$IMAGE" > /dev/null || { step "pull FAILED"; exit 1; }
docker image inspect "$IMAGE" --format '{{.Id}}' | tee "$R/image_id.txt"

step "2 fetch EuRoC"
mkdir -p "$DATA"
printf '%s\n' $SEQUENCES | xargs -P "$PAR_FETCH" -I{} sh -c \
  "python3 $SRC/infra/scripts/fetch_euroc.py {} --root $DATA > $R/fetch_{}.log 2>&1 || echo FETCH_FAILED {}" \
  | tee "$R/fetch_failures.txt"
[ -s "$R/fetch_failures.txt" ] && { step "fetch FAILED"; exit 1; }
mkdir -p "$R/provenance" && cp "$DATA"/*.provenance.json "$R/provenance/"
rm -f "$DATA"/*.zip   # extracted; the zips only cost disk
df -h / > "$R/df_after_fetch.txt"

step "3 build"
DOCKER=(docker run --rm --network host -v "$SRC":/work -v "$DATA":/data:ro -v "$OUT":/out -w /work
  -e ARIADNE_GIT_COMMIT="$GIT_COMMIT" -e ARIADNE_GIT_DIRTY=false -e ARIADNE_IMAGE_DIGEST="$(cat "$R/image_id.txt")"
  "$IMAGE")
mkdir -p "$OUT"
t0=$(date +%s)
"${DOCKER[@]}" bash -c "cmake -S /work -B /work/build/aws -G Ninja -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON \
    -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent >/dev/null \
  && cmake --build /work/build/aws --target ariadne_euroc_vo --parallel $(nproc)" > "$R/build.log" 2>&1 \
  || { step "build FAILED"; tail -40 "$R/build.log"; exit 1; }
echo "build_seconds $(( $(date +%s) - t0 ))" >> "$R/steps.txt"

step "4 flights (PAR=$PAR)"
# PAR workers, each pinned to its own disjoint CPU set (docker --cpuset-cpus). OpenCV sizes its pool from the cgroup
# cpuset, so each flight sees nproc/PAR threads and no flight competes with another; results.yaml records the pool
# size (timing_ms.opencv_threads). The flights are dealt round-robin.
N=$(nproc); PER=$(( N / PAR ))
worker() {
  local w=$1 i=0 s cpus="$(( w * PER ))-$(( (w + 1) * PER - 1 ))"
  for s in $SEQUENCES; do
    [ $(( i++ % PAR )) -eq "$w" ] || continue
    local s0; s0=$(date +%s)
    if docker run --rm --network none --cpuset-cpus "$cpus" -v "$SRC":/work -v "$DATA":/data:ro -v "$OUT":/out \
         -e ARIADNE_GIT_COMMIT="$GIT_COMMIT" -e ARIADNE_GIT_DIRTY=false \
         -e ARIADNE_IMAGE_DIGEST="$(cat "$R/image_id.txt")" "$IMAGE" \
         /work/build/aws/cpp/runtime/ariadne_euroc_vo --sequence "/data/$s" \
         --perception-config "/work/configs/perception/$PERCEPTION_CONFIG.yaml" \
         --slam-config "/work/configs/slam/$SLAM_CONFIG.yaml" --out "/out/$s" > "$R/flight_$s.log" 2>&1; then
      echo "RUN_OK $s $(( $(date +%s) - s0 ))s cpus=$cpus"
    else
      echo "RUN_FAILED $s cpus=$cpus"
    fi
  done
}
for w in $(seq 0 $(( PAR - 1 ))); do worker "$w" | tee -a "$R/flights.txt" & done
wait

step "5 collect"
for s in $SEQUENCES; do
  d=$(ls -d "$OUT/$s"/*/ 2>/dev/null | tail -1); [ -n "$d" ] || continue
  mkdir -p "$R/runs/$s"
  cp "$d"/results.yaml "$d"/manifest.yaml "$R/runs/$s/" 2>/dev/null
  cp "$d"/loop_trace.jsonl "$d"/estimate_body.tum "$R/runs/$s/" 2>/dev/null
done
python3 - "$R/runs" > "$R/summary.md" <<'PY'
import re, sys, pathlib
rows = []
for d in sorted(pathlib.Path(sys.argv[1]).iterdir()):
    t = (d / "results.yaml").read_text()
    def num(key, field):
        m = re.search(rf"^{key}: {{[^}}]*{field}: ([0-9.e+-]+)", t, re.M)
        return float(m.group(1)) if m else float("nan")
    lost = re.search(r"^vo: {lost_frames: (\d+)", t, re.M)
    thr = re.search(r"opencv_threads: (\d+)", t)
    rows.append((d.name, num("keyframe_refined", "ate_rmse_m"), num("loop_closed_keyframe_refined", "ate_rmse_m"),
                 lost.group(1) if lost else "?", thr.group(1) if thr else "?"))
print("| flight | VIO kf-refined ATE m | loops kf-refined ATE m | lost frames | opencv threads |")
print("|---|---|---|---|---|")
for r in rows: print(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]} | {r[4]} |")
if rows:
    print(f"| **avg ({len(rows)})** | {sum(r[1] for r in rows)/len(rows):.4f} | {sum(r[2] for r in rows)/len(rows):.4f} | | |")
PY
cat "$R/summary.md"
grep -c RUN_OK "$R/flights.txt" | sed 's/^/flights_ok /' >> "$R/steps.txt"
step "done"
