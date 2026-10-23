#!/bin/bash
# The 3.2/3.4 gate sweep (design SIMCOLLAB_3.2 §4-§6, D171) on one AWS instance, as root, launched by
# infra/eval/collab_aws_launch.sh through spinup.sh. Inputs staged under $STAGE as for the pre-merge
# (premerge_aws.sh): repo.bundle, images.txt (the dev image), inputs.tar (the sim exports), and the recordings as
# persistent datasets under $DATA0 / $DATA1 with their SHA256SUMS in inputs.tar.
#
# Stages (each logged to $ARIADNE_RESULTS/steps.txt; the small outputs of every run come back):
#   A  pass A: 2 scenes x 3 seeds x 2 agents (ariadne_collab_passa), P_A at a time
#   B1 the similarity arm (no budget) per (scene, seed) on collapsed_v1: R (the fetch rate) and the response ratio
#   B2 greedy + spectral x 4 budget levels per (scene, seed); levels and ratio from the OTHER scene (D171 §4-§5)
#   C  the fault campaign on the spectral arm at the 25 % level: ideal, collapsed_partition_v1, datagram faults
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${STAGE:?}" "${DATA0:?}" "${DATA1:?}" "${COMMIT:?}" "${REG:?}"
JOBS="${JOBS:-8}" P_A="${P_A:-4}" P_B="${P_B:-8}" SEEDS="${SEEDS:-1 2 3}" STAGES="${STAGES:-A B1 B2 C}"
REPO=/opt/ariadne/repo W=/opt/ariadne/work
step() { echo "=== $(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }
mkdir -p "$W"

step "0 host"
lscpu > "$R/lscpu.txt"; nproc > "$R/nproc.txt"
printf 'commit %s\nseeds %s\nstages %s\n' "$COMMIT" "$SEEDS" "$STAGES" > "$R/run_params.txt"

step "1 docker + image"
command -v docker >/dev/null || { apt-get update -qq && apt-get install -y -qq docker.io > "$R/apt.txt" 2>&1; }
aws ecr get-login-password --region "${AWS_REGION:-eu-north-1}" | docker login --username AWS --password-stdin "$REG" \
  > /dev/null || fail "ecr login"
aws s3 cp "$STAGE/images.txt" "$R/images.txt" --only-show-errors || fail "images.txt"
while read -r ref name want; do
  docker pull -q "$ref" > /dev/null || fail "pull $ref"
  docker tag "$ref" "$name"
  echo "$name $ref $(docker image inspect "$name" --format '{{.Id}}') (local $want)" >> "$R/image_ids.txt"
done < "$R/images.txt"

step "2 checkout + inputs + recordings"
aws s3 cp "$STAGE/repo.bundle" /opt/ariadne/repo.bundle --only-show-errors || fail "bundle"
git clone -q /opt/ariadne/repo.bundle "$REPO" && git -C "$REPO" checkout -q "$COMMIT" || fail "checkout"
aws s3 cp "$STAGE/inputs.tar" - | tar x -C "$REPO" || fail "inputs.tar"
(cd "$REPO" && sha256sum -c --quiet inputs.SHA256SUMS) > "$R/inputs_check.txt" 2>&1 || fail "inputs sha256"
fetch() {   # $1 = S3 prefix, $2 = local dir, $3 = checksum list: GetObject only (no ListBucket on the runner role)
  mkdir -p "$2"
  (cd "$2" && sed -E 's/^[0-9a-f]{64}  \.\///' "$3" \
    | xargs -P 4 -I{} sh -c 'mkdir -p "$(dirname "{}")" && aws s3 cp "'"$1"'/{}" "{}" --only-show-errors') || return 1
  (cd "$2" && sha256sum -c --quiet "$3")
}
REC0=$(cat "$REPO/inputs.rec0") REC1=$(cat "$REPO/inputs.rec1")
fetch "$DATA0" "$REPO/$REC0" "$REPO/rec0.SHA256SUMS" > "$R/rec0_check.txt" 2>&1 || fail "recording 0"
fetch "$DATA1" "$REPO/$REC1" "$REPO/rec1.SHA256SUMS" > "$R/rec1_check.txt" 2>&1 || fail "recording 1"
uid="$(docker run --rm ariadne/core-dev:dev id -u)"; gid="$(docker run --rm ariadne/core-dev:dev id -g)"
chown -R "$uid:$gid" "$REPO" "$W"
export HOME="${HOME:-/root}"
git config --system --add safe.directory "$REPO"
dev() {   # run in the dev container as its user, repo at /work, work dir at /w
  docker run --rm -u "$uid:$gid" -v "$REPO:/work" -v "$W:/w" -w /work \
    -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY=false ariadne/core-dev:dev bash -c "$*"
}

step "3 build (JOBS=$JOBS)"
dev "cmake -S /work -B /work/build/docker -G Ninja -DARIADNE_REQUIRE_OPENCV=ON -DCMAKE_SKIP_RPATH=ON \
     -DFETCHCONTENT_BASE_DIR=/work/.cache/fetchcontent-docker > /dev/null && \
     cmake --build /work/build/docker --target ariadne_collab_passa ariadne_collab_sim --parallel $JOBS" \
  > "$R/build.log" 2>&1 || fail "build (build.log)"
BIN=/work/build/docker/cpp/runtime
AL=/work/.cache/models/aliked_lg_sim_ref EP=/work/.cache/models/eigenplaces_sim_ref
models="--aliked-onnx $AL/aliked_n16_k1024_1280x720.onnx --lightglue-onnx $AL/lightglue_aliked.onnx"
scene_rec() { [ "$1" = 0 ] && echo "/work/$REC0" || echo "/work/$REC1"; }

if [[ " $STAGES " == *" A "* ]]; then
  step "A pass A"
  : > "$W/passa.jobs"
  for sc in 0 1; do for s in $SEEDS; do for ag in 0 1; do
    rec=$(scene_rec $sc)
    echo "$BIN/ariadne_collab_passa --rec $rec --scene $rec/scene --perception-config configs/perception/sim_v1.yaml \
      --slam-config configs/slam/sim_v2.yaml --imu-noise configs/sim/imu_noise_v1.yaml --agent $ag --seed $s \
      --place-onnx $EP/eigenplaces_r50_256_640x360.onnx --place-size 640x360 $models \
      --out /w/passa/s$sc/seed$s > /w/passa_s${sc}_seed${s}_a$ag.log 2>&1" >> "$W/passa.jobs"
  done; done; done
  mkdir -p "$W/passa"; chown -R "$uid:$gid" "$W"
  # One container per job (P_A at a time); OpenCV threads split the cores between them.
  xargs -P "$P_A" -I{} -d '\n' docker run --rm -u "$uid:$gid" -v "$REPO:/work" -v "$W:/w" -w /work \
    -e OPENCV_FOR_THREADS_NUM=$(( $(nproc) / P_A )) ariadne/core-dev:dev bash -c "{}" < "$W/passa.jobs"
  mkdir -p "$R/passa"
  for f in "$W"/passa_*.log; do tail -3 "$f" > "$R/passa/$(basename "$f")"; done
  (cd "$W" && find passa -name 'passa.yaml' -o -name 'results.yaml' | tar cf - -T -) | tar xf - -C "$R/"
  n=$(find "$W/passa" -name 'store_agent*.bin' | wc -l)
  step "A done: $n stores"
  [ "$n" -eq $(( 2 * 2 * $(echo $SEEDS | wc -w) )) ] || fail "pass A: expected every store"
fi

run_b() {   # $1 = run dir name, $2 = scene, $3 = seed, $4 = config file (in /w/cfg)
  echo "$BIN/ariadne_collab_sim --stores /w/passa/s$2/seed$3 --rec $(scene_rec $2) --scene $(scene_rec $2)/scene \
    --config /w/cfg/$4 --seed $3 --out /w/runs/$1 $models > /w/runs/$1.log 2>&1"
}
pool_b() {   # run the job file $1, P_B at a time
  mkdir -p "$W/runs"; chown -R "$uid:$gid" "$W"
  xargs -P "$P_B" -I{} -d '\n' docker run --rm -u "$uid:$gid" -v "$REPO:/work" -v "$W:/w" -w /work \
    -e OPENCV_FOR_THREADS_NUM=2 -e ARIADNE_GIT_COMMIT="$COMMIT" -e ARIADNE_GIT_DIRTY=false \
    ariadne/core-dev:dev bash -c "{}" < "$1"
}
cfg() {   # $1 = name, then key: value lines appended to the shared template
  mkdir -p "$W/cfg"; local f="$W/cfg/$1"; shift
  cp "$REPO/configs/collab/collab_v2.yaml" "$f"
  for kv in "$@"; do echo "$kv" >> "$f"; done
}

if [[ " $STAGES " == *" B1 "* ]]; then
  step "B1 similarity arm"
  cfg sim.yaml "arm: similarity"
  : > "$W/b1.jobs"
  for sc in 0 1; do for s in $SEEDS; do run_b "sim_s${sc}_seed$s" $sc $s sim.yaml >> "$W/b1.jobs"; done; done
  pool_b "$W/b1.jobs"
  step "B1 done"
fi

# R and the response ratio per scene from B1, written as scene<k>.calib: "rate ratio runs" (D171 §4-§5).
calibrate() {
  python3 - "$W/runs" > "$R/calibration.txt" <<'PY'
import csv, sys, glob, os, statistics
root = sys.argv[1]
for sc in (0, 1):
    rates, ratios, runs = [], [], []
    for d in sorted(glob.glob(f"{root}/sim_s{sc}_seed*")):
        if not os.path.isdir(d): continue
        rows = list(csv.DictReader(open(f"{d}/ledger.csv")))
        first = [r for r in rows if r["retransmit"] == "0"]
        if not first: continue
        t0 = min(float(r["t_s"]) for r in rows); t1 = max(float(r["t_s"]) for r in rows)
        fetch = sum(int(r["wire_bytes"]) for r in first if r["class"] in ("3", "4"))
        rates.append(fetch / max(t1 - t0, 1e-9)); runs.append(os.path.basename(d))
        ratio = []
        for k in ("4", "5", "6"):   # match response, pose edge, pose graph: the compressed classes
            rs = [r for r in first if r["class"] == k and r.get("resync", "0") == "0"]
            pay = sum(int(r["payload_bytes"]) for r in rs); tx = sum(int(r["transmitted_bytes"]) for r in rs)
            ratio.append(tx / pay if pay else 0.0)
        ratios.append(ratio)
    if rates:
        m = [statistics.mean(r[i] for r in ratios) for i in range(3)]
        print(f"scene{sc} {statistics.mean(rates):.1f} {m[0]:.5f} {m[1]:.5f} {m[2]:.5f} {','.join(runs)}")
PY
}

if [[ " $STAGES " == *" B2 "* ]]; then
  step "B2 greedy + spectral x budget levels"
  calibrate || fail "calibration"
  cat "$R/calibration.txt" >> "$R/steps.txt"
  : > "$W/b2.jobs"
  for sc in 0 1; do
    other=$(( 1 - sc ))
    read -r _ rate ratio eratio gratio runs < <(grep "^scene$other " "$R/calibration.txt") \
      || fail "no calibration for scene $other"
    for pct in 10 25 50 100; do
      lvl=$(python3 -c "print(round($rate * $pct / 100.0, 1))")
      burst=$(python3 -c "print(round($lvl * 5.0, 1))")
      for arm in greedy spectral; do
        cfg "${arm}_s${sc}_p$pct.yaml" "arm: $arm" "budget_bytes_per_s: $lvl" "budget_burst_bytes: $burst" \
          "budget_source: \"$pct % of the similarity arm's fetch rate $rate B/s on scene $other ($runs)\"" \
          "response_ratio: $ratio" "edge_ratio: $eratio" "graph_ratio: $gratio" \
          "calibration_run: \"scene $other similarity arm ($runs)\""
        for s in $SEEDS; do run_b "${arm}_s${sc}_p${pct}_seed$s" $sc $s "${arm}_s${sc}_p$pct.yaml" >> "$W/b2.jobs"; done
      done
    done
  done
  pool_b "$W/b2.jobs"
  step "B2 done"
fi

if [[ " $STAGES " == *" C "* ]]; then
  step "C fault campaign (spectral, 25 %)"
  : > "$W/c.jobs"
  for sc in 0 1; do
    base="$W/cfg/spectral_s${sc}_p25.yaml"
    [ -f "$base" ] || fail "C needs B2's configs"
    for v in ideal partition faults; do
      f="$W/cfg/camp_${v}_s$sc.yaml"; grep -v '^comms_scenario:' "$base" > "$f"
      case $v in
        ideal) echo "comms_scenario: configs/comms/ideal.yaml" >> "$f" ;;
        partition) echo "comms_scenario: configs/comms/collapsed_partition_v1.yaml" >> "$f" ;;
        faults) printf 'comms_scenario: configs/comms/collapsed_v1.yaml\nfaults: {duplicate_p: 0.1, reorder_p: 0.1, jitter_s: 0.05}\n' >> "$f" ;;
      esac
      for s in $SEEDS; do run_b "camp_${v}_s${sc}_seed$s" $sc $s "camp_${v}_s$sc.yaml" >> "$W/c.jobs"; done
    done
  done
  pool_b "$W/c.jobs"
  step "C done"
fi

step "collect"
mkdir -p "$R/runs" "$R/cfg"
cp "$W"/cfg/*.yaml "$R/cfg/" 2>/dev/null
for d in "$W"/runs/*/; do
  n=$(basename "$d"); mkdir -p "$R/runs/$n"
  cp "$d"/{results.yaml,outcome.yaml,manifest.yaml,ate_vs_bytes.csv,edges.csv} "$R/runs/$n/" 2>/dev/null
  gzip -c "$d/ledger.csv" > "$R/runs/$n/ledger.csv.gz" 2>/dev/null
  tail -2 "$W/runs/$n.log" > "$R/runs/$n/tool.log" 2>/dev/null
done
# The pass-A stores persist for reruns (maintainer: no re-sync per run): under runs/ (the runner role's prefix).
(cd "$W" && find passa -name 'store_agent*.bin' -print0 | xargs -0 sha256sum) > "$R/stores.SHA256SUMS"
step "done"
