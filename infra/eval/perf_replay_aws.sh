#!/bin/bash
# ---------------------------------------------------------------------------
# The pipeline profile's live-pipeline replay (A2, 2026-10-24; M-approved 44-A2-replay-perf; MEASUREMENT ONLY).
# Replays B's ma200 run (20261023T200447Z-cpux86-70a514: 200 s, 3 agents, lockstep, pre-D179) from its recorded
# inputs.mcap - no Gazebo, so the render is excluded and what remains is the pipeline's own cost - with EXACTLY that
# run's code and images: B's repo snapshot (runs/<id>/input/repo.tar.gz), its prebuilt payload (ariadne_live and the
# worker binaries, sha-checked) and its core image by id, invoked as live.sh's cmd_replay does (same flags,
# --memory=4g). Three replays, one variable: the OpenCV pool of the 3 D168 estimator workers (inherited env):
#   asis   OpenCV's default (16 = nproc; what the run recorded)       t5   OPENCV_FOR_THREADS_NUM=5 (3 x 5 = 15)
#   t2     OPENCV_FOR_THREADS_NUM=2
# with infra/eval/perf_sampler.sh beside each (mpstat, pidstat -t, PSI, docker stats, cgroup cpu.stat, perf record -g)
# and a per-symbol perf report. Runs as root on the instance (spinup.sh, cpux86), from this repo's snapshot:
#   RUN_ID=20261023T200447Z-cpux86-70a514 PAYLOAD=<s3 payload.tar.gz> BUCKET=<s3://bucket> [VARIANTS="asis t5 t2"] \
#     bash infra/eval/perf_replay_aws.sh
# Results ($ARIADNE_RESULTS): steps.txt, run_manifest.yaml, replay_<v>/{wall_s.txt, results.yaml, timing.jsonl,
#   agent<k>/{results.yaml, estimate_body.tum}, sampler/...}, b_replay1/ (B's own replay outputs, for comparison)
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${RUN_ID:?}" "${PAYLOAD:?}" "${BUCKET:?}"
SRC="$(pwd)" REP=/opt/rep RUNDIR=build/live/runs/live-20261023T200640Z-62e0 SCENE=build/live/scenes/maze_demo_0
CORE_REF=836505524290.dkr.ecr.eu-north-1.amazonaws.com/ariadne/ariadne-core:live-8668bd7ece68
CORE_ID=sha256:8668bd7ece68de59029fc47289e0034a0647f2dee1c60f96ac52a999e88017fb
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
fail() { step "FAILED: $*"; exit 1; }
kv() { printf '%s: %s\n' "$1" "$2" >> "$R/run_manifest.yaml"; }
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
export HOME="${HOME:-/root}"

step "0 host"
lscpu > "$R/lscpu.txt"
kv run_kind pipeline_profile_live_replay; kv created_utc "$(date -u +%FT%TZ)"; kv replayed_run "$RUN_ID"
kv instance_type "$(imds instance-type)"; kv nproc "$(nproc)"
kv cpu_model "\"$(lscpu | sed -n 's/^Model name: *//p' | head -1)\""

step "1 docker + sysstat + perf"
apt-get -o DPkg::Lock::Timeout=600 update -qq > "$R/apt.txt" 2>&1
apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io sysstat pigz "linux-tools-$(uname -r)" >> "$R/apt.txt" 2>&1 \
  || apt-get -o DPkg::Lock::Timeout=600 install -y -qq docker.io sysstat pigz >> "$R/apt.txt" 2>&1 || fail "apt"
systemctl start docker 2>/dev/null || true
kv perf_available "$(command -v perf >/dev/null && echo yes || echo no)"

step "2 B's core image by id"
REG=${CORE_REF%%/*}
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin "$REG" > /dev/null || fail "ecr login"
docker pull -q "$CORE_REF" > /dev/null || fail "pull $CORE_REF"
[ "$(docker image inspect "$CORE_REF" --format '{{.Id}}')" = "$CORE_ID" ] || fail "core image id differs"
kv core_image_id "$CORE_ID"

step "3 B's snapshot + payload + the recorded run (inputs.mcap)"
mkdir -p "$REP" /tmp/p
aws s3 cp "$BUCKET/runs/$RUN_ID/input/repo.tar.gz" - | tar xz -C "$REP" || fail "snapshot"
aws s3 cp "$PAYLOAD" - | tar xz -C /tmp || fail "payload"
(cd /tmp/p && sha256sum -c --quiet SHA256SUMS) || fail "payload sha256"
cp -a /tmp/p/. "$REP/"
kv payload "$PAYLOAD"; kv payload_sha256sums_sha256 "$(sha256sum /tmp/p/SHA256SUMS | cut -d' ' -f1)"
# GNU tar exits 2 when an optional member pattern is absent (shadow_map_config.yaml is not in every run: job 44's
# first attempt, 20261024T021800Z-cpux86-716179, failed exactly so), so only the S3 stream's status is fatal; the
# files that are required are checked right below.
aws s3 cp "$BUCKET/runs/$RUN_ID/output/results/run.tar.gz" - | pigz -dc | tar x -C "$REP" --wildcards --no-anchored \
  'inputs.mcap' 'directives.jsonl' 'trace.jsonl' 'live_config.yaml' 'planner_config.yaml' 'shadow_map_config.yaml' \
  "replay1/*results.yaml" "replay1/*estimate_body.tum" "replay1/*timing.jsonl" 2> "$R/run_tar.err"
st=("${PIPESTATUS[@]}"); echo "aws ${st[0]} pigz ${st[1]} tar ${st[2]}" >> "$R/run_tar.err"
[ "${st[0]}" = 0 ] && [ "${st[1]}" = 0 ] || fail "run.tar.gz stream (run_tar.err)"
[ -f "$REP/$RUNDIR/live_config.yaml" ] || fail "run.tar.gz: no live_config.yaml"
LIVE_DIR=$(cd "$REP/$RUNDIR" && ls -d */inputs.mcap | head -1 | xargs dirname)
[ -n "$LIVE_DIR" ] || fail "no inputs.mcap"
kv inputs_mcap_sha256 "$(sha256sum "$REP/$RUNDIR/$LIVE_DIR/inputs.mcap" | cut -d' ' -f1)"
mkdir -p "$R/b_replay1"
(cd "$REP/$RUNDIR/replay1" && find . -name results.yaml -o -name estimate_body.tum -o -name timing.jsonl | tar cf - -T -) \
  | tar xf - -C "$R/b_replay1"
chown -R 1000:1000 "$REP"
CFG=$RUNDIR/live_config.yaml
EXTRA=(--directives "$RUNDIR/$LIVE_DIR/directives.jsonl")
grep -q '^chooser: *llm' "$REP/$CFG" && EXTRA+=(--decisions "$RUNDIR/$LIVE_DIR/trace.jsonl")
[ -f "$REP/$RUNDIR/planner_config.yaml" ] && EXTRA+=(--planner-config "$RUNDIR/planner_config.yaml")
[ -f "$REP/$RUNDIR/shadow_map_config.yaml" ] && EXTRA+=(--shadow-map-config "$RUNDIR/shadow_map_config.yaml")
printf '%s\n' "${EXTRA[@]}" > "$R/replay_extra_args.txt"

# Symbols for the host-side perf report: the replay's binaries live in the container (/work = B's snapshot, OpenCV
# under /opt/opencv5 in the core image), gone when it exits. /work -> $REP and the image's /opt/opencv5 copied out
# make the same paths resolve on the host (job 44b's first report came back empty: the call-graph report over 346k
# samples with unresolvable paths ran into its timeout).
[ -e /work ] || ln -s "$REP" /work
if [ ! -d /opt/opencv5 ]; then
  cid=$(docker create "$CORE_REF") && docker cp "$cid:/opt/opencv5" /opt/opencv5 > /dev/null; docker rm "$cid" > /dev/null
fi
export SAMPLER_PERF_ARGS="${SAMPLER_PERF_ARGS:--a -F 199}" SAMPLER_NO_REPORT=1
kv sampler_perf_args "\"$SAMPLER_PERF_ARGS\""
for v in ${VARIANTS:-asis t5 t2}; do
  step "R replay $v"
  env=()
  [ "$v" = t5 ] && env=(-e OPENCV_FOR_THREADS_NUM=5)
  [ "$v" = t2 ] && env=(-e OPENCV_FOR_THREADS_NUM=2)
  O="$R/replay_$v"; mkdir -p "$O"
  bash "$SRC/infra/eval/perf_sampler.sh" "$O/sampler" & SP=$!
  sleep 3
  t=$(date +%s.%N)
  docker run --rm --memory=4g --memory-swap=4g -u 1000:1000 "${env[@]}" -v "$REP":/work -w /work "$CORE_REF" \
    build/docker/cpp/runtime/ariadne_live --scene "$SCENE" --perception-config configs/perception/sim_v1.yaml \
    --slam-config configs/slam/sim_v1.yaml --mission-config configs/mission/mission_v1.yaml --live-config "$CFG" \
    --out "$RUNDIR/perf_$v" --replay "$RUNDIR/$LIVE_DIR/inputs.mcap" "${EXTRA[@]}" > "$O/replay.log" 2>&1
  rc=$?
  awk -v a="$t" -v b="$(date +%s.%N)" -v rc="$rc" 'BEGIN { printf "%.1f rc=%s\n", b - a, rc }' > "$O/wall_s.txt"
  kill -TERM "$SP"; wait "$SP" 2>/dev/null
  if [ -s "$O/sampler/perf.data" ]; then
    for sort in dso,sym pid,dso,sym; do   # whole box; per thread (perf's `pid` key is the thread: tid:comm)
      timeout 900 perf report -i "$O/sampler/perf.data" --stdio --no-children -g none --sort "$sort" \
        --percent-limit 0.2 2> "$O/sampler/perf_report_${sort//,/_}.err" | head -600 > "$O/sampler/perf_report_${sort//,/_}.txt"
    done
    timeout 300 perf report -i "$O/sampler/perf.data" --stdio --no-children -g none --sort comm,dso \
      2>/dev/null | head -200 > "$O/sampler/perf_report.txt"
    ls -la "$O/sampler/perf.data" > "$O/sampler/perf_data_size.txt"   # kept: uploaded with the results
  fi
  d=$(ls -d "$REP/$RUNDIR/perf_$v"/*/ | head -1)
  cp "$d/results.yaml" "$d/timing.jsonl" "$d/manifest.yaml" "$O/" 2>/dev/null
  for a in "$d"/raw/*-agent*/; do
    n=$(basename "$a" | sed -E 's/.*-(agent[0-9]+)-.*/\1/'); mkdir -p "$O/$n"
    cp "$a/results.yaml" "$a/estimate_body.tum" "$a/manifest.yaml" "$O/$n/" 2>/dev/null
  done
  step "R $v done: $(cat "$O/wall_s.txt")"
done
kv total_seconds "$(( $(date +%s) - T0 ))"
step "done"
