#!/bin/bash
# ---------------------------------------------------------------------------
# Demo-film narration on one AWS GPU (2026-10-27): Ming-3.0-tts-0.5b, built-in voice US_James, the same image, venv
# and pins as the listening test (tts_bakeoff_aws.sh) so the voice matches the take picked in the listening test.
# Runs as root on the instance (spinup.sh --profile gpu):
#
#   CUES=s3://.../cues.jsonl bash infra/eval/narration_aws.sh
#
# Results ($ARIADNE_RESULTS): steps.txt, nvidia_smi.txt, gpu_util.csv, narration/{wav/<id>.wav, takes/, manifest.json,
# ming.log, versions.txt}.
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
: "${CUES:?}"
SRC="$(pwd)" IMG=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
O="$R/narration"
mkdir -p "$O"

step "0 host + GPU"
nvidia-smi > "$R/nvidia_smi.txt" 2>&1 || { step "FAILED: nvidia-smi"; exit 1; }
aws s3 cp "$CUES" "$O/cues.jsonl" --only-show-errors || { step "FAILED: cues"; exit 1; }
step "1 image $IMG"
docker pull -q "$IMG" > "$R/docker_pull.txt" 2>&1 || { step "FAILED: pull"; exit 1; }
extra=$(printf '{"instance_id": "%s", "instance_type": "%s", "image": "%s", "gpu": "%s"}' "$(imds instance-id)" \
  "$(imds instance-type)" "$(docker image inspect "$IMG" --format '{{index .RepoDigests 0}}')" \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)")
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$R/gpu_util.csv" 2>&1 & GU=$!

cat > "$O/inner.sh" <<'EOF'
set -uo pipefail
export HF_HOME=/opt/hf PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update -qq > /dev/null && apt-get install -y -qq git > /dev/null 2>&1
git clone -q --depth 1 https://github.com/inclusionAI/Ming-3.0-tts.git /opt/ming && git -C /opt/ming rev-parse HEAD \
  && python3 -m venv --system-site-packages /opt/v/ming \
  && /opt/v/ming/bin/pip install -q soundfile -r /opt/ming/requirements.txt huggingface_hub \
  && (/opt/v/ming/bin/pip install -q pynini==2.1.6 || echo "pynini unavailable: text normalisation off") \
  && /opt/v/ming/bin/hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir /opt/ming-0.5b > /dev/null \
  && /opt/v/ming/bin/pip freeze | grep -iE '^(torch|torchaudio|transformers|pynini)==' > /out/versions.txt \
  && timeout 3000 /opt/v/ming/bin/python /src/infra/eval/narration_ming.py --cues /out/cues.jsonl --repo /opt/ming \
       --model /opt/ming-0.5b --out /out --extra "$EXTRA"
EOF
step "2 narration"
docker run --rm --gpus all -e EXTRA="$extra" -v "$SRC":/src:ro -v "$O":/out "$IMG" bash /out/inner.sh \
  > "$O/ming.log" 2>&1
rc=$?
kill "$GU" 2>/dev/null
step "done rc=$rc $(ls "$O"/wav/*.wav 2>/dev/null | wc -l) wavs"
exit "$rc"
