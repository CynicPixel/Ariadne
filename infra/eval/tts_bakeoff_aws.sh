#!/bin/bash
# ---------------------------------------------------------------------------
# Narration TTS listening test for the demo film on one AWS GPU (2026-10-27): Ming-3.0-tts-0.5b,
# Chatterbox and Kokoro-82M, open weights only, each in its own venv over one pinned-tag PyTorch image (their pins
# conflict: Ming transformers 4.46.3, Chatterbox 5.2.0). Runs as root on the instance (spinup.sh --profile gpu):
#
#   bash infra/eval/tts_bakeoff_aws.sh
#
# Results ($ARIADNE_RESULTS): steps.txt, nvidia_smi.txt, tts/{*.wav, clips.jsonl, manifest.json, <voice>.log}.
# One stack failing (e.g. Ming's bf16 on a T4, which has no native bf16) is logged and the others still run.
# ---------------------------------------------------------------------------
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
SRC="$(pwd)" IMG=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime   # torch 2.6.0: Ming's and Chatterbox's pin
T0=$(date +%s)
step() { echo "=== $(date -u +%FT%TZ) +$(( $(date +%s) - T0 ))s $*" | tee -a "$R/steps.txt"; }
imds() { local tok; tok=$(curl -s -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
         curl -s -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$1"; }
mkdir -p "$R/tts"

step "0 host + GPU"
nvidia-smi > "$R/nvidia_smi.txt" 2>&1 || { step "FAILED: nvidia-smi"; exit 1; }
step "1 image $IMG"
docker pull -q "$IMG" > "$R/docker_pull.txt" 2>&1 || { step "FAILED: pull"; exit 1; }
digest=$(docker image inspect "$IMG" --format '{{index .RepoDigests 0}}')
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 5 > "$R/gpu_util.csv" 2>&1 & GU=$!

# Inside the container: one venv per voice (--system-site-packages reuses the image's torch), then its clips.
cat > "$R/tts/inner.sh" <<'EOF'
set -uo pipefail
B=/src/infra/eval/tts_bakeoff.py O=/out
export HF_HOME=/opt/hf PIP_ROOT_USER_ACTION=ignore PIP_DISABLE_PIP_VERSION_CHECK=1
apt-get update -qq > /dev/null && apt-get install -y -qq git espeak-ng > /dev/null 2>&1   # espeak-ng: Kokoro's OOV fallback
venv() { python3 -m venv --system-site-packages /opt/v/$1 && /opt/v/$1/bin/pip install -q soundfile "${@:2}"; }
echo "--- kokoro"
( venv kokoro kokoro==0.9.4 && timeout 900 /opt/v/kokoro/bin/python $B kokoro --out $O ) > $O/kokoro.log 2>&1
echo "--- chatterbox"
( venv chatterbox chatterbox-tts==0.1.7 && timeout 900 /opt/v/chatterbox/bin/python $B chatterbox --out $O ) > $O/chatterbox.log 2>&1
echo "--- ming"
( git clone -q --depth 1 https://github.com/inclusionAI/Ming-3.0-tts.git /opt/ming && git -C /opt/ming rev-parse HEAD \
  && venv ming -r /opt/ming/requirements.txt huggingface_hub \
  && (/opt/v/ming/bin/pip install -q pynini==2.1.6 || echo "pynini unavailable: text normalisation off") \
  && /opt/v/ming/bin/hf download inclusionAI/Ming-3.0-tts-0.5b --local-dir /opt/ming-0.5b > /dev/null \
  && timeout 1200 /opt/v/ming/bin/python $B ming --repo /opt/ming --model /opt/ming-0.5b --out $O ) > $O/ming.log 2>&1
for v in kokoro chatterbox ming; do echo "$v: $(/opt/v/$v/bin/pip freeze 2>/dev/null | grep -iE '^(kokoro|misaki|chatterbox-tts|transformers|torch|torchaudio)==' | tr '\n' ' ')"; done > $O/versions.txt
EOF
step "2 voices"
docker run --rm --gpus all -v "$SRC":/src:ro -v "$R/tts":/out "$IMG" bash /out/inner.sh > "$R/tts/inner.log" 2>&1
kill "$GU" 2>/dev/null
step "3 manifest"
extra=$(printf '{"instance_id": "%s", "instance_type": "%s", "image": "%s", "gpu": "%s"}' "$(imds instance-id)" \
  "$(imds instance-type)" "$digest" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)")
docker run --rm -v "$SRC":/src:ro -v "$R/tts":/out "$IMG" bash -c \
  "python3 -m venv --system-site-packages /opt/m && /opt/m/bin/pip install -q soundfile && /opt/m/bin/python /src/infra/eval/tts_bakeoff.py manifest --out /out --extra '$extra'" \
  > "$R/tts/manifest.log" 2>&1
rc=$?
step "done rc=$rc $(ls "$R/tts"/*.wav 2>/dev/null | wc -l) wavs"
exit "$rc"
