#!/usr/bin/env bash
# The ALIKED input-size knob's models (pipeline profile, A2 2026-10-24): aliked_n16_k1024 at 960x540 and 640x360,
# exported by the SAME exporter and pins as models.sh aliked-lg-sim (1280x720), with --size the only change. The graph
# is fixed-shape by design (the exporter's int() constants: cv::dnn ENGINE_NEW mistypes traced Shape->Gather int64
# arithmetic, SMOKE §4s), so each size is its own ONNX. The reference images are the pinned 1280x720 sim frames
# resized with INTER_AREA - exactly what the bench feeds the model - so the exporter's rewrite checks (rewritten graph
# vs upstream torch, before export) run on the knob's real inputs.
#   evaluation/runs/perf/pipeline_profile_20261024/scripts/export_sizes.sh       (from the repo root)
# Output: .cache/models/aliked_lg_sim_<W>x<H>/ (ONNX, upstream reference tensors, manifest.yaml).
set -euo pipefail
LG_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8 AL_COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
UP="$(realpath -m .cache/upstream)" MODELS="$(realpath -m .cache/models)" IMG=ariadne-model-ref:1
REF="$(realpath -m /data/code/Ariadne/.cache/models/aliked_lg_sim_ref)"   # the pinned 1280x720 export's inputs
[ -d "$UP/lightglue_src/.git" ] || git clone -q https://github.com/cvg/LightGlue.git "$UP/lightglue_src"
git -C "$UP/lightglue_src" checkout -q "$LG_COMMIT"
W="$UP/weights_aliked"; mkdir -p "$W"
fetch() { [ -f "$1" ] || curl -sfL -o "$1" "$2"; echo "$3  $1" | sha256sum -c --quiet; }
fetch "$W/aliked-n16.pth" "https://github.com/Shiaoming/ALIKED/raw/$AL_COMMIT/models/aliked-n16.pth" \
  5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2
fetch "$W/aliked_lightglue.pth" https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth \
  d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
for size in 960x540 640x360; do
  OUT="$MODELS/aliked_lg_sim_$size" FR="$MODELS/aliked_lg_sim_$size.frames"
  rm -rf "$OUT" "$FR"; mkdir -p "$OUT" "$FR"
  docker run --rm --network none -u "$(id -u):$(id -g)" -v "$REF:/ref:ro" -v "$FR:/fr" "$IMG" python -I -c "
import cv2
w, h = map(int, '$size'.split('x'))
for f in ('f200', 'f205', 'f700', 'f710'):
    im = cv2.imread(f'/ref/input_sim_a0_{f}.png', cv2.IMREAD_COLOR)
    assert im.shape == (720, 1280, 3), im.shape
    cv2.imwrite(f'/fr/a0_{f}.png', cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA))
"
  docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/python/model_tools:/tools:ro" \
    -v "$UP/lightglue_src:/lg:ro" -v "$W:/w:ro" -v "$FR:/sim:ro" -v "$OUT:/out" \
    "$IMG" python -I /tools/aliked_lightglue_reference.py --upstream /lg --weights /w --out /out \
    --size "$size" --k 1024 \
    --rgb-image sim_a0_f200=/sim/a0_f200.png --rgb-image sim_a0_f205=/sim/a0_f205.png \
    --rgb-image sim_a0_f700=/sim/a0_f700.png --rgb-image sim_a0_f710=/sim/a0_f710.png \
    --pair sim_a0_f200,sim_a0_f205 --pair sim_a0_f700,sim_a0_f710 \
    --upstream-commit "$LG_COMMIT" --aliked-commit "$AL_COMMIT"
  (cd "$OUT" && sha256sum aliked_n16_k1024_${size}.onnx lightglue_aliked.onnx)
done
