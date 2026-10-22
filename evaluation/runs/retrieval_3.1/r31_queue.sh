#!/usr/bin/env bash
# 3.1 CLAHE ablation queue (RETRIEVAL_3.1_3.6 §3): 3 conditions x 11 EuRoC flights, ONE process at a time, then the
# built-in regression (clean ep376.f32 == D62's dump, byte for byte) and the rows.
set -uo pipefail
WT=/data/code/Ariadne-2.1
cd "$WT"
OUT=build/a1/r31
for cond in clean dark1_dust0 dark2_dust0_ag8; do
  for s in MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult \
           V2_01_easy V2_02_medium V2_03_difficult; do
    deg=()
    [ "$cond" = clean ] || deg=(--degradation /work/configs/degradation/$cond.yaml --seed 20261022)
    .cache/feat-2.1-perception/dev.sh /work/build/docker/cpp/runtime/ariadne_place_eval \
      --sequence /work/.cache/datasets/euroc/$s --perception-config /work/configs/perception/euroc.yaml \
      --export /work/.cache/models/eigenplaces_ref --out /work/$OUT/$cond/$s --retrieval-3.1 "${deg[@]}" 2>&1 \
      | grep -v WARN
  done
done
for s in build/a1/place/*/; do
  n=$(basename "$s")
  cmp -s "$s/ep376.f32" "$OUT/clean/$n/ep376.f32" && echo "regression $n: identical" || echo "regression $n: DIFFERS"
done
echo QUEUE_DONE
