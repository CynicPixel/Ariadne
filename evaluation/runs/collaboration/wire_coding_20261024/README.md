# Phase 3.6 — local-feature wire codings on real verify pairs (A2, 2026-10-24)

Design `docs/design/wire-coding.md`. Rule pre-registered as D233 (M-approved before the run). Result: **D234**.
Measurement only; no coding was adopted.

## Run
- `20261024T053045Z-kaggle-a2-bytes36`: Kaggle CPU kernel via test-compute's off-AWS harness
  (`infra/eval/ci_kaggle_launch.sh`), 4 vCPU AMD EPYC 7B12 (AVX2), `OPENCV_FOR_THREADS_NUM=4`, commit `13cad72`,
  opencv build `fnv1a64:cf8ea084bb984703`. Bench 1535 s. Cost $0.
- The harness manifest is `raw_kaggle/kaggle_manifest.yaml`. It records the command verbatim, the image, the inputs
  and the platform.

## Command (verbatim, from the manifest)
    build/docker/cpp/runtime/ariadne_perf_bench bytes --stores-root build/bytes_inputs/stores \
      --pairs-dir evaluation/runs/performance/pipeline_profile_20261024/inputs --collab-config configs/collab/collab_v2.yaml \
      --seed 1 --aliked 1280x720=.cache/models/aliked_lg_sim_ref/aliked_n16_k1024_1280x720.onnx \
      --lightglue .cache/models/aliked_lg_sim_ref/lightglue_aliked.onnx --out $CI_OUT/bytes
    # defaults: --pca-fit-scene 1 --reseeds 4

## Inputs
- Pairs: `evaluation/runs/performance/pipeline_profile_20261024/inputs/pairs_s{0,1}.csv`. These are 120 real verify
  calls of run `0345a5` (`cal_s*_seed1`): per scene 30 verified + 6 per rejection reason (`pairs.py`, seed
  20261024).
- Stores: run `0345a5` pass A, `s{0,1}/seed1/store_agent{0,1}.bin`. sha256 values are in
  `raw_kaggle/stores.SHA256SUMS` (prefixes fd4e0c47, 2131da23, 74dfd305, a6110547) and match
  `inputs/stores_seed1.SHA256SUMS` of the profile.
- Models: `raw_kaggle/models.SHA256SUMS` (the `models.sh aliked-lg-sim` pins).
- Verify config: `configs/collab/collab_v2.yaml`, seed 1 (the fleet's per-pair seed).

## Files
- `raw_kaggle/` holds the bench outputs:
  - `pairs.csv`: per pair × setting × seed offset; matches, inliers, reason, GT errors.
  - `bytes_kf.csv`: per keyframe × setting; WireCodec(kFragmented) payload, zstd and wire bytes, plus today's
    predictor's estimate.
  - `geometry.csv`: the geometry codings' decode error.
  - `pca.yaml` and `pca_basis.bin`.
  - `bench.log`.
- `order_check.txt`: cv::ALIKED emits unsorted (515 inversions), so K 512 can't be taken from the stores (D233).
- `processed/` is written by `python3 -I scripts/analyze.py --raw raw_kaggle --out processed`:
  - `precondition.csv`, `seed_envelope.csv`, `bytes_summary.csv`, `coding_verdicts.csv` (D233 verbatim);
  - `combination.csv` (empty: nothing adopted);
  - `posthoc_vs_fp32.csv` (POST-HOC, report-only).
- `figures/fig_bytes.png` is written by `scripts/plot.py` in `ariadne-analysis:1` (command in the script header).

## Result (D234)
- **Precondition passes.** R (int8/1024/xyz) reproduces production on 120/120 pairs.
- **The seed envelope is zero.** Every pair is seed-stable.
- **The fp32 control fails bar 1.** It has 1 reject→accept per scene against int8, so by D233 nothing is adopted
  without M.
- **Every compression coding misses bar 1**, with one codec flip in at least one scene. Inlier ratios stay
  ≥ 0.995 (bar 2). The descriptor codings clear bar 3 (−56 to −102 kB/KF); the geometry codings don't (−8.5 to
  −12.9 kB).
- **Post-hoc** (not part of the rule): int4 (79.4 kB/KF) and bin1 (33.6 kB/KF) give fp32's exact verdict on 120/120
  pairs, while today's int8 (135.7 kB) differs from fp32 on 2. The lossless z32 also flips one pair, so a 0-flip
  bar on this sample sits below its numeric resolution.

## Honesty notes
- This is a screen. 0 flips bound the per-reason-stratified rate at ≤ 10 % per scene and direction (95 %, rule of
  three). It is not a population rate.
- Both endpoints take the codec's round trip, as today's fleet does. One-way verification can only do better.
- Codings the current `wire.fbs` has no field for are packed into the descriptors vector, a [PROXY] within ±16 B/KF.
- Scene 0's accepted edges are mostly GT-false (D176 aliasing: R has 14 GT-true of 30 accepted), so its pose medians
  (~88-90°) describe the aliasing, not the codings.
