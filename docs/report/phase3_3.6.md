# Phase 3.6 — descriptor compression (section for the Phase 3 report; A2, 2026-10-24)

Tags as in the other report files: **measured** = a run we can point to. Evidence:
`evaluation/runs/ph3/bytes_3.6_20261024/` (§36-complete: README, command, raw, processed, `scripts/analyze.py`,
`scripts/plot.py`, `figures/fig_bytes.png`). Decisions: D233 (rule, pre-registered and approved before the run) and
D234 (result).

**Global descriptor** (measured, D93, earlier): int8 with a per-vector scale (258 B) leaves every query's top-k
unchanged against fp32 (1024 B) on EuRoC. This is the wire candidate (arch §17.2).

**Local features** (measured, D234; Kaggle CPU run `20261024T053045Z-kaggle-a2-bytes36`):
- **Data.** 120 real verify calls (run `0345a5`: per scene 30 verified + 6 per rejection reason), taken from the
  stores production verified, through the fleet's exact verify path.
- **Production's int8 local descriptors differ from uncompressed fp32 on 2 of the 120 pairs:** one per scene,
  where fp32 accepts and int8 rejects. Of the two edges, scene 0's is GT-true and scene 1's is GT-false. This is a
  measured property of today's wire format.
- **Today's payload** (int8, 1024 keypoints, float XYZ) is **135.7 kB per keyframe on the wire** (WireCodec,
  fragmented, zstd ratio ~0.88).
- **Candidate codings and bytes per keyframe:** int4 79.4 kB, 1-bit 33.6 kB, PCA-64 77.6 kB, PCA-32 47.9 kB.
  Geometry codings (disparity, u16 keypoints) save only 8.5–12.9 kB.
  - All keep PnP inlier counts within 0.5 % of int8.
  - Each has one accept/reject change against int8 in at least one scene.
- **Result.** Under the pre-registered rule (0 flips against int8, with fp32 as the control), **no coding is
  adopted**. The fp32 control itself fails the 0-flip bar, so int8 is already at the resolution this sample has.
- **Post-hoc, report-only** (not part of the rule):
  - int4 and 1-bit give fp32's exact verdicts on all 120 pairs.
  - A lossless coding (identical points to the last float bit) flips one borderline pair. A 0-flip bar is
    therefore below this sample's numeric floor.
  - A re-test with fp32 as the reference is on the post-demo backlog.
- **Keypoint count** (measured, pipeline profile job 70): K 512 loses 4 of 63 verified pairs at a median inlier
  ratio of 0.53, so it is not adopted.
- **Reach.** 0 flips in 120 pairs bounds the per-reason-stratified flip rate at ≤ 10 % per scene and direction
  (95 %). This is a screen, not a population rate.
