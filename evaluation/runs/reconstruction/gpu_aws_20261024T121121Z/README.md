# Phase 6 splat training on an AWS GPU, rerun with the keying fix (2026-10-24)

Run `20261024T121121Z-gpu-ae604c`: g4dn.xlarge spot (Tesla T4), eu-north-1a, $0.0673 (2560 s). Code
`3da19cc`: FrameStore keyed by (submap, kf), with CPU unit tests; recon python 40/40. Runner
`infra/eval/recon_gpu_aws.sh`. The first run, `gpu_aws_20261024T115234Z/`, is void.

Inputs:
- image `ariadne-recon-gpu` sha256:a827de55… (id checked; compiled gsplat csrc, no JIT);
- spools of the ma200 rerun `20261024T022434Z-cpux86-626abe` (tree 95544d6, contains the D225 fix 0105f5e; lens ON;
  lit maze_demo_0, lamps idle), built in-region (`20261024T113021Z-cpux86-b83165`), scene_id 8410a254a94cc16b
  checked in every spool;
- training image: RGB (D199).

| Budget | Iterations/KF | Keyframes | Submaps | ms/KF median (p95) | ms/iteration | Training-view PSNR, before → after (median) |
|---|---|---|---|---|---|---|
| smoke | 15, first 120 KF | 120 | 5 | 171 (220) | 7.8 | 30.4 → 30.7 dB |
| live | 15 | 1,476 | 51 | 203 (276) | 8.5 | 23.1 → 25.7 dB |
| batch | 100 | 1,476 | 51 | 978 (1,103) | 8.6 | 23.3 → 27.4 dB |

- **PSNR is TRAINING-VIEW, NOT held-out.** Each value is the masked PSNR of a submap's render at the keyframe it
  just trained on, against that keyframe's own image, taken before and after its iterations (D198's diagnostic).
  It measures fitting, not novel-view quality. There is no held-out protocol for real spools yet.
- Keyframes per agent: 568 / 482 / 426. That is every keyframe of the 200 s, 3-agent run. The freeze check is
  bitwise in every budget.
- **Live training speed** (the demo limitation): at the live budget the T4 trained 1,476 keyframes in 350 s for a
  200 s mission, 0.57× real time. Mission speed would need ~7.4 KF/s at ≤ 135 ms/KF; the live budget measured
  203 ms/KF median.
- The batch splat holds 2.59 M Gaussians in 51 submaps, 83 MB of chunks.
  - `s3://ariadne-836505524290-eun1/runs/20261024T121121Z-gpu-ae604c/output/results/train/splat_batch.tar.gz`,
    sha256 8bc2db12…
  - The live and smoke splats sit beside it.
  - These are left in S3 (`SPINUP_FETCH_EXCLUDE`), and only the batch splat was fetched, for the renders.
- `bench/remote.json` repeats the AWS T4 micro-bench (D196 points), matching the first run.
