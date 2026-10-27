# Phase 6 splat training on an AWS GPU — first run (2026-10-24): image validated, training VOID (trainer bug)

Run `20261024T115234Z-gpu-130b29`: g4dn.xlarge spot (Tesla T4, driver 595.91.07), eu-north-1a, $0.0199 (756 s).
Code: `583e49b`. Runner: `infra/eval/recon_gpu_aws.sh` (capped at $0.30).
Inputs: image `ariadne-recon-gpu` (sha256:a827de55…, id checked), the 626abe spools
(`runs/20261024T113021Z-cpux86-b83165/output/spools.tar.gz`, scene_id 8410a254a94cc16b checked in every spool).

## Valid
- **The judge-box GPU image works on a real GPU** (`bench/remote.json` env, `train/remote.json`). It runs torch
  2.11.0+cu128, CUDA 12.8 and gsplat 1.5.3 from the compiled `gsplat/csrc.so`, with no JIT build at run time
  (install 0.0 s).
- **gsplat on an AWS T4 equals the Kaggle T4 (D196).** These are D196's micro-bench points (`bench/remote.json`),
  unpacked, train ms/iteration median, AWS vs Kaggle:

  | Resolution | 50k | 100k | 250k | 500k |
  |---|---|---|---|---|
  | 320x180 | 5.7 vs 5.9 | 6.6 vs 6.3 | 9.0 vs 9.1 | 12.8 vs 13.5 |
  | 640x360 | 8.4 vs 9.0 | 10.1 vs 10.9 | 14.1 vs 15.3 | 19.4 vs 20.7 |

  Render times agree within 0.3 ms. This closes HG2's "gsplat on AWS GPU" gap.

## Void: every training number in `train/remote.json`
- **The live budget crashed at keyframe 269** (`KeyError: 999` in `LiveTrainer._step`), so the batch budget never
  ran.
- **Cause.** The trainer keyed its per-keyframe training frames by the keyframe id alone. Each of the 3 agents
  numbers its own keyframes, so the ids collide. A submap closing for one agent deleted another agent's frame
  (the crash). Before that, colliding ids overwrote one agent's training image with another's, so even the smoke
  budget's PSNR is contaminated.
- **Why it never showed before.** Every earlier GPU run (D196/D198 Kaggle smokes) was single-agent, and the
  multi-agent end-to-end runs used the CPU initialiser, which keeps no training frames.
- **Fix.** Frames are keyed by (submap id, keyframe), and a submap belongs to one agent. A failed budget now also
  fails the kernel; before, the run reported rc 0 with the live budget failed.
- The splat outputs of this run are not kept: they're contaminated and stay in S3 only.
