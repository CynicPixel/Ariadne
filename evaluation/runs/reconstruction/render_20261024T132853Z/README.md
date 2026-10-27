# Demo-video renders of the trained splat (2026-10-24)

**Label: "trained Gaussian splat, novel viewpoints (rendered)".**

- **Source.** The batch-budget splat of run `20261024T121121Z-gpu-ae604c` (`../gpu_aws_20261024T121121Z/`):
  2,590,315 Gaussians in 51 submaps, trained on the 200 s 3-agent maze_demo_0 run 626abe.
- **Renderer.** `python/reconstruction/render/render_gsplat.py` (gsplat 1.5.3 rasterization) on an AWS g4dn T4.
  - Run `20261024T132853Z-gpu-1e9c60`, $0.0198, via `infra/eval/recon_gpu_aws.sh` RENDER mode, code `cfea45b`.
  - An earlier attempt, `20261024T131706Z-gpu-3c7b01` ($0.0165), failed on a `backgrounds` shape assertion that was
    fixed in `cfea45b`.
- **Command (on the instance):**
  `render_gsplat.py --splat <splat_batch> --out <render> --seconds 9 --fps 30 --w 1920 --h 1080`.
- **Committed here:** `poses.json` (every image's camera in mission world W), the 4 stills (60° elevation, poses no
  ground agent occupied), the per-agent and merged overview (`align/`), plus the log, steps, manifest and GPU
  utilisation.
- **Kept outside git:** the 270 orbit frames (244 MB with the rest). They are in S3 under the run's output prefix and
  in `/data/code/Ariadne/build/video_assets/splat_626abe_render_20261024/`, which is demo-video-session's copy.
- **Quality.** Training-view PSNR only: 27.4 dB batch, NOT held-out. There is no held-out metric for these views,
  and the holes and floaters visible in them are real.
