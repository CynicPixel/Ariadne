# Trained splat of the generated maze, for the demo film's beat 10 (A2, 2026-10-27 IST)

**Label (demo-video-session's): "trained Gaussian splat of the generated maze (rendered views, not a robot run)".**

- **What.** gsplat gradient optimisation of a seed splat against 76 WorldGen-rendered camera views (320×180, roof
  omitted) of scene `8410a254a94cc16b`. These are synthetic views of the generated world. No robot recorded them, and
  no SLAM pose is involved.
- **Why.** The maintainer asked on 2026-10-27 for the film's closing fade-in and 360° turnaround. They approved the
  spend directly; demo-video-session prepared the input.
- **Input.** `s3://ariadne-836505524290-eun1/runs/video-worldgen-gsplat-20261026/input/data.tar.gz`. It carries
  `views.npz`, `seed.splat` (sha256 04d1b678…), `data_manifest.json`, and the script
  `python/video/tools/train_worldgen_gsplat.py`.
  - The script ran byte-for-byte: sha256 `7d8da02a…`, recorded in `run_manifest.yaml`.
  - That script is demo-video-session's and is not committed on this branch.
- **Runner.** `infra/eval/recon_gpu_aws.sh` WORLDGEN mode, code `8a47c33`, through `run.py` (committed here).
  - The recon-gpu image has no Pillow; `train.log` line 1 confirms it.
  - The script uses Pillow only to save the held-out preview PNGs. `run.py` substitutes a cv2-backed
    `Image.fromarray(...).save` for it. Training is unaffected.
- **Run.** `20261026T211816Z-gpu-2705f3`: g4dn.xlarge spot, eu-north-1b, Tesla T4, image `sha256:a827de55…`.
  - 680 s instance time, **$0.0230** (`ledger_row.csv`).
  - Two launches before it got no spot capacity (eu-north-1c, then 1a). No instance started for either, $0.
- **Command (in the container).** `train_worldgen_gsplat.py train --data /data --out /out --steps 3000`, torch seed 626.
- **Result (`manifest.json`).**
  - 3,000 steps in 21.9 s on 187,405 Gaussians.
  - 70 training views and 6 held-out views (every 12th view, offset 5).
  - **Held-out PSNR: 23.12 dB before, 24.36 dB after (+1.25 dB).**
  - Output `trained_worldgen.splat` sha256 `f8a69684…`.
- **Caveats.**
  - The log samples one random training view every 100 steps, so its `depth_l1` swings between about 0.04 and 0.38
    from view to view. That is not a held-out measure, and no depth metric on held-out views was computed.
  - In `heldout_017.png`, the walls render lighter and more translucent than in `target_017.png`.
- **Kept outside git.** `trained_worldgen.splat` (6 MB) is in S3 under the run's output prefix and in
  `/data/code/Ariadne/build/video_assets/splat_626abe_render_20261024_aws/worldgen_trained/`, which is
  demo-video-session's copy.
