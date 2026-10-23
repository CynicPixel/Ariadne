# Trainer smoke 1 (D198)

- command: `infra/scripts/kaggle.sh train-smoke evaluation/runs/ph6/train_smoke_20261023T1129Z` at commit `fbbe641` (kernel `python/reconstruction/kaggle/train_smoke_kernel.py`)
- GPU: Kaggle Tesla T4, torch 2.11.0+cu128, gsplat 1.5.3
- raw: `remote.json`
- result: mechanics OK (submaps, lifecycle, GPU re-anchor invariance), held-out PSNR flat ~17 dB: seeds had a 1-px footprint scale at a 5 % sample -> holes. Superseded by the kNN-scale run.
