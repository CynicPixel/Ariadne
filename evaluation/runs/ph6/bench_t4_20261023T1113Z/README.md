# T4 microbenchmark (RECON_6 §1, D196)

- command: `infra/scripts/kaggle.sh bench evaluation/runs/ph6/bench_t4_20261023T1113Z` at commit `613cd09` (kernel `python/reconstruction/kaggle/bench_kernel.py`)
- GPU: Kaggle Tesla T4 (idle; the demo T4 is shared with gz, so this is an upper bound), torch 2.11.0+cu128, gsplat 1.5.3
  (PyPI source, JIT CUDA build)
- loss: L1 photo + 0.1 depth L1 (no D-SSIM)
- raw: `remote.json` (per point: median/p90 ms of 40 timed iterations after 8 warm-up, peak memory)
- synthetic scene: N Gaussians filling the view (4 m x 2.4 m x 1-6 m slab, ~2 px footprints) [ASSUMPTION: representative]
