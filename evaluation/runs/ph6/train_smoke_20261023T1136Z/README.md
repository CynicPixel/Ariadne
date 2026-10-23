# Trainer smoke 2 (D198)

- command: `infra/scripts/kaggle.sh train-smoke evaluation/runs/ph6/train_smoke_20261023T1136Z` at commit `da9d9d4`
- GPU: Kaggle Tesla T4, gsplat 1.5.3; raw: `remote.json`
- result: kNN scale + 5 % RTG sample + alpha 0.5 mask + sigma-normalised depth: PSNR 13-16 dB (grey floor ~12.9 dB), seeded 0 after the first keyframe, loss dominated by the depth term. Superseded by the Gaussian-SLAM recipe (smoke 3).
