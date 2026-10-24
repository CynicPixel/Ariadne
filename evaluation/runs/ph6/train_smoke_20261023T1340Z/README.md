# Trainer smoke 3 (D198)

- command: `infra/scripts/kaggle.sh train-smoke evaluation/runs/ph6/train_smoke_20261023T1340Z` at commit `c09d202`; Kaggle Tesla T4, gsplat 1.5.3; raw `remote.json`
- variants at 320x180, synthetic corridor (exact depth, analytic albedo), 60 KF, 30-KF submaps: gslam_batch (100/100 iterations), live_budget (15), gslam_batch_nodepth (diagnostic)
- reported in D198
