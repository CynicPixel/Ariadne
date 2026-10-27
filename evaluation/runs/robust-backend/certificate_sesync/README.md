# D74 certificate on the large SE-Sync graphs

`Certificate.SeSyncBenchmarkGraphs`, one graph per container, in this order: sphere2500, torus3D, cubicle, rim,
grid3D. Command, binary, image, graph digests and the grid3D go/no-go rule (written before its run): `manifest.yaml`.
Raw output: `<graph>.log` (the test's line, the gtest exit code, the cgroup `memory.peak`, `oom_kill`).

| graph | f (Table 1) | verdict | f − f* ≤ | peak | wall | gtest |
|---|---|---|---|---|---|---|
| sphere2500 | 1687.01 (1.687e3) | certified | 1.1e-3 | 79 MB | 20 s | pass |
| torus3D | 24227 (2.423e4) | certified (LM at its 200-iteration cap) | 3.6e-3 | 152 MB | 180 s | pass |
| cubicle | 717.126 (7.171e2) | certified | 2.9e-2 | 215 MB | 16 s | pass |
| rim | 5460.9 (5.461e3) | **not certified**: λ_min ≤ −0.233 at η 9.8e-6 | — | 348 MB | 89 s | **fail** |
| grid3D | 84319 (8.432e4) | certified (LM at its cap) | 0.13 | 616 MB | 7176 s | pass |

`run_one.sh` reports `docker exit 0` even when gtest fails, so the `exit <rc>` line in each log is the test
result: rim's is 1. Interpretation and the rule against re-tuning: DECISIONS D74.
