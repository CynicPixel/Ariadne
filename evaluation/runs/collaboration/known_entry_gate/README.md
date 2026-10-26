# Known-entry gate sweep (D178): pre-registered, PASS

| run | what | where |
|---|---|---|
| `20261023T191248Z-cpux86-eee456` | baseline sweep with the gate **off** (the 3.78 m aliasing failure) | `s3://ariadne-836505524290-eun1/runs/20261023T191248Z-cpux86-eee456/`; its per-cell ATE is the `ate_base_m` column of `verdict/cells.csv` |
| `20261023T213707Z-cpux86-e31421` | the same cells with the gate **on** | `gated_e31421/` |
| `20261024T042708Z-cpux86-4ec3f6` | logged rerun, identical to the gated run on every cell, with each candidate's d² | `logged_4ec3f6/` |

- **Verdict** (`verdict/verdict.yaml`):
  - scene-0 budgeted cells: worst seed-mean merged ATE is 0.014675 m, down from 3.78 m;
  - scene 1 is not worse in any of 7 cells;
  - no run is corrupt;
  - **0 of 560** tested true edges were rejected.
- **Separation of true and false edges**, from `docs/report/figures/data/chi2_separation.csv`: all 560 true edges have d² ≤ 1.1, and all 1,164 false edges have d² ≥ 6,175, against a cut of χ²₆(0.999) = 22.46.
- **Calibration:** `entry_nees_scene1.yaml`. The run command is in `command.txt`.
