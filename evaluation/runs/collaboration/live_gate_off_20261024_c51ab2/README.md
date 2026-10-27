# Recorded live run, three robots, known-entry gate OFF (collab60d)

- **AWS run:** `20261024T031036Z-cpux86-c51ab2` (c7i.4xlarge spot, 2,018 s, $0.07), the recording of `s3://ariadne-836505524290-eun1/runs/20261024T031036Z-cpux86-c51ab2/`.
- **Scene:** `sim-live/8410a254a94cc16b` (maze_demo_0). Seed 1, 60 s, lockstep.
- **Inter-robot verification** (`collab_edges.csv`, one row per candidate, with ground-truth error):
  - 491 candidates;
  - 459 refused (essential gate 355, degeneracy 90, PnP gate 11, too few matches 3);
  - 32 passed every geometric check, and **all 32 are false**: wrong by 14.8–18.7 m and 64–157°.
- **Outcome:** a false merge at t = 40 s puts robot 2 more than 1 m off from t = 45.1 s, and it ends 15.26 m off. Robots 1 and 3 stay within 7.9 cm.
- **Figure:** `docs/report/figures/data/collab60d_err.csv` (report Fig. "Why the known-entry gate exists").
- **Comparison run:** the same scene and seed with the gate on, `../live_gate_on_20261024_debba5`.
