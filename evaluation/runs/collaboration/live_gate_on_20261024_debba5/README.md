# Recorded live run, three robots, known-entry gate ON (collab60e)

- **AWS run:** `20261024T085818Z-cpux86-debba5` (c7i.4xlarge spot, 1,871 s, $0.07); recording at `s3://ariadne-836505524290-eun1/runs/20261024T085818Z-cpux86-debba5/`.
- **Scene and seed:** identical to the gate-off run, `sim-live/8410a254a94cc16b` with seed 1 (`manifest.txt`).
- **Gate outcome** (`gate_report.json`):
  - 473 verification calls;
  - 9 candidates passed geometry, and 8 reached the known-entry gate. **The gate refused all 8**, so no inter-robot edge was kept.
- **Accuracy:** every robot stayed within **7.5 cm** of ground truth in the shared frame. Per-robot maxima were 0.075, 0.065 and 0.058 m over 1,200 samples each.
- **Comparison run:** the gate-off run on the same scene and seed, `../live_gate_off_20261024_c51ab2`.
