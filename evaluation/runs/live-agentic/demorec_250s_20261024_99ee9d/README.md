# Recorded 250 s three-robot mission with the gate on (demo recording)

- **Live leg:** `20261024T123707Z-cpux86-99ee9d` (c7i.4xlarge spot, 6,448 s, $0.23).
- **Analysis pack:** `20261024T221611Z-cpux86-285fd8` ($0.02), built from the recording with no replay. Recording and pack are at `s3://ariadne-836505524290-eun1/runs/20261024T123707Z-cpux86-99ee9d/` and `s3://ariadne-836505524290-eun1/runs/20261024T221611Z-cpux86-285fd8/output/`.
- **Gate and verification** (`gate_report.json`):
  - the known-entry gate tested 11,251 candidate checks and rejected 2,686;
  - there were 3,610 verification calls, 350 of which passed geometry;
  - **291 inter-robot edges were kept** by t = 245 s, and all three robots merged into one frame.
- **Accuracy against ground truth** in the shared frame, max / final, over 5,000 samples per robot:
  - robot 1: 0.221 / 0.181 m;
  - robot 2: 0.219 / 0.130 m;
  - robot 3: 0.194 / 0.188 m.
  No aliased merge happened in 250 s.
- **Replay:** the first replay of this recording ran out of memory. Three free-running verification workers filled the 20 GB container, which the cgroup OOM report confirms. Bounding that inbox is listed in `docs/failure-cases.md`. The live leg itself completed with exit code 0.
