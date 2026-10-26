# Phase 5 — Degraded visual environments (dark, lamp-lit, dust)

DRAFT (A2, 2026-10-24). Sources are the decision entries (`docs/decisions.md`) and the evidence directories named inline. Tags: **measured**
= a run we can point to; **[DESIGN]** = built and unit-tested, not yet measured end to end; **[PROJECTION]** = a
derived number. Nothing here is a projection presented as measured.

**Status: built; exit gate NOT closed (D149).** On 2026-10-23 the maintainer ruled the dark, lamp-lit and dusty runs
STRETCH goals, to be taken up after the live demo runs in real time (D199). Dark/DVE work has been paused in every
session since (STATUS 2026-10-24). The bullets that need them are deferred by that ruling, not dropped.

### What was built
- **Radiometric pipeline at the frame boundary.**
  - The sim render is sRGB-encoded. D gains an opt-in decode (D130).
  - Stage 3 falloff normalisation produces an albedo proxy at keyframes (D134, D140, D146).
  - Stage 4 exposure normalisation (D132).
- **Lamp model and calibration.**
  - The rig lamp is inverse-square, wide and shadow-casting, as rendered (D131).
  - A lamp calibration procedure, executed for `rig_stereo_v2` (D133).
- **5.3 exposure + lamp co-controller.**
  - Runs at keyframes, with the applied state on every frame and known-gain LK (D132, D147).
  - Replay trace checks pass, 171/171 and 168/168.
  - The Failure E scenario recovers in 2.35 s (D147 Results).
  - The live `light_config` sink is B's and is not wired.
- **Worlds.**
  - `ci_tiny_dark` (co-moving lamp) and its fixed-light twin, rendered on g4dn spot (D137).
  - `confined_v1` and a one-pass baseline-sweep rig (D138).
  - Dust options (D135/D139): built; the measurement is pending, stretch.

### What was measured (with the negatives)
- **Lighting rows** (D142; AWS c7i spot, commit `380c900`; `evaluation/runs/perception/radiometric/lighting/`).
  - On the same path, dark + co-moving lamp is ~6× worse in ATE than dark + fixed light: 15.2 vs 2.5 mm (lit 3 mm).
    It also takes ×2.8 the keyframes.
  - Brightness and coverage were not matched, so the gap is **not attributed** to the lamp's motion (D145).
  - Sensor noise adds 2–3 mm.
- **§31.5 illumination rows** (D145–D147; AWS run `20261023T113916Z-cpux86-7b3f1b`, commit `649f156`;
  `evaluation/runs/perception/radiometric/d145_rows_20261023T113916Z/`). Ten rows of ATE, keyframes and lost frames, with the
  negatives kept:
  - features computed on the albedo proxy: +2.9 mm ATE (worse);
  - lamp at 15 %: 1.32 m.
  - Not measured: per-stage repeatability, match precision and loop-closure precision. These are a CPU row set
    still to be defined through preflight.
- **Real-data validation (5.6) is negative** (D141, D143, D144; `evaluation/runs/perception/radiometric/hilti_5.6/`).
  - OIVIO was offline, so the maintainer-approved substitute was Hilti 2023 `site2_robot_3`, with equidistant
    cameras through `cv::fisheye`.
  - All three rows score **0/100**: sim-tuned, EuRoC-tuned, and EuRoC + radiometric. GCP rms is 10–229 m and
    14–22 % of frames are lost.
  - The losses fall in windows where the robot turns with a dark far field. This is recorded as failure case
    FC-5.6-1.
  - The estimate is wrong; the scoring isn't (rigid-invariant GCP distances were checked).
  - Static ring edges were measured and rejected as the mechanism. The root cause is open. There was no re-tuning
    after the failure.
- **Corridor degeneracy gate ordering (5.5):** pre-registered (D148) and not measured. The sweep render failed:
  first a TF bug (fixed `82ffa00`), then a writer backlog. A 2-camera `confined_v1` render was approved, and then
  the work was paused.

### Spend
- AWS CPU ≈ $0.19 (D142 ≈ $0.07 plus the 2026-10-23 rows $0.1156).
- AWS GPU per D149: $0.2122 (renders $0.1241 plus the failed sweep render $0.0881). Add the spot-reclaimed
  `confined_v1` render reported afterwards, ~$0.084 (2026-10-24 ~00:02 IST): ≈ $0.30 GPU in total.
  To verify against `build/runs/ledger.csv` before submission.
- Kaggle: free.

### Honest summary sentence
The radiometric front end and lamp control are built, deterministic under replay and ablated on sim. On real data
(Hilti) the estimator fails in dark-far-field turns, and that negative result stands. The dark/lamp/dust evaluation
is deferred by the maintainer's real-time-first ruling.
