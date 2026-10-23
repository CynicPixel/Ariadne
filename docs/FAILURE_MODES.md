# Failure modes and failure cases

Arch §3 (repo layout) names this file; the phased plan's 5.6 and 8.2 require "failure segments documented as
failure cases". One subsection per case: where, what was observed (numbers, not adjectives), the evidence path,
what is known about the cause, and what is not. Dataset pixels are not reproduced here (Hilti 2023 is
CC BY-NC-SA 3.0; EuRoC as D35): cases are described by measurement.

---

## FC-5.6-1 — Real dark sequence with a co-moving lamp: VO loses track while the robot turns with a dark far field

**Where.** Hilti SLAM Challenge 2023 `site2_robot_3` (D141), robot front OAK-D Pro W stereo (equidistant, D143),
10 Hz, all 3592 frames. Matrix `configs/benchmark/hilti_5.6.yaml`; decision D144.

**Observed (three configurations, none re-tuned for Hilti):**

| row | Hilti score fit / loo | GCP error rms | lost frames | keyframes / 3592 |
|---|---|---|---|---|
| sim-tuned (`sim_v1` + `sim_v2`) | 0 / 0 | 10.1 m | 602 | 2167 |
| EuRoC-tuned (`euroc_v2`) | 0 / 0 | 229 m | 779 | 3006 |
| EuRoC-tuned + radiometric gate | 0 / 0 | 17.0 m | 485 | 2031 |

**Failure segments** (per 300-frame window; same in all rows):

| segment | frames (≈ s from start) | gyro p90 | far-field brightness | lost (sim-tuned) |
|---|---|---|---|---|
| A | 900–1199 (90–120 s) | 3.1 °/s | 44 (dark) | 174 / 300 |
| B | 1500–1799 (150–180 s) | 3.1 °/s | 50 (dark) | 203 / 300 |
| B′ | 1800–2399 (180–240 s) | 3.0–3.5 °/s | 71–81 | 76–97 / 300 |

Driving straight (gyro 0.2 °/s), the same rows lose nothing. In segments with a lit far field (80–94) and similar
turning, losses are 0–39. Whenever the robot moves, keyframes run at 200–300 per 300 frames (the lit sim: ~80 per
1000), so the map is rebuilt almost continuously even where nothing is lost.

**Verified not the cause:**
- **The scoring chain.** Distances between the predicted control points do not depend on any rigid transform. They
  fall apart (ratios 0.27–50 to the survey), and the estimated height spans 14–527 m on one floor.
- **Static camera-locked edges** (the lamp's ring pattern on the floor, a lens smudge). In segments A and B, 7–10 % of
  GFTT+LK tracks are static (< 0.5 px) against 36–38 % in the windows that work.

**Measured, not separating:** keyframe stereo matching (forward-backward rejections 50 %, valid 29 %). EuRoC MH_04,
which the pipeline handles (0.046 m, D87), is at 17 % valid with the same configuration.

**Open (candidates, not ranked):** the co-moving lamp's range-dependent falloff under rotation (in sim, D142, dark + lamp
was ×6 worse than dark + fixed light, but brightness and coverage were not matched, so the falloff is not isolated); 10 Hz frames; the fisheye-rectified periphery (107.6° kept of
~127°); left/right ISO lag on 253 frames.

**Evidence.** `evaluation/runs/ph5/hilti_5.6/` (results, manifests, estimates, control-point scores). The per-window
tables come from `vo_trace.csv`, the export's `imu0/data.csv` and its images, by the procedures in D144.
