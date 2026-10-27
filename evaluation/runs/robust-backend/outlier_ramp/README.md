# 2.5 outlier ramp with perceptual-aliasing groups (D75)

Evidence for `docs/decisions.md` D75 and `docs/design/robust-backend.md` §6 (local-slam.md#estimator-plan §3 "outlier ramp";
Phase 2 exit gate). Synthetic, so no dataset: the protocol is the header of `tests/unit/test_aliasing_ramp.cpp`, the
generator is `cpp/slam/robust/aliasing_generator.hpp`, and both were fixed before the first measured run.

**What it is.** 3 laps of a 20 m × 10 m corridor (180 keyframes), 60 loops per graph, 10 seeds per rate. False loops
come in groups that are mutually consistent and consistent with the odometry: one spurious transform per group
("segment B is segment A"), the grouped policy of Sünderhauf & Protzel (ICRA 2013) and the failure mode DC-GM
(Lajoie et al., RA-L 2019) models. `size10` = groups of 10; `count2` = two groups, so the largest one grows with the
rate. Success = loop precision ≥ 0.99 and recall ≥ 0.95. Breakpoint = the lowest rate with success probability < 0.5.

| policy / arm | breakpoint | at the breakpoint |
|---|---|---|
| size10 / PCM → GNC | **90 %** | 0 / 10; 6 true loops < a 10-loop false group |
| count2 / PCM → GNC | **70 %** | 0 / 10; 18 true loops < a 21-loop false group |
| size10 / GNC only | none (worst 7 / 10 at 70 %) | |
| count2 / GNC only | none (worst 5 / 10 at 70 %) | |

**Reading.** PCM keeps the maximum clique of pairwise-consistent loops, so it breaks exactly where the largest false
group outnumbers the true loops: the analytic point, reproduced. GNC alone sees the global odometry, which the
aliased groups contradict (their segments are ≥ 3 m apart), and degrades without breaking. It runs ~10× slower
(165–313 ms against 15–118 ms) and is less exact below 60 % (one 9/10 dip at 40 %). The pipeline ships PCM → GNC
(D72) and this ramp is its stated limit; it is not tuned away.

**Files:** `ramp_raw.csv` (one row per policy × arm × rate × seed), `ramp_summary.csv`, `breakpoints.json`,
`ramp.png`, `plot_ramp.py`, `manifest.yaml` (command, seeds, image digest).
