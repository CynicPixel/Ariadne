# Pipeline speed profile (A2 for M, 2026-10-24) — measurement and recommendations only

Scope: the whole pipeline. The DNN front end (ALIKED, LightGlue, EigenPlaces) runs per engine, thread count,
input size and process shape. The live multi-agent loop runs as AWS replays of B's 200 s run. Decisions:
D232 (+ its correction), D235 (the D182 verdict), D234/K 512. Owners' code was not changed: the estimator,
ingest and live runtime are B's, collab is M's.

## Runs (all AWS spot, eu-north-1; every one launched through M's slotq)

| Job | Run id | Instance | What | Cost |
|---|---|---|---|---|
| 42 | `20261024T001208Z-cpux86-6ba75f` | c7i.4xlarge | timing matrix 1–16 threads, c3 × 2 threads, engines, per-op profile (salvaged: the knobs stage hung on a `wait` bug, fixed `923f34f`) | $0.2187 |
| 43 | `20261024T003403Z-cpuarm-46e7b8` | c8g.4xlarge (Graviton4) | the same on Arm (salvaged, same bug) | $0.1408 |
| 44 | first attempt | c7i.4xlarge | sampler hang (stop-file fix `b17cd07`), nothing kept | $0.0106 |
| 44b | `20261024T024346Z-cpux86-29bfe2` | c7i.4xlarge | replay of B's ma200 `70a514` at OpenCV pools 16 / 5 / 2 | $0.0517 |
| 45 | `20261024T034527Z-cpux86-833096` | c7i.4xlarge | the same replay with `perf -a -F 199` (per-thread attribution) | $0.0225 |
| 46 | `20261024T051058Z-cpux86-f94953` | c7i.4xlarge | the same inputs with B's lens-off build (D227; payload `live-payload-20261024T043040Z`, snapshot of run `64cb9c`), pools 16 / 5 | $0.0274 |
| 70 | `20261024T051236Z-cpux86-d80b45` | c7i.4xlarge | singles at 1/8 threads incl. the l2conv graph, c3 × 4/5, ORT 1.24.4, knobs on the 120 pairs, upstream adaptive LightGlue | $0.1168 |

**Total: $0.5885.** Raw per-run output is in `aws/<run>/` (DNN jobs) and `aws_replay/<run>/` (replays), with each run's
`run_manifest.yaml`, `steps.txt` and `ledger_row.csv`. 45's 40 MB `perf.data` stays in S3 under the run's output
prefix.

## Commands
- DNN jobs: `infra/eval/perf_aws_launch.sh run cpux86|cpuarm` (the instance runs `infra/eval/perf_aws.sh`; env
  `SINGLE_THREADS`, `C3_THREADS`, `MAXMIN`; the slotq job files are quoted in the job logs). The tools are
  `ariadne_perf_bench timing|knobs|gate` and `ariadne_perf_kfdump` (`cpp/runtime/tools/`), and
  `scripts/{ort_bench,lg_adaptive}.py`.
- Replays: `infra/eval/perf_aws_launch.sh replay` → `infra/eval/perf_replay_aws.sh`. Each replay runs B's
  `ariadne_live --replay inputs.mcap` with that run's flags, plus `infra/eval/perf_sampler.sh` (mpstat, pidstat -t,
  PSI, docker stats, perf).
- Inputs: `inputs/` holds `pairs_s{0,1}.csv` (from `scripts/pairs.py`, seed 20261024) and the sha256 of stores,
  recordings and models. `scripts/l2conv_rewrite.py` and `scripts/export_sizes.sh` make the graph variants.
- Analysis, stdlib only:
  - `python3 -I scripts/analyze.py --aws aws --out processed`
  - `python3 -I scripts/replay.py --replays aws_replay --out processed`
  - `scripts/mine.py` and `scripts/mine_live.py` produce `mined/` (the existing-data pass).
- Figures: `scripts/plot.py` in `ariadne-analysis:1` (command in its header) writes `figures/`.

## Results (measured unless tagged)
1. **The live loop's critical path is the per-agent lane, and the D41 lens was half of it.**
   - Lens on (45): each lane thread runs `SimFrameAssembler::Feed` at 50 ms/frame self time (ApplyLens inlined) of
     a ~94 ms lane.
   - Lens off (46, B's D227): lane 48–52 ms; RTF 0.514 → **0.795 (16 threads) / 0.836 (5 threads)** on the same
     200 s inputs. Runtime wall 389.0 → 251.4 / 239.2 s.
   - Source: `processed/replay_{runs,agents,perf_threads}.csv`.
2. **The estimator is thread-count-invariant.** Trajectories are bit-identical at pools of 16 / 5 / 2 (44b) and
   16 / 5 (46) (D232).
3. **Pool size barely moves wall time** (D232 correction): 5 vs 16 threads is −1.6 % lens-on and −4.9 % lens-off.
   It does remove the CPU pressure: PSI 9.4 % → 0.1 % (lens-on) and 26.3 % → 1.2 % (lens-off).
4. **After the lens, the next single threads (46, per frame of the 3-agent step):**
   - each lane: FAST ~11 ms, plus SGBM and LK on the lane thread;
   - the replay reader: `ConsumedRecorder::Write` 19 ms + ZSTD 16.5 ms (the ZSTD part is replay-only);
   - the planner's ESDF `EdgeMinConvolution1D`: ~19 ms.
5. **ALIKED's dense L2 normalize is a serial op**: ~64 % of the 1280×720 forward at 8–16 threads (42,
   `profile_ops.csv`).
   - The l2conv rewrite makes it 2.37× faster at 8 threads (70).
   - It is **not adopted**: D182 (i) has 1 flip and (ii) has descriptor diff 7.08e-6 > 4.9e-6 (D235).
6. **LightGlue:** each of the 9 layers is ~10.6 % of the forward (`lightglue_layers.csv`).
   - Upstream adaptive depth (torch) exits at mean layer 4.3–5.2 and is 2.5–2.9× faster than all 9 layers.
   - It loses 1 of 63 verified pairs (70, `adaptive_summary.csv`).
   - The cv::dnn export has no adaptive exit [ASSUMPTION: export runs all 9].
7. **Keypoint count / input size** (70, `knobs_summary.csv`, 120 real pairs):
   - K 512 loses 4 verified pairs at 0.41× the match+verify time; K 256 loses 10.
   - 960×540 K1024 loses 1 and gains 2. 640×360 K1024 loses 3.
   - K 512 is not adopted (D234).
8. **Process shape:** 3 concurrent processes × 4–5 threads give 2.2–2.3× (ALIKED-1280), 1.5–1.6× (LightGlue K1024)
   and 1.2–1.4× (EigenPlaces) the calls/s of the same run's best single process. That is the fleet's shape (70,
   `dnn_c3.csv`, same-run baselines).
9. **Engines:**
   - ENGINE_CLASSIC can't run ALIKED (Gather) or import LightGlue (42, `engines.csv`).
   - ORT 1.24.4 at 1 thread vs cv::dnn (70): ALIKED equal; LightGlue K1024 1.4× slower; EigenPlaces 1.9× faster.
   - ORT reproduces keypoints exactly, with descriptors within 7.2e-6.
10. **Arm (c8g) vs x86 (c7i)** at 16 threads (43 vs 42): ALIKED-1280 is 0.64× (530 vs 829 ms); LightGlue K1024 and
    EigenPlaces are ~0.85–0.88×. At 1 thread LightGlue and EigenPlaces are 1.6–1.7× slower on Arm.

## Caveats
- **Instance spread.** Two c7i.4xlarge instances (42 and 70) differ by ~30 % single-process: ALIKED-1280 at 1
  thread is 1760 vs 2304 ms. They have the same CPU model and OpenCV commit, no steal, and an idle box. The cause
  is not determined [UNVERIFIED: host frequency]. Every ratio above pairs rows within one run. `fig_threads.png`
  draws both runs, and the vertical segments are that spread.
- The replays are of `70a514`, B's pre-D225 lockstep run. Its trajectories are valid, but its planner trace is not
  a determinism reference (D225).
- The perf per-thread split is self time from the report's top ~100 lines (`perf_report_pid_dso_sym.txt`), so
  thread totals are "listed", not complete. ApplyLens is inlined into Feed: 50 ms is an upper bound on the lens.
- Knobs and fidelity: kfdump's re-extraction reproduces the pass-A stores (every keypoint ≤ 6.1e-5 px; 70,
  `fidelity_summary.csv`), and the store replay reproduces production on 120/120 pairs (`replay_exactness.csv`).

## Recommendations (for B and the maintainer; none applied here; post-demo list under the 2026-10-24 freeze)
1. Lane pipelining (D232 a2): ingest of frame n+1 overlapping the track of frame n. The lane is now ≈ the 50 ms
   frame period, and this is the lever to RTF ≥ 1. Then copy cuts (a3).
2. Worker OpenCV pool = 5 (3 × 5 of 16 vCPU). It barely changes wall time but removes the CPU pressure (PSI 26 % →
   1 %). This is a config change on B's runtime.
3. Replay reader: the ConsumedRecorder/ZSTD thread (~36 ms per step) is the next single thread to watch as the lane
   speeds up.
4. Graph-level ALIKED speed-ups need a fidelity bar with a seed and 1-ulp envelope. D182's 0-flip and 4.9e-6 bars
   sit at the numeric floor (D234 post-hoc: a lossless coding flips a pair).
