# 4.3 ablation, job 1: the abort check (next-best-view.md §15.5 (f)): GO

**Run:** `20261024T055409Z-cpux86-8d7693`, AWS c7i.4xlarge spot (eu-north-1c, $0.1254/h), slotq
`30-A1-nbv43-abort`. Launched 11:24 IST, ended 11:42 IST. Job wall time 664 s.

**What ran:**
- ci_tiny seed 2, row `ariadne_v0` (`planner_config.yaml` = `configs/mission/ablation_4.3/mission_v2_ariadne_v0.yaml`).
  The gate config plus the D120 term log, with D119 and D125 always on. This is the seed of the gate's 25 s hold
  livelock.
- Lockstep, RTF 0 + D168, llvmpipe, 200 s. The scene is the gate's own `ci_tiny_2`, copied from the a1p tree.

**Provenance:**
- `ariadne_live` and the sim plugin were built locally from feat/ph4-nagent at `5502dde` (code; `b4cbf9d` adds only
  the job file) and shipped as the payload (`payload_check.txt`: sha256 OK).
- The instance received a snapshot without `.git`, so `manifest.yaml` has `git_commit: ""`. The commit is recorded
  here instead. The payload also contained the uncommitted FAR debug edit in `ariadne_volumetric_eval.cpp`. It is a
  different binary, not used by this run.
- EGL/llvmpipe repro check before the job: PASS (`batch.log`).

## The go/no-go conditions, fixed before the run

| condition | result |
|---|---|
| every `kind:"turn"` trace record has its mover answer `accepted: true`, `segments: 1` | **PASS.** The one turn (t = 20 s, agent 0, yaw π) was answered `accepted: true`, `segments: 1`, 20.86 → 28.69 s. The mover log shows one `turn` segment of 7.84 s and no move. |
| `answer_timeouts` = 0, every barrier answered | **PASS.** 0 timeouts; 441/441 barriers answered; reorder timeouts 0. |
| no hold at a won at-cell view | **PASS.** 0 hold decisions with a pick (80 decisions: 76 goto, 3 follow, 1 turn). |

**Verdict: GO** for the other 47 runs, which wait on the maintainer's spend approval.

## Development observation (one run; not the ablation's result and not a claim)

- At t = 20 s agent 0 chose the same at-cell view as in the gated run. This time it turned to it (D125), then moved
  on: at 1.0 m by t = 45 s. In the gated run it held from t = 20 to 45 s.
- Coverage, scored by `python/analysis/gate_coverage.py` with the gate's scenes (`gate_rows.jsonl`):

| ci_tiny seed 2 | 20 s | 40 s | 60 s | 80 s | 200 s |
|---|---|---|---|---|---|
| ariadne_v0 (this run) | 0.351 | 0.562 | **0.878** | 0.899 | 0.961 |
| gated ariadne (d59ee3d) | 0.351 | 0.531 | 0.728 | — | — |
| gated TARE-inspired / coverage-greedy | 0.406 / 0.400 | 0.772 / 0.841 | 0.929 / 0.870 | | |

- The base commit also differs from the gate's (main merged), so this is not a clean attribution to D125.
- The 20 s loss is unchanged (0.351): that is the turning cost D117 targets.
- One run at one seed. The 48-run matrix decides.
