# 2.5 intra-agent loop closure on EuRoC (D73 part 2b)

Evidence for `docs/decisions.md` D73 part 2b and `docs/design/robust-backend.md` §8. Ground truth is not copied
(EuRoC's own data, D35); `raw/` is git-ignored.

| directory | runs | what they show |
|---|---|---|
| `no_model/` | MH_01_easy, V1_01_easy (2026-10-20) | loop closure with the raw PnP covariance: ~1000x overconfident (NEES ~1e4); the data the noise model was fitted on |
| `model_train/` | MH_01_easy, V1_01_easy (2026-10-21) | the same flights with the fitted edge noise model on |
| `model_heldout/` | V1_02_medium (2026-10-21) | held-out validation: mean good-loop NEES 5.24, tail 4.4 % above chi2_6(0.99) (criterion <= 3 %: missed, accepted by the maintainer) |

Per run: `results.yaml` (`loop_closed_keyframe_refined` next to the VO's `keyframe_refined`, and the `loop_closure:`
funnel / recall / latency / PCM good-pair summary), `loop_trace.jsonl` (one line per loop event),
`loop_candidates.csv` (every gated candidate with its ground-truth error; `model_*` runs also carry the full error
vector `xi*` and covariance `cov*`), `manifest.yaml`, `estimate_body.tum`.

**Command** (inside `ariadne/core-dev:dev`, worktree mounted at `/work`):

    ariadne_euroc_vo --sequence <EuRoC>/<FLIGHT> --perception-config configs/perception/euroc.yaml \
                     --slam-config configs/slam/euroc_v2_loops.yaml --out <dir>          # seed 1 (default)

`model_*` runs used `euroc_v2_loops.yaml` with the noise model as committed; `no_model/` runs used the same file
before the `noise_*` keys existed (PnP covariance unchanged).

**Provenance, stated.** All five binaries were built from the uncommitted D73 part 2b tree on top of `4b45934`. The
manifests say `git_commit: UNRECORDED` and `git_dirty: false`; the dirty flag is WRONG (the dev container did not
pass `ARIADNE_GIT_COMMIT` / `ARIADNE_GIT_DIRTY`). `no_model/V1_01_easy-*` predates the label fix, so its `label` column
marks everything bad; `analyze_loops.py` recomputes labels from the error columns for both `no_model` runs.

**Reproduce the tables:** `python3 evaluation/runs/robust-backend/loop_closure/analyze_loops.py`
