# Phase 2 full gate: `phase1.sh` full mode on AWS (D172)

**Run:** clean commit b59a77d, on AWS c7i.4xlarge spot (run `20261023T050134Z-cpux86-8b6fdc`), 32 min.

**Render:** a fresh Kaggle render, relayed so that the token stays on the workstation (`infra/eval/render_relay.sh`).

**Command:**
```
infra/eval/premerge_aws_launch.sh stage full && infra/eval/premerge_aws_launch.sh run
```

**Result:**
- **Verdict as run: 53/54.** ctest 539/539 plus protocol 18/18. Every I1 condition and every §4 contract passes.
- **Matches the committed reference:**
  - the fault identity;
  - `bytes.json`;
  - the sha256 of the fused cloud, the render package and `renders.npz`. The fresh render is byte-identical to the
    cached one, which is D48 reproduced.
- **The one FAIL** is `report/report.yaml` against the D55 reference, and only on `inputs_clean`: here `true`, in the
  reference `false`.
  - The D55 reference run's `recon/.../manifest_score.yaml` recorded `git_dirty: 'true'`.
  - So the comparator, which compares `inputs_clean` on purpose so that a dirty run cannot pass, demanded a dirty
    run.
  - D172 re-pins that one reference to this run. Every other reference is unchanged.

`aws/` holds the run-level logs: steps, host, image ids, checksums, and the tail of `phase1.log`.
