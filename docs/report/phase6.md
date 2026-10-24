# Phase 6 — Reconstruction: incremental Gaussian splatting in the live console

DRAFT (A2, 2026-10-24). Sources are the decision entries (`docs/DECISIONS.md`) and the evidence directories named inline. Tags: **measured**
= a run we can point to; **[DESIGN]** = built and unit-tested, not yet measured end to end; **[PROJECTION]** = a
derived number. Nothing here is a projection presented as measured.

**Status: the live incremental path works end to end on the CPU initialiser. GPU training is measured on synthetic
smokes; real-scene GPU training on the 200 s demo recording is running/next (Kaggle T4). The Phase 6 exit-gate
metrics (held-out PSNR/SSIM/LPIPS, Chamfer to the GT mesh, the double-wall check, the exposure-delay ablation) are
NOT yet measured on a sim spool.**

### Architecture (D190, design `docs/design/RECON_6.md`)
- The GPU trains and the browser renders.
- Keyframe-anchored submaps (close at 30 KF). Only the active submap trains; closed submaps are frozen and sent once.
- A pose-graph update moves a submap by a new 4×4 transform: no retraining and no re-download.
- Each agent trains in its own odometry frame. Submaps are drawn at T_W_O[k] · T_O_anchor, with T_W_O from the
  briefing (known starts) or from the merge (D230).
- The renderer is Spark (MIT; user-approved, D197), with one SplatMesh per submap.
- Without a GPU, the console still grows a map, labelled "initialisation only, no GPU" (D194).
- Training images: the albedo proxy is the locked default (D195). The live demo trains on linear RGB in the lit maze,
  a user directive, and the albedo proxy is the dark-world stretch (D199).

### Measured
- **T4 cost** (D196; Kaggle T4, gsplat 1.5.3; `evaluation/runs/ph6/bench_t4_*`).
  - Full loss at 640×360: 10.9 ms/iteration at 100k Gaussians and 15.3 ms at 250k. At 320×180: 6.3 and 9.1 ms.
  - Render at 640×360: 1.7–3.0 ms.
  - The append path keeps the Adam moments bitwise.
  - This gives ~7–10 iterations per keyframe at 640×360 in a ~110 ms keyframe budget. **[PROJECTION]:** an upper
    bound on an idle T4, not measured under gz load.
- **Trainer smokes on a synthetic corridor** (D198; `evaluation/runs/ph6/train_smoke_20261023T1340Z`;
  Gaussian-SLAM's own recipe, arXiv 2312.10070).
  - Batch (100 iterations/KF): train-view 22.2 → 25.1 dB, held-out 17.8–20.1 dB. Predicting the mean scores 12.9 dB.
  - Live budget (15 iterations/KF): train-view +0.5 dB only. **At that budget the live quality is essentially the
    seed quality**, a measured property, not tuned away.
  - Freezing and re-anchoring are bitwise exact on the GPU.
- **Live end to end, real runtime to real service** (D230/D231; `evaluation/runs/ph6/live_e2e_20261024`).
  - B's 20 s 3-agent lit `maze_demo_0` run, teed over a socket: 71 keyframes, 0 drops, 4 submaps, 259,994 Gaussians.
  - The tee is determinism-neutral: the run digest equals the no-tee reference.
  - Placement from the runtime matches the briefing to ≤ 2.2e-16.
  - Seeds against the GT ESDF: 92.9 % of wall submaps within 10 cm. Agent 0's floor-heavy submaps reach 47–66 %.
  - The page renders in headless Chromium on an Intel Iris Xe with no errors.
  - **This was the CPU initialiser (cpu_init); there was no GPU training in this run.**
- **Judge-box GPU image** `ariadne-recon-gpu` (torch 2.11 + cu128, gsplat 1.5.3 with CUDA kernels compiled ahead of
  time for T4/A10G/L4; `evaluation/runs/ph6/recon_gpu_build_20261023T194812Z`). Built and uploaded to S3. **Not yet
  validated on a GPU instance.**

### Running / next (fill before submission)
- **GPU training on the 200 s demo recording**, Kaggle T4, pre-approved.
  - Dataset: B's ma200 rerun `20261024T022434Z-cpux86-626abe` (tree 95544d6, which contains the D225 barrier fix).
    It is lens-ON, since it predates D227. Per-agent spools via `ariadne_sim_vo --spool`.
  - Report: held-out PSNR/SSIM/LPIPS at the batch and live budgets, ms/KF, Gaussians.
- The exit-gate geometry checks (Chamfer to the GT mesh, the double wall) and the exposure-delay ablation are not
  started.

### Spend
- AWS: GPU image builds $0.097. Everything else ran locally or on Kaggle (free).

### Honest summary sentence
Incremental, keyframe-anchored splatting runs live from the real multi-agent runtime into the browser, with exact
placement and determinism preserved. GPU training is costed and smoke-tested on a T4. Its real-scene quality on the
demo recording is the open measurement; the live budget is known to leave the map close to its depth seeds.
