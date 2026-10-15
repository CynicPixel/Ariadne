# Design — Phase 1.7: placeholder reconstruction (spool boundary, fused cloud, gsplat init, one render)

Plan 1.7: "fused stereo point cloud, gsplat initialization from it, one offline render; the Python reconstruction
service boundary (§18.2) exists even though the content is naive". Constraint discovered on 2026-10-15
(SMOKE §4o, D44): gsplat cannot rasterize without CUDA, the dev box has no NVIDIA GPU, and every AWS GPU quota is
0 (re-requests escalated). The maintainer's decision: **run the gsplat stage on Kaggle/Colab for now**; the
realtime demo still needs a live GPU (AWS once the quota lands). So the boundary is designed to be
**backend-agnostic**: the same service code runs locally (CPU: everything except rasterization), on Kaggle (CUDA),
and later on an AWS GPU node.

## Pipeline (arch §18.1, placeholder internals)

```text
C++ replay (ariadne_sim_collab --spool DIR)      Python service (python/reconstruction/)
  both agents' VO, 1.6 merge                        validate spool (schema v1; any missing field = error)
  per keyframe: rectified RGB + depth + σ  ──────►  fuse: back-project depth, T_O0_O1 for agent 1, voxel grid
  final graph poses + graph version                 init Gaussians (gsplat simple_trainer conventions)
  held-out frames (RGB + estimated pose)            render held-out views (gsplat.rasterization, CUDA)
  merge record (T_O0_O1 from 1.6)                   metrics: held-out PSNR/SSIM/LPIPS, Chamfer, completeness
```

## The spool (the §18.2 boundary: "KeyframeBundle records over gRPC or spooled files")

One directory per run, written by C++, read only by Python; `spool.json` carries `schema_version: 1`, run/scene
ids, the 1.6 merge record (`T_O0_O1`, the edge it came from, `frame: "agent0 odometry"`), and the bundle index.
Per keyframe (`agent{k}/kf_{frame_id}.*`), every §18.2 field, explicit when not modeled:

| §18.2 field | spool | 1.7 value |
|---|---|---|
| kf_id, agent | index | VO keyframe frame id, agent |
| frame_id_of_pose | index | VO pose-graph version when the keyframe was exported (at insertion) |
| T_WiC, K | index | the rectified-left camera pose in agent i's odometry frame AT INSERTION (the same poses the 1.6 merge `T_O0_O1` was computed from, so the merge and the spool are one pose set); rectified K at the spool scale |
| rgb | `rgb.png` (8-bit) | rectified left RGB through the D41 lens + the same rectification maps; **not** albedo-proxy (lamp inert in 1.x: `rgb_kind: "raw_radiance"`) |
| depth + σ (fp16) | `depth.png`, `sigma.png` (fp16 bit patterns in 16-bit PNG: lossless, compressed) | SGBM/LK depth on the VO's support mask, NaN elsewhere; σ_Z = Z²·σ_d/(f·B), σ_d = `[PROXY]` config |
| exposure_state, lamp_state | index | `"not_modeled"`, `"inert"` |
| mask | index | `"none"` |

Held-out frames (non-keyframes at a fixed stride) carry RGB + estimated pose only, `role: "heldout"`. GT poses
never enter the spool (evaluation reads GT separately). Volume: 577 keyframes at full 1280×720 would be several GB,
so the spool has a `scale` (default 0.5 → 640×360) and a `keyframe_stride`; it lives under `build/`, never in
evidence (the evidence keeps `spool.json` + sha256 of every file).

## Gaussian initialization and render

Transcribed from gsplat's `examples/simple_trainer.py` at the pinned version (not from memory): means = fused
points, colors = SH DC from RGB, scales from the mean distance to the 3 nearest neighbours (log), opacity
logit(0.1), quaternions as simple_trainer sets them. The fused cloud is voxel-downsampled (`voxel_m`) and the
Gaussian count is capped. 1.7 renders the INITIALIZATION (no training: "one offline render"); `train_steps`
exists in the config with value 0 so Phase 6 extends rather than replaces.

## Where it runs

- **Local (Docker `ariadne-recon`, CPU torch, pinned):** spool validation, fusion, init, Chamfer/completeness, unit
  tests — the deterministic, byte-identical stages. No rasterization (gsplat cannot without CUDA; SMOKE §4o).
- **Kaggle (CUDA), the render step only:** the upload is a small RENDER PACKAGE — the initialized Gaussians
  (N × ~60 B), the held-out frames' RGB, poses and K — not the spool (a few MB instead of hundreds).
  `infra/scripts/kaggle.sh` (Kaggle CLI in a container, token from `~/.kaggle`): push the package as a private
  dataset, push the kernel (GPU on), poll, pull the renders back; metrics are computed locally from them.
  The kernel records its torch/CUDA/gsplat versions and GPU name into the manifest
  (`splat_backend: gsplat_cuda@kaggle`). Kaggle over Colab: it has a CLI, so the run is scripted and reproducible.
- **AWS GPU (later):** the same command with a different launcher.

## Evaluation frame

The fused cloud lives in agent 0's odometry frame; GT meshes live in the world frame. `T_W_O0` is the SE(3)
Umeyama alignment of agent 0's estimated trajectory to GT (the one `ComputeATE` already computes), written to the
EVALUATION record only — GT never enters the spool or the service. GT surface: the scene's meshes sampled
uniformly by area at `gt_sample_m` spacing; completeness = fraction of GT samples within `completeness_tau_m` of
the cloud (both fixed in the config before the first measurement). The agents see only part of the maze, so
completeness is expected to be low; it is reported as measured.

## Measured outputs (§30.2 mapping ids, D31)

`heldout_psnr_db`, `heldout_ssim`, `heldout_lpips` (renders of the initialization: expected poor, reported as is),
`chamfer_m` and `completeness` of the fused cloud against the scene's GT meshes (sampled surface), Gaussian count,
spool bytes. Determinism: the CPU stages are byte-identical across two runs; the CUDA render is compared by
tolerance (GPU float atomics), not bytes.

## Not in 1.7

Submaps and anchoring (§18.6, Phase 6), albedo proxy (§19, needs the lamp), training, stable/unstable lifecycle,
merge-time pruning, streaming to the console.
