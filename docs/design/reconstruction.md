# Design — Phase 6: dense reconstruction to full spec, live-incremental (D190–D209)

Status: **design proposal** (2026-10-23), after the preflight (arch §18–19, §29, D44, plan items 6.1–6.4 and the exit
gate; algorithms.md §2–4; the survey in Appendix A; the OpenCV check in Appendix B; one advisor review). Nothing is built
until M has the summary. Every constant below without a citation is [ASSUMPTION] until the T4 microbenchmark (§2)
or a gate measurement replaces it.

## 1. What runs where

```text
keyframe stream ──► recon service (python, PyTorch + gsplat v1.5.3, ONE T4)      ──► console (browser)
 live: B's #kf       ingest → submap manager → seed → train active submap         Spark renders the splats
       (D168 record)  → lifecycle → export chunk (rate-limited)                    (WebGL, client GPU);
 replay: spool dir    PGO version → re-anchor = new 4×4 per submap                  per-submap transform
 (same reader)                                                                       = re-anchoring
```

- **The GPU trains, the browser renders.** Every web splat renderer surveyed rasterises client-side (A.5), so the T4
  does no display rendering. Two budgets, kept separate:
  - **Train/update budget (T4):** ms per iteration at keyframe resolution × iterations per keyframe must fit the
    keyframe interval. **Measured keyframe rate:** dark + lamp 219 KF / 1000 frames at 20 Hz = 4.4 KF/s per agent,
    lit 78 KF = 1.6 KF/s (D142 rows). For the dark demo (arch §42–43), two agents: ~9 KF/s, **~110 ms of GPU per
    keyframe in total**; a 30-KF submap closes every ~7 s. Unknown on a T4 — nobody publishes one (A.3/A.4) — so the
    **first deliverable is a Kaggle T4 microbenchmark** (free): forward+backward ms/iter for N = 50k/100k/250k/500k
    Gaussians at 320×180 and 640×360, packed vs unpacked, plus the append-Gaussians path, with and without the
    D-SSIM term. Iterations per keyframe are then derived, not chosen.
  - **The T4 is shared** (B, 2026-10-23): demo mode today is ONE g4dn.xlarge running gz EGL + bridge + the runtime
    (parent + two estimator workers) + console. The idle-T4 benchmark is an upper bound; the budget is measured with gz
    rendering on the same GPU, and the demo's GPU sizing (a second instance, or g5/g6) is a Phase 8 decision for M and
    the user.
  - **Render budget (browser):** splat count (Spark's stated budget is 1–5 M on a computer, A.5), sort rate, and bytes
    per update (32 B/Gaussian .splat; 16 B Spark packed). Closed submaps are sent once; only the active submap is
    re-sent, at most every ~2 s [ASSUMPTION].
- **Bounded work:** only the active submap trains (Gaussian-SLAM / GLC-SLAM pattern, A.2); closed submaps are frozen,
  exported once, and never touched again except by their anchor transform.

## 2. Mechanisms (each with its backing)

| Piece | Choice | Backing |
|---|---|---|
| Submaps | anchored to their first keyframe; Gaussians stored in the anchor frame; close at 30 KF (arch 20–40) or on a loop closure that bends the trajectory | arch §18.6; LoopSplat/GLC-SLAM rigid submap update (A.2) |
| Re-anchoring | a PGO update changes the anchor's T_W_anchor only; Gaussians unchanged; the console gets one 4×4 per submap (Spark per-object transform), no re-download, no retraining | arch §18.6; MAGS-SLAM μ′ = Rμ + t, Σ′ = RΣRᵀ (s = 1); exit-gate bullet 2 becomes a CPU unit test |
| Seeding | back-project the keyframe's half-res Z where the current render's α < 0.5, only if no Gaussian lies within ρ; initial scale from the pixel footprint Z/f; colour from the training image | Gaussian-SLAM α-mask + radius check (A.2); RTG-SLAM opaque-disc init (A.2) |
| Optimisation | per keyframe: ≥ 40 % of the iterations on the new keyframe, the rest over the active submap's keyframes; L1 + D-SSIM + σ-weighted depth L1 (§19.3's L_photo + L_depth) | Gaussian-SLAM ≥ 40 % rule (A.2); arch §19.3 |
| Densify / prune | no clone/split live (seeding is the only growth); prune opacity < 0.005 | Gaussian-SLAM "no clone/prune" (A.2); gsplat DefaultStrategy prune threshold (A.4) |
| Lifecycle | NEW → UNSTABLE → SUPPORTED → STABLE by a per-Gaussian count of keyframe updates in which it was visible and consistent; STABLE are frozen (gradients masked) | arch §18.4; RTG-SLAM Tables 13/15 (A.6 item 1). RTG's δη = 100 counts consecutive FRAMES (Replica, every frame processed): ~3.3 s of observation at 30 Hz. Ours counts keyframes, so the threshold is converted at equal observation time: 3.3 s × 4.4 KF/s ≈ **15 keyframe updates** (D191) [ASSUMPTION: equal time ≈ equal evidence; keyframes are more distinct views than consecutive frames, so this errs towards freezing late] |
| Exposure | per-keyframe 3×4 affine, identity, enabled when the submap's own step counter passes 5000 | arch §18.7. Live: ~110 ms/KF total at the dark KF rate leaves order 10 iterations per keyframe [to be derived from the benchmark], so a 30-KF submap gets ~300 steps ≪ 5000 — **the transform likely never switches on live** (D192); the exposure-delay ablation and the gate's geometry-mush evidence run on the batch path |
| Merge | nearest-chunk overlap pruning between submaps/agents | arch §18.7 (Hierarchical 3DGS) — baseline; §18.5 fusion only if it beats it |
| Training image | **the albedo proxy, always by default** (§19.1 via KeyframeFalloff; locked, arch §0.2). Linear RGB on lit worlds is an **ablation row** with its own D-entry (D195), never a silent fallback: a record tagged `training_image: none` is refused unless the run is configured as that ablation | arch §19.1; D146 (the proxy refuses non-lamp-only scenes); M's review 2026-10-23 |
| Display | the console shows the albedo proxy directly and labels the panel "albedo proxy" (§19.1's "honest, slightly flat" option); relighting with a virtual lamp needs a custom shader in Spark and is not in scope | arch §19.1 |

## 3. Interfaces

- **Live input:** B's D168 `#kf` record (id, t_ns, T_O_C, intrinsics, half-res Z/S fp16, full-res rgb), forwarded by
  `ariadne_live` to a recon socket (a tee of the bytes B's worker already holds; LinkFrame length-prefixed typed
  records; unix socket, TCP if the service is off-host), plus a pose-version record when PGO re-anchors. B confirmed
  (2026-10-23) that today's live/demo worlds are LIT (ci_tiny_0, maze_short_6: per-room point lights, ambient 0.35,
  rig lamps at intensity 0). Arch §42–43 require a dark demo lit by the agents' lamps (M, 2026-10-23), so those
  worlds are not the demo and train only as the RGB ablation (D195). B added `training_image` + half-res albedo
  (fp16) + `albedo_valid` to `#kf` on 2026-10-23, tagged `none` until the chain below exists. **Live proxy training
  needs three things first:** B's 5.3 lamp sink (Normalize needs the frame's applied photometric state, and live has
  no D to provide it), a dark demo world (Phase 8, unowned), and a lamp calibration for that world's rig. Until then
  the trainer is built and gated on replay spools of `ci_tiny_dark_0` (lamp-only, calibrated; the proxy already sits
  in `KeyframeBundle`, D146). **Frames:** `#kf` carries T_O_C (per-agent odometry); B's tee adds T_W_O[k] per agent
  (the scene briefing's entry pose, the frame the console draws trajectories in). Submaps anchor in O; the console
  places one at T_W_O[k] · T_O_anchor. The live loop runs per-agent VO without global PGO today, so pose-version records will be rare until collaboration
  lands live; the re-anchoring path is exercised by the replay gate. Console: B owns `operator_console`; the splat
  panel is its own JS module and its own websocket path (`/splat`), no change to B's endpoints or telemetry sections.
- **Replay input:** the reconstruction.md#first-reconstruction-loop spool (schema bumped for the albedo proxy and pose versions) — "replay from files
  alone" stays true; the live socket and the spool feed one reader.
- **Output:** per-submap splat chunks + a submap table (id, agent, anchor kf, T_W_anchor, pose version, count,
  bytes); manifest records `splat_backend` (`gsplat_cuda@aws|kaggle`), gsplat/torch/CUDA versions, GPU name.

## 4. Without a GPU

The CPU path (reconstruction.md#first-reconstruction-loop fuse + init, D44) streams depth-seeded, **unoptimised** Gaussians to the same console panel,
labelled "initialisation only, no GPU". Geometry still grows live; appearance is the seed colour. Batch renders and
the exit-gate metrics go to Kaggle (D44). The live realtime demo still needs a GPU (maintainer, D44).

## 5. Open decisions (not mine)

- **Judging-week hosting.** G/VT on-demand quota is 0 (raise to 8 requested); g4dn spot in eu-north-1 was reclaimed
  three times on 2026-10-23 in 1a, and 1b/1c had no g4dn capacity the same morning. g4dn.xlarge on-demand in
  eu-north-1 is $0.558/h (AWS Pricing API, 2026-10-23) = $13.39/day; spot was $0.087–0.108/h the same day
  (≈ $2.1–2.6/day) with reclaims. The user decides on-demand vs spot vs the no-GPU fallback for the judging window.
- **Renderer.** Arch §29.3 names GaussianSplats3D as an example; its README now declares it unmaintained and points
  to Spark (MIT, per-object transforms, incremental `SplatMesh`). Proposed D-entry: Spark.

## Appendix A — SOTA survey: real-time / incremental 3DGS mapping, gsplat, web renderers

Collected 2026-10-23 from arXiv HTML tables, repo/doc files and the GitHub API. Every number is quoted from its
source; anything not confirmed is [UNVERIFIED]. **No candidate publishes a T4 number**; the lowest tier reported is
Photo-SLAM's laptop RTX 3080 Ti and Jetson Orin. "System FPS" = frames / total wall time, not render FPS.

### A.1 SLAM candidates — licences and inputs

| Method | ID / code | Licence (verified) | Input |
|---|---|---|---|
| MonoGS | arXiv 2312.06741 (CVPR'24); github.com/muskie82/MonoGS | Imperial NC licence + Inria NC rasterizer submodule (diff-gaussian-rasterization-w-pose) | mono / RGB-D, estimates poses |
| SplaTAM | 2312.02126 (CVPR'24); github.com/spla-tam/SplaTAM | BSD-3 repo, Inria NC rasterizer dependency (w-depth) | RGB-D |
| Photo-SLAM | 2311.16728 (CVPR'24); github.com/HuajianUP/Photo-SLAM | GPL-3.0 `LICENSE` + Inria NC `LICENSE.md`; ORB-SLAM3 | mono / stereo / RGB-D |
| RTG-SLAM | 2404.19706 (SIGGRAPH'24); github.com/MisEty/RTG-SLAM | GPL-3.0; rasterizer submodule carries the Inria NC licence; ORB-SLAM2 backend | RGB-D |
| CaRtGS | 2410.00486 (venue [UNVERIFIED]); github.com/dapengfeng/cartgs | GPL-3.0 + Inria NC `LICENSE.md` | on Photo-SLAM / GS-ICP |
| GS-ICP SLAM | 2403.12550; github.com/Lab-of-AI-and-Robotics/GS-ICP-SLAM | MIT repo, Inria NC rasterizer fork | RGB-D |
| MAGS-SLAM | 2605.10760 (v3, 2026-10-19) | no code link found [UNVERIFIED] | mono, multi-agent |
| Gaussian-SLAM | 2312.10070; github.com/VladimirYugay/Gaussian-SLAM | MIT repo, Inria NC rasterizer; simple-knn fork unlicensed | RGB-D |
| LoopSplat | 2408.10154 (venue [UNVERIFIED]); github.com/GradientSpaces/LoopSplat | MIT repo, same Inria NC rasterizers | RGB-D |
| GLC-SLAM | 2409.10982 | no code found [UNVERIFIED] | RGB-D |
| GPS-SLAM | 2509.11574; gapszju.github.io/GPS-SLAM | licence [UNVERIFIED]; C++ | RGB-D |
| LightSplat | 2609.07274 | no code found [UNVERIFIED] | RGB-D |

Consequence: every candidate is an **idea source**, never a code source (algorithms.md "Inria rasterizer trap");
gsplat (Apache-2.0) stays the only implementation base.

### A.2 Mapping mechanism

| Method | Seeding | Iterations / window | Densify / prune | Submaps / bounded set |
|---|---|---|---|---|
| MonoGS | depth back-projection per keyframe | 100 tracking / 150 mapping iters per frame; window 8 KF (TUM), 10 (Replica) (`configs/rgbd/*/base_config.yaml`) | drop last-3-KF Gaussians seen by < 3 other frames; prune opacity < 0.7 | covisibility window |
| SplaTAM | silhouette < 0.5 or large depth error | 40 / 60 per frame (SplaTAM-S 10 / 15), k overlap-selected KFs | mask-based adds | none |
| Photo-SLAM | ORB points + temporary primitives | asynchronous, Gaussian-pyramid progressive | geometric + 3DGS split/clone | none |
| RTG-SLAM | 5 % of new / high-depth-error pixels as opaque discs (α 0.99); transparent (α 0.1) for colour residual | 50 iters (75 ScanNet++), window 3–8 frames | stable once confidence count ≥ δη (100/200/400); only unstable Gaussians, only their pixels, optimised | stable set frozen |
| CaRtGS | inherited | splat-wise backprop (Taming-3DGS): 3× iterations at equal runtime; loss-chosen replay keyframes | opacity regularisation | none |
| GS-ICP | downsampled depth cloud = G-ICP source Gaussians | parallel mapping | 3DGS-style | none |
| Gaussian-SLAM | where rendered α < threshold, only if no neighbour within ρ = 0.01 m | ≥ 40 % of iterations on the new KF; only the active submap | **no clone / prune** | new submap after 0.5 m or 50° |
| LoopSplat | as Gaussian-SLAM | as Gaussian-SLAM | as Gaussian-SLAM | submaps registered on 3DGS; rigid update after PGO |
| GLC-SLAM | as Gaussian-SLAM | all active-submap Gaussians jointly | — | submap anchored to a global KF; direct map adjustment |
| MAGS-SLAM | inverse-depth back-projection per KF | sliding window | — | frozen at K_max KFs or τ_move; μ′ = sRμ + t, Σ′ = s²RΣRᵀ; occupancy-voxel fusion |
| GPS-SLAM | Gaussians only where the SDF render misses appearance | 4000 iters over 2000 frames | remove over-large / tiny / transparent; sorting-free render | SDF carries geometry |

### A.3 Reported speed (as printed)

| Method | Numbers (GPU) |
|---|---|
| MonoGS | system 1.8 FPS (Replica office1 RGB-D, multi-process), 1.1 single-process; TUM fr3/office 2.5 RGB-D / 3.2 mono; render 769 FPS; RTX 4090 (Tables 8–10) |
| SplaTAM | tracking 1.00 s/frame, mapping 1.44 s/frame; SplaTAM-S 0.19 / 0.33 s; RTX 3080 Ti (Table 6) |
| Photo-SLAM | tracking 41.6 FPS / render 911 FPS (Replica mono, RTX 4090); laptop 3080 Ti 20.0 / 353; Jetson Orin 18.3 / 95; RGB-D fr1/desk 58.4 / 2084 (Tables 1, 6) |
| RTG-SLAM | tracking 0.02 s/frame, mapping 3.5 ms/iter and 0.05 s/frame, 17.24 FPS, 2751 MB (Replica); 21.74 FPS TUM; RTX 4090 (Tables 1, 7) |
| CaRtGS | iterations-per-frame 20.6 vs Photo-SLAM 8.88 at 23.1 vs 23.5 FPS (TUM fr1/desk); RTX 4090 (Table II) |
| GS-ICP | 98.1 FPS avg, PSNR 35.93 unlimited; 30 FPS cap → PSNR 38.83 (Replica); RTX 4090 (Table 3) |
| Gaussian-SLAM | mapping 24 ms/iter, 0.93 s/frame; tracking 0.83 s/frame; render 2175 FPS; RTX A6000 (Table 8) |
| LoopSplat | tracking 0.83, mapping 0.93 s/frame; 1.36 s per registration edge; 7.0 GiB peak; A6000 (Table 7) |
| GLC-SLAM | mapping 18 ms/iter, 0.80 s/frame; tracking 1.07 s/frame; 7.0 GiB; RTX 3090 (Table VI) |
| MAGS-SLAM | 13–16 FPS tracking per agent, < 3.9 GB per agent; GPU not stated [UNVERIFIED] (Table VIII) |
| GPS-SLAM | mapping 2.6 ms/frame, 1.1 ms/iter, 137,200 Gaussians, 380.7 FPS (Replica office0); 252.6 FPS Replica avg; 151 FPS real Azure Kinect; RTX 4090 (Tables 1, 2) |
| LightSplat | tracking 0.04 s/frame, mapping 0.68 s/frame, 5.84 FPS; RTX 4080 (Table IV) |

GPS-SLAM Table 2 measures the others on Replica office0: RTG-SLAM 58.3 ms/frame, 16,700 iterations, 17.15 FPS;
GSFusion (2408.12677) 15.37 FPS; GS-ICP **843 iterations** over 2000 frames, 1.68 M Gaussians, 174 FPS — the
iteration count explains GS-ICP's speed.

### A.4 gsplat v1.5.3 (latest tag, 2025-07-04; Apache-2.0)

| Item | Finding (file at tag v1.5.3) |
|---|---|
| DefaultStrategy | duplicate/split by 2D gradient (`grow_grad2d` 2e-4), prune opacity < 0.005, opacity reset every 3000 steps, refine steps 500–15,000; optional `absgrad`, `revised_opacity` (`gsplat/strategy/default.py`) |
| MCMCStrategy | relocates low-opacity Gaussians, `sample_add` from the opacity distribution, position noise; `cap_max` 1 M; refine to step 25,000 (`mcmc.py`) |
| Adding Gaussians mid-training | public ops `duplicate`, `split`, `remove`, `reset_opa`, `relocate`, `sample_add`; `sample_add` resamples existing Gaussians only. Appending depth-seeded points = the pattern of the **private** `_update_param_with_optimizer` (concatenate params, zero-pad Adam state, extend strategy state) (`ops.py`) |
| `packed` | default True (memory); docs recommend False for small scenes |
| `sparse_grad` | COO grads for means/quats/scales; needs packed and a sparse optimiser |
| `absgrad` | `meta["means2d"].absgrad` |
| `rasterize_mode="antialiased"` | opacity compensation √(det Σ / det(Σ + εI)) |
| batching | `viewmats [..., C, 4, 4]` + `Ks`: several cameras per call (`rendering.py`) |
| 2DGS / 3DGUT | `rasterization_2dgs`; 3DGUT via `with_ut` / `with_eval3d` (packed and sparse_grad asserted off) |
| examples | trainers, viewers, `image_fitting.py`; **no SLAM / incremental example** |
| export | `export_splats`: "ply", "splat", "ply_compressed" (`gsplat/exporter.py`) |
| measured speed | eval.rst: gsplat-30k 35m49s / 5.7 GB vs Inria 37m13s / 9.0 GB; MCMC 1 M Gaussians 15.4 min / 1.98 GB (A100). profile.rst (TITAN RTX, Turing, gsplat v1.0.0): 171.8 fwd / 97.1 bwd FPS unpacked vs 160.8 / 88.4 packed (Gaussian count [UNVERIFIED]). **No T4 benchmark published.** |

`main` has moved past v1.5.3 (unreleased arguments); pin the tag. SMOKE §4o used gsplat 1.4.0 — the pin moves only
with a smoke of the append path.

### A.5 Web renderers

| Renderer | Licence | Formats | Incremental update | Sort | FPS |
|---|---|---|---|---|---|
| mkkellogg GaussianSplats3D | MIT | .ply, .splat, .ksplat | `addSplatScene` (`progressiveLoad`); `removeSplatScenes` rebuilds from saved buffers (Viewer.js); whether adding rebuilds [UNVERIFIED]; README: no longer actively developed, points to Spark | CPU WASM-SIMD worker, optional GPU distance pre-pass | [UNVERIFIED] |
| Spark v2.4.0 (2026-10-20) | MIT | .ply (incl. compressed), .spz, .splat, .ksplat, .sog, .rad | many `SplatMesh` objects sorted together, **per-object transform**; `pushSplat`/`setSplat` auto-resize; `updateVersion()`; paged LRU pool; `.RAD` streaming over HTTP Range | GPU distance readback + CPU bucket sort in a worker (`system-design.md`) | budget only: "Computer: 1–5 million splats" (`performance.md`) |
| antimatter15/splat | MIT | .splat (converts .ply) | progressive load | CPU worker; ~1 M splats ≈ 150 ms sort, sort ~4 fps vs render 60 fps (README) | as quoted |
| PlayCanvas engine v2.23.3 / SuperSplat | MIT | PLY, compressed PLY, SOG, streamed SOG, GLB, SPZ | streamed-SOG LOD chunks; runtime add [UNVERIFIED] | CPU worker (WebGL/WebGPU) or GPU radix (WebGPU hybrid) | [UNVERIFIED] |

Bytes per Gaussian: `.splat` 32 B (12 pos + 12 scale + 4 RGBA + 4 quat, SH0); gsplat float32 PLY 56 B (SH0) / 236 B
(SH3); Spark PackedSplats 16 B, ExtSplats 32 B; SOG 15–20× smaller than PLY, SPZ ~10× (PlayCanvas formats page);
.ksplat [UNVERIFIED].

### A.6 Mechanisms with measured backing

1. Optimise only unstable Gaussians and only their pixels: 9.8 → 5.2 ms/iter (Storeroom), 8.4 → 4.7, 8.9 → 4.3
   (RTG-SLAM Table 13); δη = 100 → 35.43 dB at 17.31 FPS vs 33.63 dB at δη = 50 (Table 15).
2. Cheaper backprop buys iterations: CaRtGS 20.6 vs 8.88 iterations per frame at equal FPS (Table II).
3. Few iterations, spend on appearance: GPS-SLAM 380 FPS with 137k Gaussians; GS-ICP 843 iterations / 2000 frames
   but 1.68 M Gaussians (GPS-SLAM Table 2).
4. Fusion in submap merge: without occupancy-aware fusion PSNR 40.33 → 33.16; SE(3) instead of Sim(3) 18.61 dB —
   a monocular result (MAGS-SLAM Table IX).
5. Submaps bound memory: Gaussian-SLAM 0.5 m / 50° trigger; LoopSplat 7.0 GiB peak vs SplaTAM 18.5 GiB (Table 7).
6. Packed vs unpacked: 0.35 vs 0.48 GB at 160.8 vs 171.8 forward FPS (gsplat profile.rst, TITAN RTX).
7. Browser sort lag: CPU worker sort ~4 fps against a 60 fps render (antimatter15); GaussianSplats3D rebuilds its
   whole mesh on scene removal.

Sources: arxiv.org/abs/{2312.06741, 2312.02126, 2311.16728, 2404.19706, 2410.00486, 2403.12550, 2605.10760,
2312.10070, 2408.10154, 2409.10982, 2509.11574, 2609.07274}; github.com/nerfstudio-project/gsplat (tag v1.5.3:
strategy/*.py, rendering.py, exporter.py, docs/source/tests/{eval,profile}.rst); github.com/sparkjsdev/spark
(docs/docs/*); github.com/mkkellogg/GaussianSplats3D (README, src/Viewer.js); github.com/antimatter15/splat;
github.com/playcanvas/engine; developer.playcanvas.com/user-manual/gaussian-splatting/formats/.

## Appendix B — OpenCV-native check (installed build, `ariadne/core:dev`, /opt/opencv5)

- No Gaussian-splat or differentiable rasterizer anywhere in the installed headers (grep for splat/3dgs: none).
- `ptcloud`: `loadPointCloud`/`savePointCloud`, `loadMesh`/`saveMesh` — GT mesh and cloud I/O for Chamfer.
- `geometry`: `triangleRasterize`, `triangleRasterizeDepth`, `triangleRasterizeColor` — CPU GT-mesh depth/colour
  renders for held-out evaluation; `Volume` (TSDF family) and `Odometry`.
- `quality`: `QualityPSNR`, `QualitySSIM`, `QualityGMSD` — held-out image metrics (LPIPS stays torch).
- The build has no CUDA: everything on the GPU is torch + gsplat.

---

<a id="first-reconstruction-loop"></a>

# First reconstruction loop

Plan 1.7: "fused stereo point cloud, gsplat initialization from it, one offline render; the Python reconstruction
service boundary (§18.2) exists even though the content is naive". Constraint discovered on 2026-10-15
(SMOKE §4o, D44): gsplat cannot rasterize without CUDA, the dev box has no NVIDIA GPU, and every AWS GPU quota is
0 (re-requests escalated). The maintainer's decision: **run the gsplat stage on Kaggle/Colab for now**; the
realtime demo still needs a live GPU (AWS once the quota lands). So the boundary is designed to be
**backend-agnostic**: the same service code runs locally (CPU: everything except rasterization), on Kaggle (CUDA),
and later on an AWS GPU node.

### Pipeline (arch §18.1, placeholder internals)

```text
C++ replay (ariadne_sim_collab --spool DIR)      Python service (python/reconstruction/)
  both agents' VO, 1.6 merge                        validate spool (schema v1; any missing field = error)
  per keyframe: rectified RGB + depth + σ  ──────►  fuse: back-project depth, T_O0_O1 for agent 1, voxel grid
  final graph poses + graph version                 init Gaussians (gsplat simple_trainer conventions)
  held-out frames (RGB + estimated pose)            render held-out views (gsplat.rasterization, CUDA)
  merge record (T_O0_O1 from 1.6)                   metrics: held-out PSNR/SSIM/LPIPS, Chamfer, completeness
```

### The spool (the §18.2 boundary: "KeyframeBundle records over gRPC or spooled files")

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

### Gaussian initialization and render

Transcribed from gsplat's `examples/simple_trainer.py` at the pinned version (not from memory): means = fused
points, colors = SH DC from RGB, scales from the mean distance to the 3 nearest neighbours (log), opacity
logit(0.1), quaternions as simple_trainer sets them. The fused cloud is voxel-downsampled (`voxel_m`) and the
Gaussian count is capped. 1.7 renders the INITIALIZATION (no training: "one offline render"); `train_steps`
exists in the config with value 0 so Phase 6 extends rather than replaces.

### Where it runs

- **Local (Docker `ariadne-recon`, CPU torch, pinned):** spool validation, fusion, init, Chamfer/completeness, unit
  tests — the deterministic, byte-identical stages. No rasterization (gsplat cannot without CUDA; SMOKE §4o).
- **Kaggle (CUDA), the render step only:** the upload is a small RENDER PACKAGE — the initialized Gaussians
  (N × ~60 B), the held-out frames' RGB, poses and K — not the spool (a few MB instead of hundreds).
  `infra/scripts/kaggle.sh` (Kaggle CLI in a container, token from `~/.kaggle`): push the package as a private
  dataset, push the kernel (GPU on), poll, pull the renders back; metrics are computed locally from them.
  The kernel records its torch/CUDA/gsplat versions and GPU name into the manifest
  (`splat_backend: gsplat_cuda@kaggle`). Kaggle over Colab: it has a CLI, so the run is scripted and reproducible.
- **AWS GPU (later):** the same command with a different launcher.

### Evaluation frame

The fused cloud lives in agent 0's odometry frame; GT meshes live in the world frame. `T_W_O0` is the SE(3)
Umeyama alignment of agent 0's estimated trajectory to GT (the one `ComputeATE` already computes), written to the
EVALUATION record only — GT never enters the spool or the service. GT surface: the scene's meshes sampled
uniformly by area at `gt_sample_m` spacing; completeness = fraction of GT samples within `completeness_tau_m` of
the cloud (both fixed in the config before the first measurement). The agents see only part of the maze, so
completeness is expected to be low; it is reported as measured.

### Measured outputs (§30.2 mapping ids, D31)

`heldout_psnr_db`, `heldout_ssim`, `heldout_lpips` (renders of the initialization: expected poor, reported as is),
`chamfer_m` and `completeness` of the fused cloud against the scene's GT meshes (sampled surface), Gaussian count,
spool bytes. Determinism: the CPU stages are byte-identical across two runs; the CUDA render is compared by
tolerance (GPU float atomics), not bytes.

### Not in 1.7

Submaps and anchoring (§18.6, Phase 6), albedo proxy (§19, needs the lamp), training, stable/unstable lifecycle,
merge-time pruning, streaming to the console.
