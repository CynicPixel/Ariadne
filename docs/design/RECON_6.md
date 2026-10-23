# Design — Phase 6: dense reconstruction to full spec, live-incremental (D190–D209)

Status: **preflight** (2026-10-23). The design sections follow after the advisor review and the T4 microbenchmark;
nothing is built from this file until M has the design summary.

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

Consequence: every candidate is an **idea source**, never a code source (ALGORITHMS.md "Inria rasterizer trap");
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
