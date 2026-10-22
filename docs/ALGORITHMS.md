# Ariadne — Algorithm Research

Reference catalogue for reimplementation. Sourced from five parallel deep-research passes, with key claims independently re-verified against primary sources.

**Setting:** multi-agent visual SLAM + collaborative reconstruction in post-disaster, human-inaccessible structures — darkness under self-carried illumination, texture-poor rubble, dust, GPS-denied, bandwidth-constrained. Fire/smoke is a **stretch goal** ([positioning doc](./ARIADNE-feasibility-audit.md) §2.3).

**Status:** Domains 1, 2, 4, 4b, 5 complete (§1–§21). **Domain 3 — active perception, next-best-view, multi-robot task allocation — is the only one still owed**; its agent hit API session rate limits on both attempts. Re-run when the limit resets. (§20.2 notes that TARE and FAR Planner, the two most reusable SubT exploration planners, belong to that domain.)

| # | Domain | Status |
|---|---|---|
| 1 | Visual SLAM & collaborative multi-agent SLAM | ✅ §1 |
| 2a | Classic 3DGS-SLAM systems | ✅ §2 |
| 2b | Feed-forward pointmap SLAM & multi-agent 3DGS | ✅ §3 |
| 2c | Partitioned / distributed 3DGS (merge mechanisms) | ✅ §4 |
| 3 | Active perception, NBV, multi-robot task allocation | ✅ §22 (2026-10-23; primary text except where marked) |
| 3a | Volumetric mapping for planning (TSDF, ESDF, frontiers, traversability) | ✅ §21 (2026-10-23, Phase 4.1/4.2; primary text) |
| 4 | Degraded visual environments + DARPA SubT | ✅ **§9–§20** — illumination §9 · dust/confined §10 · burst §11 · event+thermal §12 · enhancement §13 · HDR/noise §14 · modalities §15 · night data §16 · aliasing §17 · co-moving-light data §18 · ASTM protocol §19 · **DARPA SubT §20** |
| 4b | **Datasets & benchmarks** | ✅ §6, §8 |
| 5 | Distributed optimization, pose-graph consensus, robust estimation | ✅ §7 |

---

## ⚠️ License traps — read before writing any code

> **Policy update (team decision):** licensing is **deprioritised**. Everything is **reimplemented from the paper**, not copied or linked, so a restrictive licence is not a blocker — at most a reason not to crib an implementation while reading it. The already-verified findings below are kept because they cost nothing and are occasionally useful (they also flag which repos are safe to read closely). **Do not spend further research time verifying licences.**

Verified against actual `LICENSE` files, not from memory. Several widely-used methods are **non-commercial**; under the old policy these would have poisoned the project.

| Method | License | Verdict |
|---|---|---|
| **SuperPoint** (Magic Leap weights) | ⛔ *"ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY"* | **TRAP** — many downstream repos are Apache-2.0 in code but pull these weights |
| **DUSt3R / MASt3R / MASt3R-SLAM** | ⛔ CC BY-NC-SA 4.0 | **TRAP** |
| **MicKey** (Niantic) | ⛔ Non-commercial + "Patent Pending" | **TRAP** |
| **VGGT** (and VGGT-SLAM via it) | ⚠️ Bespoke; AUP prohibits *"autonomous systems that pose safety risks"* | **Live legal question for a disaster robot** |
| SALAD, ORB-SLAM3, DSO, SVO Pro, COVINS/COVINS-G | ⛔ GPL-3.0 | Study the architecture, do not link |
| D²SLAM, SE-Sync | LGPL-3 | Dynamic linking only |
| XFeat, LightGlue, DISK, LoFTR, maplab | ✅ Apache-2.0 | Clear |
| ALIKED, AnyLoc, robust_distributed_mapper | ✅ BSD-3 | Clear |
| Swarm-SLAM, DOOR-SLAM, iMESA, RoMa, DeDoDe, CosPlace, EigenPlaces, MegaLoc, BoQ, DPVO | ✅ MIT | Clear |
| GTSAM, Kimera-VIO/RPGO/Multi-LCD, Ceres | ✅ BSD-2/3 | Clear, linkable |
| DBoW2 | ⚠️ BSD-style, but clause 3 requires notifying the original author | Read before shipping |

**Sidestep the SuperPoint trap entirely:** LightGlue ships `features='aliked' | 'disk' | 'sift'` variants. OpenCV 5 does the same — it ships ALIKED, DISK, and XFeat, and **no SuperPoint**.

---

## 1. Visual SLAM & Collaborative Multi-Agent SLAM

### 1.0 Headline finding — OpenCV 5 ships the modern learned front-end natively

**Independently verified** against `modules/features/include/opencv2/features.hpp` on the `5.x` branch. Classes present:

```
KeyPointsFilter · Feature2D · AffineFeature · SIFT · ORB · MSER
FastFeatureDetector · GFTTDetector · DISK · XFeat · SimpleBlobDetector
ALIKED · DescriptorMatcher · DescriptorCollection · BFMatcher
FlannBasedMatcher · LightGlueMatcher · ANNIndex
```

`SuperPoint`: **0 mentions** — OpenCV deliberately shipped only the permissively-licensed extractors.

| Class | Detail (from header) |
|---|---|
| `cv::XFeat` | ONNX via `cv::dnn::Net`. Grayscale `N×1×H×W` in [0,1]. **64-D float descriptors.** `create(modelPath, maxKeypoints, scoreThreshold, inputSize=Size(640,640), backendId, targetId)` |
| `cv::ALIKED` | ONNX. RGB `[1,3,H,W]` (auto BGR→RGB). **128-D float**, sub-pixel keypoints. `Params{inputSize, normalizeDescriptors, engine, backend, target}` |
| `cv::DISK` | ONNX. Input named `image`, `N×3×H×W` float32. **H, W must be positive multiples of 16.** |
| `cv::LightGlueMatcher` | Subclass of `cv::DescriptorMatcher`. Call `setPairInfo(qKpts, tKpts, qSize, tSize)` **before** `match()`. Auto-context from in-process `ALIKED`. Drops into any `BFMatcher`-shaped code. |
| `cv::ANNIndex` | Annoy-backed ANN index — the modern FLANN replacement. **This is your place-recognition retrieval index.** *(Note: class is `ANNIndex`, not `AnnoyIndex`.)* |

> ⚠️ **Correction (2026-10-17, 2.1 OpenCV-native check):** the `cv::XFeat` row does not hold for the **pinned 5.0.0
> build**. There is no `XFeat` class in either tree (`SMOKE_TEST_RESULTS.md`, the module audit). Ariadne runs XFeat as
> its own `cv::dnn` graph (`cpp/perception/features/xfeat/xfeat_net.*`, D50, SMOKE §4p). `ALIKED`, `LightGlueMatcher`
> and `ANNIndex` are present as described above. ORT does not appear in `cvconfig.h`; this is
> [UNVERIFIED pending `getBuildInformation`], and ORT is treated as absent.

> ⚠️ **Correction (2026-10-21, 2.1 hard-pair preflight; SMOKE §4s, PERCEPTION_2.1 §12), the `cv::ALIKED` and
> `cv::LightGlueMatcher` rows against the pinned 5.0.0 source:**
> - **No weights ship.** OpenCV's own test loads opencv_extra testdata fetched from a personal repo with no
>   licence and no export script (`aliked-n16rot-top1k-640.onnx`, `aliked_lightglue.onnx`).
> - **Both load on ENGINE_NEW only;** ENGINE_CLASSIC fails on both.
> - **"Auto-context from in-process ALIKED" is false:** the implementation never reads it, and `setPairInfo` is
>   mandatory.
> - **`setPairInfo` normalises pixels per axis** (x/W·2−1, y/H·2−1). Upstream LightGlue keeps the aspect ratio:
>   shift W/2, H/2; scale max(W,H)/2. An empty `Size()` passes pre-normalised keypoints through.
> - **`clone()` shares the Net,** so it is not a per-thread copy.
> - **`cv::ALIKED` ignores `mask`** and resizes to `inputSize` without keeping the aspect ratio. It maps
>   normalised keypoints as (n+1)/2·W, where upstream uses (W−1).
>
> **As built:** Ariadne exports its own from upstream (`cvg/LightGlue@eb42fee`, ALIKED-n16 per glue-factory's
> official config, native 752×480, `python/model_tools/aliked_lightglue_reference.py`). Through these same two
> classes it is exact against upstream PyTorch: keypoints within 1e-4 px, descriptors within 4e-6, identical match
> sets (SMOKE §4s-2).

Also confirmed for OpenCV 5:
- **USAC is the default robust-estimation backend** for `findHomography`, `findEssentialMat`, `findFundamentalMat`, and all `solvePnP` variants → **MAGSAC++-class robustness for free**, no hand-written RANSAC.
- **LAPACK always available** (internal minimal subset if not installed) → `cv::SVD` and eigendecomposition guaranteed.
- New DNN engine covers **>80% of the ONNX spec** (vs <23% in 4.x).
- `calib3d` → **`geometry`** (`findHomography`, `solvePnP`, `triangulatePoints`, `estimateAffine*`) + **`calib`** (`calibrateCamera`, `stereoCalibrate`, new N-camera pipeline) + **`stereo`** (`StereoBM`, `StereoSGBM`, `reprojectImageTo3D`) + **`ptcloud`** (visual odometry, TSDF, ICP, `.ply`/`.obj` I/O).
- C++ back-compat: `opencv2/calib3d.hpp` still exists, includes the three new headers.

> **Consequence for Ariadne.** "Write SLAM from scratch" and "use cutting-edge algorithms" are **not in tension**. The right reading: *you* write the SLAM system — tracking, keyframing, map management, BA, pose graph, loop closure, the multi-agent protocol — and call OpenCV 5 primitives for feature extraction, matching, and robust geometry. **Do not spend two months reimplementing a CNN.** This also strengthens the 30% Technical Execution row: every one of these is an OpenCV 5 class, so the "depth of OpenCV 5 implementation" claim becomes concrete.

### 1.1 Front-end architecture — the domain call

The field has bifurcated: classical geometric SLAM (writable, CPU-capable) vs. learned dense SLAM (DROID/DPVO/MASt3R-SLAM — GPU-bound, mostly unreimplementable in two months, often non-commercially licensed). **Build the former with learned features inside it.**

| System | Year/venue | License | Difficulty | Verdict |
|---|---|---|---|---|
| **ORB-SLAM3** | T-RO 2021 | ⛔ GPL-3.0 | Med (skeleton ~3–4 wks) | **Architectural template.** Keyframe graph, covisibility, local vs global BA split, keyframe culling. Read it; don't link it. Swap ORB for learned features. |
| **DSO** (direct) | TPAMI 2018 | ⛔ GPL-3.0 | **High** | **Do not build on this.** Photometric BA with Schur marginalization is the hardest thing here — and see the domain warning below. |
| **SVO / SVO Pro** (semi-direct) | T-RO 2017 | ⛔ GPL-3.0 | Med | **The pattern to adopt.** Cheap direct alignment for frame-to-frame tracking; expensive learned features only at keyframes. Exactly right for a compute- and bandwidth-constrained fleet. Sparse image alignment on 4×4 patches ≈ 200 lines; depth filter ≈ 150 lines. |
| **DPVO / DPV-SLAM** | NeurIPS 2023 / ECCV 2024 | ✅ MIT | High to train, **Low to run** | Sparse *patch* tracking + differentiable BA. AUC 0.80 vs DROID's 0.71 at **4× the speed**; 1–4× real-time, **5–7 GB VRAM**. The one learned-VO system clean enough to deploy. **Use as a baseline to beat**, or an optional GPU-agent mode. |
| **MASt3R-SLAM / VGGT-SLAM** | CVPR 2025 / 2025 | ⛔ NC / ⚠️ AUP | — | Accuracy frontier. Cite, don't build on. |

> ⚠️ **Critical domain warning — avoid direct/photometric methods.** They assume brightness constancy. Under **self-carried illumination** (Ariadne's base case) the light source *moves with the camera*, so scene radiance changes with pose — violating the assumption at first principles, at a level the per-frame affine brightness model cannot absorb. The same holds for smoke and flame in the stretch case. **This is a hard architectural constraint, not a preference.** **→ §9 now details the mitigations, and qualifies this warning in one important way:** the correct response is not to abandon photometric information but to *model* it — near-field falloff can be divided out of the front-end (§9.8 step 3), and once modelled it becomes a **metric depth cue** (§9.2). Keep the front-end feature-based; use the photometry as a normaliser and a scale source, not as the tracking objective.

**Recommended front-end:** keyframe-based, SVO/ORB-SLAM3 mold.
- Frame-to-frame: `calcOpticalFlowPyrLK` on the previous keyframe's keypoints (2 days) → sparse direct alignment as an upgrade
- Keyframe features: **`cv::XFeat`** → matching → **`geometry::solvePnPRansac`** (USAC) → windowed BA

### 1.2 Learned features — benchmark table

MegaDepth-1500 relative pose, VGA, laptop i5-1135G7 **CPU**:

| Method | AUC@5° | AUC@10° | Desc dim | **FPS (CPU)** |
|---|---|---|---|---|
| ORB | 17.9 | 27.6 | 256-bit | 44.3 |
| **XFeat** | **42.6** | **56.4** | **64-f** | **27.1** |
| **XFeat\*** (semi-dense) | **50.2** | **65.4** | 64-f | 19.2 |
| SuperPoint | 37.3 | 50.1 | 256-f | 3.0 |
| ALIKE | 49.4 | 61.8 | 64-f | 5.3 |
| DISK | 53.8 | 65.9 | 128-f | 1.2 |

**XFeat beats SuperPoint on accuracy at 9× the speed**, and approaches DISK at 16× the speed. On embedded hardware it is **the only learned method above 1 FPS** (1.8 vs SuperPoint 0.16, ALIKE 0.58).

**The decisive number for Ariadne** — ScanNet-1500 (indoor, *low-distinctiveness*, the closest available proxy for texture-poor rubble):

| Method | AUC@5° |
|---|---|
| **XFeat** | **16.7** |
| SuperPoint | 12.5 |
| DISK | 9.6 |
| ALIKE | 8.0 |

The paper attributes this to XFeat's large receptive field and depth, which *"focus less on distinctive textures"* — precisely the failure mode of concrete and rubble.

**XFeat** — CVPR 2024, [arXiv:2404.19174](https://arxiv.org/abs/2404.19174), [code (Apache-2.0)](https://github.com/verlab/accelerated_features).
*Core idea:* prior lightweight detectors shrink networks uniformly. XFeat identifies that the `H×W` term dominates FLOPs at early high-resolution layers and **redistributes** depth — minimal channels early, then **tripling** channels as resolution halves (vs VGG's doubling). Keypoint detection is a separate branch using 1×1 convs on an 8×8 tensor-block-transformed image. Outputs keypoint heatmap **K**, **64-D dense descriptor map F**, reliability heatmap **R**. An optional match-refinement module predicts pixel offsets from coarse 1/8-resolution NN pairs → semi-dense matching at negligible cost.
*Difficulty:* **Low** (use `cv::XFeat`). If reimplementing inference for the "we wrote it" story: ~10 conv blocks, a weekend in C++/Eigen. Training from scratch is not tractable (COCO synthetic warps + MegaDepth, multi-GPU).
*Fit:* **best practical option.** Best low-texture generalization, and its 64-D descriptor is the smallest of any learned extractor — **halving inter-agent bytes vs ALIKED, quartering vs SuperPoint.**

**ALIKED** — IEEE T-IM 2023, [arXiv:2304.03608](https://arxiv.org/abs/2304.03608), [code (BSD-3)](https://github.com/Shiaoming/ALIKED).
*Core idea:* predecessors extract descriptors from a *dense* map with a fixed, geometrically rigid support region per keypoint. ALIKED's **Sparse Deformable Descriptor Head (SDDH)** *learns the deformable positions of supporting features* per keypoint and builds descriptors only at sparse locations — more expressive, cheaper. Relaxes the Neural Reprojection Error loss from dense to sparse.
*Difficulty:* **Low** (`cv::ALIKED`). SDDH is a learned-offset bilinear gather, ~80 lines if reimplementing.
*Fit:* good — deformable support adapts to blurred/attenuated structure. Costs 2× XFeat's bandwidth.

**LightGlue** — ICCV 2023, [arXiv:2306.13643](https://arxiv.org/abs/2306.13643), [code (Apache-2.0)](https://github.com/cvg/LightGlue).
*Core idea:* SuperGlue runs fixed-depth transformer + Sinkhorn optimal transport over all pairs regardless of difficulty. LightGlue is **adaptive in both depth and width**: after each layer it predicts confidence, **exits early** on easy pairs, and **prunes** unmatchable keypoints so later layers see fewer points. Replaces Sinkhorn with a lighter matchability+similarity decomposition.
*Speed:* 150 FPS @1024 kpts / 50 FPS @4096 kpts on RTX 3080; **20 FPS @512 kpts on an i7-10700K CPU**. 4–10× over SuperGlue.
*Difficulty:* **Low** to use (`cv::LightGlueMatcher`); Med-High to write (9-layer transformer, rotary positional encoding of keypoint coords, confidence/pruning heads). ONNX exports: [LightGlue-ONNX (Apache-2.0)](https://github.com/fabio-sim/LightGlue-ONNX).
*Fit:* **very good.** Context aggregation lets a keypoint's match be disambiguated by its *neighbours'* matches — exactly what rescues you when individual descriptors are degraded. Adaptive depth also means it spends more compute on hard pairs automatically.

**Also noted:** DISK (Apache-2.0, NeurIPS 2020 — trains detection+description end-to-end with **policy gradient** since keypoint selection is non-differentiable; highest raw accuracy but 1.2 FPS CPU → offline/GPU-agent only). RoMa (MIT, CVPR 2024 — dense matching on frozen DINOv2 + robust regression-by-classification; too heavy for a per-keyframe front-end, good for offline verification of a few critical inter-agent links).

### 1.3 Backend

**Windowed bundle adjustment** — *write this from scratch; highest-value "we built it" component.*
Reprojection residuals, analytic Jacobians w.r.t. SE(3) on the manifold (**left-perturbation convention — never Euler angles**), Levenberg-Marquardt, sparse normal equations. The essential trick is the **Schur complement**: marginalize the many landmark blocks to get a small reduced camera system, then back-substitute — turning `O((6N+3M)³)` into `O((6N)³ + M)`.
*OpenCV 5:* `geometry::projectPoints` **returns the analytic Jacobian as an output argument** — a free derivative check. `geometry::solvePnPRefineLM` as a single-pose reference to validate against. `cv::SVD`/`cv::solve` LAPACK-backed. Sparse Cholesky from Eigen `SimplicialLDLT` (header-only, MPL2).
*Difficulty:* Medium (windowed, 10–15 keyframes) — ~1.5 weeks with assistance.

**Pose graph optimization.** Simpler than BA: SE(3) nodes, relative-pose edges, residual `Log(Z_ij⁻¹ · X_i⁻¹ · X_j)`.
> **Critical:** initialization matters more than the solver. **Do rotation initialization first** (chordal relaxation / rotation averaging over SO(3) — a *linear* least-squares problem with a closed-form solution), then translation, then joint refinement. This two-stage pattern turns a nasty non-convex problem into a well-initialized one, and is what essentially every system below does.

**Robust kernels.** Apply a robust loss to **loop-closure edges only**; keep odometry edges quadratic (they're outlier-free). This is exactly Kimera-Multi's choice. Ladder: Huber/Cauchy (IRLS, ~20 lines) → Truncated Least Squares → **Switchable Constraints** (Sünderhauf & Protzel IROS 2012 — a scalar switch `s_ij ∈ [0,1]` per loop edge scaling its information matrix, letting the optimizer smoothly turn off bad edges; **Low difficulty, very explainable**) → Dynamic Covariance Scaling (closed form, free) → Barron's adaptive loss (CVPR 2019).

**⭐ Graduated Non-Convexity (GNC) — highest value-per-line item in this entire report.**
Yang, Antonante, Tzoumas, Carlone, **RA-L 5(2):1127–1134, 2020**, [arXiv:1909.08605](https://arxiv.org/abs/1909.08605), [reference code](https://github.com/MIT-SPARK/GNC-and-ADAPT), also in GTSAM.
*Core idea:* via the **Black–Rangarajan duality** between robust estimation and outlier processes, rewrite a robust cost as `min_w [ w·r² + Φ_ρ,μ(w) ]` over per-measurement weights `w_i ∈ [0,1]`. Then alternate: (a) solve a **weighted least-squares** problem — your existing solver, unchanged; (b) update weights in closed form; (c) increase `μ` to reintroduce non-convexity. Starts from a convex surrogate (no initial guess needed) and sharpens toward the true robust cost, avoiding the spurious local minima that directly optimizing a robust cost falls into.
*Verified performance:* **robust to 70–80% outliers, outperforms RANSAC, more accurate than specialized local solvers, faster than specialized global solvers.**
*Difficulty:* **LOW.** It is a *wrapper* around a solver you already have — initialize `μ`, loop { solve weighted PGO; update `w_i` from residuals and `μ`; `μ ← μ/1.4` }, stop when weights are binary. **~60 lines for 70–80% outlier robustness.**
*Fit:* **essential.** Perceptual aliasing in texture-poor rubble drives a high loop-closure outlier rate — Swarm-SLAM measured **10 outliers in 67 inter-robot loop closures (15%) in a benign parking lot**. Expect worse. GNC handles this regime by design.

**SE-Sync — steal the certificate, not the solver.**
Rosen, Carlone, Bandeira, Leonard, **IJRR 2019**, [arXiv:1612.07386](https://arxiv.org/abs/1612.07386), [code (LGPL-3)](https://github.com/david-m-rosen/SE-Sync).
*Core idea:* PGO is non-convex, so local solvers can return a bad minimum with no way to know. SE-Sync marginalizes translations analytically (leaving pure rotation synchronization), forms a **convex SDP relaxation**, and solves it by Riemannian truncated-Newton on the Stiefel manifold via Burer–Monteiro low-rank factorization. The minimizer is **globally optimal whenever noise is below a critical threshold** — and you can **verify this a posteriori** by checking a certificate matrix (the connection Laplacian) is PSD.
*Recommendation:* **don't reimplement the solver (High, 3+ weeks) — do implement the certificate.** After GNC-PGO converges, compute the connection-Laplacian certificate and its minimum eigenvalue via **Lanczos / shifted power iteration** (~150 lines). Non-negative ⟹ provably globally optimal.
> **"Our system emits a certificate that the merged multi-agent map is the provable global optimum" is a claim almost no competing entry will have, for ~150 lines of work.**

*Distributed variant:* Tian, Khosoussi, Rosen, How, **T-RO 2021**, "Distributed Certifiably Correct Pose-Graph Optimization" (RBCD) — Kimera-Multi's distributed workhorse.

**iSAM2 — flagged impractical.** Kaess et al., IJRR 2012; GTSAM ([BSD](https://github.com/borglab/gtsam)). Bayes tree + incremental variable reordering + fluid relinearization ≈ months to do correctly. **For Ariadne's scale (a few agents, one building), batch GNC-PGO re-solve on rendezvous is entirely adequate** — which is exactly what Swarm-SLAM concluded. Use GTSAM if you want iSAM2; don't build it.

### 1.3a Tracking loss — what a VO does while it cannot localise (added 2026-10-17, D56)

When PnP fails, the map cannot be used, but the 2-D tracks usually survive. Ariadne measured 40–360 on EuRoC's dark
and blurred stretches (`docs/design/LOST_FRAMES.md`).

| method | source / licence | what it does | Ariadne |
|---|---|---|---|
| **RECENTLY_LOST → LOST, Atlas** | ORB-SLAM3 `Tracking.cc` (⛔ GPL-3.0: read, don't link); Atlas, Elvira et al., IROS 2019 | Stereo: relocalise with DBoW2 for 3 s and make no keyframes; then start a new map (reset below 10 keyframes), merged on revisit. With an IMU: predict from it for 5 s. | No-keyframes-while-lost **tested and rejected** without relocalisation: it raised lost frames (D56). Segments are tagged; merging is Phase 3 place recognition. |
| **Failure detection → new segment** | VINS-Mono, Qin et al., T-RO 2018 (⛔ GPL-3.0) | Few features, or jumps in pose or bias → re-initialise → a separate pose-graph segment, joined by loop detection. | Segment counter + `pose_source` on every keyframe (D56). |
| **H / F model selection** | ORB-SLAM, Mur-Artal et al., T-RO 2015 §IV-B | Scores S_M = Σ (Γ − d²/σ²) over χ²-truncated transfer errors; H when S_H / (S_H + S_F) > 0.45. | **Adopted** for the rotation bridge, as published. Counting inliers instead was tried and fails (sub-pixel parallax bias). |
| **Reference-keyframe tracking** | ORB-SLAM `TrackReferenceKeyFrame` | Tracks against one reference, not frame to frame. | **Adopted:** the bridge measures from an anchor frame, so its error does not accumulate. |
| **Gap smoothing (RTS backtracking)** | SP-VIO, arXiv 2411.07551 | Smooths the interrupted stretch once tracking returns. | Needs an inertial model: a 2.4 option. |
| **IMU bridging** | Forster preintegration (arch §10) | A metric prediction over seconds. | 2.4. The bridge's rotation source is the seam the gyro replaces. |

*OpenCV 5:* `geometry::findHomography` / `findEssentialMat` take `UsacParams` (seeded, `isParallel = false`, so
replays are deterministic), and `recoverPose` takes `distanceThresh`. ⚠️ The default `recoverPose` overload counts
only points within 50 baselines, i.e. none at 1–2 cm between frames, so it can silently return 0.

**Ariadne's measured result (D56):** MH_04 2.78 → 0.36 m ATE, V2_03 2.12 → 1.27, MH_05 0.56 → 0.35. Lost-frame
counts do not move; the losses themselves are 2.1 / 2.4 territory.

### 1.3b Inertial layer — attitude first, then preintegration (added 2026-10-18, D58)

| method | source / licence | what it gives | Ariadne |
|---|---|---|---|
| **On-manifold preintegration** | Forster, Carlone, Dellaert, Scaramuzza, T-RO 2017, [arXiv:1512.02363](https://arxiv.org/abs/1512.02363) | ΔR, Δv, Δp between keyframes with first-order bias Jacobians; inertial factors in BA that touch only camera blocks, so the Schur structure is unchanged | arch §10.2, 2.4 `full` (next). Paper perturbs on the RIGHT: re-derived in our left convention, FD-tested (PHASE2_PLAN §2). |
| **Explicit complementary filter** | Mahony, Hamel, Pflimlin, IEEE TAC 53(5) 2008, [HAL](https://hal.archives-ouvertes.fr/hal-00488376), Theorem 5.1 | attitude from gyro + accel, gyro bias online; crossover at kP | **Adopted** for gravity direction (`attitude_only`), gains as in the paper's experiments: kP = 1, kI = 0.3 rad/s (§VI). Yaw bias is unobservable from gravity. |
| **Inertial-aided KLT** | Hwangbo, Kim, Kanade, IROS 2009 ([CMU project](https://www.ri.cmu.edu/project/imu-assisted-klt-feature-tracker)) | the gyro predicts each feature's motion, keeping LK in its convergence basin under sharp rotation | Seam built and **tested**: on EuRoC, lost frames −4% only, mixed ATE (D58). Off by default: our losses are landmark failures, not LK failures (D56). |
| **Inertial-only MAP initialisation** | Campos, Montiel, Tardós, ICRA 2020, [arXiv:2003.05766](https://arxiv.org/abs/2003.05766) | gravity, velocity and biases as a MAP problem from a few seconds of VO; ORB-SLAM3 initialises in < 4 s on EuRoC | Reference for `full`. Stereo gives scale, so scale is not estimated (arch §10.3). |

**Measured on EuRoC (D58):**
- The gyro integral, with the ground-truth bias removed, matches the ground-truth camera rotation to 0.019° per frame
  pair, median over 26,530 pairs. Raw, it is 0.23°: the bias dominates.
- Camera and IMU timing agree to within a sample.
- Gyro bias is estimated from VO rotations (a robust median), since Mahony cannot observe it about the gravity axis.

### 1.3c Re-association — re-finding mapped landmarks after their tracks die (added 2026-10-17, 2.1)

A landmark is useful only while some pixel is linked to it. With LK, that link is the track that created it. Once the
track dies, the point can never be observed again, unless it is re-found by appearance.

On D56's EuRoC runs, 633 of the 683 lost frames have ≥ 30 live tracks but < 15 landmark correspondences
(`docs/design/PERCEPTION_2.1.md` §4.0). That is the case this section serves.

| method | source / licence | what it does | Ariadne |
|---|---|---|---|
| **Relocalisation** | ORB-SLAM, Mur-Artal et al., T-RO 2015 §V-C (arXiv 1502.00956); ORB-SLAM2 `Tracking.cc` `Relocalization()` (⛔ GPL-3.0: read, don't link) | Candidate keyframes by BoW → `ORBmatcher(0.75)` descriptor matching (≥ 15 matches) → PnP-RANSAC (min 10 inliers, χ² 5.991) → pose optimisation; accepts at ≥ 50 inliers. `ORBmatcher::TH_LOW = 50` (Hamming). | **Adopted, restricted to the local window, with brute force instead of BoW** (the window is small): orphan tracks are described at their pixel and matched (mutual NN, ratio 0.75, ≤ 50 bits) against landmarks born at MEASURED keyframes. A seeded USAC PnP, with the frame's healthy correspondences as anchors, keeps only geometric inliers, and the tracker renames them to the landmark ids. StereoVo's own PnP then decides. |
| **Track Local Map** | ORB-SLAM §V-D | Projects covisible map points with the current pose and searches near the projection (viewing angle < 60°, scale range, representative descriptor). | Planned tracking mode, at keyframes (§4.3 of the design). It needs a real pose prior, so it is not used while lost. |
| **Representative descriptor** | ORB-SLAM §III-C | The medoid of a map point's observed descriptors. | Ablation A5. The baseline keeps the latest keyframe observation. |
| **Landmark-aware matching** | OKVIS2, Leutenegger, arXiv 2202.09199 §VI-A | Matches keypoints to existing 3-D landmarks by descriptor distance AND reprojected distance. | Same principle (appearance, then geometry). DL-VINS-Factory (arXiv 2607.01757, preprint) measures OKVIS2 best on EuRoC stereo (0.060 m) among the systems it ran. |
| **Poisoning guard** | Ariadne AUDIT_2.3 §4 / D56 `PoseSource` | A landmark triangulated at a predicted pose carries that pose's error. | Candidates are restricted to `PoseSource::kMeasured` births; `admit_bridged` is ablation A1b. |

*OpenCV 5:*
- `cv::BFMatcher` (NORM_HAMMING, `knnMatch` k = 2, plus a reverse `match` for the mutual check) is exact and
  deterministic.
- `geometry::solvePnPRansac` with `UsacParams` (seeded, `isParallel = false`).
- ⚠️ `ORB::compute` on PROVIDED keypoints does not compute orientation (`ICAngles` runs only when detecting,
  `features/src/orb.cpp`). It drops keypoints within `edgeThreshold` of the border, so descriptors from
  `OrbEngine::Describe` are unrotated BRIEF, comparable only with each other.

**Measured on EuRoC v2 (D60, merged D58 tree, 2026-10-19).** Lost-mode re-association with ORB ships:
- failing-four median lost frames 264/101/141/171 → 67/99/92/122;
- 11-flight seed-1 average ATE 0.223 → 0.204 m.

With XFeat descriptors (upstream `XFeat.match`, e92685f: mutual NN + cos > 0.82, no ratio test), V1_02 regressed.
Adding Lowe 0.75 fixed it (PERCEPTION_2.1 §9.11), so a ratio test is needed on XFeat descriptors too. XFeat as the
tracking engine regressed V2 and MH_02 ATE independently of re-association (§9.13).

### 1.4 ⭐ Swarm-SLAM — the reference system for Ariadne

Lajoie & Beltrame, **IEEE RA-L 2024**, [arXiv:2301.06230](https://arxiv.org/abs/2301.06230), [code (MIT)](https://github.com/MISTLab/Swarm-SLAM).

The only major C-SLAM system whose *central design objective* is minimizing bytes exchanged — MIT-licensed, and every mechanism is individually reimplementable.

**M1 — Two-stage exchange** (Cieslewski ICRA'18 / DOOR-SLAM). Stage 1: broadcast only *compact global descriptors* per keyframe; peers compute cosine similarity to find candidates. Stage 2: only for selected candidates, transmit the *expensive local descriptors* for 3D registration. *Difficulty: **Low** — protocol design, not an algorithm.*

**M2 — ⭐ Budgeted spectral prioritization via algebraic connectivity.** Prior systems used **greedy** selection (verify the top-B by similarity), which picks *redundant* candidates clustered in high-similarity regions. Swarm-SLAM instead frames candidate selection as **pose-graph sparsification performed before the measurements exist** (the dual of Doherty/Rosen/Leonard spectral sparsification, [arXiv:2203.13897](https://arxiv.org/abs/2203.13897)).

Build a *reduced* multi-robot pose graph from global-matching info alone — **requiring zero additional communication**. Weighted Laplacian with `κ_ij = 1` for fixed edges, `κ_ij = s_e` (similarity score) for candidates, deliberately using similarity as a confidence proxy so noise models never need transmitting. Then select the size-B subset maximizing **algebraic connectivity λ₂**:

> maximize over ω ∈ {0,1}ⁿ of λ₂(L_local + L_fixed + Σ_e ω_e L_e), subject to |ω| = B

λ₂ is used because it *"controls the worst-case error of the solutions of the SLAM MLE"*. NP-hard (Mosk-Aoyama 2008) → relax integrality, solve the convex relaxation, round. Two practical notes: **(a)** the graph must be connected first, so run greedy until at least one inter-robot loop closure exists; **(b)** use the greedy solution as the initial guess.

*Verified result:* spectral prioritization reaches near-final ATE with a small fraction of loop closures computed, while greedy plateaus — across KITTI-00, KITTI-360-09, GrAco, S3E-Square.
*Difficulty:* **Medium** — you need λ₂ and the Fiedler vector; **shifted power iteration or Lanczos on the sparse Laplacian suffices (~100 lines)**. The relaxation is a simple gradient step since `∂λ₂/∂ω_e = (v₂ᵀ L_e v₂)`. **This is the flagship reimplementation: elegant, directly a bytes-saved story, and genuinely novel.**

**M3 — Vertex-cover exchange policy** (Giamou, Khosoussi & How, ICRA 2018). When several selected candidates *share a keyframe*, that keyframe's local features need transmitting **once**. Formulate as **minimum vertex cover** on the candidate bipartite graph — **optimal for 2 robots** (bipartite ⟹ König ⟹ max-matching), approximate for 3+. *Difficulty: **Low-Med** — Hopcroft–Karp is textbook.*

**M4 — ⭐ Elected-broker backend, NOT distributed PGO.** The counterintuitive result, and the most actionable finding here. Swarm-SLAM *deliberately avoids* distributed PGO: at each rendezvous a broker is elected (lowest ID, or by compute), peers ship local pose graphs to it, it runs **centralized GNC with TLS**, and ships estimates back. Verbatim justification: *"more robust to communication failures and easier to implement."*

Verified Table II — **communication in kB**:

| Dataset | Robots | **Swarm-SLAM** | DGS+PCM | D-GNC (RBCD) |
|---|---|---|---|---|
| KITTI 00 | 2 | **280.00** | 30,045.50 | 13,499.49 |
| KITTI-360 09 | 5 | **484.91** | 3,241.21 | 4,485.13 |
| GrAco Ground | 3 | **105.82** | 44,686.69 | 78,162.42 |
| M2DGR Gate | 3 | **51.29** | 242.02 | 721.39 |
| S3E Square | 3 | **80.97** | 1,261.12 | 620.19 |
| S3E College | 3 | **150.37** | 3,420.23 | 3,012.28 |

**A 4×–740× reduction in backend bytes**, with better ATE and lower compute on 5 of 6 sequences. **Lesson: iterating a distributed optimizer over a network costs enormously more in bytes than shipping the graph once.**

**M5 — Bookkeeping + anchor propagation.** Neighbour management tracks (via heartbeats) who is in range and *exactly what has already been exchanged*, so nothing is sent twice. Reference-frame convergence uses **anchor propagation**: first rendezvous anchors on the lowest-ID robot's first pose; later ones anchor on the lowest *reference-frame* ID. All agents converge to one global frame **without ever requiring a full-team rendezvous.** *Difficulty: **Low**, large payoff.*

**Real-world validation (Table III):** 3 heterogeneous robots (Spot, Scout, Scout Mini), Jetson AGX Xavier + RealSense D455 + Ouster OS0-64 + ad-hoc WiFi, indoor parking lot. **475.42 m, 3,103 keyframes, 67 inter-robot loop closures (10 rejected by GNC), 94.95 MB total exchanged.** Optimization 5.52 ± 7.11 s; sparsification 2.71 ± 2.39 s in a separate thread.

> **Key quote:** *"The communication load is mostly attributable to the front-end and thus dependent on the number of keyframes."*
> ⇒ **Ariadne's bytes metric will be dominated by global descriptors × keyframes + local descriptors × verified candidates. Optimize there first.**

### 1.5 Other collaborative systems

**Kimera-Multi** — Tian et al., **T-RO 38(4):2022–2038, 2022**, [arXiv:2106.14386](https://arxiv.org/abs/2106.14386). Components BSD-2; index repo has no LICENSE file.
First fully-distributed system rejecting incorrect inter/intra-robot loop closures from perceptual aliasing, using only peer-to-peer comms, building a globally-consistent dense metric-**semantic** mesh in real time. Front-end: ORB + DBoW2; on match ≥ 0.1 normalized visual similarity, robot β *requests* matched keyframe's 3D keypoints + descriptors from α, matches via NN search, runs **Nistér five-point** verification. (Maps almost exactly onto `features::ANNIndex` + `geometry::findEssentialMat`.) Backend: **fully distributed GNC** with the variable-update step solved by **RBCD**; quadratic on odometry, TLS on loop closures; robust two-stage (rotation first, then PGO).
*Explicit justification for abandoning PCM, verbatim:* *"the runtime of exact maximum clique search exceeds 10 seconds already in graphs with 700 loop closures... in practice PCM has to rely on heuristic maximum clique algorithms and thus often exhibits poor recall."*
**→ Copy the outlier-rejection architecture and the request/response protocol. Skip the semantic mesh.** Kimera-VIO and Kimera-RPGO are BSD-2 and safely linkable as references.

**DOOR-SLAM** — Lajoie et al., **RA-L 5(2), 2020**, [arXiv:1909.12198](https://arxiv.org/abs/1909.12198), [code (MIT)](https://github.com/MISTLab/DOOR-SLAM). Two modules: a distributed front-end detecting inter-robot loop closures **without exchanging raw sensor data**, and an optimizer combining **Distributed Gauss-Seidel** (Choudhary et al., IJRR 2017 — robots converge to a consistent joint estimate communicating *only* poses involved in inter-robot loop closures, preserving trajectory privacy) with **distributed PCM**. *Difficulty: **Medium**; DGS is genuinely simple (block Gauss-Seidel on separator poses).* **The clearest pedagogical reference** — read it, then use Swarm-SLAM's superior mechanisms.

**COVINS-G** — Patel et al., **ICRA 2023**, [arXiv:2301.07147](https://arxiv.org/abs/2301.07147). ⛔ **GPLv3** — cannot link. But the *idea* is valuable: it generalizes the backend to be **front-end agnostic** by deriving loop-closure constraints from a **multi-camera relative pose algorithm on 2D-2D correspondences only**, so it works with any odometry source including a black-box T265, with heterogeneous front-ends across agents in one mission.
> **Adopt the interface design, write your own code: define the inter-agent interface as 2D-2D correspondences + a relative-pose constraint, not as a shared map representation.** That makes drones and ground robots interoperable despite different sensors and different local SLAM. (Practical tip from their README: start transmission only after 50 keyframes.)

**D²SLAM** — Xu et al., **T-RO 40, 2024**, LGPL-3. Splits **near-field** (precise *relative* state estimation for close-range collaboration) from **far-field** (globally consistent trajectories); supports stereo and omnidirectional. ⚠️ **Requires CUDA for the front-end** — disqualifying for cheap agents, but the near/far split is a good idea if Ariadne's drones fly in proximity.

**maplab 2.0** — Cramariuc et al., **RA-L 8(2), 2023**, [code (Apache-2.0)](https://github.com/ethz-asl/maplab). Not an algorithm but a *platform*: modular multi-modal multi-robot mapping, explicitly supporting **heterogeneous teams with different sensor setups**. **The most permissively-licensed complete C-SLAM platform in the field.** Worth studying for its VIMap data structure and plugin architecture — the right shape when several people build different modules in parallel.

**iMESA / riMESA** — McGann & Kaess, **RSS 2024** / 2026, [code (MIT)](https://github.com/rpl-cmu/iMESA). Incremental distributed optimization via **consensus ADMM**. *Difficulty: Med-High.* Given Swarm-SLAM's finding that elected-broker centralized GNC beats distributed methods on both accuracy and bytes at small team sizes, ADMM is unlikely to be right for Ariadne's scale.

### 1.6 Place recognition — the bandwidth arithmetic

Global descriptors are broadcast **per keyframe to every peer**, so descriptor dimensionality is a **first-order term in Ariadne's headline metric**.

| Method | Desc dim | fp32 bytes/kf | @3,000 keyframes |
|---|---|---|---|
| DBoW2 BoW vector | sparse | ~1–4 KB | ~3–12 MB |
| **EigenPlaces/CosPlace (R18, 128-D)** | **128** | **512 B** | **1.5 MB** |
| **EigenPlaces/CosPlace (256-D)** | **256** | **1.0 KB** | **3.1 MB** |
| EigenPlaces/CosPlace (512-D) | 512 | 2.0 KB | 6.1 MB |
| SALAD (`dino_salad_512_32`) | 544 | 2.2 KB | 6.5 MB |
| CosPlace (R50, 2048-D) | 2048 | 8.2 KB | 25 MB |
| **MegaLoc** | **8448** | **33.8 KB** | **101 MB** |
| BoQ (DINOv2) | 12288 | 49.2 KB | 148 MB |
| BoQ (ResNet50) | 16384 | 65.5 KB | 197 MB |
| AnyLoc (VLAD/DINOv2 ViT-G/14) | ~49k | ~196 KB | ~590 MB |

> **The single most decision-relevant table here.** Choosing BoQ-ResNet50 over EigenPlaces-128 costs **128× the place-recognition bandwidth** for marginal recall gain. Swarm-SLAM's entire 3-robot mission was 94.95 MB total — **AnyLoc alone would blow that by 6×.**

**⭐ EigenPlaces — the pick.** Berton, Trivigno, Caputo, Masone, **ICCV 2023**, [code (MIT)](https://github.com/gmberton/EigenPlaces). Trains on *clusters of images from different viewpoints of the same place*, building lateral-viewpoint invariance into the embedding without extra inference-time supervision. **This matters enormously for inter-agent loop closure, where a drone and a ground robot see the same corridor from radically different heights and headings.** Dims: R18 (256/512), R50 (128/256/512/2048), R101, VGG16 (512). **Use ResNet50 @ 256-D → 1 KB/keyframe.** Via `torch.hub` → ONNX → `dnn::readNetFromONNX`; index with `features::ANNIndex`.

> **As built and measured (2.1, 2026-10-20; PERCEPTION_2.1 §11; D62 draft).** R50-256 at commit `a2969f7`
> exports to ONNX (opset 17, fixed input size) and runs in `cv::dnn` exactly as upstream on both engines (cosine
> 1.0000000, SMOKE §4r). EuRoC loop-closure recall, 297 queries over 11 flights (positive = 1 m and 30°, database
> older than 30 s): R@1 0.626 at 376×240 and 0.677 at 752×480, against 0.444 for the mean-bit ORB placeholder and
> 0.167 for random. Cost ≈ 150–200 ms and 650–850 ms per descriptor (shared CPU).
> **`cv::ANNIndex` findings (pinned source):**
> - `addItems` numbers items from 0 on every call;
> - no adds after `build()`;
> - no filter during search;
> - the multithreaded build seeds every thread alike over `hardware_concurrency()`, so repeatability across core
>   counts is not guaranteed.
>
> At ≤ thousands of keyframes per agent an exact gemm is the default (D62). `ANNIndex` is the scale ablation:
> top-1 identical to exact on 297/297 queries at N ≤ 181.

**CosPlace** — Berton et al., **CVPR 2022**, [code (MIT)](https://github.com/gmberton/CosPlace). Swarm-SLAM's choice. Reframes VPR training as **classification** over geographically-partitioned "CosPlace Groups" with a cosine-margin classifier, scaling to 41M images without mining. Superseded by EigenPlaces on viewpoint robustness; same license and interface.
⚠️ *Both are trained on street-view imagery — indoor darkness is out of distribution. This is the one place where fine-tuning or domain adaptation would pay off.*

**MegaLoc** — Berton & Masone, 2025, [arXiv:2502.17237](https://arxiv.org/abs/2502.17237), [code (MIT)](https://github.com/gmberton/MegaLoc). Combines DINOv2-Base + SALAD aggregation + multi-dataset training into one model that is SOTA on many VPR datasets, strong on landmark retrieval, and **new SOTA for Visual Localization on LaMAR by changing only the retrieval method**. **8448-D → 33.8 KB/keyframe.**
**→ Use as the accuracy ceiling in your ablation, not as the deployed descriptor — unless you compress.** fp16 halves it; int8 quarters it; **PCA-whitening to 256-D before transmission gets you to 1 KB while retaining most retrieval quality. That compress-then-transmit trick is itself a cheap, defensible Ariadne contribution.**

**Also:** **AnyLoc** (RA-L 2023, BSD-3) — VLAD over **frozen DINOv2 ViT-G/14**, *no training*, works "anywhere" including subterranean and degraded domains where trained VPR collapses. **Strongest a-priori claim for disaster settings — but ~196 KB/descriptor. Run once offline as a robustness reference; too heavy to deploy.** **BoQ** (CVPR 2024, MIT) — learns *global learnable queries* probing local features via **cross**-attention rather than input-derived self-attention; excellent but 12k–16k-D. **NetVLAD** (CVPR 2016) — still the right thing to reimplement if you want to write an aggregator yourself: ~80 lines on any CNN backbone, easy to explain. **DBoW2** — hierarchical k-means vocabulary tree + inverted index; **Low-Med to reimplement (~400 lines from `cv::kmeans` + TF-IDF), and the only zero-neural-network option** — a meaningful robustness property when your ONNX runtime dies in the field. ⚠️ License clause 3 requires notifying the original author.

**SALAD** (CVPR 2024) — ⛔ GPL-3.0. Reformulates NetVLAD's feature-to-cluster assignment as **optimal transport** with a **dustbin** to discard uninformative features. *Conceptually worth stealing: in darkness and dust, most of the image is uninformative, and a method that can say so is exactly right.*

### 1.7 Three-layer outlier rejection — build all three, they're cheap

1. **USAC** — per-candidate 2-view geometric verification. **Free: it's OpenCV 5's default backend.**
2. **PCM** — Mangelson, Dominic, Eustice, Vasudevan, **ICRA 2018**, "Pairwise Consistent Measurement Set Maximization for Robust Multi-Robot Map Merging." Two inter-robot loop closures are *pairwise consistent* if composing one loop closure, the odometry between the two keyframes on each robot, and the inverse of the other returns near-identity — a **Mahalanobis/χ² test**. Build a consistency graph over candidates and take the **maximum clique**. Needs no initial trajectory estimate and no iterative optimization. *Difficulty: **Low-Med** — the check is ~30 lines; use a greedy/heuristic clique finder.* ⚠️ Use as a **pre-filter, not the sole mechanism** (see the Kimera-Multi quote in §1.5). In Kimera-RPGO (BSD-2) and DOOR-SLAM (MIT).
3. **GNC** — optimizer-level weighting. See §1.3. The primary mechanism.

### 1.8 ⭐ The bandwidth blueprint — ordered by bytes-saved per hour of work

| # | Mechanism | Source | Difficulty | Expected effect |
|---|---|---|---|---|
| 1 | **Two-stage exchange** — broadcast compact global descriptors; send local descriptors only for selected candidates | Cieslewski'18 / DOOR-SLAM | **Low** | Foundational — 100–1000× vs sharing maps |
| 2 | **Transmission bookkeeping** — never send the same thing twice; heartbeat neighbour tracking | Swarm-SLAM | **Low** | Large in practice, trivial to build |
| 3 | **Elected-broker centralized GNC** instead of distributed PGO | Swarm-SLAM Table II | **Low** | **4×–740× fewer backend bytes, *and* better ATE** |
| 4 | **Compact descriptors** — EigenPlaces @128–256D, XFeat @64D | §1.2, §1.6 | **Low** | 8–128× vs BoQ/AnyLoc; 2–4× vs SuperPoint locals |
| 5 | **Vertex-cover local-feature exchange** | Giamou'18 / Swarm-SLAM | **Low-Med** | Eliminates redundant keyframe transmission |
| 6 | **Spectral (λ₂) candidate prioritization** under budget B | Swarm-SLAM / Doherty'22 | **Medium** | Target ATE at a fraction of loop closures verified |
| 7 | **Descriptor quantization** — fp16 / int8 / PCA-whiten to 256-D pre-transmit | *your own contribution* | **Low** | Free 2–4× on top of everything above |

**Instrumentation:** wrap every peer-to-peer message in a **single serialization chokepoint** incrementing a byte counter tagged by category (`global_desc`, `local_desc`, `pose_graph`, `heartbeat`, `map`). Headline plot: **ATE vs cumulative bytes exchanged**, one curve per configuration — Swarm-SLAM's Fig. 3 but with the more honest x-axis. **That plot is the demo.**

### 1.9 Recommended stack

**Per-agent front-end** — keyframe-based, SVO/ORB-SLAM3 architecture (tracking / local mapping / loop closing threads). Frame-to-frame: `calcOpticalFlowPyrLK` on previous keyframe's keypoints → sparse direct alignment as upgrade. Keyframe features: **`cv::XFeat`** (64-D), with `cv::ALIKED` and `cv::ORB` behind the same interface for ablation. Matching: mutual-NN `BFMatcher` → **`cv::LightGlueMatcher`**. Robust geometry: `geometry::solvePnPRansac`, `findEssentialMat`, `triangulatePoints` — USAC by default.

**Per-agent backend** — windowed BA (10–15 keyframes, Schur complement), local pose graph with GNC+TLS. *Both from scratch.*

**Place recognition** — **EigenPlaces R50 @ 256-D** via ONNX; retrieval via **`features::ANNIndex`**; optional zero-network fallback: own DBoW-style vocabulary tree from `cv::kmeans`.

**Multi-agent protocol** — two-stage exchange + bookkeeping + heartbeats; **spectral λ₂ prioritization** under budget B; vertex-cover exchange policy; **USAC → PCM → GNC**; **elected-broker centralized GNC-PGO** on rendezvous + anchor propagation; optional SE-Sync-style optimality certificate.

**Highest value-per-effort, ranked:** (1) **GNC** — ~60 lines for 70–80% outlier robustness; (2) **elected-broker backend** — a design decision worth 4–740× in bytes; (3) **descriptor dimension choice** — a 128× swing, free; (4) **spectral λ₂ prioritization** — the genuinely novel reimplementation; (5) **SE-Sync certificate** — a provable-optimality claim for ~150 lines.

---


---

## ⚠️⚠️ The Inria rasterizer trap — the most important finding in the 3DGS research

**Verified independently by two research passes, by fetching `.gitmodules` and vendored `LICENSE.md` files directly.**

Nearly every 3DGS system depends on Inria/MPII's `diff-gaussian-rasterization` and/or `simple-knn` CUDA code, which carries the **non-commercial research-only "Gaussian-Splatting License."** **This survives a permissive top-level license.** Repos declaring MIT / Apache-2.0 / BSD-3 at the root that nonetheless ship Inria non-commercial code:

| Repo | Declared | Actual |
|---|---|---|
| SplaTAM | BSD-3 | ⛔ Inria rasterizer submodule |
| **`google-research/Splat-SLAM`** | **Apache-2.0** | ⛔ **Google's own release pulls the Inria rasterizer + `simple-knn`** |
| LoopSplat | MIT | ⛔ pins two Inria-lineage rasterizers in `requirements.txt` |
| GS-ICP-SLAM | MIT | ⛔ lab fork of the Inria rasterizer |
| GauS-SLAM | MIT | ⛔ pins `diff-gaussian-rasterization-w-depth` |
| S3PO-GS | MIT | ⛔ rasterizer vendored **in-tree** (so no `.gitmodules` to warn you) |
| IG-SLAM | Apache-2.0 | ⛔ pulls `simple-knn` **direct from `gitlab.inria.fr`** |
| DROID-Splat | Apache-2.0 | ⛔ three rasterizer forks + `simple-knn` |
| VastGaussian (unofficial) | Apache-2.0 | ⛔ Inria rasterizer + simple-knn + SIBR |
| DOGS | Apache-2.0 | ⛔ LICENSE itself carves out files under the 3DGS license |
| Grendel-GS | Apache-2.0 | ⛔ its rasterizer fork is the Inria license verbatim — **but it also ships a `gsplat` submodule = escape path** |
| Momentum-GS | — | ⛔ `LICENSE.md` **is** the Inria license verbatim |
| CityGaussian / V2 | — | ⛔ **CC BY-NC-SA 4.0 *and* Inria submodules** (doubly non-commercial) |
| Hierarchical 3DGS | — | ⛔ Inria non-commercial, explicit |
| Fed3DGS | — | ⛔ Denso custom: *"for non-commercial purposes"* |
| MonoGS | — | ⛔ **Imperial College non-commercial *and* Inria rasterizer** (doubly) |
| BlockGaussian, CityGS-X, EC3R-SLAM, M³, CoMA-SLAM, MASt3R-Fusion | — | ⚠️ **NO LICENSE FILE** → all rights reserved by default |
| `eriksandstroem/Splat-SLAM` | — | ⚠️ No LICENSE file → effectively unusable |
| GSORB-SLAM | README says MIT | ⚠️ GitHub detects **GPL-3.0** (`License-gpl.txt` present) — treat as GPL |

**Clean exceptions:** ✅ **`gsplat` (Apache-2.0)** — 5,602★, last commit 2026-08-20. **This is the escape hatch; build on it.** ✅ **GLORIE-SLAM (Apache-2.0 all the way down)** — but it uses a neural point cloud, not Gaussians. ✅ **VGGT-SLAM (BSD-2, no submodules at all).** ✅ **RTG-SLAM** appears to use a from-scratch renderer.

**Model weights are separately encumbered:** all **DUSt3R/MASt3R weights are CC BY-NC-SA 4.0**; default **VGGT weights are non-commercial** (only the separate `VGGT-1B-Commercial` checkpoint is cleared, as of 2025-07-29); **Pi3** splits code BSD-3 / weights CC BY-NC 4.0.

> **Calibration:** none of this blocks the hackathon — non-commercial is fine for a competition entry. It bites only if Ariadne becomes a product. **But the competition report should state the license posture explicitly** — it reads as engineering maturity, and the rubric's *"responsible operation"* row rewards it. Build on `gsplat` and the cost of staying clean is near zero.

---

## 2. Classic 3DGS-SLAM systems

### 2.1 The three that matter

**⭐ Photo-SLAM (CVPR 2024)** — [arXiv:2311.16728](https://arxiv.org/abs/2311.16728), [code](https://github.com/HuajianUP/Photo-SLAM) · **GPL-3.0** (built on ORB-SLAM3) · 761★
*Sensor:* **monocular, stereo AND RGB-D** — widest support surveyed, inherited from ORB-SLAM3.
*Speed:* **the fastest of the entire cohort.** RTX 4090: tracking **41–58 FPS**, rendering 900–2000+ FPS. Laptop RTX 3080 Ti: 19–27 FPS. **Jetson AGX Orin: tracking 18–28 FPS, rendering 93–380 FPS.**
*Memory:* models **4–35 MB**; 4–6 GB GPU on desktop *and* Jetson.
*Robustness:* **DEMONSTRATED on Replica, TUM, EuRoC, and real Jetson embedded hardware** — the broadest validation in the survey.
*Core idea:* decouples classical ORB-SLAM3 tracking from a parallel photorealistic-mapping thread that spawns Gaussians around keyframes and optimizes them via a Gaussian-pyramid multi-resolution photometric loss. **No gradient flows from photometric loss into pose.**
*Why it's 10–100× faster:* it removes the exact bottleneck that makes SplaTAM/MonoGS/GS-SLAM slow — differentiable-render-based tracking.
*Difficulty:* **Medium.**
> **This decoupled-tracking pattern is the same architectural insight as §1.1's SVO recommendation, arrived at independently. It is the right shape for Ariadne: a robust classical front-end, with Gaussians as a mapping consumer rather than a tracking mechanism.**

**⭐ RTG-SLAM (SIGGRAPH 2024)** — [arXiv:2404.19706](https://arxiv.org/abs/2404.19706), [code](https://github.com/MisEty/RTG-SLAM) · **GPL-3.0**, no Inria dependency found (custom renderer) · 506★, **last push Dec 2025 — most actively maintained**
*Sensor:* RGB-D required.
*Speed:* **17.24 FPS (Replica) / 17.90 FPS on a real ~70 m² Azure Kinect home scan**, RTX 4090 — genuinely real-time online capture, not offline replay.
*Memory:* **2.75–8.8 GB**, final model 71–399 MB. **46× faster than SplaTAM.**
*Robustness:* **DEMONSTRATED at real building scale** — corridor, storeroom, hotel room, home, office, 43–100 m². **The only method in the survey independently shown to work at building scale on real hardware.**
*Core idea:* classify every Gaussian **stable** (frozen) or **unstable** (actively optimized); force near-opaque-or-transparent Gaussians so **one Gaussian suffices per surface point** instead of many overlapping transparent ones; track by frame-to-model **ICP**, not differentiable rendering.
*Difficulty:* Medium–High (the stable/unstable state machine + custom CUDA).
> **Reimplement this architecture even if you don't use the code.** It is the direct answer to unbounded Gaussian growth, which is what kills every other system at Ariadne's scale.

**GS-ICP-SLAM (ECCV 2024)** — [arXiv:2403.12550](https://arxiv.org/abs/2403.12550), [code](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM) · MIT top-level ⛔ Inria rasterizer fork · 534★, **last push June 2026**
*Sensor:* RGB-D. *Speed:* **up to 107 FPS unlimited / 30 FPS rate-limited** on RTX 4090 — fastest RGB-D number surveyed, independently reproduced.
*Core idea:* fuses Generalized-ICP with 3DGS via **ONE shared representation** — covariances serve as both the ICP surface model and the splat map, avoiding two scene representations.
*Difficulty:* Medium (G-ICP is well-trodden; the shared-covariance exchange is the novel, well-scoped part).
⚠️ *Its own issue tracker shows real brittleness off-benchmark:* a reproducible `Floating point exception` crash on TUM freiburg1_desk "in pure vanilla source" (#61), multiple perf-collapse reports, NumPy 2.0 breakage. **Budget hardening time.**

### 2.2 The rest — idea sources only

| System | Sensor | Speed | Memory | Verdict |
|---|---|---|---|---|
| **SplaTAM** (CVPR'24) | RGB-D | **0.41 FPS** (0.14–0.51 across 5 independent reproductions) | 9.1–12.1 GB; **OOM on a ~70 m² home scan** (7.15M Gaussians) | ⛔ **Poor base.** Not real-time; documented to OOM at exactly Ariadne's scale. |
| **MonoGS** (CVPR'24 oral) | **Mono/stereo/RGB-D** | 3.2 FPS mono / 2.5 RGB-D (RTX 4090) | **2.6–3.97 MB** — smallest surveyed | Great sensor flexibility + tiny map. ⛔ **Doubly non-commercial** (Imperial + Inria) → reimplement the analytic pose-Jacobian from the paper, don't reuse code. |
| **GS-SLAM** (CVPR'24) | RGB-D | 8.34 FPS | 198 MB | ⛔ **NO CODE EVER RELEASED** — project page link is `disabled`. Design inspiration only. |
| **Gaussian-SLAM** | RGB-D | 0.57 FPS | — | MIT, likely Inria-encumbered. Good submap-structuring ideas. |
| **LoopSplat** (3DV'25) | RGB-D | 0.57 FPS + 1.36 s/loop-edge | **7.0 GiB vs SplaTAM's 18.5** | ⭐ **Loop closure as a Gaussian-native operation** — registers two Gaussian submaps directly. *Conceptually adjacent to multi-agent map merging.* MIT ⛔ Inria-pinned. |
| **Splat-SLAM** (Google) | **Monocular** | 1.24 FPS | 17.57 GiB, 6.5 MB map | DROID-style global-BA tracker + Gaussians re-anchored on BA updates. Apache-2.0 ⛔ Inria. |
| **GLORIE-SLAM** | Monocular | 0.23 FPS | 15.22 GiB | ✅ **Apache-2.0, genuinely clean** — but it's a neural point cloud, **not 3DGS**. |
| **GauS-SLAM** (RAL'26) | RGB-D | 0.59 / 1.72 FPS (fast variant) | — | 2D surfels + **submap-level bundle adjustment** — relevant to multi-robot fusion. MIT ⛔ Inria. |
| **GSORB-SLAM** (RA-L'25) | RGB-D | — | — | ORB tracking + **transmittance-guided** Gaussian expansion/pruning. Ran on an **RTX 4060 Ti 16GB** — least powerful GPU surveyed. ⚠️ License conflict (README MIT vs GPL-3.0 detected). |
| **DROID-Splat** | Mono/stereo/RGB-D | interactive | uint8 buffers for long seqs | Broadest trajectory-type validation (indoor+outdoor+driving). Apache-2.0 ⛔ Inria. High difficulty. |
| **MonoGS++** (BMVC'24) | Monocular | 2.84 FPS | −⅓ Gaussians | ⛔ No code. Three cheap densification ideas to fold into a MonoGS reimplementation. |
| **G2S-ICP SLAM**, **GLC-SLAM** | RGB-D | ~30 / 0.53 FPS | — | ⛔ No code. G2S-ICP: tangent-plane-constrained anisotropic disks. GLC-SLAM: **uncertainty-based subsampling for cheap loop closure**. |

---

## 3. Feed-forward pointmap SLAM & multi-agent 3DGS

### 3.1 ⭐ The competitive gap — read this

> **Multi-agent 3DGS-SLAM defaults to RGB-D almost everywhere.** HAMMER, coko-slam, and (by dataset/baseline inference) GRAND-SLAM and CoMA-SLAM all assume depth. **MAGS-SLAM and CGS-SLAM are the only multi-agent methods found that are explicitly monocular-first — and neither has released code.**

**This is the gap Ariadne would fill.** A working monocular multi-agent collaborative reconstruction system is not a crowded space; it is an open one. That is a defensible novelty claim for the report, and it is unusually well evidenced.

### 3.2 Backbones

| Model | Venue | Code license | Weights | Note |
|---|---|---|---|---|
| DUSt3R | CVPR 2024 | ⛔ CC BY-NC-SA 4.0 | ⛔ NC | 7.3k★. The ancestor. |
| MASt3R | CVPR 2024 | ⛔ CC BY-NC-SA 4.0 | ⛔ NC + training-dataset licenses | 3.1k★. Adds matching + metric-scale heads. |
| VGGT | CVPR 2025 **Best Paper** | ⚠️ Meta custom (Llama-style) | ⛔ NC by default; ✅ `VGGT-1B-Commercial` cleared 2025-07-29 | 14.3k★. Single forward pass → cameras + depth + pointmaps + tracks. |
| **π³ / Pi3** | ICLR 2026 | ✅ **BSD-3** | ⛔ CC BY-NC 4.0 | 2.1k★. **Permutation-equivariant** — no privileged reference frame, so frame order doesn't bias reconstruction. **57.4 FPS** vs VGGT 43.2, DUSt3R 1.25. |

### 3.3 🚨 The VGGT memory wall — architectural constraint

From **VGGT-Long** ([arXiv:2507.16443](https://arxiv.org/abs/2507.16443), ICRA 2026), verified from the paper:

> *"VGGT can just process 60 to 80 images on a 24 GiB RTX 4090 GPU… scaling to a KITTI Seq 00 trajectory (about 4,600 frames) would require an unreachable GPU memory requirement."*

Standard VGGT **OOMs on every full KITTI sequence tested**. VGGT-Long fixes this by chunking (~2.6–2.8 s/chunk) and stitching with a Sim(3) Levenberg-Marquardt solver (~3 iterations, <15 ms/step). ⛔ Its code inherits Meta's non-commercial VGGT license wholesale.

> **⇒ Budget submaps at ≤60–80 frames per 24GB card, and design a chunk-and-align architecture from day one. Never treat VGGT as a per-frame tracker.**

### 3.4 Single-agent systems, ranked

**⭐ MASt3R-SLAM (CVPR 2025)** — [arXiv:2412.12392](https://arxiv.org/abs/2412.12392), [code](https://github.com/rmurai0610/MASt3R-SLAM) · ⛔ CC BY-NC-SA 4.0 (verified from `LICENSE.md`) · 3.2k★
**Monocular, no depth.** **~15 FPS on a single RTX 4090** (paper's own number). No Inria entanglement — it is pure pointmap dense SLAM, not 3DGS. Replaces the classical geometric front-end with MASt3R's learned two-view pointmap prior, wrapped in a real-time backend.
⚠️ Open issue #96 *"Concerns about MASt3R's robustness in feature-sparse environments"* — **directly relevant to texture-poor rubble.** Recurring OOM reports (#64, #18).
**Best-verified single-agent starting point.**

**⭐ VGGT-SLAM 2.0** — [arXiv:2601.19887](https://arxiv.org/abs/2601.19887), [code](https://github.com/MIT-SPARK/VGGT-SLAM) · ✅ **BSD-2, no submodules** · 1.1k★
Monocular, **uncalibrated** (unknown intrinsics OK). **8.4 FPS** (6.3 with CLIP semantics) for 16-frame submaps on an RTX 3090. **23% lower pose error than 1.0**, and removes 1.0's KITTI planar-degeneracy divergence.
**The only method in the survey demonstrated running onboard a real robot** — *"running our entire [pipeline] onboard a Jetson Thor"* — plus a 4,200 sq-ft barn reconstruction from a handheld iPhone. Reuses one of VGGT's own attention layers, **training-free, as a loop-closure verifier** — directly reusable for cross-agent place recognition.
*(v1.0 had documented brittleness: drift #15, submap mismatch #22, singular SL(4) matrices #5/#34, KITTI divergence. 2.0 fixes these.)*

**⭐ EC3R-SLAM (ICRA 2026)** — [arXiv:2510.02080](https://arxiv.org/abs/2510.02080), [code](https://github.com/hulxgit/EC3R-SLAM) · ⚠️ no LICENSE file · 61★
**Monocular, calibration-free.** **The strongest embedded-hardware evidence in the survey:**

| Hardware | EC3R-SLAM | MASt3R-SLAM | DROID-SLAM | VGGT-SLAM |
|---|---|---|---|---|
| Desktop | **27 FPS** | — | — | — |
| Laptop RTX 4060 (8GB) | **15 FPS** | 2 FPS | ❌ failed | ❌ **failed** |
| Jetson Orin NX | **7 FPS** | 0.1 FPS | 1 FPS | ❌ **failed** |

**<10 GB GPU memory** vs "concurrent methods usually beyond 20GB."
*Core idea:* **decouples a lightweight XFeat-based sparse tracker (cheap, per-frame) from a VGGT-based feed-forward mapper (expensive, keyframe-only)** — the expensive network runs only occasionally while tracking stays real-time.
> **Note the convergence:** Photo-SLAM (§2.1), SVO (§1.1), and EC3R-SLAM independently arrive at the same architecture — *cheap classical tracking, expensive learned mapping at keyframes only.* **And EC3R-SLAM's cheap tracker is XFeat, which §1.2 already recommends and which is a native OpenCV 5 class.** This is the strongest cross-domain signal in the research.

**Also:** MASt3R-Fusion (⚠️ no license; **8 GB floor**, RTX 4080 Mobile real-time, adds IMU/GNSS via a GTSAM factor graph) · M³ ([arXiv:2603.16844](https://arxiv.org/abs/2603.16844) — conceptually the closest match to Ariadne, **but the repo is mechanically verified as a stub**: every source file 404s, only `README.md` and `assets/` exist) · S3PO-GS (ICCV'25, outdoor, MIT top-level ⛔ Inria in-tree; **explicitly no loop closure and no dynamic-object handling** — disqualifying gaps) · IG-SLAM (9.94 FPS on a 4090, 8× faster than Splat-SLAM) · OpenMonoGS-SLAM (no code; open-vocabulary semantics on monocular).

### 3.5 Multi-agent systems

**⭐⭐ MAGS-SLAM** — [arXiv:2605.10760](https://arxiv.org/abs/2605.10760) · **no code** (under double-blind review)
> *"most existing multi-agent Gaussian SLAM methods still rely on RGB-D sensors... we propose MAGS-SLAM, the first RGB-only multi-agent 3DGS SLAM framework."*

**The single most Ariadne-shaped abstract in the entire survey.** Per-agent tracking **13–16 FPS**; peak GPU **<3.9 GB per agent** ("several agents fit on one device"); coordinator-side retrieval + Sim(3) pose-graph optimization within a 33 ms budget; **communication reduced 45–520× vs transmitting raw RGB** by sending compact submap summaries.
*Architecture:* each agent runs independent monocular Gaussian SLAM locally → shares compact submap summaries → coordinator does geometry- and appearance-aware loop verification + occupancy-aware Gaussian fusion, handling monocular scale ambiguity.
**Read this closely. It is the closest existing description of what Ariadne needs to build — and it has no code, which is exactly why building it is a contribution.**

**HAMMER (RA-L 2025)** — [arXiv:2501.14147](https://arxiv.org/abs/2501.14147) · no code found
⚠️ **RGB-D**, explicitly: *"the current implementation does not account for RGB-only devices."* But: **DEMONSTRATED in real hardware trials** (>40% better MSE vs Di-NeRF). Server-centralized: robots stream posed keyframes; the server does **frame alignment with no prior relative pose needed** + continual online 3DGS + CLIP semantics. SfM alignment ~33.6 s/device one-time; streaming 15 ms (66 Hz) sparse → 38 ms (26 Hz) dense on an RTX 4090.
> **Strong architectural template for Ariadne's coordinator side** — particularly alignment without known relative poses.

**Multi-Agent Monocular Dense SLAM with 3D Reconstruction Priors** — [arXiv:2511.19031](https://arxiv.org/abs/2511.19031) · no code. Extends MASt3R-SLAM to N agents with loop-closure-based map fusion. *"MASt3R-SLAM is limited to single-agent operation... we extend [it] to introduce the first multi-agent monocular dense SLAM system."* **Design template on top of the well-verified MASt3R-SLAM base.**

**coko-slam** — [arXiv:2604.00804](https://arxiv.org/abs/2604.00804), [code](https://github.com/lemonci/coko-slam) · ✅ **MIT (verified)** · RGB-D. **85–95% reduction in transmitted data** via Gaussian compaction + centralized loop closure needing no initial pose guess. **The bandwidth-management pattern is worth porting into a monocular pipeline even though its sensor assumption doesn't match.**

**Also:** CGS-SLAM ([arXiv:2608.26868](https://arxiv.org/abs/2608.26868), 2026-08-27 — RGB+IMU, GPS-denied, hybrid decentralized; days old) · CoMo3R-SLAM ([arXiv:2605.30488](https://arxiv.org/abs/2605.30488) — *"first collaborative monocular dense RGB SLAM... for outdoor multi-agent mapping"*; **flagged for follow-up**) · GRAND-SLAM ([arXiv:2506.18885](https://arxiv.org/abs/2506.18885) — 91% lower multi-agent tracking error, but likely RGB-D/stereo by inference) · CoMA-SLAM (AAAI 2026, 1-commit repo, no license) · MCN-SLAM ([arXiv:2506.18678](https://arxiv.org/abs/2506.18678) — triplane implicit, communicates only network params + keyframe descriptors; **bandwidth-bounded map representation precedent**) · SIREN ([arXiv:2502.06519](https://arxiv.org/abs/2502.06519) — *Semantic, Initialization-Free Registration of Multi-Robot Gaussian Splatting Maps*) · [Survey on Collaborative SLAM with 3DGS](https://arxiv.org/abs/2510.23988).

---

## 4. Partitioned / distributed 3DGS — the merge mechanisms

Structurally identical to multi-agent map merging, which is why the **merge mechanism** is the transferable part.

### 4.1 ⭐ Start here — ~200 lines that solve both problems

Both from **Hierarchical 3D Gaussians** (SIGGRAPH 2024, Kerbl/Inria, [arXiv:2406.12080](https://arxiv.org/abs/2406.12080)) — ⛔ Inria non-commercial, but these two mechanisms are trivially reimplementable. Notably it runs on **1× V100, ~3 hours total** — the most affordable full pipeline surveyed.

**Merge — nearest-chunk overlap pruning.** *"If a primitive associated to — but outside — chunk i is closer to another chunk j≠i, it is deleted."* Chunks deliberately overlap during training, then boundary duplicates are pruned by nearest-chunk assignment. **The simplest correct answer to "two robots both reconstructed this corridor."** Implement this as your baseline, then measure what fancier options buy.

**Exposure — per-image 3×4 affine color transform.** `C_c = E[C|1]ᵀ`, E initialized to identity, optimized per-camera with Adam. **12 parameters per keyframe. No CNN, no embedding.**
> ⚠️ **Critical detail: delay exposure optimization by ~5000 steps** so geometry stabilizes first. Otherwise exposure absorbs geometric error and you get mush. Frozen after the per-chunk stage.

**Ariadne will hit exposure hard** — different robots, independent auto-exposure, wildly varying illumination under self-carried lighting. **The large-scale-3DGS field has largely punted on this**: CityGaussian (*"we leave solving this issue for future works"*), CityGaussianV2, DOGS (*"we did not implement"*), BlockGaussian, Grendel, Splaxel, UrbanGS all have nothing.

### 4.2 ⭐ Fed3DGS — the closest published analogue to Ariadne's merge

[arXiv:2403.11460](https://arxiv.org/abs/2403.11460), [code](https://github.com/DensoITLab/Fed3DGS) ⛔ Denso non-commercial · 111★, **abandoned since 2024-03**
Partitions **by client** (200–400 clients, 100–200 images each), not spatially. V100 at **0.29 h/client**; one A100 for global updates.

**The negative results are worth more than the positive ones:**
- *Naive append* (`G_g ∪ G_l`) grows the model monotonically until it explodes.
- **Voxel-grid filtering — the standard point-cloud merge — FAILS on 3DGS.** Their explanation: *"each Gaussian has a different role. For example, some are large and represent the base color of a covered area, while others are much smaller and capture detailed texture. Since voxel grid filtering averages Gaussians having different roles based on their position, it disrupts both appearance and geometry."*

> **That one paragraph will save you a week of building the wrong thing.** The obvious merge is wrong, and for a non-obvious reason.

**Their fix:** union, then **distill** — optimize opacity and appearance so the merged model matches renders from local cameras **plus replay cameras** drawn from an accumulated camera bank (this prevents catastrophic forgetting of earlier clients). Reset opacity logits to σ(o)=0.05, add an **entropy term pushing opacity to 0 or 1**, prune below 0.05.

### 4.3 Other transferable mechanisms

**⭐ CityGaussian's coarse global prior** (ECCV 2024, ⛔ CC BY-NC-SA + Inria) — **the cheapest seam prevention known.** Train a coarse global model on all views for 30k iterations first, then **initialize every block's finetuning from it**. *"Thanks to the global geometric prior, interference among blocks is significantly mitigated. Thus a high-quality overall model can be derived through direct concatenation."* Blocks never diverge because they all started from the same geometry. **Maps directly onto Ariadne: a coarse joint map, then per-robot refinement.**

**⭐ VastGaussian's decoupled appearance CNN** (CVPR 2024) — the best-documented appearance mechanism in the literature and **more expressive than affine** (handles spatially-varying effects like glare). Downsample the render 32×, concatenate a **64-dim per-image appearance embedding** to every pixel → CNN → an `H×W×3` transformation map. Loss on both raw and adjusted renders, so appearance variation is absorbed by the CNN and **not baked into the Gaussians**. **Discarded entirely at inference** — your delivered splat stays clean. Also: its **airspace-aware visibility** (project the cell's 3D bounding box, not just surface points) is what kills floaters — generalizes to "the volume above/around a region" in a collapsed building.

**⭐ F3DGS** ([arXiv:2604.01605](https://arxiv.org/abs/2604.01605), no code) — *"Federated 3DGS for Decentralized Multi-Agent World Modeling."* Structurally the closest paper to Ariadne. Three independently implementable ideas: **(1) freeze Gaussian center positions** throughout distributed optimization — agents update only scale/rotation/opacity/SH, so geometric drift is eliminated by construction; **(2) Umeyama Sim(3)** registration to a global anchor + **exponentially decayed SE(3) corrections over a smoothing window** for boundaries; **(3) visibility-aware weighting** `α_k,i = v_k,i / Σ_j v_j,i` — *the robot that observed a region most dominates that region's appearance.* Indoor, 9 sequences, 8,739 frames. ⚠️ Hardware and training time not reported at all.

**⭐ StitchGS** (*Remote Sensing* 18(10), 2026, no code) — the only mechanism that **blends rather than deletes** in the overlap: **Oriented Bounding Boxes defining soft transition zones**, then **confidence-driven competition for sub-pixel fusion**. Frames the exact problem: *"continuous geometric surfaces like walls and pavements are inevitably truncated by rigid partitioning, leading to visible physical seams and abrupt illumination discontinuities at block interfaces after merging."*

**OccluGaussian** ([arXiv:2503.16177](https://arxiv.org/abs/2503.16177)) — ⭐ **occlusion-aware** region partitioning instead of distance-based. **Most relevant Tier-2 entry for Ariadne: in a collapsed building, spatial proximity ≠ mutual visibility, and every other method here partitions on proximity.**

**Grendel-GS's √batch-size LR rule** — the **Independent Gradients Hypothesis** (gradients from different camera views are approximately statistically independent) yields `λ' = λ√b` with `β'₁,β'₂ = β₁^b, β₂^b`. Linear scaling performed worse. **Free and correct — apply it the moment you train with batch size > 1 anywhere.** (The rest of Grendel needs 16–32× A100 + NVLink; on one GPU it degenerates to plain 3DGS with batching.)

**Train-time schemes** (only if agents communicate during mapping): **DOGS** ADMM consensus every 100 iters, `z ← (1/K)Σx_i` — seams never form because shared Gaussians are continuously pulled toward consensus (⚠️ zero appearance handling, explicitly declined). **Momentum-GS** — shared Gaussian decoder + **momentum teacher** `θ_t ← m·θ_t + (1−m)·θ_s` + consistency loss; blocks trained in staggered 500-iter rotations; **runs on 4× RTX 3090** and beats CityGaussian by 12.8% LPIPS using 8 blocks vs 9–36. ⚠️ Consistency ≠ exposure: it makes blocks agree with each other, but won't correct a robot whose auto-exposure ran two stops hot. **CityGS-X** (ICCV 2025) — shared decoder + gradient sync *"eliminates the merge-and-partition process"*; **4× RTX 4090, 5 hours** for 5,000+ images, where CityGaussianV2 OOMs. ⚠️ **no license file.**

### 4.4 Compute reality check

🟢 **Within budget:** Hierarchical-3DGS (**1× V100, ~3h — cheapest full pipeline**) · Momentum-GS (**4× RTX 3090**) · CityGS-X (**4× RTX 4090, 5h**) · BlockGaussian (**8× RTX 4090, 52 min**) · Fed3DGS (**0.29 h/client**).
🚨 **Out of budget:** CityGaussianV2 (**8× A100**) · Splaxel (**8× RTX 6000 Ada/Blackwell**) · Grendel at scale (**16–32× A100 + NVLink + 200Gbps Slingshot**).

### 4.5 Honesty ledger

**Code exists and is verifiable:** CityGaussian/V2, Hierarchical-3DGS, Grendel-GS, DOGS, Momentum-GS, BlockGaussian, CityGS-X, Fed3DGS (abandoned), VastGaussian (**unofficial, author-declared "messy", never officially released**).
**Paper claims only, nothing independently verified:** Splaxel, F3DGS, StitchGS, UrbanGS, LoBE-GS, NVIDIA billion-splat abstraction, CoSurfGS (**stub — 20 months stale, "Brewing🍺, code coming soon"**).

> ⚠️ **Every "seamless" claim in this literature is author-asserted on aerial/urban datasets with broadly consistent lighting. None was evaluated on independently-captured, differently-exposed, indoor, occlusion-heavy data — i.e. none was evaluated on Ariadne's actual problem. Budget time to discover the seams are worse than advertised.**

---

## 5. Cross-domain synthesis

**The architecture three independent literatures converge on:** cheap classical tracking per frame, expensive learned perception at keyframes only.
- §1.1 SVO/SVO Pro (2017): direct alignment for tracking, features at keyframes
- §2.1 Photo-SLAM (2024): ORB-SLAM3 tracking, Gaussians as a mapping consumer — 10–100× faster than render-based tracking
- §3.4 EC3R-SLAM (2026): XFeat tracker, VGGT mapper at keyframes — the only method that runs at all on a Jetson Orin NX

**And EC3R-SLAM's tracker is XFeat — which §1.2 independently recommends and which is a native OpenCV 5 class (`cv::XFeat`).** Adopt this shape.

**Ranked build order, by value per unit effort:**

| # | Item | Source | Effort | Payoff |
|---|---|---|---|---|
| 1 | **GNC + TLS** on loop-closure edges | §1.3 | ~60 lines | 70–80% outlier robustness |
| 2 | **Elected-broker centralized PGO** (not distributed) | §1.4 | design decision | 4–740× fewer backend bytes |
| 3 | **Nearest-chunk overlap pruning** + **per-image 3×4 affine exposure** | §4.1 | ~200 lines | Baseline merge + exposure, both problems |
| 4 | **Compact descriptors** — XFeat 64-D, EigenPlaces 256-D | §1.2, §1.6 | selection | up to 128× bandwidth swing |
| 5 | **Two-stage exchange + transmission bookkeeping** | §1.8 | Low | foundational, 100–1000× |
| 6 | **Chunked submaps ≤60–80 frames** if using any VGGT backbone | §3.3 | architectural | avoids guaranteed OOM |
| 7 | **Spectral λ₂ candidate prioritization** | §1.4 | ~100 lines + gradient step | the flagship novel reimplementation |
| 8 | **SE-Sync optimality certificate** | §1.3 | ~150 lines | provable-global-optimum claim |
| 9 | **Fed3DGS distillation merge** with replay cameras | §4.2 | Medium | the real merge, once the baseline is measured |

**Two negative results that save weeks:**
1. **Voxel-grid averaging destroys 3DGS** — Gaussians have heterogeneous roles (§4.2). The obvious merge is wrong.
2. **Distributed PGO costs 4–740× more bytes than shipping the graph to an elected broker** (§1.4). The sophisticated option is worse on both bytes *and* accuracy at this team size.

**And one competitive finding:** monocular multi-agent collaborative 3DGS reconstruction has **two papers and zero released implementations** (§3.1). That is an open space, not a crowded one.

---

## 6. Datasets & benchmarks

**Legend.** ✅ = terms fetched and read · ⚠️ = page fetched, terms not stated · ❌ = unverified.
**PAIRED** = registered clean/degraded pairs (PSNR/SSIM/LPIPS possible) · **UNPAIRED** = no-reference metrics only.

### 6.1 ⭐ Primary: subterranean / dark-environment SLAM

**⭐⭐ SubT-MRS** — [arXiv:2307.07607](https://arxiv.org/abs/2307.07607), CVPR 2024 · download: https://superodometry.com/datasets · ⚠️ **no license statement** (email the authors)
**The single best-matched public dataset for Ariadne.** 6 primary sequences (SubT Finals UGV1-3, Urban UGV1-2, Laurel Cavern handheld); **7 sensor-degradation sequences: Multi-Floor, Long Corridor, BlockLiDAR, BlockVisual, SmokeRoom, OutdoorNight, FlashLight**; 8 SuperLoc localization sequences (4 cave, 2 corridor, 2 floor). Up to 4× RGB + Velodyne + **thermal** + IMU, on RC car / Spot / UAV / handheld. Degradations: **darkness, smoke, dust, snow, fog, geometric degeneracy, over/under-exposure, self-similar caves.** GT: 6-DoF trajectories + calibration.
> **The only dataset putting smoke, dust and total darkness on the same underground platform with thermal and LiDAR alongside RGB, with trajectory GT — so you can measure whether the restoration front-end actually improves *odometry*, not just image quality.** Note the `FlashLight` sequence: that is your self-carried-illumination case.

**DARPA SubT official ground truth** — index: [subt_resources/3-Datasets.md](https://github.com/subtchallenge/subt_resources/blob/main/3-Datasets.md) · ❌ per-repo licenses unverified
Survey-grade **course scan point clouds** = absolute map ground truth for the real environments the challenge ran in: [Tunnel (NIOSH mine)](https://github.com/subtchallenge/systems_tunnel_ground_truth) · [Urban (Satsop nuclear plant)](https://github.com/subtchallenge/systems_urban_ground_truth) · [Finals (Louisville Megacavern)](https://github.com/subtchallenge/systems_finals_ground_truth) · [Virtual](https://github.com/subtchallenge/virtual_ground_truth) · [ARL reference datasets](https://github.com/subtchallenge/tunnel_urban_reference_datasets) (LiDAR/camera/odometry from teleoperated runs).
**[SubT Virtual Testbed](https://github.com/osrf/subt/wiki)** — Gazebo/Ignition tunnel, urban and cave worlds + robot and sensor models. **The way to get arbitrary labelled degraded-underground data with controlled lighting — directly relevant to the simulator track.**

**Hilti SLAM Challenge 2023** — https://hilti-challenge.com/dataset-2023.html · ✅ **CC BY-NC-SA 3.0**
3 construction sites; **Site 3 is 4 underground sequences** (10–19 GB). Alphasense Core 5× global-shutter + Hesai PandarXT-32, <1 ms sync. **Unfinished concrete with poor lighting — the closest legally-clean analogue to a partially-collapsed interior.**

**Newer College** — https://ori-drs.github.io/newer-college-dataset/ · ⚠️ unverified
~6.7 km handheld. GT: **BLK360 survey-grade TLS, 290 M points, mm-accurate map → cm-accurate 6-DoF per scan.** Includes explicitly labelled **fast motion, aggressive shaking, rapid lighting change, and textureless surface** sequences — your perceptual-aliasing and auto-exposure stress test, with map-level GT.

**Oxford Spires** (IJRR 2025) — https://github.com/ori-drs/oxford_spires_dataset · ✅ **CC BY-NC-SA 4.0**
~10× Newer College. LiDAR + colour cameras + **Leica TLS ground-truth clouds.** Benchmarks for localization, 3D reconstruction, **radiance fields and novel-view synthesis** — the validation track for "did degraded input hurt the map."

**⭐ M2DGR** — https://github.com/SJTU-ViSYS/M2DGR · ⚠️ repo MIT is the *code* license; data terms unstated
**36 sequences, ~1 TB.** Includes **Dark Room: 6 sequences, 171 GB, 866 s**, plus Lift/elevator 4, Hall 5. Sensors: 6 fisheye + perspective RGB, **FLIR thermal 640×512 @25 Hz**, **Inivation DVXplorer event 640×480**, Velodyne VLP-32C, GNSS/RTK, multiple IMUs. GT: **Leica MS60 (1 mm + 1.5 ppm) and Vicon (1 mm, 50 Hz) indoors.**
> **The Dark Room sequences are complete-darkness indoor VO failures with mm-accurate GT and a co-mounted thermal + event camera — an almost perfect ablation rig for a dark-void agent.**

**FusionPortable V2** — https://fusionportable.github.io/dataset/fusionportable_v2/ · ⚠️ unverified. Handheld / UGV / **legged (Unitree A1) underground**; Ouster 128-beam, stereo FLIR, **stereo DAVIS346 event**, STIM300 INS. Underground parking, corridors, escalators.

**ColoRadar** (Team MARBLE) — https://arpg.github.io/coloradar/ ⚠️ *link source-attested via subt_resources; direct fetch 404'd.* **52 sequences, >145 min**, 3D FMCW radar + LiDAR + IMU, **in mines.** Radar is the only modality that ignores dust entirely — the reference set for a fallback channel.

**Could not verify:** CTU-CRAS-NORLAB SubT (expired certificate) · CERBERUS (SSL failure; NTNU-ARL has `subt_cave_sim` but **no released sensor dataset found**) · JPL/NeBula/CoSTAR (**no public dataset located**) · MineXplore ([arXiv:2606.04569](https://arxiv.org/abs/2606.04569), RL exploration benchmark for GNSS-denied mines) · CAVERS ([arXiv:2604.15052](https://arxiv.org/abs/2604.15052), karstic cave SLAM with mocap GT).

### 6.2 Indoor reconstruction GT — the controlled-ablation backbone

**Access friction decides your week-1 plan.** Start the gated applications on day one.

| Dataset | Friction | License | GT |
|---|---|---|---|
| **TUM RGB-D** | **none** | ✅ CC BY 4.0 | 8-camera mocap @100 Hz; ships ATE/RPE scripts |
| **TUM Mono VO** | **none** | ✅ CC BY 4.0 | loop-closure drift; **photometrically calibrated** |
| ETH3D | none, NC | ✅ CC BY-NC-SA 4.0 | MVS + RGB-D SLAM benchmark, public leaderboard |
| Replica | script, research-only | ✅ custom NC terms | 18 scenes, renderable meshes, **glass/mirror params** |
| ScanNet | **signed ToU, ~1 wk** | ✅ ScanNet ToU | >1,500 scans, 2.5M RGB-D views |
| ScanNet++ | **account + application + token** | ✅ ScanNet++ ToU | >1,000 scenes, **sub-mm laser + 33MP DSLR** |
| Matterport3D | **signed ToU** | ✅ in ToU doc | 90 building-scale scenes, 194,400 RGB-D |
| EuRoC MAV | direct | ❌ **unverified** (ETH record 500s) | Vicon + laser tracker |

**⭐⭐ TUM Mono VO** — https://cvg.cit.tum.de/data/datasets/mono-dataset · ✅ CC BY 4.0 · 50 sequences, >100 min, 43 GB.
**Uniquely photometrically calibrated: per-frame exposure times, camera response function, and lens vignetting.**
> **This is the only benchmark that lets you separate "the scene got darker" from "auto-exposure changed" — essential given self-carried illumination, which breaks the brightness-constancy assumption every direct VO method depends on. If you build one thing against one dataset first, build the photometric model against this.**

**Replica** — [arXiv:1906.05797](https://arxiv.org/abs/1906.05797) · ✅ custom research-only terms. 18 scenes, HDR textures, semantic+instance segmentation, **explicit glass/mirror surface parameterization**, Habitat navmeshes.
> **Photorealistic renderable GT means you can synthesize darkness/dust over known geometry and get exact reconstruction error — the only route to paired degraded/clean 3D ground truth at all.** This is the backbone of the controlled-ablation track.

**EuRoC caveat worth knowing:** the authors state per-camera **independent auto-exposure causes stereo brightness mismatch** — itself a mild DVE effect you can exploit as a free test case.

### 6.3 Low-light and dark VO — and the honest gap

> **⚠️ KEY FINDING: there is no widely-used public benchmark of headlamp-lit sequences with pose ground truth.** The nearest anchors are TUM Mono VO (photometric calibration lets you *model* the violation even though the light is fixed) and M2DGR Dark Room (complete darkness, mm GT).
>
> **⭐ But the method paper exists: [NFL-BA — Near-Field Light Bundle Adjustment for SLAM in Dynamic Lighting](https://arxiv.org/abs/2412.13176)**, which explicitly targets *"endoscopy, subterranean robotics, and search & rescue in collapsed environments"* where **the light moves with the camera.** This is the correct prior art for Ariadne's dominant condition. Read it first, and check its evaluation section for what data it used.
>
> **This gap is also an opportunity:** your simulator can generate exactly the missing thing — headlamp-lit sequences with perfect pose GT. That is a legitimate, citable contribution in its own right.

**DIDLM** — [arXiv:2404.09622](https://arxiv.org/abs/2404.09622) · ❌ license unverified · **18.5 km, 69 min, ~660 GB.** 4D mmWave radar + **infrared** + depth + 3D LiDAR + RGB + GPS/IMU, under snow, rain, **nighttime**, rough terrain. The only set combining 4D radar + IR + depth + LiDAR under low light.
**ASL Illumination-Robust Visual-Inertial Dataset** — https://projects.asl.ethz.ch/datasets/ir-dataset/ · ❌ unverified — **the most on-point remaining lead for illumination-change VIO.**

**Image-level low-light sets** (no pose): **LOL / LOLv2** (PAIRED, ⚠️/❌ unverified) · **SID/See-in-the-Dark** (PAIRED raw short↔long exposure, Sony+Fuji; ⚠️ repo MIT is code-only) · **SMID** (PAIRED video) · **ExDark** (✅ **BSD-style, verified**; UNPAIRED, 12 object classes) · **DarkFace/UG2+** (6,000 labelled + 9,000 unlabelled faces, **plus 789 paired low/normal images**) · **SDSD** (PAIRED video, ❌ no license) · LIME/NPE/MEF/DICM/VV (UNPAIRED conventional test sets).

### 6.4 Texture-poor / perceptual aliasing — no new dataset needed

The failure mode is already isolated inside sets verified above. Use these as the aliasing suite, at zero extra access friction:
- **SubT-MRS**: `Long Corridor`, `BlockVisual`, `Multi-Floor` — featureless corridors, stairs, self-similar caves
- **Hilti 2023 Site 3**: 4 underground sequences, bare concrete
- **Newer College**: the sequences the authors label *"textureless surface"* and *"rapid lighting change"*
- **M2DGR**: Hall (5) and Dark Room (6)
- **TUM Mono VO**: narrow indoor corridors, photometrically calibrated

### 6.5 ⚠️ Post-disaster / USAR / rubble — the key negative finding

> **There is essentially no public robot-collected dataset of a real collapsed structure. What exists is synthetic compositing, aerial classification, or contact-gated test-method documentation.**
>
> **State this plainly in the report rather than padding it — and note it cuts in Ariadne's favour: it is part of why the simulator track exists, and why a synthetic-but-physically-grounded pipeline is the honest approach rather than a shortcut.**

**C2A** — [arXiv:2408.04922](https://arxiv.org/abs/2408.04922), [code](https://github.com/Ragib-Amin-Nihal/C2A) · ❌ no license. **10,215 images, >360,000 human instances** — but **synthetic**: human figures (poses: bent, kneeling, sitting, upright, **lying**) composited onto AIDER backgrounds (**Collapsed Building/Rubble 320**). YOLO/COCO + 5 pose classes. **The only labelled "human body in rubble from a UAV" set that exists — but composited, so a detector trained on it has never seen a partially-occluded body under real rubble lighting. Pretraining only; never a capability claim.**

**AIDER** — https://zenodo.org/records/3888300 · ✅ **CC BY 4.0 — the least restrictive record in this whole survey.** ~500 images per class (Fire/Smoke, Flood, **Collapsed Building/Rubble**, Traffic Accident) + >4,000 normal ≈ 6,000+. UNPAIRED scene-level labels. Redistributable — useful for a demo, too small for a perception claim.

**NIST Response Robots** — https://www.nist.gov/el/intelligent-systems-division-73500/response-robots · ✅ **verified: no downloadable dataset is published.** It publishes **standard test methods** (ground, aerial with 5 proficiency levels, aquatic), standardized via **ASTM E54.09**. Apparatus specs, maze layouts and **visual-acuity targets** are contact-gated (`RobotTestMethods@nist.gov`).
> **The right source for a reproducible physical test protocol — e.g. standardized visual-acuity targets under controlled darkness. A recognized NIST/ASTM test method in your evaluation section is a strong credibility signal for the Real-World Impact row.**

**Not located:** RoboCup Rescue arena data, TRADR and ICARUS EU releases, earthquake-damage robot datasets. xBD/xView2 is satellite overhead imagery, not in-structure.

### 6.6 Thermal-visible pairs (stretch — victim detection)

| Dataset | People? | Registration | License |
|---|---|---|---|
| **KAIST Multispectral** | ✅ **103,128 annotations** | **hardware beam-splitter** | ✅ CC BY-NC-SA 4.0 |
| **LLVIP** | ✅ pedestrians, very dark scenes | strict time+space | ✅ NC, form-gated |
| **M3FD** | ✅ People class | calibration + homography | ⚠️ repo GPL-3.0 (code) |
| **MS2** | ⚠️ driving | stereo + **LiDAR depth in thermal plane** | ✅ CC BY-NC-SA 3.0 |
| **MSRS** | ✅ | aligned (125 bad pairs removed) | ❌ unverified |
| **ViViD++** | — | RGB+**thermal**+**event**+depth, synced | ✅ docs CC BY-SA 4.0, code MIT |
| **STheReO** | — | **stereo thermal** + RGB + LiDAR + radar | ⚠️ unverified |

**KAIST** (CVPR 2015) — 95k pairs, 640×480 @20 Hz, day and night. **Beam-splitter co-axial alignment is the gold standard — the only large set where thermal-visible correspondence is exact enough to serve as geometric registration GT.**
**LLVIP** (ICCV 2021) — 15,488 pairs, *"most taken at very dark scenes"*, LWIR 8–14 µm. **The highest-density "humans visible in thermal but nearly invisible in RGB" set — the direct proxy for finding a body in an unlit void.**
**MS2** (CVPR 2023) — 184,000 synchronized pairs, stereo RGB + NIR + thermal + LiDAR. **One of very few sets with depth GT in the *thermal* image plane.**
**ViViD++** — the canonical "does thermal or event beat RGB when illumination changes" comparison, all three recorded simultaneously.
**Unverified:** TNO (403), **FLIR ADAS (URL moved, 404 + NXDOMAIN — do not cite a license or frame count)**, RoadScene (no license), VOT-RGBT.

### 6.7 Event cameras

| Dataset | License | Content |
|---|---|---|
| **DSEC** | ✅ **CC BY-SA 4.0 — commercial use permitted** | **53 sequences**, favourable + challenging illumination; stereo VGA event + stereo global-shutter colour + synced LiDAR + RTK; disparity GT + optical-flow benchmark |
| **MVSEC** | ✅ CC BY-SA 4.0 | Stereo DAVIS 346B; hexacopter indoor/outdoor, handheld, **car day + night**, motorcycle; Vicon/Qualisys GT |
| **ECD** (RPG) | ✅ CC BY-NC-SA 3.0 | DAVIS240C, 25+ sequences, dedicated **HDR** subsets, mocap GT, 2 synthetic Blender scenes |
| DDD20 | ❌ unverified | DAVIS driving incl. night |

> **DSEC is the only major dataset in this entire survey that permits commercial use.**

### 6.8 Stretch: haze / smoke / fire

**Paired smoke GT exists in exactly three places:**
1. **⭐ SmokeBench** (ACM MM 2025) — [arXiv:2509.12701](https://arxiv.org/abs/2509.12701), [code](https://github.com/ncfjd/SmokeBench) · ❌ no license. **9,875 training + 100 test pairs**, real surveillance of **early-stage fire (0–15 min post-ignition)**, precisely aligned. **The only real-world paired smoke dataset that exists.** Its 100-pair test set is small — report per-image variance, not just the mean.
   > ⚠️ **Name collision:** [arXiv:2512.11215](https://arxiv.org/abs/2512.11215) is a *different* "SmokeBench" (MLLM wildfire detection eval). Always cite the arXiv ID.
2. **LSD3K** — [arXiv:2407.13132](https://arxiv.org/abs/2407.13132) · ❌ unverified. 3,000 paired **synthetic non-homogeneous** smoke images (Blender smoke over Cholec80 surgical frames). Large domain gap; pretraining only.
3. **Dense-Haze / NH-HAZE** (NTIRE 2019/2020) · ✅ **explicit grant + citation requirement.** 33 and 55 real pairs. **NH-HAZE's non-homogeneity is the right prior — real smoke is never a uniform transmission map.**

**Also:** **NT-HAZE / NTIRE 2026** ([repo ✅ MIT](https://github.com/NTIRE-HAZE/Nighttime-Image-Dehazing-NTIRE-2026)) — 40 indoor **nighttime** scenes, paired, **haze + darkness + artificial-light glow — exactly what a headlamp does inside smoke** (download link still a placeholder) · **I-HAZE/O-HAZE** (35/45 real pairs, MacBeth chart in every scene) · **RESIDE** (the default benchmark; ⚠️ no license stated, Dropbox/Baidu links with high rot risk; RTTS is the honest UNPAIRED real half) · **HazeRD** (14 images + depth + a physically-grounded MATLAB generator — **the only one letting you sweep haze density as a controlled variable**) · **BeDDE** (208 semi-paired + Visibility/Realness Index + **exBeDDE human scores** — use these to check your NR metric tracks human preference).

**Fire/aerial:** **FLAME 2** ✅ Open Access — **53,541 side-by-side RGB/IR frame pairs**, dual-camera Mavic 2 EA so **inherently aligned**, every pair annotated for fire presence and **smoke coverage**. *The dataset for "how much does thermal recover when RGB is smoke-blinded", with smoke coverage as the independent variable.* · **FLAME 3** ✅ — 622 Fire + 116 No-Fire **quartets**: raw RGB 4000×3000, raw thermal, **FOV-corrected RGB resampled to 640×512, and a Thermal TIFF with per-pixel Celsius** — true per-pixel registration plus absolute temperature, which is what a victim-vs-hot-debris discriminator needs. · **FLAME 1** ✅ (~41.6 GB, 2,003 pixel-wise fire masks) · **PYRONEAR-2025**, **FIgLib** (expired cert; `curl -k` works), **D-Fire** (⚠️ its LICENSE is an *aggregation disclaimer, not a grant*), **Nemo** (✅ Apache-2.0), **Boreal Forest Fire** (*Scientific Data* 2025 — mandated open data record, **highest-value remaining fetch**).

**Turbulence — deprioritized, all unverified:** CLEAR, OTIS, URG-T, BRIAR (government-gated), TSRWGAN. **Honest assessment: turbulence is a long-range, hot-surface phenomenon. For an agent operating inside a structure at short range it is largely irrelevant.**

### 6.9 ⚠️ No-reference metrics — required wherever GT is unpaired

Most real degraded data here (RTTS, ExDark, DarkFace, all SubT smoke/dust footage) has **no clean reference**, so PSNR/SSIM are unavailable.

| Metric | Family | Note |
|---|---|---|
| **FADE** | haze-specific NR | Fog Aware Density Evaluator (IEEE TIP 2015). ⚠️ **rewards contrast increase — gameable by naive contrast stretching.** |
| **NIQE** | NSS, opinion-unaware | No training on human scores. Unstable on heavily restored images. |
| **BRISQUE** | NSS, opinion-aware | SVR trained on human MOS. |
| **CLIP-IQA** | vision-language | Prompt-pair similarity; robust across domains, steerable with custom prompts. |
| **MUSIQ** | transformer | Handles native resolution without cropping — useful for mixed-resolution sets. |
| **Contrast gain / visible-edge ratios** | reference-free restoration | Hautière's **e** (newly visible edges), **r̄** (mean visible-gradient ratio), **Σ** (saturated %). The conventional companion to FADE. |
| **AG / SF / SD / Qabf / SCD / VIF** | fusion-specific | The standard IR-visible fusion suite. |

**Code: [pyiqa / IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)** — implements NIQE, BRISQUE, MUSIQ, CLIP-IQA, MANIQA, HyperIQA, DBCNN, NIMA, TOPIQ and more.
> ⚠️ **License: dual PolyForm Noncommercial 1.0.0 + NTU S-Lab — NONCOMMERCIAL. If this project ever commercializes, the *metric library* is a license problem before the data is.**

> **⚠️ Methodological warning, and it applies directly to Ariadne's evaluation section: NR metrics disagree with each other and with humans on restored images far more than the dehazing literature admits. Report ≥3 from different families (one hand-crafted NSS, one learned, one task-driven) — and always include a task-driven metric: detection AP, or ATE on a SubT-MRS sequence. "Does the downstream task get better" is the only claim that survives scrutiny, and it is exactly what the competition's evaluation rubric rewards.**

### 6.10 Recommended dataset plan

1. **Anchor on SubT-MRS.** Only source combining darkness, dust and smoke underground with thermal + LiDAR + trajectory GT. **Email the authors about the license** — it's the one thing to resolve.
2. **Validation backbone: TUM RGB-D + TUM Mono VO first** (both CC BY 4.0, zero friction), then **Replica** for synthetic-degradation ablation against exact geometry.
3. **Start the gated applications on day one** — ScanNet, ScanNet++, Matterport3D take up to a week and will otherwise block week 3.
4. **Build the photometric/illumination model against TUM Mono VO**, then validate darkness on **M2DGR Dark Room** (mm GT + co-mounted thermal and event cameras).
5. **Do not claim rubble-victim detection performance** from LLVIP/KAIST (standing, outdoors, unoccluded) or C2A (composited). Say what the data supports and no more.
6. **License posture:** DSEC (CC BY-SA 4.0) is the only major set permitting commercial use; nearly everything else is NC, form-gated, or unstated. Note this in the report — it reads as diligence.

---

## 7. Distributed optimization, pose-graph consensus & robust estimation

**Convention used throughout: right/local perturbation** `x ⊞ δ = x · Exp(δ)`. Sophus, manif, GTSAM and Ceres all default to this. **Mixing left and right silently transposes your Jacobians** — the solver converges, to the wrong thing. Pin this in a comment at the top of every Jacobian.

**Chordal vs geodesic, settled once.** For `R_a, R_b ∈ SO(3)` at relative angle θ: `‖R_a − R_b‖_F² = 2·tr(I − R_aᵀR_b) = 4(1 − cos θ) = 8 sin²(θ/2)`. They agree to first order, but **chordal saturates at large θ** — it is implicitly slightly robust, and correspondingly slightly biased. Every chordal-relaxation result inherits this.

### 7.1 ⚠️ Three license findings that contradict common assumptions

All verified by fetching the actual LICENSE file, not README badges.

| Library | Common assumption | **Verified reality** |
|---|---|---|
| **SE-Sync** (`david-m-rosen/SE-Sync`) | BSD/MIT | ⚠️ **LGPL-3.0.** Dynamic linking with relink ability is fine; **static linking into a closed binary triggers the relinking obligation**, and copying source into a permissive project violates it. **You never need the risk** — the math is fully public (§7.4) and permissive equivalents exist: **GTSAM `ShonanAveraging3` (BSD-3)** and **`mit-acl/dpgo` (MIT)**. |
| **g2o** | BSD | ⚠️ **Mixed minefield.** **No top-level LICENSE file.** Core BSD-3 (in `doc/license-bsd.txt`); **`csparse_extension` is LGPL-2.1+**; **`g2o_viewer`, `g2o_incremental`, `slam2d_g2o` are GPL-3+**. Worse: **CHOLMOD's supernodal factorization is GPL and g2o uses it if present — an ordinary `apt`-based build pulls GPL into your link.** **Recommendation: skip g2o entirely.** GTSAM and Ceres are clean BSD-3 and strictly better oracles. |
| **`CogRob/distributed-mapper`** (the DGS reference) | open source | ⚠️ **NO LICENSE FILE — all rights reserved.** Read it to understand DGS; **do not vendor it.** Reimplement from the paper — §7.2 has the equations and it's a ~2-day job. |

**Clean and verified:** GTSAM **BSD-3** (bundles Eigen MPL-2.0, CCOLAMD BSD-3, METIS Apache-2.0) · Ceres **BSD-3** · SymForce **Apache-2.0** · Theseus **MIT** · manif **MIT** · Sophus **MIT** · TEASER++ **MIT** · Kimera-RPGO **BSD-2** · `mit-acl/dpgo` **MIT** · `MurpheyLab/DPGO` **MIT** · GNC-and-ADAPT **BSD-2** · CLIPPER **MIT** · riSAM **MIT** · EGNC-PGO **MIT** · iMESA **MIT** · Swarm-SLAM **MIT**.

### 7.2 Distributed Gauss–Seidel — build this first

Choudhary, Carlone, Nieto, Rogers, Christensen, Dellaert, **IJRR 2017** · [arXiv:1702.03435](https://arxiv.org/abs/1702.03435) · **Difficulty: Low, ~2 days**

**Core idea:** don't solve the nonconvex problem distributedly. Solve Carlone's **two-stage chordal approximation** distributedly, because each stage is a *linear system*, and linear systems have textbook distributed solvers.

**Stage 1 — rotations.** Drop `R_i ∈ SO(3)` entirely; each edge gives the *linear* residual `R_j − R_i·R̃_ij = 0`. Solve unconstrained linear LS, then project each 3×3 block back to SO(3) **with the determinant correction** (§7.5 — this is where DGS implementations go wrong).
**Stage 2 — poses.** Rotations fixed ⇒ translations linear. Then **one Gauss–Newton iteration** on the full SE(3) objective.

**The block updates.** Partition normal equations by robot α: `A_αα y_α + Σ_{β≠α} A_αβ y_β = b_α`.

*JOR (all robots update simultaneously from the previous sweep):*
```
y_α^{k+1} = (1−γ)·y_α^k + γ·A_αα⁻¹ ( b_α − Σ_{β≠α} A_αβ · y_β^k )
```
*SOR / Gauss–Seidel (fixed order, uses the latest value from whoever already went):*
```
y_α^{k+1} = (1−γ)·y_α^k + γ·A_αα⁻¹ ( b_α − Σ_{β∈N⁺(α)} A_αβ·y_β^{k+1} − Σ_{β∈N⁻(α)} A_αβ·y_β^k )
```
**Convergence:** JOR needs `ρ(I − γD⁻¹A) < 1`, i.e. `0 < γ < 2/λ_max(D⁻¹A)`; the paper confirms convergence on the rotation subproblem for `γ ≤ 1`. **SOR for SPD A converges for `0 < γ < 2` (Ostrowski–Reich) — no spectral estimate needed, which is why SOR is preferred.**

**Flagged initialization:** each robot's *first* iteration **ignores inter-robot measurements**, landing it in a self-consistent local frame, then flags itself initialized. Cuts iteration count substantially — and iterations are the bandwidth currency.

⚠️ **Gotchas:** DGS returns *chordal + one GN step*, **not the MLE** — the gap is real at high rotation noise, exactly the texture-poor/dark regime. **Use it as an initializer, not the answer.** Gauss–Seidel needs a **global robot ordering**, which in a partition-prone network is itself a coordination problem (§7.9). And `A_αα` is singular without a gauge anchor (§7.3).

### 7.2a Single-agent initialisation: chordal relaxation vs dead reckoning — measured (D70, 2026-10-19)

Carlone, Tron, Daniilidis, Dellaert, **ICRA 2015**, "Initialization techniques for 3D SLAM: a survey on rotation
estimation and its use in pose graph optimization" · **Table I `[PAPER]`, checked at the source.** Gauss–Newton from
the odometric start fails on sphere-a, torus, cube, cubicle and rim and is fine on sphere and garage; chordal
(Martinec–Pajdla) followed by GN reaches the best observed cost on all of them.

**Measured here** on the anisotropic cost, against GTSAM LM from GTSAM's chordal start (design
`ROBUST_BACKEND_2.5.md` §9):
- Our LM from dead reckoning fails on **torus3D** (29950.006 against 12117.637) and **grid3D** = Carlone's cube
  (253372.10 against 42179.98). GTSAM's LM from the same start stops in the same local minima.
- It is fine on tinyGrid3D, smallGrid3D, sphere2500 and parking-garage. This matches Table I on the four graphs the
  two studies share; their sphere-a is not in our set, and cubicle and rim have indefinite information (§7.4, D74).
- **Chordal + LM** lands within 7e-9 relative of the reference on all six, in 4–15 iterations.
- **The `cv::buildMST` spanning-tree start** (highest-κ edges) rescues both failures too (7 and 9 iterations). Carlone
  reports the spanning tree still failing on sphere-a, cubicle and rim, so it remains a baseline, not the default.
- **Weights:** SE-Sync's isotropic conversion, `κ = 3/(2 tr(I_R⁻¹))` and `τ = 3/tr(I_t⁻¹)` from the diagonal blocks
  (the same conversion serves the certificate, §7.4). Projection to SO(3) with the determinant fix (§7.7).

⚠️ **Two oracle traps found on the way (both GTSAM 4.3, Python):**
- **Unnormalised quaternions.** `Rot3.Quaternion` does not normalise, and the SE-Sync g2o files print about 7
  digits (|q| off by up to 6.8e-7). An oracle that skips normalisation solves with non-orthogonal "rotations", and
  that looked like "different minima" on garage.
- **Soft anchor.** `computeOrientationsChordal` pins its anchor with a fixed unit precision while edges weigh about
  15, so its relaxation drifts by up to 6.5e-3 rad from the hard-anchored one. Scale the edge information by 2⁻⁵⁴
  in the oracle to make the anchor effectively hard.

⚠️ **OpenCV `LevMarqBase` damping floor.** With `clampDiagonal`, OpenCV clamps the damping `λ·diag(H)` itself to an
absolute [1e-6, 1e32] (`levmarq.cpp:192–194`). Ceres clamps `diag(JᵀJ)` before applying λ, so its damping still
vanishes as λ → 0. OpenCV's floor throttles the final Gauss–Newton steps: garage needed 95 iterations and stopped
3.7e-5 above the minimum with it, 15 iterations at the minimum without it. The pose graph runs with the clamp off
(`pgo_v2`).

### 7.3 ⚠️⚠️ Gauge freedom — and the best 20 lines in the codebase

An unanchored SE(3) pose graph has **exactly 6 DoF of gauge freedom**. Consequences:

- `H` is **rank-deficient by exactly 6** per connected component. **If Cholesky *succeeds* on an unanchored graph, you have silently added a prior somewhere — find it, it's corrupting your covariances.**
- Under partition you have `6 × #components` gauge DoF; components drift in their own frames. On rejoin you **must** explicitly align (Umeyama/Procrustes) and re-optimize — never just concatenate.
- **Exactly one robot must own the anchor.** Two anchors over-constrains and biases; zero anchors leaves a null direction that random-walks — the *cost* converges while the *iterates* don't, a genuinely confusing failure mode.

**Write this test on day one.** Right-perturbation convention; the 6-D null space of `H` has basis, per pose i:
```
translation u:   δt_i = R_iᵀ u,                δθ_i = 0
rotation    ω:   δt_i = −R_iᵀ [t_i]× ω,        δθ_i = R_iᵀ ω
```
Assert `‖H·N‖ ≈ 0` to machine precision.

> **This single test catches Jacobian sign errors, left-vs-right perturbation mismatches, and marginalization consistency bugs simultaneously. It is the highest-value 20 lines in the project, and §7.11 reuses it.**

### 7.4 ⭐ The certificate — best depth-per-line item in the project

After analytic translation elimination, PGO reduces to `min_{R ∈ SO(d)^n} tr(Q̃ Rᵀ R)`. Shor relaxation + Burer–Monteiro `Z = YᵀY`, `Y_i ∈ St(d,r)`.

**Construct exactly this:**
```
Λ   = SymBlockDiag_d( Q̃ Yᵀ Y ),     Λ_ii = ½( [Q̃ YᵀY]_ii + [Q̃ YᵀY]_iiᵀ )
S(Y) = Q̃ − Λ
```
**The check:** `Z = YᵀY` is a global minimizer of the SDP ⟺ `S(Y) ⪰ 0`. If additionally `rank(Y) = d`, rounding `Y` to `SO(d)^n` gives the **global** minimizer of the original nonconvex problem — **with a zero duality gap you can print as a number.**

> ⚠️ **THE CERTIFICATE GOTCHA.** `S(Y★)` **always** has a nontrivial null space — the global gauge freedom (§7.3), d-dimensional. So the test is **never** `λ_min(S) > 0`; it is `λ_min(S) ≥ −ε` with ε scaled to `‖Q̃‖`. **Implementations testing strict positivity reject every correct answer.** Conversely, deflate against the known null space before believing a near-zero eigenvalue.

**The connection Laplacian.** The rotation block of `Q̃` is `L(G^ρ)` with `[L]_ii = (Σ_j κ_ij)·I_d` and `[L]_ij = −κ_ij·R̃_ij`. Noiseless, its kernel is exactly d-dimensional, spanned by the true rotations. **All tightness theory is a statement about how far noise can move the spectrum before that low-eigenvalue subspace stops being separated — i.e. about the eigengap λ_{d+1}.** Full `Q̃` adds `T̃ᵀ Ω^½ Π Ω^½ T̃` where Π projects onto the weighted cycle space. **Build Π once via sparse QR of the weighted incidence matrix and cache it** — recomputing per iteration dominates runtime.

**On the noise threshold, honestly:** Rosen's theorem says *there exists* β > 0 below which the relaxation is tight. β is an **existence result** — no closed form, dependent on graph connectivity and weights, not noise alone. **There is no transferable "tight below X degrees" number; anyone quoting one is quoting a single experiment.**
> **This is exactly why you implement the certificate: it converts an unknowable a-priori threshold into a per-instance, computable a-posteriori fact.** That inversion *is* the practical value of certifiable estimation, and it's the argument to make to a judge. **Run your own sweep** — synthetic graphs at your topology, sweep σ_R, plot *fraction certified* vs σ_R. That curve is your operating envelope and a strong figure.

**The Riemannian Staircase:**
```
r ← d
loop:
  Y_r ← local_solve(rank-r BM problem)
  S ← Q̃ − SymBlockDiag_d(Q̃ Yᵀ Y)
  (λ_min, v) ← min_eigenpair(S)            # deflate known gauge null space
  if λ_min ≥ −ε: return round_to_SO(d)(Y_r)   # CERTIFIED GLOBALLY OPTIMAL
  Y_{r+1} ← [Y_r ; 0]                       # append a ZERO ROW
  Ẏ       ← [0   ; vᵀ]                      # descent direction in the NEW row
  Y_{r+1} ← line_search along Retr(Y_{r+1}, t·Ẏ)
  r ← r + 1
```
Curvature along `Ẏ` is `2·vᵀSv = 2λ_min < 0`, so the retracted line search strictly decreases f.

> ⚠️ **STAIRCASE GOTCHAS.** (a) The descent direction **must** go in the newly created row — anywhere else leaves you at the same critical point and the solver returns immediately, an infinite loop that looks like non-convergence. (b) `[Y_r; 0]` is *exactly* a critical point of the lifted problem — **line-search before calling the local solver** or it terminates at step 0. (c) Termination at r = d or d+1 in practice; **if you reach r = 8 you have a bug, not a hard problem.**

**⭐ Distributed certification (DC²-PGO Alg. 6).** `S` has the sparsity of the pose graph, so `Sv` is a distributed operation. Everything reduces to power iteration:
```
1. (λ_dom, v_dom) ← power_iteration(S)      # dominant = MAX-MAGNITUDE
2. if λ_dom < 0: return (λ_dom, v_dom)      # all eigenvalues negative
3. C ← λ_dom·I − S                          # spectral shift ⇒ C ⪰ 0
4. (θ, v) ← accelerated_power_iteration(C)
5. return (λ_dom − θ, v)                    # θ = λ_dom − λ_min(S)
```
The shift is the trick: the *minimum* eigenvalue of S — unreachable by plain power iteration — becomes the *maximum* eigenvalue of C. Acceleration: `v_{k+1} ∝ C v_k − γ v_{k−1}` with `γ ∈ (0, λ₂(C)²/4)`; **overestimating γ destroys convergence — estimate conservatively.**

**Stop on the Ritz residual, not iteration count:** if `‖Sv − λv‖ ≤ ε` with `‖v‖ = 1`, then `λ_min(S) ≥ λ − ε` by backward stability. **A rigorous certificate from finitely many inexact distributed iterations — exactly what to show a judge.** Communication: one sparse mat-vec per iteration = separator entries of v. **The certificate is nearly free in bandwidth.** ~150 lines given you have the distributed mat-vec.

### 7.5 ⚠️ The Stiefel-vs-SO(d) determinant confusion

In RBCD's single-pose block update, eliminate the lifted translation analytically, leaving `max_{Y_i ∈ St(d,r)} ⟨Y_i, B⟩`, solved by projection: **`P_St(B) = UVᵀ` from the thin SVD `B = UΣVᵀ`.**

> ⚠️ **THE GOTCHA EVERYONE HITS. This Stiefel projection has NO determinant correction.** `St(d,r)` for `r > d` is **orientation-free** — there is no proper component to enforce. **The determinant fix belongs ONLY to the final rounding step**, projecting the rank-d solution back to `SO(d)`. Applying a `det` fix inside the RBCD inner loop is a real bug that silently degrades the solution; omitting it in rounding gives you reflections. **Two different operations, two different rules, one shared SVD — write them as two clearly-named functions.**

### 7.6 The distributed solvers

**⭐ MM-PGO / AMM-PGO — best effort/robustness ratio.** Fan & Murphey, [arXiv:2108.00083](https://arxiv.org/abs/2108.00083), [code MIT](https://github.com/MurpheyLab/DPGO) · **Difficulty: Med**
Build a surrogate `G(X | X⁽ᵏ⁾) ≥ F(X)` touching F at `X⁽ᵏ⁾` and **decoupling across nodes**: replace the coupling Hessian by a block-diagonal dominating `M ⪰ Q̃` (generalized-Jacobi / row-sum bounds). Minimizing G separates per node, each a Stiefel quadratic with the same closed-form SVD projection.
**The whole convergence argument in one line:** `F(X⁽ᵏ⁺¹⁾) ≤ G(X⁽ᵏ⁺¹⁾|X⁽ᵏ⁾) ≤ G(X⁽ᵏ⁾|X⁽ᵏ⁾) = F(X⁽ᵏ⁾)`.
> **Monotone descent by construction, with ZERO stepsize tuning — the majorizer *is* the stepsize, derived from data rather than guessed.** Huge practical win when you can't afford a tuning campaign. And "assert cost never increases" is a trivially writable unit test. **Recommendation: MM-PGO as the workhorse, RBCD as the certifiable path.**

**⭐ RBCD / DC²-PGO.** Tian, Khosoussi, Rosen, How, **T-RO 2021**, [arXiv:1911.03721](https://arxiv.org/abs/1911.03721), [code MIT](https://github.com/mit-acl/dpgo) · **Difficulty: High**
Makes every piece of SE-Sync distributed. Block = robot ⇒ RTR for *sufficient decrease* (the generalization over prior BCM methods, which needed exact closed-form block minimization); block = single pose ⇒ closed form via the Stiefel projection above. **Block selection:** uniform, **importance** (`p_b ∝ ‖grad_b f‖²`, cheap and much better), or greedy Gauss–Southwell. **RBCD++** adds Nesterov + adaptive restart. **Parallel RBCD:** color the robot-level aggregation graph — same-color robots share no measurements, so their updates **commute exactly**. Free parallelism on a ≤10-node graph.

**⭐ ASAPP — the delay-tolerant one.** Tian, Koppel, Bedi, How, [arXiv:2003.03281](https://arxiv.org/abs/2003.03281), same MIT repo · **Difficulty: Med**
Each robot runs Riemannian gradient steps **without waiting**, using stale neighbor values: `X_α^{k+1} = Retr(−η · grad_α f(X_α^k, X_N^{k−d^k}))`, `0 ≤ d^k ≤ τ`. Under bounded delay there **exists a convergent stepsize η(τ, sparsity) > 0**, and **at τ = 0 it reduces exactly to the synchronous constant** — you pay nothing for the asynchrony machinery on a healthy network.
⚠️ **Bounded delay is an assumption, not a property of your radio — enforce it:** if a neighbor's value is older than τ steps, **drop that edge from your local gradient** rather than continuing with unbounded staleness.

**2023–2026 successors:** **IRBCD + METIS multi-level partitioning** ([2401.01657](https://arxiv.org/abs/2401.01657), RA-L 2024 — balances subgraph sizes and **shrinks the inter-robot cut**, the #1 bandwidth lever) · **DCORA** ([2503.03192](https://arxiv.org/abs/2503.03192), **ICRA 2025** — first distributed certifiably-correct **range-aided** SLAM; **high value if you add UWB**, since range is low-bandwidth with known data association) · **Certifiable Factor Graph Optimization** ([2603.01267](https://arxiv.org/abs/2603.01267), 2026 — Shor + Burer–Monteiro **inherit factor-graph structure**, so the Staircase runs **inside GTSAM/g2o with ordinary tooling**; the cheapest route to certification for a from-scratch factor-graph codebase) · **Robust M-Estimators + Certifiable FGO** ([2603.20932](https://arxiv.org/abs/2603.20932), **ICRA 2026** — each IRLS inner WLS solved *certifiably*; **best-in-class composition as of 2026**) · **Async distributed trajectory estimation** ([2607.01106](https://arxiv.org/abs/2607.01106), 2026 — a MAP approximation **eliminating up to 96.9% of inter-agent communication** with negligible error, exponential convergence) · **Continuous Riemannian Dynamics** ([2605.11210](https://arxiv.org/abs/2605.11210) — poses as damped particles; modeling **velocity** enables **principled neighbor prediction under delay**, a genuinely novel answer to the lossy-link problem).

### 7.7 SVD methods — the determinant details people get wrong

**Procrustes / Kabsch / Wahba.** `min_{R∈SO(d)} Σ w_i‖q_i − R p_i‖²`. Only the cross term matters: `Σ w_i q_iᵀ R p_i = tr(Rᵀ H)` with **`H = Σ w_i q_i p_iᵀ`**. Thin SVD `H = UΣVᵀ`:
```
R★ = U · diag(1, …, 1, det(U)det(V)) · Vᵀ
optimal value = σ₁ + … + σ_{d−1} + det(UV ᵀ)·σ_d
```
> ⚠️ **FOUR DISTINCT WAYS THIS GOES WRONG:**
> 1. **Wrong transpose.** `H = Σ w_i q_i p_iᵀ` gives R mapping p→q; `Σ w_i p_i q_iᵀ` gives Rᵀ. Both "work" on symmetric test data and both are wrong half the time in production. **Write the derivation in a comment above the function.**
> 2. **Using `det(H)` instead of `det(U)det(V)`.** They agree when H is full rank. **When H is rank-deficient — coplanar points, ubiquitous in rubble: flat slabs and walls — `det(H) = 0` and the sign test is undefined, while `det(U)det(V) = ±1` stays well defined. Using `det(H)` silently returns a reflection.**
> 3. **Correcting the wrong singular vector.** The −1 must multiply the column of the **smallest** singular value σ_d. Applying it to σ₁ maximizes the damage.
> 4. **Not detecting non-uniqueness.** Uniqueness needs `rank(H) ≥ d−1`. If `σ_{d−1} = σ_d`, the minimizer is a continuum (collinear points, symmetric configuration). **Detect and report** — silently returning one arbitrary element is how a degenerate corridor scan poisons a map.

**Horn's quaternion method** (JOSA A 4(4):629–642, 1987). Build `M = Σ w_i p_i q_iᵀ`, then the symmetric 4×4 `N`; the optimal unit quaternion is the eigenvector of **largest** eigenvalue, and `λ_max = max tr(RᵀH)`.
> **Why prefer it for d=3:** quaternions parameterize SO(3) **only** — no improper component exists in the search space, so **no determinant correction is possible or needed.** Structurally immune to gotchas 2 and 3. **Implementing both Kabsch and Horn and cross-checking them is a strong, cheap correctness argument for a from-scratch codebase.**
> ⚠️ **Quaternion gotchas:** **double cover** — `q` and `−q` are the same rotation and the eigenvector sign is arbitrary; **canonicalize (force w ≥ 0) before storing, averaging, interpolating or differencing**, or averaging sign-flipped quaternions cancels catastrophically toward zero. **Hamilton vs JPL** — GTSAM/Sophus/Eigen/ROS use Hamilton; much of the aerospace/MSCKF lineage uses JPL (reversed multiplication order, opposite handedness). **Mixing them transposes rotations silently.** **Storage order** — `Eigen::Quaterniond` constructs as `(w,x,y,z)` but `.coeffs()` returns `(x,y,z,w)`; serializing coeffs on one robot and deserializing with the constructor on another is a classic multi-robot interop bug. **Pin one wire format in the protocol spec and assert round-trip in a test.**

**Umeyama similarity alignment** (T-PAMI 13(4):376–380, 1991). `R = USVᵀ`, `c = tr(DS)/σ_x²`, `t = μ_y − cRμ_x`, with
```
S = I                        if det(Σ_xy) > 0
S = diag(1,…,1,−1)           if det(Σ_xy) < 0
if rank(Σ_xy) = m−1:  S = I when det(U)det(V) = +1, else diag(1,…,1,−1)
if rank(Σ_xy) < m−1:  NO UNIQUE SOLUTION — detect and error
```
> ⚠️ **The rank-deficient branch is a separate case in Umeyama's theorem and is the one dropped in reimplementations. It fires exactly when points are coplanar — the *common* case in a collapsed building, not the pathological one.** The naive `det(Σ)>0 / else` version returns a reflection there.
>
> ⚠️⚠️ **THE EVALUATION GOTCHA — read before reporting any ATE number.** Umeyama with `with_scaling=true` is Sim(3) alignment, standard for *monocular* evaluation. **Ariadne is metric.** Evaluating ATE with scaling enabled **silently absorbs your own scale error and reports a flattering number a knowledgeable judge will catch.** **Use SE(3)-only alignment (c ≡ 1) and say so explicitly in the writeup.** Being explicit is itself a depth signal.

**Spectral & chordal initialization.** Spectral: take the d eigenvectors of the d smallest eigenvalues of the connection Laplacian, project each block to SO(d). No initialization needed, one sparse eigensolve, error bounds exist (Bandeira–Singer–Spielman). **Use it as the staircase warm start — dramatically better than odometry chaining, nearly free.**

**Shonan Rotation Averaging** (Dellaert, Rosen, Wu, Mahony, Carlone, **ECCV 2020**, [arXiv:2008.02737](https://arxiv.org/abs/2008.02737)) — same staircase but climbing `SO(p)^n`, so the problem *stays* a rotation-averaging problem at every rank and you reuse an existing solver unchanged. **In GTSAM as `ShonanAveraging3` (BSD) — use it as your oracle for certified-optimal rotations.** Same §7.4(b) trap: `diag(R,1)` is already a critical point, so perturb before solving.

**⭐ Robust rotation averaging — a real novelty angle.** L1-IRLS (Chatterjee & Govindu) · HARA ([2111.08831](https://arxiv.org/abs/2111.08831), CVPR 2022 — hierarchical, ordered by **triplet support**) · **Anisotropic Coordinate Descent** ([2506.01940](https://arxiv.org/abs/2506.01940), 2025) · **Certifiably Optimal Anisotropic RA** ([2503.07353](https://arxiv.org/abs/2503.07353), 2025).
> **Relative-pose covariances in a texture-poor corridor are strongly anisotropic — the along-corridor direction is nearly unobservable while cross-corridor is tightly constrained. Feeding those to an isotropic κ_ij throws away exactly the information that distinguishes a good estimate from a bad one, and the 2025 result shows isotropic *solvers* actively fail there. Propagating full anisotropic covariances into rotation-averaging weights is a defensible, novel-ish contribution directly motivated by the collapsed-building scenario.**

### 7.8 Robust estimation — the layered architecture

**TEASER++** (Yang, Shi, Carlone, **T-RO 2020**, [arXiv:2001.07715](https://arxiv.org/abs/2001.07715), [code MIT](https://github.com/MIT-SPARK/TEASER-plusplus)). The decoupling cascade: **TIMs** `a_ij = a_j − a_i` **eliminate translation exactly, not approximately** → **TRIMs** `s_ij = ‖b_ij‖/‖a_ij‖` depend only on scale → **scale by adaptive voting** (1-D TLS solved *exactly* in O(N log N) by interval stabbing — polynomial-time exact solution of a combinatorial problem) → rotation via GNC-TLS, certified by Douglas–Rachford splitting → translation as three scalar TLS problems. **Max-clique pruning:** inliers are mutually consistent, hence form a clique.
> ⚠️ **Max clique is NP-hard** (cap correspondences, use max-**core** approximation). ⚠️ **Outliers can form cliques too — and rubble is the worst case: broken slabs look like broken slabs.** A large clique is a very strong prior, **not a guarantee**. ⚠️ **β_i must come from real sensor covariance** — this is the one parameter that matters, and ADAPT can estimate it. ⚠️ **The certificate covers the rotation subproblem only** — never report "certified registration" for the full pipeline.

**⭐ The inapproximability result — worth quoting in the writeup.** Antonante, Tzoumas, Yang, Carlone, [arXiv:2007.15109](https://arxiv.org/abs/2007.15109), T-RO 2021: **outlier-robust estimation is inapproximable** — in the worst case it is impossible to even *approximately* find the outlier set, even with slower-than-polynomial algorithms.
> **This is the principled justification for a layered architecture. There is no single correct algorithm; you buy robustness by stacking independent cheap filters, each individually defeatable but jointly very hard to fool. Saying this explicitly, with the citation, is a stronger depth signal than adding another solver.**

**ADAPT** (same repo, BSD-2) — combinatorial adaptive trimming: `ε_k = max_{i∈I_k} r_i(x_k)`, `I_{k+1} = {i : r_i(x_k) ≤ (1−α)ε_k}`. The real contribution is **"minimally tuned"**: both ADAPT and GNC extended to **unknown or time-varying inlier-noise statistics**.
> **Composition recipe: GNC wins when noise statistics are known; ADAPT wins when they aren't. → Run ADAPT to *estimate* c̄/β_i online, then hand that threshold to GNC-TLS and TEASER++.** In a collapsed building noise statistics genuinely drift (dust, dark, varying surface geometry), so a fixed hand-tuned threshold isn't defensible.

**IRLS convergence — the part usually left out.** IRLS is an **MM algorithm** whenever `ρ(√s)` is **concave in s** ("sqrt-concavity") — Huber, Cauchy, Geman–McClure, Welsch, Tukey all satisfy it, so the cost is **monotonically non-increasing**. That's the actual theorem, not "IRLS usually works." Rate is linear; for **redescending** kernels `ψ' < 0` in the tail, the surrogate is loose and fixed points include bad local minima — **precisely the gap GNC/homotopy fills, and precisely what [2603.20932](https://arxiv.org/abs/2603.20932) closes by solving each inner WLS certifiably.**
> ⚠️ **Two real IRLS bugs.** **Zero-residual blowup:** for L1, `w = 1/|r| → ∞`. Guard with `w = 1/max(|r|,δ)` or pseudo-Huber `1/√(r²+δ²)` — **in the code, not the comments.** **Weight collapse:** a redescending kernel can drive *all* weights toward zero, the weighted information matrix loses rank, the gauge becomes unconstrained, and the solve returns garbage **that may not even throw.** Guard: after each reweighting **assert `rank(H_w)` equals the expected `6n − 6`** — three lines, reusing the §7.3 null-space machinery, turning silent catastrophe into a clean error.

**GNC successors:** **riSAM** ([2209.14359](https://arxiv.org/abs/2209.14359), ICRA 2023, MIT) — **GNC made incremental**, the direct answer to "GNC is batch, Ariadne is online" · **EGNC-PGO** ([2310.06765](https://arxiv.org/abs/2310.06765), MIT) — computes the **boundary points beyond which convexity is lost**, replacing the heuristic μ-schedule; faster *and* more accurate · **AGNC-PGO** ([2308.11444](https://arxiv.org/abs/2308.11444)) — adaptive B-spline kernel shape.

**⭐ The recommended composition:**
```
per candidate loop closure   →  TEASER++ (MIT)          reject bad correspondences
across loop closures         →  PCM                     reject inconsistent closures
threshold estimation         →  ADAPT                   estimate c̄/β_i online, no hand-tuning
backend                      →  riSAM-style incremental GNC-TLS
inner WLS                    →  certifiable solve (§7.4)  λ_min(S) printed per solve
```
Each layer independently justifiable; each has a verified permissive reference implementation.

**Additions, 2026-10-16 (Phase 2 literature pass, `docs/design/POSE_GRAPH_2.md` §6):**
- **TACO** (arXiv 2606.29851, 2026): an online test-and-check robust PGO. Incremental Probabilistic Consensus tests each
  new loop closure, and periodic Switchable-Constraint "sanitization" removes inconsistent ones it admitted. It
  reports robustness comparable to offline methods at online cost. An ablation candidate next to GNC/PCM.
- **EGNC schedule** (Int. J. Control Autom. Syst., 2025; the published form of arXiv 2310.06765, above): replaces the
  heuristic μ ×1.4 with convexity boundary points. An optional schedule against Yang's.
- **COSMO-Bench** (arXiv 2508.16731, 2025): 24 real collaborative-SLAM pose-graph datasets (LiDAR front end). Real
  multi-robot graphs with real outliers, for Phase 3 and the 2.5 outlier ramp. Licence to be checked before use.
- **SE-Sync benchmark graphs** (sphere, torus, garage, cubicle, rim): the standard PGO test set, with published
  certified optima. The external check on our certificate's sign convention.

### 7.8a Loop-closure degeneracy gate — sources and the choice made (D71, 2026-10-19)

- **Zhang, Kaess, Singh, ICRA 2016**, "On degeneracy of optimization-based state estimation problems" `[PAPER]`,
  checked at the source. The degeneracy factor is λ_min of AᵀA, an **absolute** eigenvalue. The threshold is set
  from one sample dataset at the midpoint of the margin between degenerate and well-conditioned groups. Degenerate
  directions are handled by solution remapping, which Ariadne does not adopt: the gate declines the loop instead.
- **Tuna et al., X-ICP, T-RO vol. 40 (2024)** `[PAPER]`: localizability analysed on the rotational and translational
  sub-blocks of the Hessian separately.
- **Zhang et al., arXiv 2608.15532 (Aug 2026, preprint, not peer reviewed)**: eigenvalue thresholds on a
  mixed-unit information matrix depend on the body frame, because H transforms by congruence `Adᵀ H Ad`. Cited only
  for "do not threshold mixed-unit eigenvalues"; its generalized-eigenvalue method is not adopted.
- **Vakhitov, Ferraz, Agudo, Moreno-Noguer, CVPR 2021** `[PAPER]`: in PnP the 3D points are uncertain too. Their
  noise is propagated into pixel space, here as B's stereo noise in (u, v, disparity).
- **Choice.** Each PnP loop gets a covariance in the pose-graph residual tangent, with stereo depth noise propagated.
  - **Conditioning gate:** the worst-direction 1σ of the translation marginal [m] and the rotation marginal [rad],
    each against an absolute bound. Marginals keep the pan-versus-lateral-translation coupling.
  - **Parallax gate:** the number of points with a stereo triangulation angle ≥ θ_min. It is separate because the
    linearised covariance is not honest at far range: mean NEES **54.7** at 25–40 m (about 2.8 px of disparity
    against 0.65 px of noise), where the covariance claims 2.7 cm.
  - **Collinearity gate:** the 3D inliers lie on a line (λ_mid/λ_max of their covariance). Stereo noise scattering
    the points off the line fabricates rotation information about it: on a synthetic rail, conditioning reported
    only 8 mrad. Collinear *pixels* are not the criterion. A planar fan through the camera centre (pixels on one
    row) is well conditioned with known depth (6.4 mm, 1.35 mrad), and a pixel-spread gate refused it.
  - **Reported only:** pixel spread, planarity and a corridor score. Planar PnP with known 3D points is not
    degenerate.
- **Measured** (synthetic, `test_degeneracy.cpp`): mean NEES over 500 trials is **5.95** against the χ²₆ mean of 6
  at 1.5–12 m, the range the gate accepts.

### 7.8b PCM — the threshold is a per-loop decision, not a per-pair one (D72, 2026-10-19)

- **Mangelson, Dominic, Eustice, Vasudevan, ICRA 2018** `[PAPER]`: pairwise cycle consistency plus a maximum
  clique, per robot pair. **Pattabiraman et al., arXiv 1209.5818**: the CLIQUEHEU greedy heuristic used here. Which
  heuristic Kimera-RPGO's reference code uses was not checked [UNVERIFIED].
- **The finding.** A maximum clique keeps a true loop only if every one of its n − 1 pairs passes, so a per-pair
  test at 5 % loses true loops in proportion to group size. Measured on 30-loop groups with χ²₆(0.95): recall
  0.84–0.86 mean and 0.60–0.70 worst, at every outlier rate. A dropped true loop is unrecoverable downstream.
- **The choice.** Bonferroni per loop: α_pair = α_loop/(n − 1), with the threshold from the closed-form χ²₆
  survival e^(−x/2)(1 + x/2 + x²/8). α_loop = 0.001 is derived from "≤ 1 true loop lost in 30".
  - **Random outliers (10–40 %):** recall 1.000 and precision 1.000 on all 20 seeds.
  - **The cost:** near-miss outliers at 0.2–0.5 m pass more often (precision 0.75 at 0.2 m / 1°, 0.94 at
    0.5 m / 5°). They are left to GNC.
- **Shared odometry.** For intra-agent pairs the two odometry legs can share edges, so their covariances must not
  simply be added. They are split at the four endpoints, with summed coefficients. Checked: the mean statistic is
  5.75–6.15 for disjoint, nested, crossing, backward and shared-end configurations.

### 7.8c GNC-TLS — what it does and does not catch, measured (D73 part 1, 2026-10-20)

- **Yang, Antonante, Tzoumas, Carlone, RA-L 2020** `[PAPER]`, Prop. 4 and Remark 5. Conventions were cross-checked
  against GTSAM 4.3.0's `GncOptimizer.h`, which puts barcSq = ½χ² on its ½-scaled error. The closed-form weight is
  verified as the argmin of the Black–Rangarajan dual.
- **Random outliers** (the paper's protocol): on grid and sphere topologies, 20 seeds, precision 1.000 and recall
  ≥ 0.967 at 10–40 %, and 10/10 successes up to 90 % on the grid. Random SE(3) outliers are easy; this ramp does not
  locate a breakpoint.
- **Near misses** are the realistic case. At 0.5 m / 5° GNC rejects 200/200 on its own. At 0.2 m / 1° (about 4σ),
  71 % of them pass PCM → GNC, costing 11 % ATE.
- ⚠️ **A lone link is unverifiable.** One loop between two otherwise unconnected agents fits exactly, so no robust
  cost can reject it. Only redundancy (PCM over several loops) can.
- ⚠️ **Disconnection.** When rejections split the graph, each component needs its own anchor. With pgo_v2's clamp
  off, the singular system would otherwise fail the pivot guard.
- **Cost:** about 27 outer iterations, each a warm-started LM solve. Measured 1.6 s at 1,000 nodes and 4.75 s at
  3,000 per event, with GNC alone. Behind PCM: about 0.1 s and 0.2 s per event.
- ⚠️ **Reusing weights across events (D73 part 2a)** needs more than "the old split is still a TLS fixed point".
  A wrong new loop can bend the odometry and keep its own residual small. An innovation test at the full χ² bound
  still keeps early near misses that drift makes statistically consistent, which TLS slightly favours and cold GNC
  rejects. Measured safe: the innovation within ½χ²₆ₙ(0.99), Remark 5's region, so only unambiguous loops take the
  fast path (72–83 % of events).

### 7.8d ⚠️ Loop covariances on real data — measured overconfidence and the fix (D73 part 2b, 2026-10-21)

- **Measured.** The PnP-Hessian loop covariance, even with stereo depth noise propagated (D71: NEES 5.95 on
  synthetic data), gives median NEES ≈ 1e4 on EuRoC V1_01, against 6. The worst-direction σ is about right in size
  (fitted k_t ≈ 1); the narrow directions are wildly overconfident.
- **Why.** 100+ inliers are treated as independent, so σ ∝ 1/√N, but the dominant errors are shared: SGBM depth
  bias at the depth edges where ORB corners sit, calibration and rectification, and the ground truth's own
  timing and extrinsics. RANSAC keeps the best-fitting points, so the residual-based inflation never fires.
- **Literature.** COVINS-G (arXiv 2301.07147) reports analytic loop covariances NEES "several magnitudes" too large;
  its sampling fix still gives NEES ≈ 30. Censi's Hessian ICP covariance is "widely considered overoptimistic"
  (Brossard et al., RA-L 2020, arXiv 1909.05722). Common SOTA practice (ORB-SLAM [UNVERIFIED], Kimera's configured
  noise, NVIDIA's guidance) uses fixed or tuned loop noise rather than per-loop covariance.
- **Fix adopted.** var = k σ²_PnP + floor², isotropic per block, fitted by maximum likelihood on ground-truth-good
  loops. Held out: mean NEES 5.24; the tail is still heavy (4.4 % above χ²₆(0.99)), and GNC absorbs it. With it, PCM
  and GNC raise precision stage by stage (0.53 → 0.80 → 0.88 → 0.97 on V1_02).

### 7.9 ⚠️⚠️ ADMM is not loss-tolerant — an architectural constraint

**Convergence conditions, stated precisely:** convex `f_i` — Shi et al., **IEEE T-SP 2014**, linear convergence under strong convexity + Lipschitz gradient. **Nonconvex `f_i` (this is PGO)** — Wang, Yin, Zeng, **J. Sci. Comput. 78, 2019**: convergence to a *stationary point* requires ρ **sufficiently large** (roughly ρ > 2L), with **no global-optimality guarantee.** ρ tuning: **residual balancing** is what actually works (`‖r_primal‖ > 10‖r_dual‖ ⇒ ρ ← 2ρ`, reverse ⇒ `ρ ← ρ/2`).

> ⚠️⚠️ **THE DECISIVE POINT. ADMM's dual variables `y_ij` are STATEFUL** — `y_ij^{k+1}` is defined recursively from `y_ij^k` and the *current* `x_i, x_j`. **If a packet carrying `x_j^{k+1}` is lost, robot i updates `y_ij` against a stale `x_j`. The endpoints' duals desynchronize, and the constraint `x_i = x_j` is driven toward the WRONG target. The algorithm does not slow down — it converges to a wrong point, silently, with a small primal residual.** That is the worst possible failure mode.
>
> **Contrast: RBCD, MM-PGO and ASAPP are *stateless in the neighbor*** — they need only the neighbor's current value. A dropped packet means a stale value, which is exactly the bounded-delay model ASAPP **proves** convergence under.
>
> **→ Choose MM-PGO / RBCD / ASAPP over consensus-ADMM, and state this reasoning explicitly in the writeup. It is a rigorous, scenario-driven architectural decision, and defending it is worth more to a technical-depth judge than implementing ADMM would be.** If you must use ADMM, you need ack+retransmit for the x-messages — reintroducing the bandwidth cost you were avoiding. (The clean manifold formulation, if needed: [2503.06912](https://arxiv.org/abs/2503.06912), which splits the orthogonality constraint into an **analytic Procrustes** subproblem so SO(3) is never relaxed.)

**⭐ Push-sum / ratio consensus — the right primitive here.** Kempe, Dobra, Gehrke, **FOCS 2003**. Keep `(s_i, w_i)`, split into equal shares among out-neighbors and self, estimate `ŝ_i/w_i → mean`. Needs only a **column-stochastic** matrix, so it works on **directed, time-varying** graphs — **you do not need bidirectional links**, exactly the asymmetric-radio situation in a rubble pile.
> ✅ **THE SINGLE MOST USEFUL PRACTICAL TRICK: running-sum push-sum.** Send **cumulative totals** rather than per-round increments. A receiver that missed messages recovers the entire missed mass by **differencing against its last cumulative value**. **Packet loss then costs latency, not correctness** — mass is never destroyed, only delayed. Converts a lossy link from a correctness hazard into a performance parameter, for ~30 extra lines.

(Plain **average consensus** `x^{k+1} = (I − εL)x^k` converges iff connected, at a rate set by the **Fiedler value λ₂**; use **Metropolis–Hastings weights** — doubly stochastic with **no global parameter to estimate**. ⚠️ But **packet loss breaks mass conservation**: you still converge to *a* consensus, just not the average — a correctness bug, hence push-sum. **Randomized gossip** conserves mass only if exchanges are **atomic**, requiring two-phase commit per exchange, which is why push-sum usually wins.)

### 7.10 ⭐ Raft/Paxos — where they belong, and emphatically don't

**Five independent reasons they're the WRONG tool for pose/map consensus:**
1. **Wrong domain.** Raft agrees on *discrete values proposed by someone*. Pose consensus is agreement on a **continuous quantity that is the solution of an optimization over a manifold** — nobody proposes it. **You cannot elect a rotation.**
2. **Wrong objective.** Raft's criterion is "everyone holds the same log." Ariadne needs "everyone converges to the near-optimal estimate given all measurements." **Identical-but-wrong is a Raft success and an Ariadne failure.** Raft's agreement is *exact equality*; geometric estimation's is *metric closeness to the optimum*.
3. **Wrong partition behaviour.** Raft's minority side **cannot commit — it halts.** A robot that loses contact deep in a collapsed building must keep mapping. Distributed PGO **degrades gracefully**; Raft doesn't degrade, it stops. **The single most disqualifying property for search-and-rescue.**
4. **Wrong cost model.** Majority quorum per entry: O(n) messages, 2 RTTs. Replicating the measurement log to a majority is **exactly the bandwidth blow-up the project exists to avoid.**
5. **Wrong fault emphasis.** Raft handles crash faults; Ariadne's dominant fault is a **bad loop closure** — a Byzantine-ish *data* fault consensus protocols don't address and GNC/PCM/TEASER++ do.

**Where Raft-like protocols ARE legitimately useful:**
1. **Broker election.** Given the elected-broker finding (§1.4), you need exactly one broker and **split-brain is genuinely harmful.** Raft's **term numbers give you fencing for free** — a stale broker's messages carry an old term and are rejected.
2. **Configuration/membership agreement** — team roster, **wire format and frame conventions (§7.7: Hamilton vs JPL, coefficient order!)**, the agreed inlier threshold, **the anchor-pose owner (§7.3)**, and **the Gauss–Seidel robot ordering (§7.2)**. Discrete, rarely-changing values where exact agreement *is* the criterion.
3. **Task-assignment commitment** — "Robot 3 owns frontier F." Exactly-once assignment; two robots on one frontier wastes the mission.
4. **Loop-closure acceptance ledger** — recording GNC/PCM decisions in an append-only log prevents the pathological state where robots permanently disagree about whether edge e is an inlier and therefore **optimize different problems forever.** **Geometry decides; consensus records.**

> **Scoping judgment worth making explicitly: do not implement Raft from scratch in 2 months.** The corner cases (log matching, snapshot install, membership change) are where correctness dies and none of them are what you're judged on. **For ≤10 robots with a designated broker, a bully or ring election with monotonic term numbers plus lease expiry (~200 lines) suffices — render its only failure mode (transient dual brokers) harmless by making the geometric layer idempotent and term-fenced (tag every optimization round with the broker's term; discard stale terms). Frame this explicitly as a deliberate, justified weakening with a stated failure-mode argument — that reads as engineering depth. "We implemented Raft" in a SLAM competition reads as scope confusion.**

### 7.11 ⚠️ Marginalization — the two subtle killers

**Schur complement:** `Λ★ = Λ_kk − Λ_km Λ_mm⁻¹ Λ_mk`, `b★ = b_k − Λ_km Λ_mm⁻¹ b_m`.
> ⚠️ **Marginalize vs condition — people invert these.** In **information** form: marginalizing = Schur complement; conditioning = simply *dropping* the off-diagonal blocks. In **covariance** form it's the reverse. Writing the covariance-form rule while working in information form is a classic quiet error.

Store the result as a linear prior anchored at `x_lin` **at the time of marginalization** — **that anchor must be stored alongside the factor and never updated.**

**⚠️⚠️ The consistency bug (FEJ / OC).** If you relinearize marginalized states at new estimates, the prior's Jacobians and the new factors' Jacobians sit at *different points*. The consequence isn't just inaccuracy: **the null space of the total information matrix shrinks. Genuinely unobservable directions become spuriously observable, the estimator becomes overconfident, and drift in those directions is systematically under-estimated.** This is EKF-SLAM inconsistency (Huang, Mourikis, Roumeliotis) transplanted into a smoother — and **for a pose graph the relevant null space is exactly the 6-DoF gauge of §7.3.**
- **FEJ:** evaluate **all Jacobians** involving a state at its **first-ever estimate**, residuals at the current estimate. Preserves the null space exactly. Used by OpenVINS, VINS-Mono, OKVIS. **The pragmatic choice.**
- **OC:** project instead — `Ĥ = H − (HN)(NᵀN)⁻¹Nᵀ`. Slightly better, more code, requires N explicitly — which §7.3 already gave you.
> **The test that catches all of it: assert `‖H·N‖ ≈ 0` BEFORE AND AFTER every marginalization. If the norm jumps after marginalization, you have the consistency bug.** Same 20 lines as §7.3, reused — which is why it's the best test in the project.

**⚠️⚠️ Data incest — a first-class hazard for any system exchanging marginals.** Marginalization densifies over the Markov blanket, and the multi-robot payoff is **"condensed measurements"**: a robot marginalizes its private poses and transmits only the Schur complement over its separators — a small dense matrix summarizing an arbitrarily long private trajectory, a direct bandwidth win. **But: if A sends its marginal to B, and B (now containing A's information) sends its marginal back to A, the shared information is double-counted. Iterate around a cycle and the team becomes arbitrarily overconfident while the estimates look perfectly consistent with each other.**
- **Covariance Intersection** — `Λ_CI = wΛ_A + (1−w)Λ_B`, guaranteed consistent for *any* unknown correlation. Cost: discards real information.
- **Explicit provenance tracking** (channel filters) — exact, bookkeeping-heavy.
> **A decentralized system that exchanges marginals WILL become overconfident unless it handles this explicitly. Naming the hazard and stating which fix you chose and why is exactly what a technical-depth judge looks for — and exactly what most teams miss.**

**Sparsification** to fight fill-in: Mazuran et al., *Nonlinear Graph Sparsification for SLAM* (RSS 2014/IJRR 2016); Carlevaris-Bianco & Eustice, **GLC** (IROS 2012/T-RO 2014).

**iSAM2** (IJRR 2012, GTSAM BSD): Bayes tree = incrementally-maintained junction tree; adding a factor touches **only cliques on the path from the affected variables to the root** — O(1) amortized for chain-like graphs, O(n) on loop closure. **Fluid relinearization** (relinearize when `‖Δ_j‖_∞ > β`, typically 0.1) removes periodic batch relinearization; **partial back-substitution** gives the anytime property.
> ⚠️ **The variable ordering *is* the performance** — CCOLAMD with newest variables **constrained last**. Wrong ordering turns O(1) into O(n): the system works but is unusably slow and profiling points at the wrong place. ⚠️ **The classic silent bug: a too-loose relinearization threshold** accumulates linearization error the incremental machinery **never revisits**, converging to a wrong-but-stable answer with a small reported delta. **Guard: periodically re-solve the same graph with batch LM in the test harness and assert the costs match.**

**⭐ iMESA** (McGann, Rogers, Kaess, **RSS 2024**, [code MIT](https://github.com/rpl-cmu/iMESA)) — **incremental *and* distributed**: each robot runs its own iSAM2 and exchanges only shared-variable estimates via a biased-prior/consensus scheme. **Exactly the intersection Ariadne needs, and MIT-licensed. Read this closely — it's the closest existing system to Ariadne's backend requirements.**

### 7.12 ⭐ The bandwidth formula and its levers

```
Bytes ≈ N_iter × Σ_α |S_α| × r(d+1) × B
        ↑         ↑          ↑          ↑
      §7.12.1  §7.12.2    payload    §7.12.4
```
At r=5, d=3: `r(d+1) = 20` scalars = **80 B/pose/iteration in fp32.**

**1. Reduce iterations — the best lever by a wide margin.** Better init (spectral/chordal, flagged init), acceleration (RBCD++/AMM-PGO Nesterov + adaptive restart), preconditioning. **A 3× iteration reduction beats any quantization scheme and costs no accuracy. Do this first and measure it.**

**2. Reduce separators** — METIS-style multi-level partitioning ([2401.01657](https://arxiv.org/abs/2401.01657)) balances subgraph sizes and shrinks the inter-robot cut.

**3. ⭐ Event-triggered communication — best value per line.** Transmit only when `‖x_α^k ⊖ x_α^{last sent}‖ > θ_k`. **Summable threshold** (`Σθ_k < ∞`, e.g. `θ₀/(k+1)^p`, p>1) ⇒ **asymptotic convergence preserved**; constant θ ⇒ convergence to a θ-neighborhood. **Typically cuts messages 50–90% at negligible accuracy cost**, because most poses stop moving early while a few keep refining. **Difficulty: Low — and sweeping θ₀ is the easiest way to *generate the bandwidth-vs-accuracy curve*, which is your headline metric.**

**4. Quantization.** fp32→fp16 halves bytes at ~1e−3 relative error — **do it unconditionally.** **Deterministic uniform quantization biases the fixed point** by `O(Δ/λ₂)` — note the `1/λ₂`: **poorly connected networks amplify quantization bias**, exactly Ariadne's regime. **Prefer stochastic (dithered) quantization** (`E[Q(x)] = x`), which removes the bias for an `O(Δ²)` variance floor. **⭐ Error feedback / innovation compression is the best bytes-per-accuracy technique:** send `Q(x^k − x̂^{k−1})` with both endpoints maintaining `x̂`. Because increments shrink geometrically, **the same bit budget buys exponentially more precision each round** — linear convergence at constant bits/round ([2105.06697](https://arxiv.org/abs/2105.06697)). **If you implement one thing here, implement this.**
> ⚠️ **DO NOT quantize the 9 entries of R independently** — you leave SO(3) and the reprojection error is unbounded relative to your step. Two correct options: **unit quaternion "smallest three"** (drop the largest-magnitude component, recoverable from unit norm + sign; **32–40 bits gives ~0.1° resolution** — the standard game-netcode trick, excellent here), or **tangent-space increment** `ξ = log(R_lastᵀ R_new)^∨ ∈ ℝ³` — small near convergence, 3 numbers, and **it pairs perfectly with error feedback since it *is* an increment. This is the one to build.**
> Theory: [2506.07351](https://arxiv.org/abs/2506.07351) (2025), *Quantized Riemannian Gradient Tracking on Compact Submanifolds* — the rigorous foundation, and a strong citation.

**5. ⚠️ Honest byte accounting.** **Descriptor and keyframe traffic for loop-closure detection will dwarf optimizer traffic by one to two orders of magnitude.** Optimizing 80 B/pose/iteration while shipping kilobyte descriptors is optimizing the wrong term. **Report both separately** and make loop-closure prioritization (§1.4) a first-class part of the bandwidth story. **A judge will notice if you don't.**

**6. ⭐ The design rule.** Bytes-vs-ATE is a **hockey stick** — flat as you cut bytes, then a **knee**, then sharp degradation. The knee sits where communicated precision reaches the measurement noise floor.
> **There is no point transmitting a rotation more precisely than the relative-rotation measurement noise. Quantize to ≈ σ_R/3 and set the event-trigger threshold in the same units. Below that you are spending bytes to transmit noise.**
>
> **Measurement protocol for the writeup:** instrument bytes **at the serialization boundary** (not the algorithm's notion of a message); report **bytes-to-reach-an-ATE-threshold**, not bytes-at-convergence (which rewards early stopping); plot **ATE vs cumulative bytes** as the headline figure **with the knee annotated and the σ_R/3 rule marked on it.** That one plot makes the entire bandwidth argument.

### 7.13 Build order and the three decisions to defend

| Week | Component |
|---|---|
| 1–2 | Lie group library (exp, log, Adj, left/right Jacobians) + **the gauge null-space test (§7.3)** + Kabsch/Horn/Umeyama with **all** determinant branches. Validate against manif/Sophus/Eigen. |
| 2–3 | Chordal + spectral init. Batch GN/LM. **Validate cost against GTSAM on benchmark `.g2o` graphs.** |
| 3–4 | **DGS (§7.2)** — first working distributed backend, first bandwidth number. |
| 4–6 | **MM-PGO (§7.6)** as workhorse, then **RBCD/RBCD++** with parallel graph-coloring. |
| 6–7 | **The certificate (§7.4)** — Λ, S, distributed accelerated power iteration, Ritz-residual stopping, gauge deflation. **The headline technical result.** |
| 6–8 ∥ | Robustness stack (§7.8) + comms layer (§7.12). Produce the ATE-vs-bytes curve. |
| throughout | ASAPP bounded-delay handling; running-sum push-sum; weak term-fenced leader election — **not Raft**. |

**Oracles, not dependencies:** GTSAM (BSD-3 — batch LM, iSAM2, `ShonanAveraging3`, `.g2o` reader; **the primary oracle**) · `mit-acl/dpgo` (MIT — reference RBCD/DC²-PGO/ASAPP; **highest-value oracle for your core deliverable**) · `MurpheyLab/DPGO` (MIT — MM-PGO) · **Ceres `GradientChecker`** (numeric vs analytic — **the fastest way to kill Jacobian sign and left-vs-right convention bugs**) · SymForce (Apache-2.0) · manif/Sophus (MIT) · `Eigen::umeyama` (MPL-2.0, including the rank-deficient branch) · **TEASER++ (MIT) — use directly, don't reimplement; max-clique + adaptive voting is hard to do well and your differentiation is in the backend.**

**The three decisions to defend explicitly — each scenario-driven and rigorous:**
1. **Stateless-neighbor solvers (MM/RBCD/ASAPP) over consensus-ADMM**, because ADMM's stateful duals fail *silently and incorrectly* under packet loss while the others are provably delay-tolerant (§7.9).
2. **Raft-like protocols confined to broker election, config agreement and task commitment — never pose or map consensus** — with the FLP/partition-halting argument (§7.10).
3. **Anisotropic measurement covariances propagated into rotation averaging**, because corridor and rubble geometry is intrinsically degenerate and 2025 results show isotropic solvers *fail* there (§7.7).

**The three gotchas most likely to silently sink the project:** the **Stiefel-vs-SO(d) determinant confusion** (§7.5), the **marginalization consistency/FEJ bug** (§7.11), and **data incest from exchanging marginals** (§7.11).

---

## 8. Datasets, part 2 — the illumination audit

Second dataset pass, focused on construction/handheld/ground-robot SLAM and event cameras. **Method note: every size and license below was verified by direct HTTP fetch of primary sources** (dataset sites, HuggingFace API, `raw.githubusercontent.com`, arXiv PDFs) — not search snippets, not README badges.

### 8.1 ⭐⭐ THE headline find — the only verified onboard co-moving light

> **Exactly one publicly-released sequence anywhere in this survey is confirmed by its own paper to use an onboard, co-moving light in a genuinely dark environment.**
>
> **Hilti SLAM Challenge 2023, `site2_robot_3`** ("Floor 2 Large room – dark", **32 GB**). Verbatim from the RA-L paper ([arXiv:2404.09765](https://arxiv.org/abs/2404.09765), p.5):
> *"site2 robot 3 (blue): This sequence scans a similar large room one level below, **with on-robot mounted lighting in a dark environment**, and very limited map overlap with site2 robot 1."*

This is **exactly** Ariadne's photometric geometry: light rigidly co-moving with the camera, underground parking under renovation, bare graded concrete, non-parallel walls. Everything else labelled "dark" in the literature is dark under *ambient* conditions — a **materially different problem**: no radial falloff locked to camera pose, no moving specular hotspot, no shadow-flow keyed to ego-motion.

**Size correction:** `site2_robot_3.bag` is **34.87 GB** exactly (summed from the HF tree API); the website's "32 GB" is rounded. Repo totals: 2021 = 198.2 GB, 2022 = 349.8 GB, 2023 = 369.5 GB (bags alone: 198.2 / 335.9 / 344.2 GB).

**Platform:** *Trailblazer*, a 700 kg drilling robot — **Robosense BPearl hemispherical LiDAR** (32 rays, ±3 cm), **4× Luxonis OAK-D stereo pairs mounted all around**, **Xsens MTi-670 IMU**. PTP + hardware triggering; IMU↔LiDAR ≤1 ms, IMU↔camera ≤2 ms.
⚠️ **Spec corrections** (verified against arXiv Table I, second pass): the Hilti handheld **IMU is Bosch BMI085 @ 400 Hz, not 200 Hz**, and the **Alphasense cameras run at 40 Hz** for the 2022/2023 editions. The "5× 720×540 @ 40 Hz" spec is 2022/23 only — **the 2021 rig was 5× 1.3 MP global shutter @ 10 Hz** (they explicitly traded frame rate for resolution) and carried **two LiDARs** (Ouster OS0-64 **+ Livox MID70**) and **three IMUs**. GT also differs per edition: 2021 = PLT300 total station + **OptiTrack mocap** (<5 mm); 2022 = **Z+F Imager 5016 TLS** (σ_t <1 mm); 2023 = Trimble X7.

**GT:** Trimble X7 TLS + fiducial Ground Control Points; **92% / 84% / 90% of registered scans within ≤3 mm** per site. Sparse **3DoF position only** (no orientation) — the robot can't be tipped onto a survey spike, so GCPs are detected directly from LiDAR by a circular-target detector.
**Access:** ✅ **fully open HuggingFace, `gated: False`, no registration.** License **CC BY-NC-SA 3.0**.

> **⭐ The experiment this enables: `site2_robot_3` (32 GB, robot, onboard light) and `site2_handheld` (14 GB, handheld, same dark room) cover the *same scene*. That pair isolates platform from lighting — a genuinely controlled ablation you get for free, for 46 GB.**
> ⚠️ The site warns the additional Site 2 sequences *"contain rapidly flashing lights in their video streams"* — consistent with active onboard illumination, and something to handle in your exposure model.

Also from Hilti 2023: **Site 1 Underground (32 GB) and Stairs (17 GB)** — *"bright to dark transitions"*; **Site 3 Underground 1–4** (10/15/19/11 GB) — a tunnel-corridor network. Total dataset 369.5 GB.

### 8.2 The illumination audit — what is actually dark, and what lights it

| Dataset | Genuinely dark sequences | **Lit by** | Evidence |
|---|---|---|---|
| **⭐ Hilti 2023** | **`site2_robot_3`** (32 GB) | **ONBOARD, co-moving with cameras** | paper, verbatim (§8.1) |
| **Hilti 2023** | `site2_handheld` (14 GB); Site 1 Underground/Stairs; Site 3 Underground 1–4 | Likely onboard — **partially verified** (flashing-lights warning; "bright to dark transitions") | site + paper |
| **M2DGR** | **`room_dark_01–06`** (171 GB) — **complete darkness, zero-lux** | **NO LIGHT AT ALL** — deliberately unlit; the **thermal camera is the intended sensor** | [arXiv:2112.13659](https://arxiv.org/abs/2112.13659): *"recorded a few sequences in complete darkness to test SLAM systems' robustness"* |
| **M2DGR** | `gate_01/02`, night streets `street_01/03/04/05/06/08` | **Ambient/fixed** — street lamps, distant and static | paper: features came *"from far-away bright objects like street lamps or light screens"* |
| **Hilti 2021** | **Basement, Basement 3, Basement 4** (42 GB) | **Ambient/fixed** — *"No natural light, mixed illumination brightness. Concrete space with building infrastructure"* | [arXiv:2109.11316](https://arxiv.org/abs/2109.11316) |
| **Hilti 2022** | Exp03 Stairs, Exp14 Basement 2 — **dark corners only** | Ambient/fixed | [arXiv:2208.09825](https://arxiv.org/abs/2208.09825): *"a dark corner in the basement, returning few lidar points"* |
| **Ground-Challenge** | `Darkroom1/2/3` (6.1 GB) — **mild, not zero-lux** | Ambient — tags literally say **"slight light"** | README + [arXiv:2307.03890](https://arxiv.org/abs/2307.03890) |
| **SubT-MRS** | caves, tunnels, `SmokeRoom`, `OutdoorNight`, **`FlashLight`** | ⚠️ **UNVERIFIED** — a sequence is *named* `FlashLight` but **no primary source confirms onboard lights.** A sequence name is not evidence. | superodometry.com |
| **Newer College** | **NONE** | daylit Oxford college; hardest case is "lighting change" in Quad-Hard | all sequence descriptions checked |
| **Oxford Spires** | **NONE confirmed** | daylit landmarks; per-sequence badges emit only Indoor/Outdoor | page markup checked |
| **UrbanNav** | **NONE** | **zero** hits for "night"/"dark"/"illumin" in the entire README | verified |

> **⚠️ The M2DGR distinction matters and is easy to miss: zero-lux with *no* light is a fundamentally different problem from headlamp-lit.** M2DGR's entire experimental point is that **RGB fails and thermal succeeds** — the paper reports ORB-SLAM3 *"failed to extract enough feature points in those dark scenes… By contrast, using a thermal-infrared camera, ORB-SLAM3 achieved significantly better robustness."* Its Vicon GT survives the darkness because mocap is IR-based.
> **Use M2DGR `room_dark_*` to answer one specific question: does Ariadne need a thermal channel at all, or will a headlamp plus a good VO front-end do?** Nothing else can answer it — it's the only zero-lux ground-robot data with synchronised **thermal AND event** cameras plus mm-accurate mocap GT.

### 8.3 ⚠️ License landmines in this bucket

| Dataset | Badge / apparent | **Verified reality** |
|---|---|---|
| **Hilti 2021 / 2022 / 2023 / 2026** | — | ✅ **CC BY-NC-SA 3.0**, verbatim clause fetched from each site. Non-commercial **and share-alike** (virally licenses derivatives). |
| **Newer College, Oxford Spires** | — | ✅ **CC BY-NC-SA 4.0**, *"intended for non-commercial academic use."* |
| **M2DGR** | MIT badge | ⚠️ **CONTRADICTORY.** The `LICENSE` file is plain **MIT** ("without restriction"), but the README adds *"is provided for academic purpose… for commercial purposes, please contact us."* MIT covers "the Software" (tools) and is silent on data. **Unresolved — do not treat M2DGR data as MIT-clean.** |
| **Ground-Challenge** | MIT badge | ⚠️ **NO LICENSE FILE EXISTS.** Probed `LICENSE`/`.md`/`.txt`/`COPYING` on both `main` and `master` — **all 404.** The badge is a shields.io image, not evidence. The README's "License" section contains only a citation request. |
| **UrbanNav** | — | ⚠️ **NO LICENSE.** Same probe, all 404. README's "License" section contains **only contact email addresses.** No grant of rights of any kind. |
| **FusionPortable** | "MIT" on the site | ⚠️ **THE MIT REFERS TO THE MKDOCS THEME.** The site's License section reads in full: *"**Cinder** is licensed under the MIT license."* Cinder is the HTML theme. The dataset pages contain **zero** occurrences of "licen". **Do not cite FusionPortable as MIT-licensed data.** |
| **SubT-MRS** | — | ⚠️ **No license statement anywhere.** Zero occurrences of "licen"/"Creative Commons" on the datasets page; LICENSE probes on three plausible repos all 404. |

> **Everything you'd actually use here is non-commercial and share-alike.** Fine for the competition. If Ariadne has any commercial trajectory, resolve the four unlicensed datasets by email first.

### 8.4 Other findings from this pass

**Ground-Challenge** ([arXiv:2307.03890](https://arxiv.org/abs/2307.03890), ROBIO 2023) — 78.8 GB, 36 trajectories, deliberately driven into corner cases (poor illumination, textureless walls, motion blur, occlusion, pure rotation, wheel slip). Cheap and thematically aimed. ⚠️ **But its GT is Fast-LIO2 output** — verbatim: *"we use the result of Fast-LIO2 as the **pseudo-ground-truth** trajectory."* Validated against mocap at 3 cm in a 15×15 m room. **Fine for relative comparison; you cannot make absolute-accuracy claims against it.** Combined with the missing license, use it as a fast smoke test, not a benchmark.

**Hilti 2022** ships **full 3D Terrestrial Laser Scans in `.e57`** of the Construction Site and Sheldonian Theatre — genuine reconstruction ground truth, not just trajectory. GT method: **Z+F Imager 5016 TLS**, registration sigma **below 1 mm for all stations**. Its 5 cameras are explicitly **"Global shutter (Infrared)"**, which behave better in low light than typical rolling-shutter colour rigs.

**Oxford Spires** (IJRR 2025, ~1.31 TB) — environmentally wrong (daylit Oxford landmarks) but **methodologically the right benchmark**: it ships a full reconstruction **and radiance-field (NeRF/3DGS) benchmark** against **Leica RTC360 mm-accurate TLS**. **That is exactly Ariadne's output modality — use it as the recipe for how to evaluate a reconstruction against TLS ground truth.**

**ENWIDE / COIN-LIO** (ETH ASL, ICRA 2024, code ✅ **BSD-3**) — built specifically for **environments with large sections of degenerate geometry** (a `tunnel` sequence, plus smooth/dynamic variants). No camera at all, so it can't exercise the headlamp pipeline — **but COIN-LIO's trick, augmenting LiDAR with intensity images, is directly transferable, and intensity imaging is illumination-invariant, sidestepping the headlamp problem entirely.** Worth reading for the idea.

**Not covered / deprioritized:** UrbanNav (outdoor daytime car-mounted GNSS benchmark — **zero** darkness hits; no license; poor cost/benefit) · Hilti 2026 (360° consumer video + consumer IMU only, no LiDAR/depth/stereo released — wrong modality) · TUM-VI and EuRoC (not fetched this pass; **not stated from memory** per protocol).

> **⚠️ No dataset in this bucket contains an actual collapsed structure.** Nearest proxies: Hilti's under-renovation underground parking and tunnel networks, and SubT-MRS's caves. **This reinforces §6.5 — genuine rubble geometry needs the simulator, and Hilti 2022/2023's TLS scans plus Spires's benchmark give you the validated recipe for evaluating it.**

### 8.5 Event cameras — ⛔ verdict: distraction, with one narrow carve-out

**⚠️ OpenCV 5 has ZERO event-camera support — verified, not assumed.** Module directories enumerated directly from the 5.x GitHub trees (core: 23 modules; contrib: 60 modules — **none event-related**), and `videoio.hpp` grepped for `prophesee|metavision|event|dvs|davis|neuromorph` → **zero hits** in the backend enums. **No event data type, no event I/O, no event representation, no event algorithm.** The nearest neighbour, `bioinspired`, is a retina *model* operating on ordinary frames.

**The arithmetic:** event data is not images. Before a single result, you must write a representation layer from scratch — accumulating an asynchronous `(x, y, t, polarity)` stream into event-count images, time surfaces, voxel grids, or E2VID reconstructions. Plus HDF5 stream indexing via `ms_to_idx`, accumulation-window choice (differs between a night drive and a 1-lux indoor loop), event rectification, sync bookkeeping. **Realistically 4–8 days before a first plot** — and at the end you own a codepath sharing **no primitives** with the rest of Ariadne: no `goodFeaturesToTrack`, no ORB, no `calcOpticalFlowPyrLK`, no `solvePnP` on the raw stream. **Every OpenCV skill is inapplicable until that layer exists.** And with no hardware there is no deployment path.

> **⭐ The one-day carve-out that is worth doing.** Pull exactly two files — **M3ED `falcon_outdoor_night_high_beams` (7.9 GB)** and **TUM-VIE `floor2-dark` (13.2 GB)** — render both as time-surface videos beside their **co-recorded conventional frames** (both ship co-registered grayscale, so the comparison is free). **~1 day, produces a genuinely persuasive figure about why frames fail in this regime, commits you to nothing.** If anyone insists on event coverage, this is the whole of it.

The M3ED sequence note: *"the UAV flies in a parking lot, in front of a car with the high beams enabled. This sequence showcases the advantages of event cameras compared to traditional imagers."* A **drone, at night, against a harsh directional light** — closest thing anywhere to the scenario, though the light is static.

**Also relevant even if you skip events:** M3ED `spot_indoor_stairwell` — *"spot is driven inside a stairwell, very close to the walls… very challenging for VIO systems, as there are not many features on the stairwell steps."* **That is the texture-poor problem exactly.** And **night-driving headlights genuinely ARE a co-moving light rigidly attached to the sensor** — the one structural match to agent-carried lighting, available in MVSEC `outdoor_night1/2/3` and DSEC Zurich City night.

**Event dataset licenses:** ✅ DSEC, MVSEC, M3ED **CC BY-SA 4.0** (commercial OK) · TUM-VIE **CC BY 4.0** · VECtor **BSD-3-Clear** · ⚠️ **ECD is CC BY-NC-SA 3.0 — the only NC one.** Prefer M3ED or TUM-VIE. Sizes are not the blocker: single sequences are 6–19 GB over plain unauthenticated HTTPS.

**On simulating events (ESIM / v2e):** both MIT-licensed (verified). ⚠️ v2e's **SuperSloMo checkpoint is a separate 151 MB artifact on Google Drive with no license stated** — code license ≠ weights license. **Recommendation: don't.** Simulated events inherit every flaw of your simulator, and the specific things that would break Ariadne — dust scattering the beam, specular wet concrete, **a light source moving with the camera so scene illumination changes as you move** — are precisely what neither ESIM nor v2e models faithfully. A frame-based simulator with an honest scattering model answers the real question better and reuses your existing stack.

---

## 9. Degraded visual environments — vision under self-carried illumination

*Domain 4. This section answers the architectural warning raised in §1.1: direct/photometric methods assume brightness constancy, and a co-moving light violates it at first principles. Below is what the literature actually offers as mitigation, and how much of it is real.*

**Datasets for this domain are in §6 — not repeated here.** License notes in this section are informational only; per team decision, methods here are **reimplemented from the paper**, not linked.

### 9.0 ⭐⭐ The headline finding — the problem has a name, a model, and a 2025 paper

The dominant condition (power out, agent carries its own lamp) is **near-field illumination**, and it is not a vague "hard lighting" problem. It has a precise physical model and a small, high-quality literature — mostly published under a domain label the team would never search for: **endoscopy**. An endoscope is a camera with a light source rigidly bolted to it, imaging a texture-poor, non-Lambertian, close-range cavity. That is Ariadne's optical geometry exactly.

**Why brightness constancy dies.** For a distant source (sun, ceiling light), scene radiance is independent of camera pose, so a pixel's intensity is (approximately) a property of the surface. With a co-located source, irradiance at a surface point falls as **1/r²** where r is the *camera-to-surface* distance, and is further modulated by the **n·l** foreshortening term. Both r and n·l are functions of the pose being estimated. Illumination is therefore **coupled to pose**, and the coupling is multiplicative and strongly nonlinear — not an additive brightness offset. This is why **DSO's affine brightness transfer model (a·I + b) cannot absorb it**: an affine map is global per-frame, while near-field falloff is per-pixel and depth-dependent. Approaching a wall does not brighten the image uniformly; it brightens the near part quadratically and leaves the far part black.

**The flip side, and the opportunity:** the same coupling that destroys brightness constancy makes intensity an **observable of depth**. If you know the photometric model, radiometric falloff is a *depth sensor*. Several papers below exploit exactly this. For a monocular system this is unusually valuable — it attacks scale ambiguity, which is monocular SLAM's fundamental weakness.

### 9.1 ⭐⭐ NFL-BA — Near-Field Light Bundle Adjustment

**NFL-BA** — Dunn Beltran, Rho, Niethammer, Sengupta, **NeurIPS 2025**, [arXiv:2412.13176](https://arxiv.org/abs/2412.13176), [project page](https://asdunnbe.github.io/NFL-BA/), [code](https://github.com/asdunnbe/NFL-BA/) (no LICENSE file present).

**The single most on-target paper found in this domain.** Its opening sentence names Ariadne's problem verbatim: *"SLAM systems typically assume static, distant illumination; however, many real-world scenarios, such as endoscopy, subterranean robotics, and search & rescue in collapsed environments, require agents to operate with a co-located light and camera in the absence of external lighting."*

**The mechanism — worth implementing, because it is only a loss function.** Replace the photometric BA term (which compares raw RGB and therefore implicitly assumes constant lighting) with one that renders through an explicit near-field shading model. Per-point:

```
Ld(x)  = (x - xL) / ||x - xL||            light direction (xL = light position, camera-attached)
La(x)  = (Ld(x)ᵀ f)^β / ||x - xL||²        attenuation: angular falloff × inverse-square
PPS(x) = La(x) · (Ld(x)ᵀ n(x))            "per-point shading" = attenuation × Lambertian n·l
Ĉ(x)   = ρ(x) · PPS(x)                     rendered colour = albedo × shading
```

with `f` the forward optical axis and `β` an angular (spotlight) attenuation coefficient. **They set β = 0**, arguing angular falloff is negligible against the inverse-square term — so in practice `La(x) = 1/||x − xL||²`. Colour is composited along the ray in the usual volumetric way, but with `ρ·PPS` substituted for the raw radiance. Normals come from the geometry itself: `n = −∇σ/||∇σ||` for a neural field, or the shortest axis of each Gaussian for 3DGS, oriented toward the camera by forcing `nᵀLd > 0`.

**Key structural insight:** the network's per-point colour feature is *reinterpreted* as the albedo ρ rather than as final radiance. Shading is then computed analytically from the current geometry estimate. So the geometry is forced to explain the observed brightness pattern — **shading gradients become geometry supervision**, exactly what is missing on texture-poor surfaces.

**Results:** integrated into MonoGS, NICE-SLAM and EndoGSLAM. On C3VD colonoscopy, MonoGS ATE 3.48 → 2.18 mm (**−37%**) and Chamfer 1.59 → 0.99 mm (−38%); EndoGSLAM −13%. On four self-captured **indoor rooms lit only by a phone flashlight**, ~**35%** average tracking-error reduction.

*Difficulty: **Medium** as a loss term if you already have differentiable rendering; the model itself is ~30 lines. **High** if you must build the differentiable renderer first.*
*OpenCV 5: not an OpenCV algorithm — it lives in the 3DGS/optimizer layer. `ptcloud` for the point/normal handling.*

> **⚠️ Honest fit — read before adopting.** Two caveats.
> 1. **It is formulated inside differentiable-rendering SLAM**, not a classical feature front-end. Ariadne's §1.9 stack is feature-based. The transferable piece for the front-end is not the loss but the **PPS shading model used as photometric normalisation**: divide out the predicted `1/r² · (n·l)` using the current depth estimate before computing any photometric or descriptor comparison. That converts a pose-coupled intensity into an approximate albedo image — the correct pre-processing step for this environment, and cheap.
> 2. **The subterranean/search-and-rescue claims are motivational.** The experiments are endoscopy plus flashlight-lit rooms. **No subterranean or rubble experiments exist.** This is an opportunity, not a defect: applying it to a collapsed-structure setting is a genuine, defensible contribution — and the phone-flashlight room captures show the team how to build a validation set for ~zero cost.

### 9.2 ⭐ Depth from illumination decline — the physics turned into a sensor

The strongest idea in the domain: **the 1/r² falloff is a metric-depth cue**, and it is available from a single view.

**LightDepth** — Rodríguez-Puigvert, Batlle, Montiel, Martinez-Cantin, Fua, Tardós, Civera, **ICCV 2023**, [arXiv:2308.10525](https://arxiv.org/abs/2308.10525). Single-view depth **self-supervision** from illumination decline. Models `I ∝ ρ · cos θ / d²`; the network predicts depth, the model renders the expected brightness, and the photometric residual is the training signal. **No ground-truth depth needed** — decisive for a domain where GT will never exist. Reported to approach supervised performance.
*Difficulty: **Medium**. The loss is simple; the work is in calibrating ρ and the light profile, and in handling specularities.*

**LightNeuS** — Batlle, Montiel, Fua, Tardós, **MICCAI 2023**, [arXiv:2309.02777](https://arxiv.org/abs/2309.02777). NeuS + a **calibrated photometric model of the camera/light pair** + a watertight SDF prior. The inverse-square relation between brightness and depth is written directly into the reconstruction objective. The watertightness prior lets it plausibly **complete unobserved surface** — interesting for Ariadne, where "is there a void behind that slab" is a mission question, though it must be flagged as inference, not measurement.
*Difficulty: **High** (neural implicit surface + calibration).*

**Photometric single-view dense 3D reconstruction in endoscopy** — Batlle, Montiel, Tardós, **IROS 2022**, [arXiv:2204.09083](https://arxiv.org/abs/2204.09083). **Photometric stereo on a single calibrated monocular camera with a co-located light**, plus an in-place calibration procedure. Mean depth error **7% (<3 mm)** on simulation; qualitatively correct on real human colonoscopy. **The most reimplementable member of this family** — classical, no network, and it delivers *true metric scale from a monocular camera*, which is exactly what a monocular multi-agent system needs.
*Difficulty: **Medium**. ⭐ Best depth-per-effort item in this section.*
*OpenCV 5: `calib` (in-place calibration), `imgproc`, `photo`. Pairs naturally with `ptcloud`.*

**ColIAGS** — Wang et al., **ICME 2026**, [arXiv:2510.18739](https://arxiv.org/abs/2510.18739), [code](https://github.com/haowang020110/ColIAGS) (no LICENSE file). Illumination-**attenuation-aware 3DGS** for a moving light source, using *two* classes of attenuation factor rather than distance decay alone. Confirms the failure mode independently: vanilla 3DGS compensates for lighting variation by **distorting geometry** — it spawns bogus Gaussians to explain brightness changes. **Directly relevant to Ariadne's reconstruction stage**: without an illumination model, the splat map will absorb lighting into geometry and the reconstruction will be quietly wrong.

**ColonAdapter** ([arXiv:2511.22250](https://arxiv.org/abs/2511.22250), 2025) names the triple threat as *"non-Lambertian surfaces, moving light sources, and large textureless regions"* — a one-line statement of Ariadne's visual problem, useful for the report.

### 9.3 ⭐ Photometric calibration — the unglamorous prerequisite

Every method in §9.1–9.2 assumes you know the map from scene irradiance to pixel value. You do not, unless you calibrate. The model:

```
I(x) = G( t · V(x) · B(x) )
```
`B` = irradiance, `V` = vignetting (radial attenuation), `t` = exposure time, `G` = camera response function (nonlinear, usually gamma-ish). Under auto-exposure, `t` changes every frame and is **not reported** by most cameras — so raw intensities are not comparable frame to frame even under *constant* lighting.

**Online Photometric Calibration** — Bergmann, Wang, Cremers, **RA-L 2018**, [arXiv:1710.02081](https://arxiv.org/abs/1710.02081), [code](https://github.com/tum-vision/online_photometric_calibration) — **BSD-3-Clause, verified by reading `LICENSE.txt`**. Note: GitHub's API reports `NOASSERTION` for this repo, and a badge-reader would have got it wrong — a clean illustration of why the metadata is untrustworthy even when, as now, the answer is permissive.
Recovers **response function, vignetting, and per-frame exposure jointly and online**, with **no calibration target**: track KLT features, assert that a tracked point's true irradiance is constant, and solve the resulting nonlinear least-squares for `G`, `V`, and `{t_i}`. Runs in real time alongside the VO front-end.
*Difficulty: **Medium**. Self-contained, testable in isolation, and one of the cleanest "real signal processing" items available. ⭐ Strong candidate.*
*OpenCV 5: `video` (KLT `calcOpticalFlowPyrLK`), `features`, `imgproc`; the solver is yours (or Ceres).*

**A Photometrically Calibrated Benchmark for Monocular VO** — Engel, Usenko, Cremers, [arXiv:1607.02555](https://arxiv.org/abs/1607.02555). The **TUM monoVO** dataset: 50 sequences with full photometric GT (response, vignette, exposure). **The correct place to validate a calibration implementation before trusting it on your own footage.** See §6.

**OSPC — Online Sequential Photometric Calibration** — [arXiv:2305.17673](https://arxiv.org/abs/2305.17673), [code](https://github.com/8bit-nyk/OSPC) (GPL-3.0, MATLAB). Estimates parameters **sequentially rather than jointly**, which is more robust to the initialisation sensitivity of the joint solve. Useful as an idea; the MATLAB implementation is not a dependency.

**Challenges in Monocular VO** — Yang, Wang et al., **RA-L 2018**, [arXiv:1705.04300](https://arxiv.org/abs/1705.04300). Quantifies how much photometric calibration, motion bias and rolling shutter each cost. Good citation for justifying the effort.

### 9.4 ⭐ Exposure control — the actuator you already own

If illumination is coupled to pose, **exposure is a control input**, not a camera setting. Off-the-shelf auto-exposure optimises for human viewing (mean brightness), which is the wrong objective: it will happily blow out the bright near-field that the lamp is illuminating.

**Active Exposure Control for Robust VO in HDR Environments** — Zhang, Forster, Scaramuzza, **ICRA 2017**, [PDF](https://rpg.ifi.uzh.ch/docs/ICRA17_Zhang.pdf).
**Mechanism, verified from the paper.** Choose exposure to maximise **gradient information**, since gradients are what both feature and direct methods consume. They compare four metrics:
- `M_sum = Σ ||∇I(u)||` — not robust; the squared sum is dominated by a few strong edges, so it drives toward overexposure.
- `M_shim` (Shim et al.) — logarithmic gradient mapping `log(λ(G̃−σ)+1)`; better, still not robust.
- `M_perc(p)` — the **p-th percentile** of gradient magnitudes. A robust scale estimator; **best in 13 of 18 real datasets** by extracted-feature count. But it is not differentiable in exposure.
- `M_softperc` — a **smooth weighted approximation** of the percentile, which is what they actually optimise, precisely because it varies smoothly with exposure time.

The elegant part: they never bracket exposures. Using the calibrated photometric response they compute **∂M/∂Δt analytically** — via Schwarz's theorem, `∂/∂Δt[∇I] = ∇[∂I/∂Δt]`, and `∂I/∂Δt = E(u)/(g′(I)·Δt)` — then do **gradient ascent** on exposure time. One parameter (the ascent rate γ), scheduled by a brightness lookup table. They note **gain must be controlled too**, since exposure time alone saturates in extreme scenes.
*Difficulty: **Medium** — but it needs a camera whose exposure and gain you can set per frame, and the photometric response from §9.3. **In simulation it is easy and fully controllable**, which makes it an excellent ablation axis.*
*OpenCV 5: `imgproc` (Sobel/Scharr) for the metric; the control loop is yours.*

**Active Illumination for Visual Ego-Motion Estimation in the Dark** — Crocetti, Dionigi, Brilli, Costante, Valigi, **ICRA 2025**, [arXiv:2502.13708](https://arxiv.org/abs/2502.13708). The natural next step: **aim the lamp**. A detector identifies feature-rich regions, a **pan-tilt controller steers the light beam** toward them, and VO consumes the better-lit frames. Reports **up to 75% pose-error reduction versus fixed lighting** on a real platform.
> **This is the most Ariadne-shaped idea in the section after NFL-BA.** It reframes illumination as an **active perception** problem, which connects Domain 4 directly to Domain 3 (active perception / next-best-view). "Where should I point my light?" is the same class of decision as "where should I look next?", it is trivially simulated, and it is a genuinely novel thing to show a judge. Even a simple version — steer or bias the lamp toward the region of highest predicted gradient information — is a strong, cheap demo and a natural task for the LLM/agentic layer.

### 9.5 ⚠️ Cross-cutting empirical finding — restoration is not free

**Twilight SLAM: Navigating Low-Light Environments** — Singh, Mazotti, Rajani, Mayilvahanan, Li, Ghaffari, 2023, [arXiv:2304.11310](https://arxiv.org/abs/2304.11310).

The only study found that asks the question the team must answer before building a restoration front-end: *does enhancing the image actually improve localisation?* They cross ORB-SLAM3 and SuperPoint-SLAM with EnlightenGAN, Bread, Zero-DCE and Dual, on KITTI and ETH3D.

**Findings, stated honestly:**
- Semi-dark: **real gains.** ETH3D `sfm_house_loop` RMSE 0.43 m → **0.15 m** (EnlightenGAN + ORB-SLAM3).
- Dark: **gains shrink.** `sfm_lab_room2` 1.13 m → 0.53 m, and the authors' own conclusion is that in extremely dark environments *"the improvements in localization accuracy with SOTA enhancers are incremental."*
- **Enhancement sometimes hurts.** SuperPoint-SLAM *frequently performed worse* with enhancement, and failed outright in the darkest sequences regardless.

> **Consequences for Ariadne — this is the empirical backbone of risk R4.**
> 1. **Every restoration stage must be ablated against SLAM accuracy, not against PSNR.** A prettier image is not the deliverable. This is already planned (§6.1 of the positioning doc); this paper is the citation that makes it a *considered* decision rather than a hedge.
> 2. **Enhancement interacts with the front-end.** A learned detector trained on normal-exposure images can be pushed off-distribution by an enhancer. Ariadne's front-end is learned (ALIKED/DISK/XFeat), so this risk is live and must be measured per-detector.
> 3. **The generic-enhancer approach saturates in real darkness — which is precisely the argument for the physics-based route in §9.1–9.3.** Generic low-light enhancement does not know where the light is. A near-field model does. That contrast is the strongest narrative available for this domain: *we did not reach for an off-the-shelf enhancer, because the published evidence says it stops helping exactly where our problem starts.*

### 9.6 ⭐⭐ OpenCV 5 findings that affect settled decisions

Verified directly against the `5.x` branches of `opencv/opencv` and `opencv/opencv_contrib`.

**1. OpenCV 5 ships a monocular visual odometry pipeline.** `modules/ptcloud/include/opencv2/ptcloud/slam.hpp` declares a module documented as:

> *"@defgroup slam SLAM and Visual Odometry — Monocular visual odometry pipeline. Entry point is `cv::slam::VisualOdometry`. Bootstraps an initial map from two-view geometry, then tracks subsequent frames with PnP, growing the map at keyframe promotions."* (Copyright 2026, BigVision LLC.)

**API surface, read from `ptcloud/slam/visual_odometry.hpp`** — this is a real pipeline, not a stub:

```cpp
static Ptr<VisualOdometry> create(const Ptr<Feature2D>& detector,
                                  const Ptr<DescriptorMatcher>& matcher,
                                  InputArray cameraMatrix, InputArray distCoeffs,
                                  const OdometryParams& params);
bool processFrame(InputArray image);           // true if a pose was emitted
OdometryState getState();  Matx44d getLastPose();
const std::vector<Matx44d>& getTrajectory();   int getNumKeyframes(); int getNumMapPoints();
```
Documented state machine: `NOT_INITIALIZED → INITIALIZING (H/F two-view bootstrap) → TRACKING (per-frame PnP + local-map refinement)`, with tracking failure rewinding to `INITIALIZING`. Everything is `CV_WRAP`, so it is **exposed to Python**. There is a `samples/slam` example that drives it from a directory of images and **writes the result in COLMAP format**.

Two details that matter more than the existence of the module:
- **The detector and matcher are injected, not hard-coded.** It takes any `Feature2D` + `DescriptorMatcher` — so **ALIKED or DISK (§1.0) plug straight in**. That makes it a *controlled* baseline: same front-end features as Ariadne, different back-end, which isolates the contribution of the from-scratch back-end rather than confounding it with the feature choice. That is a much stronger ablation than comparing against a black box.
- **COLMAP-format output is a free bridge to the 3DGS stage.** Standard 3DGS initialisation consumes COLMAP. A working `images → COLMAP → splat` path on day 2, entirely inside OpenCV 5, is the thinnest possible end-to-end spine — precisely the R2 mitigation the positioning doc asks for.

`ptcloud` also carries `odometry.hpp` (RGB/DEPTH/RGB_DEPTH odometry, ICP-based, `Odometry::compute`), `volume.hpp` (TSDF integration), `depth.hpp` and `viz3d.hpp` — i.e. the old `rgbd`/KinectFusion machinery **promoted into the main repository**.

> **This bears directly on the settled "write SLAM from scratch" decision (positioning §4.2) — it does not overturn it, but it changes how it must be defended.** Three consequences:
> - **The judges will know this exists.** "We wrote SLAM from scratch" invites "OpenCV 5 ships one — why?" The answer is strong and should be prepared: `cv::slam::VisualOdometry` is **monocular, single-agent, and assumes brightness constancy**. Ariadne is **multi-agent, collaborative, and explicitly targets the case where brightness constancy fails.** Building on it would forfeit the entire technical claim.
> - **It is a free test oracle.** Exactly the R1 mitigation already in the plan: run `cv::slam::VisualOdometry` on the same sequence as the from-scratch front-end. Divergence localises the bug. This is a better oracle than an external dependency because it ships with the required library.
> - **It is a legitimate baseline row in every results table**, and an *in-library* baseline is more persuasive to these judges than an external one.
>
> **Recommend verifying on day 1 whether `ptcloud`/`cv::slam` is present in the OpenCV 5 Python wheels**, alongside the existing day-1 check on `sfm` (positioning §4.1). Same class of risk.

**2. Correction to the module map (positioning §4.1): `quality` is NOT in the main repository in 5.x.** `opencv/modules/quality/` returns 404; `opencv_contrib/modules/quality/` exists. The evaluation harness (PSNR/SSIM on held-out views) therefore **requires a contrib build**. Also contrib in 5.x: `xphoto`, `ximgproc`, `rgbd`, `optflow`, `intensity_transform`, `sfm`. Confirmed in main: `photo`, `features`, `calib`, `geometry`, `stereo`, `ptcloud`, `video`, `imgproc`.

**3. What `photo` (main) actually gives you for this domain**, verified from `photo.hpp`: `fastNlMeansDenoising{,Colored,Multi,ColoredMulti}` — note the **`Multi` variants take a temporal window of frames**, i.e. multi-frame denoising is already in the box; `denoise_TVL1`; `createAlignMTB`, `createCalibrate{Debevec,Robertson}`, `createMerge{Debevec,Robertson}`, `createTonemap{,Drago}` — a complete HDR path including **radiometric response estimation**, which is directly reusable for the §9.3 calibration work; plus `edgePreservingFilter`, `detailEnhance`, `decolor`, `inpaint`.
`xphoto` (contrib) adds `bm3d_image_denoising`, `dct_image_denoising`, `white_balance`, `oilpainting`, `tonemap`, `inpainting`.

### 9.7 Illumination-invariant representations — an obvious idea that does not apply here

The classical route to "lighting robustness" is an **illumination-invariant colour space**: assume a Planckian (blackbody) illuminant and narrow-band sensors, take logs of the three channels, and form a weighted log-chromaticity combination `I = log R₂ − α·log R₁ − (1−α)·log R₃` whose value is invariant to the illuminant's **colour temperature**. The weight α is fixed by the sensor's peak wavelengths, or found by entropy minimisation over projection angles (the Finlayson intrinsic-image/shadow-removal line of work). It is used in outdoor robotics to make matching robust across sun/shadow and across the day, and it works well there.

> **Verdict for Ariadne: ⛔ don't bother.** The invariance on offer is to **changes in the illuminant's spectrum**. Ariadne's lamp is a single LED of fixed spectrum; the spectrum never changes. What changes is **geometric**: `1/r²` falloff and `n·l` foreshortening, both of which are achromatic and survive log-chromaticity untouched. The technique removes exactly the variation Ariadne does not have and leaves exactly the variation it does. Worse, it discards intensity — and per §10.2 intensity **is the depth cue** in this environment.
>
> Stated positively, and this is the sentence for the report: **the right invariance here is radiometric, not spectral.** Divide out the modelled near-field falloff (§10.1) rather than projecting away chroma.
>
> *Caveat: the specific outdoor-robotics citation for this family (Maddern et al., building on Ratnasingam & Collins) is a workshop paper that is poorly indexed and was not retrieved directly. The mechanism above is standard and safe to state; pin the exact reference before citing it in the report.*

### 9.8 The practical recipe — what to actually build, in order

Synthesised from §10.0–9.7. This is the minimal path that makes the illumination story real rather than decorative.

| Order | Component | Source | Difficulty | Why it earns its place |
|---|---|---|---|---|
| 1 | **Simulate the light correctly** — a point source at the camera, intensity `∝ 1/r² · (n·l)`, optional cone falloff | §10.1 model | **Low** | Everything downstream is only as honest as this. It is ~10 lines in a shader and it is *physically correct*, unlike simulated smoke. Expose lamp power, cone angle and lamp-camera offset as sweep axes. |
| 2 | **Photometric calibration** — response, vignette, per-frame exposure | Bergmann RA-L 2018 (§10.3) | **Med** | Prerequisite for 3–5. Self-contained, unit-testable, validated against TUM monoVO (§6). Real signal processing. |
| 3 | **Radiometric normalisation in the front-end** — divide out predicted `1/r²·(n·l)` using current depth, then detect/describe on the result | PPS model from NFL-BA (§10.1) | **Low–Med** | The cheapest way to get most of the benefit into a *feature-based* pipeline. Ablate against raw intensity — this is a headline chart. |
| 4 | **Depth from illumination decline** | Batlle IROS 2022 (§10.2) | **Med** | ⭐ Gives **metric scale from a monocular camera**. Attacks monocular SLAM's core weakness using the very degradation that was supposed to be the problem. Best narrative payload in the domain. |
| 5 | **Active exposure control** | Zhang ICRA 2017 (§10.4) | **Med** | Turns exposure into a control input. Trivial to ablate in sim. |
| 6 | **Near-field-aware reconstruction loss** | NFL-BA (§10.1) | **Med–High** | Stops the splat map from baking lighting into geometry (§10.2, ColIAGS). Needed for reconstruction quality, not tracking. |
| 7 | *Stretch:* **steer the lamp** | Crocetti ICRA 2025 (§10.4) | **Med** | Connects Domain 4 to Domain 3 active perception. A genuinely novel demo. |

**The one-sentence claim this buys:** *we do not enhance the image and hope — we model the light we carry, calibrate it, divide it out of the front-end, and then exploit its falloff as a depth cue.* That is a defensible mechanism a judge can probe, which is the stated selection criterion (positioning §4.3).

**Required ablation (feeds risk R4):** trajectory error and reconstruction quality with each of stages 2–6 on and off, swept over lamp power and lamp-camera offset. Include the **negative controls** — raw intensity, and a generic off-the-shelf enhancer — because §10.5 predicts the generic enhancer will saturate, and demonstrating a predicted negative result is stronger evidence of understanding than another win.

---


---

## 10. Dust, scattering & confined-space vision

Domain 4, partial. **Module facts below were verified by fetching files on the `5.x` branch**, not from docs.

### 10.1 🚨 PLAN-BREAKING: `quasi_dense_stereo` does not exist in OpenCV 5

Verified: `opencv_contrib/4.x/modules/stereo/include/opencv2/stereo/quasi_dense_stereo.hpp` **exists**; the same path on `5.x` **404s**, and **the entire contrib `stereo` module directory is gone** in 5.x. It is not in main `5.x` either.

**If any part of the plan depends on quasi-dense stereo, that plan is already broken.** Options: vendor the 4.x sources (Apache-2.0, so this is fine), reimplement Lhuillier–Quan seed-and-grow propagation, or drop it.

**Verified OpenCV 5.x module reality** (branch reports `CV_VERSION 5.1.0-dev`; both repos Apache-2.0):

| Path | Status |
|---|---|
| main: `calib`, `features`, `geometry`, `stereo`, `ptcloud`, `photo` | ✅ present — renames confirmed |
| main: `calib3d`, `features2d`, `3d`, `xphoto` | ❌ **absent from main** |
| `calib/include/opencv2/calib3d.hpp` | ⚠️ **compat header survives** — `#include <opencv2/calib3d.hpp>` may still compile. Don't let that mislead you about where code lives. |
| contrib: `ximgproc`, `xphoto`, `ccalib`, `optflow`, `rgbd`, `sfm`, `xfeatures2d`, `line_descriptor` | ✅ present |
| contrib: **`stereo`** | ❌ **REMOVED** (was in 4.x) |
| contrib: `aruco` | ❌ removed (folded into main `objdetect` in 4.7+) |

**Confirmed useful:** `stereo` — `StereoBM`, `StereoSGBM` (`MODE_SGBM_3WAY`, `MODE_HH4`), `stereoRectify`, `reprojectImageTo3D`, **and `cv::fisheye::stereoRectify`** (fisheye stereo rectification lives in `stereo`, not `calib`). `calib` — `calibrateCamera`, `calibrateCameraRO`, `stereoCalibrate`, `solvePnP`, plus the `fisheye` namespace. `ximgproc` — `guidedFilter`, `dtFilter`, `amFilter`, `fastGlobalSmootherFilter`, **`createDisparityWLSFilter`**. `ccalib` — **`omnidir`** (Mei/UCM omnidirectional model with its own calibration/undistortion/stereo). `xphoto` — `SimpleWB`, `GrayworldWB`, `LearningBasedWB`.

> ⚠️ **Two traps in the pinned OpenCV 5.0.0 / contrib (found 2026-10-19, 2.1 D61; `docs/design/PERCEPTION_2.1.md` §10):**
> - `ximgproc::createDisparityWLSFilter(matcher)` **mutates the matcher passed in**: uniqueness → 0, `disp12MaxDiff` →
>   1e6, speckle window → 0 (`disparity_filters.cpp`). To keep an existing SGBM's output, use
>   `createDisparityWLSFilterGeneric(true)` + `createRightMatcher(sgbm)` (a new matcher) and set the ROI and radius
>   by hand.
> - `StereoSGBM` clamps a **non-positive `disp12MaxDiff` to 1** (`stereo/src/stereosgbm.cpp`: `> 0 ? v : 1`), the
>   strictest check, not "disabled". A large value disables it.

### 10.2 ⭐⭐ The headlamp physics — why every standard method breaks

**The daylight baseline (Koschmieder):** `I_c = J_c·t + A_c·(1−t)`, `t = e^{−β_c·d}`. Two assumptions do all the work, **and both fail for a headlamp**: `A_c` is a *global constant* (valid because sunlight fills the medium uniformly), and attenuation is *single-path* over the object→camera path only.

**With a co-located lamp:**

**Direct term** — light travels *out and back*, so the exponent **doubles**, plus inverse-square falloff and the lamp's angular profile:
```
D_c(x) = ρ_c(x) · [ I_L(θ,φ)·cos θ_i / d(x)² ] · e^{−2·β_c·d(x)}
```
**Backscatter term** — an *integral along the line of sight*, not a constant:
```
B_c(x) = ∫₀^{d(x)}  [ I_L(θ,φ) / r² ] · β_{b,c} · P(π) · e^{−2·β_c·r}  dr
```

**Four consequences, and they are the whole ballgame:**
1. **The `1/r²` makes the integral near-field dominated.** Dust 10 cm from the lens is enormously brighter than dust at 2 m. Backscatter is a bright veiling lobe that **saturates within tens of centimetres** and is nearly independent of scene depth beyond that. **This is why headlamp-in-dust looks like a glowing wall, not gentle depth-graded haze.**
2. **`A_c → A_c(x)`** — spatially structured, peaked on the beam axis. **Every method estimating a single global airlight is *structurally* wrong here.**
3. **⚠️ THE DEPTH CUE INVERTS.** In daylight haze, more haze = more distant — which is what DCP exploits. Under a headlamp the strongest veiling is *nearest* the camera. **The sign flips. Any method inferring depth from haze density produces inverted depth in dust.**
4. **Attenuation doubles** (`2βd`) and the `1/d²` falloff means distant objects are dark *and* washed out by near-field glow — a brutal dynamic-range problem.

> ### ⭐⭐ The single highest-value action in this entire research corpus is hardware, not software.
> **Move the lamp off the optical axis.** Backscatter is proportional to the overlap between the illumination cone and the camera FOV *in the near field*. Displacing the lamp 20–40 cm, and/or toeing it in so cone and FOV only intersect beyond ~1 m, **removes most of the near-field lobe geometrically.** No algorithm can recover what a saturated pixel destroyed. This is standard practice in underwater imaging and cave photography.
>
> **Treat lamp placement as a first-class design variable in the simulator and A/B it before writing any dehazing code.** It is also a cheap, legible, and genuinely novel experiment for the report — a lamp-offset sweep against reconstruction quality is a figure nobody else will have.

### 10.3 ⭐ Sea-thru — the best-matched method in the literature

**Akkaynak & Treibitz, CVPR 2019** ([PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf)). ⚠️ No official code; `github.com/hainh/sea-thru` is a **third-party** MIT reimplementation.

**It requires a known range map as input** — verbatim: *"it differs in the fundamental way that we utilize a known range map, rather than try to estimate it."* **For Ariadne this is a feature, not a blocker — you have stereo.** This is the hinge connecting the dust work to the confined-space stereo work.

**Mechanism, in reimplementable detail:**
1. **Backscatter estimation.** Partition the range map into 10 evenly-spaced `z` clusters. Within each, take RGB triplets in the **bottom 1st percentile** — *darkest triplets, not per-channel minima; explicitly unlike DCP, and no dark-channel image is formed.* Those pixels satisfy `I_c ≈ B_c`. Fit by NLS:
   ```
   B̂_c = B_c^∞·(1 − e^{−β_c^B·z}) + J'_c·e^{−β_c^{D'}·z}
   bounds: B_c^∞∈[0,1], β_c^B∈[0,5], J'_c∈[0,1], β_c^{D'}∈[0,5]
   ```
   The second term accounts for dark pixels that aren't perfectly black.
2. **Attenuation.** After subtracting backscatter (`D_c = I_c − B_c`), recovering `J_c` becomes *spatially-varying illuminant estimation*: `β̂_c^D(z) = −log Ê_c(z)/z`, estimated via a range-aware **Local Space Average Color** iteration whose 4-connected neighbourhood is **gated by range similarity** `‖z(x,y) − z(x',y')‖ ≤ ε`. **That range gate is what stops the illuminant map bleeding across depth discontinuities.**

**Difficulty: Medium.** A percentile search, a 4-parameter LM fit per channel, and iterative local averaging. The hard parts: **(a) it needs *linear* radiometric data — shoot RAW or carefully invert the tone curve, and disable auto-exposure/AWB**; (b) LSAC's `p` and `ε` need tuning on real depth-discontinuous scenes.
**OpenCV 5 mapping:** range map from `stereo::StereoSGBM`; NLS via `cv::LMSolver` (in `calib`) or hand-rolled; LSAC as a masked `imgproc::boxFilter`, or `ximgproc::guidedFilter`/`dtFilter` with the range map as guide for a faster edge-aware variant.

> **Fit: HIGH — make this the backbone of the dust pipeline.** It is built for exactly Ariadne's situation: artificial light, a scattering medium, an available range map. **Caveat and opportunity: it was designed for near-ambient underwater scenes, so the `1/r²` near-field lobe from a headlamp is not in its model. Adding point-source falloff to the backscatter fit is a genuine, honestly publishable extension rather than a reimplementation. Budget 2 weeks, not 1.**

**The theoretical keystone: Akkaynak & Treibitz, "A Revised Underwater Image Formation Model", CVPR 2018.** The standard model is **wrong because it uses one `β_c` for two physically different processes.** The revised model splits them:
```
I_c = J_c·e^{−β_c^D(v_D)·z} + B_c^∞·(1 − e^{−β_c^B(v_B)·z})
v_D = {z, ρ, E, S_c, β}     v_B = {E, S_c, b, β}
```
The *direct-signal* coefficient depends on range **and scene reflectance ρ**; the *backscatter* coefficient doesn't depend on ρ at all. Their error analysis states plainly that **"estimating coefficients from color charts that are close to the camera, as is often done, is a bad habit."**
> **This is not water-specific — it follows from spectral integration over a non-delta camera response, which applies equally to a broadband LED in grey dust. If Ariadne implements any physical dust model, carry two coefficients from day one. Fitting one β and expecting it to work at both 0.5 m and 3 m is the mistake this paper exists to prevent.**

### 10.4 ⚠️ Dark Channel Prior fails on grey dust — three independent ways

**He, Sun, Tang** (CVPR 2009 / TPAMI 2011). `t̃(x) = 1 − ω·min_Ω min_c [I^c(y)/A^c]`, ω=0.95, 15×15 patches; `A` from the brightest 0.1% of dark-channel pixels; recovery `J = (I−A)/max(t,t₀) + A`, t₀=0.1. Refinement: soft matting (2009) superseded by **guided filtering** (ECCV 2010, O(N), ~100× faster, near-identical).

1. **White/grey content violates the prior at the source.** DCP's known failure mode is exactly bright, low-saturation objects. For pale dust on grey concrete, `min_c min_Ω I^c` is *large*, so `t̃ → small`, so it concludes "very hazy/very distant" and divides by a tiny t → over-amplification, colour artifacts, heavy darkening of **exactly the surfaces Ariadne needs**. **Ariadne's scenes are near the DCP worst case by construction.**
2. **The airlight estimate locks onto the headlamp.** `A` comes from the brightest dark-channel pixels — under a co-located lamp that's the near-field backscatter lobe. DCP will characterize the glow 15 cm from the lens, then apply it globally.
3. **The depth cue is inverted** (§10.2). The transmission map will be roughly *anti-correlated* with true depth.

> **Verdict: implement DCP as a one-week calibrated baseline you *expect to fail*, not as the shipping algorithm.** It's cheap, it gives the team the transmission/airlight machinery and guided filter they'll reuse in Sea-thru, and **having it produce visibly bad output on real dust footage is the most persuasive possible justification for the Sea-thru route** — which is exactly the kind of negative result the competition's evaluation rubric rewards.
> **OpenCV 5:** `imgproc::erode` with a rectangular kernel *is* the patch min-filter (per-channel, then `cv::min` across channels); `ximgproc::guidedFilter` verified present in contrib 5.x.

### 10.5 ⛔ Modern dehazing successors — why almost all are wrong here

> **The decisive constraint: there is no dust training data, and there never will be. Every supervised dehazing network is trained on RESIDE/NYU-derived *synthetic haze generated with the Koschmieder model* — the exact model §10.2 shows is wrong for a headlamp. These networks have learned the daylight-haze prior, including the inverted depth cue. Fine-tuning is impossible without ground-truth dust-free pairs.**

**Don't bother:** DehazeNet (TIP 2016) · AOD-Net (ICCV 2017) · FFA-Net (AAAI 2020) · Dehamer (CVPR 2022) · DehazeFormer (TIP 2022) · GridDehazeNet · MSBDN · AECR-Net · GCANet · RefineDNet. ⚠️ **Eight of these repos have NO LICENSE file at all** — all rights reserved for code *and* weights. Only DehazeFormer is MIT (weights ship separately with no stated licence). Dehamer's "MIT" LICENSE is **"Copyright (c) 2018 Joey Litalien"** — a file copied from an unrelated project.

**Worth reading, not shipping:** **Zero-shot/unsupervised — ZID, YOLY, Double-DIP.** *Conceptually* the right family: they fit an untrained network per-image (Deep Image Prior style), decomposing into scene/transmission/airlight layers **with no training set**, sidestepping the data problem. But the decomposition is still constant-`A` Koschmieder, so they inherit the physical error, and per-image optimization takes seconds-to-minutes. ⚠️ All three: no LICENSE file.
> **Read YOLY/ZID for the layer-decomposition trick. That idea, re-derived against the *headlamp* model, is the most interesting research direction in this report if Ariadne wants a novel contribution.**

**Non-Local Dehazing / Haze-Lines** (Berman, Treibitz, Avidan, CVPR 2016 / TPAMI 2020) — genuinely the most interesting classical alternative: a haze-free image is well approximated by a few hundred colour clusters; under haze each stretches into a *line* in RGB toward `A`, and position along the line gives transmission. **More robust than DCP on white objects because it doesn't need dark pixels.** But still assumes global constant `A`, and:
> ⚠️⚠️ **DOUBLY ENCUMBERED. Code is CC BY-NC** (verbatim: *"for Non Commercial purposes only"*) **AND there is a live US patent family: US10885611B2 (granted 2021-01-05) and US11810272B2 (granted 2023-11-07), "Image dehazing and restoration", assignee Carmel Haifa University Economic Corporation Ltd.** — University of Haifa's commercialization arm, Treibitz's institution. **Reimplementation dodges the licence but NOT the patent. Flag to whoever owns IP risk.** (Inventorship unverified — Google Patents blocked the detail page.) Also verified: US9361670B2 (hybrid DCP, National Taipei Univ. of Technology, 2016). ⚠️ **A Microsoft/He-era DCP patent could not be verified — searches returned 503s and noise. Treat as unresolved, NOT as "no patent exists."** Such a filing would still be in force to ~2029–2030.

**Sand-dust enhancement literature: exists, sizeable, and does not transfer.** These target *desert sand*, whose signature is a strong **yellow/red colour cast** (sand absorbs blue). The algorithms are essentially colour-cast correction — compensate the attenuated blue channel, rebalance, guided-filter, then DCP. **Ariadne's dust is pulverized concrete and gypsum: grey/white, spectrally flat. There is little cast to correct, and blue-channel compensation would *inject* a false cast into neutral dust.** The one transferable thing is the pipeline *shape* — white-balance → guided-filter refinement → physical inversion — which `xphoto`'s WB classes support directly.

**LiDAR in dust — the key reference:** Phillips, Guenther, McAree, **"When the Dust Settles: The Four Behaviors of LiDAR in the Presence of Fine Airborne Particulates"**, *J. Field Robotics* 2017, doi:`10.1002/rob.21701`. Characterizes returns into four distinct behaviours rather than treating dust as uniform noise — **essential reading before fixing the sensor mix.** Also: Parsons et al., *"RGB-LiDAR sensor fusion for dust de-filtering"*, Automation in Construction 2024, doi:`10.1016/j.autcon.2024.105850`.

> **⚠️ SubT's consistent field lesson across teams (CERBERUS, CSIRO Data61, and the Circuits Stage synopsis, all *Field Robotics* 2022) was sensor redundancy and degradation-aware fusion — NOT dust-specific image restoration. Weigh that against spending two months on dehazing.**
>
> **Bottom line: there is no ready-made "grey concrete dust removal" literature to reimplement. The gap is real — which is precisely why Sea-thru-with-a-point-source is both the most defensible path and genuinely novel territory.**

### 10.6 Polarization — small hardware ask, real physics, decide now

Schechner/Narasimhan/Nayar (CVPR 2001; *Applied Optics* 2003) · **Treibitz & Schechner, "Active Polarization Descattering", TPAMI 2009** ← the one that matters · and their sceptical counterweight, *"Polarization: Beneficial for Visibility Enhancement?"* (CVPR 2009) — **read this before buying anything.**

**Mechanism:** backscatter is **partially linearly polarized**; diffuse object reflection largely is not. Two images at best/worst polarizer angles give `p = (I_max − I_min)/(I_max + I_min)`, isolating the scattered component and yielding airlight directly — **with no prior on scene content whatsoever.** *That is the crucial property: it does not care that the dust and the walls are both grey.* It fails where DCP fails for entirely different reasons, making it complementary rather than redundant. **Active polarization descattering** polarizes the *lamp* too — backscatter from near-field particles largely preserves source polarization, so a crossed analyzer suppresses it strongly, attacking precisely the near-field lobe of §10.2.

**Hardware:** a **polarization camera with a per-pixel micropolarizer array** (Sony IMX250MZR — captures 0/45/90/135° in one shot, no moving parts, no motion artifact) plus polarizing film over the LED. Cost: **~50%+ light loss**, a real penalty in a dark building. **Difficulty: Low-Medium** — a few per-pixel algebraic ops, far simpler than anything else here.
> ⚠️ **UNVERIFIED: whether concrete dust polarizes backscatter strongly enough to be useful.** No DOP measurements found for concrete dust under LED. **If you can borrow a polarization camera for one afternoon of dust-box testing, that experiment is worth more than a month of algorithm work — but it's a procurement lead-time decision, so raise it now or not at all.**

### 10.7 Confined-space stereo — the arithmetic, worked

`δZ = Z²·δd/(f·B)`. Depth error grows **quadratically with range**, inversely with baseline — so **at close range a short baseline is far less punishing than intuition suggests.**

**1280×800, ~90° HFOV (f ≈ 640 px), δd = 0.25 px** (realistic for SGBM with subpixel on decent texture):

| Baseline | Z=1 m | Z=2 m | Z=3 m | Z=5 m |
|---|---|---|---|---|
| **5 cm** | 7.8 mm | 31 mm | 70 mm | 195 mm |
| **8 cm** | 4.9 mm | 20 mm | 44 mm | 122 mm |
| **12 cm** | 3.3 mm | 13 mm | 29 mm | 81 mm |
| **20 cm** | 2.0 mm | 7.8 mm | 18 mm | 49 mm |

At δd = 1.0 px (the pessimistic dust/low-texture case), **multiply everything by 4.**

> **The load-bearing conclusion: in a 1–3 m corridor, even a 5 cm baseline is fine. Baseline is NOT the binding constraint at these ranges — matching quality in dust is.** The δd=1.0 column matters more than the baseline column. **Effort spent on radiometric preprocessing and WLS disparity filtering buys more than effort spent enlarging the rig.**

**⚠️ Minimum range is the real short-baseline trap**, set by `Z_min = f·B/N_disp`:

| | N=64 | N=128 | N=192 | N=256 |
|---|---|---|---|---|
| **5 cm** | 0.50 m | 0.25 m | 0.17 m | 0.12 m |
| **8 cm** | 0.80 m | 0.40 m | 0.27 m | 0.20 m |
| **12 cm** | 1.20 m | 0.60 m | 0.40 m | 0.30 m |
| **20 cm** | 2.00 m | 1.00 m | 0.67 m | 0.50 m |

**A 12 cm baseline with the common `numDisparities=64` is blind inside 1.2 m — fatal for a robot squeezing through rubble.** (`numDisparities` must be a multiple of 16; cost and memory scale linearly.)

> **⭐ Recommendation: 8 cm baseline, `numDisparities` 128–192, `StereoSGBM` (`MODE_SGBM_3WAY` for speed / `MODE_HH4` for quality), plus `ximgproc::createDisparityWLSFilter` with a left-right consistency check.** → ~0.4 m minimum range, ~5 mm at 1 m, ~44 mm at 3 m.
> **⚠️ And the headlamp bites here too: a moving light source violates stereo photometric consistency if the two cameras see different specular/backscatter patterns. Mount the lamp symmetrically between the cameras, or use census/rank transforms — census is inherently robust to monotonic intensity differences, which is exactly this failure mode.** (`PREFILTER_XSOBEL` and SGBM's census-like matching help.)

### 10.8 Wide-FOV models and field calibration

| Model | Params | Notes |
|---|---|---|
| UCM (Mei & Rives, ICRA 2007) | 5 | what OpenCV's `ccalib/omnidir` implements |
| EUCM (RA-L 2016) | 6 | ellipsoid projection; closed-form inverse |
| Kannala–Brandt (TPAMI 2006) | 6–8 | **this is `cv::fisheye` (KB4)**; inverse needs iterative root-finding |
| **Double Sphere** (3DV 2018, [arXiv:1807.08957](https://arxiv.org/abs/1807.08957)) | 6 | **closed-form inverse, NO transcendental functions in projection or unprojection**, analytic Jacobians |

**Double Sphere's selling point is engineering, not accuracy** — in a VO/BA inner loop projecting hundreds of thousands of points per second, avoiding iterative θ inversion is the whole argument. **Difficulty: Low** (four lines of algebra plus a domain check; calibration is the actual work).

> **⚠️ Rectifying a fisheye to a pinhole plane destroys the FOV you paid for** — a 180° lens can't be rectified to a plane at all, and even 140° wastes most of the sensor on stretched corners with badly non-uniform effective resolution. **In a 1 m corridor, wide FOV is precisely what keeps features in view.** Better: keep the fisheye model and do matching/VO **directly with the camera model**, never forming a rectified image; or rectify to a **cylindrical/spherical** target. For dense stereo you *do* need rectified epipolar geometry — `cv::fisheye::stereoRectify` (in `stereo` 5.x) or `omnidir::stereoRectify` (`ccalib`), accepting the crop.
>
> **⚠️ Fisheye breaks the local-affine assumption underlying patch descriptors** — a fixed pixel patch subtends wildly different solid angles at centre vs periphery, so ORB/SIFT degrade toward the rim. Mitigate by detecting on the raw image but *describing* on locally-undistorted patches, or restricting usable radius. **Direct/photometric methods handle fisheye more gracefully** because they work per-pixel with the true model — which is why Basalt and Omni-DSO use DS/EUCM natively.

**Toolboxes:** **Basalt** — `gitlab.com/VladyslavUsenko/basalt` (**GitLab, not GitHub**), ✅ **BSD-3 verified**, the reference DS implementation. **Recommended.** ⚠️ **Kalibr is BSD-*4*-Clause, not 3** — verbatim: *"All advertising materials... must display the following acknowledgement..."*. The advertising clause is a genuine obligation and is **GPL-incompatible.** Fine internally; know it before it ships.

**Field calibration — what actually breaks, in order:** (1) **stereo extrinsics** — a hard landing or thermal cycling shifts baseline/rotation a fraction of a degree, enough to ruin disparity, and **rotation error is far more damaging than translation**; (2) focus/zoom shift; (3) **lens contamination — dust on the lens, a near-certainty here**, not a calibration problem but degrades everything; (4) rolling-shutter and camera-IMU time-offset drift.
> **The realistic answer for a robot that cannot be recalibrated: don't re-estimate intrinsics in the field; make extrinsics a live state.** Standard modern VIO practice (VINS-Mono TRO 2018; ORB-SLAM3 TRO 2021).
> **⚠️ But online self-calibration needs *excitation* — rotation and translation about multiple axes to make parameters observable. A robot creeping down a straight corridor provides almost none, so extrinsics drift unobserved and the estimator quietly absorbs the error into pose.** → **Calibrate thoroughly before entry; hold intrinsics fixed; estimate extrinsics online with tight priors and an observability check that freezes them when excitation is low; and add a lens-contamination detector (blur/contrast monitor), because dusty optics masquerade as calibration drift.**

### 10.9 ⭐ Corridor degeneracy — detect it, don't let it slide

> **Title correction before anyone searches: the paper is "On Degeneracy of Optimization-Based State Estimation *Problems*" — Zhang, Kaess, Singh, ICRA 2016, doi:`10.1109/icra.2016.7487211`.**

**Mechanism.** You solve `H·δx = b` with `H = JᵀJ`. Geometric degeneracy means `H` is **ill-conditioned in specific directions** — a long featureless corridor constrains lateral and rotational motion well but leaves *translation along the corridor axis* nearly unconstrained, so `H` has a small eigenvalue whose **eigenvector points down the hallway.**

> **The damage is subtle and that is what makes it dangerous: the solver doesn't fail, it *confidently slides* along that direction, and the error enters the state as if it were a measurement.**

The fix is **eigenvalue analysis of `H`** plus **solution remapping** — solve only in the well-conditioned subspace, projecting out components along degenerate eigenvectors rather than letting noise fill them. **This is far better than naive Levenberg–Marquardt damping, because it distinguishes *which* directions are unobservable instead of shrinking everything.**

> **⭐ Apply this — it is cheap and high-value. You already form `H` in any BA or VIO backend; computing its smallest eigenvalues and gating updates costs almost nothing, and it converts a silent catastrophic failure into a detectable, reportable condition: *"I cannot localize along this corridor — fall back to IMU/wheel odometry."* Difficulty: Medium, mostly integration.** Note this composes directly with the gauge null-space machinery of §7.3 — same eigen-analysis, different null space.

**The visual equivalent compounds it:** blank walls give **no parallax and few features**, and what features exist sit on repeating structures (identical doors, tiles) generating false matches — i.e. §8.2's perceptual aliasing, arriving through a second door. **The standard remedy is to abandon point features for lines and planes**, which textureless man-made corridors have in abundance. **OpenCV 5 contrib retains `line_descriptor` (verified present)** — LSD/BD line detection and matching, a directly usable primitive.

**Motion constraints:** for the **ground robot**, imposing planar/Ackermann motion (SE(2) rather than SE(3)) removes three DoF and materially stabilizes VO; the nonholonomic **1-point RANSAC** case reduces the minimal solver to a single correspondence, dramatically more robust when inliers are scarce (Scaramuzza, *Visual Odometry Part II*, RA-M 2012). **⚠️ Do NOT apply this to the drone.**

### 10.10 Verdicts

**Do:**
1. **Move the headlamp off-axis and test it before writing any dehazing code.** Highest value-per-effort item in the corpus (§10.2).
2. **Sea-thru re-derived for a point source**, using stereo depth as the range map (§10.3). The backbone, ~2 weeks, and the one place Ariadne could make a real contribution.
3. **DCP + guided filter as a one-week baseline you expect to fail**, to justify (2) with evidence (§10.4).
4. **8 cm baseline, `numDisparities` 128–192, SGBM + WLS filter** (§10.7). Invest in matching robustness, not baseline length.
5. **Degeneracy detection via Hessian eigenvalue analysis** + line features for corridors (§10.9).
6. **Price a polarization camera now** and run one afternoon of dust-box tests (§10.6) — procurement lead time forces the decision early.

**Don't bother:** every supervised dehazing network (wrong prior, no data, unlicensed code) · sand-dust colour-cast methods (wrong colour of dust) · structured-light descattering (needs a projector) · rectifying fisheye to a plane in a 1 m corridor.

**Two things that bite if ignored:** **`quasi_dense_stereo` does not exist in OpenCV 5** (§10.1) · **Non-Local Dehazing is CC BY-NC *and* covered by a live patent family** (§10.5).

## 11. ⭐⭐ Burst imaging & temporal accumulation — the closest published work to Ariadne

**This section contains the single most on-point paper line found in the entire research effort.** All facts verified by direct primary-source fetch (arXiv API, dblp, Crossref, paper PDFs).

### 11.1 The blueprint: Burst Imaging for Light-Constrained Structure-from-Motion

**Ravendran, Bryson, Dansereau (Sydney Robotic Imaging), IEEE RA-L 7:1040–1047, 2022** · [arXiv:2108.09895](https://arxiv.org/abs/2108.09895) · doi:`10.1109/LRA.2021.3137520` · code + dataset at [roboticimaging.org/Projects/BurstSfM](https://roboticimaging.org/Projects/BurstSfM/)
*(⚠️ The repo README says "our ICRA2021 paper" — trust the DOI: RA-L vol 7, 2022. This group's READMEs consistently cite submission targets rather than final venues.)*

**This is burst imaging for 3D reconstruction on a moving robot in the dark. It is Ariadne's front-end, already published.**

The method is deliberately unoriginal, which is exactly why it's useful: capture bursts of **N = 7** short-exposure frames → **HDR+ hierarchical tile alignment** → **HDR+ robust frequency-domain merge** → feed **one merged image per burst** to COLMAP.

**Verified results:** burst-with-merge **22 images, 3.25 min/scene** vs. burst-without-merge **154 images, 49.58 min/scene** (~15×), **plus a lower non-convergence rate.** On synthetic data at σ=0.03, merged bursts yield **no spurious features**, while single frames miss true features and unmerged bursts are *"overwhelmed by spurious features."*

**Two details to copy verbatim:**
- **⭐ Merge to the MIDDLE frame of the burst, not the first.** Their caption: *"merge images in a burst ideally to the middle frame for robotic applications, to address for the relative motion between frames."* **This halves the maximum displacement any tile must search** — on a translating robot, the difference between converging and not. (Night Sight picks the *sharpest* frame; on a robot, motion is smooth and predictable, so "middle" wins.)
- **They evaluate on feature quality and pose accuracy, not PSNR** — true positives, spurious features, inlier matches, registered images, convergence rate. **Adopt this metric set wholesale; it is also what the competition rubric rewards.**

**The architectural argument, directly transferable:** *direct* (intensity-based) registration works **within** a burst because camera motion is tiny and exposure/illumination are constant over the burst; *feature-based* SfM handles the **large** motions and illumination changes **between** bursts. **Two regimes, two tools.** This is also the clean answer to Ariadne's moving-light problem (§11.5).

**Difficulty: Low, given §11.2.** This paper *is* HDR+ plus COLMAP. **Reproduce it first, then extend.**

### 11.2 HDR+ — the core to build

**Hasinoff et al., ACM TOG 35(6), SIGGRAPH Asia 2016** · [paper](https://hdrplusdata.org/hdrplus.pdf) · dataset **CC-BY-SA** · no official code; `timothybrooks/hdr-plus` (Halide, **MIT**) and `martin-marek/hdr-plus-pytorch` (**MIT**) verified.

**Capture protocol — constant exposure, deliberately underexposed, NO bracketing.** Verbatim: *"we do not bracket… Small exposure variation may compromise alignment due to differing levels of noise and motion blur, and large variation may render local alignment impossible."* Capped at **8× dynamic-range compression**. Their framing: *"we treat HDR imaging as denoising."*

**Alignment (§4):** average 2×2 Bayer blocks → align a 3 Mpix grayscale proxy; displacements constrained to **multiples of 2 px** so Bayer samples stay coincident, with the resulting aliasing *deliberately deferred to the merge*. **4-level Gaussian pyramids**, coarse-to-fine, tile-based, `D_p = Σ|T(x,y) − I(x+u+u₀, y+v+v₀)|^p`. **Split norms by scale:** coarse levels use **L2, large search radius, sub-pixel** (FFT-accelerated: `D₂ = ‖T‖² + box(I∘I,n) − 2·F⁻¹{F{I}*∘F{T}}`, sub-pixel from a **bivariate quadratic fit to the 3×3 window**); the finest level uses **L1, small radius, integer pixel only.** Pyramid upsampling evaluates **3 hypotheses** (nearest coarse tiles), keeping min-L1 — this survives motion boundaries. Handles displacements up to **169 px**.

**⭐ The robust pairwise merge (§5) — the part that matters most.** Tiles **16×16** (or **32×32** in very dark scenes), **half-overlapped**, windowed by `½ − ½cos(2π(x+½)/n)` — note denominator `n`, not `n−1`, and the half-shift, so overlapping windows **sum exactly to 1** with no zeros.
```
T̃₀(ω) = (1/N) Σ_z [ T_z(ω) + A_z(ω)·( T₀(ω) − T_z(ω) ) ]
A_z(ω) = |D_z(ω)|² / ( |D_z(ω)|² + c·σ² ),   D_z(ω) = T₀(ω) − T_z(ω)
```
A per-frequency Wiener shrinkage. `c` bundles the DFT scaling `n²`, the window factor `1/4²`, a factor 2 (D_z is a difference of two tiles), and a **hand-tuned factor of 8**.

> ### ⭐ Why this is the right choice for a safety-critical robot
> - Frame well aligned ⇒ `D_z` is pure noise ⇒ **A_z → 0** ⇒ plain temporal average ⇒ √N noise reduction.
> - Frame badly aligned ⇒ `|D_z|² ≫ cσ²` ⇒ **A_z → 1** ⇒ the reference tile. Summed over N frames, output → `T₀` exactly.
>
> **The worst case of the merge is "you get the reference frame back." It can never be worse than not merging.** For a system that must not produce hallucinated geometry, that is a hard safety guarantee — and it is why you chase *robust merging* rather than *better alignment*.

**Why frequency domain and not pixel domain — five defensible reasons:**
1. **Alignment failure is partial, and the DFT lets you accept it partially.** A misaligned tile usually still agrees at low frequencies and disagrees at high. A per-pixel robust weight must make an all-or-nothing call; per-frequency shrinkage keeps the usable half. Their Fig. 6d shows features **partially denoised despite alignment failure.**
2. **Sub-pixel misalignment is a phase ramp**, so `|D_z|` grows monotonically with frequency — the shrinkage automatically merges low frequencies and rejects high ones, **with no explicit sub-pixel estimation.** This is why integer-pixel alignment suffices at the finest level.
3. **Noise is white; signal is sparse.** Per-bin SNR is wildly non-uniform — exactly where Wiener shrinkage is near-optimal.
4. Aliasing from the 2-px Bayer constraint is **confined to a subset of bins** and rejected bin-wise.
5. **It composes with spatial denoising for free** — one DFT, two filters.

**Difficulty: Medium**, ~1.5 weeks. The merge is genuinely easy (a few hundred lines). The pyramid alignment with multi-hypothesis upsampling is where time goes. **The hard prerequisites: (a) a calibrated per-ISO noise model σ²(signal) for your actual sensor — without it `c·σ²` is meaningless and the whole thing collapses to either no-merge or ghosting; (b) tile/window bookkeeping so overlaps sum to 1; (c) staying in linear raw, never gamma-corrected.**

**OpenCV 5:** `core::dft`/`idft`/`mulSpectrums` · `imgproc::phaseCorrelate` (**a ready-made FFT translation estimator with sub-pixel peak fit — a legitimate shortcut for coarse levels**) · `imgproc::matchTemplate` (`TM_SQDIFF`) · `imgproc::buildPyramid` · `imgproc::boxFilter` · `video::findTransformECC` for optional global pre-align.

### 11.3 Night Sight — the deltas worth taking

**Liba et al., ACM TOG 38(6), SIGGRAPH Asia 2019** · [arXiv:1910.11336](https://arxiv.org/abs/1910.11336). Works to **0.3 lux**, 6–13 frames.

**"Bounded Flow" motion metering** — from `I_t = ∇I·v`, Cauchy–Schwarz gives a **lower bound on flow magnitude: ‖v‖ ≥ |ΔI| / ‖∇I‖**. One divide per pixel, **no iteration, no solver.** They only need the magnitude, because *blur direction doesn't affect image quality*. Computed on **linear** frames; pixels discarded where `‖∇I‖` is small relative to σ; downsample taking the **90th percentile per bin**; final motion map is only **16×12**.

**⭐ Mismatch maps (§3.2)** — a per-tile, per-frame **spatial-domain L1 difference**, shrunk into [0,1], **reusing the L1 residuals already computed during alignment (free)**. HDR+'s constant `c` becomes **spatially varying**: `c → c·s(tile, frame)`. Static regions merge hard; dynamic/misaligned regions merge weakly.
*Why both, when both measure tile differences?* **The DFT-domain `D_z` looks within one frequency band inside one tile; the mismatch map is spatial, across all bands, with larger support** — so it catches alignment error whose signature is spread out. Complementary, not redundant.
Also: **take the minimum weight across colour channels** and apply to all (per-channel weights diverge in low light and cause colour artifacts). And **compute post-merge noise variance from how much merging actually happened per tile**, not the idealized σ²/N, so tiles that merged little automatically get stronger spatial shrinkage.

**Difficulty: Low–Medium given HDR+.** Bounded Flow ~30 lines; mismatch maps ~50 lines on existing residuals.
> **Fit: high, and partly MORE applicable to Ariadne than to phones. You have a better IMU than a phone, you control the trigger, and you know your own commanded velocity — so you can compute predicted apparent motion *analytically* rather than estimating it from viewfinder frames. Take the mismatch maps and the blur-limiting exposure rule; skip the GMM predictor.**

### 11.4 🚨 The SNR arithmetic that decides your capture config

Signal S electrons per short frame, shot noise √S, read noise σ_r:
```
N short frames, merged:  SNR = √N · S/√(S + σ_r²)
One long frame (N×):     SNR = N·S/√(N·S + σ_r²)
ratio (long/burst) = √( (N·S + N·σ_r²) / (N·S + σ_r²) )  ≥ 1  ALWAYS
```
- **Shot-noise-limited (S ≫ σ_r²):** ratio → 1. **Bursting is free.**
- **Read-noise-limited (S ≪ σ_r²) — i.e. very dark, Ariadne's regime:** ratio → **√N.** **A single long exposure is up to √N times better**, because you pay read noise once instead of N times.

> **⚠️ Bursting is NOT free in the dark. You pay up to √N in SNR, and you pay it to buy motion-blur immunity, HDR headroom and robustness. This is usually left implicit; state it explicitly in the report — it is exactly the kind of quantified design tradeoff that reads as depth.**

**Three consequences:**
1. **Sensor selection dominates.** Choose low read noise (modern CMOS ~1–3 e⁻; avoid ≫5 e⁻). **This is why burst photography works on phones now and didn't in 2005. A high-read-noise sensor makes the entire burst pipeline counterproductive.**
2. **Don't over-split. Use N = 8.** Both Sydney papers independently found saturation at **6–10 frames**. N = 30 costs ~2× more read noise for a benefit that has already saturated.
3. **Set exposure by the blur budget, then stop.** `t_max = b_max·Z/(f·v)` for translation, `t_max = b_max/(f·ω)` for rotation; take the min. Use **b_max ≈ 1–2 px** — tighter than Night Sight's 3–4, because your consumer is a keypoint detector, not an eye. Then take the *longest* exposure respecting it; shorter is pure read-noise loss with no benefit.

> **⭐⭐ The onboard-lights wrinkle — and why HDR+ specifically, not just "burst denoising."** Ariadne's lamps fall off as 1/r², so surfaces at 30 cm are **~100× brighter** than surfaces at 3 m. A single frame **simultaneously** has blown highlights on near rubble and read-noise-floor darkness in the far field. **This is not merely a low-light problem — it is a high-dynamic-range problem**, exactly what HDR+ was designed for and what a pure denoiser (KPN, VBM4D, FastDVDnet) does not address at all. **Follow HDR+'s protocol literally: constant exposure across the burst, deliberately underexposed to protect near-field highlights, dynamic range recovered by merging in the far field.** Clipped near-field pixels are permanently lost and produce no features.
> **You also control the illumination, which no camera-phone paper does — but do NOT bracket exposure time (it breaks alignment). Vary lamp current only if you keep it constant *within* a burst.**

### 11.5 ⭐ The moving-robot failure taxonomy

**⭐⭐ The IMU pre-warp — probably the highest-leverage Ariadne-specific idea in the corpus.**
Apparent motion from **rotation** is `f·ω·Δt` — **independent of depth**, and usually dominant. At f=500 px, ω=30°/s: **8.7 px/frame at 30 fps, ~60 px across a 7-frame burst.** From **translation**, `f·v·Δt/Z` — at v=0.5 m/s, Z=2 m: 4.2 px/frame, ~25 px.
> **Pre-warp every burst frame by the gyro-derived rotation homography `H = K·R(Δt)·K⁻¹` before tile alignment.** Nearly free, **uses no image data so it cannot fail in the dark**, removes the depth-independent and typically largest motion component, and leaves the tile search to handle only the **parallax** — the part that genuinely needs to be local. Night Sight uses the gyro only for stability assessment; on a robot you can use it for actual warping, which is strictly more powerful.

| Failure | Why it breaks | Mitigation |
|---|---|---|
| **Parallax, non-planar scenes** | `d = f·B/Z` — every depth gets a different warp. **No global model can be correct** for a translating robot in rubble, the maximally non-planar case. | Small tiles (16×16) + robust merge. ⚠️ **Tension: smaller tiles are more locally valid but noisier to match, and matching in the dark is already marginal.** HDR+/Night Sight resolve it by *increasing* tile size as noise rises — accept a worse motion model to get a reliable match, let A_z clean up. |
| **Occlusion / disocclusion** | **There is no correct warp — the information does not exist.** Any method returns *something*, and it's wrong. | **Only robust merging saves you.** A_z → 1, reference passes through un-merged. **This is precisely why "better alignment" is not the answer: the correct behaviour is to detect and abstain.** |
| **Rolling shutter** | Row *r* exposed at `t₀ + r·t_row` ⇒ a temporally **sheared** sample. Displacement becomes a function of row — no per-tile translation can express it. **Worse in low light**, since long exposures increase the fraction of frame in motion during readout. | **(a) Use a global-shutter sensor — a procurement decision and by far the cheapest fix.** (b) If stuck, estimate a per-row warp from the gyro and rectify before alignment. |
| **Motion blur within frames** | **Unrecoverable by merging.** Merging N blurred frames gives a cleaner blurred image. | Sets the hard upper bound on exposure — this is the entire reason burst photography exists (§11.4). |
| **⚠️ Ghosting** | In SfM, **ghosting is worse than noise**: noise gives *spurious but unrepeatable* features (rejected by RANSAC); ghosting gives **spurious AND repeatable** features — a doubled edge appears consistently and **will be triangulated into fake geometry.** | **Under-merging is far safer than over-merging for Ariadne. Tune `c` conservatively.** |
| **⭐ Illumination change from your own lights** | **Ariadne-specific, in none of these papers.** HDR+'s L1 tile matching is explicitly **not brightness-invariant** — fine for a phone because *"illumination is unlikely to change quickly over the duration of our bursts."* **On Ariadne, a rigidly-mounted lamp changes illumination on every surface between frames** (1/r² + changing incidence angle). | Keep bursts short (change scales with baseline); **keep lamp current constant — check for PWM dimming, which will alias with exposure and destroy you**; if significant, switch tile matching to **ZNCC** or gain-compensate per tile ⚠️ but note this weakens A_z, since a brightness offset then no longer registers as mismatch. **The clean answer is §11.1's: let feature-based SfM handle inter-burst illumination change; keep direct methods strictly intra-burst.** |

**Why robust merging beats better alignment — four independent arguments:** (1) for occlusion there *is* no correct alignment; (2) alignment quality is bounded by input SNR, which is the problem you're fixing — circular, and robust merging breaks the circle; (3) the cost asymmetry is enormous (A_z is ~20 lines; parallax-correct warping needs dense depth in the dark, a research project); (4) **direct empirical evidence — HDR+ recovers clean results where alignment demonstrably failed; KPN's annealed loss independently *rediscovers* the same weighting by gradient descent; Liu & Freeman reached it in 2010; Night Sight added a second complementary robustness mechanism rather than improving alignment. Four groups, same conclusion.**
> **Corollary: spend your alignment budget on *initialization* (pyramids, IMU pre-warp, middle-frame reference) — cheap and helps a lot — and nothing on *refinement*, which is expensive and helps little because the merge absorbs the residual.**

### 11.6 Robust temporal estimators

| Estimator | Breakdown | Gaussian efficiency | Verdict |
|---|---|---|---|
| Temporal **mean** | 0% | 100% | Ghosts. Only after robust gating |
| Temporal **median** | 50% | **~64% (2/π)** | ⚠️ **Worse than it looks: a 7-frame median gives roughly the noise reduction of a 4.5-frame mean.** You paid capture time for 7 and got 4.5. Debug baseline only |
| **⭐ Trimmed mean** (α≈0.2–0.3) | α | ~90%+ | **Best simple robust choice.** With N=7, drop min+max, average 5. Nearly free, **no noise model needed** |
| Winsorized mean | α | ~92% | Marginally better, same cost |
| **HDR+ A_z** | continuous | ~100% aligned | **Best overall.** Needs calibrated σ² |

> **Implement trimmed mean first (one afternoon, no noise model, immediately better than averaging), then upgrade to A_z once σ²(signal) calibration exists. Keep the trimmed mean as a fallback for when the noise model is out of range.**

**⛔ Kalman / IIR temporal filtering: don't.** `Ŷ_t = (1−α)·W(Ŷ_{t−1}) + α·X_t` looks attractive (O(1) memory, unbounded N, causal) but **every frame resamples the accumulator, so interpolation blur compounds geometrically — after ~20 frames it is visibly softer than any single frame.** That alone disqualifies it for a feature-detection front-end. The finite-window burst is strictly better: bounded error, no compounding resample, and it maps onto the "capture burst → merge → emit one keyframe" pattern SfM wants anyway.

### 11.7 ⭐ BuFF — detect features *in* the burst (stretch goal)

**Ravendran et al., RA-L 2023** · [arXiv:2209.09470](https://arxiv.org/abs/2209.09470) · repo `RoboticImaging/BuFF` (MATLAB + python; ⚠️ **no LICENSE file**)

**Motion model:** on a robot (unlike a phone), platform dynamics are dominated by inertia, so apparent motion within a fast burst is **smooth and locally linear** — general 6-DOF ⇒ **2D linear** feature traces; motion orthogonal to the optical axis ⇒ **1D parallel** traces.
**Shift-sum motion filter:** shift every frame by a candidate slope and sum; sweep slopes to build an `M_x × M_y` **motion stack**. Each slice is a higher-SNR image correctly integrated for objects at that slope. **Structurally a focal stack — it gives depth selectivity for free** (slope ∝ inverse depth under translation).
**Joint 5D extrema search** in `(x, y, scale, slope_x, slope_y)`. Descriptor: standard SIFT 128-D, computed on the **high-SNR motion-stack image**, not any raw frame.

> **⭐ Why detecting in the burst beats denoise-then-detect:**
> - **The slope axis is an extra discriminant noise cannot fake.** To survive, a peak must be an extremum *along the slope dimension too* — consistent with a physically plausible constant apparent motion across all N frames. **Noise has no consistent slope, so it never forms a slope-extremum.** This suppresses spurious features by a mechanism entirely unavailable to a 2D detector on a merged image.
> - **No single alignment has to be correct.** Merge-then-detect commits to *one* warp per tile; anything at a different depth is degraded. BuFF evaluates **all motion hypotheses in parallel** and lets each feature pick its own — **inherently parallax-tolerant, which is precisely the failure mode that kills merge-based methods on a translating robot.**

Verified ROC ordering (peak thresholds equalized): **BuFF-2D/1D > SIFT-on-merged-burst > SIFT-on-single-noisy**. Performance saturates at **6–10 frames**.
**Difficulty: Medium–High** (shift-and-sum is trivial; 5D non-max suppression and sub-slope interpolation are fiddly; 2D slope sweep is memory-hungry). **BuFF-1D is genuinely tractable and is the right target if the robot mostly translates laterally down a corridor.**
> **Fit: excellent as a stretch goal, wrong as a starting point. Do §11.2 + §11.1 first. If you have 3 spare weeks and near-1D motion, BuFF-1D is the highest-value extension in this entire corpus.**

**Also: LBurst** (IJCNN 2025, [arXiv:2410.23522](https://arxiv.org/abs/2410.23522), repo ships 8 `.pt` weights, ⚠️ **no LICENSE**) — README advertises *"3D reconstruction in millilux conditions using captured drone imagery as burst sequences on the fly."* **The closest published match to Ariadne's exact scenario. Read it for framing and the drone dataset; don't reimplement.**

### 11.8 ⛔ Learned burst denoisers — don't bother

**KPN** (CVPR 2018) predicts per-pixel N×K×K filters, K=5, N=8. ⚠️ **±2 px of implicit alignment is useless for a translating robot** — and its annealed loss is a **learned rediscovery of HDR+'s A_z**, which you already have in closed form. Their own ablation notes the HDR+ baseline *"performs similarly to burst averaging but avoids introducing error in the case of misalignment."* Third-party code has **no LICENSE**. **Deep Burst Denoising** (ECCV 2018) — **no explicit alignment at all**; architecturally the least useful for a moving platform. **BPN** (CVPR 2020, ✅ MIT) — factorizes KPN's per-pixel filter bank into global basis kernels + per-pixel mixing coefficients, buying **much larger kernels at lower cost**; the only learned method whose core idea transfers cleanly to a classical implementation and whose alignment range is relevant to a robot. Note for the record; don't build. **Karadeniz et al.** (TIP 2021, MIT) — permutation-invariant feature-level merge; ⚠️ **its stated goal is *perceptual quality* ("perceptually pleasing", "realistic textures"), which is an anti-goal here.**

> ### ⚠️ The generalized verdict, and it applies to §9's dehazing networks too
> **Every learned method here targets human-viewable output. Ariadne's consumer is SIFT/ORB and a bundle adjuster. Optimize for feature repeatability and absence of hallucination, not PSNR or perceptual quality — hallucinated texture creates repeatable-looking keypoints that don't correspond to geometry, and bundle adjustment will happily triangulate them into geometry that isn't there.** The Burst-SfM paper validates exactly this by measuring true/spurious feature counts and pose accuracy instead of PSNR.

**BasicVSR / BasicVSR++** — the *separable* idea worth stealing is the design pattern: **"get a cheap dense flow as a base warp, then let a robust local mechanism absorb the residual, and never trust the flow as final"** — the same insight as A_z arriving from a different direction. Also: **bidirectional accumulation** (a symmetric ±k window beats a causal one at equal latency) and **don't warp twice**. **Not separable:** the gain comes from *offset diversity* in deformable convolution, which has no classical analogue; strip the learning and you're left with plain flow warping, i.e. worse. Also offline, non-causal, PSNR-optimized. **Don't implement.**

**Classical video denoising:** **VBM4D** (TIP 2012) — BM3D extended to spatiotemporal volumes with a separable 4-D transform. **Don't bother** (High/very-High reimplementation, slow, non-commercial reference code) — **but use it as your quality ceiling reference.** **Buades et al.** (TIP 2016) — non-local patch denoising where the search is **guided by optical flow**: flow proposes, patch similarity verifies. **Skip the implementation, adopt the pattern.** **Liu & Freeman** (ECCV 2010) — explicitly gates temporal filtering on **estimated flow reliability**, falling back to spatial-only where flow is unreliable. Same lesson as A_z, from 2010. **FastDVDnet** (CVPR 2020) — competitive video denoising with **no explicit motion estimation at all**; empirical evidence that explicit alignment isn't always necessary at small displacements. Cite it internally if someone insists alignment is mandatory. **Wronski et al., Handheld Multi-Frame Super-Resolution** (TOG 2019) — merges the burst **directly onto an RGB grid** with **locally adaptive kernels** steered by local gradient structure tensors, robustness **per-pixel rather than per-tile**; also *replaces demosaicing*. Don't implement, **but steal the per-pixel-merge-count → adaptive-spatial-denoise idea; it's cheap.**

### 11.9 Build order

| Wk | Deliverable | Risk |
|---|---|---|
| 1 | **Sensor noise calibration** — σ²(signal, ISO) from flat-field/dark-frame pairs. Raw path, linear, no gamma. **Verify global shutter.** | Low, **but everything downstream depends on it — skip it and A_z is meaningless** |
| 1 | Baseline harness: SIFT/ORB repeatability + COLMAP convergence on single dark frames. **Metrics must be feature/pose, never PSNR** | Low |
| 2 | **Trimmed-mean merge** with ECC or homography alignment — crude, already beats single frames | Low |
| 3–4 | **HDR+ core** — 4-level pyramid tile alignment (`phaseCorrelate` coarse, `matchTemplate` fine), multi-hypothesis upsample, **merge to middle frame**, A_z with half-overlapped raised-cosine tiles | **Medium — this is the project** |
| 5 | **IMU gyro pre-warp** + blur-limited exposure scheduler | Low, high payoff |
| 6 | **Mismatch maps** → spatially varying `c`; adaptive post-merge spatial shrinkage | Low (reuses residuals) |
| 7 | **Reproduce Burst-SfM** — merged frames → COLMAP; convergence, registered images, inlier matches vs. baselines | Low |
| 8 | Stretch: **BuFF-1D** if motion is near-lateral; else tune and harden | Med-High |

**OpenCV 5:** `video::findTransformECC` (incl. masked), `video::DISOpticalFlow`, `video::VariationalRefinement`, `video::calcOpticalFlowPyrLK`, `optflow::createOptFlow_DeepFlow`/`DualTVL1`, `photo::fastNlMeansDenoisingMulti` (⚠️ temporal NLM **requires pre-aligned frames** — a *post*-merge or baseline tool, not an alignment substitute), `core::dft`/`mulSpectrums`, `imgproc::phaseCorrelate`/`matchTemplate`/`buildPyramid`/`bilateralFilter`, `ximgproc::guidedFilter`, `quality::` for metrics.

---

## 12. Secondary modalities — event cameras & thermal/LWIR

Domain 4, Tier 2. All URLs, venues and prices fetched, not recalled.

### 12.1 ⭐⭐ Event cameras vs. a co-moving lamp — a first-principles analysis

Each DVS pixel monitors **log** photocurrent and fires when `L(u,t) − L(u,t−Δt) = p·C`. Linearizing:
```
ΔL ≈ (∂L/∂t)|_illum · Δt  +  (−∇L · v) · Δt
                              └─── motion ───┘
```
**Every event-based VO/SLAM method keeps only the second term** — that is the brightness-constancy assumption, and it yields "events are generated by moving edges." EVO, ESVO, EDS and the contrast-maximization family are all built on it.

**⚠️ The literature on a co-moving onboard lamp is genuinely empty — no paper studies event VO in Ariadne's exact configuration. But the physics is not speculative, because a whole subfield deliberately exploits the effect:**

| Work | Venue | What it proves |
|---|---|---|
| **EventPS** | **CVPR 2024** | Recovers surface normals *"from events triggered by changes in logarithmic Lambertian reflections under a **moving directional light source**."* **Camera fixed, scene fixed, only the light moves — and there are enough events to solve per-pixel normals.** |
| **PS-EIP** | **CVPR 2025** | Same premise; notes EventPS is *"sensitive to noise, **shadows**, and non-Lambertian reflections."* |
| Event-based PS via rotating illumination | 2026 | A **single light moving on a circular trajectory** is sufficient signal |
| **ESL** (event structured light) | 3DV 2021 | Event camera + scanning laser → depth |

**The derivation** (the researcher's own analysis, not from any paper — flagged as such). For a Lambertian point at distance `r` lit by a rigidly-attached source:
```
log I = log ρ + log(n·l) − 2·log r + log Φ
d/dt(−2 log r) = −2ṙ/r
⇒ illumination-induced event rate ≈ 2|ṙ| / (r·C)
```
**Ariadne numbers** — drone closing at 1 m/s, wall at 2 m, EVK4 nominal `C = 25%` (≈0.223 nat):
```
rate ≈ 2·1 / (2·0.223) ≈ 4.5 events/s/pixel
× 1280×720 = 921,600 px  ⇒  ≈ 4 MILLION events/s of pure illumination background
```
> **That is the same order as, or larger than, the motion-induced event rate in a typical scene — before a single edge moves. It is not a rounding error; it is the dominant signal.**

**Four failure mechanisms, by severity:** (1) **inverse-square drift** — fires everywhere, *including perfectly textureless surfaces*, directly violating brightness constancy; (2) **shading term `n·l`** — robot rotation swings the lamp, every surface changes brightness by its normal; this is exactly the EventPS *signal*, arriving as noise; (3) **moving cast shadows** — rubble is geometrically extreme, and a co-moving lamp drags high-contrast shadow edges that look *exactly* like real structure to a tracker but aren't attached to the world; (4) **specular highlights** that track with the light, not the scene.

> **⭐ A useful discriminator worth testing (also the researcher's own observation, unpublished as far as they could find):** motion events scale as `|∇L·v|` → **zero in textureless regions, concentrated on gradients.** Illumination-drift events scale as `2|ṙ|/(rC)` → **nonzero everywhere, independent of texture.** So the contamination is *spatially dense and low-gradient* while the signal is *sparse and gradient-aligned*. **A per-pixel event-rate normalization against a local-gradient estimate should suppress much of it.**

**Two hardware mitigations that follow directly:** **mount the lamp coaxially** (ring-light geometry) so cast shadows fall *behind* the objects casting them, hidden from the camera — kills mechanism (3), does nothing for (1) or (2). And **drive the lamp with DC/current regulation, never PWM** — a PWM-dimmed LED saturates the sensor with coherent flicker events. **The cheapest, highest-impact hardware decision in this section.** (Note this independently confirms §10.5's PWM warning for the frame pipeline.)

**⭐ The closest published configuration: NightSight** ([arXiv:2605.26330](https://arxiv.org/abs/2605.26330), 2026) — small aerial robot, **complete darkness**, monocular event camera + **coded aperture lens + IR dot projector**. Depth from the depth-dependent blur signature of the projected pattern, CNN trained *only on synthetic planar-wall data*, generalizing zero-shot. **20 Hz on a Jetson Orin Nano, L1 error 7.0 cm to 2.5 m (2.80%).**
> **Note what they did: they abandoned motion-based VO entirely and used the active illuminator as the signal. That is the honest read of this whole literature — under onboard lighting, treat the light as *structured light*, not as a nuisance to a brightness-constancy tracker.**

### 12.2 Event algorithms — and the one worth reimplementing

⚠️ **License + patent trap, verified by downloading the actual agreements.** **EVO and EMVS** ship no `LICENSE` file — instead a Word doc titled **"NON-COMMERCIAL SOURCE CODE LICENSE AGREEMENT"** with the University of Zurich: internal non-commercial research only; **§3.1 grants UZH the right to *sell* your derivative work**; §3.2 requires you to provide derivatives on request; §7.4 **auto-expires after five years**. Both READMEs also state the algorithms are **patented: WO/2018/037079, PCT/EP2017/071331**. **The patent is on the *method*, so it survives reimplementation** — fine for a research demo, flag it if anything commercial is contemplated.

| Method | Venue | License | Difficulty |
|---|---|---|---|
| EVO | RA-L 2017 | ⚠️ UZH non-commercial + patented | High |
| **Ultimate SLAM** | RA-L 2018 | GPLv3 | **Medium (front-end)** |
| ESVO / ESVO2 | TRO 2021 / 2025 | GPLv3 / **MIT** | High |
| EDS | CVPR 2022 | GPLv3 | High |
| DEVO | 3DV 2024 | **MIT** (⚠️ weights not separately checked) | High |
| **EMVS** | IJCV 2018 | ⚠️ UZH non-commercial + patented | **Low–Medium** |

**⭐ EMVS is the one to reimplement.** Given events *and known poses*, back-project each event as a ray into a **Disparity Space Image** — a 3D voxel grid anchored at a reference view — and **vote**. Rays from events generated by the same 3D edge, seen from many viewpoints, intersect at that edge, producing a local maximum in ray density. Extract semi-dense depth by finding, per pixel, the depth maximizing ray density.
> **There is no explicit data association and no photometric matching — it is a Hough-style accumulator.** Elegantly simple, robust to noise, and it exploits precisely what events give you (many views, high temporal resolution). **The most tractable serious event algorithm in this entire corpus.** ⚠️ It needs poses from somewhere — pair it with your main SLAM stack, not with EVO. OpenCV 5: the DSI is just a `cv::Mat` of dims `(W,H,N_depth)`; `ptcloud` for voxel/filtering, `imgproc` for local-max extraction, `calib`/`geometry` for rays.

**Ultimate SLAM** is the most Ariadne-relevant *design*: events are batched into **motion-compensated event frames** (warped by IMU-predicted motion so edges sharpen instead of smear), then **standard FAST + KLT run on those synthetic frames** and feed a conventional VIO. **That's what makes it reimplementable — once you have a sharp event frame, everything downstream is off-the-shelf.**

**The OpenCV bridge (≈150 lines, and it unlocks everything):** hot-pixel filter → background-activity filter → **exponentially-decayed time surface** `T(u) = exp(−(t − t_last(u))/τ)` at 50–100 Hz → `cv::normalize` to 8-bit → **the entire OpenCV toolbox applies unchanged** (`features` FAST/ORB, `video::calcOpticalFlowPyrLK`, `calib::solvePnPRansac`, `stereo`). The *negative* time surface behaves like a distance transform to recent edges, which is exactly why ESVO can run direct alignment on it.

**Simulators:** **v2e** (**MIT**) — video→events *with a low-light noise model*, the more honest choice given §12.1. **vid2e/ESIM** (GPLv3). `prophesee-ai/openeb` (**Apache-2.0**) for RAW/HDF5 reading.

**⚠️ Low light degrades the sensor itself, not just the algorithm.** *Seeing Motion at Nighttime with an Event Camera* (CVPR 2024) reports nighttime events show **"temporal trailing characteristics and spatial non-stationary distribution"** — delayed, smeared response with spatially structured noise. **SciDVS treats 1.7% contrast sensitivity at 0.7 lux as a research-grade achievement**, which tells you what commercial DVS pixels *don't* do there. **The >120 dB figure is a *dynamic range* claim, not a *low-light sensitivity* claim** — the two are routinely conflated in event-camera marketing.

### 12.3 🔴 Event camera verdict — don't buy one

Independently reached by two research passes (cf. §8.5), with more physics behind it here:

1. **The co-moving lamp breaks the core assumption of every method listed** — ~4 M events/s of illumination background at HD, comparable to or exceeding the motion signal. **No paper has characterized this failure, so the team would be doing unfunded original research, in the dark, on a deadline, while also owing multi-agent SLAM and 3D reconstruction.**
2. **Zero OpenCV support** — every byte of I/O, filtering, representation and visualization is hand-written first.
3. **Cost and lead time** — iniVation DVXplorer Micro **€2,400 academic**, DAVIS346 **€3,900**, DVXplorer **€3,400**; the Prophesee EVK4 HD you'd actually want is **quote-only**, i.e. a sales cycle in weeks. Against a 2-month budget that's a delay, not a purchase.
4. **Low light degrades the sensor**, per above.

**If the team wants an event deliverable at all, pick exactly one, timeboxed to a week:** **(best) reimplement EMVS** on ECD sequences that ship ground-truth poses — no hardware, sidesteps the UZH licence, leaves only a patent that doesn't bite a research demo; **(fallback)** build the time-surface bridge + filters and demo FAST+KLT on public data; **(headline)** use **v2e to simulate a co-moving lamp** and *measure* the illumination-induced event rate against the `2|ṙ|/(rC)` prediction — **cheap, needs no camera, and appears to be genuinely novel.**

### 12.4 ⭐⭐ Thermal victim detection — the best value-per-hour item in the entire corpus

> ### The single most important sentence: **capture 14-bit raw radiometric (TLinear), never 8-bit AGC.** It is a configuration flag, and it determines whether this is trivial or impossible.

**Why AGC destroys it:** AGC is a per-frame histogram stretch for *human viewing*. (1) **The pixel→temperature mapping changes every frame** — a 37 °C victim reads 200 in one frame and 90 in the next when a hot pipe enters view, so **absolute thresholding becomes impossible**; (2) **small hot blobs get crushed**, because equalization allocates range by pixel *population* — a distant victim on 30 pixels gets almost none while a large cold wall gets most of it; (3) radiometric precision thrown away, 14→8 bits non-linearly; (4) **temporal flicker** ruins any temporal filter or photometric VO. Raw TLinear instead gives each pixel a value **linearly proportional to absolute temperature** — a body at 37 °C reads the same number in every frame, in every scene.

**The pipeline:**
```
14-bit TLinear frame → affine map to T(u,v) [vendor constants]
  → band threshold 30 °C ≤ T ≤ 40 °C          [cv::inRange]
  → morphological open then close             [morphologyEx, MORPH_ELLIPSE]
  → connected components with stats           [connectedComponentsWithStats]
  → gate on area, aspect ratio, fill ratio, blob-vs-local-background ΔT
  → back-project centroid as a ray into the SLAM map
```

> **⭐ Why the SNR is overwhelming: a human at ~37 °C skin (ε ≈ 0.98, essentially a blackbody) in a 10 °C void is ΔT = 27 K. Against NETD ≤ 60 mK that is ~450:1. There is no ambiguity to resolve. This is not a detection problem in the machine-learning sense — it is a thresholding problem, and the honest engineering answer is a threshold.**
> **Effort: ~1 day. Pure OpenCV. No model, no training data, no GPU, no dataset licence question, runs on a Pi, and explaining it to a judge or a rescue professional takes one sentence.**

**⭐ And sidestep RGB↔thermal registration entirely.** Ariadne doesn't need dense pixel-accurate cross-modal alignment — it needs *"a hot blob is at thermal pixel (u,v); where is that in the world?"* With the cameras **rigidly mounted and calibrated once offline**, back-project the blob centroid as a **ray** and intersect it with the 3D map SLAM is already building. **One ray, one map lookup, no per-frame registration.** ~50× less work, more robust, and it exploits the fact that Ariadne is a reconstruction project.

**🔴 Honest failure modes — and this is where the intellectual content is:**

**False positives.** Sun-warmed concrete and dark metal routinely exceed 50 °C and sweep *through* the band while cooling — **the upper bound (≤40 °C) does most of the work; this is why a band, not a lower-bound threshold, is the right primitive.** Cooling engines sweep through too. **⚠️ The non-obvious one: low-emissivity metal as a thermal mirror.** Polished metal has ε ≈ 0.05, reflectivity ≈ 0.95 in LWIR — **a steel plate reflects a victim's body heat and reads as a human-temperature region *at the wrong location*. No temperature gate can remove it, because the apparent temperature really is ~37 °C.** Mitigation: **multi-view consistency — a reflection moves inconsistently with parallax as the robot translates, and Ariadne is already estimating pose, so this is nearly free.** Plus shape/planarity gating.

**False negatives — the more serious direction for a rescue tool.**
> **⚠️⚠️ LWIR is strictly line-of-sight. A fully buried victim is INVISIBLE — full stop. This is physics, not an algorithmic limitation:** 8–14 µm radiation does not penetrate concrete, brick, wood, or meaningful dust. A victim under 10 cm of rubble becomes detectable only once body heat has *conducted* to the surface — hours, producing a diffuse 1–3 K patch, not a person-shaped blob. **A partially-buried victim with an exposed hand or face is detectable; a covered one is not. State this explicitly — overclaiming here is what gets a rescue tool distrusted.**

**Hypothermia:** victims trapped for hours drift toward ambient, and *skin* temperature falls much faster than core. **A hypothermic victim at 28 °C skin falls outside a naive 30–40 °C band.** Mitigation: widen the band and lean on *relative* ΔT against local background plus shape gating. Also: wet clothing (evaporative cooling), dust on skin (changes emissivity and insulates), and range — on a 160×120 Lepton with a 57° lens a person at 10 m spans a handful of pixels, so **detection range is set by angular resolution, not sensitivity.**

**⭐ The field evidence, and it reframes the claim.** Zhang, Sessa, Kasai et al., *"Evaluation of a Sensor System for Detecting Humans Trapped under Rubble,"* **Sensors 18(3):852, 2018** — CO₂ + thermal + microphone in a high-fidelity simulated disaster area. Their finding: **"the CO₂ sensor is useful to effectively reduce the possible concerned area, while the thermal camera can confirm the correct position of the victim."**
> **Thermal is a *confirmation* sensor, not a *search* sensor, for buried victims. Frame the claim that way in the report.** ⚠️ No quantitative field ROC data for thermal victim detection in collapsed structures could be found — the USAR literature is largely qualitative pilot studies. **Say so rather than inventing numbers.**

**⛔ Learned victim detection: don't.** A YOLO variant on 8-bit AGC thermal will **underperform a radiometric threshold**, because the network is handed data from which the discriminating physical quantity — absolute temperature — **has already been destroyed by AGC**. It costs a training pipeline, a dataset licence, and explainability. **If you want one learned component, use it for *shape* verification on top of the threshold, not instead of it.**

### 12.4b ⭐⭐ CERBERUS — the actual mechanism, and the architecture worth stealing

**Khattak, Papachristos, Alexis, "Keyframe-based Direct Thermal-Inertial Odometry," ICRA 2019** ([arXiv:1903.00798](https://arxiv.org/abs/1903.00798)). DJI Matrice 100, **FLIR Tau 2 640×512 @30 fps**, validated indoors *and inside an underground mine* in darkness with airborne dust.

> **⚠️ The damning baseline first: they ran OKVIS, DSO and ROVIO on thermal imagery. "Even with significant fine tuning the odometry estimates of these approaches diverge significantly." Do not expect an off-the-shelf VO to work on thermal.**

**They don't track features at all — they go direct, on 14-bit data**, minimizing `e_radio = Σ_i Σ_{p∈N_i} ‖T(p') − T(p)‖²` where **T is the thermal value in 14-bit resolution**. Weighted Gauss–Newton, **coarse-to-fine over a scale-space pyramid, coarsest level initialized by an IMU prior**, aligned against the last two frames plus the latest two keyframes.

**Two front-end tricks that are the actual contribution, and both are cheap:**
- **⭐ Block-median adaptive gradient threshold.** Compute gradients, divide into **32×32 blocks**, take the **median gradient per block**, add a constant offset, and suppress everything below `g_med + g_off` **within that block**. **This is the right answer to thermal's uneven contrast** — a *global* gradient threshold would take all its points from whichever corner has thermal contrast and none from the rest. Points then go into a **2D point-selection grid** the size of the image, enforcing spatial spread without pairwise distance computation.
- **⭐ Residual ratio test.** For each point, compute the radiometric residual **along its motion vector** within a search window, take the ratio of **lowest to second-lowest** residual, reject if too low. **Lowe's ratio test transplanted onto direct residuals.** Its stated purpose: eliminate points that *"due to alignment errors separate from corner or edge locations onto a uniform planar surface and become unconstrained."* **This is the mechanism that stops thermal's low-texture regions poisoning the estimate.**

**⭐⭐ CompSLAM — the graded-fallback architecture, from [arXiv:2207.04914](https://arxiv.org/abs/2207.04914) (Field Robotics 2022).** *"a **hierarchical sensor fusion strategy which refines individual or multi-modal pose estimates as they propagate through the framework**, ensuring that a robot always has an updated pose estimate."*

**How it decides what to track across two modalities — steal this:**
> *"The VTIO estimator **tracks and balances a fixed number of features across multi-spectral images by evaluating the quality of each image subject to a spatial and temporal entropy criterion**."*

**The EKF feature budget is *fixed* (bounding compute) and reallocated between visible and thermal frame-by-frame according to an entropy score.** Walk into darkness, RGB entropy collapses, features migrate to thermal; walk into a thermally-flat cave, they migrate back. **A continuous per-frame reallocation rather than a discrete modality switch** — and simple enough to reimplement (spatial + temporal-difference entropy per image, normalize, allocate slots proportionally).

**How the fallback triggers — two health checks, then a graded chain:**
1. **VTIO health:** (a) is the estimate within the **motion bounds set by the robot controller**? (b) is the **covariance growth rate** acceptable, by the **D-optimality criterion**?
2. **If it fails**, Laser Odometry falls back **in order**: kinematic/leg odometry → IMU-propagated pose → constant-velocity model.
3. **LiDAR degeneracy check** via the **minimum eigenvalues of the optimization Hessian** — and then the best idea in this literature:

> ### ⭐⭐ *"If optimization degeneracy is detected, the ill-conditioned dimensions of the LO estimation output are **replaced with the prior's estimates**."*
> **Fallback is per-degree-of-freedom, not all-or-nothing.** In a long featureless tunnel, LiDAR is well-conditioned in 5 DOF and degenerate only along the tunnel axis; CompSLAM substitutes the prior **only in that direction** and keeps the good LiDAR estimate everywhere else. **This composes directly with §10.9's degeneracy detection and §7.3's null-space machinery — same eigen-analysis, now driving a graded fallback rather than just a warning.**

Output fuses with IMU in a **sliding-window factor graph (GTSAM)**, with **zero-velocity factors when stationary** to pin IMU biases.
⚠️ **Field-honest failure mode from the paper: *"for a thermal camera, cave environments presented thermal flatness."* Thermal is not a universal fallback — it has its own null space, which is why CERBERUS has a third and fourth option beneath it.**

> **⭐ Transferable verdict: do NOT build thermal odometry as a deliverable. Build the health-check + graded-fallback wrapper around whatever primary SLAM you have — a few hundred lines, the actual lesson from a DARPA-winning system, and it pays off even if thermal never ships.**

**⭐ And CERBERUS validates §12.4's victim detector almost exactly:** *"the detector **clustered pixels that were in the same temperature range as specified by DARPA for the Survivor artifact or 10 °C above the ambient temperature (defined as the average of all temperature values of the last three images)**… If enough adjacent pixels were in a similar temperature range, a thermal artifact was considered detected and **sent to the Human Supervisor for further inspection**."* Run at **3 Hz**.
**Three things to copy verbatim: (a) the ambient reference is *adaptive* — mean of the last three frames — not a fixed absolute band, which is what survives changing ambient and residual FFC drift; (b) the criterion is a *relative* +10 °C offset as an alternative to the absolute band; (c) detections go to a *human for confirmation*, never an autonomous accept.** They also **ray-cast detected artifact locations into the volumetric map** — independently confirming §12.4's recommendation.

> ⚠️ **One more failure mode §12.4 missed: warm ambient.** In summer, or a building still retaining heat, void ambient can reach **30–35 °C**. The ΔT collapses from 27 K to a few K and **the 450:1 SNR advantage largely evaporates.** The adaptive-ambient + relative-offset primitive is the right defence — but if ambient ≈ 35 °C the contrast is genuinely marginal, and the report should say so rather than quoting 450:1 unconditionally.

**⭐ If you end up with a camera that won't expose raw:** Das, Matthies, Daftry (JPL), **"Online Photometric Calibration of Automatic Gain Thermal Infrared Cameras"** ([arXiv:2012.14292](https://arxiv.org/abs/2012.14292)). Their framing matches §12.4's: methods for photometric calibration of RGB cameras *"cannot be applied to thermal-infrared cameras due to difference in data acquisition and sensor phenomenology."* An **online** algorithm estimating and inverting the AGC transform, **requiring no specific driver or hardware support** — works on any COTS thermal camera. Presented explicitly for VO/SLAM. **The escape hatch.**

**SBNUC detail (§12.5) — the two families, and the failure mode that sinks naive implementations.** *Constant-statistics* (Harris & Chiang, IEEE TIP 1999): assume every pixel sees the same intensity distribution over a moving window, so per-pixel temporal mean and std *are* the offset and gain error — corrected output `(y − μ)/σ`, maintained by EMAs. **~80 lines.** ⚠️ **Its failure mode has a name: ghosting.** Dwell on a static scene and scene structure gets absorbed into μ, burning in as a negative afterimage — Harris & Chiang wrote a dedicated paper on it (SPIE 1998). **Mitigation: gate the update on detected motion. This is the single most important practical detail.** *Registration-based* (Hardie et al., Applied Optics 2000; **gated-LMS** version, Optics Express 2009): estimate inter-frame motion, exploit that a true scene point must read the same value at whichever pixel observes it. **⭐ Ariadne already has a motion estimate from SLAM, so the gate is nearly free — build the gated-LMS variant.**

### 12.5 Thermal sensor practicalities

**Microbolometer physics:** thermally-isolated VOx/a-Si membranes whose *resistance* changes with incident IR. Consequences: **thermal time constant ~10 ms → an effective motion blur no shutter speed can fix** (on a fast drone, thermal smears); every pixel has its own resistance-vs-temperature curve that **drifts with the camera's own body temperature**; frame rates typically 9 Hz (export-restricted) or 30/60 Hz.

**⚠️ The FFC problem.** The camera periodically closes an internal shutter and re-learns per-pixel offsets. **Video freezes ~0.5–1 s every 1–3 minutes**, with a click. On a moving drone that is a dropped chunk of trajectory. **Most SDKs let you defer or manually trigger FFC — do that, and trigger while hovering, never while transiting.** Between FFCs, **fixed-pattern noise grows back as static row/column stripes that are fixed in *image* coordinates — poison for any tracker, because FPN features don't move with the scene and will anchor your tracker to the sensor rather than the world.**

**⭐ Scene-based NUC (SBNUC) — a good self-contained deliverable, ~3 days, ~100 lines.** Removes the FFC freeze by estimating correction from imagery. Two families: **registration-based** (estimate inter-frame motion; a true scene point should read the same value at different pixels, so inconsistencies *are* per-pixel offset error — accumulate and subtract) and **statistical/constant-statistics** (assume every pixel sees the same intensity distribution over a long window, true if the camera moves, so per-pixel temporal mean/variance deviations *are* the offset/gain error — update with LMS or Kalman). OpenCV 5: `imgproc`, `video`, `ximgproc::guidedFilter`, `xphoto`.

**⭐ Geometric calibration — printed checkerboards are INVISIBLE in LWIR** (black and white ink have the same emissivity and, at equilibrium, the same temperature). **The verified best answer: emissivity-contrast targets.** ElSheikh, Abu-Nabah, Hamdan, *Sensors* 23(7):3479, 2023 — a **UV-printed checkerboard on polished SS304 stainless steel** achieved **0.057 px mean reprojection error**, better than typical *optical* calibration. Polished metal ε ≈ 0.05, printed ink ε ≈ 0.95 — a huge contrast **that needs no heater and no power.** Then it's standard: `calib::findChessboardCornersSB` (more robust to thermal's soft corners), `cornerSubPix`, `calibrateCamera`, `stereoCalibrate`. Tips: upsample 2–4× before corner detection (160×120 is tiny), use a larger physical target, average many frames per pose to beat NETD noise, and **force an FFC immediately before each capture.**

**⚠️ RGB↔thermal registration is genuinely hard and is where teams lose a week.** There is **no shared appearance model** — a red wall and a blue wall look different in RGB and identical in LWIR; gradients frequently have **opposite sign** (contrast reversal). **SIFT/ORB/SURF matching across RGB↔LWIR essentially does not work. Do not plan on it.** If you must register: **Mutual Information** — maximize `MI = H(A) + H(B) − H(A,B)` over warp parameters, which is maximal at alignment *whatever* the intensity relationship, including inverted. ⚠️ **No MI implementation exists in OpenCV** — `reg` is *photometric* (it assumes exactly the shared appearance you don't have) and `rgbd` targets single-sensor RGB-D. **You implement it: `cv::calcHist` with 2 channels for the joint histogram (~64×64 bins), entropy sum in a few lines, Powell/Nelder–Mead over a similarity warp, coarse-to-fine. ~200 lines — a good, defensible deliverable.** Or align **edge maps** matching gradient orientation *modulo 180°* (invariant to contrast reversal) via `distanceTransform` chamfer matching.

**Hardware (GroupGets, verified 2026-08-29):** **⭐ FLIR Lepton 3.5, 160×120, radiometric, $164** — ~10×12×6 mm, <1 g, ~150 mW, SPI/I²C from a Pi or Jetson; PureThermal breakout gives USB-UVC (**make sure you pull the 14-bit TLinear stream, not the 8-bit AGC preview**). Radiometric Boson 320 **$2,123**, Boson+ 640 from **$4,448**. ⚠️ **Export control: Boson and higher-frame-rate cores are ITAR/EAR restricted — widely-sold variants are often frame-rate-limited to 9 Hz for this reason. Check before ordering internationally; this can add weeks.**

**Thermal VO/SLAM: stretch goal, not a deliverable.** ⭐ **The transferable lesson is architectural, from CERBERUS/ICUAS 2020** (Khattak, Nguyen, Mascarich, Alexis): **they do not trust thermal alone** — VIO and thermal-inertial odometry are used as **robust priors for LiDAR odometry**, and to **selectively propagate** the LiDAR estimate when LiDAR degenerates in a featureless tunnel. **The architecture is *complementary degradation*: each modality fails differently, so fuse at the estimator level and let whichever is healthy carry the others. Thermal is a resilience layer for the primary SLAM stack, not a standalone odometry source.** (Khattak et al. ICRA 2019 works on **direct 14-bit radiometric** data, not AGC — the field's standard practice, for the same reason as §12.4.) **ROVTIO** (`ntnu-arl/rovtio`, same NTNU lineage) is worth two ideas: it **does not rely on stereo correspondences between modalities** (each camera contributes its own patch features to a shared filter state), and it can **initialize and track with only one stream available** — exactly the graceful degradation needed when a robot passes from a lit corridor into a thermally-flat void. ⚠️ Attributed to a master's thesis; **no peer-reviewed publication found.**

### 12.6 Ranked recommendations

| # | Item | Effort | Why |
|---|---|---|---|
| **1 ✅** | **Buy a Lepton 3.5 ($164); build the classical radiometric victim detector** | **~1 day** | ~450:1 SNR, pure OpenCV, explainable in one sentence, directly on-mission. **Best value-per-hour in the entire corpus.** |
| **2 ✅** | **Ray-cast blob centroid into the SLAM map** instead of per-frame registration | ~1 day | Sidesteps the hardest problem in the section by exploiting the map you're already building |
| **3 ✅** | **Document the failure modes honestly** — line-of-sight limit, hypothermia, thermal mirrors | ~0 | **The difference between a demo and a credible rescue tool, and it's free** |
| **4 👍** | Reimplement **EMVS** on public event data | ~1 wk | Lowest-difficulty serious event algorithm; elegant Hough mechanism; produces 3D; no hardware |
| **5 👍** | **Scene-based NUC** | ~3 d | Genuinely interesting signal processing, ~100 lines, removes the FFC freeze |
| **6 👍** | **MI-based offline RGB↔thermal extrinsic calibration** | ~2 d | ~200 lines, no OpenCV builtin so it's real work, classical and defensible |
| 7 🤔 | Time-surface bridge + OpenCV tracking demo | ~3 d | Prerequisite for any future event work |
| 8 🤔 | **v2e experiment measuring co-moving-lamp event contamination** | ~3 d | Cheap, no camera, **appears genuinely novel** |
| **9 ❌** | Buy an event camera | — | €2,400+, quote-only for HD, weeks of lead time, zero OpenCV support |
| **10 ❌** | Event VO/SLAM as a deliverable | — | Co-moving lamp breaks brightness constancy; unstudied; original research on a deadline |
| **11 ❌** | Thermal VO/SLAM as a deliverable | — | High difficulty. Take the CERBERUS *architectural* lesson without the implementation |
| **12 ❌** | Event NeRF/3DGS, cross-spectral stereo, learned victim detection | — | All strictly worse than the simple option |

**Datasets:** ⭐ **ViViD++** is the sleeper — RGB + **thermal + event** with explicitly dark sequences, the only public data spanning both modalities at once. ⭐ **WiSARD** ([arXiv:2309.04453](https://arxiv.org/abs/2309.04453)) — *"A Labeled Visual and Thermal Image Dataset for Wilderness Search and Rescue."* The driving sets (KAIST, FLIR ADAS, M3FD, MFNet) are **wrong-domain** — daylight street scenes with standing pedestrians, not dark voids with prone occluded bodies. ⚠️ FLIR ADAS is registration-gated with **no verifiable dataset-specific licence**; LLVIP is **non-commercial, permission-gated**.
> **Honestly: for the classical detector you barely need a dataset. A person, a cold room and a $164 Lepton gives you a better validation set for this specific task than any of the above, in an afternoon.**

---

## 13. Low-light enhancement — does it actually help SLAM?

The question is answerable and the answer is **equivocal in a specific, actionable way.** Three independent papers split cleanly by illumination regime.

### 13.1 ⚠️ The empirical backbone — enhancement can make SLAM worse

**Summers, Jones, Seale, *Sensors* 2025** ([PMC12656054](https://europepmc.org/articles/PMC12656054), free full text; MDPI 403s direct fetches). 6 enhancers vs unenhanced, ORB features, ORB-SLAM3, 5 runs each.

| Method | ORB inliers | vs original |
|---|---|---|
| **Original (no enhancement)** | **495.88** | — |
| Ancuti 2017 | 481.55 | −2.9% |
| Demir & Kaplan 2023 | 389.42 | −21.5% |
| FUnIE-GAN | 424.67 | −14.4% |
| **WaterNet** | **517.97** | **+4.5%** |
| **WaveNet** | **530.53** | **+7.0%** |

Verbatim: *"All enhancement methods tested were detrimental to SLAM's tracking accuracy, and loop closure detection, leading to drift and incomplete map generation."* · *"the original video remained the only video that resulted in a complete survey"* · *"no enhancement method resulted in the capture of a single loop closure in five runs. Only the original video resulted in any loop closure detection."*

> ### ⚠️⚠️ **The critical nuance: WaterNet and WaveNet *raised* inlier counts (+4.5%, +7.0%) and still destroyed loop closure entirely.**
> **A front-end ablation that counts matches will mislead you.** Likely cause (the researcher's inference, not the paper's — it lists the mechanism as future work): per-frame *adaptive* enhancement applies a different intensity mapping to each frame, **breaking the photometric consistency that place recognition and BA depend on.**
>
> ⚠️ Caveat: the six enhancers are underwater-specific, not low-light methods. **The transferable findings are methodological — appearance metrics don't predict matching utility, and inlier counts don't predict trajectory quality** — not direct evidence about Zero-DCE-class methods.

**Useful by-product:** the paper proposes **local matching stability (LMS)** and **furthest matchable frame (FMF)** as cheap surrogates for full-SLAM evaluation. **A ready-made evaluation harness, cheap to reimplement.**

### 13.2 ⭐ The benefit depends on the *detector*, not the scene

**Twilight SLAM** ([arXiv:2304.11310](https://arxiv.org/abs/2304.11310), ⚠️ **preprint, no venue**). RMSE (m), enhancers = Bread / Dual / EnlightenGAN / Zero-DCE:

| Sequence | System | Original | Bread | Dual | EnGAN | Zero-DCE |
|---|---|---|---|---|---|---|
| KITTI 04 | **SuperPoint-SLAM** | **0.26** | 0.40 | 0.47 | 0.37 | 0.52 |
| KITTI 04 | ORB-SLAM3 | 1.72 | 1.30 | 1.52 | **1.19** | 1.50 |
| KITTI 06 | ORB-SLAM3 | 17.65 | 14.56 | 13.91 | 15.22 | **11.98** |
| KITTI 07 | **SuperPoint-SLAM** | **2.40** | 2.57 | **8.07** | 3.22 | 2.08 |

> **Every enhancer hurt SuperPoint-SLAM on KITTI 04 (1.4–2× worse); Dual made KITTI 07 3.4× worse. Every enhancer helped ORB-SLAM3.** The paper's own Table II selects **"Original" as best for SuperPoint-SLAM in both mild-daylight and semi-dark**; only in true dark does an enhancer win.
>
> **⇒ Hand-crafted ORB benefits from enhancement; learned SuperPoint — already trained across illumination — is harmed. Since §1.2 recommends a learned front-end (XFeat), enhancement is likely a net negative for Ariadne.**

Their own conclusion: *"in extremely dark environments, the improvements in localization accuracy with SOTA enhancers are **incremental**."*

> **⭐ And one systems finding that argues for VIO over pure monocular VO more strongly than for any enhancer:** the LaMARia benchmark ran six systems (ORB-SLAM3, DSO, Kimera-VIO, OpenVINS, DPVO, DPV-SLAM) on five low-light sequences. **Only Kimera-VIO tracked all five to completion**; ORB-SLAM3, DSO and OpenVINS *"fail outright or diverge on most of the harder and low-light sequences"*; DPVO/DPV-SLAM never lose tracking but accumulate **~100 m absolute error.** Conclusion: **RGB-only SLAM holds up in the dark only when inertial fusion AND global optimization are both present.** ⚠️ Source is [arXiv:2607.17699](https://arxiv.org/abs/2607.17699), identified in its own arXiv comments as a university certification-program project — **not peer-reviewed.** Weight accordingly, but the direction matches §1.1's architecture call.

**Counter-evidence for the extreme case: DarkSLAM** ([arXiv:2206.02199](https://arxiv.org/abs/2206.02199), IEEE VTC 2022) — GAN preprocessing + ORB-SLAM on the darkest sequences: **25.1% tracking time vs 0.6% for existing approaches.** When the baseline tracks 0.6% of a sequence, "enhancement adds spurious features" is irrelevant — you have nothing to lose. **The regimes are genuinely different.**

### 13.3 ⭐⭐ MID (ICCV 2021) — the most rigorous study, and it overturns folk wisdom

**"Matching in the Dark"** ([arXiv:2109.03585](https://arxiv.org/abs/2109.03585), **ICCV 2021 verified**). 8 enhancement methods × 11 matchers, evaluated by **relative pose error**, not match counts. `N_τ` at τ=5°, indoor ("R-" = after the standard 8-bit pipeline; "D-" = directly on RAW):

| Matcher | none | R-HistEq | R-CLAHE | R-MIRNet | D-HistEq | D-CLAHE | **D-BM3D** | SID |
|---|---|---|---|---|---|---|---|---|
| SuperPoint | 0.223 | 0.421 | 0.275 | 0.381 | 0.548 | 0.540 | **0.596** | 0.583 |
| SP+SuperGlue | 0.219 | 0.400 | 0.292 | 0.404 | 0.548 | 0.544 | 0.585 | **0.619** |
| R2D2 | 0.113 | 0.317 | 0.192 | 0.229 | 0.388 | 0.383 | **0.483** | 0.421 |
| **ORB** | 0.029 | 0.210 | 0.056 | 0.125 | 0.267 | 0.238 | **0.296** | 0.238 |

Four results that should shape the preprocessing choice:
1. **Plain global histogram equalization beats a SOTA CNN enhancer** (R-HistEq 0.421 > R-MIRNet 0.381). Verbatim: *"using classical histogram equalization or the state-of-the-art CNN-based image-enhancing method does not make a big difference."*
2. **⚠️ CLAHE is markedly WORSE than global HistEq** — 0.275 vs 0.421 for SuperPoint; for ORB nearly useless (0.056 vs 0.210, a **3.75× gap**). **This directly contradicts the folk wisdom that CLAHE is the safe default for VO front-ends.** Caveat: tile size and clip limit are not reported as tuned — and CLAHE's *local adaptivity* is exactly the property that breaks cross-frame photometric consistency, consistent with §13.1's loop-closure result.
3. **⭐ Bit depth beats algorithms.** Every "D-" column beats its "R-" counterpart. ORB goes 0.210 → 0.267 **just by moving HistEq before the 8-bit bottleneck.** Simple denoising (BM3D) on RAW is best or near-best overall — better than the learned SID for most matchers.
4. **The killer quote:** *"the manual method yields significantly better results than the image enhancers tested in this paper… **The standard image enhancers are designed to yield images that appear the most natural, which should differ from the best image for image matching.**"*

Also: ORB is catastrophically weak in the dark in absolute terms (0.029 unenhanced vs SuperPoint's 0.223).

### 13.4 ⭐⭐ What actually wins: control the light, don't fix the pixels

**"Active Illumination for Visual Ego-Motion Estimation in the Dark"** — [arXiv:2502.13708](https://arxiv.org/abs/2502.13708), **ICRA 2025 (journal-ref verified)**. A detector identifies high-texture regions; a **pan-tilt controller steers the light beam at them**, feeding information-rich images to the VO.
> ### **Result: a reduction in pose estimation error of up to 75% versus fixed lighting.**
> **That is a far larger effect than any enhancer in any paper above — by an order of magnitude.**

**"Lightning: Adaptive Illumination Control for Robot Perception"** ([arXiv:2602.15900](https://arxiv.org/abs/2602.15900), 2026 preprint) — closed-loop control of a programmable onboard light. Trains a **Co-Located Illumination Decomposition (CLID)** relighting model splitting an observation into ambient + light-contribution field, so you can **synthesize the same scene at other light intensities and generate dense multi-intensity training data without re-flying trajectories.** Its framing is worth quoting to the team: all downstream fixes *"are limited by the image captured [in] these conditions"*, and illumination *"can both reveal structure and induce failure modes such as specular highlights and saturation."*

**⭐ IRAF-SLAM** ([arXiv:2507.07752](https://arxiv.org/abs/2507.07752), **ECMR 2025**) — **the cheapest high-value idea here.** Rather than enhancing pixels, it adapts the *front-end*: entropy/intensity/gradient-driven **adaptive detection thresholds** plus density-and-lighting-based **feature culling**. Reduced tracking failures on TUM-VI and EuRoC *"without incurring significant computational overhead."* **Costs almost nothing and doesn't touch photometric consistency.**

**ULL-SLAM** (*Frontiers in Marine Science* 2023) states the design lesson directly. Figure 1 caption, verbatim: *"An image-level low-light enhancement preprocessing module (e.g., Zero-DCE) can improve human visual perception. **However, it is unlikely to improve feature point matching performance** between two adjacent frames."* And, describing **precisely Ariadne's configuration**: *"Although artificial light sources could help to some extent, **they might also cause non-uniform illumination, which may have an adverse effect on feature point matching.**"* Their fix is the architectural lesson: a **zero-reference enhancement branch trained *jointly* with a self-supervised feature detector/descriptor branch**, plus a temporal consistency term — **enhancement optimized for matchability, not for looking good, and video-level rather than per-frame.**

### 13.5 The methods, ranked

**⭐ LIME** (Guo, Li, Ling, **IEEE TIP 2017**; [arXiv:1605.05034](https://arxiv.org/abs/1605.05034)) — **best classical value-for-effort, ~1 week.** Model `L = T∘R`, recover `R = L/T`.
1. Max-RGB init: `T̂(x) = max_c L^c(x)`
2. Structure-aware refinement: `min_T ‖T̂−T‖²_F + α‖W∘∇T‖₁`, α=0.15
3. RTV-inspired weights `W`, σ=2. **Critical: `W` is computed ONCE from `T̂` and never updated — that's what makes LIME fast.**
4. **The fast solver:** replace the ℓ₁ with a reweighted quadratic → **one sparse SPD linear solve, zero iterations.**
5. `T ← T^0.8`; BM3D **on the Y channel only**; recompose `I_f = I∘T + I_d∘(1−T)` so bright regions keep detail and only dark regions get denoised.
Timings: LIME 0.44 s vs NPE 34.5 s.
> **⭐ `ximgproc::fastGlobalSmootherFilter` IS essentially step 4 already** — the fastest path is max-RGB init → FGS → gamma → denoise. Step 5's recomposition is exactly the noise-control mechanism §13.1 says you need.

**SRIE** (CVPR 2016) — the *pedagogical* piece. Its insight worth stealing: **the log transform is right for the fidelity term but wrong in the regularizer**, because `∇(log x) = (1/x)∇x` injects a spurious `1/x` weight that over-penalizes gradients in dark regions — destroying detail exactly where you need it. The `e^r`, `e^l` weights cancel it. Split-Bregman with **FFT-diagonalized derivative operators**, so no matrix is ever formed. Cleanest textbook demo of that solver template.

**RRM** (TIP 2018) — models `S = R∘L + N` with an **explicit noise map** as a decomposition variable, ALM-solved **without** log transform. ⚠️ **Adopt the idea (model the noise instead of amplifying it), skip the implementation** — LIME's recomposition captures ~80% of the benefit.

**Zero-reference deep:** ⭐ **SCI** (CVPR 2022 Oral) — cascaded illumination learning with weight sharing + a **self-calibrated module forcing convergence between stages, so at inference you run only a single basic block.** Best zero-data deep candidate; "model-irrelevant generality" means you could apply the self-calibration to your own LIME illumination estimate. **Zero-DCE++** (TPAMI) — **10K parameters, 11 FPS on CPU at 1200×900**; the only number here that sounds like an onboard budget. ⚠️ Zero-DCE was the *worst* enhancer for SuperPoint in §13.2.
**⛔ Don't bother:** RUAS (unreproducible NAS), **URetinex-Net and Retinexformer (supervised — paired data cannot exist for a collapsed building)**, all diffusion methods (inference cost + **generative hallucination is actively dangerous when the map feeds a rescue decision**), MSRCR (baseline only).

**⭐ FLARE-BO** ([arXiv:2604.22093](https://arxiv.org/abs/2604.22093), 2026 preprint, ⚠️ no venue) — **worth attention despite being unrefereed, because it is the closest published thing to what Ariadne's constraints imply.** Training-free **per-image Bayesian optimization with Gaussian Processes over 8 classical parameters**: gamma, **LIME-style illumination normalisation**, chrominance denoising, bilateral filtering, NLM denoising, gray-world AWB, adaptive post-smoothing. Explicitly framed for *"low-light robotic vision."* **Every one of its 8 components already ships in OpenCV 5** — no training data, no network, all classical primitives, tuned per-image. If the ablation in §13.6 shows classical baselines are competitive (which MID suggests), this is the natural next step.

> ⚠️ **Realism check on "reimplement and train from scratch in 2 months": only Zero-DCE / Zero-DCE++ and SCI are realistic** — small networks, no data collection, losses that are a few lines each. **RUAS is not** (its contribution *is* the architecture search). **URetinex-Net and Retinexformer are not** — and note the blocker there is **paired training data you cannot obtain**, not the code.

> ⚠️ **Venue correction — two CIDNet arXiv entries must not be conflated.** [arXiv:2402.05809](https://arxiv.org/abs/2402.05809) *"You Only Need One Color Space"* is **CoRR/arXiv ONLY per DBLP — no conference version; do not assert a venue for it.** [arXiv:2502.20272](https://arxiv.org/abs/2502.20272) *"HVI: A New Color Space for Low-light Image Enhancement"* is **DBLP-verified CVPR 2025** (doi 10.1109/CVPR52734.2025.00533). That is the published paper.

**⭐ Already in OpenCV 5 contrib and free:** `intensity_transform::BIMEF` (bio-inspired multi-exposure fusion — a complete, competitive low-light enhancer, zero implementation time). **Put it in the ablation from day one.**

### 13.6 The plan

> **Build the evaluation harness BEFORE building any enhancer.** Use LMS/FMF as cheap SLAM surrogates. **Measure trajectory and loop-closure outcomes, not inlier counts, and never PSNR/SSIM** — two independent papers show both mispredict SLAM outcome.

**Tier 1:** (1) baselines first — gamma, **global HistEq**, CLAHE, `BIMEF`; MID says global HistEq beat a SOTA CNN while CLAHE badly underperformed, **so measure rather than assume**. (2) **LIME**. (3) **IRAF-SLAM-style front-end adaptation instead of pixel enhancement.** (4) **Exposure and illumination control** — the largest measured effect in the entire literature.
**Tier 2:** one deep arm, SCI or Zero-DCE++. **Expect competitive with, not dominant over, gamma.**

> **⭐ The single most important caveat: treat "enhance, then SLAM" as a HYPOTHESIS TO TEST, not a design decision already made.** In moderate darkness enhancement was neutral-to-harmful; it helped hand-crafted ORB and hurt learned SuperPoint in the same experiment; and enhancers that *increased* match counts still eliminated loop closures entirely. **Ariadne's differentiating move is the one the strongest paper points to: optimize the light and the capture, and optimize enhancement jointly with the feature front-end — rather than optimizing pixels for human eyes.**

---

## 14. HDR under a point source, and photon-starved noise modelling

### 14.1 The intra-frame dynamic range problem

```
I(x) = R(x) · [Φ(ω)/d(x)²] · cos θ(x) · V(x) · t · g
```
1. **The dynamic range is intra-frame, not inter-frame.** A wall at 0.5 m and a corridor at 8 m differ by **(8/0.5)² = 256× — 8 stops — from falloff alone.** Add 10:1 albedo variation and you are at **11–12 stops within a single frame.** A 12-bit machine-vision sensor gives ~10–11 stops; an 8-bit JPEG pipeline ~8. **This is genuinely unwinnable by exposure choice alone at long ranges.**
2. **The illumination is not a scene property — it moves with you.** Everything Debevec/Robertson assume (a static radiance field sampled at different exposures) is false.
3. **⭐ But the exploitable structure: the illumination field is FIXED IN CAMERA COORDINATES.** Because the lamp is rigidly attached, `Φ(ω)·V(x)` is a **static function of pixel coordinates.** Only `1/d²` and `cos θ` move. **This is the single most important insight in this section and it is what makes calibration viable.**

### 14.2 ⭐ Flat-field calibration — do this

**What you CAN calibrate and remove:** the static camera-fixed part — lens vignetting `V(x)` and the lamp's angular beam profile `Φ(ω)`. Measure their product and divide it out. **This flattens the beam pattern and removes a systematic bias that otherwise makes your detector fire preferentially in the beam centre.**
**What you CANNOT remove without depth:** `1/d²` and `cos θ`. With stereo depth you *can* — which is exactly what the underwater literature does (§10.3).

**Recipe (~1 week):** dark room → image a **flat matte white target** fronto-parallel with only the robot's lamp → average heavily → normalize to max to get gain field `K(x) ∈ (0,1]` → at runtime, on the **linear** image (black-level subtracted, pre-gamma), `I_flat = I/K`. Repeat at 2–3 distances.

> ⚠️ **Two failure modes.** Dividing by a small `K` in the corners **amplifies noise by exactly `1/K`** — and your corners are already the photon-starved region. **Propagate the variance: after dividing, `Var → Var/K²`.** Clamp `K` at ~0.1. And **do this in linear space** — dividing a gamma-encoded image by a gain field is wrong and leaves residual structure.

**⭐⭐ DarkGS / NeLiS** (Zhang, Huang, Zhi, Johnson-Roberson, **IROS 2024**, [arXiv:2403.10814](https://arxiv.org/abs/2403.10814)) — **the closest published work to Ariadne's exact setup**: a robot carrying its own light, exploring in the dark, calibrating the camera-light system. Its empirical finding is decisive:
> *"when the range is close, light fall-off does not follow the inverse square law as the point light source assumption starts to fail. However, with a **Lorentzian model**… the light falloff can be better fitted with our introduced learnable parameter tau."*
>
> **⇒ DO NOT hard-code `1/d²`. They measured three real light sources and inverse-square fits poorly at close range. Fit your falloff curve empirically.** For Ariadne working in tight rubble at sub-metre ranges this is directly relevant.
**NeLiS** (the camera-light calibration half — the part Ariadne wants) is [MIT](https://github.com/tyz1030/neuralight); the DarkGS splatting half inherits the **Inria non-commercial** licence. **The useful half is the permissive half.**

**Underwater analogue:** Bryson, Johnson-Roberson, Pizarro, Williams, *J. Field Robotics* 2015 — an AUV with strobes at fixed geometry uses **range from SfM** to compensate falloff per-pixel. **The template — recover geometry first, then use the recovered range to invert the illumination model — is exactly the loop Ariadne can close, since it is already building a 3D reconstruction.**

### 14.3 ⛔ Exposure bracketing — keyframes only, not odometry

A 3-exposure bracket (1/8/64 ms) spans ~100 ms end-to-end. A ground robot at 0.5 m/s moves 5 cm; a drone at 2 m/s moves 20 cm; **30°/s yaw sweeps ~43 pixels** of global image motion at 90° HFoV on 1280 px.
- `cv::createAlignMTB` estimates **translation only, in integer pixels.** ⚠️ **It cannot model rotation or parallax — for a rotating drone it will fail.**
- ECC handles rotation but assumes planar scene or pure rotation — **parallax in rubble violates both.**
- Optical flow handles parallax but **is unreliable across a 64× exposure change, which is exactly when you need it.**
> ⚠️ **And the deeper problem: the long exposure is motion-blurred.** You cannot align your way out of blur, and fusing a blurred long exposure injects blur into exactly the dark far-field regions you were trying to recover.
>
> **Verdict: bracketing is a keyframe/mapping tool for a stop-and-stare mode, NOT an odometry tool.** For stop-and-stare it's excellent and cheap: `MergeMertens` + `AlignMTB`, ~3 days. **Frame it that way in the report.**

**Exposure Fusion (Mertens, PG 2007 / CGF 2009)** — three per-pixel quality measures (contrast via Laplacian, saturation via cross-channel std, well-exposedness via a Gaussian on 0.5), combined as `W = C^wc · S^ws · E^we`, then **blended per Laplacian-pyramid level.** **It skips radiometric calibration entirely** — no CRF, no exposure times, no radiance map. Fuses display-referred LDR straight to display-referred LDR. **Difficulty LOW, ~150 lines, 2–3 days.** ⛔ **Debevec and Robertson: don't bother** — they need a static scene, accurate exposure times, and a bracket, and **you don't actually need physical radiance for SLAM.**

**Single-image synthetic bracket:** apply several *fixed* gamma curves to one linear frame and fuse with Mertens weights. One frame, no alignment problem, no motion artifacts. ⚠️ **Fix the synthesis curves as constants — if gammas are chosen adaptively per frame you reintroduce the brightness-inconsistency problem.**

### 14.4 ⚠️ Tone mapping is a human-viewing concern, not a SLAM concern

**OpenCV 5 `photo` ships:** `TonemapDrago`, `TonemapReinhard`, `TonemapMantiuk`, `AlignMTB`, `Calibrate/Merge` × Debevec/Robertson/Mertens.
> ⚠️ **Correction most write-ups get wrong: OpenCV's `TonemapReinhard` implements Reinhard & Devlin 2005 (photoreceptor physiology), NOT the famous Reinhard et al. 2002 photographic operator.** Verified by resolving the `@cite RD05` key in `opencv/opencv:5.x/doc/opencv.bib`. **If you want the 2002 operator you must implement it yourself.**
> ⚠️ **`TonemapDurand` is NOT in `photo` in OpenCV 5** — it's in contrib `xphoto` and requires `OPENCV_ENABLE_NONFREE`. Treat as unavailable.

**Why tone mapping doesn't help the front-end:** ORB/FAST use intensity **comparisons**, invariant to any monotone map — so a **global** TMO changes essentially nothing. **Local** operators (Durand, Fattal, Reinhard dodge-and-burn) *do* change local gradients, but are **spatially adaptive in a scene-dependent way**, so the same 3D point gets a different mapping in two frames. **That actively hurts direct/photometric methods and perturbs descriptors** — the same mechanism as §13.1.
> **Use Drago/Reinhard for the operator's display, spend a day, move on. ⛔ Do not implement Fattal or Mantiuk.**

### 14.5 ⭐⭐ The Poisson-Gaussian model and the exact unbiased inverse

**This is the highest value-per-day item in this section, and the team will get it wrong by default.**

Photon arrival is Poisson: **variance equals mean.** With read noise added:
```
z = α·p + n,   p ~ Poisson(λ),   n ~ N(g, σ²)
⇒  Var(z) = a·E[z] + b,   a = α (system gain, DN per e⁻),  b = σ² − α·g
```
**Two numbers describe your camera's noise at a given gain, and you can measure them in an afternoon** — this is the classical photon transfer curve.

**Why it matters:** every standard denoiser (BM3D, NLM, wavelet, TV) assumes **AWGN of a single known sigma.** Ariadne's images violate that badly — the bright near-field is shot-dominated, the dark far-field is read-noise-dominated. **A single-sigma denoiser either over-smooths the bright regions (destroying the texture SLAM needs) or under-smooths the dark ones (leaving noise SLAM will track as features).**

**Generalized Anscombe Transform** (read out of the authors' reference code, not recalled):
```
f(z) = (2/α)·sqrt( α·z + (3/8)·α² + σ² − α·g )     → output has variance ≈ 1 everywhere
```

> ### ⚠️⚠️ THE BIAS TRAP — and it is largest exactly where Ariadne operates
> A good denoiser applied to `f(z)` returns `D = E{f(z)|y}` — the conditional mean of the **transformed** variable. It does **not** return `f(y)`. Because `f` is **concave**, Jensen gives `E{f(z)} < f(E{z})`. So the naive algebraic inverse `(D/2)² − 3/8` **systematically underestimates `y`.**
> **The bias is negligible when `y` is large and grows severely as `y` falls toward a few counts per pixel — precisely Ariadne's far-field regime.** The consequence is a spatially-varying, signal-dependent **darkening of exactly the dim regions you are trying to recover**, which biases photometric computation and suppresses genuine faint structure.

Three inverses, verbatim from the reference `.m` files:
1. **Algebraic (naive):** `(D/2)² − 3/8`. **Biased. Do not use.**
2. **Asymptotically unbiased:** `(D/2)² − 1/8`, clamped to 0 below `D = 2√(3/8) ≈ 1.2247`. **⭐ The constant is 1/8, not 3/8 — this one change removes the leading-order bias and costs nothing. If the team does only one thing, do this.**
3. **⭐ Closed-form approximation to the exact unbiased inverse — the pragmatic choice, a handful of flops:**
```
I(D) = (D/2)² + (1/4)·sqrt(3/2)·D⁻¹ − (11/8)·D⁻² + (5/8)·sqrt(3/2)·D⁻³ − 1/8
```
4. **GAT version:** same polynomial **minus σ²**, with scaling undone after. ⚠️ **Parameter bookkeeping is the most likely bug: σ is divided by α BEFORE the polynomial, and the result is multiplied by α and offset by g AFTER.**

**Papers:** Mäkitalo & Foi, IEEE TIP 20(1):99–109 2011 · TIP 20(9):2697–2698 2011 (closed form) · **TIP 22(1):91–103 2013 (generalized, Poisson-Gaussian)**.
**Difficulty: LOW — 2–3 days. `core` element-wise ops only.**

**Estimating `(a,b)` from your own camera** (Foi et al., TIP 2008 — the paper *"explicitly take[s] into account the clipping of the data"*, which matters because with a point source you will have clipped near-field pixels in most frames, and clipped pixels have artificially *reduced* variance): capture linear raw at fixed gain → subtract black level (**do not clamp negatives — that biases the variance**) → local (mean, std) pairs from detail-band coefficients where the signal is locally smooth → bin by mean into level sets → **robust MAD per bin** → fit `σ² = a·mean + b`.
> **⭐ The best sanity check available: `a` should scale LINEARLY with analog gain and `b` QUADRATICALLY. Check at 2–3 gain settings before trusting anything. If it doesn't hold, something upstream is non-linear and the whole model is invalid.**
**Difficulty: LOW for the level-set + robust-MAD + linear-fit version (2–3 days); MEDIUM for the full ML fit with clipping.**

**The pipeline:**
```
1. Capture RAW (linear, 12-bit, no gamma/WB/sharpening/denoise)
2. Subtract black level (do NOT clamp to 0 yet)
3. Estimate (a,b) once per gain setting, offline
4. Apply GAT              → output has unit variance everywhere
5. Denoise as AWGN σ=1    → any Gaussian denoiser; start with photo::fastNlMeansDenoising
6. Exact unbiased inverse GAT (closed form)
7. Divide by falloff field K(x) from §14.2, propagating Var → Var/K²
```
> **⚠️ Ordering matters: do the noise handling BEFORE the falloff division.** Dividing by a small `K` first amplifies noise by `1/K` and destroys the affine variance relation the VST depends on — the model becomes `Var = a·I/K + b/K²` with spatially-varying coefficients. **Denoise in the domain where the model is valid.**

**⚠️ Why 8-bit gamma-encoded JPEG breaks this entirely.** Under a nonlinear map, `Var(y) ≈ [G'(z)]²·Var(z)` — and gamma's `G'` is steep near black and shallow near white, so it **massively amplifies apparent shadow noise.** The variance-vs-intensity curve is no longer affine, `a` and `b` no longer exist as constants, and **the GAT stabilizes nothing.** Plus: 8-bit quantization erases Poisson structure in the far field; **lossy DCT introduces spatially-correlated noise, and every method here assumes WHITE noise**; in-camera denoising and sharpening make you model the ISP rather than the sensor; WB and demosaicing correlate neighbours.
> **Doing this on JPEG doesn't reduce accuracy — it invalidates the derivation. If forced onto 8-bit, abandon the VST route and use a plain tuned denoiser, and say so.**
>
> **✅ Good news: for machine-vision cameras (Basler, FLIR, Allied Vision), linear 12/16-bit output with gamma off and no hidden ISP is the DEFAULT, not an exotic mode** — that's what people buy them for. Verify: pixel format 12/16-bit; gamma = 1.0 (some SDKs default to 0.45); denoise/sharpen/AWB off; black level known; **and for Bayer colour, fit `(a,b)` per channel on the un-demosaiced CFA data — demosaicing correlates neighbours and breaks whiteness.** **This consideration alone justifies a machine-vision camera over a webcam**, and linear raw is exactly what the falloff division and the exposure controller both need — one capture config serves all three subsystems.

⛔ **Don't bother:** PURE-LET (more work than GAT + off-the-shelf denoiser for comparable results — mention as the road not taken), non-local PCA, **all Quanta Image Sensor work** (specialized single-photon hardware; algorithms don't transfer — read the HDR-QIS paper for framing only).

> **2.2 as built (2026-10-21; D65–D68, `docs/design/RADIOMETRIC_2.2.md`). Insert; the text above is unchanged.**
> - **Source and licence.** GAT and the exact unbiased inverse are implemented **from the thesis equations**
>   (Mäkitalo, Tampere UT 2013, eqs. 3.7, 4.3–4.14, 4.37). The invansc `.m` and `.mat` files are under a non-commercial
>   licence; they were read only to cross-check formulas, and nothing of theirs ships.
> - **Correction to item 3 above.** The `D < 2√(3/8) → 0` branch of the closed form is the thesis' (§3.2.2, below
>   eq. 3.5). The reference `.m` omits it.
> - **Evidence.** Our eq.-4.10 tables reproduce the paper's accuracy figures:
>
>   | quantity | ours | paper |
>   |---|---|---|
>   | closed form, L∞ (4.40) | 0.0470 | 0.0468 |
>   | closed form, weighted L2² (4.39) | 0.00690 | 0.0069 |
>   | I0 − σ², L∞ (4.36) | 0.0470 | 0.0470 |
>   | I0 − σ², weighted L2² (4.35) | 0.00295 | 0.0028 |
>
>   An independent exact-Poisson Monte Carlo passes the exact inverse and fails the algebraic one at low counts.
> - **The (a, b) fit** is the level-set + robust-MAD + WLS version [PROXY: no ML fit, no clipped likelihood; clipped
>   neighbourhoods are excluded]. The gain check holds (a doubles with gain).
>   - b is the weak term: 0.82 vs 1.12 DN² at the deepest gain-up config.
>   - The stabilised-domain check σ_z ≈ 1 (RPG-VST, arXiv 2607.24291) is the gate's real test of the fit.
> - **Ordering nuance (refines the ⚠️ above).** "VST before ÷K" is right when K acts on the light before photon
>   counting: real optics. The simulator's D applies PRNU and vignette AFTER the noise. There, raw
>   `var = K·a·m′ + K²·b`, and ÷K FIRST restores one model. So 2.2 flat-fields first, in the sim only. D's order is
>   an open question for Phase 5.
> - **The 8-bit caveat, measured.** Linear 8-bit (D's output) keeps the model valid wherever it isn't posterised:
>   σ_z ≈ 1 at gain-up. At unity-gain darkness it **is** posterised, and the measured-noise gate refuses the VST
>   there, as the ⚠️ says it must.
>   - The 8-bit hand-off after denoising also re-quantises without dither. The inverse choice is then invisible below
>     ≈ α/4 DN.
> - **Colour.** RGB → per-channel noise → BT.601 gray keeps the mean and scales the variance by Σw² = 0.447. Fit (a, b)
>   on the image the denoiser actually sees.
> - **OpenCV 5.0.0 facts.**
>   - `fastNlMeansDenoising` accepts 8U/16U only: the stabilised image is scaled into 8U. It costs ≈ 236 ms at
>     1280×720 on 8 threads.
>   - `xphoto::bm3dDenoising` is refused in our build (non-free off).
>   - `GaussianBlur` 5×5: 0.8 ms.

### 14.6 Priority

1. **GAT + exact unbiased inverse** — 2–3 days, highest value.
2. **`(a,b)` photon-transfer calibration**, validated by the gain-scaling check — ~1 week.
3. **Static beam/vignetting `K(x)` calibration**, informed by DarkGS's Lorentzian finding — ~1 week.
4. **Active exposure control** (§9.4, §13.4) — ~1 week.
5. MAD/wavelet sigma estimator as a diagnostic — 1 day.
6. Exposure Fusion for stop-and-stare keyframes — 3 days.

---

## 15. Alternative sensing modalities — a well-argued "why not"

> **⭐ Writing this section well is itself a stronger contribution than a half-finished radar implementation.** The numbers below make it defensible.

| Modality | Verdict | The one-line reason |
|---|---|---|
| **Active stereo / ToF** | ✅ **PRACTICAL** | **$334 D435i / $429 OAK-D Pro project their own IR texture — fixes darkness AND blank concrete with one component, zero papers to reimplement** |
| LiDAR in dust | 🟡 **MARGINAL** | Filters are cheap and dual-return is a real fix, but it's the priciest sensor and the one dust hurts most; only justified if you need >6 m or 360° |
| Radar / mmWave | ⛔ **NOT PRACTICAL** | Tens of points/frame at 15° resolution, and **60 GHz does not penetrate concrete**; the good papers need a five-figure scanning Navtech |
| Polarization | ⛔ **NOT PRACTICAL** | **Throws away >50% of light in a scene defined by having almost none**, and the dehazing model needs ambient airlight you don't have |
| Hyperspectral | ⛔ **NOT PRACTICAL** | Five-figure, sunlight-scale integration times, answers a materials question you're not asking |
| Acoustic | ⛔ **NOT PRACTICAL** | Deployed tech is a **contact geophone + trained human operator**; rotor noise kills it on a drone; no pipeline output |

### 15.1 ⭐ Active stereo is the answer — and it solves both problems at once

Ariadne has two vision-killers: **total darkness** and **texture-poor concrete.** Passive stereo fails on both. **An IR-projector stereo camera fixes both with the same component** — the projector paints pseudo-random IR dots, giving the matcher texture *and* supplying its own photons.

> **⭐ The counter-intuitive fact to state flatly: a RealSense D435i works in complete, pitch-black darkness.** The depth pipeline runs on two IR imagers plus the projector and does not care about visible light. **Depth often works *better* in the dark, because there's no ambient IR washing out the pattern.** You keep the lamps for RGB texture and reconstruction appearance — but **geometry keeps working even when the lamp points elsewhere.**

**Status verified 2026-08-29:** the 2021 Intel EOL **explicitly excluded stereo** (*"Intel WILL continue to sell and support stereo products including… D435, D435i, D455…"*), RealSense is now a **standalone company** (RealSense, Inc., under licence from Intel), and the store is live: **D415 $272 · D435 $314 · D435i $334 · D455 $419**, all in stock, with *new* models (D585 Pro, D555 PoE, D456, D436) in the catalogue. **Frame it as "spun out and continuing," not a reversal.**
**D435i:** 87°×58° FOV, **ideal range 0.3–3 m**, Min-Z ~28 cm, **<2% depth error at 2 m**, global shutter, IMU. **The ~3 m range is the real limit** — room- and corridor-scale, not large-atrium.

**⭐ OAK-D Pro — $429, in stock** — arguably better suited: it carries **both a dot projector** (depth on blank concrete) **and a flood IR illuminator** (*"enables running your AI and CV algorithms in low-light or no-light environments"*), plus onboard inference and an **IP66 enclosure** — not decorative in a dusty building, and the RealSense is not sealed.

**⚠️ Say this explicitly, because someone will suggest it: the Stereolabs ZED is passive stereo — no projector, no illuminator. It will produce nothing usable in a dark building and nothing usable on blank concrete even if lit. It is the single worst choice on this list for this environment.**

**Dust behaviour is *softer* than LiDAR's:** you get **holes** (invalid depth) rather than confident phantom surfaces. **Holes are far easier to handle downstream than lies**, and the left-right consistency check does a lot of free work.
**⚠️ Multi-device interference is a genuine multi-agent engineering issue** — overlapping dot patterns degrade matching. Mitigate by hardware-syncing and alternating projector frames, or separating viewing volumes. **It's also an interesting, publishable sub-problem for a multi-robot project.**
**Simulator note: model the IR projector and its dropout.** Simulating a perfect depth camera makes the sim results dishonest — model the 0.3–3 m valid range, holes on specular/absorbing surfaces, and dust dropout.

**Azure Kinect DK: discontinued** (docs archived under `previous-versions`, flagged `is_archived: true`; the Oct 2023 date is secondary-source). ⚠️ **Multipath is the ToF-specific weakness and a collapsed room is the worst case imaginable** — light bounces corner-to-corner and yields depths that are wrong in a *smooth, plausible-looking* way. **Active stereo is more robust: triangulation has no phase to corrupt.** Successor Orbbec Femto Bolt ($418) / Mega ($695) — **both out of stock as of 2026-08-29, and on a two-month clock a sold-out sensor is a non-sensor.**

### 15.2 The rebar physics — get this right, it's counter-intuitive

| Band | λ free-space | λ in concrete (ε_r≈6) | vs 100–300 mm rebar spacing |
|---|---|---|---|
| 400 MHz GPR | **750 mm** | ~300 mm | λ ≥ spacing → **mesh acts as a screen** |
| 1.5 GHz GPR | 200 mm | ~80 mm | comparable → strong rebar clutter |
| **60 GHz mmWave** | **5.0 mm** | ~2 mm | λ ≪ spacing by 30–60× → **mesh is irrelevant** |

> **⚠️ The naive conclusion — "60 GHz slips through the mesh, so it penetrates" — is EXACTLY BACKWARDS, and this is the most important thing to get right in the report.** At 60 GHz the rebar isn't the limiting factor; **bulk dielectric loss in the concrete is**, and it scales badly with frequency (order 100 dB through a typical wall; exact figure ⚠️ unverified). **60 GHz mmWave does NOT see through concrete — it is a surface sensor.** GPR *does* penetrate, which is precisely why GPR is a 400 MHz–2 GHz instrument dragged along the ground, and why rebar mesh is GPR's famous nemesis rather than its ally.

**And the sensor-class bait-and-switch:** chip-scale mmWave (TI IWR6843) gives **tens of points per frame at ~15° azimuth resolution** — a detection sensor, not a reconstruction sensor. The radar-SLAM papers worth reimplementing (CFEAR, RadarSLAM, Oxford/MulRan/Boreas) assume a **Navtech CTS350-X: a rooftop-mounted, kilogram-scale, five-figure vehicle sensor.** *"They do not transfer to a $300-class TI board."* ⚠️ Licences: `christopherdoer/rio` is **GPL-3.0**; CFEAR code is **CC BY-NC 4.0**. **Do not read either while writing your own version if Ariadne is permissive.**

**NASA JPL FINDER** — microwave radar detecting *"miniscule motions of the body caused by processes innate to living things, such as heartbeats or respiration"* (Jim Lux: *"Your body moves a millimeter when your heart beats. Because the rubble itself isn't moving, we can separate those motions out."*). **Cleanly out of scope** — it outputs "there is a heartbeat ~N metres that way," no image, no cloud, no pose, and it's a licensed commercial instrument, not a dev board. **Belongs in related work as motivation, nowhere in the sensor plan.** ⚠️ Its exact band is unverified (don't write "L-band" as fact) and the Nepal 2015 deployment is secondary-source.

### 15.3 LiDAR in dust — the mechanism, and the 905 vs 1550 nm myth

**Dust doesn't "add noise" — each particle is a legitimate reflector returning a genuine, correctly-timed pulse.** Four consequences, worst first: **(1) spurious near-field returns** — in first-return mode the sensor reports the *dust*, growing a phantom surface right in front of the robot (for a drone running avoidance off that cloud, a stop-forever failure or a crash into the wall it never saw); (2) two-way extinction collapses effective range; (3) intensity collapse breaks reflectivity thresholds; **(4) the dust cloud MOVES, so consecutive scans disagree and ICP/NDT diverges.**
> **Point 4 is underestimated: a camera in dust gives you a *blurry* image you can sometimes still register. A LiDAR in dust gives you a *confidently wrong* geometric surface.**

**⭐ Multi-echo / dual return is the single best mitigation** — the dust gives an early weak echo, the wall a later stronger one. **Last return** discards the dust directly; **dual return** lets your filter decide, because **the existence of a second later echo along the same ray is itself evidence the first is an aerosol.** Velodyne, Ouster and Hesai expose these modes. ⚠️ Livox Mid-360 multi-return support unverified.

> **⚠️ "1550 nm sees through fog/dust better" is NOT supported by measurement.** Duthon, Colomb & Bernardin, *Applied Sciences* 9(14):2843, 2019, controlled fogs at Cerema PAVIN, verbatim: *"the wavelengths for LiDARs should not be chosen on the basis of fog conditions: there is a **small difference (<10%)** between the extinction coefficients at 905 nm and 1550 nm for the same emitted power in fog."*
> **1550's real advantage is the eye-safety power budget, not the obscurant physics** — the cornea absorbs 1550 so Class-1 permissible exposure is far higher, letting you emit more power. **A power-budget win dressed up as a wavelength win.** Water absorption cuts the *other* way at 1550. **And for dry mineral dust — your actual case — neither wins:** particles are ≫ both wavelengths, so extinction is essentially wavelength-independent. **There is no wavelength you can buy your way out of dust with**, and 1550 needs InGaAs detectors, putting it outside a student budget.

**Filters (all cheap):** **DROR** (Charron, Phillips & Waslander, **CRV 2018**) scales the search radius with range because point density falls with distance — ~50 lines over a PCL KD-tree, 2–5 days, and the **first author's own implementation is MIT** (`nickcharron/lidar_snow_removal`), a gift for a reimplementing team. **LIOR** (*IEEE Access* 2020) gates on intensity first — aerosol returns are weak — then applies geometry only to survivors. **DSOR** (arXiv:2109.07078, ⚠️ **preprint, don't assign a venue**) claims higher recall than DROR at **28% faster**. PCL's ROR/SOR are free and BSD-3.
> ⚠️ **All of these were developed and validated on falling SNOW — large, sparse, well-separated particles. Dense airborne concrete dust is a harder regime**, and a filter tuned to delete isolated low-density returns will happily delete the real wall when the cloud is thick. **Assume partial relief, not a solution. The multi-echo last-return trick is physically stronger than any point-statistics filter, because it uses the ray rather than the neighbourhood.**

**The discriminator:** *does Ariadne need range beyond ~6 m or 360° coverage a depth camera can't give?* Exploring room- and corridor-scale voids → **no, and LiDAR is a liability.** Mapping a 30 m collapsed atrium → **it becomes the only option and you scope the project around it from day one, not in month two.**

### 15.4 Polarization — the photon argument is disqualifying on its own

**⚠️ The Schechner dehazing line models scattered *ambient* airlight, partially polarized because it is sunlight scattered through a large atmospheric volume. Your scene has NO ambient light at all** — every photon came out of a lamp on your robot. **The algorithm's core assumption is simply absent.** The version that does work is **cross-polarized active illumination** (polarize the lamp, orthogonal analyser on the camera) — real and proven, but it costs another large factor of light.

**The photon cost:** a Sony IMX250MZR micro-polarizer array **loses at least half the energy by definition** (Malus' law averaged over angle — not an engineering imperfection), **plus 4× spatial resolution**, plus QE from the filter stack.
> **The defining constraint of Ariadne is too few photons. You are proposing to throw away more than half of the photons you paid battery capacity to emit.** Compensating means longer exposure (motion blur on a moving drone — fatal) or higher gain (noise swamps the small polarization signal — self-defeating). **This modality trades your scarcest resource for information you cannot afford to buy.**
*(Note this is a harsher verdict than §10.6's, which was written before the photon budget was quantified. §10.6's suggestion — borrow a polarization camera for one afternoon of dust-box testing — remains the sane way to settle it empirically for near-zero cost.)*

### 15.5 Acoustic — what USAR actually uses

The operational tool is a **seismic/acoustic listening device** (canonically the **Delsar LifeDetector**), which *"converts the entire collapsed structure into a large sensitive microphone."* **The physics is the point: geophone sensors in direct mechanical contact with the rubble, listening for tapping conducted through the structure — not an airborne-acoustic sensor, and fundamentally human-in-the-loop** (a trained operator triangulates by moving sensors and listening on headphones).

Robot-mounted audio detection is a real but small research area — note **Zimroz et al., *Energies* 14(13):3725, 2021**, *"Application of UAV in Search and Rescue Actions in Underground Mine — A Specific Sound Detection in Noisy Acoustic Signal"*, which is **precisely Ariadne's scenario** and whose actual subject is **rejecting the drone's own noise.** A quadrotor is a loud broadband noise source with its microphones bolted to the noise source; the faint tapping sits at the bottom of the dynamic range; **and the geophone-contact trick that makes Delsar work is unavailable to a flying robot.**
> **Victim detection is the *motivation* for Ariadne, not its deliverable. The contribution is the map and the exploration policy — handing rescuers an accurate 3D model of the void structure is what makes the difference. Cite Delsar and these papers in related work as the thing your map gets handed to, and move on.** (Contrast §12.4: a $164 thermal camera *does* give a cheap, on-mission victim-detection deliverable — that is the version worth building.)

---

## 16. Datasets, part 3 — night driving & low-light restoration

> **⭐ The grading axis that matters: streetlit urban night is a *static-illumination* problem and DOES NOT TRANSFER.** The physics that transfers is **co-moving illumination** — a light rigidly attached to the sensor, so a fixed world point brightens as you approach it (inverse-square × beam pattern), the hotspot tracks the camera, and **brightness constancy between frames is violated.** Every set below is graded on that axis.

### 16.1 ⭐ The two cheapest useful downloads — ~27 GB total, both registration-free

**1. RobotCar Seasons `night.zip` — 368 MB, direct HTTPS, no forms**
`https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/images/night.zip` · CC BY-NC-SA 4.0
**The single cheapest real night imagery in existence with reference poses attached.** 1024×1024 mono Grasshopper2 frames, heavy motion blur, real read noise, blown highlights — feature detector and matcher running against genuine night data **within an hour**, with poses to check matches are geometrically consistent rather than eyeballing them. Add `overcast-reference.zip` (3.6 GB) for the day-night matching pair.
⚠️ **Streetlit, so it does NOT test co-moving-light physics — purely a sensor-noise and feature-stability smoke test.**
*(Aachen Day-Night v1.1 is the same benchmark family: 6,697 reference / 1,015 queries, **824 day + 191 night**. ⚠️ Note its night queries were shot with **software HDR** and are deliberately well-exposed — the *easy* case, much less representative of a dark sensor.)*

**2. ⭐⭐ SID Sony — 26.9 GB, direct from Google Cloud Storage, no forms**
`https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip` (26,926,662,016 bytes verified) · Fuji subset 55.4 GB · code MIT
> **The ONLY set in either bucket that gives you unprocessed sensor RAW with real read and shot noise at true low photon count** — no demosaic, no denoise, no tone curve, no auto-exposure hiding the problem. **Every other dataset hands you an ISP's opinion of the darkness.** If the question is "does my low-light front-end survive real sensor noise," SID is the honest test and everything else is a proxy.
>
> **This is also the validation set for §14.5's GAT pipeline** — you cannot calibrate `(a,b)` or verify the exact unbiased inverse on gamma-encoded 8-bit data, and SID is where you get linear RAW with matched long-exposure ground truth. **Start with Sony; skip Fuji unless you need X-Trans CFA handling.**
> ⚠️ **Maintainer's Sept 2025 notice: *"We got significant bills from Google storage. Please download the data manually and save a copy locally. Avoid re-downloading the data per experiment run. The download script is removed."* Download once and cache — the authors pay the bandwidth personally.** Known erratum: exclude images 10034, 10045, 10172 (ground-truth misalignment).

### 16.2 ⭐ Boreas — the best real co-moving-headlight data with metric ground truth

**Burnett et al., IJRR 2023** · https://www.boreas.utias.utoronto.ca/ · **open S3, `--no-sign-request`, no registration** · ✅ **CC BY 4.0 — one of the very few permissive licences in this whole corpus**

```
aws s3 sync s3://boreas/boreas-2021-09-08-21-00 . --exclude "*" \
    --include "camera/*" --include "applanix/*" --no-sign-request
```
**Measured directly against the bucket:** `boreas-2021-09-08-21-00/camera/` = **11,410 images, 16.25 GB**; `boreas-2021-11-06-18-55/camera/` = **12,034 images, 17.31 GB**.
> **⭐ Direct empirical evidence these frames are genuinely dark, not dusk: mean JPEG size is 0.77–0.82 MB at night vs 2.98 MB in the daytime comparison sequence — a 4× compression drop.**

**Sensors:** FLIR Blackfly S 5 MP mono **2448×2048 @ 10 Hz**; 128-beam Velodyne Alpha-Prime; Navtech CIR204-H radar; Applanix POS LV.
**GT:** post-processed 6-DoF at **200 Hz** from Applanix POSPac batch optimization (GPS+IMU+wheel+RTX), fixed ENU frame — **centimetre accurate.**
**Night sequences identified:** `2021-06-29-20-43`, `2021-09-08-21-00`, `2021-09-14-20-00`, `2021-11-06-18-55` (Toronto sunset on 2021-09-08 is ~19:45 and on 2021-11-06 ~17:55, so the 21:00 and 18:55 runs are fully dark).

> **Why it's the strongest physics match in this bucket: the Boreas route includes farm and forest segments that are UNLIT, so at night the only illumination is the car's own headlights — a co-moving source with inverse-square falloff and a hotspot that moves with the camera. Exactly the brightness-constancy violation Ariadne's agent-carried lights produce, with centimetre-accurate poses to validate against.** Scene geometry is open outdoor road, nothing like a collapsed interior — but the *photometry* is right, and 16 GB is very realistic.

### 16.3 The rest — and what to skip

**⭐ Dark Zurich — 214.5 MB for the val set, and one concept worth stealing.** Project page is **NXDOMAIN** (TRACE group renamed) but **the data is still live** — verified HTTP 200 today at `https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/`: `Dark_Zurich_val_anon.zip` **214,514,882 bytes** (50 annotated night + 50 day counterparts), `train_anon.zip` **16.63 GB** (8,377 images: 3,041 day / 2,920 twilight / 2,416 night), `test_anon_withoutGt.zip` 597.5 MB. No registration for the anonymised versions; ⚠️ no explicit licence — treat as research-use with no redistribution right.
Streetlit, so wrong physics — **but two things justify 215 MB: the images are genuinely deeply dark with real noise, and the invalid-region masks are a concept Ariadne should steal outright.** They mark exactly where darkness has destroyed recoverable signal — **precisely the "where do I not trust my front-end" question a rubble-exploring robot has to answer, and here it comes with human-verified ground truth.** (Annotated using a "privileged information" protocol: the corresponding *daytime* image assigns reliable labels to regions indiscernible in the night frame.)

**4Seasons** (TUM/Artisense, GCPR 2020) — structurally the most Ariadne-like of the driving sets on two counts: **countryside and highway** (unlit → true headlight-only) *and* **a multi-level parking garage — the only genuinely enclosed, GNSS-denied, artificially-lit interior geometry anywhere in this bucket.** Basler acA2040-35gm **mono global-shutter, 30 cm baseline, 30 fps** — closer to a robotics camera than a dashcam. GT centimetre-accurate from stereo DSO + RTK-GNSS.
⚠️ **Registration required (GDPR), CC BY-NC-SA 4.0, and — flagging loudly — the `www.4seasons-dataset.com` homepage is currently serving SEO-spam (injected German online-casino copy). The `/dataset` and `/documentation` pages still carry real content, but treat the domain as possibly compromised and verify anything you download.**

**ACDC** (ICCV 2021) — **1,006 nighttime images**, each **paired with a normal-condition image of the same scene** plus a binary invalid-region mask. The day/night correspondence pairs are a genuinely useful asset: same scene, same viewpoint, lit vs unlit, letting you measure directly what your front-end loses to darkness. ⚠️ But it's a *segmentation* dataset — no poses, no sequences, no VO evaluation — and a rolling-shutter consumer GoPro with aggressive in-camera processing is a poor proxy for raw sensor noise.

**⛔ Do not spend time on:**
- **nuScenes** — night is **11.6% of keyframes** (~116 of 1,000 scenes), but **night scenes are scattered across all ten trainval blob parts; there is no night-only archive.** You'd pull the entire trainval set for streetlit urban with auto-exposure that normalises away the falloff signal. **Disqualifying on the download structure alone.**
- **Ithaca365** — right physics (rural upstate-NY roads, explicitly *"including night"*), **but 59 GB per camera per traversal** (`01-16-2022/cam0.tar.gz` = 59,053,852,735 bytes verified). **Boreas gives the same physics for 16 GB with better pose GT and a permissive licence.**
- **⚠️ BDD100K — all official distribution channels are DOWN**, verified from two independent network paths: `doc.bdd100k.com` **NXDOMAIN**, `www.vis.xyz` **NXDOMAIN**, `bdd100k.com` timeout, `bdd-data.berkeley.edu` TLS cert mismatch + **403**. Unofficial HF mirrors exist but aren't authoritative. Even if obtained: **no pose, no calibration, and 720p H.264 compression that destroys the very sensor-noise statistics you'd be validating.**
- **Nighttime Driving** (ITSC 2018) — project page 404s, download link 404s, **and the training set was never released** (standing GDPR note). Superseded by Dark Zurich. *Listed only so nobody burns an afternoon rediscovering the 404.*
- **Oxford RobotCar** itself — 8 night traversals but **~200–220 GB each** (chunk-granularity pulls of ~5.5 GB are possible), registration required, and **Oxford city centre is comprehensively streetlit so headlights aren't the dominant source.** For 2014-era CCD night noise, take the 368 MB `night.zip` instead — same platform, same imagery family, 1/15th the download, no registration.

### 16.4 Low-light restoration sets — licence and availability notes

**LOL** (BMVC 2018) — ~500 pairs, Google Drive/Baidu only, **no size and no licence stated**. ⚠️ **LOL-v2 has no canonical repo** — widely-used copies are third-party HF mirrors; **the authoritative source is unverified.**
**SICE** — 589 multi-exposure sequences, **no LICENSE file at all** (all rights reserved by default).
**✅ ExDark** — **BSD-3-Clause verified**; the cleanest licensing story here.
**MIT-Adobe FiveK** — ~50 GB, 5,000 DNG RAWs + 5 expert retouches, Adobe **research licence**: *"solely for your own research purposes… not… intended for or directed toward commercial advantage."*
**SMID/DRV** (ICCV 2019) — `long.zip` 11.9 GB, `VBM4D_rawRGB.zip` **132 GB**, MIT. ⚠️ README warns *"The proposed method is designed for sensor raw data… It also does not work for images after camera ISP"* — **another independent confirmation of §14.5's RAW requirement.**
**⚠️ ELD** (CVPR 2020 Oral) — MIT code, evaluation-only data across four camera bodies, **but the physics-based noise model is WITHHELD**: *"Due to the business license, we are unable to provide the noise model as well as the calibration method."* **If you want their calibrated Poisson-Gaussian model you must reimplement it from the paper — which is exactly what §14.5 describes how to do.**
**LSRW** (R2RNet) — **Baidu Pan only** (code `wmrr`), no LICENSE file.

> **None of these five headline sets have pose, sequences, or co-moving light** — they are single-image benchmarks, useful only as a static sanity check.

### 16.5 The plan

1. **RobotCar Seasons `night.zip` (368 MB)** — sensor-noise and feature-stability smoke test, first hour.
2. **SID Sony (26.9 GB)** — the honest RAW noise test, and the validation set for the §14.5 GAT pipeline.
3. **Then, and only if those pass, one Boreas night sequence (16 GB)** — the only cheap dataset here where the light genuinely co-moves with the camera, with 200 Hz cm-accurate poses and CC BY 4.0.

> **⚠️ Honest caveat on all of §16: none of these have collapsed-interior geometry — no confined spaces, no close-range surfaces, no rubble, no dust. What transfers is (a) real sensor noise at low photon count and (b) in Boreas and 4Seasons only, the co-moving illumination photometry. The scene structure has to come from the simulator regardless — which is, again, why the simulator track exists.**

---

## 17. Texture-poor matching & perceptual aliasing

✅ **Record now complete.** §17.1–17.5 are the researcher's original summary plus independent verification of its central claim. **§17.6–17.12 are the full detail, since added** — the classical aliasing-robust back-ends with their actual formulations (§17.6–17.10), SeqSLAM as an implementable algorithm (§17.11), and the diagnostics that measure aliasing in our own environment (§17.12).

### 17.1 ✅ CORRECTION — the line front-end is essentially free in OpenCV 5

**§10.9 recommended `line_descriptor` for corridors on the assumption LSD was unavailable. That was wrong, and the correction is favourable.**

**LSD is not missing from OpenCV 5.** It was disabled in 2019 over the original code's licence ([issue #14576](https://github.com/opencv/opencv/issues/14576)), then **clean-room reimplemented in PR #20797, merged 2021-10-02**, under a BSD header. Independently verified against the `5.x` branch just now:

| Component | Location | Verified |
|---|---|---|
| `lsd.cpp` | `modules/imgproc/src/` on `5.x` | ✅ present |
| `createLineSegmentDetector`, `LineSegmentDetector`, `LSD_REFINE_*` | `modules/imgproc/include/opencv2/imgproc.hpp` | ✅ 1 / 11 / 5 mentions |
| `EdgeDrawing::detectLines()` (**EDLines**) | contrib `ximgproc/edge_drawing.hpp` | ✅ present |
| `createFastLineDetector` (Lee14) | contrib `ximgproc/fast_line_detector.hpp` | ✅ present |
| LBD descriptor + binary matcher | contrib `line_descriptor` | ✅ (§10.9) |

⚠️ Open PR #29349 (June 2026) proposes **moving LSD to `ximgproc`** — pin your include path and check on upgrade.

> **⇒ Line detection, line description and line matching are all shipping, permissively-licensed OpenCV 5 code. Reimplementation budget belongs on 3D line *geometry* — line triangulation, Plücker-coordinate parameterization, line-based PnP and BA residuals — not on the 2D front-end.** For textureless corridors and stairwells (§10.9's failure mode) this is the cheapest available win.

### 17.2 ⚠️⚠️ IMPORTANT CORRECTION — PCM has a blind spot on exactly Ariadne's failure mode

§1.7 and §7.8 recommend a three-layer stack: **USAC → PCM → GNC.** That still stands, **but with a caveat that changes how much weight PCM can carry:**

> **Perceptual aliasing creates large, mutually-consistent, WRONG cliques.** PCM's premise is that inliers are mutually consistent and therefore form a clique — but when a corridor genuinely looks like another corridor, a whole set of false loop closures is *self-consistent too*, and it can form a clique as large as the true one. **Max-clique selection will confidently pick it.**
>
> **This is not a corner case for Ariadne — it is the defining hazard of texture-poor, self-similar rubble (§2.2).** PCM was already flagged in §1.7 as a pre-filter rather than a sole mechanism because of its runtime; **this is a second, independent, and more serious reason.** Do not let PCM be the load-bearing layer. **GNC (§1.3) and the certificate (§7.4) are what actually protect you**, and the certificate is especially valuable here because a wrong-but-self-consistent solution is precisely the case where a *provable* global-optimality check earns its keep.

**Classical aliasing-robust back-ends worth knowing** (beyond §1.3's ladder): **switchable constraints**, **dynamic covariance scaling**, **max-mixtures**, and **realizing, reversing, recovering (RRR)**. These attack the problem at the optimizer rather than at the data-association stage, which is the right layer when the data association is *self-consistently* wrong. **⇒ Worked out with their actual formulations, and ranked against the GNC of §1.3, in §17.6–17.10 — including the finding that RRR is the only one of the four whose unit of decision is the revisit event rather than the edge, which is the structural answer to the hole identified just above.**

⚠️ **OpenCV 5 has no pose-graph optimizer** — use **GTSAM (BSD-2)** or **Ceres (BSD-3)**, both verified clean in §7.1.

### 17.3 SeqSLAM — and the step most reimplementations omit

Sequence-based place recognition is the classical answer to perceptual aliasing: **a single frame may be ambiguous, but a *sequence* of frames rarely is.** Directly relevant, since Ariadne's agents traverse corridors.

*(§17.11 gives the full five-step algorithm, the multi-agent fixes and the descendants; this subsection is the summary.)*

> **⭐ The implementation detail that matters: SeqSLAM's column-wise local contrast enhancement step is omitted by most reimplementations, and it is not optional** — it is what normalizes the difference matrix so that matching scores are comparable across the sequence.

**Three ways it breaks in a multi-agent setting, and the fixes:** differing **speed**, differing **direction**, and differing **path** between the agent that recorded and the agent that queries.
> **⭐ Arc-length indexing — indexing the sequence by distance travelled rather than by frame number — is a ~20-line change that removes the biggest one (speed).** Cheap, and it composes with the odometry Ariadne already has.

**Six descriptor-space diagnostics** were proposed to quantify aliasing in your *own* environment rather than trusting published numbers, with **recall-at-100%-precision versus sequence length** as the decision-relevant chart. *(Spelled out in §17.12.)* That plot answers the practical question directly: *how many frames of context do we need before place recognition becomes reliable here?*

### 17.4 ⭐ Cross-scope flag — the carried light is not just a lighting problem

> **The co-moving lamp is a brightness-constancy violation, not merely a low-texture problem. Two consequences:**
> 1. **It rules out direct/photometric methods as a primary front-end** — independently confirming §1.1's hard architectural constraint, now from a second direction.
> 2. **⚠️ It causes shading-gradient keypoints anchored to the ROBOT rather than the WORLD.** The lamp's hotspot, its falloff boundary, and the shading edges it casts all move *with the camera*. A detector will happily fire on them, and they will appear stable frame-to-frame — **which is worse than instability, because they look like good features and will be tracked, matched, and triangulated into geometry that does not exist.**
>
> **This is the same class of hazard as §11.5's ghosting ("spurious AND repeatable") and §12.4's thermal-mirror false positive — and it has the same mitigation: multi-view parallax consistency. A world-anchored feature moves with parallax as the robot translates; a lamp-anchored one does not. Ariadne is already estimating pose, so this test is nearly free — and it is worth building as a first-class filter rather than hoping RANSAC catches it.**
>
> **The flat-field calibration of §14.2 also attacks this directly** — dividing out the static beam pattern removes the falloff-boundary gradients that are the most likely source of these keypoints.

### 17.5 Licence traps

- ⚠️ **GlueStick** — its own README states verbatim that **the overall system is non-commercial**, because of the SuperPoint backbone. The point-and-line joint matching is attractive for exactly this problem; the licence is not. *(Consistent with §1.2's SuperPoint trap — the taint propagates through anything built on those weights.)*
- ⚠️ **seqNet** and **DeltaDescriptors** — **no LICENSE file** on either; all rights reserved by default.
- ⚠️ One item, *"Do we need a bag of words?"*, **could not be confirmed** and was correctly left unreconstructed rather than guessed.

### 17.6 ⭐⭐ Switchable Constraints — the mechanism

Sünderhauf & Protzel, **IROS 2012**, *"Switchable constraints for robust pose graph SLAM"*, [doi:10.1109/IROS.2012.6385590](https://doi.org/10.1109/IROS.2012.6385590). Reference implementation: `vertigo` (OpenSLAM), g2o and Ceres bindings.

§1.3 lists this one rung up the robust-kernel ladder. **It deserves more than a rung, because it is the conceptual parent of everything else in this subsection — including DCS, which is literally its closed-form solution.**

**The idea.** Data association is a *discrete* decision: each loop closure is an inlier or an outlier. Discrete decisions are combinatorial and can't be handed to a least-squares solver. Switchable Constraints **relaxes the discrete decision into a continuous latent variable and optimizes it jointly with the poses** — so the solver you already built (§1.3) decides, by itself, which loop closures to believe.

Attach to every **loop-closure** edge a scalar switch `s_ij ∈ [0,1]`, and augment the state from `X` to `(X, S)`:

```
                  ┌ odometry — never switched, quadratic, exactly as before ──────┐
X*, S* = argmin   Σ  ‖ f(x_i, x_{i+1}) − u_i ‖²_Λi
          X,S    i

                  ┌ loop closures — residual SCALED by the switch ───────────────┐
              +   Σ  ‖ Ψ(s_ij) · ( f(x_i, x_j) − z_ij ) ‖²_Λij
                 ij

                  ┌ switch prior — the term that makes it work ─────────────────┐
              +   Σ  ‖ γ_ij − s_ij ‖²_Ξij            with γ_ij = 1
                 ij
```

`Ψ` is the switch function, in practice `Ψ(s) = s` clamped to `[0,1]`. Note the switch multiplies the **residual**, so it enters the objective as `s²·χ²_ij` — it is scaling the *information matrix* by `s²`, i.e. inflating that edge's covariance by `1/s²`. `s=1` is the original constraint; `s=0` is an edge that has been continuously deleted from the graph.

> **⚠️ The switch prior is not a regularizer you can drop. It is the entire mechanism.** Without `‖γ − s‖²_Ξ`, the global minimum is `s_ij = 0` for *every* loop closure — turn them all off, incur zero loop error, done. The prior **charges a fixed price for disabling a constraint**, so each `s_ij` settles where the marginal error reduction from switching off equals the marginal price of paying the prior. Constraints that agree with the graph consensus keep `s ≈ 1`; constraints that fight it find it cheaper to switch themselves off.

**Why this is the right shape for Ariadne.** Three properties, in order of importance:

1. **It is differentiable end-to-end.** No RANSAC, no hypothesis enumeration, no discrete search. One extra scalar parameter block and one extra unary factor per loop edge in the solver you already have.
2. **`s_ij` is an interpretable output.** After convergence you can *print* which loop closures the system distrusts and by how much. For a competition demo and for field debugging in a collapsed building, "the system tells you which of its own beliefs it doubts" is worth real points — and it is free, because the variable already exists.
3. **It composes with everything else.** Switches are orthogonal to GNC, to the certificate (§7.4) and to PCM (§1.7).

**Difficulty: LOW — ~2–3 days.** In GTSAM/Ceres: add a `double` parameter block per loop edge, wrap the loop factor so it scales its residual and Jacobian by `s`, add a unary prior factor on `s`, clamp to `[0,1]`.
**Tuning:** `Ξ_ij` **is your inlier threshold in disguise** — it sets the exchange rate between "how badly does this edge disagree" and "how much does it cost to ignore it." Sweep it; do not guess it.

> ⚠️ **The aliasing caveat, and it is the same hole as §17.2's.** SC weighs each edge against **the consensus of the rest of the graph**. When perceptual aliasing produces *many mutually-consistent wrong loops* — identical corridor A ↔ identical corridor B, plus all their neighbours, all agreeing — **the wrong set can BE the consensus**, and SC will dutifully switch off the correct minority. **SC/DCS are necessary but not sufficient in a self-similar building.** They must be paired with evidence that arrives *before* the back-end: sequence matching (§17.11) and geometric verification. This is the same structural warning as §17.2 — it applies to every edge-wise method, not just PCM.

### 17.7 ⭐⭐ Dynamic Covariance Scaling — switchable constraints, solved in closed form

Agarwal, Tipaldi, Spinello, Stachniss, Burgard, **ICRA 2013**, *"Robust map optimization using dynamic covariance scaling"*, [doi:10.1109/ICRA.2013.6630557](https://doi.org/10.1109/ICRA.2013.6630557).

§1.3 calls DCS "closed form, free." That is exactly right and it undersells it. **The result:** the inner minimization of §17.6's objective over `s_ij` — holding the poses fixed — has an **analytic solution**. So you never need the extra variable at all. You evaluate the optimal switch directly from the current residual, each iteration, and you have SC's robustness with **no state augmentation, no extra parameter blocks, and no change to the sparsity pattern of your normal equations.**

**The scaling factor.** With `χ²_ij` the current squared (whitened) residual of loop edge `ij`, and a single free parameter `Φ`:

```
s_ij = min( 1 ,  2Φ / ( Φ + χ²_ij ) )
```

then scale that edge's information matrix by `s_ij²` — i.e. apply IRLS weight `w = s²` to the squared residual, exactly as SC would have.

- `χ²_ij ≤ Φ` → the constraint agrees → `s = 1` → **full weight, ordinary Gaussian factor, no robustness cost at all.**
- `χ²_ij ≫ Φ` → `s ≈ 2Φ/χ²` → weight `≈ 4Φ²/χ⁴` → **the edge's influence decays as `1/χ⁴` and vanishes.**

`Φ` has units of squared error and is the **only** parameter. It plays exactly the role `Ξ` played in SC — the price of disbelief — but as one scalar for the whole graph rather than a per-edge covariance.

**✅ Verified against the implementation of record.** GTSAM ships this as `noiseModel::mEstimator::DCS`; from `gtsam/linear/LossFunctions.cpp` on `develop`:

```cpp
double DCS::Weight(double distance, double c) {
  const double e2 = distance * distance;
  if (e2 > c) {
    const double w = 2.0 * c / (c + e2);
    return w * w;                       // note: w² — the switch enters squared
  }
  return 1.0;
}
```

and the header documents the kernel in closed form:

```
Loss        ρ(x) = (c²x² + c·x⁴)/(x² + c)²
Derivative  φ(x) = 2c²x/(x² + c)²
Weight      w(x) = φ(x)/x = 2c²/(x² + c)²   if x² > c,  else 1
```

> ⚠️ **On the factor of 2, honestly.** The published constant depends on whether the least-squares term is written `χ²` or `½χ²` — GTSAM's source notes explicitly that *"DCS uses the x^2 (not 0.5 x^2) least-squares convention."* Naïvely minimizing `s²χ² + Φ(1−s)²` over `s` gives `Φ/(Φ+χ²)`; the published DCS form carries the extra 2 from the paper's convention. **Do not re-derive it from scratch and trust your own constant — take the form above, which is the shipped, tested implementation, and validate against GTSAM's `DCS` on a synthetic graph before trusting your own.** This is a five-minute check that prevents a subtle, silent factor-of-2 miscalibration of `Φ`.

**Relation to M-estimators.** GTSAM's own header states it: *"Under the special condition of the parameter c == 1.0 and not forcing the output weight s <= 1.0, DCS is similar to Geman-McClure."* This matters practically:

> **DCS is REDESCENDING; Huber is not.** Under Huber, a gross outlier at residual `d` still contributes force `∝ d` forever — it keeps pulling on the solution no matter how obviously wrong it is. Under DCS, its influence goes to **zero**. For catastrophic wrong loop closures — §17.2's failure mode, the kind that folds a map — redescending is not a nicety, it is the requirement. **This is the argument for putting DCS above Huber/Cauchy in §1.3's ladder rather than treating them as interchangeable.**

**Difficulty: VERY LOW — ~1 day.** It is a custom `LossFunction` in Ceres or a one-line noise-model wrap in GTSAM. **~10 lines.**

> **⇒ Given §17.2's finding that PCM has a hole on exactly Ariadne's failure mode, DCS is the highest value-per-line back-end item after GNC itself. Build it first — before switchable constraints, before max-mixtures, before RRR. One day, ten lines, and every loop-closure edge in the graph becomes redescending-robust.** Use full SC (§17.6) *only* if you decide the per-edge `s_ij` diagnostic output is worth the extra variables.

### 17.8 ⭐ Max-Mixtures — representing ambiguity instead of suppressing it

Olson & Agarwal, **RSS 2012**, *"Inference on networks of mixtures for robust robot mapping"*, [doi:10.15607/rss.2012.viii.040](https://doi.org/10.15607/rss.2012.viii.040); extended in **IJRR 32(7), 2013**, [doi:10.1177/0278364913479413](https://doi.org/10.1177/0278364913479413).

**The most conceptually general of the four, and the only one that addresses aliasing on its own terms rather than as an outlier problem.**

Replace each constraint's single Gaussian with a **mixture**:

```
p(z_ij | x_i, x_j)  =  Σ_k  w_k · N( μ_k , Σ_k )
```

For a loop closure the natural two-component mixture is:

| Component | Mean | Covariance | Meaning |
|---|---|---|---|
| `k = 1` (inlier) | the measured relative pose | tight | "this loop closure is real" |
| `k = 2` (null hypothesis) | same mean | **enormous** (or uniform) | "this measurement carries no information" |

**The problem, and the trick.** A sum of Gaussians is not a Gaussian, so `−log p` is not quadratic, and the whole sparse-Cholesky/Schur machinery of §1.3 stops applying. Olson & Agarwal's move is to **replace the sum with a max**:

```
p̂(z) = max_k [ w_k · N(μ_k, Σ_k) ]
  ⟹  −log p̂(z) = min_k [ −log w_k + ½‖ f(x_i,x_j) − μ_k ‖²_{Σk⁻¹} ]
```

> **⭐ Why `max` and not `sum` — this is the whole result.** The maximum of Gaussians is, **piecewise, a single Gaussian.** So at each iteration you simply *select* the currently-dominant component per factor — an `O(K)` evaluation, K = 2 or 3 — and then solve an **ordinary Gaussian least-squares problem** with the selected components. **Cost per iteration is essentially unchanged from non-robust SLAM.** The sparsity pattern, the Schur complement, the incremental solvers: all preserved. The max is also an excellent approximation to the sum — they differ materially only near component crossovers — so you give up very little modelling fidelity for a very large computational gain.

**What it buys over SC/DCS — and this is the part that matters for Ariadne.** SC and DCS can only *downweight*. A mixture can carry **genuine multi-hypothesis ambiguity**: "this place is corridor A **or** corridor B" expressed as two *inlier* components with **different means**, and let global consistency across the rest of the graph decide which survives.

> **That is the literal shape of perceptual aliasing.** §17.2's hazard is that a self-consistent wrong hypothesis is indistinguishable *locally*. Max-mixtures is the only method in this subsection that lets you **hold both hypotheses open until distant evidence arrives** rather than committing early and hoping a robust kernel can undo it. It is the back-end counterpart of the same argument the ambiguous-localization literature makes at the front end — that in an aliased scene the pose posterior is genuinely multi-modal, and any system returning a single pose is asserting more than it knows.

**Difficulty: MEDIUM — ~1–1.5 weeks.** A custom factor that selects its component inside the linearization loop. Straightforward in GTSAM; in Ceres a custom `CostFunction` that switches residual and Jacobian on the winning component.
**⚠️ Caveat:** the `max` introduces **discontinuities** in the cost surface → local minima and initialization sensitivity. Mitigate by initializing with the null hypothesis dominant, or by annealing (§17.10).

> **⇒ Recommended use for Ariadne — do NOT multi-hypothesize everything.** Use a **2-component (inlier / null)** mixture as a general robust baseline if you want one, but reserve **k-component** mixtures with distinct means for the *specific* places your §17.12 diagnostics have **measured** to alias. Targeted, not blanket. That keeps K small, keeps the discontinuities rare, and spends the complexity exactly where the environment earned it.

### 17.9 RRR — cluster-level decisions, and the only one that can take a decision back

Latif, Cadena, Neira. **RSS 2012**, *"Robust Loop Closing Over Time"*, [doi:10.15607/rss.2012.viii.030](https://doi.org/10.15607/rss.2012.viii.030); journal version **IJRR 32(14), 2013**, *"Robust loop closing over time for pose graph SLAM"*, [doi:10.1177/0278364913498910](https://doi.org/10.1177/0278364913498910).

Structurally different from §17.6–17.8: RRR is a **hypothesis-set search around the optimizer**, not a robust cost inside it. Two consequences follow, and both matter here.

**The algorithm:**

1. **Cluster.** Group loop-closure candidates into **topological clusters** — two loops belong together if their endpoints are close *in pose-graph sequence*, i.e. they were generated by the same physical revisit event. A real revisit produces a **burst** of loop closures across consecutive keyframes.
2. **Intra-cluster consistency ("Realizing").** Optimize with only that cluster's loops active; **χ² test** the residual at the appropriate DoF. A cluster whose members disagree *with each other* is internally inconsistent → reject immediately, cheaply.
3. **Inter-cluster consistency.** For subsets of surviving clusters, optimize jointly and χ² test again. A cluster consistent alone but inconsistent with another means **at most one of them can be right.**
4. **Maximize the consistent set ("Reversing", "Recovering").** Search for the largest jointly-consistent subset of clusters, **re-run as new clusters arrive** — so a previously-accepted cluster **can be evicted later** when better evidence turns up.

> **⭐ The two properties that make RRR worth its cost for Ariadne:**
>
> **(a) The unit of decision is the revisit event, not the edge.** §17.2's hazard is that aliasing outliers are *mutually consistent* — so any method that scores edges one at a time is scoring them inside the wrong reference class. RRR asks the right question: **"did this revisit happen at all?"** and accepts or rejects the whole burst as one object, tested against the rest of the graph. **This is the structural answer to §17.2, and none of §17.6–17.8 provides it.**
>
> **(b) It is reversible.** In a collapsed building the correct answer frequently only becomes clear after further exploration — the agent finds a second stairwell and *then* knows which one it saw first. Every other method in this subsection makes a soft decision that hardens as the optimization converges. RRR can undo an accepted loop closure. **Given that a wrong inter-agent loop is otherwise unrecoverable (§17.2), reversibility is worth real engineering budget.**

**Difficulty: MEDIUM — ~1.5–2 weeks.** No new mathematics beyond χ² tests and repeated pose-graph optimization; the work is bookkeeping and search. The cost is **many optimizer calls**.
> **Cost control:** restrict the step-3 joint optimization to the **sub-graph spanned by the clusters under test**, not the full graph. Without this, RRR's runtime is dominated by re-solving a graph whose distant parts cannot change the verdict.

**Fit:** ✅ **Build it, at the inter-agent layer specifically.** Single-agent loop closures already have an odometry prior to arbitrate them; inter-agent ones do not (§17.2), which is exactly where cluster-level testing and reversibility pay for themselves.

### 17.10 ⭐⭐ How these relate to the GNC the team is already building — and what to build

**§1.3 commits to GNC as "the highest value-per-line item in this entire report."** That still stands. This subsection places the four classical methods relative to it, because they are **not** competitors — three of the four compose with it.

**The unifying view.** GNC (Black–Rangarajan duality) rewrites a robust cost as `min_w [ w·r² + Φ_ρ,μ(w) ]` over per-measurement weights, then anneals `μ` from a convex surrogate to the true kernel. **Switchable Constraints is the same augmentation, hand-built for one specific kernel**, with `s²` playing the role of `w` and the switch prior playing the role of `Φ_ρ`. **DCS is the closed-form weight update for that kernel.** In other words:

> **SC : GNC :: a specific outlier process : the general outlier-process framework.**
> **DCS : SC :: the analytic solution : the variable you no longer have to carry.**

**✅ And they literally compose — verified from GTSAM source.** `noiseModel::mEstimator::DCS` implements `graduatedWeight(distance, mu)`, and the header documents the schedule:

```
DCS loss is graduated by scaling c with λ = 1/μ, i.e. by replacing c with c/μ
(c already has units of squared error).  μ = 0 is least squares.
```

with `GradShapeSquared(csquared, mu) { return csquared / mu; }`.

> **⇒ You do not choose between GNC and DCS. You run GNC *on the DCS kernel*: `Φ ← Φ/μ`, anneal `μ` downward, and you get DCS's redescending robustness with GNC's insensitivity to initialization.** That is the single most useful sentence in this subsection, it is verified from the shipped implementation, and it costs nothing beyond what §1.3 already plans to build.

**The recommendation, given §17.2.**

| Method | Build? | Effort | What it buys, given the PCM finding |
|---|---|---|---|
| **GNC** (§1.3) | ✅ **Yes — already committed** | ~60 lines | The load-bearing layer. §17.2 raises, not lowers, its importance. |
| **DCS** as GNC's kernel | ✅ ⭐⭐ **Yes — build first** | **~10 lines, 1 day** | Redescending (Huber is not). Composes with GNC for free. Best value-per-line in this subsection. |
| **Switchable Constraints** | 🟡 **Only for the diagnostic** | 2–3 days | Same robustness as DCS but with extra variables. Worth it *only* for the interpretable per-edge `s_ij` output — which for a demo and for field debugging is a real, if secondary, benefit. |
| **Max-mixtures** | ✅ **Yes, phase 2, targeted** | 1–1.5 weeks | **The only method that represents multi-modal ambiguity** rather than suppressing it. Apply at places §17.12 measures as aliased — not everywhere. |
| **RRR** | ✅ **Yes, inter-agent layer** | 1.5–2 weeks | **Cluster-level** decisions and **reversibility** — the structural answer to §17.2's mutually-consistent-outlier hole. |
| **The certificate** (§7.4) | ✅ **Yes — already committed** | ~150 lines | §17.2 already argued this: a wrong-but-self-consistent solution is *precisely* where a provable global-optimality check earns its keep. |

**Build order:** DCS (1 day) → fold into GNC's kernel (free) → max-mixtures at measured-aliased places → RRR at the inter-agent layer. **SC only if you want the `s_ij` printout.**

**⭐ Two references that close this out:**

- **Sünderhauf & Protzel, ICRA 2013** — *"Switchable constraints vs. max-mixture models vs. RRR — A comparison of three approaches to robust pose graph SLAM"*, [doi:10.1109/ICRA.2013.6631320](https://doi.org/10.1109/ICRA.2013.6631320). **Exactly the head-to-head comparison of §17.6, §17.8 and §17.9, by the author of one of them. Read this before choosing.**
- **riSAM** — McGann, Rogers, Kaess, [arXiv:2209.14359](https://arxiv.org/abs/2209.14359) (2022). GNC made **incremental** / iSAM2-compatible. Its framing is the honest statement of the whole problem: *"Due to the NP-Hardness of data association in the presence of perceptual aliasing, tractable (approximate) approaches to data association will produce erroneous measurements... Existing robust SLAM methods either remain sensitive to outliers, become increasingly sensitive to initialization, or fail to provide online efficiency."*
- **DC-SAM** — Lajoie, Hu, Beltrame, Carlone, [arXiv:1810.11692](https://arxiv.org/abs/1810.11692), RA-L 2019. The source of §17.2's warning, verbatim: *"perceptual aliasing creates a large number of mutually-consistent outliers"*, and separately that most methods *"rely on a given trajectory guess (e.g., from odometry) to discern between inliers and outliers and this makes the resulting pipeline brittle."* Solves discrete association and continuous poses **jointly** in one graphical model. Reimplementation: High — **read it as the framing document, don't build it.**

### 17.11 ⭐ SeqSLAM in full — the algorithm, not the summary

Milford & Wyeth, **ICRA 2012**, *"SeqSLAM: Visual route-based navigation for sunny summer days and stormy winter nights"*, [doi:10.1109/ICRA.2012.6224623](https://doi.org/10.1109/ICRA.2012.6224623).

§17.3 gave the one-line version. Here is the implementable one. **The insight is information-theoretic and it is the cheapest large win available on this project: aliasing is a property of *individual observations*, but the temporal ordering along a route is extra evidence that costs nothing to collect.** If N places look identical but their *neighbourhoods* differ, a sequence of length `d_s` disambiguates them as long as the building contains no `d_s`-length identical *routes*.

**The five steps.**

**1. Preprocess — brutally.** Downsample to **64×32 or even 32×24** grayscale. Then **patch-normalize**: split into small patches (e.g. 8×8) and within each patch subtract the mean, divide by the standard deviation. The descriptor is the flattened normalized image.
> **⭐ This step is why SeqSLAM belongs in Ariadne specifically.** Patch normalization is what gives day/night invariance in the original paper — and it is *exactly* the operation that kills the co-moving lamp's brightness gradient (§17.4, §14.2). SeqSLAM's headline robustness property and Ariadne's headline nuisance parameter happen to be the same thing.

**2. Difference matrix `D`.** For query frame `i` and reference frame `j`:
```
D(i,j) = (1/n) · Σ_pixels | q_i − r_j |          # SAD over normalized pixels
```

**3. ⭐ Column-wise local contrast enhancement — the step §17.3 flagged, stated properly.** Normalize each **column** of `D` within a local window of length `R_window` (typically 10–20):
```
D̂(i,j) = ( D(i,j) − μ_local(j) ) / σ_local(j)
```
> **Why it is not optional.** Raw SAD scores have wildly different *scales* in different parts of a route. A bland concrete corridor yields uniformly low differences against everything; a cluttered rubble field yields uniformly high ones. **Without local normalization the global minimum of `D` lands wherever the environment happens to be blandest — which in Ariadne is precisely the texture-poor region you most need to get right.** Local contrast enhancement converts the question from *"how similar is this?"* to *"how much more similar is this than its neighbours?"* — and only the second question is answerable in a self-similar building. **This is the single most common reason naive SeqSLAM reimplementations fail, and it fails in the direction that looks like success on cluttered test data and collapses on the real environment.**

**4. Trajectory search.** For each candidate reference index `s` and each velocity `V ∈ [V_min, V_max]`, sum `D̂` along the straight line `j = s + V·(i − i_0)` over `d_s` frames:
```
S(s,V) = Σ_{k=0..d_s−1}  D̂( i_0 + k ,  s + round(V·k) )
S_min(s) = min_V S(s,V)          best match = argmin_s S_min(s)
```

**5. Accept / reject.** Threshold the ratio `μ = S_min(best) / S_min(second-best outside an exclusion window)` — a **sequence-level Lowe ratio test**. This is your operating point.

> **Why it defeats aliasing, quantitatively.** For a false match to survive, it must be accompanied by `d_s − 1` *further* false matches lying on a straight, consistent-velocity line in `D̂`. For roughly independent aliasing events the false-positive rate falls **geometrically in `d_s`** while the true-positive rate falls only slowly. **You are buying precision with latency and nothing else** — no GPU, no training, no extra sensor. For a system whose loss function is asymmetric (§17.2: a false positive is unrecoverable, a false negative is merely a missed correction), that is an outstanding trade.

**Difficulty: LOW — ~200 lines, ~1 week including evaluation.**
**OpenCV 5:** `imgproc::resize`, `imgproc::cvtColor`, `imgproc::boxFilter` (patch statistics **and** the column-wise local normalization — both are box filters), `core::absdiff` / `core::norm`. **No learning, no GPU, no external dependency.**

**⚠️ The critique to read first.** Sünderhauf, Neubert & Protzel, **2013**, *"Are We There Yet? Challenging SeqSLAM on a 3000 km Journey Across All Four Seasons"* — the honest evaluation: SeqSLAM is considerably more **parameter- and velocity-sensitive** than the original paper suggests, and degrades sharply when the constant-velocity assumption is violated. Which is the multi-agent problem:

**⭐⭐ The multi-agent breakages and their fixes.** §17.3 named three; here they are with the actual remedies.

| Breakage | What happens to `D̂` | Fix |
|---|---|---|
| **Different speeds** — drone at 2 m/s vs. ground robot at 0.3 m/s | True match is a line of slope `V ≈ 0.15`, outside any naive `V ∈ [0.8,1.2]` search — **it is never even evaluated** | ⭐ **Index by arc length, not frame number.** Resample both trajectories to fixed distance spacing using the odometry Ariadne already has, *before* building `D`. **The true match then has slope 1 by construction.** ~20 lines; removes the biggest failure mode outright. |
| **Opposite directions** — A goes up the corridor, B comes down it | True match has slope **−1**; a forward-only search never finds it. **And** the images are opposite-facing views of the same place, so even single-frame descriptors don't match | (a) Search negative velocities — trivial. (b) The *appearance* half is the hard part: see **Early Bird** ([arXiv:2010.01421](https://arxiv.org/abs/2010.01421)), which combines deep features with **homography-based extreme viewpoint invariance** under a ground-plane assumption, explicitly to address viewpoint change **and** perceptual aliasing together. **Two agents meeting head-on in a corridor is the normal case in a building search, not an edge case.** |
| **Different paths** — A hugs the left wall, B the right | The minimum ridge in `D̂` is **curved or broken**, not a straight line | Replace the straight-line search with a **DP / Viterbi shortest path through `D̂`** under monotonicity and slope constraints. Same asymptotic cost, strictly more general. **Recommended over the line search from the start** — it subsumes it. |
| **No shared frame between agents** | — | Not a sequence-matching problem; see §17.2 and the inter-agent discussion in §17.9/§17.10. |

**Descendants worth the read (none are required):**

| Method | Ref | One-line mechanism | Verdict |
|---|---|---|---|
| **Delta Descriptors** | [arXiv:2006.05700](https://arxiv.org/abs/2006.05700), RA-L 2020 | Match the **temporal derivative** `Δ(t) = φ(t+w) − φ(t−w)` of any global descriptor, L2-normalized. Constant offsets — which is what a global appearance/lighting change *is* — **cancel by construction.** | ✅ **Do it. ~1 day.** Under a co-moving lamp your descriptor carries a large common-mode lighting offset (§17.4); this cancels common mode for free. Known shortcoming per the paper: frame-to-frame volatility — so combine with sequence matching, as the paper does. |
| **SeqNet** | [arXiv:2102.11603](https://arxiv.org/abs/2102.11603), RA-L 2021 | Learnt 1-D temporal convolution over descriptor windows → short sequential descriptor as a **fast hypothesis generator**, then per-image descriptors verify the shortlist | 🟡 **Steal the hierarchy, skip the learning.** Cheap sequence score to shortlist → expensive geometric verification on top-k. The architecture is free; the training is not. |
| **ConvSequential-SLAM** | [arXiv:2009.13454](https://arxiv.org/abs/2009.13454) | **Training-less.** SeqSLAM's sequential + block normalization blended with regional-convolutional matching for viewpoint invariance; analyses content-overlap between query frames to pick a **minimum sequence length adaptively** | ✅ Good fit — handcrafted, no GPU, and adaptive sequence length is directly useful. |
| **Dynamically Modulating VPR Sequence Length** | [arXiv:2407.00863](https://arxiv.org/abs/2407.00863), 2024 | Fits a model on a **calibration set** that modulates `d_s` on-line to just meet a target performance level, rather than fixing it | ✅ **Very good fit.** Ariadne wants *precision ≈ 1.0 at minimum latency*; this is the principled dial instead of a hand-tuned constant. |
| **DeepSeqSLAM** | [arXiv:2011.08518](https://arxiv.org/abs/2011.08518) | CNN+RNN learns description and sequence matching jointly | ❌ **Don't.** You lose the mechanistic explainability that is the entire reason to use sequences. |

### 17.12 ⭐⭐ Descriptor-space aliasing diagnostics — measure YOUR building

§17.3 promised six diagnostics and a chart. Here they are. **Total effort ~1 week, and it belongs in week 1, before any method selection in §17.6–17.11 is committed** — because these measurements are what tell you whether you need max-mixtures at all, and how long `d_s` has to be.

**Prerequisite.** A small labelled set: traverse a self-similar concrete site (a parking garage, a stairwell core, a multi-floor building) and label frames with ground-truth place identity — total station, AprilTags at surveyed positions, or careful manual annotation of a known loop. You need a binary *same place / different place* label per frame pair. **200–500 frames is enough.** One afternoon collecting, one labelling.

**1 — The similarity-matrix picture (do this first, it takes an hour).**
Compute `S(i,j) = sim(φ(I_i), φ(I_j))` for all pairs, render as a heatmap with the ground-truth match mask overlaid.
- Healthy: bright diagonal, bright blocks exactly where GT says revisit, dark elsewhere.
- **Aliased: bright off-diagonal blocks with no GT support. Those blocks ARE your perceptual aliasing, and their positions name the physical places that alias.** This single image will be the most persuasive figure in the project, and it directly feeds §17.8's "apply mixtures only where measured."

**2 — ⭐ The separability plot. This is the chart.**
Over all frame pairs, split similarity scores by GT label and overlay two normalized histograms: `P(sim | same place)` and `P(sim | different place)`.
> **The overlap between those two distributions IS the perceptual aliasing of your environment, quantified.** Report:
> - **Overlap area** `∫ min(p_same, p_diff)` — one number in `[0,1]`, read directly as *"fraction of decisions that are irreducibly ambiguous with this descriptor."*
> - **d′** `= (μ_same − μ_diff) / √(½(σ²_same + σ²_diff))` — effect size, comparable across descriptors.
>
> **Run it for every descriptor you are considering, on your own data.** This is how you choose a descriptor — not from a published leaderboard on outdoor city imagery.

**3 — Nearest-neighbour ratio distributions.**
For each query compute `r = d_NN1 / d_NN2`, where **NN2 must be the best match outside a spatial exclusion window around NN1** — otherwise NN2 is just the adjacent frame, `r ≈ 1` trivially, and the diagnostic is meaningless. Plot `P(r | NN1 correct)` vs. `P(r | NN1 wrong)`.
- Healthy: correct matches concentrate at low `r`; wrong matches near 1.
- **Aliased: both distributions pile up near 1 — the ratio test carries no information.**
> **If you see that, it is your empirical proof that single-image retrieval cannot work at your site — and it is the justification for the sequence-matching (§17.11) and multi-hypothesis (§17.8) budget.** Run it at **both** layers: global descriptors, and the Lowe ratio over local descriptors. Which layer the collapse happens at tells you whether the problem is in aggregation or all the way down in the features.

**4 — ⭐ Precision–recall at the operating point that actually matters.**
Sweep the accept threshold, plot PR. **Do not report mAP.** Ariadne's loss is asymmetric (§17.2), so report the metric that matches it:
> **`R@100P` — recall at 100% precision.** *"We retrieve X% of true loop closures with zero false positives"* is the claim that matters when a single false positive folds the map.
>
> **⭐⭐ Then plot `R@100P` as a function of sequence length `d_s`.** **This is the decision-relevant chart of the entire section.** It answers, in one figure, *how many frames of context we must buy before place recognition is safe here* — which is exactly the question §17.11 exists to answer, and exactly the number that justifies (or kills) the sequence-matching effort.

**5 — Intrinsic structure of the descriptor space.**
- **Eigenvalue spectrum** of the descriptor covariance over the site. A texture-poor, self-similar environment collapses onto a **low-dimensional manifold**; a fast-decaying spectrum says your `d`-dimensional descriptor is really using ~3 dimensions, and quantifies how little the building varies.
- **Pairwise distance distribution.** Narrow and unimodal ⟹ everything is equidistant from everything ⟹ the geometric signature of *no information*.
- **Burstiness.** Count how often a single visual word / cluster dominates an image's descriptor. **Repeated rebar, formwork joints, window mullions and identical door frames produce exactly this**, and it is the aggregation-level manifestation of self-similar structure. See **VLAD-BuFF** ([arXiv:2409.19293](https://arxiv.org/abs/2409.19293), ECCV 2024), whose framing is precise: VLAD's *"feature-to-cluster weighting does not account for over-represented repetitive structures within a cluster... this phenomenon is also referred to as the 'burstiness' problem, classically solved by discounting repetitive features before aggregation."*
> **Burst discounting — power/`sqrt` normalization of visual-word counts before aggregation — is a Low-difficulty classical fix that you should apply regardless of what else you build.** Rubble is *made of* repeated structural elements.

**6 — Use the published aliasing metric so your numbers mean something.**
**TOPO-Bench** — Wang, Chen, Liu, Soh, [arXiv:2510.04100](https://arxiv.org/abs/2510.04100) (2025). *"a key challenge — perceptual aliasing — remains under-quantified, despite its strong influence on system performance... proposing the first quantitative measure of dataset ambiguity to enable fair comparisons across environments"*, with a curated benchmark at **calibrated ambiguity levels**. **The only work found that claims a standard quantitative aliasing measure — adopt their metric so your numbers are comparable to something outside this project.** Difficulty: Low (a metric and a protocol).

**OpenCV 5:** `core` (`norm`, `batchDistance`, `PCA`, `reduce`), `features` (Annoy-based ANN — the FLANN replacement — for the NN searches at scale), `imgproc` for the heatmaps.

**Supporting infrastructure, all verified permissive:**
- **VPR-Bench** — [arXiv:2005.08135](https://arxiv.org/abs/2005.08135), IJCV 2021, [MIT](https://github.com/MubarizZaffar/VPR-Bench). Standardized VPR evaluation; saves you writing PR-curve plumbing.
- **Visual Place Recognition: A Tutorial** — Schubert, Neubert, Garg, Milford, RA-M 2023, [doi:10.1109/MRA.2023.3310859](https://doi.org/10.1109/MRA.2023.3310859) / [arXiv:2303.03281](https://arxiv.org/abs/2303.03281). *"the first tutorial paper on visual place recognition"*, with an evaluation-methodology section and accompanying code. **Give this to every team member on day 1** — it is the on-ramp for everything in §17.11–17.12.
- **GV-Bench** — [arXiv:2407.11736](https://arxiv.org/abs/2407.11736), 2024. Benchmarks local feature matching **for geometric verification of loop closures specifically**, on the premise that *"a false loop closure can be fatal"* and that existing matcher benchmarks target localization instead. ⭐ **Verification and odometry are different jobs and want different matchers — use this to choose the verification one.**

**Knowing when you can't tell — worth a half-day each:**
- **On the Estimation of Image-matching Uncertainty in VPR** — Zaffar, Nan, Kooij, [arXiv:2404.00546](https://arxiv.org/abs/2404.00546) (2024). Compares retrieval-based, data-driven aleatoric, and geometric-verification uncertainty — and proposes a **simple baseline**. *"Highly certain but incorrect retrieval can lead to catastrophic failure of VPR-based localization pipelines."* **Implement the baseline; it tells you empirically whether geometric verification is worth its compute on your data.**
- **Quantile Transfer for Reliable Operating Point Selection in VPR** — [arXiv:2602.04401](https://arxiv.org/abs/2602.04401) ⚠️ *arXiv preprint, venue unverified*. A short **calibration traversal** with known correspondences, then quantile normalization of similarity-score distributions to transfer the threshold to deployment, aiming to *"maximize recall whilst aiming to achieve 100% precision."* ⭐ **That objective is literally Ariadne's, and a calibration traversal on arrival at a site is operationally realistic.** Low difficulty.

---

## 18. ⭐⭐ Co-moving-light datasets — and where the prior art actually lives

### 18.1 The framing finding

> **"SLAM with a headlamp" returns ZERO results. It is not a term of art.** Nor is "active illumination SLAM dataset" or "handheld light SLAM." The one phrase that returns real hits is **"co-located light source"** — and the work under it is in **endoscopy** and **deep-sea imaging**, not robotics.
>
> **The subterranean-robotics community solved darkness by moving to LiDAR and thermal rather than by modelling their own lamps.** Not one SubT dataset (SubT-MRS, CERBERUS, CSIRO, LAMP 2.0, Hilti) documents its illumination rig — wattage, count, or position relative to the cameras. **That gap is itself the finding: Ariadne is taking the harder, less-trodden path, and the relevant prior art is in medicine and oceanography.**

### 18.1b ⭐⭐⭐ OIVIO — the dataset that matches Ariadne's premise exactly

**Found by chasing a loose end in §20.2.** Ebadi et al. mention it in passing; MARBLE built it and then abandoned the vision-primary architecture it was made to evaluate. **It is the single best-matched public dataset in this entire corpus, and it was nearly missed.**

**OIVIO — Onboard Illumination Visual-Inertial Odometry.** Kasper, McGuire, Heckman, **IROS 2019**, [doi:10.1109/iros40897.2019.8968554](https://doi.org/10.1109/iros40897.2019.8968554) (48 citations). Page: **<https://arpg.colorado.edu/oivio/>** (live; note `github.com/arpg/marble` 404s, this does not).

| | |
|---|---|
| **Content** | **36 sequences in mines, tunnels and other dark environments**, >145 minutes of stereo video + IMU |
| **Cameras** | Intel RealSense D435i stereo, **1280×720 grayscale, 30 Hz** |
| **IMU** | LORD Microstrain 3DM-GX5-15 @ 100 Hz |
| **⭐ Illumination** | **9000 lumen / 100 W white LED, carried onboard** — and each trajectory is recorded **three times at ~100%, 50% and 15% of full power** |
| **Ground truth** | 3 sequences: Leica TCRP1203 R300, 3D positions @ 10 Hz. Remainder: **start and end at the same position**, so *total accumulated drift* is the metric |
| **Format** | **ASL (EuRoC-style) — and it ships photometric calibration files** |
| **Terms** | No explicit licence stated on the page |
| ⚠️ **Risk** | The page marks trajectory downloads **"temporarily unavailable."** **Verify the data actually downloads on day 1 before planning around it.** |

> **Why this matters more than anything else in §18.**
> 1. **It is the premise, not an analogue.** Not endoscopy, not underwater, not night driving — a robot carrying a lamp through a mine. R5 (*"algorithms tuned to synthetic darkness fail on anything real"*) is answered by this dataset more directly than by any other.
> 2. **⭐ The three light levels are a free, real-world version of the exact ablation §9.8 asks for.** The same trajectory at 9000 / 4500 / 1350 lumens is a controlled illumination sweep on *real* data — a chart the team would otherwise only be able to produce in simulation, and one a judge cannot dismiss as synthetic.
> 3. **⭐⭐ It ships photometric calibration files**, so §9.3 (response, vignette, exposure) can be *validated against ground truth* rather than merely implemented — and the calibration work becomes measurable rather than assumed.
> 4. **The drift-based ground truth is a feature, not a shortcoming.** Closed loops mean every sequence yields an honest drift number without survey equipment, which suits a 2-month schedule.
> 5. **⚠️ It is stereo D435i, so illumination and *stereo* depth are both available** — the falloff-derived depth of §9.2 can be checked against stereo disparity on the same frames. That is a direct empirical test of the depth-from-illumination-decline idea on real data, which no paper in §9.2 performs outside endoscopy.

**⇒ Recommended as the #1 real-data target for this project**, ahead of the Hilti pairing in §8.1 — Hilti has better ground truth, OIVIO has the right physics and the illumination sweep. Take both if bandwidth allows; take OIVIO first.

### 18.2 ⭐⭐ EndoMapper — the only dataset anywhere with a *calibrated* co-located light

**Azagra et al., *Scientific Data* 10, 671 (2023)** · [syn26707219](https://www.synapse.org/Synapse:syn26707219) · [open-access paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10547713/)

> **This ships a validated photometric calibration alongside the geometric one — the complete set of unknowns Ariadne must solve, already parameterized, with the calibration videos included so you can reproduce the procedure on your own rig.**

Verbatim from the paper:
- *"The light source is co-located with the endoscopes, and hence is **moving and is close to the body surfaces**."*
- *"we assume that these sources can be modelled as a single virtual light and adopt the **Spot Light Source model (SLS)**… In addition, the **light spread function and the camera vignetting are jointly estimated assuming radial symmetry**."*
- *"the endoscope applies an **automatic gain g_t, that can vary at every t-th time instant**, and a **gamma curve (γ = 2.2)**"* — relative auto-gain estimated w.r.t. the first frame, so σ₀ is unobservable and fixed to one.
- *"The parameters are estimated by optimising a photometric loss on the white areas of the Vicalib pattern."* **Validation RMSE: 2.9 grey levels (colonoscopes), 3.3 (gastroscopes).**
- The result is a **2D weighting map of vignetting + light spread function that can be used to compensate them** — exactly §14.2's `K(x)`, with a published recipe.

> **⚠️ The time-varying automatic gain is a failure mode your team will hit with any auto-exposure camera, on top of the geometric falloff. Design for it now.** (This is the same hazard §13.1 identified from the enhancement side, arriving from the sensor side.)

96 sequences, >24 h video, **1440×1080 @ 40–50 fps**, Olympus CV-190 + CLV-190 light source. GT: COLMAP pseudo-GT on a subset, plus **simulated colon sequences from a CT-derived model with full GT geometry and pose.**
⚠️ **Access is the bottleneck — plan for it on day 1:** Synapse **Certified User** status (quiz), profile with **full name + ORCiD + institutional affiliation**, and a **Statement of Intended Use**. *"Requests that do not follow these instructions will automatically be rejected."* Turnaround unverified — assume days-to-weeks.
⚠️ **And read this clause before building a plan around it:** use is *"Limited to research on how to obtain relevant medical information from images or video."* **A disaster-robotics use is not obviously covered. Ask the authors directly.**

### 18.3 ⭐⭐ C3VD / C3VDv2 — best supervision per unit effort, and the physics is independently validated

**Bobrow et al. (JHU Durr Lab)** · v1 *Medical Image Analysis* 2023, **v2 *Scientific Data* 2026** · https://durrlab.github.io/C3VDv2/

**✅ Free direct download, no registration, no form, no DUA** — per-file Google Drive links from **0.59 GB**, so you can prototype on one sequence. v1 = **118.71 GB** total, CC BY-NC-SA 4.0; **v2 is CC BY 4.0 — NC and SA both gone.**

**The best ground truth in the entire corpus:** multimodal 2D–3D registration of real colonoscope video against a *known* 3D phantom — **0.321 mm mean translation, 0.159° mean rotation.** Per frame: **depth (16-bit), surface normals, optical flow, occlusion mask, 6-DoF pose, coverage map, and the GT mesh.**

**Three reasons it transfers better than "colonoscopy" suggests:**
1. **The phantom is RIGID — like rubble, unlike live tissue.** Deformation isn't confounding your photometric residuals. *(Every other endoscopy set has this problem.)*
2. Dense per-frame depth + normals + pose lets you **isolate photometric error from geometric error.**
3. **⭐ NFL-BA independently validated that C3VD images conform to a near-field inverse-square light model** (their "Image Formation Validation" figure). **Third-party evidence that the physics you want to fit is demonstrably present in this data.**

**⭐ v2 adds 192 videos / 169,371 frames (8× v1), plus deliberately introduced artifacts: fecal debris, mucus, blood, and lens contamination — with PAIRED clean/contaminated frames of the same scene. That is the closest published analogue to dust settling on your lens mid-flight** (and §14.8's lens-contamination detector now has training data).

⚠️ **Does not transfer:** depth clamped **0–100 mm** vs your metres — your intra-frame falloff dynamic range is an order of magnitude larger; wet specular silicone vs dry matte concrete; **no airborne particulate backscatter at all — C3VD's medium is clear.** No IMU.

### 18.4 ⭐⭐ The two papers to read first — and the datasets to email for

**NFL-BA** ([arXiv:2412.13176](https://arxiv.org/abs/2412.13176), UNC + UCSD) — already the centrepiece of §9. Two things §9 didn't have:
- It reduces the attenuation term to *"a simple inverse-square fall-off"*, **arguing the directional falloff component is negligible in this regime** — a useful simplification to inherit.
- **⚠️ It contains an unreleased dataset that is almost exactly Ariadne:** *"we capture a dataset of four indoor scenes (Guitars, Porch, Pool, and Stairs) using an iPhone 15 Pro… Scenes are captured using a **co-located point light source mounted to camera**"*, with **ground-truth camera trajectories from motion capture.** The repo TODO reads `[ ] Release self-captured data` — **not released.**
> **⭐ Email the authors (asdunnbe@cs.unc.edu, dnl03c1@cs.unc.edu, ronisen@cs.unc.edu) in week 1. Four mocap-tracked indoor scenes with a light bolted to the camera would be worth more to Ariadne than any 100 GB medical download, and an academic request costs nothing.**

**GLOW** ([arXiv:2511.22857](https://arxiv.org/abs/2511.22857), Nov 2025) — *"We capture images using a smartphone with a flashlight turned on in a dark room."* Same model: *"the flashlight is co-located with the camera and acts as a point light… an inverse-square falloff, i.e. 1/t²."*
> **⭐ Its transferable contribution: a surface-angle-weighted radiometric loss specifically to suppress the specular highlights that co-located flashlight captures induce.** That is directly applicable to §17.4's moving-specular-highlight problem and to wet concrete. 3 synthetic + 4 real scenes, *"will be publicly released"* — **not yet available. Ask.**

### 18.5 Real-world analogues, ranked

**⭐ VAROS** ([Zenodo 5567209](https://doi.org/10.5281/zenodo.5567209), NTNU, ICCVW 2021) — **17.96 GB, CC BY 4.0, anonymous download.** Synthetic, Blender-raytraced: *"physically realistic motion… including **moving illumination**"* and *"integrating direct light, and **indirect volumetric scattering**."* **Exact and dense GT: poses, depth, surface normals, IMU @200 Hz, depth gauge.**
> **The best development harness in the corpus. The only entry combining (a) a light that moves with the vehicle, (b) volumetric scattering, (c) *perfect* pose+depth+normals, and (d) a permissive licence — in 18 GB.** Build and unit-test the photometric model where you can measure error exactly, *then* move to real data. ⚠️ You can overfit Blender's scattering approximation; one sequence; water.

**⭐ Eiffel Tower** ([SEANOE 10.17882/92226](https://www.seanoe.org/data/00810/92226/), Ifremer, IJRR 2023) — **54.46 GB byte-exact**, free, CC BY-**NC-ND** 4.0 (⚠️ the **NoDerivatives** clause constrains redistributing relit or modified imagery). ROV Victor6000 with **12 LED panels at 20,000 lumens each** in absolute darkness; 2020 sequences are **3840×2160 @ 30 fps**. The paper names your problem: *"it also creates **strong differences in illumination depending on the distance between the robot and the scene**."* Four visits over five years gives a long-term appearance-change axis that maps onto "the rubble shifted between passes."

**⭐ AQUALOC** (LIRMM/ONERA, IJRR 2019) — ~69 GB, free Seafile, ⚠️ **no licence stated anywhere.** Two 250,000-lumen LEDs at **270–380 m depth with zero ambient light**. **Two documented phenomena that map exactly onto Ariadne:**
- *"A shadow is also omnipresent in these sequences in the left corner of the recorded images, **because of the limits of the lighting system**."* — **a genuine camera-fixed vignette/shadow that translates with the rig.**
- *"These floating particles are **stirred up from the seabed by the water flows of the ROV's thrusters**"* — **self-induced particulate backscatter, i.e. your rotor downwash stirring dust.**
⚠️ Use the **archaeological** sequences; the harbour ones are sunlit. 640×512 @ 20 Hz monochrome is low-fidelity; GT is COLMAP scaled by pressure.

**EndoSLAM** (Mendeley, **CC BY 4.0**, free) — the only endoscopy set with **independent pose GT** (OptiTrack mocap + 3D-scanned organs + a CT-referenced phantom), not SfM. Multiple camera/lighting configs of the *same* trajectories = a rare ablation axis. Its method contributes a **"brightness-aware photometric loss… consistent under various illumination condition"** — directly reusable.

⚠️ **SCARED** — excellent da Vinci **joint-encoder pose GT** (genuinely independent), but *"the illuminator was turned off"* during the structured-light GT capture. **The frames with the best geometry are not the frames with the co-moving-light photometry.** For a project whose whole question is how co-moving light corrupts photometric residuals, that decoupling blunts it. Plus a login wall (Downloads → 403).
⛔ **Hamlyn — skip.** Official host unreachable (TCP timeout, no Wayback snapshot); GT is Libelas stereo pseudo-depth capped at 300 mm; no pose truth. **C3VD dominates it on every axis.**
⛔ **Sea-thru and FLSea are SUNLIT** — the premise that "underwater ⇒ co-moving light" does not hold for shallow-water sets. **But Sea-thru remains the cleanest published treatment of range-dependent backscatter separation with per-image depth, and that math has the same structure as your airborne-dust term** (cf. §10.3). Borrow the model, not the illumination geometry.

### 18.6 ⚠️ Rubble data — the blunt verdict

> **Real, downloadable, sequential, sensor-grade data of actual collapsed buildings does not exist in public form. The team must accept a proxy, and should say so in the report.**

**The one corpus that ever existed is dead, and I have the forensics.** TRADR's **Amatrice 2016 earthquake ROS bags** (EU FP7, CTU Prague) — the description page is live and lists exactly what Ariadne wants, including *"We e.g. use fog machines for our exercises"* and RGB + laser + stereo + **thermo-camera**. But: the Amatrice path **404s**, the master bagfile index **404s**, and the directory listing now contains only two unrelated 2017 outdoor experiments. **Wayback CDX proves the Amatrice directory was returning 200 as recently as 2023-04-05.** Even the survivors resolve to `/tradr/data/`, which returns **403 Forbidden, not 404.**
> **⭐ ACTION: the 403-not-404 suggests the files may still be on disk behind an access rule. Email Tomáš Svoboda (CTU Prague) requesting the Amatrice and Dortmund power-plant bags. Long latency, zero cost, and recovering them would be worth more than everything else in this category combined. Send it in week 1.**

**Every other EU USAR project is gone:** ICARUS **DNS failure**, RESPOND-A **DNS failure**, NIFTi **403**, SHERPA **51 bytes**, CURSOR's downloads page **empty**. **CRASAR** — the organisation with the most real collapse deployments on Earth, including Surfside — has a datasets page whose entire body text is **"Coming Soon"** (since a 2018 copyright footer). **NIST, RoboCup Rescue and TEEX Disaster City publish arena specs and ASTM test methods, not logs** (grepping all three for "dataset"/"download" returns zero hits). Fukushima/JAEA/Sellafield: nothing downloadable.

**What you can actually get:**
- **⭐ Gazebo Fuel collapsed-building meshes — CC0 1.0:** Collapsed House, Collapsed Fire Station, Collapsed industrial building, Collapsed police station (~1.3–1.8 MB each), plus Urban Wall Debris (20 MB) and Tunnel Wall Debris (13 MB). Open REST download. ⚠️ **Honest caveat: at ~1.5 MB per building these are low-poly SubT/DRC competition props, not photogrammetric scans. Surface detail and self-similar debris texture — exactly what a feature-based front-end keys on — are not realistic. Fine for pipeline plumbing and coverage-planning; misleading for reconstruction-quality benchmarking.**
- **⭐ Mill 19 "Rubble" (Mega-NeRF, CMU) — 9.77 GB, direct download, no registration.** A genuine debris field at a former industrial site, UAV RGB with **PixSFM/COLMAP-refined poses**. **The closest thing to real unstructured rubble geometry with reconstruction ground truth that downloads freely.** ⚠️ Daylight, outdoor, aerial-nadir, no carried light, no dust.
- **CRASAR-U-DROIDs** — 218.79 GB, **CC BY 4.0**, open HuggingFace. 265 orthomosaics, 122,502 views, 21,716 building polygons across 10 disasters — **including one nadir orthomosaic of the Champlain Towers collapse.** ⚠️ Top-down photogrammetric output: **no per-frame poses, no interior views, no darkness.** Damage-classification priors only.

### 18.7 ⭐ The gap no dataset fills — and the plan

> **Nothing published combines a *calibrated* co-moving light with *airborne particulate backscatter* at *metre scale* in *air*.**
> **Endoscopy gives you the calibrated light without the dust. Deep-sea gives you the dust and the darkness in the wrong medium. SubT gives you the medium and scale without a characterized light.**
> **⇒ Expect to calibrate your own rig — and EndoMapper's Vicalib-pattern procedure (§18.2) is the recipe to copy. This gap is also, precisely, Ariadne's contribution.**

**Day 1 (long latency, zero cost — do these before writing code):**
1. Submit the **EndoMapper** Synapse request (certification + ORCiD + Statement of Intended Use), and ask about the medical-use clause.
2. **Email the NFL-BA authors** for the four mocap-tracked co-located-light indoor scenes. **Highest expected value of anything in this section.**
3. Email the **GLOW** authors for their flashlight-in-a-dark-room scenes.
4. **Email Tomáš Svoboda** for the TRADR Amatrice bags.

**Week 1 downloads — ~30 GB, all free and unregistered:** **VAROS** (17.96 GB — exact GT, moving light, scattering) + two small **C3VD** sequences (0.59 + 0.97 GB — dense per-frame GT) + one **AQUALOC** archaeological sequence (~0.8 GB) + **Eiffel Tower 2020** (10.63 GB, 4K, 12 co-moving LED panels).

**Read before writing code:** **NFL-BA** for the near-field BA loss · the **EndoMapper** paper's photometric-calibration section for the SLS + vignetting + auto-gain parameterization · **GLOW** for the surface-angle-weighted specular-suppression loss.

---

## 19. ⭐⭐ A citable evaluation standard — ASTM E2566 and the degraded-conditions protocol

**This corrects §6.5, which said NIST publishes test methods but no data and left it there. The test method itself is the finding — it gives Ariadne a standards-traceable degraded-perception metric, which is exactly what the competition's evaluation rubric rewards.**

**ASTM E2566/E2566M-24**, *"Standard Test Method for Evaluating Response Robot Sensing: **Visual Acuity**"*, under subcommittee **E54.09 "Response Robots"**.
⚠️ **Cite E54.09, not E54.08** — that designation does not exist today; the historical body was E54.08.01 and still appears in older NIST PDF footers. E54.09 currently has **27 active standards, 16 proposed, 0 withdrawn.**

### 19.1 ⭐ The clause that makes this your degraded-conditions protocol

> **§1.7** *"This test method may be **embedded in other tests and operational scenarios**, to provide an indication of **how well the robot can maintain its tested visual acuity under different test and operational situations**."*
> **§1.8** *"…the required performance in the embedded test may be expressed as an operational requirement… **or as a relative requirement, such as achieving 80 % of the performance that the robot achieved on the standard visual acuity test**."*

> **There is no ASTM dust or obscurant test method — none exists in the active or proposed lists. The citable route is exactly this: measure baseline acuity with the standard chart in clean lit conditions, embed the *identical* chart in your dark / dusty / self-illuminated scenario, and report performance as a ratio to baseline. Defensible, reproducible, standards-traceable — and the standard supplies the worked example ratio itself.**

Other quoted scope worth having: **§1.1** measures acuity *"as displayed to a remote operator **or vision algorithm**… with a specified statistical significance."* **§1.2** secondary metrics are **field of view, resolution, aspect ratio.** **§5.1** covers **far-field and near-field**. And **§5.4**, which anticipates your exact problem:
> *"…helps evaluate the **effect of illumination on visual acuity**. In dark environments, robots typically need to illuminate the scene… **Far-field objects downrange require much greater light intensity than near-field objects**… Variable illumination helps ensure the scene is neither too dark nor overwhelmingly lighted so as to thwart the camera's ability to discern visual details (so-called **'washout'**)."*

**Pin §5.4 to §14.1's 8-stops-from-falloff analysis and to CERBERUS's overexposure failure — the standard already names the washout failure mode that non-adaptive lighting produces.**

### 19.2 The apparatus — free, and buildable

⚠️ **Correction: the optotype is the Landolt C, not a 1951 USAF chart** — a full-text search across the NIST guides, aerial Level 1–3 materials and RRL rules found **zero** occurrences of "USAF" or "1951". And the chart specs, optotype geometry and scoring rules are **published free** in NIST's *Guide for Evaluating, Purchasing, and Training with Response Robots*; only E2566 §6 itself is paywalled.

From that guide (Figure 11): **Landolt "C" / Tumbling C charts** — *"essentially line resolution tests… A correct reading of a particular line of 5–10 Tumbling C's produces a numeric measurement of the visual acuity that can be referenced to average human vision."* **Concentric C** versions *"scaled to any size"* appear in a **5-dice pattern to emphasize at least some statistical significance**. Plus **hazmat labels, 10 cm per side**, with a clean scoring rule: *"The labels contain **four attributes including color, icon, text, and number**… **Identification of any three of four attributes is considered successful identification**."*

**Illumination conditions:** *"lights on (**> 300 lux**)"* and *"lights off (**< 0.1 lux**, almost unnavigable without additional light)."* ⚠️ Aerial materials use **< 1 lux** — keep the two attributed separately, don't blend.

**Nested Landolt C calibration** (NIST *Level 1–3 Presentations: Open Environments v2024A*, p.29). Calibration: *"the outer diameter of the largest Landolt C optotype should be **200 mm**."* Scored as **percent of average human vision at a 6 m reference distance**:

| Gap size | Equivalent avg. human vision at 6 m |
|---|---|
| 40 mm | 4.5 % |
| 16 mm | 11 % |
| 6.4 mm | 28 % |
| 2.6 mm | 70 % |
| 1.0 mm | 180 % |
| 0.41 mm | 440 % |

Six levels, ~2.5× per step. ⚠️ *(This table is NIST **aerial** test-method material, not the text of E2566.)* Note the current metric is **not** a Snellen fraction — "Snellen" appears only in the superseded E2566-08 scope.

### 19.3 What exists and what doesn't

**Perception/comms-relevant active standards:** **E2566/E2566M-24** (Visual Acuity) · E2521-24e1 (Terminology) · E2853/E2853M-22 (Search Tasks) · **E2854/E2854M-21 (Radio LOS Range)** and **E2855-12(2021) (Radio Non-LOS Range)** — *directly relevant to §2.1's bandwidth-budget premise* · E3380/E3426 (Endurance) · E3479 (Aerial: Fly Through Apertures) · E3480 (Aerial: Land and Takeoff).

⚠️ **Do NOT cite these as standards — they are WK drafts only:** WK49478 (Video Latency), WK60783 (Audio Speech Intelligibility), WK85837 (Map Indoor Environments).
⚠️ **Confirmed non-existent:** depth perception (no standard, no draft, phrase absent from the corpus) · colour recognition (only via the hazmat colour attribute) · low light (*a test condition inside E2566*, not a standard) · field of view (*a secondary metric of §1.2*) · motion/dynamic-obstacle detection (a RoboCup Rescue task only) · **dust/obscurants (nothing — use the §1.7/§1.8 embedding route)**.
⚠️ **Count reconciliation:** NIST advertises "more than 50 test methods" but only **27 are published ASTM standards**. Cite the E-number where one exists, a NIST document otherwise. Ground-method assembly drawings are contact-gated (`RobotTestMethods@nist.gov`, Adam Jacoff).

### 19.4 RoboCup Rescue — where the apparatus gets stressed

Current rules **2026D (updated 2026-06-03)**; 23 teams in the 2026 season. **Adam Jacoff — the NIST project leader — is International Chair of the RRL Organizing Committee**, and NIST states the test methods *"rel[y] heavily on robot competitions to refine proposed test apparatuses."* **RRL is where NIST designs are stressed before entering the ASTM ballot** — so its tasks preview the next standards.

**Align-Inspect targets** — direct descendant of the concentric C: **5 cm circles inside 5 cm inner-diameter pipes**, on walls at **60 cm elevation**. The operator must align with the pipe, identify the colour, then read **at least two concentric ring gap orientations** (LEFT 90°, LEFT 45°, CENTER, RIGHT 45°, RIGHT 90°). Explicitly: ***"Zooming, exposure control, and image processing score more points."*** — i.e. the competition *rewards* exactly the active-exposure work §14.4 recommends.

**⭐ Mapping/object-detection scoring is directly relevant to Ariadne's reconstruction deliverable:** per-class weights (`AR = 1, Hazmat Signs = 2, Object = 10, **Heat = 15**`), **AprilTag Standard41h12**, detection log types `{ar_code, hazmat_sign, real_object, heat_sig}`, and **bonus multipliers for maps containing sensible RGB colour data and/or heat data.** *(Note heat is the highest-weighted class — consistent with §12.4's victim-detection framing.)*
⚠️ Gating rule: *"In order to score sensor points (victim crate), you must score at least 2 points mobility and 2 points in dexterity."* ⚠️ The 2025B fabrication guide contains **no darkness/lux specification** — whether the current arena has a lights-out lane is unconfirmed.

### 19.5 The recommendation

**Cite ASTM E2566/E2566M-24 under E54.09.** Build the **Landolt C** apparatus (near + far linear charts, concentric C's in the 5-dice pattern, largest optotype **200 mm OD**) plus **10 cm hazmat labels scored 3-of-4**. Score as **percent of average human vision at 6 m** using the nested-gap table. Run at **>300 lux** and **<0.1 lux**.

> **For dust, darkness and self-illumination — where no standard exists — use E2566 §1.7–1.8: baseline acuity, then the identical chart embedded in the degraded scenario, reported as a ratio to baseline.**
>
> **⭐ This is the single cheapest credibility win available to the project. A recognized ASTM/NIST test method in the evaluation section is something almost no hackathon entry will have, it costs a printed chart and an afternoon, and the relative-to-baseline framing is exactly how to make a vision-primary claim honestly — which is also what §13.6 and §6.9 independently concluded about measuring degraded performance.**

**Other USAR benchmarks, verified:** ENRICH (Zwentendorf NPP, Austria — *"completely built but never activated"*; current edition status unconfirmed, several sub-pages 404) · TRADR and ICARUS are **research projects, not scored benchmarks** · euRathlon **domain is dead** · DARPA Robotics Challenge is a task-scored challenge with **no rebuildable apparatus**, unlike E54.09 · ⚠️ **ERL Emergency Robots could not be verified** — both euRobotics league URLs 404.

---

## 20. ⭐⭐ DARPA SubT — the systems, and what actually broke

The canonical prior work for exactly Ariadne's problem: 2018–2021, $82 M, ~300 competitors, 20 teams, courses with **motion-triggered smoke machines**, trap doors and falling debris. Everything below is from primary sources.

⚠️ **Sourcing note:** `fieldrobotics.net` returns **403** to this egress and its DOIs now redirect to **paywalled IEEE Xplore** (Field Robotics content was absorbed into IEEE T-FR). All quotes below are from **arXiv preprints**, with venue/DOI verified independently via the **OpenAlex API**. Preprint and published text may differ.

⚠️ **Venue corrections to the common citations:** the CERBERUS "Wins" paper is **Field Robotics 2024** (`10.55417/fr.2024009`), *not* Science Robotics. Ebadi et al. is **IEEE T-RO 2023** (`10.1109/tro.2023.3323938`) and was **retitled** on publication — the arXiv preprint says *"…in Extreme **Underground** Environments"*, the journal says *"…in Extreme Environments: **The DARPA SubT Challenge**"*. Cite the T-RO title.

### 20.0 🚨🚨 THE FINDING THAT THREATENS ARIADNE'S PREMISE — read before anything else

**Every top SubT team was LiDAR-primary. One team started vision-primary and abandoned it mid-competition.**

Ebadi et al. (T-RO 2023), the canonical cross-team survey, states it flatly:
> *"particular emphasis is put on LIDAR-centric SLAM solutions (**the go-to approach for virtually all teams in the competition**)"*
> *"Most teams relied on **LIDAR and IMU as the dominant sensing modalities**; IMUs are not sensitive to perceptual aliasing…; LIDARs afford accurate and long-range depth measurements **even in the absence of external illumination**."*

The reason is *precisely* §9's problem, named in the survey:
> *"poor lighting conditions make it challenging to deploy visual and visual-inertial SLAM solutions; while the lack of illumination can be **partially compensated by onboard light sources**, the resulting illumination is **either tenuous or creates specular reflections that interfere with visual feature tracking**."*
> *"changing lighting conditions and **specular highlights caused by onboard illumination in dark scenes often lead to instability**."*

And the abandonment, stated explicitly:
> *"A dataset for benchmarking visual-inertial SLAM systems with onboard illumination was released as part of these tests, however **team MARBLE switched to a LIDAR based architecture soon after**."*

Even the **winner's** vision pipeline was its weakest link. CERBERUS on their own multi-robot mapping server (M3RM):
> *"our multi-robot pipeline (M3RM) was **severely challenged** both in terms of computational requirements and with respect to the **resilience of its solution against conditions of visual degradation**."*

> **The team must have a prepared answer to "SubT went LiDAR — why are you doing vision?" Do not claim vision beats LiDAR underground; the evidence says otherwise and a judge may know it.** The three defensible counters are below, and all three are evidence-backed rather than rhetorical.

#### ⭐ Counter 1 — the winners' own post-hoc analysis says their feature detector was the problem

This is the most directly actionable finding in the whole SubT corpus, and it lands squarely on §1.2. CERBERUS analysed their winning run and found the failure was **BRISK**, not vision as such:

> *"in many situations, the employed detection and description of **BRISK features resulted in only a handful of good landmarks**, since most of them were directly discarded due to having a **too short feature track**."*
> *"**It is evident that a keypoint detection and description using BRISK has limited performance and efficiency in subterranean environments.**"*
> *"An experimental processing of the Prize Round using different feature types such as **Superpoint** showed that **the multi-robot map significantly improves**."*
> (Fig. 34) *"landmarks detected using BRISK are **barely present in the corridor**. **Despite the lack of structure, Superpoint still detects several good landmarks** in this challenging environment."*

> **⭐⭐ Read this as: the DARPA-winning team ran handcrafted features underground, watched them fail in a texture-less corridor, identified learned features as the fix, and shipped too late to use it. Ariadne starts where they finished.**
>
> ⚠️ **But do NOT follow them to SuperPoint** — per §1.2 and the licence table, the Magic Leap weights are **non-commercial** and **OpenCV 5 does not ship SuperPoint**. The transferable claim is *learned > handcrafted in texture-poor subterranean scenes*; route it to **ALIKED / DISK / XFeat**, which OpenCV 5 does ship. This is the single strongest external validation the project has for the §1.2 decision, and it costs nothing to cite.

#### ⭐ Counter 2 — the overexposure they suffered is the failure mode §9.4/§14 already fixes

CERBERUS names non-adaptive lighting as a direct cause of their vision failure:
> *"the narrow passages… yielded **many overexposed scenes since the onboard lighting system was not adaptively controlled**."*

They *did* get the hardware synchronisation right, and it is worth copying verbatim:
> *"a **tightly-coupled illumination board** was used to trigger a set of LEDs according to the cameras' exposure times… **The LEDs are only turned on between the start of the first camera and the end of the last camera's exposure.**"*
> *"since multi-camera systems inherently yield different exposure times for each camera, the trigger board performs **exposure compensation using the mid-exposure times**."*

And a dust-specific lighting-geometry trick from their Gagarin aerial robot:
> *"a custom lighting system involving **LEDs at different angles designed to reduce the effect of dust on the image**, increasing the visual range and enhancing the environment texture."*

⚠️ **Their honest limit on self-illumination in open space — this bounds §9.2's depth-from-falloff idea:**
> *"in large open-spaces such as the cave environment it **only illuminated the nearby structure and effectively missed regions further away**. Such areas posed a challenge for visual odometry as **features were only detected [and] tracked in the near vicinity of the camera, typically on the planar ground directly in front of the robot**."*

> **Ariadne's illumination–exposure co-control (§9.4, §14) is not a nicety — it is the named, unfixed cause of the SubT winner's vision failure. That is a genuine novelty axis, externally validated.**

#### ⭐ Counter 3 — the survey lists Ariadne's problem as an OPEN problem, and the cost argument is theirs, not ours

> *"Achieving this goal entails both engineering efforts… and **more research on vision-based SLAM in degraded perceptual conditions (e.g., dust or fog)**."* — Ebadi et al., Future Research Directions
> *"All solutions examined in this paper leverage one or multiple **LIDARs, and powerful embedded computers**. More work is required to enable [these] capabilities… **but with low-cost components** that might be suitable for **smaller, cheaper, expendable systems**… or to design **more affordable robots to increase adoption by first responders**."*

CSIRO — who **tied for the top score** — concede the same weakness in their own winning stack:
> *"The sensor payload with the spinning lidar is **both expensive and heavy**. In many applications, it is desirable or essential to use **smaller, lighter and cheaper sensor configurations**."*

> **⭐ Recommended framing for the submission:** *"SubT solved underground SLAM with heavy, expensive, LiDAR-centric suites and 64-core base stations. Ebadi et al. explicitly name vision-based SLAM in degraded conditions as unsolved and low-cost sensing as the open frontier. Ariadne attacks that named gap, with the fix CERBERUS identified but shipped too late (learned features) and the fix they never made (adaptive illumination–exposure co-control)."*

### 20.1 The hard lessons — what actually broke

#### Dust and obscurants

DARPA *engineered* obscurants into the course: *"Different areas in the course featured **motion-triggered smoke machines** that produced artificial smoke visible in the visual camera **and LiDAR sensors**."*

**On cameras** — the direct statement:
> *"The artificial smoke **introduced many incorrect feature detections for visual tracking** and led to the **loss of good visual landmarks** in the scene."* (CERBERUS)

**On LiDAR** — and the mitigation, which is a one-line sensor config:
> *"the artificial smoke might lead to **incorrect measurements for the LiDAR sensors**, for which the configuration of **only using the strongest return signal per beam** is a possible remedy."* (CERBERUS)

**⭐ Team Explorer's dust filter — the most concrete anti-dust algorithm in the corpus, and it generalises to vision:**
> *"…being able to handle the **dust that arises due to the robot's propellers**. This solution was based on the assumption that **dust would usually accumulate circularly around the robot**, but usually there are still **other distant features** in the environment that allow the robot to solve the optimization correctly. If the robot were capable of performing estimation and continuing operation, it would usually escape the dusty area. **Otherwise, dust would eventually cover the robot from all sides and a catastrophic failure would occur.**"* (via Ebadi)

**CTU's intensity filter** — note it was needed in **both** circuits and was **self-inflicted by the drones**:
> *"an **adaptive intensity-based filter** removes irrelevant data from the Ouster OS1 lidar in order to **filter out clouds of whirled dust, which emerged in both the circuits due to the aerodynamic influence of the UAVs**."*
> Ebadi: *"a **local intensity-threshold filter**… proved to be a **highly robust method for filtration of dust even in the harshest conditions**."*

⚠️⚠️ **CSIRO's caveat is aimed directly at Ariadne — do not over-generalise from SubT's fog machines:**
> *"SLAM performance appeared sufficient in the dust and smoke (e.g., **fog machine**) obscurants tested in SubT. **This is specific to the obscurants encountered; each obscurant may react differently with the lidar signals.**"*

> **Theatrical glycol fog ≠ concrete dust from a collapsed building. §10's scattering physics is the right level to reason at; SubT's dust results are an existence proof, not a transfer guarantee.** Two ideas do transfer: **(a)** the **radial/annular dust prior** — self-generated dust surrounds the robot, so **down-weight near features and trust distant ones**, which composes with §10.7's confined-space stereo arithmetic; **(b)** the vision analogue of "strongest return only" is **rejecting low-contrast / low-gradient-energy image regions before feature extraction** — cheap in OpenCV, and complementary to §13's finding that restoration is not free.

#### Perceptual aliasing and false loop closures — corroborates §17.2

> *"the presence of **self-similar and symmetric areas, and the lack of distinctive visual texture increase the number of false positives** in the place recognition methods that fuel loop closure detection in SLAM, and map fusion and merging in multi-robot systems."* (Ebadi)

**⭐ The strategic move — don't just reject bad loop closures, decline to attempt them:**
> *"While loop closures are crucial for map merging and drift reduction…, it is **equally crucial to avoid closing loops in ambiguous areas with high degree of geometric degeneracy**, as it could lead to spurious loop closure detections."* (Ebadi)

> **This is the missing piece §17.2 asks for.** PCM's blind spot is that self-consistent-but-wrong cliques survive max-clique selection. SubT's answer is to **gate loop-closure *attempts* on a degeneracy/ambiguity measure** — if §10.9's eigen-analysis says the local geometry is degenerate, **suppress place-recognition queries there entirely** rather than trusting the outlier rejector downstream. Cheap, and it attacks the problem before PCM's assumption is violated.

Mechanisms actually deployed: **switchable constraints** (CERBERUS: *"robust outlier rejection using **switchable constraints**"* — the §17.2 / §1.3 optimizer-level family), and **prior-consistency gating** (CERBERUS: *"if the resulting transformation **differs significantly from the prior, it is rejected**… as we expect that the drift between consecutive nodes is relatively small"*).

⚠️ **Calibration on detector false-positive rates** — sobering for any artifact-reporting pipeline:
> CSIRO: *"Post analysis of the sent reports showed that **13 reports were true positives and the remaining 175 were false positives**."* And in another run, *"**11 were true positives and the remaining 95 were false positives**."* Cause of a quarter of them: *"**Approximately one quarter of the false positives were due to artefacts being detected on the robot itself.**"*
> CTU: *"Two false positives are depicted, **Backpack as the red sign in water reflection** and **cellphone as a piece of paper**."*

#### Geometric degeneracy — the corridor that beat the winners

> *"the **lack of perceptual features (e.g., long corridors, large open spaces, and chambers) induces failures in LIDAR-odometry** approaches based on feature or scan matching."* (Ebadi)

CERBERUS's Final Event corridor is the canonical case, and note it degraded **both** modalities at once — violating the very assumption their architecture rests on:
> *"the entrance to the urban area featured a **long white and texture-less corridor with no salient regions nor structure**."*
> *"the **smooth planar walls created a featureless self-similar environment along the principal direction of the corridor**."*
> *"this rendered the underlying **LiDAR pose estimation optimization briefly ill-conditioned**, thus enforcing the use of fallback solutions."*

⭐ **CTU's fix is nearly free and directly applicable:** *"the maps created during the Tunnel Circuit showed **bending effects in the initial long corridor**, so we applied **gravity-vector-based constraints** in the Urban Circuit."* — an IMU gives roll/pitch almost for nothing, and pinning them kills the dominant corridor drift mode. **Pairs directly with §10.9.**

⚠️ Dynamic obstacles broke registration too: *"the **trap doors made it particularly difficult for the registration-based approaches**… since the current pointcloud **differs significantly from the previously acquired pointcloud**."*

#### Comms dropouts and map merging without connectivity

All teams converged on the same three-part answer: **mesh radios + physically deployed relay nodes + onboard autonomy that tolerates disconnection.**

- **Transactional submap sync (⭐ copy this):** *"A **synchronization logic handles partial transmissions and connection losses** and ensures that **only completed submap transmissions will be integrated** into the multi-robot map."* Submaps ≈ **2 MB** each after subsampling and compression — *directly comparable to §1.8's bandwidth blueprint.*
- **Store-and-forward:** *"If the robot was **out of communication range**…, the artifact's class and location was **stored on the robot and transferred once communication was re-established**."*
- **Comms-aware behaviour (⭐ copy this):** *"This module checks if the agent is in connection to the Base Station, and if the agent is **not assigned a time budget for exploration out of communication range**, it commands the robot to **backtrack its path until the connection is reestablished**."* An agent was "Out of Comms" if *"it could not ping the Base Station for more than **3 s**."*
- **Data muling (CSIRO):** *"While out of communications, Bluey had climbed stairs… In the process, **Bluey briefly reconnected with Bingo, and was subsequently able to mule part of its data back** to the ground station."*
- **Breadcrumbs:** each a *"water and dustproof self-contained unit… **2 h battery pack**, BMS, magnetic switch, an **LED to aid recovery of the module** after a mission, WiFi patch antenna, and a radio."*

⚠️ **The honest failures:**
> CERBERUS: *"we deployed **only three of them** at the Final Round (given the eight available modules)… the [Human Supervisor] **couldn't really focus on if and where to deploy the communication breadcrumbs**… This could have been possible only if an **automatic mechanism to decide when to deploy a breadcrumb node** had been implemented."*
> CSIRO: *"**We did not have regular access to a representative underground communication environment**, which made development of… autonomous communications node dropping difficult to validate (**thus this task remained manual**)."*
> CSIRO: *"**communications were not well-modelled in simulation**, so extensive difficulties were experienced when the fleet was scaled up."*
> CSIRO: *"Unfortunately, **global SLAM optimisation had significant latency, often exceeding 10 sec, leading to poor associations**."*

⭐ **The strategic conclusion, and it should shape Ariadne's architecture:**
> *"achieving an always-reliable communication link is particularly challenging in underground environments. To that end, in our experience **a high level of autonomy in the robot agents allowed to mitigate this problem** by enabling exploration when not in communication range."*

> **Design for disconnection as the DEFAULT state, not the exception — this is exactly Swarm-SLAM's premise in §1.4, now with field validation.** Note also that **automating breadcrumb placement is an open gap the winners explicitly named** — a legitimate novelty axis adjacent to Domain 3.

#### Drone failures — the bluntest reporting in the corpus (mostly CTU-CRAS-NORLAB)

> *"Track '**ARMY**' was **not favorable for drones due to the enormous amount of dust that hindered all platforms when the UAVs tried to lift off**. The liftoff also caused the dust to be perturbed, **making it impossible for the small hexapods to navigate**."*
> *"**One of the UAVs crashed about 60 m in** while the other went 100 m in and then 80 m back, **landing out of the crew's reach to gather any data**."*
> *"During this run, **both drones crashed**, entering the course to the door frames."*
> *"one **crashed after about 50 meters to a wall due to a glitch in localization**."*
> (figure caption) *"**crashed UAV being observed by hexapod**"*
> CSIRO: *"the robot **crashed spectacularly after ingesting debris soon after launch**."*
> CSIRO: *"The tunnel was narrow and the **ceiling height too low to launch the UAV without risking catastrophic failure of the map**."*
> CSIRO: *"**Over-exposure of the image due to reflections from UAV legs.**"*

⚠️⚠️ **The critical operational finding: downwash-generated dust is self-inflicted and it degraded not just the drone but the GROUND ROBOTS AROUND IT.** Two consequences for Ariadne's multi-agent plan: **(1)** never launch a drone in a confined space while ground agents are simultaneously relying on vision there; **(2)** the **marsupial** pattern (ground robot carries drone deep, then launches) was independently identified by CERBERUS as the right answer and by CSIRO as blocked by ceiling height — **launch-site selection is itself a perception task.**

CERBERUS on why they barely used drones at all:
> *"we **did not use flying robots extensively** in this competition event and we deployed them after ground robots as opposed to an early scouting operation."*
> *"…**Unfortunately, due to limited development time**, our team could complete the marsupial integration of ANYmal C with RMF-Owl **only after the competition was over**."*

#### ⭐ The "we tried X and it did not work" ledger

| What was tried | Outcome | Source |
|---|---|---|
| **BRISK features underground** | Failed in texture-less corridor; SuperPoint fixed it in post-hoc analysis | CERBERUS |
| **Vision-primary VIO with onboard illumination** | **Abandoned for LiDAR** | MARBLE (via Ebadi) |
| **Non-adaptive onboard lighting** | Widespread overexposure/washout | CERBERUS |
| **Bespoke hexapod platform** | Abandoned for COTS quadrupeds — *"the engineering effort… was **difficult to sustain under the resource constraints**"* | CSIRO |
| **Teleoperation over autonomy on rough terrain** | Caused the rollovers — *"**Most… occurred under teleoperation when the operator intervened to force the robot to navigate to areas where autonomy was refusing to go.** The conclusion… was that **autonomous navigation performed better on this terrain than teleoperation**"* | CSIRO |
| **Simulation as a substitute for field testing** | Repeatedly insufficient — *"**productivity was noticeably slower on features that were not adequately modelled in simulation**"*; *"Not all sensing modalities can be captured in simulation accurately (e.g., **thermal camera frames**)"* | CSIRO, CERBERUS |
| **Hand-labelled training data only** | *"**slow and cumbersome, resulting in the collection of a limited amount of data**"* — they recommend synthetic data **with simulated obscurants and lighting** | CERBERUS |
| **Multi-robot coordination** | Built but **never deployed** — *"for our team **only single-robot autonomy was deployed** despite multi-robot coordination methods being developed"* | CERBERUS |

⭐ **CERBERUS's meta-lesson on testing — the one most likely to save a 2-month project:**
> *"**It is insufficient to handle failures locally in individual components** of the whole pipeline. Failures need to be considered and treated **holistically**."*
> *"**isolated multi-robot testing for scalability and degeneracy issues would yield more information** on its weaknesses without having the full system-of-systems running."*
> *"The day before each field test, a functionality test of all the robots and features… was conducted. **If issues were found in any feature, it was not tested in the field experiment.**"*

⚠️ **And their honest admission about the graded-fallback architecture §12.4b recommends:**
> *"CompSLAM… employed a hierarchical sensor fusion architecture which entails that we had to **define a set of decision functions and heuristics**. Despite its satisfactory performance…, **this approach may present limitations in terms of generalizability**."*
> Ebadi corroborates: *"the **degeneracy detection relies on a hand-tuned set of parameters and is robot-dependent**."*
> **Budget tuning time for the §12.4b wrapper. It is not parameter-free, and the people who won with it say so.**

### 20.2 The teams and their stacks

**CERBERUS** (ETH Zurich / NTNU / UNR / Oxford / UC Berkeley / Flyability / Sierra Nevada) — winner. **CompSLAM's fallback mechanism is already covered in §12.4b; not repeated here.** Complementary facts: the original CompSLAM paper is **ICUAS 2020** (`10.1109/icuas48674.2020.9213865`); a **full method paper landed 2025** ([arXiv:2505.06483](https://arxiv.org/abs/2505.06483)) with code at `leggedrobotics/compslam_subt`. Two mechanism details §12.4b does not carry: the health check is a **D-optimality criterion on covariance growth rate** plus a **motion-bounds check against the controller's commanded motion**; and the smoother is a **GTSAM fixed-lag smoother with a 3-second horizon** with IMU pre-integration, adding **roll/pitch priors from bias-compensated IMU during periods of no motion** (the same trick as CTU's gravity constraint). LiDAR depth is used to **initialise feature depth in the visual/thermal images**, *"providing robustness for scale estimation **without the need for computationally expensive stereo-matching**"* — a cheap idea worth stealing for §10.7.

**Team CoSTAR / JPL–Caltech–MIT–KAIST — NeBula.** LOCUS (RA-L 2021, [arXiv:2012.14447](https://arxiv.org/abs/2012.14447)) = multi-sensor LiDAR odometry; **LOCUS 2.0** (RA-L 2022, `10.1109/LRA.2022.3181357`); LAMP (ICRA 2020, [arXiv:2003.01744](https://arxiv.org/abs/2003.01744)) = multi-robot pose-graph back-end; **LAMP 2.0** (RA-L 2022, `10.1109/lra.2022.3191204`). NeBula system paper: Field Robotics 2022 `10.55417/fr.2022047` ([arXiv:2103.11470](https://arxiv.org/abs/2103.11470)). ⚠️ **Centralized, and expensively so** — *"CoSTAR's base station relied on an **AMD Ryzen Threadripper 3990X with 64 cores/128 threads**"*, and *"**None of the deployed robots performed onboard loop closure detection.**"* **Relevance:** LiDAR-only, so the odometry does not transfer; **LAMP's outlier-rejection layer does** (PCM lineage — §1.7, §17.2).

**Team Explorer (CMU / Oregon State).** System paper: *"Resilient and Modular Subterranean Exploration with a Team of Roving and Flying Robots"*, Scherer et al., **Field Robotics 2022** `10.55417/fr.2022023`; finals paper *"Exploring the Most Sectors at the DARPA Subterranean Challenge Finals"*, Cao/Best et al., **Field Robotics 2023** `10.55417/fr.2023025`. **Super Odometry** (Zhao et al., **IROS 2021**, `10.1109/iros51168.2021.9635862`) is architecturally the interesting contrast to CompSLAM — **IMU-centric rather than cascaded**: *"a **central fusion factor graph**, taking the output of two further factor graphs for VIO and LIO as measurements, and **feeding back the high-rate fused estimate to the given submodules**"*, running at **200 Hz**. Their SLAM held up: *"Explorer's SLAM solution achieved accurate localization and mapping despite… **low light, long corridors, heavy dust/fog, and even dynamic scenes**."* ⭐ **TARE** (**RSS 2021**, `10.15607/rss.2021.xvii.018`) and **FAR Planner** (**IROS 2022**, `10.1109/iros47612.2022.9981574`) are the two most reusable *exploration* algorithms from SubT — **both are sensor-agnostic** (they consume an occupancy/traversability map, not raw LiDAR), so they transfer to a vision-primary system unchanged. **These belong to the still-owed Domain 3.**

**Team MARBLE (CU Boulder / MIT LL).** Field Robotics 2022 `10.55417/fr.2022035` ([arXiv:2110.04390](https://arxiv.org/abs/2110.04390)) and 2023 `10.55417/fr.2023004`. The cautionary tale of §20.0. ⭐ **Chase their benchmark paper — it is the closest thing in the literature to a dataset for Ariadne's exact problem:** *"A benchmark for visual-inertial odometry systems employing **onboard illumination**," IROS 2019* (Heckman et al.). **Cross-check against §18 before assuming it is new.** ⚠️ `github.com/arpg/marble`, cited by Ebadi, now **404s** — MARBLE's code is no longer public at that path.

**CTU-CRAS-NORLAB** — the most candid failure reporting (Field Robotics 2022 `10.55417/fr.2022055`, [arXiv:2110.05911](https://arxiv.org/abs/2110.05911); UAV paper 2023 `10.55417/fr.2023001`). ⭐ **One of the very few vision-primary localization deployments in SubT:** *"The localisation of the hexapods is based on the **Intel T265 camera module**… proprietary grey-scale visual-inertial SLAM… equipped with a secondary RGBD camera (**Intel D435**) which builds a 3D map."* Also ran **without loop closure at all**: *"does not use loop closures, it assumes low odometry drift and **can reuse its global map for scan-to-map matching** when revisiting"* — and admit the cost: *"when loop closure is not possible, **it is hard to avoid localization drift**."* ⭐ Their failure-recovery philosophy is worth adopting wholesale: *"**component-level failures are inevitable**… they design their stack to **continually log system state and automatically restart components from the last known safe state** when a failure is detected."*

**CSIRO Data61 — Wildcat** (tied for top score). Final Event: Field Robotics 2024 `10.55417/fr.2024010` **and** IEEE T-FR `10.1109/tfr.2024.3522063` ([arXiv:2302.13230](https://arxiv.org/abs/2302.13230)). ⚠️ **Wildcat itself is arXiv-preprint-only** ([arXiv:2205.12595](https://arxiv.org/abs/2205.12595)) — no journal version found, **and it was never open-sourced** (`csiro-robotics/wildcat` does not exist). Distinguishing idea: **continuous-time trajectory representation** for point-cloud undistortion, *"resulting in more sharp maps and robust registration even under agile motion."* ⭐ Striking data point on how little loop closure they actually needed: *"**The total number of loop closure edges in this graph is three**"* over a 60-minute run, and *"**place recognition was not found to be necessary for loop closure**."* Their common-sensor-pack discipline paid off: *"All ground platforms utilised the same sensing solution and autonomy varied only through minor configuration parameter changes."*

⚠️ **Team Coordinated Robotics** — **no paper found** in the Field Robotics special-issue index, OpenAlex, or arXiv. Nothing to report.

⭐ **DARPA-side after-action material — found, and still unread:** Orekhov & Chung, *"The DARPA Subterranean Challenge: A Synopsis of the Circuits Stage"*, Field Robotics 2022 `10.55417/fr.2022024`; and Orekhov, Maio, Daniel & Chung, *"Inspiring Field Robotics Advances through the Design of the DARPA Subterranean Challenge"*, Field Robotics 2023 `10.55417/fr.2023018`. **Both blocked by the 403/paywall. High-value targets if anyone has institutional access.**

### 20.3 ⚠️ Fleet heterogeneity — a specific warning for a multi-agent project

CERBERUS documented a failure mode that maps directly onto Ariadne's drone + ground-robot plan:
> *"robots equipped with similar sensing modalities such as visual cameras **differed in sensor types and lens setups, producing images of varying quality** in poorly illuminated environments and **limiting the potential for loop closure detection for multi-robot mapping**."*
> *"small changes such as **different camera types, lenses, and placements of illumination sources** on the robot **play a critical role**… **especially for loop closure and map merging**."*
> Their corrective: *"we aim to equip them with **analogous sensors, lenses, synchronization hardware, and illumination**."*

> **If Ariadne's drones and ground robots carry different cameras or differently-placed lights, cross-agent visual place recognition will silently degrade — and §1.6's descriptor budget assumes it works. Standardise the optical stack across the fleet, or make map merging geometric rather than appearance-based.**

### 20.4 Reusability — ROS1 is the practical barrier, not licensing

*(Licensing skipped per the policy update at the top of this doc. The one licence fact worth carrying: `superxslam/SuperOdom` is **GPL-3.0-only with a paid commercial alternative** — architecture only.)*

| Repo | ROS | Transfers to vision-primary? |
|---|---|---|
| `leggedrobotics/compslam_subt` | ROS1 / catkin | **Architecture yes** (§12.4b); code is LiDAR-centric |
| `ntnu-arl/gbplanner_ros` (branch `gbplanner2`) | **ROS1 Melodic/Noetic only** | ✅ **Yes** — consumes a volumetric map, sensor-agnostic |
| `unr-arl/gbplanner_ros`, `unr-arl/mbplanner_ros` | ROS1 | ✅ Yes |
| `caochao39/tare_planner`, `MichaelFYang/far_planner` | ROS1 Melodic/Noetic (TARE has an AMD64/OR-Tools note) | ✅ **Yes** — reimplement from RSS'21 / IROS'22 |
| `NeBula-Autonomy/LOCUS`, `/LAMP` | ROS1 Melodic/Noetic | ❌ odometry; ✅ **back-end outlier rejection** |
| `csiro-robotics/ohm` (GPU occupancy mapping) | — | ✅ Yes — mapping is sensor-agnostic |
| `superxslam/SuperOdom` | ROS1 | ⚠️ Study only (GPL-3) |
| `osrf/subt` (DARPA simulator) | Ignition/Gazebo | Environment assets useful |
| `arpg/marble`, `csiro-robotics/wildcat` | — | ⛔ **Not public / 404** |

> 🚨 **The dominant practical finding: essentially the ENTIRE SubT open-source corpus is ROS1 Melodic/Noetic, and ROS1 Noetic reached end-of-life in May 2025.** For a 2026 project these are **reference implementations to read, not dependencies to install** — which aligns with the reimplement-from-paper policy anyway, and conveniently moots the licence questions.

### 20.5 Priority — what to actually take from SubT

| Rank | Item | Effort | Why |
|---|---|---|---|
| **1 ⭐⭐** | **Cite the BRISK→learned-features finding to justify §1.2** | ~0 | External validation from the DARPA winner, for free. Route to ALIKED/DISK/XFeat, **not** SuperPoint (§1.2 trap) |
| **2 ⭐⭐** | **Gate loop-closure *attempts* on §10.9's degeneracy measure** | Low | Closes §17.2's PCM blind spot at the right layer — *before* the outlier rejector's assumption breaks |
| **3 ⭐** | **Gravity-vector roll/pitch prior** (CTU; also in CompSLAM at zero velocity) | 1–2 days | Kills the dominant corridor drift mode. Nearly free given an IMU |
| **4 ⭐** | **Comms: transactional submap sync + explore-out-of-comms budget with auto-backtrack** | 2–3 days | Field-validated; composes with §1.4/§1.8 |
| **5** | **Radial dust prior** — down-weight near features, trust distant ones | 1–2 days | Explorer's assumption, restated for vision |
| **6** | **"Strongest return only" analogue** — reject low-gradient-energy regions pre-extraction | 1 day | Cheap; pairs with §13's "restoration is not free" |
| **7** | **CTU's auto-restart-from-last-safe-state supervisor** | 2–3 days | The cheapest reliability win in the corpus |
| **8** | TARE / FAR-style exploration | 2–3 wks | Sensor-agnostic and transfers cleanly — **Domain 3's problem** |
| ⛔ | Reimplementing CompSLAM, Super Odometry, LAMP or Wildcat | >2 months each | **Read, don't build.** Take the wrapper (§12.4b), not the system |

⚠️ **Honesty ledger for §20:** fieldrobotics.net was inaccessible (403) so all quotes are from arXiv preprints with DOIs cross-verified via OpenAlex; the two **DARPA-side after-action papers were located but not read** (paywalled); **Team Coordinated Robotics** could not be found at all; and `arpg/marble` and `csiro-robotics/wildcat` are **not publicly available** despite being cited as such in the T-RO survey.

---

## 21. Volumetric mapping for planning — TSDF, ESDF, frontiers, traversability (Phase 4.1 / 4.2)

Added 2026-10-23, from primary text: the papers were downloaded and read, sections cited (`docs/design/PHASE4_PREFLIGHT.md`). Consumed by D110 (4.1) and D111 (4.2).

| Method | Venue | What Ariadne takes | What it does not |
|---|---|---|---|
| **Voxblox** (Oleynikova et al.) | IROS 2017, [arXiv:1611.03631](https://arxiv.org/abs/1611.03631) | 8³-voxel block hash; weighted TSDF fusion; merged raycasting (one update per voxel per scan); **eq. 5 weighting**: `1/z²` in front, a linear drop-off behind the surface to 0 at δ = 4v (after Bylow et al.); the projective-distance error `r_p = d sin θ − d` and the rule *"increase the safety distance by half of one voxel"* | its quasi-Euclidean ESDF wavefront |
| **nvblox** (Millane et al.) | 2023, [arXiv:2311.00626](https://arxiv.org/abs/2311.00626) | sites = voxels within ε of the TSDF zero crossing; full Euclidean ESDF; an ESDF updated slower than the sensor rate is acceptable; no depth interpolation across foreground–background edges | projection-mapping integration (Voxblox: it aliases at large voxel/pixel ratios); its GPU parallel banding |
| **FIESTA** (Han et al.) | IROS 2019, [arXiv:1903.02144](https://arxiv.org/abs/1903.02144) | separate insert and delete queues for an incremental ESDF (the design comparison) | occupancy-based, "near-optimal" distances |
| **Felzenszwalb & Huttenlocher** | Theory of Computing 8:415–428, 2012 | Algorithm 1, the exact 1D lower envelope, for the **centre-to-centre** metric | not the cube-edge metric: Ariadne's clearance convention (centre to the nearest point of a blocked cube, the GT's) needs `(2|Δ|−1)²/4` per axis, an exact windowed min-convolution (D110 correction) |
| **FUEL** (Zhou et al.) | RA-L 2021, [arXiv:2010.11561](https://arxiv.org/abs/2010.11561) | the frontier information structure is updated only where space was newly observed (D110's incremental frontiers) | its hierarchical planner (Domain 3) |
| **OctoMap** (Hornung et al.) | Auton. Robots 2013 | the 1.8 placeholder's sensor model (D45), kept as the ablation baseline | — |
| **STEP** (Fan et al.) | JPL / DARPA SubT, [arXiv:2303.01614](https://arxiv.org/abs/2303.01614) (abstract) | a risk map consumed as a planning cost, not a binary gate | its slope and step terms (flat floors here) |
| **GBPlanner2** (Kulkarni et al.) | CERBERUS 2022 (secondary) | traversable ground beneath every edge: the 4.2 capsule's observed-column half | its sampling planner (Domain 3) |

**OpenCV 5 adjacency** (checked in the installed headers, SMOKE §4u):
- `cv::Volume` (TSDF / HashTSDF / ColorTSDF): integrate and raycast only, with no voxel or weight read. Used as the
  4.1 test oracle.
- `cv::distanceTransform`: 2D and centre-to-centre.
- `cv::RgbdNormals` and `cv::findPlanes` (`ptcloud/depth.hpp`): the candidate plane-based overhang term for 4.2
  (pending calibration).
- `cv::triangleRasterizeDepth`: mesh → depth, for GT-geometry test depth.
- `cv::Octree`: points, not occupancy.

**Domain 3 (NBV, exploration, allocation, routing):** §22.

---

## 22. Domain 3 — active perception / NBV, exploration, multi-agent allocation, routing (Phase 4.3–4.6)

Added 2026-10-23. Papers were downloaded and their method sections read, except where marked *secondary*. This closes
the domain owed since the start (rate-limited twice) and since D45.

### 22.1 Next-best-view and information gain (4.3)

| Method | Source | Content (as read) | Ariadne |
|---|---|---|---|
| **Isler et al.**, *An Information Gain Formulation for Active Volumetric 3D Reconstruction* | ICRA 2016 (*secondary*: via SEE's related work and abstracts; no open PDF found) | per-voxel IG from occupancy probability, visibility (rays stop at occupied) and distance; variants weight unobserved voxels and "rear side" voxels | §21.3's `Δcoverage` (unknown voxels hit by a 200–500-ray frustum bundle) and `Δentropy` (occupancy entropy along the same rays) are this family |
| **Delmerico et al.**, *A Comparison of Volumetric Information Gain Metrics for Active 3D Object Reconstruction* | Auton. Robots 42(2), 2018, DOI 10.1007/s10514-017-9634-0 (*secondary*: abstract and summaries; no open PDF) | the first comparative study. Counting unobserved voxels, and weighting unobserved voxels near observed surfaces, did best | supports §21.3's choice of an unknown-count term over a pure-entropy term; Ariadne self-normalises per term (p90), so the terms are never compared in raw units |
| **SEE** (Border & Gammell) | IJRR 2023, [arXiv:2207.13684](https://arxiv.org/abs/2207.13684) | measurement-direct NBV on a density representation; frontier points with shared visibility; also the field's own survey: raycast visibility evaluation is the cost bottleneck | the ray budget per candidate (200–500) is §21.3's answer to that cost |

### 22.2 Exploration planning (4.3 candidates, 4.6 scheduling)

| Method | Source | Content (as read) | Ariadne |
|---|---|---|---|
| **TARE** (Cao, Zhu, Choset, Zhang) | RSS 2021, [PDF](https://frc.ri.cmu.edu/~zhangji/publications/RSS_2021.pdf) | Two levels. In a **local planning horizon**, sample a minimum viewpoint set covering the uncovered surfaces (`|p_s − p_v| ≤ D` plus an orientation bound) and solve a TSP through them. At the **global** level, sparse subspaces (cubes) marked exploring / explored / unexplored, with a TSP over them; the two paths join at the horizon boundary | 4.6: D110's 0.5 m coarse layer is the global subspace grid; the local layer is 4.3's candidates and a short TSP |
| **FUEL** (Zhou et al.) | RA-L 2021, [arXiv:2010.11561](https://arxiv.org/abs/2010.11561) | A **frontier information structure** per cluster: cells, AABB, the mean, viewpoints sampled in cylindrical coordinates around the cluster, and connection costs to all other clusters. Large clusters split by PCA. Incremental update by AABB intersection. A global ATSP over clusters, then local viewpoint refinement, then a min-time trajectory | 4.1 already takes the incremental rule; 4.3's generator A takes the per-cluster viewpoint sampling and the PCA split |
| **GBPlanner / GBPlanner2** (Dang et al. 2020; Kulkarni et al. 2022) | JFR / CERBERUS (*secondary*: the CERBERUS system papers) | a local RRG for gain, a global graph for relocation and return home; GBPlanner2 checks traversable ground under every edge | the return-home relocation mirrors §21.4's hard energy rule; ground under edges is D111's observed column |
| **RACER** (Zhou, Xu, Shen) | T-RO 39(3), 2023, [arXiv:2209.08533](https://arxiv.org/abs/2209.08533) | multi-UAV: an **online hgrid decomposition** of unknown space; allocation by **pairwise interaction** over asynchronous, unreliable communication, with per-pair timestamps; coverage paths and workload balance as a **CVRP** | the comms-constrained comparator for 4.4: pairwise, asynchronous, timestamped. CBBA is the arch's choice (§21.7); RACER's pairwise hgrid is the documented alternative if CBBA's consensus traffic does not fit the byte budget |

### 22.3 Multi-agent allocation (4.4)

| Method | Source | Content | Ariadne |
|---|---|---|---|
| **CBBA** (Choi, Brunet, How) | T-RO 25(4):912–926, 2009, DOI 10.1109/TRO.2009.2022423. **Read in primary text** (2026-10-23): MIT Open Access copy, DSpace bitstream `9954d25d-991a-4c8d-be1d-3574ee986f43`; Table I transcribed from the rendered p. 917, since its text layer does not extract | bundle construction by marginal score under diminishing marginal gain (DMG); consensus on winning bids, winners and timestamps; conflict-free convergence with a worst-case guarantee under DMG. Algorithm 3, eqs. (3)–(6) and Table I are transcribed below (§22.3.1) | §21.7: bundles of 1–2, event-triggered. 4.4 implements §22.3.1 exactly |
| **HRCA**, a CBBA variant | ACC 2011 ([PDF](https://skoge.folk.ntnu.no/prost/proceedings/acc11/data/papers/1172.pdf)) | read: CBBA's two phases, the DMG condition, a marginal score with insertion `⊕_n`, the `update / reset / leave` action names and a two-bidder rule table (its own, not CBBA's) | confirms the shape of the rules; not a substitute for Table I |
| **GACA** (grouping auction-consensus) | [arXiv:2608.15884](https://arxiv.org/abs/2608.15884) | nearest-neighbour task grouping before allocation; group-level claim, acquire and contest actions; a per-agent conflict graph resolved by component; timestamps to compensate for communication discrepancies | §21.7's "experimental extension"; Ariadne's spatial clustering step plays the same role |
| **Exact optimum** | `cv::solveLP` (OpenCV core, `optim.hpp`) | the assignment LP is integral (totally unimodular), so the simplex gives the optimal assignment | **4.4's oracle:** CBBA and broker-greedy are graded against the exact optimum on every logged round |

#### 22.3.1 CBBA, transcribed (Choi, Brunet & How 2009, §IV, Algorithm 3, eqs. (3)–(6), Table I)

**Per-agent state:**
- winning bids y_i ∈ R₊^{N_t};
- winners z_i ∈ I^{N_t};
- bundle b_i and path p_i, each of length ≤ L_t;
- timestamps s_i ∈ R^{N_u}.

**Phase 1, BuildBundle (Algorithm 3).** While |b_i| < L_t:
- c_ij = max_{n ≤ |p_i|} S_i^{p_i ⊕_n {j}} − S_i^{p_i} for every j ∉ b_i (eq. 3; c_ij = 0 if j ∈ b_i);
- h_ij = 𝟙(c_ij > y_ij);
- J_i = argmax_j c_ij · h_ij;
- n_{i,J_i} = argmax_n S_i^{p_i ⊕_n {J_i}};
- b_i = b_i ⊕_end {J_i}; p_i = p_i ⊕_{n_{i,J_i}} {J_i};
- y_{i,J_i} = c_{i,J_i}; z_{i,J_i} = i.

The recursion ends when |b_i| = L_t or h_i = 0. Ties are broken lexicographically by agent and task id (§III-B).

**Timestamps (eq. 5):** on each message, s_ik = τ_r (the reception time) if g_ik = 1, else max_{m: g_im = 1} s_mk.

**The three actions on task j:**
- update: y_ij = y_kj, z_ij = z_kj;
- reset: y_ij = 0, z_ij = ∅;
- leave: no change.

**Table I: receiver i's action on task j, from sender k's message** (default: leave). m, n are other agents.

| k thinks z_kj | i thinks z_ij | receiver's action |
|---|---|---|
| k | i | if y_kj > y_ij → update |
| k | k | update |
| k | m ∉ {i,k} | if s_km > s_im or y_kj > y_ij → update |
| k | none | update |
| i | i | leave |
| i | k | reset |
| i | m ∉ {i,k} | if s_km > s_im → reset |
| i | none | leave |
| m ∉ {i,k} | i | if s_km > s_im and y_kj > y_ij → update |
| m ∉ {i,k} | k | if s_km > s_im → update; else → reset |
| m ∉ {i,k} | m | if s_km > s_im → update |
| m ∉ {i,k} | n ∉ {i,k,m} | if s_km > s_im and s_kn > s_in → update; if s_km > s_im and y_kj > y_ij → update; if s_kn > s_in and s_im > s_km → reset |
| m ∉ {i,k} | none | if s_km > s_im → update |
| none | i | leave |
| none | k | update |
| none | m ∉ {i,k} | if s_km > s_im → update |
| none | none | leave |

**Bundle release (eq. 6):** if an update or reset touched a task in b_i, let n̄_i = min{n : z_{i,b_in} ≠ i}. Then:
- y_{i,b_in} = 0 and z_{i,b_in} = ∅ for every n > n̄_i;
- b_in = ∅ for every n ≥ n̄_i.

The tasks added after the outbid one are released, because their marginal scores assumed it. With L_t = 1, CBBA reduces to CBAA.

**Guarantees, as stated (checked against §V–VI, not from memory):**
- **Theorem 1** (DMG, synchronized conflict resolution, static network of diameter D): CBBA produces the **same
  solution as sequential greedy (SGA)** and converges to a conflict-free assignment within N_min·D iterations, with
  N_min = min{N_t, N_u·L_t}.
- **Dynamic network:** if every window of ρ consecutive graphs is jointly connected, the bound is ρ·N_min.
- **Theorem 2** (DMG, accurate situational awareness): M_OPT ≤ 2·CBBA, i.e. 50 % optimality.
- **Consequence for 4.4:** under DMG, CBBA and a broker's centralized sequential greedy give the **same
  assignment**. Their difference is communication: bytes, rounds and robustness to a lost broker. It is not
  solution quality, so §21.7's "both logged identically" can be checked as an exact equality.

### 22.4 Routing (4.5 egress, 4.6 exploration-time)

| Method | Source | Content (as read) | Ariadne |
|---|---|---|---|
| **D\* Lite** (Koenig & Likhachev) | AAAI 2002, Figure 3 transcribed from the rendered page | CalculateKey, UpdateVertex and ComputeShortestPath as transcribed in `EGRESS_4.5.md`; search from the goal; `k_m` for a moving start | 4.5: egress over D111's lattice; the start is fixed, so `k_m` = 0 |
| **FAR Planner** (Yang, Cao, Zhu, Oh, Zhang) | IROS 2022, [arXiv:2110.09460](https://arxiv.org/abs/2110.09460) | polygons from edge points around obstacles; a **two-layer** dynamically updated global visibility graph; "attemptable" routing through unknown space with fast replanning; reported 12–47 % less travel time than A* / D* Lite baselines | 4.6, the exploration-time router and §31.6's comparator. OpenCV-native polygon extraction: `findContours` + `approxPolyDP` on the 2D slab, visibility by `LineIterator` |

### 22.5 Traversal and risk (4.2)

See §21: STEP (risk as a CVaR cost map), GBPlanner2 (ground beneath edges). No published TSDF-based structural
collapse-risk proxy was found. D111's proxy is ours and is calibrated against D112's GT zones.

### 22.6 OpenCV 5 adjacency for Phase 4 (installed headers, 2026-10-23)

- `cv::solveLP`: the exact allocation oracle.
- `cv::partition` (equivalence classes by predicate) and `cv::kmeans`: task clustering (§21.7).
- `cv::findContours`, `cv::approxPolyDP`, `cv::pointPolygonTest`, `cv::convexHull`: FAR polygons.
- `cv::LineIterator`: line of sight on the grid.
- `cv::Subdiv2D`: Delaunay / Voronoi graphs (a candidate topology for the coarse layer).
- `cv::connectedComponents`: 2D clusters.

**No** OpenCV graph search, TSP or auction exists. Those are from scratch, with oracles.

