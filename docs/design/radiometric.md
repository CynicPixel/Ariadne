# Radiometric front end: flat field, VST denoising, exposure normalisation

Plan item 2.2. Decisions D65–D69. Arch §6.1–§6.4, algorithms.md §14.2/§14.5.

**Status (2026-10-21):** the components are implemented and unit-verified against the paper (§5). The sim matrix and
the EuRoC runs (§6) are **not yet measured**. Falloff (stage 3) is Phase 5 and out of scope.

---

## 1. What the stage is

**Data in → out:** raw 8-bit gray stereo pair → raw 8-bit gray stereo pair, between the source and the rectifier.

| arch §6.2 stage | here | what it does |
|---|---|---|
| 1 flat field | `flat_field.hpp`, `RadiometricStage` | `z = raw · (1/K)`; `1/K` stored fp16, one `cv::multiply` |
| 2 Poisson–Gaussian | `gat.hpp`, `noise_estimate.hpp`, `RadiometricStage` | GAT → denoise at σ = 1 → exact unbiased inverse, gated on measured noise (§6.4) |
| 3 falloff | — | Phase 5: needs lamp calibration and depth |
| 4 exposure / gain | inert, recorded in the manifest | needs the §6.5 photometric state; no source carries one yet |

**Why before rectification:**
- the rectifier's interpolation mixes neighbouring pixels' noise, so the GAT's per-pixel model no longer holds after it;
- `K(x)` lives in raw pixel coordinates.

**Mode off is a call-through.** With no `radiometric:` section, `RadiometricSource::ForEachFrame` forwards to the
wrapped source without touching a frame. Default runs therefore change no pixel and no compared output (§7).

## 2. Preflight measurements

### 2.1 What D produces (temporal photon-transfer probe, single channel, 400 frames)

- **Gain and floors:** α = 0.0255 DN/e⁻ at unity gain; b_read = 0.004 DN²; the 8-bit rounding adds 0.083 DN².
- **Unity-gain darkness is posterised.**
  - dark2 spans 0–15 DN with σ 0.2–0.7 DN, mostly rounding.
  - dark3 spans 0–4 DN, and P(0) = 1 at the darkest inputs.
  - So the 16 shipped configs give the VST nothing to stabilise (D65).
- **Gain-up follows the model:** dark2 × gain 8 and dark3 × gain 16 follow `var = a·m + b` to within 0.5%.
- **Corner vs centre:** the corner's var/mean is ≈ K × the centre's. D scales the noise by K, because PRNU and
  vignette are applied after the noise.
- **Cost:** D takes 1.05 s per 1280×720 RGB frame on one thread. `DegradeParallel` row bands are byte-identical
  (test `ParallelRowBandsAreByteIdentical`).

### 2.2 The sim's real path is RGB → D per channel → BT.601 gray

Gray keeps the mean and scales the variance by Σw² = 0.447, so every SNR is 1.5× the mono one.

Single-image fit on the gray path (`preflight/graycheck`, 640×480 synthetic scene, calibrated flat field):

| config | fit a (pred. 0.447·a_D) | b DN² | σ_z | SNR @ median | gate (SNR < 30) |
|---|---|---|---|---|---|
| dark0 | 0.0119 (0.0114) | 0.155 | 0.996 | 91.2 | off: SNR |
| dark1 | 0.0125 (0.0114) | 0.131 | 0.987 | 40.0 | off: SNR |
| dark2 | invalid (4 bins) | — | — | — | off: no fit |
| dark3 | invalid (1 bin) | — | — | — | off: no fit |
| dark2_ag8 | 0.0928 (0.0912) | 0.233 | 1.000 | 23.4 | **on** |
| dark3_ag16 | 0.1892 (0.1824) | 0.467 | 1.003 | 11.3 | **on** |

### 2.3 OpenCV 5.0.0 (pinned build, 8 threads, 1280×720)

| function | result |
|---|---|
| `fastNlMeansDenoising` | 8U (L2) or 16U (L1) only; **refuses 32F**. t5/s11: 236 ms; t7/s21: 536 ms; at 752×480: 119 ms |
| 16U, L1 norm | 180 ms |
| `GaussianBlur` 5×5 (32F) | 0.8 ms |
| guided filter | 13 ms |
| `bilateralFilter` | 16 ms |
| `xphoto::dctDenoising` | 4.3 s |
| `xphoto::bm3dDenoising` | **refused**: patented, `OPENCV_ENABLE_NONFREE` is off in our build |
| `multiply`, `sqrt` | ≈ 1 ms |

## 3. Decisions (detail in DECISIONS D65–D68)

- **Order within the stage: flat field, then VST** (arch §6.2).
  - **For D, this order is correct.** D applies K after the noise, so raw `var = K·a·m' + K²·b`, and z/K restores
    one (a, b). Test `DividingByKRestoresOneNoiseModel` shows it.
  - **For real optics, algorithms.md §14.5's order is the right one** (VST first, then ÷K). There, vignetting acts on
    the light before photon counting, so the raw variance is already homogeneous.
  - So D's order is itself unphysical: a D9-style open question. **D is not changed**, because changing it moves the
    degradation digest. Revisit with the Phase-5 lamp falloff.
- **Stage 4 vs D54's exposure match: no duplication.**
  - Stage 4 is metadata-driven, temporal and per camera: it normalises to a canonical (t₀, g₀) from the §6.5
    controller's state. Inert until a source carries that state (EuRoC has none; the sim's D has fixed exposure).
  - D54's `MatchExposure` is statistics-driven: a left/right gain/bias match at keyframes, for EuRoC's per-camera
    auto-exposure. It keeps that job; `stereo_lk_matcher.*` is untouched.
- **Noise model fitted online, not offline (a deviation from arch §6.2).**
  - The arch says (α, σ) are calibrated offline per (exposure, gain) from static captures.
  - EuRoC has no static captures, and a per-(t, g) table needs §6.5's photometric state.
  - 2.2 fits single-image Foi-style on the first `gate_frames` pairs.
- **Per-(agent, camera) D seeds,** derived from the run seed outside D. Otherwise left and right would carry identical
  noise and PRNU fields, and stereo would match the noise.
- **Dust refused:** no GT depth for the right camera.
- **D frame id = per-camera arrival index,** which equals the pair index.
- **Flat-field calibration frames** use ids `0xC0FFEE…`, so they never share a noise stream with a run frame.
- **The gate**, decided once per run, jointly for both cameras. ON iff every camera has:
  - a valid fit (≥ 6 bins of ≥ 200 px, spanning ≥ 16 DN, a > 0);
  - σ_z ∈ [0.8, 1.25] in the stabilised domain;
  - SNR at the median level < **30**.

  **30, not the draft's 20: corrected 2026-10-21 before any VO measurement.** 20 was derived on mono D output; the
  sim path is gray, 1.5× higher SNR (§2.2), so the same boundary is 30.

  `gate_frames` is 8, not the draft's 30. 8 pairs is ~2.4 M sampled pixels; 30 bought only warm-up latency.

## 4. Components

- **`perception/radiometric/gat.hpp`** (OpenCV-free)
  - Forward transforms: eq. 4.3, and eq. 4.7 after the reduction of eq. 4.4.
  - Inverses: algebraic, asymptotic (4.38), closed form (3.7/4.37, with the `D < 2√(3/8) → 0` branch the invansc
    `.m` omits).
  - `ExactInverse`: E{f|y,s} by eq. 4.10, tabulated on a quadratic y-grid over [0, 200] with 1201 nodes.
    - Poisson sum truncated at 12 SD.
    - The Gaussian part is integrated over u, with z = t0 + u², which removes the square-root kink.
    - The ML zero below E{f|0,s} (eq. 4.14); the closed form beyond the grid.
- **`noise_estimate.hpp`** (OpenCV-free) [PROXY of Foi 2008: no ML fit, no clipped likelihood]
  - Per sampled pixel:
    - level: 3×3 mean;
    - detail: 3×3 second difference ÷ 6;
    - edge test: scale-5 curvature with its exact noise covariance, Mahalanobis χ²₂ at 99%.
  - Per-bin 1.4826·MAD, two passes; WLS fit.
  - `StabilisedSigma` re-measures in the GAT domain, with the mask scaled by the measured variance (never by the
    hoped-for 1).
- **`flat_field.hpp`** (OpenCV-free): K from 16 D frames of a flat-200 target under the camera's own sensor and seed,
  normalised to the central block's median, clamped ≥ 0.1. `AnalyticFlatField` is D's own vignette × PRNU, the check.
- **`radiometric_config.hpp`:** the optional `radiometric:` section.
  - Modes: `off | flat_only | auto | on`.
  - Denoisers: `gaussian` (default: 0.8 ms per image, realtime) `| nlm | none`. NLM is an ablation row only
    (≈ 360 ms per image on this path; the demo needs near-realtime).
  - Inverses: `exact | closed_form | algebraic`.
- **`radiometric_stage.{hpp,cpp}`** (OpenCV): `Decide` (fits + gate) and `Apply`/`ApplyFloat`.
  - NLM runs on the stabilised image scaled to [0, 250] in 8U, with h = `nlm_h`·scale.
  - The inverse is a 4096-node LUT.
- **`runtime/datasets/radiometric_source.hpp`:** the decorator. It holds back `gate_frames` pairs, decides, then
  emits. It forwards every `StereoSource` virtual, including the 2.4 IMU ones. It writes the manifest summary and the
  results block.
- **Sim wiring:** `SimSource::SetDegradation` (lens → D on RGB → gray; calibrates both flat fields).
  `ariadne_sim_vo --degradation YAML`.

## 5. Verification (unit level, measured)

**The paper's figures, reproduced** (`test_gat.cpp`):

| quantity | ours | paper |
|---|---|---|
| closed form, L∞ (4.40) | 0.0470 at s = 0.38, y = 0 | 0.0468 at (0.4, 0) |
| closed form, weighted L2² (4.39) | 0.00690 | 0.0069 |
| I0 − s², L∞ (4.36) | 0.0470 | 0.0470 |
| I0 − s², weighted L2² (4.35) | 0.00295 | 0.0028 |

**Unbiasedness against an independent exact-Poisson Monte Carlo:**
- the exact inverse returns y within 4 SE;
- the algebraic inverse misses by more than 5× that tolerance at y ≤ 5. That's the negative control: the test can
  tell a biased inverse from an unbiased one.
- The other checks: E{f} vs Monte Carlo, variance stabilised to [0.9, 1.1] for y ≥ 5, a round trip with α ≠ 1 and
  μ ≠ 0, and continuity into pure Poisson.

**The noise fit recovers D** (`test_noise_estimate.cpp`):
- a within 10%, b within 0.1 DN² at dark2 × gain 8;
- a gain doubling doubles a;
- a wrong model shows σ_z > 1.3;
- posterised dark2 is refused;
- the gray path scales a by 0.447.

**[MEASURED] weak spots, recorded rather than tuned away:**
- b is 0.82 DN² vs D's 1.12 at dark3 × gain 16. It's harmless: σ_z is 1.01, because the forward and inverse
  transforms share b.
- dark1's a reads 14% high, from only 14 bins over a 63 DN range. The gate is off there by SNR either way.

**Flat field** (`test_flat_field.cpp`):
- ≤ 0.5% RMS against the analytic K, for RGB and mono;
- another camera's seed fails the match;
- z/K brings the corner's var/mean from ≈ 0.72 × the centre's to 1.

**Stage** (`test_radiometric_stage.cpp`):
- off is the same buffer;
- flat_only brings the corner's mean to the centre's within 2%;
- the gate's three outcomes;
- `on` throws without a fit;
- on a dim patch (≈ 15 e⁻, α 0.82 DN/e⁻), the exact inverse keeps the mean, while the algebraic darkens it by
  ≈ α/4 and NLM halves the std. This is measured **before** the 8-bit rounding.
  - **Honesty note:** after `Apply`'s 8-bit rounding, the algebraic inverse's 0.2 DN bias shrinks to 0.075 DN. A
    smooth denoised patch re-quantises without dither.
  - So on the 8-bit hand-off, the inverse choice is invisible below ≈ 0.25·α DN.
  - With α_gray ≈ 0.09–0.19 on the gain-up configs, the algebraic-inverse matrix row is an **expected null**.

**D:** `ParallelRowBandsAreByteIdentical` at 1, 2, 3, 8, 61 and 200 bands with every stage on. The golden digest is
unchanged (8535765251477677907).

## 6. Measurements

### 6.0 Gate decisions on real frames (measured 2026-10-21; 12 frames, `*_rad_auto`)

**Sim** (`ci_tiny_0`, agent 0, left camera shown; right agrees):

| degradation | decision | a | b | σ_z | SNR @ median |
|---|---|---|---|---|---|
| dark0_dust0 | off: SNR | 0.0158 | −0.018 | 0.979 | 57.7 |
| dark2_dust0 | off: no fit (3 bins) | — | — | — | — |
| dark2_dust0_ag8 | **on** | 0.0917 | 0.264 | 1.002 | 15.5 |
| dark3_dust0_ag16 | **on** | 0.1894 | 0.533 | 1.000 | 7.3 |

- a matches 0.447 × D's α to within 1% at gain-up.
- The real scene is darker than the synthetic ramp of §2.2, so its SNRs are lower. **The draft's threshold of 20 would
  have given the same four decisions here.** The correction to 30 rests on the gray-path physics, not on these
  outcomes.

**EuRoC** (default `euroc_rad_auto`): **off on all 11 flights.**
- SNR at the median level is 34–80; the darkest flights are V1_03 at 35.6 and V2_03 at 33.8.
- **The affine model fits EuRoC's 8-bit images less well than D's.**
  - σ_z is 0.73–0.79 on V1_01 and V1_02, with a negative b; the right camera of V2_03 reads 0.66.
  - The likely cause is a non-linear (companded) sensor response [UNVERIFIED]. algorithms.md §14.5's 8-bit warning
    applies.
  - So the forced-on EuRoC rows test the VST on a poorly fitting model, and are reported as such.

**Cost, measured:**
- with NLM, the stage takes ≈ 355–365 ms per 1280×720 image (flat field + GAT + NLM + LUT);
- flat-field only: ≈ 1.8 ms;
- gate-off EuRoC: ≈ 0.2–0.5 ms.

### 6.1 Matrix (fixed here before running; trimmed 2026-10-21)

**Trimmed (decision): run only what cannot be done without.** The one essential comparison is
`dark3_dust0_ag16` × {off, VST gated with the Gaussian denoiser}. The other rows are defined in
`configs/benchmark/radiometric_2.2.yaml` but not run. The EuRoC forced-on runs are dropped: the gate is off on all
11 flights, so default EuRoC is unaffected.

**Result** (agent 0, 1000 frames, `sim_v2`, seed 1; D69):

| row | ATE | lost | keyframes | landmarks alive | stereo valid | stage ms per image | wall time |
|---|---|---|---|---|---|---|---|
| clean recording | 3 mm | 0 | 80 | 770 | 0.776 | — | 296 s |
| dark3_ag16, off | 5 mm | 0 | 76 | 238 | 0.747 | — | 1087 s |
| dark3_ag16, flat + VST + Gaussian | 4 mm | 0 | 104 | 197 | 0.722 | 14.1 | 1243 s |

- **No measurable benefit; slightly worse secondary metrics** (more keyframes, fewer landmarks, lower stereo-valid
  fraction). The 1 mm ATE difference is not claimed (one seed).
- The extra wall time of the degraded rows is mostly D itself (≈ 0.26 s per pair), not the stage.

**Sim.** Recording `ci_tiny_0`, agent 0, `--frames` limited (the budget is ≈ 0.75 s per pair with NLM). Configs:
`dark2_dust0_ag8`, `dark3_dust0_ag16`, plus `dark2_dust0` as the gate-off control.

| row | perception config |
|---|---|
| off | `sim_v1` (run on dark3_ag16 only) |
| flat only | `sim_v1_rad_flat` — **NOT RUN** |
| flat + Gaussian (gated): **the proposal** | `sim_v1_rad_auto` (run on dark3_ag16 only; dark2_ag8 **NOT RUN**) |
| flat + NLM (gated): not realtime | `sim_v1_rad_nlm` — **NOT RUN** |
| flat + Gaussian, algebraic inverse, forced on | `sim_v1_rad_alg`; expected null (§5) — **NOT RUN** |

Metrics: ATE, lost frames, stereo-match valid fraction, landmarks alive, radiometric ms per image.

**EuRoC.**
- Default configs: byte-identical by construction (§1), so the local-slam.md §10 table stands.
- `euroc_rad_auto`: the gate decision on all 11 flights (short runs). Run (§6.0).
- `euroc_rad_on`: forced on, on 2–3 flights. **NOT RUN** (dropped; no result exists).

## 7. Phase-1 gate

- The gate compares results, CSVs, the spool, and the recon and mission outputs, never manifests
  (`configs/benchmark/phase1_gate.yaml`).
- With no `radiometric:` section and no `--degradation`, no pixel and no compared byte changes.
- The manifest gains one field, `radiometric: off`.
- Determinism digests: the degradation digest is unchanged, and the others are untouched.

## 8. Not done / open

- **Falloff (stage 3)** and **stage 4's real normalisation:** Phase 5 (§6.5 controller).
- **Offline per-(t, g) noise tables:** need §6.5.
- **D's K-after-noise order** (§3): physically wrong for optics; revisit in Phase 5.
- **Dust under D in SimSource:** needs right-camera GT depth.
- **`ariadne_sim_collab`** goes through `RunStereoVo`, so it gets the stage via its perception config, but it has no
  `--degradation` flag yet.
- **NLM cost:** NLM alone is ≈ 236 ms per 720p image, and the whole stage with NLM ≈ 360 ms. That's far above a
  20 Hz two-camera budget, so the **default is the Gaussian denoiser** (0.8 ms). NLM is an ablation only; COOL-path
  timing is Phase 3.
- **D's own cost** (≈ 130 ms per RGB frame on 8 threads) matters only for a live-sim demo; a real camera doesn't pay
  it.
- **The existing sim VO** ran 296 s for 1000 clean frames, ≈ 0.3 s per frame at 1280×720, before any 2.2 stage.
  That's a demo-latency issue outside 2.2.
- **The 8-bit hand-off** discards sub-DN precision of the denoised image. A float hand-off to the frontend is a wider
  change, not taken.

## 9. Sources

- M. Mäkitalo, A. Foi, "Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise", IEEE
  TIP 22(1):91–103, 2013. Read via the doctoral thesis (Tampere UT 2013), ch. 3–4,
  https://trepo.tuni.fi/bitstream/10024/114970/1/makitalo.pdf.
- invansc package and licence (reference only, nothing used or shipped): https://webpages.tuni.fi/foi/invansc/.
- A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian, "Practical Poissonian-Gaussian noise modeling and fitting for
  single-image raw-data", IEEE TIP 17(10), 2008.
- RPG-VST, arXiv 2607.24291: the stabilised-domain σ ≈ 1 reliability check.
- P. Bergmann, R. Wang, D. Cremers, "Online photometric calibration…", RA-L 2018, arXiv 1710.02081: the road not taken
  for EuRoC exposure.

---

<a id="degraded-environments"></a>

# Degraded visual environments

Branched from `main` f8c7030. Decisions D130–D149. Status is dated in `the development log (archive/dev-history branch)`.

### 1. What the preflight established

**Order, honestly:**
- The literature pass, the OpenCV-native check and the render probe below came first.
- The algorithms.md survey (§9, §10, §13–14, §17–18) and the design review did NOT. They were done on 2026-10-23,
  after D130–D138 were committed, on request. They changed:
  - D135 (D139);
  - Stage 3 (D140);
  - the D132 framing;
  - the 5.6 substitute.

**Literature.**
- Exposure metric: Shim et al. 2014/2018 (arXiv 1708.07338, eq. 1 transcribed). Comparator: Zhang, Forster &
  Scaramuzza, ICRA 2017.
- Near-light calibration from planes: Park et al., CVPR 2014.
- Lamp-backscatter veil: the underwater artificial-light models (Jaffe–McGlamery).
- Co-moving-light real data: OIVIO (Kasper et al., IROS 2019) and CAVERS (arXiv 2604.15052, 2026).

**OpenCV-native check** (OpenCV 5.0.0 + contrib 5.0.0 headers):
- Present: CRF calibration (Debevec/Robertson), CLAHE, NLM/BM3D, `intensity_transform`.
- Absent: dehazing, exposure control, light-source calibration. Those are ours, built on OpenCV primitives (Sobel,
  resize, the calib3d geometry).

**Render probe** (SMOKE §4u). Four facts no document stated:
1. The sim's RGB is **sRGB-encoded** (D130).
2. The renderer's lamp falloff is **1/(c′ + r²)**. The shipped rig made it nearly flat (D131).
3. SDF spot angles are **full** cone angles, and the lobe is **flat-topped** (D131).
4. Shadows render on llvmpipe.

### 2. Pieces and where they stand

| Plan | Piece | State | Evidence |
|---|---|---|---|
| 5.1 | Dark worlds (`ci_tiny_dark`: ambient 0, no zone lights, shadows) + lit lamp (`rig_stereo_v2`) | built; v1 worldgen byte-identical | 0 differing files vs main for ci_tiny/rig v1; a 4 s dark recording PASSED (RTF 0.035 on llvmpipe) |
| 5.1 | Lamp calibration procedure, executed | done | D133, `configs/sim/lamp_calib_v2.yaml`; cos decision by held-out fit |
| 5.2 | Stage 3 → albedo proxy | built, synthetic tests | D134; not yet wired into the runner (seam, after the 2.6/L2 merge) |
| 5.2 | Geometry-vs-retrieval split | exists for CLAHE (3.1); Stage 3 is geometry-legal | the §31.5 row needs dark renders |
| 5.3 | Controller + Stage 4 + replay lamp actuator | built, unit + closed-loop tests | D132; runner wiring and the live loop (light_config) to do |
| 5.4 | Configs frozen + L9 dust fidelity | options built (D135); decision and freeze wait on one measured pair | — |
| 5.5 | Confined worlds, baseline sweep 4/8/12/20 cm | not started | §9.4 validator first |
| 5.6 | OIVIO | **blocked**: the official page says downloads are unavailable; every Drive ID returns 404 (2026-10-22) | decision needed: email ARPG, or approve a substitute (§49) |

### 3. The sensor model after D130

The clean render is 8-bit sRGB. D decodes it (`input_transfer: srgb`), then applies the linear sensor chain. D's
output is linear raw, which the §6.2 Poisson–Gaussian fit and the GAT assume. The 8-bit sRGB render holds about 12
bits of linear range in the dark: enough for the lamp's ~400× falloff, and L16 adds nothing (§4u).

### 4. Replay vs live (exit-gate comparison)

| Actuator | Live | Replay | Exact? |
|---|---|---|---|
| exposure t, gain g | D's parameters | D's parameters | identical (gz has no camera exposure) |
| lamp power | gz `light_config` | D `lamp_power` (radiance scale) | exact in light transport for ambient 0 and a linear decode; differs only by pre-render clipping/quantisation |
| motion blur from t | not rendered | not modelled | the same omission in both |

So the live-vs-replay gap is expected to be small and concentrated where the render clipped or quantised. The
comparison is designed to find those pixels and is reported whichever way it comes out.

### 5. Open, and why

- D135 freeze needs one measured pair on a dark recording.
- 5.5 needs worldgen presets for narrow voids, plus a multi-baseline rig: extra right cameras in one render, not one
  pass per baseline.
- 5.6 needs data access.
- The live closed loop needs the controller in the sim runner and the `light_config` wrapper (§5.3).
