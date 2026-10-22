# Phase 5 — Degraded visual environments (plan 5.1–5.6, arch §5.3, §5.6, §6.2–6.5, §9)

Session A2, branch `feat/ph5-dve` (from `main` f8c7030). Decisions D130–D149. Status is dated in `docs/STATUS.md`.

## 1. What the preflight established (before any design)

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

## 2. Pieces and where they stand

| Plan | Piece | State | Evidence |
|---|---|---|---|
| 5.1 | Dark worlds (`ci_tiny_dark`: ambient 0, no zone lights, shadows) + lit lamp (`rig_stereo_v2`) | built; v1 worldgen byte-identical | 0 differing files vs main for ci_tiny/rig v1; a 4 s dark recording PASSED (RTF 0.035 on llvmpipe) |
| 5.1 | Lamp calibration procedure, executed | done | D133, `configs/sim/lamp_calib_v2.yaml`; cos decision by held-out fit |
| 5.2 | Stage 3 → albedo proxy | built, synthetic tests | D134; not yet wired into the runner (seam, after the 2.6/L2 merge) |
| 5.2 | Geometry-vs-retrieval split | exists for CLAHE (3.1); Stage 3 is geometry-legal | the §31.5 row needs dark renders |
| 5.3 | Controller + Stage 4 + replay lamp actuator | built, unit + closed-loop tests | D132; runner wiring and the live loop (light_config) to do |
| 5.4 | Configs frozen + L9 dust fidelity | options built (D135); decision and freeze wait on one measured pair | — |
| 5.5 | Confined worlds, baseline sweep 4/8/12/20 cm | not started | §9.4 validator first |
| 5.6 | OIVIO | **blocked**: the official page says downloads are unavailable; every Drive ID returns 404 (2026-10-22) | maintainer decision: email ARPG, or approve a substitute (§49) |

## 3. The sensor model after D130

The clean render is 8-bit sRGB. D decodes it (`input_transfer: srgb`), then applies the linear sensor chain. D's
output is linear raw, which the §6.2 Poisson–Gaussian fit and the GAT assume. The 8-bit sRGB render holds about 12
bits of linear range in the dark: enough for the lamp's ~400× falloff, and L16 adds nothing (§4u).

## 4. Replay vs live (exit-gate comparison)

| Actuator | Live | Replay | Exact? |
|---|---|---|---|
| exposure t, gain g | D's parameters | D's parameters | identical (gz has no camera exposure) |
| lamp power | gz `light_config` | D `lamp_power` (radiance scale) | exact in light transport for ambient 0 and a linear decode; differs only by pre-render clipping/quantisation |
| motion blur from t | not rendered | not modelled | the same omission in both |

So the live-vs-replay gap is expected to be small and concentrated where the render clipped or quantised. The
comparison is designed to find those pixels and is reported whichever way it comes out.

## 5. Open, and why

- D135 freeze needs one measured pair on a dark recording.
- 5.5 needs worldgen presets for narrow voids, plus a multi-baseline rig: extra right cameras in one render, not one
  pass per baseline.
- 5.6 needs data access.
- The live closed loop needs the controller in the sim runner and the `light_config` wrapper (§5.3).
