# 3.1 / 3.6 — retrieval-path enhancement and global-descriptor compression (A1)

Status: design fixed **before** the runs (2026-10-22). Decisions: D92 (3.1), D93 (3.6). Builds on D62 (EigenPlaces
R50-256 through `cv::dnn`, exact index) and its EuRoC protocol (PERCEPTION_2.1 §11.5).

## 1. Preflight

**Literature pass (2026-10-22, web search):**
- No study was found that measures CLAHE or histogram equalisation as a preprocessing step for learned VPR descriptors
  (NetVLAD, EigenPlaces, MegaLoc). Night-time VPR work reports large drops at night ([NPR](https://arxiv.org/pdf/2304.00276))
  and studies illumination-robust retrieval with learned normalisation
  ([No Fear of the Dark](https://arxiv.org/pdf/1908.08999); its CLAHE comparison is `[UNVERIFIED]`: not read).
  So the arch's framing holds: a cheap ablation with an informative outcome either way (§6.3). Nothing is claimed in
  advance.
- No benchmark was found that reports PCA plus int8 descriptor bytes against VPR recall. Adjacent work: network-weight
  quantisation for VPR ([Southampton](https://eprints.soton.ac.uk/503089), [TAT-VPR](https://arxiv.org/html/2505.16447v1)),
  autoencoder compression of CNN descriptors, and 8-byte hashing with sequence matching. The Deep Visual
  Geo-localization Benchmark ([Berton et al., CVPR 2022](https://arxiv.org/pdf/2204.03444)) measures resolution
  against time and memory. So the §17.2 rows are measured here, not imported.

**OpenCV-native:** `cv::createCLAHE` (imgproc) and `cv::PCA` (core) are both in the pinned build (§3.1 below, checked
against the installed headers). The 3.6 harness is offline numpy over stored descriptors; `cv::PCA` is the runtime
path once a consumer exists.

**Wire:** `wire.fbs` `GlobalDescriptor.descriptor` is `[float]`. Shipping any fp16/int8/PCA row therefore needs a
schema change owned by 3.2 (main session). 3.6 does not touch `core/serialization`, `cpp/collaboration` or the §15.7
byte table, and its bytes are **payload bytes `[PROXY]`**: no framing, no compression.

## 2. 3.6 — global-descriptor compression (D93), fixed before the run

- **Data:** the D62 dumps (`build/a1/place/<flight>/{ep376,ep752}.f32`, 11 EuRoC flights, 297 queries), offline. No
  network is re-run.
- **Protocol:** identical to PERCEPTION_2.1 §11.5: positives within 1 m and 30°, a database of samples older than 30 s,
  exact cosine top-k.
- **Rows (§17.2):**

  | row | per-keyframe payload |
  |---|---|
  | fp32 256-D | 1024 B |
  | fp16 256-D | 512 B |
  | int8 256-D | 256 B + fp16 scale = 258 B |
  | PCA→128 fp16 | 256 B |
  | PCA→256 int8 | 258 B |

  Two extra curve points, labelled as extra: PCA→64 fp16 (128 B) and PCA→128 int8 (130 B).
- **int8:** symmetric with a per-vector scale, `s = max|x| / 127`, `q = round(x / s)`. The decoded `q·s` is
  L2-renormalised. Per-vector because the scale travels with its descriptor and needs no shared table; a per-dimension
  table is the alternative, not run.
- **PCA:** mean-centred, no whitening (whitening is not a §17.2 row), fitted on the **other environment**. The Machine
  Hall flights' samples fit the basis used on the Vicon Room flights, and vice versa, so the fit never sees the
  flights it is scored on. Output is truncated, then L2-renormalised. The basis (256×256 + mean) is a one-time shared
  model, not per-keyframe bytes.
- **Built-in regression:** the fp32 row must reproduce D62's pooled R@1 exactly (ep376 0.626, ep752 0.677).
- **Admissibility bar:** a row is admissible if its pooled R@1 is within **0.01** of fp32 (3 of 297 queries).
  Reported with it: R@5, the paired McNemar exact p on R@1 against fp32, and bytes.
- **Not in 3.6, a stated gap:** the §31.9 XFeat local-descriptor fp16/int8 row. D60 ships ORB (binary, 32 B), so ORB
  is what crosses the wire today; the XFeat row lands with the first XFeat-on-the-wire configuration (2.x/3.x
  ablation, register).

## 3. 3.1 — retrieval-path CLAHE (D92), fixed before the run

- **Where:** `PlaceRecognizerParams.enhance = none | clahe`, default `none`.
  - The image is enhanced inside the recogniser, on a copy, so the caller's frame (the geometry path's pixels) is never
    touched.
  - Gray input: CLAHE directly. RGB: CLAHE on the L channel of Lab.
  - Applied at the input resolution, before the INTER_AREA resize.
  - `Name()` carries `+clahe`, so the manifest records it.
- **One setting, fixed now:** clip limit 2.0, 8×8 tiles. This is the common choice in the CLAHE literature, not
  OpenCV's default 40. No sweep: choosing after seeing recall would be tuning on the test set.
- **Tests:**
  - `none` reproduces the D62 dumps byte for byte.
  - The input Mat is unchanged after `Describe` with `clahe`.
  - `clahe` changes the network input.
- **Ablation on EuRoC** (`ariadne_place_eval --retrieval-3.1`): methods `ep376` and `ep376_clahe`, under three
  conditions fixed now:
  - **clean**: the raw frames;
  - **`dark1_dust0`**: radiance 0.25, gain 1;
  - **`dark2_dust0_ag8`**: radiance 0.06 with analog gain 8, so the camera compensates and the frame is noisy.

  `EurocSource` has no degradation hook, so the tool applies the seeded operator D (arch §5.6) to the left frame at
  consumption time (seed 20261022, frame id = the stream index). This is a darkened-EuRoC `[PROXY]`: D assumes linear
  radiance and EuRoC is 8-bit with gamma. Same protocol as §2. Reported either way; no success bar. **Built-in
  regression:** the clean `ep376.f32` must equal D62's dump byte for byte.
- **The §31.5 row** ("dark + co-moving light + retrieval-path enhancement only") is a **sim** row. It needs a sim
  retrieval evaluation with revisit ground truth, which also settles D90's 640×360 vs 1280×720 question. Not in this
  pass: it is the remaining part of 3.1 (register).

## 4. 3.6 result (2026-10-22, `evaluation/runs/retrieval_3.6/20261021T212015Z`)

The regression holds: fp32 reproduces D62's pooled R@1 exactly (ep376 0.626, ep752 0.677). 297 queries. Rows marked *
are the extra curve points.

| row | bytes | ep376 R@1 / R@5 | ΔR@1 (McNemar p) | ep752 R@1 / R@5 | ΔR@1 (p) |
|---|---|---|---|---|---|
| fp32 | 1024 | 0.626 / 0.875 | — | 0.677 / 0.902 | — |
| fp16 | 512 | 0.626 / 0.875 | 0 (1.0) | 0.677 / 0.902 | 0 (1.0) |
| int8 | 258 | 0.626 / 0.875 | 0 (1.0) | 0.677 / 0.902 | 0 (1.0) |
| PCA→128 fp16 | 256 | 0.633 / 0.875 | +0.007 (0.80) | 0.663 / 0.865 | −0.013 (0.48) ✗ |
| PCA→256 int8 | 258 | 0.640 / 0.879 | +0.013 (0.29) | 0.684 / 0.882 | +0.007 (0.73) |
| PCA→64 fp16 * | 128 | 0.640 / 0.872 | +0.013 (0.57) | 0.640 / 0.869 | −0.037 (0.07) ✗ |
| PCA→128 int8 * | 130 | 0.640 / 0.875 | +0.013 (0.48) | 0.663 / 0.865 | −0.013 (0.50) ✗ |

✗ = not admissible (more than 0.01 below fp32).

**Reading:**
- fp16 and int8 (per-vector scale) change **no** query's hit: 4× fewer bytes for free, and no shared model.
- PCA→256 is only a rotation. Its gain is **mean-centring** (the domain-shift effect of removing the training mean),
  not compression, and it is not significant.
- Truncating PCA helps nothing at 376×240 (all differences n.s.) and loses at 752×480.

**So:** the wire candidate is **int8 with a per-vector scale, 258 B** (fp16 at 512 B if the schema prefers a float
type). PCA is not worth a shared basis at this scale. Every difference here is within noise on 297 correlated
queries.
