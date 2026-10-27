# Place recognition: retrieval enhancement and global-descriptor compression

Status: design fixed **before** the runs (2026-10-22). Decisions: D92 (3.1), D93 (3.6). Builds on D62 (EigenPlaces
R50-256 through `cv::dnn`, exact index) and its EuRoC protocol (perception.md §11.5).

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
schema change owned by 3.2. 3.6 does not touch `core/serialization`, `cpp/collaboration` or the §15.7
byte table, and its bytes are **payload bytes `[PROXY]`**: no framing, no compression.

## 2. 3.6 — global-descriptor compression (D93), fixed before the run

- **Data:** the D62 dumps (`build/a1/place/<flight>/{ep376,ep752}.f32`, 11 EuRoC flights, 297 queries), offline. No
  network is re-run.
- **Protocol:** identical to perception.md §11.5: positives within 1 m and 30°, a database of samples older than 30 s,
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

## 4. 3.6 result (2026-10-22, `evaluation/runs/retrieval/euroc_descriptor_coding/20261021T212015Z`)

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

## 5. 3.1 result on EuRoC (2026-10-22, `evaluation/runs/retrieval/euroc_clahe_ablation/`)

The regression holds: the clean `ep376.f32` equals D62's dump byte for byte on 11/11 flights. 297 queries per
condition, on the same samples throughout.

| condition | ep376 R@1 / R@5 | ep376 + CLAHE R@1 / R@5 | ΔR@1 (McNemar p) |
|---|---|---|---|
| clean | 0.626 / 0.875 | 0.640 / 0.865 | +0.013 (0.50) |
| `dark1_dust0` | 0.667 / 0.865 | 0.657 / 0.869 | −0.010 (0.65) |
| `dark2_dust0_ag8` | 0.667 / 0.869 | 0.680 / 0.865 | +0.013 (0.45) |

**Reading:**
- **CLAHE has no measurable effect** on retrieval on EuRoC: |Δ| ≤ 4 queries, with changing sign.
- **Unexpected:** the darkened proxy retrieves *better* than clean: 0.667 vs 0.626 (McNemar p = 0.023, optimistic
  because the queries are correlated). The cause is `[UNVERIFIED]`. One candidate is the operator's tone and
  vignetting on gamma-encoded EuRoC frames (D assumes linear radiance). Either way, darkened EuRoC is **not** a
  harder retrieval condition for EigenPlaces at 376×240. So it cannot show whether CLAHE helps in the dark, and this
  result says nothing about that question.
- **So:** `enhance` stays `none` by default. CLAHE is kept as the selectable ablation (`enhance: clahe`). The question
  CLAHE exists for is answered only by the §31.5 sim row: real darkness with a co-moving lamp, where falloff and
  noise are physical. That row is the remaining part of 3.1.

---

<a id="sim-retrieval"></a>

# Retrieval in simulation

Status: protocol and decision rule fixed **before** any descriptor was computed (2026-10-22). Decision: D90.
Carries the clean part of 3.1's §31.5 row; the dark-scene row waits for Phase 5.1's dark worlds (register).

### 1. Question

D90 (draft) proposed 640×360 as EigenPlaces' default input on the sim rig, the analogue of EuRoC's half size
(D62: 376×240). L5 measured the cost: 312 ms at 640×360 vs 1280 ms at 1280×720 (p50, 4 threads, g4dn). What was
missing is recall on sim views. This note measures it on the one clean recording there is.

### 2. Data

- The Phase 1 evidence recording `evaluation/runs/gates/phase1_thin_e2e/sim_record/ci_tiny_0-20261001T133914Z` (ci_tiny, seed 0,
  two agents, 199 s, 20 Hz, clean: lamp off, no degradation), read through `SimSource`: the RGB frames the pipeline
  sees, with the D41 lens.
- GT: the recorded `gt/pose` stream × the rig's body←camera extrinsic, nearest pose within 10 ms (§11.5's rule).
- One sample per second (`--every 20`), as on EuRoC.

### 3. Protocol (fixed now)

- **Motion thinning, from GT only.** A sample is kept only if the camera moved ≥ 0.25 m or turned ≥ 15° since the last
  kept sample of that agent, the way keyframes enter the real database. Without it, an agent standing still makes
  "revisits" of the identical image 30 s later: measured, every one of agent 1's 115 raw positives is such a repeat.
  Kept: agent 0 127 samples, agent 1 44.
- **Positive:** within 1 m and 30° (perception.md §11.5).
- **Two query sets, pooled:**
  - *intra-agent:* the database is the same agent's samples older than 30 s (§11.5 exactly);
  - *cross-agent:* the database is the other agent's samples older than the query (causal: what the peer could have
    sent by then; no exclusion window, since a different camera cannot produce a trivial repeat).
  Counted from GT before any descriptor: intra 33 queries (all agent 0), cross 28 (agent 0 against agent 1's
  database). **61 queries in all.**
- **Methods:** `ep640` (EigenPlaces R50-256 at 640×360) and `ep1280` (1280×720), both through the RGB path
  (`DescribeRgb`), the sim exports of `models.sh eigenplaces-sim`, `cv::dnn` ENGINE_CLASSIC, exact cosine top-k.
- **Decision rule:** 640×360 becomes the default **unless** 1280×720's pooled R@1 is higher with an exact McNemar
  p < 0.05. 1280×720 costs 4.1× the time per keyframe, so it has to earn it.
- **Power, stated in advance:** 61 correlated queries from one small scene detect only a large difference. A null
  result here means "no evidence that full resolution helps", not "they are equal".

### 4. Commands

```bash
ariadne_place_eval --rec REC/rec --scene REC/scene --agent {0,1} --sim-d90 \
  --perception-config configs/perception/euroc.yaml --export .cache/models/eigenplaces_sim_ref --out OUT/agent{0,1}
python3 -I python/analysis/retrieval_rows_eval.py sim OUT --base ep640 --methods ep1280 --out OUT/rows
```

(`--perception-config` only names the ORB config the tool loads; no ORB method runs in `--sim-d90`.)

### 5. Result (2026-10-22, `evaluation/runs/retrieval/sim_cross_agent/`)

Query counts match §3's GT-only count: 61. **All 61 are agent 0's.** After thinning, agent 1 contributes no intra-agent
query (its raw positives were all stationary repeats) and no cross-agent query (its path never enters agent 0's
earlier views within 1 m and 30°); it contributes only the database of the 28 cross queries.

| query set | queries | ep640 R@1 / R@5 | ep1280 R@1 / R@5 | ΔR@1 (McNemar p) |
|---|---|---|---|---|
| intra, agent 0 | 33 | 0.667 / 0.818 | 0.606 / 0.818 | −0.061 (0.63) |
| cross, agent 0 vs agent 1's database | 28 | 0.750 / 0.964 | 0.821 / 0.929 | +0.071 (0.63) |
| **pooled** | **61** | **0.705 / 0.885** | **0.705 / 0.869** | **0 (1.0)** |

Cost on the dev host (shared, 8 OpenCV threads, indicative): 331 ms vs 1297 ms per frame, the same 4× as L5.

**Reading:**
- The two sizes tie on pooled R@1. The two query sets move in opposite directions by 2 queries each, which is noise
  at this n.
- **Decision rule outcome:** 1280×720 did not win with p < 0.05, so **640×360 is the default (D90 settled).**
- The cross-agent recall, 0.75 R@1 and 0.96 R@5 at 640×360, is the first measurement of the collaborative case on
  sim views. It is on clean frames, in one small scene.
- **Not measured:** the dark-scene row with a co-moving lamp (§31.5). It waits for Phase 5.1's dark worlds (register,
  carried to 5.1).
