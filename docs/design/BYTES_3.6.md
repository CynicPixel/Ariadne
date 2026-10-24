# BYTES_3.6 — local-feature payload compression on real verify pairs (Phase 3.6, A2, 2026-10-24)

Owner A2 (delegated by M, 2026-10-24). Decision block D233 (pre-registration) and D234 (result).
Status: DESIGN. Rule pre-registered (D233) before any run. Advisor-reviewed 2026-10-24: in-run controls, fragment
accounting, per-component PCA scale, corrected reach.

## 1. What and why

A loop-closure candidate between two agents is checked by sending one keyframe's local features (the MATCH_RESPONSE,
arch §15.7) to the verifier, which runs LightGlue + essential/PnP RANSAC + the D71 degeneracy gate. That message
is the expensive one: about 1,024 × (8 B keypoint + 132 B int8 descriptor + 12 B XYZ) ≈ 155 kB per keyframe today,
against ~150 B for the pose edge it buys (arch §15.7: "it is cheap to send the answer and expensive to send the
evidence"). M's survey (`collab_fast_survey.md` §1-2) puts the published like-for-like at 20.1 kB (D2SLAM, PCA'd
SuperPoint) and states that no paper tests the *stock* LightGlue-ALIKED weights on quantised or PCA'd inputs
(**[UNVERIFIED]**). This study measures that on our data, as arch §31.9 requires: accuracy against bytes, never a
byte saving alone.

Data in: the 120 real verify calls of the 3.2 calibration arm (run `0345a5`, `cal_s{0,1}_seed1`, ideal comms,
`pairs.py`: per scene 30 verified plus 6 of each rejection reason). The pass-A stores hold exactly the features
production verified. The `store` row of the knobs harness reproduces the production verdict exactly, e.g. 195/172/92
verified on (0,210)-(1,6) (local check, 2026-10-23). Data out: per setting, the exact codec bytes per keyframe and,
per pair, matches / inliers / verdict / GT pose error, compared with today's encoding in the same process.

Not in scope: the global descriptor (done, D93: int8 at 258 B), retraining (MAD-DR-class Tier B), changing the
feature family (locked).

Conservative by construction: both endpoints' features take the codec's round trip, as today's fleet does (the
verifier decodes its own response too). One-way endpoint verification with the verifier keeping its own float
features (survey A2, MAD-DR's asymmetry) can only do better than these numbers.

## 2. Inputs and why the stores, not re-extraction

- Features: the pass-A stores (`store_agent{0,1}.bin` per scene, sha256 in `inputs/stores_seed1.SHA256SUMS`; all four
  are on the workstation from run 0345a5's results). These are exactly the features production verified, so ALIKED is
  not run at all and no re-extraction question arises. (A 2026-10-24 fidelity alarm about kfdump re-extraction was
  withdrawn: the local smoke had used fixture images, not kfdump output; job 70's fidelity.csv measures the real
  case.)
- K = 512 cannot be measured from the stores. They keep no scores, and `cv::ALIKED` emits its 1,024 keypoints
  unsorted: the order check (`ariadne_perf_bench order`, local, 2026-10-24, on `aliked_lg_sim_ref/input_sim_a0_f200.png`)
  found 515 score increases in 1,023 steps, so the first 512 are not the top 512. As pre-registered, the K = 512 rows
  are dropped here. The keypoint-count question is answered by job 70's knobs stage (`1280x720:512` vs `:1024`,
  re-extracted with scores, on the same 120 pairs), whose own `fidelity.csv` states how closely its re-extraction
  reproduces the stores.
- LightGlue: the pinned `aliked_lg_sim_ref` export (`models.sh`), cv::dnn ENGINE_NEW, exactly as the fleet runs it.
- Verify, gate, per-pair seed: `configs/collab/collab_v2.yaml` (production's), the fleet's per-pair seed formula. The
  code path is the knobs harness's `Verify()`, verbatim. GT: the store's `gt_T_W_B · T_B_C` per keyframe,
  `T_Ca_Cb = Ca⁻¹ Cb` (CollabFleet::GtRel). Error = `GtRel⁻¹ · T_A_B`: translation norm, rotation angle, and D63's
  pose_err_deg (max of rotation and translation-direction angle; rotation only under a 0.1 m true baseline).

## 3. The settings (full factorial, 6 × 4 = 24 at K 1024, all fixed here)

Descriptor coding D (sender encodes, verifier decodes to 128-D float, then stock LightGlue):

| id | encoding | bytes/kp + per-KF header | decode | source |
|---|---|---|---|---|
| `fp32` | float | 512 | as is | upper reference |
| `int8` | **today**: q = round(127 d / s), s = max\|d\| over the KF | 128 + 4 | s q / 127 | D171 |
| `int4` | uniform 16-level midrise, step Δ = 0.3352 σ, σ = 1/√128 (the RMS of any unit 128-D row, so no header); q = clamp(floor(d/Δ), −8, 7), 2 per byte | 64 | (q + ½) Δ | optimum uniform step for a unit Gaussian at 16 levels, Max 1960 **[SECONDARY]**: arXiv 1811.10489 Table, citing Max |
| `bin1` | sign bit, 8 per byte | 16 | ±1/√128 (unit norm) | CDVS SCFV 1-bit; ZippyPoint |
| `pca64` | y = Wᵀ(d − μ), 64 comps; int8 per component, q_j = round(127 y_j / s_j), s_j = max \|y_j\| over the scene-1 fit set (stored with the basis) | 64 | μ + W ŷ, then L2-normalised | Omni-swarm 256→64, D2SLAM |
| `pca32` | same, 32 comps | 32 | same | MAD-DR's direction, no training |

- PCA basis: `cv::PCA` (OpenCV-native, covariance via `cv::calcCovarMatrix`) fitted on **scene 1's stores only**
  (every keyframe of both agents, every feature). Evaluated on both scenes: scene 0 is held out, scene 1 is in-sample
  (the D178 discipline). The basis (μ + W, 33 kB at 64 comps) is model state that both ends load, like the ONNX. It
  never goes on the wire, and its sha256 goes in the manifest.
- A per-component scale is used because one per-keyframe max would let the first component (largest variance) set
  the step for all of them; this is today's max rule applied per component. PCA decode renormalises to unit length because ALIKED's descriptors are unit rows, and that is the input
  manifold LightGlue was trained on. `int4` and `int8` decode without renormalising, as today's codec does.

Keypoint count K: 1024 (today) only; K 512 (the LightGlue README's operating point, survey A4) comes from job 70 (§2).

Geometry coding G (what carries each keypoint's pixel and 3D point; it affects only the verifier's PnP side
and the bytes):

| id | keypoint | 3D | bytes/kp | note |
|---|---|---|---|---|
| `xyz` | 2 × f32 | x, y, z f32 (**today**) | 20 | |
| `z32` | 2 × f32 | z f32; x, y = (u − c) z / f at the verifier | 12 | the production formula on the same floats; expected bit-identical, checked |
| `disp16` | 2 × f32 | disparity u16 at 1/16 px = round(16 f_x b / z), 0 = none | 10 | SGBM's native disparity resolution |
| `kp16` | 2 × u16 at 1/32 px | disp16 | 6 | arch §15.7 has u16 keypoints at 1/4 px; 1/32 px fits in 16 bits at 1280 × 720 |

The finite / non-finite pattern of the 3D points is preserved (VerifyMatched skips NaN x or z). The tool reports
the max |Δz| and max |Δx| that each G introduces, over all pair keyframes.

## 4. Bytes: exact, at the codec

- Each setting's response is built and encoded with `WireCodec(Framing::kFragmented).EncodeFeatureResponse`, the
  framing the transport requires. The ByteRecord's `payload_bytes` (serialized), `transmitted_bytes` (after zstd if
  the codec compresses MATCH_RESPONSE) and `wire_bytes` (+ (28 B UDP/IP + 6 B fragment header) per 1,466 B fragment)
  are recorded per keyframe; wire_bytes is the §15.7 / gate quantity. A response over 255 fragments cannot be sent
  (`Fragment` refuses it): fp32 at K1024 is ~545 kB ≈ 372 fragments, so it is a control only.
- The codings that the current `wire.fbs` has no field for (int4, bin1, pca, disp16, kp16) are packed into the
  existing `descriptors` byte vector, with `feature_bytes`/`header_bytes` set to match. Unused vectors are left
  empty. This is a **[PROXY]** of a dedicated schema field: a FlatBuffers vector costs a 4 B length plus ≤ 3 B
  alignment and a 2 B vtable slot, so the proxy is within ±16 B per keyframe of a real schema.
- K is fixed per setting, so bytes per keyframe are near-constant. The table reports the mean over the pair
  keyframes as an absolute number next to the reduction against R. For scale, the 3.2 sweep measured today's
  int8/K1024/xyz at ~136 kB/KF (M).
- Prediction gate (I6, ≤ 15 % predicted-vs-measured): for the combination row, the tool also reports wire bytes
  predicted from the closed-form size of the new layout (per-KF header + K × bytes/kp + FlatBuffers envelope, plus
  34 B per fragment) against the measured ones, so the predictor can take the new formula if it is adopted.

## 5. Pre-registered adoption rule (D233)

**Reference R** = `int8 / K1024 / xyz` (today), run in the same process.

**Controls** (in-run, measured before any setting is judged):
- `R±seed`: R re-verified with the same matches and 4 other RANSAC seeds (per-pair seed + 1..4). This measures how
  far a verdict or pose moves when only the RANSAC samples change. Any codec changes the match set and so the
  samples, so a difference inside this envelope is not attributable to the codec.
- `fp32 / K1024 / xyz`: decode-free, i.e. what int8 itself costs.

**Precondition.** R reproduces the production verdict (`prod_reason`) on all 120 pairs. If it doesn't (Kaggle AVX2
vs production's AVX-512 can move a float near a threshold), say so. Everything stays measured against the in-run R.

**Definitions.**
- A pair is *seed-stable* if R and all 4 reseeds agree on accept/reject.
- A *codec flip* is an accept ↔ reject change against R on a seed-stable pair. Flips on seed-unstable pairs are
  reported, never counted.
- Reason changes between two rejections are reported only.

**Bars, per coding.** Each of the 5 non-today descriptor codings and the 3 non-today geometry codings is judged on
its own row, with the other factor at today's value. K 512 is judged by the same bars on job 70's knobs rows against
their 1280x720:1024 row. Each bar holds on each scene separately;
scene 0 is the held-out claim for PCA.
1. **Flips:** 0 codec flips reject → accept and 0 accept → reject.
2. **Inliers:** median over R-accepted pairs of pnp_inliers(S) / pnp_inliers(R) ≥ 0.90. Mechanism: the edge's
   information prior is ProxyInformation(n) with σ ∝ √(30/n), so a 10 % inlier loss widens the edge's σ by ≤ 5.4 %.
3. **Bytes:** wire bytes per keyframe (fragmented, §4) at least 15 kB below R. That is ≥ 1 s of airtime at the
   15 kB/s low end of the link budget (survey §6).

A coding is **adopted** only if it clears 1, 2 and 3. It goes to **M decides** if it clears 2 and 3 with no
reject → accept codec flip and at most 1 accept → reject codec flip per scene. Otherwise it is **not adopted**.

**Pose accuracy (reported, flags to M, not a bar).** D63's pose_err_deg and gt_err_t/r on the pairs both accept.
If S's median pose_err_deg in a scene exceeds the largest median of R's 5 seeds, the coding is flagged to M even if
it clears the bars.

**Combination.** The recommended wire format is the adopted descriptor coding + adopted geometry (+ K 512 if job 70 adopts it, then re-measured). Its
factorial row must itself clear 1-3 (interactions are measured, not assumed). If it doesn't, the drop-one rows
of the factorial decide, and M picks.

**If the controls fail.** If fp32 itself has a codec flip or misses bar 2 against R, int8-today is already at or
below the floor this sample can resolve. Nothing is adopted without M, and the floor is reported.

**Reach (stated with every claim).** Each scene has 30 accepted and 30 rejected pairs. Rejects were sampled 6 per
reason, so the counts are not population rates: 0 codec flips bound the per-reason-stratified flip rate at ≤ 10 %
per scene and direction, ≤ 5 % over both scenes (95 %, rule of three). This is a screen. A larger pair sample is
the follow-up before an adopted coding goes in a gate.

## 6. Where it runs

Kaggle CPU kernel via test-compute-session's off-AWS harness (`infra/eval/ci_kaggle_launch.sh`): 4 vCPU, AVX2,
`OPENCV_FOR_THREADS_NUM=4`. D233 approved by M as written, plus the two reporting adds of §4 (2026-10-24). It is an accuracy study, so it is hardware-agnostic. Timing is not measured: LightGlue
always sees 128-D input, so its cost changes with K only, and job 42 already measured K 1024 vs 512. Inputs: the 4
local stores (~650 MB, uploaded as a private Kaggle dataset; no AWS egress), the pairs CSVs, the pinned LightGlue
ONNX.

## 7. Outputs (evaluation/runs/ph3/bytes_3.6_<date>/, §36)

`bytes.csv` (setting, keyframes, payload / transmitted / wire bytes mean), `pairs.csv` (pair × setting: matches,
essential, pnp, reason, gt_err_t_m, gt_err_r_deg, pose_err_deg), `geometry.csv` (max |Δz|, |Δx| per G),
`pca.yaml` (basis sha256, explained variance at 32/64, fit set size), `order_check.txt`, `manifest.yaml`. Then
`analyze.py` produces the summary (per setting: bytes, flips by direction and scene, inlier ratios, pose error
deltas, verdict per §5), and `plot.py` plots accuracy against bytes.
