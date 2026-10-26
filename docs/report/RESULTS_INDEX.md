<!-- Compiled 2026-10-24 by M from main (3c3befb+); every number quoted from the cited D-entry or run file. Source for the video claims and the report results section. -->
# Ariadne: best stand-behind result per subsystem (for demo video + technical report)

Compiled 2026-10-24 from `main` (post-consolidation 59fe978 + video commits). Read-only pass over
`docs/DECISIONS.md`, `docs/COMPLETION_REGISTER.md`, `docs/STATUS.md`, `docs/report/*.md`,
`docs/POST_DEMO_BACKLOG.md` and the `evaluation/runs/**` files named below. Numbers are quoted as written. Where I
re-derived a number from a run file, the row says "derived here".

**Tags:** `measured` = a run file or a D-entry result we can point to; `published` = a number taken from a paper
table as cited in the repo (not re-run by us); `design` = built and unit-tested only, or a projection;
`UNVERIFIED` = no file found that backs it (what I searched is listed).

## Headline per subsystem (the one line to use; details and caveats in the sections below)

| # | subsystem | headline | comparator strength |
|---|---|---|---|
| 1 | Perception | ALIKED+LightGlue loop verification: pose AUC@10° 0.607 vs ORB 0.284, better on 11/11 EuRoC flights (D63) | matched baseline |
| 2 | Restoration | Active exposure under a +2-stop event: saturation cleared in 2.35 s, ATE 16.1 vs 19.1 mm with fixed exposure (D147) | matched baseline (one replay run) |
| 3 | VIO / SLAM | EuRoC full system 0.066 m avg, at or better than ORB-SLAM3 on 7/11 flights; in class 10/11 (D87) | **published** |
| 4 | Place recognition | EigenPlaces R@1 0.626 vs mean-bit 0.444 / random 0.167 on EuRoC (D62); sim cross-agent R@1 0.75 has no baseline | matched baseline (EuRoC only) |
| 5 | Robust back end | Certified (bounded gap, chordal surrogate) on 7 SE-Sync benchmark graphs, objectives = SE-Sync Table 1 to 4 s.f.; rim not certified (D74) | **published** |
| 6 | Comms | Known-entry gate: merged ATE 3.78 m → ≤ 0.0147 m (seed-mean), 0/560 tested true edges rejected (D178) | matched baseline |
| 7 | Allocation | CBBA = SGA 40/40, CBBA/optimum mean 0.994, worst 0.983 (vs ≥ 0.5 guaranteed); 226 B/msg (D115, D115a) | **published bound** + exact oracle |
| 8 | Egress | 12/12 responder routes GT-valid; final route = GT geodesic (4.44142 m) (D113) | GT |
| 9 | Exploration | maze_short final coverage 0.757 vs 0.713 / 0.681. The pre-registered time-axis claim FAILS | weak (gate FAIL) |
| 10 | Reconstruction | Live 3-agent splat: 259,994 Gaussians, 0 drops, determinism-neutral, wall seeds 92.9 % within 10 cm (cpu_init) | none (no real-scene quality metric) |
| 11 | Agentic | Matched Bedrock/classical pair: Bedrock agent-0 choice differs on 6/18 replans; both replay verdicts PASS | behavioral comparison only; one scene/seed, no mission-quality outcome |
| 12 | Perf / cost | Collab sweep $0.0875; pre-merge ~$0.035; replay RTF 0.836 (below real time) | none (absolute numbers) |
| 13 | Determinism | Recorded live runs replay byte-identically (live vs replay ×2); estimator invariant to thread count | self-check |

**Comparator split:**
- **Published comparator:**
  - EuRoC VIO / full system: ORB-SLAM3 T-RO 2021 Table II, plus Kimera / VINS-Fusion / BASALT from the same table.
  - TUM fr2/xyz: ORB-SLAM2 T-RO 2017 Table III class.
  - Certificate: SE-Sync Table 1.
  - CBBA: the 50 % bound (Theorem 2), plus an exact oracle.
  - GTSAM 4.3 oracle (D52).
- **Matched baseline:**
  - ALIKED+LG vs ORB.
  - D178 gated vs ungated.
  - Active vs fixed exposure.
  - EuRoC retrieval vs mean-bit / random.
  - PCM→GNC vs GNC-only.
  - Egress vs the GT geodesic.
  - Volumetric / traversability vs 1.8.
  - Global descriptor int8 vs fp32.
  - Bedrock vs classical chooser on the same scene/seed (matched decisions and replay; mission outcome not scored).
- **Weak:**
  - Spectral vs greedy: mm-level, 3/4 cells.
  - Per-metre exploration: only vs the 3 cost-free rows.
  - Late coverage: margins ≤ 0.006 on ci_tiny.
- **No usable comparator:**
  - Reconstruction: no real-scene quality metric.
  - Agentic mission performance: a 500 s matched chooser pair has completed and replayed deterministically, but map/route/task outcomes were not scored against ground truth.
  - Sim retrieval: a size ablation only.
  - COOL: no run.
  - Exploration on its pre-registered claim: it loses.
  - XFeat: negative vs ORB.
  - SGBM / LK.

---

## 1. Perception / registration

| claim (on-screen line) | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| ALIKED + LightGlue (through cv::ALIKED / cv::LightGlueMatcher) doubles loop-candidate pose accuracy over ORB | pose AUC@10° **0.607 vs 0.284** (+0.322); C ≥ A on **11/11** flights; AP_geom (fixed denominator, all 676 positives) **0.807 vs 0.303** | matched baseline: ORB + ratio, the 1.6 verifier's RANSAC E, on the same 4815 EP-376 top-5 pairs | `docs/DECISIONS.md` D63; design `docs/design/PERCEPTION_2.1.md` §12 | EuRoC, 752×480 export. **4 confident wrong verifications** (210–264 inliers, 37–59° rotation error, Vicon rooms), so E inliers alone are not a loop gate. Zero-false-accept recall ≈ 0 for every row. LightGlue over mutual NN: only +0.042 AUC@10° | measured |
| Our OpenCV-dnn exports are exact against upstream PyTorch | ALIKED keypoints 100 % within 9.3e-5 px, descriptors within 4e-6, LightGlue match sets identical (D63). XFeat graph: maps ≤ 8e-6 relative, 4096/4096 keypoints, min descriptor cosine 1.000000 (D50). EigenPlaces cosine 1.0000000 (D62 / SMOKE §4r) | upstream PyTorch reference outputs | D50, D62, D63; `tests/unit/test_xfeat_reference.cpp` | Fidelity, not accuracy | measured |
| DNN front-end timings on a quiet CPU host | g4dn.xlarge **CPU** (Xeon 8259CL, 4 vCPU), p50 at 4 threads: XFeat 736×480 **41.5 ms**; EigenPlaces 640×360 (classic) **311.8 ms**; ALIKED-n16 K1024 1280×720 **1998.6 ms**; LightGlue pair 1280×720 **921.0 ms** | none (absolute) | `evaluation/runs/l5_timing/20261021T211117Z-gpu-e2b3e3/timing_t4.csv` (`t4` = 4 threads, not a T4 GPU) | Do not mix with D50's "12–15 ms with 8 threads" (dev host, under different load). D60: the XFeat ≤ 16 ms budget is **unmet** | measured |
| Retrieval-side CLAHE has no effect on EuRoC | ΔR@1 +0.013 / −0.010 / +0.013 (clean / dark1 / dark2), all n.s. (p ≥ 0.45) | CLAHE off, same queries | D92; `evaluation/runs/retrieval_3.1/` | The darkened proxy is easier than clean (cause UNVERIFIED), so not a dark test | measured (neutral) |
| ALIKED l2conv rewrite: 2.37× faster but not adopted | p50 at 8 threads 1112.7 → 469.9 ms | production graph | D235; `evaluation/runs/perf/pipeline_profile_20261024/` | NOT adopted: 1 flip, descriptor diff 7.08e-6 > 4.9e-6 | measured (negative adoption) |
| Local descriptor wire coding (3.6) | today's int8 payload **135.7 kB/KF**; post hoc: int4 79.4 kB and 1-bit 33.6 kB reproduce fp32 verdicts 120/120 | fp32 control | `docs/report/phase3_3.6.md`; D234; `evaluation/runs/ph3/bytes_3.6_20261024/` | **No coding adopted** under the pre-registered rule; int4/1-bit results are post hoc only | measured |

- **XFeat vs ORB inliers / registration success / ms per frame: UNVERIFIED, never measured as such.** The only
  measured XFeat-vs-ORB result is NEGATIVE: XFeat at keyframes regressed 6 of 7 passing EuRoC flights, so ORB ships
  on the EuRoC VO front end (D60). Searched: D50, D60, STATUS 2.1 close-out table, `l5_timing`.
- **SGBM accuracy: UNVERIFIED.** WLS confidence is built, off by default, "confidence AUC not measured" (STATUS
  2.1 close-out, D61).
- **LK accuracy: none measured.** Known-gain LK is checked only through the D147 replay traces.
- **D233** is a pre-registration only; its result is D234 (row above).

## 2. Restoration / dark world (Phase 5)

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Active exposure recovers from a +2-stop overexposure event in 2.35 s | Failure E: reacts at t = 20.0 s, saturation 0 by **t = 22.35 s**; ATE **16.1 mm vs 19.1 mm**; **169 vs 249** keyframes | matched baseline: the same run with fixed exposure (r07) | D145–D147 Results; `evaluation/runs/ph5/d145_rows_20261023T113916Z/` | Replay only, sim `ci_tiny_dark`, agent 0, 1000 frames, seed 1, one run per row. Live `light_config` sink not wired | measured |
| The exposure controller is deterministic under replay | trace checks **171/171** (r05) and **168/168** (r06); one-frame latency (50 ms at 20 Hz) | its own pre-stated checks | same | | measured |
| Lamp calibration: Lambertian incidence term is needed | held-out log-RMS **0.048 with vs 0.167 without** | the model without the cosine term | D133 | Sim render calibration (`rig_stereo_v2`) | measured |
| Finding: a co-moving lamp costs accuracy by itself | fixed light, brightness-matched **4.2 mm** (r10) vs carried lamp **14.8 mm** (r01); lamp at 50 % **150.6 mm**, at 15 % **1317 mm** | brightness-matched fixed light | D145–D147 Results | A finding about lamps, not a win for Ariadne. Cause (coverage of the far field) UNVERIFIED | measured |

- **Falloff normalisation (Stage 3) on vs off: no measured gain.** r02's trajectory is byte-identical to r01's
  (14.8 mm both). By design it only feeds the splat. **Features on the proxy are NEGATIVE: +2.9 mm** (r04 vs r03).
- D142 lighting rows (dark + lamp 15.2 mm vs fixed light 2.5 mm vs lit 3 mm) are PRELIMINARY with brightness not
  matched (D142, D145). Use r10 vs r01 above instead.

## 3. Pose estimator / VIO

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Full system (stereo-VIO + loops) at or better than ORB-SLAM3 on 7 of 11 EuRoC flights | avg ATE **0.066 m** (summary.md 0.0656); MH_01 0.021, MH_02 0.028, MH_03 0.033, MH_04 0.046, MH_05 0.044, V1_01 0.038, V2_01 0.030 | published: ORB-SLAM3 T-RO 2021 Table II (avg 0.035; per flight 0.036 / 0.033 / 0.035 / 0.051 / 0.082 / 0.038 / 0.032) | D87; `evaluation/runs/2.6/euroc_full_aws/20261022T193510Z-cpux86-655144/summary.md` | Keyframe-refined ATE (D57), single run, seed 1. V1_01 is a tie at 3 decimals (0.0382 vs 0.038). Average is between BASALT 0.051 and Kimera 0.119, **not at ORB-SLAM3**. V1_03 **0.288 = 12× ORB-SLAM3**. V1_02, V2_02, V2_03 are 2–5× ORB-SLAM3. In class on 10/11 | measured vs published |
| VIO only (loops off) is in the published stereo-inertial range | avg **0.120 m**, **9/11** in class | published (PHASE2_PLAN §1 reference table = ORB-SLAM3 T-RO 2021 Table II, per D86): Kimera 0.119, VINS-Fusion 0.138 (class bar); not at BASALT 0.051 (BASALT averaged over 10 flights, D87) | D85, D86, D87 | "At the weak end, not state of the art" (D86 wording). V1_03 0.329 and V2_02 0.110 out | measured vs published |
| TUM fr2/xyz in class | ATE **0.0139 m** vs target **0.029** | published: worst of ORB-SLAM2 T-RO 2017 Table III RGB-D class | D100; `configs/benchmark/tum_regression.yaml` | VO, loops off (the class is full SLAM) | measured vs published |
| TUM fr1 drift | 1 s translation RPE fr1/xyz **10.7 mm**, fr1/desk **16.2 mm**, fr2/xyz **2.7 mm** | none | D136 | Analysis of the pinned outputs; no rerun | measured |
| Solver cores agree with the oracle | PGO vs **GTSAM 4.3: 100/100** (energy ≤ 1.5e-12 relative); windowed BA one free pose vs `cv::solvePnPRefineLM` **1.3e-9 m / 2.8e-9 rad** | oracle: GTSAM 4.3; OpenCV PnP refine | D52 | Synthetic graphs | measured |
| Estimator is thread-count invariant | `estimate_body.tum` bit-identical at OpenCV pools 16 / 5 / 2 (3 agents × 4001 poses) | — | D232; `evaluation/runs/perf/pipeline_profile_20261024/` | x86 only | measured |

- **TUM fr1 is NOT in class:** fr1/xyz 0.0496 vs 0.014 (Kerl 2013), fr1/desk 0.0581 vs 0.037 (D100). Cause open.
- **Ceres oracle: UNVERIFIED.** D52 names GTSAM only. Searched D52 and the register "oracle 100/100" row.
- **Initialisation time: UNVERIFIED.** Not found in D59/D85/D87. The only init-related number is sim_v2 IMU-full
  initialising with ATE 3.06 mm on 1000 frames (register L2, D101), which is not a time.
- **EuRoC RPE: not found** (only MH_05's "D58 has 20 % less RPE than coasting", D60).

## 4. Place recognition

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| EigenPlaces (cv::dnn) beats a mean-bit baseline on EuRoC | pooled R@1 **EP-376 0.626**, EP-752 0.677 | matched: mean-bit 0.444, random 0.167, same 297 queries | D62 | EuRoC, 11 flights, correlated queries; bar 2 missed by 15/297 | measured |
| Retrieval in our sim, including across agents | pooled R@1 **0.7049**, R@5 0.8852 (640×360); cross-agent R@1 **0.7500**, R@5 0.9643 | 1280×720 (R@1 0.7049, McNemar p = 1.0): **size ablation only, no method baseline** | `evaluation/runs/retrieval_sim_d90/rows/rows.csv`; D90 | 61 queries, all agent 0's, one clean scene. "No evidence full resolution helps", not "equal" | measured |
| Global descriptor 4× smaller with no retrieval loss | int8 + per-vector scale **258 B vs 1024 B**; identical hits on all 297 queries | fp32 | D93; `evaluation/runs/retrieval_3.6/20261021T212015Z` | EuRoC only. The wire carries fp16; int8 stays a row (D171) | measured |
| Exact index loses nothing at this scale | Annoy top-1 = exact on 297/297 | cv::ANNIndex | D62 | | measured |

- **No baseline on the sim data** (no mean-bit or random row in `retrieval_sim_d90`). For a method comparison, use
  only the EuRoC rows.

## 5. Robust back end

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Certified on 7 SE-Sync benchmark graphs (f − f* ≤ bound, for SE-Sync's isotropic chordal cost) | certified: parking-garage (f − f* ≤ 0.024 %), tinyGrid3D, smallGrid3D, sphere2500 (≤ 1.1e-3), torus3D (≤ 3.6e-3), cubicle (≤ 2.9e-2), grid3D (≤ 0.13); objectives reproduce **SE-Sync Table 1 to 4 s.f.** on the six graphs that report f (e.g. sphere2500 1687.01 vs 1.687e3; tinyGrid3D/smallGrid3D report no f) | published: SE-Sync Table 1 objectives | D74; `evaluation/runs/2.5/certificate_sesync/README.md` | **rim NOT certified** ("unknown, not wrong"; gtest exit 1). torus3D and grid3D stopped at the LM iteration cap and still certify (weak duality). grid3D took 7176 s. The runtime flag is off | measured vs published |
| PCM → GNC survives perceptual-aliasing groups up to the analytic limit | breakpoint **90 %** (groups of 10), **70 %** (two growing groups) | GNC alone: no breakpoint (worst 7/10, 5/10 at 70 %) but **~10× slower** (165–313 ms vs 15–118 ms) | D75; `evaluation/runs/2.5/outlier_ramp/README.md` | Synthetic, 60 loops/graph, 10 seeds/rate. PCM breaks exactly where the largest false group outnumbers the true loops | measured |
| GNC rejects planted outliers | 10/10 outliers rejected, 10/10 certified after GNC | before GNC: 6/10 certified | D74 | Synthetic 60-node | measured |
| Robust joint multi-agent solve | 10/10 true inter-agent loops kept, 0 false, on 5 seeds | true-loops-only solve (equal ATE) | D76; `tests/unit/test_joint_robust.cpp` | Unit test | measured |
| Loop noise model held out | mean good-loop NEES **5.24**; tail 4.4 % above χ²₆(0.99) | criterion ≤ 3 % (**missed**, accepted by the maintainer) | `evaluation/runs/2.5/loop_closure/README.md` | V1_02 held out | measured |

## 6. Comms protocol

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Known-entry gate turns a 3.78 m aliasing failure into a ≤ 1.5 cm map (seed-mean per cell), rejecting no true loop | scene 0 budgeted cells worst seed-mean merged ATE **0.014675 m** (4/4 < 0.10 m; worst single run 0.0211969 m), down from **3.78–3.79 m**; true edges rejected **0 of 269 + 0 of 291 = 0/560 tested** | matched baseline: the same cells without the gate (sweep `eee456`) | D178 RESULT; `evaluation/runs/collab_d178/verdict/verdict.yaml`, `cells.csv` | Pre-registered, all 4 criteria PASS. Known starts only: unknown-start aliasing is still 3.5. 10 scene-0 true edges were never tested (ungated). The similarity arm is ungated and keeps its 3.09 m alias | measured |
| Spectral selection is near-optimal on small instances | selected/optimum mean **0.970**, worst 0.650 | greedy alone **0.936 / 0.533**; exact oracle, 60 instances | D170 | Unit, synthetic. Bound valid 60/60 but loose (1.3–6.4×) | measured (unit) |
| Spectral vs greedy in sim (gated sweep) | spectral ≤ greedy in **3/4** cells (s0 p25 0.013494 vs 0.014675 m; s0 p50 0.014423 vs 0.014469; s1 p50 equal; **s1 p25 worse** 0.007682 vs 0.007598) | greedy, same budget | `evaluation/runs/collab_d180_bytes/gate.yaml` (dominance), `collab_d178/verdict/cells.csv` | Differences are sub-mm to ~1 mm, 3 seeds: not a strong claim | measured |
| Budgeted exchange vs the request-everything arm (derived here) | scene 1, collapsed_v1: budgeted p25 first-transmission **2.27–2.45 MB**, merged ATE 6.8–8.6 mm, all seeds merged; similarity arm **14.8–16.0 MB**, seed 2 aliased at 3.09 m, seeds 1 and 3 report `merged_agents: 1` | similarity arm (unbudgeted, ungated) | `evaluation/runs/collab_d178/gated_e31421/*/results.yaml` (`total_wire_bytes.first`, `ate_merged`) | **Derived here, not a pre-registered claim.** Two variables differ (budget and gate). The meaning of `merged_agents: 1` is UNVERIFIED. Scene 0 similarity: 7.95–8.27 MB, 17–37 mm | measured (derived) |
| Per-message byte prediction is accurate | match_response per message **−0.4 % to +0.4 %**; match_request all 24 budgeted rows ≤ 15 % (worst 10.65 %, median 0.77 %) | measured at the serializer | D180 RESULT; `evaluation/runs/collab_d180_bytes/` | **Class-total byte table FAILS** (below) | measured |
| Unit byte prediction on a disjoint scene | response **7.2 %**, request **0.29 %** (gate ≤ 15 %) | — | D170 | Unit, synthetic descriptors that compress to 0.21. Real ORB is nearly incompressible, so not the I6 evidence | measured (unit) |
| Global descriptor int8 is lossless | 258 B, identical top-k on 297 queries | fp32 1024 B | D93 | see §4 | measured |

- **Vertex-cover (D181):** **0.0 % saving on 24/24** budgeted rows. Not a win; deferred.

## 7. Allocation

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| CBBA matches the centralised greedy every time and is within 2 % of the exact optimum on every linked replan (mean 0.994, worst 0.983) | CBBA = SGA **40/40**; CBBA/opt mean **0.994** (0.9938), worst **0.983** | greedy pick 0.985 / 0.900; exact oracle per round; published guarantee: CBBA ≥ 50 % of optimal (Theorem 2, ALGORITHMS §22.3) | `evaluation/runs/allocation_4.4_replay/phase4_allocation.jsonl` (verified here); D115 | Replay, one spool, 2 agents. Only 12 of the 40 replans are linked (where the ratio is computed). L_t = 1 | measured |
| Sparse consensus messages | **226 B/msg** (15820 B / 70 msgs, verified here); 11.8 KB per spool (was 123.4 KB) | the dense message, 2.4 KB | D115a; same jsonl | | measured |
| Live: auction picks over the radio | 6 coordinated / 3 no-consensus / 3 no-assignment picks; bid 6880 B, consensus 1744 B | — | D218 | 20 s, 3 agents, ideal link | measured |

- Event triggering saves only 26 % of bytes (D115). Live gate rows for 4.4 are open.

## 8. Planner / egress

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Every egress route was valid for a 0.40 m responder; the final route is GT-optimal | **12/12** GT-valid; final 4.44142 m vs GT geodesic 4.4414215 m (ratio 0.9999997) | GT geodesic; the 1.8 A* placeholder gives the same final route | `evaluation/runs/egress_4.5/egress_vs_gt.yaml`; D113 | One spool (ci_tiny, 2 agents). 0 reroutes happened. **Confidence calibration degenerate** (0.8996–0.9017) | measured |
| D* Lite = fresh Dijkstra, with far fewer expansions | equal on **180** random flip rounds, **7.6× fewer expansions** | fresh Dijkstra | D113 (unit test) | Unit test. On the spool: 771 incremental expansions over 40 replans vs 342 for one fresh search (raw; not a speed-up claim) | measured (unit) |
| Risk-aware routing moved a route out of a hazard at no extra length (1 of 3 hazard seeds) | seed 1: route risk **0.340 → 0.000**, length 4.991 → 4.991 m | w_r = 0 | `docs/design/EGRESS_4.5.md` §10 table; lengths in `evaluation/runs/egress_4.5_hazard/s1_w{0,1}/results.yaml` | The 0.340 figure is in the design doc, not a run file. Seed 0 unchanged (4.48284 m both). **Seed 2: +4.4 % length for nothing** (5.61213 → 5.86066 m; proxy saturation) | measured |
| Map quality vs the Phase 1 placeholder | coverage **0.942 vs 0.933**; occupied precision **0.865 vs 0.707** | matched: the 1.8 OctoMap placeholder, same spool | D110; `evaluation/runs/volumetric_4.1/` | | measured |
| Responder traversability | precision **1.000**, recall **0.900** | 1.8: 1.000 / 0.898 | D111; `evaluation/runs/traversability_4.2/` | | measured |
| Structural-risk proxy passes its bar | AUC **0.831 ≥ 0.75**, control FPR 0 | pre-registered bar | D112; `evaluation/runs/risk_calibration_4.2/` | Saturates: FPR 0.34 at threshold 1.0 | measured |

- **Nothing beats FAR / TARE / A\*.** FAR and A* agree on all 12 spool routes (FAR 4.403 m = the string-pulled
  lattice). "FAR ≤ lattice" is FALSE in general (see do-not-claim).

## 9. Exploration (4.3 gate FAIL)

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Highest final coverage on the maze | maze_short coverage at 200 s **0.757** | coverage-greedy 0.713, TARE-inspired 0.681 (mean of 3 seeds) | `docs/report/phase4.md` Table 4.1; `evaluation/runs/nbv_4.3_gate/gate_summary.yaml` | Best only at 180–200 s; below coverage-greedy at every T ≤ 160 s | measured |
| Most coverage per metre flown, against the cost-free baselines | ci_tiny 10 m **0.837** vs greedy 0.735 / random 0.790 / uncertainty 0.797; maze_short 10 m **0.349** vs 0.262 / 0.288 / 0.275 | coverage-greedy, random, uncertainty-only only | `evaluation/runs/nbv_4.3_gate/coverage_per_metre.txt` | Holds 5–30 m on both scenes **only against those three rows**. Against TARE-inspired it loses at ci_tiny 10 m (0.849 vs 0.837) and maze_short 15 m (0.428 vs 0.422). Against 1.8 it loses at ci_tiny 5 m (0.482 vs 0.440) | measured |
| Late-mission lead on ci_tiny | best of all six rows at every T from 80 s (80 s 0.938; 100 s 0.952 vs 0.946; 200 s 0.962 vs 0.960) | all 6 policies | `gate_summary.yaml` (80 s checked here); phase4.md Table 4.1 | Margins at 100–200 s ≤ 0.006 | measured |
| Phase 4 beats the Phase 1 placeholder | AUC **0.884 vs 0.789** (ci_tiny), **0.440 vs 0.237** (maze_short) | the 1.8 nearest-frontier placeholder | `gate_summary.yaml` | Weak: every baseline, random included (0.876 / 0.374), also beats 1.8. 1.8 still leads ci_tiny at 20 and 40 s | measured |
| Livelock fix (dev) | ci_tiny seed 2 coverage at 60 s **0.878 vs 0.728** gated | the same seed in the gate | `evaluation/runs/nbv_4.3_ablation/abort_check/` | One development run, not gate evidence; the 47 other ablation runs are pending | measured (dev) |

- **AUC: third of six on both scenes** (ci_tiny 0.884 vs TARE-inspired 0.895 / greedy 0.892; maze_short 0.440 vs
  greedy 0.495 / TARE-inspired 0.443). "TARE parity" is a narrow loss on both, not a win. The TARE row is
  "TARE-inspired" (no goal commitment, no waypoint extension).

## 10. Reconstruction

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Live splat from the real 3-agent runtime, determinism-neutral | 71 KF, **0 drops**, 4 submaps, **259,994 Gaussians**, 8.3 MB; run digest = the no-tee reference; placement ≤ **2.2e-16** vs briefing | no-tee reference run | `evaluation/runs/ph6/live_e2e_20261024/README.md`; D230/D231 | **cpu_init only, no GPU training**; 20 s lit `maze_demo_0` | measured |
| Full 200 s mission trained on AWS T4 | **1,476 KF**, 51 submaps, **2,590,315 Gaussians** (batch, 100 it/KF); 270 novel-view orbit frames | — | `evaluation/runs/ph6/shared_model_video_20261026/` (recovered training/render records and S3 manifests) | Recorded 3-agent `maze_demo_0` mission 626abe; gsplat 1.5.3, briefing placement. The reveal is editorial, not live reconstruction footage | measured |
| Incremental GPU training throughput on that recording | live budget 15 it/KF: **349.7 s for a 200 s mission**, 203 ms/KF median (276 p95) | mission keyframe arrival rate | same evidence, `training_remote.json` | **0.57× real time**, not a real-time training claim. Batch model is used in the video | measured |
| Seed geometry vs ground truth | wall submaps (agents 1, 2): seeds **92.9 % within 10 cm** of the GT ESDF; agent 0's submaps 65.8 % and 47.3 % | GT ESDF | same README | Percentage of seed points in those submaps, not of submaps. Agent 0's submaps are floor-heavy | measured |
| gsplat cost on a T4 | full loss at 640×360: **10.9 ms/iter** (100k), **15.3 ms** (250k); render 1.7–3.0 ms | — | D196; `evaluation/runs/ph6/bench_t4_*` | Kaggle idle T4, synthetic scene. "7–10 iterations/KF" is a **[PROJECTION]** | measured / design |
| Trainer recipe works (synthetic) | batch 100 it/KF: train-view **22.2 → 25.1 dB**, held-out 17.8–20.1 dB vs mean predictor 12.9 dB | predicting the mean (12.9 dB) | D198; `evaluation/runs/ph6/train_smoke_20261023T1340Z` | Synthetic corridor. **Live budget (15 it/KF): +0.5 dB only**, so live quality ≈ seed quality | measured |

- **The recovered AWS ma200 run has training-view PSNR only:** live budget 23.14 → 25.71 dB,
  batch 23.34 → 27.40 dB (median before/after on the trained keyframe). These are fitting diagnostics,
  not held-out or novel-view quality scores. No Phase 6 held-out PSNR/SSIM/LPIPS, Chamfer, double-wall or
  exposure-delay result is evidenced. The Phase 1 placeholder has held-out PSNR 17.5 dB, SSIM 0.718,
  LPIPS 0.68; accuracy 21 mm, completeness 0.66 (STATUS 2026-10-15), from a different untrained-init run.
  No published comparator is used anywhere.

## 11. Agentic

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Matched chooser behavior and replay | 90 s pair: Bedrock answered **36/36** assignments over 18 replans; agent-0 choice differed on **6/18**; 118 model calls, 2,936,424 cumulative input tokens. 500 s pair: 100 joint replans and 200 commands per arm; both two-replay checks PASS, with 200 trace records and identical replay1/replay2 hashes | Same source revision `c143db14`, generated scene `89e61fecbb277f69`, runtime, renderer, and mission settings; chooser-only config difference | `evaluation/runs/agentic_planning_pair_20261025/` and `evaluation/runs/agentic_planning_pair_500s_20261025/`; full 500 s archives in the linked S3 objects | Both runs are on `maze_short_6`; 500 s run recorded 10,001 frames/agent and zero unpaired-frame drops. The 90 s arm had 34/36 movement commands accepted and two unanswered; no approval event. No paired ground-truth coverage, route-quality, or mission-completion metrics were emitted for the 500 s pair. Replay establishes reproducibility, not a mission-performance advantage | measured |

## 12. Performance / cost

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Three-agent live pipeline replay speed | RTF **0.836** (5 threads) / 0.795 (16), lens off; was 0.514 lens-on | the lens-on replay of the same 200 s inputs | `evaluation/runs/perf/pipeline_profile_20261024/README.md` | c7i.4xlarge replay. **Below real time.** ~30 % instance spread between c7i boxes | measured |
| Arm (Graviton4 c8g) vs x86 (c7i) | ALIKED-1280 at 16 threads **0.64×** time (530 vs 829 ms); LightGlue / EigenPlaces ~0.85–0.88× | c7i.4xlarge, same build | same README, item 10 | Stock arm64 OpenCV already includes KleidiCV + carotene HAL. **Not COOL.** At 1 thread LightGlue / EigenPlaces are 1.6–1.7× slower on Arm | measured |
| A full multi-agent collaboration sweep costs about 9 cents | gated sweep **$0.0875**; logged rerun $0.0882 | — | D178 RESULT | AWS spot | measured |
| A full AWS pre-merge gate costs about 3.5 cents | **~$0.035** (999 s, 54/54, ctest 708/708) | — | STATUS 2026-10-24 14:35; `evaluation/runs/premerge_aws/20261024T083919Z-cpux86-73e3fa/` | Render reused | measured |
| 11-flight EuRoC full system on AWS | $0.0439 (run 2, 1371 s) | — | `evaluation/runs/2.6/euroc_full_aws/20261022T193510Z-cpux86-655144/ledger_row.csv` | | measured |
| Per-keyframe and per-frame cost | lane ≈ 52 ms/frame (track + barrier wait); keyframe ~126 ms on 30 % of 3-agent steps; FAST ~11 ms; TUM track 16–18 ms, keyframe 24–30 ms (640×480) | — | perf README; D100 | | measured |

- **COOL (m8g, Arm-vs-Arm): not started** (0.3/7.3 blocked). No COOL number exists.
- **Lane pipelining (D229): NEGATIVE**, 0.833 vs 0.839.

## 13. Determinism

| claim | number | comparator | evidence path | caveat / scope | tag |
|---|---|---|---|---|---|
| Recorded live runs replay byte-identically (live vs replay ×2) | LLM run 5 replay 15/15; D175 3-agent POC double replay PASS + oracle A/B digest `3d10d011018c42d9` identical; D229 pipelined and kill-switch arms byte-identical on AWS; estimator bit-identical across pools 16/5/2. Cross-arch: degradation operator D digests match between Graviton4 and x86 | live vs replay ×2 | `evaluation/runs/L1/llm_live_run5_maze_short_90s/`; D218 (its runs are in `build/live/`, so the DECISIONS entry is the only evidence); D229 outcome; D232; register Phase 0 gate row / SMOKE §4e | Trajectories are compared on x86 only. `cv::ALIKED` output depends on the thread count. **Wire bytes are not bit-portable across arch** (#185: arm64 pose-graph message 419 vs 417 B) | measured |

---

## Do NOT claim (known FAIL / NEGATIVE / unsupported)

1. **"XFeat is our primary feature" on the EuRoC VO**: XFeat regressed ATE on 6/7 passing flights; ORB ships (D60).
   The XFeat ≤ 16 ms budget is unmet.
2. **"State-of-the-art / ORB-SLAM3 parity" on EuRoC**: average 0.066 vs 0.035; V1_03 0.288 = 12× (D87). Say "at or
   better on 7/11".
3. **TUM fr1 in class**: 0.0496 / 0.0581 vs 0.014 / 0.037 (D100, cause open).
4. **"Globally optimal"** without the scope "certified for the isotropic chordal surrogate"; **rim not certified**.
   The certificate proves optimality of the cost, not outlier-freeness (6/10 planted-outlier graphs certified).
5. **"≤ 15 % byte table passed"**: sim byte table FAIL, worst 0.2757 (match_response 8/24 rows over 15 %, D180).
   The 7.2 % / 0.29 % are unit-level only.
6. **3.2 ATE-vs-bytes gate**: FAIL on scene 0 before D178. Without the gate, aliasing wins PCM (3.78 m). The D178
   fix holds **with known starts only**.
7. **Place-grouped PCM (D176)**: NEGATIVE (3.7837 vs 3.7835 m).
8. **Vertex-cover saving (D181)**: 0.0 % on the budgeted rows.
9. **"Spectral beats greedy" in sim**: only 3/4 cells, mm-level. Verified-edge precision before the robust stage
   is 0.3271 (`collab_d180_bytes/gate.yaml`).
10. **4.3 NBV gate**: FAIL on both scenes, all 36 runs. Third of six on AUC. **The phase4.md sentence "coverage per
    metre ... best at every point from 5 to 30 m on both scenes" is stale for the 36-run table** (TARE-inspired and
    1.8 beat it at three points). Do not call the TARE row "TARE".
11. **"FAR ≤ string-pulled lattice"**: FALSE (hazard seed 0: 11.115 vs 4.424 m; random map #9: +12.5 %). The test is
    quarantined. FAR stays `[DESIGN]`.
12. **Egress confidence calibrated**: degenerate (0.900–0.902). No live reroute has been exercised.
13. **Hilti real-data validation (5.6)**: **0/100 on all 3 rows** (D144).
14. **Falloff normalisation improves tracking**: byte-identical trajectory; features on the proxy NEGATIVE
    (+2.9 mm). The dark / lamp / dust rows are a paused stretch (D199). Phase 5 exit NOT closed (D149).
15. **Splat quality / PSNR on the demo scene**: not measured. The live run was cpu_init. The live 15 it/KF budget
    gives +0.5 dB. The recon-gpu image was not validated on a GPU instance. gsplat has not run on AWS GPU.
16. **Agentic approval / fallback exercised**: never fired. A matched classical-vs-Bedrock pair exists, but mission-level benefit is unmeasured.
17. **Real time**: RTF 0.836 replay; ~0.147 classical and ~0.146 Bedrock for the matched 90 s live pair.
18. **COOL speed-up**: no measurement. The c8g vs c7i 0.64× is stock (KleidiCV) Arm vs x86, not COOL.
19. **ALIKED l2conv 2.37×, K 512, int4 / 1-bit codings**: none adopted (D234, D235).
20. **Cross-architecture bit-identity**: #185 wire bytes differ on arm64. The GravityPgo fixture fix (D123/D124) was
    merged, but I found no arm64 re-run evidence in this pass, so arm64 ctest-green is also not claimable.
21. **Retrieval-path CLAHE helps in the dark**: unmeasured in sim. The EuRoC proxy was easier than clean.
22. **Loop noise model calibrated**: held-out tail 4.4 % vs ≤ 3 % criterion (missed, accepted).

## Not verifiable in a file (UNVERIFIED)

- **Ceres oracle agreement.** Searched D52, the register 2.5 row and `evaluation/runs/2.5/`; only GTSAM is named.
- **XFeat inliers / registration success / ORB ms per frame.** Searched D50, D60, STATUS 2.1 close-out and
  `l5_timing`. The rows do not exist.
- **SGBM / WLS accuracy (AUC).** STATUS says not measured.
- **VIO initialisation time and EuRoC RPE.** Searched D59, D85, D87 and the `euroc_full_aws` summary.
- **D171 "two-stage vs send-everything" saving as a stated result.** Only my derivation in §6, from results.yaml.
- **Route risk 0.340 → 0.000.** Found in `docs/design/EGRESS_4.5.md` §10, not in a run file.
