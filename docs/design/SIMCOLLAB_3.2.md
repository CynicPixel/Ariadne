# Design — 3.2 sim integration: the v2 protocol over CommsNode on sim recordings (D171)

Plan 3.2/3.3/3.4 meet here. COLLAB_3.2 §9 says the sim tool switches to the new protocol under `protocol: v2` once
3.4's policy exists; D102 says no protocol traffic runs over the real link until this exists. All three Phase 3 gate
rows wait on it:
- fault injection and split-brain on the sim;
- the §15.7 byte table, with a hard 15% gate;
- ATE against cumulative bytes, spectral vs greedy vs similarity, on 2 or more scenes × 3 seeds.

Status: revised after advisor review 1 (2026-10-23). Nothing is built yet.

## 0. Preflight

**Literature.**
- **COSMO-Bench** (McGann, Potokar & Kaess, 2025, arXiv 2508.16731, §IV-B/C) is the closest methodological precedent.
  - It runs each robot's front-end over real recorded data.
  - Inter-robot exchange (scans, then loop-closure detection) is gated over time by a communication model.
    Connectivity is a function of inter-robot distance; bandwidth is shared under spatially local interference.
  - The model is fitted to the real ad-hoc Wi-Fi traces of Lajoie et al. 2026 (arXiv 2601.21063).
  - They also criticise "single-robot benchmarks partitioned to simulate multi-robot". Our recordings are genuinely
    multi-agent: two rigs in one Gazebo world, on one clock.
- **Swarm-SLAM** (Lajoie & Beltrame, RA-L 2024).
  - The front-end is local; inter-robot results feed only the back-end (verified in `lajoiepy/cslam`). So the
    front-end can run open-loop on its recording.
  - The evaluation compares spectral vs greedy at equal candidate budgets, reporting λ₂ and ATE. Our budget is bytes
    (arch §16.3), so the x-axis is cumulative wire bytes.
- **Tian et al., RSS 2018:** budgeted exchange, where the vertex cover decides what must be sent. This is the
  `sent_local_feature_set` that 3.4 prices.
- **RFComms** (D102): PER is a function of the datagram size, `1 − (1 − BER)^(8L)`. That is why §3 puts the size on
  the seam.

**OpenCV-native check.**
- Retrieval: EigenPlaces through `cv::dnn` (3.1, D90 sim export).
- Hard-pair local features: `cv::ALIKED` + `cv::LightGlueMatcher` through cv::dnn (D63, with D90's sim export).
- Geometry: `cv::findEssentialMat` with USAC, `cv::solvePnPRansac`, and SGBM depth (2.1).
- Nothing in the event simulation or the link is image analysis, so there is nothing OpenCV-native to reuse there.

**Arch.** §5.7, §15.2–15.7, §16.3, §31.1, and COLLAB_3.2 §6 (outcome) and §7 (byte gate, fixed earlier), D102.

## 1. Time model: two passes, causal release

**Pass A, per agent, per seed; cached by input sha256.** The agent's shipped front-end runs over its recording:
sim_v1 perception plus sim_v2 SLAM with D101 IMU noise, the D69 configuration. Per keyframe it writes a store record:
- `id`;
- `t_final`: the frame time at which the keyframe left the BA window, so its odometry edge became final. This is
  what IntraLoopClosure already waits for (D73);
- `T_O_C` at `t_final`;
- the odometry edge to the previous keyframe, with its covariance;
- the EigenPlaces descriptor;
- the hard-pair local features with stereo 3D points (§2).

**Pass B: a discrete-event run on the recording clock.**
- `CollabNode::AddKeyframe` takes the pose, the odometry edge and the descriptor in one call (collab_node.hpp:187).
  So a keyframe is released whole at `t_final`.
  - That is causal: nothing is known before the agent would know it.
  - The cost is one BA window of latency on the descriptor. This is stated, not modelled away.
- Each 20 ms step of sim time runs `Network::DeliverUntil(t)`, then every `CollabNode::Tick(t, net)`, then the
  releases due by `t`.
- The 3.4 selection runs inside `Tick` on its 5 s period.
- After the last release comes a drain period (`drain_s` 30 s), then the COLLAB_3.2 §6 outcome.

**The startup prefix (D87's carry).** VO keyframes before the window anchor never get a final exit edge, so they
are **never released**.
- An agent's graph starts at its first released keyframe id.
- 3.4's Swarm-SLAM chain construction changes from "ids 0..max" to "ids min..max seen". Swarm-SLAM sends every
  keyframe from 0, so the two are the same there. Here this is a one-line change, with a test.
- The count of unreleased prefix keyframes per agent goes into the manifest.

**Seeds.** "3 seeds" means 3 values of `seed`, and each one drives both passes:
- in pass A: the D101 IMU noise operator and the VO's RANSAC seeds;
- in pass B: `DeriveSeed(seed, ·)` for Philox link loss and the verifier's RANSAC.
- D is `none`: the gate scenes are lit. Dark rows belong to Phase 5.
- Pass A therefore runs 2 scenes × 3 seeds = 6 times. All arms, budgets and comms scenarios of one (scene, seed)
  share that pass A.

**Why two passes.**
- **The front-end does not consume collaboration output:** arch §15.5 says stored geometry stays in W_i and is never
  rewritten. That is Swarm-SLAM's split, and COSMO-Bench's method.
- **Cost:** pass A is shared across arms, budgets and scenarios.

**Stated [ASSUMPTION]s.**
- Broker compute (verify, solve) takes zero sim time. Measured compute time is reported beside each run (it shifts
  when edges become available, not bytes). The live runtime (L1) is the closed-loop number.
- No lookahead: a unit test perturbs every store value after its release time and checks that the outputs before
  that time are unchanged.

## 2. Hooks (full spec; the Phase-1 ORB/mean-bit path becomes the `features: orb_meanbit` ablation)

- **`retrieve`:**
  - EigenPlaces R50-256 at the D90 sim size;
  - on the wire as int8 with a per-vector scale (D93);
  - exact cosine search (one gemm, D62), threshold from the 3.1 sim evaluation.
- **`features` and `verify`: ALIKED-n16 + LightGlue, the hard-pair path (D63; locked: "ALIKED+LightGlue hard pairs").**
  - Inter-agent candidates are hard pairs by construction: a different agent, a different viewpoint, a different
    time.
  - Evidence (D63, EuRoC, 4815 EP-376 top-5 pairs with the 1.6 verifier's RANSAC E):
    - pose AUC@10°: 0.607 against the ORB placeholder's 0.284;
    - C ≥ A on 11/11 flights;
    - AP_geom 0.807 against 0.303.
  - D63 lists "3.x calls it" as its first runtime consumer. D90 supplies the 1280×720 RGB sim export.
  - **Why not XFeat:** XFeat ships in no runtime path. D60 measured it as the VO engine and did not ship it, and no
    loop-verification row for XFeat exists. It stays a §31.3 row.
  - **Extraction:** at keyframe cadence in pass A, K = 1024 (D63), 3D points from the masked SGBM depth.
  - **Wire encoding:** fp16 descriptors (128 × 2 B). int8 is a 3.6/§17.2 row, gated on a LightGlue-match-set check
    before it may become the default. The predictor prices `n_features × feature_bytes` (3.4).
  - **Verify chain (COLLAB_3.2 §11):** LightGlue matches, then the RANSAC essential-matrix gate, then PnP for the
    metric SE(3), then the D71 degeneracy gate. The broker then runs degeneracy → PCM → GNC-TLS (2.5) over the batch.
  - **Thresholds** (min matches/inliers) are re-calibrated for ALIKED+LG on the other scene (§5), never on gate data.
    D63 states that τ does not transfer MH→V, so this cannot be skipped.
  - **Cost:** about 1.2 s per warm pair (D63). Broker CPU time per run is reported, and it becomes a §16.3 secondary
    budget (time) row.
- **`solve`:** the broker's chordal init plus PGO with the 2.5 robust stage, gauge per COLLAB_3.2 §5.
- **Carry, stated:** intra-agent loop closure (D87's EuRoC numbers) stays ORB/mean-bit. Moving it to the same stack
  is a separate item that would re-run D87. It is registered, not done here.

## 3. Link

- `CommsNode(MakeLinkModel(scenario, tracks, walls), seed)`.
- Positions come from GT body tracks. The physical channel depends on where the radios really are; this is not an
  estimation leak.
- **Seam change (D102's open item): datagram size goes on the seam.**
  - `LinkModel::Sample` returns the link state. Loss is evaluated per datagram at its actual size with
    `rf::ExpectedPer(rx_dbm, σ, noise, L)`, the RFComms expression D102 already transcribes.
  - `kPathMtuBytes` stays 1500 (D102 L8, under RFComms as modelled).
  - `ScriptedLinkModel` and `FaultPlan` keep size-independent loss, so the protocol tests are unchanged.
- **Scenarios:** `ideal`, `collapsed_v1`, `collapsed_partition_v1` (configs/comms), plus `DatagramFaults`
  (duplicate, reorder, jitter) for the campaign.

## 4. Arms, budgets, runs

**Arms**, all sharing the same pass A and link seed. The plan's gate plot names three arms: spectral, greedy and
similarity.

| plot label | D170 name | §31.1 row | meaning |
|---|---|---|---|
| — | — | solo | each agent alone, per-agent ATE, no exchange |
| — | — | independent multi-agent | both agents, no exchange: per-agent ATE in its own frame, no merged map |
| similarity | `similarity` (select unset) | (reference) | every candidate above the retrieval threshold fetched; no budget |
| greedy | `greedy` (`SelectBySimilarity`) | greedy similarity | top similarity under the byte budget |
| — | — | greedy coverage | **after the gate arms**: definition pending (open item 1) |
| spectral | `spectral` (`SelectSpectral`) | Ariadne spectral + vertex-cover exchange | D170 |
| spectral, no VC | `spectral` + `vertex_cover: off` | Ariadne spectral | features re-sent per candidate |

**Budget levels: derived, not chosen.**
- Measure the similarity arm's mean fetch rate R (bytes/s of classes 3–4) on the calibration scene (the other
  scene, as in §5).
- The levels are {10, 25, 50, 100} % of R, with burst = one 5 s selection period of the level.
- The runtime default (D170's unsourced 20 kB/s and 200 kB) is replaced by 25 % of R from scene 0, recorded with
  the run that measured it.

**Scenes:** ci_tiny seed 0 and seed 1 (two worldgen seeds, 2 agents each). Scene 0 has 17 true inter-agent
opportunities (v1, the 1.6 retrieval row).
- **Precondition, measured first on AWS:** scene 1's true opportunities, from pass A and GT co-visibility.
- If scene 1 has fewer than 10, record a third scene (maze_short, a GPU render on g4dn spot) before the gate runs.
  The threshold comes from scene 0's 17: it keeps the cross-scene calibration from resting on a handful of pairs.

**Gate matrix:**
- 2 scenes × 3 seeds × {similarity, greedy, spectral} × 4 budgets, on `collapsed_v1`;
  - the similarity arm has no budget, so it runs once per (scene, seed);
- the §6 campaign, on the spectral arm.

## 5. Outputs and the byte gate

**Per run:**
- `ledger.csv`: every datagram, tagged first-transmission / retransmit / ack / heartbeat;
- `byte_table.yaml`: predicted vs measured per class 1–6, overhead rows separate (COLLAB_3.2 §7);
- `ate_vs_bytes.csv`: one row per broker epoch. Columns:
  - `t`;
  - cumulative first-transmission bytes;
  - cumulative all bytes;
  - merged ATE over released keyframes, in G;
  - per-component counts.
  The merged ATE uses one SE(3) Umeyama alignment to GT per row, and agents not yet merged are reported separately,
  never dropped silently.
- `outcome.yaml`: `consistent` / `explicitly incomplete` / `corrupt`, with reasons;
- `manifest.yaml` (§28.2).

**Byte gate (COLLAB_3.2 §7, completed here):**
- **Pairing:** every message the protocol plans carries its predicted size, keyed by message id. The ledger pairs
  each first transmission with its prediction. Predictions are made where the message is planned, by its sender:
  the broker for requests, a member for descriptors, responses, edges and graph batches. They are summed over
  every broker the run had: 3.4's open item.
- **Predictors per class:**
  - heartbeat, descriptor and request: the codec's size function on the planned fields. Exact by construction; a
    test pins it;
  - feature response: 3.4's `FetchCostPredictor` (n_features × feature_bytes plus framing, response ratio);
  - pose edge and graph batch: the §15.7 rows times a calibrated zstd ratio.
- **Overhead rows, never prediction error:**
  - retransmits;
  - acks;
  - **resync after an election.** Broker state is soft, so a new broker receives the latest graph batch and the
    full edge set again (COLLAB_3.2 §4). These sends are tagged `resync` at plan time.
- **Calibration** of the compression ratios and verify thresholds is cross-scene: scene 1's runs calibrate scene 0's
  gate and vice versa.
- The gate fails if any class is off by more than 15%.
- `python/analysis/collab_gate.py` produces the plot, the table and the verdict from the run directories, with no
  hand-edited numbers (§36).

## 6. Fault campaign and split-brain on the sim

- **Campaign:**
  - loss levels {0, 0.1, 0.3};
  - duplicate, reorder and jitter;
  - `collapsed_partition_v1` (partition 20–40 s, agent 1 disconnected 60–65 s);
  - a forced split-brain: each agent alone elects itself, then reconnects under a new epoch.
- **Gate:** every run ends `consistent` or `explicitly incomplete`; none ends `corrupt`. A replay with the same seeds
  is byte-identical: the ledger and `outcome.yaml` hashes are compared.

## 7. What stays

- `protocol: v1` (`ariadne_sim_collab` today) stays the 1.6 baseline and the Phase-1 gate path, unchanged; `bytes.json`
  is unchanged.
- v2 lives in its own translation unit, behind `protocol: v2` in the collab config.

## 8. Order and cost

1. 3.4 merges first. It needs a rebase onto main and its own AWS pre-merge.
2. Then this, on `feat/3.2-sim`:
   - the seam change and tests;
   - pass A with its store;
   - pass B over CommsNode with its tests: no lookahead, determinism, outcome;
   - the gate harness.
3. The sweep runs on AWS (cpux86): pass A once per scene, then pass B in parallel there.

## Open items

1. **"greedy coverage" (§31.1).** Check Tian et al. RSS 2018 / IJRR 2021 for a budgeted maximum-coverage
   formulation with a (1 − 1/e) greedy. If it is there, that is the sourced definition. It is built after the three
   gate arms.
2. **Seam change.** Confirm that `ExpectedPer(L)` is D102 L8's "RFComms as modelled" case, so the 1500 MTU stands.
   Under the physical case (PER in bits, slow fading) the recommendation flips to capping fragment size, and that
   becomes its own decision.
   - After the change, these must be byte-identical: the protocol tests, the 54-run campaign, and Phase 1's
     `bytes.json`.
