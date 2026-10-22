# Sim retrieval evaluation with revisit GT — settles D90's EigenPlaces sim size (A1)

Status: protocol and decision rule fixed **before** any descriptor was computed (2026-10-22). Decision: D90.
Carries the clean part of 3.1's §31.5 row; the dark-scene row waits for Phase 5.1's dark worlds (register).

## 1. Question

D90 (draft) proposed 640×360 as EigenPlaces' default input on the sim rig, the analogue of EuRoC's half size
(D62: 376×240). L5 measured the cost: 312 ms at 640×360 vs 1280 ms at 1280×720 (p50, 4 threads, g4dn). What was
missing is recall on sim views. This note measures it on the one clean recording there is.

## 2. Data

- The Phase 1 evidence recording `evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T133914Z` (ci_tiny, seed 0,
  two agents, 199 s, 20 Hz, clean: lamp off, no degradation), read through `SimSource`: the RGB frames the pipeline
  sees, with the D41 lens.
- GT: the recorded `gt/pose` stream × the rig's body←camera extrinsic, nearest pose within 10 ms (§11.5's rule).
- One sample per second (`--every 20`), as on EuRoC.

## 3. Protocol (fixed now)

- **Motion thinning, from GT only.** A sample is kept only if the camera moved ≥ 0.25 m or turned ≥ 15° since the last
  kept sample of that agent, the way keyframes enter the real database. Without it, an agent standing still makes
  "revisits" of the identical image 30 s later: measured, every one of agent 1's 115 raw positives is such a repeat.
  Kept: agent 0 127 samples, agent 1 44.
- **Positive:** within 1 m and 30° (PERCEPTION_2.1 §11.5).
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

## 4. Commands

```bash
ariadne_place_eval --rec REC/rec --scene REC/scene --agent {0,1} --sim-d90 \
  --perception-config configs/perception/euroc.yaml --export .cache/models/eigenplaces_sim_ref --out OUT/agent{0,1}
python3 -I python/analysis/retrieval_rows_eval.py sim OUT --base ep640 --methods ep1280 --out OUT/rows
```

(`--perception-config` only names the ORB config the tool loads; no ORB method runs in `--sim-d90`.)

## 5. Result (2026-10-22, `evaluation/runs/retrieval_sim_d90/`)

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
