# LIVE_COLLAB: the 3.x collaboration stack inside the live runtime (D174)

**Owner:** M. **Runtime side:** B (ariadne_live, console). **Status:** design, before build.

## 0. Why

The live judge endpoint (Phase 8, D213) runs the agents in the live loop (`ariadne_live`, B), but collaboration
runs only offline: pass A/B over the recording (3.2, D171/D173). The live runtime has no comms node, no broker and no
inter-agent loops, and every agent's world alignment is the briefing's constant (`linked = true`). The demo needs
these live, because the judges drive it:
- the comms panel (bytes against budget, keyframes exchanged, the merge moment);
- the "cut comms" switch;
- the "inject false closure" button.

Maintainer decision 2026-10-23:
- **For Oct 24 (B):** a concurrent pass A/B over the growing recording, labelled with its measured lag.
- **For the final submission (M):** real live collaboration, specified here.

## 1. One implementation (the governing rule)

Live and replay must run the same code. Otherwise the replay evidence says nothing about what the judges see. Two
refactors make the 3.2 tools thin drivers of shared classes:

| Class | Extracted from | Live use | Replay use |
|---|---|---|---|
| `runtime/agent/collab_frontend.hpp` `CollabFrontend` | `ariadne_collab_passa`'s `on_keyframe`: per-keyframe global descriptor (EigenPlaces), local features (ALIKED) with points3d from the full-res masked SGBM depth, and release at t_final through the `LoopBackend` mirror, with the final odometry edge and its information | in each **estimator worker** (D168), on the worker's `on_keyframe` | pass A (writes the store) |
| `runtime/agent/collab_fleet.hpp` `CollabFleet` | `ariadne_collab_sim`'s node set-up: hooks (retrieve, similarity, features, verify, solve, select), CommsNode, robust solve, injection, evidence (estimates, kept edges) | in the **parent**, one fleet, ticked on sim time | pass B |

Acceptance for the refactor:
- On the 3.2 smoke inputs (`build/smoke`), pass A's `store_agent<K>.bin` is byte-identical before and after.
- Pass B's results.yaml (excluding timing fields), ledger.csv, edges.csv and outcome.yaml are byte-identical before
  and after.

Why the frontend runs in the worker and not on the parent's `#kf` record:
- `#kf` carries half-res depth, and pass A's points3d use full-res depth.
- The odometry edge's information lives in the worker's VO graph and is not exported.
- Recomputing either from `#kf` would be a second implementation.

## 2. Live data flow

```
worker k (estimator process, D168)                      parent (ariadne_live)
  VO -> on_keyframe -> CollabFrontend                     reader thread k
     place descriptor (side thread)                         #ckf -> CollabFleet::Release(k, record at t_final)
     release at t_final  ---------------- #ckf -------->
     keeps image + full-res depth per released kf          CollabFleet (one, parent)
  <------------------- #freq kf -------------------------     features hook: request -> pending
     ALIKED + points3d (side thread) ---- #fresp ------>      pending -> FeatureResponse when it arrives
                                                            Tick(sim t): CommsNode, CollabNodes, broker solve
                                                            T_G_W(k) -> T_W_O[k] (B's mutex-guarded refactor)
                                                            telemetry -> console panel (B)
```

- **#ckf** is a `CollabStoreRecord` without local features: id, t_create, t_final, T_W_C, the odometry edge and its
  information, the global descriptor, camera, T_B_C, and GT (evaluation).
- **Lazy local features.** CollabNode asks for features only for budgeted, selected candidates (3.4), so ALIKED
  (~2.3 s/kf on the 8-core laptop, cv::dnn CPU) runs per request, not per keyframe. The worker keeps the image and
  full-res depth of released keyframes for `candidate_ttl_s` (120 s, D170). A request older than that is answered
  "gone", and the broker gives up as it does today.
  - Eager and lazy give identical features for the same image, because the model is deterministic. Test: lazy ==
    eager on the smoke keyframes.
- **Deferred features in CollabNode (protocol change).** `NodeHooks::features` may answer "not ready". The node then
  keeps the request and answers it on a later `Tick` when the hook is ready. Today a synchronous hook is assumed: a
  slow answer would block the tick, and a dropped one costs the broker's 10 s `request_retry_s`.
  - Replay hooks are always ready, so replay behaviour and bytes are unchanged (protocol tests plus the §1
    acceptance).

## 3. Clock and determinism

- **The fleet runs on the sim clock in both modes.** Releases are inserted at the record's `t_final` (sim time), never
  at arrival.
- **Lockstep (gate runs):**
  - The barrier at T also waits until every worker has sent all #ckf with `t_final <= T` (a per-lane collab
    watermark, like the image progress).
  - A deferred feature answer goes out at a fixed SIM tick: the first tick at or after the request's tick plus
    `feature_latency_s` (config; a sim-time model of the worker's compute, recorded in the manifest). In lockstep the
    fleet blocks at that tick until the worker's #fresp is in, so wall-clock arrival never decides the order. #ckf is
    likewise inserted at its `t_final`.
  - Result: replays equal the live run, the same acceptance as D152a/D168: three replays equal live.
- **Demo mode (RTF 1, no lockstep):** releases and responses apply when they arrive. The panel shows the measured
  lag: #ckf arrival minus t_final, and #fresp latency. Demo runs are not claimed reproducible (D169 already says so).

## 4. Feedback into the live loop

- `T_G_W(k)` from the fleet's adopted estimate maps agent k's odometry frame into the shared frame G.
  B's refactor replaces the briefing's constant `T_W_O[k]` with a mutex-guarded value:
  - **Initial value:** the known entry pose.
  - **Updated:** to `T_G_W(0)^-1 * T_G_W(k)` after each adopted broker estimate (the same formula as the D173
    composer).
- A1's planner gets `linked[k]` from the fleet's comms state (a partitioned agent is unlinked), not a constant.

**Decided (maintainer, 2026-10-23): known starts.** Think of drones inserted through the building's windows: the
operator places them, so each entry pose is known when it goes in.
- The briefing's start alignment `T_W_O[k]` is a legitimate prior, labelled as the known entry pose. It is not hidden
  ground truth.
- Arrows, the top-view map and the planner are coherent from t = 0.
- Inter-agent loops then correct accumulated drift, rather than merge unknown frames: the panel shows "drift corrected
  by loop agent i <-> agent j". The robust stage still rejects false closures, which the inject button shows.
- The fleet's broker solve therefore gets each agent's entry pose as a prior on its first keyframe.
  - Today: the gauge agent is fixed, and the other agents are free until a loop exists.
  - New: a weak entry-pose prior per agent, its information a config value, recorded in the manifest.
  - Without that prior, an agent with no loops would be unplaced, contradicting the known entry.
  - This changes the fleet's solve input, so it ships behind a config key (`entry_prior`, default off). Replays and
    the 3.2 gate are unchanged.

## 5. Panel controls

- **Cut comms:** a runtime override on CommsNode: a partition of {all agents apart} or {one agent apart}, from the
  button press until it is released. It is logged in link.jsonl as a scripted fault, so the run's record shows it.
  Reconnection follows the existing 3.3 path: heartbeat, then the new epoch, then resync.
- **Inject false closure:** `CollabFleet::InjectFalse()`, the D173 path.
  - Pair: the agent pair with the most kept loops.
  - Keyframes: the latest released keyframe of each.
  - Offset: 4 m and 45° (fixed, so every press is the same experiment).
  - Armed only when that pair has kept true loops. The panel shows kept_solves and the final weight.
- **Telemetry** (JSON lines, 1 Hz plus events), for B's panel:
  - bytes per class, first transmission vs overhead;
  - budget and token level;
  - each selection round: candidates, selected, predicted vs measured bytes;
  - verify outcomes by reason;
  - merges: agent joins frame, with time;
  - kept / rejected inter-agent loops;
  - injection fate;
  - lag (§3).

## 6. Compute budget (sizing for the judge instance)

Per agent, at the lit-maze keyframe rate of ~1.6 KF/s (D142: 78 KF per 1000 frames at 20 Hz):
- EigenPlaces at 640×360 is ~0.5 s/kf on the laptop, i.e. **~0.8 core per agent**.
- ALIKED: per request only, a few per 5 s selection period under the budget.
- This model work is cv::dnn on CPU: the image has no ONNX Runtime, and the pinned OpenCV 5 build has no CUDA.

With gz, 3 estimators and 3 collab frontends, a g4dn.xlarge (4 vCPU) will not hold real time. A g4dn.2xlarge
(8 vCPU) or g4dn.4xlarge (16 vCPU), with the same single T4, is the realistic target. The maintainer is filing the on-demand
G/VT quota at 16 vCPU (one g4dn.4xlarge). Measured, not assumed, in the M2-style acceptance below.

A GPU path for the two models (ONNX Runtime with CUDA; CLAUDE.md §1 allows an ORT fallback if it logs which ran)
would cut this to a few ms per keyframe. It is a separate image change and an [UNVERIFIED] gain, so it is out of scope
unless the measured CPU budget fails.

## 7. Build order and acceptance

1. **Refactor** `CollabFrontend` and `CollabFleet`. Pass A/B stay byte-identical on the smoke inputs (§1). The 3.2
   tests and protocol tests stay green.
   - **Done (as built):** `runtime/agent/collab_frontend.hpp` (`CollabFrontend`: eager for pass A, lazy with
     `keep_s` for live; `Features(id)` computes on request) and `runtime/agent/collab_fleet.hpp` (`CollabFleet` +
     `CollabConfig`/`LoadCollabConfig`, which now rejects duplicate keys). Pass B on the D173 smoke: ledger, edges,
     outcome, ate_vs_bytes byte-identical, results.yaml identical but for `ms_total`. Pass A: store byte-identical to
     the pre-refactor binary at the same OpenCV thread count; `--check-lazy 1`: lazy == eager, 38/38 keyframes bitwise.
   - **Finding (determinism):** `cv::ALIKED` output depends on the OpenCV thread count (`OPENCV_FOR_THREADS_NUM`):
     the same keypoint set (1023-1024 of 1024 shared), in a different order, with descriptors differing up to 0.53.
     Run to run at a fixed thread count it is bitwise stable. So pass-A stores, and live features, are reproducible
     only with the thread count pinned. Pass A's yaml now records `opencv_threads`; the sweep pins it per job
     (nproc / P_A). Live lockstep (§3) must pin it per worker too.
2. **Deferred features** in CollabNode, plus a protocol test (a request answered two ticks later). Protocol tests stay
   green.
   - **Done (as built):** `NodeHooks::features` returns `std::optional`; nullopt keeps a peer's request
     (`AnswerDeferred`, request-id order, dropped after `response_ttl_s`) or the broker's own keyframe
     (`ResolveOwnFeatures`, dropped after `request_give_up_s`). Stats `answers_deferred` / `answers_given_up`.
     protocol 72/72 (DeferredFeatures 3). Pass B on the D173 smoke (`spec_inject`, seed 1): ledger.csv, edges.csv,
     outcome.yaml and ate_vs_bytes.csv byte-identical to 15426f8; results.yaml differs only in `ms_total` timings.
3. **Worker side:** the frontend in the worker behind `collab: true` (default false, so the D168 digests are
   unchanged), plus the #ckf/#freq/#fresp codecs. Test: lazy == eager features.
4. **Parent side:** the fleet in ariadne_live, the lockstep watermark, the T_W_O feedback (B), and telemetry.
5. **Acceptance, lockstep:** a 3-agent lit maze, 60 s, `collab: true`.
   - Three replays equal live.
   - Merged ATE and kept loops reported.
   - With `collab: false`, the D168 digest is unchanged.
6. **Acceptance, demo mode, on the GPU instance (with B's M2):**
   - Holds ≥ 0.95× real time with 3 agents and collab on.
   - Lag reported.
   - The cut-comms and inject buttons exercised once each, with their outcome in the trace.

## 8. §49

- **catalogue:** no new algorithm. This wires 3.1–3.4 and D76 live.
- **arch:** a §5.7 / §15 as-built note on live collaboration at merge.
- **ablation:** `collab: true|false` in the live config.
- **benchmark:** the §7 acceptance runs.
- **manifest:** `collab`, the comms scenario, the budget level, lag statistics, and which model backend ran.
