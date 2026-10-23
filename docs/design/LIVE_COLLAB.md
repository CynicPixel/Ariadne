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

**Amendment (2026-10-24, D174): the DNN work runs in its own process per agent, not in the estimator worker.**
- D152a/D168: OpenCV's parallel_for pool is per process, and a busy pool changes stripe partitioning, hence the order
  of floating-point reductions. EigenPlaces (~0.5 s/kf) and ALIKED (~2.3 s/kf) on a side thread of the estimator
  worker would contend for its pool, and VO would stop being reproducible (the D168 digests). Measured on ALIKED itself
  (§7 step 1 finding): its output depends on the thread count.
- So `CollabFrontend` splits into a cheap **release tracker** (the LoopBackend mirror: needs the VO graph, no DNN) and
  a **feature extractor** (EigenPlaces at t_create, ALIKED lazily). Pass A composes both in one process; live puts the
  tracker in the estimator worker and the extractor in an `ariadne_collab_worker` process per agent (one thread, sim
  order, `OPENCV_FOR_THREADS_NUM` pinned and recorded). Same classes either way.
- Per-agent processes, not one shared: three agents' place descriptors serialized in one process exceed real time.

## 2. Live data flow

```
estimator worker k (D168, no DNN)        ariadne_collab_worker k (DNN, own pool)        parent (ariadne_live)
  VO -> on_keyframe                         #ckin: place descriptor, keep rgb+depth
     #ckin (id, t, full-res rgb, masked       (until keep_s after t_final)
      full-res depth, pinhole, T_B_C) ---->
     release tracker (mirror)
     #crel (id, t_final, T_W_C, odometry   -> assemble record ------ #ckf ------------->  CollabFleet::Release at t_final
      Z + information) ------------------->  then #cwm (watermark t) ---------------->  lockstep watermark (§3)
                                           <------------------------ #freq (kf) -------  features hook: pending
                                            ALIKED + points3d ------ #fresp ---------->  answered on a later Tick
                                                                                         Tick(sim t): CommsNode, nodes
                                                                                         T_G_W(k) -> T_W_O[k] (B)
```

- The collab worker listens on its own Unix socket (LinkFrame, `runtime/live/link_frame.hpp`); the estimator worker
  and the parent each connect to it. Messages to one peer stay in order, so a keyframe's #ckin is always processed
  before the #crel that releases it.
- **#ckf** is a `CollabStoreRecord` without local features: id, t_create, t_final, T_W_C, the odometry edge and its
  information, the global descriptor, camera, T_B_C, GT (evaluation), plus the keypoint count to advertise (unknown
  before extraction: the model's K, 1024, is advertised; the byte predictor prices fetches with it).
- **#cwm t**: every release with t_final <= t has been sent. The parent's barrier waits on it like the image progress.
- **Lazy local features** as before: ALIKED runs per request; the collab worker keeps the image and full-res depth of
  released keyframes for `candidate_ttl_s` (120 s, D170); older requests are answered "gone" (#fresp with no
  features), and the broker gives up as it does today.
  - Eager and lazy give identical features for the same image at the same thread count (§7 step 1: 38/38 bitwise).
- **Deferred features in CollabNode (protocol change)**: done (§7 step 2).

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
- It runs in the collab workers (§1 amendment), one process per agent: 3 estimator workers + 3 collab workers + gz +
  the parent.

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
     the same keypoints in a different order, with float-level value differences (4 threads vs the default, 38 keyframes:
     every keypoint matched within 1.8e-4 px, descriptors within 4.9e-6 once rows are aligned by keypoint).
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
   - **Steps 3-4 as built (2026-10-24):** `CollabReleaseTracker` + `CollabFeatureExtractor` (split of
     `CollabFrontend`), `runtime/live/collab_link.hpp` (messages), `collab_worker_core.hpp` + `ariadne_collab_worker`,
     `collab_tee.hpp` (estimator side), `collab_live.hpp` (parent). Pass A `--check-live 1` on the smoke runs the live
     split in-process through the encoders: every #ckf record and #fresp reply bitwise equal to the store, 38/38; the
     store byte-identical to step 1's at the same thread count; lazy == eager 38/38. CollabLink round-trip tests.
     Call sites in ariadne_live (spawn, tee calls, barrier tick, T_W_O feedback, panel): B, after the D175 POC.
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

## 9. Asynchronous collaboration (D179, 2026-10-24) - replaces §3's lockstep

**Live (never waits).** One fleet thread owns CollabFleet and its radio nodes. The runtime calls, all non-blocking:
- `Step(t)`: enqueue sim tick t. `Send(k, fn, t)`: enqueue a D175 send (a closure on node k) stamped t.
- `Drain(k, T)`: take agent k's coordination deliveries the fleet thread has produced so far (ticks <= T, per-sender
  FIFO, ticks in order).
- `Snapshot(T)`: the latest published {T_O0_O, Linked, Telemetry, version}, one atomically swapped value.
The fleet thread processes queued items in order. At each tick t it commits everything that arrived from the collab
workers since the previous tick (#ckf records released at t, #fresp replies made available at t), then delivers the
network up to t, ticks every node, and publishes a new snapshot version. Feature hooks never block: a reply not yet
committed answers "not ready" (CollabNode's deferred path).

**Log (`collab_events.jsonl`).** Every nondeterministic choice is recorded with the sim tick it was taken at:
`commit` (t, agent, kind ckf|fresp, keyframe), `drain` (T, agent, the delivered message keys in order) and `snapshot`
(T, version read). Payloads are NOT logged: features are recomputed by the collab workers, which are deterministic at
a pinned OpenCV thread count (§7 step 1).

**Replay (deterministic).** The same classes run synchronously from the log. A tick commits exactly the logged
arrivals (waiting for the workers if needed, which costs replay wall time only). `Drain(k, T)` returns exactly the
logged keys, keeping the rest for later drains. `Snapshot(T)` returns the logged version (the fleet keeps its published
versions in replay). So replays equal the live run they log - B's invariant 1 (drain order) and invariant 2 (one
consistent snapshot) hold both ways.

**What it costs.** Live decisions now depend on wall-clock arrival (as on real robots); determinism is a property of
the logged run, not of re-running live. Acceptance: a live 60 s 3-agent run with collab on, then 3 replays from its log
give identical fleet decisions (estimates digest, kept edges, ledger) and identical runtime digests.
