# ARIADNE — Phased Build Plan

**Model:** thin-Ariadne-first. Phase 0 lays plumbing, Phase 1 stands up the entire system end to end with
placeholder algorithms, and every later phase swaps placeholders for the full-spec components in vertical slices.
At the end of every phase the whole system still runs; nothing is ever "integrated later."

**Section references (§) point to `ARIADNE_IMPLEMENTATION_ARCHITECTURE.md` (v2-integrated).**
**Iteration references (I#) map to its §33 program; this plan is the execution regrouping of those iterations.**

**Calendar:** 57 days to the platform deadline (Oct 26, 11:45 pm PDT); internal submission target Oct 24.
Durations below are guidance, not commitments; the hard rule is the exit gate, not the date.

**Tracking convention:** every numbered item gets a checkbox in this file's working copy; a phase is closed by a
tagged commit `phase-N-exit` whose CI run demonstrates the gate.

**Parallelization:** items marked `∥A/∥B/∥C` are independent workstreams safe to run as concurrent agents within
a phase; unmarked items are on the phase's critical path in listed order.

---

## Phase 0 — Plumbing and determinism substrate (I0 + smoke tests) · ~4 days

- [ ] **0.1** Monorepo per §3: CMake superbuild, `pyproject.toml`, config schema loader, `LICENSES/`, `docs/` stubs
  including `SMOKE_TEST_RESULTS.md`.
- [ ] **0.2 Day-1 smoke tests (gates everything):** build OpenCV 5 + contrib from the pinned tag on amd64 and arm64;
  record in `SMOKE_TEST_RESULTS.md`: actual module/header layout (§0.2 naming discrepancy), presence of
  `features` classes (XFeat/ALIKED/DISK/LightGlueMatcher/ANNIndex), HashTSDF API shape (§20.1), `ximgproc`
  DisparityWLSFilter, `quality`; PyPI wheel status; DNN engine runs each ONNX model or falls back to the ORT path
  (§7.4). `∥A`
- [ ] **0.3** COOL day-1 check on the free trial: launch AMI, run one stock pipeline unmodified, first microbenchmark,
  confirm result-publication terms. `∥B`
- [ ] **0.4** `core/`: SE(3)/SO(3) math with one perturbation convention (§11.3), timestamp types, covariance
  transport, FlatBuffers schemas + single serialization boundary with ByteRecord accounting (§4.4, §15.7). `∥C`
- [ ] **0.5** Logging/event schema (§41), run-manifest writer (§28.2), deterministic seed plumbing.
- [ ] **0.6** CI per §27.2 (amd64 + arm64 jobs, ccache, opencv-base layer, ECR push by digest); Terraform modules
  applied for network/storage/registry (§27.3). `∥B`
- [ ] **0.7** MCAP record/replay skeleton (§5.9) and the seeded degradation operator `D(image, params, seed,
  frame_id)` with the sensor-response chain of §5.6 — built now, against synthetic images, before any simulator
  exists.

**Phase 0 testing / exit gate**
- Unit + numerical: SE(3) exp/log/adjoint round-trips, covariance transport, serialization round-trip (§37.1).
- Determinism: a synthetic trajectory is generated, serialized, replayed, and scored bit-identically twice; the
  degradation operator is bit-identical for fixed (seed, frame_id) across machines.
- CI green on both architectures; both container images boot and print `cv::getBuildInformation()` hashes matching
  the manifest.

---

## Phase 1 — Thin Ariadne, end to end (I1) · ~6 days

Everything below is deliberately the simplest correct version; the point is the wiring, contracts, and metrics.

- [ ] **1.1** Gazebo Harmonic world: one hand-authored two-room + corridor scene, normal lighting; agent model SDF with
  stereo pair + IMU + (inert) spot lamp per §5.3; ros_gz bridge YAML, `/clock`, `use_sim_time` (§5.4).
- [ ] **1.2** Kinematic mover with `goto/follow/hold` and collision envelope (§5.8); scripted trajectories for 2 agents.
- [ ] **1.3** Record the scene to MCAP with clean images + GT streams (§5.9); all downstream work in this phase runs
  from the bag as well as live.
- [ ] **1.4** Placeholder perception: grayscale, ORB, LK tracking, `StereoSGBM` with default-ish params. `∥A`
- [ ] **1.5** Placeholder SLAM: pose chain from PnP odometry, naive triangulation, a two-frame BA stub behind the real
  `KeyframeBundle`/pose-graph interfaces (§4.2, §12). `∥A`
- [ ] **1.6** Placeholder collaboration: broadcast every global descriptor (raw ORB-BoW-ish or mean-pooled patch
  vector), verify one inter-agent loop with `findEssentialMat`, exchange pose edges through the real serializer so
  bytes are counted from day one; trivial broker = agent 0 (§15). `∥B`
- [ ] **1.7** Placeholder reconstruction: fused stereo point cloud, gsplat initialization from it, one offline render;
  the Python reconstruction service boundary (§18.2) exists even though the content is naive. `∥C`
- [ ] **1.8** Placeholder mission layer: occupancy from raw depth into a plain voxel grid, nearest-frontier policy,
  A* route to exit at ρ_agent (§20–§22 interfaces, minimal internals). `∥B`
- [ ] **1.9** Evaluation harness v1: ATE/RPE vs GT (Umeyama alignment, `evo` as oracle), bytes by class, route
  validity vs GT; one auto-generated report page per run (§30, §36). `∥C`
- [ ] **1.10** Console stub: WebSocket telemetry, 2D top-down map, byte gauges (§29).

**Phase 1 testing / exit gate**
- The I1 exit condition verbatim: two agents explore, build local maps, merge via at least one verified
  inter-agent closure, render a splat, and produce an exit route — in one deterministic replay run.
- Every §4 data contract exercised end to end; every metric in §30.2 emits a number (however bad).
- Protocol tests: duplicate/out-of-order message injection is idempotent (§15.3, §37.7).

---

## Phase 2 — Single-agent estimator to full spec (I2 + I3 + I3.5 + I4) · ~10 days

- [ ] **2.1** Perception swap-in (I2): XFeat keyframes + LK tracking per §7.2/§8, spatial-bin keypoint selection
  (§7.3), DNN execution contract with per-thread Nets (§7.4), stereo calibration + SGBM/WLS to §9.3 spec with the
  confined-space config validator (§9.4), depth uncertainty contract (§9.6), ANNIndex retrieval, ALIKED+LightGlue
  hard path. `∥A`
- [ ] **2.2** Radiometric stages 1–2 (flat-field, exposure normalization) and VST denoise with GAT + exact unbiased
  inverse, gated on measured noise (§6.2, §6.4) — falloff normalization waits for Phase 5's lamp calibration. `∥A`
- [ ] **2.3** Real local SLAM (I3): keyframe policy (§8.3), landmark lifecycle (§11.2), analytic-Jacobian Schur LM
  windowed BA (§11.3–11.6), window-exit policy (§11.7), covariance recovery, local pose graph. 
- [ ] **2.4** Inertial layer (I3.5, §10): preintegration with bias states, stationary initialization, gravity-prior
  edges, motion-predicted LK seeding; `attitude_only` fallback wired and selectable.
- [ ] **2.5** Robust single-agent backend (I4): chordal init with determinant fix (§13.1), degeneracy detector
  (§14.4), PCM (§14.5), GNC-TLS with exposed weight trajectories (§14.6), certificate (§14.8) behind a flag. `∥B`
- [ ] **2.6** Dataset program starts (§34.1): EuRoC (stereo+IMU), TUM RGB-D; regression suite pinned in CI.

**Phase 2 testing / exit gate**
- Jacobian finite-difference suite green including near-zero depth and wide-FOV rays (§11.4, §37.1).
- Oracle agreement: Ariadne PGO vs GTSAM/Ceres within tolerance on 100 random synthetic graphs (§11.6, §37.5);
  certificate returns `certified` on known-optimal graphs and `not certified` on planted-outlier graphs.
- Outlier ramp (10→90% false constraints): plot success probability/ATE/runtime; breakpoint reported (§37.3).
- EuRoC: stereo-VIO ATE within the published error class of reference stereo-inertial systems; TUM: stereo/RGB
  pipeline ATE regression pinned. Fallback decision point: if 2.4 exceeds its budget, ship `attitude_only`,
  record it in the manifest, and proceed — do not stall the phase.

---

## Phase 3 — Multi-agent collaboration to full spec (I5 + I6 + I7) · ~8 days

- [ ] **3.1** EigenPlaces ONNX retrieval + descriptor pipeline, retrieval-path CLAHE split (§6.3, §17.1). `∥A`
- [ ] **3.2** Full protocol (I5): two-stage exchange, bookkeeping/high-water marks, store-and-forward with priority
  classes and TTLs (§15.3), heartbeats, anchor propagation and frame-tagged poses (§15.5), broker election with
  epochs (§15.4).
- [ ] **3.3** Comms broker node with RFComms-derived link state + wall-penetration attenuation + scenario partitions;
  all payloads and faults flow through it (§5.7, §15.6). `∥B`
- [ ] **3.4** Spectral prioritization (I6): candidate Laplacian, warm-started Fiedler vector, `gain/predicted_cost`
  selection with vertex-cover sharing, exact enumeration oracle ≤15 candidates (§16).
- [ ] **3.5** Aliasing laboratory (I7): repetition-operator worlds from worldgen (§5.5), descriptor-space aliasing
  diagnostics (§17.3), degeneracy-gate-before-PCM ordering enforced and tested.
- [ ] **3.6** Descriptor compression experiment rows (fp16/int8/PCA) wired into the harness (§17.2, §31.9). `∥A`

**Phase 3 testing / exit gate**
- Network fault injection: loss/latency/partition/reorder/disconnect; every run ends `consistent` or
  `explicitly incomplete`, never silently corrupted (I5 gate); split-brain test: two partitions each elect a
  broker, reconnect, merge under a new epoch with no pose-graph corruption.
- **Byte-table hard gate (I6):** predicted vs measured wire bytes per class; >15% error fails the phase (§15.7).
- Core plot exists: ATE vs cumulative bytes, spectral vs greedy vs similarity, on ≥2 scenes × 3 seeds; spectral
  dominates or the deviation is investigated before phase close.
- Aliasing campaign: injected false closures rejected at ≥ target precision; accepted-false-closure rate and map
  deformation reported (I7 metrics).

---

## Phase 4 — Mission intelligence: volumetric, NBV, allocation, egress (I8 + I12) · ~7 days

- [ ] **4.1** Volumetric stack (§20): HashTSDF integration (or the from-scratch fallback recorded in
  `SMOKE_TEST_RESULTS.md`), tri-state occupancy, incremental Felzenszwalb ESDF, frontier clustering, coarse global
  layer + RLE submap summaries.
- [ ] **4.2** Traversability/risk at ρ_agent and ρ_responder; risk proxy calibrated against GT-labeled zones (§20.5). `∥A`
- [ ] **4.3** NBV (I8): candidate generators A–E, raycast IG terms with self-normalization, λ scheduling with the hard
  energy-reserve rule (§21.3–21.4).
- [ ] **4.4** Allocation: event-triggered CBBA bundle 1–2 + broker-greedy shortcut, both logged identically (§21.7). `∥B`
- [ ] **4.5** Egress (I12): ρ_responder lattice, D* Lite incremental replanning, 5th-percentile route confidence,
  invalidate-and-reroute path (§22).
- [ ] **4.6** TARE-style coarse/fine scheduling and FAR-style route graph for agent motion (§21.5–21.6). `∥B`

**Phase 4 testing / exit gate**
- ESDF vs brute-force distance oracle on random grids; frontier detection vs GT unknown boundary (§37.6).
- Planner ablation plot v1: reconstruction-coverage vs mission budget for random / coverage-greedy /
  uncertainty-only / Ariadne (I8 gate), ≥2 scenes × 3 seeds.
- Egress: obstacle-discovered-after-route scenario reroutes (I12); time-to-first-valid-route and route validity
  computed at ρ_responder only; route-confidence calibration plot emits.
- Allocation: no duplicate task ownership under partition + reconnect; auction-vs-broker-greedy row measured.

---

## Phase 5 — Degraded visual environments (I9 + I10) · ~7 days

- [ ] **5.1** Dark worlds: near-zero ambient, spot lamp active with shadows, lamp calibration procedure (P, cosine
  lobe decision) executed and stored (§5.3, §19.2).
- [ ] **5.2** Falloff normalization goes live (§6.2 Stage 3) producing the albedo-proxy image; geometry-vs-retrieval
  preprocessing split enforced (§6.3).
- [ ] **5.3** Active exposure + lamp co-control (§6.5) actuating sim exposure params and `light_config`; overexposure
  failure-and-response demo scenario scripted. 
- [ ] **5.4** Degradation configs `dark0..3 × dust0..3` named and frozen; dust compositing uses recorded GT depth
  (§5.6). `∥A`
- [ ] **5.5** Confined-space worlds: narrow voids, corridor degeneracy scenarios; baseline sweep 4/8/12/20 cm (§9.1,
  §31.8). `∥B`
- [ ] **5.6** Real-data validation: OIVIO co-moving-light sequences through the full estimator; report honestly where
  sim-tuned settings transfer and where they do not (§34.2).

**Phase 5 testing / exit gate**
- DVE ablation matrix rows (§31.5) measured: each radiometric stage's effect on repeatability, match precision,
  ATE, loop-closure precision — negative results kept.
- Live-vs-replay exposure-control comparison quantified (§5.6 stated approximation).
- OIVIO: tracking survives lamp-lit sequences with reported ATE; failure segments documented as failure cases.
- Corridor degeneracy detector fires on the degenerate scenes and suppresses loop attempts (gate ordering test).

---

## Phase 6 — Dense reconstruction to full spec (I11) · ~5 days

- [ ] **6.1** Albedo-proxy training path from normalized keyframes; held-out evaluation renders with the light model
  applied at the held-out lamp pose (§19.1).
- [ ] **6.2** Submap lifecycle: keyframe-anchored frames, PGO re-anchoring, stable/unstable Gaussian lifecycle,
  geometry-first rule (§18.3–18.6).
- [ ] **6.3** Merge: nearest-chunk overlap pruning baseline; per-keyframe 3×4 affine exposure enabled only after the
  ~5k-step delay; NFL-aware residual term; specular downweighting (§18.7, §19.3–19.4). `∥A`
- [ ] **6.4** Console upgrade to full spec: splat streaming in-browser, raw-vs-restored feeds, uncertainty overlay,
  the thread, negotiation timeline, approval modal (§29). `∥B`

**Phase 6 testing / exit gate**
- Held-out PSNR/SSIM/LPIPS + Chamfer vs GT mesh per scene; multi-agent merged scene has no double-wall artifacts
  on the overlap test scene; exposure-delay ablation (on-time vs early) demonstrates the geometry-mush failure.
- PGO update mid-training re-anchors submaps with zero retraining (assert Gaussians unchanged in local frame).
- Console live against a replayed mission; approval round-trip works.

---

## Phase 7 — Agentic layer + COOL + AWS scale-out (I13 + I14) · ~6 days

- [ ] **7.1** MCP tool surface over the deterministic core; Bedrock-hosted orchestrator choosing among
  planner-presented candidates only; out-of-set fallback logged; decision-trace JSONL artifact (§25).
- [ ] **7.2** Classical-planner ablation configured as "skip the choose step" (§25.3). `∥A`
- [ ] **7.3** COOL benchmark (I14): stock-vs-COOL on the same m8g.4xlarge, per-op and per-pipeline p50/p95,
  utilization, cost/frame, concurrent-agent-pipeline knee; OpenCV4-vs-5 Arm context measurement; workload defined
  per §0.2/C4 (always-on pipeline, restoration as labeled extra). `∥B`
- [ ] **7.4** Sweep infrastructure live: SQS + worker ASG consuming run manifests, artifacts to S3, budget alarms
  (§27.3, §28).

**Phase 7 testing / exit gate**
- A complete agentic trace: OpenCV output → decision → tool call → action → outcome, replayable offline; one
  human-approval escalation end to end (Agentic Vision evidence set complete).
- LLM-vs-classical planner comparison measured on ≥2 scenes × 3 seeds; result reported whichever way it lands.
- COOL benchmark pack reproducible from a clean instance via one script; numbers in the manifest format.
- A 20-run sweep launches, completes, and lands metrics in S3 untouched by hand.

---

## Phase 8 — Full-system missions, failure campaign, evidence, submission (I15 + I16) · ~9 days + buffer

- [ ] **8.1** Full missions (I15): 3–5 agents, dark + dust + attenuation + injected false closures + agent loss +
  active perception + reconstruction + egress; complete run artifact per §33/I15.
- [ ] **8.2** Failure campaign (I16): scenarios A–G executed with expected-behavior assertions; each becomes a report
  failure-case subsection.
- [ ] **8.3** Headline evaluation sweeps: quality-vs-bytes per policy, budget-to-threshold reporting, full ablation
  matrix rows prioritized by report need (§31, §35).
- [ ] **8.4** Report: architecture, OpenCV 5 module table, evaluation with failure cases, limitations,
  responsible-use; three diagrams (system, agent workflow, hybrid deployment); §44 claims discipline. `∥A`
- [ ] **8.5** Video (§43 shot list) from one coherent recorded mission; console hosted live for judging. `∥B`
- [ ] **8.6** Submission mechanics: repo access, pinned deps + build/test instructions for both architectures, both
  Special Award boxes ticked, Devpost form complete by **Oct 24**.

**Phase 8 testing / exit gate**
- The §42 "done" list, all 17 items, in one deterministic replayable run.
- Every figure directory satisfies §36 (config, seed, command, raw + processed metrics, plot script).
- A clean-machine reviewer rebuild: clone → build → replay one mission → reproduce one headline figure, following
  only the written instructions.

---

## Standing rules across phases

1. A phase never opens while the previous phase's gate is red; a placeholder is only replaced, never deleted —
   old versions remain as ablation baselines.
2. The thin end-to-end run from Phase 1 is executed in CI nightly for the rest of the project; it breaking is a
   stop-the-line event (this is R2/integration-debt insurance).
3. Evidence artifacts (decision trace, byte tables, COOL pack, DVE ablations, failure cases) are produced by the
   phase that builds the capability, never reconstructed in Phase 8 (R3 insurance).
4. Dataset regressions (Phase 2.6) stay green through every later phase; the simulator never becomes the only
   validator (R1 insurance).
5. Stretch items (thermal witness, event cameras, GLOW-full, feed-forward comparator) are admissible only after
   Phase 8.1 is green, as isolated branches.
