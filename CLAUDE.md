# Ariadne — Implementation Repo Guidelines (Claude)

**This is the implementation repo for Ariadne:** collaborative visual SLAM + 3D Gaussian
reconstruction + risk-weighted egress routing for autonomous agents in dark, GPS-denied,
comms-constrained collapsed structures. Competition entry: **OpenCV AI Competition 2026
(powered by AWS)** — build phase ends **Oct 26, 2026 11:45 pm PDT; internal submit Oct 24**.
Targets: Overall prize + Best Use of COOL + Agentic Vision (both Special boxes get ticked).

> **Provenance chain:** proposal (PDF) → algorithm research (`docs/ALGORITHMS.md`) →
> architecture (`docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md`, v2-integrated) → phased plan
> (`docs/ARIADNE_PHASED_PLAN.md`) → **code (here)**. The architecture doc is the source of
> truth; the phased plan is the execution order. When code and doc disagree, the doc wins
> unless you surface the conflict and get a decision. Never silently substitute an algorithm —
> §49 of the arch doc defines the update ritual (catalogue entry → arch decision → ablation
> matrix → benchmark config → manifest schema).

---

## 0. Teaching protocol (brief, applies to every code-producing turn)

The maintainer is an experienced systems engineer; assume full CS/systems background and
**teach the domain, not the language.** Concretely:
1. Before code: one short paragraph — what this piece is, why it exists, data in → data out.
2. Teach each estimation/geometry/graphics concept the first time it appears (preintegration,
   Schur complement, chordal relaxation, GNC, λ₂/Fiedler vector, TSDF/ESDF, Gaussian
   splatting, gz/ROS bridge mechanics) — first-principles, short, concrete.
3. No unexplained magic: non-obvious lines, config parameters, and CLI flags get a one-line
   what+why. State alternatives briefly when a choice had them.
4. After code: how to run it, what correct output looks like, how to tell it broke.
5. Walk large files in reviewable chunks; don't dump. Done = the maintainer can explain every
   file and function unaided.

## 1. Locked decisions (do NOT re-litigate; arch §0.2 records why)

| Area | Decision |
|---|---|
| Sensor rig | **Stereo RGB (1280×720) + IMU + co-mounted controllable spot lamp.** Vision-only, no LiDAR. Monocular is an *ablation*, never "monocular-first". |
| Sim stack | **ROS 2 Jazzy + Gazebo Harmonic**, Ubuntu 24.04 (official pairing; gz-sim8/sdformat14/gz-transport13). No PX4, no physics locomotion — kinematic movers. |
| Perception/geometry | **OpenCV 5 + contrib built from a pinned git tag** (not pip). XFeat primary features; ALIKED+LightGlue hard pairs only; EigenPlaces 256-D retrieval via cv::dnn (ORT fallback allowed, log which ran). |
| SLAM | From scratch on OpenCV 5 primitives: keyframe stereo-VIO (Forster preintegration; `attitude_only` fallback), Schur-LM windowed BA, drop-plus-odometry-edge window exit (**no FEJ marginalization**), chordal-init pose graph, degeneracy gate → PCM → GNC-TLS, SE-Sync-style certificate. GTSAM/Ceres are **test oracles only**, never runtime. |
| Edges/wire | Inter-agent edges are **SE(3)** (Sim3 only in the monocular ablation + Umeyama eval). Serializer = **FlatBuffers**, one boundary, bytes counted there. |
| Collaboration | Swarm-SLAM pattern: two-stage exchange, spectral λ₂ prioritization, vertex-cover sharing, **elected broker with epochs** — no distributed PGO in production. |
| Reconstruction | **gsplat** service (Python/GPU), keyframe-anchored submaps, **albedo-proxy training images** (falloff-normalized), 3×4 affine exposure after ~5k-step delay, nearest-chunk overlap pruning baseline. |
| Determinism | **Record/replay is the substrate:** Gazebo renders clean MCAP bags once; all degradation is a seeded consumption-time operator; all baselines/ablations consume identical pixels. Live runs only for closed-loop + demo. |
| AWS | Graviton4 m8g.4xlarge (COOL) for CPU pipeline + sweeps; g5/g6 for Gazebo (EGL headless) + gsplat; SQS+ASG sweeps; orchestrator LLM via **Bedrock**; Terraform, no Kubernetes. |

## 2. Invariants — true things that must not regress

1. **Competition hard gates:** substantive OpenCV 5 image/video analysis + a meaningful AWS
   component. Every prize depends on both.
2. **COOL benchmark is Arm-vs-Arm:** same binary, same m8g instance, stock OpenCV 5 vs
   `/opt/cool` via loader switch only. The benchmarked workload is the **always-on per-frame
   pipeline** (rectify/pyramids/LK/VST/serialization) — restoration is a labeled extra, so the
   COOL claim survives the restoration ablation either way.
3. **Agentic claim rests exclusively on the runtime loop** (vision output → decision → tool
   call/action/approval, traced in JSONL). The rules state AI-assisted *development* does not
   qualify — never mention the dev workflow near the agentic evidence.
4. **Egress claims are computed at ρ_responder (0.40 m default), never ρ_agent.** An
   agent-passable void is not a human route.
5. **Datasets before simulator** (EuRoC/TUM → OIVIO → sim). A from-scratch estimator and a
   from-scratch sim can be wrong together; the dataset regression suite stays green forever.
6. **Bytes are counted at the serializer** (payload + compression + framing), tagged by class.
   The spectral policy optimizes `gain/predicted_cost`, so the §15.7 byte table has a hard
   ≤15% prediction-error gate (Phase 3).
7. **Honesty rules:** never claim "provably optimal" without a passing certificate; never
   present a projection as measured; negative ablation results get reported, not buried.
   Judges may reject entries that misrepresent capabilities.
8. Verified-live facts (2026-08-30) are tagged in the arch doc: `[VERIFIED web]` /
   `[PAPER]` / `[DESIGN]` / `[SMOKE-TEST]`. `[PAPER]` equations are sketches — transcribe from
   the paper and land the numerical test in the same commit. `[SMOKE-TEST]` items gate
   dependent code and their results go in `docs/SMOKE_TEST_RESULTS.md` first.

## 3. Repo structure (target; only `docs/` exists today)

Arch §3 is normative. Summary: `cpp/` (core → perception/slam/collaboration/reconstruction/
planning → runtime), `python/` (reconstruction service, orchestration/MCP, worldgen, analysis,
console), `ros2/`, `simulator/`, `evaluation/`, `infra/` (docker, terraform), `tests/`
(unit/numerical/integration/protocol/regression/e2e), `configs/`, `docs/`.

**Dependency rule (enforced):** core ← domain libs ← runtime ← ROS2/sim/AWS adapters. No gz or
ROS message type in the SLAM core; no AWS SDK type in any algorithm library. Replay must work
from files alone, ROS-free.

## 4. Execution model

Follow `docs/ARIADNE_PHASED_PLAN.md`: Phase 0 plumbing/determinism → Phase 1 **thin Ariadne
end-to-end with placeholders behind real interfaces** → Phases 2–7 swap placeholders for
full-spec components in vertical slices → Phase 8 missions/failure campaign/submission.
Standing rules: a phase opens only on the previous gate; placeholders are demoted to ablation
baselines, never deleted; the Phase-1 thin run is a **local pre-merge gate** (D49, no nightly CI):
`infra/scripts/phase1.sh --premerge` before handing over commits for any pipeline-touching change
and before checking off a plan item, the full run (fresh GPU render) at phase gates and before
submission; breaking it is stop-the-line; evidence artifacts (decision trace, byte tables, COOL pack, DVE ablations,
failure cases) are produced by the phase that builds the capability, never retrofitted.
Items marked `∥A/∥B/∥C` in a phase are safe for parallel agent sessions; unmarked items are
critical-path in order.

## 5. Conventions

- **Manifests or it didn't happen:** every run records the §28.2 manifest (git commit, image
  digests, opencv build hash, scene/seed, configs, feature/backend/planner selections). Every
  figure directory satisfies §36 (config, seed, command, raw+processed metrics, plot script);
  no hand-edited plots.
- Version pinning: OpenCV 5 tag, gz/ROS apt versions, gsplat/torch, Eigen, gcc — in
  `infra/docker/` and mirrored in manifests. Container images referenced by digest.
- One SE(3) perturbation convention globally (left-multiplicative); every analytic Jacobian
  ships with a finite-difference test, including near-zero depth and wide-FOV rays.
- Honesty tags in code/docs: `[UNVERIFIED]`, `[ASSUMPTION]`, `[PROXY]`. No guessed URLs or
  API names — verify against the installed build or search.
- Git: branch per phase item; small, reviewable commits (pairs with the teaching protocol).
  **Claude may NOT perform git write operations** — no `git add`/`commit`/`push` or any other
  state-changing git command; Claude may draft a commit message, the maintainer runs the
  commands. **Claude must never add itself as co-author** (no `Co-Authored-By: Claude`
  trailer), no exceptions — revisit at end of project, not before.
- Discussion-first: after a significant step, surface what is still unknown and ask **one**
  focused question; if exploration exceeds ~30 min without convergence, check in.
- Verify before claiming done: run it, observe behavior, report actual output. A skipped step
  or red test is stated plainly.
- **Context Mode** (the `context-mode` MCP plugin) is the default for anything that would dump
  large raw output into context — web/codebase search results, log tails, GitHub-issue lists,
  file listings/counts, Playwright-style snapshots. Prefer its sandboxed `ctx_execute()` path
  (write a small script that filters/counts/greps and prints only the result) over reading raw
  data into context directly — this matters most here for large `[SMOKE-TEST]` outputs
  (`cv::getBuildInformation()`, header dumps, build logs), MCAP/log inspection, and evaluation
  artifact scans (`evaluation/runs/**`, `docs/SMOKE_TEST_RESULTS.md` companions). It only
  routes *where data goes*, never how the final answer reads — brevity/formatting stays
  governed by this file and normal judgment. One-time setup (if not already installed):
  `/plugin marketplace add mksglu/context-mode` then `/plugin install context-mode@context-mode`
  (requires Claude Code v1.0.33+; restart or `/reload-plugins` after).

## 6. Source-of-truth corpus (link, don't duplicate)

- Architecture (primary): `docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md`
- Execution order: `docs/ARIADNE_PHASED_PLAN.md`
- Algorithm research + licenses + benchmark evidence: `docs/ALGORITHMS.md`
- Competition rules/rubrics capture: `docs/opencv-ai-competition-2026.md`
- Positioning/risk audit: `docs/ARIADNE-feasibility-audit.md`
- Day-1 gate results (create in Phase 0): `docs/SMOKE_TEST_RESULTS.md`
- Local decisions: `docs/DECISIONS.md`, status: `docs/STATUS.md` (create in Phase 0; dated
  entries; mirror anything that changes the architecture back into the arch doc per §49)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and
cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when
  `graphify-out/graph.json` exists. Use `graphify path "<A>" "<B>"` for relationships and
  `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually
  much smaller than GRAPH_REPORT.md or raw grep output.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source
  browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when
  query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API
  cost).
