# Demo video brief (≤ 5 min, Manim): grounded in the 2026-10-24 state

Owner: **demo-video-session**. Coordinator: M (master session).

The maintainer's outline (the creative direction, quoted in full in M's message to you) asks for a 3Blue1Brown-style
technical film: Manim animations of geometry, algorithms, evolving graphs and plots, told as a mission story. This
brief roots that outline in what the stack actually does today and points to the data behind each beat.

## Rules of the road

- **Presentation may be bold; facts may not.** The maintainer allows a slightly oversold *presentation*: cinematic
  framing, confident narration, idealised schematic animations, best-case runs chosen as the hero shot. Every
  **number** and every claim that *Ariadne does X* must be true of the stack.
  - The competition rules say judges may reject entries that misrepresent capabilities. Our honesty invariant says
    a projection is never presented as measured.
  - If a capability is designed but not built or measured, either leave it out, or show it labelled "design" or
    "next".
- **Label sources on screen:** a small tag on each number, either `measured (Ariadne)`, `published (cite)` or
  `design target`.
- **The live demo footage is a recorded lockstep run played back at real speed.** Label it "recorded run" once. The
  pipeline runs at about 0.84× real time on a c7i CPU, so never call it real time.
- **Determinism:** one line at most, e.g. "every run replays byte-for-byte".
- **The video must show, per the competition (`docs/opencv-ai-competition-2026.md`):** the team, the application
  working, its architecture, and its principal results. Make an architecture beat explicit, with OpenCV 5 and AWS
  visible (both are hard gates).
- **Agentic Vision award rule:** the agentic claim rests only on the runtime loop (vision output → decision →
  tool call, action or approval). Never mention AI-assisted development anywhere near it.
- **References:** look at Manim examples in the community repo and its gallery, 3b1b's own `videos` repo, and
  robotics/SLAM explainer videos made with Manim. Reuse patterns, not assets with unclear licences.

## Beat map: outline → what we have → data

| # | outline beat | state | use this |
|---|---|---|---|
| 1 | Seeded WorldGen maze, GPS-denied, degraded radio, 3 agents at distinct entrances, a target inside | **built** | `python/dataset_tools/worldgen` (D212: 3 agents, greedy-maximin starts); scene `build/live/scenes/maze_demo_0/` (B's worktree `/data/code/Ariadne-llm`); demo picks: maze_demo_0 defaults, seed 0, 3 agents, 90 s. Radio: RF link model with wall attenuation and partitions (D102). **Battery is not a modelled consumable** (only a heartbeat field). Show a mission *time* budget and a *byte* budget, which are real; a battery gauge would be illustrative, so label it or drop it. |
| 2 | Lamp-lit stereo capture, restoration, keypoints, LK tracking, XFeat, correspondences, triangulation | **built** (lit world in the demo; dark + lamp exists, paused, D217) | Stereo 1280×720 + IMU. Pipeline: rectification, SGBM/WLS, FAST + pyramidal LK, `cv::dnn` XFeat (built and exact, D50), ALIKED + LightGlue for hard pairs and inter-agent verification (D171). Console feature overlay (D215): 491/371/298 points on 3 agents. Restoration: radiometric stages and VST (2.2), falloff normalisation (D146), active exposure (D147); these are replay results on the dark worlds (`evaluation/runs/ph5/`). |
| 3 | From-scratch VIO/SLAM: trajectories, reprojection, BA residuals, local pose graph | **built, measured** | EuRoC: full system 0.066 m, 10/11 in class, ≥ ORB-SLAM3 on 7/11 (D87; cite ORB-SLAM3's published numbers as `published`). Schur-LM BA, Forster preintegration. Phase 2 gate: `evaluation/runs/phase2/exit_gate/`. |
| 4 | Cross-agent reconciliation: place recognition, candidate inter-agent loops, descriptor exchange, byte budget, spectral (λ₂/Fiedler) edge prioritisation | **built, measured** | EigenPlaces retrieval (sim R@1 0.705, `evaluation/runs/retrieval_sim_d90/`); two-stage exchange; spectral prioritisation D170 (bound valid 60/60). Bytes are counted at the serializer; per-message prices are accurate to ±0.4 % (D180). The match_response prediction gate FAILS (8/24 > 15 %): don't claim byte-table accuracy. |
| 5 | Two-view verification, pairwise consistency (PCM), GNC suppressing an outlier | **built, measured** | Verify chain E → PnP → degeneracy gate, then PCM → GNC (D71–D76). Outlier ramp: breakpoint 90 % (groups of 10) (`evaluation/runs/2.5/outlier_ramp/`). Certificate: certified on 7 benchmark graphs (`evaluation/runs/2.5/certificate_sesync/`). **Hero result, D178:** with a byte budget, aliasing had pulled the merged map 3.78 m off; the known-entry gate brings it to 0.0147 m, 0 true edges rejected (`evaluation/runs/collab_d178/verdict/`). False-closure injection is rejected (D173). |
| 6 | Resource gauges: bandwidth, uncertainty, mission time, task ownership; observations competing; reallocation | **partly built** | Real: bytes per message class (the comms panel shows live byte counts, D222), CBBA task allocation (D115: CBBA/opt 0.994, `evaluation/runs/allocation_4.4_replay/`), NBV information gain. **Caution, 4.3 NBV gate FAILS** against simpler baselines (`evaluation/runs/nbv_4.3_gate/`, `docs/report/phase4.md`). Do NOT claim uncertainty-aware selection beats baselines on coverage. Show the mechanism; the negative goes in results honestly, or is left out. |
| 7 | Human-in-the-loop orchestration: evidence → proposal → explanation → risk-gated approval → human decision → action; conservative on risk breach or lost link | **built; firing shown as a mechanism** | Bedrock Nova 2 Lite chooser with C++ validation, MCP, JSONL decision trace. Live run 5: 36/36 decisions, 0 fallbacks (`evaluation/runs/L1/llm_live_run5_maze_short_90s/trace.jsonl`). Approval hold and timeout fallback were demonstrated with a lowered threshold (B runs 82-B-gate-mech f964fe and gate-mech2 45609e). Present them as "the safety envelope at work (test configuration)", not as an organic event. Classical vs LLM comparison (7.2): B runs 147c59 / b00318, results pending from B. |
| 8 | Ariadne Thread: risk/clearance-weighted graph, search, first valid route + exit, obstruction → incremental replan | **built; replan only in replay** | D* Lite egress at responder width ρ = 0.40 m (invariant: human-passable, not agent-passable); 12/12 routes GT-valid (`evaluation/runs/egress_4.5/`); hazard seeds (`egress_4.5_hazard/`). **No live reroute has occurred** (4.5 open). Animate the replan from the D* Lite replay mechanism (D113: equals a fresh Dijkstra on 180 flip rounds) and label it replay/simulation. Time-to-first-route: ask B whether the demo run logs it. |
| 9 | Climax: multi-agent Gaussian splat from keyframes, submaps aligning, novel viewpoints | **built; GPU training running now** | Live splat POC: 3 agents → 4 submaps, 259,994 Gaussians, 92.9 % of wall points within 10 cm of GT, placement exact (`evaluation/runs/ph6/live_e2e_20261024/`). AWS g4dn training on the 626abe spools: A2, run id coming. Novel-view renders from that run. Training-view PSNR only: say "training views". |
| 10 | Close: operational stakes | narrative | Ground it in what was measured: a shared map that doesn't break under aliasing (D178), human-width validated routes, a traced agentic loop. |

## Hard-gate beats (must appear)

- **OpenCV 5:** a module table on screen (rectify, SGBM, LK, `cv::dnn`, `cv::ALIKED` + `LightGlueMatcher`,
  findEssentialMat/PnP, LevMarq, `cv::Volume` oracle). The HG1 row of `docs/COMPLETION_REGISTER.md` lists them.
- **AWS:** the architecture diagram shows c7i spot pipeline runs, g4dn GPU (render + gsplat), Bedrock (the agentic
  loop), S3 and Terraform. COOL: **do not show** unless M confirms the AISPL Pay-By-Invoice route came through by
  Oct 25 18:00 IST.

## Numbers you must NOT use

- "real time" (0.84×);
- "provably optimal" for any graph not certified (rim is not certified);
- NBV beating baselines;
- byte-table ≤ 15 % accuracy;
- battery figures;
- held-out splat PSNR;
- live reroute;
- Hilti real-data success (it's 0/100: leave it out or show it as a limitation).

## Sources

- **State and claims:** `docs/COMPLETION_REGISTER.md`, `docs/STATUS.md` (newest first), `docs/DECISIONS.md`,
  `docs/POST_DEMO_BACKLOG.md` (what is not done).
- **Report drafts:** `docs/report/phase3_3.6.md`, `phase4.md`, `phase5.md`, `phase6.md`.
- **Architecture:** `docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md` (arch §0.2 decisions; diagrams to redraw).
- **Algorithms and published comparisons:** `docs/ALGORITHMS.md`.
- **Footage:** B's 90 s demo recording (pending), the console (`python/console`), and per-agent camera panes and
  overlays in the run dirs.

## Process

1. **Storyboard first:** send M a ≤ 1-page storyboard (beats, durations, claims with their source) before
   rendering. M checks the claims, not the art.
2. **Manim:** Community Edition, pinned in a requirements file, under `python/video/` (or a path you choose).
   Render locally at low quality for review. A final 1080p60 render may need an AWS CPU slot: ask M.
3. **No new runs** for the video without M. Data comes from the evidence directories above.
4. **Deadline:** internal submit Oct 24 (slipping); hard deadline Oct 27 12:15 IST. Plan for a full cut by
   Oct 26 noon IST.
