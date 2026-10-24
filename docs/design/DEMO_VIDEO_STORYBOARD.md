# Demo video storyboard v1 (≤ 5 min, Manim CE): for M's claim check

Owner: demo-video-session, 2026-10-24. Brief: `DEMO_VIDEO_BRIEF.md` (0031e15). Total **288 s** (12 s margin).
Tags shown on screen: **M** = measured (Ariadne), **P** = published (cite), **D** = design, **I** = illustrative
(schematic, no number). Footage from B's 90 s maze_demo_0 seed-0 pack is labelled "recorded run" once. Everything
from other sources carries its own label ("replay", "sweep", "EuRoC", "test configuration").

| # | t (s) | Visual (real data unless marked I) | On-screen claim | Tag | Source |
|---|---|---|---|---|---|
| 1 | 0–12 | Seeded WorldGen draws maze_demo_0 wall by wall; 3 agents at 3 entrances; target marker; radio rings shadowed by walls | "No GPS. Radio through walls. A fixed mission clock and byte budget." | D/I | scene `maze_demo_0/` (B), RF model D102. **No battery gauge.** |
| 2 | 12–19 | Title + the question (one line) | none | none | none |
| 3 | 19–47 | Lamp cone (I) → real stereo frame → feature overlay → LK tracks → XFeat match lines → stereo triangulation into points (I geometry, real points) | "XFeat on OpenCV 5 `cv::dnn`; LK tracking; SGBM stereo." 1 shot: dark-world restoration, labelled "replay, dark world" | M (built) | pack frames, D50, D215; `evaluation/runs/ph5/` |
| 4 | 47–75 | Camera frusta along a trajectory; reprojection rays and residual springs (I); BA "snaps" them; per-flight EuRoC bar chart | "EuRoC, 11 flights: mean ATE **0.066 m**; ORB-SLAM3 **0.035 m** (P). At or better on **7/11**; in class 10/11." V1_03 bar kept (0.288) | M + P | D87; ORB-SLAM3 T-RO 2021 Tab. II |
| 5 | 75–110 | Three local maps float apart. Retrieval: query → top-k strip. Candidate funnel animated from real edges: 491 candidates → essential / degeneracy / PnP gates → verified. Byte counter ticks per message class. λ₂ / Fiedler: one bridge edge between weak clusters lifts λ₂ more than a redundant edge (I) | "Place recognition R@1 **0.705** (sim)"; "every byte counted at the serializer"; spectral shown as mechanism only, **no comparative claim** | M / I | `retrieval_sim_d90/`; collab_edges (pack) **pending B's GT check**; D170 |
| 6 | 110–145 | Two-view check (epipolar lines) → PCM consistency graph → clique; GNC: one outlier spring's weight anneals to 0 while the map stays put (I). Payoff chart: merged-map error under the byte budget, without vs with gate | False-loop rejection: "Under the byte budget, aliased loops pulled the merged map **3.78 m** off. Known-entry gate + GNC: **0.0147 m**; **0 of 560** tested true edges rejected." Subtitle: "replay sweep, known entry poses." Breakpoint: robust to **90 %** outliers (groups of 10). Credit goes to the gate, **never to spectral** | M | `collab_d178/verdict/` (on main, 26864da); `2.5/outlier_ramp/` |
| 7 | 145–165 | Gauges: bytes per class, mission clock, task owner per agent (CBBA picks from trace); frontier candidates as bids | "Task allocation: CBBA within **0.994** of optimal (replay)." No coverage-beats-baseline claim | M | `allocation_4.4_replay/`; pack trace |
| 8 | 165–193 | Decision-trace card animates: vision summary → candidates → LLM pick + explanation → C++ validation → approval gate → command. Then the same card in red: risk above envelope → held for approval; link lost → classical fallback | "Live on Bedrock (Nova 2 Lite): **36/36** decisions answered, **0** fallbacks." Red half: "safety envelope, test configuration" | M | `L1/llm_live_run5_maze_short_90s/trace.jsonl`; B runs f964fe, 45609e |
| 9 | 193–223 | Occupancy → clearance field → ρ = 0.40 m responder lattice; D* Lite expands; first route + exit drawn as a gold thread; obstruction drops in → only nearby vertices update → new thread | "Routes at **human** width (0.40 m): **12/12** GT-valid." "Incremental = fresh Dijkstra, 180 random rounds, **7.6×** fewer expansions." Replan labelled "replay" | M | `egress_4.5/`, D113; replan from the replay mechanism, **not live** |
| 10 | 223–253 | Submaps (one colour per agent) fly into the shared frame; splat fades in; camera orbits to an unseen viewpoint | "71 keyframes → 4 submaps, **259,994** Gaussians; wall points **92.9 %** within 10 cm (agents 1–2; agent 0 lower)." Swap in A2's g4dn trained run if it lands: training-view metrics only | M | `ph6/live_e2e_20261024/` (cpu_init), A2 run id TBD |
| 11 | 253–273 | Architecture: OpenCV 5 module table; AWS diagram (c7i spot pipeline, g4dn render, Bedrock, S3/ECR, Terraform); "every run replays byte-for-byte" (one line) | Modules and services only where evidenced. gsplat on AWS drawn only if A2's run lands. **No COOL** unless M confirms by Oct 25 18:00 IST | M | HG1/HG2 register rows |
| 12 | 273–288 | Return to the maze: thread lit, three agents, the shared map. Team card. One honest line: "Open: exploration that beats simpler baselines" | The stakes, no new numbers | none | `docs/report/phase4.md` (4.3 FAIL) |

**Rendering notes.** All geometry, paths, edges, bytes, trace cards and routes are re-drawn in Manim from the pack's
JSON/CSV. Camera frames and splat views are imported images or clips (Manim can't rasterise splats). Visual grammar
follows 3b1b `videos` and the Manim CE gallery: a dark field, a 3-agent palette held throughout, no logos.

**Excluded per brief:** real time, battery, byte-table accuracy, NBV win, live reroute, held-out PSNR, Hilti,
"provably optimal" (the certificate isn't shown on a graph it didn't certify).
