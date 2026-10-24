# Demo video script v2 (≤ 5 min, Manim CE)

Owner: demo-video-session, 2026-10-24. Brief: `DEMO_VIDEO_BRIEF.md`. Replaces storyboard v1 (2484d00).

**How to read this.** Each beat has three layers:
- **Narration:** the voice, which is TTS for now and a voiceover later. It is also the subtitle text. It should be plain and warm, for someone new to robotics, and carry no numbers it doesn't need.
- **On screen:** what the animation shows. Algorithms are named and run step by step as highlighted pseudo-code. Numbers appear only in charts and result tables, each with a small tag: *measured*, *published* or *illustrative*.
- **Ledger:** the claims and their sources, for M's check. Not shown on screen.

The narration budget is about 720 words, roughly 4:50 at TTS pace, which leaves room for music-only moments.

---

### 1 · The scene (≈ 30 s)
**Narration.** Imagine a building after an earthquake. The lights are out, and somewhere inside, someone may need
help. Before we send people in, we send three small robots. None of them has a map. GPS can't reach them in here,
and their radios barely make it through the walls. Each one will only ever see a small piece of the building. So the
question is: can they put those pieces together, and find a safe way in, and out?

**On screen.** The maze draws itself from its seed. Three robots wait at three doors, and a soft pulse marks the possible
casualty. Radio shading stops at the walls (*illustrative*). The only text is a small corner label, "procedurally
generated · seed 0".

**Ledger.** Scene maze_demo_0 (B). Radio wall attenuation D102. No battery.

### 2 · Title (≈ 6 s)
**Narration.** This is Ariadne, named for the thread that once led a hero out of a labyrinth.

**On screen.** "Ariadne". A gold thread runs through the maze and fades.

### 3 · Seeing (≈ 28 s)
**Narration.** Let's ride along with one of them. Each robot looks through two cameras, like a pair of eyes, and
carries its own lamp for the dark. In every frame it picks out distinctive points, like corners, edges and scuffs on
the wall, and follows them as it moves. Because it has two eyes, it can judge how far away each point is, and a sparse
3D sketch of the corridor begins to appear.

**On screen.**
- A real camera frame (*recorded run*). Features pop in, then tracks streak between frames.
- Labels: **FAST corners + pyramidal Lucas–Kanade** · **XFeat descriptors** (learned, via `cv::dnn`) · **stereo
  rectification → SGBM → triangulation**.
- One dark-world shot before and after restoration (*replay, dark world*).
- OpenCV 5 logo-free tag: "all OpenCV 5".

**Ledger.** D50 (XFeat exact), D215 overlay, pack frames, ph5 replay.

### 4 · Knowing where you are (≈ 30 s)
**Narration.** Next, the robot has to work out where it is, using only what it sees and what its motion sensor
feels. Each new view adds a little tension: points that should line up don't quite. So, again and again, it nudges
its recent path until everything agrees as well as it possibly can. We built this from scratch, and before trusting
it here, we tested it on real drone flights. On seven of eleven, it matched or beat ORB-SLAM3, one of the field's
best-known systems.

**On screen.**
- Camera frusta along a path, with reprojection "springs" that relax.
- Pseudo-code run-through of **windowed bundle adjustment**: IMU preintegration (Forster) → build the normal
  equations → **Schur complement** eliminates the points → solve the poses → Levenberg–Marquardt step.
- **EuRoC chart:** 11 flights, Ariadne vs ORB-SLAM3. The 7 wins light up. No mean lines.

**Ledger.** `evaluation/runs/2.6/euroc_full_aws/…655144` (measured); ORB-SLAM3 T-RO 2021 Tab. II (published). 7/11
at published precision (V1_01 a tie). The maintainer's choice: show the wins, no average. "7 of 11" stays
visible, so the four losses are implied; no wording suggests we beat ORB-SLAM3 overall (M).

### 5 · Sharing a map, on a budget (≈ 35 s)
**Narration.** But one robot's map is only one piece of the puzzle. To join them, the robots need to notice when
they've seen the same place. Sending pictures over a weak radio is far too expensive, so they talk in two steps:
first a tiny fingerprint of each place, and only when two fingerprints look alike, the details. Even then, every byte
counts. So Ariadne asks which link would hold the team's map together best. A bridge between two loosely joined
pieces is worth far more than one more link where the map is already solid.

**On screen.**
- **EigenPlaces** turns an image into a 256-number fingerprint. Fingerprints fly between robots; a byte counter ticks.
- **Two-stage exchange:** descriptor → match request → match response.
- **Spectral prioritisation:** the team's map as a graph whose "stiffness" is λ₂ (algebraic connectivity). Pseudo-code
  run-through: compute the Fiedler vector → score each candidate edge by gain per byte → greedy pick → 1-swap refine.
  The bridge edge raises λ₂; the redundant edge barely moves it.
- **Table:** selection quality vs the exact optimum, spectral 0.970 vs greedy 0.936. Labelled "selection objective on
  synthetic graphs (D170)".
- **Chip:** "per-message cost model accurate to ±0.4 %" (*measured*, D180).

**Ledger.** R@1 not shown. D170: 60 synthetic instances, selected/optimum mean 0.970 (greedy alone 0.936). It is
never shown as map quality (M approved, labelled). D180: the per-message response price is within ±0.4 %. The I6
byte-table gate is not shown: the response class FAILS, so either omit it or show both rows. Spectral is never
credited with the D178 fix.

### 6 · Trust, but verify (≈ 40 s)
**Narration.** Matching places is risky. Many corridors here look identical, and a robot can be fooled into thinking,
"I've been here before." So every match runs a gauntlet. Does the geometry between the two views really add up? Do
the matches agree with one another? Then the remaining evidence pulls gently, tightening step by step until the
false matches simply let go. And one last guard: Ariadne remembers where each robot came in. A match that would
teleport a robot outside its uncertainty just isn't believed.

**On screen.**
- **Two-view check:** epipolar lines → `findEssentialMat` → PnP → degeneracy gate.
- **PCM (pairwise consistency maximisation):** a consistency graph, with the largest consistent clique kept.
- **GNC-TLS (graduated non-convexity):** pseudo-code run-through with μ annealing; outlier weights fade to 0 while the
  map holds still.
- **Known-entry gate:** the alias drags agent 1's map through walls; the uncertainty ellipse; the link is cut.
  (Rendered: 6b part A.)
- **Results tables:**
  - Merged-map error, no gate vs gate: 3.78 m → 1.5 cm or better (3-seed means); 0 of 560 real matches discarded;
    with no aliasing, nothing got worse (0 of 7).
  - Robustness: PCM → GNC still exact with **90 %** false loops (groups of 10).
  - **Certified globally optimal** pose graphs on 7 public SE-Sync benchmarks: garage, tinyGrid3D, smallGrid3D,
    sphere2500, torus3D, cubicle, grid3D.
  - If collab60e holds: the live before/after, where an agent jumps 15 m without the gate and holds with it.

**Ledger.** `collab_d178/verdict/` (approved wording). `2.5/outlier_ramp/` breakpoint 90 % size10 (70 % with two
growing groups, not shown). Certificate D74: 7 certified (M; the source file is confirmed in M's results table).
rim is NOT certified: if it appears, it is labelled so; the film never says "all". Entry-gate wording: "would put
a robot somewhere its own odometry says it cannot be"; "teleport" is fine as narration flavour (M).

### 7 · Sharing the work (≈ 20 s)
**Narration.** With one shared map, the robots can share the work too. Every unexplored doorway becomes a task, and
the robots bid for them, each weighing what it would learn against how far it would have to travel. So they spread
out instead of crowding into the same corridor.

**On screen.**
- Frontiers glow and become task cards; bids rise; ownership colours the map.
- Pseudo-code run-through of **CBBA (consensus-based bundle algorithm)**: build bundle → share bids → resolve conflicts.
- **Table:** allocation quality vs the exact optimum, CBBA mean 99.4 % (worst 98.3 %) vs greedy 98.5 % (worst 90.0 %).

**Ledger.** `allocation_4.4_replay/`, D115 (replay, linked replans). No claim that exploration beats baselines.

### 8 · A human stays in charge (≈ 28 s)
**Narration.** Some decisions shouldn't be left to a robot alone. Ariadne has an AI assistant that reads what the
robots have seen, proposes the next move, and explains why. Every proposal is checked by plain code before anything
moves. And if a move looks too risky, or the link to the command post drops, the robot waits for a human, or falls
back to the safe choice.

**On screen.**
- One real decision record unfolds: what the cameras found → options → **Amazon Bedrock** (Nova 2 Lite) choice and
  its explanation → C++ validation → command. Tool calls via **MCP**.
- Result chip: 36 of 36 live decisions answered, 0 fallbacks.
- The same card turns red: held for approval / link lost → safe fallback (*test configuration*).

**Ledger.** `L1/llm_live_run5_maze_short_90s/trace.jsonl`; B runs f964fe / 45609e for the red half. Nothing about
AI-assisted development.

### 9 · The Ariadne Thread (≈ 32 s)
**Narration.** Now for the reason we're here: a way out. A robot can squeeze through gaps a person can't, so Ariadne
only plans routes wide enough for a person. It searches outward from the casualty until it reaches an exit, and lays
down the thread. If something collapses across that path, it doesn't start over. It repairs just the part of the plan
that changed.

**On screen.**
- Clearance field (**ESDF**) → robot-width vs person-width (0.40 m) lattice. The one narrow gap turns red, and the
  robot's short route is set against the person's longer one.
- Pseudo-code run-through of **D\* Lite**, with ComputeShortestPath and UpdateVertex highlighted as the wavefront spreads.
  The gold thread appears. Rubble falls, only the changed cells re-light, and a new thread forms (*illustrative, on
  simulator ground truth*).
- **Table:** 12 of 12 routes valid against ground truth (replay); D\* Lite repair = full replan on 180 random
  changes, with 7.6× less search.

**Ledger.** scene_gt.npz (mechanism); `egress_4.5/egress_vs_gt.yaml`; D113 unit test. No live reroute claimed.

### 10 · One shared model (≈ 30 s)
**Narration.** And as they explore, the robots' pieces come together into something you can walk through. Each
robot's views become a cloud of tiny, soft, coloured blobs, called Gaussian splats. Because their maps agree, the
pieces slot into one model, which we can view from angles no robot ever stood at.

**On screen.**
- Submaps fly in, one colour per agent, and lock into place. **3D Gaussian splatting** (gsplat).
- The camera orbits to a novel view.
- Chip: 259,994 Gaussians from 71 keyframes.
- Table: wall accuracy, 92.9 % within 10 cm (agents 1–2).

**Ledger.** `ph6/live_e2e_20261024/` (cpu_init). The novel-view orbit needs A2's trained run; held-out PSNR is never
shown.

### 11 · Under the hood (≈ 18 s)
**Narration.** Under the hood, the vision is built on OpenCV 5, and the heavy lifting runs on AWS: simulation, model
training and the AI assistant. And every run can be replayed, byte for byte.

**On screen.**
- **OpenCV 5 module table:** rectify, SGBM, LK, `cv::dnn` (XFeat, EigenPlaces), ALIKED + LightGlue, findEssentialMat /
  PnP, LevMarq.
- **AWS diagram:** EC2 c7i spot (pipeline), g4dn (render; gsplat only if A2's run lands), Bedrock, S3/ECR, Terraform.
  No COOL unless M confirms.

### 12 · Why it matters (≈ 20 s)
**Narration.** In the end, Ariadne isn't about a pretty 3D model. It's about whether a small team of robots, each
seeing only part of the picture, can build one map they can trust, and hand a responder a way in, and a way out.

**On screen.**
- Back to the maze: the thread lit, three robots, one map.
- Team card: Suvojit Das: design, engineering, evaluation (wording to confirm).
