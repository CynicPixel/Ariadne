# Demo video script v3 (≤ 5 min, Manim CE)

Owner: demo-video-session, 2026-10-24. Brief: `DEMO_VIDEO_BRIEF.md`. **Numbers: only from
`docs/report/RESULTS_INDEX.md`** (cited "RI §n"). Replaces v2 (f808c97).

**Three layers per beat:**
- **Narration:** the voice and the subtitles. Plain and warm, for a first-time viewer, with few numbers.
- **On screen:** named algorithms, pseudo-code run-throughs, real data, and results tables where we do well. Each number
  carries a tag: *measured*, *published* or *illustrative*.
- **Ledger:** claims and sources for M. Not shown.

**Rules-required content:** the team (12), the application working (the recorded run in 7 and 12), the architecture
(11, also exported as a standalone SVG/PNG), and the principal results (tables throughout).

**Priority order (M):** animated algorithms with a metric → the console top-view panel (recorded run) → real camera
frames → the splat render → one real agentic decision.

---

### 1 · The scene (≈ 30 s)
**Narration.** Imagine a building after an earthquake. The lights are out, and somewhere inside, someone may need
help. Before we send people in, we send three small robots. None of them has a map. GPS can't reach them in here,
and their radios barely make it through the walls. Each one will only ever see a small piece of the building. So the
question is: can they put those pieces together, and find a safe way in, and out?

**On screen.** The maze draws itself from its seed. Three robots wait at three doors, and a soft pulse marks the casualty.
Radio shading stops at the walls (*illustrative*). Corner label: "procedurally generated · seed 0". (Rendered: s01.)

### 2 · Title (≈ 6 s)
**Narration.** This is Ariadne, named for the thread that once led a hero out of a labyrinth.

### 3 · Seeing in the dark (≈ 30 s)
**Narration.** Let's ride along with one of them. Each robot looks through two cameras, like a pair of eyes, and
carries its own lamp. In every frame it picks out distinctive points, like corners and scuffs on the wall, and
follows them as it moves. With two eyes, it can judge how far away each point is. And if a sudden flash blinds it,
it turns its own exposure down, and is seeing clearly again in a couple of seconds.

**On screen.**
- A real camera frame with its feature overlay (pack frames).
- Labels: **FAST corners + ORB**, tracked by **pyramidal Lucas–Kanade**; **stereo rectification → SGBM → triangulation**
  into a sparse 3D sketch.
- **Active exposure:** a +2-stop flash whites out the frame; the exposure control loop recovers.
- **Chart:** saturation cleared in 2.35 s; ATE 16.1 mm vs 19.1 mm with fixed exposure.

**Ledger.** RI §2 row 1 (replay, sim ci_tiny_dark, one run per row: label "replay"). No falloff-normalisation claim.
**XFeat is NOT named as the front end** (RI do-not-claim 1).

### 4 · Knowing where you are (≈ 30 s)
**Narration.** Next, the robot has to work out where it is, using only what it sees and what its motion sensor feels.
Each new view adds a little tension: points that should line up don't quite. So, again and again, it nudges its
recent path until everything agrees as well as it possibly can. We built this from scratch, and before trusting it
here, we tested it on real drone flights. On seven of eleven, it matched or beat ORB-SLAM3, one of the field's
best-known systems.

**On screen.**
- Camera frusta and reprojection "springs" that relax.
- Pseudo-code run-through of **windowed bundle adjustment**: IMU preintegration (Forster) → normal equations →
  **Schur complement** eliminates the points → solve the poses → **Levenberg–Marquardt** step.
- **EuRoC chart:** 11 flights; the 7 wins light up; no mean line.
- **Chip:** "solver matches GTSAM on 100 of 100 synthetic graphs".

**Ledger.** RI §3 row 1 (D87; published ORB-SLAM3 Table II). "7 of 11" stays visible, with no parity wording. GTSAM
100/100 from RI §3 (D52, synthetic).

### 5 · Sharing a map, on a budget (≈ 35 s)
**Narration.** But one robot's map is only one piece of the puzzle. To join them, the robots need to notice when
they've seen the same place. Sending pictures over a weak radio is far too expensive, so they talk in two steps:
first a tiny fingerprint of each place, and only when two fingerprints look alike, the details. Even then, every byte
counts. So Ariadne asks which link would hold the team's map together best. A bridge between two loosely joined
pieces is worth far more than one more link where the map is already solid.

**On screen.**
- **EigenPlaces** turns an image into a 256-number fingerprint.
  - **Table:** recall@1 on EuRoC, EigenPlaces 0.626 vs mean-bit 0.444 vs random 0.167.
- **Two-stage exchange:** fingerprint → match request → match response, with a byte counter ticking.
  - **Chip:** "per-message cost model accurate to ±0.4 %".
- **Spectral prioritisation:** the map as a graph whose stiffness is λ₂ (algebraic connectivity).
  - Pseudo-code run-through: Fiedler vector → score candidate edges by gain per byte → greedy pick → 1-swap.
  - The bridge raises λ₂; the redundant edge barely moves it.
  - **Table:** spectral 0.970 vs greedy 0.936 of the exact optimum, labelled "selection objective, synthetic graphs".

**Ledger.**
- RI §4 row 1 (EuRoC only; no sim baseline).
- RI §6: ±0.4 % per message; the I6 table FAIL is not shown.
- D170 is unit/synthetic and never shown as map quality.
- Fingerprint byte size is not shown: the wire carries fp16, and int8 is only a row.

### 6 · Trust, but verify (≈ 40 s)
**Narration.** Matching places is risky. Many corridors here look identical, and a robot can be fooled into thinking,
"I've been here before." So every match runs a gauntlet. Does the geometry between the two views really add up? Do
the matches agree with one another? Then the remaining evidence pulls gently, tightening step by step until the
false matches let go. And one last guard: Ariadne remembers where each robot came in. A match that would teleport a
robot somewhere its own sense of motion says it can't be just isn't believed.

**On screen.**
1. **Two-view registration:** **ALIKED + LightGlue** matches → `findEssentialMat` → PnP → degeneracy gate.
   - **Table:** pose accuracy (AUC@10°), ALIKED + LightGlue 0.607 vs ORB 0.284, better on 11 of 11 flights.
2. **PCM:** a consistency graph, with the largest consistent clique kept.
3. **GNC-TLS:** pseudo-code with μ annealing; outlier weights fade to 0.
   - **Design-choice table:** PCM → GNC holds up to 90 % false loops, and is ~10× faster than GNC alone.
4. **Known-entry gate** (rendered: s06b part A), then the **live before/after, same maze, same aliasing**:
   - gate off: agent 1 dragged **15.3 m** and never recovers;
   - gate on: **8 of 8** false matches refused; every robot within **7.5 cm** of ground truth.
   - Both labelled *recorded run*, side by side from the two packs.
5. **Sweep table:**
   - merged-map error 3.78 m → 1.5 cm or better (3-seed means);
   - 0 of 560 real matches discarded;
   - with no aliasing, nothing got worse (0 of 7).
6. **Certificate table:** "certified optimal for SE-Sync's cost" on 7 public benchmark graphs (garage, tinyGrid3D,
   smallGrid3D, sphere2500, torus3D, cubicle, grid3D). Our objectives match the paper's Table 1.

**Ledger.**
- RI §1 row 1: 4 confident wrong verifications exist, so E inliers alone are not a gate; this is not shown as a
  zero-error claim.
- RI §5: certificate scope is "SE-Sync's isotropic chordal cost". rim is never shown as certified. The words
  "provably optimal" never appear.
- RI §6 D178 and the collab60e gate_report (B): known starts only. collab60e kept 0 inter-agent edges, so no
  cross-agent closure is claimed there.

### 7 · Sharing the work (≈ 25 s)
**Narration.** With one shared map, the robots can share the work too. Every unexplored doorway becomes a task, and
the robots bid for them, each weighing what it would learn against how far it would have to travel. So they spread
out instead of crowding into the same corridor.

**On screen.**
- **Recorded run:** the console's top-view panel, cropped, sped up and labelled "recorded run": robots fan out over the map.
- Frontiers become task cards, bids rise, and ownership colours the map.
- Pseudo-code run-through of **CBBA (consensus-based bundle algorithm)**: build bundle → share bids → resolve conflicts.
- **Table:** quality vs the exact optimum. CBBA mean 0.994, worst 0.983; greedy 0.985 / 0.900; CBBA's published
  guarantee ≥ 0.5.

**Ledger.** RI §7 (replay, one spool, 2 agents, 12 linked replans). No exploration-vs-baseline claim (RI §9 is weak).

### 8 · A human stays in charge (≈ 28 s)
**Narration.** Some decisions shouldn't be left to a robot alone. Ariadne has an AI assistant that reads what the
robots have seen, proposes the next move, and explains why. Every proposal is checked by plain code before anything
moves. And when a move would be too risky, or the link to the command post drops, the robot is built to wait for a
human, or fall back to the safe choice.

**On screen.**
- One real decision record unfolds: vision summary → options → **Amazon Bedrock (Nova 2 Lite)** choice and its
  explanation → C++ validation → command. Tools via **MCP**.
- **Chip:** 36 of 36 live decisions answered, 0 fallbacks.
- The safety envelope (red half): rendered from the real trace lines **only if B confirms them**, labelled "test
  configuration: threshold lowered to force the safety path". Otherwise it's an *illustrative* mechanism, with nothing
  implying it fired.

**Ledger.**
- RI §11 (run 5).
- M ruling: f964fe's approval hold fired genuinely; its fallback fired because of an IAM deny, not a timeout.
  45609e's timeout fallback awaits B's trace records.
- The narration's "built to wait / fall back" is OK as a capability statement, but never implies it fired in the demo
  mission.

### 9 · The Ariadne Thread (≈ 32 s)
**Narration.** Now for the reason we're here: a way out. A robot can squeeze through gaps a person can't, so Ariadne
only plans routes wide enough for a person. It searches outward from the casualty until it reaches an exit, and lays
down the thread. If something collapses across that path, it doesn't start over. It repairs just the part of the plan
that changed.

**On screen.**
- Clearance field (**ESDF**) → person-width lattice (0.40 m); the one narrow gap turns red; the robot's short route
  is set against the person's longer one.
- Pseudo-code run-through of **D\* Lite**: ComputeShortestPath and UpdateVertex light up as the wavefront spreads; the
  gold thread is drawn; rubble falls; only the changed cells re-light; a new thread forms. Labelled *illustrative, on
  simulator ground truth*. (Rendered: s09, being fixed.)
- **Table:**
  - 12 of 12 routes valid for a 0.40 m responder; the final route equals the ground-truth shortest path (ratio
    0.9999997);
  - responder traversability precision 1.000;
  - occupied-map precision 0.865 vs 0.707 for the placeholder;
  - D\* Lite repair = full replan on 180 random changes, 7.6× fewer expansions (*unit test*).

**Ledger.** RI §8 (ci_tiny spool, 2 agents, label "replay"; 0 reroutes happened). No FAR/TARE claim. Risk 0.340 →
0 not used (design doc only).

### 10 · One shared model (≈ 30 s)
**Narration.** And as they explore, the robots' pieces come together into something you can walk through. Each
robot's views become a cloud of tiny, soft, coloured blobs, called Gaussian splats. Because their maps agree, the
pieces slot into one model, which we can view from angles no robot ever stood at.

**On screen.**
- Submaps fly in, one colour per agent, and lock into place. **3D Gaussian splatting** (gsplat).
- An orbit to a novel view.
- **Chip:** 259,994 Gaussians, 0 dropped keyframes.
- **Table:** 92.9 % of wall seed points (agents 1–2) within 10 cm of ground truth.

**Ledger.** RI §10. A2's AWS GPU render replaces the cpu_init view when it lands. No PSNR or quality metric is shown.

### 11 · Under the hood (≈ 20 s)
**Narration.** Under the hood, the vision is built on OpenCV 5, and the heavy lifting runs on AWS: simulation,
experiments and the AI assistant. A complete collaboration experiment, across two buildings and three trials, costs
about nine cents. And every run can be replayed, byte for byte.

**On screen.**
- **Architecture diagram** (also exported standalone, SVG/PNG):
  - OpenCV 5 modules: rectify, SGBM, LK, ORB/FAST, `cv::dnn` (EigenPlaces), ALIKED + LightGlue, findEssentialMat /
    PnP, LevMarq;
  - agent components;
  - AWS: EC2 c7i spot (pipeline, sweeps), g4dn (render; gsplat once A2's run lands), Bedrock, S3/ECR, Terraform.
- **Cost table:**
  - collaboration sweep $0.0875;
  - 11-flight EuRoC $0.0439;
  - pre-merge gate ~$0.035.
- No COOL; Arm 0.64× is not shown.

**Ledger.** RI §12, §13. There is no real-time claim anywhere.

### 12 · Why it matters (≈ 20 s)
**Narration.** In the end, Ariadne isn't about a pretty 3D model. It's about whether a small team of robots, each
seeing only part of the picture, can build one map they can trust, and hand a responder a way in, and a way out.

**On screen.**
- The recorded top view, then the maze with the thread lit.
- Team card: Suvojit Das: design, engineering, evaluation (wording to confirm).
