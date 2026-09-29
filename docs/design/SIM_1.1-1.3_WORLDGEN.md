# Design — Phase 1.1–1.3: procedural room-maze worlds, kinematic mover, MCAP recording

Status: **DRAFT for maintainer approval** (2026-09-30). Implements arch §5.3–§5.5, §5.8, §5.9 and plan items
1.1–1.3. Where this deviates from the arch text it says so and lists the deviation in §8 for a DECISIONS entry.

Evidence behind it: `SMOKE_TEST_RESULTS.md` §4f, §4g (measured in this image), and the research list in §10.

---

## 1. What the worlds must do (requirements derived, not assumed)

| Consumer | Requirement on the world | Where it comes from |
|---|---|---|
| ORB/FAST + LK (1.4) | Corners at every scale on every surface, visible at 0.5–6 m; no large uniform areas | §7.3, D34 |
| SGBM (1.4) | Horizontal texture variation in every row, non-periodic (periodic = disparity aliasing) | §9.3, D34 |
| Stereo range | Most surfaces between Z_min 0.25 m and ThDepth 6 m (60·B) | §9.4, D36 |
| Intra-agent loop closure | Each agent revisits some place with a similar heading | §12, §13 |
| Inter-agent closure + consensus (1.6, PCM/GNC in Ph.3) | **Several** same-heading, co-visible overlaps between agents, in **≥2 distinct rooms** (PCM needs mutually consistent loops; closures all in one room are correlated evidence) | §14, Swarm-SLAM |
| Place recognition hygiene | Distinct places look distinct (Phase 1); aliasing is a *dial* for the I7 lab (3.5), not an accident | §5.5 repetition operators |
| Egress (Invariant 4) | Routes defined at ρ_responder = 0.40 m; at least one agent-only passage so ρ_agent and ρ_responder routes differ | §20.5, §22 |
| Record/replay (C6) | World, trajectories and IMU are pure functions of (seed, params, version) | §5.9 |

Key research consequence: ORB/BoW place recognition fails on opposite-direction revisits, so "the two agents
walked the same corridor" is **not** an overlap unless headings agree. Overlap is therefore *measured* (§4.3),
not hoped for.

---

## 2. Floorplan generator — `python/dataset_tools/worldgen` (arch §5.5 stage 1)

A **macro-cell maze with embedded rooms**, combining three established constructive methods:

1. **Lattice.** `nx × ny` macro-cells of size `s` (default 4×4, s = 6 m → ~24×24 m site).
2. **Uniform spanning tree** over the lattice with **Wilson's algorithm** (loop-erased random walks; unbiased
   over all spanning trees, the standard "perfect maze" generator).
3. **Rooms.** Choose `n_rooms ∈ {3,4}` cells as rooms; each gets a rectangle of 3.5–5.5 m per side inside its cell.
   Other cells are corridor junctions (T/X/L intersections: the maze part).
4. **Braid** (the cyclic-dungeon idea from *Unexplored*): add `k_cycles ≥ 1` non-tree edges so entrance→target
   has two independent arcs. Cycles give intra-agent loops and let two agents take different arcs to one target.
5. **Cull** dead-end corridor cells that lead nowhere, keeping `n_dead_ends` short stubs (maze decision points,
   and later the dead-end primitive of §5.5).
6. **Passages.** Each lattice edge becomes an axis-aligned corridor of width `w ∈ [1.0, 1.4] m` (≥ 2ρ_responder +
   0.2 m). `n_narrow` (default 1) edges **on a cycle** get `w ∈ [0.55, 0.70] m`: agent-passable (≥ 2·ρ_agent +
   margin) but not responder-passable. Because narrow edges sit on a cycle, a ρ_responder route still exists
   (asserted). A 0.55 m passage leaves ~0.275 m clearance, at the stereo Z_min (0.25 m) and inside D34's
   near-range SGBM alias zone, with v_max ≈ 0.26 m/s. Their purpose is ρ_agent-vs-ρ_responder route evaluation,
   so **scripted Phase 1 routes do not traverse narrow edges** (the planner runs at ρ_responder for scripts);
   traversing them is a Phase 5.5 confined-space scenario.
7. **Entrances.** Two exterior openings on boundary cells, maximally separated (one per agent).
8. **Target** room chosen so the **GT geodesic length at ρ_responder from the primary entrance (agent 0's)**
   lies in the requested band `[L_lo, L_hi]`; the secondary entrance's length is recorded, not constrained.
   Candidate presets short / medium / long; the bands themselves are **set from a seed sweep** (§9 step 2),
   not guessed, and `max_attempts` makes an unsatisfiable band fail loudly.

**Rejection sampling, deterministic.** Any failed constraint (length band, ρ_responder connectivity, overlap
thresholds of §4.3) increments `attempt`; all randomness for attempt `a` is drawn from the stream keyed
`(seed, a, site)`. The accepted `attempt` is recorded, so the world is recoverable from the seed alone and the
manifest also states it explicitly.

**Seeded randomness in Python (the D18 lesson).** NumPy guarantees bit-generator *streams* but not the outputs
of `Generator.integers/choice/random` across versions. Worldgen therefore draws only `random_raw()` words from
`numpy.random.Philox` keyed as above and owns every mapping (integers via rejection on raw words, floats from the
top 53 bits). Texture noise uses **hash-based** lattice noise (a pure function of integer coordinates), not RNG
streams.

**Scene identity** = `(generator_version, seed, params_hash, attempt)`; `params` is a named preset file
`configs/worldgen/<preset>.yaml` (full vector, including the length band), so "two numbers reproduce a world"
(§5.5) holds as `(seed, preset)`.

---

## 3. Geometry, textures, lights

**Geometry.** All dimensions snapped to 5 cm. The free space is a union of axis-aligned rectangles; it is
rasterized at 5 cm, and walls are the **maximal boundary runs** between free and occupied cells, each extruded
into a 0.2 m-thick wall outside the free space (corners closed by extension). Walls, floor tiles and ceiling
tiles (one per free rectangle) are emitted as **OBJ meshes with explicit UVs** at a fixed texel density
(256 px/m ≈ 3.9 mm/texel), because gz `<box>` stretches textures (§4g). **Trade-off, stated:** at the 1 m
corridor's ~0.5 m clearance a pixel covers ~0.8 mm, so texture is magnified 3–6×; matching texel to pixel there
would need ~1300 px/m, i.e. ~25× the texture memory, which llvmpipe cannot afford for a whole site. Fine
corner-bearing detail therefore comes from geometry-scale layers (§ below), and the texture gate is evaluated
on the close-range frames where magnification bites. Wall height 2.6–3.0 m. Props (crates,
pillars, cabinets) sit against walls inside rooms, keeping a central free disk for surveys; props enter the GT
occupancy.

**Textures** — generated per surface from `(seed, surface_id)`, so no two walls are alike in Phase 1:
- base: multi-octave (1/f) hash value noise (concrete mottling);
- **corner-bearing layers** the §4g render showed are needed: cellular/Worley "aggregate" pits, multi-scale
  speckle, hard-edged stains, crack polylines, occasional stencil marks/bolt rows. These put high-contrast
  corners at 2–40 px scales, which FAST/LK and SGBM need;
- one **richness** knob `τ ∈ [0,1]` scales the corner layers' contrast/density (Phase 5 dials toward §5.5's
  "low-texture concrete"), and one **repetition** knob `r ∈ [0,1]` (default 0) clones textures/props across
  rooms for the I7 aliasing lab (3.5). Same code path for both; nothing is a second texture system.
- PNGs are written deterministically (no metadata, fixed zlib level); the pixel-data hash, not the file hash, is
  the golden value.

**Texture acceptance, measured not eyeballed:** a small core-image tool renders nothing itself; it runs the 1.4
front-end on recorded frames and reports ORB keypoints per 8×6 grid cell and SGBM valid-and-supported
density, **stratified by GT clearance** (corridor/minimum-clearance frames reported separately from room frames,
so room frames cannot carry the average). Gate for Phase 1 presets at τ = 1, on the lowest-clearance stratum:
≥ 90 % of grid cells hold ≥ 10 keypoints, and SGBM support density ≥ 0.6. Thresholds are revisited after the
first measurement, and any change is recorded.

**Lighting (Phase 1 = "normal").** Ambient 0.3 plus one point light per room/corridor segment, no shadows
(llvmpipe cost). Each agent carries the §5.3 spot lamp, inert (power 0) until Phase 5.

**Calibration target.** Each entrance room has a **ChArUco** board (objdetect `CharucoDetector`), with
**disjoint marker-ID ranges per board**. Identical checkerboards in two rooms would be exactly the kind of
repeated landmark that produces a false inter-agent loop (§1). Calibration-dance frames are tagged in the
manifest so loop-closure evaluation can exclude them. The boards serve the 1.1 acceptance test (§6) and later
lamp calibration (5.1).

---

## 4. Ground truth and scripted trajectories

### 4.1 GT layer (Phase 1 subset of §5.5 stage 5)
Because Phase 1 worlds are extruded 2D plans, a 2D GT is exact:
`scene_gt.npz` = occupancy (5 cm), ESDF (exact Euclidean distance transform, own implementation),
traversability at ρ_agent and ρ_responder, the GT egress route (A* on the 8-connected grid with ESDF ≥ ρ),
semantic zone ids (room/corridor/junction/dead-end), per-surface material ids. `scene.json` holds the manifest
(identity, attempt, graph, lengths, overlap metrics). Voxel/3D GT arrives with the collapse operators (Phase 5).

### 4.2 Motion model and speed limits (derived from the tracker)
Per-frame image motion must stay inside LK's capture range. The placeholder tracker (win 21, 3 levels) is
budgeted at **d_max = 30 px/frame** (well inside 10·2³):
- translation: flow ≈ f·v·Δt/Z at the nearest visible surface ⇒ `v_max = d_max·c / (f·Δt)` with `c` = GT ESDF
  clearance at the pose; with f = 640, Δt = 0.05 s: 0.47 m/s at 0.5 m clearance (1 m corridor), capped at
  0.6 m/s in rooms;
- rotation: pure yaw flow at the image edge ≈ 2·f·ω·Δt ⇒ `ω_max ≈ 0.44 rad/s` (25°/s).
Both are mover-enforced, and recorded in the scene manifest.

Primitives (arch §5.8): `goto(p, R)` = minimum-jerk (quintic) profile `s(t)` shared by translation and SLERP
rotation — C², so the IMU is continuous. The quintic's peak rate is **1.875×** its average, so segment
duration is set from the peak: `T = 1.875·max(L/v_max, θ/ω_max)`. When a segment both translates and rotates,
the pixel budget applies to the **sum** of both flows; `follow(path)` = turn-to-heading + goto segments along the planner
path; `hold(T)`; `survey()` = 360° yaw sweep at ω_max (gives same-heading views of every room regardless of
arrival direction). Altitude varies ±0.2 m between waypoints for IMU excitation (planar yaw-only motion leaves
IMU states unobservable in Phase 2).

### 4.3 Routes and the overlap metric
Route template per agent k: `entrance_k → target (survey) → entrance_k by the other arc`, with room surveys on
the way. This gives each agent an intra-agent loop (at its entrance), both agents the target room, and shared
rooms on the cycle.

**Closure opportunities**, computed on GT at keyframe spacing: pose pairs (i from agent A, j from B) with
distance ≤ 1.5 m, heading difference ≤ 30°, **and GT line of sight** between them over the occupancy grid (so
two poses on opposite sides of a wall do not count). Clustered by semantic zone. Scene gate:
**≥ 3 inter-agent opportunities in ≥ 2 distinct rooms**, and ≥ 1 intra-agent revisit per agent. Failing scenes
are rejected (next attempt), so every accepted scene can exercise PCM.

**Collision envelope (1.2).** The mover asserts `ESDF ≥ ρ_agent` at every 1 cm of every commanded segment before
executing it; a script that violates it fails the run. Worldgen runs the same check offline when it writes the
script.

---

## 5. Simulation: agent SDF, mover plugin, IMU, bridge (1.1, 1.2)

**Agent model** `ariadne/agent_<k>` (§5.3): one link with left/right cameras (1280×720, HFOV 90°, B = 0.10 m,
20 Hz, identical intrinsics), co-located depth camera (GT only, never visible to perception), and the spot lamp.
Cameras are free-running at the same rate: the §4f run showed stereo pairing 1.0 at identical stamps, so
triggered cameras are not needed [the §5.3 SMOKE-TEST fallback]. Mild plumb-bob distortion per §5.3, verified by
§6's recovery test.

**Mover** = C++ gz system plugin `ariadne::sim::KinematicMover` (built in the sim image; gz headers and g++ are
present). No Physics system in the world. Each `PreUpdate` it evaluates the active trajectory at sim time and
writes `components::Pose` (proven in §4g). It accepts commands from a script file (Phase 1) and a gz-transport
topic (for live runs later).

**IMU — synthesized, not gz's** (gz's is blind under kinematic motion, §4g). The mover publishes a **clean**
IMU at 200 Hz from the closed-form derivatives of the trajectory: ω_B = ṡ·θ·u for the SLERP axis u; specific
force f_B = Rᵀ(a_W − g_W). Stamps are on the sim clock. §4g measured render time = pose time on the depth camera
under light load; zero camera–IMU offset is **claimed only after** §6 re-checks it on RGB under full load. **Noise and bias random walk are a seeded consumption-time operator** (like D,
keyed per (seed, agent, sample block)), reading the same `configs/sensors/imu_*.yaml` that the Phase 2
estimator reads. See deviation §8-1.

**GT pose** is published by the mover at 200 Hz from the same trajectory (exact), plus static camera extrinsics.

**Bridge** (§5.4): a `parameter_bridge` YAML with images, camera_info, IMU, GT pose and GT depth per agent under
`/ariadne/agent_<k>/…`, and `/clock`. `use_sim_time:=true` everywhere.

---

## 6. 1.1 acceptance test — calibration recovery, not self-consistency

From a recorded calibration dance in front of the checkerboard (≥ 15 views), the core image runs
`findChessboardCornersSB` + `calibrateCamera` + `stereoCalibrate` and compares the **recovered** K, D, and
left→right extrinsic against the SDF-derived values; then `solvePnP` per frame against GT pose composed with the
explicit, tested gz-link (+X forward) → optical (+Z forward) rotation. The dance includes a
**constant-velocity pass** at two speeds: a render/pose time offset δ shows up as a PnP-vs-GT translation
residual ≈ v·δ that grows with speed. The run is under the full two-agent load, since a render-thread race
only appears at low RTF. Together this catches a gz/OpenCV distortion convention mismatch, wrong camera_info,
wrong extrinsics and timing errors. Thresholds are set when the
first run is measured; I will not pre-commit numbers.

---

## 7. Recording (1.3) and the D25 gap

`ros2 bag record -s mcap --storage-preset-profile zstd_fast`, one bag per `(scene, trajectory set, seed)`.
Measured (§4g): 74.6 MB per sim-second for two agents with full-rate depth; RTF 0.217 on llvmpipe.

**Sizing proposal:** GT depth at **640×360, 20 Hz** (a quarter of the pixels, every frame). Depth is 40 % of
the raw bytes (§4g); the compressed saving is to be measured, not assumed. The dust model's transmission
`exp(−β·d)` is smooth in d, so bilinear upsampling is adequate except at depth edges (a stated approximation);
evaluation uses depth only at its own resolution. Phase 1 missions ~2–3 min per agent, recorded once, sizes
re-measured on the first real bag.

**"Runs from the bag" needs a ROS-free reader.** The core image has no ROS and the sim image has no OpenCV. The
choice is to add a **minimal CDR decoder** in `cpp/runtime/replay/` for exactly `sensor_msgs/Image`,
`CameraInfo`, `Imu`, `geometry_msgs/PoseStamped` and `tf2_msgs/TFMessage` (XCDR1 little-endian, fixed layouts),
reading MCAP through the existing Foxglove SDK. Sim bags stay standard rosbag2 (viewable in Foxglove/rviz). The
alternative, a recorder node writing Ariadne FlatBuffers, would put a custom writer on the render host and lose
tool compatibility. This closes D25's design gap.

---

**Nightly CI (CLAUDE.md §4).** A 10+ GB bag taking ~18 min on llvmpipe cannot run in GitHub Actions, and
nightly S3 egress to GitHub runners would spend the credit budget. CI uses a dedicated **`ci_tiny` preset**:
2 rooms + 1 corridor (the plan's hand-authored topology), 1 agent plus a 5 s second-agent segment for the
inter-agent path, ~20–30 s of motion, depth off, rendered **inside the CI job** from (seed, preset). That keeps
it deterministic in world and trajectory, and it never downloads a bag. Its render cost is measured and budgeted
before it enters nightly. Full-size bags live in S3 `datasets/` and are consumed only on the AWS nodes.

## 8. Deviations from the arch text (each becomes a DECISIONS entry on approval)

1. **IMU noise lives in a consumption-time operator, not SDF `<imu><noise>` (§5.3).** Forced by §4g: gz's IMU
   produces no signal under kinematic motion. The operator is more consistent with §5.9 (clean recording, seeded
   degradation). Arch §10.2's "simulator and estimator read the same numbers" is kept: one IMU config file,
   recorded once in the manifest.
2. **1.1 uses the procedural generator, not a hand-authored two-room scene.** The simplest preset (2 rooms + 1
   corridor, no cycle) reproduces the plan's hand-authored scene exactly; the 3–4-room maze presets are the
   default.
3. **Phase 1 textures are rich and unique, not §5.5's "low-texture concrete".** Richness τ is the dial toward
   §5.5 in Phase 5; aliasing r is the I7 dial.
4. **Phase 1 GT is 2D (exact for extruded plans), not voxel/SDF.** 3D GT arrives with collapse operators.
5. **GT depth recorded at 640×360** rather than the colour resolution (sizing, §7).
6. **GT pose comes from the mover** (exact, from the trajectory it executes), not gz's pose-publisher system
   (§5.3). They are the same numbers in a physics-free world; the mover's has no publish-rate jitter.
7. **Walls are OBJ meshes, not trimesh** (§5.5 stage 3 names trimesh; a ~60-line OBJ writer is sufficient for
   extruded rectangles and adds no dependency).

---

## 9. Build order (risk first)

1. **OBJ UV probe:** one OBJ wall with an asymmetric test texture (gradient + a letter), rendered. Check
   V-flip, orientation and aspect before the generator depends on OBJ output.
2. **Floorplan generator + seed sweep:** ~1k seeds per preset, recording acceptance rate and the entrance→target
   length histogram. This sets the length bands and `max_attempts`. The histogram is the evidence for "varying
   path length recoverable on seed". Golden scene-hash test.
3. **Textures + GT layer + scripts** (overlap metric, clearance assertions), all offline Python, unit-tested.
4. **ROS-free CDR decoder** in `cpp/runtime/replay/` (gates 1.1 as well as 1.3: calibration recovery and the
   texture gate both run in the core image from bags).
5. **Mover plugin + synthesized IMU + agent SDF + bridge YAML**, with an IMU numerical test: finite differences
   of the GT pose vs the published ω and f.
6. **First real bag** on `ci_tiny` and one full preset: re-measure RTF with the real light count (~10), and
   bytes with depth at 640×360; run §6 calibration recovery and the texture gate.
7. **MCAP evidence run** with manifest; the 1.4/1.5 pipeline consumes it (the "runs from the bag" criterion).

## 10. Research basis

- SubTGraph (arXiv 2605.20917): grid discretization, topometric tiles and constraint-driven Dijkstra paths for
  controllable topology in generated SubT worlds.
- Unexplored cyclic dungeon generation (Boris the Brave, 2021): entrance/goal splitting a loop into two arcs.
- Wilson's algorithm: uniform spanning trees (perfect mazes); braiding for cycles.
- Two-step constructive dungeon generation (arXiv 1906.04660); constrained-growth floor plans (Lopes et al.,
  GAMEON 2010).
- pcg_gazebo (Bosch Research): procedural Gazebo worlds from Python.
- OpenVINS `Simulator`: SE(3) B-spline trajectory, IMU from spline derivatives plus seeded bias/white noise.
- gz-sim8 `Imu.cc` / `Physics.cc`: the IMU reads Physics-owned velocity/acceleration components.
- ORB-SLAM/DBoW2: similar-viewpoint limits of binary-descriptor place recognition; Swarm-SLAM 15 % outlier rate
  (ALGORITHMS.md) ⇒ several overlaps needed for consensus.
