# Procedural room-maze worlds, kinematic mover, MCAP recording

Status: **worldgen implemented and frozen (GENERATOR_VERSION 1); mover, CDR decoder and recorder implemented (D38); full-mission evidence pending**
(2026-09-30). Implements arch §5.3–§5.5, §5.8, §5.9 and plan items 1.1–1.3. Deviations from the arch text are
listed in §8 and recorded as DECISIONS D37.

Evidence: `validation.md` §4f–§4j (measured in the pinned image); census runs in
`evaluation/runs/gates/phase1_thin_e2e/worldgen_census/`; research list in §10. Code: `python/dataset_tools/worldgen/`,
tests `python/tests/worldgen/` (run `infra/scripts/worldgen.sh test`), presets `configs/worldgen/`.

---

## 1. What the worlds must do (requirements derived, not assumed)

| Consumer | Requirement on the world | Where it comes from |
|---|---|---|
| ORB/FAST + LK (1.4) | Corners at every scale on every surface, visible at 0.5–6 m; no large uniform areas | §7.3, D34 |
| SGBM (1.4) | Horizontal texture variation in every row, non-periodic (periodic = disparity aliasing) | §9.3, D34 |
| Stereo range | Most surfaces between Z_min 0.25 m and ThDepth 6 m (60·B) | §9.4, D36 |
| Inter-agent closure + consensus (1.6; PCM/GNC Ph.3) | **Several** same-heading, co-visible overlaps between agents in **≥ 2 distinct rooms** | §14, Swarm-SLAM |
| Place recognition hygiene | Distinct places look distinct in Phase 1; aliasing is a *dial* for the I7 lab (3.5) | §5.5 |
| Egress (Invariant 4) | Routes defined at ρ_responder = 0.40 m; agent-only passages so ρ_agent and ρ_responder routes differ | §20.5, §22 |
| Record/replay (C6) | World, trajectories and IMU are pure functions of (GENERATOR_VERSION, seed, preset) | §5.9 |

ORB/BoW place recognition fails on opposite-direction revisits, so "both agents walked this corridor" is not an
overlap unless headings agree. Overlap is **measured** (§4.4), never assumed.

---

## 2. Floorplan (arch §5.5 stage 1) — `layout.py`, `scene.py`

Everything is in integer grid units (grid_m = 0.05 m): exact rasterization, identical on every machine.

1. **Lattice** `nx × ny` of 6 m macro-cells (presets 2×1 … 5×5).
2. **Uniform spanning tree** over the lattice by **Wilson's algorithm** (loop-erased random walks; uniform over
   all spanning trees — tested: each of the 4-cycle's four trees appears 25 % ± 1.9 % over 8000 draws).
3. **Rooms**: `count` cells get a rectangle (3.5–5.0 m sides) containing the cell centre with a margin, so every
   corridor (they run between cell centres) opens into the room.
4. **Entrances**: boundary rooms; the second maximizes tree hop distance from the first. Each gets a vestibule
   (corridor out through an exterior side, 2.5 m past the lattice); the agent starts 0.6 m from its closed end.
5. **Target candidates** in seeded order. For each (`complete_for_target`):
   - **route-cycle cover** (`topology.route_cycle = 1`): add non-tree edges until every tree edge on every
     entrance→target path lies on a cycle. *Theorem:* in tree + extra edges, s and t are 2-edge-connected iff no
     bridge separates them, and the bridges are exactly the tree edges on no added edge's fundamental cycle.
     Greedy set cover (seeded tie-break), at most `cover_edges_max` edges; verified by unit-capacity max-flow;
   - `extra_edges` more random braids (maze loops), dead-end culling, `dead_end_stubs` re-attached;
   - **narrow passages** (0.55–0.70 m: agent-passable, responder-blocking): non-bridge edges whose removal keeps
     every entrance 2-edge-connected to the target;
   - corridor widths 1.0–1.4 m (≥ 2ρ_responder + one cell: the exact passability rule, §2.1).
6. **Length band**: the ρ_responder geodesic (8-connected Dijkstra, no corner squeezing) from the primary
   entrance start to the target room centre must lie in `target.path_length_m`. An octile lower bound prunes
   first; the search raster's ESDF is capped just above ρ.
7. **Furnish and re-verify** (§3), **script routes** (§4), **gate** (§4.4).

Any failure increments `attempt`; all randomness of attempt *a* is Philox-keyed `(seed, a, site)` with a fixed
site id per decision. The accepted attempt is a function of (seed, preset) and is recorded anyway. Rejection
reasons are counted and reported; an unsatisfiable preset fails loudly after `max_attempts`.

**Scene identity** = `(GENERATOR_VERSION, seed, preset digest, attempt)`; the preset digest is over the parsed
values (comments and key order never matter). Golden digests of two full scene directories are pinned in
`test_golden.py`; any output change must bump `GENERATOR_VERSION`.

**Randomness (D18 lesson, Python side):** a bit-exact port of the C++ Philox4x32-10 stream (Random123 KATs and
words from the C++ class are tests); every mapping from bits to values is owned (rejection-sampled integers,
53-bit floats, Fisher–Yates). Texture noise is hash-addressed (a pure function of position), never a stream.

### 2.1 Passability on the grid, exactly
The ESDF is the exact Euclidean distance from each cell centre to the obstacle squares (separable transform with
edge offsets; tested against brute force). An *n*-cell corridor's maximum clearance is (⌈n/2⌉ − ½) cells, so
width ≥ 2ρ + g always passes ρ and width ≤ 2ρ always blocks it (tested for both ρ values and n = 4…39).

### 2.2 Presets and length bands — from the census, not guessed
Census (1000 seeds per preset; generator before props/routes existed; to be re-run from a clean commit): the
ρ_responder geodesics achievable from the primary entrance. Bands chosen non-overlapping; the per-attempt hit
rate makes 50 attempts effectively certain.

| Preset | Lattice | Band (m) | Census median | Per-attempt hit | Notes |
|---|---|---|---|---|---|
| `maze_short` | 3×3 | 10–20 | 15.4 | 73 % | |
| `maze_medium` | 4×4 | 20–30 | 20.4 | 51 % | |
| `maze_long` | 5×5 | 30–45 | 26.2 (p95 44.7) | 37 % | "40–60 m" was infeasible |
| `ci_tiny` | 2×1 | 4–16 | — | — | two rooms + one corridor, no cycle (§7) |

---

## 3. Geometry, textures, boards, lights — `geometry.py`, `texture.py`, `furnish.py`, `build.py`

**Surfaces are single-sided textured quads** (OBJ with explicit UVs), because gz `<box>` stretches textures (§4g)
and the camera is always in free space. Walls = maximal runs of free/solid cell edges (split ≤ 8 m), facing into
free space; floors/ceilings = a greedy rectangle partition of the free mask. The quad convention (p0 bottom-left
seen from the front, CCW, UV (0,0) at p0) is the one verified on a render (§4h). Texel density 256 px/m walls,
128 px/m floors/ceilings: at the 1 m corridor's ~0.5 m clearance texture is magnified 3–6× — a stated trade-off
(texel-matched ~1300 px/m would be ~25× the memory).

**Textures**, per surface from `(seed, surface id)`: 1/f value noise (concrete) plus the corner-bearing layers the
§4g render proved necessary — hard-edged pits/aggregate (2–14 px), thresholded stains, random-walk cracks,
sharp-cornered paint marks. Rooms get distinct hue/brightness styles; corridors a grey family. Richness `τ`
(appearance.tau) scales every layer but the base; `repetition` > 0 (the I7 aliasing dial) is refused until 3.5.
Pixel data is integer Philox + IEEE `+ − × ÷`, so it is identical everywhere; PNGs carry no metadata and are
compared by pixel digest.

**Props**: boxes against room walls (0.4–1.2 m wide, 0.4–1.8 m tall), in the GT occupancy but never in wall
extraction. Keep-outs: squares around the room centre (GT target point) and the lattice-cell centre (survey
spot), every corridor mouth + 0.6 m, the volume in front of each board, 0.3 m between props. After furnishing
the raster is rebuilt and the band, start/target clearance and every responder passage's reachability are
re-verified.

**ChArUco boards** (DICT_5X5_250, 7×5 squares of 8 cm, 6 cm markers, one-square white margin; 0.72×0.56 m) are
drawn by OpenCV itself (`cpp/runtime/tools/ariadne_charuco_assets.cpp`) into `simulator/assets/charuco/` (PGM +
`boards.json`). Board *k* uses marker ids [17k, 17k+16]: **disjoint**, so the two entrance boards can never be a
false inter-agent loop. Each entrance room gets its board on the wall facing the entering agent, at camera
height, clear of mouths, 2 mm proud of the wall, at its exact physical size. `scene.json` records the world
corners of each board's squares area (TL TR BR BL seen from the front), so evaluation assumes no OpenCV
board-frame convention.

**Lights** (Phase 1 "normal"): ambient 0.35 + a point light over every room and junction (9 in `maze_medium`
seed 0), no shadows. Agents carry the §5.3 spot lamp, inert until Phase 5.

**Scene directory**: `model.sdf`/`model.config` (for `model://` use), `model_inline.xml` + `world.sdf` (the scene
model **inlined**: sdformat14 does not resolve relative `<include>`, §4i), `meshes/`, `textures/`,
`esdf_f32.bin` (for the mover), `scene_gt.npz`, `routes/agent_<k>.json`, `scene.json`, `preview.png`,
`manifest.json` (command, git commit/dirty, image id, versions).

---

## 4. Ground truth, motion, routes — `raster.py`, `motion.py`, `routes.py`

### 4.1 GT layer (Phase 1 subset of §5.5 stage 5)
Exact for extruded 2D plans: structural and navigable free masks, ESDF (capped 4 m), traversability at ρ_agent and
ρ_responder, geodesic distance fields and routes from every start at both ρ, target and start cells. 3D GT
arrives with the collapse operators.

### 4.2 Motion primitives
A segment moves a scalar u from 0 to D with an **S-curve**: velocity ramps with σ(τ) = 10τ³ − 15τ⁴ + 6τ⁵,
cruises, ramps down. σ′ and σ″ vanish at both ends, so acceleration **and jerk** are continuous: the synthesized
IMU has no steps. Long segments cruise at v_max after ramps of Ta = 1.875·v_max/a_max; short ones use the
cruise-less member whose peak acceleration is exactly a_max (time-optimal within the family). A pure
minimum-jerk move would average only v_max/1.875 (tested: long segments average > 80 % of v_max).

Pose is (x, y, z, yaw, pitch), roll 0, R = Rz(yaw)·Ry(pitch) (positive pitch = nose down). A **turn** moves
(yaw, pitch) on a straight line in angle space; for this Euler pair ω_B = (−ψ̇ sin θ, θ̇, ψ̇ cos θ) and
|ω_B| = |(ψ̇, θ̇)|, so the S-curve's rate limit is exactly the angular-speed limit. A **move** translates in 3D at
fixed orientation (the body is holonomic). Specific force f_B = Ryᵀ Rzᵀ (a_W + g ẑ), g = 9.80665. Both are
tested against finite differences of the pose (ω from the skew part of RᵀṘ).

**Speed limits from the tracker** (placeholder LK: win 21, 3 levels), budget d_max = 30 px/frame at f = 640,
Δt = 0.05 s: v ≤ d_max·c/(f·Δt) with c the minimum over the move of the navigable ESDF and the floor/ceiling
distance (0.47 m/s at 0.5 m clearance), capped at 0.6 m/s; yaw rate at the image edge 2f·ω·Δt ≤ d_max ⇒ ω ≤ 0.44
rad/s. Every 50 ms sample of every script is tested against both limits and against the collision envelope
ESDF ≥ ρ_agent; altitude varies ±0.2 m per move (IMU excitation).

### 4.3 Routes
Routes run on the responder cell graph (narrow passages excluded) through lattice-cell centres (corridor axes).
For each entrance *j*, **(A_j, B_j) = the least-length pair of edge-disjoint routes to the target** (Suurballe's
problem as a 2-unit min-cost flow, Bellman–Ford on the residual graph, opposite flows cancelled). Deleting one
specific shortest path is *not* equivalent and can wrongly disconnect.

| Template | Agent k | Use |
|---|---|---|
| `cross` (default) | A_k out → target → reversed B_{other} → other entrance room → **out through the other entrance** to its vestibule | Every return leg is edge-disjoint from the outbound leg it would meet head-on; each agent surveys the target and both entrance rooms |
| `loop` | A_k out → target → reversed B_k → own vestibule | Intra-agent loops (Phase 2) |
| `return` | out and back along one route → own vestibule | Maps without a route cycle (`ci_tiny`) |

**Surveys**: a 360° spin at a seeded per-agent offset (≤ 0.5 m, heading kept) from the cell centre, so the two
agents' views of a shared room differ by a real baseline, not only altitude. `routes.surveys` = `all` (entrance
rooms + target) or `target`.

**Agent–agent separation**: Phase 1 agent models carry **no visual geometry** (agents are invisible to each
other; a stated simplification, D37). Physical plausibility is still enforced: the minimum 3D distance between
the agents over the mission (50 ms sampling) must be ≥ 1.0 m; the scheduler tries start delays (5 s steps up to
60 s) on agent 1, then agent 0, and records the delays and the achieved separation.

**Calibration dance** (`routes.calibration = 1`, agent 0 only — one rig model): three view positions in front of
the entrance board (near-left, near-right, far-centre), each aimed at the board plus two **diagonal** tilts
(±15° yaw with ±12° pitch: board planes in general position; rotations about one axis only are a degenerate
Zhang configuration), short holds; then constant-speed passes parallel to the board at 0.2 and 0.5 m/s,
clamped to the clear span. Segments are tagged `calib:*`; evaluation excludes them from loop-closure metrics.

### 4.4 Closure-opportunity gate
Keyframe-like samples (0.5 m / 15°); an inter-agent opportunity is a pair within 1.5 m and 30° heading with
**grid line of sight** (props occlude). Gate: `min_inter_pairs` (3) in ≥ `min_inter_rooms` distinct rooms (2; 1
for `ci_tiny`), `min_intra_pairs` (0 for `cross`). On the seeds inspected: ~400–850 pairs in 3 rooms per
full-preset scene (the accept sweep records the distribution).

---

## 5. Simulation: agent SDF, mover plugin, IMU, bridge (1.1, 1.2) — *next*

**Agent model** `ariadne/agent_<k>` (§5.3): one link with left/right cameras (1280×720, HFOV 90°, B = 0.10 m,
20 Hz, identical intrinsics, mild plumb-bob distortion verified by §6), co-located depth camera (GT only), inert
spot lamp; no visual geometry (§4.3). Free-running cameras at one rate (§4f: stereo pairing 1.0); triggered
cameras not needed.

**Mover** = C++ gz system plugin, built in the sim image (gz-sim 8.15 headers + g++ present). No Physics system.
Each `PreUpdate` it evaluates the script at `info.simTime` and writes `components::Pose` (proven §4g); skipped
while paused. The script arrives as `<segment>` elements inside the plugin's SDF block (written by the mission
composer; parsed with `sdf::Element`, no JSON dependency). It evaluates exactly the §4.2 formulas from each
segment's stored (D, Ta, vc) — it never re-plans — and re-checks the collision envelope against
`esdf_f32.bin`. Python↔C++ agreement is tested on golden samples with a 1e-9 tolerance (libm differs in last
bits).

**Lens (D41).** The render is a pinhole (ogre2 ignores SDF distortion, §4n). The camera the estimator sees is
`rig.lens`, a Brown-Conrady model applied at read time by `perception/degradation/lens.hpp` before the sensor
stages; its focal length (680) exceeds the render's (640) so barrel distortion never samples outside the render.

**IMU — synthesized, not gz's** (gz's is blind under kinematic motion, §4g): clean ω_B and f_B at 200 Hz from the
§4.2 closed forms at the same `simTime`, header-stamped explicitly. Noise and bias random walk are a seeded
consumption-time operator reading the same `configs/sensors/imu_*.yaml` the estimator reads (D37).
Zero camera–IMU offset is *claimed only after* §6 re-checks render timing on RGB under full two-agent load.

**GT pose** from the mover at 200 Hz (exact), plus static camera extrinsics. **Bridge** (§5.4): one
`parameter_bridge` YAML under `/ariadne/agent_<k>/…`, `/clock`, `use_sim_time:=true`.

**Canonical IMU / GT streams (D39).** The live 200 Hz streams cross gz-transport → bridge → recorder and can
lose a sample under load (§4k/§4l), which no replay can repair. Both are exact functions of the scripts, so the
recording's IMU and GT pose come from `ariadne_canonical_streams` (`simulator/plugins/tools/`): it parses the
mission SDF with the plugin's own parser and samples with the plugin's own `SampleAt` (`MoverScript.hh`, both
targets `-ffp-contract=off`), on the grid t_k = k·period, k = 0..⌊T/period⌋ (the plugin publishes k ≥ 1; k = 0
brackets the first camera frame at 2 ms). Messages are encoded by `runtime/replay/ros_cdr_writer.hpp` with
zeroed CDR padding (rclcpp leaves padding uninitialised), each round-tripped through the runtime decoder
bit-exactly before writing. Reproducibility is scoped in D39: container bytes identical run-to-run in one
image (checked on every recording), content digest pinned by `worldgen.sh simtest`, and across images/arches
only the ≤ 1e-9 Python agreement is claimed. The live streams stay
in the sensor bag as a cross-check. This is "analytic IMU/GT from script" [DESIGN], never presented as sensor
output; realism (noise, bias, jitter) is the consumption-time operator's job.

---

## 6. 1.1 acceptance test — calibration recovery, not self-consistency

From agent 0's dance frames the core image runs `CharucoDetector` + `calibrateCamera` + `stereoCalibrate` and
compares the **recovered** K, D and left→right extrinsic with the SDF values; then per-frame `solvePnP` against
GT pose composed with the explicit, tested gz-link (+X forward) → optical (+Z forward) rotation and the board's
recorded world corners. The constant-speed passes expose a render/pose time offset δ as a PnP-vs-GT residual
≈ v·δ growing with speed; the run is under full two-agent load. Thresholds are set from the first measurement.

**Built** (`cpp/runtime/tools/ariadne_sim_calib.cpp`, SMOKE §4n): A GT reprojection without calibration (radial
trend → distortion, mean → principal point, correlation with image velocity → time offset), B free calibration
with standard deviations and a pinhole-only RMS, C stereo (free and with the hypothesised true K), D PnP vs GT
with a 180°-flip check, E board-frame self-check on the asset. First measurement: renders are an exact pinhole
(0.06 px), distortion is not rendered under ogre2 (D41 pending), camera_info cx/cy are +0.5 px, δ ≈ 0.

---

## 7. Recording (1.3), sizing, CI

`ros2 bag record -s mcap --storage-preset-profile zstd_fast`, one bag per `(scene, scripts, seed)`. Measured on
a real scene (§4j): **~84 MB per sim-second** for two agents (RGB dominates; depth resolution is irrelevant to
compressed size), **RTF ~0.17** on this 8-core llvmpipe host (one scene). On the seeds inspected, scripted
missions are 2–7 min per agent (`maze_short` ~2–4, `maze_long` ~4–7 incl. start delays), i.e. ~10–35 GB and
15–40 wall-minutes per bag.

**Runs from the bag** needs a ROS-free reader: a minimal CDR decoder in `cpp/runtime/replay/` for exactly
`sensor_msgs/Image`, `CameraInfo`, `Imu`, `geometry_msgs/PoseStamped`, `tf2_msgs/TFMessage` (XCDR1 LE), over the
existing Foxglove MCAP SDK. Sim bags stay standard rosbag2. This closes D25's gap and gates §6 as well.

**Recording path (D40)**: not `ros2 bag record` via the bridge — every loss in three logged full runs was on
that DDS link (SMOKE §4l). `ariadne_gz_recorder` subscribes to gz-transport directly, converts with
ros_gz_bridge's own `convert_gz_to_ros`, writes synchronously through `rosbag2_cpp`, and fails loudly on any
stamp gap, duplicate, late start, incomplete end or backlog overflow (D40). The world starts paused until the
recorder has discovered every publisher. The bridge remains for live runs.

**Recording layout (D39, D40)**: `RECDIR/sensors/` (rosbag2: rendered topics + live IMU/GT cross-check, log
time = stamp), `RECDIR/canonical/` (rosbag2: IMU + GT pose + `/tf_static`, offline, source of truth for those
topics), `RECDIR/logs/` (gz, recorder log + JSON report), `RECDIR/recording.json`.
Verification: canonical — exact stamp grid, log time = stamp, values = the Python evaluator (≤ 1e-9); cameras /
infos — strict (a frame cannot be regenerated: count, period, monotone, stereo pairing 1.0; remedy = re-record);
live IMU/GT — every sample bit-identical to canonical at its stamp (fields compared as IEEE bit patterns), no
off-grid or duplicate stamps; lost live samples are counted (`live_gaps`) and, since D40, any loss is also a
recorder error that fails the run; `/clock` strict; `/tf_static` equal to the manifest's extrinsics.
**Replay rule**: IMU, GT and `/tf_static` from `canonical/`, the rest from `sensors/`, merged by header stamp
(ties: topic name); since D40 log time = header stamp in both bags, so log-time order agrees.
The merge lands with the 1.4/1.5 sim consumer. Live (closed-loop/demo) runs have no canonical file: the
estimator must bridge a missing IMU sample (integrate across the gap, log it) — a Phase 2 requirement.
The §4g "frame stamped t shows pose t" result the canonical GT relies on was measured at 2 agents on llvmpipe;
re-check it whenever the agent count grows.

**Nightly CI — open.** `ci_tiny` missions are ~200 s (agent 0 with the dance) and ~60 s: ~16 GB, too large for
GitHub caches and too slow to render on a hosted runner nightly. Options to decide when the thin run is wired:
render once per (GENERATOR_VERSION, preset digest, seed, image id) on AWS and fetch a trimmed segment; or a
reduced-resolution CI rig. Recorded here, not decided.

---

## 8. Deviations from the arch text (DECISIONS D37)

Items 1–6 were in the approved draft; items 7–9 arose during implementation and
are **pending confirmation**.

1. **IMU synthesized by the mover; noise as a consumption-time operator**, not gz's IMU and SDF `<imu><noise>`
   (§5.3). Forced by §4g. One IMU config serves simulator-side noise and estimator.
2. **1.1 uses the procedural generator**; the plan's hand-authored two-room scene is the `ci_tiny` preset.
3. **Phase 1 textures are rich and unique**, not §5.5's "low-texture concrete"; τ is the dial toward it.
4. **Phase 1 GT is 2D** (exact for extruded plans), not voxel/SDF.
5. **GT pose comes from the mover**, not gz's pose-publisher (§5.3): same numbers, no publish jitter.
6. **Walls are OBJ quads written in-house**, not trimesh (§5.5 stage 3).
7. **Scene worlds inline the scene model** (sdformat14 relative `<include>` does not resolve, §4i).
8. **Phase 1 agents have no visual geometry** (invisible to each other); separation is enforced in the scripts.
9. **Motion primitives are S-curves with yaw+pitch**, not minimum-jerk, and the rig pitches during calibration.

Withdrawn: "GT depth at 640×360" (§4j showed it saves ~1 % of compressed bytes).

---

## 9. Build order and status

1. ✅ OBJ UV probe (§4h). 2. ✅ Floorplan generator, census, bands, golden. 3. ✅ Textures, GT, props, boards,
routes, gate. 4. ✅ ROS-free CDR decoder + rosbag reader (D38). 5. ✅ Mover plugin + synthesized IMU + agent SDF +
mission composer + bridge YAML; golden Python↔C++ samples; recorded GT/IMU equal the script exactly (§4k).
5b. ✅ Canonical IMU/GT streams + CDR encoder + two-bag recording layout and verifier (D39, §4l).
5c. ✅ Loss attribution (DDS link), direct gz-transport recorder with loud failures, `/tf_static` in canonical (D40).
6. ✅ 1.3 evidence recording from a clean commit (`evaluation/runs/gates/phase1_thin_e2e/sim_record/ci_tiny_0-20261001T042105Z`).
6b. ✅ §6 tool + first measurement (§4n); ✅ D41: pinhole render + read-time lens operator; §6 verdict with
thresholds (`configs/sim/acceptance_s6.yaml`). ✅ D41-rig 1.3 evidence recording; §6 evidence PASS on seed 0 and on
the independent seed 1 (`evaluation/runs/gates/phase1_thin_e2e/sim_calib/`).
7. ⏳ A full bag of one larger preset (RTF, bytes), the texture gate, and 1.4/1.5 consuming the sim bag
(camera model from `rig.lens`, never from camera_info).
Evidence to re-run from a clean commit: census + accept sweeps for all presets.

---

## 10. Research basis

- SubTGraph (arXiv 2605.20917): grid discretization, topometric tiles, constraint-driven Dijkstra paths.
- Unexplored cyclic dungeon generation (Boris the Brave, 2021): entrance and goal split a loop into two arcs.
- Wilson (1996): uniform spanning trees by loop-erased random walks; braiding for cycles.
- Suurballe (1974) / min-cost flow: shortest pair of edge-disjoint paths.
- Two-step constructive dungeon generation (arXiv 1906.04660); constrained-growth floor plans (GAMEON 2010).
- pcg_gazebo (Bosch Research): procedural Gazebo worlds from Python.
- OpenVINS `Simulator`: IMU synthesized from trajectory derivatives plus seeded noise.
- gz-sim8 `Imu.cc` / `Physics.cc`: the IMU reads Physics-owned velocity/acceleration components.
- Zhang (2000) calibration; Sturm & Maybank (1999) degenerate configurations (planes parallel to one line).
- ORB-SLAM/DBoW2 viewpoint limits; Swarm-SLAM 15 % inter-robot outlier rate (algorithms.md).
