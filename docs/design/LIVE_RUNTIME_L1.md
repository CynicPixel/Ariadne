# L1 — the live closed-loop runtime (design; decisions D150–D155)

Status: implemented 2026-10-23 (session B); acceptance on the local llvmpipe host, evidence `evaluation/runs/L1/`;
the AWS GPU path is built, not run. Amendments from the runs: D152a (estimators take turns), D153b (standoff goals,
the previous outcome is an input), D154 fix (console directory).

Register row L1. Arch §5.4 (bridge, topics, sim clock), §5.8 (agent motion),
§5.9 (record/replay; the live exemption), §5.10 (live topology), §25 (the choose step), §29 (console).

## 1. What it is

Today every Ariadne result comes from a recording: Gazebo renders a scripted trajectory once, and the estimator,
the mission layer and the console consume the bag afterwards. The mission's decisions are logged but never executed
(`trace.jsonl`: `mode replay, executed false`). L1 closes the loop: a decision moves the agent, the agent's cameras
see the consequence, and the estimator's next output feeds the next decision.

```text
gz sim (GPU EGL)                   ── one sim container ──────────────────────────────┐
  cameras / IMU / GT pose ──gz-transport──> RosGzBridge ─┐ intra-process (one         │
  KinematicMover <──gz-transport── Pose_V ──── RosGzBridge <┘ component container)      │
                                               │     ^                                  │
                                               v     │ PoseArray (goto/follow/hold)     │
                                         ariadne_bridge::RuntimeLink (typed subs ->     │
                                         CDR bytes; CDR bytes -> typed pubs)            │
                                               │     ^                                  │
──────────────────────────────── Unix socket: framed (topic, CDR bytes) ───────────────┘
                                               v     │
ariadne_live (core image, ROS-free) ── reorder by stamp ── consumed-stream MCAP (the recording)
   per agent: LiveSimSource -> RunStereoVo (the replay estimator path, unchanged)
   mission: keyframe depth -> occupancy -> candidates -> chooser -> command   ──> trace.jsonl
   telemetry.jsonl + map snapshots ──> operator console (live mode, WebSocket)
```

## 2. Decisions

**D150 — where ROS stops.** ROS 2 exists only in the sim container: `ros_gz_bridge` (the 1.1 bridge YAML, plus
the command and mover-state entries) and one relay component, `ariadne_bridge::RuntimeLink`. The relay subscribes
to the bridged topics with their real types and serializes each message to CDR with `rclcpp::Serialization`. It
writes `(topic, CDR bytes)` frames to a Unix socket. The Ariadne runtime decodes them with its own ROS-free decoder
(`runtime/replay/ros_cdr`, D38): the same decoder and the same frame assembly the bag replay uses. No ROS type
enters `cpp/` (CLAUDE.md §3). The sim image ships no OpenCV and the core image ships no ROS; neither changes.

- *Why not run the estimator inside a ROS node:* that needs OpenCV 5 in the ROS image (a third image, and an
  ambiguous `opencv_build_hash`) and moves ROS types next to the estimator.
- *Why not read gz-transport directly, as the recorder does (D40):* the brief says ros_gz_bridge, and live runs are
  where the full ROS graph belongs (arch §5.10). D40's loss finding applies to the DDS hop, so that hop is removed:
  `ros_gz_bridge::RosGzBridge` is a registered rclcpp component (verified in the installed 1.0.24 package) and
  RuntimeLink is loaded into the same component container with `use_intra_process_comms`. Every input topic still
  counts its stamp-grid gaps, so a loss is measured, not assumed away.
- *Precedent:* domain_bridge and rosbag2 relay serialized messages the same way.
- **ros2/ packages.** `ariadne_bridge` (RuntimeLink, the launch file composing it with RosGzBridge) and
  `ariadne_sim` (the live world launch, render-mode selection). No `ariadne_interfaces`: commands and mover state
  ride on standard types that ros_gz_bridge maps (`geometry_msgs/PoseArray` ↔ `gz.msgs.Pose_V`,
  `std_msgs/String` ↔ `gz.msgs.StringMsg`; header stamp and frame_id verified to pass through, 2026-10-22). No
  `ariadne_agent` / `ariadne_operator`: the agent runtime and the console are not ROS nodes by D150.

**D151 — the command interface (arch §5.8).** One topic per agent, `/ariadne/agent_<k>/cmd`, PoseArray in the world
frame (`frame_id: world`): 0 poses = `hold()`, 1 = `goto(pose)`, N = `follow(path)`. The header stamp is the sim
time of the decision: `(agent, stamp)` is the command id that links a trace record to the motion it caused. The
mover answers on `/ariadne/agent_<k>/mover_state` (String, JSON): accepted or refused (with the reason), and the
segments it scheduled.

- **Motion = appended script segments.** The mover keeps its D38 script and appends turn/move segments from the
  state where the new motion begins: turn to face each waypoint, move straight, then turn to the commanded final
  yaw. Profiles are the D38 family (`plan_profile`: `v_cap`, `a_max`, `omega_max`, `alpha_max`, with the
  image-flow speed cap of `routes.py`), so a commanded path is as trackable as a scripted one. One
  dependency-free planner (`cpp/core/trajectory/command_planner.hpp`) is used by the plugin and tested in the core
  suite.
- **Preemption.** A new command replaces every segment not yet started. The segment in progress finishes, so the
  rig never stops mid-ramp (all segments end at rest; no velocity discontinuity).
- **Safety net.** Before accepting, the mover samples the new segments every 10 ms against the scene ESDF at
  `rho_agent`, as at start-up (D38). A violation refuses the whole command and the agent holds. The planner plans on
  the *estimated* map; the GT check is the bumper, and every refusal is in the trace.
- **The executed script is logged** (every accepted segment, exact doubles) so the canonical IMU/GT of a live run
  can be regenerated offline (D39's tool, same `SampleAt`).

**D152 — the recording is what was consumed.** `ariadne_live` reorders the incoming frames by header stamp behind
a watermark: a message is released when every input topic has delivered a stamp ≥ it, or after a hold-back
timeout, which is logged. Each released message is appended to `inputs.mcap` (rosbag2 MCAP layout, so
`RosbagReader` reads it) with log time = release sequence number, strictly increasing. Replay
(`ariadne_live --replay inputs.mcap`) feeds that order verbatim through the same code. The first replay check is a
digest over the consumed sequence (topic, stamp, sha256 of the payload), before any output is compared.

- **"Reproduces" is defined before the run.** For each agent: sha256-identical `estimate_body.tum`, identical
  keyframe count. Mission: identical `trace.jsonl` once wall-clock fields are stripped. Replay runs twice (two VO
  threads share OpenCV's thread pool; the second run shows the replay itself is deterministic).
- **Live-only tolerance.** A frame whose stereo partner never arrives is dropped and counted (`unpaired_frames`),
  never thrown (SimSource's strict rule stays for bags). A dropped frame enters neither the estimator nor the
  recording.

**D153 — the mission layer runs incrementally, on a sim-time barrier.** The 1.8 logic (OctoMap-model grid, slab,
frontiers, A* at both ρ) moves out of `ariadne_mission.cpp` into `cpp/planning/mission/frontier_mission.hpp`.
The batch tool calls it, with output unchanged, and the live runtime calls it per keyframe and per replan. A replan
at sim time T runs only after BOTH agents' estimators have passed T, so it sees the same keyframes live and in
replay.

- **Anchoring, live.** Each agent is anchored by its OWN briefed entry pose (the `--agents k` rows of 1.8). The
  merged anchor through the 1.6 loop is not used live. Phase 3's map merge will replace this.
- **Candidates + chooser (§25.1, built now).** The deterministic layer enumerates, per agent, the frontier
  clusters reachable at `rho_agent`, each with its path, and the safety envelope (`hold` is always a candidate). A
  chooser picks one: `classical` = the nearest (the 1.8 policy); `llm` (7.1) = the orchestrator's pick, validated
  against the set (out-of-set → classical, logged). 7.2 is the config switch `chooser: classical`. Replay re-applies
  the RECORDED choice from the trace and never calls a model.
- **Phase 4 replaces** `frontier_mission.hpp`'s candidate generator (NBV, allocation). The chooser interface and
  the command path stay.
- **The executed command.** `go` → `follow` along the A* path to the cluster's nearest reachable cell, simplified
  with `cv::approxPolyDP` (ε = 1 cell, each chord re-checked on the traversable grid), truncated to
  `max_command_m`, ending with a yaw that faces the cluster centroid (observe what was approached). Trace records
  carry `mode: live, executed: true` and the command id.

**D154 — console fed live.** `ariadne_live` appends `telemetry.jsonl` (5 Hz in sim time: estimated poses in W, the
latest decision per agent, mover state, byte counters) and writes the map snapshot at each replan. The console
server gains a live mode that tails the run directory and pushes the same frame schema over its WebSocket. It logs
every frame it served (wall time, sim time) as the evidence that it was fed during the run.

**D155 — where live runs execute.** The host has no GPU. Live runs go to one g4dn.xlarge spot instance (the D84
quota: 4 vCPU, T4) through `infra/scripts/spinup.sh` (three brakes, D27). Images are pushed to ECR by digest and
pulled. Binaries built locally in `ariadne/core-dev` ship as a payload with SHA256SUMS (the L5 pattern). The NVIDIA
EGL path of `ariadne/sim` has never run, so it is a `[SMOKE-TEST]` gate first, recorded in SMOKE_TEST_RESULTS.
CPU is the budget, not GPU: the GT depth cameras are off in the live world (the estimator never sees them), and the
world's real-time factor is set from the smoke's measurement.

**D165 — lockstep (2026-10-23; A1's 4.3 gate, maintainer's choice via A1).** Unpaused, gz keeps running while
the runtime plans and chooses, so a command lands ~1.7 s of sim time after its barrier (measured `applied_ns −
stamp`) and that lag depends on wall latency: two live runs of one seed differ. `lockstep: true` in the live config:
- **gz does the pausing, sim-side.** `[SMOKE-TEST §4w: the bridge forwards run_to_sim_time; /clock stops at T exactly]` The runtime asks for `WorldControl.run_to_sim_time = T` (gz-msgs10 field 7; gz-sim8
  `SimulationRunner::SetRunToSimTime`): gz runs to T and pauses itself, independent of how far behind the runtime's
  processing is. The call is the `/world/<w>/control` service, bridged by `ros_gz_bridge` 1.0.24
  (`ros_gz_interfaces/srv/ControlWorld` ↔ `gz.msgs.WorldControl`/`gz.msgs.Boolean`, one service entry in the bridge
  YAML), called by RuntimeLink when the runtime sends a `#run_to` control frame (int64 ns). No ROS or gz type in the
  core: the frame carries a number. RuntimeLink answers `#run_to_ack` with the sim clock at the moment of the request
  (the paused time; `/clock`, subscribed locally, not forwarded) and the service's verdict.
- **The barrier needs a frame at T.** `[SMOKE-TEST §4w: paused at T, the frame at T is delivered, none after; a world starting paused has no t = 0 frame]` Measured on the first LLM run: the cameras tick every 50 ms exactly, and a frame
  is stamped at every 5 s barrier (1801 frames, all 18 barrier stamps present), so a world paused at T has delivered
  everything the barrier waits for.
- **Zero lag, not "one step".** `[TESTED: test_mover_live.py::test_command_taken_while_paused, applied_ns = the paused time]` The mover now takes queued commands in `PreUpdate` even while paused (gz-sim calls
  systems while paused; motion still advances only when unpaused), so a command sent at barrier T is applied at sim
  time T and answered. The runtime waits for both answers BEFORE asking for `run_to T + replan`: every command's
  `applied_ns` equals its barrier in lockstep. Unpaused runs are unchanged (the next update applies it, as before).
- **Logged per barrier** in `lockstep.jsonl`: T, `paused_sim_ns` (clock when the next run_to was requested),
  `applied_sim_ns` per agent, `resume_to_ns`, the wall seconds the world spent paused. The figure states the lag from
  these, never assumes it.
- live.sh does not unpause a lockstep run: the runtime's first `run_to` (after `#ready`) starts it. Replay is
  unchanged: it consumes the recording, which is the same in either mode.

## 3. Acceptance (register L1)

1. One live two-agent run: decisions are executed (`executed: true`), each command id appears in the mover's state
   as accepted (or refused, with the reason), and the agent's estimated motion follows it: decide → move → observe.
2. Its `inputs.mcap`, replayed twice, reproduces §D152's outputs exactly.
3. The console served frames during the run (its frame log).
4. Bag replay (`SimSource`, the Phase 1 chain) is unchanged: targeted tests, plus the 1.8 tool's output on the
   committed spool, byte-identical before and after the extraction.
