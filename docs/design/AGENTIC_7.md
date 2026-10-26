# 7.1–7.2 — the agentic layer (design; decisions D156–D162)

Status: design 2026-10-23, session B. Plan items 7.1, 7.2; invariant 3. Arch §23 (human authority / risk gate),
§25 (agentic orchestration: §25.1 the LLM only chooses, §25.2 MCP tool surface, §25.3 the ablation), §29.3 (approval
REST). Builds on L1 (`LIVE_RUNTIME_L1.md`): the live runtime already computes the candidate set and calls a chooser.

## 0. Preflight (2026-10-23): literature, ALGORITHMS.md, OpenCV 5, advisor

**ALGORITHMS.md.** Domain 3 (active perception / NBV / multi-robot allocation) is still owed there, and it holds no
LLM-in-the-loop survey; this pass is that survey for the choose step only. Pointers it does hold: TARE / FAR Planner
(§20.2: sensor-agnostic SubT exploration planners, Phase 4's), "task-assignment commitment: exactly-once, two robots
on one frontier wastes the mission" (line 1165), lamp aiming as active perception for the agentic layer (line 1430),
CERBERUS's comms-aware backtrack rule (line 3197).

**Literature (web, 2026-10-23; abstracts and project pages read, full texts not).**
- *LLM picks among frontiers:* L3MVN (Yu et al. 2023, arXiv:2304.05501) — an LLM selects one frontier of a semantic
  map as the long-term goal, a classical planner drives there. LFG (Shah et al. 2023) — LLM scores of frontiers used
  as a planning HEURISTIC, the planner keeps feasibility. SayNav (arXiv:2309.04077) — the opposite end: the LLM plans
  over a 3D scene graph. Ariadne takes the L3MVN/LFG side: the LLM only ranks what the planner proved reachable.
- *Multi-robot:* Co-NavGPT (Yu et al., arXiv:2310.07937, RA-L 2025) — one LLM call over the MERGED map assigns
  frontiers to all robots at once; COMRES-VLM / LLM-MCoX (arXiv:2509.26324) — frontier clusters + doorway detection
  + a multimodal LLM assign waypoints to robots (self-reported 22.7 % faster with 6 robots; not checked). Survey:
  arXiv:2502.03814 classes Co-NavGPT's LLM as a task-allocation mechanism.
- *Guardrails:* RoboGuard (arXiv:2503.07885) — LLM plans checked by temporal-logic control synthesis against safety
  specs (unsafe executions 92 % → < 2.5 %). Human-in-the-loop large action models (arXiv:2512.11620) — static
  capabilities hard-coded, the plan shown to a person before execution. Ariadne's shield is simpler and stronger
  for its scope: a CLOSED candidate set computed by the planner plus a code-enforced risk gate (§23).
- *MCP for robots:* robotmcp/ros-mcp-server (rosbridge; listed as ROS 2 Jazzy-compatible) gives an LLM raw topic /
  service / action control, and its listings note no safety validation of generated commands. Ariadne deliberately
  does not: its MCP tools are read-only views plus choose-among (§25.1).

**What the pass changes in this design.**
1. *Joint assignment (Co-NavGPT; ALGORITHMS line 1165).* One decision request per replan covers BOTH agents (in W
   through their briefed anchors), and the answer assigns one candidate per agent. The C++ validation adds
   exactly-once: two agents are not sent to the same cluster while another eligible cluster exists. The classical
   chooser gets the same rule (greedy, agent 0 first), so the ablation stays "skip the choose step" and nothing else.
2. *Images, not only numbers (COMRES-VLM, Co-NavGPT's VLM version; arch §25.2 `observe_image`).* Nova 2 Lite takes
   images. `observe_image` returns the map rendered with OpenCV (tri-state slab, agents, numbered candidates, the
   egress route; `cv::imencode` PNG) and each agent's latest keyframe image (rectified left, 640×360). That makes
   "vision output → decision" literal and keeps the image path in OpenCV 5.

**OpenCV 5 adjacency.** The deterministic layer's geometry is OpenCV: `cv::distanceTransform` (standoff),
`cv::approxPolyDP` + `cv::LineIterator` (command paths), the clearance maps; the images the model sees are rendered
and encoded with OpenCV. Nothing in OpenCV 5 replaces the chooser itself (`cv::dnn` runs vision models, not a tool-
using language model), so the LLM stays on Bedrock (D84), outside every algorithm library (CLAUDE.md §3).

**Advisor.** Reviewed the first draft (2026-10-23): bottleneck clearance instead of the path minimum for the risk
(the minimum saturates on every turn), eligible candidates only, approvals verified by C++ from the console's file,
replay copies the recorded choice block. All four are in §2.

## 0b. Revision B (2026-10-23, maintainer: "go with B") — the seams around Phase 4, not a planner

**Why.** The planner I had in `ariadne_live` (frontier standoffs, one-shot A* paths, per-run knobs) duplicates arch
§21 / plan 4.3–4.6, which A1 owns (D110–D129) and is preflighting. The maintainer: "don't hard code the paths, let
the agents reach consensus and locomote, informed by llm, and directable/guidable by the user". Under B, A1 builds
NBV, CBBA consensus and the route graph (4.3–4.6); this item builds the seams those plug into, the LLM's task-level
role, and operator direction. The 1.8 + standoff code becomes `PlaceholderPlanner`, an ablation baseline (standing
rule), and its knobs stay inside it.

**Spec read first.** Arch Plane E (consumes maps, uncertainty, coverage, traversability, connectivity, energy, time,
existing assignments; produces candidate observation poses, task assignments, rendezvous tasks, revisit requests,
communication actions, human escalation requests); §21.7–21.8 (event-triggered CBBA, bundles 1–2, broker-greedy
ablation; FAR / D* Lite within-task path); §23 (risk gate, approval packet, envelope without comms); §25.1 (the LLM
owns task selection, negotiation, explanation, exception handling, escalation; never poses, paths or thresholds);
§29.3 (approval REST). A1's `VOLUMETRIC_4.1.md` §4 already fixes the INPUT: `planning::MissionInput` (per-keyframe
depth + agent states), "the L1 live runtime fills the same struct".

**Literature for the seams (2026-10-23; abstracts / project pages).**
- *Directed autonomy:* the SubT human–robot team study (arXiv:2212.05626) — across 16 missions the best mode was
  autonomy on a multi-agent task-allocation system that the operator steers by assigning tasks directly or by
  geometric prioritisation regions. CSIRO's SubT GUI (IROS 2022) encoded operator intent as tasks on its allocator.
  CERBERUS (arXiv:2207.04914): supervisor commands = explore an area within comms, explore beyond comms with a time
  budget, go to a frontier near unmapped space, and a waypoint only as a last resort.
- *Language → constraints, checked:* AutoTAMP (arXiv:2306.06531) — an LLM translates instructions into formal
  constraints and an LLM checker repairs them before a motion planner consumes them; Lang2LTL (arXiv:2302.11649) —
  grounded translation of navigation commands to LTL. Here: the LLM translates operator text into the SAME
  structured directives the console produces, validated by schema and map, and shown to the operator to confirm.

**The seams (what this item builds).**
1. *Planner interface* (`runtime/live/planner.hpp`): `Integrate(MissionInput keyframe)`, `Plan(PlanContext) ->
   PlanOutput`, where `PlanContext` = time, agent states, active directives, the previous output, and the mover's
   answers; `PlanOutput` = Plane E's products: per-agent task (kind, region or pose, utility terms, risk), route
   (waypoints the planner computed, or none = hold), the ranked alternatives the LLM may choose among, and
   escalation requests. `PlaceholderPlanner` wraps the current code; A1's Phase 4 planner implements the same
   interface (proposal to A1, not imposed: STATUS + `AGENTIC_7.md` §0b).
2. *Directives* (`planning/mission/directives.hpp`, deterministic, applied by the runtime to every planner):
   `priority_region(polygon, weight)`, `keep_out(polygon)`, `assign(agent, region)`, `explore(agent, region,
   time_budget_s)`, `goto(agent, pose)` (last resort), `hold(agent)` / `release(agent)`, `approve` / `reject`
   (§23). Sources: the console (operator) and the orchestrator (LLM, proposed → operator-confirmed). Each directive
   has an id, author, wall and sim time, and appears in the trace; the planner's output records which directives
   shaped each task.
3. *LLM role (task level, §25.1):* per replan, rank the planner's alternatives (joint assignments) with a reason, or
   escalate; translate operator text into directive proposals; explain. Never routes, poses or thresholds. The
   C++ validation of §2 stays (in-set, risk gate with a console-written approval, exactly-once comes from the
   allocator, not from a rule here).
4. *Consensus* is the allocator's (A1, 4.4). Until it lands, `PlaceholderPlanner` allocates by the joint greedy
   rule with exactly-once (ALGORITHMS line 1165), labelled placeholder.

**Review (advisor, 2026-10-23) — folded in.**
- *Directives are recorded inputs.* Console clicks arrive at wall time; the runtime takes directives only at the
  replan barrier, stamps each with the replan T at which it takes effect, writes `directives.jsonl`, and replay
  re-applies that file (the mover-answer pattern, D153b). `live_replay_check.py` compares it. Confirmed LLM
  proposals take the same path.
- *No second `planning::MissionInput`.* A1 owns that name on `feat/ph4-mission`; the live side gets
  `live::KeyframeInput` with the same fields plus an adapter, naming A1's struct so the swap is one line.
- *`PlanOutput` has "continue".* A command is sent only when an agent's route changes materially; otherwise the
  planner answers "no change" and the mover keeps executing (the 45 s run's turn-in-place thrash, solved by the
  event-triggered structure of §21.7, not by a knob). Replans stay on fixed sim-time ticks (determinism).
- *Approvals stay C++-verified files* (`approvals/<id>.json` written by the console); a directive may reference one,
  never replace it.
- *The gate never fires on ci_tiny* (risk ≤ 0.49): approval is exercised on a preset with narrow passages
  (`narrow_count > 0`), never by lowering the threshold.

## 1. What it is

At every replan the deterministic layer (`ariadne_live`, D153) has, per agent, a candidate set (frontier clusters
reachable at ρ_agent from the map built from keyframe depth) and the classical choice. 7.1 puts an LLM orchestrator
on Amazon Bedrock in the CHOOSE step, through an MCP tool surface over the deterministic core. It never computes
poses, paths or maps (§25.1): it reads what the core computed, picks one presented candidate, and may escalate to a
human. 7.2 is the same run with the choose step skipped (the classical rule).

```text
ariadne_live (C++)                         orchestrator (python/orchestration, Python)
  replan -> candidates + risk  ── decision request (JSON line, Unix socket) ──>  Bedrock converse (Nova 2 Lite,
  wait (timeout -> classical)                                                     DeepSeek V3.2 fallback) with the
  validate choice ∈ set        <── choice {candidate, reason, approval} ──────    MCP tools as toolConfig; MCP server
  command -> mover                                                                (stdio) answers each toolUse from
  trace: vision -> decision -> tool calls -> action / approval                    the request's snapshot
```

## 2. Decisions

**D156 — where the LLM sits: a chooser process, not inside the C++ runtime.** `ariadne_live` gets
`chooser: llm`: at each replan it writes one JSON decision request per agent to `<run>/chooser.sock` and waits up to
`timeout_s` (config, 20 s wall; 120 s in the live config, D163) for the answer. Out of set, malformed, timeout or a dead orchestrator → the classical
choice, with the reason logged (`fallback`). The C++ side validates; the LLM cannot inject a pose. The runtime stays
ROS-free and AWS-free (CLAUDE.md §3: no AWS SDK type in any algorithm library).

**D157 — the MCP tool surface (§25.2).** (Amended by §0: one request per replan for both agents; `observe_image`.) `python/orchestration/mcp_server.py`, mcp 2.3.0 `MCPServer`, stdio. Tools
are views over the decision request's snapshot (what the deterministic core computed at this replan) plus two
actions. The §25.2 names are kept where the semantics match; the rest are named for what they do:
- `observe_map` (§25.2 observe_image at map level): known free / occupied / unknown fractions, frontier clusters,
  the agent's estimated cell — derived from keyframe depth;
- `query_uncertainty`: tracking quality of the agent's estimator (keyframes, lost frames, tracks);
- `query_route`: the egress route at ρ_responder (found, length, exit) — invariant 4;
- `list_candidates`: id, path length, cluster size (information proxy), risk score and its reason;
- `request_human_approval(candidate, reason)`: files a §23 HUMAN_APPROVAL_REQUEST; returns the operator's verdict or
  `timeout`;
- `choose_candidate(candidate, reason)`: the decision. Must be called exactly once; ends the turn.
The other §25.2 tools (estimate_pose, compute_depth, retrieve_place_candidates, verify_loop,
request_peer_observation, allocate_task, request_rendezvous) belong to capabilities this loop does not exercise yet
(Phase 3/4); they are not exposed rather than stubbed.

**D158 — the risk gate (§23), deterministic.** Each candidate gets `risk_score` from the map: the narrowest free
clearance along its path c_min (the mission's EdgeClearance at ρ_agent) mapped linearly to
`risk = clamp((ρ_responder − c_min) / (ρ_responder − ρ_agent), 0, 1)`: 0 where a responder fits, 1 at the agent's
own limit. `autonomous_threshold` = 0.5 (config). A chosen candidate above the threshold needs approval; the
orchestrator must call `request_human_approval` first, and the C++ validation REJECTS an above-threshold choice
without an `approved` verdict (fallback: the classical choice restricted to below-threshold candidates, else hold —
§23's "outside envelope → hold"). The rule is in code, not in the prompt.

**D158a — approval path.** The console gets `POST /api/approval/{id}` and shows pending requests; the orchestrator
polls `<run>/approvals/<id>.json`. Unattended runs: no auto-approver. A timeout is a `timeout` verdict, the
candidate is rejected, and the trace says so (failure-handling evidence, §25.1). An approval during a demo is a
human click, logged with wall time.

**D159 — models (D84).** Primary `eu.amazon.nova-2-lite-v1:0`, fallback `deepseek.v3.2` (on a Bedrock error or
throttling, not on a "bad" answer). boto3 pinned; credentials from the instance role (AWS) or the `aws login` cache
(dev host); region eu-north-1. `temperature 0`. Every call's model id, latency, token usage and stop reason go in
the trace. Prices `[UNVERIFIED]` until pulled (D84); the budget per run is bounded by `max_calls`.

**D160 — the trace (invariant 3).** One JSONL record per decision in `trace.jsonl` (the same file as L1):
`vision` (the map/estimator facts the request carried) → `candidates` (with risk) → `orchestrator` {model, tool
calls in order with arguments and results, approval verdicts, latency, tokens} → `choice` + `validated` /
`fallback` reason → `command` → the mover's answer (mover_state, joined by command id). Nothing about how the code
was written appears near it.

**D161 — replay.** `--replay` never calls the model: the chooser re-applies the choice recorded in the live trace
(by replan time and agent). Estimator outputs are chooser-independent (they depend on pixels); the replay check of
D152 then also confirms the decisions.

**D162 — the ablation (§25.3, 7.2).** `chooser: classical` is "skip the choose step": the same binary, scene,
seeds, replan cadence and candidate generator. Compared per run: coverage of the GT free space (cells observed free
∩ GT free), time to first valid egress route, path length driven, approvals requested, fallbacks, wall time per
decision. Reported over seeds with variance (live runs are not bit-deterministic, §5.9); a loss for the LLM is
reported, not buried (invariant 7).

**D163 — the live wiring (2026-10-23).** `infra/scripts/live.sh run` with `ARIADNE_LIVE_CONFIG=configs/mission/live_llm_v1.yaml`
(live_v1 + `chooser: llm`, the D158 threshold 0.5, `llm.timeout_s` 120) starts `ariadne-orchestrator:1`
(`infra/docker/orchestrator.Dockerfile`: the console's base by digest, boto3/botocore 1.43.118, mcp 2.3.0) BEFORE the
runtime; the runtime connects to `<run>/chooser.sock`. **Timeout 120 s, not D156's 20 s:** `request_human_approval`
blocks up to 60 s (the operator's window) and a turn makes several model calls; 20 s would turn every approval into
a classical fallback. Credentials: the session's short-lived keys exported to a mode-600 env file, deleted once docker
has read it (aws.sh's Terraform pattern). Every run records its config beside it (`live_config.yaml`), so replay uses
the same chooser and gate. **Fixed before any real call:** mcp's stdio client passes only HOME/LOGNAME/PATH/SHELL/
TERM/USER to the server process (`DEFAULT_INHERITED_ENV_VARS`), which dropped `PYTHONPATH`; the servers now start from
`python/` with the orchestrator's interpreter. Tested without Bedrock (`python/tests/orchestration/
test_orchestrator_stub.py`, botocore Stubber, real MCP servers): inspect → approval → assign on Nova; throttled Nova →
DeepSeek with the text-only tool list; no assign → null answer and no fallback. 3/3.

**D164 — what the chooser sees beyond the candidates (2026-10-23, after the first real Bedrock run).** That run
(`maze_short` seed 6, 90 s, Nova 2 Lite, 18 replans, every pick valid, no fallback) showed two gaps in the request,
not in the model:
1. *No memory.* Each replan is a fresh conversation and the planner renumbers clusters, so agent 1 was sent north,
   south, west, east on successive replans and spent t = 20–50 s turning in place (0.5 m net). The request now carries
   per agent `current_task`: the goal its last command serves, since when, the chooser's reason, the straight-line
   distance left, and `candidate` = the option whose goal lies within `refused_radius_m` of it (the radius the
   refused-goal rule already uses; no new knob), else −1. A pick within that radius continues the task (`since`
   kept). The prompt says a switch costs a turn and must name its cause. **No hysteresis rule in C++:** commitment
   is the Phase 4 planner's (A1, CBBA bundles, arch §21.7); the classical chooser keeps its 1.8 behaviour.
2. *Directives invisible.* The operator drew seven `priority_region`s; the planner reweights only candidates INSIDE a
   region, none of agent 1's ever was, and the request had no directive field, so Nova could not act on them. The
   request now has the active directives, and each candidate the region directives covering its goal, the product of
   their priority weights and `to_directive_m` (distance to the nearest region for that agent; 0 inside), all
   computed by the core (`Directive::Distance`). The prompt ranks operator directives above the model's preferences.
The trace carries `current_task` (without the reason: replay has no answer text, and live = replay must hold) and the
per-candidate directive facts. Same run, also found and fixed: `CommandPath` replaced the whole path by its raw A*
cells when one simplified chord clipped an obstacle (45 waypoints, 85 s for 2.6 m); the console's streams died when
the operator endpoints and the websockets polled the telemetry tail concurrently (D154 fix).

## 3. Acceptance

1. A live two-agent run with `chooser: llm` in which the trace shows, per decision: vision facts → MCP tool calls →
   the choice → the command → the mover's answer; at least one fallback or approval event exercised for real.
2. Its replay reproduces the estimator outputs and the decisions (D152 check).
3. The same scene with `chooser: classical`, and the §D162 comparison table, whatever it shows.

## 4. Demo mode (D169): judges drive it live, so the loop must hold wall pace

Gate runs stay lockstep + replay: that substrate is untouched and keeps producing the quantitative evidence. Demo
mode is a separate live configuration (`mode: demo` in the live config) for the web endpoint, where the world runs
at `real_time_factor` 1 and judges interact in real time. It changes four things, nothing else:

1. **No lockstep.** The world never pauses. The mover keeps executing its current command until a newer one arrives
   (receding horizon), so a slow replan degrades to a longer leg of the old plan, never a freeze.
2. **Integration off the replan path.** Keyframes integrate into the planner's map on a background thread as they
   arrive, serialized with `Plan` by one planner mutex (A1: `Plan` is a pure function of the integrated map plus the
   `PlanContext`, so "the map at plan time" stays well-defined). The replan takes whatever is integrated when it
   fires; `keyframes_integrated` in results.yaml says how far behind it ran.
3. **Gate-only artifacts off.** The shadow TSDF and the per-replan slab/egress PNGs are skipped (`snapshots: false`);
   the console renders from telemetry, not files.
4. **The LLM is supervisory and non-blocking (maintainer direction, 2026-10-23).** Planning stays deterministic: at
   every replan the classical pick (A1 planner's own choice within the risk gate) is commanded IMMEDIATELY - there is
   no `ChooserLink::Wait` on the control path. The decision request still goes to the orchestrator with the same
   facts; its answer is collected non-blocking at the NEXT replan and, if it names a different candidate that still
   exists and passes the same validation (risk gate, approvals, reserve-return exemption), it is applied as an
   override for that horizon; otherwise it is recorded `expired`. Approval-gated options are never taken
   autonomously, exactly as in gate mode. The trace per decision still shows vision facts -> request -> answer ->
   (applied | expired | fallback) -> command -> mover answer, which is what the agentic claim rests on (invariant 3);
   `applied_at_s` names the horizon that acted on it.

Honesty: demo runs are not the evidence substrate (no determinism claim is made from them); results.yaml carries
`mode: demo`, and every number quoted in the submission keeps coming from lockstep + replay runs.

Acceptance (M2, g4dn EGL, renderer string must name the NVIDIA device): a 2-agent then 3-agent run at RTF 1 where
gz_stats holds >= 0.95x wall, no mover stall waits on a missing command, the per-decision trace is complete, and
frame handling keeps up (drops, if any, are counted in the link stats and said out loud).

Planner facts folded in (A1, 2026-10-23, read from phase4_planner.cpp, not memory): `Plan` is NOT pure - it carries
incremental egress state (D* Lite + visibility graph updated in place) - so demo keeps ONE Plan caller and never
reorders or reconstructs; "deterministic" means deterministic given the SEQUENCE of (map at plan time, context).
`Integrate` calls may interleave arbitrarily between Plans under the single mutex; per-agent keyframe order must stay
time order (track_ is the own-track polyline for the flown return route); cross-agent interleaving makes the demo map
not bit-reproducible across runs, which is accepted and is exactly why the gate stays lockstep. Replan-while-moving:
the mover's arrival rule and commanded-cell risk read the commanded cell, so a goal switch mid-leg is safe; the moving
START is handled runtime-side - the planned path is trimmed to the waypoint nearest the agent's LATEST estimated pose
before the command is sent (demo only), so a follow never backtracks to where the agent stood at plan time.


## 5. Hosted demo amendment — 2026-10-26

`LIVE_DEMO_CONSOLE.md` supersedes §4 items 3–4 for the hosted demo. Announced map
snapshots are now written, and a single nonblocking supervisory request remains
pending through inference/human review until its wall timeout. The next answer is
revalidated against current candidates; original approval IDs are retained across
candidate reindexing, with an exact approved-goal and non-increasing-risk check.
The controller rejects expired or superseded requests. Lockstep behavior is unchanged.
