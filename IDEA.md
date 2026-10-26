# CLEARWAY

**A robot walks into a dark, half-collapsed building. It comes back with a route a human can
actually take — and it shows its work.**

CLEARWAY drives a single ground robot through an unlit, unmapped, structurally compromised interior
and returns the one artefact an incident commander can act on: a **responder-grade egress corridor**.
Not "a map." A specific route, validated at the clearance a person in breathing apparatus needs while
dragging someone out, with every risk judgment along it surfaced, justified, sourced, and
individually approvable or rejectable by the human in charge.

**The technical thesis, one level down.** A frontier reasoning model is too slow and too expensive to
sit inside a 30 Hz robot control loop, and too capable to leave out of the decisions that actually
matter. CLEARWAY makes cloud reasoning *load-bearing* rather than decorative by **gating it on value
of information**: a small physical-reasoning model triages every keyframe on the robot, and a gate
decides — per observation — whether escalating to a large cloud model would change the recommended
route enough to be worth its latency, its tokens, and its bytes over a degraded radio link. We build
that gate and we measure the frontier it buys.

---

## 0. Target and constraints

**Nebius x NVIDIA Global AI Hackathon — Physical AI Track.** Requirements verified against the
Devpost page and official rules on 2026-09-09:

- **Deadline: Fri Oct 30, 2026, 10:00 am PDT** — a morning cutoff, not end of day.
- **Stage-One pass/fail gate:** the project must *run on Nebius Token Factory or Nebius AI Cloud* —
  defined as "makes a runtime call to the Token Factory inference API, or is deployed/run using
  Nebius AI Cloud compute (Serverless Jobs, Serverless Endpoints, or DevPods)" — **and** use at least
  one NVIDIA open-source model, **and** be a genuine attempt at the track's stated goal, "not a
  superficial rebrand of an unrelated idea."
- **Physical AI Track, verbatim:** *"Build embodied and edge agents that sense and act in the real
  world: robotics, IoT, and on-device intelligence, powered by Nemotron, GROOT, Cosmos and Sonic
  models and coordinated through an agent runtime. Use Nebius Serverless Jobs to run simulations,
  generate synthetic data, evaluate robot policies, or process sensor data at scale, and Nebius
  Serverless Endpoints to serve real-time inference when needed. Your demonstration video must
  include at least a 1-minute clip showing the physical hardware/robot actually operating, or — if
  your project has no physical hardware component — the key application modules in action."*
- **Judging — four equally weighted criteria:** Technological Implementation · **Design** ("a
  complete, coherent product experience *not just a technical proof of concept*") · **Potential
  Impact** ("a credible, specific case for solving a real problem for a real audience *based on what's
  demonstrated*") · Quality of the Idea.
- **Prizes:** Grand $20,000 / 2nd $10,000 / 3rd $6,000; Physical AI Track Winner = Jetson Orin Nano;
  Best Use of Tavily = $3,000. One project may take **one Overall award OR one Track award plus one
  Bonus award** — so Physical AI + Tavily is a legitimate paired target.
- **Also required:** public repository with an OSS licence visible at the top, README with setup and
  run instructions, and a ≤3-minute public YouTube video whose audio covers how Nebius and the NVIDIA
  models were used. A hosted-demo URL is *not* required for Physical AI.

**Team constraints:** no physical hardware of any kind and no local RTX GPU — **everything runs on
Nebius**. Two to three builders. ~7.3 weeks from 2026-09-09.

Half the rubric is explicitly hostile to a bare systems-research demo. So this is built and presented
as a *product with a named user*, whose technical depth happens to be a genuinely hard systems
problem.

---

## 1. The problem, and the specific person who has it

Urban search and rescue inside a compromised structure is gated on one question: **can we send people
in there, and by which way?** Today that answer comes from sending a human to look. Existing robot
answers are occupancy maps and object detections — they tell a commander what is *there*, not what is
*survivable to walk through*.

The gap is not perception. It is **consequence**. A void with 0.6 m of clearance is geometrically
passable and might still be the worst place in the building to send a two-person team, because the
column beside it is cracked and carrying load. Geometry cannot see that. A fixed-vocabulary detector
cannot see it either — it reports "column, crack" in the safe case and the dangerous case alike.

CLEARWAY's output is aimed at that gap: a route, a per-segment risk with a written justification, and
a control that lets the commander refuse it.

### The design parameter that makes it a product, not a demo

Routes are planned and validated at **ρ_responder = 0.40 m**, not at the robot's own radius.

Justification, stated as a chosen parameter rather than a magic constant: a firefighter in SCBA is
roughly 0.60 m across the shoulders; gear plus a drag-rescue allowance gives an ~0.80 m corridor, so
a 0.40 m planning radius. **An agent-passable void is not a human route**, and every metric in this
project is computed at the responder radius. ρ_responder is a config value, swept in the appendix
(0.30 / 0.40 / 0.50 m) so the sensitivity is on the record.

---

## 2. Why the language model cannot be removed

Three couplings into the closed loop. Each is independently measurable, and deleting the model
measurably breaks each one.

**2.1 Context-dependent risk.** Detection yields "concrete column, crack." The *risk* depends on
whether that column is load-bearing and currently under load — an inference about physical
consequence over the surrounding scene, not a property of the detected object. Identical object
inventories imply different correct routes. This is precisely the capability Cosmos Reason exists to
provide, and precisely what a per-class cost table cannot encode. The model's verdict enters the
planner's cost field; changing the verdict changes the route.

**2.2 Active re-inspection.** A low-confidence verdict emits a **look-again** action: the robot
re-approaches the region, re-aims its lamp, and captures a multi-view bundle for a second opinion.
Vision → judgment → *robot motion* → new vision. Delete the model and the robot has nothing to be
uncertain about, so it never re-inspects and never changes course. This is the closed loop, and it is
visible in the video.

**2.3 Retrieval-grounded keep-out geometry.** A hazmat placard is four digits on a wall. Turning
"UN1075" into *a number of metres of isolation distance* requires retrieving an authority document,
reading it, and resolving it to a scalar — and that scalar becomes a hard obstacle in the cost map.
Retrieval materially changes the robot's geometry. See Section 7.

### The ablation that decides whether any of this is defensible

The sharpest objection a robotics-literate judge will raise is *not* "a detector could do it." It is:
**"a fixed hazard vocabulary plus a hand-written per-class cost table gets you 90% of this — why is a
reasoning model in the loop?"** So that configuration is built as a first-class baseline (**A′**,
Section 4), in the POC, not the MVP. If A′ ties the full system, we learn it in week 3 and re-centre —
not in the submission.

Beating A′ requires scenes where **the same detected objects imply different risk**. Authoring those
is therefore a core deliverable, not scene dressing (Section 6.2).

---

## 3. Architecture

Three tiers, separated by clock rate. The separation *is* the design: each tier may touch only what
it can touch at its own latency.

```
                          ┌──────────────────── NEBIUS ────────────────────┐
                          │                                                │
  Isaac Sim (L40S)        │  Serverless Endpoint      Token Factory        │
  headless render         │  Cosmos-Reason2-8B        Nemotron 3 Super     │
        │                 │  (vLLM, multi-view        (orchestrator,       │
        │  clean MCAP     │   visual verdict)          tool use, schema)   │
        v                 │         ▲                      ▲              │
  seeded degradation      │         └──────────┬───────────┘              │
  operator (consume-time) │                    │ T2: only when gate fires │
        │                 └────────────────────┼──────────────────────────┘
        v                                      │  (link model: delay/loss/cap)
  ┌─ T0 REFLEX ~30 Hz ─────────┐               │
  │ stereo+IMU VIO             │        ┌──────┴───────┐
  │ TSDF → ESDF                │        │ VoI GATE     │
  │ local planner              │        │ Δrisk × cons │
  │ safety envelope            │        │ ÷ pred. cost │
  └────────────┬───────────────┘        └──────▲───────┘
               │ keyframes                     │
               v                               │
  ┌─ T1 TRIAGE ~0.5 Hz ────────┐               │
  │ Cosmos-Reason2-2B          ├───────────────┘
  │ open-vocab hazard          │  hypotheses + confidence
  │ hypotheses + escalate score│
  └────────────┬───────────────┘
               v
      RiskVerdict → risk field ⊕ ESDF → planner @ ρ_responder = 0.40 m
               │
               ├──► recommended route + per-segment justification
               ├──► look-again goal back into T0's queue
               └──► operator console: accept / reject / hold
```

**T0 — Reflex (robot process, ~30 Hz, no model).** Stereo-visual-inertial odometry, TSDF → ESDF,
local planner, safety envelope. **This tier alone keeps the robot alive.** Every tier above it only
changes *where it chooses to go* and *what it tells the human*. Fully deterministic and replayable.
Built on proven components — NVIDIA cuVSLAM / Isaac ROS Visual SLAM as first choice, since it is
NVIDIA-native and Jetson-deployable. We are not writing a SLAM system from scratch; none of this
project's claims live in the estimator.

**T1 — Triage (edge-class, ~0.5 Hz on keyframes).** **Cosmos-Reason2-2B**. Per keyframe it emits
open-vocabulary hazard hypotheses, each with a class, a support region, a calibrated confidence, and
an *escalate-worthiness* score. Runs in a VRAM- and latency-capped container sized to an Orin-class
envelope.

> `[PROXY]` **Stated plainly in the README and the video:** there is no Jetson in this build. T1 is an
> edge-*class* model under an enforced edge-*class* budget, and we report its measured latency and
> VRAM against Orin Nano's published envelope. We claim deployability, not on-device execution. NVIDIA
> publishes a Jetson deployment path for Cosmos-Reason2-2B, which is why this is the model chosen.

**T2 is asynchronous and non-blocking by construction.** The robot never waits on the cloud: it keeps
driving under T0's safety envelope while an escalation is in flight, and a returning verdict revises
the risk field and the recommended route rather than releasing a stalled controller. This is why the
tier split is an *architectural* necessity rather than a cost optimisation — it is the only structure
in which a multi-second cloud model can participate in a real-time loop at all.

**T2 — Deliberation (Nebius; invoked only when the gate fires).**

- **Cosmos-Reason2-8B** on a **Nebius Serverless Endpoint** (vLLM container on L40S). Input: 3–5
  views of the same region plus a compact geometry summary. Output: *is this passable for a
  responder, and why* — physical-consequence reasoning, which is the model's design purpose.
- **Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b`, verified live on Token Factory;
  OpenAI-compatible at `https://api.tokenfactory.us-central1.nebius.com/v1/`) as the
  **orchestrator**: owns tool use, fuses the visual verdict with retrieval and geometry into a
  structured `RiskVerdict`, composes the operator escalation packet, and compiles the commander's
  natural-language standing orders into a machine-checkable mission constraint set (Section 8).
- **Nemotron 3 Nano** for high-rate cheap calls (schema repair, summarisation) so credits stretch.
  `[UNVERIFIED]` — exact Nano and Ultra model IDs to be confirmed against the Token Factory model
  list at signup; only the Super ID above is verified.

**The gate — the actual technical contribution.** For each candidate escalation:

```
score = (expected Δ route-risk if the verdict flips) × (consequence at this location)
        ─────────────────────────────────────────────────────────────────────────────
                    predicted cost (tokens + measured p50 latency + bytes)
```

Consequence is not uniform: a segment *on* the currently recommended route, or one gating the only
known egress, is worth far more than a segment in a side void. Escalate above threshold. **The
threshold is swept and reported as a curve**, never tuned to one lucky value. Token cost is
predictable; end-to-end cloud latency is not — so cost-prediction error is reported as a *measured
result*, not promised as a contract.

**Safety invariant, enforced in code and not by convention:**

```
the deterministic planner computes the candidate action set and the safety envelope;
the models may only CHOOSE AMONG presented candidates, ANNOTATE risk, and COMPOSE escalations.
```

An out-of-schema or out-of-set model response falls back to the deterministic choice and **logs the
event**. Those log lines are failure-handling evidence — they show the guardrail firing rather than
asserting that one exists.

---

## 4. Evaluation — the money table

| ID | Configuration | What it tests |
|---|---|---|
| **A** | Geometry only — ESDF + clearance, no semantics | Does semantics matter at all? |
| **A′** | Closed-vocab detector + static per-class cost table | **Does *reasoning* matter, or just detection?** |
| **B** | T1 always, never escalate | Is the small model enough on its own? |
| **C** | Escalate every keyframe to T2 | Quality ceiling; cost floor |
| **D** | **VoI-gated — CLEARWAY** | Does the gate recover C's safety at a fraction of C's cost? |

**Metrics**, all computed at ρ_responder:

- **False-clear rate — primary, safety-critical.** Fraction of route segments marked safe that ground
  truth says are hazardous. This is the number that matters and the number we lead with.
- Route hazard exposure: integral of ground-truth risk along the recommended route.
- Time-to-first-valid-route.
- Route validity against ground-truth geometry at ρ_responder.
- Cloud tokens per mission-minute; escalations per minute; escalation latency p50/p95; bytes over the
  link model.
- **T1 deployment envelope — a result, not a caveat.** Measured T1 p50/p95 latency and peak VRAM,
  reported against NVIDIA Orin Nano's published envelope. This is the number that turns "we had no
  Jetson" into "here is the edge budget the triage tier actually fits in," and it belongs in the
  results table rather than a README footnote.
- Human-approval precision: of escalations shown to the operator, what fraction were genuine
  decisions rather than noise.

**Target claim shape:** *D matches C's false-clear rate within noise while spending a small fraction
of C's tokens and latency, and strictly dominates A, A′ and B.* Plotted as a Pareto frontier of
safety against cloud spend, with the gate threshold as the swept parameter.

**Negative results are reported.** If A′ ties D, that goes in the README and the design re-centres on
the escalation and approval reasoning instead of the risk field.

---

## 5. Determinism — what makes the comparison a comparison

Two mechanisms, both cheap, both non-negotiable.

1. **Clean render once, degrade at consumption.** Isaac Sim renders clean frames into an MCAP bag.
   *All* sensor degradation — low light, photon and read noise, motion blur, dust scattering,
   exposure — is applied by a **seeded operator at consumption time**. GPU rendering is not a
   reproducible function; a seeded post-process is. Every configuration in Section 4 therefore
   consumes **identical pixels**.
2. **Cache every model response**, keyed by `(prompt_hash, image_hash, model_id, seed)`. Ablations
   replay from cache, so configurations that overlap consume identical *model outputs* too. This is
   what converts "we ran it four times" into a controlled comparison, and it keeps the sweep
   affordable.

Every run writes a **manifest**: git commit, container digests, scene and seed, config hash, model
IDs, gate threshold, ρ_responder. No figure without its manifest, config, seed, command, and raw
metrics.

**Honesty tags** are used in code, README and video: `[UNVERIFIED]`, `[ASSUMPTION]`, `[PROXY]`.
Nothing is claimed as measured that was projected.

---

## 6. Simulation

### 6.1 Stack, and the risk that comes with it

**Isaac Sim 5**, headless, on **Nebius L40S**. This is the single largest schedule risk in the
project and it is named up front:

- Isaac Sim **requires RT cores**. H100 / H200 / B200 do not have them and are explicitly unsupported
  for Isaac Sim rendering. **L40S is the only viable Nebius GPU** for this — available on Nebius
  Compute from ~$1.55–1.82/hr.
- `[UNVERIFIED]` L40S availability as a **Serverless AI** GPU option; Serverless Jobs and Endpoints
  are in **public preview**. L40S is verified only for Nebius *Compute*.

**Day 1–2 go/no-go probe:** launch the `nvcr.io/nvidia/isaac-sim` container on Nebius — a Serverless
Job if L40S is offered there, an L40S Compute VM otherwise — render one frame headless with EGL,
print its hash. Timebox: 2 days.

**Fallback if it fails:** author scenes in a lighter renderer and keep the **evaluation sweeps** on
Serverless Jobs over recorded bags, since replay needs no RT cores. This matters less than it looks,
because **the Nebius eligibility requirement is already satisfied independently** by the Token
Factory runtime call *and* the Cosmos Reason Serverless Endpoint. Isaac-on-Nebius is track alignment,
not a gate.

### 6.2 The discriminating scene pairs — core authoring deliverable

Four pairs. Each pair has an **identical detected-object inventory** and a **different correct
responder route**. A detector sees the same scene twice; a physical reasoner should not.

| # | Left (safe route exists) | Right (route must divert) |
|---|---|---|
| 1 | Cracked column lying as debris beside an intact column | Cracked column **bearing load**, ceiling deflection visible above |
| 2 | Standing water in an open bay | Standing water adjacent to a **severed conduit** |
| 3 | Beam lying on the floor | Same beam **spanning and supporting** a slab |
| 4 | Wide void behind an intact support | Same void behind a **compromised** support |

These pairs are where the entire defensibility argument is settled. They are built first.

### 6.3 Scene content

Partially collapsed two-storey interior, unlit, airborne dust, camera-mounted spot lamp with correct
falloff and shadowing. Fidelity is spent on **what affects the algorithms** — camera-mounted
illumination, short-baseline stereo geometry, low-light sensor noise — not on visual polish. A
physically plausible illumination model is worth more than a beautiful renderer.

**Nebius Serverless Jobs** run the scenario sweeps and ablation replays at scale; **Nebius Serverless
Endpoints** serve Cosmos Reason. This is exactly the division the track text asks for.

---

## 7. Tavily — retrieval that changes the robot's geometry

Robot reads a hazmat placard → **Tavily** retrieves the emergency-response isolation guidance for
that UN number → the isolation distance becomes a **hard keep-out radius in the planner's cost map**.

This is a rare kind of retrieval integration: the web result is never shown to a human, it is
*converted into metres of obstacle*. Removing it changes where the robot will drive.

Retrievals are **cached to a local hazard pack pre-mission**, and the cache is what runs at mission
time — a robot in a basement has no internet, so the offline cache is the honest architecture rather
than a convenience. Timeboxed to ~1 day. Targets the $3,000 Best Use of Tavily bonus, which is
compatible with a Track award.

---

## 8. Operator console — not optional scope

Half the rubric is Design and Potential Impact, judged on *what is demonstrated*. The console is the
demonstration surface, so it is promoted into the POC.

Live structural map with robot pose · raw vs. degraded camera side by side · the risk field · **the
recommended responder route with per-segment risk and its written justification** · budget gauges
(tokens, latency, bytes) · the escalation feed · **accept / reject / hold on every risk-gated entry**.

It pays into four things at once: the Design score, the human-control evidence, the "key application
modules in action" video requirement, and making the named user visible for Potential Impact.

**Standing orders.** The commander types constraints in natural language — *"do not enter anything
with standing water; prioritise voids near the collapse face"* — and Nemotron compiles them into a
machine-checkable mission constraint set the planner consumes. The commander sets the risk policy in
their own words; the planner enforces it deterministically.

---

## 9. POC / MVP boundary

### POC — end of week 3. Done = the loop closes and A′ is answered.

- Scene pair #1 authored and rendering
- T0 on replayed bags: VIO + ESDF + planner at ρ_responder
- T1 and T2 wired; gate stubbed to always-escalate
- `RiskVerdict` → risk field → route
- **A, A′, B, C measurable on pair #1**
- Console v0: map, route, escalation feed, approve/reject
- Decision-trace JSONL + run manifests
- Response cache operational

**Decision rule — written now, before there are results to rationalise against.** On scene pair #1,
across 3 seeds, **D's false-clear rate must be at least 20 percentage points below A′'s, with
non-overlapping per-seed ranges.** If the gap is smaller, or the ranges overlap, the reasoning tier is
not earning its place in the risk field and we re-centre in week 4: the models keep the escalation,
justification and human-approval role (Sections 2.2 and 8), the risk field reverts to a
detector-plus-cost-table baseline, and the README says so plainly. The 20-point bar is deliberately
high — the pairs are *designed* to be discriminating, so a narrow margin means the pairs failed, the
premise failed, or both.

### MVP — end of week 6; week 7 is buffer, video and README.

- VoI gate + threshold sweep → the Pareto plot
- **4 scene pairs in total** (not 4 per interior), distributed across 2 interiors — pairs #1–2 in the
  first, #3–4 in the second — each run at 3 seeds, swept on Serverless Jobs. Scene authoring is the
  deliverable most likely to overrun week 5; the count is capped here deliberately.
- Full A–D table
- Tavily keep-out radius
- Standing-orders compilation
- Console v1, ≤3-min video, public Apache-2.0 repo with run instructions

### Explicitly out of scope, and why

| Cut | Why |
|---|---|
| Multi-robot collaboration | Multiplies every subsystem; adds nothing to the LLM-necessity argument |
| Gaussian-splat reconstruction | Expensive and photometric; the product is a route, not a render |
| From-scratch SLAM | No claim of this project lives in the estimator |
| **GR00T** | A manipulation VLA for humanoids. There is nothing here to manipulate — forcing it in would be exactly the cosmetic use we are arguing against |
| **Cosmos world-generation for synthetic data** | A whole workstream; the scene pairs matter more than scene volume |
| Audio / speech models | No organic role in this loop |

Naming a sponsor model we deliberately did *not* use, with the reason, is stronger than using it
badly.

---

## 10. Component justification — what each piece buys

| Component | Capability it enables | Removable? |
|---|---|---|
| **Cosmos-Reason2-2B** (T1) | Open-vocab hazard triage at edge latency; produces the *uncertainty* the gate consumes | No — without it there is nothing to gate |
| **Cosmos-Reason2-8B** on Serverless Endpoint (T2) | Multi-view physical-consequence reasoning: is this passable *for a person* | No — this is what beats A′ |
| **Nemotron 3 Super** on Token Factory | Tool use, verdict fusion, escalation packets, NL→constraint compilation | No — it is the agent runtime |
| **Nemotron 3 Nano** | Cheap high-rate schema repair and summarisation; keeps credits alive | Yes, but costly to remove |
| **Nebius Serverless Endpoints** | Self-hosted Cosmos Reason with controlled latency; the "real-time inference" the track asks for | No |
| **Nebius Serverless Jobs** | Scenario sweeps and ablation replays at scale — the ablation table is impossible by hand | No |
| **Isaac Sim on Nebius L40S** | Ground truth for hazards and geometry; RT-core rendering of dark scenes with a camera-mounted lamp | Degradable — see Section 6.1 fallback |
| **Tavily** | Converts a placard into metres of keep-out geometry | Yes — bonus-prize scope, timeboxed |

---

## 11. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Isaac Sim will not run headless on an available Nebius GPU | Day-1/2 probe; Section 6.1 fallback; Nebius eligibility already met independently |
| R2 | **A′ ties D — reasoning adds nothing over a cost table** | Scene pairs built first; decided by the week-3 rule, not at submission |
| R3 | Serverless Jobs/Endpoints are public preview; quota or region gaps | Probe both in week 1; Compute-VM fallback for the endpoint |
| R4 | Escalation latency makes the loop feel unusable on video | The gate exists for this; T2 is non-blocking and the robot continues under T0 while it is in flight |
| R5 | Credit burn on sweeps | Response cache + Nano for cheap calls + 2 interiors rather than 3 |
| R6 | Six good subsystems that meet each other in week 6 | The week-3 POC gate requires the *whole* loop end-to-end before anything is deepened |
| R7 | No hardware footage → Physical AI judges discount it | Rules explicitly permit "key application modules in action"; console + sim footage carries it, and the `[PROXY]` framing is stated honestly rather than hidden, with the T1 envelope reported as a measured result |

---

## 12. Sequencing for 2–3 builders

**Week 0 — immediate, blocking.** Nebius signup + Builder Program credits (Token Factory, Tavily);
register on Devpost; create the public Apache-2.0 repository.

**Week 1.** Three probes in parallel — (a) Token Factory Nemotron call end-to-end, (b) Cosmos-Reason2
on a Serverless Endpoint, (c) Isaac Sim headless on L40S. Plus scene pair #1 and the replay/manifest
substrate.

**Weeks 2–3.** POC to the gate in Section 9.

**Weeks 4–6.** MVP — gate, sweeps, remaining pairs, Tavily, console v1.

**Week 7.** Video, README, submission. **Submit Oct 28, not Oct 30 — the deadline is 10:00 am PDT, a
morning cutoff, not end of day.** Treating Oct 30 as "we have all of Thursday" loses the project.

**Suggested split.** **(A)** edge stack — T0, degradation operator, replay, manifests · **(B)** cloud
reasoning — T1/T2, gate, schemas, caching, Tavily · **(C, or shared)** Isaac scenes, evaluation
harness, console, video.
