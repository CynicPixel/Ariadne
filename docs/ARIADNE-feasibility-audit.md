# Ariadne — Positioning, Coverage & Competitive Strategy

**Project:** Collaborative visual reconstruction of a space that humans cannot safely enter, by autonomous perceptive mobile agents
**Application framing:** Post-disaster structural triage and responder egress mapping — earthquake-collapsed / structurally compromised buildings, confined voids, rubble
**Target:** OpenCV AI Competition 2026, powered by AWS ([brief](./opencv-ai-competition-2026.md))
**Revised:** 2026-08-29 · **Days to deadline:** 58 (Oct 26, 11:45pm PDT)

> **Revision note.** Two corrections from earlier drafts are folded in. First, "human adversarial" meant *an environment hostile to human presence*, not adversarial humans. Second, the setting is **generic post-disaster** — collapsed and unstable structures — **not specifically an active fire**. Fire, smoke, flame artifacts and heat shimmer are retained as an explicit **stretch goal** (§2.3), not the base case. Scope decisions below are the team's and are recorded as settled; this document keeps the risk analysis and drops the argument.

**Scope decisions — settled, not open:**

| Decision | Status |
|---|---|
| Build a simulator **and** source from a dataset | ✅ Both. §5.1 defines how they divide labour. |
| Write SLAM from scratch | ✅ From scratch. A **scoring asset** — see §4.2. |
| Reimplement cutting-edge algorithms from reference papers | ✅ Research running; lands in [`ALGORITHMS.md`](./ALGORITHMS.md). |
| GCNN over point clouds | ⚪ Likely dropped, superseded by the above. |
| Post-disaster triage narrative | ✅ Locked. Generic collapse/USAR, not fire-specific. |
| Fire, smoke, thermal artifacts | 🔵 **Stretch goal.** §2.3. |

---

## 1. Verdict

**Ariadne is positioned to win rather than merely to place**, and the narrowing from fire to generic collapse improves the project rather than costing it anything.

| Question | Verdict |
|---|---|
| Hard requirements met? | **Yes, deeply.** Degraded-visibility imaging keeps OpenCV 5 load-bearing at the pixel level, not only at the geometry level. |
| Can it win? | **Yes — credible first-place profile, and the strongest COOL candidate I can construct.** The row that was bleeding (Real-World Impact, 20%) is now a strength. |
| Buildable in 58 days? | **The team's call, reasonable given the stated resources.** The risk has changed shape rather than vanished — §7. |

**Why dropping fire was the right trade.** Active fire demands participating-media scattering, flame flicker, radiometric saturation, heat shimmer, and a thermal channel to be worth anything — a large modelling burden whose payoff is mostly visual. Collapse keeps every property that made the framing valuable — non-navigable by humans, hard budgets, degraded comms, forced autonomy, a named beneficiary — while replacing that burden with degradations that are **cheaper to simulate and algorithmically deeper** (§2.2).

---

## 2. The post-disaster framing

### 2.1 What it buys

**Constraints become physical rather than academic.**
- **Time budget** is the survival window — the "golden 72 hours" of urban search and rescue — plus platform endurance measured in tens of minutes.
- **Power budget** is battery in a space that cannot be re-entered to swap one.
- **Bandwidth budget** is RF propagation through reinforced concrete and rubble. This is a genuinely documented, unsolved problem in response robotics, not a modelling convenience — which is what makes bytes-exchanged a legitimate headline metric rather than a contrived one.

**Autonomy becomes a requirement, not a convenience.** The space is defined by human inability to enter it, and comms are degraded by premise, so teleoperation is not available. Agents *must* decide among themselves who observes what. The agentic layer is motivated by the problem rather than by the prize category — precisely what the Agentic Vision rubric selects for.

**The output has an obvious consumer.** Incident command needs: which volumes are structurally intact, where the voids are, where survivors may be, and **what the route in and out is**. That is a spatial digital twin with a named user, which is what both the Real-World Impact and UX rows ask for.

**There is a canonical precedent to anchor against.** The **DARPA Subterranean Challenge** was exactly this problem — multi-robot autonomous exploration and mapping of degraded, GPS-denied, comms-constrained underground and collapsed environments. It gives Ariadne published reference systems, documented hard lessons, public datasets, and instant credibility with a robotics-literate judge. Cite it explicitly and position Ariadne against it: SubT teams largely produced occupancy maps and object detections; Ariadne targets a **photorealistic reconstruction plus a navigable egress route under an explicit communication budget**. That is a clear, honest delta.

**Responsible use goes from liability to asset.** Multi-agent visual mapping carries an obvious dual-use surveillance edge, and the rules let judges reject entries creating *"surveillance risks without appropriate safeguards."* Disaster response is unambiguously beneficial and the report can say so plainly.

**It is culturally on-target for these judges.** The panel includes OpenCV's founder and Director of Content. Search-and-rescue robotics built on deep classical computer vision sits squarely in OpenCV's tradition.

### 2.2 The degradations — cheaper to simulate, deeper to solve

This is the substantive win from the change. Collapse-specific degradations are more tractable *and* more technically interesting than fire's.

| Degradation | Why it matters | Why it is good for you |
|---|---|---|
| **Darkness with active illumination** — power is out, agents carry their own lights | **The dominant condition.** Moving light sources violate the brightness-constancy assumption that direct VO and photometric bundle adjustment are built on. Illumination is coupled to pose. | The single most interesting problem in the project. Deep, well-posed, has real literature (photometric calibration, illumination-invariant representations, explicit light modelling), and is **trivial to simulate correctly** — it is just a light attached to the camera. Fire's participating media is neither. |
| **Texture-poor, self-similar geometry** — concrete, rubble, repeated structural elements | Causes **perceptual aliasing**: different places look identical. Poison for place recognition and inter-agent loop closure, which is exactly where a multi-agent system is most fragile. | A genuinely hard problem that your robust-estimation workstream (outlier-rejected loop closures, PCM, graduated non-convexity) exists to solve. The failure mode and the machinery that addresses it are in the same story. |
| **Dust and airborne particulates** | Mild scattering, reduced contrast. | Keeps the descattering workstream alive at a fraction of smoke's modelling cost — the physical model is the same, the density is forgiving. |
| **Confined spaces, narrow voids** | Close-range stereo, short baselines, wide-FOV optics, constrained motion. | Motivates careful calibration and geometry work; cheap to simulate. |
| **Structural instability** | Some volumes are unsafe to enter even for a robot. | Motivates the risk model behind the egress thread **and** the human-approval gate — one physical fact underwriting two rubric items. |
| **No GPS, degraded comms** | The premise. | Bandwidth-constrained consensus, justified. |

**The headline technical claim this enables:** *visual SLAM and multi-agent map consensus under self-carried illumination in texture-poor, GPS-denied, comms-constrained collapsed structures.* That is specific, hard, honest, and not a description of anyone's weekend hackathon project.

### 2.3 Fire as a stretch goal

Keep it scoped as a clearly-marked extension, attempted only once the base system is green:

- Smoke as denser participating media — the dust descattering model extended, not replaced
- Thermal/LWIR as a **victim-detection** channel (body heat), which is the stronger USAR motivation anyway, plus RGB-thermal registration
- Heat shimmer / atmospheric turbulence mitigation
- Flame radiometric saturation and flicker rejection

Framing it this way is strictly better than cutting it: the report gets to say *"the architecture extends to active-fire conditions; here is the additional sensing and restoration required, and here is a preliminary result"* — which reads as engineering judgment rather than as a missing feature. If time runs out, it is a documented roadmap item, not a hole.

### 2.4 Make the thread literal

The name is doing more work than it was hired for. Ariadne's thread is what leads someone **back out of the labyrinth alive** — which in a collapsed structure is not a metaphor. It is the deliverable, and it is the canonical USAR output: a route for rescuers to reach a location and return.

**Recommendation: make egress a first-class output alongside the splat.** A continuously-updated, risk-weighted route from any point in the explored volume back to an exit, with confidence attached, degrading gracefully as the map changes. This gives you:

- A **product**, not a demo. "The system produces a map" is weaker than "the system tells a responder which way to go."
- A **legible metric**: does a proposed route remain valid and traversable under ground truth? Time-to-first-valid-route and path validity communicate to a non-specialist judge in a way PSNR never will.
- The **video's closing shot**: agents disperse into darkness, the splat assembles, a glowing thread resolves through it.
- It naturally consumes the uncertainty machinery already being built — a route through poorly-observed volume is a route you should not recommend.

Cost is modest: a traversability cost field over the reconstruction plus a graph search. Payoff spans Real-World Impact, UX, and the video.

---

## 3. Requirements coverage

### 3.1 Hard gates

| Gate | Coverage | Assessment |
|---|---|---|
| **OpenCV 5 for substantive image/video analysis** | ✅ **Very strong** | Spans the full stack: low-light restoration and denoising, illumination compensation, dust descattering, feature extraction and matching on texture-poor surfaces, stereo depth at short baselines, from-scratch SLAM geometry, point-cloud merging. OpenCV 5 is the substrate of the entire perception path. |
| **A meaningful component on AWS** | ✅ **Strong** | Simulation fleet, per-agent perception workers, bandwidth-modelled message bus, splat training, orchestration, operator console. Lands all three Devpost themes: IoT, ML/AI, Serverless. |

### 3.2 Suggested project areas — five hits

- ✅ *"Active perception for autonomous inspection with agentic orchestration or MCP"*
- ✅ *"Multi-agent visual SLAM with distributed context"*
- ✅ *"Real-time spatial digital twins"*
- ✅ *"COOL-based server, serverless, container, or hybrid x86/Arm vision pipelines"*
- ✅ *"…safety, accessibility…"* — the disaster-response framing adds a fifth the abstract version lacked

### 3.3 Submission checklist

| Required artifact | Status | Notes |
|---|---|---|
| Technical report | ✅ | Responsible-use section is now easy and genuinely favourable (§2.1). |
| Judge-accessible repo (may be private) | ✅ | |
| Pinned deps + build/deploy/test instructions | ⚠️ | Arm + x86 split means two lockfiles, two images. Budget explicitly. |
| Architecture diagrams | ✅ | Need **three**: system, agent workflow (perception→decision→action), hybrid x86/Arm deployment. |
| Working endpoint **or** live screen-share | ✅ | The incident-command console (§6.2) satisfies this. |
| Video ≤ 5 min | ✅ **Standout** | See §2.4. |
| Evaluation evidence incl. failure cases | ✅ **Standout** | The ablation design produces it natively (§6.1). |

---

## 4. Technical depth — where the 30% lives

*"Technical execution — 30%: correctness and depth of the **OpenCV 5** implementation, architecture, reliability, and evaluation."* Largest single row, and the one most affected by the team's decisions.

### 4.1 OpenCV 5 module map

OpenCV 5's headline change is the `calib3d` teardown. Using the new structure is free evidence of genuine 5.x engagement — and these judges will notice the difference between OpenCV 5 code and OpenCV 4 code that happens to compile.

| Stage | Module | Notes |
|---|---|---|
| Restoration: denoise, low-light enhance, illumination/vignette compensation, dust descatter | `photo`, `xphoto`, `imgproc`, `ximgproc` | The DVE front-end. Also largely **COOL-accelerated op classes** — §5.3. |
| Undistort, resize/pyramids, adaptive threshold, contours | `imgproc` | COOL's advertised strengths. |
| Feature detect/describe/match | **`features`** | Renamed from `features2d` in 5.x. |
| Calibration, `solvePnP`/`solvePnPRansac` | **`calib`** | New in the 5.x split. |
| Essential/fundamental matrix, `recoverPose`, triangulation | **`geometry`** | New in the 5.x split. Core of cross-agent relative pose. |
| Disparity → depth, short-baseline stereo | **`stereo`** | Now top-level. |
| Point-cloud handling, merging, registration | **`ptcloud`** | **New top-level module in 5.x** — citing it is a clear "we are actually on 5" signal. |
| Optical flow, tracking | `video`, `optflow` | |
| Multimodal registration | `reg`, `rgbd` | Thermal is stretch-goal; keep the hook. |
| PSNR / SSIM on held-out views | `quality` | Run evaluation metrics through OpenCV 5 too. |
| Bundle adjustment | `sfm` | ⚠️ Likely absent from PyPI wheels (Ceres dep). Verify day 1; moot given the from-scratch backend. |

**Put this table in the report.** It converts a subjective 30%-weighted judgment into something verifiable at a glance.

### 4.2 From-scratch SLAM is a scoring asset

Implementing SLAM rather than integrating ORB-SLAM3 changes the Technical Execution calculus favourably. The criterion rewards *"correctness and depth of the OpenCV 5 implementation"* — a hand-built front-end and back-end on `features`/`geometry`/`calib`/`stereo` is the most direct possible evidence of exactly that. Integrating someone else's SLAM is, to this rubric, comparatively shallow however well it performs.

Two practices convert ambition into defensible evidence:

1. **Validate against a dataset oracle before trusting it in your own simulator.** From-scratch SLAM *and* a from-scratch sim means two systems that can be silently wrong at once, and disentangling a SLAM bug from a sim bug is the classic multi-week sink. Prove trajectory error against known ground truth on public data **first**, then take the validated estimator into the sim.
2. **Use GTSAM or g2o as a correctness oracle, not a dependency.** Solve the same pose graph both ways and report agreement. That is a *stronger* depth signal than either using the library or avoiding it — it shows the implementation is correct, not merely that it runs. Keep the oracle in the test suite, out of the runtime path.

### 4.3 Algorithm sourcing

Deep research is running across five domains; results land in [`ALGORITHMS.md`](./ALGORITHMS.md) with paper links, licenses, reimplementation difficulty, and OpenCV 5 mapping:

1. Visual SLAM & collaborative/multi-agent SLAM
2. 3D Gaussian Splatting frontier — SLAM-integrated, incremental, sparse-view, uncertainty-aware
3. Active perception, next-best-view, multi-robot task allocation
4. **Degraded visual environments** — retargeted to darkness under self-carried illumination, low-SNR denoising, texture-poor matching and perceptual aliasing, dust; plus DARPA SubT systems and USAR datasets. Fire/smoke/thermal demoted to a stretch section.
5. Distributed optimization, pose-graph consensus, robust estimation on manifolds

**Selection criterion:** prefer algorithms whose *mechanism* you can explain and defend over algorithms that merely benchmark well. A judge assessing depth will probe why a method works; depth that cannot be articulated does not score.

---

## 5. Architecture of the build

### 5.1 Simulator and dataset — division of labour

Running both is right, provided each has a defined job. Ambiguity about which produces which result is what costs time.

| Track | Job | Produces |
|---|---|---|
| **Dataset replay** (Replica / ScanNet / Matterport3D / TUM-RGBD / EuRoC / SubT data) | **Validation and controlled ablation.** Known ground-truth geometry and trajectories. All coordination regimes draw from an identical observation pool, so differences are attributable to policy, not simulation noise. | Trajectory error, registration error, quality-vs-budget curves, baseline comparisons — **the quantitative claims.** |
| **Simulator** (collapsed structure, darkness, dust, comms attenuation) | **The application.** Conditions no public dataset provides, with free ground truth. | The demo, the video, DVE algorithm evaluation, egress-path validation — **the narrative and impact claims.** |

Sequence matters: **validate on data, then deploy into sim** (§4.2).

For the simulator, prioritise fidelity in what actually affects the algorithms — **camera-mounted light sources with correct falloff and shadowing**, sensor noise at low light, short-baseline stereo geometry, comms attenuation and packet loss — over visual polish. A physically-plausible illumination model is worth more than a beautiful renderer, and it is the cheapest high-fidelity thing you can build.

### 5.2 Failure injection

The rules require reporting failure cases, and the Agentic rubric spends 15% on *"failure handling, observability, security, and human control."* Every failure mode here is realistic rather than synthetic:

- Comms partition and RF attenuation through rubble — **the premise, not an edge case**
- Agent loss mid-mission (entrapment, structural shift, battery)
- A degraded agent reporting bad poses after a hard impact
- **Perceptual aliasing producing a false inter-agent loop closure** — the highest-value failure case, because it is realistic, catastrophic, and your robust-estimation machinery exists precisely to catch it
- Total tracking loss in a featureless void — the case where the system must know it does not know

### 5.3 COOL — highest expected value

Verified: **Cloud Optimized OpenCV For AWS Graviton4**, sold by OpenCV. Ubuntu 24.04 Arm **AMI**, v3.1, built on OpenCV 5 with **Arm KleidiCV**. **7-day free trial**, then **$0.04/hr on m8g.4xlarge** plus EC2 infra. Vendor benchmarked **78 imgproc/core/I-O functions against the default OpenCV 5.0.0 build**.

Three things make this the most winnable prize on the board:

1. **Integration is launching an AMI**, not a porting project. Existing OpenCV 5 code runs as-is.
2. **The vendor hands you the rubric's required baseline**: identical code, identical Graviton4 instance, COOL vs. stock OpenCV 5.0.0. Arm-vs-Arm isolates the library rather than the silicon — far more defensible than the Arm-vs-x86 comparison most entrants will reach for.
3. **The restoration front-end is exactly COOL-shaped, and if anything more so than under the fire framing.** COOL accelerates pre/post-processing primitives — resize, adaptive threshold, contour detection, core ops. A low-light pipeline (denoise → illumination compensation → enhance → threshold → contour → pyramid → feature prep) is that class of work, it runs on **every frame from every agent**, and it is the highest-volume stage in the system. "COOL executes our core workload" becomes straightforwardly true rather than something to be argued.

**Benchmark story:** sweep agent count and report throughput and **cost-per-frame scaling** on Graviton, plus frames-per-instance-dollar. "How many agents can one instance serve, and at what cost" maps directly onto the rubric's *"latency, throughput, utilization, cost"* language and beats any single-op microbenchmark.

**Treat it as a discrete timeboxed workstream (~1 week), not something woven throughout.**

### 5.4 Agentic Vision

Qualifying bar: *"image or video results must influence a subsequent plan, tool call, action, or request for human approval."* Ariadne clears this structurally — the coverage/uncertainty estimate derived from OpenCV 5 output determines which agent moves where next.

**The trap, stated precisely:**
> *"Using an AI coding assistant to write the entry does not qualify as an agentic workflow."*

The team is using heavy AI assistance to build this. That is fine and irrelevant to the award — but the agentic claim must rest **exclusively on the runtime loop**, and the report and video must keep the two narratives strictly separate. Do not mention the development workflow anywhere near the agentic evidence.

**Design split:** classical estimators underneath (pose, depth, registration, uncertainty, coverage utility — deterministic and measurable); LLM/agent layer on top for task allocation, inter-agent negotiation, budget reasoning, exception handling, and **MCP tool invocation into OpenCV 5 operations**. MCP is named explicitly in both the agentic path and the suggested project areas — use it and say so.

**Human control is worth real points and the setting motivates it perfectly:** incident command retains authority over risk. *"Agent 3 requests permission to enter a structurally uncertain void"* is simultaneously a required evidence type, a UX feature, and a realistic operational protocol.

**Required evidence — instrument from week 1, do not retrofit:**
1. Agent workflow diagram (perception → decision/orchestration → action)
2. **A trace showing an OpenCV 5 output changing a later decision, tool call, or action** — the award in a single file
3. Evaluation of task success, failure handling, observability, human control

---

## 6. Where the points are

### 6.1 Evaluation design

Build the harness early — every chart depends on it, and it is what converts an ambitious build into a defensible one.

**Primary: reconstruction quality vs. budget spent.**
- **Quality:** PSNR / SSIM / LPIPS on held-out views (`quality`), geometric completeness, Chamfer distance to GT mesh
- **Budget:** captures, distance travelled, energy proxy, **and bytes exchanged**
- **Conditions:** solo · p2p consensus · orchestrator-worker
- **Baselines:** random view selection (table stakes) **and a naive greedy frontier/coverage heuristic**. Beating greedy is the actual claim; without that baseline a judge will ask whether the coordination layer earns its keep, and they will be right to.

**Second — mission-outcome metrics, more legible than photometrics.**
- Time-to-first-valid-egress-route
- Egress path validity against ground truth
- Structural coverage of the traversable volume
- Void/survivor-location detection rate

**Lead the report and video with these; keep photometric metrics for the technical section.**

**Third: DVE ablations.** Reconstruction and trajectory accuracy with and without each restoration stage, across illumination levels and dust densities. This proves the signal-processing work was necessary rather than decorative — the cheapest possible demonstration of depth.

**Fourth: COOL throughput and cost-per-frame vs. agent count.**

**Compute cost is not a constraint.** Room-scale 3DGS sweeps run roughly 3 regimes × 3 scenes × 3 seeds × ~30 min ≈ 14 GPU-hours ≈ **~$5 on g5.xlarge spot**. Ten full sweeps ≈ $50. The $150 grant plus Free Tier credits covers it comfortably.

### 6.2 The incident-command console

The remaining soft spot is **User Experience (10%)** — *"usability, accessibility, and quality of interaction."*

Build an **incident-command console**: live structural map with per-agent position and camera feeds (**raw vs. restored side by side — this sells the DVE work instantly**), coverage and uncertainty heatmap, budget gauges (battery/bandwidth/time), the inter-agent negotiation transcript, **the egress thread**, the growing splat, and a **human-approval control** for risk-gated actions.

Pays into five places at once:
1. UX row (10%)
2. *"A working web endpoint"* — a mandatory submission requirement
3. Agentic rubric's *"human control"* (part of 15%)
4. The qualifying condition's *"request for human approval"*
5. Real-World Impact — it makes the named user visible

Build once, bill five times. Highest ratio in the project.

### 6.3 Scoring projection

Heuristic, calibrated against the published weights.

| Criterion | Weight | Original abstract framing | **Current positioning** |
|---|---|---|---|
| Technical execution | 30 | 25–28 | **27–29** |
| Innovation | 20 | 15–17 | **17–19** |
| Real-world impact | 20 | 10–12 | **17–19** |
| User experience | 10 | 3–4 | **8–9** |
| Documentation & presentation | 10 | 7–9 | 8–9 |
| Cloud delivery & reproducibility | 10 | 6–8 | 8–9 |
| **Total** | **100** | ~68–76 | **~85–94** |

Unchanged by the fire→collapse narrowing: USAR is at least as compelling a beneficiary as firefighting, and the technical claim got sharper. The gain came from framing plus one console — not from additional systems work.

---

## 7. Risk register

Scope is the team's call. These are the failure modes that remain regardless of build capacity.

| # | Risk | Sev | Mitigation |
|---|---|---|---|
| R1 | **Two-systems ambiguity** — from-scratch SLAM debugged against a from-scratch sim; a bug in either looks like a bug in the other | 🔴 High | **Sequence it: validate SLAM on public datasets with known GT first, then move into the sim (§4.2/§5.1).** Keep the dataset regression suite green throughout. Highest-value single practice in the plan. |
| R2 | **Integration debt** — many strong subsystems, assembled late | 🔴 High | Build the thinnest end-to-end spine early (one agent → one splat) and keep it green. Every later capability lands on a working system. The failure mode is not "too ambitious," it is "integrated in the last fortnight." |
| R3 | Evidence artifacts retrofitted late | 🟠 Med | Decision trace, COOL benchmark pack, DVE ablations, failure cases are **required deliverables**. Instrument week 1. |
| R4 | Restoration degrades the geometry it was meant to help | 🟠 Med | Ablate. Measure SLAM accuracy with/without each stage. A negative result honestly reported scores better than an unmeasured pipeline. |
| R5 | Sim realism gap — algorithms tuned to synthetic darkness/dust fail on anything real | 🟠 Med | Validate restoration on **real** low-light and dark-environment imagery where public data exists (research agent 4 is sourcing SubT and USAR datasets). Even a small real-data validation set defuses the obvious judge question. **Lower now than under the fire framing** — synthetic illumination is far easier to make physically faithful than synthetic smoke. |
| R6 | Perceptual aliasing breaks inter-agent loop closure | 🟠 Med | This is a **planned failure case** (§5.2), not a surprise. Robust back-end (PCM, GNC, max-clique inlier selection) is the mitigation and the story. |
| R7 | LLM layer decorative | 🟠 Med | Give it decisions that change outcomes; log every decision with its vision-derived inputs; **include an ablation with the LLM layer removed**. Report honestly if classical planning wins. |
| R8 | From-scratch backend subtly incorrect | 🟠 Med | GTSAM/g2o as test-suite oracle (§4.2); convergence tests on synthetic graphs with known optima. |
| R9 | Two-architecture repro burden (Arm + x86) | 🟡 Low-Med | Two lockfiles, two images, one Makefile, IaC from the COOL week. |
| R10 | `sfm`/`viz` missing from OpenCV 5 wheels | 🟡 Low | Day-1 smoke test. Largely moot given the from-scratch backend. |
| R11 | COOL claim doesn't survive scrutiny | 🟡 Low | Front-end is genuinely COOL-shaped (§5.3); Arm-vs-Arm baseline; publish method, inputs, instance config. |
| R12 | 3DGS compute blowout | 🟢 Low | ~$5/sweep. |
| R13 | Responsible-use objection | 🟢 Low | Largely defused by disaster-response framing. Still address: no person re-identification, scope/consent, deployment authority. |
| R14 | **Stretch-goal creep** — fire work started before the base system is green | 🟢 Low | §2.3 is gated: fire is attempted only after the collapse pipeline is end-to-end and evaluated. Documented as a roadmap item costs nothing; a half-built smoke model costs weeks. |

---

## 8. Competitive assessment

**Field-size heuristic (my estimate, not from the competition page):** 776 registrants; typical Devpost registration-to-submission conversion runs ~10%. Expect **50–80 eligible final entries**, most single-model demos.

| Target | Assessment |
|---|---|
| **Best Use of COOL** ($1,000) | **Strongest odds.** 30% of that rubric is *verified COOL integration on Graviton* — mechanical, checkable, and most teams won't spend the week. The AMI plus free trial makes it cheap, and the restoration front-end is genuinely the workload COOL accelerates. The rules note the award **may go unawarded** if nobody qualifies — the organizers expect a thin field. |
| **Agentic Vision** ($1,000) | **Strong.** Thicker field, but the *"visual evidence must change what the system does next"* bar disqualifies most entries. Ariadne clears it structurally. Keep runtime and development narratives strictly separate (§5.4). |
| **Overall top-3** ($2,000–$5,000) | **Credible first-place profile.** Technical depth is top-decile; the application is one few judges would not want to see funded; the video will be the best in the field. |

Chasing all three is correct — one Overall plus both Specials is permitted, and **both Special Award Consideration boxes must be ticked at submission**.

**Ignore public voting.** It appears in the schedule but **no prize in the rules attaches to it.**

**Where this loses, if it loses:** not on ambition and not on ideas — on arriving with six excellent subsystems that met each other in the last week, and a report written in a hurry. R1, R2, and R3 are the whole game.

---

## 9. Immediate actions

- **A1.** Register the team on Devpost. Join the OpenCV Slack. Adopt `#OpenCVComp26`.
- **A2. Submit the grant proposal — the Jotform is open and accepting** (verified live, PDF upload). Include all nine required elements; the **team bio is explicitly weighted**. Declare **both** focus paths. Non-selection costs nothing, and submitting moots any question about whether a proposal gates final submission.
- **A3.** Email the hackathon manager to confirm the proposal is not a submission gate. Five minutes.
- **A4. OpenCV 5 smoke test.** `pip install opencv-contrib-python==5.0.0.93`; inspect `cv2.getBuildInformation()` and `dir(cv2)`; mark each §4.1 module wheel-verified or needs-source-build.
- **A5. COOL day-1 check.** Launch the Graviton4 AMI on the free trial, run stock OpenCV 5 code unmodified, confirm drop-in behaviour, take a first microbenchmark, confirm the license permits publishing results.
- **A6.** Choose datasets and lock the simulator's fidelity targets — **camera-mounted illumination first** (§5.1).
- **A7.** Stand up the end-to-end spine and the evaluation harness before adding capability (R2).

**Submit on Oct 24, not Oct 26.** The platform deadline reads 11:45pm PDT while the description says 11:59pm — do not let a fourteen-minute ambiguity end a two-month project.

---

## 10. Bottom line

The post-disaster framing does more for this submission than additional engineering could, and narrowing from fire to generic collapse keeps all of that while removing the one workstream whose cost was mostly cosmetic.

What remains is a sharper technical claim than the fire version supported: **visual SLAM and multi-agent map consensus under self-carried illumination, in texture-poor, GPS-denied, comms-constrained collapsed structures, producing a reconstruction and a navigable egress route under an explicit communication budget.**

Three things decide the outcome, and none is scope:

1. **Sequence the validation.** Datasets prove the estimator; the sim proves the application. Getting that order wrong is the one mistake that could still cost weeks (R1).
2. **Integrate early and continuously.** Six excellent subsystems meeting in the final week score worse than five running together since September (R2).
3. **Treat evidence artifacts as deliverables from day one** — decision trace, COOL benchmark pack, DVE ablations, failure cases. Each is explicitly required by a published rubric and nearly impossible to reconstruct at the end (R3).

Do those three, and the open question is not whether Ariadne places — it is how many of the three prizes it takes.
