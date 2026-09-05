# OpenCV AI Competition 2026, powered by AWS

**Source:** https://opencv26.devpost.com/ (Devpost challenge ID `30984`)
**Captured:** 2026-08-28 — pages read: Overview, Rules, Resources, Schedule, Updates, Project Gallery, Discussions.
**Tagline:** Build vision systems that see, reason, and act with OpenCV 5 and Amazon Web Services (AWS).

**Status as of capture:** Registration open; submission/build phase in progress (opened Aug 26). 776 participants. Project gallery not yet published; no discussion topics posted; Updates page empty.

- **Sponsor:** Amazon Web Services ("Amazon" / "Sponsor")
- **Administrator:** OpenCV Foundation / The Open Source Vision Foundation ("OpenCV")
- **Format:** Online, public. Themes: OpenCV, IoT, Machine Learning/AI, Serverless
- **Total prizes:** $12,000 cash
- **Register:** https://opencv26.devpost.com/register?flow[data][challenge_id]=30984&flow[name]=register_for_challenge

---

## 1. Key dates

| Milestone | Window |
|---|---|
| Competition launch + registration + proposal submission | From **Aug 12, 2026**; proposals accepted starting **Aug 13** |
| Build phase (2 months) | **Aug 26 → Oct 26, 2026** |
| Devpost "Submissions" period | Aug 26, 12:00am PDT → **Oct 26, 11:45pm PDT** |
| Grant-recipient 30-min Zoom check-in | **Sept 21 → Oct 2, 2026** |
| Public voting | Oct 26, 12:00am PDT → Nov 9, 11:45pm PST |
| Judging | **Oct 27 → Nov 9, 2026** |
| Winners announced | **Nov 10, 2026, 9:00am PST** (on OpenCV Live! webinar or another announced OpenCV channel) |

> ⚠ **Deadline discrepancy (verbatim from source):** the Devpost header/schedule states **Oct 26 @ 11:45pm PDT**, while the description text twice says **Oct 26 at 11:59 p.m. Pacific**. The platform-enforced 11:45pm PDT is the safe one to treat as binding.
>
> ⚠ Source artifact: the launch bullet reads literally "runs August 12- 2026" (end date garbled on the page). Recorded as-is.

Calendar links: [Apple](https://opencv26.devpost.com/calendar) · [Google](https://opencv26.devpost.com/calendar/google) · [Outlook](https://opencv26.devpost.com/calendar/outlook)

---

## 2. Prizes ($12,000 total)

**Overall Awards**
| Place | Amount | Winners |
|---|---|---|
| First Place Grand Prize | $5,000 | 1 |
| Second Place | $3,000 | 1 |
| Third Place | $2,000 | 1 |

**Special Awards** (optional focus paths)
| Award | Amount | Winners |
|---|---|---|
| Best Use of COOL Award | $1,000 | 1 |
| Agentic Vision Award | $1,000 | 1 |

Rules on stacking:
- An entry may receive **no more than one Overall Award**, and **may also receive one or both Special Awards**.
- Each Special Award is judged independently under its own rubric and goes to the highest-scoring eligible entry; **if no entry qualifies, the Special Award may go unawarded**.
- Payment within 30 days of final judging, conditional on timely eligibility + tax documentation.
- For teams, the **designated team representative** receives payment and distributes it. Prizes are non-transferable; Sponsor may substitute a prize of equal/greater value.

---

## 3. Hard requirements (every entry)

1. **Must use OpenCV 5 for substantive image or video analysis.**
2. **Must run a meaningful component on AWS.**
3. Any programming language or supporting hardware is allowed, provided 1 & 2 hold.
4. Emphasis: **Physical AI and Generative AI** systems where visual understanding drives useful decisions, actions, predictions, or interactions.
5. The two featured paths are **optional** — entries may pursue either, both, or neither and remain fully eligible for Overall prizes.

### Final submission checklist
- **Technical report** covering: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible-use considerations.
- **Code repository or archive**, public or private but judge-accessible. **Does not have to be open source.**
- **Pinned dependencies** plus clear build, deployment, and test instructions.
- **Architecture diagram** showing OpenCV 5 + AWS components (and COOL / agent components where relevant).
- **A working web endpoint** *or* an arranged **live screen-share demonstration**.
- **Video, ≤ 5 minutes**, public or unlisted and judge-accessible, showing: the team, the application working, its architecture, its principal results.
- **Evaluation evidence** appropriate to the project, **including failure cases or limitations**.

---

## 4. Focus paths (special prizes)

### 4a. Cloud-Optimized Vision with COOL — *Best Use of COOL Award*
COOL = **Cloud-Optimized OpenCV Library**, available via AWS Marketplace, optimized for **AWS Graviton (Arm)**.

Qualifying condition: **COOL must execute the claimed core workload on AWS Graviton**, or on the Arm component of a documented hybrid architecture.

Strong submissions:
- Show COOL executing the core image/video workload on the **Arm path**.
- Report **reproducible measurements** — latency, throughput, utilization, cost, or developer productivity — **against an appropriate baseline**.
- Explain any x86/Arm coexistence, container, server, or serverless architecture.

Additional required evidence:
- COOL version + AWS instance/deployment configuration.
- Reproducible evaluation method, inputs, baselines, results.
- Evidence that COOL executes the claimed core workload.

### 4b. Agentic Vision with OpenCV 5 — *Agentic Vision Award*
Build a workflow where an agent uses OpenCV 5 tools inside a multi-step **perception → decision → action** loop.

Qualifying condition: **image/video results must influence a subsequent plan, tool call, action, or request for human approval.**
- ❌ A chatbot that only *explains* a fixed vision result is **not enough** — visual evidence must change what the system does next.
- ❌ **Using an AI coding assistant to write the entry does not qualify as an agentic workflow.**
- ✅ A **developer-agent** workflow qualifies *only* when it iteratively invokes, evaluates, and adjusts an OpenCV 5 workload (with COOL where applicable) — e.g. using visual test outputs or runtime measurements to change a later tool call, config, or deployment — **and culminates in a running image/video workload**.
- Any agent framework or model is allowed; OpenCV 5 / COOL may be invoked through APIs, tools, or **Model Context Protocol (MCP)**.
- Example domains: autonomous inspection, visual troubleshooting, embodied assistants, active perception, safety monitoring, human-in-the-loop operations.

Additional required evidence:
- **Agent workflow diagram** showing perception, decision/orchestration, and action.
- A **trace or demonstration** showing OpenCV 5 output changing a later decision, tool call, or action.
- Evaluation of task success, failure handling, observability, and appropriate human control.

---

## 5. Suggested project areas (not exclusive)

- Active perception for autonomous inspection with agentic orchestration or MCP.
- Physics-informed video prediction for process control.
- Multi-agent visual SLAM with distributed context.
- Real-time spatial digital twins.
- COOL-based server, serverless, container, or hybrid x86/Arm vision pipelines.
- Developer agents integrating COOL with tools such as **Codex, Claude Code, Kiro, or MCP-compatible systems**.
- Healthcare, safety, accessibility, agriculture, environmental monitoring, smart cities, education, retail, sports analytics.

---

## 6. Judging

Judges are drawn from the AI and computer vision community; **AWS may nominate up to two judges**. Every eligible final entry receives **at least two independent, conflict-free scores**; final ranking uses the **average**. Judges must disclose conflicts and recuse. Judges' decisions are final.

**Named judges on the page:**
- **Phil Nelson** — Director of Content & Creative, OpenCV
- **Gary Bradski** — Founder, OpenCV

### Overall rubric (100 points)
| Criterion | Weight | What it covers |
|---|---|---|
| Technical execution | **30%** | Correctness and depth of the OpenCV 5 implementation, architecture, reliability, evaluation |
| Innovation | **20%** | Originality and thoughtful use of computer vision or AI |
| Real-world impact | **20%** | Importance of the problem, usefulness, evidence of potential benefit |
| User experience | **10%** | Usability, accessibility, quality of interaction |
| Documentation and presentation | **10%** | Clarity of report, code, architecture, instructions, video |
| Cloud delivery, reproducibility, responsible operation | **10%** | AWS deployment quality, repeatability, observability, security, responsible-use practices |

> COOL or agentic methods are **not required** for Overall Awards. Where present, judges credit them within the overall criteria only to the extent they improve the project.

### Best Use of COOL rubric
| Criterion | Weight |
|---|---|
| Verified COOL integration on AWS Graviton (or Arm component of a documented hybrid architecture) | **30%** |
| Architecture and technical quality | **25%** |
| Measured performance, cost, reliability, or developer-productivity value | **20%** |
| Innovation | **15%** |
| Reproducibility and demonstration | **10%** |

### Agentic Vision rubric
| Criterion | Weight |
|---|---|
| Substantive OpenCV 5 and agent integration | **30%** |
| Orchestration and appropriate autonomy | **25%** |
| Task effectiveness and evaluation | **20%** |
| Failure handling, observability, security, human control | **15%** |
| User experience, documentation, demonstration | **10%** |

### Tie-breaking
- **Overall:** Technical Execution → Real-World Impact → a conflict-free additional judge scores all tied entries under the same rubric.
- **Special awards:** first criterion of the applicable rubric → second criterion → a conflict-free additional judge.

---

## 7. AWS cloud compute grant

- **50 teams** will be selected for a cloud compute grant valued at **$150**.
- **All other teams — including those that applied and were not selected — remain fully eligible** to continue into the two-month build phase and to win prizes.
- **Apply:** https://www.jotform.com/form/262145877145059
- Grant recipients must complete **one 30-minute Zoom check-in** with an organizing committee member during **Sept 21 – Oct 2** to receive the **other 50%** of the grant, and must maintain active development and give a concise progress update at the check-in.

### Required contents of a grant proposal
1. A team name.
2. A clear problem statement and intended real-world impact.
3. The planned OpenCV 5 image or video analysis.
4. The planned AWS architecture and services.
5. A high-level architecture diagram or technical description.
6. The target users or beneficiaries.
7. The proposed evaluation method and judge demonstration.
8. Whether the team intends to pursue the **COOL path, the Agentic Vision path, both, or neither**.
9. **Team bio** for all members, including participation in prior hackathons/competitions — *team strength is used to judge grant recipients.*

### Free AWS credits (separate from the grant)
- New AWS customers get **$100 in AWS Free Tier credits** on signing up for a Free Plan account, and **may earn up to $100 more** through eligible activities.
- Usable for eligible AWS services in this competition, subject to Free Tier terms, service availability, and expiration rules.
- Governed by the **AWS Promotional Credit Terms & Conditions**; credits are **not redeemable for cash** and apply only to AWS services.

---

## 8. Eligibility

- Devpost overview states: **above the legal age of majority** in country of residence; **all countries/territories except Devpost standard exceptions**.
- Rules page states: participants must be **at least 13 years old**; anyone below the local age of majority must obtain **parent/guardian permission** and complete required platform forms. *(Both statements recorded as written — they differ in framing.)*
- **Every team member** must satisfy all eligibility requirements.
- A team must **designate one representative** authorized to submit and to receive notices and prizes.
- **Not eligible:** employees and contractors directly involved in administering or judging, plus their immediate household members.
- Entries must not manipulate registration, voting, judging, platform traffic, or contest operations via bots, fraud, hacking, or other abuse.
- **Disclosed agents and automated programs integral to the submitted project are allowed and encouraged** — the prohibition targets abuse of the competition, not legitimate project functionality.

---

## 9. IP, ownership, and privacy

- **Participants retain all right, title, and interest** in their entries and underlying IP: source code, models, model weights, datasets, inventions, methods, documentation, images, know-how (subject to third-party licenses). Participation does **not** transfer ownership to OpenCV or AWS.
- **License granted:** by submitting any proposal, report, presentation, or video ("Submitted Materials"), each participant grants OpenCV and AWS — **each independently** — a perpetual, irrevocable, worldwide, royalty-free, fully paid-up, non-exclusive, transferable, sublicensable license to use, reproduce, host, store, edit, modify, adapt, translate, create derivative works from, publish, distribute, transmit, publicly display/perform, promote and otherwise use those materials, in any media, for any lawful commercial or noncommercial purpose, without further notice, approval, attribution, or compensation.
- **Scope limit:** the license covers only the Submitted Materials themselves. It grants **no rights** to private source code, model weights, datasets, credentials, confidential materials, or other underlying assets that are merely referenced, linked, described, depicted, or demonstrated — unless those are themselves included in a submission.
- **Personal data** may be processed/shared among the administrator, OpenCV, and AWS only as reasonably necessary to administer the competition, verify eligibility, communicate, and deliver benefits/prizes. Use for unrelated marketing requires a **separate, informed opt-in**.

---

## 10. Responsible use — grounds for rejection

Entries must be safe, lawful, respectful, and appropriate for public demonstration. Judges may reject an entry that:
- Uses data, models, or media without necessary rights or consent.
- Creates material safety, privacy, security, discrimination, or surveillance risks without appropriate safeguards.
- Contains unlawful, defamatory, sexually explicit, gratuitously violent, hateful, harassing, or otherwise abusive content.
- **Misrepresents capabilities, results, benchmarks, or the role of human review.**
- Violates applicable OpenCV, AWS, Devpost, or third-party terms.

---

## 11. Administration, changes, liability

- Organizers may **disqualify** ineligible, incomplete, misleading, unsafe, unlawful, or rules-violating entries.
- Organizers may modify, suspend, or cancel the competition to preserve fairness, security, or legal compliance, communicating material changes through official channels.
- Potential winners must respond to notification and complete eligibility, release, and tax documentation by the stated deadline **or risk replacement by the next eligible entry**.
- Winners are responsible for applicable taxes; teams handle internal division of prizes.
- **Final participant-facing governing-law, venue, eligibility, platform, and privacy terms appear in the Devpost Official Rules; those control if the overview conflicts with them.**
- Liability: Released Parties are not liable for indirect/incidental/special/consequential/exemplary damages or lost profits, revenue, customers, opportunities, or goodwill; **aggregate liability capped at USD $100**, to the maximum extent permitted by law.

---

## 12. Links

**Official resources (from the Resources tab)**
| Resource | URL |
|---|---|
| OpenCV 5 documentation | https://docs.opencv.org/5.x/ |
| Official Competition Website | (linked on Resources tab, opencv.org) |
| OpenCV COOL overview | https://opencv.org/COOL/ |
| COOL on AWS Marketplace | https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs |
| AWS account and Free Tier information | https://aws.amazon.com/free/ |
| AWS Promotional Credit Terms | https://aws.amazon.com/awscredits/ |
| OpenCV community Slack server | https://bit.ly/opencv-slack |

**Other**
| Item | URL |
|---|---|
| AWS compute grant application (Jotform) | https://www.jotform.com/form/262145877145059 |
| AWS Free Tier terms | https://aws.amazon.com/free/terms/?p=ft&z=subnav&loc=6 |
| Full rules | https://opencv26.devpost.com/rules |
| Schedule | https://opencv26.devpost.com/details/dates |
| Resources | https://opencv26.devpost.com/resources |
| Participants (776) | https://opencv26.devpost.com/participants |
| Project gallery (not yet published) | https://opencv26.devpost.com/project-gallery |
| Updates (empty at capture) | https://opencv26.devpost.com/updates |
| Discussions (no topics at capture) | https://opencv26.devpost.com/forum_topics |
| Manage submissions | https://devpost.com/submit-to/30984-opencv-ai-competition-2026-powered-by-aws/manage/submissions |

**Community:** join the official OpenCV Slack channel to collaborate; use the hashtag **#OpenCVComp26** on social media to share progress and milestones. Questions go to the hackathon manager via the "Email the hackathon manager" link on the Devpost overview page.

---

## 13. Practical takeaways for a team building now

1. **Two non-negotiables gate every prize:** substantive OpenCV 5 image/video analysis **and** a meaningful AWS component. Everything else is scoring.
2. **Technical execution is 30%** of the overall score — depth of the OpenCV 5 work, architecture, reliability, and evaluation carry the most weight; UX, docs, and cloud-ops are 10% each.
3. **Evaluation evidence including failure cases is explicitly required** — a demo without measured results and stated limitations loses points in two rubric rows.
4. **Both special awards are cheap to stack**: an entry can win an Overall Award *and* both $1,000 special awards, but each requires its own hard evidence (Graviton/Arm execution proof for COOL; a decision-changing trace for Agentic Vision).
5. **The Agentic bar is behavioral, not architectural** — vision output must demonstrably change the next action. Build the trace/logging for this from the start; it is a submission artifact, not an afterthought.
6. **Ship a reproducible deployment**: pinned dependencies, build/deploy/test instructions, and a live endpoint or scheduled screen-share are all separately required.
