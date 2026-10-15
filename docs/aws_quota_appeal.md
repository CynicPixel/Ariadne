# Appeal text: EC2 G/VT quotas (cases 179069316400187 and 179069316600303)

**2026-10-15 re-file.** The 2026-09-29 cases auto-closed on 2026-10-09 (no reply from us within 10 days). New
CLI requests filed 2026-10-15, both 4 vCPU in eu-north-1, status PENDING:
- spot "All G and VT Spot Instance Requests": request id `2ea5cdb12c3a4f848341f75c88fd9bc5eeqrJ1OG`
- on-demand "Running On-Demand G and VT instances": request id `7e1e3706c4ed48f1a6104034571eb296gs50z3DA`
AWS opens a support case per request; reply on each case with the text below (CLI requests carry no
justification). The "History" paragraph is still accurate as written (one 67 s Graviton4 spot run, per
`build/runs/ledger.csv`); add that the deadline is now 11 days away (26 October 2026) and that the final
simulation and reconstruction runs need the GPU.

Original 2026-09-29 requests, both denied with "ramp up gradually":
- 179069316400187: "All G and VT Spot Instance Requests" -> 4
- 179069316600303: "Running On-Demand G and VT instances" -> 4

Reopen both and reply with the text below. For the on-demand case, replace the "What" and "How much"
paragraphs with the on-demand variants at the bottom.

---

Hello,

Thank you for reviewing this. I'd like to appeal with a detailed use case, since the request is small
and bounded.

**What:** one g4dn.xlarge (4 vCPU, a single NVIDIA T4) as a spot instance in eu-north-1. That is the
entire request: 4 vCPU, the minimum for any G instance.

**Why:** I am building an entry for the OpenCV AI Competition 2026 (powered by AWS, deadline
26 October 2026). The project reconstructs 3D scenes with Gaussian splatting (the gsplat library),
which requires an NVIDIA GPU with CUDA. Everything else in the project already runs on this account's
CPU instances, including Graviton4 (c8g) for the competition's COOL track.

**How much:** roughly 25 instance-hours in total over the next four weeks, in short test runs of
10 to 60 minutes each. At current Stockholm spot prices that is about $2 to $3 in total.

**Safeguards already in place on this account:**
- Every instance is launched by a script that sets the instance to terminate on shutdown and schedules
  a shutdown at a hard time limit (at most 4 hours) as the first boot action. The launching host
  terminates the instance on exit. A separate sweep terminates any project instance past its deadline.
  Root volumes are deleted on termination.
- An AWS Budgets cost alarm on gross spend (credits excluded) alerts at 25%, 50% and 80% of the account's
  credit balance, plus a forecast alert.
- No NAT gateways, Elastic IPs or long-lived resources; the account's only persistent resources are
  S3 and empty ECR repositories.

**History:** the account has already run a Graviton4 (c8g.large) spot instance with this same
tooling. It ran for 67 seconds and terminated itself when the test finished.

I'd be grateful if you could reconsider a limit of 4 vCPU for "All G and VT Spot Instance Requests"
in eu-north-1.

Thank you.

---

**On-demand variant (case 179069316600303).** Replace "What" and "How much" with the two paragraphs
below, and change the closing request to: *a limit of 4 vCPU for "Running On-Demand G and VT instances"
in eu-north-1.*

**What:** one g4dn.xlarge (4 vCPU, a single NVIDIA T4) on-demand in eu-north-1, used only when spot
capacity is unavailable. 4 vCPU is the minimum for any G instance.

**How much:** at most about 25 instance-hours over the next four weeks, in runs of 10 to 60 minutes.
At the Stockholm on-demand price ($0.558/h) that is under $15 in total.
