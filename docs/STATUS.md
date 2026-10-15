# Status

Dated entries. Newest first. Phase tracking checkboxes live in `ARIADNE_PHASED_PLAN.md`.

---

## 2026-10-15 — 1.4/1.5 run from the sim recording (D42)

- `StereoSource` (EuRoC, sim + read-time lens) and a shared `stereo_vo_runner.hpp`; EuRoC MH_01 reproduces
  byte-identically through the refactor. Sim configs `sim_v1` (240 disparities: Z_min 0.281 m, §9.4).
- Evidence, clean commit 869b946 (`evaluation/runs/phase1/sim_vo/`), ci_tiny seed 0: agent 0 ATE 9.9 mm over
  31.9 m (0.03 %, calibration spins/tilts included, 0 lost); agent 1 ATE 4.4 mm over 10.7 m (0.04 %, 1 lost
  frame, not yet investigated). Identical to the dev runs. Sensor degradation after the lens: Phase 5.
- Next: 1.6 placeholder collaboration (`docs/design/COLLAB_1.6.md`).

## 2026-10-01 — 1.1–1.3: mover, recording, canonical IMU/GT, direct gz recorder (D38–D40); full missions clean

- ROS-free CDR decoder + rosbag reader (D38); kinematic mover plugin running `scripted.hpp`; mission composer
  and rig; `sim_record` with strict verification.
- Deep queues did not make the 200 Hz path lossless (one IMU sample lost in a 199 s mission), so IMU and GT
  pose now come from `ariadne_canonical_streams` (D39): same parser and sampler as the plugin, own CDR encoder
  (rclcpp leaves padding uninitialised), byte-reproducible, content digest pinned by `worldgen.sh simtest`.
  Live streams are a bit-exact cross-check: 159,199 + 159,200 samples over two full runs, 0 mismatches.
- Three logged full runs put every loss on the bridge → `ros2 bag record` DDS link (residual 0 elsewhere,
  queues 1000 deep not enough). D40: `ariadne_gz_recorder` reads gz-transport directly, converts with the
  bridge's own functions, writes synchronously, fails loudly on any gap/duplicate/late start/incomplete end/
  backlog overflow; world starts paused until every publisher is discovered; `/tf_static` now canonical.
- Two full `ci_tiny` missions PASSED strict verification: 0 recorder errors, every frame and `/clock` step
  present, live IMU/GT byte-identical to canonical. Open: render determinism across runs (files differ;
  content not yet compared). All runs from a dirty tree: re-run from a clean commit for evidence.
- 1.3 evidence recording from clean commit 9cdb7eb PASSED (`evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T042105Z`).
- §6 calibration recovery (SMOKE §4n): renders are an exact pinhole to ~0.06 px; lens distortion is NOT rendered
  under ogre2 while camera_info claims it; camera_info cx/cy are +0.5 px off the pixel-centre convention;
  render/pose time offset ≈ 0. gz warnings are now a recording gate.
- D41 (option A): pinhole render + read-time Brown lens (`lens.hpp`, fx 680). §6 with the lens passes thresholds
  derived from the same dev run (circular until the evidence run; GT reprojection 0.07-0.09 px, k1 recovered
  -0.0506 vs -0.05) and fails a 5 mm board error. D41-rig 1.3 evidence recording PASSED from clean commit
  0f7b0e4. §6 on it reproduced the dev run to every digit (renders reproducible); the independent §6 test,
  ci_tiny seed 1, PASSED with margin on every check (clean commit 5de3f70). 1.1 acceptance met. Lens digest
  added to manifests and the determinism gate. Next: 1.4/1.5 consume the sim recording (sim adapter).
- Tests: core image 242/242, worldgen 104/104, simtest 19/19.

## 2026-09-30 — 1.1–1.3: worldgen implemented and frozen

Procedural room-maze worlds (D37, `docs/design/SIM_1.1-1.3_WORLDGEN.md`): floorplan, exact 2D GT, textured quad
geometry, props, ChArUco boards, lights, S-curve yaw+pitch scripts with an analytic IMU, closure-opportunity and
separation gates, four presets, 100 tests green in the sim image, golden digests pinned. Smoke results §4g–§4j:
kinematic pose writes work physics-free, render time = pose time (depth camera), gz IMU is blind under
kinematic motion, OBJ UV convention verified, relative `<include>` fails, two full rigs in a real scene run at
RTF ~0.17 and ~84 MB/sim-s. Next: ROS-free CDR decoder, mover plugin + synthesized IMU + agent SDF + mission
composer, first real bag, §6 calibration-recovery test. To re-run from a clean commit: census + accept sweeps.

## 2026-09-30 (latest) — 1.5 placeholder stereo VO: MH_01 ATE 0.126 m (0.16% of path), evo-verified

- Conventions pinned by tests: FD Jacobians (near-zero depth, wide FOV, right camera); Monte Carlo NEES 5.93 over 2000 trials (~6);
  pose-graph edge information checked by Monte Carlo.
- One LM (Huber, Schur, §11.6 damping, S X = E covariance) for PnP refinement and the two-frame BA; agrees with
  `solvePnPRefineLM` to 1e-6.
- Stereo VO: seeded USAC PnP, ORB-SLAM3 keyframe rule, §11.2 triangulation with an SGBM cross-check, pose graph with
  a version counter; `KeyframeBundle` (§18.2) and `Keyframe` (§4.2) records.
- Synthetic CI test: 4.6 mm RMSE over 3.5 m, robust to 10% corrupted tracks, deterministic across thread counts.
- MH_01 (3682 frames): ATE 0.126 m RMSE, Sim(3) scale 1.004, 0 lost frames, rotation drift 4.9 deg/182 s (no loop
  closure yet). evo reproduces the ATE translation and rotation numbers exactly.
- Tests: image 210/210, host 185/185. Evidence run: `core.sh euroc-vo` from a clean tree after committing.

---

## 2026-09-30 — 1.4 complete on EuRoC MH_01_easy

- EuRoC fetched by a ranged single-member download (1.57 GB of a 12.7 GB archive), CRC and sha256 verified.
  Licence InC-NC: no EuRoC pixels in submitted material (D35).
- Rectifier (alpha 0, fixed-point maps) verified synthetically (1e-3 px) and on real data: median row error
  0.135 px vs 13.1 px raw and 8.1 px for a 1-degree error.
- Full sequence, 3682 frames: 377 tracks/frame, 0 DEGRADED, 22.9 ms/frame tracking, 31.6 ms/keyframe SGBM.
- Near-range aliasing on real texture is episodic (worst keyframe 26%); the density gate catches 87%. 1.8 must pick
  "no stereo free-space carving" or a near-field pass.
- Tests: image 186/186, host 166/166. Evidence run: re-run `core.sh euroc` from a clean tree after committing.

---

## 2026-09-29 — 1.4 placeholder perception: components built, EuRoC run pending

- ORB engine with §7.3 grid selection, pyramidal-LK tracker with forward-backward and epipolar rejection, SGBM
  depth to the §9.6 contract, §9.4 range validator. D34.
- 178/178 tests in the image (13 new OpenCV perception tests); host 161/161 (5 new config/validator tests).
- Deterministic across OpenCV thread counts and global RNG state (verified bit-for-bit).
- **Hazard found and bounded:** SGBM reports phantom FAR depths (0.3 m to >3 m) for surfaces nearer than Z_min.
  Speckle window 200 cuts them ~10-30x; a measured dense-support gate (`DenseSupportMask`) rejects all survivors.
  Rule for 1.8: carve free space only through supported depth. Synthetic numbers; on EuRoC the gate catches 87%
  (D35).
- Open for 1.4: EuRoC adapter + rectification + a real-sequence run (Invariant 5).

---

## 2026-09-29 — `ariadne-core` image; Phase 0 gate 4 closed on amd64

- `infra/docker/core.Dockerfile` (`dev` + `ci` stages) FROM the pinned `opencv-base`; `infra/scripts/core.sh
  image|gate|dev`; CI job `core-image` via `docker-bake.hcl` (`[UNVERIFIED]` on GitHub). D33.
- Inside the image: **160/160 tests pass**, including the first OpenCV-linked code (`ariadne_runtime_opencv`,
  `ariadne_identity`, `test_opencv`). The host build without OpenCV still passes 156/156.
- **Phase 0 gate 4 PASS (amd64):** the `cv::getBuildInformation()` hash recorded in a run manifest at run time equals
  the one baked in at build time (`fnv1a64:cf8ea084bb984703`). The manifest also records `opencv_cpu_features`.
- The COOL loader switch is protected: no RPATH/RUNPATH on anything we build (enforced at build time); OpenCV is
  resolved through `ld.so.conf.d`.
- Still open: arm64 image (with 0.2), ECR push by digest, COOL soname check (0.3).

---

## 2026-09-29 — AWS live: tests-only spin-ups; determinism proven on real Graviton4

- Account on the PAID plan with $200 of credits kept (D26, D27). Region **eu-north-1**, the cheapest region
  meeting every need (D28). Minimal network: default VPC plus one egress-only security group (D29).
- Terraform applied: bootstrap (state bucket, state migrated to S3) and main (23 resources + the
  adopted budget, raised to $200; converged).
- `infra/scripts/spinup.sh`: spot by default, cheapest AZ, three independent brakes, results pulled back,
  cost ledger. First real run: **determinism PASS on Graviton4 with gcc 11.5 / clang 15**, identical digests
  to x86-64 gcc 13 / clang 18, for **$0.0003**. See SMOKE_TEST_RESULTS §4e.
- Pending with AWS: GPU quota (G/VT 4 vCPU, spot + on-demand) in eu-north-1. Bedrock is blocked until
  post-upgrade account verification completes.
- Still open: 0.2 arm64 OpenCV 5 build; 0.3 COOL trial (deliberately not started: it lasts 7 days and
  can't be paused, so start it when the benchmark exists).

---

## 2026-09-29 — `ariadne-sim` image builds and renders headlessly

Parallel track to Phase 1, started on the maintainer's go-ahead. `infra/docker/sim.Dockerfile`
(ROS 2 Jazzy + Gazebo Harmonic, gz-sim8 8.15.0), run by `infra/scripts/sim.sh`, acceptance test
`infra/scripts/sim_smoke.py`. Full evidence in `SMOKE_TEST_RESULTS.md` §4c.

- **PASS on Mesa llvmpipe** (this box has no GPU): headless camera frames rendered, bridged to
  ROS 2, and recorded to MCAP. The test judges pixel statistics, and each of its three failure
  branches has been made to fire.
- **Not verified: NVIDIA EGL on g5/g6.** Needs a GPU instance; first task there.
- **No OpenCV in the image** (D25): the `ros-jazzy-ros-gz` metapackage would have pulled Ubuntu's
  OpenCV 4.6, so the image uses the bridge/sim subset and the build fails if OpenCV appears.
- ROS packages pinned via the dated snapshot `jazzy/2026-09-11` (D25).

Honest record of the bring-up: the first background run was reported green by its log wrapper
while the test itself had failed (`SMOKE_EXIT=1`). The cause was the test's colour detector, not the
renderer. Two further test/tooling mistakes were caught by looking at frames rather than trusting
exit codes: `<sky>false</sky>` enables a sky in SDF, and `sim.sh` did not forward the timeout,
which made a normal 5 Hz camera look like 15 Hz.

**Not done by this:** item 1.1 (the two-room corridor scene) is still open; the smoke world is an
acceptance fixture, not a Phase 1 scene. **Design gap for 1.3:** rosbag2 writes CDR images, the
replay path reads Ariadne FlatBuffers (D25).

---

## 2026-09-08 — Phase 0 end-to-end audit: six claim-vs-reality gaps found and fixed

A full re-read of Phase 0 with the advisor, hunting for claims the code does not
support. Every finding below is that same shape: **not a crash, but a comment,
test name or job name asserting a guarantee the code did not provide.** None of
them produced a wrong published result, because nothing has been published.

| # | Claim made | Reality | Status |
|---|---|---|---|
| 1 | `DetExp` accurate "to a few ulp" over its whole declared domain | returned `inf` for x in [709.44, 709.78] and `0` for x in [-745.13, -745.0], both INSIDE its own guard | fixed |
| 2 | Philox block advance "without ever colliding with a neighbouring site's stream" | `(site=0, block=1)` and `(site=2^32, block=0)` produced byte-identical output | fixed |
| 3 | CI determinism gate enforces bit-identity | `script \| tee file` under `bash -e` without `pipefail` exited with `tee`'s status; a `DETERMINISM=FAIL` went green | fixed |
| 4 | `poisson_exact_max_lambda` is a safe config knob | above ~50 Knuth's loop hits its 10000-iteration cap and silently truncates every bright pixel | fixed |
| 5 | manifest digests exist "so drifted math is identifiable after the fact" | `MissingRequiredFields` passed a manifest with all three digests `UNRECORDED` | fixed |
| 6 | `CoeffC` cancellation is "~1e-9 relative" at t=1e-2 | actual is `eps*t/(t - sin t)` ~ 1.3e-11, off by ~100x | fixed |
| 7 | `EventLog` chain localizes where two runs diverged | `state_hash_ != 0 ? ... : basis` used zero as a sentinel, so a zero chain value would silently restart the chain and let two histories converge | fixed |
| 8 | "28 B emulated UDP/IP header per datagram" (§15.7) | charged once per MESSAGE; a 4266 B pose graph is 3 datagrams, so large classes were under-billed | fixed |
| 9 | D17 (written in this same audit): Philox bounds "are asserted, so the guarantee is structural" | `assert` is compiled out under `NDEBUG`, and this project builds `Release`; in the shipped binary `site=2^48` aliased onto `site=0` and block 65536 wrapped onto block 0 | fixed |
| 10 | `require("execution_mode", ...)` | defaults to `"replay"`, so the check could never fire; it read as a guarantee and verified nothing | fixed |

Row 9 is the sharpest instance of the pattern: it is a false guarantee **introduced
by this audit**, in the commit that fixed the original one. The bounds are now
enforced by code that survives `NDEBUG` -- `site` is masked (documented 48-bit
truncation, with `SiteFromHash64()` for callers holding 64-bit ids), and block
exhaustion throws rather than silently wrapping onto a stream it already used.
The determinism gate could not have caught it: `check_determinism.sh` compiles
with bare `-O0/-O2/-O3` and no `-DNDEBUG`, so asserts are live there and dead in
the binary that would run on Graviton.

### The finding behind the findings

Re-running the digests after fixing 1 and 2, **two of the three were unchanged**:

- `det_math_digest` swept only x in [-500, 500] — the `DetExp` bug lived entirely outside it.
- `philox_digest` used sites < 2^32 drawing 2 words each, so `site >> 32 == 0` and
  `block == 0` — it never varied the two counter fields that aliased.

Both golden digests were byte-identical across genuinely broken code. A digest only
guards the inputs it evaluates, and these evaluated the safe interior of both
functions. The sweeps now cover the overflow/underflow boundaries and the
high site bits / multi-block draws respectively, and all three digests were
re-recorded. This is the durable lesson from the audit, more than any individual bug.

### Re-verification after the fixes

```
x86-64   PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   6/6 bit-identical
aarch64  PASS g++ -O0/-O2/-O3 (QEMU)                  3/3, identical to x86-64
det_math_digest = 11806531094783593156
philox_digest   = 14182439601419266695
degrade_digest  = 8535765251477677907
```

Control still fires: on aarch64 with `-ffp-contract=fast` at -O2/-O3 the det_math digest
becomes `1750406000263991614`, so `-ffp-contract=off` remains load-bearing and the
extended sweep is still a sensitive canary.

`DetExp` is now within **1 ulp** of libm across both previously-broken bands.

**146 tests green** (was 137; +9 regression tests pinning the defects). Note the
suite builds `Release`, so the two new Philox bound tests exercise exactly the
configuration where the old asserts were absent.

Byte accounting moved with the per-datagram fix: the compressed 200n/40e pose graph
goes 4294 -> 4350 wire bytes (ratio 4.93x unchanged, since that is payload over
transmitted), and the Phase 0 gate 8647 -> 8759. MCAP still reproduces the gate
byte-for-byte.

Phase 0 gate re-run: `poses=200 messages=4 wire_bytes=8647`, ATE rmse 0.150672 m,
RPE trans rmse 0.0175348 m, rot rmse 0.0033766 rad.

The ATE moved from 0.0633 m because trajectory noise draws from Philox and the
stream changed. **Measured, not assumed:** ATE rmse across 8 seeds spans
0.0717 - 0.1507 m (median 0.1139), a 2.1x spread, so this metric is dominated by
which noise realization it drew. Stated precisely: the new value sits at the top
of that sampled spread and the old one at or below its bottom, so neither is
anomalous but neither is central either. ATE is a single accumulated random-walk
statistic and is high-variance by construction.

The load-bearing check is RPE, which is local and averaged over 199 steps:
trans rmse 0.0175 m still matches the model's sqrt(3) x 0.01 = 0.0173. That is
what validates the noise model; ATE is reported, not gated.

### Correction to an earlier claim in this file

`cross-arch-determinism` was labelled "THE JOB THAT MATTERS". It is not: each
architecture already asserts against the committed constants, so if both per-arch
steps pass they necessarily agree with each other and the job cannot fail. It has
been downgraded to a second, independent comparison that does not route through
those constants, and given `if: always()` so it runs when a per-arch gate failed —
which is the only case where it is informative.

### Still open from the audit

- **§30.2 "Core metrics" (arch line 4201) has never been read.** ATE/RPE were
  implemented from §30.1 plus judgement. Phase 1.9 requires every §30.2 metric to
  emit a number, so this is a **Phase 1 prerequisite**, not a Phase 0 defect. Two
  things to check specifically: whether rotational ATE is wanted (only translation
  is computed today, which is `evo`'s default but not universal), and whether RPE
  deltas should be keyframe-, time-, or distance-based (only keyframe exists).
- `IsCompressedClass` was checked and is sound: the policy is stated once, asserted
  by a test, and D14 documents the POSE_GRAPH extension to §15.7's classes 4-5.
  `ChainHash` and the framing arithmetic were both defective and are now fixed
  (rows 7 and 8 above).
- The MTU model is an `[ASSUMPTION]`: 1500 B Ethernet, IPv4 headers, no jumbo
  frames. It should be revisited against the real link model in Phase 3.

---

## 2026-09-06 — Phase 0 exit gate built and passing

### The gate

`tests/end_to_end/test_phase0_gate.cpp` runs the loop the plan names: a synthetic
trajectory is generated, serialized to a record file, replayed back, and scored against
ground truth. Two independent runs are compared with exact equality, not a tolerance,
because a tolerance would hide precisely the drift the gate exists to detect.

```
poses=200  messages=4  wire_bytes=12632
ATE  rmse=0.0633155 m   max=0.130391 m
RPE  trans rmse=0.0174459 m   rot rmse=0.00357854 rad
```

The RPE figure validates the noise model rather than merely reporting it: per-step sigma is
0.01 m on three independent axes, and sqrt(3) * 0.01 = 0.0173 against a measured 0.01745.

**gcc 13.3.0 and clang 18.1.3 produce identical metrics**, which is stronger than the gate
requires (it asks only for reproducibility twice within a run).

### What the gate covers beyond the headline

- Replay reconstructs the recorded trajectory to float32 exactness.
- A different seed changes the metrics, so the gate can fail.
- Umeyama alignment recovers a known rigid transform, and recovers a known scale in the
  Sim(3) variant used by the monocular ablation.
- **Reflections are rejected.** An unconstrained SVD can return an improper transform that
  fits mirrored points better than any rotation; the determinant fix is asserted directly.
- A truncated record file is reported as truncated, not passed off as a clean end of file.

### Pieces added

- `core/trajectory/synthetic.hpp` -- Lissajous ground truth (self-intersecting, so it
  produces genuine revisits for later loop-closure work) plus a noisy estimate built by
  corrupting relative motions and re-integrating, so error compounds into realistic drift.
- `runtime/replay/record_file.hpp` -- length-delimited framed messages written through the
  SAME `WireCodec` the network uses, so replay exercises the production encode path and a
  wire-format bug cannot hide behind a separate file format.
- `runtime/metrics/trajectory_metrics.hpp` -- Umeyama alignment, ATE, RPE.

**101 tests green.**

### Honest limits of this gate

- The record file is **not MCAP**. §5.9 specifies MCAP as the simulator recording container;
  that remains open. This format is the deterministic substrate the gate needs, and MCAP will
  wrap the same framed messages when it lands.
- Cross-ARCHITECTURE identity of the METRICS is not claimed. The dominant reason is that
  `GenerateGroundTruth` calls `std::sin`/`std::cos`/`std::atan2`, which libm is not required
  to round correctly; Eigen's SVD in Umeyama is a second, smaller contributor. (An earlier
  version of this file named only the SVD, which understated the cause.) This is harmless:
  the gate asserts equality WITHIN a run, and CI never compares gate metrics across
  architectures. Cross-architecture bit-identity IS proven for the degradation operator and
  the deterministic math, which is what the plan's gate wording actually requires.

### 0.5 done — observability and seed plumbing

- `core/random/seed.hpp` — no global RNG anywhere. Each subsystem derives its seed from the
  run's root seed plus a NAME-hashed purpose id, so adding a subsystem later cannot shift an
  existing one's stream and silently move every ablation baseline. splitmix64 finalizer, so
  sweep seeds 1..N decorrelate rather than producing structured sub-seeds.
- `runtime/logging/event_log.hpp` — §41's 20 event types, JSONL, with a chained `state_hash`.
  The chain is the diagnostic: comparing two runs gives the exact `sequence_id` where they
  first diverged, turning a determinism failure in a long mission into a bisection.
- `runtime/manifest/run_manifest.hpp` — §28.2 fields. Images by digest not tag; OpenCV by
  build hash not version string (C7); `execution_mode` recorded because replay is
  bit-deterministic and live is not (§5.9); `imu_mode` recorded so §10.7's fallback is an
  honest downgrade. `MissingRequiredFields` refuses a run whose tree was dirty, since the
  commit then does not identify the code that ran.

### 0.6 partial — CI and infrastructure as code

- `.github/workflows/ci.yml` — build and test on amd64 and arm64 hosted runners, ccache and
  dependency caches keyed on the pins file, the determinism gate on each arch, and then the
  job that matters: **`cross-arch-determinism`**, which compares the two architectures'
  digests against EACH OTHER. Each arch already checks itself against recorded values; this
  proves they agree, which is what lets a Graviton result be compared with a dev-box result.
  The digest-extraction logic was verified against real script output rather than assumed.
  `[UNVERIFIED]` the workflow has not executed on GitHub. arm64 hosted runners are free for
  public repos; if unavailable to this account, switch to a self-hosted Graviton label per
  §27.2 and serialize benchmark jobs behind a lock.
- `infra/aws/terraform/` — network, storage, registry modules (§27.3's stated 0.6 scope).
  `terraform validate` passes. **Not applied**: no AWS credentials here, and applying is a
  billable action that is the maintainer's to take.
  - network: one VPC, one public subnet, no NAT gateway (~$32/month standing for nothing
    this design needs). Worker SG is egress-only with SSM access rather than an open SSH
    port. Console SG is closed until `allowed_console_cidrs` is set deliberately.
  - storage: one bucket with §28.1 prefixes; versioning ON because run artifacts are
    evidence and an overwritten metrics file with no history is indistinguishable from a
    fabricated one; `runs/` to IA at 30 d, noncurrent versions expired at 90 d, incomplete
    multipart uploads aborted at 7 d so they cannot bill invisibly.
  - registry: five repos per §27.1, scan-on-push, untagged images expired at 14 d. Tags stay
    mutable only because manifests pin digests (§27.2).

### 0.1 done — config loader, named configs, licences, Python env

- `core/config/config.hpp` — strict loading. **An unknown key is an error**, which is the
  whole point: a permissive loader turns `radiance_scal:` into "use the default", so a
  dark-scene ablation silently measures a lit scene and no test notices. Missing required
  keys and wrong types fail the same way, and the message names the offending key.
- `configs/degradation/dark{0..3}_dust{0..3}.yaml` — the 16 named configs §5.6 requires,
  referenced by name from run manifests instead of inline constants. The sensor is identical
  across darkness levels and only scene radiance changes, so read noise stays constant while
  signal falls. That is what makes low light hard rather than merely dim, and a test asserts
  it rather than trusting the files.
- `LICENSES/THIRD_PARTY.md` — every dependency's licence verified from the fetched source,
  not from memory. Two findings: zstd is dual BSD-OR-GPLv2 and we take BSD; Eigen ships some
  LGPL code, so the build now defines **`EIGEN_MPL2_ONLY`**, making inclusion of any LGPL
  Eigen header a compile error. Permissive licensing is therefore proven by compiling rather
  than asserted. Ariadne's own licence is still `[UNDECIDED — maintainer]`; the competition
  does not mandate one.
- `pyproject.toml` — analysis environment with bounded pins, and `evo` as the trajectory
  oracle (§30.1) so our ATE/RPE is cross-checked rather than self-certified.

### 0.7 done — MCAP container

`runtime/replay/mcap_recorder.hpp` over the standalone Foxglove SDK (D3), so replay works
from files alone with no ROS present while bags stay rosbag2-readable.

Bags store the FRAMED message, so replay exercises the production decode path rather than a
parallel on-disk format. Verified:
- Replaying through MCAP reproduces the record-file gate **exactly**: ATE 0.0633155, RPE
  0.0174459, 8626 wire bytes. Adopting the container costs no determinism.
- Frames survive the container byte for byte.
- Messages replay in log-time order, and log time comes from the message stamp rather than a
  wall clock, which would otherwise make every bag unique.
- **Bags written twice from identical inputs are byte-identical**, which is stronger than the
  gate requires and is what makes a bag checksum in a §28.2 manifest meaningful.
- A missing or non-MCAP file is rejected rather than silently replayed as empty.

### Remaining in Phase 0

0.5 logging/event schema, run manifest, seed plumbing; 0.6 CI and Terraform; 0.7 MCAP
container; 0.1 config schema loader, `pyproject.toml`, `LICENSES/`; 0.2 arm64 OpenCV build;
0.3 COOL day-1.

---

## 2026-09-06 (later) — degradation operator and serialization boundary

### 0.7 — the operator `D` (§5.6)

`cpp/perception/degradation/degradation.hpp` implements the full sensor chain:
radiance scale, Poisson shot noise, read noise, PRNU and vignette, exposure and gain,
quantization and clipping, dust compositing. Signature is
`D(image, depth_gt, params, seed, frame_id)`: the plan's four-argument form has nowhere
for GT depth to enter, which §5.6's dust stage requires.

Physical properties that are asserted, not assumed:
- PRNU is keyed frame-invariantly, so fixed-pattern noise stays fixed. Drawing it per
  frame would make it removable by frame averaging and inflate the apparent value of
  §6.2's flat-field stage.
- Each stage draws from its own stream, so disabling one stage in a Phase 5 ablation does
  not reshuffle the noise every later stage sees.
- Shot-noise variance scales with signal, read noise does not; both verified.
- Vignette follows cos^4 and matches the requested corner response.
- Dust matches the closed form `I = J*t + A*(1-t)`, `t = exp(-beta*d)`.
- Locality: perturbing one input pixel changes exactly one output pixel, which is what
  makes the operator safe to parallelize over any partition.

### 0.4 — serialization boundary (§4.4, §15.7)

`wire.fbs` plus `serializer.hpp`. FlatBuffers headers are generated by `flatc` at build
time and never committed, so the schema cannot drift from the code. zstd is applied to
message classes 4 and 5 (MATCH_RESPONSE, POSE_EDGE) per §15.7. Byte accounting is computed
in exactly one place. §15.5 frame tagging is enforced: untagged poses are rejected rather
than defaulted, since a pose with an unknown frame cannot be composed with anything.

### Determinism gate

Now covers `D` end to end, which is what the Phase 0 exit criterion names. Six
configurations (gcc and clang, -O0/-O2/-O3) are bit-identical. `-ffast-math` still changes
the det_math digest, confirming the control has teeth.

**90 tests green.**

### Not started

0.5 logging/manifest/seed plumbing, 0.6 CI + Terraform, 0.7 MCAP replay skeleton,
the thin end-to-end trajectory loop that the Phase 0 gate names, 0.3 COOL day-1.

---

## 2026-09-06 — Phase 0: OpenCV verified, determinism substrate landed

### 0.2 smoke tests — now `[BUILD-VERIFIED]` on amd64

`ariadne/opencv-base:5.0.0` built and verified. `infra/scripts/smoke_test.cpp` compiles,
links and runs inside it; **the compile-and-link succeeding is the evidence**. Full results
in `SMOKE_TEST_RESULTS.md` §4b. Highlights:

- **HashTSDF constructs** → §20.1 satisfied, the from-scratch TSDF fallback is not needed.
- `ALIKED`/`DISK`/`LightGlueMatcher`/`ANNIndex` present; `ALIKED::create` resolves in the
  installed `.so`. **XFeat still absent** (issue I1 below).
- `DisparityWLSFilter` and the `quality` module link → C7 contrib path works.
- `build_info_hash = 9127192543017384188` for the §28.2 manifest.
- Headers install to `include/opencv5/`, **not** `opencv4/`.

Two build corrections came out of this (D7, D8): `xfeatures2d` removed from `BUILD_LIST`
(it stalled the first build on configure-time blob downloads and nothing in the
architecture uses it), and manifests must pin the **peeled commit**, not the tag-object
SHA that `git ls-remote --refs` returns for an annotated tag.

### 0.4/0.7 — deterministic primitives, the substrate C6 rests on

- `core/random/philox.hpp` — Philox4x32-10 counter-based RNG. **Passes all three
  authoritative known-answer vectors** from the Random123 reference implementation, so our
  stream matches the published algorithm rather than merely being self-consistent.
  Stateless by construction: value = f(key, counter), so thread count and iteration order
  cannot change a single bit (both asserted under real concurrency).
- `core/math/det_math.hpp` — `DetExp`/`DetLog` built from correctly-rounded operations
  only. libm's exp/log are *not* specified to be correctly rounded and legitimately differ
  between glibc versions and between x86-64 and AArch64; calling them inside `D` would
  break cross-machine bit-identity silently, one ulp at a time. libm is used as the
  **accuracy oracle in tests**, never as the runtime path.

### Determinism gate — measured, not asserted

`infra/scripts/check_determinism.sh` (repeatable; belongs in CI):

```
PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   →  6/6 bit-identical
det_math_digest = 8409701359450404215
philox_digest   = 215287054852263674
```

**The control proves the test has teeth:** `-ffast-math` *changes* the det_math digest
(`8409…` → `12633…`), which is exactly the failure D5's `-ffp-contract=off` guards against.
The Philox digest is unchanged even under `-ffast-math`, as an integer-only algorithm should be.

Precise caveat: on this host the *default* (no `-ffp-contract=off`) also matched, because
the default `-march` has no FMA. **On Arm, FMA is baseline and the default would contract**,
so the flag is load-bearing for the arm64 arm, not decorative.

**Tests: 48/48 green.**

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seed plumbing,
0.6 CI + Terraform, 0.7 MCAP replay and the `D` operator itself (its primitives now exist),
0.3 COOL day-1.

---

## 2026-08-31 — Phase 0 opened


### Done and verified

- **0.1 (partial)** Monorepo tree per arch §3 (106 directories); CMake superbuild with
  hermetic `FetchContent` pins; `.gitignore` / `.dockerignore`; this doc plus
  `DECISIONS.md` and `SMOKE_TEST_RESULTS.md`.
- **0.4 (partial)** `cpp/core` — Eigen-only, no OpenCV, per the §3 dependency rule:
  - `core/math/lie.hpp` — SO(3)/SE(3) exp/log/adjoint, left-multiplicative convention,
    `[rho; phi]` ordering (D1), stable small-angle and near-pi branches.
  - `core/time/timestamp.hpp` — integer-nanosecond `Timestamp`/`Duration`, §10.6
    time-offset contract carried explicitly as a stated simplification.
  - `core/uncertainty/covariance.hpp` — SE(3) covariance transport, composition and
    inverse rules derived for the left convention.
  - **22/22 tests green** (`ctest`). Covariance composition and inverse are checked
    against 200k-sample Monte Carlo, not against a re-derivation; a control test
    confirms the naive no-adjoint rule is rejected by the same check.
- **0.2 (partial)** Smoke tests at `[SOURCE-VERIFIED]` level — see `SMOKE_TEST_RESULTS.md`.
  C3's module-naming discrepancy is **resolved**; HashTSDF **confirmed present**, so the
  from-scratch TSDF fallback is not needed.

### In flight

- `ariadne/opencv-base:5.0.0` amd64 image building locally (`-j4`). Promotes the
  smoke-test rows from `[SOURCE-VERIFIED]` to `[BUILD-VERIFIED]`.

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seeds,
0.6 CI + Terraform, 0.7 MCAP replay + degradation operator `D`, 0.3 COOL day-1.

---

## Open issues

### I1 `[RESOLVED 2026-09-06]` — XFeat will be implemented on OpenCV 5 primitives

`cv::XFeat` does not exist in OpenCV 5.0.0. Maintainer decision (D15): implement the forward
pass directly on OpenCV 5 primitives rather than routing upstream weights through ONNX, since
substantive OpenCV 5 implementation scores better and it removes the `cv::dnn` ONNX-coverage
risk. Inference is ours, weights are upstream (Apache-2.0) and credited. Phase 2.1 work, with
validation against upstream PyTorch outputs as the acceptance test.

Original finding retained below for provenance.

### I1-orig `[HISTORICAL]` — XFeat is not in OpenCV 5

CLAUDE.md §1 locks XFeat as the primary feature extractor; it has no implementation in
OpenCV 5.0.0 or contrib. Evidence in `SMOKE_TEST_RESULTS.md` §2, decision record D6.
Needs the §49 update ritual. **Does not block Phase 0.**

### I2 `[SCHEDULE]` — the phase durations do not sum to the calendar

Phase durations total **62 days** (`4+6+10+8+7+7+5+6+9`) plus Phase 8's stated buffer,
against **57 days** to the Oct 26 platform deadline and **55** to the Oct 24 internal
target. The plan's own mitigations are gates-not-dates and the `∥A/∥B/∥C` parallel
workstreams; the compressible slack is Phase 2 (10 d) and Phase 8 (9 d), and §10.7
`attitude_only` is the named fallback for a Phase 2 overrun. Flagged now so the decision
is taken deliberately at the Phase 1 exit gate rather than discovered in October.

### I3 `[RESOLVED for determinism]` — arm64 bit-identity proven under emulation

QEMU aarch64 (gcc 13.4.0) reproduces all three digests exactly, at -O0/-O2/-O3. Bit-identity
is not a timing property, so emulation is valid evidence here. The control fired: without
`-ffp-contract=off`, Arm's baseline FMA changes `det_math_digest`, which x86-64 could not
have revealed. See DECISIONS.md D5.

Still open on real Arm hardware, for reasons unrelated to determinism: the OpenCV 5 arm64
BUILD, and every COOL timing number (§26.3 is Arm-vs-Arm and emulated timings are worthless).

### I3b `[OPEN]` — arm64 has no local host for builds and timing

Cross-**toolchain** determinism is now proven (gcc + clang, 3 optimization levels each);
cross-**architecture** is not, and is the half that actually matters for Graviton. The
digests above are committed so the arm64 check is a one-line comparison when a box exists.

Phase 0.2/0.6 require an arm64 arm. This box is amd64-only; QEMU timings would be
meaningless and the COOL benchmark must run on a real `m8g` regardless. Deferred to a
Graviton box or GitHub Arm runner. Recorded as `[PENDING]`, not as passing.

### I5 `[PARTIALLY RESOLVED]` — §15.7 byte predictions vs measurement

Resolved for the aggregate: POSE_GRAPH is now compressed (D14), taking a 200n/40e batch from
21.0 KB to 4.3 KB, under §15.7's 12 KB prediction rather than 75% over it.

Still open: the per-element figures. Node measures 65.8 B against a predicted 28 B, edge
193 B against 150 B. The doc amendment (D12) is unmade, so the §15.7 table still states
numbers this schema does not produce.

Original finding below.

### I5-orig `[HISTORICAL]` — §15.7 byte predictions are low by 29-135%

Measured: pose-graph node 65.8 B against a predicted 28 B; pose edge 193 B against 150 B.
The node figure is unreachable, not merely optimistic: §15.5 requires frame-tagged poses,
and 28 B is exactly a bare pose with no identity, timestamp or tag. Phase 3 gates
predicted-vs-measured at 15% and §16's policy divides by predicted cost, so this needs
resolving before spectral selection is tuned. Detail and options in `DECISIONS.md` D12.

### I6 `[OPEN]` — dust model fidelity, grouped for one decision

D9 (dust composited after the sensor, so it cannot affect shot-noise statistics) and D11
(the dust field is temporally white, so frame averaging removes it). Both make Phase 5's
restoration ablations easier than reality. Neither blocks Phase 0.

### I4 `[COSMETIC]` — doc filename drift

Arch §3's tree names `docs/ARCHITECTURE.md`; the real file is
`docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md`. Source of truth not renamed.
