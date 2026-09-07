# Status

Dated entries. Newest first. Phase tracking checkboxes live in `ARIADNE_PHASED_PLAN.md`.

---

## 2026-09-06 (latest) — Phase 0 exit gate built and passing

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
- Cross-ARCHITECTURE identity of the METRICS is not claimed: Umeyama uses Eigen's SVD, which
  is not guaranteed bit-identical across architectures. Cross-architecture bit-identity IS
  proven for the degradation operator and the deterministic math, which is what the plan's
  gate wording actually requires.

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
