# Local Decisions

Dated decisions taken during implementation. Anything here that changes the
architecture must be mirrored back into `ARIADNE_IMPLEMENTATION_ARCHITECTURE.md` per §49.

---

## 2026-08-31 — D1: Tangent-vector ordering is `[rho; phi]` (translation first)

Arch §11.3 pins the *perturbation convention* (left-multiplicative) but not the
*ordering* of the 6-vector. Chosen: `[translation; rotation]`, matching Sophus.

**Why it matters:** the SE(3) adjoint's block layout depends on this ordering. GTSAM
orders it `[rotation; translation]`, so when GTSAM is used as a test oracle (§11.6) the
ordering must be swapped **at that boundary**, never inside `cpp/core`.
Enforced by `SE3.AdjointMovesPerturbationsBetweenFrames` in `tests/numerical/test_lie.cpp`,
which fails on a wrong ordering while every round-trip test still passes.

## 2026-08-31 — D2: Eigen pinned at 3.4.1, not 5.0.x

Arch §27.1 says "Eigen 3.4". Upstream now also offers 5.0.1. Doc wins: pinned **3.4.1**
(latest 3.4.x). Revisit only if a needed feature is 5.x-only; it would be an arch change.

## 2026-08-31 — D3: MCAP via the standalone C++ SDK, not rosbag2

Arch §5.9 names "rosbag2 MCAP" as the format; arch §3's dependency rule requires replay
to work from files alone, ROS-free. These are compatible because MCAP is a container, not
a ROS API: we read/write it with the standalone Foxglove MCAP C++ SDK, so `cpp/core` never
sees a ROS type and replay needs no ROS install. Bags stay rosbag2-compatible.

## 2026-08-31 — D4: `SE3Log` inverts the left Jacobian numerically

`V^-1` is computed with a 3x3 `Eigen::inverse()` rather than its closed form. `V` is
always invertible for `|phi| < 2*pi`, the cost is negligible, and this removes a class of
transcription bugs from a coefficient formula that is easy to get subtly wrong.
Alternative (closed form) may be added later **with a test against this version**.

## 2026-08-31 — D5: `-ffp-contract=off` project-wide, no fast-math

The Phase 0 gate requires the degradation operator `D` to be bit-identical across
machines. Compilers may fuse `a*b+c` into an FMA, changing the last bit depending on
target ISA — differently on amd64 vs Graviton. Contraction is therefore disabled
globally rather than only for `D`, so the property cannot be lost by a file moving
between targets.

## 2026-08-31 — D6 `[OPEN — needs maintainer decision]`: `cv::XFeat` does not exist

CLAUDE.md §1 locks XFeat as the primary feature extractor. `docs/ALGORITHMS.md` records
this as low-risk because XFeat is "a native OpenCV 5 class", and at line 63 documents a
concrete signature:
`cv::XFeat::create(modelPath, maxKeypoints, scoreThreshold, inputSize, backendId, targetId)`.

**That class does not exist in OpenCV 5.0.0 or contrib** (evidence:
`SMOKE_TEST_RESULTS.md` §2 — exhaustive case-insensitive search of both trees). The
catalogue's claim is a factual error, and it is load-bearing in six places there
(§1.2 shortlist, the OpenCV-5 class table, the recommended per-agent front-end, and the
EC3R-SLAM "strongest cross-domain signal" argument).

**The locked decision itself may still survive.** What is false is "OpenCV ships a class
for it", not "XFeat is usable". Verified upstream facts (2026-08-31):
- `github.com/verlab/accelerated_features`, Apache-2.0 (license confirmed in catalogue L36).
- Ships **PyTorch** weights only: `weights/xfeat.pt` (6.2 MB), `xfeat-lighterglue.pt`.
  **No ONNX export is published** — we would own that step.
- The model is small (~10 conv blocks; `modules/model.py` is 4.5 KB).

**Relevant fact that applies to every option:** OpenCV 5 ships ALIKED/DISK/LightGlue as
classes but **not** their models — all three take a `modelPath` to an ONNX file. So a
learned feature path requires sourcing/exporting ONNX weights no matter which is chosen;
that cost is not unique to XFeat, and §7.4's DNN execution contract governs all of them.

Options, none yet chosen:
- **(a) Keep XFeat primary, via ONNX + `cv::dnn`.** Export `xfeat.pt` ourselves under
  §7.4 (ORT fallback allowed, log which ran). Preserves the locked decision and the
  catalogue's benchmark case. Risk: the export must survive `cv::dnn`'s ONNX coverage —
  a `[SMOKE-TEST]` gate, not an assumption.
- **(b) Promote ALIKED to primary** via the §49 ritual, demoting XFeat to an ablation.
  Native class, but C8 rates ALIKED ~5 FPS-class on CPU, and the catalogue's speed
  argument for XFeat (16.7 ms; the only learned method above 1 FPS on embedded) is
  exactly what motivated the choice.
- **(c) Defer to the Phase 2 opening**, recorded now, decided when 2.1 starts.

Does not block Phase 0. Blocks Phase 2.1.

## 2026-09-06 — D7: manifests pin the PEELED COMMIT, never the tag-object SHA

`5.0.0` is an annotated tag in both OpenCV repos. `git ls-remote --tags --refs` returns the
**tag object** SHA; the commit is the peeled `refs/tags/5.0.0^{}` row. The two differ, and an
earlier revision of `SMOKE_TEST_RESULTS.md` recorded tag objects as commits — which would have
looked like a provenance mismatch against the image's own recorded SHAs during a reproducibility
audit (§36).

**Rule:** every §28.2 manifest field naming an upstream revision records the commit, obtained
from `git rev-parse HEAD` inside the build (as `opencv-base.Dockerfile` already does) or from
the `^{}` row of `git ls-remote --tags <url> '<tag>*'`. Never from `--refs`.

Correct commits for OpenCV 5.0.0: core `40738fb16ceddb5fb3fea747585f7ce6abb0605b`,
contrib `755e50675d97db9b7d449d8bd6b09888646f6c6e`.

## 2026-09-06 — D8: contrib build set is exactly C7's four modules

`xfeatures2d` was in the first `BUILD_LIST` speculatively and **removed**: nothing in the
architecture uses SURF/BRIEF/FREAK/DAISY, and it fetches `boostdesc`/`vgg` blobs at *configure*
time from a slow third-party host, which stalled the first build entirely. Built contrib set is
`ximgproc`, `xphoto`, `optflow`, `quality` — exactly the modules C7 names.

## 2026-09-06 — D9 `[OPEN — needs maintainer decision]`: dust is composited after the sensor chain

§5.6 states the degradation chain in this order:

```
radiance scale -> shot noise -> read noise -> PRNU/vignette
  -> exposure/gain -> quantization/clipping -> optional dust compositing
```

`Degrade()` implements that literally, because the doc wins. Recording the physical
objection so the choice is deliberate rather than accidental:

**Dust is a scene-level effect and physically precedes the sensor.** Airborne particulate
attenuates scene radiance and adds airlight *before* photons reach the detector. Applying
`I = J*t + A*(1-t)` after quantization makes dust a pure post-process, which loses two real
couplings: attenuation should REDUCE photon count and therefore reduce shot noise, and
airlight should ADD photons and therefore add its own shot noise. In the implemented order a
heavily dusted pixel carries the shot-noise statistics of the clean radiance it no longer has.

**Why it matters beyond tidiness:** Phase 5 measures each radiometric stage's effect on
repeatability, match precision and ATE (§31.5). A dehazing or contrast-recovery stage
evaluated against post-sensor dust is being tested against a strawman that is easier to
invert than real dust, because the noise field it must fight was generated from
undegraded radiance.

The same objection applies, more weakly, to PRNU and vignette sitting after read noise:
vignetting is optical and precedes photon conversion, and read noise is added by the readout
circuit and is not vignetted. The implemented order multiplies read noise by the vignette
field.

Options: (a) keep §5.6's order and note the approximation in the report; (b) amend §5.6 to
the physical order under the §49 ritual and re-record all golden digests; (c) implement both
behind a flag and measure the difference once, which turns the question into a number.
Recommendation is (c) then (b) if the difference is material. Not blocking: the operator is
deterministic and self-consistent either way, and no published result depends on it yet.

## 2026-09-06 — D10: one keying discipline for all seeded fields

Every seeded value in the degradation chain is keyed as
`(seed, frame_id, site)` where `site` encodes pixel, channel and stage, and
multi-octave noise folds the octave index BENEATH a caller-owned `salt_base`
(`salt = salt_base * 64 + octave`).

Fixed a violation of this in the dust field, which had been keyed on
`frame_id ^ NoiseStage::kDustField`. Mixing a stage id into the frame dimension
permutes frames against one another (frame 0 and frame 3 exchange fields), and
the `salt` argument was already consumed by the octave index, so any second
caller would have collided with octaves. Stage now travels in `salt_base` and
`frame_id` passes through unmodified.

Consequence: the degradation golden digest changed from `9169008833890659974`
to `12016663175328231476`. No published result depended on the old value. The
determinism gate caught the change and refused, which is the intended behaviour.

## 2026-09-06 — D11 `[PROXY — grouped with D9 for decision]`: the dust field is temporally white

The particulate field is keyed on `frame_id`, so it is statistically independent
from frame to frame. Real airborne dust advects and evolves over seconds, so
consecutive frames are strongly correlated.

This is the same class of modelling error as drawing PRNU per frame, which is
explicitly avoided (see `kFixedPatternFrameId`): a temporally white field can be
removed by simple frame averaging, so a dehazing or contrast-recovery stage
evaluated against it is being tested against an unrealistically easy target. The
Phase 5 DVE ablation (§31.5) is where this would distort a result.

Options: (a) static field per sequence, correlated but not advecting; (b)
interpolate between fields keyed on `frame_id / N` for a chosen N, giving a
tunable correlation time; (c) keep it white and state the limitation. (b) is
cheap and closest to the physics. Grouped with D9 because both are questions
about how faithful the dust model needs to be before Phase 5 depends on it.

## 2026-09-06 — D12 `[OPEN — needs maintainer decision]`: §15.7 size table vs measurement

Measured on `cpp/core/serialization/wire.fbs` (envelope overhead 96 B excluded by
differencing against an empty graph; messages fully POPULATED, see the trap below):

| Element | §15.7 predicts | Measured | Delta |
|---|---|---|---|
| Pose-graph node | 28 B | 65.8 B | +135% |
| SE(3) pose edge | 150 B | 193 B | +29% |
| Graph 200n+40e, uncompressed | ~12 KB | 21.0 KB | +75% |
| **Graph 200n+40e, zstd-3** | ~12 KB | **4.3 KB** | **-64%** |

### The finding that reframes this

**Pose graphs compress 4.93x** (21008 -> 4260 B at zstd level 3). Node fields are
near-sequential keyframe ids, closely spaced timestamps and smoothly varying poses, which is
almost ideal for an entropy coder. §15.7 currently compresses classes 4-5 only
(MATCH_RESPONSE, POSE_EDGE) and leaves POSE_GRAPH uncompressed.

So the real gap is not that the schema is bloated. Uncompressed it is 75% over the
prediction; compressed it is 64% under. Either way §15.7's 12 KB is not what the link
carries, and the direction depends entirely on a compression policy the doc sets elsewhere.

**This strengthens rather than weakens the §15.7 thesis** that front-end descriptor traffic
dominates back-end traffic by orders of magnitude: back-end traffic is even cheaper than
predicted once compressed.

### Correction to an earlier version of this entry

An earlier revision claimed 28 B/node was "unreachable" because §15.5 requires frame-tagged
poses. That was too strong. The tag need not be per node: `PoseGraph` already carries a
batch-level `anchor:FrameTag`, and nodes from one agent share a frame, so hoisting the tag to
the batch satisfies §15.5 while removing 8 B/node. Combined with a `struct` node (no vtable,
-10 B) and delta-encoded ids and timestamps, 28 B is plausibly reachable.

### The actual question for the maintainer

Is 28 B a *prediction* of what this schema costs, or a *design target* the schema must meet?

- If **prediction**: amend §15.7 to the measured values under the §49 ritual, and state the
  compression policy per class alongside them. Cheapest, and the numbers become honest.
- If **target**: apply the levers in increasing cost of properties surrendered:
  1. compress POSE_GRAPH (zero schema change, measured 4.93x, already exceeds the target);
  2. hoist `frame` and `agent` to batch level (-12 B/node, no property lost);
  3. `struct` instead of `table` (-10 B/node, forfeits schema evolution that §15.7 wants);
  4. varint ids and quantized rotations (large, costs precision and readability).

Recommendation: 1 and 2, then amend §15.7 to the resulting measured numbers. Lever 1 alone
takes the graph well below the prediction, so 3 and 4 are unnecessary for now.

### Measurement trap, recorded because it nearly published a wrong number

A first measurement put the edge at 148.8 B, apparently confirming the prediction. It used
default-constructed edges, and **FlatBuffers omits any scalar equal to its schema default**,
so unset fields cost nothing. Populating the message realistically gives 193 B. Every Phase 3
byte measurement must use populated messages.

## 2026-09-06 — D13: an uncompressed frame header precedes every payload

`Decode` previously required the caller to supply the message class in order to know whether
to decompress, but the class also lived inside the envelope: a receiver could not read it
without already knowing it. §15.3's store-and-forward triages queued messages by priority
class and TTL and must do so without decompressing, so class and compression flag now travel
in a 6-byte uncompressed header (`class`, `flags`, `payload_length`), readable via
`PeekFrameHeader`. `wire_bytes` now includes it.

Also: a compressed form is kept only when it is actually smaller. High-entropy descriptors can
expand under zstd, and transmitting the larger form would corrupt the byte table.

## 2026-09-06 — D14 `[DECIDED]`: POSE_GRAPH is compressed, extending §15.7's classes 4-5

§15.7 applies zstd-3 to classes 4-5 (MATCH_RESPONSE, POSE_EDGE) and leaves POSE_GRAPH
uncompressed. Measured, that leaves a 200-node/40-edge batch at 21.0 KB against a 12 KB
prediction. Compressed it is **4.3 KB**, a 4.93x ratio, comfortably inside the budget.

Pose graphs compress well because their content is structurally redundant: keyframe ids are
near-sequential, timestamps are evenly spaced, and consecutive poses differ slightly. This is
close to the best case for an entropy coder, and unlike descriptor payloads there is no
high-entropy content to defeat it.

**Decision:** `IsCompressedClass` covers MATCH_RESPONSE, POSE_EDGE and POSE_GRAPH. A
compressed form is kept only when smaller, so a pathological payload cannot inflate traffic.

Consequences:
- The Phase 0 gate's recorded traffic fell from 12632 B to 8626 B for 200 poses.
- D12's headline gap closes: the graph now lands under the §15.7 prediction rather than 75%
  over it. The per-element figures (node 65.8 B, edge 193 B) are unchanged and still differ
  from §15.7's 28 B / 150 B, so **D12 remains open** for the doc amendment.
- `PoseGraphsRemainHighlyCompressible` asserts the ratio stays above 3x, so a schema change
  that destroys compressibility (random uuids for ids, say) fails here rather than as a blown
  byte budget in Phase 3.

This extends §15.7 rather than contradicting it, and needs the §49 ritual to land in the
architecture doc.

## 2026-09-06 — D15 `[DECIDED by maintainer]`: XFeat is implemented on OpenCV 5 primitives, not run through ONNX

Resolves D6. XFeat has no implementation in OpenCV 5.0.0 despite `ALGORITHMS.md` L63 claiming
`cv::XFeat` exists. Two routes were available: export upstream's PyTorch weights to ONNX and
run them through `cv::dnn`, or implement the network directly on OpenCV 5 primitives.

**Chosen: implement on OpenCV 5 primitives.** The competition rewards substantive technical
OpenCV 5 usage, and a hand-built modern learned extractor running on OpenCV 5 is a far
stronger claim than loading someone's ONNX file through a general-purpose runtime. It also
removes the `cv::dnn` ONNX-coverage risk flagged for XFeat's keypoint head entirely.

**What this does and does not mean.** It is a from-scratch INFERENCE implementation. Training
is not attempted: `ALGORITHMS.md` L124 records that training needs COCO synthetic warps plus
MegaDepth on multiple GPUs. Weights come from upstream `verlab/accelerated_features`
(Apache-2.0, `weights/xfeat.pt`, 6.2 MB), extracted to a plain tensor format we load. The
report must say exactly this: the architecture and forward pass are ours, the weights are
upstream and credited. Claiming more would breach the §44 claims discipline.

Scope estimate from the catalogue (L124): ~10 conv blocks, "a weekend in C++/Eigen". Work
items when Phase 2.1 opens:
1. Extract weights from `xfeat.pt` to a versioned binary blob plus a manifest of shapes.
2. Implement the forward pass: conv/BN/ReLU blocks, the depth-redistribution trunk, the
   1x1 keypoint branch over an 8x8 tensor-block-transformed image, and the 64-D dense
   descriptor head.
3. Validate against upstream PyTorch outputs on fixed inputs, to a stated tolerance. This is
   the acceptance test: without it, "we implemented XFeat" is unverified.
4. Benchmark against the catalogue's published figures (L103, L115: 16.7 ms class) and report
   the delta honestly, including if ours is slower.

Risk: this is a Phase 2.1 critical-path item with real depth. If step 3 does not converge,
the fallback is the ONNX route (unchanged in cost) or promoting ALIKED, which is native.
Record whichever ships in the run manifest.

## 2026-09-08 — D16: bags store FRAMED messages, and chunk compression is a storage choice

`McapRecordWriter` stores the framed wire message (frame header plus
possibly-compressed FlatBuffer), not the raw payload. Replay therefore exercises the
identical decode path, frame-tag validation and byte accounting as a live run, so a
wire-format bug cannot hide behind a different on-disk representation.

MCAP chunk compression is set to zstd. This is a FILE-SIZE decision with no effect on wire
byte accounting: bytes are counted at the serializer (§4.4), and a bag is storage rather
than a link. Classes 4-6 arrive already compressed, so the additional gain falls mostly on
the uncompressed classes.

Log time comes from the message stamp, never a wall clock. A wall clock would make every
bag unique and destroy the reproducibility §5.9 exists to provide. Asserted by
`LogTimesComeFromMessageStampsNotWallClock`.

Measured: bags written twice from identical inputs are **byte-identical**, which is stronger
than the Phase 0 gate requires (identical metrics) and is what makes a bag checksum in a
§28.2 manifest meaningful. Also measured: replaying through MCAP reproduces the record-file
gate exactly (ATE 0.0633155, RPE 0.0174459, 8626 wire bytes), so adopting the container
costs no determinism.

LZ4 is disabled (`MCAP_COMPRESSION_NO_LZ4`): nothing selects it and vendoring a codec we
would never use adds a dependency for nothing.

## 2026-09-08 — D17: Philox counter is PARTITIONED, and `site` is bounded to 48 bits

`PhiloxStream` mixed its refill block index in with `ctr.v[3] ^= block`, while the
constructor wrote `site >> 32` into that same word. The two therefore aliased:
`(site=0, block=1)` and `(site=2^32, block=0)` produced an identical counter and
byte-for-byte identical output. The code comment claimed a site could draw
"without ever colliding with a neighbouring site's stream", which was false.

Philox4x32's counter is 128 bits and has to hold `frame_id`, `site` and `block`,
so one of the three must be bounded — there is no layout that gives all three
64 bits. The options considered:

| Layout | Cost | Verdict |
|---|---|---|
| Steal `block` from `frame_id`'s high word | `site` keeps 64 bits | **rejected**: `kFixedPatternFrameId` is the all-ones sentinel, so `frame_id` genuinely uses all 64 bits |
| Bound `block` only | would cap draws per site very low | rejected |
| **`site` < 2^48, `block` < 2^16, `frame_id` keeps 64** | 262144 words per site | **chosen** |

`SiteId` for a 1280x720x3 image with 4 noise stages peaks around 11M, four orders
of magnitude inside 2^48. `LatticeNoise` hashes a 2D cell to 64 bits and is now
masked to 48; among the ~10^3-10^4 cells one image touches the birthday collision
probability in 2^48 is ~1e-11, and a collision would mean two lattice cells sharing
a value, not a correlated stream.

The map from `(frame_id, site, block)` to the counter is injective on the declared
domain. **Corrected same-day:** the first version of this entry claimed the bounds
were "asserted, so the guarantee is structural". `assert` is compiled out under
`NDEBUG`, which this project's `Release` build defines -- so in the shipped binary
neither bound was checked, `site = 2^48` aliased onto `site = 0`, and block 65536
wrapped onto block 0 and reused the stream. That is the same defect class the entry
was written to close, reintroduced by its own fix. Now:

- `site` is **masked in the constructor** in every build. It is a documented 48-bit
  identifier; passing more bits is defined truncation, not an aliasing surprise.
  Callers holding a 64-bit id (Phase 1 keyframe/landmark/submap hashes) must use
  `SiteFromHash64()`, which mixes all 64 bits down with a splitmix64 finalizer so
  the high bits still matter.
- Block exhaustion **throws**, in every build. Wrapping would silently repeat a
  site's numbers, and a determinism substrate that quietly repeats itself is worse
  than one that stops. Cost is one comparison per four draws.

Note the asymmetry that hid this: `check_determinism.sh` compiles with bare
`-O0/-O2/-O3` and no `-DNDEBUG`, so asserts are live in the gate and dead in the
release binary. The gate structurally could not have caught it. This matches how Random123-derived APIs partition counter words
(JAX's `fold_in`/`split`, cuRAND's Philox usage) rather than XOR-mixing them.

Consequence: `philox_digest` and `degrade_digest` changed. Nothing published
depended on the old values.

## 2026-09-08 — D18: golden digests must evaluate the boundaries they claim to guard

`DetExp` returned `inf` for x in [709.44, 709.78] and `0` for x in [-745.13, -745.0]
— inside its own guard, where the true results are finite and representable. The
range reduction produces k=1024 (resp. -1075) there and `Exp2Int` saturates. Fixed
by splitting the scaling so neither factor leaves the normal range; now within
1 ulp of libm across both bands.

The decision worth recording is not the fix but what it exposed. After fixing both
`DetExp` and Philox, **two of the three golden digests were unchanged**:

- `det_math_digest` swept x in [-500, 500] only — the bug lived outside it entirely.
- `philox_digest` used sites < 2^32 drawing 2 words each, so `site >> 32` and `block`
  were both always zero — it never varied the fields that aliased.

Both digests were byte-identical across genuinely broken code, and would have stayed
green through a release. **A digest guards only the inputs it evaluates.** The sweeps
in `check_determinism.sh`, `test_det_math.cpp` and `test_philox.cpp` now include the
overflow/underflow boundaries and high-site/multi-block draws respectively.

Standing rule from this: when a function's contract names a domain, the digest
sweep must reach the ends of that domain, not just its comfortable interior.

## 2026-09-08 — D19: CI determinism gate must not be piped

`run: ./infra/scripts/check_determinism.sh | tee determinism-$arch.txt` was green on
failure. GitHub's default shell is `bash -e {0}` **without** `pipefail`, so the step
exited with `tee`'s status; verified locally that even a compilation failure inside
the script yielded exit 0. Replaced with redirect, capture status, `cat`, `exit`.
`if: always()` added to the artifact upload, since the log matters most when the gate
failed.

Related correction: `cross-arch-determinism` was commented as "THE JOB THAT MATTERS".
With per-arch assertion working it cannot fail — both arches are compared against the
same committed constants, so reaching the job implies they agree. It is now described
as a second, independent comparison, runs with `if: always()`, and the script emits a
machine-readable `det=... philox=... degrade=...` line on **both** its pass and fail
paths so two failing runs can still be diffed. Previously the failure path printed no
`=` at all, so the extracting `grep` matched nothing and the job died on an empty
match rather than reporting the mismatch.

## 2026-09-08 — D20: `poisson_exact_max_lambda` is bounded at both the config and the sampler

The field was config-exposed and unvalidated. Knuth's exact branch is hard-capped at
10000 iterations; setting the threshold high enough would make every bright pixel
return exactly 10000 photons — a plausible-looking image built on fabricated
statistics, with no test failing.

Bounded to (0, 50] in `ValidateDegradation`, and clamped again inside `SamplePoisson`
for callers that do not come through a config. The Gaussian approximation is already
accurate by lambda=30, so nothing is lost. With the clamp, `target = e^-lambda > 1e-22`
and the 10000-iteration cap is unreachable short of a broken RNG, which turns it from
a routine truncation path into a genuine guard.

`dust_field_frequency` is now required `> 0` for the same reason: zero maps every pixel
onto lattice cell 0, so the field silently becomes a spatial constant while still
reading like a field in the config.

## 2026-09-08 — D21: manifest required-field set now covers provenance, not just identity

`MissingRequiredFields` returned empty for a manifest whose `seed`,
`degradation_config`, `execution_mode` and all three determinism digests were
`UNRECORDED`. §5.6 requires degradation parameters to be named configs *referenced by
run manifests*, and the digests exist "so a result produced by drifted math is
identifiable after the fact" — both guarantees were empty as written.

All six added to the required set. `seed` needed a separate `seed_recorded` flag
because 0 is a legitimate seed and the value alone cannot distinguish "seeded with 0"
from "nobody recorded it".

## 2026-09-08 — D22: `EventLog` chain hash must not use zero as a sentinel

`ChainHash` began with `state_hash_ != 0 ? state_hash_ : kFnvOffsetBasis`, using
zero to mean "no previous event". Zero is a legitimate hash value, so a chain that
happened to produce it would silently RESTART, and two different histories could
converge onto the same hash from then on. That destroys the property the chain
exists for: localizing WHERE two runs diverged, not merely that they did.

Replaced with a `chain_` member seeded at construction and always carried forward.
`state_hash_` still reports 0 before the first event, so the public surface is
unchanged. Same bug shape as treating `seed == 0` as unset in the run manifest
(D21), which is why both were fixed in the same pass.

## 2026-09-08 — D23: framing is charged per DATAGRAM, not per message

Arch §15.7 (line 2699) specifies "28 B emulated UDP/IP header **per datagram**".
The serializer charged it once per message, so any message exceeding the path MTU
was under-billed. Doc wins per CLAUDE.md §6.

Added `kPathMtuBytes = 1500`, `kDatagramPayloadBytes = 1472` and `DatagramCount()`;
`wire_bytes = framed + 28 * DatagramCount(framed)`.

Measured effect — it engages exactly where the byte table is load-bearing:

| Message | framed | datagrams | wire before | wire after |
|---|---|---|---|---|
| pose graph 200n/40e (compressed) | 4266 | 3 | 4294 | **4350** |
| Phase 0 gate, 4 messages total | — | — | 8647 | **8759** |

Small messages are unaffected (one datagram, 28 B, as before). The correction is
~1.3% on the compressed pose graph, so it does not threaten §15.7's <=15%
prediction-error gate — but the Phase 3 spectral policy optimizes
`gain / predicted_cost`, and systematically under-pricing the largest classes is
exactly the bias that would distort which messages it chooses to send.

Checked for a second mismatch, since Phase 3's gate measures against §15.7's table:
**the table is payload-only.** Its rows are encoding arithmetic with no framing term
at all (200 nodes x 28 B + 40 edges x 150 B = 11600 B, the printed "~12 KB"), and the
28 B appears only in the formula line, which already says "per datagram". So the fix
aligns the code with both the formula and the table; it does not trade one mismatch
for another.

Assumption recorded: 1500 B Ethernet MTU with no jumbo frames, and IPv4 (20 B)
rather than IPv6 (40 B). Both are `[ASSUMPTION]` and should be revisited when the
real link model lands in Phase 3.

## 2026-09-08 — D24: `execution_mode` is validated by membership, not presence

`MissingRequiredFields` called `require("execution_mode", ...)`, but the field
defaults to `"replay"` -- non-empty and not `UNRECORDED` -- so the check could never
fire. It read as a guarantee and verified nothing.

§5.9 makes the distinction load-bearing: replayed runs are bit-deterministic, live
closed-loop runs are not and must be reported over seeds. What matters is therefore
that the value is one of the two known modes, which is now what is checked.

**Follow-up for Phase 1:** nothing in the tree populates the six newly-required
manifest fields yet. Whoever writes the first real manifest writer must wire
`seed_recorded`, `degradation_config` and the three determinism digests, or the
first real run will correctly report itself unpublishable.

## 2026-09-29 — D25: sim image installs the ros_gz bridge/sim subset, not the `ros-jazzy-ros-gz` metapackage

Arch §5 says "install `ros-jazzy-ros-gz` for the bridge and sim launch integration". Measured
2026-09-29: the metapackage pulls `ros_gz_image -> image_transport_plugins -> cv_bridge`, which
installs **Ubuntu's OpenCV 4.6 (43 packages)**. That puts a second OpenCV in `ariadne-sim`, against
C7's "one OpenCV, built from the pinned tag", and makes a sim-image manifest's `opencv_build_hash`
ambiguous.

The image installs `ros-jazzy-ros-gz-bridge`, `ros-jazzy-ros-gz-sim` and `ros-jazzy-ros-gz-interfaces`
instead: **zero** OpenCV packages, and still gz-sim8 8.15.0 / gz-rendering / gz-sensors / sdformat14 /
ogre-next. This serves the doc's stated purpose (bridge + sim launch). Camera frames cross as
`sensor_msgs/Image` through `parameter_bridge`, which needs no OpenCV. What is lost is
`ros_gz_image`'s image_transport compression plugins, which Ariadne is not planning to use.

**Open for Phase 1.3 `[DESIGN GAP]`:** rosbag2 records CDR-serialized `sensor_msgs/Image`
(verified in the image), but `McapRecordReader` (D3) replays Ariadne-framed FlatBuffers on
`/ariadne/wire`. Nothing converts between the two yet. 1.3 must decide whether sim bags stay
rosbag2/CDR and a ROS-free CDR reader is added, or a recorder node writes the Ariadne wire form
directly. The Dockerfile fails the build if any OpenCV package
ever enters the image through a transitive dependency.

Also recorded: ROS packages come from the dated snapshot `snapshots.ros.org/jazzy/2026-09-11`,
signed by "ROS Snapshot builder <rosbuild@ros.org>", fingerprint
`4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA`. The fingerprint was verified against keyserver.ubuntu.com
and against the snapshot's own InRelease signature, and the key is vendored at
`infra/docker/keys/ros-snapshots.asc` with a build-time fingerprint check. The live ROS repo keeps
only the newest build of each package, so `pkg=version` pins break on its next sync. The snapshot
does not move.

## 2026-09-29 — D26: AWS runs on the free plan, ~$100-200 of credits, in one standalone account

The grant request did not go through. The account (`836505524290`, `ap-southeast-2`) is on the
AWS **free plan**. Consequences, from AWS's own documentation:

- **Never join AWS Organizations or set up Control Tower.** Either upgrades the account to paid
  and **expires the free credits immediately** (Free Tier FAQ). This rules out org-instance IAM
  Identity Center, and with it SSO permission sets. The account-instance Identity Center that was
  created does not support permission sets (IAM Identity Center docs) and is unused.
- The free plan **closes the account** when credits run out or after 6 months (data kept 90
  days), and excludes "certain AWS Marketplace offers that can incur charges" — which may include
  the COOL AMI. `[UNVERIFIED]`: attempt the COOL subscription to find out.
- Credits: $100 at signup, up to $100 more from activities.

Setup chosen (standard for a standalone account):

| Concern | Choice |
|---|---|
| Human access | IAM user + MFA; `aws login --remote` (CLI 2.37.5) for short-lived credentials, no access keys |
| Terraform credentials | provider cannot read the `aws login` cache (terraform-provider-aws #45316); `infra/scripts/aws.sh` exports current short-lived credentials into a mode-600 env file per run |
| CI | GitHub OIDC -> IAM role scoped to ECR push on the project repos. Zero stored secrets |
| State | S3 bucket `ariadne-tfstate-836505524290` (versioned, SSE, TLS-only, public access blocked), `use_lockfile = true`, Terraform >= 1.11; bootstrapped with local state then migrated |
| Tooling | official `amazon/aws-cli:2.37.5` and `hashicorp/terraform:1.16.0` images, pinned by digest |

**OIDC subject.** GitHub repos created after 2026-07-15 use the immutable subject
`repo:OWNER@OWNER_ID/REPO@REPO_ID:ref:...` (GitHub OIDC reference). The owner id (`95163434`) is
pinned. The repo id is wildcarded because the repository is private and its id was not readable
without credentials. Tighten it once the first CI run shows the actual `sub` in CloudTrail.

**Budget pulled forward from Phase 7.** The account closing on credit exhaustion would take the
bags, ECR images and judging console with it. The budget uses `include_credit = false`: a cost
budget nets credits by default, which would hold it at $0 so it never fires. Thresholds are 25/50/80%
actual plus 100% forecast, and the limit is a variable (100 now, 200 after activities).
`[UNVERIFIED]` whether Budgets sees gross spend on the free plan; check after the first billable run.

**The budget is the backstop, not the brake.** Budget data lags by hours, and on a GPU instance
the credits are gone before the email arrives. Every Phase 7 launch template MUST set
`instance_initiated_shutdown_behavior = "terminate"` and a max-lifetime shutdown in user-data.

**Locked-decision tension, for the maintainer.** CLAUDE.md §1 puts Gazebo on g5/g6. That may not fit
the credit pool. Record-once rendering could move to local Mesa llvmpipe (proven at 320x240 / 5 Hz,
unmeasured at 1280x720 stereo), but gsplat needs CUDA regardless. The maintainer decides once real
prices are pulled.

**D26 addendum (2026-09-29), credit activities done.** All five Free Tier credit activities were
COMPLETED and the balance is **$200.00** (FREE plan). EC2, Budgets, Lambda and RDS were done via the CLI;
completion is tracked by API calls, not console clicks. Bedrock was done in the console playground.
All temporary resources were deleted and the account verified empty of billable resources.
Two follow-ups:
- The budget `ariadne-free-credits` ($100, gross spend, 25/50/80% + 100% forecast) was created via CLI.
  Before the main stack's first apply: `terraform import module.budget.aws_budgets_budget.credits
  836505524290:ariadne-free-credits`, and raise `credit_budget_usd` to 200.
- Bedrock `converse` via the API returned `Operation not allowed` / `authorizationStatus: NOT_AUTHORIZED`
  for Nova Micro on the free plan, although the model is region-available. The Phase 7 orchestrator
  needs API access: re-test after the plan upgrade. `[UNVERIFIED]` cause.

## 2026-09-29 — D27: tests-only AWS usage; cheapest configurations that run each workload

The account is on the PAID plan with $200 of credits (upgraded via the console button, so the credits
were kept). Graviton4 launches were verified after the upgrade. AWS is used ONLY for short, self-terminating
test runs; development, CI and simulator bag rendering stay local.

| Workload | Config | Sydney $/h (AWS price data 2026-09-29) | Quota |
|---|---|---|---|
| COOL / Arm CPU pipeline | **c8g.large** (Graviton4, 2 vCPU, 4 GiB) | 0.1038 on-demand, + $0.01 COOL software fee | Standard 5 vCPU: fits |
| GPU: gsplat, NVIDIA EGL rendering | **g4dn.xlarge spot** (T4 16 GB, 4 vCPU) | 0.2622 spot (0.684 on-demand) | G/VT 0: **request 4** |

Why these and not smaller:
- GPU quotas count vCPUs, and g4dn.xlarge (4 vCPU) is the smallest standard NVIDIA type with a full GPU.
  The only 2-vCPU GPU, g6f.large (1/8 L4), costs the same ($0.2626 on-demand) but has ~2.8 GiB VRAM and
  needs GRID drivers with unverified headless EGL. That's the wrong trade for gsplat.
- c8g.large is the smallest size on COOL's published price list (c8g.medium is not among the visible
  dimensions).

**Locked-decision tension, for the maintainer.** CLAUDE.md §1 / Invariant 2 name **m8g.4xlarge** for the
COOL benchmark, and §27.1's concurrent-pipeline "knee" sweep is not meaningful on 2 vCPUs. c8g.large is
right for development and smoke runs. Sizing the FINAL benchmark is the maintainer's call. m8g.4xlarge needs
16 vCPU, above the current Standard quota of 5, so a request would be needed.

**Biggest cost unknown:** whether local Mesa llvmpipe renders 1280x720 stereo at a usable rate. If it does,
the GPU is needed for gsplat only. Measuring it is free.

Mechanism: `infra/scripts/spinup.sh run --profile cpu|gpu -- '<cmd>'`. It uses spot by default; on-demand only via
an explicit flag, never as a silent fallback. There are three independent brakes: (1) `shutdown -h +MAX` as the
first user-data line, with shutdown meaning terminate; (2) a host trap that terminates on exit/Ctrl-C; (3) `sweep`,
which terminates stray instances past their deadline and lists unattached volumes. Root volumes are forced to
`DeleteOnTermination=true`. IMDSv2 is required, with hop limit 2 so containers can reach the instance role. The
role can only read/write `runs/` in the artifact bucket and pull from the project's ECR repos. Debugging uses SSM,
never SSH. Every run appends type, market, AZ, duration, the actual spot price and cost to `build/runs/ledger.csv`,
which is also the cost evidence the COOL rubric asks for.

~~`[UNVERIFIED]` until the first real run~~ **Verified 2026-09-29** (run `20260929T145114Z-cpu-762cb1`): AWS accepts
spot + `instance-initiated-shutdown-behavior=terminate`, and an in-instance `shutdown -h` ends in `terminated`
(reason `Client.InstanceInitiatedShutdown`), not `stopped`. Two silent-exit bugs were found on the way and fixed
(a `read` at EOF-without-newline under `set -e`, and a SIGPIPE-prone `sort | while break`). The script now has an
ERR trap that names the failing line.

Open: the Bedrock API (`converse`) still returns `Operation not allowed` after the upgrade and as the IAM user
(so neither the free plan nor root identity is the cause). Needed by the Phase 7 orchestrator; to investigate.

## 2026-09-29 — D28: region is eu-north-1 (Stockholm), the cheapest one that meets every need

Requirements, all in ONE region (cross-region traffic would distort cost-per-frame): c8g.large (Graviton4,
COOL), g4dn.xlarge (T4 16 GB), the COOL Graviton4 AMI, and Bedrock with Anthropic models.

Method: all 18 enabled regions scanned via the live API (offerings, current spot, Bedrock model lists).
Then 7-day spot history (median / p90) for the leaders, plus on-demand from AWS's published price data.

| Region | c8g.large spot 7d median | g4dn.xlarge spot 7d median (p90) | on-demand c8g / g4dn |
|---|---|---|---|
| **eu-north-1** | **0.0206** | **0.0814** (0.0899) | 0.0851 / 0.558 |
| sa-east-1 | 0.0342 | 0.0973 | 0.122 / 0.894 |
| ap-south-1 | 0.0262 | 0.2121 | 0.054 / 0.579 |
| eu-west-3 | 0.0446 | 0.1906 | 0.094 / 0.615 |
| ap-southeast-2 (previous choice) | ~0.046 | ~0.296 | 0.104 / 0.684 |

ap-south-2 (Hyderabad) has the cheapest Graviton spot (0.011) but no g4dn at all, so it's excluded.
ap-south-1 (Mumbai) has the cheapest on-demand Graviton: if the final COOL benchmark runs on-demand on
m8g.4xlarge, Mumbai is $0.51/h against Stockholm's $0.76/h. Worth revisiting for that single run.

- **COOL availability:** the listing page shows no region list, but Marketplace AMIs are public EC2 images.
  Product code `aajkmdd4qo3r7yhqg61a7aah9` (image `Graviton5-COOL-v3-prod-k6u24vxijyzvg`, where
  `prod-k6u24vxijyzvg` is the Graviton4 listing) is published in **all 18 regions**; in eu-north-1 it's
  `ami-0a302ff85af8d8764`. (The earlier "0 images" result came from a case-sensitive name filter guess.)
- **Spot placement scores were discarded.** AWS returns an artificially low score when fewer than three
  instance types are specified, and the query named one type. Real capacity is proven only by a real launch.
- **Bedrock "Operation not allowed" explained:** in eu-north-1 the same call returns "Your account is
  currently being verified ... normally takes less than 2 hours". That's post-upgrade account verification,
  not a configuration problem.
- **Absolute savings are small** (a few dollars against $200 on tests-only use). The region choice matters less
  than spot capacity and single-region co-location.
- **Quotas are per region:** eu-north-1 has Standard 5 vCPU, and G/VT 0 for both on-demand and spot.

## 2026-09-29 — D29: minimal network: the region's default VPC, one egress-only security group

The maintainer asked for the network to be as minimal as possible to keep costs down. The main stack no longer
creates a VPC. It uses eu-north-1's **default VPC** (172.31.0.0/16, one public subnet per AZ, internet
gateway), which every region ships with, free and already present. The only network object Ariadne
creates is `ariadne-test-runner`, an egress-only security group with no inbound rules; debugging goes
through SSM, not SSH.

Deliberately not used: NAT gateways (~$0.045/h each, the classic silent cost), Elastic IPs, interface
endpoints, and a custom VPC. Runners use their auto-assigned public IPv4 (billed only while an instance runs).
`modules/network` is kept, unused, in case the Phase 7 hosted console justifies its own VPC.

`spinup.sh` launches spot runs into the AZ with the **lowest current spot price** (it reads all three
AZs before launch), since spot prices differ per AZ.

Applied 2026-09-29: bootstrap (6 resources; state migrated to `s3://ariadne-tfstate-836505524290/bootstrap/`)
and main (1 imported, 23 added, 1 changed, 0 destroyed; post-apply plan converged). Verified by that
apply: AWS accepted the GitHub OIDC provider without a `thumbprint_list`. The CLI-created budget was
adopted with a declarative `import` block, because `terraform import` failed on the default-subnet
`for_each`. The block was removed after the apply.

GPU quota requests filed in eu-north-1 (quotas are per region): G/VT spot and on-demand to 4 vCPU,
both PENDING at filing.

## 2026-09-29 — D30: a GPU is needed ONLY for gsplat; the GPU quota is 0 and the spot request was denied

- **Quota, authoritative:** `Running On-Demand G and VT instances` and `All G and VT Spot Instance Requests`
  are both applied **0** in eu-north-1, and the AWS default is also 0. The account has no GPU access today.
  **Both requests were denied** on 2026-09-29 with "ramp up gradually": case 179069316400187 (spot, 4 vCPU)
  and case 179069316600303 (on-demand, 4 vCPU). The quota API still reported both as `CASE_OPENED` after the
  denial emails, so the API status lags the support case; the applied value (0) is what counts. The appeal
  text for both cases is in `docs/aws_quota_appeal.md`.
- **What needs a GPU:** gsplat (CUDA-only). The Phase 1 exit gate includes "render a splat", which puts a GPU
  on Phase 1's critical path, not just Phase 6's.
- **What does not:** bag rendering (SMOKE_TEST_RESULTS §4f: software rendering records the full rig
  completely at RTF 0.57), SLAM, COOL, Bedrock.
- CLAUDE.md §1 places Gazebo on g5/g6. §4f shows that is not necessary for record-once bags, so the GPU
  requirement shrinks to gsplat alone. The locked decision is not changed here; this only records the evidence.
- Fallbacks if both cases fail (e.g. off-AWS GPU time for gsplat) conflict with the locked "gsplat service on
  AWS g5/g6" and are the maintainer's call.

## 2026-09-29 — D31: §30.2 becomes a code registry; rotational ATE added; "rotation/translation error" interpreted

`cpp/runtime/metrics/metric_registry.hpp` holds the 27 §30.2 metrics (4 localization, 3 registration, 6 mapping,
7 communication, 7 mission) as stable ids with units. `MissingMetrics()` treats absent AND non-finite values as
missing, since "emits a number, however bad" must not be satisfiable by a NaN. `UnknownMetrics()` rejects
unregistered ids, so a typo is an error rather than an orphaned value. The Phase 1 exit gate becomes
`MissingMetrics(report).empty()` on the thin run's report.

`[ASSUMPTION]` §30.2's localization line ("ATE RMSE, RPE, rotation error, translation error") does not say which
quantity each error refers to. It is read the way evo reports trajectories: translation and rotation parts of
both ATE and RPE (`ate_rmse_m`, `ate_rotation_rmse_rad`, `rpe_translation_rmse_m`, `rpe_rotation_rmse_rad`).

`ComputeATERotation` closes the audit carry-over. It uses the same position-based Umeyama alignment as
`ComputeATE` (factored into `AlignTrajectories`, so the two can never describe different registrations), then
takes the angle of R_gt^T·(R_align·R_est) per pose, which is evo's APE rotation part. The test that motivates it:
positions exact, every orientation off by 0.05 rad gives translation ATE = 0 and rotation ATE = 0.05.
`energy_proxy` carries no invented unit; §30.2 gives none, and its definition lands with the planner (4.3).

## 2026-09-29 — D32: determinism digests exist once, and the run manifest measures them

Each digest (det_math, Philox, degradation) previously existed twice, once in `check_determinism.sh`'s inline C++
and once in a test, and the two agreed only because they were kept in sync by hand. They now live once:
`core/determinism/digests.hpp` (`DetMathDigest`, `PhiloxDigest`, `Fnv1a64`) and
`perception/degradation/degradation_digest.hpp` (`DegradeDigest`). The tests, the script, and the run manifest
all call them, and the D18 boundary-coverage rationale moved with them.

`runtime/manifest/determinism.hpp::RecordDeterminismDigests(RunManifest&)` fills the manifest by MEASURING
the digests at run time. Copying the committed constants would make D21's "a result produced by drifted math is
identifiable after the fact" empty again: the manifest would claim the recorded values whatever the host
actually computed. `ariadne_runtime` now links `ariadne::perception`, which is allowed because runtime sits above
the domain libraries in the §3 dependency order.

Verified: 156 tests pass; `check_determinism.sh` PASS on 6/6 configurations using the shared code; committed
constants unchanged.

## 2026-09-29 — D33: the ariadne-core image, and what the OpenCV build hash does and does not identify

`infra/docker/core.Dockerfile` builds FROM `opencv-base` (tag locally; CI passes a reference through the
`OPENCV_BASE` ARG, by digest once ECR push is wired). Two stages: `dev` (toolchain only; the repo is bind-mounted
and built into `build/docker/`, never the host's `build/`) and `ci` (sources bind-mounted read-only for the build step so they never enter an image layer, configured with
`ARIADNE_REQUIRE_OPENCV=ON`, full `ctest` run inside the build, `ariadne_identity` installed). ONNX models from the
§27.1 row are deliberately absent until XFeat (Phase 2) consumes one. `infra/scripts/core.sh image|gate|dev` is
the only entry point; `docker-bake.hcl` builds `opencv-base` then `core` in one CI invocation with separate GHA
cache scopes.

OpenCV stays optional on the host: `find_package(OpenCV 5 CONFIG QUIET PATHS /opt/opencv5 NO_DEFAULT_PATH)`, and
OpenCV-dependent code lives in targets that exist only when it is found (`ariadne_runtime_opencv`,
`ariadne_identity`, `test_opencv`). `core`, the degradation operator, replay and metrics remain OpenCV-free.

**The build hash identifies a build artifact, not a configuration.** `cv::getBuildInformation()` contains
`Timestamp:` and `Host:` lines, so its hash (`fnv1a64:<hex>`, via the existing `core::Fnv1a64`) changes on every
rebuild of the same tag and differs between amd64 and arm64. No golden value of it is ever committed. It is
measured at run time from the LOADED `libopencv_core`, which is what makes it useful for COOL: the one binary run
against `/opt/cool` records a different hash than against stock, so every manifest says which OpenCV produced it.
The Phase 0 gate 4 check is therefore "the hash a manifest records at run time equals the hash baked into the
image at build time by the same binary" (`core.sh gate`).

**Loader linkage protects Invariant 2.** DT_RPATH is searched before `LD_LIBRARY_PATH`, so a binary carrying
`RPATH=/opt/opencv5/lib` would ignore the COOL loader switch and benchmark stock OpenCV against itself. The image
resolves OpenCV through `/etc/ld.so.conf.d` (searched after `LD_LIBRARY_PATH`), builds with `CMAKE_SKIP_RPATH=ON`,
and fails the build if `readelf -d` finds RPATH or RUNPATH on any installed binary or test. (The stock OpenCV
libraries themselves carry RUNPATH, which `LD_LIBRARY_PATH` does override.) Open for 0.3: confirm the COOL
libraries use the same sonames (`libopencv_core.so.500`).

**OpenCV outputs are not cross-machine golden values.** OpenCV compiles several SIMD variants of hot kernels and
selects one per call from the running CPU (the dev box dispatches up to AVX512_SKX). The manifest gains
`opencv_cpu_features` (`cv::getCPUFeaturesLine()`) so a cross-machine difference can be attributed. Consequence for
the thin-run determinism item: OpenCV-stage determinism is defined run-vs-run on the same machine; only the
OpenCV-free digests (D18/D32) are committed as cross-machine constants.

`image_digest` currently holds the LOCAL image ID (the config digest, `docker image inspect .Id`). After an ECR
push the registry manifest digest is a different value; when push is wired, manifests must record the registry
digest, and the two kinds must not be compared.

Also: `.dockerignore` now excludes `*.tfvars`, Terraform state and `.venv`, since the core build context is the repo
root and `terraform.tfvars` holds the alert email. `opencv-base` still uses an unpinned `FROM ubuntu:24.04`; pinning
it forces an OpenCV rebuild, so it is deferred to the 0.2 arm64 multi-arch rebuild that needs one anyway.

Verified (amd64, 2026-09-29): the `ci` stage builds and passes **160/160** tests inside the image (the 156
OpenCV-free tests plus 4 `OpenCvIdentity` tests), so the committed D32 digests also hold on the image's gcc 13.3.
`core.sh gate`: baked and manifest-recorded hash both `fnv1a64:cf8ea084bb984703`, `CORE_IMAGE_GATE=PASS`. The
RPATH detector was checked against binaries deliberately linked with RPATH and with RUNPATH (both detected);
`LD_DEBUG=libs` shows `libopencv_core.so.500` resolved from `/opt/opencv5/lib` via the loader cache. Not yet
demonstrated: a hash CHANGE under a different OpenCV (needs a second build, e.g. COOL at 0.3). The first image
build also exposed a latent bug: `run_manifest.hpp` used `uint64_t` without `<cstdint>` and compiled only through
transitive includes; fixed, and every header under `cpp/` now compiles standalone.

## 2026-09-29 — D34: placeholder perception front-end (1.4), and a near-range stereo hazard

Built from OpenCV 5 primitives with the algorithm logic owned by Ariadne (§8.1's split), following the reference
implementations read for it (VINS-Fusion `feature_tracker.cpp`, OpenVINS `TrackKLT.cpp` / `Grider_GRID`,
ORB-SLAM3 `ORBextractor`) and the papers behind them (Rublee 2011 ORB; Shi & Tomasi 1994; Bouguet's pyramidal LK;
Kalal 2010 forward-backward error; Hirschmueller 2008 SGM). API names verified against the installed 5.0.0
headers: `features.hpp` (ORB), `video/tracking.hpp` (`buildOpticalFlowPyramid`, `calcOpticalFlowPyrLK`),
`stereo.hpp` (`StereoSGBM`), `geometry/3d.hpp` (`findFundamentalMat`, `UsacParams`).

- **`OrbEngine`** (§7.1 interface, with `Detect` split from `DetectAndCompute` so tracking never pays for
  descriptors): ORB detect oversampled 4x; cells left starved re-detected at a lower FAST threshold through ORB's
  mask argument (ORB-SLAM's iniThFAST/minThFAST); our own `GridSelect` keeps top-k per 8x6 cell (§7.3) with a
  total tie-break order; descriptors computed on exactly the selected set. ORB's internal `nfeatures` cap is not
  the selection, because it is a global top-N, which §7.3 forbids.
- **`KltTracker`**: one pyramid per frame reused for the forward, backward and next-frame passes; forward-backward
  check at 0.5 px; border check; epipolar rejection; VINS-style spacing mask with older tracks winning;
  replenishment through the §7.1 engine (so the XFeat swap changes tracking too); `DEGRADED` below a configured
  track count.
- **Determinism:** `findFundamentalMat`'s `FM_RANSAC` path samples from `cv::theRNG()`, per-thread state that any
  earlier OpenCV call advances. The tracker uses the `UsacParams` overload with `isParallel = false` and
  `randomGeneratorState` drawn from Philox at `(seed, frame_id, reserved site)`. Verified: tracks, ORB descriptors
  and SGBM disparity are bit-identical across 1, 2 and 8 OpenCV threads and across arbitrary `cv::theRNG()` state.
- **§9.6 deviation:** `StereoDepth::disparity` is SGBM's native `CV_16S` (1/16 px), not fp16. fp16 spacing is
  0.125 px above 128 px, i.e. coarser than SGBM's output for every depth under 0.5 m at f = 640 px, B = 0.10 m.
  `CV_16S` is exact and also 2 bytes. Z and sigma_Z are computed in float32, then stored as fp16. `confidence` is an
  all-zero `[PLACEHOLDER]` flagged by `confidence_is_placeholder` until the WLS filter (Phase 2).
- **§9.4 validator** implemented (`ValidateStereoAgainstRig`), refusing e.g. B = 0.20 m at 256 disparities.
- Parameters live in `configs/perception/placeholder.yaml`, recorded per run in the new manifest field
  `perception_config` (§7.3: manifest fields, not constants). The OpenCV tests load that shipped file rather than
  struct defaults, so the tests exercise what runs.

**Approved 2026-09-29:** the §9.6 `CV_16S` disparity and the §7.1 `Detect`/`DetectAndCompute` split; both are
mirrored into the architecture doc.

**Hazard found: SGBM aliases surfaces nearer than its search range into phantom FAR depth.** A textured plane at
0.20 m (true disparity 320, beyond the 255 searched) still yields "valid" pixels: 1.38% of the image at uniqueness
10 and speckle window 100. The reported depths spread from 0.3 m to past 3 m rather than clustering at the range
limit, so **they cannot be recognised by their value**. A consumer that carves free space along depth rays would
carve it straight through a real obstacle: an error in the unsafe direction for egress.

What does separate them was measured rather than assumed (probe over planes at 1.5, 0.30, 0.26 m true and 0.20,
0.15, 0.10 m aliased):
- **Structure:** aliases form ~48 blobs of 116-312 px; true surfaces form one connected region.
- **Speckle window:** 100 lets those blobs through; 200 (top of §9.3's 100-200) cuts aliased pixels from 6518 to
  720 / 215 / 0 at 0.20 / 0.15 / 0.10 m, with no loss of true depth on the planes. Set to 200. `uniqueness_ratio` is
  set to 15 (top of §9.3's 10-15), which halves the rate relative to 10.
- **Local density:** the fraction of valid pixels in a 31x31 window is >= 0.84 for true depth (1st percentile;
  median 1.0) and <= 0.42 for the surviving aliases. `DenseSupportMask` (window 31, density >= 0.75, configured in
  `stereo.support_*`) passes > 90% of true depth and none of the aliases (tested).

Hand-off rule, testable: **1.8's occupancy placeholder carves free space only along rays whose depth is in
`DenseSupportMask`.** Unsupported valid depth may mark occupancy but never free space. **Superseded in part by
D35:** on real EuRoC texture the gate catches 87% of aliases, not all of them, so 1.8 must also choose between
"stereo never carves free space" and a near-field pass. `[ASSUMPTION]` These numbers
come from synthetic random-texture planes. Real scenes have genuinely sparse true depth (low texture, occlusion
edges), where the gate discards true depth, which errs in the safe direction. They must be re-measured on EuRoC,
and Phase 2's WLS confidence plus a near-field check (e.g. a reduced-resolution pass whose effective range covers
the blind zone) replaces the gate before any egress claim rests on stereo free space.

Measured (amd64, 8 threads, 1280x720 synthetic texture): 697 initial tracks; 689 survive a (3.3, -1.7) px shift
with median error 0.013 px (p95 0.028 px); dark-half keypoints 33 -> 600 with the low-threshold rescue; SGBM valid
fraction 0.80 = every non-blind column. Timings: pyramid 0.5 ms, tracking 18 ms/frame, ORB 48 ms/keyframe, SGBM
235 ms/keyframe. Known shortfall: the first frame reaches 697 of the 1200 target tracks (20 px spacing with ORB's
clustered corners); later frames reach about 1035.

Not yet done for 1.4 (Invariant 5, datasets before simulator): a EuRoC sequence through rectification
(`stereoRectify` / `initUndistortRectifyMap` / `remap`) and this front-end.

## 2026-09-30 — D35: EuRoC adapter, rectification, and the 1.4 real-sequence run

**Source and licence.** EuRoC MAV (Burri et al., IJRR 2016), DOI 10.3929/ethz-b-000690084. The old ASL page now
redirects to ethz-asl.github.io, whose EuRoC page points to the ETH Research Collection. That collection's HTML
frontend returns HTTP 500 or 429, but its DSpace REST API works (`/server/api/pid/find?id=hdl:20.500.11850/690084`
-> bundles -> bitstreams). It lists one zip per environment: `machine_hall.zip` is 12.7 GB and holds each MH
sequence as a ROS bag plus an ASL-format zip. **Licence: "In Copyright - Non-Commercial Use Permitted"** (dc.rights,
rightsstatements.org InC-NC 1.0). The competition licenses every *submitted* material to OpenCV and AWS "for any
lawful commercial or noncommercial purpose" (`opencv-ai-competition-2026.md` line 209); data only referenced is
not covered (line 210); third-party-terms violations disqualify (line 222). Therefore **no EuRoC pixels in any
submitted material** (report figures, demo video, S3/ECR artifacts). Derived metrics are treated as facts that may
be reported (confirmed by the maintainer 2026-09-30: report EuRoC numbers, never its pixels). The data lives only in `.cache/` (gitignored and
dockerignored; `spinup.sh` snapshots via `git ls-files --exclude-standard`, so it never uploads).

**Fetch** (`infra/scripts/fetch_euroc.py`, stdlib only). The server honours HTTP Range. The script reads the outer
zip's central directory (2 requests), downloads exactly the member's bytes in one streaming ranged GET (1.57 GB
deflated for MH_01 instead of 12.7 GB), checks the central-directory CRC-32, and writes a provenance file with
the sha256 (`5f4ecbc5...` for MH_01). It retries with backoff on 429/5xx and resumes stored members. That sha256
goes into the manifest's `bag_checksum`.

**Reader** (`runtime/datasets/euroc.hpp`, OpenCV-free): strict `sensor.yaml` parsing (pinhole + radial-tangential
only, rigid `T_BS` checked, OpenCV's `%YAML:1.0` directive tolerated although MH_01's files do not carry it), a
strictly increasing `data.csv` index, and stereo pairing by EXACT timestamp, never by row. **Frame convention
pinned by test**: composing the Kalibr `T_BS` values reproduces ORB-SLAM3's independently published
`Stereo.T_c1_c2` to 1e-6, and the `stereoRectify` input `T_right_left` has t.x = -0.110 m.

**Rectifier** (`perception/stereo/stereo_rectifier.*`), in the idiom of ORB-SLAM3 (`precomputeRectificationMaps`)
and Kimera-VIO (`StereoCamera`): `stereoRectify(..., STEREO_ZERO_DISPARITY, alpha = 0)`, then
`initUndistortRectifyMap` straight to `CV_16SC2` fixed-point maps, then `remap`. `alpha = 0` follows Kimera-VIO (the
default -1 leaves black borders that create invalid keypoints, opencv/opencv#7240). `CALIB_ZERO_DISPARITY` is
deprecated in OpenCV 5 in favour of `STEREO_ZERO_DISPARITY`. The rectifier refuses a non-horizontal result or a
negative baseline (the inverted-extrinsic failure). Synthetic geometric test: 3D points projected through a raw,
distorted, toed-in rig land on one rectified row within 1e-3 px, with depth from disparity within 1e-4 relative.
The negative control, a 1-degree pitch error, produces > 5 px. `remap` output is bit-identical across thread counts.

**`configs/perception/euroc.yaml`** is a DATASET config: 96 disparities (Z_min 0.505 m at the rectified
fB = 436.2 x 0.1101 m; 13% blind columns), 400 target tracks (VINS-Fusion ~150, OpenVINS ~200 at 752x480), and
`required_min_range_m` 0.60. The mission's 0.30 m requirement stays in `placeholder.yaml`.

**MH_01_easy, all 3682 frames** (`ariadne_euroc_frontend`; stereo, alias and row checks on every 10th frame, i.e.
369 keyframes; 8 OpenCV threads, amd64):
- Pairing: 3682 pairs, 0 unmatched on either side.
- Rectification: sub-pixel LK left->right with a 0.5 px reverse check gives **median |dy| 0.135 px** (p90 0.68 px)
  over 96742 points. The controls must fail it and do: raw pair 13.1 px, 1-degree pitch error 8.1 px (f*theta
  ~ 7.6 px). ORB mutual matches give median 0 at their ~1 px resolution (integer FAST positions) and 96% positive
  disparity.
- Tracker: 377 tracks/frame mean (p10 333); 0 DEGRADED frames. Rejections over 1.39 M attempted:
  forward-backward 35.0k, LK lost 9.4k, epipolar 8.8k, crowding 33.7k, border 0.8k.
- Stereo: valid 70.4% of pixels, dense-supported 62.3%.
- **Aliasing on real texture** (pseudo-truth = 288-disparity SGBM, counted only where its own support gate keeps
  it; SGBM judging SGBM, so this is an estimate, not ground truth): 0.70% of pseudo-truth pixels are aliases.
  They are **episodic, not diffuse**: max 26% in one keyframe, and 43 of 369 keyframes above 1%, i.e. genuinely near
  surfaces. **The gate catches 87%**, leaving a residual of 0.09% of pixels (synthetic planes: 100%). Gate cost on
  agreeing depth: 0.14% in the interior, 0.24% within 15 px of edges. That cost is measured only on densely
  supported pixels, so its cost on sparse true depth remains unmeasured.
- Timing: rectify 1.4 ms/pair, tracking 22.9 ms/frame (p95 40 ms), SGBM 31.6 ms/keyframe at 752x480x96.

**Consequence for 1.8 (decision deferred to 1.8, recorded now):** stereo either never carves free space in the
placeholder, or a near-field pass is added (e.g. half-resolution SGBM at the same 96 disparities covers the blind
zone at ~1/4 the cost), with any carving gated by it.

Also: the run manifest now records `compiler`/`architecture` (shared `build_identity.hpp`), `imu_mode: none` (the
default `attitude_only` would claim an inertial estimator ran), `loop_backend: none`, ISO `created_utc`, and the
exact command in `extra.command` (§36). `core.sh euroc` reproduces the run with provenance.

## 2026-09-30 — D36: placeholder stereo VO (1.5), conventions, and the MH_01 trajectory

Reference implementations read for it: ORB-SLAM3 (`StereoInitialization`, `ThDepth` = 60 B close points,
`NeedNewKeyFrame`'s 0.75 reference ratio), VINS-Fusion (`solvePoseByPnP`, stereo DLT `triangulatePoint`), and evo
(`associate_trajectories` max_diff 0.01 s, TUM format). OpenCV 5 signatures were verified in the installed headers:
`solvePnPRansac(..., UsacParams)`, `solvePnPRefineLM`, `triangulatePoints`.

**Conventions (one place each).** The solver state is world->camera `T_C_W` with a LEFT perturbation
`T <- exp([rho; phi]) T` (§11.3). Nodes and keyframes store `T_W_C` (§4.2); their covariance is converted once
with `core::InvertCovariance`. Pinhole Jacobians (`core/geometry/pinhole.hpp`) are finite-difference tested at
typical depth, near-zero depth (1 cm), a wide-FOV ray (X/Z = 5), per tangent direction, and for the right camera
of a rectified pair. **Monte Carlo covariance consistency**: the mean NEES over 2000 noisy PnP solves is 5.93 (400 trials gave 6.37, by chance) (chi^2
with 6 dof, i.e. about 6) for both `T_C_W` and the transported `T_W_C`. The §12.2 edge residual
`Log(Z^-1 X_i^-1 X_j)` is a right-side error on Z, so an edge stores its information in the residual's tangent,
converted through `Adj(Z^-1)`. A Monte Carlo test with a 1-rad relative rotation confirms it (mean r^T Info r ~ 6).

**One optimizer** (`slam/bundle_adjustment/reprojection_lm.hpp`): LM (§11.6 Madsen-Nielsen damping, lambda_0 =
1e-4 max diag H, x1/3 on accept, x2 on reject, a relative-cost OR step-size stop, 15 iterations), Huber (§11.3),
Schur elimination of landmark blocks (§11.5), and pose covariance by solving `S X = E` (§11.6). With landmarks
fixed it is the PnP refinement, and it agrees with `cv::solvePnPRefineLM` to 1e-6. With the older keyframe fixed as
the gauge it is the two-frame BA stub. Dense S for now; sparse Cholesky arrives with the Phase 2 window.

**Stereo VO** (`slam/odometry/stereo_vo.*`, plain inputs per §3; runtime glues perception to it):
constant-velocity prediction; `solvePnPRansac` with the `UsacParams` overload seeded from Philox at a new
reserved site (`kMaxSite - 2`), single-threaded; the shared LM for the pose and its covariance; ORB-SLAM3's keyframe
rule plus min-tracked and max-frames limits; at keyframes, LK left->right correspondences (promoted to
`perception/stereo/stereo_lk_matcher`, the idiom validated at 0.135 px in D35), `triangulatePoints` with the §11.2
rejections (depth range, reprojection in both views, sigma_Z/Z, SGBM consistency where SGBM passes
`DenseSupportMask`), the two-frame BA, and an ODOMETRY edge carrying the BA's relative covariance. When PnP fails it
holds the motion model, flags the frame, and forces a keyframe with an uninformative (1 m / 1 rad) edge; it never
silently resets. Parameters live in `configs/slam/{placeholder,euroc}.yaml` (strict loader), with `slam_config`
added to the run manifest. **Limits:** a landmark lives only as long as its LK track (no re-association, so no
loop closure), two-keyframe BA, no IMU.

**Interfaces:** `slam::PoseGraph` (§12.1 edge types, §12.2 residual, a version counter bumped on every mutation, and
ODOMETRY's GNC weight fixed at 1); the §4.2 `Keyframe` and §18.2 `KeyframeBundle` records live in
`runtime/agent/keyframe_records.hpp`, because they join slam and perception outputs. `KeyframeBundle` is stamped with
the graph version (its `frame_id_of_pose`), and fields without a producer are listed and left unset.

**Synthetic end-to-end test (CI, no dataset):** a textured room projected through a rectified stereo rig along an
S-curve, with 0.3 px noise: 14 keyframes, **translation RMSE 4.6 mm over 3.5 m**, never lost. It stays lost-free
with 10% of tracks jumping 20 px per frame, and is bit-identical across 1, 2 and 8 OpenCV threads and arbitrary
`cv::theRNG()` state. (The test first tripped the D18 Philox exhaustion guard by drawing a whole sequence from one
stream; it now keys noise per frame as (seed, frame_id, site).)

**EuRoC MH_01_easy, all 3682 frames** (`ariadne_euroc_vo`; poses mapped to the body frame via
`T_B_Crect = T_B_C0 [R1^T | 0]`, a round trip tested with a negative control; associated to 3638 ground-truth
samples):
- **ATE (SE(3)): 0.126 m RMSE over an 80.5 m path (0.16%)**, mean 0.110 m, max 0.365 m; rotation 0.077 rad.
- **evo 1.31.1 as the oracle reproduces 0.126268 m and 0.077268 rad exactly** (ATE translation and rotation;
  the evo RPE command was not captured), so the ATE metrics and the frame conversion agree with the
  field-standard tool.
- Sim(3) scale 1.004 (no scale drift). RPE over 1 s: 1.39 cm and 0.0025 rad.
- The rotation error is drift, not a frame bug: anchored at the first pose it grows 0 -> 4.9 deg over 182 s. The
  3.5 deg at t = 0 in the aligned view comes from the position-based alignment spreading the end drift.
- 0 lost frames, 556 keyframes, 555 BA runs (mean final/initial cost 0.28). 22.6k landmarks created from 113.5k
  candidates (rejections: 55.5k no stereo match, 34.0k depth range, 1.45k SGBM-inconsistent).
- Timing: tracking 16.3 ms/frame, VO 4.0 ms/frame, keyframe (stereo + SGBM + triangulation + BA) 34.8 ms.

Review fixes (2026-09-30): node covariance is ABSOLUTE, composed along the odometry chain from the gauge keyframe
(`ComposeCovariance`); the conditional BA/PnP covariance stays on the edge, and `VoTrackResult` names its field
`covariance_given_map`. Tested: the node covariance trace never decreases, and the final keyframe's NEES under the
composed covariance is 1.01 (< 22.46). A blackout frame with no tracks is flagged LOST, holds the constant-velocity
pose, forces a keyframe with an uninformative edge, and tracking resumes within 3 frames. `MatchLeftToRight` is the
single stereo LK implementation (the frontend tool calls it with the row gate off) and has its own unit test. Every
shipped config is load-tested on the host. The EuRoC ground-truth copy is written under the gitignored `raw/`.

## 2026-09-30 — D37: procedural room-maze worlds (1.1–1.3 worldgen) and their deviations from arch §5.3/§5.5

Design: `docs/design/SIM_1.1-1.3_WORLDGEN.md` (the source of this entry). Code: `python/dataset_tools/worldgen/`
(numpy + pyyaml only; runs in the pinned ariadne-sim image via `infra/scripts/worldgen.sh`), presets
`configs/worldgen/{maze_short,maze_medium,maze_long,ci_tiny}.yaml`, ChArUco assets
`simulator/assets/charuco/` from `cpp/runtime/tools/ariadne_charuco_assets.cpp`. 100 worldgen tests pass in the
image, including two pinned golden scene-directory digests (GENERATOR_VERSION 1).

**What a scene is.** A seeded maze of 6 m lattice cells (Wilson uniform spanning tree) with 2–4 rooms, dead-end
stubs, two entrances with vestibules, and a target room whose ρ_responder geodesic from the primary entrance lies
in the preset's length band (short 10–20 m, medium 20–30 m, long 30–45 m, set from a 1000-seed census run on a
dirty tree before props and routes existed; to be re-run from the clean commit). Every
entrance is made 2-edge-connected to the target by a fundamental-cycle cover (so loop routes exist by
construction), and agent-only narrow passages (0.55–0.70 m) are placed off those cycles so ρ_agent and
ρ_responder routes genuinely differ (Invariant 4). Identity: (GENERATOR_VERSION, seed, preset digest, attempt).

**Consensus-aware scripting.** Scripted routes use the least-length edge-disjoint route pair per entrance
(min-cost flow); under the default `cross` template each return leg is edge-disjoint from the other agent's
outbound leg, so no head-on corridor meetings. Agents survey the target and both entrance rooms with 360° spins
at per-agent offsets. A scene is accepted only with ≥ 3 inter-agent same-heading, line-of-sight closure
opportunities in ≥ 2 rooms (~400–850 in 3 rooms on the seeds inspected), and only if a start delay keeps the agents ≥ 1.0 m
apart throughout.

**Deviations from the arch text.** Items 1–6 were in the design draft the maintainer told me to implement ("continue with the implementation", 2026-09-30). Items 7–9 arose during implementation and are **pending the maintainer's confirmation**:
1. IMU is synthesized by the kinematic mover from the trajectory's closed-form derivatives; IMU noise and bias
   random walk are a seeded consumption-time operator, not SDF `<imu><noise>` (§5.3). Forced by SMOKE §4g: gz's
   IMU reports (0, 0, 9.8) and zero rate on a pose-written body. One IMU config file serves the noise operator
   and the estimator (§10.2's "same numbers" holds).
2. 1.1 uses the generator; the plan's hand-authored two-room + corridor scene is the `ci_tiny` preset.
3. Phase 1 textures are rich and unique per surface, not §5.5's low-texture concrete; `appearance.tau` is the
   dial toward it (Phase 5), `appearance.repetition` the I7 aliasing dial (refused until 3.5).
4. Phase 1 GT is 2D (exact for extruded plans): occupancy, exact ESDF, traversability at both ρ, geodesics.
5. GT pose is published by the mover (exact), not gz's pose-publisher.
6. Surfaces are in-house OBJ quads, not trimesh meshes (§5.5 stage 3).
7. Scene worlds inline the scene model (SMOKE §4i: sdformat14 does not resolve relative `<include>`).
8. Phase 1 agent models carry no visual geometry: agents are invisible to each other (a stated simplification;
   a visible peer is a dynamic-object problem for later phases). Separation is enforced in the scripts.
9. Motion primitives are S-curves (C² velocity, continuous jerk) with yaw+pitch, not minimum-jerk; the rig
   pitches during the calibration dance (views rotated about one axis only are a degenerate Zhang set).

Withdrawn: GT depth at 640×360 (SMOKE §4j: saves ~1 % of compressed bytes).

**Measured sizing (SMOKE §4j, one scene):** ~84 MB per sim-second for two agents, RTF ~0.17 on the 8-core
llvmpipe host; missions 2–7 min per agent on the seeds inspected (the accept sweep records the distribution).
**Open:** how the `ci_tiny` bag (~16 GB) reaches nightly CI.

## 2026-10-01 — D38: ROS-free bag reading, the kinematic mover, and verified recording (1.1–1.3)

**ROS-free CDR decoding** (`cpp/runtime/replay/ros_cdr.{hpp,cpp}`, `rosbag_reader.hpp`): XCDR1 decoding of exactly
the recorded types (Image, CameraInfo, Imu, PoseStamped, TFMessage, Clock) into Ariadne's own structs, over the
Foxglove MCAP SDK (zstd chunks). Bounds-checked; every truncation of every fixture throws. Fixtures are bytes
produced by `rclpy.serialize_message` and a `rosbag2_py`-written zstd MCAP. Closes D25's design gap.
`McapRecordReader` now requests LOG-TIME order explicitly (the SDK default is file order); the Phase 0 replay
gate is unchanged (226/226 image, 199/199 host).

**Kinematic mover** (`simulator/plugins/kinematic_mover/`): a gz-sim system on each agent model, no Physics.
The trajectory math is `cpp/core/trajectory/scripted.hpp` — one dependency-free implementation that the core
suite checks against the Python generator's golden samples (808 samples incl. pitched ones, 1e-9) and the
plugin compiles. The script arrives as `<segment>` elements in the mission SDF (exact float repr), is validated
for continuity and against the scene ESDF (collision envelope, 10 ms sampling) at start-up, and a rotation
self-check pins gz's Euler convention; any failure aborts the run. The IMU message's orientation field is
identity and documented as unmeasured (an IMU does not measure attitude; GT stays out of perception).

**Mission composer** (`python/dataset_tools/worldgen/mission.py`) + rig config `configs/sim/rig_stereo_v1.yaml`:
the mission world inlines the scene and adds per-agent body link (IMU frame), stereo cameras, a GT depth camera
co-located with the left camera, an inert spot lamp, and the mover; the manifest carries T_B_C to the optical
frames and nominal intrinsics (the recorded camera_info and the §6 recovery are authoritative).

**Recording** (`infra/scripts/sim_record.{sh,py}`): bridge from the generated YAML, latched `/tf_static` with the
rig extrinsics, `ros2 bag record -s mcap --storage-preset-profile zstd_fast --use-sim-time`, then a STRICT
verification of the bag it wrote (every expected message present, period-exact gaps, stereo pairing 1.0, GT pose
and IMU equal to the script at their stamps) and `recording.json` with sha256, versions and identity. Deep queues
on the 200 Hz streams after SMOKE §4k found a stall-induced drop.

## 2026-10-01 — D39: canonical IMU / GT-pose streams written from the scripts; the live ones become a cross-check

**Accepted by the maintainer 2026-10-01** ("yes go ahead, make sure it is deterministically correct").

**Problem.** Deep queues (D38) did not make the live 200 Hz path lossless: the full `ci_tiny` mission (199 s)
lost one of 39,800 agent-1 IMU samples (t = 174.435 s) between gz-transport and the bag, while every sample that
arrived equalled the analytic value exactly (SMOKE §4k/§4l). The loss rate rises with agents (message rate and
host load both grow), and a missing IMU sample corrupts preintegration.

**Decision.** The recording's IMU and GT pose come from `ariadne_canonical_streams`
(`simulator/plugins/tools/canonical_streams.cc`), which evaluates the mission SDF's scripts offline and writes
them as their own rosbag2 MCAP bag, `RECDIR/canonical/`. The rendered bag moves to `RECDIR/sensors/` and keeps
the live IMU/GT as a cross-check.

- **Same code, same numbers.** The parser and the per-tick sampler (`MoverScript.hh`: `ParseMoverConfig`,
  `SampleAt`) are shared by the plugin and the tool; both targets compile with `-ffp-contract=off` so no FMA
  contraction can differ between inlining contexts (GCC fuses by default where the ISA has FMA, e.g. aarch64).
- **Stamp grid** t_k = k·period for k = 0..⌊T/period⌋. The plugin publishes k ≥ 1 (its first PreUpdate is one
  step in); k = 0 is added so GT brackets the first camera frame (stamped 2 ms). log time = header stamp.
- **Own CDR encoder** (`cpp/runtime/replay/ros_cdr_writer.hpp`), not rclcpp: Fast-CDR skips alignment padding
  without writing it, so rclcpp output carried heap garbage (observed "at" in the 2 pad bytes after
  "agent_0/base_link") and two runs of the tool differed. With zeroed padding every message is decoded back
  by the runtime decoder and compared bit-for-bit (a -0.0 stays -0.0) before writing.
- **What "reproducible" means, scoped.** (a) *Container bytes*: identical run-to-run in the same image and
  arch — checked on every recording (generated twice, compared) and the recording aborts before gz if not.
  (b) *Content*: `canonical_content_sha256` = sha256 over (topic, stamp, payload) in (stamp, topic) order,
  independent of rosbag2/MCAP/zstd versions; pinned for ci_tiny seed 0 / 8 s by
  `python/tests/sim/test_canonical_streams.py` (`worldgen.sh simtest`), so a one-bit change anywhere in the
  trajectory code, encoder, composer or routes fails in ~15 s instead of on the next render. (c) *Across images
  or arches* (libm, codegen) the content may move in the last bits; the guard there is the ≤ 1e-9 agreement
  with the independent Python evaluator, not bit equality. The manifest pins the inputs: mission SDF sha256,
  `end_ns`, tool binary sha256 (built from `git_commit` in `image_id`).
- **Verification** (`infra/scripts/sim_record.py`): canonical — exact grid, log time = stamp, values = the
  independent Python evaluator (≤ 1e-9), regenerated a second time and compared byte-for-byte; live IMU/GT —
  every sample bit-identical to canonical (bytes, else decoded fields as IEEE bit patterns), no off-grid or
  duplicate stamps, losses counted (`live_gaps`, with stamps) and fatal only below 99 % coverage; cameras and
  infos — unchanged STRICT (a frame cannot be regenerated; remedy = re-record). The world's `<max_step_size>`
  is asserted to be the 1 ms the recorder counts in. The C++ `RosbagReader` reads a committed 1 s canonical bag
  ROS-free (`test_ros_cdr`), and the encoder reproduces live bridge bytes including padding.
- **Honesty.** Manifest `streams.canonical.source` = "analytic IMU/GT from script [DESIGN] … not sensor output".
  gz never measured this IMU; the mover already computed it analytically (OpenVINS-simulator pattern); noise,
  bias and jitter stay a seeded consumption-time operator (D37).
- **Replay rule.** IMU + GT from `canonical/`, everything else from `sensors/`, merged by header stamp (ties:
  topic name), not log time (sensor log times are receive times). Implemented with the 1.4/1.5 sim consumer.
- **Cost.** 5.1 MB of canonical MCAP against 12.6 GB of sensors for the 199 s mission (0.04 %).

**Consequences / open.** Live closed-loop and demo runs have no canonical file: the estimator must bridge a
missing IMU sample (integrate across the gap, log it) — Phase 2 requirement. The §4g "frame stamped t shows the
pose at t" measurement that the canonical GT relies on was made at 2 agents on llvmpipe; re-check it whenever
the agent count grows. `live_gaps` per run is the trend line. The bridge and recorder logs are now kept
(`RECDIR/logs/`) and their loss reports parsed into the manifest; transport loss is measured on `/clock`
(SMOKE §4l: 27 on an 8 s run, matching the bag exactly). The first fresh full recording under D39 then lost
one **camera frame** (agent 1 right, 3980/3981) and failed strict — the remaining exposure, since frames
cannot be regenerated. Its layer (DDS transport, recorder cache, gz → bridge) is unknown: that run predates
log capture. Open: attribute with logged runs, then size image queues / recorder cache; until then the
remedy is re-record.

## 2026-10-01 — D40: the sim recorder reads gz-transport directly; no bridge or DDS in the recording path

**Accepted by the maintainer 2026-10-01** ("go with A, but with loud failures"). Deviates from design §7 /
arch §5.9's `ros2 bag record` via `ros_gz_bridge`; the bridge stays for live (closed-loop, demo) runs.

**Why.** Three logged full recordings (SMOKE §4l) put every lost message on the DDS link between the bridge
and `ros2 bag record`: per topic, expected − in bag = the recorder's own transport-lost count, residual 0; no
cache drops; isolated single losses, including GT pose with queues 1000 deep on both sides. Depth is not a
guarantee, and frames cannot be regenerated.

**What.** `ariadne_gz_recorder` (`simulator/plugins/tools/gz_recorder.cc`) subscribes to the mission's gz
topics (the bridge YAML's GZ→ROS set), converts with `ros_gz_bridge::convert_gz_to_ros` from the installed
`libros_gz_bridge.so` (the bytes the bridge would publish; ROS CDR serialized into zero-filled buffers), and
writes `RECDIR/sensors/` through `rosbag2_cpp` with `max_cache_size = 0` (synchronous; no cache drop path),
log time = header stamp. gz-transport's ZeroMQ high-water marks are 0 (unbounded) on both ends
(`GZ_TRANSPORT_{RCV,SND}HWM`; default 1000 drops silently). The world starts paused and is unpaused only
after the recorder has discovered every publisher. `/tf_static` moves into `canonical/`, built from the SDF
camera sensor poses (identity rotation asserted) and checked against the mission manifest's extrinsics.

**Loud.** Every way a message can go missing is an error printed when it happens, counted in
`logs/gz_recorder.json`, and a non-zero exit that fails the recording: a stamp interval that is a multiple
of the period (k−1 LOST), any other irregular interval, duplicate, out-of-order, a first stamp other than the
expected one (start missed; 0 for rendered topics and `/clock` with the paused start, one period for the
mover's IMU/pose, as measured), a topic not reaching floor(end/period)·period (incomplete), the writer backlog
exceeding its cap (default 4 GiB: ABORT rather than drop), no publisher discovered (exit 2), an unsupported
type pair (exit 2). Sole exception, from measurement: a paused world republishes `/clock` = 0, so repeats
of stamp 0 before a topic's first non-zero stamp are counted as `paused_repeats` and not written.
`/clock` becomes strict in the verifier (one message per 1 ms step). Each failure mode is exercised against
the real binary by `python/tests/sim/test_gz_recorder.py` (13 tests, fake gz publisher). Log time = header
stamp in the sensor bag too (it was the recorder's receive time), so both bags merge in stamp order.
Not logged by the recorder: running out of memory (the ZeroMQ queues are now unbounded); it still surfaces
as a non-zero exit of gz or the recorder, which fails the run. Recordings made before D40 (no recorder report,
no canonical `/tf_static`) no longer pass `--verify-only`; they are superseded, not migrated.

**Confirmed** (SMOKE §4m): two full `ci_tiny` missions recorded through it both PASSED strict verification
with 0 recorder errors and every message present, where the bridge path lost 3–9 non-clock messages per run.

## 2026-10-01 — D41: lens distortion in the simulated rig is a read-time operator (option A)

**Finding (SMOKE §4n).** Under gz Harmonic's ogre2 renderer the SDF `<distortion>` is ignored (no
`DistortionPass` in ogre2) while `camera_info` still advertises the coefficients; the rendered camera is an
exact pinhole. `camera_info` also states cx = W/2 where the pixel-centre convention needs (W-1)/2. Arch §5.3
asks for a mild plumb-bob so the undistort path is exercised.

**Until decided:** the recorder's gz-warning gate fails every recording with non-zero rig distortion; the
1.4/1.5 sim adapter must not take D (nor cx, cy) from `camera_info` — the truth is pinhole fx = fy = 640,
cx = 639.5, cy = 359.5. The 1.3 evidence recording (`evaluation/runs/phase1/sim_record/ci_tiny_0-20261001T042105Z`)
predates the gate: it is valid evidence for the lossless recording path; its camera_info D is false.

**Options.** (A, recommended) zero distortion in the SDF so camera_info is honest about the pixels (cx
convention documented and corrected in the adapter), and Brown distortion applied as a seeded consumption-time
operator (`cv::remap` with the forward model), consistent with D37's degradation-at-read; the undistort path is
exercised with exactly known coefficients. (B) pinhole only in sim; EuRoC (real radtan) exercises undistort.
(C) the Ogre1 backend — not verified to render headless on EGL here; not recommended.

**Accepted by the maintainer 2026-10-01 ("go with a"). As built:**
- **Render:** `configs/sim/rig_stereo_v1.yaml` has no `camera.distortion` (the composer refuses one, naming
  this decision); the SDF carries no `<distortion>`, so gz has nothing to ignore and `camera_info` D is zero.
  The render truth is pinhole fx = fy = 640, cx = 639.5, cy = 359.5 (pixel-centre); the mission manifest
  records it as `render_intrinsics_pixel_centre` and states gz's W/2 camera_info convention.
- **Lens:** `rig.lens` (plumb_bob, fx = fy = 680, cx = 639.5, cy = 359.5, k1 -0.05, k2 0.01, p1 0.0005,
  p2 -0.0002, k3 0) is applied at read time by `cpp/perception/degradation/lens.hpp`: per output pixel, invert
  Brown-Conrady (fixed-point, 30 iterations, verified to 1e-12) and sample the render bilinearly; depth is
  resampled nearest-neighbour. OpenCV-free, IEEE double with `-ffp-contract=off`, pinned by a digest. Optics
  precede the sensor stages of `Degrade()`.
- **Why fx 680, not 640:** with barrel distortion the inverted ray lies outside x_d, so a lens with the render's
  focal length would sample beyond the render (black corners). The minimum for these coefficients is 674.05;
  680 leaves >= 3.37 px. Both `BuildLensMap` and the composer refuse a lens that samples outside the render.
- **Verified** (`tests/unit/test_lens.cpp`, SMOKE §4n): the model equals `cv::projectPoints` to < 1e-9 px;
  identity and linear-image exactness; refusal of fx 670. §6 on a full recording with the lens on: GT
  reprojection through the lens model 0.07-0.09 px RMS, no radial trend; OpenCV's free calibration recovers
  k1 -0.0506, k2 0.0104 (truth -0.05, 0.01) and fx within 0.4 %; stereo with the truth intrinsics within
  0.09 mm. §6 now has a verdict (`configs/sim/acceptance_s6.yaml`) and exits 1 on FAIL. That PASS is CIRCULAR:
  the thresholds were derived from the same dev run; the first independent test is the evidence run on a fresh
  D41-rig recording. Demonstrated FAIL: a 5 mm board misplacement (every A check). The other failure modes are
  argued from magnitude, not demonstrated. The free calibration's fx bias (+0.2-0.4 %) persists with k1/k2/k3
  held at truth `[ASSUMPTION: fx-distance ambiguity of near-frontal planar views]`.
- **Consumers** take the camera model from `rig.lens` (via the mission manifest), never from `camera_info`.
  Nothing applies the lens yet outside §6: the 1.4/1.5 sim adapter must run `ApplyLens` then `Degrade`.
- **Evidence:** the committed recording `ci_tiny_0-20261001T042105Z` used the pre-D41 rig (camera_info D false);
  it stays evidence for the lossless recording path only. The upcoming D41-rig recording replaces it as the
  1.3 evidence and is the input of the §6 evidence run.
- **Open, recorded:** (1) the route speed budget uses `focal_px: 640` (worldgen presets); the estimator's camera
  is now 680, ~6 % more pixel motion at the image centre than planned — changing it moves the worldgen goldens
  (GENERATOR_VERSION bump), so it is a deliberate later change; the same 6 % applies to SGBM disparity ranges in
  the 1.4/1.5 sim config. (2) The lens digest is pinned only by an x86 unit-test golden; add it to
  `RecordDeterminismDigests` (D32) before any Graviton/COOL run so manifests and the Arm gate check it.
  DONE 2026-10-01: `perception/degradation/lens_digest.hpp` (map bits + output bytes, all coefficients
  non-zero), manifest field `lens_digest` (required), `check_determinism.sh` EXPECTED_LENS
  1253886791886396038 (g++/clang++, -O0..-O3).
- **Evidence status:** the D41-rig recording `ci_tiny_0-20261001T133914Z` (clean) is the 1.3 evidence. §6 on it
  reproduces the dev run exactly (renders reproducible), so it does not break the threshold circularity; the
  independent §6 test is ci_tiny seed 1 (other rooms, other board pose) — PASSED with margin on every check
  (SMOKE §4n). Thresholds `s6_v1` stand.


## 2026-10-15 — D42: 1.4/1.5 run from a sim recording through one StereoSource interface

**What.** `cpp/runtime/datasets/stereo_source.hpp` defines `StereoSource` (raw calibration, T_B_C0, GT body poses,
time-ordered RAW gray stereo pairs) with two implementations: `EurocSource` and `SimSource`. The 1.4 + 1.5 loop and
its scoring moved verbatim from `ariadne_euroc_vo.cpp` into `cpp/runtime/tools/stereo_vo_runner.hpp`
(`RunStereoVo`); `ariadne_euroc_vo` and the new `ariadne_sim_vo` only bind a source. **Regression:** the EuRoC
MH_01 trajectory reproduces byte-identically (estimate sha256 22d3a40d…; results.yaml identical except timing).

**SimSource** reads one agent of a two-bag recording ROS-free: frames from `sensors/` through the read-time lens
(D41) then RGB→gray; calibration from `rig.lens` (both cameras) and `T_right_left` from the canonical
`/tf_static`; GT from the canonical pose at each image stamp (images sit on the 5 ms grid, so association is
exact). It never reads camera_info. It refuses a pre-D41 mission, k3 != 0 (PinholeRadtan carries four
coefficients) and any stereo pair that completes out of order (a missing frame). Not yet: the sensor stages of
`Degrade()` after the lens (Phase 5; needs GT depth and a named config); manifests say degradation "none".

**Configs `sim_v1`** (perception, slam): euroc.yaml with only resolution/geometry-dependent values changed. The
§9.4 stereo guard refused the EuRoC disparity range on this rig (Z_min 0.71 m at rectified f 671.5, B 0.1);
240 disparities give Z_min 0.281 m, meeting the 0.30 m mission requirement (the sim is the mission geometry,
unlike a dataset), at the cost of 19 % blinded columns and ~0.28 s SGBM per keyframe at 720p.


## 2026-10-15 — D43: 1.6 placeholder collaboration — one broker, verified SE(3) edges, bytes from day one

**What.** `ariadne_sim_collab` (design `docs/design/COLLAB_1.6.md`) runs both agents' 1.4/1.5 VO in replay from one
sim recording and then the two-stage exchange of arch §15.1 with placeholder internals: agent 1 sends a
GLOBAL_DESCRIPTOR per keyframe (256-D mean-bit ORB, `collaboration/retrieval/mean_bit_descriptor.hpp`) to the broker
(agent 0, lowest AgentId); the broker takes cosine top-1 per query, keeps up to `max_candidates`, sends
MATCH_REQUEST; agent 1 answers MATCH_RESPONSE (ORB keypoints + descriptors + stereo 3D points + camera); the broker
verifies (`collaboration/broker/edge_verifier.hpp`: ratio test → `findEssentialMat` RANSAC gate → `solvePnPRansac`
for the metric SE(3) edge, RANSAC seeded by rank) and sends POSE_EDGE (`LoopEdge` table, status GEOMETRY_VERIFIED).
Merge: T_O0_O1 from the best-ranked verified edge; both trajectories scored in one frame. Every message goes through
`WireCodec` (three new `wire.fbs` tables appended to `Envelope`, decode validates array lengths) and `ByteLedger`,
and through the idempotent `collaboration/protocol/inbox.hpp` (per-(sender, class) sequence, out-of-order buffer,
duplicates dropped). The VO runner gained `on_keyframe` / `on_body_pose` hooks (called only when set).

**Choices.** The essential matrix is the plan's geometric gate but has no scale, so the edge is PnP of agent 1's
stereo 3D points against agent 0's pixels (SE(3), locked decision; never Sim3). Edge information is a `[PROXY]`
diagonal (σt 2 cm, σr 0.01 rad at 30 PnP inliers, ∝ 1/√inliers). Outputs are built from a rank-keyed outcome map
after the exchange, so they cannot depend on arrival order; any unexpected message class, lost edge, undrained link
or leftover inbox message fails the run. GT is evaluation-only (opportunity labels, edge error), never fed back.

**Fault injection** (`--inject-faults 1`): duplicate every 5th send, hold every 3rd delivery back past the next
flush's messages, drain at the end; the run fails unless both inboxes saw duplicates AND buffered reorders. The
first version (adjacent swaps) mostly swapped messages of different classes, which the per-class inbox never had
to reorder; replaced before any result was taken.

**Evidence** (clean commit 2ed0a49, `git_dirty: false`): `evaluation/runs/phase1/sim_collab/`
`aadffb8df5790ca0-20261015T083420Z` (no faults) and `aadffb8df5790ca0-faults-20261015T083420Z`;
`infra/scripts/check_collab_fault_identity.sh` → `fault_identity.txt` **VERDICT PASS** (CSVs byte-identical,
results.yaml identical except inject_faults/protocol, both agents' estimates sha-identical, faults exercised:
broker 54 dup / 4 buffered, agent 1 8 / 8). Every pre-diagnostic column and results.yaml are identical to the
dev runs (the diagnostics do not change the verifier).

**Measured (ci_tiny seed 0, `collab_v1`; evidence = dev runs).** 327 + 250 keyframes. 20 candidates (the cap), 20 verified,
**20/20 edges within 5 cm / 1° of GT** (max 3.3 cm / 0.57°); 17 inside the 30° opportunity cone, the other 3 are
correct edges at 33–48° view angle (the cone is worldgen's guarantee, not a correctness test; gates were NOT
tightened to exclude them). Merged two-agent ATE 9.8 mm (one alignment, 7962 poses). Bytes (§15.7 first numbers):
GLOBAL_DESCRIPTOR 250 × 1170 B = 292 KB; MATCH_REQUEST 20 × 137 B; MATCH_RESPONSE 20 × 48.6 KB = 971 KB (76 % of
1.27 MB); POSE_EDGE 20 × 207 B. Core suite 248/248 (4 protocol + 2 verifier-diagnostic tests new); EuRoC MH_01
estimate reproduces byte-identically (22d3a40d…). Fault run: broker 54 duplicates / 4 buffered, agent 1 8 / 8, `candidates.csv`
byte-identical and `results.yaml` identical except `inject_faults`/`protocol`; both runs' VO trajectories
byte-identical to the committed `sim_vo` evidence (the hooks do not perturb VO).

**What the numbers do not show (open items).**
1. Retrieval is weak and the threshold is inert: over all 250 queries the top-1 cosine spans 0.884–0.996
   (median 0.893), so `retrieval_threshold: 0.5` passes everything and the cap selects (effectively top-20 by
   score). 71 queries have a GT opportunity; top-1 is a true opportunity for 51 (72 %). EigenPlaces + λ₂
   prioritization replace this in Phase 3; the threshold is not re-tuned to look meaningful.
2. The proxy information is overconfident: rank 4 (333 inliers, σt ≈ 0.6 cm) is off by 3.3 cm ≈ 5σ, rotation
   ≈ 3σ. Harmless for the 1.6 merge; PCM/GNC (Phase 3) need a real edge covariance.
3. Candidates cluster: agent 1 parks for ~70 % of its run (GT, 200 Hz: stationary at (9.0, 7.9) from ~57.5 s to
   the end at 199 s; camera 20 Hz, hence its constant 159 VO tracks), so many of its keyframes view one place.
   Vertex-cover / dedup is Phase 3.
4. MATCH_RESPONSE dominates bytes; 1000 ORB features with float32 3D points per request. Budgeting is Phase 3.
5. `edge_correct_t_m`/`_r_deg` (5 cm / 1°) were set after the first run, to separate correctness from the cone.
6. **The verifier accepts a wrong edge** (negative probe, below). Not gated in 1.6: the exchanged edges are all
   correct, the degeneracy gate → PCM → GNC is the locked Phase 3 design, and any threshold picked now would be
   fit to one negative (`min_essential_inlier_ratio` 0.57–0.67 would reject it and keep all 25 correct edges;
   `collab_v1` is NOT changed). This exact pair becomes a Phase 3 regression case.

**Negative probe** (evaluation only; no wire, bytes, edges or merge): the verifier is fed the 20 highest-scoring
top-1 pairs that are NOT opportunities (`negative_probe.csv`); a false accept is a verified edge that is not correct
vs GT. Dev run: **14 rejected** — including all 12 pairs at different places (4.9–7.7 m apart; 7 at the essential
gate, 5 for too few matches) plus 2 near pairs (too few matches / 3D points) — **5 accepted, all correct** (out-of-cone overlaps), **1 false
accept**: agent 0 frame 3372 vs agent 1 frame 382, 0.48 m apart at 54.5° view angle, 289 matches / 162 essential /
129 PnP inliers, edge error 1.10 m / 51.6°. So the gate rejects appearance-only matches between different places
and fails on a near pair. A yaw-only bearing check suggests both cameras face ChArUco board 1 (room (1,0), wall S)
from ~1.3–1.4 m (a planar target, where PnP has a flip ambiguity). **Tested and NOT confirmed as a separable
cause:** the non-gating diagnostics `h_over_e` (homography / essential inliers on the same matches) and
`planarity` (λmin/λmax of the PnP inliers' 3D covariance) put the false accept (0.611, 0.00104) INSIDE the range
of the 25 correct edges (h_over_e 0.19–0.71, planarity 0.00041–0.167; ranks 1, 2 and 4 are more planar, rank 6
has a higher h/E). In a room maze most matches lie on one wall, so correct edges are near-planar too: neither
scalar can gate this. Cause `[UNVERIFIED]`; the Phase 3 answer is consistency (PCM against odometry and the other
edges), not a per-edge threshold.


## 2026-10-15 — D44: gsplat runs off-AWS (Kaggle) until the GPU quota lands; no CPU rasterizer

**Facts.** No NVIDIA GPU on the dev box. AWS GPU quota is 0 for EC2 G/VT (on-demand and spot) in eu-north-1,
us-east-1, us-west-2, eu-west-1 and eu-central-1, and for every SageMaker g4dn/g5/g6 training/processing/notebook/
Studio quota checked (eu-north-1, us-east-1). The 2026-09-29 cases auto-closed unanswered; re-filed 2026-10-15
(spot case 179205877700888, on-demand case 179205877900026, 4 vCPU each); AWS escalated both to an internal team the
same day. gsplat 1.4.0 cannot rasterize without CUDA: even `_torch_impl._rasterize_to_pixels` calls a CUDA kernel and
needs `nerfacc` (SMOKE §4o).

**Decision (maintainer).** Delegate the gsplat render to Kaggle for now. Kaggle rather than Colab because it has a
CLI (`kaggle datasets` / `kaggle kernels push|status|output`), so the run is scripted and its environment
recorded. This deviates from the locked compute placement ("gsplat on AWS g5/g6", arch §18.2) and is **interim**;
arch §18.2 carries the note. The maintainer also stated that **the demo needs realtime rendering**, so a live GPU
(AWS once the quota lands) stays on the critical path and the quota cases stay open.

**Consequences.** No Ariadne-written CPU rasterizer (it would be throwaway once any GPU exists). The C++→Python
boundary is the spool (design RECON_1.7); validation, fusion, initialization and geometry metrics run locally on
CPU and deterministically; only a small render package goes to Kaggle. Manifests record
`splat_backend: gsplat_cuda@kaggle` (later `@aws`) with the remote torch/CUDA/gsplat versions and GPU name. Remote
renders are compared by tolerance, not bytes.


## 2026-10-15 — D45: 1.8 placeholder mission layer — OctoMap-model occupancy, nearest frontier, egress at ρ_responder

**What.** `ariadne_mission` (design `docs/design/MISSION_1.8.md`) consumes the 1.7 spool and the mission briefing
and, in replay, integrates keyframe depth into a 0.05 m voxel grid (OctoMap sensor model, Hornung et al. 2013:
log-odds hit 0.7 / miss 0.4, clamp [0.12, 0.97], fixed-point; rays walked with Amanatides & Woo 1987), collapses a
standing-body z band into a 2D tri-state, and every 5 s computes frontiers (Yamauchi 1997), a nearest-frontier
decision per agent (JSONL trace, `mode: replay, executed: false`), the **egress route at ρ_responder** and the
**agent route at ρ_agent** (A*, octile, worldgen's corner rule). `python/analysis/mission_eval.py` scores against
`scene_gt.npz` (evaluation only).

**Choices (each fixed before measuring).**
1. *ρ discipline:* the plan's wording ("A* route to exit at ρ_agent") is superseded by invariant 4 / arch §20.5:
   the egress product is at ρ_responder (0.40 m); the ρ_agent route is reported beside it, labelled.
2. *Clearance convention = GT's* (cell centre to blocked-square edge, ≥ ρ − 1e-9), ported exactly; egress clearance
   counts unknown as blocked ("observed, not merely not contradicted"), agent clearance counts occupied only.
3. *World anchor:* agent 0's briefed entry pose × the rig's body←camera extrinsic (`[ASSUMPTION]` a team knows
   where it entered; R1 ≈ I on the ideal rig). Agent 1's entry pose is NOT used: it enters W only through the
   verified 1.6 loop, at t_merge = the later of the two matched keyframes (`[ASSUMPTION]`: that edge was picked
   with hindsight). The evaluator's oracle: agent 0's first 10 s through this anchor vs GT.
4. *Depth gates from the noise model:* σ_Z = 0.00745 Z² m (σ_d 0.5 px, f 671.5, B 0.1) → σ ≤ 0.06 m ⇒ Z ≤ 2.84 m, so
   wall hits land within ~1 cell and the 1.05 m entrance's 2.5-cell slack at ρ_responder survives; free carving
   stops 2σ short of each surface.

**Measured (dev, ci_tiny seed 0, the 1.7 evidence spool).** Frame anchors (first 10 s vs GT): agent 0 5.0 mm /
0.07°, agent 1 (own entry pose, solo run) 6.2 mm; agent 1 placed through the 1.6 merge: 3.0 cm / 0.47° (the merge's
accuracy). Merged map vs GT: coverage 0.93, free precision 0.97, occupied precision 0.71. Three runs (first-valid
times are replan instants: the route appeared within the preceding 5 s):

| run | first valid egress route | final exit, length (GT geodesic) | coverage |
|---|---|---|---|
| agent 0 alone | 195 s | west, 11.26 m (11.21) | 0.89 |
| agent 1 alone (its own entry anchor) | **55 s** | north, 4.44 m (4.44) | 0.43 |
| merged (agent 1 joins at t_merge 138 s) | 140 s | north, 4.44 m (4.44) | **0.93** |

Every route is valid on every cell against GT `trav_responder`. **Reading:** the target is in agent 1's room, so
agent 1 alone has a valid route at 55 s; the merge brings it into the shared (broker) map only at 140 s, because
the one verified inter-agent loop arrives at 138 s. What collaboration adds here is the joint map (coverage 0.93 vs
0.89 / 0.43) and, for agent 0, the shorter exit 55 s before its own exit leg would have found any. The 85 s of
merge latency is a measured cost of a single late loop: the Phase 3 case for earlier, prioritized exchange.
Deterministic: trace, routes and maps byte-identical across runs. 257/257 C++ tests (7 new planning tests).

**What the numbers do not show (open items).**
1. Occupied precision 0.71 is a convention effect, measured: of the 783 map-occupied, GT-free cells, 726 (93 %)
   are exactly one cell from GT structure, 56 two cells, 1 three, none further. Walls are thin planes on cell
   boundaries, so their hits land in the last GT-free cell. It costs clearance along walls (the north entrance's
   traversable band is 5 cells at ρ_responder) but blocked no route here.
2. Stray "free" cells beyond the walls: rays slipping past the thin wall meshes into space GT counts as structure.
   Disconnected from every route; costs ~3 % free precision.
3. The policy is logged, not executed (replay follows the scripts): there is no exploration-efficiency number yet;
   that needs closed-loop live runs. Nearest frontier is the baseline for the owed Domain 3 survey (FUEL, TARE, FAR).
4. One scene, one seed. The egress timing depends on where the scripts face; seed 1 is the next check.


## 2026-10-15 — D46: 1.9 evaluation harness v1 — every §30.2 metric, gated by the C++ registry, cross-checked by evo

**What.** `infra/scripts/report.sh` over one Phase 1 run chain (sim_collab → recon → mission): the C++ tool
`ariadne_report_check` dumps the D31 registry (single source of truth) and later gates the report with
`MissingMetrics` / `UnknownMetrics`; `python/analysis/report.py compute` (pinned analysis image, evo 1.31.1)
computes the 27 metrics with the definitions in `docs/design/HARNESS_1.9.md`, checks the gates (all input manifests
clean, the mission's and the spool's T_O0_O1 identical) and three oracles; `render` draws the figures and a
self-contained `report.html` (figures embedded, every number beside its definition, inputs with sha256).

**Choices.** (1) RPE pairing = the C++ `rpe_1s` pairing, (i, i+20) for every i, which is evo with
`all_pairs=True, delta_unit=frames`; evo's default (non-overlapping pairs) would differ. (2) Closure metrics use
denominators the harness cannot inflate: precision over everything ACCEPTED (25/26, the probe's false accept stays
visible), false-closure rate over the edges that reached the map (0/20). (3) Distance travelled from GT (VO jitter while
agent 1 is parked would inflate the estimate; both reported). (4) Calibration and energy are `[PROXY]`, labelled on the
page. (5) evo sees agent 1's timestamps offset by 10⁶ s: both agents share the sim clock and evo associates by time.
(6) Rotation angles via atan2: arccos lost ~1e-6 relative at mrad level and failed the first evo comparison.

**Measured (dev, the committed 1.6–1.8 evidence chain).** All 27 metrics emit a number: **gate PASS**. Oracles:
per-agent ATE/RPE equal the C++ results; merged ATE 0.0097643 m over 7962 poses = 1.6; evo APE translation and
rotation and RPE translation and rotation agree to ≤ 4.5e-14 relative (APE rotation 1.5e-11) over 7962 / 7922
samples. This is the **first
RPE oracle** (1.5's evo run never captured RPE). Headline values: ATE 9.76 mm / 0.26°, RPE (1 s) 0.86 mm / 0.018°,
inter-agent edge RMSE 12.0 mm, closure precision 0.962, false closure rate 0, Chamfer 0.126 m, completeness 0.66,
coverage 0.93, PSNR 17.5 dB (untrained init), 1.27 MB total (2.2 kB/keyframe), first valid egress 140 s, route
validity 1.0. Harness output byte-identical across reruns (except the analysis image id recorded in inputs.yaml).

**Open.** Single run chain, single seed: no intervals, no cross-run calibration plot, no §31 matrix yet; outputs YAML
not parquet. `time_to_first_usable_map_s` is defined as the first agent-ρ route `[ASSUMPTION]`.


## 2026-10-16 — D47: 1.10 operator console stub — replayed telemetry over WebSocket, causal message times

**What.** `python/operator_console/` (FastAPI + a Canvas 2D page, pinned `infra/docker/console.Dockerfile`,
`infra/scripts/console.sh check|serve`): a committed run chain replayed as telemetry frames at 5 Hz over
`/ws/telemetry` (agents from ESTIMATED poses through the mission's world anchor, agent 1 only once merged; cumulative
wire bytes per class + rate; the latest map snapshot and routes; incident events), REST `/api/meta`, `/api/map/<i>.png`,
`/api/frame?t=` (a static frame with all events so far: screenshots and thumbnails). Design `CONSOLE_1.10.md`.

**Producers gained time-resolved outputs, with identity proven.** `ariadne_sim_collab` writes `ledger.csv` (one row per
message the ByteLedger counted) and `ariadne_mission` writes a map snapshot per replan. A full dev rerun of the 1.6/1.7
chain is byte-identical to the committed evidence (candidates, probe, results, both VO estimates, and the render
package npz `9ec2f672…`, so the 1.7 Kaggle render stays valid); the mission rerun is identical to the 1.8 evidence.

**Causal message time (a correction, not a choice).** Wire stamps are keyframe times and are not causal: a match
request carries agent 0's keyframe (~138 s), its response and edge agent 1's (~20 s), so a stamp-keyed console would
show edges before the requests that triggered them. The ledger therefore also records a causal send time —
descriptor = its keyframe; request = the later of the two keyframes; response = request; edge = response — the rule
1.8 uses for t_merge. Wire stamps are unchanged, so encoded bytes and the 1.6 evidence are identical.
`[ASSUMPTION]` zero processing and link latency.

**Checks (`console.sh check`, PASS on the dev chain).** Ledger totals = results.yaml (310 messages, 1,270,717 B, per
class bytes and counts); the console's merge event = the mission's t_merge_s (137.95 s) and its first-valid-route event
= 140 s (one time base); no edge before its request, the first edge not after the merge; frames byte-identical across
builds. Visual check by headless screenshot at 150 s: map, both agents, egress route, gauges, timeline. Frontier
decisions appear only when the choice changes, labelled "logged decision · replay · not executed" (invariant 3).

**Not in 1.10.** 3D splat view in the console (the separate 1.7 viewer page exists), approvals (§23), raw vs restored
feeds, live hosting (Phase 8), budget gauges against a policy budget (Phase 3). A claude.ai artifact cannot reach a
localhost WebSocket, so the console is served locally (`127.0.0.1`, ssh tunnel from elsewhere).

## 2026-10-16 — D48: Phase 1 exit gate — one replay command, the §4 contracts wired and counted, determinism vs evidence

**What.** `infra/scripts/phase1.sh` replays the recorded bag through every Phase 1 stage from one commit, into one
directory (`evaluation/runs/phase1/exit_gate/<scene>-<UTC>/`; large outputs in `build/phase1_gate/`, sha256
committed): build + full ctest, the protocol tests, 1.6 clean + fault-injected runs concurrently, fault identity,
1.7 prepare → fresh Kaggle render → score, 1.8 mission ×3 + eval, 1.9 report, 1.10 console check. Then
`python/analysis/phase1_gate.py` (config `configs/benchmark/phase1_gate.yaml`) writes `gate.yaml` and one verdict.
Design `PHASE1_GATE.md`.

**The §4 contracts were not exercised before (a correction).** `runtime::Keyframe` (§4.2) and `KeyframeBundle` (§18.2)
were defined, but no run built them: the VO hook passed loose fields, and `ariadne_mission` declared its own `Keyframe`.
Now `RunStereoVo` builds both at each keyframe. The Keyframe holds the graph node pose, its absolute covariance, SGBM
depth and the observed landmarks (`StereoVo::KeyframeLandmarks`); the Bundle is built by `BuildKeyframeBundle` at the
graph's version. The 1.6 exchange completes the Keyframe with ORB and the mean-bit descriptor, and derives its wire
features from it; the spool writes the Bundle. `StereoFrame` (§4.1) gains `agent` and `id`, asserted in sequence.
Agent 1 now checks every decoded POSE_EDGE field for field against the encoded one, and checks that the merge's edge
was received. The merge itself still comes from the broker's double-precision verifier output: the broker owns the
global estimate. Integrating received edges into agent 1's graph is Phase 3, and merging from the float32 wire copy
would change results. `sim_collab` writes `contracts.yaml` with the per-hop counts, the per-field gaps, and the
declared-unset fields. Every output stays byte-identical: the EuRoC MH_01 estimate and results, and the 1.6 CSVs,
results and both estimates, all match. The gate re-proves this against all committed evidence.

**Determinism means agreement with independent runs, with named exclusions only.** The gate compares its outputs
file by file with the 1.6–1.9 evidence, which came from four other commits and processes. Exclusions are per-file
line regexes, and each must carry a reason, `run_identity` or `timing`; the checker refuses one without a reason.
The dev run's 12 differing files were all run ids, paths, commit, the spool.yaml hash (it embeds the run id), or
wall clock. `report.yaml`'s `inputs_clean` stays compared on purpose, so a dirty run cannot pass.

**Dev result (dirty tree, 2026-10-15).** Every I1, contract, stage and determinism check passes, except
`inputs_clean` (expected on a dirty tree). The **fresh Kaggle render is sha256-identical to the 1.7 render** (`cc3a2498…`):
gsplat rasterization is deterministic across T4 sessions for this package. Gauge note: keyframe 0's absolute covariance
is zero by definition (the gauge), counted as `gauge_zero_cov`, not as missing.

**Not gate items, reported.** §37.7 round trips for message classes with no schema yet, the ≤15% byte prediction
and stale-epoch rejection (Phase 3). Nightly CI of the thin run is open: it needs the 12 GB bag and a GPU.

## 2026-10-16 — D49: no nightly CI — the thin run is a local pre-merge gate

**Decision (maintainer).** No nightly CI for the end-to-end run: "this is just a hackathon project", and "before we
check off each feature, we do run e2e once". Nightly infrastructure (a self-hosted runner, the 12 GB recording, a GPU)
would cost more than it saves before Oct 24. Unit tests alone are not enough: the 257 tests stayed green while §4.2 and
§18.2 went unused (D48). CLAUDE.md §4 is updated to say this.

**What.** `infra/scripts/phase1.sh --premerge` runs the same chain and checks on the working tree, into
`build/premerge/`, in ~20 min with no network. It runs before commit commands are handed over for any change that
touches the pipeline, and before a plan item is checked off. The full run (a clean commit and a fresh Kaggle render)
runs at phase gates and before submission. The GitHub `ci.yml` (build, test, cross-architecture determinism) is
unchanged.

**The render is reused, never faked.** Three Kaggle sessions gave the same `renders.npz` (`cc3a2498…`), so renders are a
deterministic function of the render package. Full runs store them in `build/render_cache/<package sha256>/`.
Pre-merge reuses them only for a byte-identical package. Otherwise it stops and asks for a full run (that branch was
tested in isolation). The splat check and the verdict line say "REUSED … NOT exit-gate evidence". The cache was seeded
from the exit-gate evidence run's render, which predates the cache code.

**One mode-only exclusion.** In pre-merge, `report.yaml`'s `inputs_clean` is excluded, under its own reason
`premerge_tree`. Full runs still require it to be true.

**Verified.** A pre-merge run on the working tree (commit e10dd8e plus these edits) gave VERDICT PASS, 54/54, in
19.5 min. From Phase 2 on, each phase's gate compares against its own committed evidence through a new config.

## 2026-10-16 — D50: XFeat runs as a cv::dnn graph WE assemble from upstream's weights; a GEMM reference checks it

**Context.** D15 chose "XFeat on OpenCV 5 primitives, no ONNX". The Phase 2 smoke test (SMOKE §4p) verified it.

**What.** `cpp/perception/features/xfeat/`: `atb.hpp` (a tensor blob, no dependencies) and `xfeat_net.{hpp,cpp}`. The
weights are upstream's (`verlab/accelerated_features` e92685f, Apache-2.0, credited), exported offline by
`python/model_tools/xfeat_reference.py` in the pinned `ariadne-xfeat-ref:1` image (`infra/scripts/models.sh xfeat`:
pinned commit, sha256-checked `.pt`, `--network none`, `weights_only=True`). BatchNorm (affine=False) folds into the
convolutions at load. Two engines run the same folded tensors:
- **`opencv_dnn_graph` (runtime):** the network is built layer by layer with `Net::addLayer`/`connect` (Convolution,
  ReLU, Pooling, Eltwise, Resize, Sigmoid). The 8×8 unfold is an exact one-hot stride-8 convolution. 12–15 ms per
  640×480/736×480 frame with 8 threads, 27–33 ms with one.
- **`gemm_reference`:** every op written out (im2col + `cv::gemm`). It is the readable reference, ~2.5–2.9 s because
  `cv::gemm` without external BLAS runs at 0.88 GFLOP/s.

**Why the graph API and not ONNX.** It keeps D15's claim intact: the architecture and forward pass are ours, OpenCV
5's dispatched convolution kernels execute it, and no importer coverage risk exists. Why not only the GEMM path: it
is ~200× slower.

**Acceptance (both engines, `tests/unit/test_xfeat_reference.cpp`, against upstream's own PyTorch outputs):** maps
≤ 8e-6 relative, 4096/4096 keypoints at the same pixel on both inputs, min descriptor cosine 1.000000. Upstream's
quirks are reproduced, not fixed, and are listed in SMOKE §4p. The test SKIPS without the export, but FAILS under
`ARIADNE_REQUIRE_MODELS=1`, which the pre-merge gate now sets. The engine that ran becomes a manifest field when the
engine is wired into the pipeline (2.1).

## 2026-10-16 — D51: the Phase 2 solvers run on OpenCV 5's `LevMarqBase`; our residuals, Jacobians and sparse algebra behind it

**Context.** Arch §11.6 specifies LM as `(H + λI)δ = −g` with its own λ schedule (λ₀ = 1e-4·max diag H, ×2 on
reject, ÷3 on accept, 15 iterations). The maintainer asked that Phase 2 use OpenCV natively as far as it honestly
goes. OpenCV 5 ships `cv::detail::LevMarqBase`, a Ceres-style LM driver over a pluggable `LevMarqBackend`.

**Decision.** The pose-graph solver (and, when 2.3 lands, the windowed BA) implements `LevMarqBackend`; OpenCV's
driver runs the LM loop. Ours: residuals, exact SE(3) Jacobians (`J_l⁻¹` with Barfoot's Q), block assembly, the
sparse `SimplicialLDLT` decomposition with a positive-pivot guard, the gauge (the anchor is excluded from the
parameter vector), and the left `exp` update. Design: `POSE_GRAPH_2.md`.

**This deviates from §11.6, and why that is acceptable.** OpenCV damps with Marquardt's `λ·diag(JᵀJ)` (clamped to
[1e-6, 1e32]) and adapts λ by step quality, not ×2/÷3. The optimum depends on the cost, not on the damping path, so
the GTSAM oracle (the §11.6 reference check) is unaffected. Every `Settings` field is set explicitly from config and
recorded in the manifest. Geodesic acceleration starts OFF and becomes an ablation. The arch doc §11.6 is updated
(§49 ritual).

**Not changed.** The two-frame BA stub keeps its own §11.6 LM (Phase 1 outputs stay byte-identical).
`cv::detail::PoseGraph` is NOT used: our pinned OpenCV has no `HAVE_EIGEN`, so its `optimize()` throws. Rebuilding
OpenCV would change the build hash, breaking the Phase 0 gate and the COOL baseline. Its quaternion-space residual
also differs from §12.2.

**Risk.** `cv::detail` is an internal namespace; the OpenCV tag is pinned (D7), so it cannot drift under us. The
contract (energy = ½Σ‖√Ω r‖², `jtb = +Jᵀr`, `solveDecomposed` solves `(JᵀJ+D)x = −right`) was read from
`levmarq.cpp` and OpenCV's own `PoseGraphLevMarqBackend`. A one-step linear test pins it.

## 2026-10-16 — D52: the Phase 2 solver cores, verified: PGO agrees with GTSAM 100/100; windowed BA matches OpenCV's PnP refine

**PGO** (`POSE_GRAPH_2.md`):
- exact `J_l⁻¹`, verified against its definition at every angle class;
- gauge null space exactly 6, "the same ξ on every node";
- OpenCV `LevMarqBase` driver with the contract pinned by a one-step linear test;
- **GTSAM 4.3 oracle 100/100**, half the graphs with full correlated information: energy ≤ 1.5e-12 relative, poses
  ≤ 1e-6 absolute (99) or ≤ 1e-6 Mahalanobis² (all).

Deviations recorded in the design:
- dead-reckoning shared init, because GTSAM's `InitializePose3` threw on one case [UNVERIFIED, open for 2.5];
- the anisotropic-noise generator bug, which drove both solvers to different local minima (an initialisation test,
  not a solver test);
- the scale-invariant pose criterion.

`pgo_digest` is required in every manifest; it is same-architecture only, because the SE(3) maths uses libm.
After a solve, node covariances are the marginals at the optimum, or NaN by explicit choice, never stale.

**Windowed BA core** (`LOCAL_SLAM_2.3.md`):
- `window_ba.hpp` (OpenCV-free): left-perturbation `T_W_C`, Huber/IRLS assembly, Schur elimination after damping;
  the guard refuses non-PD systems;
- `window_ba_levmarq` on OpenCV's driver;
- Jacobians match finite differences, including 0.05 m depth and |x/z| up to 2.5 (exit-gate item);
- gauge exactly 6, with landmark rows `[I, −[X]×]`;
- the gradient is exactly the robust cost's; Schur equals the dense solve;
- **one free pose agrees with `cv::solvePnPRefineLM` to 1.3e-9 m / 2.8e-9 rad**;
- the camera covariance equals the dense-inverse block.

**Not yet:** BA integration into `StereoVo` (window, keyframe score, lifecycle, exit edges) and the EuRoC/sim runs
against the targets fixed in `LOCAL_SLAM_2.3.md` §8.

## 2026-10-16 — D53: Phase-1 correctness fixes found by the first full EuRoC sweep (byte-identical against the Phase 1 evidence)

The Phase 1 evidence covered MH_01 only. Running all 11 EuRoC flights exposed four defects in 1.x code. Each is
fixed with a test. The pre-merge gate stayed byte-identical (54/54, ctest 286/286,
`build/premerge/aadffb8df5790ca0-20261016T173330Z`). That was verified on an intermediate working tree, before the
D54 changes. D53 and D54 land in the same commit, so no commit carries D53 alone:

- **OpenCV 5.0.0 USAC defect (SMOKE §4q).** `findFundamentalMat` (`UsacParams`) can throw `!model.empty()`. It
  aborted all six V-room flights; reproduced standalone, 62/200 seeds on V1_01 frame 1009. Only that throw is
  handled: up to 3 redraws from the same Philox stream, then the check is skipped for that frame. Both outcomes are
  counted. The first draw is unchanged. Env-gated replay test: `ARIADNE_USAC_FIXTURE`.
- **Unstable coasting.** While lost, the constant-velocity model re-derived its velocity from its own extrapolations.
  Rounding error in R's orthonormality grew ≈ 2.4× per frame, and MH_04 went non-finite after 48 lost frames.
  The measured velocity is now frozen while coasting.
- **Silent NaN.** A non-finite pose or ATE now stops the run with the frame, the lost-streak length and the last
  finite position. It previously produced a NaN score with exit 0.
- **`PoseGraph::AddEdge`** refuses information that is not finite, symmetric and PSD. The SE-Sync cubicle/rim files
  carry 5021 and 8815 indefinite matrices; our LM had correctly refused to factor them, while GTSAM did not check.
  The SE-Sync test excludes exactly those two files and switches to an energy + stationarity criterion; evidence in
  `POSE_GRAPH_2.md` §5.

## 2026-10-17 — D54: frontend re-baseline — tracked points become map points again (maintainer: option A)

`docs/design/AUDIT_2.3.md` traced the hard flights' lost frames to how tracked points become 3-D points. Feature
quality and the 2.3 backend were not the cause.

- Fixes 1 and 2 were proven by one-setting interventions before landing.
- Fix 3 (exposure) was proven by an on/off pair after implementation.
- Fix 4 (time-based) is unit-tested, but **no intervention showed it matters on EuRoC**. It is a correctness fix for
  dropped frames.

All are done with the current features. These are Phase-1 frontend fixes, so the v1 numbers and the Phase-1 sim outputs
change **by design**, and the gate is re-pinned from a clean-commit full run.

1. **Online stereo row refinement.** The V2 recordings sit ≈ 1.7 px off their published calibration (rows), and the
   1 px row gate threw away 97% of correct stereo matches.
   - The residual `dy = a + b x + c y` is fitted from round-trip-consistent LK matches with no row gate. Samples
     pool across keyframes, fitting is trimmed least squares, and estimates beyond the search bound are refused
     loudly.
   - The fit is folded into the right rectification map as `A·P2`. P2, Q and `Rig()` are untouched.
   - Acceptance: V2_01's residual drops from 1.46 px to 0.09 px.
   - It is disclosed as online calibration refinement in `results.yaml` and in the manifest field
     `stereo_row_refinement`.
2. **Far points kept.** The 60-baseline discard misread ORB-SLAM, which classifies far points rather than dropping
   them (ORB-SLAM2 §III-A). The relative-σ gate is now the depth bound (≈ 9.6 m on EuRoC). Multi-view triangulation
   beyond it is not implemented.
3. **Exposure match** before stereo LK and SGBM: a global gain/bias match, needed for EuRoC's per-camera
   auto-exposure. It lands on evidence: V2_03 v1 lost 1026 → 192 frames.
4. **Time, not frame counts.** Coasting predicts by elapsed time (velocity per second), the keyframe cap is
   `max_interval_s`, and RPE pairs samples 1 s apart by timestamp. V2_03 drops 414 frames.

σ was re-measured by the same protocol (MH_01 at σ = 1): 0.453 px, within 10% of 0.46, so 0.46 is kept.

**Result** (11 EuRoC flights; v1 = the 1.5 placeholder, v2 = 2.3; ATE in m):

| flight | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 before | 0.126 | 0.081 | 0.140 | 3.93 | 0.82 | 0.152 | 0.218 | 1.18 | 2.36 | 2.47 | 3.72 |
| v1 D54 | 0.112 | 0.128 | 0.151 | 1.60 | 0.257 | 0.162 | 0.226 | 1.74 | 0.105 | 0.213 | 1.65 |
| v2 D54 | **0.075** | **0.075** | **0.093** | 2.78 | 0.560 | **0.078** | **0.113** | 0.439 | **0.083** | **0.124** | 2.12 |

- **2.3 against its target** (`LOCAL_SLAM_2.3.md` §8): v2 beats v1 on **8/11** and lies within the worst published
  stereo class on **7/11**.
- **v1 got worse on 5 of the 6 already-clean flights**: MH_02 0.081 → 0.128, V1_03 1.18 → 1.74, and slightly
  MH_03, V1_01 and V1_02. **[UNATTRIBUTED]**; the likely cause is far points feeding the two-frame stub. Stated, not
  hidden (invariant 7).
- After D54, **55.6% of window solves stop at the 10-iteration cap** (`found = false`); it was 23% at σ = 1 before
  D54. §11.6 says 15. Open.
- **Every remaining failure is a flight that still loses more than 100 frames** (MH_04 266, V2_03 181, V1_03 139,
  MH_05 100). There the window is worse than v1: lost stretches poison the map (AUDIT §4), and a 10-keyframe window
  spreads the damage.

**Next:** lost-frame handling, then 2.1 / 2.4.

- **Lost-frame handling (AUDIT §4):** no keyframes and no new points at a coasted pose, a new map segment on loss,
  and a lost-status field on keyframe records.
- **2.1** (XFeat + re-association) and **2.4** (IMU bridging), for the losses that remain.

## 2026-10-17 — D55: the gate's determinism references re-pinned to a clean post-D54 run; the sim change attributed

D53/D54 change the Phase-1 VO **by design**. The full gate on the dev tree had 32 determinism failures, all
VO-derived; the digests (det_math, philox, degradation, lens) were identical. This decision re-pins
`configs/benchmark/phase1_gate.yaml` to a full run from clean commit 150ce28 with a fresh Kaggle render, and records
what changed on the sim and why, so the new references do not bake in an unexplained regression.

**The headline sim number got worse, but it measures one loop edge, not the VO.** `merged_two_agent_ate_rmse_m`
rose from 9.8 to 14.4 mm.
- The 1.6 placeholder merges the two maps with ONE edge: the best-ranked verified candidate
  (`ariadne_sim_collab.cpp`, "merge with the best verified edge").
- The keyframe timing changed, so a different keyframe pair now ranks first. The reference edge was 11.6 mm / 0.19°;
  the new one is 18.8 mm / 0.33°.
- In every variant below, the top three verified edges span 10–29 mm and 0.18–0.53°. The merged number is one draw
  from that spread. The depth-cap variant drew a 10 mm / 0.18° edge and scored 9.1 mm.
- The real merge is the Phase 3 pose graph over all edges. Until then the merged number is reported **next to** the
  per-agent numbers, never alone.

**Per-agent attribution** (the same bag, the same build; ATE translation mm / rotation mrad):

| run | agent 0 | agent 1 | merged mm | merge edge |
|---|---|---|---|---|
| reference (Phase-1 evidence) | 9.9 / 3.2 | 4.4 / 2.6 | 9.8 | 11.6 mm, 0.19° |
| D54 (current) | **8.4** / 4.9 | **4.3** / 4.6 | 14.4 | 18.8 mm, 0.33° |
| row refinement off | 7.6 / 4.3 | 4.6 / 4.2 | 13.3 | 18.1 mm, 0.31° |
| exposure match off | 8.5 / 4.8 | 4.5 / 2.9 | 15.2 | 21.4 mm, 0.37° |
| 60-baseline depth cap restored | 10.5 / 4.5 | 5.6 / 3.2 | 9.1 | 10.0 mm, 0.18° |

- **Translation improved** on both agents.
- **Rotation RMSE rose** by about 50% on both agents: 3.2 → 4.9 and 2.6 → 4.6 mrad.
  - Restoring the depth cap recovers part of it on agent 1 (3.2 mrad), but costs translation on both agents (10.5,
    5.6 mm). Far points are kept.
  - Row refinement and exposure match each move it by less than the agent-to-agent spread.
  - The rest is **[UNATTRIBUTED]** among the remaining sim-relevant D53/D54 changes: time-based motion and keyframe
    timing, the live-track culling exemption, the χ² statistics. sim_v1 runs no window BA.
  - EuRoC does not show it. v2 rotation RMSE held or improved after D54 on every clean flight (e.g. MH_01 59 → 47,
    V1_03 231 → 177 mrad), and so did v1 (MH_01 77 → 54, MH_02 65 → 39 mrad).
- **Row refinement on a perfect rig** fits a −0.24 px offset (agent 0) from one 115-match pool. The post-refinement
  check reads +0.18 px, so that first estimate was mostly scene-dependent LK bias.
  - On EuRoC the same check confirms the real offsets: V2 at 1.3–1.7 px, and V1_01 at −0.30 px with a residual of
    0.01 px.
  - A magnitude gate (e.g. 0.5 px) would wrongly drop V1_01, so none is added.
  - A residual of ≤ 0.3 px is within the 1 px row gate and is harmless here: sim with refinement off is better on
    agent 0 and worse on agent 1.
  - Open: decide from several independent pools instead of the first one to reach `row_min_matches`.

Variant runs: `build/sim_attr/{current,norow,noexp,zcap}`. The depth-cap variant came from a temporary, env-gated
diagnostic build, reverted at once; it is not in any commit.

**The re-pin.** `configs/benchmark/phase1_gate.yaml` now points every reference at
`evaluation/runs/phase1/exit_gate/aadffb8df5790ca0-20261017T072201Z`. That is one full gate run from clean commit
150ce28 (`git_dirty: false`) with a fresh Kaggle render.
- Its own verdict against the old references was FAIL on exactly the 32 VO-derived comparisons, and PASS on the
  other 22: I1, the §4 contracts, every stage.
- The det_math, philox, degradation and lens digests were unchanged.
- A pre-merge run against the new references passed **54/54**, render reused (`build/premerge/…-20261017T080657Z`).
  It ran on a working tree whose only code change writes a new per-frame `vo_trace.csv` (D56), which no gate
  comparison reads.
- The old per-item evidence stays committed where it is.

## 2026-10-17 — D56: lost-frame handling — the visual rotation bridge, a decaying velocity, tagged segments

Design and evidence: `docs/design/LOST_FRAMES.md`. It answers `AUDIT_2.3.md` §4.

**Diagnosis** (new per-frame `vo_trace.csv`):
- In a lost stretch the landmarks fail, but the 2-D tracks survive (40–360).
- Coasting at a frozen constant velocity turned MH_04's 10 s, 9 cm dark hover into 7.6 m and 117°.

**Decision (v2 configs; v1 unchanged, it stays the 1.5 baseline):**
- **`lost.motion: bridge`.**
  - Rotation is MEASURED from the 2-D tracks by seeded USAC H and E (OpenCV 5 `geometry`), selected by ORB-SLAM's
    published score rule (T-RO 2015 §IV-B).
  - It is measured from an ANCHOR (the last measured frame, or the last frame 20 tracks reach back to), so there is
    one two-view estimate per frame, not a chain.
  - Translation is the last measured velocity, clamped to 3 m/s and decaying as `exp(−t/0.5 s)`.
- **`lost.reanchor: every_frame`.** The `gated` alternative (ORB-SLAM3's no-keyframes-while-lost) RAISED lost
  frames (V2_01 20 → 120 bridged) and is rejected: it is kept as an ablation.
- **Provenance:** `PoseSource {measured, bridged, coasted}` and a segment counter (loss onsets) on every frame and in
  `runtime::Keyframe` (`pose_source`, `segment`). Also in `results.yaml` `lost:` and manifest `lost_handling`.

**Evidence** (11 EuRoC flights):
- `coast` / `every_frame` reproduces D54 byte for byte, except wall-clock timing.
- The bridge: MH_04 2.78 → **0.36** m (rotation 55° → 1.8°), V2_03 2.12 → **1.27**, MH_05 0.56 → **0.35**.
  - Small costs: V1_03 0.44 → 0.56, attributed to the decay (no decay 0.46, τ 0.2 0.70); V1_02 0.113 → 0.137.
- **Against `LOCAL_SLAM_2.3.md` §8:** within the published class **9/11** (was 7/11); better than the placeholder
  **10/11**.
- **Held-out per-pair accuracy** (frame-to-frame bridge on 25,606 tracked pairs vs ground truth): H median 0.21°,
  E median 0.036°, p99 ≈ 1.1°; robust σ ≈ (0.09°, 0.09°, 0.04°) per pair.
- Lost-frame COUNTS barely move (MH_04 278 → 264). This limits the damage of a loss; 2.1 and 2.4 must cut the losses.
- Cost: VO tracking 9.3 → 12.4 ms per frame (the bridge runs on every frame, for the held-out check).

**Rejected on the way, each by a failing test or a measurement (LOST_FRAMES.md §4):**
- ORB-SLAM's ratio on inlier counts (blind to sub-pixel parallax; 0.1°/frame H bias, chained).
- "E unless it disagrees with H by > 5°" (breaks at anchor baselines).
- ReconstructF's cheirality test: `recoverPose`'s default overload counts only points within 50 baselines, none at
  1–2 cm/frame. This silently dropped E until `distanceThresh` was passed.

**Open:**
- Bridged edges keep the uninformative covariance until a NEES check. The measured σ is the input.
- The worst blur gaps need 2.4's gyro.
- Limit the bridge to lost frames once it is validated.

## 2026-10-17 — D57: two cheap accuracy levers measured and closed; the window keeps 10 iterations

Run from the committed D56 tree; the shipped `euroc_v2.yaml` reproduced the D56 sweep exactly on all 11 flights.

- **ORB-SLAM's trajectory protocol** (frames re-composed on the final keyframe poses,
  `System::SaveTrajectoryEuRoC`) is now reported next to the causal score, as `results.yaml` `keyframe_refined`
  (window backend only, so v1 bytes are unchanged). It differs by ≤ 1 mm on every flight.
  - The protocol caveat in `LOCAL_SLAM_2.3.md` §10 is closed: scoring is not part of the gap to ORB-SLAM3.
- **Window max_iterations 15** (§11.6) vs 10:
  - The share of solves that end at the cap drops from 45–69% to 28–47%.
  - ATE: MH_01 0.075 → 0.072, MH_03 0.093 → 0.086, V2_01 0.071 → 0.069; but MH_05 0.349 → 0.360 and **V1_03 0.56 →
    0.96**. Still 9/11 within the published class.
  - **Kept at 10.** At 15, a third of the solves still hit the cap, so the cap is not what limits them. The likely
    cause is the termination tolerances (`rel_energy_tolerance` 1e-6) [UNVERIFIED].
- **What remains of the gap to ORB-SLAM3:** data association (re-association, 2.1), loop closure (Phase 3), the IMU
  (2.4). It is not the protocol, and not the iteration budget.

## 2026-10-18 — D58: 2.4 step 1 — attitude_only ships for EuRoC v2: gyro rotation while lost, gravity on keyframes

Design and evidence: `docs/design/INERTIAL_2.4.md`.

- **Validated first.** The gyro integral matches the ground-truth camera rotation to **0.019° per frame pair** with
  the ground-truth bias removed (26,530 pairs, 11 flights); 0.23° raw. Timing is aligned within a sample.
- **Gyro bias is VO-aided** (a robust median of `(Log R_gyro,raw − Log R_vo)/dt` over 200 tracked pairs; seeded by a
  stationary test). Mahony cannot observe the gravity-axis bias.
- **While lost, the gyro replaces the D56 visual bridge's rotation** (`PoseSource::kGyro`).
  - EuRoC v2 average ATE 0.289 → **0.223 m**; V2_03 1.27 → **0.68**, V1_03 0.56 → 0.42, V1_02 0.14 → 0.10.
  - Costs: V1_01 +17 mm, MH_04 +30 mm.
  - Stereo class: **10/11**. Stereo-inertial class: not yet (`full` needed, as planned).
- **Gyro-seeded LK** (Hwangbo 2009) is built as a one-hunk seam in `klt_tracker.*` (maintainer-approved), tested,
  and **off**. Lost frames −3% only; V1_03 0.42 → 0.82.
- **Gravity** (Mahony 2008 Theorem 5.1, the paper's kP = 1, kI = 0.3) is on every keyframe record. Its error against
  ground truth is a median 1.3–4.2° per flight, 15° worst: MAV dynamics leak through. Recorded so that a
  GRAVITY_PRIOR uses an honest σ. The factor itself is 2.5's (pose graph). [Corrected in D59: the plan puts
  producing gravity-prior edges in 2.4; 2.5 consumes them.]
- `imu.mode` defaults to `none`: v1, `sim_v2` (no sim IMU plumbing yet) and the sim gate are unchanged. Pre-merge
  **PASS**.
- **Maintainer decision recorded for `full`:** window exit carries a loose bias prior forward; poses and velocity
  still drop and emit an edge.

## 2026-10-19 — D59 (part 2): 2.4 full — inertial window BA, initialisation + refinement, opt-in `euroc_v2_full`

Design and evidence: `docs/design/INERTIAL_2.4.md` §4 (as built). Part 1 (8691fb3): the preintegrator and factor.

- **Window BA.** Velocity and biases are a separate segment of the reduced system (`[poses | v,b_g,b_a | landmarks]`):
  Schur, marginals and the exit edge unchanged; with no inertial blocks every number is bit-identical (MH_01 on
  `euroc_v2` reproduces D58 byte for byte). Energy ½rᵀΩr, no kernel; bias random walk; window[0]: pose fixed,
  v and b free, bias prior = previous marginal × 10 (carried by the random walk when degenerate or unsolved).
  Tests: FD gradient and Hessian of every inertial block, Schur = dense, noise-free VI window → truth.
- **Correction while building (§4.2):** the IMU does not observe velocity (Galilean null space, tested), so an inertial
  chain ties keyframes only out of a group with two pose-determined keyframes. **Maintainer decision:** in a window
  whose window[0] has no covisible partner, v₀ is held at the previous solve's value instead of holding lost
  keyframes at predicted poses. On EuRoC this triggers in up to ~25% of solves on the dark flights (MH_04, V2_03),
  ~0–5% on the easy ones: the oldest keyframe often shares fewer than `pnp.min_correspondences` landmarks.
- **Initialisation** (`slam/inertial/initialization.hpp`) after 10 measured keyframes: b_g by Gauss–Newton on the
  keyframe rotations (**changed from §4.3's "from the attitude tracker"**, recorded there), then v, g, b_a as a linear
  MAP (ORB-SLAM3's inertial-only MAP, scale known), g on the 9.81 sphere; refused on |g| ±10 % or χ²/dof > 100.
  The threshold comes from the 11 EuRoC inits checked against GT: the one off by 8.4° (MH_04) had χ²/dof 1296,
  all others ≤ 55. **Refinement** re-solves over all keyframes since init at +5 s and +15 s (ORB-SLAM3 §V-B).
  Synthetic: exact with a flat b_a prior (1e-6); b_a/gravity-tilt trade-off quantified (0.44° ↔ 0.076 m/s²).
- **EuRoC (`euroc_v2_full`, opt-in):** average ATE **0.223 → 0.137 m** (target ≤ 0.138 on the average); **per flight
  7/11** inside the stereo-inertial class (V1_03 0.254, V2_01 0.077, V2_02 0.107, V2_03 0.281 out). Inertial
  χ²/dof at the solutions 0.38–1.23 (validates the ×3 / ×9 inflation, frames and lever arm). Costs vs
  attitude_only: MH_01 +2 mm, V2_01 +7 mm.
- **Open defect (part 3):** the estimated gravity, in each keyframe's own body frame, degrades over a run on the hard
  flights (V2_03 median 3.5°, final 5.6°): window[0]'s full 6-DoF pose is the gauge, so gravity cannot correct the
  trajectory's roll/pitch drift (a VIO gauge is 4-DoF).
- **§4.4 not decided.** Fake outages (0.5–10 s, MH_01/MH_02/V1_01/V2_01): IMU propagation drifts less to 2 s (0.24 vs
  0.45 m), then diverges (10 s: 11 m vs 3.1 m), consistent with the gravity defect (δg T²/2). The metric also scored
  the wrong instant. Re-measured after part 3 with drift to ~2–3 s after re-anchoring; `decay` stays the default.
- Pre-merge **not run** (maintainer, 2026-10-18: only when absolutely needed); the EuRoC sweeps and unit tests are
  the evidence. `euroc_v2` (attitude_only) and `sim_v2` (none) unchanged.

## 2026-10-19 — D59 (part 3): bias prior never looser than the carried prior; gravity-in-window is a negative result

Design and evidence: `docs/design/INERTIAL_2.4.md` §4.8–4.9.

- **Gravity's direction in every window + rotate-back (a 4-DoF gauge in effect): built, tested, NEGATIVE, off**
  (`imu.full.estimate_gravity: false`, kept as an ablation). EuRoC average 0.137 → 0.566 m, 0/11: a 10-keyframe
  window cannot separate tilt from b_a, and the ×10 prior bookkeeping loosened the unobserved tilt every window
  (σ 3.6° → 12° on MH_01). ORB-SLAM3's local VI BA also anchors on fixed keyframes (§V-C, §VI-C); long-term tilt
  there comes from map reuse and loop closure (2.1 / Phase 3 here), so part 2's gauge stands.
- **Bias prior fixed by the same lesson.** The ×10 deflation applied to the whole marginal lost the prior's own
  information on unobserved directions (b_a in dark windows; V2_01 propagate 10 s: |b_a error| 0.15 → 2.0 m/s²).
  Now Λ_next = Λ_carried + PSD(Λ_marg − Λ_carried)/κ: only the window's added information is deflated.
  **EuRoC `euroc_v2_full`: average 0.137 → 0.121 m, per flight 7/11 → 9/11** (V1_03 0.357 and V2_02 0.101 out).
  Bias and velocity errors vs GT drop on every hard flight; V1_03's ATE rises 0.254 → 0.357 despite better states
  (reported as measured).
- **§4.4 `lost_motion`: decided by measurement, stays `decay`.** With `reanchor: every_frame` each lost frame is an
  inertial-window keyframe, so the prediction only seeds the solve: decay and propagate are identical to the mm.
  Full vs attitude_only across outages (2.5 s after re-anchoring): better up to 5 s (2 s: 0.18 vs 0.61 m), worse at
  10 s (4.6 vs 3.1 m): long IMU-only stretches diverge; bounding them belongs with map segments (2.1 / Phase 3).
- ctest 328/328. `euroc_v2` (attitude_only) byte-identical to D58 on MH_01; `euroc_v2_full` stays opt-in pending
  the maintainer's switch decision. Pre-merge not run (maintainer's instruction).

## 2026-10-19 — D59 (switch): `euroc_v2` is imu.mode full; attitude_only kept as `euroc_v2_attitude`

- **Maintainer decision.** EuRoC average ATE 0.121 vs 0.223 m, 9/11 vs 4/11 flights in the stereo-inertial class
  (INERTIAL_2.4 §4.9). Costs: MH_03 +6 mm; V1_03 (0.357) and V2_02 (+1 mm) stay out of class.
- `configs/slam/euroc_v2.yaml` takes the full-mode values of `euroc_v2_full` (removed: identical apart from its name;
  the D59 evidence directories keep that name). `configs/slam/euroc_v2_attitude.yaml` is the D58 configuration,
  kept as the baseline/ablation.
- **Verified by targeted runs, not the pre-merge (maintainer's instruction):** the new `euroc_v2` reproduces the
  committed full-mode V2_03 trajectory byte for byte, and `euroc_v2_attitude` reproduces D58 on MH_01 byte for byte.
  `sim_v2` (imu none) and the sim gate are untouched.
- **Ownership corrected:** gravity-prior edges (arch §10.4) are a 2.4 deliverable per the plan, produced here and
  consumed by 2.5's pose graph; the D58 entry had assigned them to 2.5. They are this session's next item.
- **Merge note for 2.1:** 2.1's D60 bars were measured against D58 (attitude_only). After merging, EuRoC needs one
  combined sweep; the two gains (re-association, IMU) both act on the lost stretches and need not add.

## 2026-10-19 — D70: chordal initialisation for the pose graph; pgo_v2 drops OpenCV's absolute damping floor; robust backend placed in `cpp/slam/robust/`

**Context.** Plan 2.5 opens with arch §13.1's chordal initialisation, ahead of the degeneracy gate → PCM → GNC-TLS
order of §14.5. Three open items came with it: (a) GTSAM `InitializePose3` throwing in the oracle, (b) parking-garage
needing 1000 LM iterations where GTSAM needs 8, (c) cubicle and rim refused for indefinite information. Design:
`ROBUST_BACKEND_2.5.md` §1–§3, §9.

**Decision.**
1. **Chordal initialisation** (`cpp/slam/pose_graph/chordal_init.{hpp,cpp}`). It solves one sparse linear least-squares
   problem for the rotation rows (κ-weighted, hard anchor, three right-hand sides). Each block is projected to SO(3)
   with `cv::SVDecomp` and the determinant fix, then translations come from a τ-weighted Laplacian. κ and τ follow
   SE-Sync's isotropic conversion (`τ = 3/tr(I_t⁻¹)`, `κ = 3/(2 tr(I_R⁻¹))`), shared with D74.
   - **Baselines (§49 ablation):** dead reckoning and a `cv::buildMST` spanning tree (Prim, highest κ).
   - **Gate (maintainer decision 2026-10-18, design §9):** on the SE-Sync graphs where our LM from dead reckoning
     fails, chordal + LM must reach GTSAM's minimum.
     - The failure set is **torus3D and grid3D**, where GTSAM's LM from the same start stops in the same local
       minima (29950.006 and 253372.10).
     - Chordal + LM reaches the reference (12117.637 and 42179.979) in 6 and 8 iterations. On all six graphs it
       ends within 7.0e-9 relative, stationary, in 4–15 iterations. MST rescues both as well.
     - This matches Carlone et al. 2015, Table I, on the four graphs the two studies share. **PASSED.**
2. **pgo_v2 = pgo_v1 with `clamp_diagonal` off**, as the preset `PgoV2()` and `SolvePoseGraph`'s default.
   - **Why:** OpenCV clamps the damping `λ·diag(H)` to an absolute [1e-6, 1e32]. Ceres, which it mirrors, clamps
     `diag(JᵀJ)` before applying λ.
   - **Effect on garage:** with the floor, 95 iterations and a stop 3.7e-5 above the minimum at production
     tolerances; without it, 15 iterations, at the minimum.
   - **Open item (b) is reduced, not at parity:** 15–18 iterations against GTSAM's 7.
   - **No regressions:** the 100-case oracle 100/100 (worst 1.48e-12 relative); the SE-Sync graphs reach the same
     minimum as GTSAM on all six; ctest 300/300 (8 env-gated skips, the oracle results cited separately).
   - **The struct defaults stay pgo_v1** because the windowed BA builds on them. Flipping them would have changed
     odometry silently.
   - **Guard owed to D73:** the floor also regularised a rank-deficient H, so GNC must anchor every connected
     component of the active graph, with its own test.
3. **Oracle corrections** (`python/oracle/pgo_gtsam.py`, oracles only):
   - GTSAM's `Rot3.Quaternion` does not normalise, and the g2o files are off unit length by up to 6.8e-7. The old
     SE-Sync oracle therefore solved a slightly different problem, which was garage's "different minimum". Now
     normalised, and `SeSyncBenchmarkGraphs` asserts equal initial energies.
   - GTSAM's chordal soft-anchors node 0 at unit precision, against edge weights near 15, which costs 6.5e-3 rad.
     The oracle scales the edge information by 2⁻⁵⁴ to make the anchor effectively hard.
   - New cases go in the branch cache only. The shared `.cache/oracle/*` is untouched.
4. **Open item (a), diagnosed.** `InitializePose3` throws on every case without a `PriorFactorPose3`, because
   `NonlinearEqualityPose3` is dropped and the anchor is left disconnected. With a unit prior, 3 of 100 cases throw.
   - The throw is in multifrontal Cholesky only: QR and sequential Cholesky succeed on the same linearisation, and H
     is positive definite. So this is a false positive in GTSAM, not a problem with the graphs.
   - The reference start falls back to GTSAM's chordal rotations. The original "case 2" was never committed and is
     not reproduced [UNVERIFIED].
5. **Placement (arch §3 updated, §49).** The degeneracy gate, PCM, GNC and the certificate go in `cpp/slam/robust/`,
   not `perception/` and `collaboration/`. All of them act on pose-graph edges and must also run intra-agent, with no
   broker. `loop_closure/` holds the intra-agent candidate source (D73).

**Not done here, stated.** No runtime path solves the pose graph yet, so the `pgo_init` config selection, a
`configs/slam` PGO entry and the manifest fields land with the first caller (D73). Open item (c) belongs to D74.
grid3D dominates the 2.6 h gate. The gate is an env-gated evidence run, never part of `--premerge`.

**Evidence.**
- `PgoInit.SeSyncDeadReckoningGate`, `PgoOracle.SeSyncBenchmarkGraphs`, `PgoOracle.DampingAblation`,
  `PgoInit.ChordalRotationsAgreeWithGtsam` (50 isotropic cases within 2.6e-10 rad of GTSAM's chordal).
- `test_chordal_init.cpp`, 4 tests: the determinant fix, κ/τ, exact noise-free recovery, refusal of a graph
  disconnected from the anchor.
- ALGORITHMS §7.2a.

## 2026-10-19 — D71: loop-closure degeneracy gate on the loop's own covariance (marginal σ_t / σ_r, parallax, collinearity), stereo depth noise propagated

**Context.** Arch §14.4 puts a degeneracy gate before PCM. A degenerate PnP loop carries an honest but huge
covariance, so it is consistent with almost anything and PCM cannot discriminate it. The arch criterion is
`λ_min/λ_max < τ` on the Hessian, which mixes metres and radians and depends on the frame. The design also assumed
that 3D point noise could be ignored. Design: `ROBUST_BACKEND_2.5.md` §4.

**Decision.** `cpp/slam/robust/degeneracy.{hpp,cpp}`, `AssessLoop`, a pure library with no wiring.
- **Pose refinement:** the PnP pose is refined with `cv::solvePnPRefineLM`, then Gauss–Newton weighted by per-point
  pixel covariances that carry B's stereo noise in (u, v, disparity) (Vakhitov et al., CVPR 2021).
- **Covariance:** H⁻¹ inflated by max(1, σ̂²), in the left tangent, converted to the pose-graph residual tangent by
  `OdometryEdge`'s formula.
- **Gates:**
  - worst-direction 1σ of the translation marginal [m] and the rotation marginal [rad], absolute bounds (Zhang 2016);
  - too few inliers with a stereo triangulation angle ≥ θ_min;
  - collinearity: the 3D inliers lie on a line (λ_mid/λ_max of their covariance).
- **Reported only:** the within-block ratios, pixel spread, planarity and the corridor score.
- **§49:** the arch §14.4 criterion is replaced, with a dated update recorded there; ALGORITHMS §7.8a.

**Evidence** (`tests/unit/test_degeneracy.cpp`, 8/8; ctest 308/308, 8 env-gated skips):
- the finite-difference check of the stereo back-projection Jacobian;
- the information equals `OdometryEdge`'s conversion;
- Monte Carlo mean NEES **5.95** over 500 trials (χ²₆ mean 6) at 1.5–12 m with pixel and disparity noise. At
  25–40 m it is **54.7**: the linearised covariance is about 9× overconfident at far range, which is the measured
  reason parallax is its own gate;
- the control passes. A 3D line is named collinear, far-only points are named for parallax, and the small object
  is named for conditioning;
- **Collinearity moved from pixels to 3D after measurement.** A pixel-spread gate refused a well-conditioned planar
  fan (6.4 mm, 1.35 mrad). Conditioning alone missed a 3D rail (8.3 mrad reported), because stereo noise fabricates
  the missing rotation information. The fan is now a must-not-refuse test.
- **Scene correction:** 120 points on a 40 px patch measured 2.7 cm and 0.4°, a usable loop. The scene was corrected
  to 25 points on a 20 px patch; the threshold was not loosened.

**Not done, stated.**
- **Thresholds** (σ_t 0.10 m, σ_r 0.02 rad, 10 points at ≥ 1°, spread 0.01, σ_px 0.46, σ_d = √2 σ_px) are
  [ASSUMPTION]. Calibration follows Zhang's margin-midpoint rule on real candidates with ground truth, protocol fixed
  in design §4, run in D73.
- **Not wired.** `VerifyResult` does not export inlier correspondences, and the broker's `ProxyInformation` stays,
  because replacing it changes the Phase-1 sim_collab outputs. Both need a maintainer decision before D73.
- **Honest covariance is shown at 1.5–12 m only.** On the small-object scene the residuals are about 3× the model
  (σ̂² 3.2), so the covariance is honest there only through the inflation.
- **Not timed.** It is O(N) per loop with a 6×6 solve [UNVERIFIED].

## 2026-10-19 — D72: PCM against the odometry-only chain, with a per-loop (Bonferroni) pair threshold

**Context.** Arch §14.5: pairwise cycle consistency (Mangelson et al., ICRA 2018) at χ²₆(0.95), a maximum consistent
set by greedy maximal clique, exact for ≤ 25, after the degeneracy gate (D71) and before GNC (D73).
Design: `ROBUST_BACKEND_2.5.md` §5.

**Decision.** `cpp/slam/robust/pcm.{hpp,cpp}`, `Pcm(PoseGraph)`, a pure library with no wiring.
- **Odometry legs:** prefix-composed adjoints on the odometry-only chain (`kOdometry` and `kStereoOdometry`).
  Non-chain components throw.
- **Shared odometry edges:** when both legs run along one chain, each shared edge is counted once with summed,
  signed coefficients.
- **Groups:** per pair of odometry components. One candidate is accepted as "untested". Two or more candidates with
  a maximum clique of 1 accept nothing ("no_consensus").
- **Clique:** greedy CLIQUEHEU (Pattabiraman et al.); exact Bron–Kerbosch behind a flag. Loops are ordered by
  (from, to), so the result does not depend on insertion order.
- **Threshold (§49 deviation, arch §14.5 updated):** per-loop false rejection α_loop/(n − 1) per pair, with
  α_loop = 0.001 derived from the recall requirement. The fixed χ²₆(0.95) is kept as the ablation baseline.

**Why the threshold changed.** A maximum clique keeps a true loop only if all of its pairs pass. At 5 % per pair the
fixed quantile measured recall 0.84–0.86 mean and 0.60–0.70 worst. The acceptance criterion was fixed beforehand:
recall ≥ 0.95 on every seed at ≤ 40 % outliers, because a dropped true loop is unrecoverable and outliers are GNC's
job.

**Evidence** (`tests/unit/test_pcm.cpp`, 12/12; ctest 355/355 on the merged tree):
- **Odometry legs:** Monte Carlo NEES 6.01 and 6.24.
- **Pair statistic** (mean, χ²₆: 6): 5.90, 6.15, 5.95, 5.93 and 5.75 for disjoint, nested, crossing, backward and
  shared-end legs; 6.07 and 5.96 for inter-agent pairs with loop 2 forward and reversed.
- **Cliques:** greedy = exact on 200/200 planted-clique graphs.
- **Random outliers, 10–40 %:** recall **1.000** on every seed, precision 1.000.
- **Near-miss outliers:** precision 0.75 at 0.2 m / 1°, 0.94 at 0.5 m / 5°, 1.00 at 1 m / 10°. Recall stays
  0.99–1.00, against 0.83–0.85 for the fixed quantile.
- **Blind spot:** 5 aliased, mutually consistent wrong loops beat 3 true ones. Asserted and labelled.
- **Timing:** 186 ms for 300 loops on 3000 nodes.

**Maintainer decision (2026-10-19), following the advisor's recommendation: the per-loop threshold is the default.**
- **The two failures are not symmetric.** True loops dropped by the fixed quantile cannot be recovered; near misses
  passed by the default are GNC's job. A 0.5 m / 5° near miss has rᵀΩr ≈ 176, against GNC's c̄² = 16.8.
- **Stated trade:** the bound loosens with group size (27.6 at 10 loops, 30.3 at 30, 33.1 at 100, 35.6 at 300,
  against 12.59).
- **Blind spot:** unchanged under either threshold.
- **Condition for D73:** GNC's acceptance runs on PCM's output with 0.5 m / 5° near misses and requires precision
  ≥ 0.99 on them; 0.2 m / 1° near misses are scored by ATE. If GNC fails, the first remedy is α_loop = 0.01, not the
  fixed quantile.

**Not done, stated.**
- Near misses near the loop noise pass PCM by design; D73's GNC test must include them.
- D56 lost stretches are bridged by identity-covariance edges, so PCM is blunt across them. Not measured on real
  data.
- The greedy clique can return a smaller set than exact. It didn't on the test graphs, and a shuffled-order run
  before the tie fix showed 12 against 13.
- Not wired: D73 builds the loop source → gate → PCM → GNC.

## 2026-10-20 — D73 part 1: GNC-TLS on PCM's output; every connected component anchored

**Context.** Arch §14.6: GNC-TLS (Yang et al., RA-L 2020) on the loops that survive PCM, odometry at w = 1, weight
trajectories exposed. pgo_v2 (D70) removed OpenCV's damping floor, which had been the only regulariser for a graph
split by rejections. Design: `ROBUST_BACKEND_2.5.md` §6. D73 is split into three parts (design §6); this is part 1.

**Decision.**
- **`cpp/slam/robust/gnc.{hpp,cpp}`:** whitened r² = rᵀΩr, c̄² = χ²₆(0.99) = 16.81, Prop. 4 weights, μ₀ =
  c̄²/(2r²max − c̄²), μ ×1.4.
- **Stopping:** on weights within 1e-4 of {0, 1}, a relative cost change below 1e-5, or 100 outer iterations.
- **Execution:** runs on a working copy. PCM's accepted set arrives as an edge mask, so indices stay stable. The
  input graph gets the final poses and weights once (new `PoseGraph::SetGncWeight`, loop edges only, w ∈ [0, 1]).
  Inner solves skip marginals.
- **Conventions:** checked against GTSAM 4.3.0's `GncOptimizer.h`, and they match. The oracle run is deferred to
  part 2 [UNVERIFIED].
- **`SolvePoseGraph` anchors every connected component** of the active graph (loops at w ≤ 1e-4 are inactive). The
  common single-component path is unchanged.

**Evidence** (`tests/unit/test_gnc.cpp`, 10/10; existing PGO tests 10/10 numerical and 9/9 opencv, with the GTSAM
oracle at 100/100):
- **Weights:** the Prop. 4 weight equals the dual's argmin.
- **No outliers:** energy equals plain LM.
- **Gate met:** grid and sphere at 10–40 % random outliers, precision 1.000 and recall worst 0.967. The ramp gives
  10/10 up to 90 %.
- **Near misses:** at 0.5 m / 5°, PCM → GNC accepts none (PCM removed them all), and GNC alone accepts 0/200. At
  0.2 m / 1°, 143/200 accepted, +11 % ATE.
- **Self-consistent 8-loop cluster:** 0/8 accepted.
- **Split graphs:** two agents linked by 3 random loops → PCM no_consensus → 2 components, B held at its anchor.
- **Bit-identical twice:** yes.
- **Timing:** 1.6 s at 1,000 nodes and 4.75 s at 3,000 per event.

**Not done, stated.**
- **A lone inter-agent link is unverifiable** by GNC: measured, it is kept. PCM tags it "untested", so the first
  inter-agent loop rests on the degeneracy gate alone.
- **Hard outliers:** the 10–90 % ramp uses random outliers, so no breakpoint was found. A harder protocol (aliasing
  clusters, near misses at scale) belongs to part 2's real candidates.
- **The demo-latency estimate given earlier (0.3–0.8 s at 1,000 nodes) was optimistic.** Measured: 1.6 s.
  Warm-starting the weights across events is the unimplemented lever.
- **Not wired;** parts 2 and 3 follow.

## 2026-10-20 — D73 part 2a: GNC weight reuse across loop events (fast path = fixed point + half-bound innovation test)

**Context.** The maintainer asked for weight reuse before the wiring, to cut per-event latency. Design §6 part 2a
holds the protocol, fixed before the first run.

**Decision.**
- **State.** `SolveGnc(..., GncState*)`: the caller carries per-edge decisions. A loop that was PCM-active last event
  and still is starts at its previous decision; other active loops start at 1; the mask always wins.
- **Fast path:** accepted only if all of these hold:
  - every loop held at 1 has r² ≤ c̄²;
  - every active loop held at 0 has r² > c̄²;
  - the n new loops raise the full cost by 2ΔE ≤ ½χ²₆ₙ(0.99). This takes one extra warm solve with them at 0.
- **Fallback.** Otherwise the fast-path solve is discarded and cold GNC runs from the event's input poses.
- **Path dependent** (deterministic under replay).

**Why the half bound (measured, design §6 table).**
- **The fixed point alone diverged from cold at 25 events:** a wrong loop bent the odometry.
- **Adding the innovation test at χ²₆ₙ(0.99) diverged at 45 events** and kept a near miss for good: precision
  0.986, ATE +3.6 %.
- **Half the bound (Remark 5's region)** matched cold at every one of 84 events, with ATE identical.

**Evidence** (`tests/unit/test_gnc.cpp`, GncWarm 3/3):
- **1,000 nodes:** 72 % fast path, −22 % time, and PCM's rejected-then-accepted case covered.
- **3,000 nodes:** 83 % fast path, 227 ms mean per event, 655 ms worst.
- **Determinism:** bit-identical over a fixed sequence.

**Correction, stated.** Part 1's 1.6 s / 4.75 s per event was GNC with outliers present and no PCM in front. In
the wired order PCM removes them first, and a loop event costs about 0.1–0.2 s even cold.

## 2026-10-20 — D59 (gravity priors): GRAVITY_PRIOR edges produced; Mahony is the source, σ 2.3°

Design and evidence: `docs/design/INERTIAL_2.4.md` §4.10–4.11.

- **Edge** (`slam/pose_graph/gravity_prior.hpp`): unary on a keyframe's T_W_C, r = B(g_C)ᵀ R_WCᵀ ĝ_W (2-dof, blind to
  yaw), the body-side form of GTSAM's `AttitudeFactor<Pose3>` (same cost for isotropic information). FD-tested
  Jacobian; kept in `PoseGraph::gravity_priors()`, apart from `edges()` and `Cost()`, which are 2.5's.
- **Emission:** one edge per 10 keyframes (non-overlapping windows), both inertial modes; none without an IMU.
- **Source decided by 2-dof NEES vs ground truth, never ATE** (22 EuRoC runs): the VIO-derived candidates are accurate
  (0.4–2.4°) but inherit the reference gravity's constant per-flight error, overconfident 2–70×. **Mahony** is
  independent and consistent: σ = 2.3° calibrated on Machine Hall (NEES 2.00), held-out Vicon flights NEES 3.0;
  heavy tail (p99 ≈ 18).
- `euroc_v2` and `euroc_v2_attitude` emit them; trajectories unchanged (byte-identical), `graph_version` changes.
- **Handoff to 2.5 (consumer):** (1) add the edges' cost to the PGO from `gravity_priors()` (residual and Jacobian in
  `LinearizeGravityPrior`); (2) robustify them (heavy tail); (3) estimate the agent's gravity direction as a 2-DoF
  variable initialised at `g_ref_W` (its error is shared by every edge); (4) wire field: Phase 3 (inter-agent).


## 2026-10-21 — D73 part 2b: intra-agent loop closure wired (off by default); calibrated loop noise model

**Context.** The loop backend (gate → PCM → GNC on a VO mirror) existed but was not wired. The first EuRoC run
showed the PnP loop covariance far too confident for PCM and GNC. Design: `ROBUST_BACKEND_2.5.md` §8.

**Decision.**
- **The adapter** (`runtime/agent/intra_loop_closure.hpp`), wired into `stereo_vo_runner.hpp` behind an optional
  `loop_closure:` section (`configs/slam/euroc_v2_loops.yaml`). Per keyframe:
  - ORB features plus SGBM 3D points, and the mean-bit placeholder descriptor;
  - retrieval, with the last 50 keyframes and the last 10 s excluded;
  - verification with `edge_verifier`, whose new inlier export was a maintainer-approved hunk;
  - each loop queued until its keyframes leave the BA window, then one backend event.
  The VO is never written. Outputs: `loop_trace.jsonl`, `loop_candidates.csv` (ground-truth errors, full error
  vector and covariance), and a summary block with the funnel, recall, latency and the PCM good-pair statistic.
- **Edge noise model** (`DegeneracyParams::noise_*`): var = k σ²_PnP + floor², per block.
  - Fitted jointly on MH_01 + V1_01: k_t 1.33, floor_t 0.014 m, k_r 0.178, floor_r 0.0095 rad.
  - The gate's criteria still read the PnP σ.
  - The labels' "3σ of its own covariance" prong was dropped; labels are 0.2 m / 2° against interpolated ground truth.

**Why.** On V1_01 the PnP covariance gave median NEES ≈ 1e4, about 1000× overconfident. Correlated errors don't
average down; COVINS-G reports the same. With that covariance, PCM did not improve on the gate (0.67 against 0.63).
The MH_01-only fit failed held-out V1_01, which is stated; hence the joint fit.

**Evidence** (keyframe-refined ATE, VO → loops):
- MH_01 0.0676 → 0.0207 m (−69 %)
- V1_01 0.0563 → 0.0382 m (−32 %)
- V1_02, held out: 0.0517 → 0.0254 m (−51 %)
- **V1_02 funnel precision:** 0.53 → 0.80 → 0.88 → 0.97 (GNC: 142 good, 4 bad).
- **Held-out NEES:** mean 5.24 passes; the tail is 4.4 % against ≤ 3 %, which misses. The maintainer accepted the
  model with the miss stated; GNC absorbs the tail.
- **PCM good-pair statistic:** 2.1–7.8, so no odometry scale is needed.
- **MH_01 without the model:** GNC kept 51 good of 55 (0.93); with it, 118 good of 177 (0.67), at the same ATE
  (0.0205 / 0.0207 m). On V1_01 the model raised recall (21 → 86 good) and precision (0.81 → 0.91).
- **Evidence:** `evaluation/runs/2.5/loop_closure/` (five runs, no ground truth, `analyze_loops.py`). The manifests'
  `git_dirty: false` is wrong: the binaries were built from the uncommitted tree on 4b45934.
- **Latency:** 1.3–2.4 s mean, 7.3 s worst, mostly waiting for the BA window.
- **Tests:** `Degeneracy` 9/9, including the noise-model test.

**Not done, stated.**
- **Sim calibration**, which the demo needs.
- **ORB per-octave σ.**
- **MH_01's 59 accepted loops labelled bad:** small translation error, a ~3.8° systematic rotation offset, none
  gross. Cause [UNVERIFIED].
- **The gate's σ thresholds** are unchanged (the Youden fit barely moved them).
- **Retrieval recall** 51–66 % (placeholder descriptor).

## 2026-10-21 — D73 part 2c: GRAVITY_PRIOR edges consumed by the pose graph (per-agent 2-DoF gravity, Huber)

**Context.** The D59 handoff to 2.5: (1) add the gravity priors' cost to the PGO; (2) robustify them (heavy tail);
(3) estimate each agent's gravity direction as a 2-DoF variable starting at g_ref_W; (4) wiring, which is Phase 3.
Design: `ROBUST_BACKEND_2.5.md` §8.

**Decision.**
- **`PgoProblem` / `PgoSettings::gravity`** (off by default):
  - one 2-DoF gravity variable per agent with priors, on the sphere;
  - priors cost ½ρ(s) with a Huber kernel at k² = χ²₂(0.99) and information scaled by 0.6;
  - they are not in GNC.
- **`SolvePoseGraph`** reports `gravity_W` per agent and carries the priors through the per-component split.
- **The loop mirror** syncs the VO's priors (queued, never dropped) and re-solves when they land.
- **Config:** `loop_closure.gravity_priors` (`euroc_v2_loops` false, `euroc_v2_loops_gravity` true).

**Why 0.6:** the lag-1 autocorrelation of Mahony's prior errors is 0.12–0.36 on three EuRoC flights, measured
offline from the run traces, so the effective number of independent priors is lower than the count.

**Evidence** (`tests/unit/test_gravity_pgo.cpp`, 4/4):
- the gradient matches finite differences, Huber included;
- the solve is bit-identical when off;
- a 1.5° reference error is estimated to 0.62°, and tilt RMS goes 3.78° → 2.01°;
- a 30° outlier prior moves its keyframe by 1.85°.
`PgoLevMarq` 4/4 and numerical `PgoProblem`/`PoseGraph` 10/10 pass; the oracle was not rerun (the off path is
unchanged).

**Not done, stated.**
- No EuRoC run yet, by the maintainer's priority (sim calibration and part 3 first).
- The 4-DoF-pose-graph comparison is recorded, not built [UNVERIFIED citations].

## 2026-10-21 — D65: gain-up degradation configs for the VST; the 16 shipped configs stay

The 2.2 preflight measured what D produces (`docs/design/RADIOMETRIC_2.2.md` §2). At unity gain the 8-bit output
posterises dark2 (0–15 DN, σ 0.2–0.7 DN, mostly rounding) and dark3 (0–4 DN). The shipped dark configs therefore carry no
measurable sensor noise, and a measured-noise gate correctly refuses the VST on all of them.

- **Added:** `dark2_dust0_ag8` and `dark3_dust0_ag16`: the same sensor and scene radiance, with the analog gain raised to
  8 and 16. That's what a real camera does in the dark. Their noise follows `var = a·m + b` to within 0.5%.
- **Not chosen:**
  - re-tuning the 16 shipped configs (they feed every Phase-1 baseline);
  - a >8-bit D output (the frontend takes 8-bit only; a wide change).

## 2026-10-21 — D66: D applied in SimSource, opt-in; per-camera seeds; byte-identical parallel D

- **Opt-in.** `SimSource::SetDegradation(named config, run seed, threads)` and `ariadne_sim_vo --degradation YAML`.
  Without them nothing changes, and the manifest still says `degradation_config: none`.
- **Order:** lens → D on the RGB frame → BT.601 gray, the order a real camera has.
- **Seeds:** each (agent, camera) gets `DeriveSeed(DeriveSeed(run_seed, "degradation"), "agentK/left|right")`, so left
  and right do not share noise or PRNU fields.
- **D's frame id** is the per-camera arrival index, which equals the pair index (a missing frame aborts the run as
  before).
- **Dust is refused:** it needs GT depth for both cameras (Phase 5).
- **`DegradeParallel`:** row bands over `std::thread`. It is byte-identical to `Degrade` by the determinism contract,
  and that is tested, not assumed. The degradation digest is unchanged (8535765251477677907).
- **Flat field:** each camera's K is calibrated at `SetDegradation` from 16 D frames of a flat target under its own
  sensor and seed. The frame ids are `0xC0FFEE…`, disjoint from run frames. The calibration agrees with D's analytic
  vignette × PRNU to ≤ 0.5% RMS.

## 2026-10-21 — D67: the radiometric front end: flat field + gated GAT/exact-inverse VST, before rectification

Arch §6.2 stages 1–2 (§6.4 gate). Raw 8-bit gray in and out, between the source and the rectifier: the rectifier's
interpolation would mix the noise the GAT models, and K lives in raw coordinates.

- **Config.** An optional `radiometric:` section in the perception config.
  - Absent means `mode: off`: a call-through decorator, so no pixel and no gate-compared byte changes.
  - Modes: `flat_only | auto | on`. Denoisers: `nlm | gaussian | none`. Inverses: `exact | closed_form | algebraic`.
  - Shipped ablation configs: `sim_v1_rad_{flat,auto,gauss,alg}`, `euroc_rad_{auto,on}`.
- **Order: flat field, then VST.** Correct for D, which scales the noise by K. ALGORITHMS §14.5's opposite order
  (VST, then ÷K) is the right one for real optics. **D's K-after-noise order is therefore unphysical:** recorded as an
  open question for Phase 5, not changed now, because it would move the degradation digest.
- **Noise model fitted online** (single-image, simplified Foi 2008, [PROXY]) on the first 8 pairs, per camera.
  **Deviation from arch §6.2**, which calibrates (α, σ) offline per (exposure, gain): EuRoC has no static captures, and a
  per-(t, g) table needs the §6.5 photometric state (Phase 5).
- **Gate,** decided once per run, jointly for both cameras. ON iff every camera has:
  - a valid fit;
  - σ_z ∈ [0.8, 1.25] after the GAT (the RPG-VST check);
  - SNR at the median level < **30**.

  **The threshold was corrected from the draft's 20 on 2026-10-21, before any VO measurement.** 20 was derived from
  single-channel D output. The sim path is RGB → D → gray, which scales the variance by Σw² = 0.447, so SNR is 1.5×
  higher: the same boundary is 30. Test `GrayPathScalesTheModel` pins the scaling.
- **Stage 4 (exposure/gain normalisation) is inert** and says so in the results. D54's `MatchExposure` keeps the
  statistics-driven left/right match. Stage 4 is the metadata-driven temporal normalisation to (t₀, g₀); no source
  carries (t, g) yet. No duplication, and `stereo_lk_matcher.*` is untouched.
- **Manifest:** a new field `radiometric` (`off`, or mode/decision/denoiser/inverse/flat-field summary). The fits and
  the gate's reasons go in the `radiometric:` block of results.yaml, written only when enabled.
- **Denoiser: `GaussianBlur` 5×5 by default** (0.8 ms per 1280×720 image; the demo must be near-realtime,
  maintainer 2026-10-21).
  - `cv::fastNlMeansDenoising`, arch §6.2's suggestion, is an ablation row only. It takes 8U/16U only, so the
    stabilised image is scaled into 8U, and the stage costs ≈ 360 ms per image with it.
  - BM3D is unavailable: our build has `OPENCV_ENABLE_NONFREE` off.

## 2026-10-21 — D68: the exact unbiased inverse is ours, implemented from the paper and checked against its figures

The invansc package (Mäkitalo & Foi) is under a non-commercial licence. None of its code or `.mat` tables is used or
shipped; the inverse is implemented from the thesis' equations.

- **The table.** E{f_s(z) | y, s} (eq. 4.10) by quadrature: the Poisson sum truncated at 12 SD; the Gaussian part
  integrated in u, with z = t0 + u², which removes the square-root kink. It is tabulated on a quadratic y-grid over
  [0, 200]. Below the table: the ML zero (eq. 4.14). Above it: the closed form.
- **Evidence** (`tests/numerical/test_gat.cpp`): our tables reproduce the paper's accuracy figures.

  | quantity | ours | paper |
  |---|---|---|
  | closed form, L∞ | 0.0470 | 0.0468 |
  | closed form, weighted L2² | 0.00690 | 0.0069 |
  | I0 − s², L∞ | 0.0470 | 0.0470 |
  | I0 − s², weighted L2² | 0.00295 | 0.0028 |

  The exact inverse passes an independent exact-Poisson Monte Carlo unbiasedness test that the algebraic inverse fails.
- **Correction to ALGORITHMS §14.5** (recorded there by an insert): the closed form's `D < 2√(3/8) → 0` branch comes
  from the thesis (§3.2.2), not the `.m` file, which omits it.

## 2026-10-21 — D69: 2.2 verdict: the gate works as designed; on the measured segment the VST buys nothing measurable

**What was measured.** Per the maintainer: only what cannot be done without, and the demo must be near-realtime.
`ariadne_sim_vo`, `ci_tiny_0` agent 0, 1000 frames, `sim_v2`, seed 1. Full numbers: `RADIOMETRIC_2.2.md` §6.

| row | ATE | lost | keyframes | landmarks alive | stereo valid |
|---|---|---|---|---|---|
| clean recording | 3 mm | 0 | 80 | 770 | 0.776 |
| `dark3_dust0_ag16`, stage off | 5 mm | 0 | 76 | 238 | 0.747 |
| `dark3_dust0_ag16`, flat field + VST, Gaussian (gate on) | 4 mm | 0 | **104** | **197** | **0.722** |

- **The gate decides from measured noise, as designed.**
  - Sim: ON at both gain-up configs (σ_z 1.00, SNR 7–16); OFF at dark0 (SNR) and at posterised dark2 (no valid fit).
  - EuRoC: OFF on all 11 flights (SNR 34–80).
  - The draft's threshold of 20 would have given the same decisions on these scenes. The correction to 30 (D67) was
    physics, not outcome.
- **No measurable benefit on this segment** (negative result, reported).
  - The noise costs little here: 0 lost frames either way, ATE 5 mm vs 3 mm clean.
  - The 1 mm ATE difference is one seed on an easy 50 s segment, and is **not** claimed.
  - Secondary metrics are slightly **worse** with the stage: keyframes +37%, landmarks alive −17%, stereo-valid
    fraction −2.5 points. The smoothing plausibly shortens LK tracks [UNVERIFIED: not isolated].
  - What would show a benefit, if any: a segment where `stage off` loses frames, i.e. deeper darkness or harder motion.
    Not run.
- **Cost.** The stage takes 14 ms per 1280×720 image with the Gaussian denoiser; most of it is the single-threaded
  inverse-LUT loop, which can be parallelised. The default denoiser is Gaussian; NLM (≈ 360 ms per image) is an
  ablation row only.
- **Defaults stay off.** The shipped perception configs carry no `radiometric:` section. 2.2 lands as an opt-in stage
  with measured gating. Making it default-on is not justified by this evidence.
- **Not run (defined in `configs/benchmark/radiometric_2.2.yaml`):** the rest of the sim matrix (dark2_ag8,
  flat-only, NLM, algebraic) and the EuRoC forced-on rows.

## 2026-10-21 — D80: Phase 3 opened before the Phase 2 gate (maintainer instruction); decision IDs D80–D89 claimed

- **Opened early, explicitly.** CLAUDE.md §4 opens a phase only on the previous gate. Phase 2's gate is not passed:
  2.1 and 2.2 are unmerged, 2.5's D74 certificate is open, and the phase-gate premerge has not run.
  The maintainer started Phase 3 on 2026-10-21 ("start phase 3 collaboration"), given the deadline.
- **What the early opening must not do.** Phase 3 must not change the Phase 2 code paths the gate will measure.
  The Phase-1 gate's pinned references (incl. `figures/bytes.json`) keep describing the Phase-1 exchange until a
  recorded re-pin. The new protocol lives beside it (`docs/design/COLLAB_3.2.md` §9).
- **Ownership.** This session takes the critical path in order: 3.2, then 3.4, then 3.5. 3.1 and 3.6 (∥A) and 3.3
  (∥B) are for parallel sessions; 3.3 implements the `LinkModel` seam published in COLLAB_3.2 §2.
- **IDs.** D80–D89 are Phase 3's critical path. In use elsewhere: D60s (2.1), D70–D74 (2.5).

## 2026-10-21 — D81: the 3.2 protocol design — per-class delivery, fragment ARQ, one-hop election with epochs

Design: `docs/design/COLLAB_3.2.md`; advisor-reviewed twice. Preflight: Swarm-SLAM (code-verified), Tian et al. 2018
(vertex cover), CERBERUS (completed-transmission rule), Lajoie 2026 (real link drop-out), COSMO-Bench. No DTN+C-SLAM
paper found. OpenCV has nothing native here.

- **Delivery semantics differ by class.** Phase 1's strict per-stream order would stall forever on a TTL-expired or
  cancelled message.
  - Heartbeat/ack: latest-wins.
  - Descriptors, requests, feature responses, submaps: idempotent sets, best effort, re-requestable.
  - Edges and graphs: reliable and in order.
- **Fragment-level reassembly with selective retransmit.** Whole-message delivery cannot converge under loss
  (0.7^15 ≈ 0.5%).
  - Reliable sequence numbers are assigned at first transmission, so superseding/expiry/cancel never opens a hole.
  - Unacked messages to an old broker are abandoned and counted on election change.
- **Deviation from §15.3, superseding:** a newer queued pose-graph snapshot replaces an unsent older one from the same
  agent. "Never expire" still holds in substance, because the snapshot contains the older one. Edges are never
  superseded.
- **Deviation from §15.4 (§49 note), the election rule:** the broker is the lexicographic min of (anchor_id, agent_id)
  over the agent and its symmetric one-hop neighbours.
  - This is Swarm-SLAM's exact `is_optimizer()` rule, not "lowest AgentId of the component".
  - There is no relaying, so a broker serves only agents it reaches directly.
  - Neighbour links use hysteresis (in after 2 of 3 periods, out after 4 misses).
- **Epochs and staleness:**
  - An epoch is `(counter, members_hash, broker)`, with counter = 1 + max over members' heartbeats. That makes a
    merge strictly newer than both sides.
  - An agent accepts an estimate only if it is a member, the estimate comes from the broker it elected, and the
    counter is newer.
  - The corruption check keys on the full epoch, with digests over the wire-encoded (float32) estimate.
- **Gauge:** the anchor's first pose is fixed in full. A side that lacks its anchor fixes its lowest member's
  `T_G_Wi`. Yaw-only waits behind a switch for 2.5's gravity-prior consumer.
- **The I6 byte gate is fixed now, before any data:**
  - predicted = the §16 cost model before sending, with a compression-ratio model calibrated on disjoint runs;
  - measured = first-transmission wire bytes, per class, classes 1–6, >15% fails;
  - retransmit, ack and heartbeat overhead get their own rows and their own curve.
- **Store-and-forward = hold until the destination is reachable.** Multi-hop relay is out of scope.

## 2026-10-21 — D82: 3.2 protocol, the fixes the fault campaign forced; I5 campaign 54/54 consistent

Design: `docs/design/COLLAB_3.2.md` §10. Code: `cpp/collaboration/protocol/` (`network`, `transport`, `membership`,
`collab_node`, `outcome`); tests `tests/protocol/test_{network,transport,membership,collab_v2}.cpp`.

- **Descriptors are acked and resent until their TTL.** A broker cannot re-request a descriptor it never heard of.
  An expired descriptor leaves the sent set, so the next broker is sent it again.
- **Acks are repeated 3×, and the retransmit timeout is capped at 8 s.** Measured: 1–4 unacked edge messages per
  30%-loss run before the change; none after.
- **Reliable stream incarnations** (`seq = incarnation << 40 | n`). Abandonment on a broker change would otherwise leave
  a permanent sequence hole (measured: 435 edges held in every partition run).
- **Heartbeat hysteresis is sized from the loss rate:** 0.5 s period, out after 12 missed periods. `k = 4` at 1 Hz
  flapped the broker 40 times in 290 s at 30% loss.
- **The broker re-solves on a new edge or a member-set change.**
- **I5 evidence, unit-level (synthetic 4-agent scenario, ground-truth hooks):** 6 faults × loss {0, 10, 30%} × 3 seeds
  = 54 runs, **54 consistent, 0 incomplete, 0 corrupt**. Split-brain {0,1}|{2,3} merges under a new epoch, and the
  frames match ground truth within 2 mm. An unknown-frame pose is rejected.
- **Not yet:**
  - the production broker solve (multi-agent PGO, then 2.5's robust stage);
  - runtime/sim integration with real XFeat features;
  - 3.3's link model.
  - **Snapshots carry nodes only.** They lack odometry edges and their information. The oracle solve reads node poses
    as exact, so nothing failed, but a multi-agent PGO needs odometry factors, and the I6 POSE_GRAPH sizes need the
    §15.2 payload (nodes, edges, covariances).
    - The next step changes the API first: `AddKeyframe` takes the odometry edge with its information, and the snapshot
      carries the intra-agent edges.
  - The hooks are oracles: this evidence covers the protocol, not estimation accuracy.
- **The classifiers are proven to fire.**
  - A rogue estimate (an unissued epoch, a perturbed T_G_W) that a member adopts is classified `corrupt`.
  - A run that ends while partitioned is classified `explicitly_incomplete`, and its reasons name the missing pieces.
  - Same seed ⇒ identical ledgers and outcome reasons at the node level.

## 2026-10-21 — D83: the production broker solve — one multi-agent PGO; snapshots carry odometry edges

Code: `slam/pose_graph/multi_agent.{hpp,cpp}` (slam_opencv), adapter `collaboration/broker/broker_solve.hpp`
(`NodeHooks::solve` for the runtime); tests `tests/unit/test_multi_agent.cpp`.

- **What it solves.** One joint pose graph over the members' subgraphs, with node ids `(agent << 40) | keyframe`:
  their nodes, their odometry edges with information, and the verified inter-agent edges.
- **Placement and initialisation.** Agents with no edge path to the gauge are reported unplaced. The rest start from a
  spanning-tree alignment, then the chordal initialiser runs (D70), then LM on `LevMarqBase` (D51).
- **Gauge.** The gauge agent's first node is fixed at `T_G_W_gauge · T_W_C` (§15.4). The solve is skipped when the
  joint graph has no edge: a lone broker with one keyframe has zero free variables, and that crashed LevMarq
  (a segfault, found by the protocol-level test).
- **Per-agent T_G_W.** The gauge agent reports the gauge value (G = W_anchor). Every other agent reports a frame fit
  of its W-frame poses onto their optimised G poses: the chordal rotation mean, then the mean translation residual.
  - A positions-only Kabsch fit was rejected: it is ambiguous on a straight trajectory.
  - Per-keyframe corrections are returned as `T_G_C` but are **not yet on the wire**. The estimate carries only the
    rigid T_G_W; keyframe-local consumers (Phase 6) will need the node poses.
- **Snapshots now carry the agent's odometry edges.** `EdgeStatus::ODOMETRY` was appended to the schema, the
  information is packed as the 21-float upper triangle, and `AddKeyframe` takes the odometry edge. This is the §15.2
  payload the I6 byte gate prices.
- **Measured (synthetic, 3 agents, 60 keyframes, odometry noise 1 cm / 0.4° per step):**
  - node ATE: 0.228 m dead reckoning → 0.019 m after the solve;
  - each T_G_W lies within 5 cm / 0.6° of the best rigid fit of that agent's drifted trajectory to ground truth.
- **Protocol with the production solve:** a 4-agent run with a partition {0,1}|{2,3} at 10–30 s and 10% loss ends
  `consistent`, with every T_G_W within 5 mm / 5 mrad of ground truth. The 54-run oracle campaign is still 54/54
  consistent with odometry edges in the snapshots.
- **Not yet:**
  - the robust stage (degeneracy gate → PCM → GNC-TLS, 2.5's `cpp/slam/robust/`) between the edge list and the solve;
    inter-agent edges currently enter as verified;
  - the gravity-prior yaw-only gauge.
  - Both wait on 2.5 merging.

## 2026-10-21 — D84: AWS unblocked: Bedrock works (the 2026-09-29 verification block has cleared); GPU spot quota approved

Re-tested live as `user/pixel` in eu-north-1. Superseded: the D26–D28 "Bedrock: Operation not allowed / account being
verified" open item, and STATUS's "Pending with AWS: GPU quota … Bedrock is blocked".

- **Bedrock runtime works.** `converse` on `eu.amazon.nova-micro-v1:0` answered (9 tokens). No support case is needed.
- **Orchestrator model (maintainer: "any cheap model that's good enough").** Primary: **Amazon Nova 2 Lite** via the
  EU profile `eu.amazon.nova-2-lite-v1:0`. Fallback: **DeepSeek V3.2** (`deepseek.v3.2`, in-region on-demand).
  - Quotas (cross-region, EU): Nova 2 Lite 200 RPM / 8M TPM. In-region on-demand: DeepSeek V3.2 and Qwen3-32B at
    1000 RPM / 100M TPM.
  - Nova is first-party, needs no use-case form or Marketplace subscription, and keeps data in the EU profile.
  - Tool-use smoke test: both made a correct structured `toolUse` call on a frontier-assignment prompt. They chose the
    adjacent safe frontier over a high-gain one behind a 0.95-risk collapse, with a stated reason.
  - `[UNVERIFIED]` per-token prices: pull them before Phase 7 budgets.
- **Claude on Bedrock is not used.** The Anthropic use-case form is not filed, and the 5.5-family models have 0 TPM on
  this account. Haiku 4.5 has quota (50 RPM / 5M TPM) if it is ever wanted.
- **GPU: EC2 Spot "All G and VT" in eu-north-1 approved at 4 vCPU (AWS, 2026-10-19).** One 4-vCPU GPU instance; spot,
  so reclaimable: checkpoint gsplat and keep a demo fallback. On-demand G/VT (case 179205877900026) is still pending.

## 2026-10-19 — D60: 2.1 — ORB + lost-mode re-association ships for EuRoC v2; XFeat stays a measured ablation (overrides "XFeat primary" for this phase)

> **Approved by the maintainer as scoped, 2026-10-19.** It overrides CLAUDE.md §1's "XFeat primary features" for the
> EuRoC v2 VO front-end only. "XFeat primary" stands for the learned paths. CLAUDE.md itself is not amended.

Design, every run and every bar: `docs/design/PERCEPTION_2.1.md` (§3 rev 3, §9).

**What ships.** `configs/perception/euroc_v2.yaml` becomes the EuRoC v2 perception config, paired with
`configs/slam/euroc_v2.yaml`. It is `euroc.yaml` plus a `reassociation:` section. **No file in the repo defines the
v2 pair** (D58's v2 sweep ran `perception/euroc.yaml` + `slam/euroc_v2.yaml`). So from D60 on, every EuRoC v2
sweep, the critical-path session's included, must pass `--perception-config configs/perception/euroc_v2.yaml`, or
it silently runs without re-association.
- Lost-mode re-association, the ORB-SLAM §V-C relocalisation idea restricted to the BA window. When fewer than 30
  live tracks hold a landmark, tracks holding none are described at their pixel and matched against healthy (born at
  MEASURED keyframes) untracked landmarks: mutual NN, Lowe 0.75, Hamming ≤ 50 (ORB-SLAM2 `ORBmatcher`, `TH_LOW`).
- The matches are verified by seeded USAC PnP together with the healthy correspondences.
- Accepted tracks are renamed (`KltTracker::Relabel`), so StereoVo finds the old landmark by id.
- Off when the section is absent.
  - Verified byte-identical on EuRoC v2: section absent, D58 seed 1 equals D58's own run on all 11 flights on the
    merged binary; on the D56 tree, ORB `euroc_v2` re-runs equal A1, and with the code present but the section
    absent, runs equal D56.
  - Identical **by construction, not run** for v1 (`slam/euroc.yaml`) and the sim tools that include the runner:
    every new call returns early when its config section is absent.

**Evidence (merged tree with D58's gyro; §3 rev 3: 3-seed medians on the failing four, seed 1 on the passing seven
by the maintainer's time-gated amendment):**
- Failing four, against min(D54-equivalent, D58):
  - lost frames pass on all four (medians 264 → 67, 101 → 99, 141 → 92, 171 → 122);
  - ATE passes on MH_04, V1_03 and V2_03 (V2_03 0.595 → 0.535 on every seed).
- Passing seven: six hold at seed 1. V2_02 missed at seed 1 (0.126 against 0.118); a post-hoc 3-seed check gives a
  median of 0.1169 against D58's 0.1165, with lost 16 → 8.
- 11-flight seed 1: average ATE **0.223 → 0.204 m**, lost frames 743 → 631 (MH_04's seed 1 is its bimodal bad draw:
  266 against 66/67).
- **Against the brief's literal D54 table** (seed 1, §3.4):
  - A1 beats it on ATE and lost frames on all four failing flights: MH_04 2.777/278 → 0.365/266, MH_05
    0.560/107 → 0.314/101, V1_03 0.439/146 → 0.391/89, V2_03 2.118/183 → 0.569/130.
  - It equals or beats it on six passing flights.
  - V2_02's ATE is 1.9% worse (0.1238 → 0.1262), with lost frames 17 → 8.

**Accepted with one stated miss: MH_05 ATE** (0.314 against the bar of 0.2274).
- The bar is the D54-equivalent (coasting) median. Coasting crosses MH_05's frame-1242 loss 25× worse than D58's gyro
  (0.5 m / 4.3° against 0.02 m / 0.14°), but on seeds 2 and 3 that error happens to cancel earlier drift. Seed 1
  of the same coasting scores 0.560.
- Each half of the flight scores about 0.08 m in every configuration, and D58 has 20% less RPE than coasting.
- A1 beats D58 on MH_05 on every seed (0.314 against 0.343). Analysis: PERCEPTION_2.1 §9.15. Nothing is wrong with
  D56/D58's lost handling on MH_05.

**Why not XFeat (CLAUDE.md §1 "XFeat primary features", arch §7.2).** It is built and verified, and stays
config-selectable (`features.engine: xfeat`):
- `XFeatEngine` over D50's net: one cached forward per image, mask before §7.3 grid selection, `Describe` at any
  pixel with upstream's sampler, upstream's 752 → 736 preprocessing reproduced to float precision.
- Keyframe cadence (`tracker.replenish: keyframe`, `replenish_min_tracks`) and the §7.2 image-gradient gate
  (`replenish_min_eig_quality`).

Measured, it cuts lost frames hardest but costs accuracy:
- On the D56 tree, XFeat at keyframes (A4) regressed 6 of the 7 passing flights.
- An engine × cadence 2×2 showed both contribute.
- The best fix (A4t-r: resupply below 300 tracks + ratio 0.75 on XFeat's matcher) failed rev 3 on the merged tree:
  V2_01, V2_02, V2_03 and MH_02 ATE.
- With re-association OFF, XFeat alone still regresses V2_03 and MH_02 (§9.13), so the regressor is the engine
  (or its cadence), not the association.
- It also runs at 0.53–0.88 forwards per frame on the V2 flights, near C8's frame-rate limit.

XFeat remains the planned primary for the learned paths it was chosen for. This decision is about the EuRoC v2 VO
front-end in 2.1, revisited if a later phase removes the regression.

**Not done in 2.1 here, stated plainly:**
- **XFeat timing within the §4p budget: unmet (unmeasured).** Forward passes were only timed under load (28–78 ms
  against ≤ 16 ms).
- **The PRIORITY item is half built.** Descriptor re-association with PnP verification exists. Projection-guided
  search into the window (ORB-SLAM's Track Local Map) does not.
- The XFeat input-size decision is not taken; `resize` (upstream) is the default.
- The rest of plan 2.1 is not started: WLS/§9.4/§9.6, ANNIndex, ALIKED+LightGlue.

**Correction to D54 (append-only).** D54's quoted lost frames "MH_04 266, V2_03 181, V1_03 139, MH_05 100" are
v1's, taken from the v1 rows of the same sweep. v2's are 278, 183, 146 and 107 (PERCEPTION_2.1 §2.1).

**Merge note.** `feat/phase2-foundations` (D57, D58) is merged into `feat/2.1-perception` (`dbbf8d5`).
`ALGORITHMS.md`: D58's inertial section keeps §1.3b; 2.1's re-association catalogue entry is §1.3c.

## 2026-10-19 — D61: §9.3's confidence — the WLS map wired without touching our SGBM; off by default until consumed

Design: `docs/design/PERCEPTION_2.1.md` §10.

- **What:** `stereo.confidence: none | wls` (default `none`, the placeholder). Under `wls`, `SgbmStereo` runs
  `ximgproc::createRightMatcher(sgbm)` (a new SGBM) and `createDisparityWLSFilterGeneric(true)`, with the ROI
  (min+num, 0, W − (min+num) − max(0, −min), H) and discontinuity radius ceil(0.5·blockSize) that the native SGBM
  wiring sets. `getConfidenceMap()` (CV_32F 0–255) → u8, 0 wherever `valid` is 0. The left disparity is
  byte-identical unless `wls_filtered_disparity: true`.
- **Why not `createDisparityWLSFilter(sgbm)`, the documented route:** it sets our matcher's uniqueness to 0,
  disp12MaxDiff to 1e6 and speckle window to 0 (pinned `disparity_filters.cpp`). That would silently change every
  disparity, the SGBM gate and all v1/v2 outputs.
- **Deviation from OpenCV's intent, stated:** the map is computed from OUR left disparity (uniqueness 15,
  disp12MaxDiff 1, speckle 200), which already passed SGBM's 1 px left-right check. On valid pixels WLS's 1.5 px
  check is therefore mostly redundant, and the confidence is dominated by its variance term (depth edges).
- **Found:** OpenCV 5.0.0 SGBM clamps a non-positive `disp12MaxDiff` to 1 (`stereosgbm.cpp`: `> 0 ? v : 1`), so
  "non-positive disables" is false in this build.
- **Evidence:**
  - 9 unit tests: placeholder default; byte-identity of disparity/Z/σ_Z/valid with confidence on; high on smooth
    surfaces, low at a depth step; the left-right check rejects an occlusion band once SGBM's own check is off (1e6);
    filtered disparity opt-in; 1 vs 8 threads byte-identical; config validation; §9.4's width and boundary cases.
  - EuRoC MH_01 and V2_01 with `confidence: wls`: results and trajectories byte-identical to the v2 config.
  - Cost per stereo pass at 752×480: right SGBM 23 ms + filter 14 ms, best of 5 at load 3.8 (pipeline median 40–52 ms
    under load).
- **Not measured: quality.** The AUC against EuRoC's Leica scans is deferred by the maintainer (2026-10-19) and
  recorded as unmeasured. Nothing in 2.1 consumes the map. Turning it on in a shipped config waits for a consumer
  (TSDF §20.1 or §11.2) and that measurement.


## 2026-10-20 — D62 (DRAFT, pending the maintainer): place retrieval — EigenPlaces R50-256 via cv::dnn; exact index default, cv::ANNIndex the ablation

Design: `docs/design/PERCEPTION_2.1.md` §11. Catalogue: `ALGORITHMS.md` §1.6 (as-built note).

- **What:**
  - `perception::PlaceRecognizer` runs EigenPlaces R50-256 (Berton et al., ICCV 2023, MIT, `a2969f7`). The export
    comes from `python/model_tools/eigenplaces_reference.py` (opset 17, fixed input size). It runs in `cv::dnn`
    `ENGINE_CLASSIC`, one Net per thread, and outputs a 256-D unit vector (the existing `GLOBAL_DESCRIPTOR`
    contract).
  - `perception::PlaceIndex` provides top-k cosine with an admission filter.
- **Deviation, stated (§49):** the arch says "Use `cv::ANNIndex`". The default here is **`exact`** (one gemm, a few
  µs at thousands of keyframes, deterministic). `annoy` (`cv::ANNIndex`) is kept behind the same interface as the
  persistence/scale ablation. Reason: three findings from the pinned source (ids restart at 0 per `addItems`, no add
  after build, a per-thread seed over `hardware_concurrency()`).
- **Evidence:**
  - SMOKE §4r: `cv::dnn` matches upstream at cosine 1.0000000, both engines, both sizes.
  - Unit tests: `test_place` (exact, filter, annoy agreement 199/200, repeat 200/200, rebuild after add,
    preprocessing ≤ 1e-5, end-to-end cosine ≥ 0.9999) and `test_eigenplaces_reference`.
  - EuRoC, 11 flights, 297 queries: pooled R@1 EP-376 0.626, EP-752 0.677, mean-bit 0.444, random 0.167. Annoy
    top-1 equals exact on 297/297.
- **Acceptance (§11.5, fixed before the run):** bar 1 PASS. Bar 2 MISSED by 15/297 = 0.0505 against 0.05: half a
  query, consistent direction, paired McNemar p = 0.028 (optimistic, the queries are correlated).
- **Size (§11.7): DECIDED by the maintainer 2026-10-20, (a) 376×240 is the default.** The bar-2 miss is accepted
  and stated. 752×480 stays a selectable export. A size in between was not measured.
- **Limits:**
  - EuRoC-scoped: the rig is 1280×720 RGB, and sim needs its own export and an RGB path.
  - No consumer in 2.1 (2.5 owns loop closure).
  - CLAHE not run.
  - Annoy determinism across core counts is unverified.

## 2026-10-21 — D63: the hard-pair path — ALIKED-n16 + LightGlue from our own upstream export, through cv::ALIKED / cv::LightGlueMatcher

**Approved as scoped by the maintainer, 2026-10-21.**

Design: `docs/design/PERCEPTION_2.1.md` §12. Smoke: `SMOKE_TEST_RESULTS.md` §4s / §4s-2. Catalogue: `ALGORITHMS.md`
(the correction note under the OpenCV-5 inventory).

- **What:** `perception::HardPairMatcher`, with ALIKED-n16 at K = 1024, native 752×480, and LightGlue running all
  9 layers with `filter_threshold` 0.1. Upstream keypoint normalisation goes through `setPairInfo(..., Size(),
  Size())`; one instance per thread. The models are exported by `python/model_tools/aliked_lightglue_reference.py`
  from `cvg/LightGlue@eb42fee` with the weights `Shiaoming/ALIKED@683d7c6` `aliked-n16.pth` and LightGlue
  `v0.1_arxiv` `aliked_lightglue.pth`, configured as glue-factory's official setting.
- **Why our own export** (the maintainer's call, 2026-10-20): opencv_extra's files are an unlicensed personal
  export, of ALIKED-**n16rot** (LightGlue's ALIKED weights were trained on n16), at a fixed 640×640 square.
- **Deviations from OpenCV's documented use, stated:**
  - pre-normalised keypoints, not OpenCV's per-axis rule;
  - keypoints emitted in cv::ALIKED's own convention, so its (n+1)/2·W mapping returns upstream's (W−1) pixels;
  - export-only rewrites for cv::dnn ENGINE_NEW (DCN as GridSample + MatMul, gathers, explicit reshapes, a Floor
    before a Cast), each checked against unmodified upstream.
- **Evidence:**
  - **Exact against upstream PyTorch through OpenCV's classes:** keypoints 100% within 9.3e-5 px, descriptors
    within 4e-6, identical LightGlue match sets.
  - **Tests:** 2 reference tests, 3 wrapper tests (normalisation, size and parameter checks, upstream match set
    reproduced), 2 geometry convention tests (a synthetic pose recovered within 0.0003°).
  - **EuRoC loop-candidate verification**, 4815 EP-376 top-5 pairs on 11 flights, with the 1.6 verifier's own
    RANSAC essential matrix, against the ORB placeholder:

    | | ORB placeholder (A) | ALIKED + LightGlue (C) |
    |---|---|---|
    | pose AUC@10° | 0.284 | 0.607 (+0.322) |
    | flights where C ≥ A | | 11/11 |
    | AP_geom, fixed denominator (all 676 positives) | 0.303 | 0.807 |
    | AP_geom, per-row denominator (as first scored) | 0.578 | 0.891 |

    LightGlue over mutual NN: AUC@10° +0.042. Both §12.5 bars **PASS**, bar 1 under both AP denominators.
    The per-row one was the scorer as first written; the fixed one is like-for-like (§12.6).
  - **Reported:** OpenCV's per-axis normalisation is a wash, so upstream stays the default. K = 2048 brings no pose
    gain on all positives at ≈ 2.8× the cost, so K = 1024 stays. E's subset AP is biased against C by
    construction and does not decide K.
- **Stated limits:**
  - Zero-false-accept recall as fixed is ≈ 0 for every row: tiny-baseline revisits with correct rotation fail the
    translation term. A rotation-only diagnostic gives C 0.381.
  - **4 confident wrong verifications** (210–264 inliers, rotation error 37–59°) in the Vicon rooms. An
    essential-matrix inlier count alone is not a loop-closure gate; the downstream PnP and degeneracy gates (1.6,
    D71) must stay.
  - The inlier threshold τ does not transfer stably from MH to V.
  - The exports are 752×480 only; sim at 1280×720 needs its own export plus an RGB path.
  - No runtime consumer in 2.1 (2.5 and 3.x call it).
  - Cost ≈ 1.2 s per warm pair and ≈ 3 s per cold pair under load (≈ 0.6 / 1.7 s quiet).
  - The location of cv::dnn's rounding behind the Floor fix is `[UNVERIFIED]`.
- **Ablation baselines kept:** ORB + ratio (1.6 verifier), ALIKED + mutual NN, OpenCV per-axis normalisation,
  K = 2048.
- **§49 ritual for this adoption** (locked by CLAUDE.md §1, adopted here):

  | step | status |
  |---|---|
  | catalogue | the ALGORITHMS.md correction and as-built note |
  | arch decision | D63, plus the §7.4 blockquote |
  | ablation matrix | PERCEPTION_2.1 §12.4 rows A–E |
  | benchmark config | **deferred to 2.5**, the first runtime consumer: no `configs/` entry and no `HardPairParams` in `perception_config` yet. The offline tool takes its paths on the command line. |
  | manifest schema | **deferred to 2.5:** §28.2 has no hard-pair field yet. `ariadne_loop_verify`'s manifest records `hard_pair: [<HardPairMatcher::Name()>…]`. |
- **The rotation-only zero-false-accept numbers are a diagnostic chosen after seeing the data, not a bar.**

## 2026-10-22 — 2.1 close-out notes (bookkeeping; no new decision)

- **D62 stays DRAFT.** The maintainer decided its size question (376×240, 2026-10-20), but its deviation from the
  arch's "Use `cv::ANNIndex`" (exact index as the default) has not been explicitly approved. 2.5, the first
  consumer, should take that call.
- **Projection-guided re-association** (the PRIORITY item's second half, ORB-SLAM's track-local-map search into the
  window) is a **stated gap, deferred.** Trigger: revisit if the combined EuRoC run shows lost stretches still
  dominating V1_03/V2_02. Descriptor re-association with PnP verification (D60) is what shipped.
- **Reference exports:** the regeneration commands for `.cache/models/{eigenplaces_ref,aliked_lg_ref}` are in
  PERCEPTION_2.1 §13.

## 2026-10-21 — D85: the combined EuRoC run (2.4 full VIO + 2.1 re-association), `main` 62441c4: neutral overall; V1_03 still out

Run: `build/d85_tools/combined_euroc.sh`, with `euroc_v2` perception (ORB + lost-mode re-association, D60) and `euroc_v2` slam
(full VIO, D59). 11 flights run sequentially, all `RUN_OK`. Metric: keyframe-refined ATE (D57 protocol), the same
metric as `INERTIAL_2.4.md` §4.9.

| flight | MH01 | MH02 | MH03 | MH04 | MH05 | V101 | V102 | V103 | V201 | V202 | V203 | avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.4 only (D59 part 3) | 0.068 | 0.039 | 0.099 | 0.222 | 0.128 | 0.056 | 0.052 | 0.357 | 0.039 | 0.101 | 0.168 | 0.121 |
| combined | 0.068 | 0.044 | 0.099 | 0.223 | 0.140 | 0.054 | 0.047 | 0.329 | 0.039 | 0.110 | 0.169 | **0.120** |
| lost frames / re-associations accepted | 0/2 | 13/11 | 0/0 | 262/32 | 96/15 | 1/8 | 3/34 | 92/125 | 16/4 | 7/28 | 118/112 | 608 total |

- **Effect: neutral.** The average drops 1 mm. Per flight it is mixed: V1_03 −28 mm, V1_02 −5 mm; MH_05 +12 mm,
  V2_02 +9 mm, MH_02 +5 mm. It is still **9/11 in the stereo-inertial class**, with the same two flights out:
  V1_03 (0.329 vs 0.120) and V2_02 (0.110 vs 0.100). The ±10 mm moves are single runs; V1_03 has flipped on small
  changes before (D58).
- **The Phase 2 EuRoC gate condition stands at 9/11, average 0.120 ≤ 0.138.**
- **A1's trigger for projection-guided re-association, evaluated:**
  - V1_03 still has 92 lost frames and is out of class, so the trigger fires for V1_03.
  - V2_02 has 7 lost frames: its miss is drift, not lost stretches, so the trigger does not fire there.
  - The 2.4-only per-flight lost counts were not recorded in §4.9, so "still dominate" is judged on the combined run
    alone.
  - **Maintainer's call** whether V1_03 alone justifies building it (register: 2.1 row).

## 2026-10-21 — D62 approved: the exact index is PlaceIndex's default; cv::ANNIndex stays as the scale ablation

**Approved as scoped by the maintainer, 2026-10-21.** This supersedes the "D62 stays DRAFT" line in the 2.1 close-out
notes above. The D62 entry itself is unchanged.

- **What is approved:** `PlaceIndex` defaults to `exact` (one gemm, deterministic, a few µs at thousands of
  keyframes). `annoy` (`cv::ANNIndex`) stays behind the same interface as the persistence/scale ablation. The arch
  line "Use `cv::ANNIndex`" now reads as the ablation, not the default (§6 blockquote).
- **Why still exact:** the three pinned-source findings in D62 (ids restart at 0 per `addItems`, no add after build,
  a per-thread seed over `hardware_concurrency()`), and top-1 agreement 297/297 on EuRoC, so exact loses nothing at
  this scale.
- **§49 ritual:**

  | step | status |
  |---|---|
  | catalogue | `ALGORITHMS.md` §1.6 as-built note (D62) |
  | arch decision | D62, plus the §6 and §31.4 blockquotes |
  | ablation matrix | arch §31.4 as-built note: index `exact` (default) vs `annoy`/trees (scale); PERCEPTION_2.1 §11.5 |
  | benchmark config | `ariadne_place_eval` + `place_recall_eval.py` (PERCEPTION_2.1 §11.5). No `configs/` entry until the first runtime consumer (2.5 / 3.1) |
  | manifest schema | `ariadne_place_eval`'s manifest records `place_index: [exact, annoy/trees<N>/seed1]`. §28.2's `place_recognition` field exists; a `place_index` field is **deferred to the first runtime consumer**, like D63's hard-pair field |
- **Open:** Annoy determinism across core counts is still unverified. It matters only if the ablation is promoted.

## 2026-10-22 — D90: sim-rig exports — EigenPlaces and ALIKED-n16 + LightGlue at 1280×720 RGB, with an RGB path

**Settled 2026-10-22** by the sim retrieval evaluation (`docs/design/RETRIEVAL_SIM_D90.md`, rule fixed before the
run): EigenPlaces' sim default is **640×360**. Everything else follows D62/D63 as approved. The maintainer's
approval of the size was replaced by that pre-registered rule, as the Phase 4 brief asked; it is re-openable.

- **What:**
  - Sim-rig exports from the same pinned upstreams and weights as D62/D63, made by `models.sh eigenplaces-sim` /
    `aliked-lg-sim` and checked against sha256s pinned in the script:
    - EigenPlaces R50-256 at 1280×720 and 640×360;
    - ALIKED-n16 K = 1024 at 1280×720;
    - LightGlue (byte-identical to D63's).
  - The reference frames are recorded `left_rgb` from the phase-1 bag (`ariadne_frame_dump`).
  - The wrappers gain an RGB path whose channel order is in the name: `PlaceRecognizer::DescribeRgb`,
    `HardPairMatcher::ExtractRgb`. Both take RGB order, which is what the sim gives (`StereoFrame::left_rgb`).
    `ExtractRgb` converts RGB → BGR because `cv::ALIKED` swaps back.
- **Evidence:** SMOKE §4t.
  - Exact against upstream at the new size, which re-runs D63's 752×480-only validation of the export rewrites.
  - The regeneration is bit-exact.
  - A channel-swap guard shows an R/B swap changes the result instead of passing silently.
- **Default (settled):** EigenPlaces' sim default is **640×360**, the analogue of D62's half-size 376×240. It costs
  a quarter of 1280×720 (L5: 312 vs 1280 ms p50; here 331 vs 1297 ms on the dev host). ALIKED runs at the native
  1280×720: hard pairs only, so cost matters less than localisation.
- **Measured (clean ci_tiny recording, RETRIEVAL_SIM_D90 §5):** 61 revisit queries after motion thinning, all of them
  agent 0's (33 intra-agent, 28 against agent 1's database). Pooled R@1 is **0.705 at both sizes** (McNemar p = 1.0),
  R@5 0.885 vs 0.869. The rule kept 640×360 unless 1280×720 won with p < 0.05; it did not. With 61 queries from
  one small scene, this is "no evidence that full resolution helps", not "they are equal".
- **Limits:**
  - Four frames from one scene, with weak colour.
  - The size question was settled on one clean scene only; the dark-scene row (§31.5) is carried to 5.1.
  - There is no runtime consumer yet (3.1/3.5).
- **§49:**

  | step | status |
  |---|---|
  | catalogue | unchanged: same models as D62/D63 (`ALGORITHMS.md` §1.6 notes) |
  | arch decision | D90, plus the §7.4 blockquote |
  | ablation matrix | arch §31.4 sizes (EigenPlaces 640×360 / 1280×720); ALIKED K as D63 |
  | benchmark config | `models.sh eigenplaces-sim` / `aliked-lg-sim`; no `configs/` entry until a runtime consumer |
  | manifest schema | each export's `manifest.yaml` (`preprocessing_rgb`, `rgb_images`); `Name()` carries the size |

## 2026-10-22 — D91: the DNN execution contract object — `dnn::PerThread<T>` in every cv::dnn wrapper, and the callers own the parallelism

- **What:**
  - `perception/dnn/per_thread.hpp`: `PerThread<T>::Local()` returns the calling thread's instance, built by a
    factory on that thread's first call.
  - XFeatNet (graphs per size), PlaceRecognizer (Net) and HardPairMatcher (the ALIKED + LightGlue pair) hold their
    networks in it. One wrapper object can be shared by a worker pool, as §7.4 requires: "the feature worker pool
    owns N nets loaded from the same file".
  - `ModelRecord {model, file, engine}` (`Record()` / `Records()`) is what the manifest records about which engine
    ran.
- **Second rule, found by the test:** per-thread instances are not enough for byte identity.
  - Every Net shares OpenCV's single process-wide `parallel_for_` pool. Two concurrent ENGINE_NEW ALIKED forwards at
    the default pool (8 threads) **intermittently** differed: 462/1024 keypoints by up to 6.1e-5 px, descriptors by
    up to 1.3e-6, LightGlue scores by up to 4.4e-5, match indices identical.
  - Sequential runs on another thread: identical.
  - At `cv::setNumThreads(1)`: identical in every repeat (5 × 4 tests).
  - So when a wrapper serves several threads, **the parallelism is the callers'**: the runtime sets
    `cv::setNumThreads(dnn::kPoolOpenCvThreads = 1)` once and records it. It is a process-wide setting, so it is the
    runtime's call and the library does not set it. A single caller may keep the pool.
- **Evidence:** `test_dnn_contract`. Two threads are released together on one shared wrapper, and both match a
  single-thread reference byte for byte, for XFeat (1.1 MB of output), PlaceRecognizer and HardPairMatcher.
  `PerThread` has its own unit test. The 2.1 wrapper and reference suites still pass (31/31).
- **Not done:** no runtime pool uses it yet; Phase 7 multithreading and 7.3 COOL are the consumers. The pool rule's
  cost is unmeasured: N single-threaded nets against one N-threaded net is an L5/7.3 question.
- **§49:** the arch decision is the §7.4 blockquote. A manifest field (the pool size, `ModelRecord`s) is **deferred to
  the first runtime consumer**, like D63's.

## 2026-10-22 — D93: global-descriptor compression (3.6) — int8 with a per-vector scale is free on EuRoC; PCA is not worth its shared basis

Design and protocol fixed before the run: `docs/design/RETRIEVAL_3.1_3.6.md` §2. Result: §4.

- **What:** the §17.2 rows (fp32, fp16, int8, PCA→128 fp16, PCA→256 int8, plus PCA→64 fp16 and PCA→128 int8) are
  scored offline on D62's EuRoC dumps by `python/analysis/retrieval_rows_eval.py compression`. The protocol is
  §11.5's; PCA is fitted on the other environment.
- **Result:**
  - fp32 reproduces D62 exactly.
  - fp16 and int8 (258 B) give identical hits on all 297 queries, at both sizes.
  - PCA truncation is admissible at 376×240 but not at 752×480. PCA→256's gain is mean-centring.
  - All differences are n.s. (McNemar p ≥ 0.07).
- **Recommendation for the wire (3.2's call, main session):** int8 with a per-vector fp16 scale, 258 B per keyframe
  against 1024 B. `wire.fbs` `GlobalDescriptor.descriptor` is `[float]`, so this needs a schema change; 3.6 changed
  no comms file. The bytes here are payload `[PROXY]`.
- **Limits:**
  - EuRoC only, with 297 correlated queries.
  - Sim retrieval rows come with 3.1's sim evaluation.
  - **Stated gap:** §31.9's XFeat local-descriptor fp16/int8 row is not run. D60 ships ORB (32 B binary) on the wire,
    so it lands with the first XFeat-on-the-wire configuration (register).
- **§49:**

  | step | status |
  |---|---|
  | catalogue | no new algorithm |
  | arch decision | D93, plus the §17.2 blockquote |
  | ablation matrix | §31.9 global rows: done (§4); XFeat row: stated gap |
  | benchmark config | `retrieval_rows_eval.py compression` with its fixed defaults (bar 0.01) |
  | manifest | the run directory's `manifest.yaml` (input sha256, image, command) |

## 2026-10-22 — D92: retrieval-path CLAHE (3.1, arch §6.3) — built behind `enhance`, default off; no effect on EuRoC; the deciding row is sim

Design fixed before the run: `docs/design/RETRIEVAL_3.1_3.6.md` §3. Result: §5.

- **What:** `PlaceRecognizerParams.enhance = none | clahe` (clip 2.0, 8×8 tiles; gray directly, RGB on Lab L). It
  runs inside the recogniser on a copy, so the geometry path's pixels are untouched (unit test). `Name()` and
  `Record()` carry `+clahe`. `ariadne_place_eval --retrieval-3.1` runs the ablation, with the seeded operator D
  applied for darkened conditions.
- **Result (EuRoC, 297 queries):**
  - CLAHE's ΔR@1 is +0.013, −0.010 and +0.013 on clean / `dark1_dust0` / `dark2_dust0_ag8`. All n.s. (p ≥ 0.45).
  - The clean dump equals D62's byte for byte.
  - The darkened proxy is *easier* than clean (0.667 vs 0.626, p = 0.023), cause `[UNVERIFIED]`. So it is not a test
    of CLAHE in the dark.
- **Decision:** the default stays `none`. CLAHE is a selectable retrieval-path ablation, never applied to the geometry
  path.
- **Remaining part of 3.1:** the §31.5 sim row (dark + co-moving light + retrieval-path enhancement), which needs a sim
  retrieval evaluation with revisit ground truth. That evaluation also settles D90's 640×360 vs 1280×720 question.
- **§49:**

  | step | status |
  |---|---|
  | catalogue | CLAHE is OpenCV-native (`cv::createCLAHE`); no new learned model |
  | arch decision | D92, plus the §6.3 blockquote |
  | ablation matrix | §31.5 retrieval-path row: EuRoC proxy done (§5), sim row open |
  | benchmark config | `ariadne_place_eval --retrieval-3.1 [--degradation]` + `retrieval_rows_eval.py methods` |
  | manifest | `place_recognition` names carry `+clahe`; the tool's manifest records the degradation name, seed and digest |

## 2026-10-22 — D74: the global-optimality certificate, behind a flag (chordal surrogate, grounded Cholesky, relative η)

**Context.** Arch §14.8 asks for an SE-Sync-style certificate as a verification artifact. Design:
`ROBUST_BACKEND_2.5.md` §7. Code: `cpp/slam/robust/certificate.{hpp,cpp}`.

**Decision.**
- **What is certified:** SE-Sync's isotropic chordal cost of the active edge set (κ/τ from the information diagonal
  blocks, D70), weighted by GNC weight. It is not our anisotropic Log cost. Reports say "certified for the isotropic
  chordal surrogate", and the certificate is one-directional: `not certified` means unknown, not wrong.
- **The point:** LM on the chordal residuals (OpenCV `LevMarqBase`, anchor fixed), then an exact translation solve. The
  distance moved and the remaining Riemannian gradient are reported.
- **The matrix:**
  - the translation-explicit M, assembled from the residuals rather than transcribed;
  - Λ_i = sym((M Xᵀ)_{R_i} R_i);
  - S = M − Λ̃, SE-Sync's sign. Arch §14.8's Λ − Q̃ is the other sign; the noise-free test fixes it.
- **§49 substitution of the test.** Arch §14.8 says shifted Lanczos. Instead:
  - Cholesky of S with the **anchor's translation grounded** and **η added to the rotation diagonal only**;
  - by Schur, this is exactly SE-Sync's reduced test Q̃ − Λ + ηI ≻ 0;
  - Lanczos (full reorthogonalisation) runs only on failure, to report λ_min;
  - a certified point carries the bound f − f* ≤ |gap| + 3nη.
- **η is relative:** η = 1e-10·‖M‖∞. Also reported: the absolute-η verdict (SE-Sync's 1e-3 [UNVERIFIED default]) and
  the smallest passing decade of ‖M‖.
- **The flag:** `loop_closure.certificate` (optional key, false when absent; no shipped config sets it).
  - When on, it runs after each solved loop event on the mirror, read-only.
  - The trace gains a `certificate` object only when it ran.

**Why grounded, not a full shift.** Shifting the translation blocks too is a weaker test, since (A + ηI)⁻¹ ⪯ A⁺. On
a non-certified planted-outlier graph a full shift would pass at 0.61× the reduced η, so it over-certifies.

**Evidence** (`tests/unit/test_certificate.cpp`, 8/8 plus the env-gated SE-Sync test):
- the objective equals tr(XMXᵀ), and the gradient matches finite differences;
- noise-free 5 nodes: explicit null space 4 (3 rotation + 1 translation shift), reduced null space 3, the other sign
  indefinite, certified;
- grounded test = reduced test at ±5 % of −λ_min, and the Ritz value matches the dense reduced λ_min;
- a stationary non-global point (a wound ring, max|grad| 1.6e-12) and a perturbed point are not certified;
- low noise: 10/10 certified, worst bound 4e-4 of f;
- the loop backend is bit-identical with the flag on, and 12/12 events are certified.
- **SE-Sync, run so far** (chordal init → refine):

  | graph | f | Table 1 | bound |
  |---|---|---|---|
  | parking-garage | 1.26252 | 1.263 | f − f* ≤ 0.024 % |
  | tinyGrid3D | — | — | certified |
  | smallGrid3D | — | — | certified |

  Sphere, torus, grid3D, cubicle and rim are pending (the user is asked first: heavy).

**Planted outliers, reported, not tuned** (PHASE2_PLAN §3):
- 3 uniform SE(3) loops kept in a 60-node cost: **6/10 certified**, from each of the truth, dead-reckoning and chordal
  starts. All starts reached the same stationary point, so the "strictly worse point must fail" assertion had
  nothing to bite on here; the wound ring covers that.
- After GNC: 10/10 outliers rejected and 10/10 certified.
- The phased-plan gate's wording "not certified on planted-outlier graphs" is therefore **not met as worded**. A
  certified outlier-corrupted optimum is correct behaviour: the certificate proves optimality of the given cost, not
  correctness.

## 2026-10-22 — D75: the outlier ramp uses perceptual-aliasing groups; breakpoint reported

**Context.** D73's ramp used random SE(3) outliers and found no breakpoint up to 90 %. The exit gate needs one.

**Decision.** A reusable generator, `cpp/slam/robust/aliasing_generator.hpp` (header-only, test support), plants
groups of false loops Z_k = T_{a_k}⁻¹ Δ T_{b_k} on two runs of consecutive keyframes. These are Sünderhauf &
Protzel's grouped outliers: mutually consistent through the true odometry.
- Its own test asserts that PCM accepts a planted group whole; otherwise it would have rebuilt random outliers.
- The protocol (in the test header, fixed before the run): 60 loops per graph, policies `size10` and `count2`, arms
  GNC-only and PCM → GNC, 10 seeds per rate, success = precision ≥ 0.99 and recall ≥ 0.95, breakpoint = the lowest rate
  with success < 0.5.
- 3.5 reuses the generator with JointId chains, `cross_chain_only` and `kInterAgentLoop`.

**Result** (`evaluation/runs/2.5/outlier_ramp/`):
- PCM → GNC breaks at **90 %** (`size10`) and **70 %** (`count2`). That is exactly where the largest false group
  outnumbers the true loops, PCM's maximum-clique limit.
- GNC alone: no breakpoint (worst 7/10 and 5/10, both at 70 %), at ~10× the runtime.
- The shipped order stays PCM → GNC (D72); this is its stated limit.

## 2026-10-22 — D76: an entry point for the robust stage on a joint multi-agent graph; shared gravity option

**Context.** COLLAB_3.2 §5 says the broker's solve uses 2.5's robust pipeline (main's row). §11 puts the gate at the
verifying agent, so the broker sees pre-gated edges.

**Decision.**
- **`RobustJointSolve`** (`cpp/slam/robust/joint_robust.hpp`): PCM → chordal re-init over the anchor's PCM-accepted
  component → GNC-TLS. Nodes are JointId(agent, kf).
  - The re-init removes the hazard that `SolveMultiAgent` places agents along unfiltered inter-agent edges.
  - `GatedInterAgentEdge` gates on the broker when it holds the geometry.
  - Preconditions are in the header: noise-model information; GncState only for an append-only joint graph; GNC
    decides every loop; PCM is per agent pair.
- **`GravityOptions::shared`:** one gravity variable for all agents (key `kSharedGravityAgent`). The per-agent default
  decouples the agents' tilt in a joint graph.
  - A fixed 6-DoF anchor plus a free shared gravity *is* the yaw-only gauge, up to a global rotation; do not also fix
    gravity.

**Evidence** (`tests/unit/test_joint_robust.cpp`, 2/2):
- **Aliasing rejected:** agent 1 placed along a false edge of a 6-loop inter-agent aliasing group; on 5 seeds, 10/10
  true inter-agent loops kept, 0 false. The ATE equals the true-loops-only solve's.
  - Re-init changed nothing in these seeds: PCM had already removed the false edges, and LM recovered from the bad
    placement.
- **Tilt bias:** a 2° tilt bias on all inter-agent edges.
  - Per-agent gravity leaves agent 1 at 2.59°/2.77° tilt.
  - Shared gravity: 2.34° (−10 %) with 0.01 rad edges, 0.90° (−68 %) with 0.05 rad edges. The correction follows the
    priors' confidence against the edges'.

**Not done, stated.** `AgentSubgraph` carries no gravity priors, and `SolveMultiAgent` does not call the entry point.
Both are main's (3.2 / 3.5).


## 2026-10-23 — D110: 4.1 volumetric stack — from-scratch voxel-hash TSDF at 5 cm (HashTSDF the oracle), incremental FH ESDF, 3D frontiers, coarse RLE layer

- **What** (`docs/design/VOLUMETRIC_4.1.md`, `cpp/planning/volumetric/{tsdf_map,esdf,frontiers3d}`):
  - 8³-voxel blocks in a hash; along-ray SDF, clamped to τ = 3 voxels; per-keyframe averaging, then one fusion per
    voxel, with weights `(σ_ref/σ_Z)²`; a silhouette guard.
  - Tri-state: unknown = weight 0; occupied = SDF < half a voxel.
  - ESDF: distance to the nearest point of a blocked cube (the GT's and D45's convention), by three exact windowed
    min-convolution passes, capped at ρ_responder + 2 voxels. Two fields: egress (unknown blocks) and agent.
    Recomputed per dirty box (± reach), never globally.
  - **Correction (2026-10-23):** as first committed (`f3eb6bb`), clearance was `edt·v − v/2` on an FH
    (centre-to-centre) transform, mislabelled as the GT convention. It is optimistic on diagonals, gave 4 false
    egress claims at GT clearance 0.395 m, and the brute-force test had copied it. The map metrics are unaffected
    (they read the TSDF slab). The 5 cm decision was re-run with worldgen's `_edt_to` and survives. The preflight
    was incomplete: `docs/design/PHASE4_PREFLIGHT.md`.
  - Frontiers in the motion band, 26-connected, with centroid, normal and a 64-ray visible-unknown fan; updated
    incrementally (FUEL's rule).
  - Coarse layer: 0.5 m cells, RLE codec.
- **Why from scratch:** SMOKE §4u. `cv::Volume` has no voxel or weight read and no per-pixel weight. It stays the
  oracle (a test).
- **Supersedes the arch's 8–10 cm voxel:** measured on both scenes' GT, a one-cell wall bleed closes agent 1's
  entrance at ρ_responder for 8 and 10 cm, not for 5 cm.
- **Evidence:** 8 tests (§7). On the evidence spool vs 1.8: coverage 0.942 / 0.933, free precision 0.979 / 0.961,
  occupied precision 0.865 / 0.707; about 105 ms per keyframe on the dev host. Corrections made after measuring,
  each reported: unknown = weight 0 (the first run used `w_min = 0.5`, coverage 0.861); the oracle tolerance is two
  voxels (the oracle's own p95 error is 1.35 voxels).
- **Not here:** the live keyframe path (L1's `frontier_mission.hpp` seam, once committed) and the wire field for the
  coarse summary (3.2's).
- **§49:**

  | step | status |
  |---|---|
  | catalogue | Voxblox (block hash, weighting, projective error), nvblox, FIESTA, FH 2012, FUEL: `VOLUMETRIC_4.1.md` §1, `PHASE4_PREFLIGHT.md` §2; **`ALGORITHMS.md` entry owed** |
  | arch decision | D110, plus the §20.1 blockquote |
  | ablation matrix | 1.8 OctoMap placeholder (D45) vs D110, same spool and scorer |
  | benchmark config | `configs/mission/mission_v1.yaml` (stride, gates, band); D110 defaults in `TsdfParams` |
  | manifest schema | `ariadne_volumetric_eval` manifest (`layer: phase4_volumetric`), spool sha256 |


## 2026-10-23 — D111: 4.2 traversability at ρ_agent / ρ_responder — egress = observed column ∧ 3D occupied capsule; structural risk [PROXY] built, uncalibrated

**Partial: traversability is done in replay; the risk proxy is not calibrated.**

- **What** (`docs/design/TRAVERSABILITY_4.2.md`, `cpp/planning/volumetric/traversability.{hpp,cpp}`):
  - **Egress at ρ_responder** (the product, invariant 4) is the intersection of two tests:
    - D45's observed-column rule on 4.1's TSDF slab, with 1.8's `EdgeClearance` ≥ ρ;
    - a standing-body capsule (axis 0.45–1.40 m) on the occupied-only 3D field.
  - **Agent at ρ_agent:** a capsule over the flying rig's 0.8–1.2 m altitude band on the agent field.
  - **Risk `[PROXY]`:** saturating counts of down-facing undersides and of mass over open columns, in a 1 m disc,
    in the 1.8–2.4 m band.
  - Outputs on the briefing grid: traversability, per-column clearance, risk.
- **Supersedes arch §22.1's 26-connected 3D lattice:** 2.5D, because a responder walks on a flat floor. The 3D test
  survives as the capsule.
- **Why not strict 3D:** the rig cannot observe a standing body's volume. On the evidence spool the strict rule has
  recall 0.354 and closes the target. Reported, not the product.
- **Evidence:** 5 tests, with independent brute force. On the evidence spool: precision 1.000, recall 0.900,
  starts → target both, against 1.8's 1.000 / 0.898 / both (`evaluation/runs/traversability_4.2/`).
- **Changed after measuring:**
  - the strict → hybrid rule (advisor review);
  - the clearance convention (D110's correction);
  - the support test (before any calibration data existed).
- **Open:**
  - risk calibration against GT zones (worldgen hazards + renders; GPU spend to be asked);
  - Voxblox's half-voxel inflation (P2), decided separately.
- **§49:**

  | step | status |
  |---|---|
  | catalogue | STEP (risk map as cost), GBPlanner2 (ground beneath edges): `TRAVERSABILITY_4.2.md` §1; **`ALGORITHMS.md` entry owed** |
  | arch decision | D111, plus the §20.5 / §22.1 blockquotes |
  | ablation matrix | strict 3D vs hybrid vs 1.8, same spool and scorer |
  | benchmark config | `TravParams` defaults; ρ from `configs/mission/mission_v1.yaml` |
  | manifest schema | `ariadne_volumetric_eval` results (`traversability`, `risk_terms`) |


## 2026-10-23 — D112: worldgen structural hazards (cantilever slabs, hung masses) with GT risk zones, for calibrating D111's risk proxy

- **What** (`docs/design/RISK_ZONES_4.2.md`, `python/dataset_tools/worldgen/hazards.py`):
  - An opt-in preset section `hazards`. Its own Philox stream (`S_HAZARDS = 40`) places cantilever slabs (held by a
    room wall) and hung masses, with undersides ≥ 1.95 m: above a standing responder, so they threaten a route
    without blocking it.
  - Six-face meshes, bottom included (`K_HAZARD`).
  - `scene_gt.npz` gains `hazard_footprint` and `risk_zone` (footprint ⊕ 0.5 m, `[ASSUMPTION]` a debris-fall
    margin); `scene.json` gains `hazards`.
  - New preset `configs/worldgen/ci_tiny_hazard.yaml` (ci_tiny + 2–3 cantilevers + 1–2 hung masses).
- **Why these two:** the US&R collapse patterns (secondary sources, §1 of the design). Cantilever collapse is "the
  most dangerous type of collapse to operate in". Lean-to and V slabs reach the floor and would need a 3D GT
  traversability: out of v1, stated.
- **Existing scenes unchanged, proven:**
  - at its default the section is left out of `Preset.to_canonical()`, so all four preset digests equal their
    pre-D112 values (pinned in `test_hazards.py`);
  - every golden scene-directory digest still matches (`test_golden.py`);
  - a no-hazard scene's `scene_gt.npz` gains no layers.
- **Tests:** 112/112 worldgen tests. The new ones:
  - digests;
  - a twin check: `ci_tiny_hazard` seed 0 has the same attempt, `free`, `esdf`, `trav_*`, starts, target and
    scripts as `ci_tiny` seed 0, since the streams are keyed by (seed, attempt, site);
  - placement constraints;
  - outward faces, with the bottom normal at −z;
  - the GT margin by hand.
- **Scenes:** seeds 0 / 1 / 2 have 5 / 4 / 5 hazards (scene ids `f232a40b77b3538f`, `d0ab877b3a06fc04`,
  `3406f0792c57a975`).
- **Next:** render on AWS g4dn (maintainer, 2026-10-23), run the Phase 1 estimator chain to a spool, then the 4.1
  map, then calibration by the protocol fixed in the design §3: the threshold on seed 0, the report on seeds 1–2,
  and AUC ≥ 0.75 to enter planning.

## 2026-10-23 — D113: 4.5 egress = D* Lite on D111's ρ_responder lattice, many-exit backward search, 5th-percentile body-support confidence

- **What** (`docs/design/EGRESS_4.5.md`, `cpp/planning/route/`):
  - **Search:** Koenig & Likhachev Fig. 3 (transcribed). The search runs from the exits, each seeded rhs = 0, to
    the fixed target. k_m = 0, and h = octile(target, ·), which is consistent because every edge costs ≥ its length.
  - **Lattice:** 1.8's, 8-connected with the corner rule. A non-traversable column has no edge (invariant 4 as a
    constraint).
  - **Edge cost:** |x−y|·(1 + mean endpoint penalty), with penalty = w_u(1 − support) + w_r·risk +
    w_c·max(0, ρ_safe − clearance)/ρ_safe. The weights are 1 / 0 / 1, fixed a priori; w_r stays 0 until D112's
    calibration passes.
  - **Support** = the observed fraction of the ρ-capsule body envelope.
  - **Replan:** every changed cell re-runs UpdateVertex on its 3×3 block.
- **Why:**
  - arch §22.1/§22.3 name D* Lite;
  - the map changes under a fixed plan, and each replan touches only the vertices whose costs changed;
  - the 1.8 A* stays as the ablation baseline (unchanged).
- **Measured:**
  - tests: D* Lite = fresh Dijkstra exactly on 180 random flip rounds (7.6× fewer expansions);
  - evidence spool: 12/12 routes GT-valid and the final route GT-optimal, the same as 1.8 (both first route at 145 s);
  - incremental = fresh at the final replan.
- **Pre-registered range check fired** (support spread 0.0999 < 0.1). The geometric observable band
  ([−0.31, 2.31] m, from altitude 0.8–1.2 m, half-FOV 28° and the 2.84 m depth gate) contains the whole envelope,
  so the switch is a no-op (byte-identical support).
- **Calibration:** degenerate (all valid, confidence 0.900–0.902), reported as degenerate.
- **Exit-gate reroute:** open; no run-level instance yet.

## 2026-10-23 — D114: 4.3 NBV design (generators A–E, a shared FUEL viewpoint sampler, 256-ray IG, p90 self-normalisation, one path cost, the energy-reserve rule); gate on lockstep L1

- **What** (`docs/design/NBV_4.3.md`):
  - **Generators A–E** feed one cylindrical viewpoint sampler (FUEL): radii 1.0 / 1.5 / 2.0 m, 16 azimuths,
    z = 1.0 m; kept if in `trav_a` with line of sight; best 2 per target.
  - **IG:** a 256-ray frustum bundle to 2.84 m.
    - Δcoverage: unknown voxels;
    - Δentropy: with p_occ = ½e^(−w) for free voxels and 1 − ½e^(−w) for occupied, the weight's unit being one
      σ_ref observation;
    - Δegress: unknown voxels in the route envelope;
    - Δpose_connectivity: a planning-side `ConnectivityOracle`, the Δλ₂ of 3.4's selector through a runtime
      adapter;
    - Δloop_disambiguation: `AliasSource`.
  - **Normalisation:** each term by its candidate-set p90, and G = the sum.
  - **Costs:** the path-proportional costs (motion, time, energy) are counted once, as
    λ_path(t) = (1 + λ_e(t) + λ_t(t))/3 ∈ [1, 5/3], with the arch's cubic schedules. Bandwidth and risk are
    p90-normalised with λ = 1.
  - **Hard rule:** if E_remaining < 1.3·E(return), only {return, hold}, as a supervisor event.
- **Why:**
  - arch §21.2–§21.4 as written, except cost normalisation, an extension flagged in the design: without it every
    λ⁰ is unit-bearing;
  - counting the costs once avoids the hidden 3× motion weight that per-term normalisation of ∝-path costs would
    create.
- **Measured before adoption:** Δentropy is not saturated on the evidence-spool map (known-voxel w p50 0.74–0.83;
  mean H 0.61–0.66 bit; a third over 0.9 bit).
- **Conflict surfaced:** two §21.3 producers are absent on this branch. 3.4's spectral gain is unmerged
  (feat/3.4-spectral), and §17.3's alias flags don't exist anywhere. Those terms are 0, with the reason in the trace,
  until they land. They're never a stand-in value.
- **Gate:** the coverage-vs-budget ablation has pre-registered axes (§5). It runs on **lockstep L1** (maintainer):
  B builds a gz pause at each barrier, so runs are reproducible per seed. The replay decision log tests the
  machinery only.

## 2026-10-23 — D115: 4.4 allocation = CBBA (L_t ≤ 2, event-triggered) and broker SGA on a time-discounted DMG score with common auction scales

- **What** (`docs/design/ALLOCATION_4.4.md`; the paper transcribed in ALGORITHMS §22.3.1):
  - **Algorithms:** CBBA (Algorithm 3, Table I, eq. 6 release), run only on a new task, a completed task, or a
    topology change. Broker sequential greedy (SGA) on the same inputs. Both logged identically.
  - **Score:** S_i(p) = Σ_j c̄_ij · exp(−λ_path · d_ij(p) / P_path).
    - c̄_ij = the D114 value net of bandwidth and risk;
    - d_ij(p) = the geodesic distance along the path;
    - P = common scales: the max over agents of each term's p90, frozen per allocation event.
- **Why:**
  - **DMG holds** (paper eqs. 11–12, triangle inequality on geodesics). D114's linear U is not DMG for two-task
    bundles, because an insertion shrinks a neighbour's detour.
  - **Under DMG, Theorem 1 makes CBBA's assignment equal SGA's.** So the arch's "auction vs broker-greedy" row
    measures communication (rounds, messages, bytes), not quality, and "logged identically" becomes an exact test.
  - **Common scales make bids comparable across agents.** Per-agent p90s would not.
- **Oracle:** exact every round. L_t = 1 by `cv::solveLP` (total unimodularity). L_t = 2 with N_u = 2 by a sorted
  disjoint scan.
- **Wire:** `TaskConsensus` (MessageClass TASK_CONSENSUS = 10 already exists) is to be added with the master
  session, which holds open 3.4 edits to `wire.fbs`. Until then the bytes come from a test codec, labelled, and are
  never claimed.
- **Replay** exercises only the connected two-agent case. The disconnection and staleness regimes come from the
  synthetic tests and live L1.

- **D115 correction (2026-10-23):** "CBBA = SGA under DMG" is exact at **L_t = 1** (2000 / 2000) but not at L_t = 2
  (5 / 1000 diverge through a stale second-slot bid, each with an eq. 6 tail release; a hand-built instance pins
  it). **Default L_t = 1.** L_t = 2 is a reported row, and its round and 50 % bounds are empirical there.
