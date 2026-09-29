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
be reported, `[ASSUMPTION]` pending the maintainer's confirmation. The data lives only in `.cache/` (gitignored and
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
