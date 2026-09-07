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
