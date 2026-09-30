# Day-1 Smoke Test Results

**Purpose:** resolve the `[SMOKE-TEST]` items of arch §0.2 that gate dependent code.
Phase 0 item 0.2. Nothing downstream may assume an API shape that is not recorded here.

**Pinned subject:** OpenCV `5.0.0` + `opencv_contrib` `5.0.0` — same tag, per C7.

### Commit identity — READ THIS BEFORE PINNING ANYTHING

`5.0.0` is an **annotated tag** in both repos, so `git ls-remote --tags --refs` returns the
**tag object** SHA, *not* the commit. The peeled `refs/tags/5.0.0^{}` value is the commit.
An earlier revision of this file recorded the tag-object SHAs as if they were commits.

| Repo | Tag object | **Commit (pin this)** |
|---|---|---|
| `opencv` | `9e2ede9628a55ec2742a1b180d3e69b0322281b9` | **`40738fb16ceddb5fb3fea747585f7ce6abb0605b`** |
| `opencv_contrib` | `7b6d3c42cc2eab1774c922f932df8ca61560a1cb` | **`755e50675d97db9b7d449d8bd6b09888646f6c6e`** |

The built image records the **commit** SHAs at `/opt/opencv5/opt-opencv-core.sha` and
`opt-opencv-contrib.sha`; §28.2 manifests must use those. Verify with
`git ls-remote --tags <url> '5.0.0*'` and read the `^{}` row.

## Evidence levels — read this before using anything below

| Tag | Meaning |
|---|---|
| `[SOURCE-VERIFIED]` | Read directly out of the pinned source tree. The declaration exists. |
| `[BUILD-VERIFIED]` | Confirmed against a **built and installed** OpenCV 5 — `infra/scripts/smoke_test.cpp` compiles, links and runs inside `ariadne/opencv-base:5.0.0`. |
| `[IMAGE-VERIFIED]` | Confirmed by running a built Ariadne container image other than the OpenCV one (e.g. `ariadne-sim`, which carries no OpenCV). Same strength as `[BUILD-VERIFIED]`, different artifact. |
| `[PENDING]` | Not yet checked. |

`[SOURCE-VERIFIED]` is deliberately weaker than `[BUILD-VERIFIED]`: a class can exist
in the tree and still be excluded by our `BUILD_LIST`, gated behind a CMake option, or
fail to link. Do not promote a row without re-checking it against the image.

---

## 1. Module layout — resolves C3 item 1 `[SOURCE-VERIFIED]` 2026-08-31

**The discrepancy is resolved, and both prior descriptions were half-right.**

Core modules in 5.0.0:
`calib core dnn features flann geometry highgui imgcodecs imgproc objdetect photo ptcloud stereo stitching video videoio world`

There is **no `3d` module**. There *is* a `3d.hpp` **header inside `geometry`**, which is
what the third-party write-ups describing "a 3d module" were actually looking at.

Where the functions this architecture names actually live:

| Function | Module | Header |
|---|---|---|
| `findEssentialMat`, `recoverPose`, `solvePnPRansac`, `triangulatePoints` | `geometry` | `opencv2/geometry/3d.hpp` |
| `stereoRectify`, SGBM | `stereo` | `opencv2/stereo.hpp` |
| `calibrateCamera` | `calib` | `opencv2/calib.hpp` |

`USAC_DEFAULT` and `USAC_MAGSAC` are present in `geometry/3d.hpp`, consistent with the
C3 claim that USAC is the default robust backend.

**Consequence:** every arch reference to "the geometry module" resolves to `geometry`;
`features2d` is now `features`.

## 2. Feature classes — one BLOCKING result `[SOURCE-VERIFIED]` 2026-08-31

Classes declared in `opencv2/features.hpp`:
`AffineFeature ALIKED ANNIndex BFMatcher DescriptorCollection DescriptorMatcher DISK
FastFeatureDetector Feature2D FlannBasedMatcher GFTTDetector KeyPointsFilter
LightGlueMatcher MSER ORB SIFT SimpleBlobDetector`

| Class | Status |
|---|---|
| `ALIKED` | **Present.** `create(const String& modelPath, const ALIKED::Params&)` — ONNX-backed |
| `DISK` | **Present.** `create(modelPath, maxKeypoints, ...)` — ONNX-backed |
| `LightGlueMatcher` | **Present**, derives from `DescriptorMatcher` |
| `ANNIndex` | **Present**, with a `Distance` metric enum |
| **`XFeat`** | **ABSENT — no implementation anywhere in either tree** |

### XFeat absence — verification and impact

Exhaustive case-insensitive search over both source trees returns only: the
`xfeatures2d` contrib module, AVX `XFEATURE_MASK_*` CPU-feature macros, and tutorial
prose. There is no XFeat detector, no ONNX loader for one, and no contrib equivalent.

This **contradicts a locked decision** (CLAUDE.md §1) *and* a specific claim in
`docs/ALGORITHMS.md` L63, which documents `cv::XFeat::create(modelPath, maxKeypoints,
scoreThreshold, inputSize, backendId, targetId)` as a native OpenCV 5 class. The
catalogue entry is in error. Per CLAUDE.md an algorithm may not be silently substituted;
see `DECISIONS.md` D6 for the three options, one of which (XFeat as our own ONNX export
through `cv::dnn`, §7.4) **preserves** the locked decision.
It does **not** block Phase 0, whose `cpp/core` is OpenCV-free by construction; it
blocks the Phase 2.1 perception swap-in. Tracked in `docs/STATUS.md`.

**Second-order consequence, and it applies to every option:** OpenCV 5 ships these
classes but **not their models** — `ALIKED::create` and `DISK::create` both take a
`modelPath` to an ONNX file. A learned feature path therefore requires sourcing or
exporting ONNX weights regardless of which detector is chosen. ALIKED/DISK/LightGlue are all ONNX-model-driven, so
§7.4's DNN execution contract (per-thread `Net`s, ORT fallback, log which ran) governs
the *primary* feature path, not just the hard-pair path. C8's "ALIKED never runs at
frame rate" constraint becomes sharper, not looser.

## 3. Volumetric — HashTSDF confirmed `[SOURCE-VERIFIED]` 2026-08-31

`opencv2/ptcloud/volume_settings.hpp` declares `enum class VolumeType { TSDF=0, HashTSDF=1, ColorTSDF=2 }`
with `class CV_EXPORTS_W Volume` in `ptcloud/volume.hpp`, plus truncation distance,
running-average frame count, and resolution settings.

**Arch §20.1 is satisfied — the from-scratch TSDF fallback is NOT required.**
Noted constraint, straight from the header: *HashTSDF supports only equal resolution in
all three dimensions* (non-uniform resolution is TSDF-only). Phase 4.1 must respect this.

## 4. Contrib modules `[SOURCE-VERIFIED]` 2026-08-31

| Need | Module | Status |
|---|---|---|
| DisparityWLSFilter (§9.3) | `ximgproc` | Present — `createDisparityWLSFilter(Ptr<StereoMatcher>)` and `createDisparityWLSFilterGeneric(bool)` in `disparity_filter.hpp` |
| PSNR/SSIM harness (§30) | `quality` | Present — `qualitypsnr/ssim/gmsd/mse/brisque` |
| Radiometric (§6) | `xphoto` | Present |
| Tracking baselines | `optflow` | Present |
| Legacy descriptors (C3) | `xfeatures2d` | Present (SURF/BRIEF/FREAK/DAISY moved here as C3 stated); also `xstereo` is new |

---

## 4b. Built-image verification `[BUILD-VERIFIED]` 2026-09-06

Image `ariadne/opencv-base:5.0.0` (amd64, 1.72 GB). Reproduce with:

```bash
docker run --rm -v "$PWD/infra/scripts:/w" -w /w ariadne/opencv-base:5.0.0 sh -c '
  g++ -std=c++17 -O2 smoke_test.cpp -o /tmp/smoke \
    -I/opt/opencv5/include/opencv5 -L/opt/opencv5/lib \
    $(ls /opt/opencv5/lib/libopencv_*.so | sed "s|.*/lib|-l|;s|\.so$||")
  LD_LIBRARY_PATH=/opt/opencv5/lib /tmp/smoke'
```

**The compile-and-link succeeding is the evidence.** Every class below is referenced as a
complete type, and `cv::ALIKED::create`'s address is taken to force real symbol
resolution; a missing class is a build error, not a runtime surprise.

| Key | Value | Meaning |
|---|---|---|
| `opencv_version` | `5.0.0` | |
| `build_info_hash` | `9127192543017384188` | goes in the §28.2 manifest (C7) |
| `features_{ALIKED,DISK,LightGlueMatcher,ANNIndex,ORB,SIFT}` | `PRESENT` | all complete types |
| `features_ALIKED_create_symbol` | `LINKED` | resolves in the installed `.so` |
| `ptcloud_HashTSDF` | `CONSTRUCTED` | **§20.1 satisfied; no from-scratch TSDF needed** |
| `ptcloud_voxel_size` / `truncated_distance` / `max_weight` | `0.005859` / `0.041016` / `64` | HashTSDF defaults |
| `geometry_USAC_DEFAULT` / `USAC_MAGSAC` | `32` / `38` | C3's USAC claim confirmed |
| `stereo_StereoSGBM` | `CONSTRUCTED` | §9.3 |
| `ximgproc_DisparityWLSFilter` | `CONSTRUCTED` | §9.3, contrib linked (C7) |
| `quality_PSNR_black_vs_white` | `0.000000` | **correct, not a failure** — maximum possible error is 0 dB by definition |
| `core_setNumThreads_pinned` | `1` | thread pinning works; required by §27.1 COOL discipline |
| `core_numThreads_default` | `8` | this host |

### Install layout facts

- Headers install to **`/opt/opencv5/include/opencv5/opencv2/`** — note `opencv5`,
  not the OpenCV-4-era `opencv4`. Any `-I` path copied from OpenCV 4 docs is wrong.
- Libraries: `/opt/opencv5/lib/libopencv_*.so`, shared (required — §27.1's COOL arm
  switches implementations by loader path, which static linking would prevent).

### Build-list note

The contrib set built is **exactly C7's four modules** (`ximgproc`, `xphoto`, `optflow`,
`quality`). `xfeatures2d` was tried and **removed**: nothing in the architecture uses
SURF/BRIEF/FREAK/DAISY, and it triggers `boostdesc`/`vgg` blob downloads at *configure*
time that stalled the build behind a slow third-party host (>136 s per blob, then hung).
If a future need arises, pre-seed those blobs into the image rather than fetching at configure.

## 4c. Simulator image `ariadne-sim` `[IMAGE-VERIFIED]` 2026-09-29

`infra/docker/sim.Dockerfile`, built on this dev box (x86-64, **no GPU**), and exercised by
`infra/scripts/sim.sh`, whose default command runs the acceptance test `infra/scripts/sim_smoke.py`.

| Fact | Value |
|---|---|
| Base | `ros:jazzy-ros-base@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8` |
| ROS packages | dated snapshot `snapshots.ros.org/jazzy/2026-09-11`, signing key fingerprint verified (D25) |
| Gazebo | `ros-jazzy-gz-sim-vendor 0.0.13` = gz-sim8 **8.15.0** (Harmonic) |
| OpenCV in image | **none**; build fails if one appears (D25) |
| Image | `sha256:90a1e5a6...3dda`, 740 MB (local dev tag, not yet pushed) |
| Full package list | `/opt/ariadne/sim-versions.txt` inside the image |

**Headless camera rendering: PASS on Mesa llvmpipe (software EGL).**
`gz sim -s -r --headless-rendering`, camera bridged to ROS 2 by `ros_gz_bridge parameter_bridge`,
frames received by rclpy: 320x240 rgb8, luma stddev 73.9, red box 31.8% / blue box 19.5% of the
frame. The test judges pixel statistics, not exit codes. **Each failure branch has fired**:

| Control | Result |
|---|---|
| No EGL vendor (`__EGL_VENDOR_LIBRARY_FILENAMES` -> nonexistent) | gz segfaults (-11), "exited early", exit 1 |
| Clean render of a uniform frame (`smoke_blank.sdf`: no light, black background, camera at empty sky) | luma stddev 0.0, "near-uniform", exit 1 |
| Real structure, wrong scene (a textured cloud sky, see below) | "without the expected red and blue boxes", exit 1 |

The blank-world control only proved its branch on the second try. The first version of
`smoke_blank.sdf` contained `<sky>false</sky>`, and in SDF the mere presence of `<sky>` ENABLES a
textured cloud sky; the text is ignored. The control rendered clouds (stddev 33.1) and failed
through the scene-mismatch branch instead. Removing the element gave the uniform frame intended.

**Timing:** sensor stamps are exactly 0.200 s apart in sim time (5 Hz as configured), and wall-clock
receive rate is ~4.8 Hz, so the world runs at RTF ~1. An earlier reading of "870 frames in 60 s"
(3x rate) was wrong: `sim.sh` did not forward `ARIADNE_SMOKE_TIMEOUT`, so that run lasted the 180 s
default. Fixed; the script now forwards the test's knobs.

**MCAP recording works in the image:** `ros2 bag record -s mcap /smoke/camera` for ~11 s gave 55
`sensor_msgs/msg/Image` messages, storage `mcap`, serialization `cdr`, 12.1 MiB.

Bring-up defects in the TEST, both fixed: the first colour detector used a brightness floor of 100,
but the shaded red box renders at (95, 30, 30), so a correct render was rejected for 870 frames
with a message blaming black frames. The detector now uses channel dominance, and the failure
message distinguishes a uniform frame from an unrecognised scene.

**NOT verified: the NVIDIA EGL path on g5/g6** `[UNVERIFIED]`. It depends on the NVIDIA Container
Toolkit injecting the driver's EGL vendor library via glvnd (`NVIDIA_DRIVER_CAPABILITIES` includes
`graphics`). No GPU exists here. First task on a g5/g6: `infra/scripts/sim.sh` must print
`render mode: nvidia-egl` and `SIM_SMOKE=PASS`.

---

## 5. Open items

| Item | Status |
|---|---|
| Built-image verification (amd64) | **DONE** — §4b |
| `cv::getBuildInformation()` hash into manifest | hash captured (`9127192543017384188`); **manifest writer itself is 0.5, not yet written** |
| DNN engine runs each ONNX model, or ORT fallback (§7.4) | `[PENDING]` — needs models |
| **arm64 build + full smoke test** | `[PENDING]` — no local arm64; deferred to Graviton / Arm CI runner |
| PyPI wheel status for OpenCV 5 | `[PENDING]` — not needed; we build from tag per C7 |
| COOL day-1 trial check (0.3) | `[PENDING]` — needs AWS account |
| `ariadne-sim` NVIDIA EGL path on g5/g6 | `[PENDING]` — needs a GPU instance; Mesa path PASSES (§4c) |

## 4d. AWS account capability probe `[ACCOUNT-VERIFIED]` 2026-09-29

Account `836505524290`, `ap-southeast-2`, **free plan**. Read-only checks plus one real launch attempt.

| Check | Result |
|---|---|
| Free-tier-eligible types (`describe-instance-types --filters free-tier-eligible=true`) | `c7i-flex.large m7i-flex.large t3.micro t3.small t4g.micro t4g.small t8i.micro t8i.small` |
| Offered in region | c8g.medium/large, m8g.large, c7g.large, g4dn.xlarge, g6f.large: all yes |
| Quota: Standard family (on-demand / spot) | 5 / 5 vCPU |
| Quota: G and VT (on-demand / spot) | **0 / 0**: no GPU launch possible without an increase |
| `run-instances --dry-run` | "would have succeeded" for every type, **including g4dn.xlarge at 0 quota**, so dry-run checks IAM only and cannot test plan or quota restrictions |
| **Real launch, c8g.medium** (Graviton4) | **REFUSED**: `InvalidParameterCombination: The specified instance type is not eligible for Free Tier.` No resources created (verified: 0 instances, 0 volumes). |

**Consequence:** on the free plan, COOL cannot run at all. It requires Graviton3/4 (c7g/m7g/r7g, c8g/m8g/r8g), and none are free-eligible. The Marketplace subscribe page rendering on the account does not change this, because the EC2 instance-type gate is separate. The only arm64 types this plan permits are `t4g.micro/small` (Graviton2), which the COOL listing does not support.

## 4e. Determinism on real Graviton4 hardware `[HARDWARE-VERIFIED]` 2026-09-29

Run `20260929T145114Z-cpu-762cb1` via `infra/scripts/spinup.sh run --profile cpu -- "$(cat infra/scripts/tests/determinism_graviton.sh)"`: **c8g.large spot,
eu-north-1c, AWS Graviton4** (SVE2, AL2023 kernel 6.18), `infra/scripts/check_determinism.sh`.

```
g++ (GCC) 11.5.0   /   clang 15.0.7 (AWS)
PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   det=11806531094783593156 philox=14182439601419266695 degrade=8535765251477677907
arch=aarch64  configurations=6   DETERMINISM=PASS
```

Identical to the digests recorded on x86-64 with gcc 13.3 / clang 18.1. This is stronger than the earlier
QEMU result in two ways: real silicon rather than an ISA emulator, and **different compiler major
versions** (gcc 11 vs 13, clang 15 vs 18). Bit-identity now holds across architecture, compiler vendor,
compiler version and optimization level.

Cost: 67 s wall, spot $0.0151/h, **$0.0003 compute** (ledger: `build/runs/ledger.csv`).

Guard verification: the instance ended `terminated` with state reason
`Client.InstanceInitiatedShutdown`, so an in-instance `shutdown -h` TERMINATES rather than stops
(brake 1 works). AWS accepts spot together with `instance-initiated-shutdown-behavior=terminate`.
Post-run sweep: 0 stray instances, 0 EBS volumes.

Scope: this covers the deterministic primitives (`det_math`, Philox, degradation operator). Item 0.2's
arm64 **OpenCV 5 build + smoke test** is still open.

## 4f. Full-rig recording throughput on software rendering `[IMAGE-VERIFIED]` 2026-09-29

`infra/scripts/sim.sh python3 /opt/ariadne/bin/sim_throughput.py`, world `rig_throughput.sdf`: the arch 5.3 rig
(stereo 1280x720 at 20 Hz, 10 cm baseline, IMU at 200 Hz) run for exactly 10 000 physics steps (10.0 sim-s),
bridged by `ros_gz_bridge` and recorded by `ros2 bag record -s mcap`, then read back and checked.
Host: 8-core x86-64, no GPU, Mesa llvmpipe.

| Measure | Result |
|---|---|
| Realtime factor | **0.566** (10.0 sim-s in 17.7 wall-s) |
| Left / right frames | 201 / 201 of 200 expected (includes the t=0 frame); max gap = nominal 0.05 s; 0 duplicate stamps |
| Stereo pairing | **1.0**: every left frame has an identically-stamped right frame |
| IMU samples | 1999 of 2000; max gap 0.010 s = **one dropped sample** |

**Conclusion:** software rendering records the full rig **completely**. It slows the sim (RTF < 1) rather than
dropping frames, which is the acceptable failure mode for record-once bags. Bag recording does not need a GPU;
a 5-minute single-agent mission would take ~9 wall-minutes.

Caveats, stated rather than assumed away:
- One run, one simple scene. The Phase 1 two-room corridor, multiple agents (render cost scales roughly with
  camera count), and Phase 5's shadow-casting lamp will all be slower. Re-measure there.
- The single missing IMU sample is real. It happened in the bridge/recorder path, not the renderer, since the
  cameras were complete. Preintegration tolerates a 10 ms gap, but completeness should be asserted per bag
  rather than assumed.

## 4g. Kinematic mover, render/pose timing, gz IMU, and 2-agent bag size `[IMAGE-VERIFIED]` 2026-09-30

Probes in `simulator/smoke/kinematic_probe/` (README there), gz-sim 8.15.0, Mesa llvmpipe, 8-core x86-64.

**Physics-free kinematic pose writes work.** A system plugin that overwrites `components::Pose` of the rig model
in `PreUpdate` (and marks it `PeriodicChange`) moves the rendered cameras, the depth camera and a spot light
attached to the link. Identical results with the model `<static>true` or `false`, and with or without the
Physics system loaded. (`/world/<w>/set_pose` was not used: it goes through `WorldPoseCmd`, which only Physics
consumes.)

**Render time equals pose time.** The rig moved at 5 m/s toward a wall; the depth at the image centre implies
the pose time of each frame. Offset between implied pose time and header stamp: **0.00 ms** on every frame
(resolution better than 0.05 ms, i.e. far below one 1 ms step). A frame stamped t_k shows pose(t_k): no
one-step lag. Scope of this claim: measured on the **depth camera, 320x180, one rig, light load**. RGB-camera
timing was not isolated (the lamp-centroid pass cannot reveal RGB lag: the lamp moves with the camera), and
a render-thread race would only show under load. The 1.1 acceptance test re-checks it on RGB under the full
two-agent load before "zero camera-IMU offset" is claimed.
Lateral motion kept the lamp's brightness centroid at the exact image centre column (159.5 of 320): the light
follows the link.

**gz's own IMU is blind to kinematic motion.** Over 401 samples spanning 5 m/s motion and a velocity
discontinuity, the gz IMU sensor reported constant specific force (0, 0, 9.8), std 0, and zero angular
velocity, in all three variants. The `Imu` system reads `AngularVelocity`/`LinearAcceleration` components that
only Physics fills (gz-sim8 `src/systems/imu/Imu.cc`); a pose-written body never gets them. **Consequence:** under
kinematic movers the IMU must be synthesized from the analytic trajectory (the OpenVINS simulator pattern:
B-spline/closed-form derivatives plus seeded noise), not taken from gz.

**Two-agent full-rig bag, 5.0 sim-s** (each agent: stereo 1280x720 RGB + co-located 1280x720 float32 depth, all
20 Hz; textured 8x12 m room, two point lights, no shadows):

| Measure | Result |
|---|---|
| Frames | 101/101 per camera (all 6), none dropped |
| RTF | **0.217**, zstd_fast run, compressor sharing the CPU (vs 0.566 for one stereo rig without depth, §4f); the uncompressed run's RTF was lost to output truncation. Real worlds have ~10 lights, so re-measure |
| Raw message bytes | 372 MB per sim-s (RGB 55.9 MB/s per camera, depth 74.5 MB/s per depth camera) |
| MCAP `zstd_fast` | **74.6 MB per sim-s** (~5x) |

A 4-minute two-agent mission is therefore ~18 GB and ~18 wall-minutes on llvmpipe. Recorded as a sizing input,
not yet a decision.

**Also observed:** PBR `<albedo_map>` loads under llvmpipe, but gz's `<box>` maps the full texture onto every
face regardless of face aspect, so long thin faces and large floors are visibly stretched/streaked. Textured
geometry needs explicit UVs (meshes), not primitive boxes.

## 4h. OBJ quad texture mapping `[IMAGE-VERIFIED]` 2026-09-30

`simulator/smoke/obj_uv_probe/run_obj_probe.py` (sim image, llvmpipe). Two 2.0x1.5 m quads written by
`python/dataset_tools/worldgen/mesh.py` at 3 m from a 1280x720, HFOV 90° camera, one wound towards the camera
and one away, with a 512x384 texture (R = column gradient, G = 255 at the top row, B = 64 px checker), loaded
through **relative** `meshes/…` and `textures/…` URIs.

| Check | Result |
|---|---|
| u direction | red 9.2 at left edge → 239.0 at right edge, corr > 0.99: u runs left→right seen from the front |
| v direction (V-flip) | green 238.1 at top → 14.6 at bottom: image row 0 is v = 1, **no flip** |
| aspect | checker cells 53.5 × 53.5 px (expected 640·0.25/3 = 53.3): square, no stretch |
| back face | region of the away-facing quad: mean 0.0 (background): **back faces are culled** |
| URIs | relative mesh and albedo URIs resolve against the SDF file's directory |

**Consequence:** the quad convention in `mesh.py` (p0 = bottom-left seen from the front, CCW triangles,
UV (0,0) at p0) is what gz renders. Walls are single-sided quads facing into free space.

## 4i. Scene model inclusion and first procedural-scene render `[IMAGE-VERIFIED]` 2026-09-30

**`<include>` does not resolve relative paths in sdformat14.** A world in the scene directory with
`<include><uri>model.sdf</uri></include>` was resolved against sdformat's own share directory
(`.../share/sdformat14/1.11/model.sdf`, the schema file) and failed. Tested forms, errors reported by
`gz sim` for each: `./model.sdf` 5, `.` 2, `./` 2, `file://model.sdf` 5,
`model://<scene-dir>` with `GZ_SIM_RESOURCE_PATH=<parent>` **0**. **Consequence:** generated worlds *inline*
the scene `<model>` and live in the scene directory, where mesh/texture URIs relative to the SDF file resolve
(§4h). `model.sdf` remains for `model://` use with the resource path set.

**First render of a generated scene** (`maze_medium`, seed 0: 68 wall quads, 33 floor + 33 ceiling tiles,
20 prop faces, 9 point lights, 34 Mpx of albedo), via `simulator/smoke/scene_render/run_scene_render.py`
from three GT poses: 0 gz errors; frame means 70–97, std 25–31 (grey levels), black pixels ≤ 0.9 %. Walls
face into free space, back faces are culled, textures carry the §3 layers at the expected scale.
Close walls are visibly magnified, the documented 256 px/m trade-off. Render
throughput with the full light count is measured with the agent rig in step 6, not here.

## 4j. Two full rigs in a real generated scene: RTF and bytes `[IMAGE-VERIFIED]` 2026-09-30

`simulator/smoke/kinematic_probe/run_scene_rtf.py`, scene `maze_medium` seed 0 (9 point lights, 34 Mpx of
albedo textures, 154 textured quads), two rigs (stereo 1280x720 R8G8B8 + depth, all 20 Hz) circling in two
rooms, 20.0 sim-s, recorded with `ros2 bag record -s mcap --storage-preset-profile zstd_fast`, llvmpipe.

| Depth resolution | RTF (incl. ~15 s startup) | MCAP MB per sim-s | Raw MB/s per depth topic | Frames |
|---|---|---|---|---|
| 1280x720 | 0.150 | 84.7 | 73.9 | 401/401 on all 6 topics |
| 640x360 | 0.168 | 83.5 | 18.5 | 401/401 on all 6 topics |

**Consequences.** (1) Quartering the depth pixels saves ~1 % of the *compressed* bytes: float depth of planar
walls compresses extremely well, and the bytes are dominated by the richly textured RGB (4 x 55.4 MB/s raw,
~3x under zstd). The design's "depth at 640x360" sizing deviation is therefore **withdrawn**: GT depth stays at
the colour resolution. (2) Honest sizing: ~84 MB per sim-second for two agents, RTF ~0.17 on this 8-core host.
A 5-minute two-agent mission is ~25 GB and ~30 wall-minutes on llvmpipe. No frame drops; the renderer slows
the sim rather than dropping, as in §4f.

## 4k. Kinematic mover plugin, mission recording, and bridge drops `[IMAGE-VERIFIED]` 2026-10-01

`simulator/plugins/kinematic_mover/KinematicMover.cc` (gz-sim 8.15, built with -Wall -Wextra -Werror),
driving `ci_tiny` seed 0 (two agents, stereo + depth + IMU + GT pose) through `infra/scripts/sim_record.py`.

- **sdformat does not propagate `FilePath()` to `<plugin>` elements**: a relative ESDF path resolved against the
  process cwd and the mover aborted (loudly, as designed). gz-sim sets `components::SourceFilePath` on model
  entities from the model element's file path (`SdfEntityCreator.cc:512`); the mover resolves relative paths
  through it. Verified by launching from an unrelated cwd.
- **The gz rotation convention matches the scripts**: the mover's start-up self-check (gz `Quaterniond(0, pitch,
  yaw)` vs R = Rz(yaw) Ry(pitch)) passes, and the recorded GT pose and IMU equal the Python script evaluated at
  their own stamps with **worst absolute error 0.0** over 3198 pose and 3199 IMU samples (8 sim-s slice).
- **All camera frames present** (161/161 per camera, depth and camera_info topics), stereo pairing 1.0, RTF 0.13–0.14.
- **A transient downstream stall dropped one message on two 200 Hz topics at the same stamp (4.565 s)** with
  default queue depths (agent 0's IMU was complete, so the mover published every sample). Fix: bridge
  `subscriber_queue`/`publisher_queue` 1000 and a reliable depth-1000 recorder QoS override on IMU and GT pose;
  the verifier is now strict (count == expected, max gap == period). The 8 s slice then passed strict checks;
  the full-mission run did NOT confirm the fix (§4l).

## 4l. Full `ci_tiny` mission; canonical IMU/GT streams (D39) `[IMAGE-VERIFIED]` 2026-10-01

- **Deep queues are not lossless.** Full mission (199 sim-s, RTF 0.213, 12.6 GB): every camera, depth and
  camera_info topic complete (3981/3981), stereo pairing 1.0, GT pose 79,600/79,600; agent-1 IMU **39,799 /
  39,800** — one sample lost at t = 174.435 s between gz-transport and the bag. Every sample that arrived
  equalled the analytic value (worst 0.0). Not root-caused; now counted per run as `live_gaps`.
- **rclcpp leaves CDR padding uninitialised.** The first canonical writer used `rclcpp::Serialization`: the 2
  alignment bytes after "agent_0/base_link" held `"at"` (heap residue), so two runs differed and every IMU
  sample mismatched the live bridge bytes (whose padding happened to be 0). Fixed with an own encoder that
  zeroes padding (`ros_cdr_writer.hpp`), tested against rclpy bytes, captured live bridge bytes and 18
  frame-id lengths.
- **Canonical vs live, 8 s fresh recording** (plugin + tool from the same build): 6,400/6,400 live samples
  byte-identical to canonical, 0 gaps; canonical vs Python evaluator worst 0.0 (6,404 samples); canonical bag
  regenerated byte-identical; PASSED.
- **Canonical vs live, the 199 s bag above** (re-verified with `--verify-only`; bag from the previous plugin
  build): 159,199/159,199 live samples byte-identical, 0 mismatched / off-grid / duplicate, `live_gaps` 1 at
  174,435,000,000 ns (coverage 0.99997); canonical 4 × 39,801 samples, 5.1 MB (0.04 % of the sensor bag),
  vs Python worst 3.3e-16 (GT quaternion) / 0.0 (IMU); PASSED.
- **Fresh full mission with the D39 pipeline — FAILED on a camera frame** (RTF 0.224, 887.7 wall-s):
  `/ariadne/agent_1/right/image_raw` 3980/3981 (one 100 ms gap), agent-1 stereo pairing 0.9997. The live IMU/GT
  were complete this time: 159,200/159,200 byte-identical to canonical, `live_gaps` 0. The canonical bag's
  sha256 (`4052ae3d…`) equals the one generated for the older 199 s bag: reproducible across two recordings and
  two plugin builds. The verifier failed the recording as designed (an image cannot be regenerated).
- **Transport loss measured on `/clock`; the layer of the IMU and image drops is unknown.** The recorder's and
  bridge's output had been discarded; it is now kept in `RECDIR/logs/`. On an 8 s run that PASSED, the recorder
  reported "Number of messages lost on the transport layer: 27" at close, and the bag's `/clock` (1 kHz, not
  consumed by replay) misses exactly 27 of 7,998 steps. The runs that lost an IMU sample and a camera frame
  predate log capture, so their layer is not known: DDS transport, rosbag2's cache (100 MB double-buffered
  by default; it logs "Cache buffers lost messages per topic" on close), or gz-transport → bridge (silent in
  both logs). The bridge's default queues are 10 (`ros_gz_bridge/bridge_config.hpp:37,40`, installed headers),
  i.e. ~0.5 s of stall tolerance on a 20 Hz image topic. `[ASSUMPTION]` the 200 Hz and image topics are hit
  less often than `/clock` because they carry fewer messages or deeper queues. The next full run with logs gives
  per run: (expected − in bag) = recorder transport-lost + recorder cache-lost + silent residual.
  For scale, the fresh full run's `/clock` misses 1,150 of ~199,000 steps (re-verified; that run has no logs).
  Its canonical content digest `06054e72…` equals that of the older 199 s bag's canonical.

