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

This **contradicts a locked decision** (arch §0.2) *and* a specific claim in
`docs/algorithms.md` L63, which documents `cv::XFeat::create(modelPath, maxKeypoints,
scoreThreshold, inputSize, backendId, targetId)` as a native OpenCV 5 class. The
catalogue entry is in error. Per arch §0.2 an algorithm may not be silently substituted;
see `decisions.md` D6 for the three options, one of which (XFeat as our own ONNX export
through `cv::dnn`, §7.4) **preserves** the locked decision.
It does **not** block Phase 0, whose `cpp/core` is OpenCV-free by construction; it
blocks the Phase 2.1 perception swap-in. Tracked in `the development log (archive/dev-history branch)`.

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

**Verified 2026-10-22 on g4dn.xlarge (Tesla T4, driver 595.91.07; DLAMI base OSS NVIDIA, Ubuntu 24.04):**
`infra/scripts/sim.sh` printed `render mode: nvidia-egl` and `SIM_SMOKE=PASS` with the unmodified image
`sha256:8b780f52…` (run `20261022T200422Z-gpu-b04560`, D137). The NVIDIA Container Toolkit was already the docker
runtime on the DLAMI. A g5/g6 instance has not been tried. One prerequisite on an instance: the shipped tree must
belong to the image's user (uid 1000), or containers fail EACCES writing into `/work`.

~~**NOT verified: the NVIDIA EGL path on g5/g6** `[UNVERIFIED]`.~~ It depends on the NVIDIA Container
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
- **Attribution: every loss is on the recorder's DDS subscription.** Three full runs at unchanged settings,
  recorder at debug level (it names the topic of each transport loss; `[rcl]` traces filtered), RTF 0.230–0.231.
  Per run, expected − in bag = recorder transport-lost + residual, and the residual is **0 on every topic**; no
  cache-overflow lines. So gz-transport → bridge and rosbag2's cache lost nothing.

  | run | non-clock losses (topic: n) | `/clock` lost |
  |---|---|---|
  | 1 | a0 left 1, a0 right 1, a1 left 1, a1 right 3, a1 gt/pose 1 | 1,014 |
  | 2 | a1 depth 2, a1 right 1 | 1,058 |
  | 3 | a0 depth 2, a0 gt/pose 1, a1 left 1 | 1,078 |

  Losses are isolated single samples minutes apart, not a stall burst. Every topic has one publisher, all
  reliable (bag metadata; DDS discovery does not carry history depth, so the depth in effect is not visible
  there). GT pose was lost twice with bridge `publisher_queue` and recorder depth both 1000 (5 s at 200 Hz),
  so depth alone is not a guarantee `[ASSUMPTION: consistent with reliable keep-last eviction after dropped
  datagrams on a CPU-saturated host with rmem_default 208 KB; Fast DDS repair timing not yet read]`.
  Debug logging may raise the rate (these runs lost more than earlier info-level runs).
  Both remedies are available in the installed build: `libros_gz_bridge.so` exports
  `convert_gz_to_ros<Image|CameraInfo>` (a direct gz-transport → rosbag2 writer can reuse the bridge's
  conversions), and the bridge honours `qos_overrides.*` parameters (keep-all is configurable).

## 4m. Direct gz-transport recorder (D40): first clean full-mission recordings `[IMAGE-VERIFIED]` 2026-10-01

`ariadne_gz_recorder` replaces ros_gz_bridge + `ros2 bag record` in the recording path (D40).

- **Paused start is real and observable.** With the world started paused, gz republishes `/clock` = 0 (~1,100
  messages in 8 s of discovery wait, ~1,250 in a full run); the recorder's first version flagged them as
  duplicates. They are now counted as `paused_repeats` and not written; any other repeat is still an error.
  Camera frames now start at t = 0 (they started at 2 ms with `-r`), still 161/161 per 8 s.
- **Failure modes fire.** 13 tests drive the real binary with a fake gz publisher: clean stream written
  exactly (log time = stamp); gap → "LOST 1 message(s)", exit 1; duplicate, out-of-order, irregular interval,
  late start, only the t = 0 message missing (first stamp checked exactly), stamp 0 after start, incomplete on
  interrupt → exit 1; backlog cap → ABORT, exit 1; no
  publisher, unsupported type pair → exit 2.
- **Two full `ci_tiny` missions (199 sim-s), both PASSED strict verification.** Per run: recorder 0 errors,
  405,973 messages written; all 6 image/depth topics 3,981/3,981 and stereo pairing 1.0; `/clock` 199,001/199,001
  (0 missing steps); live IMU and GT pose 4 × 39,800, all byte-identical to canonical, `live_gaps` 0; canonical
  reproducible, content digest identical across the runs (`a4a56220…`). RTF 0.243 / 0.241 (bridge path:
  0.21–0.23); peak recorder backlog 150 / 38 MB against a 4 GiB cap; 12.62 GB each. Against the bridge path's
  3–9 non-clock losses per full run (§4l), two clean runs are strong evidence (≈1 % by chance at a mean of 5).
- **Open: render determinism.** The two sensor bags differ as files (sha256, and 26 KB in size); chunk
  layout follows arrival order, so this says nothing yet about pixels. A per-topic content comparison is
  needed before claiming identical renders across runs. Both runs from a dirty tree: not evidence yet.

## 4n. §6 calibration recovery: what gz actually renders `[IMAGE-VERIFIED]` 2026-10-01

`cpp/runtime/tools/ariadne_sim_calib.cpp` on a full `ci_tiny` recording (agent 0's ChArUco dance: 1,644
calib-window frames per camera, ~1,040 with a board detected; dev run, uncommitted tool). All five checks:

- **Lens distortion is NOT rendered (ogre2).** gz-rendering 8 has a `DistortionPass` only in the Ogre1 backend;
  under ogre2 gz-sensors logs "ImageBrownDistortionModel is not supported in ogre2" (seen once per camera with
  `-v 2`; invisible at the default verbosity) and renders a pure pinhole image — while `camera_info` still
  advertises `plumb_bob` [-0.05, 0.01, 0.0005, -0.0002, 0] from the SDF. Measured: the GT-reprojection residual
  has no radial trend (apparent k1 = -0.002 / +0.0006 with corners out to normalized radius 0.94, where the SDF's
  k1 would give ~23 px); a free calibration recovers k1 = 0.0001 ± 0.0007 and its RMS equals the pinhole-only
  RMS (0.054 px). **The recorded camera_info D is false; no consumer may use it.**
- **camera_info's principal point is half a pixel off the OpenCV/ROS convention.** gz writes cx = W/2 = 640,
  cy = 360; the GT residual mean is (-0.50, -0.50) px in every segment, both cameras. With cx = 639.5,
  cy = 359.5 (pixel-centre convention) and no distortion, the GT reprojection RMS drops from 0.72 px to
  0.058-0.067 px (p95 0.08-0.12 px): the renderer is an exact pinhole fx = fy = 640 to ~0.06 px.
- **Timing re-confirmed under the paused start** (§4g was measured with `-r`): fitting residual = δ · image
  velocity over moving corners gives δ = -7 µs (left) / +47 µs (right); a frame stamped t shows the pose at t.
- **Stereo extrinsic**: with the pixel-centre K and no distortion, stereoCalibrate recovers T = (-0.100002,
  0.000011, -0.000049) m vs /tf_static (-0.1, 0, 0): 0.05 mm, 0.0004°. (With each camera's own free
  calibration, fx ± 0.5 px, the error is 2 mm — calibration noise, not the rig.)
- **Board frame**: the asset texture's corners sit exactly where getChessboardCorners() puts them (x right, y
  down; worst 1.5e-5 px); PnP vs GT over 1,025 frames has no 180° flips (rotation error mean 0.22°, translation
  mean 1.0 mm — planar-PnP noise).
- **gz warnings are now a gate.** gz runs at `-v 2` and every `[Wrn]`/`[Err]` fails a recording unless it is on
  `GZ_WARNING_ALLOWLIST` with its reason (OBJ files without .mtl; ogre2's reserved visibility bits). With the
  current rig every recording therefore fails on the distortion warning until the rig decision (D41) lands.
- **D41 lens operator, end to end** (same recording, frames passed through `lens.hpp` before detection; truth =
  rig.lens): GT reprojection RMS 0.072-0.093 px per segment, |mean| <= 0.016 px, apparent residual k1 ~1e-4,
  time offset ~1 µs. Free calibration (80 views): k1 -0.0506 / -0.0506, k2 0.0104 / 0.0113, fx 681.8 / 682.6
  (truth 680), rms 0.070 px vs 0.126 px pinhole-only. With 400 views fx stays +1.3 / +1.8 px while the formal σ
  shrinks (z up to 6), and with k1/k2/k3 held at truth fx is still 682.1 / 682.6: a systematic calibration bias,
  not fx-radial coupling and not the pixels (check A, through the GT poses, is exact) `[ASSUMPTION: fx-distance
  ambiguity of near-frontal planar views]` — so §6 judges B by tolerance, not σ. Stereo with the
  truth intrinsics: 0.094 mm, 0.0036°. Verdict PASS against `configs/sim/acceptance_s6.yaml` — circular: the
  thresholds came from this run; the evidence run on a fresh D41-rig recording is the first independent test.
  The same run with the board moved 5 mm in scene.json FAILS (A residual 2.1-2.3 px).
- **The D41 rig records cleanly through the warning gate** (8 s: no distortion warning; only the two
  allowlisted kinds).
- **D41-rig evidence recording, clean commit 0f7b0e4** (`evaluation/runs/gates/phase1_thin_e2e/sim_record/ci_tiny_0-20261001T133914Z`):
  strict PASS, RTF 0.243, no unallowed gz warning, 0 recorder errors, camera_info D = [0, 0, 0, 0, 0].
- **Renders are reproducible across recordings (on the frames §6 reads).** §6 on this recording reproduced the
  dev run on an earlier, independent recording of the same scene to every printed digit (e.g. left fx
  681.7801654, stereo T_x -0.0999075673): the ~1,644 calib-window frames per camera produced identical ChArUco
  detections. Not yet a whole-bag pixel comparison. Consequence: the evidence run is NOT an independent test of
  thresholds derived from the dev run (identical inputs); the independent test is another seed.
- **False dirty flag.** That §6 run recorded `git_dirty: true` at a clean commit: the recording step had copied
  its manifests into `evaluation/runs/`, which the dirty check counted. Run outputs are now excluded from the
  dirty check (`core.sh`, `worldgen.sh`, `sim_record.sh`) and from the Docker build context (`.dockerignore`,
  which also stops `core.sh image` from sending ~25 GB of raw evidence bags as build context). The run was
  discarded and is redone from the next commit.
- **§6 evidence runs, clean commit 5de3f70** (`evaluation/runs/gates/phase1_thin_e2e/sim_calib/`): seed 0 on the 1.3 evidence
  recording PASS (identical to the dev run); **ci_tiny seed 1, the independent test** (new scene 5d02793f…,
  recorded clean: strict PASS, RTF 0.213) PASS with margin on every check — A RMS <= 0.091 px (limit 0.25),
  |mean| <= 0.014 px (0.05), time offset ~6 µs (0.5 ms); B fx +0.19 / +0.22 % (1 %), k1 -0.0489 / -0.0511
  (±0.003); C 0.093 mm, 0.003°; D 0.26°, 1.3 mm, no flips. The free-calibration fx bias recurs with the same
  sign (+1.3 / +1.5 px) and, with the radial terms held at truth, is 680.8 here vs 682.1 on seed 0: it varies
  with the view geometry, consistent with the fx-distance `[ASSUMPTION]`.


## 4o. gsplat without a GPU (1.7): what runs on CPU `[SMOKE-TEST]` 2026-10-15

Context: no NVIDIA GPU on the dev box (Intel Iris Xe only); AWS G/VT quotas are 0 in eu-north-1, us-east-1,
us-west-2, eu-west-1 and eu-central-1, and every SageMaker g4dn/g5/g6 quota checked is 0 (re-requests filed
2026-10-15, `docs/architecture.md`). Throwaway image: python:3.12-slim + torch 2.5.1+cpu + gsplat 1.4.0
(`--no-deps`) + jaxtyping/rich/numpy/ninja.

- `import gsplat` works on CPU-only torch; `gsplat.cuda._torch_impl` exposes pure-torch `_quat_scale_to_covar_preci`,
  `_fully_fused_projection`, `_isect_tiles`, `_isect_offset_encode` — these ran on CPU (2000 Gaussians, 320x240).
- **`_torch_impl._rasterize_to_pixels` does NOT run on CPU:** it calls `_wrapper.rasterize_to_indices_in_range`, a
  CUDA kernel (`AttributeError: 'NoneType' object has no attribute 'rasterize_to_indices_in_range'`), and its
  compositing (`accumulate`) needs the separate `nerfacc` package. So gsplat cannot produce a pixel without CUDA.
- The forward compositing rules, transcribed from `cuda/csrc/rasterize_to_pixels_fwd.cu` (gsplat 1.4.0), for a CPU
  stand-in that must later match the CUDA kernel: per tile, Gaussians front to back by depth (the `_isect_tiles`
  order); pixel centre (x + 0.5, y + 0.5); σ = ½(a·dx² + c·dy²) + b·dx·dy with conic (a, b, c);
  α = min(0.999, opacity·exp(−σ)); skip if σ < 0 or α < 1/255; stop (exclusive) when T·(1 − α) ≤ 1e-4;
  colour += c·α·T; T ← T·(1 − α); background added as T·bg.
- Consequence (D44, decision 2026-10-15): no CPU rasterizer is written; the render runs with gsplat on
  CUDA off-AWS (Kaggle) until the AWS G/VT quota lands. These transcribed rules stay as the reference for any later
  parity check.
- **Remote CUDA path, Kaggle `[SMOKE-TEST]` PASS (2026-10-15):** `infra/scripts/kaggle.sh smoke` → kernel
  `suvojit42/ariadne-recon-smoke-gpu` (private): Python 3.13.15, torch 2.11.0+cu128, **Tesla T4**; gsplat 1.4.0 has
  no prebuilt wheel for pt211/cu128, so it installed from PyPI source and **JIT-compiled its CUDA extension on the
  first `rasterization` call: 767 s** for a 1000-Gaussian 320×240 render (the render itself is milliseconds).
  Every fresh kernel session pays this compile. Three failures before it, all environmental: (1) Kaggle CLI 2.x
  ignores `enable_gpu` unless the accelerator is named (`--accelerator NvidiaTeslaT4` / `machine_shape`); (2)–(3)
  an account without phone verification silently runs a GPU-requested kernel on the CPU image
  (`gcr.io/kaggle-images/python`, torch `+cpu`, no `nvidia-smi`, GPU quota used 0.00 h) with no API error.

## 4p. XFeat on OpenCV 5 primitives (D15): exact against upstream PyTorch, 12–33 ms `[SMOKE-TEST]` 2026-10-16

Gates Phase 2.1. Upstream `verlab/accelerated_features` at commit `e92685f57f8318b18725c5c8c0bd28c7fe188d9a`
(Apache-2.0), `weights/xfeat.pt` sha256 `0f5187fd…ea10b`. `python/model_tools/xfeat_reference.py`, run in the pinned
`ariadne-xfeat-ref:1` image (torch 2.5.1+cpu, 1 thread, deterministic algorithms) with `--network none` and the
upstream checkout mounted read-only, loads the weights with `weights_only=True` (no pickle code runs). It exports
them to an ATB tensor blob, and runs upstream's OWN `XFeatModel.forward` and `XFeat.detectAndCompute` (top-k 4096,
threshold 0.05) on a seeded synthetic 640×480 card and on one EuRoC MH_01 cam0 frame centre-cropped to 736×480.

- **The architecture is 25 convolutions** (BasicLayer = conv / BatchNorm(affine=False) / ReLU, so BN folds into the
  conv at load), a 4×4 average-pool skip, two bilinear upsamplings (`align_corners=False`), an 8×8 space-to-depth
  unfold into the keypoint head, InstanceNorm on the input, and a sigmoid. Nothing exotic.
- **Upstream quirks reproduced, not "fixed":** the raw 0..255 input (InstanceNorm makes the scale nearly
  irrelevant); sampling normalises positions by (W−1, H−1) but samples with `align_corners=False`, so a pixel x lands
  at x·W_map/(W−1) − 0.5, with zero-padded taps (the last column scores 0); a keypoint at exactly (0,0) gets score
  −1 (padding marker); `nearest` rounds half to even. Upstream's `argsort` is unstable; ours breaks ties by
  detection order.
- **Two engines, both checked** (`tests/unit/test_xfeat_reference.cpp`):

  | engine | 736×480 | 640×480 | max\|Δ\|/max\|ref\| (maps) | keypoints at the same pixel | min descriptor cosine |
  |---|---|---|---|---|---|
  | `opencv_dnn_graph`, 8 threads | 15.3 ms | 12.0 ms | ≤ 8.0e-6 | 4096 / 4096 (both inputs) | 1.000000 |
  | `opencv_dnn_graph`, 1 thread | 32.6 ms | 26.9 ms | (same) | (same) | (same) |
  | `gemm_reference`, 8 threads | 2873 ms | 2504 ms | ≤ 2.5e-6 | 4096 / 4096 | 1.000000 |

  Host: i5-1135G7. `opencv_dnn_graph` builds the network with the `cv::dnn` graph API (`Net::addLayer`/`connect`:
  Convolution, ReLU, Pooling, Eltwise, Resize with half-pixel bilinear, Sigmoid). It uses our folded weight tensors,
  **no ONNX file and no importer**. The unfold is an exact one-hot 8×8 stride-8 convolution. OpenCV 5's dispatched
  AVX2/AVX-512 convolution kernels run it. `gemm_reference` writes every op out (im2col + `cv::gemm`); without an
  external BLAS, `cv::gemm` runs at 0.88 GFLOP/s here, so it is the readable reference, not a runtime path.
- **Consequence:** D15 holds. Ours is the architecture and forward pass, and the weights are upstream's (credited).
  The catalogue's 16.7 ms figure is matched in the same class: 12–15 ms with 8 threads, 27–33 ms with one. That is
  keyframe-rate capable, and C8 (no neural net at frame rate) is unchanged. The engine that ran is a manifest field
  (§7.4).

## 4q. OpenCV 5.0.0 USAC: `findFundamentalMat` can throw on an empty model `[SMOKE-TEST]` 2026-10-16

Found by the first full EuRoC sweep. Earlier evidence covered MH_01 only. All six Vicon-room sequences aborted inside
the tracker's epipolar check (`klt_tracker.cpp`, the `UsacParams` overload of `cv::findFundamentalMat`):
`estimator.cpp:418: (-215:Assertion failed) !model.empty() in function 'setModelParameters'`. That assert is in
`SampsonErrorImpl`, the fundamental-matrix error. So the throw comes from F estimation, not PnP.

- **Reproduced standalone.** The input of V1_01_easy frame 1009 was dumped (285 correspondences, flow 10.8–26.3 px,
  so there is real parallax). It was replayed against the pinned build (tag `5.0.0`, commit `40738fb1`) with the
  tracker's parameters (threshold 1 px, confidence 0.99, 1000 iterations, single-threaded). It throws for **62 / 200**
  `randomGeneratorState` values. With `final_polisher = NONE_POLISHER` it still throws, and with
  `loMethod = LOCAL_OPTIM_NULL` it still throws (69 / 200). So the cause is neither the polisher nor local
  optimisation.
- **Likely path `[UNVERIFIED]`.** `FundamentalDegeneracy::recoverIfDegenerate` (`usac/degeneracy.cpp`) can return
  "degenerate, recovered" with a reset score. It does this without writing the model when neither calibrated
  DEGENSAC nor plane-and-parallax recovers one. V1/V2 are rooms with large planar walls and floor, where the
  degeneracy test fires. The exact sequence inside `ransac_solvers.cpp` was not traced to the line, and no upstream
  issue was found (searched 2026-10-16).
- **Handling (`klt_tracker.cpp`).** Only that throw is handled: `cv::Error::StsAssert` whose message contains
  `!model.empty()`. Any other exception propagates. On the throw, the check redraws `randomGeneratorState` from the
  same Philox stream `(seed, frame_id, site)`, up to 3 draws in total. If all three throw, the check is skipped for
  that frame. Both outcomes are counted: `TrackStats::usac_defect_retries` / `usac_defect_skipped`, and the
  `tracker_opencv_usac_defect` line in `results.yaml`, which is written only when non-zero. The first draw is the
  draw the tracker always made, so every frame that never threw is byte-identical, and the Phase 1 evidence stands.
  V1_01_easy now completes, with 1 redraw over 2912 frames.
- **Not done.** No change to OpenCV itself; the pin stays and the COOL baseline stays. An upstream report needs a
  go-ahead.

## 4r. EigenPlaces R50-256 through `cv::dnn` (ONNX): exact against upstream PyTorch `[SMOKE-TEST]` 2026-10-19

Gates 2.1 retrieval (`docs/design/perception.md` §11).

**Source and environment**
- Upstream: `gmberton/EigenPlaces` (MIT) at commit `a2969f71d5ea31017443af490b15273ca4c50af1`.
- Weights: `releases/download/v1.0/ResNet50_256_eigenplaces.pth`, sha256 `2923c102…dc9d7e7`. Downloaded once to the
  local cache; upstream publishes no checksum, so it is recorded on first download.
- `python/model_tools/eigenplaces_reference.py` runs in the reference image `ariadne-eigenplaces-ref:feat-2.1`
  (`ariadne-xfeat-ref:1` + torchvision 0.20.1+cpu + onnx 1.17.0; the Dockerfile is local, an open ask
  to move it into `infra/docker`).
- Settings: `--network none`, upstream and weights mounted read-only, torch 1 thread, deterministic algorithms.
- Upstream's constructor fetches CosPlace weights over `torch.hub` only to initialise the backbone. That one call is
  stubbed, because `load_state_dict(strict=True)` replaces those values entirely.

**What is exported**
- The whole model and the backbone up to `layer4`, at FIXED sizes 752×480 (EuRoC native, upstream's full-resolution
  test protocol) and 376×240 (`INTER_AREA`), opset 17.
- Head weights (GeM p, fc) as ATB, the fallback if the head failed to import (it did not).
- Reference inputs: the exact preprocessed tensors (gray/255 → resize → RGB replicate → ImageNet normalisation) of
  EuRoC MH_01 frame 200 and V1_01 frame 500, with upstream's descriptors.

**Result** (`tests/unit/test_eigenplaces_reference.cpp`; `ARIADNE_EIGENPLACES_REF` = the export directory)

| engine | 752×480 cosine (both frames) | 376×240 cosine (both frames) |
|---|---|---|
| `readNetFromONNX` `ENGINE_NEW` | 1.0000000 | 1.0000000 |
| `readNetFromONNX` `ENGINE_CLASSIC` | 1.0000000 | 1.0000000 |

- Both engines import and run the graph, head included, with |d| = 1.000000. No fallback is needed.
- **Forward cost (indicative only, host load 23 from other jobs):**

  | engine | 752×480 | 376×240 |
  |---|---|---|
  | ENGINE_NEW | 1.0–2.7 s | 0.25–0.5 s |
  | ENGINE_CLASSIC | 1.4–2.2 s | 0.2–0.38 s |

- **Consequence:** ResNet-50 cannot run on every keyframe on this CPU. Retrieval descriptors are computed on a
  spaced subset (≥ 1 s apart) at 376×240. A quiet-host timing replaces these figures before any budget claim.
- ONNX Runtime is absent from the build (`ENGINE_ORT` unavailable), so `cv::dnn` is the only engine, as §7.4 plans.

## 4s. ALIKED + LightGlue through OpenCV 5.0.0's `cv::ALIKED` / `cv::LightGlueMatcher` `[SMOKE-TEST]` 2026-10-20

- **Models.** The pinned build ships no weights. OpenCV's own test (`modules/features/test/test_aliked_lightglue.cpp`)
  loads two opencv_extra testdata files, `aliked-n16rot-top1k-640.onnx` (sha1 `41faa7bf…`, sha256 `b46868de…`) and
  `aliked_lightglue.onnx` (sha1 `02723aa5…`, sha256 `a1bd99d2…`). `download_models.py` fetches them from a personal
  GitHub repo (`YangGuanyuhan/lightglue_opencv_project`, HEAD `9bf886bf`, **no licence file**, export script not
  published). Upstream weights: ALIKED BSD-3 (Shiaoming/ALIKED), LightGlue Apache-2.0 (cvg/LightGlue `eb42fee`).
  Provenance of the export is therefore unverified. Downloaded with matching sha1 to
  `.cache/feat-2.1-perception/models/opencv_extra_aliked/`.
- **Load** (`readNetFromONNX`, our image, no ORT):

  | engine | ALIKED | LightGlue |
  |---|---|---|
  | ENGINE_NEW | ok (41 ms) | ok (453 ms) |
  | ENGINE_CLASSIC | **fails**: "Multidimensional unsqueeze" (`/desc_head/Unsqueeze_1`) | **fails**: Concat shape (`transformers.0/self_attn/Concat_2`) |

  The model author's compatibility table (ENGINE_NEW "unsupported" for `aliked_lightglue.onnx`) is stale for 5.0.0.
- **Run** (MH_01 cam0 frames 200 and 205, BGR, ENGINE_NEW, shared CPU): ALIKED 1000/1000 keypoints (fixed top-1k),
  128-D, **786 ms/image** at 640×640; LightGlue **644 matches** at score ≥ 0.1, **977 ms** for 1000×1000.
- **Wrapper behaviour, read from the pinned source** (`feature2d_aliked.cpp`, `matchers_lightglue.cpp`):
  - `ALIKED` ignores `mask` and `useProvidedKeypoints`. It resizes to `inputSize` without keeping the aspect ratio
    (752×480 is stretched to 640×640) and maps normalised keypoints to pixels as (k+1)/2·W. Upstream
    (`aliked.py`) uses (k+1)/2·(W−1): up to 1 px at the far edge.
  - `LightGlueMatcher` normalises pixel keypoints per axis (x/W·2−1, y/H·2−1). Upstream `normalize_keypoints` keeps
    the aspect ratio (shift W/2, H/2; scale max(W,H)/2). At 752×480 the matcher's positional encoding sees y
    stretched 1.57×. Passing pre-normalised keypoints with an empty `Size()` bypasses it (public API).
  - The header promises "automatic context from in-process ALIKED". The implementation never reads it:
    `setPairInfo` is mandatory.
  - `clone()` shares the `dnn::Net`, so it is **not** a per-thread copy (§7.4: create one matcher per thread).
  - `knnMatch` supports k = 1 only; `radiusMatch` throws.
- **The graphs** (`onnx` 1.17, both opset 17, producer PyTorch 2.1.0):
  - **ALIKED:** input FIXED at [1,3,640,640]; outputs `keypoints` [1000,2] (a fixed top-1k), `descriptors`
    [N,128], `scores` [1000]; 398 nodes. There is no DeformConv: the deformable conv is decomposed by hand
    (GatherND ×16, GridSample ×2, Floor). No control flow. `Params.inputSize` 736×480 or 752×480 fails at load
    (Reshape assertion), so **native aspect needs a re-export**.
  - **LightGlue:** dynamic N (`kpts0` [1,N,2], `desc0` [1,N,128]); outputs `matches0` [M,2] int64 and `mscores0`
    [M]; 2341 nodes, 36 LayerNorm (= 9 layers, all run), and no `If`/`Loop`, so **adaptive depth/width is not in
    the graph**. Match filtering is in-graph (NonZero/Where; the threshold value is not yet read).
- **Not yet verified:** equality with upstream PyTorch (keypoints, descriptors, matches) on identical 640×640
  inputs.
- **Decision (2026-10-20): export our own from upstream** (option b), not the opencv_extra files.

### 4s-2. Our export: ALIKED-n16 + LightGlue from upstream, exact through cv::ALIKED / cv::LightGlueMatcher `[SMOKE-TEST]` 2026-10-21

- **Sources.**
  - `cvg/LightGlue` at `eb42fee` (Apache-2.0; its `aliked.py` is ALIKED, BSD-3).
  - Weights: `Shiaoming/ALIKED@683d7c6` `models/aliked-n16.pth` (sha256 `5be87048…`), and LightGlue
    `v0.1_arxiv` `aliked_lightglue.pth` (sha256 `d975e965…`).
  - Configuration from glue-factory `configs/aliked+lightglue-official.yaml`: ALIKED-**n16**, which is what
    LightGlue's ALIKED weights were trained on (`ALIKED-k2048-n16`). OpenCV's file is n16rot. Top-K
    (`detection_threshold` 0); LightGlue `depth_confidence` = `width_confidence` = −1; `filter_threshold` 0.1.
  - Native 752×480 (upstream pads to a multiple of 32 by edge replication), no resize.
- **Tool:** `python/model_tools/aliked_lightglue_reference.py`, reference image `ariadne-eigenplaces-ref:feat-2.1`,
  `--network none`, kornia stubbed (raises if called; only resize/gray paths use it). Outputs in
  `.cache/feat-2.1-perception/models/aliked_lg_ref/`: `aliked_n16_k{1024,2048}_752x480.onnx`,
  `lightglue_aliked.onnx` (dynamic N), references and a manifest with every sha256.
- **Rewrites for cv::dnn**, each checked in torch against UNMODIFIED upstream before export:
  - **DCN v1** (no DeformConv importer): written as a sampling grid, one GridSample (zeros, align_corners), then a
    MatMul. Max |diff| 7.6e-6 alone; dense maps 1.5e-6.
  - **DKD top-K:** Unfold becomes gathers; the border is a constant mask. Exact (0) on the same score map.
  - **SDDH:** the gather plus one MatMul; 8.9e-7.
  - **LightGlue:** `unflatten`/`repeat_interleave`/einsum respelled as explicit reshapes and MatMuls (the traced
    rank was lost with a dynamic N). Matches are identical; score difference 0.
  - **Static dims frozen with `int()`.** Otherwise traced Shape→Gather int64 arithmetic hits ENGINE_NEW's
    "all inputs should have equal types" in an elementwise layer.
  - **SDDH patch corner computed with Floor, then cast.** With the truncating `.long()` spelling, cv::dnn gave a
    different patch on ~76–79% of keypoints, as if fractional parts ≥ 0.5 rounded up (cosine ≥ 0.976, max element
    0.05). The pinned `cast2_layer.cpp` CPU path truncates correctly, so the location of the rounding is not pinned
    down `[UNVERIFIED]`. After the change: exact.
  - Keypoints are emitted in cv::ALIKED's convention, n = 2·px/W − 1, so its (n+1)/2·W mapping returns upstream's
    (k+1)/2·(W−1) pixels.
- **Keypoints are compared as sets, not index order.** A ~1e-6 change of the score map (any engine's float error)
  flips near-tied NMS maxima (`score == maxpool(score)`) and the top-K order. Each upstream keypoint is paired with
  the nearest exported one within 1e-3 px.
- **Result** (`tests/unit/test_aliked_lightglue_reference.cpp`, `ARIADNE_REQUIRE_MODELS=1`, 2/2 PASS; ENGINE_NEW,
  shared CPU):

  | | K = 1024 | K = 2048 |
  |---|---|---|
  | ALIKED input tensor | identical (0) | identical (0) |
  | ALIKED keypoints matched (≤ 1e-3 px) | 100%, max 9.2e-5 px | 100%, max 9.3e-5 px |
  | ALIKED scores / descriptors (max element) | 1.6e-5 / 4.0e-6 | 1.9e-5 / 4.0e-6 |
  | ALIKED time per image | 0.48–0.59 s | 0.52–0.60 s |
  | LightGlue match sets (MH_01 0.25 s apart; V1_01 revisit 102 s apart) | 679 / 551, identical | 1314 / 1159, identical |
  | LightGlue scores | ≤ 6.2e-5 | ≤ 6.4e-5 |
  | LightGlue time per pair | 0.59–0.65 s | 1.86–1.97 s |

- **For information** (upstream with adaptive depth/width ON, its defaults): it stops at layer 5–7 and keeps 92–99%
  of the full model's matches (V1_01 k1024: 540 matches, 505 shared with the full 551). The export runs all 9
  layers (the official evaluation config).
- **Usage contract:** `cv::ALIKED::Params.inputSize` = the export's size (752×480); `cv::LightGlueMatcher`
  `setPairInfo(k0n, k1n, Size(), Size())` with keypoints normalised as upstream (shift W/2, H/2; scale max(W,H)/2).
  One matcher per thread (`clone()` shares the Net).
- **Export sha256:**
  - `aliked_n16_k1024_752x480.onnx` `cc5bf19d…`
  - `aliked_n16_k2048_752x480.onnx` `ee9bc548…`
  - `lightglue_aliked.onnx` `51906eca…`

  The full hashes are in the export's `manifest.yaml`.

## 4t. Sim-rig exports (1280×720 RGB): EigenPlaces and ALIKED-n16 + LightGlue, exact through cv::dnn `[SMOKE-TEST]` 2026-10-22

**Input:** four `left_rgb` frames of the phase-1 `ci_tiny` recording (agent 0, frames 200/205 at 10 s and 700/710 at
35 s, bag `sha256:21399732…`), written by `ariadne_frame_dump` exactly as the pipeline sees them: rgb8 decoded,
D41 lens applied, no degradation. They are 1280×720. Their colour is weak (mean |R − B| ≈ 7/255).

**Exports:** `models.sh eigenplaces-sim` (1280×720 and 640×360) and `models.sh aliked-lg-sim` (K = 1024 at
1280×720). Inputs are RGB order straight from the frame; nothing is converted from gray. A second regeneration was
bit-exact: 11/11 and 12/12 files. `lightglue_aliked.onnx` is byte-identical to the EuRoC export.

| check | result |
|---|---|
| ALIKED export rewrite vs upstream (Python, set-based) | matched 1.0 on all 4 frames; max 1.8e-4 px, desc 8.6e-6, score 2.4e-5 |
| LightGlue rewrite vs upstream | identical match sets: 946 and 949 |
| `cv::ALIKED` vs upstream (`test_opencv`) | input diff 0; matched 1024/1024; px ≤ 1.8e-4; desc ≤ 8.9e-6; no truncation flips |
| `cv::LightGlueMatcher` vs upstream | 946/946 and 949/949 shared; score ≤ 4.9e-5 |
| EigenPlaces, both engines, both sizes | cosine 1.0000000 (8/8) |
| `PlaceRecognizer::DescribeRgb` | preprocessing ≤ 7.2e-7; cosine 1.0000000. BGR-swapped: input moves 0.26–0.77, cosine 0.996–0.998 |
| `HardPairMatcher::ExtractRgb` | byte-equal to `cv::ALIKED` on the BGR file. BGR-swapped: 0/1024 keypoints kept exactly. The wrapper reproduces upstream's sim match sets |

**Cost:** indicative only, shared host (load 2–5). EigenPlaces classic is 1.9 s (1280×720) and 0.46 s (640×360) on
1 thread, 0.76 s and 0.17–0.23 s on 8. ALIKED at 1280×720 is 1.6–2.4 s per frame and LightGlue about 0.95 s per pair,
both at the default pool. Quiet-host numbers are L5.

## 4w. Lockstep through `ros_gz_bridge`: `run_to_sim_time` stops gz exactly at T, with the frame at T `[SMOKE-TEST]` 2026-10-23

Gates D165 (live-runtime.md). Sim image `ariadne/sim:dev` (ROS 2 Jazzy, `ros_gz_bridge` 1.0.24, gz-sim8, gz-msgs10),
Mesa llvmpipe, scene `ci_tiny_0` (`mission_live.sdf`), world started paused (`gz sim -s --headless-rendering`).
Bridge YAML: `/clock` and `/ariadne/agent_0/left/image_raw` gz→ROS, plus one service entry
`{service_name: /world/<w>/control, ros_type_name: ros_gz_interfaces/srv/ControlWorld, gz_req_type_name:
gz.msgs.WorldControl, gz_rep_type_name: gz.msgs.Boolean}`; the bridge logs "Creating ROS->GZ service bridge".
A probe node (rclpy) calls the service with only `world_control.run_to_sim_time` set, then watches `/clock` until it
has not moved for 15 s of wall time.

| request | service result | `/clock` stopped at | frames received | frame stamped AT T | frames after T |
|---|---|---|---|---|---|
| run_to 2 s | true | 2 000 000 000 ns | 40 | yes | 0 |
| run_to 4 s | true | 4 000 000 000 ns | 80 | yes | 0 |

- The bridge copies `run_to_sim_time` (a plain `pause: false` would have run free): gz pauses itself at T exactly.
- The 20 Hz camera's frame at T is published before the pause, so a barrier at T has its frame.
- A world that STARTS paused publishes no frame at t = 0 (first frame 0.05 s): lockstep runs have one frame fewer per
  agent than unpaused ones (1800 vs 1801 over 90 s).
- Systems' `PreUpdate` while paused (the mover taking a command at paused time T): covered by
  `python/tests/sim/test_mover_live.py::test_command_taken_while_paused` (applied_ns = 0 in a world paused at 0).
Re-run (~2 min): `infra/scripts/tests/lockstep_smoke/run.sh` (header has the docker command; probe.py beside it).

## 4u. HashTSDF (`cv::Volume`) for the §20 volumetric stack: usable as the oracle, not as the planner's map `[SMOKE-TEST]` 2026-10-22

§4b showed `cv::Volume(VolumeType::HashTSDF)` constructs. 4.1 (arch §20.1) needs more than that, so this checks what
the planner would read from it. Reproduce with `infra/scripts/smoke_hashtsdf.cpp` (command in its header).

**The public API** (`opencv2/ptcloud/volume.hpp`, `volume_settings.hpp`, `detail/`): `integrate(depth, pose)`
(also with colour or an `OdometryFrame`), `raycast`, `fetchNormals`, `fetchPointsNormals[Colors]`, `getBoundingBox`,
`getVisibleBlocks`, `getTotalVolumeUnits`, `reset`, `setEnableGrowth`. It does **not** have:

- a read of a voxel's TSDF value or weight, or any iteration over blocks or voxels;
- a per-pixel sample weight: every pixel integrates with weight 1, capped by `maxWeight`.

§20.2's tri-state needs the per-voxel weight (unknown = weight 0). §20.3's ESDF and §20.4's frontiers need the
voxels. §20.1's sample weights are `1/σ_Z²`. None of these is expressible through this API: this is §20.1's "API
proves awkward" case. §4b's note "no from-scratch TSDF needed" is superseded.

**Synthetic check** (known answer: a fronto-parallel wall at Z = 2 m, 640×360 pinhole, 5 identical integrations,
truncation 4 voxels, raycast from the same pose):

| voxel | raycast valid | raycast \|ΔZ\| median / max | `fetchPointsNormals` points, mean \|ΔZ\| |
|---|---|---|---|
| 0.05 m | 230317 / 230400 | 0.020 / 0.050 m | 21573, 0.087 m |
| 0.08 m | 229694 / 230400 | 0.042 / 0.160 m | 8672, 0.141 m |

The raycast surface sits about half a voxel off the true wall. The fetched points spread further, by 1.7 voxels on
average. This is `[UNVERIFIED]` as to cause; one candidate is a voxel-corner vs voxel-centre convention.

**So:** 4.1 builds the from-scratch voxel-hash TSDF of §20.1's fallback, and `cv::Volume` HashTSDF stays as its
correctness oracle: same frames, same voxel size and truncation, unit weights, compared on the raycast surface at a
tolerance of one voxel. The oracle's own half-voxel bias, measured here, is why the tolerance is not tighter.

## 4v. NVIDIA EGL headless render path on AWS g4dn.xlarge `[SMOKE-TEST]` 2026-10-23

**Gate:** does Gazebo Harmonic render camera frames headless through the NVIDIA EGL path (not a silent llvmpipe
fallback) on the GPU instance class the arch names for rendering (§5.10)?

**Run:**
- infra/scripts/sim.sh acceptance (`sim_smoke.py`), as step 3 of the D112 render job, on one g4dn.xlarge spot
  instance in eu-north-1a;
- driver 595.91.07, CUDA 13.2; the ariadne/sim:dev image by id `sha256:8b780f52…`;
- evidence in `evaluation/runs/mapping/risk_calibration/aws_run/egl_smoke/`.

**Result: PASS.**
- `[sim_smoke] render mode: nvidia-egl`, then `SIM_SMOKE=PASS (3 good frames, render mode nvidia-egl)`.
- Frames are 320×240 rgb8, luma std 73.9, with the scene's colour structure (red 0.318, blue 0.195).
- The job then rendered three full ci_tiny_hazard missions on the same path: wall 2188 / 1323 / 469 s for seeds
  0 / 1 / 2, about 200 s of sim each.
  - The spread is not explained here; seed 0 also paid the first-render shader warm-up. `[UNVERIFIED]`

**Notes:**
- The `wall=` field in sim_smoke's timing lines is `time.monotonic()`, the host's monotonic clock, not time since
  start: 192.9 s on frame 1 is uptime, not a slow frame.
- Two failed launches before this one were both the job script's: an S3 prefix outside the instance role's
  `runs/*`, then output dirs owned by root while the containers run as uid 1000. Neither was the render path.
- **Unblocks:** the live runs on g4dn (L1 `live_aws.sh`). The 4.3 / 4.6 lockstep gate on g4dn needs an EGL
  lockstep check first: lockstep is verified on llvmpipe only.
- 2026-10-24: the same gate re-confirmed through `live_aws.sh`'s own smoke (`cmd_smoke`, run
  `20261024T141057Z-gpu-6ced25`, g4dn.xlarge spot, image `ariadne-sim:live-8b780f52cb9a`): exit 0,
  `sim_smoke_exit=0`, `smoke_pass.png` 320×240 RGB written. The first attempt (d986bf) had already rendered but
  died writing `/out` under a root-owned results dir — the runner bug fixed in 65caaf4 (chmod before render), not
  the render path. Evidence: `build/runs/20261024T141057Z-gpu-6ced25/output/` + the run's S3 output.
## 4u. Lamp photometry of the gz-sim8 / ogre2 camera (Phase 5 preflight) `[SMOKE-TEST]` 2026-10-22

`simulator/smoke/lamp_probe/run_lamp_probe.py` (ariadne/sim:dev, llvmpipe; 17 launches plus a 2-launch rerun, every
`gz_rc` 0). Each world: ambient 0, no light but a spot lamp 5 cm forward and 4 cm up of the camera, a matte grey plane
(diffuse 0.5, roughness 1) facing the camera, and an R8G8B8 and an L16 camera at the same pose. Raw frames are in
`.cache/feat-2.2-radiometric/runs/lamp_probe/` (`lamp_probe.json`, `lamp_probe_rerun.json`, `*_c8.npy`, `*_c16.npy`).

| Question | Measured | Consequence |
|---|---|---|
| Q1 linearity | R8G8B8 is **sRGB-encoded**: doubling lamp intensity at 1.5 m gives 49.6 → 71 → 99 → 137 (×1.43, ×1.39, ×1.38; 2^(1/2.2) = 1.37). L16 is linear (2061 → 4129 → 8177 → 16394, ×2.00) and equals sRGB-decode(R8) to 1e-5 | D's input is not linear radiance: D130 adds an sRGB decode |
| Q3 headroom | L16 has exactly as many distinct values as R8 in every frame: it is the same 8-bit render, decoded. No extra precision exists | Keep 8-bit RGB recordings; sRGB is a compander (~12 bits of linear range at the dark end), enough for the lamp's ~400× falloff |
| Q2 falloff | Centre pixel (lamp axis), linear: I = A / (c′ + r²) with r from the lamp. SDF attenuation (0, 0, 1): A = 0.327, **c′ ≈ 0.5 m²** (residuals < 1 % from 0.75 to 4 m). The rig's current (0.2, 0, 0.05): **c′ ≈ 10 m²**, nearly flat to 3 m | Not inverse-square as the rig ships; D131 sets (0, 0, 1) and calibrates c′ |
| Lobe | Divided by 1/(c′ + r²) and **one** cos (Lambertian incidence): flat to ±0.6 % inside the inner half-angle, smooth roll-off to 0 at the outer half-angle; dividing by cos² over-corrects by 7 % at 22°. SDF `inner_angle`/`outer_angle` are **full** cone angles (25°/40° lit only to a 12.5°/20° half-angle) | The renderer's lobe is a flat top, not cos^μ; calibrate the edge as a table. The rig lamp lit only the centre of an ~86° FOV |
| Q4 shadows | Rerun `shadow2`, row 360: lit plane 57–76, **hard shadow 0** for ~112 px, occluder face 153–155 | ogre2 spot shadows work on llvmpipe; with ambient 0 a shadow is exactly black |

The first `shadow` case placed the occluder just outside the 20° half-angle cone, so it could not show a shadow. It is
kept in the JSON and superseded by `shadow2`.
