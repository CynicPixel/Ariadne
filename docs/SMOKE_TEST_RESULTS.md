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
