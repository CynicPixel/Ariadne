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

## 5. Open items

| Item | Status |
|---|---|
| Built-image verification (amd64) | **DONE** — §4b |
| `cv::getBuildInformation()` hash into manifest | hash captured (`9127192543017384188`); **manifest writer itself is 0.5, not yet written** |
| DNN engine runs each ONNX model, or ORT fallback (§7.4) | `[PENDING]` — needs models |
| **arm64 build + full smoke test** | `[PENDING]` — no local arm64; deferred to Graviton / Arm CI runner |
| PyPI wheel status for OpenCV 5 | `[PENDING]` — not needed; we build from tag per C7 |
| COOL day-1 trial check (0.3) | `[PENDING]` — needs AWS account |
