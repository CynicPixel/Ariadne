# Ariadne: Best Use of COOL evidence

The typeset version is `docs/report/COOL_EVIDENCE.pdf`, built from `COOL_EVIDENCE.tex`.

## Workload
Each stereo agent gets a frame every 50 ms of simulated time. Every frame goes through rectification (`cv::remap`), pyramid construction and `cv::calcOpticalFlowPyrLK`, then colour conversion and resize, then stereo. The cost scales with the number of agents.

That per-frame OpenCV path is the COOL target. Two things sit outside it:
- **Restoration:** flat-field and VST are labelled auxiliary configurations.
- **Keyframe work:** keyframe DNN calls (EigenPlaces, ALIKED, LightGlue) are timed separately.

The ablations did not change this choice of workload. ORB + LK still carries the high-rate geometry, because XFeat regressed EuRoC VO on 6 of 7 flights.

## Version and AWS configuration

| item | value | source |
|---|---|---|
| COOL | AWS Marketplace *Cloud Optimized OpenCV for Graviton4*, COOL 3.1 (OpenCV 5.x, KleidiCV + ArmPL, Ubuntu 24.04), SDK `/opt/cool/cpp_sdk` | https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs |
| AMI | `Graviton5-COOL-v3-prod-k6u24vxijyzvg`, eu-north-1 `ami-0a302ff85af8d8764` (18 regions) | `infra/cool/cool-regions.tsv`, DECISIONS D28 |
| instance | c8g.4xlarge (Graviton4, Neoverse-V2, 16 vCPU) spot, eu-north-1; m8g.4xlarge is the design's throughput instance | `infra/scripts/spinup.sh` (`cool` profile), arch §26.3 |
| OpenCV pin | 5.0.0 at opencv `40738fb16ced`, contrib `755e50675d97`; C++17, Eigen 3.4.1, gcc 13.3, `-O3 -ffp-contract=off` | `infra/docker/opencv-base.Dockerfile` |
| stock Arm HAL | carotene + KleidiCV 26.03, already on in the stock build; dispatch `NEON FP16 *NEON_DOTPROD *NEON_FP16 *NEON_BF16` | `evaluation/runs/arm64/20261023T193621Z-cpuarm-7f8c96/` |

## Execution identity: one binary, loader switch only
- **No hidden library paths:** `ARIADNE_REQUIRE_OPENCV=ON` and `CMAKE_SKIP_RPATH=ON`. A `readelf` gate refuses any RPATH or RUNPATH (`infra/docker/core.Dockerfile`).
- **The manifest records what actually loaded:** the OpenCV version, `cv::getCPUFeaturesLine()`, and an FNV-1a digest of `cv::getBuildInformation()` taken from the loaded `libopencv_core`. `/proc/self/maps` lists the shared objects that were mapped.
- **Switching libraries:** `LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib` selects COOL, and `/opt/stock/lib` selects stock. To build directly against COOL, use `-DOpenCV_DIR=/opt/cool/cpp_sdk/lib/cmake/opencv4`.

## Evaluation method
```
infra/scripts/spinup.sh run --profile cool --type c8g.4xlarge --max-minutes 90 -- 'bash infra/cool/day1.sh'
```
`infra/cool/day1.sh` and `probe.cpp` run four steps. Each step writes its evidence as soon as it finishes.
1. Inspect `/opt/cool`: sonames, the CMake package, COOL's build information and its modules.
2. Build stock OpenCV 5.0.0 + contrib natively, at the pinned commits and with the project's module list and flags, into `/opt/stock`.
3. Run the loader-switch probe: build one binary, run it on both libraries, and check `/proc/self/maps`.
4. Record output checksums and timings for each operation, at 1 thread and at all cores, on a 1280×720 frame. The operations are `cvtColor`, `remap`, `resize` (area), `GaussianBlur` 5×5, `pyrDown`, `buildOpticalFlowPyramid`, `calcOpticalFlowPyrLK` (window 21, 3 levels), `Sobel` and Otsu `threshold`.

**Whole-pipeline stage:** replay one recorded stereo MCAP through the same binary on both libraries. Report p50/p95 latency, FPS, CPU utilisation, the trajectory digest and pose tolerances.

**Cost per frame:** C = (instance $/h + software $/h) / (3600 · FPS).

## Graviton4 results (stock OpenCV 5.0.0, measured)

| check | result | evidence |
|---|---|---|
| native arm64 build + suite | **708/708** C++ tests, image gate PASS, commits match the Dockerfile; 901 s, ~$0.02 | `evaluation/runs/arm64/20261024T100530Z-cpuarm-602802/` |
| determinism | x86 and Graviton replay digests are identical | `docs/validation.md` §4e |
| wire bytes | arm64 FlatBuffers bytes made identical to x86 goldens | commit `6a16717`, arm64 run `05b8cc` |
| learned kernels | ALIKED 1280×720 takes **530 ms on c8g vs 829 ms on c7i** (0.64×); LightGlue and EigenPlaces run at 0.85–0.88× | `evaluation/runs/performance/pipeline_profile_20261024/` |
| thread invariance | estimator output identical at 2, 5 and 16 threads; 3-agent replay RTF 0.836 (c7i.4xlarge) | main report |

## Published COOL figures (vendor)
- **End to end:** AWS measured an image-analysis workflow on Graviton4 at 1.6 fps with COOL against 1.2 fps with pip OpenCV 4.12, about 30 % faster ([AWS blog](https://aws.amazon.com/blogs/spatial/accelerating-opencv-on-graviton-the-cool-framework)).
- **Per operation:** up to 70 % faster across 78 functions ([opencv.org/cool](https://opencv.org/cool/)).
- **Ariadne's baseline is stronger:** it is OpenCV 5.0.0 with KleidiCV already on, not pip OpenCV 4.12.

## How a judge verifies it
Subscribe to the AMI, run the command above, and compare the two manifests:
- the loaded-library list plus the build digest show stock versus COOL;
- the input digest shows both runs saw identical frames;
- the tables give the speedup at matched outputs.
