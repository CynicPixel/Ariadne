# Plan 0.2 arm64 arm: first native Graviton run (c8g.4xlarge spot, about $0.013)

- OpenCV 5.0.0 + contrib (the pinned tag) built natively in 244 s; smoke test PASS.
- Stock arm64 build: Custom HAL = carotene + KleidiCV 26.03, ON by default. This matters for the COOL Arm-vs-Arm
  comparison: "stock" already includes KleidiCV.
- ariadne-core dev image, Release build, full ctest: 595/596 pass, 64 skipped (model exports not shipped to this job).
- **FAILED: GravityPgo.RecoversTheReferenceGravityErrorAndReducesTilt.** On the same synthetic chain (odometry tilt RMS
  3.78 deg on both), x86 recovers gravity to 0.62 deg (tilt 2.01 deg) but arm64 gets 2.71 deg (tilt 3.17 deg), against
  the test's 1.0 deg bound. The pose-graph solve diverges across architectures. Not yet diagnosed [OPEN]: candidates are
  vectorized reductions (Eigen NEON vs SSE/AVX) or OpenCV's linear solve via the HAL.
