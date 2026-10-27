# arm64 full ctest of the consolidated tree (c8g.4xlarge, Graviton4)

- **Run:** `20261024T100530Z-cpuarm-602802`, M's slot-queue job 17. c8g.4xlarge spot, eu-north-1b, $0.083/h, 901 s, compute ~$0.0208.
- **Tree:** consolidate/2026-10-24 at `ad99edf`.
- **OpenCV:** 5.0.0, opencv `40738fb16ceddb5fb3fea747585f7ce6abb0605b`, contrib `755e50675d97db9b7d449d8bd6b09888646f6c6e`. `opencv_commits_match_dockerfile: true`.
- **Result:** `100% tests passed, 0 tests failed out of 708`, `core_image_gate: PASS`.
- **Images built:** `ariadne/opencv-base:5.0.0-arm64`, `ariadne/core-dev:arm64`, `ariadne/core:arm64` (533 MB tarball, S3 only).

The source is `slotq_job.log`, the job's local log, copied from `build/slotq/done/17-M-arm64-main.log`. The full results tarball is in S3 under `runs/20261024T100530Z-cpuarm-602802/`.

This is stock OpenCV 5.0.0 on Graviton4. COOL is not involved. The COOL protocol is `infra/cool/day1.sh`.
