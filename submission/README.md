# Ariadne: submission package

This folder maps every deliverable required by the OpenCV AI Competition 2026 (powered by AWS) to a file or link.

## Required deliverables

| requirement | where |
|---|---|
| Technical report: problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation, limitations, responsible use | [`report/ariadne_technical_report.pdf`](report/ariadne_technical_report.pdf) (LaTeX source: `docs/report/main.tex`) |
| Judge-accessible code repository | https://github.com/CynicPixel/Ariadne (this repository) |
| Pinned dependencies; build, deployment and test instructions | [`docs/build.md`](../docs/build.md) and the [`ariadne`](../ariadne) command (`./ariadne build`, `test`, `gate`, `run`, `replay`, `demo`, `cloud …`) |
| Architecture diagram: OpenCV 5, AWS, agent components | [`architecture/ariadne_architecture.png`](architecture/ariadne_architecture.png); detailed system diagram [`architecture/ariadne_system_diagram.pdf`](architecture/ariadne_system_diagram.pdf) |
| Working web endpoint or arranged live screen share | https://www.ariadne.cynicpixel.xyz/ (operator console). A live screen-share demonstration is available on request |
| Video, at most 5 minutes: team, application working, architecture, principal results | [`video/`](video/) |
| Evaluation evidence, including failure cases and limitations | [`docs/results/index.md`](../docs/results/index.md) (best result per subsystem, with the do-not-claim list); [`docs/failure-cases.md`](../docs/failure-cases.md); every AWS run in [`runs/`](runs/); run directories under [`evaluation/runs/`](../evaluation/runs/) |

## Best Use of COOL

| requirement | where |
|---|---|
| COOL version and AWS instance or deployment configuration | [`evidence/cool.md`](evidence/cool.md) §Version and AWS configuration ([`report/cool_evidence.pdf`](report/cool_evidence.pdf)) |
| Reproducible evaluation method, inputs, baselines and results | `evidence/cool.md` §Evaluation method and §Graviton4 results; protocol in `infra/cool/day1.sh` and `infra/cool/probe.cpp`; Graviton runs in `evaluation/runs/arm64/` |
| Evidence that COOL executes the claimed core workload | `evidence/cool.md` §Execution identity: one binary, a loader switch, and a manifest that records the library actually loaded (`/proc/self/maps` plus the build-information digest); `./ariadne cloud cool` runs the protocol |

## Agentic Vision

| requirement | where |
|---|---|
| Agent workflow diagram: perception, decision or orchestration, action | [`evidence/agentic-vision.md`](evidence/agentic-vision.md) §Workflow ([`report/agentic_vision_evidence.pdf`](report/agentic_vision_evidence.pdf), figure 1) |
| Trace showing that OpenCV 5 output changes a later decision, tool call or action | `evidence/agentic-vision.md` §OpenCV output changing the next action: the final integrated run's `trace.jsonl`, `decisions/`, `mover_state.jsonl` in `evaluation/runs/live-agentic/final_stack_e2e/` |
| Evaluation of task success, failure handling, observability and human control | `evidence/agentic-vision.md` §Runs and §Failure handling, observability, human control; chooser pairs in `evaluation/runs/live-agentic/` |

## Also here

- **`media/`:** GIFs of each algorithm, and stills taken from recorded runs. Each item's source is listed in `media/README.md`.
- **`runs/`:** `RUNS.md` lists the runs behind each headline result, and `runs.csv` lists all 94 AWS runs (cost, outcome, evidence, S3).
