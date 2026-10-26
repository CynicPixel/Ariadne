# AWS runs behind the headline results

- **All runs:** `runs.csv` lists all 94 AWS runs, with instance, duration, spot cost, exit code, committed evidence and S3 location. Regenerate it with `python3 infra/scripts/runs_index.py`.
- **Total compute:** about **$4.8** of spot time. EBS, IPv4 and S3 are excluded; Bedrock tokens are billed separately.
- **S3 layout:** every run's inputs (repository snapshot, payload) and full outputs are in `s3://ariadne-836505524290-eun1/runs/<id>/`.

| result | AWS run | instance, time, cost | committed evidence |
|---|---|---|---|
| Pre-merge gate: 54/54 checks, 708/708 C++ tests, two-agent mission vs ground truth | `20261024T083919Z-cpux86-73e3fa` | c7i.4xlarge, 999 s, $0.035 | `evaluation/runs/gates/premerge_aws/20261024T083919Z-cpux86-73e3fa/` |
| Native Graviton4 build, 708/708 C++ tests | `20261024T100530Z-cpuarm-602802` | c8g.4xlarge, 901 s, $0.021 | `evaluation/runs/arm64/20261024T100530Z-cpuarm-602802/` |
| Full EuRoC stereo-inertial runs (11 flights) | `20261022T165035Z-cpux86-679a8b`, `20261022T193510Z-cpux86-655144` | c7i.4xlarge, 1,513 s + 1,371 s, $0.09 | `evaluation/runs/vio/euroc_full_aws/` |
| Known-entry gate: 3.78 m → 1.5 cm, 0/560 true loops lost | gate off `…-eee456`; gate on `20261023T213707Z-cpux86-e31421`; logged `20261024T042708Z-cpux86-4ec3f6` | c7i.4xlarge, about 2,550 s each, about $0.09 each | `evaluation/runs/collaboration/known_entry_gate/` |
| Recorded live run, gate off: 32/32 geometry-verified loops false, a robot 15.3 m off | `20261024T031036Z-cpux86-c51ab2` | c7i.4xlarge, 2,018 s, $0.07 | `evaluation/runs/collaboration/live_gate_off_20261024_c51ab2/` |
| Recorded live run, gate on, same scene and seed: 8/8 refused, all robots ≤ 7.5 cm | `20261024T085818Z-cpux86-debba5` | c7i.4xlarge, 1,871 s, $0.07 | `evaluation/runs/collaboration/live_gate_on_20261024_debba5/` |
| 250 s three-robot mission: 291 inter-robot edges kept, all merged, ≤ 0.22 m | `20261024T123707Z-cpux86-99ee9d` (+ pack `…-285fd8`) | c7i.4xlarge, 6,448 s, $0.23 | `evaluation/runs/live-agentic/demorec_250s_20261024_99ee9d/` |
| Final integrated stack: full VIO, 80/80 Bedrock decisions validated, GPU splat | `20261026T001000Z-gpu-5bebcd` | g4dn, 200 s mission | `evaluation/runs/live-agentic/final_stack_e2e/20261026T000432Z/` |
| Bedrock vs classical chooser, 90 s and 500 s matched pairs | `…-3f6e0b` / `…-c4e130`; `…-cd139e` / `…-ecf2b0` | c7i.4xlarge | `evaluation/runs/live-agentic/chooser_pair_{90s,500s}_20261025/` |
| Gaussian splat trained and rendered on GPU (2.59 M Gaussians) | `20261024T121121Z-gpu-ae604c`, `20261024T132853Z-gpu-1e9c60` | g4dn.xlarge (T4) | `evaluation/runs/reconstruction/shared_model_video_20261026/` |
| WorldGen splat, held-out PSNR 24.36 dB | `20261026T211816Z-gpu-2705f3` | g4dn.xlarge, 680 s, $0.023 | `evaluation/runs/reconstruction/worldgen_gsplat_20261026T211816Z/` |
| Pipeline profile: stage timings, thread scaling, Graviton vs x86 | 7 runs | c7i / c8g | `evaluation/runs/performance/pipeline_profile_20261024/` |
| DNN timings: XFeat, EigenPlaces, ALIKED, LightGlue | `20261021T211117Z-gpu-e2b3e3` | g4dn.xlarge, 1,520 s, $0.035 | `evaluation/runs/performance/dnn_timing/` |
