# Phase 5 §31.5 rows (D145–D147), AWS run 20261023T113916Z-cpux86-7b3f1b

- command: `infra/scripts/spinup.sh run --profile cpux86 --type c7i.2xlarge --az eu-north-1c --max-minutes 90 -- "IMAGE=836505524290.dkr.ecr.eu-north-1.amazonaws.com/ariadne/ariadne-core@sha256:91643838031ab6cbda6ae2a5dec9de51591b4b433ff26c390e3fbef83f39985a GIT_COMMIT=649f156… BUCKET=ariadne-836505524290-eun1 DARK_PREFIX=s3://…/runs/ph5-dark/631b43b… HILTI=skip ROWS=d145 bash infra/eval/ph5_aws.sh"`
- commit `649f156` (clean tree); image digest in `image_id.txt`; recordings `ci_tiny_dark_0` / `ci_tiny_dark_fixed_0` (D137), sizes checked (`fetch_check.txt` empty)
- rows: agent 0, 1000 frames, seed 1, slam `sim_v2_vo` (= D142's sim_v2); configs per row in `infra/eval/ph5_aws.sh` and each `replays/<row>/manifest.yaml`
- raw: `replays/<row>/{results.yaml, manifest.yaml, estimate_body.tum, vo_trace.csv, exposure_trace.csv}`
- cost: c7i.2xlarge spot $0.0789/h, 2717 s, ~$0.0595 compute (spinup). Earlier attempts the same day: $0.0022 (image tag gone), $0.0117 (sim_v2 needs --imu-noise, all rows failed), $0.0422 (spot reclaim, nothing kept)
- analysis: inline in D145–D147 Results (decisions.md); trajectory equality by sha256 of estimate_body.tum; trace checks over exposure_trace.csv + vo_trace.csv stamps
