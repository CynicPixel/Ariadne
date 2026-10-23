# Phase 5 evidence (A2)

## lighting/ — D142 no-dust lighting rows (AWS session 5, commit 380c900, clean tree)
Produced by `infra/eval/ph5_aws.sh` step 4b on c7i.2xlarge (spot, eu-north-1c), image
`sha256:91643838031ab6cbda6ae2a5dec9de51591b4b433ff26c390e3fbef83f39985a`:

    ariadne_sim_vo --rec <rec>/rec --scene <rec>/scene --agent 0 --frames 1000 --seed 1 \
      --perception-config configs/perception/sim_v1.yaml --slam-config configs/slam/sim_v2.yaml \
      [--degradation configs/degradation/lampdark_sensor_v0.yaml]

Recordings: `s3://ariadne-836505524290-eun1/runs/ph5-dark/631b43b…/{ci_tiny_dark_0,ci_tiny_dark_fixed_0}` (D137).
Rows: `lamp_*` = co-moving lamp, `fixed_*` = fixed lights; `_noD` = no degradation, `_sensor` = lampdark_sensor_v0.

## hilti_5.6/ — 5.6 real-data rows (configs/benchmark/hilti_5.6.yaml; D141, D143, D144)
Local, commit 1468344 (clean tree), `ariadne/core-dev` image:

    ariadne_euroc_vo --sequence site2_robot_3_asl --perception-config configs/perception/<p>.yaml \
      --slam-config configs/slam/<s>.yaml --seed 1 --out <dir>
    python -m analysis.hilti_score <dir>/estimate_body.tum site2_robot_3_asl/site2_robot_3_gt.txt \
      --gcp site2_robot_3_asl/gcp.csv --calib site2_robot_3_asl/hilti_calibration/calibration_robot --out hilti_score.yaml

The sequence (Hilti SLAM Challenge 2023, CC BY-NC-SA 3.0) is not copied here: only our estimates and derived numbers.
