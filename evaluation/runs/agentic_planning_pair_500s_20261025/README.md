# Classical and Bedrock chooser pair: 500-second run

This is the durable, compact evidence index for the matched AWS pair. The complete run archives are retained in the project's S3 artifact bucket; the large sensor recordings are not copied into this repository.

## Setup

Both arms used source revision `c143db14eb9dfbd7ffef003cf80ab3c344c64dec`, the generated `maze_short_6` scene with identity `89e61fecbb277f69`, the same renderer and runtime settings, and a 500-second live horizon. Their saved `live_pair500_*.yaml` files differ only in `chooser` (`classical` or `llm`). Both ran the Ariadne Phase 4 planner over the `mission_v1` mapping base. The runs were executed in parallel on c7i.4xlarge Spot instances with llvmpipe rendering.

## Recorded result

| Measure | Classical | Bedrock |
|---|---:|---:|
| Simulated mission duration | 500 s | 500 s |
| Joint replans | 100 | 100 |
| Commands decided / sent | 200 / 200 | 200 / 200 |
| Frames per agent | 10,001 | 10,001 |
| Integrated keyframes, agent 0 / 1 | 829 / 717 | 783 / 767 |
| Unpaired frames dropped | 0 | 0 |
| Late or out-of-order messages | 0 | 0 |
| End-to-end wall time | 3,359.42 s | 3,367.06 s |
| End-to-end real-time factor | 0.148835 | 0.148497 |
| Replay check | PASS | PASS |

Each replay check covers 15 conditions, including both agents' pose outputs, consumed-input digest, directive and mover-state hashes, trace records and hashes, and equality of replay 1 with replay 2. Each arm has 200 trace records. The per-arm replay JSON and replay logs are included in the complete S3 archives.

The run did not emit the route and map evaluation artifacts (`routes.jsonl`, `slab.png`, `eval.yaml`) required to score ground-truth coverage, route quality, or mission completion. This pair therefore adds long-horizon integration and deterministic replay evidence. It does not establish that Bedrock improved mission performance over the classical chooser. Approval-path activation and model-call/token accounting are also not claimed from this pair.

## Durable artifacts

The full archives and the small runner logs are in S3:

- Classical run `20261025T114148Z-cpux86-cd139e`: `s3://ariadne-836505524290-eun1/runs/20261025T114148Z-cpux86-cd139e/output/results/`. `run.tar.gz` is 30,523,293,173 bytes.
- Bedrock run `20261025T114156Z-cpux86-ecf2b0`: `s3://ariadne-836505524290-eun1/runs/20261025T114156Z-cpux86-ecf2b0/output/results/`. `run.tar.gz` is 29,842,891,150 bytes.

Each S3 prefix also contains `live_run.log`, `replay.log`, `SHA256SUMS`, runtime and host information, and the payload check. `replay_check.json` is inside each run archive. The manifest and hash values below were read from those completed archives and runner summaries.

The small local copies of `live_run.log`, `replay.log`, `SHA256SUMS`, and `exit_code` are stored in the `classical/` and `bedrock/` subdirectories here. They were copied out of the dedicated run worktrees so the status evidence does not depend on temporary-worktree retention.

## Replay digests

| Digest | Classical | Bedrock |
|---|---|---|
| Consumed-input digest | `3f9caafdd6d89bff` | `e7f2c3bf0f7dbad6` |
| Trace SHA-256 | `3d0044bb55f774103d75ed44329e98bf04e1de353ee70262fd3295ac0a0203c3` | `eb930e844cdee76e57363f2493cba19c247a9ac570c2ebe7a34621a33e1135f6` |
| Mover-state SHA-256 | `ce70053666b8ae9df62d68bce7a5c0110dcf738cf732afce7ce55435c4fe6181` | `c32dd01ec79074fbc6a2b3fe20b5ea8a5c0f0db443dbb1bcc592c186d63abaf2` |
| Agent 0 estimate SHA-256 | `b2315c572d71c0327cd280af319e5e97fb15c6a46d6a512dd750d6dff4be7111` | `012bc7bdcd6f036c27412d15d29d0c1efa8f50bedc0b6579db2e1fcd3116cd71` |
| Agent 1 estimate SHA-256 | `455e8492f12b536beeb32a5e83493badfe4eeaf75d041252863019e308059e4f` | `ac14a39c1d5710847e3f04d413cf7963cd2ce13d0af0d9b7f6646429a17515de` |
