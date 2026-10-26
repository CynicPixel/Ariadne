# Matched classical vs Bedrock planning run evidence

This directory preserves the compact evidence extracted from the paired AWS runs. The repository source tree was not used as the run worktree; the evidence is copied here so it survives cleanup of `/tmp`.

## Pair setup

- Scene: `maze_short`, seed 6; scene ID `89e61fecbb277f69`
- Source revision: `c143db14eb9dfbd7ffef003cf80ab3c344c64dec`
- Paired configurations differ only in chooser (`classical` vs `llm`); both ran 90 simulated seconds on the same instance class and renderer.
- AWS region / instance: `eu-north-1`, `c7i.4xlarge` Spot, Mesa llvmpipe
- Bedrock model: `eu.amazon.nova-2-lite-v1:0`

## AWS run identifiers and original archives

- Classical: run `20261025T001119Z-cpux86-3f6e0b`, runtime `live-20261025T001308Z-a688`, instance `i-0ca3266dffc8f2ef3`
  - Full original: `s3://ariadne-836505524290-eun1/runs/20261025T001119Z-cpux86-3f6e0b/output/results/run.tar.gz`
- Bedrock: run `20261025T001221Z-cpux86-c4e130`, runtime `live-20261025T001411Z-5ccc`, instance `i-0c8b3c272d81bfb6b`
  - Full original: `s3://ariadne-836505524290-eun1/runs/20261025T001221Z-cpux86-c4e130/output/results/run.tar.gz`

The full archives include MCAP recordings and remain in S3. The local compact archives contain traces, decision/tool records, telemetry, snapshots, runtime/simulator/orchestrator logs, replay outputs, and manifests; MCAP is excluded to keep the evidence bundle small.

## Key findings from the traces

- Both 90-second runs completed: 18 joint replans and 36 agent decision records each.
- Bedrock differed from the classical choice on 6/18 joint replans. No Bedrock fallback or tool error was recorded.
- Both runs sent 36 movement commands: 34 accepted, zero refused, and 2 unanswered at run end.
- Bedrock trace usage: 118 model calls and 2,936,424 cumulative input tokens across 18 joint decisions (163,135 mean per decision, counting repeated context across calls); median reported orchestration latency 5.8 s, maximum 9.03 s.
- Replay verification passed for both runs.

This is one matched scene/seed pair. It demonstrates integration and replayability, not a general mission-performance advantage. Candidate indices are agent-local; matching indices do not by themselves establish duplicate spatial assignments.

## Files

- `classical-evidence.tar.gz`, `bedrock-evidence.tar.gz`: compact evidence bundles
- `*-input-manifest.txt`: files included in each compact bundle
- `pair_manifest.json`: run and configuration pairing metadata
- `analysis.json`: parsed per-run summary and per-turn Bedrock trace statistics
- `SHA256SUMS`: integrity hashes for preserved files
