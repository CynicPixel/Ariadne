# L1 lockstep (D165): two live runs of one seed are the same run

`ci_tiny` seed 0, 20 s of sim time, classical chooser, Mesa llvmpipe (dev host, no GPU), `ARIADNE_LOCKSTEP=1`:

    infra/scripts/live.sh compose ci_tiny 0 0.15
    ARIADNE_LOCKSTEP=1 infra/scripts/live.sh run build/live/scenes/ci_tiny_0 build/live/runs_lockstep3 20   # twice
    python3 python/analysis/lockstep_repro_check.py RUN_A RUN_B repro_check.json

Result (`repro_check.json`): **PASS**. Both runs: lockstep on, 4 barriers, every command applied AT its barrier
(applied_ns = T), gz paused exactly at T (ack paused_clock − T = 0 ns at every barrier), longest pause 0.098 / 0.057 s
of wall time. Identical between the runs: consumed stream (19 257 messages, digest f693e2b2edf95b00), per-agent
estimate_body.tum sha256 and frame/keyframe counts, trace.jsonl, mover_state.jsonl.

How it got there (each found by this check, fixed, re-run):
1. the end-of-run holds were sent after the last resume and raced the clock (applied 20.000 vs 20.001 s): they now go
   out at the last barrier while paused (stamp end + 1 ns) and are answered before the resume;
2. every IMU and image message differed in 2 bytes, the CDR alignment padding after `frame_id`, which the ROS
   serializer leaves uninitialised: RuntimeLink zeroes its per-topic serialization buffer.

Scope: one renderer (llvmpipe), one scene, 20 s, classical chooser. Not claimed for NVIDIA EGL or for the LLM chooser
(the model's answers are not deterministic; lockstep removes the latency confound, not the model's choice). The run
directories (with inputs.mcap) stay under build/; this directory keeps the per-run logs and the check.
