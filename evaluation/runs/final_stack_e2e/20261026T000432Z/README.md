# Final integrated stack, recorded live run (AWS, 2 agents, 200 s)

- **Scene:** `maze_short_6`.
- **Configs** (`selection.txt`): perception `sim_v1_rad_flat`, SLAM `sim_v2_loops` (10-keyframe window VIO with IMU and loop closure), IMU noise `imu_noise_v1`, live `live_final_e2e`, chooser llm (Bedrock Nova 2 Lite).
- **Source:** `c143db14` plus the working-tree diff listed in `source_status.txt`.
- **Outcome** (`run/results.yaml`, `campaign.log`):
  - 40 replans, 80 commands decided and sent;
  - `answer_timeouts: 0`, `refused_goals: 3`;
  - 4,001 frames per agent;
  - replay PASS;
  - GPU reconstruction from 677 keyframes.
- **Kept here (a slim subset):**
  - `run/trace.jsonl`: one record per agent per replan, holding the vision summary, candidates, choice, tool calls and command;
  - `run/mover_state.jsonl`: command accept and refuse records;
  - `run/commands.jsonl` and `run/manifest.yaml`;
  - the t = 50 s decision images and three occupancy slabs.
- **Not kept here:** the full run (about 380 MB, with raw estimator output, collaboration payloads and all per-decision images) stays outside the repository.
