# Ariadne media (devpost + report)

Every image here comes from a recorded run or from a rendered scene of the demo film. Each label below says which.

- **Regenerate the GIFs:** `docs/Latex/figures/src/make_gifs.sh`. It runs in `manimcommunity/manim:v0.22.0`, which provides PyAV.
- **Regenerate the stills:** `python3 docs/Latex/figures/src/make_stills.py`, run on the host (needs numpy and Pillow).

## GIFs (`gifs/`)

These are cut from the film's 480p renders (`python/video/media/videos/*/480p15/*.mp4`). The scene sources are in `python/video/scenes/`.

| file | content | label |
|---|---|---|
| `feature_tracking_lk.gif` | FAST + ORB detections followed frame to frame by pyramidal LK on recorded camera frames | recorded frames; tracks replayed offline with OpenCV 5 |
| `frontend_lk_fb_ransac_stereo.gif` | coarse-to-fine LK, forward–backward check, RANSAC epipolar fit, left–right stereo at keyframes | illustrative, on real frames |
| `window_ba_schur.gif` | 10-keyframe visual–inertial bundle adjustment, Schur complement, window exit to the pose graph | illustrative; full stack as benchmarked on EuRoC |
| `two_stage_place_recognition.gif` | EigenPlaces fingerprint, then ALIKED + LightGlue detail on demand | EuRoC numbers are measured |
| `exchange_funnel_recorded.gif` | 1,436 candidates → 205 fetched → 128 verified → 59 accepted; merged map 13.1 mm | one recorded simulation run |
| `spectral_lambda2_selection.gif` | λ₂ (Fiedler value) gain per predicted byte, greedy pick + one swap | λ₂ derived from the run's accepted edges |
| `aliasing_known_entry_gate.gif` | a look-alike pair that passes geometry (wrong by 14.9 m and 138°), stopped by the known-entry gate | recorded run, gate off |
| `chi2_pcm_gnc.gif` | χ² separation, PCM maximum clique, GNC-TLS weights | d² panel measured; mechanism panels illustrative |
| `cbba_task_allocation.gif` | consensus-based bundle algorithm on the maze | CBBA/optimum 0.994, measured on replay |
| `responder_width_dstar_lite.gif` | person-width traversability, multi-exit D* Lite, incremental repair | 12/12 routes valid, measured on replay |
| `agentic_decision_chain.gif` | what the cameras found → candidates → Bedrock Nova 2 Lite via MCP → C++ validation | live run, t = 5 s |
| `splat_submap_assembly.gif` | per-agent Gaussian submaps assembling in one frame | trained submaps |
| `splat_orbit_626abe.gif` | orbit around the trained splat (AWS g4dn, T4) | training views |

## Stills (`stills/`; also in `docs/Latex/figures/`)

| file | source |
|---|---|
| `fig_tracking.png` | `python/video/data/tracks/tracks.json` drawn on recorded agent-0 frame 360 (run c51ab2) |
| `fig_premerge_*.png` | AWS pre-merge gate run `20261024T083919Z-cpux86-73e3fa` |
| `fig_slabs.png`, `fig_decision.png` | final-stack recorded run, 2 agents, 200 s: `/data/code/Ariadne-e2e-final/evaluation/runs/final_stack_e2e/20261026T000432Z/` |
| `fig_splat.png`, `splat_626abe_still_*.png` | GPU training `20261024T121121Z-gpu-ae604c`, render `20261024T132853Z-gpu-1e9c60` |
| `fig_worldgen_heldout.png` | `evaluation/runs/ph6/worldgen_gsplat_20261026T211816Z` (branch `feat/ph5-dve`) |
| `fig_outlier_ramp.png` | `evaluation/runs/2.5/outlier_ramp/` (synthetic) |
| `fig_nbv_coverage_time.png` | `evaluation/runs/nbv_4.3_gate/` |
| `fig_bytes_3_6.png` | `evaluation/runs/ph3/bytes_3.6_20261024/` |
| `frame_agent{0,1}_f600.png` | recorded camera frames, collab60d pack |
| `console_operator_recorded_run.png` | operator console (top-down map, approvals panel, LLM proposals, directives, incident timeline) replaying recorded run `aadffb8df5790ca0` at t = 45 s; screen capture from 2026-10-23 |
