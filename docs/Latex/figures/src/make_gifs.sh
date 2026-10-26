#!/usr/bin/env bash
# Devpost/report GIFs cut from the demo film's rendered scenes (python/video/media/videos/*/480p15) and from the
# splat orbit render. Runs inside manimcommunity/manim:v0.22.0 (PyAV + Pillow). Usage, from the repo root:
#   docker run --rm -u $(id -u):$(id -g) -v $PWD:/r -w /r --entrypoint bash manimcommunity/manim:v0.22.0 \
#     docs/Latex/figures/src/make_gifs.sh
set -euo pipefail
T="python3 docs/Latex/figures/src/media_tools.py"
V=python/video/media/videos
O=submission/media/gifs
g() { $T gif "$V/$1" "$O/$2" "$3" "$4" --fps "${5:-10}" --width "${6:-720}"; }
g s03_seeing/480p15/Seeing.mp4             feature_tracking_lk.gif        10.0 20.1 10 854
g s04a_ba/480p15/BundleAdjust.mp4          frontend_lk_fb_ransac_stereo.gif 9.5 34.0  8
g s04a_ba/480p15/BundleAdjust.mp4          window_ba_schur.gif           37.5 58.0  8
g s05_sharing/480p15/Sharing.mp4           two_stage_place_recognition.gif 15.0 40.0 8
g s05_sharing/480p15/Sharing.mp4           exchange_funnel_recorded.gif  40.0 49.0  8
g s05_sharing/480p15/Sharing.mp4           spectral_lambda2_selection.gif 49.0 68.7 8
g s06_trust/480p15/Trust.mp4               aliasing_known_entry_gate.gif  0.0 21.5  8
g s06_trust/480p15/Trust.mp4               chi2_pcm_gnc.gif              21.5 42.1  8
g s07_cbba/480p15/SharingWork.mp4          cbba_task_allocation.gif       5.4 24.7  8
g s09_thread/480p15/Thread.mp4             responder_width_dstar_lite.gif 0.0 34.0  8
g s08_agentic/480p15/HumanInLoop.mp4       agentic_decision_chain.gif     0.0 17.5  8
g s10_splat/480p15/SharedModel.mp4         splat_submap_assembly.gif      0.0 21.5  8
$T frames 'build/video_assets/splat_626abe_render_20261024/orbit/frame_*.png' "$O/splat_orbit_626abe.gif" \
  --step 3 --fps 10 --width 720 --label 'Gaussian splat · AWS g4dn training · training views'
ls -la "$O"
