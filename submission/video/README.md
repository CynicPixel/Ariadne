# Video (at most 5 minutes)

- **File:** [`ariadne_final_1080p60.mp4`](ariadne_final_1080p60.mp4): 1080p60, 5:00, narrated, subtitles burned in.
- **Public link:** https://youtu.be/yrbcvQKnfkA ("Ariadne: Collaborative Visual SLAM and Safe Egress for Robot Teams | OpenCV AI Competition 2026").
- **Narration:** Ming-3.0-tts-0.5b, voice US_James (`infra/eval/narration_aws.sh`, AWS g4dn run `20261027T005304Z-gpu-f68f8d`, $0.05).
- **Rebuild:** `python/video/assemble.py 1080p60`, then `python/video/tools/mix_final.py --target 300`.
- **Contents:** the team, the application working on recorded runs, the architecture, and the principal results.
- **Production:** every scene is a Manim animation built from the project's own run data and code replays. The scene sources are in `python/video/scenes/`, and the build scripts are `python/video/render.sh` and `python/video/assemble.py`.
- **Labels:** each number on screen carries a tag saying whether it is measured, published or illustrative.
