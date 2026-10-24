#!/usr/bin/env bash
# Render one Manim scene in the pinned Manim CE image (cairo, pango, ffmpeg inside; nothing installed on the host).
#   python/video/render.sh scenes/s01_worldgen.py WorldGen [-ql|-qm|-qh|--fps 60 ...]
# Quality defaults to -ql (480p15) for review. Output lands in python/video/media/ (git-ignored).
set -euo pipefail
IMAGE="manimcommunity/manim:v0.22.0@sha256:3c3ce944c09d6fa73de59fe1e7a95e334e193ae9c2275d65937054e675f87b03"  # = requirements.txt pin
HERE="$(cd "$(dirname "$0")" && pwd)"
file="$1"; scene="$2"; shift 2
q=("$@"); [ ${#q[@]} -eq 0 ] && q=(-ql)
# --user: write media as the host user, not root. ARIADNE_VIDEO_DATA: the copied run data, read-only.
exec docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$HERE":/manim -w /manim -e ARIADNE_VIDEO_DATA=/manim/data \
  "$IMAGE" manim "${q[@]}" "$file" "$scene"
