#!/usr/bin/env bash
# Optional acceleration for the CPU Gaussian display renderer; numpy is the fallback.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
mkdir -p "$ROOT/build/video_assets"
c++ -O3 -shared -fPIC "$ROOT/python/video/ariadne_video/gaussian_raster.cpp" -o "$ROOT/build/video_assets/gaussian_raster.so"
