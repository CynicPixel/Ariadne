#!/usr/bin/env bash
# Read-only AWS retrieval; no instance launches or GPU work.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
DEST=build/video_assets/splat_626abe_render_20261024_aws
BUCKET=s3://ariadne-836505524290-eun1/runs
mkdir -p "$DEST"
infra/scripts/aws.sh aws s3 sync "$BUCKET/20261024T132853Z-gpu-1e9c60/output/results/" "$DEST/" --only-show-errors
infra/scripts/aws.sh aws s3 cp "$BUCKET/20261024T121121Z-gpu-ae604c/output/results/train/remote.json" "$DEST/training_remote.json" --only-show-errors
infra/scripts/aws.sh aws s3 cp "$BUCKET/20261024T121121Z-gpu-ae604c/output/results/train/splat_batch/manifest.json" "$DEST/training_manifest.json" --only-show-errors

# The 3D assembly uses actual trained chunks, in their recorded anchor frames.
if [ ! -f "$DEST/splat_batch.tar.gz" ]; then
  infra/scripts/aws.sh aws s3 cp "$BUCKET/20261024T121121Z-gpu-ae604c/output/results/train/splat_batch.tar.gz" "$DEST/splat_batch.tar.gz" --only-show-errors
fi
python3 - "$DEST" <<'PYEXTRACT'
import sys, tarfile
from pathlib import Path
root = Path(sys.argv[1])
with tarfile.open(root / "splat_batch.tar.gz") as archive:
    archive.extractall(root, filter="data")
PYEXTRACT
