#!/bin/bash
# The instance half of the full gate's render on AWS (premerge_aws_launch.sh stage full). phase1.sh calls it in
# place of `kaggle.sh render PKG_DIR OUT` (ARIADNE_RENDER_CMD): the Kaggle token stays on the launching host, so the
# render package goes to S3, the host renders it on Kaggle (the same kaggle.sh, a fresh session) and hands
# renders.npz + remote.json back. Keys under $RELAY (GetObject/PutObject only; the runner role has no ListBucket or
# DeleteObject, so "not there yet" is a failed GetObject and the markers are written last):
#   package.tar  package.sha256  package.done      instance -> host
#   render.tar   render.done | render.failed       host -> instance
set -euo pipefail
: "${RELAY:?}"
pkg="${1:?PKG_DIR}" out="${2:?OUT}"
deadline=$(( $(date +%s) + ${RELAY_TIMEOUT_MIN:-75} * 60 ))
log() { echo "[relay $(date -u +%H:%M:%S)] $*"; }
got() { aws s3 cp "$RELAY/$1" "$2" --only-show-errors > /dev/null 2>&1; }

mkdir -p "$out"
want="$(sha256sum "$pkg/render_package.npz" | cut -d' ' -f1)"
tar cf - -C "$pkg" . | aws s3 cp - "$RELAY/package.tar" --only-show-errors
echo "$want" | aws s3 cp - "$RELAY/package.sha256" --only-show-errors
date -u +%FT%TZ | aws s3 cp - "$RELAY/package.done" --only-show-errors
log "package $want handed to the host; waiting for its Kaggle render"

t="$(mktemp -d)"
while :; do
  if got render.done "$t/done"; then break; fi
  if got render.failed "$t/failed"; then log "host render failed:"; cat "$t/failed"; exit 1; fi
  [ "$(date +%s)" -lt "$deadline" ] || { log "no render after ${RELAY_TIMEOUT_MIN:-75} min"; exit 1; }
  sleep 20
done
got render.tar "$t/render.tar" || { log "render.done without render.tar"; exit 1; }
tar xf "$t/render.tar" -C "$out"
[ -f "$out/renders.npz" ] && [ -f "$out/remote.json" ] || { log "render.tar lacks renders.npz/remote.json"; exit 1; }
grep -q "$want" "$t/done" || { log "host rendered a different package: $(cat "$t/done")"; exit 1; }
log "render back: $(sha256sum "$out/renders.npz" | cut -d' ' -f1)"
