#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Learned-model assets, reproducibly (D15, SMOKE §4p). Nothing here is committed: the outputs live in .cache/models/.
#
#   infra/scripts/models.sh xfeat     clone verlab/accelerated_features at the PINNED commit into
#                                     .cache/upstream/xfeat_src (untrusted: own directory, read-only mount), check the
#                                     weights' sha256, then export weights + PyTorch reference outputs OFFLINE
#                                     (--network none) into .cache/models/xfeat_ref/
#   infra/scripts/models.sh test      the D15 acceptance test against that export (both engines; fails if absent)
#   infra/scripts/models.sh pgo-oracle    100 seeded random pose graphs (half with full correlated information) solved by
#                                     GTSAM 4.3 in the pinned ariadne-oracle:1 image, OFFLINE -> .cache/oracle/pgo/
#   infra/scripts/models.sh pgo-sesync    the SE-Sync 3-D benchmark graphs (pinned commit, sha256-checked) solved by GTSAM
#                                     -> .cache/oracle/pgo_sesync/ (slow: rim, grid3D; not part of the pre-merge gate)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
XFEAT_COMMIT=e92685f57f8318b18725c5c8c0bd28c7fe188d9a
XFEAT_PT_SHA256=0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b
SRC=.cache/upstream/xfeat_src OUT=.cache/models/xfeat_ref IMAGE=ariadne-xfeat-ref:1
die() { echo "models.sh: $*" >&2; exit 1; }

case "${1:-}" in
  xfeat)
    if [ ! -d "$SRC/.git" ]; then
      mkdir -p .cache/upstream
      git clone -q https://github.com/verlab/accelerated_features.git "$SRC"
    fi
    git -C "$SRC" fetch -q --depth 1 origin "$XFEAT_COMMIT" 2>/dev/null || true
    git -C "$SRC" checkout -q "$XFEAT_COMMIT" || die "cannot check out $XFEAT_COMMIT"
    echo "$XFEAT_PT_SHA256  $SRC/weights/xfeat.pt" | sha256sum -c --quiet || die "xfeat.pt sha256 mismatch"
    docker image inspect "$IMAGE" >/dev/null 2>&1 \
      || docker build -q -t "$IMAGE" -f infra/docker/xfeat_ref.Dockerfile infra/docker >/dev/null
    seq=.cache/datasets/euroc/MH_01_easy/mav0/cam0/data
    [ -d "$seq" ] || die "missing $seq (infra/scripts/fetch_euroc.py MH_01_easy)"
    frame="$(ls "$seq" | sed -n 200p)"
    mkdir -p "$OUT"
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/$SRC:/xf:ro" \
      -v "$PWD/python/model_tools:/tools:ro" -v "$PWD/$seq:/img:ro" -v "$PWD/$OUT:/out" "$IMAGE" \
      python -I /tools/xfeat_reference.py --upstream /xf --out /out --image "euroc_mh01_f200=/img/$frame"
    echo "exported to $OUT (manifest.yaml has every sha256)" ;;
  test)
    [ -f "$OUT/xfeat_weights.atb" ] || die "no export in $OUT (run: infra/scripts/models.sh xfeat)"
    infra/scripts/core.sh dev bash -c "cmake --build /work/build/docker --target test_opencv --parallel 8 >/dev/null \
      && ARIADNE_XFEAT_REF=/work/$OUT ARIADNE_REQUIRE_MODELS=1 /work/build/docker/tests/test_opencv \
         --gtest_filter='*XFeatReference*'" ;;
  pgo-oracle)
    docker image inspect ariadne-oracle:1 >/dev/null 2>&1 \
      || docker build -q -t ariadne-oracle:1 -f infra/docker/oracle.Dockerfile infra/docker >/dev/null
    rm -rf .cache/oracle/pgo && mkdir -p .cache/oracle/pgo
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD:/work" ariadne-oracle:1 \
      python -m oracle.pgo_gtsam --out /work/.cache/oracle/pgo --random 100 --seed 20261016 ;;
  pgo-sesync)
    SESYNC_COMMIT=9b631b6b82d8aa7d32ab846412ae1c070412b7b6 D=.cache/datasets/pgo/se_sync
    mkdir -p "$D" .cache/oracle/pgo_sesync
    args=()
    for f in tinyGrid3D smallGrid3D sphere2500 torus3D cubicle parking-garage rim grid3D; do
      [ -f "$D/$f.g2o" ] || curl -sfL -o "$D/$f.g2o" \
        "https://raw.githubusercontent.com/david-m-rosen/SE-Sync/$SESYNC_COMMIT/data/$f.g2o" || die "fetch $f"
      args+=(--g2o "$f=/work/$D/$f.g2o")
    done
    # Pinned sha256 of the data at SESYNC_COMMIT (recorded 2026-10-16); the check is against THESE, not a generated file.
    (cd "$D" && sha256sum -c --quiet) <<'SUMS' || die "SE-Sync data sha256 mismatch"
f7781d485383cec86d47d7650970132c36d6f3a1f4e5d62a49b7f8245c0a6465  cubicle.g2o
f5dd7a17bf8b9c39890b14f9bb0c1e8366a7490f01714e43e3bc03f52cf414e7  grid3D.g2o
3ac0a31bfb601d7455d451e2546655cb5dececf51a7823f57c8a7e0fe1ca6527  parking-garage.g2o
9cc9b0d8e5f080c5a029605e6cffd304f3c74d404d13f6329d942c473591155b  rim.g2o
9ea56c2ad1ebcc322560eb2f8d83cb3a60f99e2e2acc35e097b1162cdbafd649  smallGrid3D.g2o
104ab57593394f24351d9f692f3b923f8b98fff1eb638c64356cf5049e06cf3c  sphere2500.g2o
c341eb0d09f7556b337be5a62b9354384885333a25fa718fd699fafb19620493  tinyGrid3D.g2o
1ef8831023b372188d33ca63ce0da079d51df89364df71f2bce1c5c7c69ed5ec  torus3D.g2o
SUMS
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD:/work" ariadne-oracle:1 \
      python -m oracle.pgo_gtsam --out /work/.cache/oracle/pgo_sesync --random 0 "${args[@]}" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
