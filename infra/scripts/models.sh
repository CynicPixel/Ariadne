#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Learned-model assets, reproducibly (D15, SMOKE §4p). Nothing here is committed: the outputs live in .cache/models/.
#
#   infra/scripts/models.sh xfeat     clone verlab/accelerated_features at the PINNED commit into
#                                     .cache/upstream/xfeat_src (untrusted: own directory, read-only mount), check the
#                                     weights' sha256, then export weights + PyTorch reference outputs OFFLINE
#                                     (--network none) into .cache/models/xfeat_ref/
#   infra/scripts/models.sh test      the D15 acceptance test against that export (both engines; fails if absent)
#   infra/scripts/models.sh eigenplaces   EigenPlaces R50-256 (D62): pinned clone + weights (sha256-checked) -> export
#                                     OFFLINE into $MODELS/eigenplaces_ref/, then every output checked against the
#                                     sha256s PINNED below (the bit-exact re-run claim of perception.md §13)
#   infra/scripts/models.sh aliked-lg     ALIKED-n16 + LightGlue (D63), same contract -> $MODELS/aliked_lg_ref/
#   infra/scripts/models.sh eigenplaces-sim  the sim-rig export (L3): 1280x720 and 640x360, RGB reference frames
#                                     -> $MODELS/eigenplaces_sim_ref/
#   infra/scripts/models.sh aliked-lg-sim    ALIKED-n16 K=1024 at 1280x720 + LightGlue, RGB -> $MODELS/aliked_lg_sim_ref/
#     The sim frames are four left_rgb frames of the phase-1 recording, written by ariadne_frame_dump into
#     $ARIADNE_SIM_FRAMES (default $MODELS/sim_rgb_frames; command in perception.md §14) and sha256-checked here.
#     Overrides: ARIADNE_UPSTREAM (clones/weights, default .cache/upstream), ARIADNE_MODELS (default .cache/models).
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
UP="$(realpath -m "${ARIADNE_UPSTREAM:-.cache/upstream}")" MODELS="$(realpath -m "${ARIADNE_MODELS:-.cache/models}")"
EUROC="$(realpath -m .cache/datasets/euroc)" DNN_IMAGE=ariadne-model-ref:1
# pin_clone DIR URL COMMIT
pin_clone() {
  [ -d "$1/.git" ] || git clone -q "$2" "$1"
  git -C "$1" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "$1" checkout -q "$3" || die "cannot check out $3 in $1"
}
# fetch_pinned FILE URL SHA256 (the network is used only here)
fetch_pinned() {
  [ -f "$1" ] || { mkdir -p "$(dirname "$1")"; curl -sfL -o "$1" "$2" || die "fetch $2"; }
  echo "$3  $1" | sha256sum -c --quiet || die "$1 sha256 mismatch"
}
# The torch + onnx export image (perception.md §13). Inline so infra/docker/ stays untouched; built on the xfeat image.
dnn_image() {
  docker image inspect ariadne-xfeat-ref:1 >/dev/null 2>&1 \
    || docker build -q -t ariadne-xfeat-ref:1 -f infra/docker/xfeat_ref.Dockerfile infra/docker >/dev/null
  docker image inspect "$DNN_IMAGE" >/dev/null 2>&1 || docker build -q -t "$DNN_IMAGE" - >/dev/null <<'DOCKERFILE'
FROM ariadne-xfeat-ref:1
RUN pip install --no-cache-dir --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu \
      torchvision==0.20.1+cpu \
 && pip install --no-cache-dir --root-user-action=ignore onnx==1.17.0 \
 && python -c "import torch, torchvision; assert torch.__version__ == '2.5.1+cpu', torch.__version__; assert torchvision.__version__ == '0.20.1+cpu', torchvision.__version__"
DOCKERFILE
}
SIMF="$(realpath -m "${ARIADNE_SIM_FRAMES:-$MODELS/sim_rgb_frames}")"
# The phase-1 ci_tiny recording (bag sha256:21399732...), agent 0, frames 200/205 (10 s) and 700/710 (35 s).
sim_frames() {
  [ -d "$SIMF" ] || die "no sim frames in $SIMF (ariadne_frame_dump, perception.md §14)"
  (cd "$SIMF" && sha256sum -c --quiet) <<'SUMS' || die "sim frames differ from the pinned ones"
c4dc66bb6b5c3a586132c56b494e80e0ffbdc25837bac681c8ee00f5a98e1758  agent0_f200.png
fb12ec0d6bf28cf64efe1133a8ceb959249799b8b65a64c4c9c63bd81eef6aa5  agent0_f205.png
f5d1895903fa9d7289b9a9e9524b6711a90844c1a39916cc6cff9e9a9837f909  agent0_f700.png
3b53ad0e3dabaf6c1e632ba5f8e4f966e6bae62cae0c14fe61bd6c8c2c233c93  agent0_f710.png
SUMS
}
# check_pinned DIR: every output against the sums on stdin, and nothing unpinned beside them
check_pinned() {
  local sums; sums="$(cat)"
  (cd "$1" && sha256sum -c --quiet <<<"$sums") || die "$1 differs from the pinned export (see the FAILED lines)"
  local want got
  want="$(awk '{print $2}' <<<"$sums" | sort)" got="$(ls "$1" | grep -vx manifest.yaml | sort)"
  [ "$want" = "$got" ] || die "$1 file set differs from the pinned list"
  echo "$1: all $(wc -l <<<"$sums") outputs match the pinned sha256"
}

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
    infra/scripts/core.sh dev bash -c "cmake --build /work/build/docker --target test_opencv --parallel ${ARIADNE_BUILD_JOBS:-2} >/dev/null \
      && ARIADNE_XFEAT_REF=/work/$OUT ARIADNE_REQUIRE_MODELS=1 /work/build/docker/tests/test_opencv \
         --gtest_filter='*XFeatReference*'" ;;
  eigenplaces)
    EP_COMMIT=a2969f71d5ea31017443af490b15273ca4c50af1 OUT="$MODELS/eigenplaces_ref"
    pin_clone "$UP/eigenplaces_src" https://github.com/gmberton/EigenPlaces.git "$EP_COMMIT"
    fetch_pinned "$UP/eigenplaces_weights/ResNet50_256_eigenplaces.pth" \
      https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_256_eigenplaces.pth \
      2923c10295563ea15efa1571b499c67a7aab4d1ee6e19eba5e8b92756dc9d7e7
    dnn_image
    rm -rf "$OUT" && mkdir -p "$OUT"
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/python/model_tools:/tools:ro" \
      -v "$UP/eigenplaces_src:/ep:ro" -v "$UP/eigenplaces_weights:/w:ro" -v "$EUROC:/euroc:ro" \
      -v "$OUT:/out" "$DNN_IMAGE" python -I /tools/eigenplaces_reference.py --upstream /ep \
      --weights /w/ResNet50_256_eigenplaces.pth --out /out --size 752x480 --size 376x240 \
      --image euroc_mh01_f200=/euroc/MH_01_easy/mav0/cam0/data/1403636589713555456.png \
      --image euroc_v101_f500=/euroc/V1_01_easy/mav0/cam0/data/1403715298212143104.png \
      --upstream-commit "$EP_COMMIT"
    # Pinned 2026-10-19 (SMOKE §4r): the export the D62 numbers were measured with.
    check_pinned "$OUT" <<'SUMS' ;;
c61617ed5ee3ce813b255b543478b5dc45b71f80d3131af4e618a625c8b39fde  eigenplaces_r50_256_376x240.onnx
b38f7ce9e1e234340c77f8ea26fe59970bc34af71621a9ed362181c6e9ca4fec  eigenplaces_r50_256_752x480.onnx
2782e449003e1de343d886a865391995ff6ed3cb61e5351de0f2d5cf6cc3b36a  eigenplaces_r50_layer4_376x240.onnx
d5f84a6eb73d498639e048362a21d595b75fd5a070e5dfe398a4233214647eeb  eigenplaces_r50_layer4_752x480.onnx
785a83793d65eb9935791dfc712102f2b281f4d5b140d09f3b5867ea7dafe8fb  head.atb
7f4a0533db6d836d666076ba39bf92eb701dbdbcacd8bb5fa719d561f685a9e0  input_euroc_mh01_f200.png
3200d23cdf26b2793d3fe752b20473ba5f02ff0a0db4ae33b46e0d8fdc9ebef2  input_euroc_v101_f500.png
024a5d71067ce5273131c0015fc89d6a281e1a16780710ac967e45bef116768c  ref_euroc_mh01_f200_376x240.atb
c748630d500677a7fa85a5aaa9b96d8b3865a5021e2f03b7e97d8fc7b4308788  ref_euroc_mh01_f200_752x480.atb
daac585dafdb36bb43febbbb4992a94a045079cc6c050ffedd045bbba086a701  ref_euroc_v101_f500_376x240.atb
f7aef7b9d19023dc103384bbb79296560ace85d39e948aba34ce5c19a7d4711a  ref_euroc_v101_f500_752x480.atb
SUMS
  aliked-lg)
    LG_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8 AL_COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
    OUT="$MODELS/aliked_lg_ref" W="$UP/weights_aliked"
    pin_clone "$UP/lightglue_src" https://github.com/cvg/LightGlue.git "$LG_COMMIT"
    fetch_pinned "$W/aliked-n16.pth" "https://github.com/Shiaoming/ALIKED/raw/$AL_COMMIT/models/aliked-n16.pth" \
      5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2
    fetch_pinned "$W/aliked_lightglue.pth" \
      https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth \
      d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
    dnn_image
    rm -rf "$OUT" && mkdir -p "$OUT"
    m=/euroc/MH_01_easy/mav0/cam0/data v=/euroc/V1_01_easy/mav0/cam0/data
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/python/model_tools:/tools:ro" \
      -v "$UP/lightglue_src:/lg:ro" -v "$W:/w:ro" -v "$EUROC:/euroc:ro" -v "$OUT:/out" \
      "$DNN_IMAGE" python -I /tools/aliked_lightglue_reference.py --upstream /lg --weights /w --out /out \
      --size 752x480 --k 1024 --k 2048 \
      --image mh01_f200=$m/1403636589713555456.png --image mh01_f205=$m/1403636589963555584.png \
      --image v101_a=$v/1403715282262142976.png --image v101_b=$v/1403715384262142976.png \
      --pair mh01_f200,mh01_f205 --pair v101_a,v101_b --upstream-commit "$LG_COMMIT" --aliked-commit "$AL_COMMIT"
    # Pinned 2026-10-21 (SMOKE §4s-2): the export the D63 numbers were measured with.
    check_pinned "$OUT" <<'SUMS' ;;
cc5bf19d9a8d629dd474f619128bd330362e8d876889a9bbd2270f0f1a45f575  aliked_n16_k1024_752x480.onnx
ee9bc54897a91c6ee782d2429f4e35824c76cdc676eaf69bfa480e039958fb6f  aliked_n16_k2048_752x480.onnx
7f4a0533db6d836d666076ba39bf92eb701dbdbcacd8bb5fa719d561f685a9e0  input_mh01_f200.png
717a5b8f4d5ae180ff3f20e80ddfa0c9ebfa94962ce70f338ede003686e0d818  input_mh01_f205.png
85231821eed56bb52fb778b0f711ecdc117f0f8502e66944d83da44e0e728726  input_v101_a.png
a1fc6321fc43f2204ad7d6f285b26e68254545a06324750671979ebcacfba314  input_v101_b.png
51906eca1158a1467b2306777ca0ac48a09f26c23dda72d2ee7b02ba54def83a  lightglue_aliked.onnx
cace65972b596119c996adebc678da7c9fd4646ac7115a1f1d7a911af6960a8f  ref_img_mh01_f200_k1024.atb
a0b1bd0569a5e0f97596bacca630d8b41b4e22e1282162bbba1895d552df2b47  ref_img_mh01_f200_k2048.atb
228eb4826ea06edc2c441252bb4e2b2a9fe4b9640c2bbd7262072b333628d711  ref_img_mh01_f205_k1024.atb
1ea7fc66924880101f4af353b8de4d24c9896602f1a47657be6409ff2c00aeca  ref_img_mh01_f205_k2048.atb
bb573f5ec92a16207f2badab3c066b34402a4415614cf54c844dc9c28545157a  ref_img_v101_a_k1024.atb
44f07ef4ed7f78443d70ae0e91c49700c274289baaaac2ff4bd83ecd98c99db7  ref_img_v101_a_k2048.atb
61283c2b5f23f0cebb51c0420a2a948646feebba68356c7b6e683e95702b5b98  ref_img_v101_b_k1024.atb
d0ac273449ca82b2a81d857e023fec83e8ab997ff23600ea3d6cf1f0ea9bbc88  ref_img_v101_b_k2048.atb
b549d386ce45e331212059cfff63d17fed4d9856386c9da258d234753d207086  ref_pair_mh01_f200_mh01_f205_k1024.atb
738116773fc5059843cb96dd2f897a294a7b77c2973c3239b54f332d069434da  ref_pair_mh01_f200_mh01_f205_k2048.atb
cf56fffdd1a36b4b83ea99c329b3d48d346a0d0c0f162427bb77f973c531c00f  ref_pair_v101_a_v101_b_k1024.atb
11dc74504b470a2294f57851d9ded512f68c368bf0bd39c9798eb9c36a146e6f  ref_pair_v101_a_v101_b_k2048.atb
SUMS
  eigenplaces-sim)
    EP_COMMIT=a2969f71d5ea31017443af490b15273ca4c50af1 OUT="$MODELS/eigenplaces_sim_ref"
    pin_clone "$UP/eigenplaces_src" https://github.com/gmberton/EigenPlaces.git "$EP_COMMIT"
    fetch_pinned "$UP/eigenplaces_weights/ResNet50_256_eigenplaces.pth" \
      https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_256_eigenplaces.pth \
      2923c10295563ea15efa1571b499c67a7aab4d1ee6e19eba5e8b92756dc9d7e7
    sim_frames
    dnn_image
    rm -rf "$OUT" && mkdir -p "$OUT"
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/python/model_tools:/tools:ro" \
      -v "$UP/eigenplaces_src:/ep:ro" -v "$UP/eigenplaces_weights:/w:ro" -v "$SIMF:/sim:ro" \
      -v "$OUT:/out" "$DNN_IMAGE" python -I /tools/eigenplaces_reference.py --upstream /ep \
      --weights /w/ResNet50_256_eigenplaces.pth --out /out --size 1280x720 --size 640x360 \
      --rgb-image sim_a0_f200=/sim/agent0_f200.png --rgb-image sim_a0_f700=/sim/agent0_f700.png \
      --upstream-commit "$EP_COMMIT"
    # Pinned 2026-10-22 (L3, perception.md §14).
    check_pinned "$OUT" <<'SUMS' ;;
50e78c74f59e432244d83250077122f0b54a162461f0606cfe540dca8e5d04a6  eigenplaces_r50_256_1280x720.onnx
60b2c65af30e3eb742010111c6f4714f29b4c67035e3a538917b8a38e6f2e555  eigenplaces_r50_256_640x360.onnx
a80f6ee4c27e120c1c6629fe53d3078841fe9c8f2bc9035006231ac14e4a17fc  eigenplaces_r50_layer4_1280x720.onnx
7bee21c20ff15081f3c8dfeca19d91d5e57ab216f0868cf4a57b21062872c035  eigenplaces_r50_layer4_640x360.onnx
785a83793d65eb9935791dfc712102f2b281f4d5b140d09f3b5867ea7dafe8fb  head.atb
c4dc66bb6b5c3a586132c56b494e80e0ffbdc25837bac681c8ee00f5a98e1758  input_sim_a0_f200.png
f5d1895903fa9d7289b9a9e9524b6711a90844c1a39916cc6cff9e9a9837f909  input_sim_a0_f700.png
7fbed799edb34640e868a1b34e0e6eb91144af5549c635c0a86d3a199a3e5c53  ref_sim_a0_f200_1280x720.atb
c205d00a2a2141d057cab4070bdce2ca755a7d49c7d30590245e9721b3d05ca1  ref_sim_a0_f200_640x360.atb
ac6a41399f25753a5cf5ee4b35a4ed593077d5b6c2dd0070da380f43ca182fe1  ref_sim_a0_f700_1280x720.atb
617cc0dfdb41e890ca8b007339af6c9ce3bfcc8b51148138f51c9934082b2a58  ref_sim_a0_f700_640x360.atb
SUMS
  aliked-lg-sim)
    LG_COMMIT=eb42fee2d71449efb0aa5c10549752b5d75384d8 AL_COMMIT=683d7c65197395c0b3f01ebe76e1084a27e73a65
    OUT="$MODELS/aliked_lg_sim_ref" W="$UP/weights_aliked"
    pin_clone "$UP/lightglue_src" https://github.com/cvg/LightGlue.git "$LG_COMMIT"
    fetch_pinned "$W/aliked-n16.pth" "https://github.com/Shiaoming/ALIKED/raw/$AL_COMMIT/models/aliked-n16.pth" \
      5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2
    fetch_pinned "$W/aliked_lightglue.pth" \
      https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth \
      d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503
    sim_frames
    dnn_image
    rm -rf "$OUT" && mkdir -p "$OUT"
    docker run --rm --network none -u "$(id -u):$(id -g)" -v "$PWD/python/model_tools:/tools:ro" \
      -v "$UP/lightglue_src:/lg:ro" -v "$W:/w:ro" -v "$SIMF:/sim:ro" -v "$OUT:/out" \
      "$DNN_IMAGE" python -I /tools/aliked_lightglue_reference.py --upstream /lg --weights /w --out /out \
      --size 1280x720 --k 1024 \
      --rgb-image sim_a0_f200=/sim/agent0_f200.png --rgb-image sim_a0_f205=/sim/agent0_f205.png \
      --rgb-image sim_a0_f700=/sim/agent0_f700.png --rgb-image sim_a0_f710=/sim/agent0_f710.png \
      --pair sim_a0_f200,sim_a0_f205 --pair sim_a0_f700,sim_a0_f710 \
      --upstream-commit "$LG_COMMIT" --aliked-commit "$AL_COMMIT"
    # Pinned 2026-10-22 (L3, perception.md §14). lightglue_aliked.onnx is byte-identical to the EuRoC export's.
    check_pinned "$OUT" <<'SUMS' ;;
98a1a3b43acca1d5e6ccb2b93d6eb1a74c65c9fa174afc53c24e6cbbdcbaab0a  aliked_n16_k1024_1280x720.onnx
c4dc66bb6b5c3a586132c56b494e80e0ffbdc25837bac681c8ee00f5a98e1758  input_sim_a0_f200.png
fb12ec0d6bf28cf64efe1133a8ceb959249799b8b65a64c4c9c63bd81eef6aa5  input_sim_a0_f205.png
f5d1895903fa9d7289b9a9e9524b6711a90844c1a39916cc6cff9e9a9837f909  input_sim_a0_f700.png
3b53ad0e3dabaf6c1e632ba5f8e4f966e6bae62cae0c14fe61bd6c8c2c233c93  input_sim_a0_f710.png
51906eca1158a1467b2306777ca0ac48a09f26c23dda72d2ee7b02ba54def83a  lightglue_aliked.onnx
7f4a8a861f9af694ea970ee0f395d8c478130850727786fef9504c4a428863c6  ref_img_sim_a0_f200_k1024.atb
2db777728842c08b1ab3024b2625a5293702ffc364d05233d6760c1ee74ea54f  ref_img_sim_a0_f205_k1024.atb
c09c5c6b451c8d1076fcbcd0eae294c4b910516dadfa4fdd683150436271012e  ref_img_sim_a0_f700_k1024.atb
f0fed01a08abf3eb7d17be5c0fb0d0959735a790912892492b9c8d308d2c3173  ref_img_sim_a0_f710_k1024.atb
aefc96bacb0ce4886f2eec143c50da31d608a9d6206eb37680985736a2a5eb19  ref_pair_sim_a0_f200_sim_a0_f205_k1024.atb
7fa6804373ee1295e5e0e25df3fb31c06926017400a226f9f83f3fbb70d531fe  ref_pair_sim_a0_f700_sim_a0_f710_k1024.atb
SUMS
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
  *) sed -n '2,25p' "$0"; exit 2 ;;
esac
