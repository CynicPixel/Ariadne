# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne opencv-base — the pinned OpenCV 5 layer (arch §27.1, C7).
#
# This layer exists on its own, separate from the Ariadne code, for one reason:
# it is expensive to build and almost never changes. CI rebuilds it only when
# OPENCV_TAG moves, and every other image FROMs it. Multi-arch by construction
# (buildx amd64 + arm64) because the COOL benchmark of §26.3 is Arm-vs-Arm and
# must run the *same* source tree the amd64 dev box runs.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04

# Pinned together, deliberately: C7 requires opencv and opencv_contrib at the
# SAME tag, else the contrib ABI can skew against core.
ARG OPENCV_TAG=5.0.0
ARG BUILD_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain per §27.4 (OpenCV 5 requires C++17). zlib/png/jpeg/tiff are the
# imgcodecs backends we actually use; we deliberately do NOT pull GTK/Qt — the
# pipeline is headless everywhere (§5.10) and highgui is disabled below.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ cmake ninja-build git ca-certificates pkg-config \
      libzstd-dev zlib1g-dev libpng-dev libjpeg-turbo8-dev libtiff-dev \
      python3 python3-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch ${OPENCV_TAG} https://github.com/opencv/opencv.git opencv \
 && git clone --depth 1 --branch ${OPENCV_TAG} https://github.com/opencv/opencv_contrib.git opencv_contrib \
 && git -C opencv rev-parse HEAD > /opt-opencv-core.sha \
 && git -C opencv_contrib rev-parse HEAD > /opt-opencv-contrib.sha
# NOTE: 5.0.0 is an ANNOTATED tag, so `git ls-remote --refs` reports the tag
# OBJECT sha, not the commit. `rev-parse HEAD` above records the COMMIT, which
# is what a §28.2 manifest must pin. Peeled commits for 5.0.0:
#   opencv         40738fb16ceddb5fb3fea747585f7ce6abb0605b
#   opencv_contrib 755e50675d97db9b7d449d8bd6b09888646f6c6e

# BUILD_LIST is the set of modules Ariadne actually links, using the REAL
# OpenCV 5 module names verified against the 5.0.0 source tree (arch C3
# smoke-test item 1: the release layout is geometry/calib/stereo/ptcloud, NOT
# a single "3d" module — the 3d.hpp *header* inside geometry is what third-party
# write-ups were describing). Restricting the list cuts build time roughly in
# half; CMake pulls in transitive module deps automatically.
#   geometry -> findEssentialMat/recoverPose/solvePnPRansac/triangulatePoints
#   calib    -> calibrateCamera        stereo -> stereoRectify/SGBM
#   features -> ALIKED/DISK/LightGlueMatcher/ANNIndex (§7.2)
#   ptcloud  -> TSDF/HashTSDF volumes (§20.1)
#   ximgproc -> DisparityWLSFilter (§9.3)   quality -> PSNR/SSIM harness (§30)
#   xphoto/optflow -> radiometric + tracking baselines (C7)
# The contrib set is EXACTLY the four modules C7 names. xfeatures2d is
# deliberately excluded: nothing in the architecture uses SURF/BRIEF/FREAK/
# DAISY, and it triggers boostdesc/vgg blob downloads at CONFIGURE time that
# stall the build behind a slow third-party host.
RUN cmake -S opencv -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv5 \
      -DOPENCV_EXTRA_MODULES_PATH=/src/opencv_contrib/modules \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,features,flann,dnn,video,geometry,calib,stereo,ptcloud,photo,objdetect,ximgproc,xphoto,optflow,quality \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
      -DBUILD_opencv_python3=OFF -DBUILD_opencv_apps=OFF \
      -DWITH_GTK=OFF -DWITH_QT=OFF -DHIGHGUI_ENABLE_PLUGINS=OFF \
      -DWITH_CUDA=OFF \
      -DWITH_IPP=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DBUILD_SHARED_LIBS=ON \
 && cmake --build build --parallel ${BUILD_JOBS} \
 && cmake --install build \
 && mv /opt-opencv-core.sha /opt-opencv-contrib.sha /opt/opencv5/ \
 && rm -rf /src /root/.cache
