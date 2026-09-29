# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-core (arch §27.1): the Ariadne C++ tree built against the pinned
# OpenCV 5 of opencv-base. Build context = the repo root.
#
#   dev  toolchain + OpenCV, no sources. The repo is bind-mounted at /work and
#        built into build/docker/ (never the host's build/: CMakeCache holds
#        absolute paths and compiler ids, and sharing one tree corrupts both).
#   ci   (default) sources copied in, configured with OpenCV REQUIRED, built,
#        the full test suite run, ariadne_identity installed and its output
#        baked into /opt/ariadne/opencv_identity.env.
#
# NOT here yet: the ONNX models §27.1 lists. Nothing loads one until XFeat
# lands in Phase 2, and a model with no consumer is only an unverified blob.
# ---------------------------------------------------------------------------

# The base is an ARG so CI can pass the ECR reference BY DIGEST (§27.2) while a
# local build uses the tag of the locally built layer.
ARG OPENCV_BASE=ariadne/opencv-base:5.0.0
FROM ${OPENCV_BASE} AS dev

# ccache: here rather than in opencv-base so adding it did not force a rebuild
# of the multi-hour OpenCV layer. binutils provides readelf for the RPATH check.
RUN apt-get update && apt-get install -y --no-install-recommends ccache binutils \
 && rm -rf /var/lib/apt/lists/*

# Resolve OpenCV through the loader cache, never through RPATH. The COOL
# benchmark (Invariant 2) switches OpenCV with LD_LIBRARY_PATH only. A binary
# carrying DT_RPATH=/opt/opencv5/lib would ignore that switch (RPATH is searched
# BEFORE LD_LIBRARY_PATH) and silently benchmark stock OpenCV against itself.
# ld.so.cache is searched AFTER LD_LIBRARY_PATH, so the switch always wins.
RUN echo /opt/opencv5/lib > /etc/ld.so.conf.d/opencv5.conf && ldconfig

# uid 1000 so files written through the dev bind mount belong to the host user.
# ubuntu:24.04 ships a user `ubuntu` at 1000; it is replaced, as in the sim image.
RUN (id -u ubuntu >/dev/null 2>&1 && userdel -r ubuntu || true) \
 && useradd -m -u 1000 -s /bin/bash ariadne \
 && mkdir -p /opt/ariadne && chown ariadne:ariadne /opt/ariadne

ENV CCACHE_DIR=/home/ariadne/.ccache \
    CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache
USER ariadne
WORKDIR /work

# ---------------------------------------------------------------------------
FROM dev AS ci

# Provenance only the host knows. The launcher passes them; if it does not,
# the manifest says UNRECORDED rather than guessing.
ARG ARIADNE_GIT_COMMIT=UNRECORDED
ARG ARIADNE_GIT_DIRTY=true
ARG BUILD_JOBS=4
ENV ARIADNE_GIT_COMMIT=${ARIADNE_GIT_COMMIT} \
    ARIADNE_GIT_DIRTY=${ARIADNE_GIT_DIRTY}

# The sources are BIND-MOUNTED (read-only) for this one step, not COPYed: a
# COPY layer stays in the image even if a later step deletes it, so everything
# the build context carries would ship with the image. The build is
# out-of-source, and test_unit reads configs from /src while the mount exists.
#
# Two cache mounts keep rebuilds cheap without leaking into the image:
#   ccache       compiled objects, keyed on preprocessed source + flags
#   fetchcontent the pinned third-party checkouts (Eigen, FlatBuffers, zstd,
#                yaml-cpp, MCAP, GoogleTest), so a source edit does not re-clone
# CMAKE_SKIP_RPATH: no RPATH or RUNPATH on anything we build (see ldconfig
# above); the readelf step below turns that into an enforced property.
RUN --mount=type=bind,target=/src \
    --mount=type=cache,target=/home/ariadne/.ccache,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/ariadne/.fetchcontent,uid=1000,gid=1000 \
    cmake -S /src -B /home/ariadne/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DARIADNE_REQUIRE_OPENCV=ON \
      -DCMAKE_SKIP_RPATH=ON \
      -DFETCHCONTENT_BASE_DIR=/home/ariadne/.fetchcontent \
      -DCMAKE_INSTALL_PREFIX=/opt/ariadne \
 && cmake --build /home/ariadne/build --parallel ${BUILD_JOBS} \
 && ctest --test-dir /home/ariadne/build --output-on-failure --parallel ${BUILD_JOBS} \
 && cmake --install /home/ariadne/build --component ariadne \
 && for f in /opt/ariadne/bin/* /home/ariadne/build/tests/test_*; do \
      if readelf -d "$f" | grep -qE '\((RPATH|RUNPATH)\)'; then \
        echo "RPATH/RUNPATH found in $f; the COOL loader switch would be ignored"; \
        readelf -d "$f" | grep -E 'RPATH|RUNPATH'; exit 1; fi; done \
 && /opt/ariadne/bin/ariadne_identity > /opt/ariadne/opencv_identity.env \
 && cat /opt/ariadne/opencv_identity.env \
 && rm -rf /home/ariadne/build

ENV PATH=/opt/ariadne/bin:${PATH}
WORKDIR /home/ariadne
CMD ["ariadne_identity"]
