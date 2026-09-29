# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-sim: Gazebo Harmonic + ROS 2 Jazzy, headless (arch §27.1, §5.10).
#
# amd64 only. Graviton runs no Gazebo (§5.10): rendering-bearing sim runs
# live on g5/g6, record a clean MCAP bag once, and every CPU consumer replays
# the bag. So there is no arm64 variant and no buildx multi-arch here.
#
# ONE image, TWO render paths, chosen at `docker run` time (infra/scripts/sim.sh):
#   - g5/g6: the NVIDIA Container Toolkit injects the driver's EGL vendor
#     library through glvnd when the container has the `graphics` capability.
#   - dev box with no GPU: Mesa's llvmpipe software EGL device.
# The image carries glvnd + Mesa, so both paths resolve through the same
# libEGL and the same Gazebo command line.
#
# REPRODUCIBILITY: the base is pinned by digest, and ROS packages come from a
# DATED SNAPSHOT (snapshots.ros.org/jazzy/2026-09-11) rather than the live
# repo. The live repo keeps only the newest build of each package, so
# `pkg=version` pins break the next time it syncs; a snapshot never changes.
# The exact installed versions are written into the image at build time
# (/opt/ariadne/sim-versions.txt) for the run manifest (§5, §28.2). The image
# DIGEST is the reproducibility handle, per §27.2: a tag is not a version.
# ---------------------------------------------------------------------------
FROM ros@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8
# ^ ros:jazzy-ros-base, resolved 2026-09-29.

ARG ROS_SNAPSHOT=2026-09-11
# Fingerprint of "ROS Snapshot builder <rosbuild@ros.org>", verified against
# keyserver.ubuntu.com and against the snapshot's own InRelease signature.
ARG ROS_SNAPSHOT_KEY_FPR=4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Flaky mirrors and CDN blips are routine on this link. Retries are fine; do
# NOT force IPv4: this host sits behind DNS64/NAT64, where IPv4-only fails.
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries

# Swap the live ROS repo for the dated snapshot. The key is vendored and its
# fingerprint checked here, so a substituted key fails the build instead of
# silently authenticating a different repository.
COPY infra/docker/keys/ros-snapshots.asc /tmp/ros-snapshots.asc
RUN set -e; apt-get update; apt-get install -y --no-install-recommends gnupg; \
    fpr="$(gpg --show-keys --with-colons /tmp/ros-snapshots.asc | awk -F: '/^fpr/{print $10; exit}')"; \
    if [ "$fpr" != "${ROS_SNAPSHOT_KEY_FPR}" ]; then \
      echo "ROS snapshot key fingerprint mismatch: got '$fpr'"; exit 1; fi; \
    gpg --dearmor < /tmp/ros-snapshots.asc > /usr/share/keyrings/ros-snapshots.gpg \
 && rm -f /etc/apt/sources.list.d/ros2.sources /tmp/ros-snapshots.asc \
 && printf 'Types: deb\nURIs: http://snapshots.ros.org/jazzy/%s/ubuntu\nSuites: noble\nComponents: main\nSigned-By: /usr/share/keyrings/ros-snapshots.gpg\n' \
      "${ROS_SNAPSHOT}" > /etc/apt/sources.list.d/ros2-snapshot.sources \
 && rm -rf /var/lib/apt/lists/*

# The bridge and sim integration, NOT the `ros-jazzy-ros-gz` metapackage.
# The metapackage pulls ros_gz_image -> image_transport_plugins -> cv_bridge,
# which installs Ubuntu's OpenCV 4.6 (43 packages, measured 2026-09-29). That
# would put a second OpenCV in the image, against C7's "one OpenCV, built from
# the pinned tag", and make the manifest's opencv_build_hash ambiguous. Camera
# frames cross the bridge as sensor_msgs/Image via parameter_bridge, which
# needs no OpenCV. The subset installs zero OpenCV packages. See DECISIONS D25.
#
# glvnd + Mesa EGL/DRI: the libEGL dispatcher plus the llvmpipe software
# device. On a GPU host the toolkit adds the NVIDIA vendor library next to it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-ros-gz-bridge \
      ros-jazzy-ros-gz-sim \
      ros-jazzy-ros-gz-interfaces \
      ros-jazzy-rosbag2-storage-mcap \
      libglvnd0 libegl1 libegl-mesa0 libgl1-mesa-dri libgles2 \
 && rm -rf /var/lib/apt/lists/*

# Enforced, not just intended: fail the build if any OpenCV slipped in through
# a transitive dependency after a snapshot bump.
RUN if dpkg -l | grep -qiE '^ii +(libopencv|python3-opencv|ros-jazzy-cv-bridge)'; then \
      echo "An OpenCV package entered the sim image; C7 violated:"; \
      dpkg -l | grep -iE '^ii +(libopencv|python3-opencv|ros-jazzy-cv-bridge)'; exit 1; fi

# Record exactly what was installed, for the run manifest. Written as a file
# (machine-readable) and summarized in labels (visible via `docker inspect`).
RUN mkdir -p /opt/ariadne \
 && { echo "ros_snapshot=${ROS_SNAPSHOT}"; \
      dpkg-query -W -f='${Package}=${Version}\n' \
        'ros-jazzy-ros-gz*' 'ros-jazzy-gz-*-vendor' 'ros-jazzy-sdformat-vendor' \
        'ros-jazzy-rosbag2-storage-mcap' 'libegl-mesa0' 'libglvnd0'; \
    } | sort > /opt/ariadne/sim-versions.txt

LABEL org.opencontainers.image.title="ariadne-sim" \
      org.opencontainers.image.description="Gazebo Harmonic + ROS 2 Jazzy, headless EGL (NVIDIA or Mesa llvmpipe)" \
      ariadne.ros.distro="jazzy" \
      ariadne.ros.snapshot="${ROS_SNAPSHOT}" \
      ariadne.gz.release="harmonic"

# Non-root: Gazebo writes ~/.gz caches, and nothing here needs root at run time.
# uid 1000 matches the default dev-box user so bind-mounted output is writable.
RUN (id -u ubuntu >/dev/null 2>&1 && userdel -r ubuntu || true) \
 && useradd -m -u 1000 -s /bin/bash ariadne

COPY --chown=ariadne:ariadne simulator/ /opt/ariadne/simulator/
COPY --chown=ariadne:ariadne infra/scripts/sim_smoke.py infra/scripts/sim_throughput.py /opt/ariadne/bin/

# graphics = the capability that makes the toolkit inject EGL. Harmless on a
# host with no NVIDIA runtime: nothing reads these variables there.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    GZ_SIM_RESOURCE_PATH=/opt/ariadne/simulator/worlds:/opt/ariadne/simulator/models

# The Phase 1 live consumer (degradation applied to live camera frames, §5.9)
# does not exist yet, so the OpenCV-free core is not built into this image yet.
# It lands as a separate build stage with that consumer, so it can never block
# or invalidate the Gazebo layers above.

USER ariadne
WORKDIR /home/ariadne
# ros_entrypoint.sh (from the base) sources /opt/ros/jazzy before any command.
CMD ["python3", "/opt/ariadne/bin/sim_smoke.py"]
