#!/usr/bin/env python3
"""ariadne-sim acceptance test: prove the container renders a camera headlessly.

Runs INSIDE the ariadne-sim image (it is the image's default CMD):

  1. start `gz sim -s -r --headless-rendering` on smoke_camera.sdf (arch §5.10),
  2. bridge the camera to ROS 2 with ros_gz_bridge's parameter_bridge, the same
     path Phase 1 records into MCAP,
  3. receive frames with rclpy and judge them by PIXEL STATISTICS.

Why statistics and not an exit code: a headless render that has lost its GL
context commonly keeps publishing, and what it publishes is uniformly black or
uniformly the clear colour. A frame only passes if it has real spatial
variation AND contains the red and the blue box the world places in view.

Exit 0 = PASS. Writes the first good frame to $ARIADNE_SMOKE_OUT (a PNG) when
that directory is mounted, so a human can look at it too; the first frame
is always written as smoke_first.png, pass or fail.
"""
import os
import signal
import struct
import zlib
import statistics
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Overridable so the negative control (smoke_blank.sdf) runs through the same
# code path and must FAIL on the near-uniform branch.
WORLD = os.environ.get("ARIADNE_SMOKE_WORLD",
                       "/opt/ariadne/simulator/worlds/smoke_camera.sdf")
GZ_TOPIC = "/smoke/camera"
TIMEOUT_S = float(os.environ.get("ARIADNE_SMOKE_TIMEOUT", "180"))
FRAMES_NEEDED = 3
OUT_DIR = os.environ.get("ARIADNE_SMOKE_OUT", "")

# A rendered scene with two saturated boxes on grey has a luminance stddev in
# the tens. A dead frame has ~0. The threshold sits far from both.
MIN_LUMA_STDDEV = 8.0
# Fraction of the frame each box must cover. A correct render measures
# ~0.32 red and ~0.20 blue; the bar sits far below that.
MIN_COLOUR_FRACTION = 0.005


def log(msg):
    print(f"[sim_smoke] {msg}", flush=True)


def analyse(msg):
    """Return (luma_stddev, red_fraction, blue_fraction) for an rgb8 frame."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unexpected encoding {msg.encoding}")
    data = bytes(msg.data)
    r_off, b_off = (0, 2) if msg.encoding == "rgb8" else (2, 0)
    luma, red, blue = [], 0, 0
    n = msg.width * msg.height
    for row in range(msg.height):
        base = row * msg.step
        for col in range(msg.width):
            px = base + 3 * col
            r, g, b = data[px + r_off], data[px + 1], data[px + b_off]
            luma.append(0.299 * r + 0.587 * g + 0.114 * b)
            # Dominance, not brightness: shaded faces of a saturated box
            # render near (95, 30, 30) under this light, so a fixed floor of
            # 100 rejected a correct render (measured 2026-09-29).
            if r > 60 and r > 1.8 * max(g, b):
                red += 1
            if b > 60 and b > 1.8 * max(r, g):
                blue += 1
    return statistics.pstdev(luma), red / n, blue / n


def write_png(msg, path):
    """Minimal RGB PNG writer (stdlib only: the image carries no imaging libs)."""
    data = bytes(msg.data)
    raw = bytearray()
    for row in range(msg.height):
        line = data[row * msg.step: row * msg.step + 3 * msg.width]
        if msg.encoding == "bgr8":
            line = bytes(v for i in range(0, len(line), 3)
                         for v in (line[i + 2], line[i + 1], line[i]))
        raw += b"\x00" + line          # filter type 0 per scanline

    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body
                + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", msg.width, msg.height,
                                           8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
        f.write(chunk(b"IEND", b""))


class Collector(Node):
    def __init__(self):
        super().__init__("ariadne_sim_smoke")
        self.frames = []
        self.create_subscription(Image, GZ_TOPIC, self.frames.append, 10)


def main():
    render_mode = os.environ.get("ARIADNE_RENDER_MODE", "unknown")
    log(f"render mode: {render_mode}")
    try:
        with open("/opt/ariadne/sim-versions.txt") as f:
            for line in f:
                if "gz-sim-vendor" in line or "snapshot" in line:
                    log(f"version: {line.strip()}")
    except OSError:
        log("WARNING: /opt/ariadne/sim-versions.txt missing")

    procs = []
    try:
        procs.append(subprocess.Popen(
            ["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", WORLD],
            start_new_session=True))
        procs.append(subprocess.Popen(
            ["ros2", "run", "ros_gz_bridge", "parameter_bridge",
             f"{GZ_TOPIC}@sensor_msgs/msg/Image[gz.msgs.Image"],
            start_new_session=True))

        rclpy.init()
        node = Collector()
        deadline = time.monotonic() + TIMEOUT_S
        good, seen = 0, 0
        last_std = None
        while time.monotonic() < deadline and good < FRAMES_NEEDED:
            rclpy.spin_once(node, timeout_sec=0.5)
            for p in procs:
                if p.poll() is not None:
                    log(f"FAIL: {p.args[0]} exited early with code {p.returncode}")
                    return 1
            while node.frames:
                msg = node.frames.pop(0)
                seen += 1
                std, red, blue = analyse(msg)
                last_std = std
                # Timing diagnostic: sim-time stamp vs wall clock. Bag timing
                # is the product at Phase 1.3, so the frame rate must be
                # explained, not just observed.
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if seen <= 10:
                    log(f"timing frame {seen}: stamp={stamp:.3f}s "
                        f"wall={time.monotonic():.3f}s")
                ok = (std >= MIN_LUMA_STDDEV and red >= MIN_COLOUR_FRACTION
                      and blue >= MIN_COLOUR_FRACTION)
                if seen == 1 and OUT_DIR and os.path.isdir(OUT_DIR):
                    # Always keep the first frame, pass or fail: a rejected
                    # frame is exactly the one a human needs to look at.
                    write_png(msg, os.path.join(OUT_DIR, "smoke_first.png"))
                log(f"frame {seen}: {msg.width}x{msg.height} {msg.encoding} "
                    f"luma_std={std:.1f} red={red:.3f} blue={blue:.3f} "
                    f"{'OK' if ok else 'REJECT'}")
                if ok:
                    if good == 0 and OUT_DIR and os.path.isdir(OUT_DIR):
                        path = os.path.join(OUT_DIR, "smoke_pass.png")
                        write_png(msg, path)
                        log(f"wrote {path}")
                    good += 1

        if good >= FRAMES_NEEDED:
            log(f"SIM_SMOKE=PASS ({good} good frames, render mode {render_mode})")
            return 0
        if seen == 0:
            log(f"SIM_SMOKE=FAIL: no frames within {TIMEOUT_S:.0f}s "
                "(render engine did not start, or the bridge is not forwarding)")
        elif last_std is not None and last_std < MIN_LUMA_STDDEV:
            log(f"SIM_SMOKE=FAIL: {seen} frames received, all near-uniform "
                f"(luma_std={last_std:.1f}): the classic headless black frame")
        else:
            log(f"SIM_SMOKE=FAIL: {seen} frames received with real structure "
                "but without the expected red and blue boxes: the scene or the "
                "camera pose is wrong, not the renderer")
        return 1
    finally:
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
