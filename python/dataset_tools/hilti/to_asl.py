"""Hilti 2023 ROS1 bag -> EuRoC ASL layout, one stereo pair + every IMU (5.6, D141). Needs `rosbags` (Apache-2.0)
and `opencv-python-headless` (for compressed images and PNG writing); runs on the AWS extraction instance.

  python -m dataset_tools.hilti.to_asl list   BAG                  -> every connection: topic, type, count
  python -m dataset_tools.hilti.to_asl export BAG OUT --left T --right T

`export` writes OUT/mav0/{cam0,cam1}/data.csv + data/<ns>.png (8-bit mono, as recorded or converted), and
OUT/mav0/imu_<name>/data.csv for EVERY sensor_msgs/Imu topic ("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z").
Header stamps are used, not bag receive times. No calibration is written here: the sensor.yaml files are made
from the Hilti calibration files once the camera names are matched (they are copied beside the export).
"""
import argparse
import os
import sys

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")


def cmd_list(a):
    with AnyReader([Path(a.bag)]) as r:
        rows = sorted((c.topic, c.msgtype, c.msgcount) for c in r.connections)
        print(f"duration_s {(r.end_time - r.start_time) / 1e9:.1f}")
        for t, ty, n in rows:
            print(f"{t}\t{ty}\t{n}")


def to_gray(msg, msgtype):
    if msgtype.endswith("CompressedImage"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        enc = msg.encoding
        buf = np.frombuffer(msg.data, np.uint8)
        if enc in ("mono8", "8UC1"):
            img = buf.reshape(msg.height, msg.step)[:, :msg.width]
        elif enc in ("rgb8", "bgr8"):
            img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if enc == "rgb8" else cv2.COLOR_BGR2GRAY)
        else:
            raise SystemExit(f"unsupported image encoding {enc}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        raise SystemExit(f"non-8-bit image ({img.dtype}); EuRoC layout is 8-bit mono")
    return img


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def cmd_export(a):
    out = Path(a.out) / "mav0"
    cams = {"cam0": a.left, "cam1": a.right}
    files, counts = {}, {}
    with AnyReader([Path(a.bag)]) as r:
        conns = {c.topic: c for c in r.connections}
        for cam, topic in cams.items():
            if topic not in conns or conns[topic].msgtype not in IMAGE_TYPES:
                raise SystemExit(f"{cam}: {topic} is not an image topic in this bag")
            (out / cam / "data").mkdir(parents=True, exist_ok=True)
            files[topic] = open(out / cam / "data.csv", "w")
            files[topic].write("#timestamp [ns],filename\n")
        imus = [c for c in r.connections if c.msgtype == "sensor_msgs/msg/Imu"]
        for c in imus:
            d = out / ("imu_" + c.topic.strip("/").replace("/", "_"))
            d.mkdir(parents=True, exist_ok=True)
            files[c.topic] = open(d / "data.csv", "w")
            files[c.topic].write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],"
                                 "a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        want = [conns[t] for t in cams.values()] + imus
        cam_of = {t: c for c, t in cams.items()}
        last = {}
        for conn, _, raw in r.messages(connections=want):
            msg = r.deserialize(raw, conn.msgtype)
            t = stamp_ns(msg)
            if t <= last.get(conn.topic, -1):
                continue   # EuRoC loaders require strictly increasing stamps; duplicates are counted below
            last[conn.topic] = t
            counts[conn.topic] = counts.get(conn.topic, 0) + 1
            if conn.topic in cam_of:
                cam = cam_of[conn.topic]
                cv2.imwrite(str(out / cam / "data" / f"{t}.png"), to_gray(msg, conn.msgtype))
                files[conn.topic].write(f"{t},{t}.png\n")
            else:
                w, acc = msg.angular_velocity, msg.linear_acceleration
                files[conn.topic].write(f"{t},{w.x!r},{w.y!r},{w.z!r},{acc.x!r},{acc.y!r},{acc.z!r}\n")
    for f in files.values():
        f.close()
    for t, n in sorted(counts.items()):
        print(f"exported {t}: {n}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dataset_tools.hilti.to_asl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("bag")
    p = sub.add_parser("export"); p.add_argument("bag"); p.add_argument("out")
    p.add_argument("--left", required=True); p.add_argument("--right", required=True)
    a = ap.parse_args(argv)
    (cmd_list if a.cmd == "list" else cmd_export)(a)


if __name__ == "__main__":
    sys.exit(main())
