#!/usr/bin/env python3
"""Measure whether this host can record the real rig (arch 5.3) to MCAP.

Runs INSIDE the ariadne-sim image:
    infra/scripts/sim.sh python3 /opt/ariadne/bin/sim_throughput.py

It runs rig_throughput.sdf for a FIXED number of physics iterations (so sim
duration is exact), bridges stereo + IMU + /clock to ROS 2, records them with
rosbag2 MCAP, then reads the bag back and reports:

  realtime factor   sim seconds per wall second (1.0 = real time)
  completeness      frames recorded vs update_rate x sim duration, per sensor
  gaps              largest stamp gap vs nominal period, per sensor
  stereo pairing    fraction of left frames with an identically-stamped right

A slow renderer may either SLOW the sim (RTF < 1, frames complete) or DROP
frames (RTF ~ 1, frames missing). Both are measured, neither is assumed; for
record-once bags the first is acceptable and the second is not.
"""
import json
import os
import signal
import subprocess
import sys
import time

WORLD = "/opt/ariadne/simulator/worlds/rig_throughput.sdf"
SIM_SECONDS = float(os.environ.get("ARIADNE_TP_SIM_SECONDS", "10"))
STEP = 0.001
TOPICS = {"/rig/left": ("sensor_msgs/msg/Image", "gz.msgs.Image", 20.0),
          "/rig/right": ("sensor_msgs/msg/Image", "gz.msgs.Image", 20.0),
          "/rig/imu": ("sensor_msgs/msg/Imu", "gz.msgs.IMU", 200.0)}
BAG = "/tmp/rig_bag"
OUT = os.environ.get("ARIADNE_SMOKE_OUT", "")


def log(m):
    print(f"[throughput] {m}", flush=True)


def main():
    iterations = int(round(SIM_SECONDS / STEP))
    log(f"render mode {os.environ.get('ARIADNE_RENDER_MODE', '?')}, "
        f"{SIM_SECONDS:.1f} sim-s ({iterations} iterations)")
    bridge_args = [f"{t}@{ros}[{gz}" for t, (ros, gz, _) in TOPICS.items()]
    bridge_args.append("/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock")
    procs = []
    try:
        procs.append(subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", *bridge_args],
                                      start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        procs.append(subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", BAG, *TOPICS.keys()],
                                      start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        time.sleep(4)  # let the bridge and recorder subscribe before sim starts
        # Bridge and recorder are already subscribed; run exactly N physics steps.
        t0 = time.monotonic()
        gz = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering",
                             "--iterations", str(iterations), WORLD],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600)
        wall = time.monotonic() - t0
        log(f"gz exited {gz.returncode} after {wall:.1f} wall-s")
        time.sleep(3)  # let the recorder drain its queue
    finally:
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=BAG, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    stamps = {t: [] for t in TOPICS}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in stamps:
            msg = deserialize_message(data, get_message(types[topic]))
            stamps[topic].append(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)

    report = {"sim_seconds": SIM_SECONDS, "wall_seconds": round(wall, 2),
              "realtime_factor": round(SIM_SECONDS / wall, 3), "sensors": {}}
    for t, (_, _, hz) in TOPICS.items():
        s = sorted(stamps[t])
        expected = int(SIM_SECONDS * hz)
        gaps = [(b - a) / 1e9 for a, b in zip(s, s[1:])]
        report["sensors"][t] = {
            "expected": expected, "recorded": len(s),
            "completeness": round(len(s) / expected, 3) if expected else None,
            "nominal_period_s": round(1 / hz, 4),
            "max_gap_s": round(max(gaps), 4) if gaps else None,
            "duplicate_stamps": len(s) - len(set(s))}
    left, right = set(stamps["/rig/left"]), set(stamps["/rig/right"])
    report["stereo_pairing"] = round(len(left & right) / len(left), 3) if left else None
    print(json.dumps(report, indent=2), flush=True)
    if OUT and os.path.isdir(OUT):
        with open(os.path.join(OUT, "rig_throughput.json"), "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
