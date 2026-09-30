#!/usr/bin/env python3
"""Capture LIVE bridge-serialized IMU / PoseStamped messages from a verified sim
recording, as byte fixtures for tests/unit/test_ros_cdr.cpp (the CDR encoder test).

These bytes were produced by ros_gz_bridge (rmw_fastrtps) from the KinematicMover's
gz messages, so they exercise what the gen_fixtures.py samples cannot: a frame_id
("agent_0/base_link", 18 bytes with NUL) that forces 2 bytes of alignment padding
before the first float64, and a -0.0 gyro component. The padding in these
captures is zero; the encoder must reproduce them exactly.

Run in the sim image against a recording made by infra/scripts/sim_record.sh:
    docker run --rm -v $PWD:/work -w /work --entrypoint bash ariadne/sim:dev -lc \
      'source /opt/ros/jazzy/setup.bash && python3 tests/fixtures/ros_cdr/capture_live_samples.py \
       build/bags/ci_tiny_0_short_d39/sensors'
"""
import sys

import rosbag2_py

WANT = {("/ariadne/agent_0/imu", 5_000_000): "kLiveImuFirst",
        ("/ariadne/agent_0/imu", 3_000_000_000): "kLiveImuMoving",
        ("/ariadne/agent_0/gt/pose", 3_000_000_000): "kLivePoseMoving"}


def main(uri):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    rd.set_filter(rosbag2_py.StorageFilter(topics=sorted({t for t, _ in WANT})))
    got = {}
    while rd.has_next():
        topic, data, _ = rd.read_next()
        sec, nsec = int.from_bytes(data[4:8], "little", signed=True), int.from_bytes(data[8:12], "little")
        key = (topic, sec * 1_000_000_000 + nsec)
        if key in WANT:
            got[WANT[key]] = bytes(data)
    missing = set(WANT.values()) - set(got)
    if missing:
        raise SystemExit(f"not in bag: {sorted(missing)}")
    out = ["// CAPTURED by tests/fixtures/ros_cdr/capture_live_samples.py from a verified sim",
           "// recording (ros_gz_bridge serialization of the KinematicMover's messages). Do not edit.",
           "#pragma once", "#include <cstdint>", "#include <vector>", "namespace ros_cdr_live {"]
    for name in sorted(got):
        out.append(f"inline const std::vector<uint8_t> {name} = {{{', '.join(f'0x{x:02x}' for x in got[name])}}};")
    out.append("}  // namespace ros_cdr_live")
    open("tests/fixtures/ros_cdr/ros_cdr_live_samples.inc", "w").write("\n".join(out) + "\n")
    print("ok", {k: len(v) for k, v in sorted(got.items())})


if __name__ == "__main__":
    main(sys.argv[1])
