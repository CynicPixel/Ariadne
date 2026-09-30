#!/usr/bin/env python3
"""Golden samples of worldgen scripts for tests/numerical/test_scripted.cpp.

    infra/scripts/worldgen.sh python tests/fixtures/scripted/gen_scripted_golden.py
(or any python with PYTHONPATH=python). For each case: the script's segments (as the
mover receives them) and samples of pose/rate/acc and the ideal IMU at segment
boundaries +-1 us and at uniform times. Floats are written with repr (exact round trip).
"""
import json
import math

from dataset_tools.worldgen import routes
from dataset_tools.worldgen.params import load_preset
from dataset_tools.worldgen.scene import generate

CASES = [("ci_tiny", 0, 0), ("maze_short", 7, 1)]
out = []
for preset, seed, agent in CASES:
    s = generate(load_preset(f"configs/worldgen/{preset}.yaml"), seed)
    sc = s.scripts[agent]
    times = []
    for seg in sc.segments:
        for dt in (-1e-6, 0.0, 1e-6):
            t = seg.t0 + dt
            if 0.0 <= t <= sc.duration:
                times.append(t)
    n = 300
    times += [sc.duration * (i + 0.5) / n for i in range(n)]
    samples = []
    for t in sorted(set(times)):
        pose, rate, acc = sc.pose(t)
        gyro, f = routes.imu(sc, t)
        samples.append({"t": t, "pose": list(pose), "rate": list(rate), "acc": list(acc),
                        "gyro": list(gyro), "accel": list(f)})
    pitched = sum(1 for x in samples if abs(x["pose"][4]) > 1e-6)
    out.append({"case": f"{preset}/seed{seed}/agent{agent}", "gravity": routes.GRAVITY,
                "script": sc.to_json(), "samples": samples, "pitched_samples": pitched})
    print(preset, seed, agent, "segments", len(sc.segments), "samples", len(samples), "pitched", pitched)
with open("tests/fixtures/scripted/scripted_golden.json", "w") as f:
    json.dump(out, f)
