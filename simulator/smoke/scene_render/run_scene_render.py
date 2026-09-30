#!/usr/bin/env python3
"""Render a generated scene from a few GT poses (sim image). Usage:
    run_scene_render.py <scene_dir> <out_dir>
Composes a world that <include>s the scene model by relative path, adds static
1280x720 HFOV-90 cameras at: agent 0's start pose; the target room centre looking
+x; the midpoint of the first responder passage looking along it. Saves PNGs and
prints per-frame mean/std, so "all black" (culling/orientation bug) or "flat"
(missing textures) is visible in numbers as well as images.
"""
import json, math, os, shutil, signal, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/work/python")
from dataset_tools.worldgen.png import write_png

scene_dir, out_dir = os.path.abspath(sys.argv[1]), sys.argv[2]
m = json.load(open(os.path.join(scene_dir, "scene.json")))
cams = []
e0 = m["entrances"][0]
cams.append(("start0", e0["start_xy_m"][0], e0["start_xy_m"][1], math.radians(e0["start_yaw_deg"])))
tx, ty = m["target_xy_m"]
cams.append(("target", tx, ty, 0.0))
p = next(p for p in m["topology"]["passages"] if not p["narrow"])
cell = 6.0
ax, ay = (p["a"][0] + 0.5) * cell, (p["a"][1] + 0.5) * cell
bx, by = (p["b"][0] + 0.5) * cell, (p["b"][1] + 0.5) * cell
cams.append(("corridor", (ax + bx) / 2, (ay + by) / 2, math.atan2(by - ay, bx - ax)))

def cam_model(name, x, y, yaw):
    return f'''<model name="cam_{name}"><static>true</static><pose>{x:.4f} {y:.4f} 1.0 0 0 {yaw:.6f}</pose><link name="l">
  <sensor name="c" type="camera"><camera><horizontal_fov>1.5708</horizontal_fov>
    <image><width>1280</width><height>720</height><format>R8G8B8</format></image><clip><near>0.05</near><far>60</far></clip></camera>
    <always_on>1</always_on><update_rate>5</update_rate><topic>render/{name}</topic></sensor></link></model>'''

world = open(os.path.join(scene_dir, "world.sdf")).read()
world = world.replace("  </world>", "\n".join(cam_model(*c) for c in cams) + "\n  </world>")
wpath = os.path.join(scene_dir, "_render_world.sdf")     # beside model.sdf so the relative include resolves
open(wpath, "w").write(world)
topics = [f"/render/{c[0]}" for c in cams]
bag = "/tmp/render_bag"; shutil.rmtree(bag, ignore_errors=True)
procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge",
                           *[f"{t}@sensor_msgs/msg/Image[gz.msgs.Image" for t in topics]],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
         subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", bag, *topics],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
time.sleep(4)
t0 = time.time()
r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", "1000", wpath],
                   capture_output=True, text=True, timeout=1800)
wall = time.time() - t0
time.sleep(3)
for pr in procs: os.killpg(pr.pid, signal.SIGINT)
for pr in procs: pr.wait(timeout=30)
os.remove(wpath)
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
last = {}
while rd.has_next():
    t, d, _ = rd.read_next()
    last[t] = deserialize_message(d, Image)
os.makedirs(out_dir, exist_ok=True)
res = {"gz_rc": r.returncode, "wall_s_for_1s_sim": round(wall, 1),
       "gz_errors": [l for l in r.stderr.splitlines() if "rror" in l][:5], "frames": {}}
for t, msg in sorted(last.items()):
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
    name = t.split("/")[-1]
    write_png(os.path.join(out_dir, f"{name}.png"), img)
    g = img.mean(2)
    res["frames"][name] = {"mean": round(float(g.mean()), 1), "std": round(float(g.std()), 1),
                           "black_frac": round(float((g < 3).mean()), 3)}
print(json.dumps(res, indent=1))
