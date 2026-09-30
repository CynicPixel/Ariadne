#!/usr/bin/env python3
"""RTF and MCAP size for two full rigs in a REAL generated scene (SMOKE §4j).
Usage (sim image, repo at /work): run_scene_rtf.py <scene_dir> <sim_seconds> <depth_w> <depth_h>
Each rig: stereo 1280x720 R8G8B8 + depth (given size) at 20 Hz, circling r=1.0 m around
a room's cell centre (render cost only; collisions irrelevant). All scene lights on.
"""
import glob, json, os, shutil, signal, subprocess, sys, time
scene_dir, sim_s, dw, dh = os.path.abspath(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
m = json.load(open(os.path.join(scene_dir, "scene.json")))
rooms = sorted(m["topology"]["rooms"].keys())[:2]
centres = [((int(k.split(",")[0]) + 0.5) * 6.0, (int(k.split(",")[1]) + 0.5) * 6.0) for k in rooms]
subprocess.run("mkdir -p /tmp/b && cd /tmp/b && cmake -S /work/simulator/smoke/kinematic_probe -B . >/dev/null && make -s",
               shell=True, check=True)
def rig(k, cx, cy):
    cam = lambda n, y: f'''<sensor name="{n}" type="camera"><pose>0 {y} 0 0 0 0</pose><camera><horizontal_fov>1.5708</horizontal_fov>
      <image><width>1280</width><height>720</height><format>R8G8B8</format></image><clip><near>0.05</near><far>60</far></clip></camera>
      <always_on>1</always_on><update_rate>20</update_rate><topic>a{k}/{n}</topic></sensor>'''
    return f'''<model name="a{k}"><static>true</static><pose>{cx} {cy} 1 0 0 0</pose><link name="link">
    {cam("left", 0.05)}{cam("right", -0.05)}
    <sensor name="depth" type="depth_camera"><pose>0 0.05 0 0 0 0</pose><camera><horizontal_fov>1.5708</horizontal_fov>
      <image><width>{dw}</width><height>{dh}</height><format>R_FLOAT32</format></image><clip><near>0.05</near><far>60</far></clip></camera>
      <always_on>1</always_on><update_rate>20</update_rate><topic>a{k}/depth</topic></sensor>
    </link></model>
    <plugin filename="KinematicRing" name="KinematicRing"><model_name>a{k}</model_name><x0>{cx}</x0><y0>{cy}</y0><radius>1.0</radius><speed>0.5</speed></plugin>'''
world = open(os.path.join(scene_dir, "world.sdf")).read()
world = world.replace("  </world>", rig(0, *centres[0]) + rig(1, *centres[1]) + "\n  </world>")
wpath = os.path.join(scene_dir, "_rtf_world.sdf")
open(wpath, "w").write(world)
topics = {f"/a{k}/{n}": "sensor_msgs/msg/Image[gz.msgs.Image" for k in (0, 1) for n in ("left", "right", "depth")}
bag = "/tmp/rtf_bag"; shutil.rmtree(bag, ignore_errors=True)
procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", *[f"{t}@{v}" for t, v in topics.items()]],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
         subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "--storage-preset-profile", "zstd_fast", "-o", bag, *topics],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
time.sleep(5)
env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH="/tmp/b")
t0 = time.monotonic()
r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", str(int(sim_s * 1000)), wpath],
                   env=env, capture_output=True, text=True, timeout=7200)
wall = time.monotonic() - t0
time.sleep(10)
for p in procs: os.killpg(p.pid, signal.SIGINT)
for p in procs: p.wait(timeout=120)
os.remove(wpath)
import rosbag2_py
rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
counts, raw, first, last = {}, {}, None, None
while rd.has_next():
    t, d, ts = rd.read_next()
    counts[t] = counts.get(t, 0) + 1; raw[t] = raw.get(t, 0) + len(d)
size = sum(os.path.getsize(f) for f in glob.glob(f"{bag}/*.mcap"))
print(json.dumps({"scene": m["scene_id"], "sim_s": sim_s, "wall_s_incl_startup": round(wall, 1),
                  "rtf_incl_startup": round(sim_s / wall, 3), "depth": [dw, dh], "lights": len(m["lights"]),
                  "file_MB_per_sim_s": round(size / 1e6 / sim_s, 1),
                  "raw_MB_per_sim_s": {t: round(v / 1e6 / sim_s, 1) for t, v in sorted(raw.items())},
                  "counts": counts, "expected_per_topic": int(sim_s * 20) + 1,
                  "gz_errors": [l for l in r.stderr.splitlines() if "rror" in l][:3]}, indent=1))
