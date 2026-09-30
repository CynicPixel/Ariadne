#!/usr/bin/env python3
"""OBJ texture-mapping probe (SMOKE §4h). Runs inside ariadne/sim with the repo's
python/ mounted at /py. Checks, on a real render, the quad convention documented in
python/dataset_tools/worldgen/mesh.py:
  1. u runs left->right seen from the front (red gradient increases left->right),
  2. v runs bottom->top (green is high at the top: no V-flip),
  3. aspect is preserved (the 64 px checker cells of the texture render square),
  4. a quad seen from its back is culled (right half shows background),
  5. relative mesh/texture URIs resolve against the SDF file's directory.
"""
import json, os, shutil, signal, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/py")
from dataset_tools.worldgen.mesh import Quad, write_obj
from dataset_tools.worldgen.png import write_png

W = "/tmp/uvw"
shutil.rmtree(W, ignore_errors=True)
os.makedirs(f"{W}/meshes"); os.makedirs(f"{W}/textures")

# 512x384 texture = 2.0 m x 1.5 m at 256 px/m. R: column gradient; G: 255 at the
# TOP row falling to 0 at the bottom (so "green high at top" <=> no V-flip);
# B: a 64 px checker (cells must render square if the aspect is right).
h, w = 384, 512
cols = np.arange(w)[None, :].repeat(h, 0)
rows = np.arange(h)[:, None].repeat(w, 1)
tex = np.zeros((h, w, 3), np.uint8)
tex[..., 0] = (cols * 255 // (w - 1)).astype(np.uint8)
tex[..., 1] = (255 - rows * 255 // (h - 1)).astype(np.uint8)
tex[..., 2] = np.where(((cols // 64) + (rows // 64)) % 2 == 0, 255, 0).astype(np.uint8)
write_png(f"{W}/textures/probe.png", tex)

# Camera at (0,0,1) looking +x (gz: +y is image-left). Seen from the camera, the
# front quad's left->right is -y, so eu = (0,-2,0), ev = (0,0,1.5), p0 = its
# bottom-left corner; normal = eu x ev = -x (towards the camera).
write_obj(f"{W}/meshes/front.obj", Quad((3.0, 2.2, 0.25), (0.0, -2.0, 0.0), (0.0, 0.0, 1.5)), "front")
# Same size, right half of the view, wound the other way: normal +x (away).
write_obj(f"{W}/meshes/back.obj", Quad((3.0, -2.2, 0.25), (0.0, 2.0, 0.0), (0.0, 0.0, 1.5)), "back")

def visual(n):
    return f'''<visual name="{n}"><geometry><mesh><uri>meshes/{n}.obj</uri></mesh></geometry>
      <material><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
      <pbr><metal><albedo_map>textures/probe.png</albedo_map><roughness>1</roughness><metalness>0</metalness></metal></pbr>
      </material></visual>'''

open(f"{W}/probe.sdf", "w").write(f'''<?xml version="1.0" ?><sdf version="1.10"><world name="uvprobe">
<physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<scene><ambient>1 1 1 1</ambient><background>0 0 0 1</background><grid>false</grid></scene>
<light type="directional" name="key"><direction>1 0 0</direction><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular><cast_shadows>false</cast_shadows></light>
<model name="quads"><static>true</static><link name="l">{visual("front")}{visual("back")}</link></model>
<model name="cam"><static>true</static><pose>0 0 1 0 0 0</pose><link name="l">
  <sensor name="c" type="camera"><camera><horizontal_fov>1.5708</horizontal_fov>
    <image><width>1280</width><height>720</height><format>R8G8B8</format></image><clip><near>0.05</near><far>50</far></clip></camera>
    <always_on>1</always_on><update_rate>10</update_rate><topic>uv/cam</topic></sensor></link></model>
</world></sdf>''')

bag = "/tmp/uvbag"; shutil.rmtree(bag, ignore_errors=True)
procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", "/uv/cam@sensor_msgs/msg/Image[gz.msgs.Image"],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
         subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", bag, "/uv/cam"],
                          start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
time.sleep(4)
r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", "500", f"{W}/probe.sdf"],
                   capture_output=True, text=True, timeout=600)
time.sleep(3)
for p in procs: os.killpg(p.pid, signal.SIGINT)
for p in procs: p.wait(timeout=30)

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
img = None
while rd.has_next():
    t, d, _ = rd.read_next()
    m = deserialize_message(d, Image)
    img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
errs = [l for l in r.stderr.splitlines() if "rror" in l][:5]
out = {"gz_rc": r.returncode, "gz_errors": errs, "frame": img is not None}
if img is not None:
    write_png("/out/uv_probe.png", img)
    f, cx, cy = 640.0, 639.5, 359.5
    # Projected front-quad box: y in [0.2, 2.2] at x = 3 -> columns; z in [0.25, 1.75].
    c_lo, c_hi = int(cx - f * 2.2 / 3) + 8, int(cx - f * 0.2 / 3) - 8
    r_lo, r_hi = int(cy - f * 0.75 / 3) + 8, int(cy + f * 0.75 / 3) - 8
    q = img[r_lo:r_hi, c_lo:c_hi].astype(float)
    red_cols = q[..., 0].mean(0); green_rows = q[..., 1].mean(1)
    out["red_left_right"] = [round(red_cols[:20].mean(), 1), round(red_cols[-20:].mean(), 1)]
    out["green_top_bottom"] = [round(green_rows[:20].mean(), 1), round(green_rows[-20:].mean(), 1)]
    out["u_left_to_right"] = bool(np.corrcoef(np.arange(len(red_cols)), red_cols)[0, 1] > 0.99)
    out["v_bottom_to_top"] = bool(np.corrcoef(np.arange(len(green_rows)), green_rows)[0, 1] < -0.99)
    # Aspect: checker cell size in px along a row and a column of the blue channel.
    b = q[..., 2] > 127
    def run_len(v):
        edges = np.flatnonzero(np.diff(v.astype(int)) != 0)
        return float(np.median(np.diff(edges))) if len(edges) > 2 else float("nan")
    out["checker_cell_px_h_v"] = [run_len(b[b.shape[0] // 2]), run_len(b[:, b.shape[1] // 2])]
    # Back quad region (mirror image in y): should be background (black).
    bc_lo, bc_hi = int(cx + f * 0.2 / 3) + 8, int(cx + f * 2.2 / 3) - 8
    out["back_region_mean"] = round(float(img[r_lo:r_hi, bc_lo:bc_hi].mean()), 2)
print(json.dumps(out, indent=1))
