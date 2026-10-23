"""Lamp calibration CLI (5.1, D133).

  render --rig RIG.yaml --out DIR     (inside ariadne/sim: one gz launch per plane pose; DIR/pose_<k>.npy + poses.json)
  fit    --rig RIG.yaml --frames DIR --out CALIB.yaml

The plane is matte grey (SDF diffuse 0.5, roughness 1): gain_A is per that albedo. The lamp pose is the rig's
(sim truth, exactly what the SDF places); the fit calibrates what the renderer adds on top of it: gain, c', lobe,
and whether Lambertian incidence belongs in the model.
"""
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

from . import model

# Distances x tilts: tilts make incidence differ from the lamp angle, which is what separates RID from cos(inc)
# (Park et al., CVPR 2014). Even poses train, odd poses are held out for the cosine decision.
POSES = [dict(d=d, pitch=math.radians(p), yaw=math.radians(y))
         for d in (0.75, 1.25, 2.0, 3.0, 4.5) for (p, y) in ((0, 0), (0, 30), (-25, 0))]


def lamp_in_camera_g(rig):
    """Lamp centre in the left camera's link frame G (body axes; the cameras are not rotated in the body)."""
    return [a - b for a, b in zip(rig["lamp"]["in_body_m"], rig["left_camera_in_body_m"])]


def world(rig, pose):
    lamp = rig["lamp"]
    lx, ly, lz = lamp_in_camera_g(rig)
    c, l, q = lamp.get("attenuation", (0.2, 0.0, 0.05))
    cam = rig["camera"]
    shadows = "true" if lamp.get("cast_shadows", 0) else "false"
    return f'''<?xml version="1.0" ?><sdf version="1.10"><world name="lampcalib">
<physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<scene><ambient>0 0 0 1</ambient><background>0 0 0 1</background><grid>false</grid><shadows>true</shadows></scene>
<model name="plane"><static>true</static><pose>{pose["d"]} 0 1 0 {pose["pitch"]} {pose["yaw"]}</pose><link name="l">
  <visual name="v"><geometry><box><size>{2 * model.PLANE_HALF_THICK} 14 10</size></box></geometry>
  <material><diffuse>0.5 0.5 0.5 1</diffuse><specular>0 0 0 1</specular>
  <pbr><metal><roughness>1</roughness><metalness>0</metalness></metal></pbr></material></visual></link></model>
<model name="cam"><static>true</static><pose>0 0 1 0 0 0</pose><link name="l">
  <sensor name="c" type="camera"><camera><horizontal_fov>{math.radians(cam["hfov_deg"])}</horizontal_fov>
    <image><width>{cam["width"]}</width><height>{cam["height"]}</height><format>R8G8B8</format></image>
    <clip><near>{cam["near_m"]}</near><far>{cam["far_m"]}</far></clip></camera>
    <always_on>1</always_on><update_rate>10</update_rate><topic>lc/cam</topic></sensor>
  <light type="spot" name="lamp"><pose>{lx} {ly} {lz} 0 0 0</pose><direction>1 0 0</direction>
    <intensity>{lamp["intensity"]}</intensity><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
    <attenuation><range>{lamp.get("range_m", 20)}</range><constant>{c}</constant><linear>{l}</linear><quadratic>{q}</quadratic></attenuation>
    <spot><inner_angle>{math.radians(lamp["inner_deg"])}</inner_angle><outer_angle>{math.radians(lamp["outer_deg"])}</outer_angle><falloff>1.0</falloff></spot>
    <cast_shadows>{shadows}</cast_shadows></light>
</link></model></world></sdf>'''


def render_one(rig, pose, work):
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    open(f"{work}/w.sdf", "w").write(world(rig, pose))
    bag = f"{work}/bag"
    procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", "/lc/cam@sensor_msgs/msg/Image[gz.msgs.Image"],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", bag, "/lc/cam"],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(4)
    r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", "500", f"{work}/w.sdf"],
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
        _, d, _ = rd.read_next()
        m = deserialize_message(d, Image)
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[..., 0].copy()
    if r.returncode != 0 or img is None:
        raise RuntimeError(f"render failed: rc {r.returncode}, frame {img is not None}")
    return img


def cmd_render(a):
    rig = yaml.safe_load(open(a.rig))
    os.makedirs(a.out, exist_ok=True)
    for k, pose in enumerate(POSES):
        np.save(f"{a.out}/pose_{k}.npy", render_one(rig, pose, "/tmp/lampcalib"))
        print(f"pose {k} {pose}", flush=True)
    json.dump({"rig": rig["name"], "poses": POSES}, open(f"{a.out}/poses.json", "w"), indent=1)


def cmd_fit(a):
    rig = yaml.safe_load(open(a.rig))
    meta = json.load(open(f"{a.frames}/poses.json"))
    if meta["rig"] != rig["name"]:
        sys.exit(f"frames were rendered for {meta['rig']}, not {rig['name']}")
    poses = meta["poses"]
    frames = [np.load(f"{a.frames}/pose_{k}.npy") for k in range(len(poses))]
    lamp_g = lamp_in_camera_g(rig)
    s = model.samples(frames, poses, lamp_g)
    train, test = s[s[:, 0] % 2 == 0], s[s[:, 0] % 2 == 1]
    held = {cos: model.heldout_rms(train, test, cos) for cos in (True, False)}
    use_cos = held[True] <= held[False]
    m = model.fit(s, use_cos)
    out = {
        "name": a.name, "rig": rig["name"], "render_transfer": "srgb",
        "lamp_in_camera_optical_m": [round(float(v), 6) for v in model.g_to_optical(lamp_g)],
        "lamp_axis_optical": [0.0, 0.0, 1.0],
        "intensity_ref": float(rig["lamp"]["intensity"]),
        "calib_albedo": 0.5,
        "gain_A": round(m["gain_A"], 6),
        "c_prime_m2": round(m["c_prime_m2"], 4),
        "cos_incidence": bool(use_cos),
        "rid": {"theta_deg": [round(float(v), 3) for v in m["theta_deg"]],
                "value": [round(float(v), 5) for v in m["rid"]]},
        "fit": {"rms_log_all": round(m["rms_log"], 5),
                "heldout_rms_log_with_cos": round(held[True], 5),
                "heldout_rms_log_without_cos": round(held[False], 5),
                "poses": len(poses), "samples": int(len(s))},
    }
    with open(a.out, "w") as f:
        f.write("# Lamp calibration (5.1, D133): generated by `python -m dataset_tools.lamp_calib fit`; do not hand-edit.\n"
                "# I = gain_A * (intensity / intensity_ref) * RID(theta) * [cos(inc)] * (albedo / calib_albedo) / (c' + r^2),\n"
                "# I linear in [0, 1] (the sRGB render decoded); r, theta from the lamp centre and axis (optical frame).\n")
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None, width=120)
    print(json.dumps({k: v for k, v in out.items() if k != "rid"}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dataset_tools.lamp_calib")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("render"); p.add_argument("--rig", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("fit"); p.add_argument("--rig", required=True); p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True); p.add_argument("--name", default="lamp_calib_v2")
    a = ap.parse_args(argv)
    (cmd_render if a.cmd == "render" else cmd_fit)(a)


if __name__ == "__main__":
    main()
