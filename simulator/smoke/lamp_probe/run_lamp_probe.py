#!/usr/bin/env python3
"""Lamp render probe (Phase 5, SMOKE). Runs inside ariadne/sim with the repo's python/ at /py and writes to /out.

Questions it answers on a real gz-sim8/ogre2 render, before Stage 3 (falloff normalisation) or the dark worlds are
designed:
  Q1 linearity  -- is the camera output linear in radiance or gamma-encoded? Lamp intensity k*P at a fixed plane
                   distance: a linear camera gives pixel ratio k, sRGB-like gives about k^(1/2.2).
  Q2 falloff    -- what distance law does the renderer apply? Plane at several distances, with the attenuation the
                   rig SDF emits today (constant 0.2, quadratic 0.05) and with pure quadratic (0, 0, 1).
  Q3 headroom   -- does 8-bit hold the lamp's dynamic range? Every launch renders an R8G8B8 and an L16 camera at the
                   same pose; L16 values off the 257 grid mean real extra precision.
  Q4 shadows    -- does the spot lamp cast shadows on this host? An occluder in front of the plane, lamp offset 0.3 m
                   sideways so the shadow is visible from the camera.
Each launch is one world: ambient 0, no other light, a matte grey plane facing the camera. Raw frames go to /out as
.npy so the lamp calibration fit can reuse them.
"""
import json, os, shutil, signal, subprocess, sys, time
import numpy as np

OUT = os.environ.get("ARIADNE_SMOKE_OUT", "/out")
W = "/tmp/lampw"
ATT_RIG = (0.2, 0.0, 0.05)     # what python/dataset_tools/worldgen/mission.py emits today
ATT_QUAD = (0.0, 0.0, 1.0)     # inverse-square
LAMP = (0.05, 0.0, 0.04)       # rig_stereo_v1 lamp offset from the body, used here relative to the camera

CASES = (
    [dict(name=f"lin_k{k}", d=1.5, intensity=k, att=ATT_QUAD) for k in (0.25, 0.5, 1.0, 2.0)]
    + [dict(name=f"fall_quad_d{d}", d=d, intensity=1.0, att=ATT_QUAD) for d in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    + [dict(name=f"fall_rig_d{d}", d=d, intensity=1.0, att=ATT_RIG) for d in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)]
    + [dict(name="shadow", d=2.0, intensity=1.0, att=ATT_QUAD, occluder=(1.0, 0.15), lamp=(0.05, -0.3, 0.04))]
    # Rerun 2026-10-22: in "shadow" the occluder sat just outside the 20 deg half-angle cone (no shadow possible).
    + [dict(name="shadow2", d=2.0, intensity=1.0, att=ATT_QUAD, occluder=(1.0, -0.25), lamp=(0.05, -0.3, 0.04)),
       # SDF spot angles are FULL cone angles (probe: flat to 12.5 deg, dark by 20 deg); 50/80 = half 25/40.
       dict(name="wide_d2", d=2.0, intensity=1.0, att=ATT_QUAD, cone_deg=(50.0, 80.0))]
)
if os.environ.get("PROBE_CASES"):
    CASES = [c for c in CASES if c["name"] in os.environ["PROBE_CASES"].split(",")]


def world(c):
    lx, ly, lz = c.get("lamp", LAMP)
    ac, al, aq = c["att"]
    occ = ""
    if c.get("occluder"):
        ox, oy = c["occluder"]
        occ = (f'<model name="occ"><static>true</static><pose>{ox} {oy} 1 0 0 0</pose><link name="l">'
               '<visual name="v"><geometry><box><size>0.2 0.2 0.2</size></box></geometry>'
               '<material><diffuse>0.5 0.5 0.5 1</diffuse><specular>0 0 0 1</specular></material></visual></link></model>')
    cam = lambda n, fmt: (
        f'<sensor name="{n}" type="camera"><camera><horizontal_fov>1.5708</horizontal_fov>'
        f'<image><width>1280</width><height>720</height><format>{fmt}</format></image>'
        f'<clip><near>0.05</near><far>50</far></clip></camera>'
        f'<always_on>1</always_on><update_rate>10</update_rate><topic>lp/{n}</topic></sensor>')
    return f'''<?xml version="1.0" ?><sdf version="1.10"><world name="lampprobe">
<physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size><real_time_factor>1.0</real_time_factor></physics>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<scene><ambient>0 0 0 1</ambient><background>0 0 0 1</background><grid>false</grid><shadows>true</shadows></scene>
<model name="plane"><static>true</static><pose>{c["d"]} 0 1 0 0 0</pose><link name="l">
  <visual name="v"><geometry><box><size>0.02 12 8</size></box></geometry>
  <material><diffuse>0.5 0.5 0.5 1</diffuse><specular>0 0 0 1</specular>
  <pbr><metal><roughness>1</roughness><metalness>0</metalness></metal></pbr></material></visual></link></model>
{occ}
<model name="cam"><static>true</static><pose>0 0 1 0 0 0</pose><link name="l">
  {cam("c8", "R8G8B8")}{cam("c16", "L16")}
  <light type="spot" name="lamp"><pose>{lx} {ly} {lz} 0 0 0</pose><direction>1 0 0</direction>
    <intensity>{c["intensity"]}</intensity><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
    <attenuation><range>30</range><constant>{ac}</constant><linear>{al}</linear><quadratic>{aq}</quadratic></attenuation>
    <spot><inner_angle>{np.radians(c.get("cone_deg", (25.0, 40.0))[0])}</inner_angle><outer_angle>{np.radians(c.get("cone_deg", (25.0, 40.0))[1])}</outer_angle><falloff>1.0</falloff></spot>
    <cast_shadows>true</cast_shadows></light>
</link></model>
</world></sdf>'''


def render(c):
    """One gz launch -> last R8G8B8 and L16 frame (or None), plus gz stderr errors."""
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    open(f"{W}/w.sdf", "w").write(world(c))
    bag = f"{W}/bag"
    topics = ["/lp/c8", "/lp/c16"]
    procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge"]
                              + [f"{t}@sensor_msgs/msg/Image[gz.msgs.Image" for t in topics],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", bag] + topics,
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(4)
    r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", "500", f"{W}/w.sdf"],
                       capture_output=True, text=True, timeout=600)
    time.sleep(3)
    for p in procs: os.killpg(p.pid, signal.SIGINT)
    for p in procs: p.wait(timeout=30)

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    frames = {}
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    while rd.has_next():
        topic, d, _ = rd.read_next()
        m = deserialize_message(d, Image)
        if m.encoding == "mono16":
            a = np.frombuffer(bytes(m.data), "<u2" if not m.is_bigendian else ">u2").reshape(m.height, m.width)
        else:
            a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
        frames[topic] = (m.encoding, a)
    return frames, r.returncode, [l for l in r.stderr.splitlines() if "rror" in l][:3]


def patch(a, cx=640, cy=360, h=10):
    return float(a[cy - h:cy + h, cx - h:cx + h].astype(float).mean())


results = {}
for c in CASES:
    frames, rc, errs = render(c)
    row = {"gz_rc": rc, "errors": errs, "encodings": {t: e for t, (e, _) in frames.items()}}
    if "/lp/c8" in frames:
        a8 = frames["/lp/c8"][1]
        np.save(f"{OUT}/{c['name']}_c8.npy", a8)
        g = a8[..., 0]
        row["c8_centre"] = round(patch(g), 3)
        row["c8_sat_frac"] = round(float((g >= 255).mean()), 5)
        row["c8_max"] = int(g.max())
        row["c8_rgb_equal"] = bool((a8[..., 0] == a8[..., 1]).all() and (a8[..., 1] == a8[..., 2]).all())
        # Horizontal profile along the centre row: the cone profile (inner 25 / outer 40 deg) plus cos incidence.
        row["c8_row_profile"] = [int(v) for v in g[360, ::64]]
    if "/lp/c16" in frames:
        a16 = frames["/lp/c16"][1]
        np.save(f"{OUT}/{c['name']}_c16.npy", a16)
        row["c16_centre"] = round(patch(a16), 2)
        row["c16_max"] = int(a16.max())
        nz = a16[a16 > 0]
        # Real 16-bit precision shows as values that are not multiples of 257 (an 8-bit value expanded to 16).
        row["c16_off_257_grid_frac"] = round(float((nz % 257 != 0).mean()), 4) if nz.size else None
        row["c16_distinct"] = int(np.unique(a16).size)
    if c.get("occluder") and "/lp/c8" in frames:
        g = frames["/lp/c8"][1][..., 0].astype(float)
        row["shadow_min_row_profile"] = [int(v) for v in g[360, 200:1100:25]]
    results[c["name"]] = row
    print(c["name"], json.dumps({k: v for k, v in row.items() if "profile" not in k}), flush=True)

json.dump(results, open(f"{OUT}/lamp_probe{os.environ.get('PROBE_TAG', '')}.json", "w"), indent=1)
print("WROTE", f"{OUT}/lamp_probe.json")
