"""Mission composer: scene directory + rig config -> a runnable gz world (design §5).

Writes into the SCENE directory (relative mesh/texture/ESDF paths resolve there):
  mission_<name>.sdf   scene model inlined + one agent model per script, each with a
                       body link (= IMU frame: x fwd, y left, z up), left/right cameras
                       (baseline along -y), a GT depth camera co-located with the left,
                       an inert spot lamp, and the KinematicMover plugin carrying the
                       script as <segment> elements. No Physics system, no agent visuals.
  bridge_<name>.yaml   ros_gz_bridge parameter_bridge config (GZ -> ROS only)
  mission_<name>.json  topics, frames, extrinsics T_B_C (body -> OPTICAL frame), expected
                       intrinsics, the rig config verbatim, scene identity.

gz cameras look along the sensor link's +x with image-right = -y and image-down = -z;
the ROS optical frame is x right, y down, z forward. So R_B_optical has columns
(x_opt, y_opt, z_opt) = (-y_B, -z_B, +x_B).
"""
from __future__ import annotations

import json
import math
import os

import yaml

from .build import world_xml

R_B_OPTICAL = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]   # rows of R (B <- optical)


class MissionError(ValueError):
    pass


def load_rig(path: str) -> dict:
    rig = yaml.safe_load(open(path))
    need = {"name", "camera", "stereo", "left_camera_in_body_m", "depth", "imu", "lamp", "lens"}
    missing = need - set(rig)
    extra = set(rig) - need
    if missing or extra:
        raise MissionError(f"rig: missing {sorted(missing)} unknown {sorted(extra)}")
    cam = rig["camera"]
    for k in ("width", "height", "hfov_deg", "rate_hz", "near_m", "far_m"):
        if k not in cam:
            raise MissionError(f"rig.camera.{k} missing")
    if "distortion" in cam:
        raise MissionError("rig.camera.distortion: the render is pinhole (ogre2 ignores it, SMOKE 4n); "
                           "put the lens under rig.lens, applied at read time (D41)")
    check_lens(rig)
    if not 10.0 < cam["hfov_deg"] < 170.0 or cam["rate_hz"] <= 0 or rig["stereo"]["baseline_m"] <= 0:
        raise MissionError("rig: hfov in (10, 170) deg, rate > 0, baseline > 0")
    if 1_000_000_000 % rig["imu"]["period_ns"] != 0:
        raise MissionError("rig.imu.period_ns must divide one second")
    return rig


LENS_KEYS = ("model", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")


def render_intrinsics(rig: dict) -> dict:
    """The pinhole render in the pixel-centre convention (SMOKE §4n measured it)."""
    c = rig["camera"]
    fx = (c["width"] / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)
    return {"fx": fx, "fy": fx, "cx": (c["width"] - 1) / 2.0, "cy": (c["height"] - 1) / 2.0}


def check_lens(rig: dict) -> None:
    """The read-time lens must be well formed and sample only inside the render, the
    same border test perception/degradation/lens.hpp's BuildLensMap applies (fail at
    composition time, not after a render)."""
    lens = rig["lens"]
    if set(lens) != set(LENS_KEYS) or lens["model"] != "plumb_bob":
        raise MissionError(f"rig.lens needs exactly {LENS_KEYS} with model plumb_bob")
    if not (lens["fx"] > 0 and lens["fy"] > 0):
        raise MissionError("rig.lens focal lengths must be > 0")
    import numpy as np
    c, r = rig["camera"], render_intrinsics(rig)
    W, H = c["width"], c["height"]
    u = np.concatenate([np.arange(W), np.arange(W), np.zeros(H), np.full(H, W - 1)]).astype(float)
    v = np.concatenate([np.zeros(W), np.full(W, H - 1), np.arange(H), np.arange(H)]).astype(float)
    xd, yd = (u - lens["cx"]) / lens["fx"], (v - lens["cy"]) / lens["fy"]
    x, y = xd.copy(), yd.copy()
    k1, k2, k3, p1, p2 = (lens[k] for k in ("k1", "k2", "k3", "p1", "p2"))
    for _ in range(30):                      # the operator's fixed-point inverse
        r2 = x * x + y * y
        rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad
        y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad
    su, sv = r["fx"] * x + r["cx"], r["fy"] * y + r["cy"]
    if not (np.isfinite(su).all() and su.min() >= 0 and sv.min() >= 0 and su.max() <= W - 1 and sv.max() <= H - 1):
        raise MissionError(f"rig.lens samples outside the {W}x{H} render (u {su.min():.2f}..{su.max():.2f}, "
                           f"v {sv.min():.2f}..{sv.max():.2f}): raise lens fx/fy")


def _f(x: float) -> str:
    return repr(float(x))            # exact round trip into the plugin's parser


def intrinsics(rig: dict) -> dict:
    c = rig["camera"]
    fx = (c["width"] / 2.0) / math.tan(math.radians(c["hfov_deg"]) / 2.0)
    return {"fx": fx, "fy": fx, "cx_nominal": c["width"] / 2.0, "cy_nominal": c["height"] / 2.0,
            "note": "nominal from hfov; the gz camera_info on the bag and the §6 recovery are authoritative"}


def camera_poses(rig: dict) -> dict:
    lx, ly, lz = rig["left_camera_in_body_m"]
    b = rig["stereo"]["baseline_m"]
    return {"left": [lx, ly, lz], "right": [lx, ly - b, lz]}


def topics(k: int) -> dict:
    ns = f"/ariadne/agent_{k}"
    return {"left_image": f"{ns}/left/image_raw", "left_info": f"{ns}/left/camera_info",
            "right_image": f"{ns}/right/image_raw", "right_info": f"{ns}/right/camera_info",
            "depth_image": f"{ns}/depth/image_raw", "depth_info": f"{ns}/depth/camera_info",
            "imu": f"{ns}/imu", "gt_pose": f"{ns}/gt/pose"}


def _camera(name, pose, rig, topic, info_topic, depth=False) -> str:
    c = rig["camera"]
    dist = ""   # pinhole render; the lens is a read-time operator (D41)
    kind, fmt = ("depth_camera", "R_FLOAT32") if depth else ("camera", "R8G8B8")
    x, y, z = pose
    return (f'        <sensor name="{name}" type="{kind}"><pose>{_f(x)} {_f(y)} {_f(z)} 0 0 0</pose>'
            f'<always_on>1</always_on><update_rate>{c["rate_hz"]}</update_rate><topic>{topic}</topic>'
            f'<camera><horizontal_fov>{_f(math.radians(c["hfov_deg"]))}</horizontal_fov>'
            f'<image><width>{c["width"]}</width><height>{c["height"]}</height><format>{fmt}</format></image>'
            f'<clip><near>{_f(c["near_m"])}</near><far>{_f(c["far_m"])}</far></clip>{dist}'
            f'<camera_info_topic>{info_topic}</camera_info_topic></camera></sensor>\n')


def _segments(script) -> str:
    out = []
    for s in script.segments:
        st = " ".join(_f(v) for v in s.start)
        en = " ".join(_f(v) for v in s.end)
        out.append(f'          <segment kind="{s.kind}" t0="{_f(s.t0)}" duration="{_f(s.duration)}" '
                   f'D="{_f(s.profile.D)}" Ta="{_f(s.profile.Ta)}" vc="{_f(s.profile.vc)}" '
                   f'start="{st}" end="{en}" tag="{s.tag}"/>\n')
    return "".join(out)


def agent_xml(scene, k: int, rig: dict, scene_json: dict) -> str:
    sc = scene.scripts[k]
    x, y, z, yaw, pitch = sc.segments[0].start
    t = topics(k)
    cams = camera_poses(rig)
    lamp = rig["lamp"]
    lx, ly, lz = lamp["in_body_m"]
    # D131: optional keys; their defaults reproduce rig v1's SDF byte for byte.
    att_c, att_l, att_q = lamp.get("attenuation", (0.2, 0.0, 0.05))
    shadows = "true" if lamp.get("cast_shadows", 0) else "false"
    es = scene_json["esdf"]
    g = es["cell_m"]
    ox, oy = (round(v / g) for v in es["origin_m"])
    body = f"agent_{k}/base_link"
    depth = (_camera("depth", cams["left"], rig, t["depth_image"], t["depth_info"], depth=True)
             if rig["depth"]["enabled"] else "")
    return (
        f'    <model name="agent_{k}"><static>true</static>'
        f'<pose>{_f(x)} {_f(y)} {_f(z)} 0 {_f(pitch)} {_f(yaw)}</pose>\n'
        f'      <link name="base_link">\n'
        + _camera("left", cams["left"], rig, t["left_image"], t["left_info"])
        + _camera("right", cams["right"], rig, t["right_image"], t["right_info"])
        + depth
        + f'        <light type="spot" name="lamp"><pose>{_f(lx)} {_f(ly)} {_f(lz)} 0 0 0</pose>'
          f'<direction>1 0 0</direction><intensity>{_f(lamp["intensity"])}</intensity>'
          f'<diffuse>1 1 1 1</diffuse><specular>0.2 0.2 0.2 1</specular>'
          f'<attenuation><range>{_f(lamp["range_m"]) if "range_m" in lamp else "20"}</range><constant>{_f(att_c)}</constant>'
          f'<linear>{_f(att_l)}</linear><quadratic>{_f(att_q)}</quadratic></attenuation>'
          f'<spot><inner_angle>{_f(math.radians(lamp["inner_deg"]))}</inner_angle>'
          f'<outer_angle>{_f(math.radians(lamp["outer_deg"]))}</outer_angle><falloff>1.0</falloff></spot>'
          f'<cast_shadows>{shadows}</cast_shadows></light>\n'
        + '      </link>\n'
        + f'      <plugin filename="ariadne_kinematic_mover" name="ariadne::sim::KinematicMover">\n'
          f'        <frame_id>{body}</frame_id><gravity>{_f(9.80665)}</gravity>'
          f'<imu_period_ns>{rig["imu"]["period_ns"]}</imu_period_ns>\n'
          f'        <imu_topic>{t["imu"]}</imu_topic><pose_topic>{t["gt_pose"]}</pose_topic>\n'
          f'        <rho_agent>{_f(scene.preset.clearance.rho_agent_m)}</rho_agent>\n'
          f'        <esdf><file>{es["file"]}</file><rows>{es["rows"]}</rows><cols>{es["cols"]}</cols>'
          f'<cell_m>{_f(g)}</cell_m><origin_x_cells>{ox}</origin_x_cells><origin_y_cells>{oy}</origin_y_cells></esdf>\n'
          f'        <script>\n' + _segments(sc) + '        </script>\n'
        + '      </plugin>\n    </model>\n')


def bridge_entries(n_agents: int, depth: bool) -> list:
    out = [{"ros_topic_name": "/clock", "gz_topic_name": "/clock", "ros_type_name": "rosgraph_msgs/msg/Clock",
            "gz_type_name": "gz.msgs.Clock", "direction": "GZ_TO_ROS"}]
    for k in range(n_agents):
        t = topics(k)
        pairs = [("left_image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
                 ("left_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
                 ("right_image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
                 ("right_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
                 ("imu", "sensor_msgs/msg/Imu", "gz.msgs.IMU"),
                 ("gt_pose", "geometry_msgs/msg/PoseStamped", "gz.msgs.Pose")]
        if depth:
            pairs += [("depth_image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
                      ("depth_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo")]
        for key, ros_t, gz_t in pairs:
            e = {"ros_topic_name": t[key], "gz_topic_name": t[key], "ros_type_name": ros_t,
                 "gz_type_name": gz_t, "direction": "GZ_TO_ROS"}
            if key in ("imu", "gt_pose"):
                # 200 Hz streams must survive a transient stall downstream (SMOKE 4k: a
                # default-depth run lost one sample of two such topics at the same instant).
                e["subscriber_queue"] = 1000
                e["publisher_queue"] = 1000
            out.append(e)
    return out


def write_mission(scene, scene_dir: str, rig_path: str, name: str = "default") -> dict:
    rig = load_rig(rig_path)
    scene_json = json.load(open(os.path.join(scene_dir, "scene.json")))
    if scene_json["scene_id"] != scene.scene_id:
        raise MissionError("scene directory does not hold this scene")
    model_xml = open(os.path.join(scene_dir, "model_inline.xml")).read()
    agents = "".join(agent_xml(scene, k, rig, scene_json) for k in range(len(scene.scripts)))
    world = world_xml(scene, model_xml, extra=agents)
    world = world.replace("<physics name=\"1ms\"", "<gravity>0 0 -9.80665</gravity>\n    <physics name=\"1ms\"", 1)
    with open(os.path.join(scene_dir, f"mission_{name}.sdf"), "w", newline="\n") as f:
        f.write(world)
    depth = bool(rig["depth"]["enabled"])
    with open(os.path.join(scene_dir, f"bridge_{name}.yaml"), "w", newline="\n") as f:
        yaml.safe_dump(bridge_entries(len(scene.scripts), depth), f, sort_keys=False)
    cams = camera_poses(rig)
    manifest = {
        "mission": name, "scene_id": scene.scene_id, "rig": rig, "rig_path": rig_path,
        "world_file": f"mission_{name}.sdf", "bridge_file": f"bridge_{name}.yaml",
        "duration_s": max(s.duration for s in scene.scripts),
        "gravity_mps2": 9.80665,
        "frames": {"world": "world (scene frame: origin lattice SW corner, x east, y north, z up)",
                   "body": "agent_<k>/base_link = IMU frame: x forward, y left, z up",
                   "optical": "x right, y down, z forward (ROS optical)"},
        "R_B_optical_rows": R_B_OPTICAL,
        "t_B_camera_m": cams,
        "depth": "32FC1 metres, pinhole, co-located with the left camera, no distortion; GT only" if depth else None,
        "intrinsics_nominal": intrinsics(rig),
        "render_intrinsics_pixel_centre": render_intrinsics(rig),
        "camera_info_convention": "gz camera_info states cx = W/2, cy = H/2; the pixel-centre truth is "
                                  "(W-1)/2, (H-1)/2 (SMOKE 4n). Its D is [0]*5: the render is pinhole.",
        "lens": dict(rig["lens"], applied="read time, perception/degradation/lens.hpp (D41)"),
        "agents": [{"agent": k, "topics": topics(k), "frame_id": f"agent_{k}/base_link",
                    "script_duration_s": s.duration, "start_delay_s": scene.start_delays_s[k]
                    if k < len(scene.start_delays_s) else 0.0}
                   for k, s in enumerate(scene.scripts)],
        "imu": {"period_ns": rig["imu"]["period_ns"], "noise": "none (clean); consumption-time operator, D37",
                "orientation_field": "identity, NOT measured: consumers must ignore it"},
    }
    with open(os.path.join(scene_dir, f"mission_{name}.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest
