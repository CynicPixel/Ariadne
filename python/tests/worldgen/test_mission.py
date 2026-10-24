import json
import xml.etree.ElementTree as ET

import yaml

from dataset_tools.worldgen.build import write_scene
from dataset_tools.worldgen.mission import load_rig, topics, write_mission
from dataset_tools.worldgen.params import load_preset
from dataset_tools.worldgen.scene import generate

ROOT = __file__.rsplit("/python/", 1)[0]
RIG = f"{ROOT}/configs/sim/rig_stereo_v1.yaml"


def test_mission_world_bridge_and_manifest(tmp_path):
    s = generate(load_preset(f"{ROOT}/configs/worldgen/ci_tiny.yaml"), 0)
    write_scene(s, str(tmp_path), textures=False)
    m = write_mission(s, str(tmp_path), RIG)
    world = ET.parse(tmp_path / "mission_default.sdf").getroot()        # well-formed XML
    names = [p.get("name") for p in world.iter("plugin")]
    assert "gz::sim::systems::Sensors" in names
    assert not any("Physics" in (n or "") for n in names)               # kinematic: no physics
    for k, sc in enumerate(s.scripts):
        model = next(md for md in world.iter("model") if md.get("name") == f"agent_{k}")
        segs = list(model.iter("segment"))
        assert len(segs) == len(sc.segments)
        for e, seg in zip(segs, sc.segments):                           # exact float round trip
            assert float(e.get("t0")) == seg.t0 and float(e.get("D")) == seg.profile.D
            assert tuple(float(v) for v in e.get("start").split()) == tuple(seg.start)
        sensors = {x.get("name"): x.get("type") for x in model.iter("sensor")}
        assert sensors == {"left": "camera", "right": "camera", "depth": "depth_camera"}
        assert not list(model.iter("visual"))                           # agents invisible (D37)
    bridge = yaml.safe_load(open(tmp_path / "bridge_default.yaml"))
    ros = {e["ros_topic_name"] for e in bridge}
    for k in range(len(s.scripts)):
        assert set(topics(k).values()) <= ros
    assert "/clock" in ros
    js = json.load(open(tmp_path / "mission_default.json"))
    assert js["scene_id"] == s.scene_id and js["duration_s"] == m["duration_s"]
    assert abs(js["intrinsics_nominal"]["fx"] - 640.0) < 1e-9
    assert js["render_intrinsics_pixel_centre"]["cx"] == 639.5 and js["render_intrinsics_pixel_centre"]["cy"] == 359.5
    assert js["lens"]["fx"] == 680.0 and js["lens"]["k1"] == -0.05
    assert not list(world.iter("distortion"))                           # pinhole render (D41)


def test_rig_validation(tmp_path):
    rig = yaml.safe_load(open(RIG))
    rig["imu"]["period_ns"] = 3000000          # does not divide 1 s
    p = tmp_path / "rig.yaml"
    p.write_text(yaml.safe_dump(rig))
    try:
        load_rig(str(p))
        raise AssertionError("expected MissionError")
    except ValueError:
        pass


def _write(tmp_path, rig):
    p = tmp_path / "rig.yaml"
    p.write_text(yaml.safe_dump(rig))
    return str(p)


def test_lens_must_stay_inside_the_render(tmp_path):
    rig = yaml.safe_load(open(RIG))
    load_rig(_write(tmp_path, rig))                                      # the shipped rig is valid
    rig["lens"]["fx"] = rig["lens"]["fy"] = 670.0                       # minimum is 674.05 (barrel)
    try:
        load_rig(_write(tmp_path, rig))
        raise AssertionError("expected MissionError")
    except ValueError as e:
        assert "outside" in str(e)


def test_sdf_distortion_is_refused(tmp_path):
    rig = yaml.safe_load(open(RIG))
    rig["camera"]["distortion"] = {"k1": -0.05, "k2": 0.0, "k3": 0.0, "p1": 0.0, "p2": 0.0}
    try:
        load_rig(_write(tmp_path, rig))
        raise AssertionError("expected MissionError")
    except ValueError as e:
        assert "pinhole" in str(e)


def test_baseline_sweep_cameras(tmp_path):
    """5.5 (D138): sweep cameras sit at the stated baselines, are bridged and recorded, and v1 gets none."""
    s = generate(load_preset(f"{ROOT}/configs/worldgen/ci_tiny.yaml"), 0)
    write_scene(s, str(tmp_path), textures=False)
    write_mission(s, str(tmp_path), f"{ROOT}/configs/sim/rig_stereo_sweep_v1.yaml")
    world = ET.parse(tmp_path / "mission_default.sdf").getroot()
    model = next(md for md in world.iter("model") if md.get("name") == "agent_0")
    poses = {x.get("name"): [float(v) for v in x.find("pose").text.split()[:3]] for x in model.iter("sensor")}
    assert set(poses) == {"left", "right", "depth", "right_b04", "right_b08", "right_b12", "right_b20"}
    for name, b in (("right_b04", 0.04), ("right_b08", 0.08), ("right_b12", 0.12), ("right_b20", 0.20)):
        assert abs((poses["left"][1] - poses[name][1]) - b) < 1e-12        # +y is left: right cameras sit at -b
        assert poses[name][0] == poses["left"][0] and poses[name][2] == poses["left"][2]
    ros = {e["ros_topic_name"] for e in yaml.safe_load(open(tmp_path / "bridge_default.yaml"))}
    js = json.load(open(tmp_path / "mission_default.json"))
    t0 = js["agents"][0]["topics"]
    assert t0["right_b08_image"] == "/ariadne/agent_0/right_b08/image_raw" and t0["right_b08_image"] in ros
    assert js["t_B_camera_m"]["right_b20"][1] == js["t_B_camera_m"]["left"][1] - 0.20
    assert "right_b04" not in topics(0, load_rig(RIG))


def test_sweep_baselines_must_be_distinct(tmp_path):
    import pytest
    from dataset_tools.worldgen.mission import MissionError
    rig = yaml.safe_load(open(RIG))
    rig["stereo"]["sweep_baselines_m"] = [0.04, 0.041]
    p = tmp_path / "rig.yaml"
    p.write_text(yaml.safe_dump(rig))
    with pytest.raises(MissionError):
        load_rig(str(p))
