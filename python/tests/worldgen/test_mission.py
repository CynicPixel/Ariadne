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
