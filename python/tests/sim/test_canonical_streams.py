"""Frozen canonical IMU / GT-pose streams (DECISIONS D39).

The canonical bag of a recording is a pure function of (mission SDF, end time, the
trajectory code). This pins it: ci_tiny seed 0 with the v1 rig, 8 s, must give
exactly GOLDEN_CONTENT (container-independent digest over topic, stamp, payload),
be byte-reproducible, and equal the independent Python evaluator (<= 1e-9).

If this fails, a change to scripted.hpp, MoverScript.hh, ros_cdr_writer.hpp, the
mission composer or the route generator changed the IMU/GT a recording would hold.
Unintended: fix it. Intended: re-record affected bags and replace GOLDEN_CONTENT.
The ci_tiny scene itself is pinned by tests/worldgen/test_golden.py.

Runs in the sim image with ROS sourced (it builds and runs the C++ tool):
    infra/scripts/worldgen.sh simtest
"""
import json
import os
import sys

import pytest

ROOT = __file__.rsplit("/python/", 1)[0]
sys.path.insert(0, os.path.join(ROOT, "infra", "scripts"))

import sim_record as S  # noqa: E402
from dataset_tools.worldgen.build import write_scene  # noqa: E402
from dataset_tools.worldgen.mission import write_mission  # noqa: E402
from dataset_tools.worldgen.params import load_preset  # noqa: E402
from dataset_tools.worldgen.scene import generate  # noqa: E402

END_NS = 8_000_000_000
GOLDEN_CONTENT = "fc11e8760f9a84201becfb8fe2432497d335904a63efbb7700417d44a6339a74"  # measured 2026-10-01


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("ci_tiny_0"))
    s = generate(load_preset(f"{ROOT}/configs/worldgen/ci_tiny.yaml"), 0)
    write_scene(s, d)
    write_mission(s, d, f"{ROOT}/configs/sim/rig_stereo_v1.yaml")
    S.build_plugins()
    return d


def test_canonical_streams_golden(scene, tmp_path):
    mission = json.load(open(os.path.join(scene, "mission_default.json")))
    world = os.path.join(scene, mission["world_file"])
    S.check_world_step(world, mission["imu"]["period_ns"])
    out = str(tmp_path / "canonical")
    summary = S.write_canonical(world, END_NS, out)
    period = mission["imu"]["period_ns"]
    assert summary == {"agents": 2, "period_ns": period, "samples_per_topic": END_NS // period + 1,
                       "last_ns": END_NS}

    can = S.read_bag(out, keep_bytes=lambda t: True)
    grid = [k * period for k in range(END_NS // period + 1)]
    checks = {}
    assert S.check_canonical(can, mission, scene, grid, checks) == []
    assert checks["imu"][0] == checks["gt_pose"][0] == 2 * len(grid)

    assert S.canonical_reproducible(world, END_NS, out)
    got = S.content_digest(can)
    assert got == GOLDEN_CONTENT, f"canonical streams changed: measured {got}"


def test_content_digest_sees_one_flipped_bit(scene, tmp_path):
    """The digest is the regression guard, so it must react to a single bit."""
    mission = json.load(open(os.path.join(scene, "mission_default.json")))
    out = str(tmp_path / "canonical")
    S.write_canonical(os.path.join(scene, mission["world_file"]), 1_000_000_000, out)
    can = S.read_bag(out, keep_bytes=lambda t: True)
    base = S.content_digest(can)
    topic = mission["agents"][1]["topics"]["imu"]
    ns = sorted(can[topic]["raw"])[100]
    b = bytearray(can[topic]["raw"][ns])
    b[-1] ^= 1
    can[topic]["raw"][ns] = bytes(b)
    assert S.content_digest(can) != base
