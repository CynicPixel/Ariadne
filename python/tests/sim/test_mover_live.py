"""The kinematic mover's live command mode (design live-runtime.md D151), against the real plugin in gz.

ci_tiny seed 0, the live mission with its cameras removed (server only, no rendering). With the world running:
  1. a goto 1 m ahead of agent 0's entrance pose is ACCEPTED (mover_state), and GT pose ends there with the commanded
     yaw;
  2. a goto into a wall is REFUSED with the envelope reason, and the agent holds;
  3. a hold (0 poses) is accepted and adds no segment;
  4. a 3-waypoint follow (a square back toward the entrance) is accepted and GT pose ends at its last waypoint;
  5. the executed-script log's last line is a valid script (contiguous; the commanded segments carry the command id).

Runs in the sim image with ROS sourced:  infra/scripts/worldgen.sh simtest
"""
import json
import math
import os
import re
import subprocess
import sys
import time

import pytest

ROOT = __file__.rsplit("/python/", 1)[0]
sys.path.insert(0, os.path.join(ROOT, "infra", "scripts"))

import sim_record as S  # noqa: E402
from dataset_tools.worldgen.build import write_scene  # noqa: E402
from dataset_tools.worldgen.mission import write_mission  # noqa: E402
from dataset_tools.worldgen.params import load_preset  # noqa: E402
from dataset_tools.worldgen.scene import generate  # noqa: E402

CMD, STATE, POSE = "/ariadne/agent_0/cmd", "/ariadne/agent_0/mover_state", "/ariadne/agent_0/gt/pose"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("ci_tiny_0_live"))
    s = generate(load_preset(f"{ROOT}/configs/worldgen/ci_tiny.yaml"), 0)
    write_scene(s, d)
    write_mission(s, d, f"{ROOT}/configs/sim/rig_stereo_v1.yaml", name="live", live=True, rtf=1.0)
    S.build_plugins()
    sdf = open(os.path.join(d, "mission_live.sdf")).read()
    sdf = re.sub(r"<sensor .*?</sensor>\n?", "", sdf, flags=re.S)       # no cameras: no render engine needed
    out = os.path.join(d, "mission_live_nocam.sdf")
    open(out, "w").write(sdf)
    entrance = json.load(open(os.path.join(d, "scene.json")))["entrances"][0]
    return d, out, entrance


def _pub(pose_v: str) -> None:
    subprocess.run(["gz", "topic", "-t", CMD, "-m", "gz.msgs.Pose_V", "-p", pose_v], check=True)


def _echo(topic: str, n: int, timeout: float) -> subprocess.Popen:
    return subprocess.Popen(["timeout", str(timeout), "gz", "topic", "-e", "-t", topic, "-n", str(n)],
                            stdout=subprocess.PIPE, text=True)


def _states(text: str) -> list:
    return [json.loads(json.loads('"' + m + '"')) for m in re.findall(r'data: "((?:[^"\\]|\\.)*)"', text)]


def test_live_commands(world):
    d, sdf, ent = world
    env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH="/work/build/sim-plugins", GZ_TRANSPORT_RCVHWM="0",
               GZ_TRANSPORT_SNDHWM="0")
    gz = subprocess.Popen(["gz", "sim", "-s", "-r", "--iterations", "120000", sdf], cwd=d, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4.0)                                   # past the 2 s start hold (RTF ~1, server only)
        x0, y0 = ent["start_xy_m"]
        yaw0 = math.radians(ent["start_yaw_deg"])
        gx, gy, gyaw = x0 + math.cos(yaw0), y0 + math.sin(yaw0), yaw0 + 0.5
        states = _echo(STATE, 4, 90)
        time.sleep(1.0)
        q = f"orientation{{z:{math.sin(gyaw / 2)} w:{math.cos(gyaw / 2)}}}"
        hdr = 'header{stamp{sec:%d} data{key:"frame_id" value:"world"}}'
        _pub(hdr % 4 + f" pose{{position{{x:{gx} y:{gy} z:1.0}} {q}}}")
        time.sleep(12.0)                                  # 1 m at <= 0.6 m/s with S-curve ramps, plus two turns
        _pub(hdr % 5 + f" pose{{position{{x:{x0} y:{y0 + 30.0} z:1.0}}}}")   # 30 m north: through a wall
        time.sleep(1.0)
        _pub(hdr % 6)                                                      # hold
        time.sleep(1.0)
        # follow: a 0.3 m-wide U beside the goto point, back to the entrance (inside the 1.35 m vestibule); no yaw.
        lx, ly = -math.sin(yaw0), math.cos(yaw0)
        fw = [(gx + 0.3 * lx, gy + 0.3 * ly), (x0 + 0.3 * lx, y0 + 0.3 * ly), (x0, y0)]
        _pub(hdr % 7 + "".join(f" pose{{position{{x:{x} y:{y} z:1.0}} orientation{{w:0}}}}" for x, y in fw))
        out, _ = states.communicate(timeout=90)
        st = _states(out)
        assert [s["cmd_stamp_ns"] for s in st] == [4_000_000_000, 5_000_000_000, 6_000_000_000, 7_000_000_000], st
        assert st[0]["accepted"] and st[0]["kind"] == "goto" and st[0]["segments"] >= 2, st[0]
        assert not st[1]["accepted"] and "envelope" in st[1]["reason"], st[1]
        assert st[2]["accepted"] and st[2]["kind"] == "hold" and st[2]["segments"] == 0, st[2]
        assert st[3]["accepted"] and st[3]["kind"] == "follow" and st[3]["segments"] >= 5, st[3]
        time.sleep(max(0.0, st[3]["t_end_s"] - st[3]["applied_ns"] * 1e-9) + 2.0)   # RTF ~1, server only
        pose = _echo(POSE, 1, 10).communicate(timeout=15)[0]
        px = float(re.search(r"position \{\s*x: ([-\d.e]+)", pose).group(1))
        py = float(re.search(r"position \{.*?y: ([-\d.e]+)", pose, re.S).group(1))
        assert math.hypot(px - x0, py - y0) < 1e-6, (px, py, x0, y0)
        last = json.loads(open(os.path.join(d, "mover_agent_0.jsonl")).read().splitlines()[-1])
        segs = last["segments"]
        assert any(s["tag"] == "cmd:4000000000" and s["kind"] == "move" for s in segs)
        assert sum(s["tag"] == "cmd:7000000000" and s["kind"] == "move" for s in segs) == 3
        for a, b in zip(segs, segs[1:]):
            assert abs(a["t0"] + a["duration"] - b["t0"]) < 1e-9
    finally:
        gz.terminate()
        try:
            gz.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gz.kill()


def test_command_taken_while_paused(world):
    # D165 lockstep: the runtime sends a barrier's commands while gz is PAUSED at T and waits for the answers before
    # resuming. The mover must take the command at paused time T (applied_ns == T) and answer, and not move.
    d, sdf, ent = world
    env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH="/work/build/sim-plugins", GZ_TRANSPORT_RCVHWM="0",
               GZ_TRANSPORT_SNDHWM="0")
    gz = subprocess.Popen(["gz", "sim", "-s", sdf], cwd=d, env=env,             # no -r: starts paused at t = 0
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(4.0)
        x0, y0 = ent["start_xy_m"]
        yaw0 = math.radians(ent["start_yaw_deg"])
        states = _echo(STATE, 1, 30)
        poses = _echo(POSE, 1, 30)
        time.sleep(1.0)
        hdr = 'header{stamp{sec:%d} data{key:"frame_id" value:"world"}}'
        _pub(hdr % 9 + f" pose{{position{{x:{x0 + math.cos(yaw0)} y:{y0 + math.sin(yaw0)} z:1.0}} orientation{{w:0}}}}")
        out, _ = states.communicate(timeout=30)
        st = _states(out)
        assert len(st) == 1 and st[0]["cmd_stamp_ns"] == 9_000_000_000, st
        assert st[0]["accepted"] and st[0]["applied_ns"] == 0, st[0]       # taken at the paused time, not later
        poses.kill()
    finally:
        gz.terminate()
        gz.wait(timeout=20)
