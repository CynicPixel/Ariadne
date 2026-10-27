"""Where each agent's submaps go in the mission world W: T_W_O per agent (reconstruction.md §3; D230).

O_k is agent k's odometry frame = its rectified-left camera at the first frame. Two sources, both briefing-based (no
ground truth):

  briefing  T_W_Ok = T_W_Bk x T_B_C for EVERY agent: entrance k's start pose (scene.json entrances[k].start_xy_m,
            start_yaw_deg; z = routes/agent_k.json segments[0].start[2]) times the rig extrinsic
            (mission_default.json R_B_optical_rows, t_B_camera_m.left). A live scene's mission_live.json is read
            first: its live.demo.entrances[k] names the WORLD entrance mission agent k ran (worldgen mission.py; B,
            2026-10-24), and that entrance's route gives z; without the block, agent k = entrance k.
            The live demo's known-starts rule (D216):
            what the runtime tee sends until live-collaboration.md's adopted estimates move it. Transcribed from ariadne_mission.cpp's
            world anchor (T_W_Oa = T_W_B0 * T_B_C), generalised from the anchor agent to agent k.
  merges    agent 0 from the briefing, agent k through the spool's merge into agent 0's frame:
            T_W_Ok = T_W_O0 x T_O0_Ok (schema 3 `merges:`, or schema < 3's agent-1 `merge:`). Agent k's own entrance
            is NOT used (ariadne_mission.cpp: "agent 1 enters W only through the verified loop"). An agent with no ok
            merge has no placement and is never drawn.

Without a scene, W = agent 0's odometry frame (the earlier replay convention) and only `merges` is possible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..spool import Spool


def briefing_T_W_O(scene_dir: str | Path, agent: int) -> np.ndarray:
    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text())
    live = scene_dir / "mission_live.json"
    mission = json.loads((live if live.is_file() else scene_dir / "mission_default.json").read_text())
    demo = mission.get("live", {}).get("demo", {})
    entrance = int(demo["entrances"][agent]) if "entrances" in demo else agent
    route = json.loads((scene_dir / "routes" / f"agent_{entrance}.json").read_text())
    e = scene["entrances"][entrance]
    if int(e["agent"]) != entrance:
        raise ValueError("briefing: entrance order != entrance id")
    yaw = np.deg2rad(float(e["start_yaw_deg"]))
    T_W_B = np.eye(4)
    T_W_B[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    T_W_B[:3, 3] = [float(e["start_xy_m"][0]), float(e["start_xy_m"][1]), float(route["segments"][0]["start"][2])]
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = np.asarray(mission["R_B_optical_rows"], float)
    T_B_C[:3, 3] = np.asarray(mission["t_B_camera_m"]["left"], float)
    return T_W_B @ T_B_C


def placements(spools: list[Spool], mode: str, scene_dir: str | Path | None
               ) -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """(agent -> T_W_O for every placeable agent, agent -> why it is not placed). `briefing` takes any number of
    spools (one per agent is the usual case); `merges` reads the merges of ONE spool that holds every agent."""
    agents = sorted({b.agent for sp in spools for b in sp.bundles})
    if mode == "briefing":
        if scene_dir is None:
            raise ValueError("placement 'briefing' needs the scene (--scene)")
        return {k: briefing_T_W_O(scene_dir, k) for k in agents}, {}
    if mode != "merges":
        raise ValueError(f"placement {mode!r} is not briefing|merges")
    if len(spools) != 1:
        raise ValueError("placement 'merges' reads one composed spool (compose_mission_spool.py), not several")
    (sp,) = spools
    T_W_O0 = briefing_T_W_O(scene_dir, 0) if scene_dir is not None else np.eye(4)
    placed, unplaced = {0: T_W_O0} if 0 in agents else {}, {}
    for k in agents:
        if k == 0:
            continue
        if sp.merges is not None:
            ok, T_O0_O, edge = sp.merges.get(k, (False, None, "no merges entry"))
        else:
            ok, T_O0_O, edge = (sp.merge_ok, sp.T_O0_O1, sp.merge_edge) if k == 1 else (False, None, "schema < 3")
        if ok:
            placed[k] = T_W_O0 @ T_O0_O
        else:
            unplaced[k] = f"no ok merge ({edge})"
    return placed, unplaced
