"""Replay a Phase 1 run chain as console telemetry (design CONSOLE_1.10).

A frame is a pure function of (chain, tick index, speed): the console never invents state. Sources:
- agents: the ESTIMATED body trajectories (sim_collab raw/*/estimate_body.tum) through the mission's world anchor
  (T_W_Oanchor = agent 0's briefed entry) and, for agent 1, the 1.6 merge T_O0_O1 — shown only once merged;
- bytes: sim_collab ledger.csv by CAUSAL send time (wire stamps are keyframe times and not causal);
- map / routes: the mission's per-replan snapshots and routes.jsonl;
- events: verified edges (ledger), the merge (mission t_merge_s), route milestones, and replay-logged frontier
  decisions when the choice changes (labelled: logged, not executed — invariant 3).
Mission time = t_ns - t0, t0 = the mission's first keyframe = stamp 0 of the sim clock (asserted).
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

TICK_HZ = 5.0


def _pose12(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, :] = np.array(v, float).reshape(3, 4)
    return T


def _tum(path: Path):
    rows = np.array([[float(x) for x in l.split()] for l in path.read_text().splitlines()
                     if l.strip() and not l.startswith("#")])
    return np.round(rows[:, 0] * 1e9).astype(np.int64), rows[:, 1:4], rows[:, 4:8]


def _yaw_from_quat(q: np.ndarray, R_W: np.ndarray) -> np.ndarray:
    x, y, z, w = q.T
    # body x axis in the estimate frame, rotated into the world: heading of the agent
    bx = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], -1)
    hw = bx @ R_W.T
    return np.arctan2(hw[:, 1], hw[:, 0])


@dataclass
class Chain:
    meta: dict
    agents: dict                      # id -> (t_s array, xy (N,2), yaw (N,))
    t_merge_s: float
    ledger: list                      # (t_s, class, wire_bytes, sender, receiver), causal order
    snapshots: list                   # (t_s, path)
    routes: list                      # replan records (t_s, kind, cells, found, exit)
    events: list = field(default_factory=list)
    end_s: float = 0.0


def load_chain(collab_run: Path, mission_run: Path, scene: Path) -> Chain:
    res = yaml.safe_load((mission_run / "results.yaml").read_text())
    if str(res["agents"]) != "merged":
        raise ValueError("the console replays the merged mission run")
    T_W_O0, T_O0_O1 = _pose12(res["T_W_Oanchor"]), _pose12(res["T_O0_O1"])
    t_merge_s = float(res["t_merge_s"])
    sc = json.loads((scene / "scene.json").read_text())
    e = sc["esdf"]
    ox, oy, g = e["origin_m"][0], e["origin_m"][1], e["cell_m"]
    cell_xy = lambda r, c: [round(ox + (c + 0.5) * g, 3), round(oy + (r + 0.5) * g, 3)]

    agents, t0 = {}, None
    for k in (0, 1):
        d = next((collab_run / "raw").glob(f"*-agent{k}-*"))
        t, p, q = _tum(d / "estimate_body.tum")
        T = T_W_O0 if k == 0 else T_W_O0 @ T_O0_O1
        pw = p @ T[:3, :3].T + T[:3, 3]
        agents[k] = (t * 1e-9, pw[:, :2], _yaw_from_quat(q, T[:3, :3]))
        t0 = t[0] if t0 is None else min(t0, t[0])
    if t0 != 0:
        raise ValueError(f"mission time base: expected the first stamp at 0 ns, got {t0}")

    ledger = []
    with open(collab_run / "ledger.csv") as f:
        for r in csv.DictReader(f):
            ledger.append((int(r["causal_ns"]) * 1e-9, r["class"], int(r["wire_bytes"]), int(r["sender"]),
                           int(r["receiver"]), int(r["seq"])))
    ledger.sort(key=lambda x: (x[0], x[5]))

    snaps = []
    for pth in (mission_run / "snapshots").glob("slab_*.png"):
        snaps.append((int(re.search(r"slab_(\d+)\.png", pth.name).group(1)) / 1000.0, pth))
    snaps.sort()
    routes = []
    for l in (mission_run / "routes.jsonl").read_text().splitlines():
        r = json.loads(l)
        routes.append({"t_s": r["t_s"], "kind": r["kind"], "found": r["found"], "exit": r["exit"],
                       "length_m": r["length_m"], "xy": [cell_xy(a, b) for a, b in r["cells"]]})

    events = []
    edges = [x for x in ledger if x[1] == "POSE_EDGE"]
    for i, x in enumerate(edges):
        events.append({"t_s": x[0], "kind": "edge", "text": f"verified inter-agent edge {i + 1}/{len(edges)} sent to agent 1"})
    events.append({"t_s": t_merge_s, "kind": "merge", "text": "agent 1 merged into the shared map through the verified loop"})
    first, last_exit = None, None
    for r in routes:
        if r["kind"] != "egress":
            continue
        if r["found"] and first is None:
            first = r
            events.append({"t_s": r["t_s"], "kind": "route",
                           "text": f"first valid egress route (0.40 m clearance): {r['length_m']:.2f} m via exit {r['exit']}"})
        if r["found"] and last_exit is not None and r["exit"] != last_exit:
            events.append({"t_s": r["t_s"], "kind": "route", "text": f"egress route switches to exit {r['exit']}"})
        if r["found"]:
            last_exit = r["exit"]
    last_choice = {}
    for l in (mission_run / "trace.jsonl").read_text().splitlines():
        d = json.loads(l)
        if d.get("result") != "go":
            continue
        xy = tuple(round(v, 1) for v in d["frontier_xy_m"])
        if last_choice.get(d["agent"]) != xy:
            last_choice[d["agent"]] = xy
            events.append({"t_s": d["t_s"], "kind": "decision",
                           "text": f"agent {d['agent']}: nearest frontier at ({xy[0]:.1f}, {xy[1]:.1f}) — logged decision · replay · not executed"})
    events.sort(key=lambda e: (e["t_s"], e["kind"], e["text"]))

    end_s = max(max(a[0][-1] for a in agents.values()), routes[-1]["t_s"] if routes else 0.0)
    meta = {"grid": {"origin_m": [ox, oy], "cell_m": g, "rows": e["rows"], "cols": e["cols"]},
            "target_xy": cell_xy(*res["target_cell"]), "exits_xy": [cell_xy(r, c) for r, c in res["exit_cells"]],
            "classes": sorted({x[1] for x in ledger}), "end_s": end_s, "tick_hz": TICK_HZ,
            "chain": {"collab_run": collab_run.name, "mission_run": mission_run.name, "scene": sc["scene_id"]},
            "snapshots": len(snaps)}
    return Chain(meta, agents, t_merge_s, ledger, snaps, routes, events, end_s)


def frame(ch: Chain, n: int, speed: float) -> dict:
    """Telemetry frame n (pure): mission time n * speed / TICK_HZ."""
    t = round(n * speed / TICK_HZ, 6)
    prev = round((n - 1) * speed / TICK_HZ, 6) if n > 0 else -1.0
    agents = []
    for k, (ts, xy, yaw) in sorted(ch.agents.items()):
        merged = k == 0 or t >= ch.t_merge_s
        i = int(np.searchsorted(ts, t, side="right")) - 1
        if not merged or i < 0:
            agents.append({"id": k, "merged": merged, "x": None, "y": None, "yaw": None})
            continue
        agents.append({"id": k, "merged": True, "x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
                       "yaw": round(float(yaw[i]), 4)})
    by_class = {c: 0 for c in ch.meta["classes"]}
    rate = 0
    for ts, cls, b, *_ in ch.ledger:
        if ts > t:
            break
        by_class[cls] += b
        if ts > t - 1.0:
            rate += b
    by_class["total"] = sum(v for k, v in by_class.items() if k != "total")
    mv = max((i for i, (ts, _) in enumerate(ch.snapshots) if ts <= t), default=None)
    route = {"egress": None, "agent": None}
    for r in ch.routes:
        if r["t_s"] <= t:
            route[r["kind"]] = r["xy"] if r["found"] else None
    events = [e for e in ch.events if prev < e["t_s"] <= t]
    return {"t_s": t, "seq": n, "agents": agents, "bytes": by_class, "rate_Bps": rate, "map_version": mv,
            "route": route, "events": events}


def frames_json(ch: Chain, speed: float) -> list[str]:
    n_max = int(np.ceil(ch.end_s * TICK_HZ / speed))
    return [json.dumps(frame(ch, n, speed), sort_keys=True, separators=(",", ":")) for n in range(n_max + 1)]
