"""The console fed live (design LIVE_RUNTIME_L1 D154): tail a running ariadne_live run directory.

ariadne_live appends telemetry.jsonl (estimated poses at 5 Hz of sim time, every decision, every mover answer, a line
per map snapshot) and writes snapshots/slab_<ms>.png at each replan. LiveRun.poll() reads whatever was appended since
the last call; frame() builds the SAME frame schema as replay.frame(), so index.html is unchanged. Nothing is invented:
a field with no source yet is null.

Every frame the server sends is logged to <run>/console_frames.jsonl (wall time, sim time, seq, websocket client):
the evidence that the console was fed during the run, not afterwards.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

TICK_HZ = 5.0


@dataclass
class LiveRun:
    run: Path
    scene: Path
    offset: int = 0
    t_s: float = 0.0
    poses: dict = field(default_factory=dict)        # agent -> (t_s, x, y, yaw)
    snapshots: list = field(default_factory=list)    # (t_s, path)
    egress: list | None = None
    events: list = field(default_factory=list)       # (t_s, kind, text)
    ended: bool = False
    # poll() is called from the event loop (websockets) AND from FastAPI's thread pool: one reader at a time, or two
    # threads advance `offset` past each other and a half line reaches json.loads (it killed every stream on 2026-10-23)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def src(self) -> Path | None:
        """The directory ariadne_live writes: `run` itself, or its one subdirectory holding telemetry.jsonl (live.sh
        passes the outer run directory; the runtime creates <scene>-live-<UTC>/ inside it). None until it exists."""
        if (self.run / "telemetry.jsonl").exists():
            return self.run
        subs = sorted(d for d in self.run.glob("*/telemetry.jsonl"))
        if len(subs) > 1:
            raise RuntimeError(f"{self.run}: more than one ariadne_live run directory")
        return subs[0].parent if subs else None

    def meta(self) -> dict:
        sc = json.loads((self.scene / "scene.json").read_text())
        e = sc["esdf"]
        ox, oy, g = e["origin_m"][0], e["origin_m"][1], e["cell_m"]
        cell = lambda x, y: [round(x, 3), round(y, 3)]
        return {"grid": {"origin_m": [ox, oy], "cell_m": g, "rows": e["rows"], "cols": e["cols"]},
                "target_xy": cell(*sc["target_xy_m"]),
                "exits_xy": [cell(*en["start_xy_m"]) for en in sc["entrances"]],
                "classes": [], "end_s": None, "tick_hz": TICK_HZ,
                "chain": {"collab_run": "live", "mission_run": self.run.name, "scene": sc["scene_id"]},
                "snapshots": len(self.snapshots), "live": True}

    def poll(self) -> None:
        with self._lock:
            self._poll()

    def _poll(self) -> None:
        src = self.src()
        if src is None:
            return
        # The end marker first: a finished run may append no telemetry after it, and the stream must still end.
        self.ended = (src / "results.yaml").exists()
        p = src / "telemetry.jsonl"
        with open(p, "rb") as f:
            f.seek(self.offset)
            data = f.read()
        nl = data.rfind(b"\n")
        if nl < 0:
            return
        self.offset += nl + 1
        for line in data[: nl + 1].decode().splitlines():
            d = json.loads(line)
            kind = d.get("type")
            if kind == "pose":
                self.poses[d["agent"]] = (d["t_s"], d["xy"][0], d["xy"][1], d["yaw"])
                self.t_s = max(self.t_s, d["t_s"])
            elif kind == "snapshot":
                self.snapshots.append((d["t_s"], src / d["png"]))
                self.egress = d.get("egress_xy")
            elif kind == "decision":
                r = d["record"]
                c = r.get("command", {})
                wp = c.get("waypoints_xy_m") or []
                to = f" to ({wp[-1][0]:.1f}, {wp[-1][1]:.1f})" if wp else ""
                self.events.append((r["t_s"], "decision",
                                    f"agent {r['agent']}: {r['result']} ({c.get('kind', 'none')}{to}) — "
                                    f"{r['chooser']} choice of {r.get('candidates_total', 0)} candidates · "
                                    f"{'executed' if r['executed'] else 'not executed'}"))
            elif kind == "mover_state":
                s = d["state"]
                verdict = "accepted" if s["accepted"] else f"REFUSED ({s['reason']})"
                self.events.append((s["applied_ns"] * 1e-9, "action",
                                    f"agent {d['agent']} mover: {s['kind']} {verdict}, "
                                    f"{s['segments']} segments until t={s['t_end_s']:.1f} s"))

    def frame(self, seq: int, since: float) -> dict:
        agents = []
        for k in (0, 1):
            if k in self.poses:
                t, x, y, yaw = self.poses[k]
                agents.append({"id": k, "merged": True, "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4)})
            else:
                agents.append({"id": k, "merged": True, "x": None, "y": None, "yaw": None})
        mv = len(self.snapshots) - 1 if self.snapshots else None
        events = [{"t_s": t, "kind": k, "text": x} for t, k, x in self.events if t > since]
        return {"t_s": round(self.t_s, 3), "seq": seq, "agents": agents, "bytes": {"total": 0}, "rate_Bps": 0,
                "map_version": mv, "route": {"egress": self.egress, "agent": None}, "events": events,
                "live": True, "ended": self.ended}


class FrameLog:
    def __init__(self, run: Path):
        self.f = open(run / "console_frames.jsonl", "a", buffering=1)

    def put(self, client: str, f: dict) -> None:
        self.f.write(json.dumps({"wall_unix_s": round(time.time(), 3), "client": client, "seq": f["seq"],
                                 "t_s": f["t_s"], "events": len(f["events"]), "map_version": f["map_version"]}) + "\n")
