"""Collect a live run's committable evidence (design live-runtime.md §3; arch §0.2 "manifests or it didn't happen").

  python3 python/analysis/live_evidence.py RUN OUT

RUN = live.sh's live-<id> directory (after `live.sh replay`); OUT = evaluation/runs/live-agentic/live_loop/<name>. Copies the small files
(trace, mover answers and executed scripts, telemetry, console frame log, link log, results, manifest, replay check,
per-agent estimates, snapshots, logs) and records inputs.mcap by sha256 and size only (it is tens to hundreds of MB).
Writes loop.json: per decision, the command, the mover's answer, and the agent's ESTIMATED distance to its goal at the
decision and 5 s of sim time later (decide -> move -> observe, read from the files, not the logs). std-lib only.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    inner = [d for d in run.iterdir() if d.is_dir() and (d / "inputs.mcap").exists()]
    if len(inner) != 1:
        raise SystemExit(f"{run}: expected one live run directory with inputs.mcap")
    d = inner[0]
    out.mkdir(parents=True, exist_ok=True)
    for f in ("trace.jsonl", "commands.jsonl", "mover_state.jsonl", "telemetry.jsonl", "link.jsonl", "results.yaml",
              "manifest.yaml"):
        shutil.copy2(d / f, out / f)
    for f in ("console_frames.jsonl", "replay_check.json", "live_config.yaml", "link.ready", "container_ids.txt",
              "runtime.log", "sim.log", "console.log", "mover_agent_0.jsonl", "mover_agent_1.jsonl"):
        if (run / f).exists():
            shutil.copy2(run / f, out / f)
    shutil.copytree(d / "snapshots", out / "snapshots", dirs_exist_ok=True)
    for a in sorted((d / "raw").iterdir()):
        dst = out / "raw" / a.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("estimate_body.tum", "results.yaml", "vo_trace.csv"):
            if (a / f).exists():
                shutil.copy2(a / f, dst / f)
        if (a / "raw" / "groundtruth_body.tum").exists():
            shutil.copy2(a / "raw" / "groundtruth_body.tum", dst / "groundtruth_body.tum")
    mc = d / "inputs.mcap"
    rec = {"file": f"{run.name}/{d.name}/inputs.mcap (not committed)", "bytes": mc.stat().st_size, "sha256": sha256(mc)}

    poses = {0: [], 1: []}
    for l in (d / "telemetry.jsonl").read_text().splitlines():
        x = json.loads(l)
        if x.get("type") == "pose":
            poses[x["agent"]].append((x["t_s"], x["xy"]))

    def est(k, t):
        p = [v for v in poses[k] if v[0] <= t + 1e-9]
        return p[-1][1] if p else None

    answers = {}
    for l in (d / "mover_state.jsonl").read_text().splitlines():
        s = json.loads(l)["state"]
        answers[(int(s["frame_id"].split("_")[1].split("/")[0]), s["cmd_stamp_ns"])] = s
    loop = []
    for l in (d / "trace.jsonl").read_text().splitlines():
        r = json.loads(l)
        c = r.get("command", {})
        k, T = r["agent"], c.get("stamp_ns")
        s = answers.get((k, T))
        e = {"t_s": r["t_s"], "agent": k, "result": r["result"], "kind": c.get("kind"),
             "waypoints_xy_m": c.get("waypoints_xy_m"),
             "mover": None if s is None else {"accepted": s["accepted"], "reason": s["reason"],
                                              "segments": s["segments"], "t_end_s": s["t_end_s"]}}
        wp = c.get("waypoints_xy_m") or []
        if wp:
            p0, p1 = est(k, r["t_s"]), est(k, r["t_s"] + 5.0)
            if p0 and p1:
                e["est_dist_to_goal_m"] = [round(math.dist(p0, wp[-1]), 3), round(math.dist(p1, wp[-1]), 3)]
        loop.append(e)
    go = [e for e in loop if e["result"] == "go"]
    acc = [e for e in go if e["mover"] and e["mover"]["accepted"]]
    closer = [e for e in acc if "est_dist_to_goal_m" in e and e["est_dist_to_goal_m"][1] < e["est_dist_to_goal_m"][0]]
    cf = [json.loads(l) for l in (run / "console_frames.jsonl").read_text().splitlines()] if (run / "console_frames.jsonl").exists() else []
    summary = {
        "decisions": len(loop), "go": len(go), "go_accepted": len(acc),
        "go_refused": sum(1 for e in go if e["mover"] and not e["mover"]["accepted"]),
        "go_unanswered": sum(1 for e in go if e["mover"] is None),
        "accepted_and_estimate_moved_closer_within_5s": len(closer),
        "console": {"frames": len(cf), "nonzero_sim_time_frames": sum(1 for f in cf if f["t_s"] > 0),
                    "wall_span_s": round(cf[-1]["wall_unix_s"] - cf[0]["wall_unix_s"], 1) if cf else 0,
                    "max_sim_t_s": max((f["t_s"] for f in cf), default=0),
                    "clients": sorted({f["client"] for f in cf}),
                    "note": "served to live.sh's probe WebSocket client, not a human"},
        "recording": rec,
    }
    (out / "loop.json").write_text(json.dumps({"summary": summary, "decisions": loop}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
