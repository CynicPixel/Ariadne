#!/usr/bin/env python3
"""manim_pack: one live run dir + its scene -> a small JSON/CSV/PNG data pack for the demo film.

The Manim film (demo-video-session, docs/design/DEMO_VIDEO_BRIEF.md) re-renders REAL run data: worldgen
geometry, agent paths, keyframes, inter-agent edges, decisions, routes. This exporter reads only files the
run already wrote (plus the scene dir) and writes plain JSON/CSV plus copied PNGs - no runtime code touched.

    python3 python/analysis/manim_pack.py RUN_INNER_DIR SCENE_DIR OUT_DIR

RUN_INNER_DIR is the <sceneid>-live-<ts> dir (trace.jsonl beside raw/ and snapshots/). Camera frames are NOT
decoded here (inputs.mcap needs the C++ decoder): dump them with ariadne_frame_dump into OUT_DIR/frames/
afterwards (see README.md it writes). Known gaps are listed in meta.json["gaps"].
"""
import csv
import json
import shutil
import sys
from pathlib import Path


def read_jsonl(p):
    for line in open(p):
        line = line.strip()
        if line:
            yield json.loads(line)


def export_scene(scene_dir: Path, out: Path) -> dict:
    sc = json.load(open(scene_dir / "scene.json"))
    walls = []
    for s in sc.get("surfaces", []):
        if s.get("kind") != "wall":
            continue
        walls.append({k: v for k, v in s.items() if k not in ("pixel_sha256",)})
    geo = {
        "scene_id": sc["scene_id"],
        "seed": sc["seed"],
        "preset": sc.get("preset"),
        "grid": sc.get("grid"),
        "lengths_m": sc.get("lengths_m"),
        "wall_height_m": sc.get("wall_height_m"),
        "clearance": sc.get("clearance"),  # carries rho_responder / rho_agent
        "entrances": sc.get("entrances"),
        "target_xy_m": sc.get("target_xy_m"),
        "routes": sc.get("routes"),        # worldgen GT routes (segments)
        "walls": walls,
        "lights": sc.get("lights"),
    }
    json.dump(geo, open(out / "scene.json", "w"), indent=1)
    # The mission the run actually used (entrance subset, live.demo block when present).
    for m in ("mission_live.json", "mission_default.json"):
        if (scene_dir / m).exists():
            shutil.copy(scene_dir / m, out / m)
    return geo


def export_paths(run: Path, out: Path) -> dict:
    d = out / "paths"
    d.mkdir(exist_ok=True)
    est, gt = {}, {}
    for r in read_jsonl(run / "telemetry.jsonl"):
        t = r.get("type")
        if t == "pose":
            est.setdefault(r["agent"], []).append((r["t_ns"], r["xy_m"][0], r["xy_m"][1], r.get("yaw", "")))
        elif t == "gt_pose":
            gt.setdefault(r["agent"], []).append((r["t_ns"], r["xy_m"][0], r["xy_m"][1], r.get("yaw", "")))
    agents = sorted(set(est) | set(gt))
    for k in agents:
        for name, rows in (("est", est.get(k, [])), ("gt", gt.get(k, []))):
            with open(d / f"agent{k}_{name}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_ns", "x_m", "y_m", "yaw_rad"])
                w.writerows(sorted(rows))
    # Keyframes: vo_trace marks them; the pose comes from the estimator's own TUM at that stamp.
    for raw in sorted((run / "raw").iterdir()):
        if not raw.is_dir():
            continue
        agent = None
        for part in raw.name.split("-"):
            if part.startswith("agent"):
                agent = int(part[len("agent"):])
        if agent is None:
            continue
        tum = {}
        tum_file = raw / "estimate_body.tum"
        if tum_file.exists():
            for line in open(tum_file):
                v = line.split()
                if len(v) >= 8:
                    tum[int(round(float(v[0]) * 1e9))] = v[1:8]
        rows = []
        vo = raw / "vo_trace.csv"
        if vo.exists():
            with open(vo) as f:
                for r in csv.DictReader(f):
                    if r.get("keyframe") not in ("1", "true", "True"):
                        continue
                    t = int(r["t_ns"])
                    near = min(tum, key=lambda u: abs(u - t)) if tum else None
                    pose = tum.get(near, [""] * 7) if near is not None and abs(near - t) < 50_000_000 else [""] * 7
                    rows.append([r["frame"], t] + pose)
        with open(d / f"agent{agent}_keyframes.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "t_ns", "x", "y", "z", "qx", "qy", "qz", "qw"])
            w.writerows(rows)
    return {"agents": agents}


def export_collab(run: Path, out: Path) -> list:
    d = out / "collab"
    d.mkdir(exist_ok=True)
    gaps = []
    if (run / "collab_edges.csv").exists():
        shutil.copy(run / "collab_edges.csv", d / "edges.csv")
    else:
        gaps.append("no collab_edges.csv (collab off in this run)")
    # The CollabLive telemetry line (untyped): per-agent anchor/merged/verified + fleet counters.
    state, merges, prev_anchor = [], [], {}
    for r in read_jsonl(run / "telemetry.jsonl"):
        if r.get("type") is not None or "agents" not in r or "kept_inter_edges" not in r:
            continue
        t = r["t_ns"]
        for a in r["agents"]:
            state.append([t, a["agent"], a.get("anchor"), a.get("merged"), a.get("broker"),
                          a.get("edges_verified"), a.get("solves"), a.get("answers_deferred"),
                          r.get("kept_inter_edges"), r.get("verify", {}).get("calls"),
                          r.get("verify", {}).get("gated"), r.get("entry_gate", {}).get("tested"),
                          r.get("entry_gate", {}).get("rejected")])
            k = a["agent"]
            if k in prev_anchor and prev_anchor[k] != a.get("anchor"):
                merges.append([t, k, prev_anchor[k], a.get("anchor")])
            prev_anchor[k] = a.get("anchor")
    with open(d / "state_timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "agent", "anchor", "merged", "broker", "edges_verified", "solves",
                    "answers_deferred", "kept_inter_edges", "verify_calls", "verify_gated",
                    "entry_gate_tested", "entry_gate_rejected"])
        w.writerows(state)
    with open(d / "merge_events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "agent", "anchor_from", "anchor_to"])
        w.writerows(merges)
    # Per-barrier bytes by class, from the planner block of each decision record.
    seen, rows, classes = set(), [], []
    for r in read_jsonl(run / "trace.jsonl"):
        b = (r.get("planner") or {}).get("bytes")
        t = r.get("command", {}).get("stamp_ns", r.get("t_s"))
        if not isinstance(b, dict) or t in seen:
            continue
        seen.add(t)
        for c in b:
            if c not in classes:
                classes.append(c)
        rows.append((t, b))
    with open(d / "bytes_timeline.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns"] + classes)
        for t, b in sorted(rows):
            w.writerow([t] + [b.get(c, "") for c in classes])
    # Event counts per tick/kind from the D179 log (arrival-order substrate).
    if (run / "collab_events.jsonl").exists():
        counts = {}
        for e in read_jsonl(run / "collab_events.jsonl"):
            key = (e.get("t"), e.get("e"), e.get("kind", ""))
            counts[key] = counts.get(key, 0) + 1
        with open(d / "events_timeline.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_ns", "event", "kind", "count"])
            for (t, e, kind), n in sorted(counts.items(), key=lambda x: (x[0][0] or 0, str(x[0][1]))):
                w.writerow([t, e, kind, n])
    return gaps


def export_decisions(run: Path, out: Path):
    recs = []
    for r in read_jsonl(run / "trace.jsonl"):
        v = r.get("vision")
        if isinstance(v, dict):
            v = dict(v)
            v.pop("snapshot_png", None)  # path/blob the film does not need; slabs ship separately
            r = dict(r)
            r["vision"] = v
        recs.append(r)
    json.dump(recs, open(out / "decisions.json", "w"), indent=1)


def export_planning(run: Path, out: Path) -> list:
    d = out / "planning"
    d.mkdir(exist_ok=True)
    gaps = []
    idx = []
    for p in sorted((run / "snapshots").glob("slab_*.png")):
        shutil.copy(p, d / p.name)
        try:
            idx.append([int(p.stem.split("_")[1]), p.name])
        except ValueError:
            idx.append(["", p.name])
    for p in sorted((run / "snapshots").glob("p4slab_*.png")):
        shutil.copy(p, d / p.name)
    with open(d / "slab_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "file"])
        w.writerows(idx)
    gaps.append("egress route polylines over time are not logged as geometry; per-decision command waypoints"
                " are in decisions.json, the responder-route context is in the slab PNGs and scene routes")
    return gaps


def main():
    run, scene_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    gaps = ["per-keyframe feature overlays are not logged historically; only the final frame's observations"
            " exist (frames/overlay_final_agent<k>.json). Raw keyframe-time frames: dump with ariadne_frame_dump"
            " (see README.md)"]
    geo = export_scene(scene_dir, out)
    agents = export_paths(run, out)
    gaps += export_collab(run, out)
    export_decisions(run, out)
    gaps += export_planning(run, out)
    fr = out / "frames"
    fr.mkdir(exist_ok=True)
    for j in sorted((run / "snapshots").glob("cam_*.json")):
        shutil.copy(j, fr / ("overlay_final_" + j.stem.replace("cam_", "agent") + ".json"))
    for j in sorted((run / "snapshots").glob("cam_*.jpg")):
        shutil.copy(j, fr / ("final_" + j.stem.replace("cam_", "agent") + ".jpg"))
    manifest = {}
    if (run / "manifest.yaml").exists():
        shutil.copy(run / "manifest.yaml", out / "run_manifest.yaml")
    meta = {
        "run_dir": str(run),
        "scene_id": geo.get("scene_id"),
        "seed": geo.get("seed"),
        "agents": agents.get("agents"),
        "gaps": gaps,
        "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    (out / "README.md").write_text(
        "# manim data pack\n\n"
        "One live run, exported for the demo film. All positions are world metres (the scene frame);\n"
        "stamps are sim-time ns unless a column says otherwise.\n\n"
        "- scene.json: walls (worldgen surfaces), entrances, target_xy_m, clearance (rho_responder), GT routes.\n"
        "- mission_live.json / mission_default.json: the mission the run used (agents, topics, demo block).\n"
        "- paths/agent<k>_{est,gt}.csv: estimated and ground-truth 2D tracks; agent<k>_keyframes.csv: keyframe\n"
        "  frame index, stamp and SE(3) pose (TUM order x y z qx qy qz qw).\n"
        "- collab/edges.csv: every inter-agent candidate with verify outcome, reason and GT errors.\n"
        "- collab/state_timeline.csv: per-agent anchor/merged/broker + fleet counters every 5 s;\n"
        "  merge_events.csv: anchor transitions (the merges); bytes_timeline.csv: per-barrier bytes by class;\n"
        "  events_timeline.csv: D179 event counts per tick.\n"
        "- decisions.json: every trace record (vision summary -> decision -> command), snapshot_png dropped.\n"
        "- planning/slab_*.png (+ p4slab) and slab_index.csv: the occupancy/ESDF slabs over time.\n"
        "- frames/: final per-agent camera jpg + feature overlays; dump keyframe-time frames with:\n"
        "    ariadne_frame_dump --rec <live_to_rec OUT> --scene <SCENE> --agent k --frames <from keyframes.csv> --out frames/agent<k>\n"
        "- meta.json: agents, scene, the honest gaps list.\n")
    print(f"pack: {out} ({sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/1e6:.1f} MB,"
          f" {len(meta['files'])} files)")


if __name__ == "__main__":
    main()
