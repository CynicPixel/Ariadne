"""Corridor degeneracy gate ordering, Phase 5 gate bullet 4, scored exactly as pre-registered in D148 part A. numpy only.

  python3 -I corridor_gate.py SCENE_DIR RUN_DIR [RUN_DIR ...] [--out corridor_gate.json]

SCENE_DIR is the worldgen scene (scene.json + esdf_f32.bin). Each RUN_DIR is one agent's ariadne_sim_vo output:
loop_candidates.csv (every gated candidate with its GT label), loop_trace.jsonl, vo_trace.csv and
raw/groundtruth_body.tum.

Labels come from GT only. Clearance d = the scene ESDF at a keyframe's GT xy: narrow d <= 0.30 m, corridor
0.30 < d <= 0.50 m, open d > 0.50 m. A candidate takes the more confined label of its two keyframes; it is
*confined* if that label is narrow or corridor.
Hard criterion (ordering): no candidate with degenerate = true has pcm or gnc_inlier true.
Met: the ordering holds, >= 1 confined candidate is refused, and no GT-bad confined candidate ends as a GNC inlier.
No confined candidate at all -> "not exercised". Thresholds are D148's and are not arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

NARROW_M, CORRIDOR_M = 0.30, 0.50
RANK = {"open": 0, "corridor": 1, "narrow": 2}


def load_esdf(scene_dir: Path):
    meta = json.loads((scene_dir / "scene.json").read_text())["esdf"]
    grid = np.fromfile(scene_dir / meta["file"], dtype="<f4").reshape(meta["rows"], meta["cols"])
    return grid, float(meta["cell_m"]), np.asarray(meta["origin_m"], float)


def clearance(grid, cell, origin, xy):
    """ESDF value of the cell holding xy. Row 0 is south: the row index grows with y."""
    c = int(np.floor((xy[0] - origin[0]) / cell))
    r = int(np.floor((xy[1] - origin[1]) / cell))
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        raise ValueError(f"GT position {xy} is outside the ESDF grid")
    return float(grid[r, c])


def label_of(d: float) -> str:
    return "narrow" if d <= NARROW_M else "corridor" if d <= CORRIDOR_M else "open"


def keyframe_times(run: Path) -> list[int]:
    """Keyframe id -> t_ns. The VO numbers keyframes 0, 1, 2, ... in creation order, one per keyframe=1 row."""
    with open(run / "vo_trace.csv") as f:
        return [int(r["t_ns"]) for r in csv.DictReader(f) if r["keyframe"] == "1"]


def gt_xy(run: Path):
    a = np.loadtxt(run / "raw" / "groundtruth_body.tum", comments="#")
    t, xy = a[:, 0], a[:, 1:3]

    def at(t_ns: int):
        i = int(np.argmin(np.abs(t - t_ns * 1e-9)))
        if abs(t[i] - t_ns * 1e-9) > 0.05:
            raise ValueError(f"{run}: no GT pose within 50 ms of t = {t_ns} ns")
        return xy[i]
    return at


def score_run(run: Path, grid, cell, origin):
    kf_t = keyframe_times(run)
    at = gt_xy(run)
    kf_d = [clearance(grid, cell, origin, at(t)) for t in kf_t]
    # Consistency of the id -> time mapping: every loop event's t_ns is its query keyframe's time.
    with open(run / "loop_trace.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for c in e.get("candidates", []):
                if kf_t[c["a"]] != e["t_ns"]:
                    raise ValueError(f"{run}: keyframe {c['a']} maps to {kf_t[c['a']]} ns, event says {e['t_ns']}")
    rows = []
    with open(run / "loop_candidates.csv") as f:
        for r in csv.DictReader(f):
            a, b = int(r["a"]), int(r["b"])
            la, lb = label_of(kf_d[a]), label_of(kf_d[b])
            rows.append({
                "agent_run": run.name, "a": a, "b": b, "d_a": kf_d[a], "d_b": kf_d[b],
                "label": max(la, lb, key=RANK.get),
                "degenerate": r["degenerate"] == "1", "reasons": [s for s in r["reasons"].split("|") if s],
                "pcm": r["pcm"] == "1", "gnc_inlier": r["gnc_inlier"] == "1", "gt": r["label"],
                "corridor": float(r["corridor"]) if "corridor" in r else float("nan"),
            })
    return rows, kf_d


def summarise(rows, kf_d_all):
    out = {"keyframes": {k: int(sum(label_of(d) == k for d in kf_d_all)) for k in RANK},
           "min_keyframe_clearance_m": float(min(kf_d_all)) if kf_d_all else None}
    violations = [r for r in rows if r["degenerate"] and (r["pcm"] or r["gnc_inlier"])]
    out["ordering_violations"] = len(violations)
    per = {}
    for k in RANK:
        sel = [r for r in rows if r["label"] == k]
        ref = [r for r in sel if r["degenerate"]]
        acc = [r for r in sel if not r["degenerate"]]
        cor = np.array([r["corridor"] for r in sel], float)
        per[k] = {
            "candidates": len(sel), "refused": len(ref),
            "reasons": dict(Counter(s for r in ref for s in r["reasons"])),
            "corridor_score_median": float(np.nanmedian(cor)) if np.isfinite(cor).any() else None,
            "refused_gt": dict(Counter(r["gt"] for r in ref)), "accepted_gt": dict(Counter(r["gt"] for r in acc)),
            "accepted_bad_removed_by_pcm_gnc": sum(r["gt"] == "bad" and not r["gnc_inlier"] for r in acc),
            "gnc_inlier_bad": sum(r["gt"] == "bad" and r["gnc_inlier"] for r in sel),
        }
    out["per_label"] = per
    confined = [r for r in rows if r["label"] != "open"]
    if not confined:
        out["bullet"] = "not exercised"
    else:
        met = (not violations and any(r["degenerate"] for r in confined)
               and not any(r["gt"] == "bad" and r["gnc_inlier"] for r in confined))
        out["bullet"] = "met" if met else "not met"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene", type=Path)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    grid, cell, origin = load_esdf(a.scene)
    rows, kf_d_all = [], []
    for run in a.runs:
        r, d = score_run(run, grid, cell, origin)
        rows += r
        kf_d_all += d
    res = summarise(rows, kf_d_all)
    res["inputs"] = {"scene": str(a.scene), "runs": [str(r) for r in a.runs]}
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
        with open(a.out.with_suffix(".candidates.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["agent_run"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "reasons": ";".join(r["reasons"])})
    if res["min_keyframe_clearance_m"] is not None and res["min_keyframe_clearance_m"] < 0:
        print("WARNING: a keyframe's GT position lies inside an obstacle: GT and ESDF frames disagree", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
