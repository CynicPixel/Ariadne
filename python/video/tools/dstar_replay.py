"""Beat 9 data: the project's D* Lite (cpp/planning/route/dstar_lite.cpp via tools/dstar_trace.cpp) on maze_demo_0.

Lattice: the scene's person-width traversability (scene_gt.npz trav_responder, simulator ground truth, 0.40 m
responder); start = the casualty's cell, goals = every exit (the robots' entry doors); no risk penalty. The
obstruction is a disc of rubble at the tightest cell of the middle of the first route, wide enough to close it.
The first search, the Update() repair and a fresh search on the blocked grid all run in the C++; this script only
prepares the grids and stores g (cost-to-exit, cells) before and after, plus the routes and expansion counts, in
data/dstar/.

  python3 python/video/tools/dstar_replay.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from ariadne_video.data import load_scene  # noqa: E402

IMAGE = "ariadne/core:dev"  # has OpenCV 5 in /opt/opencv5 (the binary links it)


def run(tmp: Path, trav0: np.ndarray, trav1: np.ndarray, start, goals) -> dict:
    trav0.astype(np.uint8).tofile(tmp / "trav0.u8")
    trav1.astype(np.uint8).tofile(tmp / "trav1.u8")
    H, W = trav0.shape
    args = [str(v) for v in (H, W, *start, *[x for g in goals for x in g])]
    out = subprocess.run(["docker", "run", "--rm", "--user", f"{__import__('os').getuid()}:{__import__('os').getgid()}",
                          "-v", f"{REPO}:/repo", "-v", f"{tmp}:/tmp/d", "--entrypoint", "/repo/build/video_tools/dstar_trace",
                          IMAGE, "/tmp/d", *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    data = ROOT / "data" / "maze_demo_0"
    sc = load_scene(data)
    z = np.load(data / "scene_gt.npz")
    tr, esdf = z["trav_responder"].astype(bool), z["esdf_m"]
    start = tuple(int(v) for v in z["target_cell"])
    goals = [tuple(int(v) for v in s) for s in z["start_cells"]]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        first = run(tmp, tr, tr, start, goals)
        route = [tuple(p) for p in first["path0"]]
        n = len(route)
        k = min(route[n // 5: 4 * n // 5], key=lambda rc: esdf[rc])  # the tightest corridor cell mid-route
        rr, cc = np.ogrid[:tr.shape[0], :tr.shape[1]]
        block = (rr - k[0]) ** 2 + (cc - k[1]) ** 2 <= ((float(esdf[k]) + 0.15) / sc.cell_m) ** 2
        res = run(tmp, tr, tr & ~block, start, goals)
        g0 = np.fromfile(tmp / "g0.f64").reshape(tr.shape)
        g1 = np.fromfile(tmp / "g1.f64").reshape(tr.shape)
    dst = ROOT / "data" / "dstar"
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "fields.npz", g0=g0.astype(np.float32), g1=g1.astype(np.float32), block=block,
                        trav=tr)
    meta = dict(source="cpp/planning/route/dstar_lite.cpp DStarLite (first search, Update, fresh search), "
                       "maze_demo_0 trav_responder", start=start, goals=goals, block_center=k, cell_m=sc.cell_m, **res)
    (dst / "trace.json").write_text(json.dumps(meta))
    settled = lambda g: int(np.isfinite(g).sum())  # noqa: E731
    changed = int((~np.isclose(g0, g1, equal_nan=True) & (np.isfinite(g0) | np.isfinite(g1))).sum())
    print({k_: res[k_] for k_ in ("cost0", "cost1", "fresh_cost", "expanded_first", "expanded_repair",
                                   "expanded_fresh", "changed_cells", "exit0", "exit1")},
          "len", len(res["path0"]), len(res["path1"]), "finite g", settled(g0), settled(g1), "g changed", changed,
          "trav cells", int(tr.sum()))


if __name__ == "__main__":
    main()
