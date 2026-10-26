"""Beat 7 data: the project's CBBA (cpp/planning/mission/allocation.cpp via tools/cbba_trace.cpp) run on maze_demo_0.

Instance, built the way BuildAuction does it (phase4_planner.cpp), with two-task bundles (L_t = 2, the design's
upper bound and a reported row; the recorded missions ran L_t = 1):
  tasks     the scene's unexplored rooms (scene.json topology), goal = room centre      [illustrative task set]
  value     c̄ = the room's free floor area / its p90 over rooms (what a robot would learn there)
  distance  geodesic on the agent-width lattice (scene_gt.npz trav_agent), start -> goal
  rate      lambda_path / p90(path) with lambda_path = 1 (nbv.hpp default)
  legs      the same geodesic, goal -> goal
  bid       the marginal of S_i(p) = sum_j c̄ exp(-rate d_j(p)), warped (allocation.cpp BuildBundle)
Writes data/cbba/trace.json: per-round snapshots, the converged winners, the exact optimum and SGA's total, and the
geodesic polylines (metres) start -> room and room -> room.

  g++ -std=c++20 -O2 -I cpp python/video/tools/cbba_trace.cpp cpp/planning/mission/allocation.cpp \
      -o build/video_tools/cbba_trace
  python3 python/video/tools/cbba_replay.py
"""
from __future__ import annotations

import heapq
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT))
from ariadne_video.data import load_scene  # noqa: E402

NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]


def dijkstra(trav: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    R, C = trav.shape
    d = np.full((R, C), np.inf)
    d[src] = 0.0
    h = [(0.0, *src)]
    while h:
        dd, r, c = heapq.heappop(h)
        if dd > d[r, c]:
            continue
        for dr, dc, w in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and trav[rr, cc] and (dr == 0 or dc == 0 or (trav[r + dr, c] and trav[r, c + dc])):
                if dd + w < d[rr, cc]:
                    d[rr, cc] = dd + w
                    heapq.heappush(h, (dd + w, rr, cc))
    return d


def descend(d: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    path, (r, c) = [start], start
    while d[r, c] > 0:
        r, c = min(((r + dr, c + dc) for dr, dc, _ in NB), key=lambda q: d[q])
        path.append((r, c))
    return path


def p90(v) -> float:
    return float(np.percentile(np.asarray(v, dtype=float), 90))


def main():
    data = ROOT / "data" / "maze_demo_0"
    sc = load_scene(data)
    rooms = json.loads((data / "scene.json").read_text())["topology"]["rooms"]
    ta = np.load(data / "scene_gt.npz")["trav_agent"]
    ox, oy = sc.origin_xy_m
    cell = lambda x, y: (int((y - oy) / sc.cell_m), int((x - ox) / sc.cell_m))  # noqa: E731
    names = sorted(rooms)
    goals = [((rooms[k][0] + rooms[k][2]) / 2, (rooms[k][1] + rooms[k][3]) / 2) for k in names]
    area = []
    for k in names:
        x0, y0, x1, y1 = rooms[k]
        (r0, c0), (r1, c1) = cell(x0, y0), cell(x1, y1)
        area.append(float(sc.free[r0:r1, c0:c1].sum()) * sc.cell_m ** 2)
    value = np.array(area) / p90(area)
    U, N = len(sc.starts), len(names)
    gf = [dijkstra(ta, cell(*g)) for g in goals]  # a field per room; geodesics are symmetric
    d0 = np.array([[gf[j][cell(*e["start_xy_m"])] * sc.cell_m for j in range(N)] for e in sc.starts])
    leg = np.array([[gf[k][cell(*goals[j])] * sc.cell_m for k in range(N)] for j in range(N)])
    rate = 1.0 / p90(d0.ravel())
    bid = value[None, :] * np.exp(-rate * d0)

    exe = REPO / "build/video_tools/cbba_trace"
    stdin = f"{U} {N} 2\n" + " ".join([f"{rate:.12g}"] * U) + "\n"
    stdin += "\n".join(" ".join(f"{v:.12g}" for v in value) for _ in range(U)) + "\n"
    stdin += "\n".join(" ".join(f"{v:.12g}" for v in row) for row in d0) + "\n"
    stdin += "\n".join(" ".join(f"{v:.12g}" for v in leg.ravel()) for _ in range(U)) + "\n"
    trace = json.loads(subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True).stdout)

    path = lambda j, xy: sc.cell_to_xy(np.array(descend(gf[j], cell(*xy)))).round(3).tolist()  # noqa: E731  xy -> room j
    routes = [[path(j, e["start_xy_m"]) for j in range(N)] for e in sc.starts]
    legs = [[path(k, goals[j]) if j != k else [] for k in range(N)] for j in range(N)]
    out = dict(source="cpp/planning/mission/allocation.cpp RunCbba (L_t = 2, all-to-all link), maze_demo_0",
               rooms=names, goals=goals, area_m2=area, value=value.tolist(), d0_m=d0.tolist(), rate=rate,
               bid=bid.tolist(), leg_m=leg.tolist(), routes=routes, legs=legs, **trace)
    dst = ROOT / "data" / "cbba"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "trace.json").write_text(json.dumps(out))
    np.set_printoptions(precision=3, suppress=True)
    print("rooms", names, "\narea", np.round(area, 1), "\nd0\n", d0, "\nbid\n", bid)
    for t, snaps in enumerate(trace["rounds"], 1):
        print("round", t, [(s["z"], np.round(s["y"], 3).tolist()) for s in snaps])
    print({k: trace[k] for k in ("winner", "total", "optimum", "sga_total", "n_rounds", "messages", "bytes")})


if __name__ == "__main__":
    main()
