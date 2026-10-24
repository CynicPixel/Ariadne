"""Read-only loaders from real run data to plain numpy/dict structures for the Manim scenes (DEMO_VIDEO_BRIEF).

  python3 python/video/ariadne_video/data.py SCENE_DIR [RUN_DIR]   (prints a summary; numpy only)

SCENE_DIR is a worldgen output (scene.json + scene_gt.npz, e.g. maze_demo_0). RUN_DIR is a live run dir
(telemetry.jsonl, trace.jsonl, collab_edges.csv, snapshots/), or B's data pack once it has the same files.
Nothing here invents a number: every value a scene draws comes from these files, so the on-screen tag "measured"
traces back to a path. World frame as scene.json: metres, x east, y north, grid row 0 = SOUTH.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Scene:
    scene_id: str
    seed: int
    cell_m: float
    origin_xy_m: tuple[float, float]  # SW corner of grid cell (0, 0)
    free: np.ndarray  # bool [rows, cols], structure_free (props excluded)
    esdf_m: np.ndarray  # float32 [rows, cols], metres to the nearest obstacle
    trav_responder: np.ndarray  # bool, passable at rho_responder (invariant 4)
    walls: list[tuple[float, float, float, float]]  # x0, y0, x1, y1 (m): free/occupied boundary, merged runs
    starts: list[dict]  # scene.json entrances
    target_xy_m: tuple[float, float]
    rho_responder_m: float
    gt_routes_responder: dict[int, np.ndarray] = field(default_factory=dict)  # agent -> [n, 2] xy (m)

    def cell_to_xy(self, rc: np.ndarray) -> np.ndarray:
        """[n, 2] (row, col) grid cells -> [n, 2] cell-centre xy in metres."""
        rc = np.asarray(rc, dtype=float)
        return np.stack([self.origin_xy_m[0] + (rc[:, 1] + 0.5) * self.cell_m,
                         self.origin_xy_m[1] + (rc[:, 0] + 0.5) * self.cell_m], axis=1)


def boundary_segments(occ: np.ndarray, cell_m: float, ox: float, oy: float) -> list[tuple[float, float, float, float]]:
    """Axis-aligned free/occupied boundary of a grid, each straight run merged into one segment.

    Horizontal edges sit between rows r-1 and r where occupancy flips; vertical ones between columns. The grid is
    padded with its own edge values, so the lattice border itself is not drawn as a wall."""
    occ = np.pad(occ, 1, mode="edge")
    segs = []
    for axis in (0, 1):
        flip = occ[1:, :] != occ[:-1, :] if axis == 0 else occ[:, 1:] != occ[:, :-1]
        for i, line in enumerate(flip if axis == 0 else flip.T):
            idx = np.flatnonzero(line)
            if idx.size == 0:
                continue
            breaks = np.flatnonzero(np.diff(idx) > 1)
            for a, b in zip(np.r_[idx[0], idx[breaks + 1]], np.r_[idx[breaks], idx[-1]]):
                # line i lies between padded rows/cols i and i+1 = unpadded edge coordinate i; padded cells a..b
                # along it are unpadded cells a-1..b-1, so the run spans [a-1, b) in cell units
                k0, k1 = a - 1, b
                if axis == 0:
                    y = oy + i * cell_m
                    segs.append((ox + k0 * cell_m, y, ox + k1 * cell_m, y))
                else:
                    x = ox + i * cell_m
                    segs.append((x, oy + k0 * cell_m, x, oy + k1 * cell_m))
    return segs


def load_scene(d: str | Path) -> Scene:
    d = Path(d)
    s = json.loads((d / "scene.json").read_text())
    z = np.load(d / "scene_gt.npz")
    cell = float(s["grid"]["cell_m"])
    ox, oy = (u * cell for u in s["grid"]["origin_units"])
    free = z["structure_free"]
    routes = {}
    for a in range(len(s["entrances"])):
        k = f"route_responder_start{a}"
        if k in z.files:
            rc = z[k]
            routes[a] = np.stack([ox + (rc[:, 1] + 0.5) * cell, oy + (rc[:, 0] + 0.5) * cell], axis=1)
    return Scene(scene_id=s["scene_id"], seed=int(s["seed"]), cell_m=cell, origin_xy_m=(ox, oy), free=free,
                 esdf_m=z["esdf_m"], trav_responder=z["trav_responder"],
                 walls=boundary_segments(~free, cell, ox, oy), starts=s["entrances"],
                 target_xy_m=tuple(s["target_xy_m"]), rho_responder_m=float(s["clearance"]["rho_responder_m"]),
                 gt_routes_responder=routes)


@dataclass
class Run:
    path: Path
    scene_id: str
    end_s: float
    poses: dict[int, np.ndarray]  # agent -> [n, 4] t_s, x, y, yaw (estimate)
    gt_poses: dict[int, np.ndarray]  # agent -> [n, 4] t_s, x, y, yaw (ground truth)
    collab: list[dict]  # telemetry records without a type: per-5 s broker/verify/entry-gate counters
    trace: list[dict]  # decision records, as written
    edges: list[dict]  # collab_edges.csv rows (header lines repeated in the file are dropped)
    snapshots: list[tuple[float, Path]]  # (t_s, occupancy slab png)


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v


def load_run(d: str | Path) -> Run:
    d = Path(d)
    res = {k.strip(): v.split("#")[0].strip() for k, _, v in
           (l.partition(":") for l in (d / "results.yaml").read_text().splitlines()) if v and not k.startswith(" ")}
    est: dict[int, list] = {}
    gt: dict[int, list] = {}
    collab, snaps = [], []
    for line in (d / "telemetry.jsonl").read_text().splitlines():
        r = json.loads(line)
        t = r.get("type")
        if t in ("pose", "gt_pose"):
            (est if t == "pose" else gt).setdefault(r["agent"], []).append((r["t_s"], *r["xy"], r["yaw"]))
        elif t == "snapshot":
            snaps.append((r["t_s"], d / r["png"]))
        elif t is None and "t_ns" in r:
            collab.append(r)
    edges = []
    with open(d / "collab_edges.csv", newline="") as f:
        rows = csv.reader(f)
        hdr = next(rows)
        edges = [{k: _num(v) for k, v in zip(hdr, row)} for row in rows if row and row != hdr]
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines() if l.strip()]
    arr = lambda m: {a: np.array(v) for a, v in sorted(m.items())}  # noqa: E731
    return Run(path=d, scene_id=res.get("scene_id", "").strip("'\""), end_s=float(res.get("end_s", "nan")),
               poses=arr(est), gt_poses=arr(gt), collab=collab, trace=trace, edges=edges, snapshots=sorted(snaps))


def main(argv: list[str]) -> int:
    sc = load_scene(argv[1])
    print(f"scene {sc.scene_id} seed {sc.seed}: grid {sc.free.shape} @ {sc.cell_m} m, origin {sc.origin_xy_m}, "
          f"{len(sc.walls)} wall segments, {len(sc.starts)} entrances, target {sc.target_xy_m}, "
          f"rho_responder {sc.rho_responder_m} m, GT responder routes {[len(v) for v in sc.gt_routes_responder.values()]}")
    if len(argv) > 2:
        run = load_run(argv[2])
        same = "SAME scene" if run.scene_id == sc.scene_id else f"DIFFERENT scene ({run.scene_id})"
        print(f"run {run.path.name}: {same}, end {run.end_s} s, poses {[len(v) for v in run.poses.values()]}, "
              f"gt {[len(v) for v in run.gt_poses.values()]}, decisions {len(run.trace)}, edges {len(run.edges)}, "
              f"slabs {len(run.snapshots)}")
        reasons: dict[str, int] = {}
        for e in run.edges:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        print("edge verify reasons:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))
        if run.collab:
            c = run.collab[-1]
            print(f"last collab record t={c['t_ns'] / 1e9:.0f} s: kept_inter_edges {c['kept_inter_edges']}, "
                  f"verify {c['verify']}, entry_gate {c['entry_gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
