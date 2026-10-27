"""Evaluate a 1.8 mission run against the scene ground truth (design mission.md). Evaluation
only: GT is read here and nowhere in the planner.

  python -m analysis.mission_eval --run MISSION_RUN --scene SCENE --collab-run SIM_COLLAB_RUN \
      --config configs/mission/mission_v1.yaml [--compare SOLO_RUN ...]

Checks and metrics (written to MISSION_RUN/eval.yaml, image to MISSION_RUN/map_vs_gt.png):
- frame anchor: the anchor agent's estimated body poses over the first `anchor_window_s`, mapped through
  the briefing anchor, against GT; must be <= `anchor_max_err_m` or the run is rejected (exit 1). In the
  merged run agent 1's first seconds are also mapped through the merge (informational: merge accuracy).
- route validity vs GT: fraction of route cells in GT trav_responder (egress) / trav_agent (agent
  route); route length vs the GT geodesic from the same exit to the target.
- the first valid egress route (time, validity vs GT); route confidence (support p5).
- coverage: GT-free cells the map observed free; map precision: map-free cells GT-free, map-occupied
  cells GT-not-free.
- occupied-precision cause: distance (cells) from each map-occupied, GT-free cell to GT structure.
- collaboration comparison: time-to-first-valid-route of merged vs each agent alone (+-1 replan period).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

UNKNOWN, FREE, OCC = 0, 1, 2


def read_tum(p: Path) -> np.ndarray:
    return np.array([[float(v) for v in l.split()[:8]] for l in p.read_text().splitlines()
                     if l.strip() and not l.startswith("#")])


def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    return np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                     [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                     [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])


def pose12(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, :] = np.array(v, float).reshape(3, 4)
    return T


def anchor_check(T: np.ndarray, collab_run: Path, agent: int, window_s: float) -> dict:
    d = next((collab_run / "raw").glob(f"*-agent{agent}-*"))
    est, gt = read_tum(d / "estimate_body.tum"), read_tum(d / "raw" / "groundtruth_body.tum")
    gmap = {round(r[0], 6): r for r in gt}
    t0 = est[0, 0]
    pos_err, rot_err = [], []
    for r in est:
        if r[0] - t0 > window_s:
            break
        g = gmap.get(round(r[0], 6))
        if g is None:
            continue
        pw = T[:3, :3] @ r[1:4] + T[:3, 3]
        pos_err.append(float(np.linalg.norm(pw - g[1:4])))
        Rw = T[:3, :3] @ quat_to_R(*r[4:8])
        dR = Rw.T @ quat_to_R(*g[4:8])
        rot_err.append(float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))))
    if not pos_err:
        raise ValueError("anchor check: no estimate/GT pairs in the window")
    return {"poses": len(pos_err), "max_pos_err_m": max(pos_err), "max_rot_err_deg": max(rot_err)}


def route_metrics(route: dict, trav: np.ndarray, geodesic: np.ndarray, target: tuple, g: float) -> dict:
    if not route["found"]:
        return {"found": False}
    cells = np.array(route["cells"])
    ok = trav[cells[:, 0], cells[:, 1]]
    gt_len = float(geodesic[target])   # geodesic from the route's exit, evaluated at the target
    return {"found": True, "length_m": route["length_m"], "gt_geodesic_m": gt_len,
            "length_ratio": route["length_m"] / gt_len if np.isfinite(gt_len) and gt_len > 0 else None,
            "valid_fraction": float(ok.mean()), "valid": bool(ok.all()), "support_p5": route["support_p5"],
            "exit": route["exit"], "t_s": route["t_s"]}


def draw(slab, gtfree, routes_final, gt_route, target, exits, out: Path, scale=3):
    H, W = slab.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[slab == UNKNOWN] = (58, 52, 48)      # BGR
    img[slab == FREE] = (222, 226, 228)
    img[slab == OCC] = (24, 24, 24)
    edge = gtfree.astype(np.uint8) - cv2.erode(gtfree.astype(np.uint8), np.ones((3, 3), np.uint8))
    img[edge > 0] = (60, 60, 210)          # GT free-space boundary (red)
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    def poly(cells, color, th):
        pts = np.array([[c * scale + scale // 2, r * scale + scale // 2] for r, c in cells], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, color, th, cv2.LINE_AA)
    poly(gt_route, (200, 120, 30), 5)       # GT route: blue, wide
    for kind, color in (("agent", (200, 200, 40)), ("egress", (30, 140, 245))):
        r = routes_final.get(kind)
        if r and r["found"]:
            poly(r["cells"], color, 2)       # agent route: teal; egress route: orange
    for (r, c), color in [((target[0], target[1]), (40, 40, 230))] + [((e[0], e[1]), (60, 170, 60)) for e in exits]:
        cv2.circle(img, (c * scale + 1, r * scale + 1), 6, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.flipud(img))  # north up


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--run", "--scene", "--collab-run", "--config"):
        ap.add_argument(k, required=True)
    ap.add_argument("--compare", nargs="*")
    a = ap.parse_args(argv)
    run_dir, scene = Path(a.run), Path(a.scene)
    cfg = yaml.safe_load(Path(a.config).read_text())["evaluation"]
    res = yaml.safe_load((run_dir / "results.yaml").read_text())
    gt = np.load(scene / "scene_gt.npz")
    g = float(gt["grid_m"])
    slab = cv2.imread(str(run_dir / "slab.png"), cv2.IMREAD_UNCHANGED)
    if slab.shape != gt["free"].shape:
        raise ValueError("map grid and GT grid differ")
    target = tuple(int(v) for v in res["target_cell"])
    if target != tuple(int(v) for v in gt["target_cell"]):
        raise ValueError(f"target cell {target} != GT {tuple(gt['target_cell'])}")

    T_W_Oa = pose12(res["T_W_Oanchor"])
    anchor = anchor_check(T_W_Oa, Path(a.collab_run), int(res["anchor_agent"]), cfg["anchor_window_s"])
    anchor["agent"] = int(res["anchor_agent"])
    if anchor["max_pos_err_m"] > cfg["anchor_max_err_m"]:
        print(yaml.safe_dump({"anchor": anchor}), file=sys.stderr)
        raise SystemExit("frame anchor check FAILED: briefing T_W_O0 disagrees with GT")

    routes = [json.loads(l) for l in (run_dir / "routes.jsonl").read_text().splitlines()]
    final = {r["kind"]: r for r in routes if r["final"]}
    geo = {("egress", k): gt[f"geodesic_responder_from_start{k}_m"] for k in (0, 1)}
    geo.update({("agent", k): gt[f"geodesic_agent_from_start{k}_m"] for k in (0, 1)})
    trav = {"egress": gt["trav_responder"], "agent": gt["trav_agent"]}
    out = {"agents": str(res["agents"]), "anchor": anchor}
    if str(res["agents"]) == "merged":   # agent 1 through the 1.6 merge: how well the loop placed it
        out["agent1_via_merge"] = anchor_check(T_W_Oa @ pose12(res["T_O0_O1"]), Path(a.collab_run), 1,
                                               cfg["anchor_window_s"])
    for kind in ("egress", "agent"):
        r = final[kind]
        out[f"final_{kind}"] = route_metrics(r, trav[kind], geo[(kind, r["exit"])] if r["found"] else None, target, g)
    first = next((r for r in routes if r["kind"] == "egress" and r["found"]), None)
    out["first_valid_egress"] = (route_metrics(first, trav["egress"], geo[("egress", first["exit"])], target, g)
                                 if first else {"found": False})
    gtfree = gt["free"].astype(bool)
    out["map"] = {"coverage": float(((slab == FREE) & gtfree).sum() / gtfree.sum()),
                  "free_precision": float(((slab == FREE) & gtfree).sum() / max((slab == FREE).sum(), 1)),
                  "occupied_precision": float(((slab == OCC) & ~gtfree).sum() / max((slab == OCC).sum(), 1))}
    # Occupied-precision cause: how far is each false-occupied cell from GT structure? 1 cell = a wall plane on
    # a cell boundary putting hits into the last GT-free cell (convention); further = depth noise.
    from scipy.ndimage import distance_transform_cdt
    dist = distance_transform_cdt(gtfree, metric="chessboard")   # cells to the nearest GT-not-free cell
    fo = dist[(slab == OCC) & gtfree]
    out["map"]["false_occupied_cells"] = int(fo.size)
    out["map"]["false_occupied_by_distance_cells"] = {int(k): int((fo == k).sum()) for k in range(1, 4)}
    out["map"]["false_occupied_beyond_3_cells"] = int((fo > 3).sum())
    out["time_to_first_valid_route_s"] = res["time_to_first_valid_route_s"]
    out["replan_period_s"] = float(yaml.safe_load(Path(a.config).read_text())["replan_period_s"])
    rows = [{"agents": str(res["agents"]), "first_valid_route_s": res["time_to_first_valid_route_s"],
             "final_exit": res["final_egress"]["exit"], "final_length_m": res["final_egress"]["length_m"],
             "coverage": out["map"]["coverage"]}]
    for c in a.compare or []:
        cr = yaml.safe_load((Path(c) / "results.yaml").read_text())
        cs = cv2.imread(str(Path(c) / "slab.png"), cv2.IMREAD_UNCHANGED)
        rows.append({"agents": str(cr["agents"]), "first_valid_route_s": cr["time_to_first_valid_route_s"],
                     "final_exit": cr["final_egress"]["exit"], "final_length_m": cr["final_egress"]["length_m"],
                     "coverage": float(((cs == FREE) & gtfree).sum() / gtfree.sum())})
    out["comparison"] = {"note": "first-valid times are replan instants: the route appeared in (t - period, t]",
                         "rows": rows}
    draw(slab, gtfree, final, gt[f"route_responder_start{final['egress']['exit']}"] if final["egress"]["found"]
         else gt["route_responder_start0"], target, res["exit_cells"], run_dir / "map_vs_gt.png")
    import os
    out["provenance"] = {"git_commit": os.environ.get("ARIADNE_GIT_COMMIT", "unknown"),
                         "git_dirty": os.environ.get("ARIADNE_GIT_DIRTY", "unknown"), "command": " ".join(sys.argv)}
    (run_dir / "eval.yaml").write_text("# mission_eval (design mission.md). Generated.\n" +
                                       yaml.safe_dump(out, sort_keys=False))
    print(yaml.safe_dump(out, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
