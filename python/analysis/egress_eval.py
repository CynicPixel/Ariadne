"""4.5 egress scoring (design EGRESS_4.5 §5/§7, D113): ariadne_volumetric_eval --egress 1's egress_routes.jsonl
against the scene ground truth (scene_gt.npz), with the 1.8 baseline's routes.jsonl on the same spool beside it.

Per replan: route found, length vs the GT responder geodesic from the route's exit, the fraction of route cells in GT
trav_responder (validity), and the predicted confidence. Summary: time to the first route and to the first GT-valid
route, the final route, every reroute with its cause, and the calibration samples (confidence, valid) pooled over the
replans (a degenerate plot, e.g. all valid, is reported as such, not hidden).

    python3 -I python/analysis/egress_eval.py --scene SCENE --run OUT [--baseline 1.8_RUN_DIR] --out OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def score(routes: list[dict], gt, target: tuple, conf_key: str) -> dict:
    trav = gt["trav_responder"].astype(bool)
    geo = {k: gt[f"geodesic_responder_from_start{k}_m"] for k in (0, 1)}
    rows = []
    for r in routes:
        row = {"t_s": float(r["t_s"]), "found": bool(r["found"])}
        if r["found"]:
            cells = np.array(r["cells"])
            ok = trav[cells[:, 0], cells[:, 1]]
            gl = float(geo[r["exit"]][target])
            row.update(exit=int(r["exit"]), length_m=float(r["length_m"]), gt_geodesic_m=gl,
                       length_ratio=float(r["length_m"]) / gl if np.isfinite(gl) and gl > 0 else None,
                       valid_fraction=float(ok.mean()), valid=bool(ok.all()), confidence=float(r[conf_key]),
                       invalid_cells=int((~ok).sum()))
        for k in ("reroute", "triggers", "expanded", "changed_cells", "ms"):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    found = [x for x in rows if x["found"]]
    valid = [x for x in found if x["valid"]]
    out = {"replans": len(rows), "routes_found": len(found), "routes_valid": len(valid),
           "time_to_first_route_s": found[0]["t_s"] if found else None,
           "time_to_first_gt_valid_route_s": valid[0]["t_s"] if valid else None,
           "final": rows[-1] if rows else None,
           "all_found_routes_valid": bool(found) and len(valid) == len(found)}
    conf = np.array([x["confidence"] for x in found]), np.array([x["valid"] for x in found], bool)
    out["calibration"] = {"samples": len(found), "degenerate": bool(found) and bool(conf[1].all() or not conf[1].any()),
                          "confidence_min": float(conf[0].min()) if found else None,
                          "confidence_max": float(conf[0].max()) if found else None}
    if found and not out["calibration"]["degenerate"]:
        bins = np.linspace(0, 1, 6)
        idx = np.clip(np.digitize(conf[0], bins) - 1, 0, 4)
        out["calibration"]["bins"] = [{"lo": float(bins[b]), "hi": float(bins[b + 1]), "n": int((idx == b).sum()),
                                       "valid_rate": float(conf[1][idx == b].mean()) if (idx == b).any() else None}
                                      for b in range(5)]
    return {"summary": out, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, help="1.8 ariadne_mission run dir (routes.jsonl, results.yaml)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    gt = np.load(a.scene / "scene_gt.npz")
    target = tuple(int(v) for v in gt["target_cell"])
    res = {"phase4_dstar": score([json.loads(l) for l in (a.run / "egress_routes.jsonl").read_text().splitlines()],
                                 gt, target, "confidence")}
    vol = yaml.safe_load((a.run / "results.yaml").read_text())
    res["phase4_dstar"]["summary"]["support_band_m"] = vol["egress"]["support_band_m"]
    res["phase4_dstar"]["summary"]["final_route_support"] = vol["egress"]["final_route_support"]
    if a.baseline:
        b = [json.loads(l) for l in (a.baseline / "routes.jsonl").read_text().splitlines()]
        res["baseline_1.8_astar"] = score([r for r in b if r["kind"] == "egress"], gt, target, "support_p5")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "egress_vs_gt.yaml").write_text(
        "# python/analysis/egress_eval.py (EGRESS_4.5 §5, D113). Generated.\n"
        + yaml.safe_dump({k: v["summary"] for k, v in res.items()}, sort_keys=False))
    with (a.out / "egress_vs_gt_rows.jsonl").open("w") as f:
        for k, v in res.items():
            for row in v["rows"]:
                f.write(json.dumps({"method": k, **row}) + "\n")
    print(yaml.safe_dump({k: v["summary"] for k, v in res.items()}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
