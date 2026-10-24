"""Numbers for the film read straight from committed evidence dirs (never retyped from DECISIONS.md tables).

  python3 python/video/ariadne_video/evidence.py   (prints every number the scenes will draw, with its source)

Published comparators are constants here with their citation, tagged "published" on screen.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EUROC_RUN = REPO / "evaluation/runs/2.6/euroc_full_aws/20261022T193510Z-cpux86-655144"  # D87 run 2 (loop fix)
FLIGHTS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy",
           "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
# ORB-SLAM3 stereo-inertial ATE (m), Campos et al., T-RO 2021, Table II; same order as FLIGHTS
ORB_SLAM3 = [0.036, 0.033, 0.035, 0.051, 0.082, 0.038, 0.014, 0.024, 0.032, 0.014, 0.024]
ORB_SLAM3_CITE = "ORB-SLAM3, T-RO 2021, Tab. II"


def euroc() -> list[dict]:
    """Per flight: full-system (loops on) keyframe-refined ATE RMSE (m), D57 metric, plus the comparator."""
    out = []
    for f, orb in zip(FLIGHTS, ORB_SLAM3):
        t = (EUROC_RUN / "runs" / f / "results.yaml").read_text()
        ate = float(re.search(r"loop_closed_keyframe_refined: \{ate_rmse_m: ([0-9.e-]+)", t).group(1))
        out.append({"flight": f, "short": f[:5].replace("_", ""), "ate_m": ate, "orb_m": orb})
    return out


D178 = REPO / "evaluation/runs/collab_d178/verdict"


def d178() -> dict:
    """Known-entry gate sweep (D178): per-cell merged ATE gated vs ungated, and true edges tested/rejected."""
    with open(D178 / "cells.csv", newline="") as f:
        cells = [{"scene": int(r["scene"]), "cell": r["cell"], "budgeted": r["budgeted"] == "True",
                  "seeds": int(r["seeds"]), "gated_m": float(r["ate_gated_m"]), "base_m": float(r["ate_base_m"])}
                 for r in csv.DictReader(f)]
    edges: dict[int, dict] = {}
    with open(D178 / "true_edges.csv", newline="") as f:
        for r in csv.DictReader(f):
            e = edges.setdefault(int(r["scene"]), {"true": 0, "tested": 0, "rejected_last": 0, "rejected_ever": 0})
            e["true"] += 1
            if int(r["tested"]) > 0:
                e["tested"] += 1
                e["rejected_last"] += int(r["rejected_at_last"])
                e["rejected_ever"] += int(int(r["rejected"]) > 0)
    return {"cells": cells, "true_edges": edges}


def main() -> int:
    g = d178()
    for c in g["cells"]:
        print(f"D178 scene {c['scene']} {c['cell']:15s} budgeted={c['budgeted']!s:5s} gated {c['gated_m']:.4f} m  "
              f"ungated {c['base_m']:.4f} m")
    print("D178 true edges:", g["true_edges"], "source", D178.relative_to(REPO))
    rows = euroc()
    for r in rows:
        print(f"{r['flight']:16s} ariadne {r['ate_m']:.4f}  orb-slam3 {r['orb_m']:.3f}  "
              f"{'<=' if round(r['ate_m'], 3) <= r['orb_m'] else '> '}")
    n = sum(round(r["ate_m"], 3) <= r["orb_m"] for r in rows)
    print(f"mean {sum(r['ate_m'] for r in rows) / len(rows):.4f} vs {sum(ORB_SLAM3) / len(ORB_SLAM3):.4f}; "
          f"at or better (published 3-dp precision) {n}/{len(rows)}; source {EUROC_RUN.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
