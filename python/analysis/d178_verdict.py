"""The pre-registered D178 verdict (DECISIONS: D178 and its scoring amendment).

  python3 -I d178_verdict.py --gated SWEEP --base SWEEP --logged SWEEP --out DIR

--gated   the D178 sweep as staged (entry gate on in B2 + C): the ATE and corruption criteria are scored here
--base    the same sweep without the gate (3.2 rerun): scene 1's "not worse" reference
--logged  the logged rerun of the same cells (entry_gate.csv): the true-edge criterion, only if every gated run
          reproduces --gated exactly (merged ATE, and first-transmission bytes with D180's re-sends folded back in)

Each SWEEP is a collab_aws.sh results/runs directory. Writes DIR/verdict.yaml, DIR/cells.csv, DIR/true_edges.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

ROW = re.compile(r"^(greedy|spectral)_s(\d)_p(\d+)_seed(\d)$")
CAMP = re.compile(r"^camp_(\w+)_s(\d)_seed(\d)$")
GOOD_DEG = 10.0       # D171 §5's label: a true edge has pose error <= 10 deg
S0_LIMIT_M = 0.10     # scene 0 budgeted cells: merged ATE below this
S1_REL, S1_ABS = 0.10, 0.002   # scene 1: not worse by > 10 % (or 2 mm)
TRUE_REJ_LIMIT = 0.01


def load(d: Path) -> dict:
    out = {}
    for p in sorted(d.iterdir()):
        y = p / "results.yaml"
        if not y.exists():
            continue
        r = yaml.safe_load(y.read_text())
        oc = yaml.safe_load((p / "outcome.yaml").read_text())["outcome"] if (p / "outcome.yaml").exists() else None
        out[p.name] = {"dir": p, "r": r, "outcome": oc}
    return out


def gated(name: str, run: dict) -> bool:
    return bool(run["r"].get("entry_gate", {}).get("enabled")) and bool(ROW.match(name) or CAMP.match(name))


def ate(run: dict):
    return run["r"].get("ate_merged", {}).get("rmse_m")


def first_tx_with_resend(run: dict) -> int:
    """First-transmission wire bytes as a pre-D180 build counted them (re-sends were first transmissions then)."""
    r = run["r"]
    return int(r["total_wire_bytes"]["first"]) + sum(int(b.get("wire_resend", 0)) for b in r.get("bytes", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--logged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    G, B, L = load(a.gated), load(a.base), load(a.logged)

    # --- cells: arm x level (budgeted) and campaign variants, seed means ---------------------------------------
    cells = defaultdict(lambda: {"gated": [], "base": []})
    for name, run in G.items():
        m = ROW.match(name)
        key = (f"{m[1]}_p{m[3]}", int(m[2])) if m else None
        if not m and (c := CAMP.match(name)):
            key = (f"camp_{c[1]}", int(c[2]))
        if key is None or not gated(name, run):
            continue
        cells[key]["gated"].append(ate(run))
        if name in B:
            cells[key]["base"].append(ate(B[name]))
    rows = []
    for (cell, scene), v in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        g = statistics.mean(v["gated"])
        b = statistics.mean(v["base"]) if v["base"] else None
        rows.append({"scene": scene, "cell": cell, "budgeted": not cell.startswith("camp_"), "seeds": len(v["gated"]),
                     "ate_gated_m": round(g, 6), "ate_base_m": None if b is None else round(b, 6)})
    with open(a.out / "cells.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    s0 = [r for r in rows if r["scene"] == 0 and r["budgeted"]]
    s0_pass = sum(r["ate_gated_m"] < S0_LIMIT_M for r in s0)
    s1 = [r for r in rows if r["scene"] == 1 and r["ate_base_m"] is not None]
    s1_bad = [r for r in s1 if r["ate_gated_m"] > r["ate_base_m"] * (1 + S1_REL)
              and r["ate_gated_m"] - r["ate_base_m"] > S1_ABS]
    corrupt = sorted(n for n, run in G.items() if run["outcome"] == "corrupt")

    # --- true edges: the logged rerun, only if it is the same experiment ------------------------------------------
    mismatch = []
    for name, run in G.items():
        if not gated(name, run):
            continue
        lr = L.get(name)
        if lr is None:
            mismatch.append(f"{name}: missing in the logged rerun")
        elif ate(lr) != ate(run):
            mismatch.append(f"{name}: ATE {ate(run)} vs {ate(lr)}")
        elif first_tx_with_resend(lr) != int(run["r"]["total_wire_bytes"]["first"]):
            mismatch.append(f"{name}: first-tx {run['r']['total_wire_bytes']['first']} vs {first_tx_with_resend(lr)}")
        elif not (lr["dir"] / "entry_gate.csv").exists():
            mismatch.append(f"{name}: no entry_gate.csv")
    per_scene = defaultdict(lambda: {"true": 0, "tested": 0, "rejected_last": 0, "rejected_ever": 0, "ungated": 0})
    erows = []
    if not mismatch:
        for name, run in L.items():
            if not gated(name, run):
                continue
            scene = int((ROW.match(name) or CAMP.match(name))[2])
            dec = {}
            for e in csv.DictReader(open(run["dir"] / "entry_gate.csv")):
                k = frozenset({(int(e["agent_a"]), int(e["kf_a"])), (int(e["agent_b"]), int(e["kf_b"]))})
                dec[k] = e
            for e in csv.DictReader(open(run["dir"] / "edges.csv")):
                if e["reason"] != "verified" or e["pose_err_deg"] in ("", "nan"):
                    continue
                if float(e["pose_err_deg"]) > GOOD_DEG:
                    continue
                k = frozenset({(int(e["agent_a"]), int(e["kf_a"])), (int(e["agent_b"]), int(e["kf_b"]))})
                s = per_scene[scene]
                s["true"] += 1
                d = dec.get(k)
                if d is None:
                    s["ungated"] += 1
                    continue
                s["tested"] += 1
                rej_last = d["last_accepted"] == "0"
                s["rejected_last"] += rej_last
                s["rejected_ever"] += int(d["rejected"]) > 0
                erows.append({"run": name, "scene": scene, **{x: e[x] for x in ("agent_a", "kf_a", "agent_b", "kf_b",
                              "pose_err_deg")}, "tested": d["tested"], "rejected": d["rejected"],
                              "last_d2": d["last_d2"], "rejected_at_last": int(rej_last)})
        with open(a.out / "true_edges.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(erows[0].keys()) if erows else ["run"])
            w.writeheader()
            w.writerows(erows)
    te = {}
    for sc, s in sorted(per_scene.items()):
        frac = s["rejected_last"] / s["tested"] if s["tested"] else None
        te[f"scene{sc}"] = {**s, "rejected_fraction": None if frac is None else round(frac, 4),
                            "verdict": "PASS" if frac is not None and frac <= TRUE_REJ_LIMIT else "FAIL"}

    verdict = {
        "scene0_budgeted_cells": {"cells": len(s0), "below_0.10m": s0_pass,
                                  "worst_m": max(r["ate_gated_m"] for r in s0) if s0 else None,
                                  "verdict": "PASS" if s0 and len(s0) - s0_pass <= 1 else "FAIL",
                                  "note": "pre-registered as >= 5 of 6 cells; the sweep has the cells listed in "
                                          "cells.csv, scored as at most one failing cell"},
        "scene1_not_worse": {"cells": len(s1), "worse": [r["cell"] for r in s1_bad],
                             "verdict": "PASS" if s1 and not s1_bad else "FAIL"},
        "no_corrupt": {"corrupt_runs": corrupt, "verdict": "FAIL" if corrupt else "PASS"},
        "true_edges_rejected": ({"verdict": "UNSCORED", "reason": "logged rerun is not the same experiment",
                                 "mismatches": mismatch} if mismatch else
                                {"per_scene": te, "verdict": "PASS" if te and all(v["verdict"] == "PASS"
                                                                                for v in te.values()) else "FAIL"}),
        "inputs": {"gated": str(a.gated), "base": str(a.base), "logged": str(a.logged)},
    }
    parts = [verdict[k]["verdict"] for k in ("scene0_budgeted_cells", "scene1_not_worse", "no_corrupt",
                                              "true_edges_rejected")]
    verdict["D178"] = "FAIL" if "FAIL" in parts else ("INCOMPLETE" if "UNSCORED" in parts else "PASS")
    (a.out / "verdict.yaml").write_text(yaml.safe_dump(verdict, sort_keys=False))
    print(yaml.safe_dump(verdict, sort_keys=False))


if __name__ == "__main__":
    main()
