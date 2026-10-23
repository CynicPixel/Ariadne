"""The Phase 3 gate rows from a 3.2/3.4 sweep (design SIMCOLLAB_3.2 §4-§6, D171; COLLAB_3.2 §6-§7).

  python3 -I collab_gate.py SWEEP --out DIR

SWEEP is a collab_aws.sh results directory: runs/<name>/{results.yaml, outcome.yaml, ate_vs_bytes.csv, edges.csv,
ledger.csv.gz}, with names sim_s<scene>_seed<s> (the similarity arm), <arm>_s<scene>_p<pct>_seed<s> (greedy,
spectral at a budget level) and camp_<variant>_s<scene>_seed<s> (the fault campaign).

Writes DIR/:
  byte_table.csv     per gate run and class: measured first-transmission wire bytes vs predicted, error, verdict;
                     retransmit wire bytes as the overhead row (never folded into the error). A class with no
                     predictor yet is listed as such: the gate cannot pass on it.
  ate_vs_bytes.png   per scene: final merged ATE against total first-transmission bytes, one point per arm and budget
                     level (mean over seeds, seed range as bars), and the overhead-inclusive total as hollow markers
  ate_vs_bytes.csv   the plotted numbers
  edges.csv          accepted inter-agent edges and their ground-truth errors per run (precision)
  gate.yaml          the three verdicts: byte table (<= 15 % per class), outcome (no run corrupt), arm dominance
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1d1d1b", "#6b6a63", "#e4e3de"
ARM = {"similarity": "#9a9990", "greedy": "#eb6834", "spectral": "#2a78d6"}
CLASS = {0: "heartbeat", 1: "global_descriptor", 3: "match_request", 4: "match_response", 5: "pose_edge",
         6: "pose_graph", 12: "ack"}
POLICY = {3: "match_request", 4: "match_response"}   # priced by the selection policy (D170): aggregate per run
GATED = (1, 3, 4, 5, 6)                              # COLLAB_3.2 §7 classes; heartbeats and acks are overhead rows
GATE = 0.15


def parse(name: str):
    m = re.match(r"^sim_s(\d)_seed(\d+)$", name)
    if m:
        return {"arm": "similarity", "scene": int(m[1]), "pct": 100, "seed": int(m[2]), "campaign": None}
    m = re.match(r"^(greedy|spectral)_s(\d)_p(\d+)_seed(\d+)$", name)
    if m:
        return {"arm": m[1], "scene": int(m[2]), "pct": int(m[3]), "seed": int(m[4]), "campaign": None}
    m = re.match(r"^camp_(\w+?)_s(\d)_seed(\d+)$", name)
    if m:
        return {"arm": "spectral", "scene": int(m[2]), "pct": 25, "seed": int(m[3]), "campaign": m[1]}
    return None


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for d in sorted((a.sweep / "runs").iterdir()):
        meta = parse(d.name)
        if meta is None or not (d / "results.yaml").exists():
            continue
        r = yaml.safe_load((d / "results.yaml").read_text())
        meta["results"], meta["dir"] = r, d
        meta["outcome"] = yaml.safe_load((d / "outcome.yaml").read_text()) if (d / "outcome.yaml").exists() else None
        runs[d.name] = meta

    # --- byte table: budgeted gate runs (greedy, spectral) are the I6 evidence (calibrated, cross-scene) ----------
    rows, worst = [], 0.0
    missing_predictor = set()
    for name, m in runs.items():
        if m["campaign"] or m["arm"] == "similarity":
            continue
        r = m["results"]
        meas = {int(b["class"]): b for b in r.get("bytes", [])}
        pred = r.get("predicted", {})
        for k, b in sorted(meas.items()):
            if k in POLICY:
                p = pred.get(POLICY[k])
            else:
                p = b.get("predicted_first", -1)
                p = None if p is None or p < 0 else p
            err = abs(b["wire_first"] - p) / b["wire_first"] if p is not None and b["wire_first"] else None
            if p is None and k in GATED:
                missing_predictor.add(CLASS.get(k, str(k)))
            if err is not None and k in GATED:
                worst = max(worst, err)
            rows.append({"run": name, "class": CLASS.get(k, str(k)), "measured_wire_first": b["wire_first"],
                         "predicted": "" if p is None else round(p), "error": "" if err is None else round(err, 4),
                         "verdict": "" if err is None else ("PASS" if err <= GATE else "FAIL"),
                         "overhead_wire_retransmit": b["wire_retransmit"], "overhead_wire_resync": b.get("wire_resync", 0),
                         "uncalibrated_prices": pred.get("uncalibrated_prices", ""),
                         "uncalibrated_class_prices": pred.get("uncalibrated_class_prices", "")})
    with open(a.out / "byte_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run"])
        w.writeheader()
        w.writerows(rows)

    # --- ATE vs bytes ---------------------------------------------------------------------------------------------
    pts = defaultdict(list)   # (scene, arm, pct) -> [(first, all, ate)]
    for m in runs.values():
        if m["campaign"]:
            continue
        r = m["results"]
        ate = r.get("ate_merged", {}).get("rmse_m")
        tot = r.get("total_wire_bytes", {})
        if ate is None or (isinstance(ate, float) and math.isnan(ate)):
            ate = float("nan")
        pts[(m["scene"], m["arm"], m["pct"])].append((tot.get("first", 0), tot.get("all", 0), ate,
                                                      r["ate_merged"].get("merged_agents", 0)))
    plotted = []
    scenes = sorted({k[0] for k in pts})
    fig, axes = plt.subplots(1, max(1, len(scenes)), figsize=(6.4 * max(1, len(scenes)), 4.4), dpi=150,
                             facecolor=SURFACE, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        _style(ax)
        for arm in ("similarity", "greedy", "spectral"):
            series = sorted((pct, v) for (s, a_, pct), v in pts.items() if s == sc and a_ == arm)
            xs, ys, xa, lo, hi = [], [], [], [], []
            for pct, v in series:
                ok = [t for t in v if not math.isnan(t[2])]
                if not ok:
                    continue
                x, y = statistics.mean(t[0] for t in ok), statistics.mean(t[2] for t in ok)
                xs.append(x / 1e6), ys.append(y), xa.append(statistics.mean(t[1] for t in ok) / 1e6)
                lo.append(y - min(t[2] for t in ok)), hi.append(max(t[2] for t in ok) - y)
                plotted.append({"scene": sc, "arm": arm, "budget_pct": pct, "seeds": len(v), "seeds_merged": len(ok),
                                "first_tx_MB": round(x / 1e6, 4), "all_MB": round(xa[-1], 4), "ate_m": round(y, 4)})
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[lo, hi], color=ARM[arm], marker="o", ms=4, lw=1.2, capsize=2, label=arm)
            ax.scatter(xa, ys, facecolors="none", edgecolors=ARM[arm], s=18, lw=0.8)
        ax.set_title(f"scene {sc}", color=INK, fontsize=10)
        ax.set_xlabel("cumulative wire bytes (MB): filled first transmission, hollow incl. overhead", color=MUTED,
                      fontsize=8)
        ax.set_ylabel("merged ATE RMSE (m)", color=MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "ate_vs_bytes.png", facecolor=SURFACE)
    with open(a.out / "ate_vs_bytes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plotted[0].keys()) if plotted else ["scene"])
        w.writeheader()
        w.writerows(plotted)

    # --- dominance: at each budget level and scene, spectral ATE <= greedy ATE (mean over seeds) -----------------
    dom = []
    for sc in scenes:
        for pct in (10, 25, 50, 100):
            g = [t[2] for t in pts.get((sc, "greedy", pct), []) if not math.isnan(t[2])]
            s = [t[2] for t in pts.get((sc, "spectral", pct), []) if not math.isnan(t[2])]
            if g and s:
                dom.append({"scene": sc, "pct": pct, "greedy": statistics.mean(g), "spectral": statistics.mean(s),
                            "spectral_le_greedy": statistics.mean(s) <= statistics.mean(g)})

    # --- accepted-edge precision ----------------------------------------------------------------------------------
    erows = []
    for name, m in runs.items():
        p = m["dir"] / "edges.csv"
        if not p.exists():
            continue
        for e in csv.DictReader(open(p)):
            if e["reason"] == "verified":
                erows.append({"run": name, **e})
    with open(a.out / "edges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0].keys()) if erows else ["run"])
        w.writeheader()
        w.writerows(erows)
    good = [e for e in erows if e["gt_err_t_m"] not in ("", "nan") and float(e["gt_err_t_m"]) <= 0.20
            and float(e["gt_err_r_deg"]) <= 2.0]   # the D73 loop label: bad beyond 0.2 m / 2 deg

    outcomes = defaultdict(int)
    for m in runs.values():
        if m["outcome"]:
            outcomes[m["outcome"]["outcome"]] += 1
    gate = {
        "byte_table": {"worst_error": round(worst, 4), "limit": GATE,
                       "classes_without_predictor": sorted(missing_predictor),
                       "verdict": "FAIL" if worst > GATE else ("INCOMPLETE" if missing_predictor else "PASS")},
        "outcomes": {"counts": dict(outcomes), "verdict": "FAIL" if outcomes.get("corrupt") else "PASS"},
        "dominance": {"rows": dom, "spectral_le_greedy": sum(d["spectral_le_greedy"] for d in dom),
                      "of": len(dom)},
        "edges": {"accepted": len(erows), "good_within_0.2m_2deg": len(good),
                  "precision": round(len(good) / len(erows), 4) if erows else None},
        "runs": len(runs),
    }
    (a.out / "gate.yaml").write_text(yaml.safe_dump(gate, sort_keys=False))
    print(yaml.safe_dump(gate, sort_keys=False))


if __name__ == "__main__":
    main()
