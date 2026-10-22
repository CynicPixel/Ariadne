"""4.2 risk-proxy calibration against GT risk zones (design RISK_ZONES_4.2 §3, D112). numpy only.

  python3 -I risk_calibration.py --seed-dir 0=MAP0:SCENE0 --seed-dir 1=MAP1:SCENE1 --seed-dir 2=MAP2:SCENE2 \\
      [--control MAPC:SCENEC] --out DIR

MAPk is an ariadne_volumetric_eval output (risk_proxy.png, risk_overhang.png, risk_unsupported.png as uint16 x
65535, slab.png); SCENEk the scene directory (scene_gt.npz with risk_zone). Fixed before the data existed:
  population  GT trav_responder cells that the map observed free (slab == 1): cells a route can use and the map saw
  label       risk_zone; score: the proxy (and each term alone, the ablation)
  threshold   at recall 0.9 on seed 0 ONLY; reported on seeds 1-2: AUC, recall and FPR at that threshold
  control     a hazard-free map (the evidence spool): FPR at the seed-0 threshold
  bar         evaluation-seed AUC >= 0.75 for the proxy to enter planning; a term with seed-0 AUC <= 0.6 is dropped
AUC is the Mann-Whitney statistic (ties count one half). Writes DIR/calibration.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volumetric_eval import read_png_u8  # noqa: E402

TERMS = {"proxy": "risk_proxy.png", "overhang": "risk_overhang.png", "unsupported": "risk_unsupported.png"}
RECALL = 0.9
AUC_BAR, TERM_BAR = 0.75, 0.6


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label], score[~label]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    while i < sv.size:   # average ranks over ties
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load(map_dir: Path, scene_dir: Path):
    gt = np.load(scene_dir / "scene_gt.npz")
    slab = read_png_u8(map_dir / "slab.png")
    pop = gt["trav_responder"].astype(bool) & (slab == 1)
    label = gt["risk_zone"].astype(bool) if "risk_zone" in gt.files else np.zeros_like(pop)
    scores = {k: read_png_u8(map_dir / f).astype(np.float64) / 65535.0 for k, f in TERMS.items()}
    return pop, label, scores


def threshold_at_recall(score: np.ndarray, label: np.ndarray, recall: float) -> float:
    pos = np.sort(score[label])
    if pos.size == 0:
        return float("nan")
    return float(pos[int(np.floor((1 - recall) * pos.size))])   # >= this keeps >= recall of the positives


def at(score, label, thr):
    pred = score >= thr
    return {"recall": float(pred[label].mean()) if label.any() else float("nan"),
            "fpr": float(pred[~label].mean()) if (~label).any() else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", action="append", required=True, help="SEED=MAPDIR:SCENEDIR")
    ap.add_argument("--control", help="MAPDIR:SCENEDIR of a hazard-free map")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = {}
    for item in a.seed_dir:
        seed, rest = item.split("=", 1)
        m, sc = rest.split(":", 1)
        data[int(seed)] = load(Path(m), Path(sc))
    if 0 not in data:
        raise SystemExit("seed 0 (the threshold seed) is required")
    out = {"protocol": {"population": "GT trav_responder & map slab free", "label": "risk_zone",
                        "threshold": f"recall {RECALL} on seed 0", "auc_bar": AUC_BAR, "term_bar": TERM_BAR}, "terms": {}}
    lines = []
    for term in TERMS:
        pop, lab, sc = data[0]
        s0, l0 = sc[term][pop], lab[pop]
        thr = threshold_at_recall(s0, l0, RECALL)
        row = {"seed0": {"cells": int(pop.sum()), "positives": int(l0.sum()), "auc": auc(s0, l0), "threshold": thr,
                         **at(s0, l0, thr)}}
        evals = []
        for seed in sorted(k for k in data if k != 0):
            p, l, s = data[seed]
            r = {"cells": int(p.sum()), "positives": int(l[p].sum()), "auc": auc(s[term][p], l[p]), **at(s[term][p], l[p], thr)}
            row[f"seed{seed}"] = r
            evals.append(r)
        if evals:
            pooled_s = np.concatenate([data[k][2][term][data[k][0]] for k in data if k != 0])
            pooled_l = np.concatenate([data[k][1][data[k][0]] for k in data if k != 0])
            row["eval_pooled"] = {"auc": auc(pooled_s, pooled_l), **at(pooled_s, pooled_l, thr)}
        if a.control:
            m, sc_dir = a.control.split(":", 1)
            pc, lc, scc = load(Path(m), Path(sc_dir))
            row["control_fpr"] = float((scc[term][pc] >= thr).mean())
        row["dropped"] = bool(term != "proxy" and not (row["seed0"]["auc"] > TERM_BAR))
        if term == "proxy":
            ev = row.get("eval_pooled", {}).get("auc", float("nan"))
            row["enters_planning"] = bool(ev >= AUC_BAR)
        out["terms"][term] = row
        lines.append(f"{term:12s} seed0 AUC {row['seed0']['auc']:.3f} thr {thr:.4f} (recall {row['seed0']['recall']:.2f}, "
                     f"FPR {row['seed0']['fpr']:.3f}) | eval pooled AUC {row.get('eval_pooled', {}).get('auc', float('nan')):.3f} "
                     f"recall {row.get('eval_pooled', {}).get('recall', float('nan')):.2f} FPR "
                     f"{row.get('eval_pooled', {}).get('fpr', float('nan')):.3f} | control FPR {row.get('control_fpr', float('nan')):.3f}")
    a.out.mkdir(parents=True, exist_ok=True)
    import json
    (a.out / "calibration.yaml").write_text("# risk_calibration (RISK_ZONES_4.2 §3, D112). Generated.\n" +
                                            json.dumps(out, indent=1) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
