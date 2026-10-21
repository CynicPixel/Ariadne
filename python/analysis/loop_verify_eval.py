"""Loop-candidate verification scores (design PERCEPTION_2.1 §12.4-12.5). numpy only.

  python3 -I loop_verify_eval.py ROOT                     # score rows A-E, check the §12.5 bars
  python3 -I loop_verify_eval.py ROOT --select-e 500       # write <flight>/e_pairs.csv (row E's declared subset)

ROOT holds one directory per flight written by ariadne_loop_verify (pairs.csv, verify_<row>.csv, timing.yaml).
Labels from GT: positive = centres <= 1 m AND axes <= 30 deg; negative = > 3 m OR > 60 deg; else in between.
Verification is judged by GEOMETRY (§12.4): relevant = positive with pose error <= 10 deg; false = pose error
> 10 deg or failure, whatever the label; neutral = non-positive but correct (excluded). Score = essential inliers.
Pose AUC = glue-factory's cal_error_auc (cvg/glue-factory@2d17e3b, utils/tools.py) on positive-labelled pairs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROWS = "ABCDE"
GOOD_DEG = 10.0


def cal_error_auc(errors, thresholds):
    """Transcribed from glue-factory utils/tools.py (np.trapz -> explicit trapezoid, same result)."""
    errors = np.sort(np.asarray(errors, float))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        aucs.append(round(float(np.sum((r[1:] + r[:-1]) * np.diff(e)) / 2 / t), 4))
    return aucs


def load_csv(p: Path) -> dict[str, np.ndarray]:
    a = np.genfromtxt(p, delimiter=",", names=True)
    a = np.atleast_1d(a)
    return {k: a[k] for k in a.dtype.names}


def labels(pairs) -> np.ndarray:
    pos = (pairs["dist_m"] <= 1.0) & (pairs["axis_deg"] <= 30.0)
    neg = (pairs["dist_m"] > 3.0) | (pairs["axis_deg"] > 60.0)
    return np.where(pos, 1, np.where(neg, -1, 0))


def pr_sweep(score, relevant, false):
    """Precision/recall at every distinct threshold (accept score >= s), ties grouped (sklearn's AP convention)."""
    keep = relevant | false
    s, rel = score[keep], relevant[keep]
    n_rel = int(rel.sum())
    out = []   # (threshold, tp, fp)
    for th in np.unique(s)[::-1]:
        acc = s >= th
        out.append((float(th), int((acc & rel).sum()), int((acc & ~rel).sum())))
    return out, n_rel


def ap(sweep, n_rel) -> float:
    if n_rel == 0:
        return float("nan")
    a, prev_r = 0.0, 0.0
    for _, tp, fp in sweep:
        r = tp / n_rel
        p = tp / (tp + fp) if tp + fp else 1.0
        a += (r - prev_r) * p
        prev_r = r
    return a


def recall_at_fa(sweep, n_rel, max_fp: int) -> float:
    best = 0.0
    for _, tp, fp in sweep:
        if fp <= max_fp:
            best = max(best, tp / n_rel if n_rel else float("nan"))
    return best


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("root", type=Path)
    ap_.add_argument("--select-e", type=int, default=0, help="write e_pairs.csv: positives + top-N non-positives by C")
    a = ap_.parse_args()
    flights = sorted(p for p in a.root.iterdir() if (p / "pairs.csv").exists())

    data = {}
    for f in flights:
        pairs = load_csv(f / "pairs.csv")
        d = {"label": labels(pairs), "pair": pairs["pair"].astype(int)}
        for r in ROWS:
            p = f / f"verify_{r}.csv"
            if p.exists():
                d[r] = load_csv(p)
        data[f.name] = d

    if a.select_e:
        cand = []   # (C inliers, flight, pair) of non-positives
        for name, d in data.items():
            c = d["C"]
            lab = d["label"][c["pair"].astype(int)]
            for pid, inl, l in zip(c["pair"].astype(int), c["inliers"], lab):
                if l != 1:
                    cand.append((float(inl), name, int(pid)))
        cand.sort(key=lambda x: (-x[0], x[1], x[2]))
        hard = {(n, p) for _, n, p in cand[: a.select_e]}
        total = 0
        for name, d in data.items():
            ids = sorted(set(d["pair"][d["label"] == 1].tolist()) | {p for n, p in hard if n == name})
            (a.root / name / "e_pairs.csv").write_text("pair\n" + "".join(f"{i}\n" for i in ids))
            total += len(ids)
        print(f"row E subset: {total} pairs (all positives + the {a.select_e} highest-C non-positives)")
        return 0

    def pooled(row, subset=None):
        sc, rel, fal, lab, err, err_r, fl = [], [], [], [], [], [], []
        for name, d in data.items():
            if row not in d:
                continue
            v = d[row]
            pid = v["pair"].astype(int)
            if subset is not None:
                k = np.isin(pid, subset[name])
                v = {key: val[k] for key, val in v.items()}
                pid = pid[k]
            l = d["label"][pid]
            e = np.where(v["ok"] > 0, v["err"], np.inf)
            sc.append(v["inliers"]); lab.append(l); err.append(e)
            err_r.append(np.where(v["ok"] > 0, v["err_r"], np.inf))
            rel.append((l == 1) & (e <= GOOD_DEG)); fal.append(e > GOOD_DEG); fl.append(np.full(len(pid), name))
        cat = lambda x: np.concatenate(x) if x else np.array([])
        return {"score": cat(sc), "rel": cat(rel).astype(bool), "false": cat(fal).astype(bool), "label": cat(lab),
                "err": cat(err), "err_r": cat(err_r), "flight": cat(fl)}

    def metrics(row, subset=None):
        p = pooled(row, subset)
        if len(p["score"]) == 0:
            return None
        sw, n_rel = pr_sweep(p["score"], p["rel"], p["false"])
        lab_sw, lab_n = pr_sweep(p["score"], p["label"] == 1, p["label"] == -1)
        pos = p["label"] == 1
        n_pos = int(pos.sum())   # FIXED denominator: every positive-labelled pair, the same for every row
        m = {"pairs": int(len(p["score"])), "relevant": n_rel, "false": int(p["false"].sum()), "positives": n_pos,
             "ap_geom": ap(sw, n_rel), "recall_fa0": recall_at_fa(sw, n_rel, 0),
             "recall_fa2": recall_at_fa(sw, n_rel, 2),
             "ap_geom_fixed": ap(sw, n_pos), "recall_fa0_fixed": recall_at_fa(sw, n_pos, 0),
             "recall_fa2_fixed": recall_at_fa(sw, n_pos, 2), "ap_label": ap(lab_sw, lab_n),
             "auc": cal_error_auc(p["err"][pos], [5, 10, 20]), "auc_rot": cal_error_auc(p["err_r"][pos], [5, 10, 20])}
        # tau on MH (smallest inlier count with zero false accepts), applied to V
        mh = np.char.startswith(p["flight"].astype(str), "MH")
        sw_mh, _ = pr_sweep(p["score"][mh], p["rel"][mh], p["false"][mh])
        taus = [th for th, _, fp in sw_mh if fp == 0]
        tau = min(taus) if taus else float("inf")
        v = ~mh & (p["rel"] | p["false"])
        acc = v & (p["score"] >= tau)
        tp, fp = int((acc & p["rel"]).sum()), int((acc & p["false"]).sum())
        nv = int((v & p["rel"]).sum())
        m.update(tau_mh=tau, v_precision=tp / (tp + fp) if tp + fp else float("nan"), v_recall=tp / nv if nv else float("nan"),
                 v_false_accepts=fp)
        m["per_flight_auc10"] = {}
        for name in data:
            k = (p["flight"] == name) & pos
            if k.any():
                m["per_flight_auc10"][name] = cal_error_auc(p["err"][k], [10])[0]
        return m

    res = {r: metrics(r) for r in ROWS if any(r in d for d in data.values())}
    print("AP_geom / R@FA: per-row denominator (that row's correct positives); AP_fix / Rf@FA2: fixed denominator "
          "(all positive-labelled pairs)")
    print(f"{'row':4s} {'pairs':>6s} {'rel':>5s} {'false':>6s} {'AP_geom':>8s} {'AP_fix':>7s} {'Rf@FA2':>7s} {'R@FA0':>6s} {'R@FA2':>6s} "
          f"{'AP_lab':>7s} {'AUC@5/10/20':>20s} {'rotAUC@5/10/20':>20s} {'tau_MH':>7s} {'V P/R (FA)':>16s}")
    for r, m in res.items():
        if m is None:
            continue
        print(f"{r:4s} {m['pairs']:6d} {m['relevant']:5d} {m['false']:6d} {m['ap_geom']:8.3f} {m['ap_geom_fixed']:7.3f} "
              f"{m['recall_fa2_fixed']:7.3f} {m['recall_fa0']:6.3f} "
              f"{m['recall_fa2']:6.3f} {m['ap_label']:7.3f} {'/'.join(f'{x:.3f}' for x in m['auc']):>20s} "
              f"{'/'.join(f'{x:.3f}' for x in m['auc_rot']):>20s} {m['tau_mh']:7.0f} "
              f"{m['v_precision']:6.3f}/{m['v_recall']:.3f} ({m['v_false_accepts']})")

    out = ["# loop_verify_eval (PERCEPTION_2.1 §12.4-12.5). Generated.", "rows:"]
    for r, m in res.items():
        if m:
            out.append(f"  {r}: {m}")

    if "A" in res and "C" in res and res["A"] and res["C"]:
        A, C = res["A"], res["C"]
        d10 = C["auc"][1] - A["auc"][1]
        wins = sum(C["per_flight_auc10"].get(f, 0) >= A["per_flight_auc10"].get(f, 0) for f in A["per_flight_auc10"])
        nfl = len(A["per_flight_auc10"])
        b1 = d10 >= 0.05 and wins >= 8 and C["ap_geom"] >= A["ap_geom"]
        b1f = d10 >= 0.05 and wins >= 8 and C["ap_geom_fixed"] >= A["ap_geom_fixed"]
        print(f"\nBar 1 (C vs A): AUC@10 {C['auc'][1]:.3f} - {A['auc'][1]:.3f} = {d10:+.3f} (>= +0.05); "
              f"per-flight C >= A on {wins}/{nfl} (>= 8); AP_geom per-row {C['ap_geom']:.3f} vs {A['ap_geom']:.3f}, "
              f"fixed {C['ap_geom_fixed']:.3f} vs {A['ap_geom_fixed']:.3f} -> "
              f"{'PASS' if b1 else 'MISS'} (per-row) / {'PASS' if b1f else 'MISS'} (fixed)")
        out.append(f"bar1: {{auc10_delta: {d10:.4f}, flights_c_ge_a: {wins}, ap_geom_c: {C['ap_geom']:.4f}, "
                   f"ap_geom_a: {A['ap_geom']:.4f}, ap_geom_fixed_c: {C['ap_geom_fixed']:.4f}, "
                   f"ap_geom_fixed_a: {A['ap_geom_fixed']:.4f}, pass: {str(b1).lower()}, pass_fixed: {str(b1f).lower()}}}")
        if "B" in res and res["B"]:
            dcb = C["auc"][1] - res["B"]["auc"][1]
            print(f"Bar 2 (C vs B): AUC@10 delta {dcb:+.3f} -> "
                  f"{'LightGlue earns its cost' if dcb >= 0.02 else 'below 0.02: to the maintainer'}")
            out.append(f"bar2: {{auc10_delta_c_minus_b: {dcb:.4f}, earned: {str(dcb >= 0.02).lower()}}}")
    if "E" in res and res["E"]:
        sub = {}
        for f in flights:
            p = f / "e_pairs.csv"
            sub[f.name] = np.loadtxt(p, skiprows=1, dtype=int, ndmin=1) if p.exists() else np.array([], int)
        Cs = metrics("C", sub)
        E = res["E"]
        print(f"E vs C on E's subset ({E['pairs']} pairs): AUC@10 {E['auc'][1]:.3f} vs {Cs['auc'][1]:.3f}; "
              f"AP_geom {E['ap_geom']:.3f} vs {Cs['ap_geom']:.3f}; false accepts at R@FA2 {E['recall_fa2']:.3f} vs "
              f"{Cs['recall_fa2']:.3f}")
        out.append(f"e_vs_c_subset: {{e: {E}, c: {Cs}}}")
    (a.root / "loop_verify.yaml").write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
