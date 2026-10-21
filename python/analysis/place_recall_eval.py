"""Recall@k of global place descriptors on EuRoC (design PERCEPTION_2.1 §11.5). numpy only.

  python3 -I place_recall_eval.py ROOT [--dist 1.0] [--angle 30] [--exclude-s 30]

ROOT holds one directory per flight written by ariadne_place_eval (samples.csv, <method>.f32, annoy_<method>.csv).
For a query at t_q the database is the samples with t < t_q - exclude_s (loop closure against the past). A positive
is a camera centre within --dist m AND an optical axis within --angle deg (GT). Queries without a positive are not
counted. Exact top-k = cosine over every admitted sample. Random = the analytic expectation
1 - C(N-P, k) / C(N, k) per query. Writes ROOT/recall.yaml and prints the table.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

METHODS = ["ep376", "ep752", "meanbit"]
KS = (1, 5)


def load(d: Path):
    s = np.loadtxt(d / "samples.csv", delimiter=",", skiprows=1)
    t, c, ax = s[:, 1].astype(np.int64), s[:, 2:5], s[:, 5:8]
    desc = {m: np.fromfile(d / f"{m}.f32", dtype="<f4").reshape(len(s), 256) for m in METHODS}
    ann = {m: np.loadtxt(d / f"annoy_{m}.csv", delimiter=",", skiprows=1, dtype=np.int64).reshape(-1, 6)
           for m in METHODS}
    return t, c, ax, desc, ann


def random_recall(n: int, p: int, k: int) -> float:
    if n <= 0:
        return 0.0
    k = min(k, n)
    return 1.0 - math.comb(n - p, k) / math.comb(n, k)


def evaluate(d: Path, dist: float, angle: float, excl_s: float):
    t, c, ax, desc, ann = load(d)
    n = len(t)
    cos_thr = math.cos(math.radians(angle))
    hit = {m: {k: 0 for k in KS} for m in METHODS + [f"annoy_{m}" for m in METHODS]}
    rnd = {k: 0.0 for k in KS}
    agree = {m: 0 for m in METHODS}
    queries = 0
    for q in range(n):
        admit = np.where(t < t[q] - int(excl_s * 1e9))[0]
        if admit.size == 0:
            continue
        pos = (np.linalg.norm(c[admit] - c[q], axis=1) < dist) & (ax[admit] @ ax[q] > cos_thr)
        if not pos.any():
            continue
        queries += 1
        posset = set(admit[pos].tolist())
        for k in KS:
            rnd[k] += random_recall(admit.size, int(pos.sum()), k)
        for m in METHODS:
            sims = desc[m][admit] @ desc[m][q]
            order = admit[np.lexsort((admit, -sims))]       # best first, ties by index
            for k in KS:
                hit[m][k] += bool(posset & set(order[:k].tolist()))
            row = ann[m][ann[m][:, 0] == q][0, 1:]
            top = [int(x) for x in row if x >= 0]
            for k in KS:
                hit[f"annoy_{m}"][k] += bool(posset & set(top[:k]))
            agree[m] += bool(top) and top[0] == int(order[0])
    return queries, n, hit, rnd, agree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--angle", type=float, default=30.0)
    ap.add_argument("--exclude-s", type=float, default=30.0)
    a = ap.parse_args()
    flights = sorted(p for p in a.root.iterdir() if (p / "samples.csv").exists())
    rows, tot_q = {}, 0
    tot = {m: {k: 0 for k in KS} for m in METHODS + [f"annoy_{m}" for m in METHODS]}
    tot_rnd = {k: 0.0 for k in KS}
    tot_agree = {m: 0 for m in METHODS}
    for f in flights:
        q, n, hit, rnd, agree = evaluate(f, a.dist, a.angle, a.exclude_s)
        rows[f.name] = (q, n, hit, rnd, agree)
        tot_q += q
        for m in tot:
            for k in KS:
                tot[m][k] += hit[m][k]
        for k in KS:
            tot_rnd[k] += rnd[k]
        for m in METHODS:
            tot_agree[m] += agree[m]

    def r(x, q):
        return x / q if q else float("nan")

    out = ["# place_recall_eval (PERCEPTION_2.1 §11.5). Generated.",
           f"protocol: {{dist_m: {a.dist}, angle_deg: {a.angle}, exclude_s: {a.exclude_s}}}", "flights:"]
    print(f"{'flight':18s} {'q/n':>9s}  " + "  ".join(f"{m+'@'+str(k):>12s}" for m in METHODS for k in KS)
          + f"  {'random@1':>9s} {'random@5':>9s}")
    for name, (q, n, hit, rnd, agree) in list(rows.items()) + [("POOLED", (tot_q, sum(v[1] for v in rows.values()), tot, tot_rnd, tot_agree))]:
        line = f"{name:18s} {q:4d}/{n:<4d}  " + "  ".join(f"{r(hit[m][k], q):12.3f}" for m in METHODS for k in KS)
        line += f"  {r(rnd[1], q):9.3f} {r(rnd[5], q):9.3f}"
        print(line)
        out.append(f"  {name}: {{queries: {q}, samples: {n}, "
                   + ", ".join(f"{m}_r{k}: {r(hit[m][k], q):.4f}" for m in hit for k in KS)
                   + f", random_r1: {r(rnd[1], q):.4f}, random_r5: {r(rnd[5], q):.4f}, "
                   + ", ".join(f"annoy_top1_agrees_exact_{m}: {r(agree[m], q):.4f}" for m in METHODS) + "}")
    print("annoy vs exact, pooled R@1: " + ", ".join(
        f"{m} {r(tot['annoy_'+m][1], tot_q):.3f} vs {r(tot[m][1], tot_q):.3f} (top-1 agrees {r(tot_agree[m], tot_q):.3f})"
        for m in METHODS))
    (a.root / "recall.yaml").write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
