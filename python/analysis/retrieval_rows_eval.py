"""Retrieval rows on EuRoC: 3.6 descriptor compression and the 3.1 CLAHE ablation (RETRIEVAL_3.1_3.6.md). numpy only.

  python3 -I retrieval_rows_eval.py compression ROOT --out DIR     # §17.2 rows on the D62 dumps (ep376, ep752)
  python3 -I retrieval_rows_eval.py methods ROOT --base ep376 --methods ep376_clahe --out DIR   # 3.1 rows

ROOT holds one directory per flight from ariadne_place_eval (samples.csv, <method>.f32). The protocol is
place_recall_eval.py's (PERCEPTION_2.1 §11.5): a positive is within --dist m and --angle deg, the database is the
samples older than --exclude-s, exact cosine top-k, ties broken by index. Each row is scored on the SAME queries as
its baseline; McNemar is the exact two-sided binomial test on the discordant R@1 queries. Writes DIR/rows.csv and
DIR/rows.yaml.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

KS = (1, 5)
MACHINE_HALL = "MH_"   # the 3.6 PCA environment split: fit on one environment, score on the other


def load(d: Path, methods):
    s = np.loadtxt(d / "samples.csv", delimiter=",", skiprows=1)
    t, c, ax = s[:, 1].astype(np.int64), s[:, 2:5], s[:, 5:8]
    desc = {m: np.fromfile(d / f"{m}.f32", dtype="<f4").reshape(len(s), 256) for m in methods}
    return t, c, ax, desc


def hits(t, c, ax, desc, dist, angle, excl_s):
    """Per query (with a positive): (hit@1, hit@5) as booleans."""
    cos_thr = math.cos(math.radians(angle))
    out = []
    for q in range(len(t)):
        admit = np.where(t < t[q] - int(excl_s * 1e9))[0]
        if admit.size == 0:
            continue
        pos = (np.linalg.norm(c[admit] - c[q], axis=1) < dist) & (ax[admit] @ ax[q] > cos_thr)
        if not pos.any():
            continue
        posset = set(admit[pos].tolist())
        sims = desc[admit] @ desc[q]
        order = admit[np.lexsort((admit, -sims))]
        out.append(tuple(bool(posset & set(order[:k].tolist())) for k in KS))
    return np.array(out, dtype=bool).reshape(-1, len(KS))


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def int8_roundtrip(x):
    """Symmetric per-vector scale s = max|x|/127 (the scale stored as fp16), q = round(x/s), decoded q*s."""
    s = (np.abs(x).max(axis=1, keepdims=True) / 127.0).astype(np.float16).astype(np.float32)
    q = np.clip(np.round(x / s), -127, 127)
    return unit(q * s)


def fit_pca(x):
    mu = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt   # rows of vt: principal axes, by decreasing variance


def mcnemar_p(a, b):
    """Exact two-sided McNemar on paired booleans."""
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


ROWS = [   # name, bytes per keyframe (payload [PROXY]), transform kind, PCA dims, extra?
    ("fp32", 1024, "fp32", None, False),
    ("fp16", 512, "fp16", None, False),
    ("int8", 258, "int8", None, False),
    ("pca128_fp16", 256, "fp16", 128, False),
    ("pca256_int8", 258, "int8", 256, False),
    ("pca64_fp16", 128, "fp16", 64, True),
    ("pca128_int8", 130, "int8", 128, True),
]


def encode(x, kind, pca, dims):
    if dims is not None:
        mu, vt = pca
        x = unit((x - mu) @ vt[:dims].T)
    if kind == "fp32":
        return unit(x)
    if kind == "fp16":
        return unit(x.astype(np.float16).astype(np.float32))
    return int8_roundtrip(x)


def compression(a) -> int:
    flights = sorted(p for p in a.root.iterdir() if (p / "samples.csv").exists())
    data = {f.name: load(f, ["ep376", "ep752"]) for f in flights}
    table = []
    for m in ("ep376", "ep752"):
        env = {n: n.startswith(MACHINE_HALL) for n in data}
        pca = {e: fit_pca(np.concatenate([data[n][3][m] for n in data if env[n] == e])) for e in (True, False)}
        base = None
        for name, nbytes, kind, dims, extra in ROWS:
            h = np.concatenate([hits(t, c, ax, encode(desc[m], kind, pca[not env[n]], dims), a.dist, a.angle,
                                     a.exclude_s) for n, (t, c, ax, desc) in data.items()])
            if base is None:
                base = h
            r1, r5 = h[:, 0].mean(), h[:, 1].mean()
            table.append({"method": m, "row": name, "bytes": nbytes, "extra": extra, "queries": len(h),
                          "r1": r1, "r5": r5, "d_r1": r1 - base[:, 0].mean(),
                          "mcnemar_p_r1": mcnemar_p(base[:, 0], h[:, 0]),
                          "admissible": bool(r1 >= base[:, 0].mean() - a.bar - 1e-12)})
    return write(a, table, "compression (RETRIEVAL_3.1_3.6 §2, D93); PCA fit on the other environment (MH <-> V)")


def methods(a) -> int:
    flights = sorted(p for p in a.root.iterdir() if (p / "samples.csv").exists())
    ms = [a.base] + a.methods.split(",")
    h = {m: [] for m in ms}
    for f in flights:
        t, c, ax, desc = load(f, ms)
        for m in ms:
            h[m].append(hits(t, c, ax, desc[m], a.dist, a.angle, a.exclude_s))
    h = {m: np.concatenate(v) for m, v in h.items()}
    table = [{"method": m, "row": m, "bytes": 1024, "extra": False, "queries": len(h[m]), "r1": h[m][:, 0].mean(),
              "r5": h[m][:, 1].mean(), "d_r1": h[m][:, 0].mean() - h[a.base][:, 0].mean(),
              "mcnemar_p_r1": mcnemar_p(h[a.base][:, 0], h[m][:, 0]), "admissible": None} for m in ms]
    return write(a, table, f"methods (RETRIEVAL_3.1_3.6 §3, D92), base {a.base}")


def write(a, table, what) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    keys = ["method", "row", "bytes", "extra", "queries", "r1", "r5", "d_r1", "mcnemar_p_r1", "admissible"]
    with open(a.out / "rows.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in table:
            f.write(",".join(f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    y = [f"# retrieval_rows_eval: {what}. Generated.", f"root: {a.root}",
         f"protocol: {{dist_m: {a.dist}, angle_deg: {a.angle}, exclude_s: {a.exclude_s}}}", "rows:"]
    print(f"{'method':12s} {'row':14s} {'bytes':>6s} {'q':>4s} {'R@1':>6s} {'R@5':>6s} {'dR@1':>7s} {'McN p':>7s}  adm")
    for r in table:
        print(f"{r['method']:12s} {r['row'] + ('*' if r['extra'] else ''):14s} {r['bytes']:6d} {r['queries']:4d} "
              f"{r['r1']:6.3f} {r['r5']:6.3f} {r['d_r1']:+7.3f} {r['mcnemar_p_r1']:7.3f}  {r['admissible']}")
        y.append("  - {" + ", ".join(f"{k}: {r[k]:.4f}" if isinstance(r[k], float) else f"{k}: {r[k]}" for k in keys) + "}")
    (a.out / "rows.yaml").write_text("\n".join(y) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compression", "methods"])
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="ep376")
    ap.add_argument("--methods", default="ep376_clahe")
    ap.add_argument("--dist", type=float, default=1.0)
    ap.add_argument("--angle", type=float, default=30.0)
    ap.add_argument("--exclude-s", type=float, default=30.0)
    ap.add_argument("--bar", type=float, default=0.01, help="admissible if R@1 >= fp32 R@1 - bar (fixed before the run)")
    a = ap.parse_args()
    return compression(a) if a.mode == "compression" else methods(a)


if __name__ == "__main__":
    sys.exit(main())
