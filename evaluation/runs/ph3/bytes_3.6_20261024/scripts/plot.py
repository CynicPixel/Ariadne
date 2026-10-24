#!/usr/bin/env python3
"""Phase 3.6 figure from processed/ only (§36; arch §31.9: accuracy against bytes, never bytes alone). Run in
ariadne-analysis:1 (matplotlib):

  docker run --rm -u $(id -u):$(id -g) -v $PWD:/w -w /w ariadne-analysis:1 python -I scripts/plot.py --in processed --out figures

  fig_bytes.png   per setting at K 1024: wire kB per keyframe (log x) against the verdict differences over the 120
                  pairs, vs today's int8 (D233's codec flips, both directions) and vs the fp32 control (post-hoc)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    wire = {(r["desc"], r["geo"]): float(r["wire_kB"]) for r in csv.DictReader(open(a.inp / "bytes_summary.csv"))}
    diff = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(a.inp / "posthoc_vs_fp32.csv")):
        diff[r["reference"]][(r["desc"], r["geo"])] += int(r["verdict_differences"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ref, mk, col in (("int8_R", "o", "#1f77b4"), ("fp32", "s", "#d62728")):
        pts = sorted((wire[s], n, s) for s, n in diff[ref].items() if s in wire)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=mk, color=col,
                   label="vs today's int8 (D233)" if ref == "int8_R" else "vs fp32 control (post-hoc)", zorder=3)
        for x, y, s in pts:
            if ref == "int8_R":
                ax.annotate(f"{s[0]}/{s[1]}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axvline(wire[("int8", "xyz")], color="0.5", lw=0.8, ls="--")
    ax.text(wire[("int8", "xyz")], ax.get_ylim()[1] * 0.95, " today (int8/xyz)", fontsize=7, color="0.4", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("wire kB per keyframe (WireCodec, fragmented, zstd; K 1024)")
    ax.set_ylabel("accept/reject differences over 120 pairs")
    ax.set_title("Local-feature wire codings: verdict agreement against bytes (Phase 3.6)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "fig_bytes.png", dpi=120)
    print(f"wrote {a.out / 'fig_bytes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
