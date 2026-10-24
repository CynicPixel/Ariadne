#!/usr/bin/env python3
"""The pipeline profile's figures, from processed/ only (no hand edits; §36). Run in ariadne-analysis:1 (matplotlib):

  docker run --rm -u $(id -u):$(id -g) -v $PWD:/w -w /w ariadne-analysis:1 python -I scripts/plot.py --in processed --out figures

  fig_threads.png    single-process p50 vs OpenCV threads, per stage, x86 vs Arm
  fig_box_n3.png     the N=3 shape: box throughput (3 concurrent processes) vs the best single process, per stage
  fig_knobs.png      per (ALIKED size, K): median match+verify ms vs verified pairs lost / gained vs 1280x720:1024
  fig_lg_layers.png  LightGlue forward split by transformer layer (K 1024 / 512)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = [("aliked_extract", "1280x720", "cv_dnn_new"), ("aliked_extract", "640x360", "cv_dnn_new"),
          ("lightglue_match", "K1024", "cv_dnn_new"), ("lightglue_match", "K256", "cv_dnn_new"),
          ("eigenplaces_describe", "640x360", "cv_dnn_classic")]


def rows(p: Path):
    return list(csv.DictReader(open(p))) if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = rows(a.inp / "dnn_matrix.csv")

    fig, axs = plt.subplots(1, len(STAGES), figsize=(4 * len(STAGES), 3.4), sharey=False)
    for ax, (stage, var, eng) in zip(axs, STAGES):
        for arch, style in (("x86_64", "o-"), ("aarch64", "s--")):
            pts = sorted((int(r["threads"]), float(r["p50_ms"])) for r in m if r["shape"] == "single" and
                         r["arch"] == arch and r["stage"] == stage and r["variant"] == var and r["engine"] == eng)
            if pts:
                ax.plot(*zip(*pts), style, label={"x86_64": "c7i (x86)", "aarch64": "c8g (Graviton4)"}[arch])
        ax.set_xscale("log", base=2)
        ax.set_title(f"{stage}\n{var} {eng}", fontsize=9)
        ax.set_xlabel("OpenCV threads (one process)")
        ax.set_ylabel("p50 ms")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "fig_threads.png", dpi=120)

    c3 = rows(a.inp / "dnn_c3.csv")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    labels, box, best = [], [], []
    for r in c3:
        if (r["stage"], r["variant"], r["engine"]) in STAGES:
            labels.append(f"{r['arch'][:3]} {r['stage'].split('_')[0]} {r['variant']}\n3x{r['threads_per_proc']}t")
            box.append(float(r["box_calls_per_s"]))
            best.append(float(r["best_single_calls_per_s"] or 0))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], box, 0.4, label="3 concurrent processes (box)")
    ax.bar([i + 0.2 for i in x], best, 0.4, label="best single process")
    ax.set_xticks(list(x), labels, fontsize=6, rotation=90)
    ax.set_ylabel("calls / s on the 16-vCPU box")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "fig_box_n3.png", dpi=120)

    ks = [r for r in rows(a.inp / "knobs_summary.csv") if r["setting"] != "store"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in ks:
        ms = float(r["median_ms_match_verify"])
        ax.scatter(ms, -int(r["lost_vs_ref1280k1024"]), color="C3")
        ax.scatter(ms, int(r["gained_vs_ref1280k1024"]), color="C2")
        ax.annotate(r["setting"], (ms, -int(r["lost_vs_ref1280k1024"])), fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("median LightGlue + verify ms per pair")
    ax.set_ylabel("verified pairs gained (+, green) / lost (-, red)\nvs 1280x720 K=1024")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out / "fig_knobs.png", dpi=120)

    lg = rows(a.inp / "lightglue_layers.csv")
    by = defaultdict(dict)
    for r in lg:
        if r["label"].startswith("single"):
            by[(r["arch"], r["threads"], r["K"])][r["part"]] = float(r["ms"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    parts = [f"layer{i}" for i in range(9)] + ["other"]
    keys = sorted(by)
    for j, k in enumerate(keys):
        bottom = 0
        for i, p in enumerate(parts):
            v = by[k].get(p, 0)
            ax.bar(j, v, bottom=bottom, color=plt.cm.viridis(i / len(parts)), label=p if j == 0 else None)
            bottom += v
    ax.set_xticks(range(len(keys)), [f"{k[0][:3]} {k[1]}t {k[2]}" for k in keys], fontsize=6, rotation=90)
    ax.set_ylabel("ms per forward (per-op profile)")
    ax.legend(fontsize=6, ncol=5)
    fig.tight_layout()
    fig.savefig(a.out / "fig_lg_layers.png", dpi=120)
    print("figures ->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
