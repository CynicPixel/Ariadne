"""Aliasing outlier ramp (D75): ramp_raw.csv -> ramp_summary.csv, breakpoints.json, ramp.png.

Run from the repo root (matplotlib: the ariadne-analysis:1 image):
  docker run --rm -v "$PWD":/work -w /work ariadne-analysis:1 python3 evaluation/runs/2.5/outlier_ramp/plot_ramp.py
Breakpoint (protocol, tests/unit/test_aliasing_ramp.cpp): the lowest rate whose success probability is < 0.5.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = list(csv.DictReader(open(HERE / "ramp_raw.csv")))
    cells = defaultdict(list)
    for r in rows:
        cells[(r["policy"], r["arm"], float(r["rate"]))].append(r)
    summary, curves = [], defaultdict(list)
    for (policy, arm, rate), rs in sorted(cells.items()):
        s = {
            "policy": policy, "arm": arm, "rate": rate, "seeds": len(rs),
            "success": sum(int(r["success"]) for r in rs) / len(rs),
            "precision_mean": statistics.mean(float(r["precision"]) for r in rs),
            "recall_mean": statistics.mean(float(r["recall"]) for r in rs),
            "ate_median_m": statistics.median(float(r["ate_m"]) for r in rs),
            "ate_mean_m": statistics.mean(float(r["ate_m"]) for r in rs),
            "ms_mean": statistics.mean(float(r["ms_pcm"]) + float(r["ms_gnc"]) for r in rs),
            "largest_group": max(int(r["largest_group"]) for r in rs),
            "n_true": int(rs[0]["n_true"]),
        }
        summary.append(s)
        curves[(policy, arm)].append(s)
    with open(HERE / "ramp_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    bp = {}
    for key, ss in curves.items():
        below = [s["rate"] for s in ss if s["success"] < 0.5]
        bp[f"{key[0]}/{key[1]}"] = min(below) if below else None
    (HERE / "breakpoints.json").write_text(json.dumps(bp, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    styles = {"gnc": "--", "pcm_gnc": "-"}
    colors = {"size10": "tab:blue", "count2": "tab:orange"}
    for (policy, arm), ss in sorted(curves.items()):
        x = [100 * s["rate"] for s in ss]
        lab = f"{policy}, {arm}"
        kw = dict(ls=styles[arm], color=colors[policy], marker="o", ms=3, label=lab)
        ax[0].plot(x, [s["success"] for s in ss], **kw)
        ax[1].plot(x, [s["ate_median_m"] for s in ss], **kw)
        ax[2].plot(x, [s["ms_mean"] for s in ss], **kw)
    ax[0].set_ylabel("success probability")
    ax[0].axhline(0.5, color="grey", lw=0.6)
    ax[1].set_ylabel("ATE median [m]")
    ax[1].set_yscale("log")
    ax[2].set_ylabel("PCM + GNC runtime, mean [ms]")
    for a in ax:
        a.set_xlabel("false loops [% of 60]")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Aliasing outlier ramp (D75): groups of mutually consistent false loops, 10 seeds per rate")
    fig.tight_layout()
    fig.savefig(HERE / "ramp.png", dpi=130)
    print(json.dumps(bp))


if __name__ == "__main__":
    main()
