"""4.3 + 4.6 gate figures from gate_rows.jsonl (the 36 lockstep runs; NBV_4.3 §14).

  coverage_time.png    coverage vs sim time per scene: per-policy mean over seeds 0-2, min-max band
  coverage_metre.png   coverage vs metres flown (the secondary view), per run, mean per policy

Run (analysis image; matplotlib is not on the host):
  docker run --rm -v "$PWD":/w -w /w ariadne-analysis:1 python3 -I evaluation/runs/nbv_4.3_gate/plot_gate.py \
      evaluation/runs/nbv_4.3_gate
Colours: the dataviz reference categorical slots 1-6, validated (light surface); three slots sit below 3:1 contrast,
so every series also has its own line style and marker, and the README gives the numbers as a table.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["ariadne", "tare", "coverage_greedy", "uncertainty_only", "random", "nearest_frontier_1.8"]
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
STYLE = dict(zip(ORDER, ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]))
MARK = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P"]))
LABEL = {"ariadne": "Ariadne (4.3)", "tare": "TARE-inspired (4.6)", "coverage_greedy": "coverage-greedy",
         "uncertainty_only": "uncertainty-only", "random": "random", "nearest_frontier_1.8": "nearest frontier (1.8)"}
SCENES = ["ci_tiny", "maze_short"]


def main(d: Path) -> None:
    rows = [json.loads(line) for line in open(d / "gate_rows.jsonl")]
    by = defaultdict(lambda: defaultdict(dict))   # (scene, policy) -> seed -> t -> (coverage, flown)
    for r in rows:
        by[(r["scene"], r["policy"])][r["seed"]][r["t_s"]] = (r["coverage"], r["flown_m"])
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, sc in zip(axes, SCENES):
        for pol in ORDER:
            seeds = by.get((sc, pol))
            if not seeds:
                continue
            ts = sorted(next(iter(seeds.values())).keys())
            M = np.array([[seeds[s][t][0] for t in ts] for s in sorted(seeds)])
            ax.fill_between(ts, M.min(0), M.max(0), color=COLOR[pol], alpha=0.12, linewidth=0)
            ax.plot(ts, M.mean(0), color=COLOR[pol], linestyle=STYLE[pol], marker=MARK[pol], markersize=4,
                    linewidth=2, label=LABEL[pol])
        ax.set_title(f"{sc}  (3 seeds; band = min-max)", color="#0b0b0b")
        ax.set_xlabel("sim time (s)")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
        ax.set_xlim(0, 205)
    axes[0].set_ylabel("coverage (fraction of reachable GT free space)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(d / "coverage_time.png", dpi=150, facecolor="#fcfcfb")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, sc in zip(axes, SCENES):
        for pol in ORDER:
            seeds = by.get((sc, pol))
            if not seeds:
                continue
            for k, s in enumerate(sorted(seeds)):
                pts = sorted(seeds[s].values(), key=lambda v: v[1])
                ax.plot([p[1] for p in pts], [p[0] for p in pts], color=COLOR[pol], linestyle=STYLE[pol],
                        marker=MARK[pol], markersize=3, linewidth=1.2, alpha=0.9,
                        label=LABEL[pol] if k == 0 else None)
        ax.set_title(f"{sc}  (each run; the secondary view)", color="#0b0b0b")
        ax.set_xlabel("metres flown by both agents (GT)")
        ax.grid(True, color="#e6e5e0", linewidth=0.6)
    axes[0].set_ylabel("coverage")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(d / "coverage_metre.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
