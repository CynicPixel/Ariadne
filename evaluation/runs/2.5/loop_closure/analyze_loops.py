#!/usr/bin/env python3
"""Reproduce the D73 part 2b tables (ROBUST_BACKEND_2.5.md §8) from the copied runs: funnel by stage, good-loop
NEES of the edge noise model, and the joint maximum-likelihood fit of var = k sigma_PnP^2 + floor^2.

    python3 evaluation/runs/2.5/loop_closure/analyze_loops.py

Labels: good = within 0.2 m and 2 deg of the ground-truth relative pose (the CSV 'label' column for model_* runs;
recomputed from err_t_m / err_r_deg for no_model runs, whose files predate the label fix)."""
import csv, glob, math, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CHI2_6_99 = 16.812


def rows(run_dir):
    out = []
    for r in csv.DictReader(open(os.path.join(run_dir, "loop_candidates.csv"))):
        if r["err_t_m"] in ("nan", ""):
            continue
        r["good"] = float(r["err_t_m"]) <= 0.2 and float(r["err_r_deg"]) <= 2.0
        out.append(r)
    return out


def funnel(rs):
    stages = [("verified", lambda r: True), ("gate", lambda r: r["degenerate"] == "0"),
              ("pcm", lambda r: r["pcm"] == "1"), ("gnc", lambda r: r["gnc_inlier"] == "1")]
    return " | ".join(f"{n} {sum(r['good'] for r in rs if f(r))}/{sum(1 for r in rs if f(r))}" for n, f in stages)


def fit(rs):
    E = {"t": lambda r: float(r["err_t_m"]), "r": lambda r: math.radians(float(r["err_r_deg"]))}
    S = {"t": lambda r: float(r["sigma_t_m"]), "r": lambda r: float(r["sigma_r_rad"])}
    best = {}
    for c in "tr":
        bl = None
        for k in [0.0] + [10 ** (x / 8) for x in range(-16, 33)]:
            for f in [x * (0.001 if c == "t" else 0.00025) for x in range(120)]:
                nll = 0.0
                for r in rs:
                    v = k * S[c](r) ** 2 + f * f
                    if v <= 0:
                        nll = math.inf
                        break
                    nll += 3 * math.log(v) + E[c](r) ** 2 / v
                if bl is None or nll < bl[0]:
                    bl = (nll, k, f)
        best[c] = bl[1:]
    return best


if __name__ == "__main__":
    for group in ("no_model", "model_train", "model_heldout"):
        for d in sorted(glob.glob(os.path.join(HERE, group, "*/"))):
            rs = rows(d)
            line = f"{group}/{os.path.basename(d.rstrip('/'))}: {funnel(rs)}"
            if group != "no_model":
                n = [float(r["nees6"]) for r in rs if r["good"] and r["nees6"] != "nan"]
                line += f" | good-loop NEES mean {st.mean(n):.2f}, >chi2_6(0.99) {100 * sum(x > CHI2_6_99 for x in n) / len(n):.1f}%"
            print(line)
    train = [r for d in glob.glob(os.path.join(HERE, "no_model", "*/")) for r in rows(d) if r["good"]]
    b = fit(train)
    print("joint fit on no_model MH_01 + V1_01 good loops: k_t %.3g floor_t %.4f m | k_r %.3g floor_r %.4f rad"
          % (b["t"] + b["r"]))
