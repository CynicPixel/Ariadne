#!/usr/bin/env python3
"""The knob study's pairs: real verify calls of the 3.2 gate sweep (run 0345a5, commit 0620218), not synthetic ones.

Source: the calibration arm's runs cal_s<scene>_seed1 (ideal comms, similarity arm: every candidate the broker picked
was verified, 1311-1705 calls per run), whose edges.csv logs EVERY verify call with its reason. Per scene a seeded
sample (random.Random(20261024)): N_VERIFIED verified pairs and N_PER_REJECT of each rejection reason (fewer if the run
has fewer), so the study sees both directions of a flip (verified -> rejected, rejected -> verified).

  python3 scripts/pairs.py --runs <0345a5 results>/runs --out inputs
Writes inputs/pairs_s<scene>.csv (agent_a,kf_a,agent_b,kf_b, then the production columns prefixed prod_) and
inputs/ids_s<scene>_a<agent>.txt (the comma-separated keyframe ids ariadne_perf_kfdump must dump).
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

N_VERIFIED, N_PER_REJECT, SEED = 30, 6, 20261024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    for scene in (0, 1):
        rows = list(csv.DictReader(open(a.runs / f"cal_s{scene}_seed1" / "edges.csv")))
        by = {}
        for r in rows:
            by.setdefault(r["reason"], []).append(r)
        pick = []
        for reason in sorted(by):
            n = N_VERIFIED if reason == "verified" else N_PER_REJECT
            pool = by[reason]
            pick += rng.sample(pool, min(n, len(pool)))
        pick.sort(key=lambda r: (int(r["agent_a"]), int(r["kf_a"]), int(r["agent_b"]), int(r["kf_b"])))
        cols = ["matches", "essential_inliers", "pnp_inliers", "reason", "score", "gt_err_t_m", "gt_err_r_deg"]
        with open(a.out / f"pairs_s{scene}.csv", "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["agent_a", "kf_a", "agent_b", "kf_b"] + ["prod_" + c for c in cols])
            for r in pick:
                w.writerow([r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]] + [r[c] for c in cols])
        for agent in (0, 1):
            ids = sorted({int(r["kf_a"]) for r in pick if int(r["agent_a"]) == agent} |
                         {int(r["kf_b"]) for r in pick if int(r["agent_b"]) == agent})
            (a.out / f"ids_s{scene}_a{agent}.txt").write_text(",".join(map(str, ids)) + "\n")
        print(f"scene {scene}: {len(pick)} pairs " +
              ", ".join(f"{k} {sum(r['reason'] == k for r in pick)}/{len(v)}" for k, v in sorted(by.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
