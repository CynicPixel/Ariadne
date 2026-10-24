#!/usr/bin/env python3
"""collab_gate_report: the live gate/alias verdict of one run dir, as JSON on stdout.

    python3 python/analysis/collab_gate_report.py RUN_INNER_DIR

From telemetry.jsonl alone: the D178 entry-gate counters, kept_inter_edges over time, and per-agent
shared-frame pose error vs GT (max and final) - the number that showed collab60d's aliased merge
(agent 1 jumped 15.3 m at t=45 s with the gate off). Used by the collab60e diag job and the demo film
pack (beat 6's before/after).
"""
import json
import math
import sys
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    est, gt, collab = {}, {}, []
    for line in open(run / "telemetry.jsonl"):
        d = json.loads(line)
        t = d.get("type")
        if t == "pose":
            est.setdefault(d["agent"], []).append((d["t_s"], d["xy"][0], d["xy"][1]))
        elif t == "gt_pose":
            gt.setdefault(d["agent"], []).append((d["t_s"], d["xy"][0], d["xy"][1]))
        elif t is None and "kept_inter_edges" in d:
            collab.append(d)
    for k in est:
        est[k].sort()
    for k in gt:
        gt[k].sort()

    def gt_at(k, t):
        return min(gt[k], key=lambda r: abs(r[0] - t))

    agents = {}
    for k in sorted(est):
        errs = []
        for t, x, y in est[k]:
            g = gt_at(k, t)
            if abs(g[0] - t) <= 0.5:
                errs.append((t, math.dist((x, y), g[1:])))
        mx = max(errs, key=lambda e: e[1]) if errs else (None, None)
        agents[f"agent{k}"] = {
            "max_err_m": round(mx[1], 3) if errs else None,
            "max_err_t_s": mx[0],
            "final_err_m": round(errs[-1][1], 3) if errs else None,
            "samples": len(errs),
        }
    timeline = [{"t_s": c["t_ns"] / 1e9, "kept_inter_edges": c.get("kept_inter_edges"),
                 "entry_gate": c.get("entry_gate"), "verify": c.get("verify"),
                 "anchors": [a.get("anchor") for a in c.get("agents", [])]} for c in collab]
    gate_final = collab[-1].get("entry_gate") if collab else None
    print(json.dumps({"run": str(run), "entry_gate_final": gate_final,
                      "shared_frame_vs_gt": agents, "collab_timeline": timeline}, indent=1))


if __name__ == "__main__":
    main()
