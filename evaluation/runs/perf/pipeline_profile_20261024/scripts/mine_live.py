#!/usr/bin/env python3
"""The live runs' own timers (pipeline profile step 1, live half). Reads the small files extracted read-only from B's
live_aws.sh run.tar.gz (tar -xzf ... --wildcards for timing/lockstep/profile/gz_stats/results/collab_events), never the
inputs.mcap. One --run per live run: ID=LABEL=DIR, LABEL saying which build it measured (e.g. pre-D179-lockstep).

  python3 -I scripts/mine_live.py --run 70a514-live=ma200-pre-D179-lockstep-live=<extract>/8410a254a94cc16b-live-20261023T200701Z \
      --run 70a514-replay1=ma200-pre-D179-replay-from-mcap=<extract>/replay1 --run 70a514-replay2=...=<extract>/replay2 --out mined
  (<extract> = build/live_extract/<run id>/build/live/runs/live-20261023T200640Z-62e0, from tar -xzf of B's run.tar.gz)

mined/:
  live_runs.csv      run, label, sim_end_s, wall_s, end_to_end_rtf, gz_rtf_median (gz_stats realTimeFactor, steady part),
                     barriers, paused_total_s (sum of lockstep wall_paused_s), paused_share (of wall_s)
  live_frames.csv    run, label, agent, frames, track_p50/p95/max_ms, wait_p50/mean_ms (timing.jsonl, per agent)
  live_barriers.csv  run, label, field, n, mean/p50/p95/max ms, share of total (profile.jsonl, D224, when present;
                     else lockstep wall_paused_s as field "paused")
  live_collab.csv    run, label, event type, count (collab_events.jsonl)
  live_sources.csv   path, sha256
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

read: dict[str, str] = {}


def lines(p: Path):
    b = p.read_bytes()
    read[str(p)] = hashlib.sha256(b).hexdigest()
    for l in b.decode(errors="replace").splitlines():
        l = l.strip()
        if l:
            try:
                yield json.loads(l)
            except json.JSONDecodeError:
                pass


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def one(d: Path, name: str) -> Path | None:
    c = sorted(d.rglob(name))
    return c[0] if c else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs, frames, bars, coll = [], [], [], []
    for spec in a.run:
        rid, label, d = spec.split("=", 2)
        d = Path(d)
        wall = rtf = end = ""
        top = [p for p in d.rglob("results.yaml") if "wall_s:" in p.read_text(errors="replace")]
        if top:
            t = top[0].read_text(errors="replace")
            read[str(top[0])] = hashlib.sha256(top[0].read_bytes()).hexdigest()
            m = re.search(r"^wall_s: ([0-9.e+-]+)", t, re.M); wall = float(m.group(1)) if m else ""
            m = re.search(r"^end_to_end_rtf: ([0-9.e+-]+)", t, re.M); rtf = float(m.group(1)) if m else ""
        gz = one(d, "gz_stats.jsonl")
        gzr = []
        if gz:
            for j in lines(gz):
                if "realTimeFactor" in j and j.get("simTime", {}).get("sec"):
                    gzr.append(float(j["realTimeFactor"]))
        ls = one(d, "lockstep.jsonl")
        paused = [float(j["wall_paused_s"]) for j in lines(ls) if "wall_paused_s" in j] if ls else []   # acks carry none
        if paused:
            end = max(j.get("T_ns", 0) for j in lines(ls) if "T_ns" in j) * 1e-9
        runs.append([rid, label, end, wall, rtf, round(st.median(gzr), 4) if gzr else "", len(paused),
                     round(sum(paused), 1) if paused else "", round(sum(paused) / wall, 3) if paused and wall else ""])
        tm = one(d, "timing.jsonl")
        if tm:
            per = defaultdict(lambda: ([], []))
            for j in lines(tm):
                per[j["agent"]][0].append(1000 * j["track_s"])
                per[j["agent"]][1].append(1000 * j["wait_s"])
            for ag, (tr, wt) in sorted(per.items()):
                frames.append([rid, label, ag, len(tr), round(q(tr, .5), 1), round(q(tr, .95), 1), round(max(tr), 1),
                               round(q(wt, .5), 1), round(st.mean(wt), 1)])
        pr = one(d, "profile.jsonl")
        series = defaultdict(list)
        if pr:
            for j in lines(pr):
                if j.get("type") == "barrier":
                    for k, v in j.get("ms", {}).items():
                        series[k].append(float(v))
        elif paused:
            series["paused"] = [1000 * p for p in paused]
        tot = sum(series.get("total", [])) or sum(sum(v) for v in series.values())
        for k, v in series.items():
            bars.append([rid, label, k, len(v), round(st.mean(v), 2), round(q(v, .5), 2), round(q(v, .95), 2),
                         round(max(v), 2), round(sum(v) / tot, 4) if tot else ""])
        ce = one(d, "collab_events.jsonl")
        if ce:
            for k, n in sorted(Counter(str(j.get("type") or j.get("ev") or j.get("kind")) for j in lines(ce)).items()):
                coll.append([rid, label, k, n])

    def put(n, h, r):
        with open(a.out / n, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(h)
            w.writerows(r)
    a.out.mkdir(parents=True, exist_ok=True)
    put("live_runs.csv", ["run", "label", "sim_end_s", "wall_s", "end_to_end_rtf", "gz_rtf_median", "barriers",
                          "paused_total_s", "paused_share"], runs)
    put("live_frames.csv", ["run", "label", "agent", "frames", "track_p50_ms", "track_p95_ms", "track_max_ms",
                            "wait_p50_ms", "wait_mean_ms"], frames)
    put("live_barriers.csv", ["run", "label", "field", "n", "mean_ms", "p50_ms", "p95_ms", "max_ms", "share"], bars)
    put("live_collab.csv", ["run", "label", "event", "count"], coll)
    put("live_sources.csv", ["path", "sha256"], sorted(read.items()))
    print(f"live runs {len(runs)}  frame rows {len(frames)}  barrier fields {len(bars)}  collab {len(coll)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
