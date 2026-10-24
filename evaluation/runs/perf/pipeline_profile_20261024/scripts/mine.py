#!/usr/bin/env python3
"""Step 1 of the pipeline profile (M's delegation, 2026-10-24): mine the timers that recent AWS runs already wrote.

Reads only; runs nothing. Every source file read is listed with its sha256 in mined/sources.csv, so a re-run on the same
inputs reproduces mined/*.csv byte for byte.

  python3 scripts/mine.py --out mined  [--root /data/code]

Sources (run ids M named; paths are where each owner's session downloaded them):
  collab sweeps   3.2 gate 0345a5 (pass A + B1/B2/C) and eee456 (B1/B2/C on 0345a5's stores), infra/eval/collab_aws.sh
  premerge        48e0a5 and 083c02 (584415e = main a66dac6's merged B work), infra/eval/premerge_aws.sh
  live            B's live_aws.sh cpux86 runs: ma200 70a514 (200 s, 3 agents) and collab60 8447a1 (60 s, collab on);
                  their run.tar.gz is mined separately (mine_live.py) once downloaded

Outputs (mined/; not raw/: .gitignore drops evaluation/runs/**/raw/):
  stage_walls.csv   run, source, stage, start_utc, end_utc, seconds        (steps.txt / gate.log stage boundaries)
  passa.csv         run, scene, seed, agent, keyframes, ms_aliked_mean, ms_place_mean, place_model, aliked_model,
                    opencv_threads, concurrent_procs
  collab_b.csv      run, job, arm, duration_s, verify_calls, verify_geometric, verify_gated, verify_ms_total,
                    solve_calls, solve_ms_total, candidates, edges_verified, opencv_threads, concurrent_procs
  vo_timing.csv     run, job, sequence, frames, keyframes, track_mean_ms, vo_track_mean_ms, keyframe_mean_ms,
                    opencv_threads
  live_walls.csv    run, end_s, runtime_start_utc, runtime_exit_utc, wall_s, wall_per_sim_s
  instances.csv     run, instance, market, az, launched_utc, seconds, usd_per_h, usd, exit   (ledger_row.csv)
  sources.csv       path, sha256
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

SWEEP = {  # run id -> (results dir relative to --root, opencv threads per pass-A proc, pass-A procs, pass-B procs)
    "20261023T163202Z-cpux86-0345a5": "Ariadne-3x/build/runs/20261023T163202Z-cpux86-0345a5/output/results",
    "20261023T191248Z-cpux86-eee456": "Ariadne-3x/build/runs/20261023T191248Z-cpux86-eee456/output/results",
}
# collab_aws.sh at 0620218 (both sweeps): P_A=4 containers at OPENCV_FOR_THREADS_NUM = nproc / P_A, P_B=8 at 2.
P_A, P_B, THREADS_B = 4, 8, 2
PREMERGE = {
    "20261023T202525Z-cpux86-48e0a5": "Ariadne-pm5/build/runs/20261023T202525Z-cpux86-48e0a5/output/results",
    "20261023T212435Z-cpux86-083c02": "Ariadne/evaluation/runs/premerge_aws/20261023T212435Z-cpux86-083c02",
}
LIVE = {  # run id -> (results dir, simulated seconds)
    "20261023T200447Z-cpux86-70a514": ("Ariadne-llm/build/runs/20261023T200447Z-cpux86-70a514/output/results", 200),
    "20261023T204820Z-cpux86-8447a1": ("Ariadne-llm/build/runs/20261023T204820Z-cpux86-8447a1/output/results", 60),
}
LEDGERS = ["Ariadne/evaluation/runs", "Ariadne-3x/evaluation/runs", "Ariadne-llm/evaluation/runs",
           "Ariadne-pm5/evaluation/runs", "Ariadne-2.2-radiometric/evaluation/runs"]

read_files: dict[str, str] = {}


def read(p: Path) -> str:
    b = p.read_bytes()
    read_files[str(p)] = hashlib.sha256(b).hexdigest()
    return b.decode(errors="replace")


def utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def flow(text: str, key: str) -> dict[str, str]:
    """The top-level `key: {a: 1, b: 2, c: {...}}` flow mapping of a results.yaml, flat (nested values kept raw).
    The LAST one wins: collab_sim's results.yaml has `verify:` twice (the config echo, then the counters)."""
    ms = re.findall(rf"^{key}: \{{(.*)\}}\s*$", text, re.M)
    if not ms:
        return {}
    out, depth, cur = {}, 0, ""
    for ch in ms[-1] + ",":
        if ch == "," and depth == 0:
            k, _, v = cur.partition(":")
            out[k.strip()] = v.strip()
            cur = ""
            continue
        depth += ch in "{[" or -(ch in "}]")
        cur += ch
    return out


def scalar(text: str, key: str) -> str:
    m = re.search(rf"^{key}: (.*)$", text, re.M)
    return m.group(1).split("#")[0].strip() if m else ""


def steps(run: str, path: Path, rows: list) -> None:
    """steps.txt: '=== <UTC> [+Ns] <stage>' lines; each stage ends where the next begins."""
    marks = []
    for line in read(path).splitlines():
        m = re.match(r"=== (\S+Z)(?: \+\d+s)? (.*)", line)
        if m:
            marks.append((utc(m.group(1)), m.group(2).strip()))
    for (t0, name), (t1, _) in zip(marks, marks[1:]):
        rows.append([run, path.name, name, t0.isoformat(), t1.isoformat(), int((t1 - t0).total_seconds())])


def gate_log(run: str, path: Path, day: str, rows: list) -> None:
    """premerge gate.log: '[HH:MM:SS] <stage>' lines (UTC, same day as the run)."""
    marks = [(utc(f"{day}T{m.group(1)}Z"), m.group(2).strip())
             for m in re.finditer(r"^\[(\d\d:\d\d:\d\d)\] (.*)$", read(path), re.M)]
    for (t0, name), (t1, _) in zip(marks, marks[1:]):
        rows.append([run, "gate.log", name, t0.isoformat(), t1.isoformat(), int((t1 - t0).total_seconds())])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/code"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stage_rows, passa, coll, vo, live, inst = [], [], [], [], [], []

    for run, rel in SWEEP.items():
        d = a.root / rel
        steps(run, d / "steps.txt", stage_rows)
        nproc = int(read(d / "nproc.txt").strip())
        for y in sorted((d / "passa").glob("s*/seed*/passa_agent*.yaml")):
            t = read(y)
            passa.append([run, y.parts[-3][1:], y.parts[-2][4:], scalar(t, "agent"), scalar(t, "keyframes"),
                          scalar(t, "ms_aliked_mean"), scalar(t, "ms_place_mean"), flow(t, "place").get("model", ""),
                          flow(t, "aliked").get("model", ""), nproc // P_A, P_A])
        for r in sorted((d / "runs").glob("*/results.yaml")):
            t = read(r)
            v, s, p = flow(t, "verify"), flow(t, "solve"), flow(t, "protocol")
            coll.append([run, r.parent.name, scalar(t, "arm"), scalar(t, "duration_s"), v.get("calls"),
                         v.get("geometric"), v.get("gated"), v.get("ms_total"), s.get("calls"), s.get("ms_total"),
                         p.get("candidates"), p.get("edges_verified"), THREADS_B, P_B])

    for run, rel in PREMERGE.items():
        d = a.root / rel
        steps(run, d / "steps.txt", stage_rows)
        day = re.search(r"=== (\d{4}-\d\d-\d\d)T", read(d / "steps.txt")).group(1)
        gate_log(run, d / "gate" / "gate.log", day, stage_rows)
        for r in sorted((d / "gate").glob("**/results.yaml")):
            t = read(r)
            tm = flow(t, "timing_ms")
            if not tm:
                continue
            vk = flow(t, "vo")
            vo.append([run, str(r.parent.relative_to(d / "gate")), scalar(t, "sequence"), scalar(t, "frames"),
                       vk.get("keyframes", ""), tm.get("track_mean"), tm.get("vo_track_mean"),
                       tm.get("keyframe_mean"), tm.get("opencv_threads")])

    for run, (rel, end_s) in LIVE.items():
        t = read(a.root / rel / "live_run.log")
        day = run[:4] + "-" + run[4:6] + "-" + run[6:8]
        t0 = re.search(r"^\[live (\d\d:\d\d:\d\d)\] lockstep", t, re.M)
        t1 = re.search(r"^\[live (\d\d:\d\d:\d\d)\] runtime exit", t, re.M)
        if t0 and t1:
            w = (utc(f"{day}T{t1.group(1)}Z") - utc(f"{day}T{t0.group(1)}Z")).total_seconds()
            live.append([run, end_s, f"{day}T{t0.group(1)}Z", f"{day}T{t1.group(1)}Z", int(w), round(w / end_s, 2)])

    seen = set()
    ledgers = [f for base in LEDGERS for f in sorted((a.root / base).glob("**/ledger_row.csv"))]
    ledgers += sorted(a.root.glob("Ariadne*/build/runs/ledger.csv"))   # spinup.sh's per-worktree ledger
    for f in ledgers:
        if True:
            for row in csv.reader(read(f).splitlines()):
                if row and row[0] not in seen:
                    seen.add(row[0])
                    inst.append(row)
    for rid in list(SWEEP) + list(PREMERGE) + list(LIVE):
        if rid not in seen:
            out = None
            for rel in (SWEEP.get(rid), PREMERGE.get(rid), (LIVE.get(rid) or (None,))[0]):
                if rel and (a.root / rel / "ledger_row.csv").exists():
                    out = list(csv.reader(read(a.root / rel / "ledger_row.csv").splitlines()))[0]
            inst.append(out or [rid, "", "", "", "", "", "", "", "no ledger_row.csv"])

    def put(name, header, rows):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(rows)

    put("stage_walls.csv", ["run", "source", "stage", "start_utc", "end_utc", "seconds"], stage_rows)
    put("passa.csv", ["run", "scene", "seed", "agent", "keyframes", "ms_aliked_mean", "ms_place_mean", "place_model",
                      "aliked_model", "opencv_threads", "concurrent_procs"], passa)
    put("collab_b.csv", ["run", "job", "arm", "duration_s", "verify_calls", "verify_geometric", "verify_gated",
                         "verify_ms_total", "solve_calls", "solve_ms_total", "candidates", "edges_verified",
                         "opencv_threads", "concurrent_procs"], coll)
    put("vo_timing.csv", ["run", "job", "sequence", "frames", "keyframes", "track_mean_ms", "vo_track_mean_ms",
                          "keyframe_mean_ms", "opencv_threads"], vo)
    put("live_walls.csv", ["run", "end_s", "runtime_start_utc", "runtime_exit_utc", "wall_s", "wall_per_sim_s"], live)
    put("instances.csv", ["run", "instance", "market", "az", "launched_utc", "seconds", "usd_per_h", "usd", "exit"],
        [r for r in inst if r[0] in set(SWEEP) | set(PREMERGE) | set(LIVE)])
    put("sources.csv", ["path", "sha256"], sorted(read_files.items()))
    print(f"stage_walls {len(stage_rows)}  passa {len(passa)}  collab_b {len(coll)}  vo {len(vo)}  live {len(live)}  "
          f"sources {len(read_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
