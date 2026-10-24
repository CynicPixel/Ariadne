#!/usr/bin/env python3
"""The replay rows of the pipeline profile (jobs 44b, 45, 46): aws_replay/<run>/replay_<variant>/ -> processed tables.
Stdlib only; reads what perf_replay_aws.sh uploaded, measures nothing.

  python3 -I scripts/replay.py --replays aws_replay --out processed

processed/:
  replay_runs.csv       run, variant, code (snapshot the replay ran), container_wall_s (wall_s.txt: the docker run, incl.
                        start-up and teardown), runtime_wall_s + end_s + end_to_end_rtf (the runtime's own results.yaml),
                        consumed digest, PSI cpu "some" mean of avg10 (% of wall time runnable tasks waited for a core)
  replay_agents.csv     run, variant, agent: frames, keyframes, lane track_mean_ms (results.yaml: the per-agent lane, ingest
                        + the worker round trip), the worker's timing_ms (track_mean, keyframe_mean, opencv_threads), ATE
                        rmse, and whether estimate_body.tum is byte-identical to the same run's asis variant
  replay_perf_threads.csv  (runs with perf_report_pid_dso_sym.txt) per thread: its listed symbols' share of all samples, and
                        per symbol ms per frame = share x samples / 199 Hz / frames per agent
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path


def put(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def yget(text: str, key: str) -> str:
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^#\n]+)", text, re.M)
    return m.group(1).strip().strip("'\"") if m else ""


def flow(text: str) -> dict[str, str]:   # "{a: 1, b: x}" -> dict (results.yaml's one-line maps)
    return {k.strip(): v.strip() for k, v in (kv.split(":", 1) for kv in text.strip("{} \n").split(",") if ":" in kv)}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs, agents, perf = [], [], []
    for run in sorted(p for p in a.replays.iterdir() if p.is_dir()):
        man = (run / "run_manifest.yaml").read_text() if (run / "run_manifest.yaml").exists() else ""
        code = yget(man, "code_snapshot") or f"{yget(man, 'replayed_run')} (its own snapshot)"
        asis = {}
        for v in sorted(p for p in run.glob("replay_*") if p.is_dir()):
            var = v.name.removeprefix("replay_")
            res = (v / "results.yaml").read_text() if (v / "results.yaml").exists() else ""
            psi = []
            if (v / "sampler" / "psi_cpu.txt").exists():
                for line in open(v / "sampler" / "psi_cpu.txt"):
                    m = re.search(r"some avg10=([0-9.]+)", line)
                    if m:
                        psi.append(float(m.group(1)))
            cw = (v / "wall_s.txt").read_text().split()[0] if (v / "wall_s.txt").exists() else ""
            runs.append([run.name, var, code, cw, yget(res, "wall_s"), yget(res, "end_s"), yget(res, "end_to_end_rtf"),
                         yget(res, "digest_fnv1a64") or flow(yget(res, "consumed")).get("digest_fnv1a64", "").strip("'"),
                         round(sum(psi) / len(psi), 2) if psi else ""])
            frames = {}
            for line in res.splitlines():
                if line.strip().startswith("- {agent:"):
                    f = flow(line.strip()[2:])
                    k = int(f["agent"])
                    frames[k] = int(f.get("frames", 0))
                    ad = v / f"agent{k}"
                    ar = (ad / "results.yaml").read_text() if (ad / "results.yaml").exists() else ""
                    tm = flow(yget(ar, "timing_ms")) if yget(ar, "timing_ms") else {}
                    h = sha(ad / "estimate_body.tum")
                    if var == "asis":
                        asis[k] = h
                    agents.append([run.name, var, k, f.get("frames"), f.get("keyframes"), f.get("track_mean_ms"),
                                   tm.get("track_mean"), tm.get("keyframe_mean"), tm.get("opencv_threads"),
                                   flow(yget(ar, "ate_se3")).get("rmse_m", "") if yget(ar, "ate_se3") else "",
                                   "" if var == "asis" or k not in asis else int(h == asis[k])])
            rep = v / "sampler" / "perf_report_pid_dso_sym.txt"
            log = v / "sampler" / "perf_record.log"
            if rep.exists() and log.exists():
                m = re.search(r"\((\d+) samples\)", log.read_text())
                n = int(m.group(1)) if m else 0
                fpa = max(frames.values()) if frames else 0
                for line in open(rep):
                    m = re.match(r"\s+([\d.]+)%\s+(\d+):(\S+)\s+(\S+)\s+\[.\]\s+(.*)", line)
                    if m and n and fpa:
                        p = float(m.group(1))
                        perf.append([run.name, var, m.group(2), m.group(3), m.group(4), m.group(5)[:90], p,
                                     round(p / 100 * n / 199 / fpa * 1000, 2)])
    put(a.out / "replay_runs.csv", ["run", "variant", "code", "container_wall_s", "runtime_wall_s", "end_s", "end_to_end_rtf", "consumed_digest",
                                    "psi_cpu_some_avg10_mean_pct"], runs)
    put(a.out / "replay_agents.csv", ["run", "variant", "agent", "frames", "keyframes", "lane_track_mean_ms",
                                      "worker_track_mean_ms", "worker_keyframe_mean_ms", "opencv_threads", "ate_rmse_m",
                                      "trajectory_identical_to_asis"], agents)
    put(a.out / "replay_perf_threads.csv", ["run", "variant", "tid", "comm", "dso", "symbol", "share_pct",
                                            "ms_per_frame"], perf)
    print(f"replay: {len(runs)} run-variants, {len(agents)} agent rows, {len(perf)} perf rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
