#!/usr/bin/env python3
"""Step 2 of the pipeline profile: the AWS replay's raw outputs -> processed tables (stdlib only; plot.py draws them).

  python3 -I scripts/analyze.py --aws aws --out processed

--aws holds one directory per replay run (aws/<run id>/, the job's results as perf_aws.sh wrote them: run_manifest.yaml,
timing/<label>/{timing,profile,engines}.csv, ort/*.csv, knobs/s<k>/{knobs,fidelity,extract}.csv, knobs/t2_s<k>/,
knobs/adaptive_s<k>/gate.csv, mpstat/*.txt).

processed/:
  dnn_matrix.csv      every timing row: run, arch, instance, shape (single | c3), procs, threads (per process), stage,
                      variant, engine, p50/p95 ms, cpu/wall (the pool's effective parallelism), stable (run-to-run digest)
  dnn_c3.csv          the N=3 shape: per (run, arch, threads, stage, variant, engine) the median p50 over the 3 concurrent
                      processes, the SAME RUN's single-process p50 at the same threads (blank if that run had none), the
                      slowdown, and the box throughput (3 / p50, calls per second) next to the same run's best single
  profile_ops.csv     per-op-type share of one forward (SUMMARY rows)
  lightglue_layers.csv  per transformer layer (0-8) + "other" (input proj, posenc, final assignment): ms and share,
                      per (arch, threads, K) - the adaptive-depth payoff's cost side
  ort.csv             the ORT rows with their output checks
  knobs_summary.csv   per (ALIKED size, K) setting: pairs, verified, flips vs the 1280x720:1024 recomputed reference and vs
                      the production store replay, median match / inlier ratio vs the reference, median ms
  knobs_transitions.csv  reference reason -> setting reason counts
  replay_exactness.csv   the store replay at 2 threads vs production edges.csv: reason and match-count agreement
  fidelity_summary.csv   recomputed 1280x720 ALIKED vs the pass-A store, per scene
  adaptive_summary.csv   upstream LightGlue all9 vs adaptive: exit layer distribution, verdict flips, matches, active
                      keypoints per layer
  variants_summary.csv  a graph variant at the reference size (1280x720l2) vs the reference, over every pair keyframe:
                      keypoints / scores bitwise counts, the worst descriptor |diff| (D182 criterion ii)
  mpstat_summary.csv  per phase: mean and p95 busy % of all cores, mean %steal
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() else []


def manifest(d: Path) -> dict:
    out = {}
    if (d / "run_manifest.yaml").exists():
        for line in open(d / "run_manifest.yaml"):
            if ":" in line and not line.startswith("#"):
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def put(path: Path, header: list[str], data: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(data)


def f(x, nd=3):
    return "" if x is None else (round(x, nd) if isinstance(x, float) else x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    runs = sorted(d for d in a.aws.iterdir() if d.is_dir())

    matrix, ops, layers, ort, mp = [], [], [], [], []
    for d in runs:
        m = manifest(d)
        arch, inst = m.get("arch", ""), m.get("instance_type", "")
        for t in sorted((d / "timing").glob("*/timing.csv")):
            label = t.parent.name
            mm = re.match(r"(single|c3)_t(\d+)(?:_p(\d+))?$", label)
            shape, thr = (mm.group(1), int(mm.group(2))) if mm else ("?", 0)
            for r in rows(t):
                p50, cpu, mean = float(r["p50_ms"]), float(r["cpu_mean_ms"]), float(r["mean_ms"])
                matrix.append([d.name, arch, inst, shape, 3 if shape == "c3" else 1, thr, label, r["stage"],
                               r["variant"], r["engine"], f(p50), f(float(r["p95_ms"])), f(cpu / mean if mean else 0, 2),
                               int(r["digest1"] == r["digest2"]), r["out"]])
            for r in rows(t.parent / "profile.csv"):
                if r["op"] == "(empty profile)":
                    continue
                if r["model"].startswith("lightglue"):
                    continue
                ops.append([d.name, arch, label, r["model"], r["engine"], r["threads"], r["op"], r["time_ms"], r["count"]])
            lg = defaultdict(lambda: defaultdict(float))
            for r in rows(t.parent / "profile.csv"):
                if r["model"].startswith("lightglue") and r["time_ms"]:
                    ly = re.search(r"transformers\.(\d+)", r["op"])
                    lg[r["model"]][f"layer{ly.group(1)}" if ly else "other"] += float(r["time_ms"])
            for model, parts in lg.items():
                tot = sum(parts.values())
                for part, ms in sorted(parts.items()):
                    layers.append([d.name, arch, label, thr, model.replace("lightglue_", ""), part, f(ms), f(ms / tot, 4)])
        for o in sorted((d / "ort").glob("*.csv")):
            for r in rows(o):
                ort.append([d.name, arch, r["label"], r["stage"], r["variant"], r["threads"], r["p50_ms"], r["p95_ms"],
                            r["check"]])
        for t in sorted((d / "mpstat").glob("*.txt")):
            busy, steal = [], []
            for line in open(t):
                c = line.split()
                if len(c) >= 12 and c[1] == "all":
                    try:
                        busy.append(100 - float(c[-1]))
                        steal.append(float(c[-4]))
                    except ValueError:
                        pass
            if busy:
                busy.sort()
                mp.append([d.name, arch, t.stem, len(busy), f(st.mean(busy), 1), f(busy[int(0.95 * (len(busy) - 1))], 1),
                           f(st.mean(steal), 2)])

    put(a.out / "dnn_matrix.csv", ["run", "arch", "instance", "shape", "procs", "threads", "label", "stage", "variant",
                                   "engine", "p50_ms", "p95_ms", "cpu_over_wall", "stable", "out"], matrix)
    # Baselines are paired WITHIN a run only: two c7i instances of the same type differed by ~30 % single-process
    # (jobs 42 vs 70, same CPU model, OpenCV commit and no steal), so a cross-run ratio would mix instance spread in.
    single = {(r[0], r[1], r[5], r[7], r[8], r[9]): r[10] for r in matrix if r[3] == "single"}
    best = defaultdict(float)
    for (run, arch, thr, stage, var, eng), p50 in single.items():
        best[(run, arch, stage, var, eng)] = max(best[(run, arch, stage, var, eng)], 1000.0 / p50 if p50 else 0)
    c3 = defaultdict(list)
    for r in matrix:
        if r[3] == "c3":
            c3[(r[0], r[1], r[5], r[7], r[8], r[9])].append(r[10])
    put(a.out / "dnn_c3.csv", ["run", "arch", "threads_per_proc", "stage", "variant", "engine", "p50_median_ms",
                               "single_p50_same_threads_same_run_ms", "slowdown", "box_calls_per_s",
                               "best_single_calls_per_s_same_run"],
        [[*k, f(st.median(v)), single.get(k, ""), f(st.median(v) / single[k], 2) if single.get(k) else "",
          f(3000.0 / st.median(v), 2), f(best[(k[0], k[1], k[3], k[4], k[5])], 2) or ""] for k, v in sorted(c3.items())])
    put(a.out / "profile_ops.csv", ["run", "arch", "label", "model", "engine", "threads", "op", "time_ms", "count"], ops)
    put(a.out / "lightglue_layers.csv", ["run", "arch", "label", "threads", "K", "part", "ms", "share"], layers)
    put(a.out / "ort.csv", ["run", "arch", "label", "stage", "variant", "threads", "p50_ms", "p95_ms", "check"], ort)
    put(a.out / "mpstat_summary.csv", ["run", "arch", "phase", "samples", "busy_mean_pct", "busy_p95_pct",
                                       "steal_mean_pct"], mp)

    # ---- knobs (x86 run only has them)
    ks, trans, exact, fid, ad, var = [], [], [], [], [], []
    for d in runs:
        allk = []
        for sc in (0, 1):
            for r in rows(d / "knobs" / f"s{sc}" / "knobs.csv"):
                r["scene"] = sc
                allk.append(r)
            for r in rows(d / "knobs" / f"t2_s{sc}" / "knobs.csv"):
                if r["setting"] == "store":
                    exact.append([d.name, sc, r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"], r["prod_reason"],
                                  r["reason_here"], r["prod_matches"], r["matches_here"],
                                  int(r["prod_reason"] == r["reason_here"]), int(r["prod_matches"] == r["matches_here"])])
            vr = rows(d / "knobs" / f"s{sc}" / "variants.csv")
            for v in sorted({r["variant"] for r in vr}):
                sel = [r for r in vr if r["variant"] == v]
                var.append([d.name, sc, v, len(sel), sum(r["keypoints_bitwise"] == "1" for r in sel),
                            sum(r["scores_bitwise"] == "1" for r in sel), max(float(r["max_desc_abs_diff"]) for r in sel)])
            fr = rows(d / "knobs" / f"s{sc}" / "fidelity.csv")
            if fr:
                full = [r for r in fr if r["n_store"] not in ("missing", "") and r["matched_1e-3px"] == r["n_store"]]
                fid.append([d.name, sc, len(fr), len(full), sum(r["n_store"] == "missing" for r in fr),
                            max(float(r["max_px"]) for r in fr if r["max_px"])])
        if not allk:
            continue
        key = lambda r: (r["scene"], r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"])
        ref = {key(r): r for r in allk if r["setting"] == "1280x720:1024"}
        sto = {key(r): r for r in allk if r["setting"] == "store"}
        for s in sorted({r["setting"] for r in allk}):
            sel = [r for r in allk if r["setting"] == s]
            v = lambda r: r["reason_here"] == "verified"
            lost_r = sum(v(ref[key(r)]) and not v(r) for r in sel if key(r) in ref)
            gain_r = sum(not v(ref[key(r)]) and v(r) for r in sel if key(r) in ref)
            lost_s = sum(v(sto[key(r)]) and not v(r) for r in sel if key(r) in sto)
            gain_s = sum(not v(sto[key(r)]) and v(r) for r in sel if key(r) in sto)
            mr = [int(r["matches_here"]) / int(ref[key(r)]["matches_here"]) for r in sel
                  if key(r) in ref and int(ref[key(r)]["matches_here"]) > 0]
            er = [int(r["pnp_here"]) / int(ref[key(r)]["pnp_here"]) for r in sel
                  if key(r) in ref and int(ref[key(r)]["pnp_here"]) > 0]
            ks.append([d.name, s, len(sel), sum(v(r) for r in sel), lost_r, gain_r, lost_s, gain_s,
                       f(st.median(mr)) if mr else "", f(st.median(er)) if er else "",
                       f(st.median(float(r["ms_match_verify"]) for r in sel), 1)])
            for (x, y), n in sorted(Counter((ref[key(r)]["reason_here"], r["reason_here"]) for r in sel
                                            if key(r) in ref).items()):
                trans.append([d.name, s, x, y, n])
        for sc in (0, 1):
            g = rows(d / "knobs" / f"adaptive_s{sc}" / "gate.csv")
            if not g:
                continue
            by = defaultdict(dict)
            for r in g:
                by[(r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"])][r["mode"]] = r
            for mode in ("all9", "adaptive"):
                sel = [p[mode] for p in by.values() if mode in p]
                v = lambda r: r["reason_here"] == "verified"
                lost = sum(v(p["all9"]) and not v(p["adaptive"]) for p in by.values()) if mode == "adaptive" else 0
                gain = sum(not v(p["all9"]) and v(p["adaptive"]) for p in by.values()) if mode == "adaptive" else 0
                stops = Counter(int(r["stop"]) for r in sel)
                mr = [int(p["adaptive"]["n_matches"]) / int(p["all9"]["n_matches"]) for p in by.values()
                      if int(p["all9"]["n_matches"]) > 0] if mode == "adaptive" else []
                ad.append([d.name, sc, mode, len(sel), sum(v(r) for r in sel), lost, gain,
                           f(st.mean(int(r["stop"]) for r in sel), 2), " ".join(f"{k}:{n}" for k, n in sorted(stops.items())),
                           f(st.median(mr)) if mr else "", f(st.median(float(r["torch_ms"]) for r in sel), 1)])
    put(a.out / "knobs_summary.csv", ["run", "setting", "pairs", "verified", "lost_vs_ref1280k1024",
                                      "gained_vs_ref1280k1024", "lost_vs_store", "gained_vs_store",
                                      "median_match_ratio_vs_ref", "median_pnp_inlier_ratio_vs_ref",
                                      "median_ms_match_verify"], ks)
    put(a.out / "knobs_transitions.csv", ["run", "setting", "ref_reason", "reason", "pairs"], trans)
    put(a.out / "replay_exactness.csv", ["run", "scene", "agent_a", "kf_a", "agent_b", "kf_b", "prod_reason",
                                         "replay_reason", "prod_matches", "replay_matches", "reason_equal",
                                         "matches_equal"], exact)
    put(a.out / "fidelity_summary.csv", ["run", "scene", "keyframes", "all_keypoints_within_1e-3px", "missing_in_store",
                                         "max_px"], fid)
    put(a.out / "variants_summary.csv", ["run", "scene", "variant", "keyframes", "keypoints_bitwise", "scores_bitwise",
                                         "max_desc_abs_diff"], var)
    put(a.out / "adaptive_summary.csv", ["run", "scene", "mode", "pairs", "verified", "lost_vs_all9", "gained_vs_all9",
                                         "mean_exit_layer", "exit_layer_hist", "median_match_ratio_vs_all9",
                                         "median_torch_ms"], ad)
    print(f"matrix {len(matrix)}  ops {len(ops)}  layers {len(layers)}  ort {len(ort)}  knobs {len(ks)}  "
          f"exact {len(exact)}  adaptive {len(ad)}  mpstat {len(mp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
