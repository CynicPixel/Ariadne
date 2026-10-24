#!/usr/bin/env python3
"""Phase 3.6 bytes study: D233's pre-registered verdicts from ariadne_perf_bench bytes output (docs/design/BYTES_3.6.md §5).

  python3 -I analyze.py --raw <bench out dir> --out processed

Reads pairs.csv, bytes_kf.csv and geometry.csv; writes processed/:
  bytes_summary.csv     per setting: mean payload / transmitted / wire bytes per keyframe, the reduction vs R (kB) and the
                        zstd ratio
  precondition.csv      per scene: R (int8/1024/xyz, seed 0) vs the production verdict
  seed_envelope.csv     per scene: seed-stable pairs, R's accept count and median pose_err per seed
  coding_verdicts.csv   per coding row and scene: codec flips by direction, unstable flips, reason changes, median
                        pnp-inlier ratio, median pose_err vs R's envelope, bytes, and the D233 verdict
  posthoc_vs_fp32.csv   POST-HOC, report-only: verdict differences of every setting vs int8 R and vs the fp32 control,
                        with the GT class of each differing pair
  combination.csv       the adopted codings' joint row: the bars again, and predicted vs measured wire bytes
Report-only (not bars): GT-true / GT-false accepts per setting (split at 1 m / 10 deg, see GT_TRUE_T).
Verdict logic is D233 verbatim: adopted = bars 1 (0 codec flips) + 2 (inlier ratio >= 0.90) + 3 (>= 15 kB below R);
M decides = bars 2 + 3, 0 reject->accept and <= 1 accept->reject codec flip per scene; else not adopted. Each bar on
each scene separately. Pose is a flag, not a bar.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

R = ("int8", "1024", "xyz")
DESC_ROWS = [("fp32", "1024", "xyz"), ("int4", "1024", "xyz"), ("bin1", "1024", "xyz"), ("pca64", "1024", "xyz"),
             ("pca32", "1024", "xyz")]
GEO_ROWS = [("int8", "1024", "z32"), ("int8", "1024", "disp16"), ("int8", "1024", "kp16")]
INLIER_BAR, BYTES_BAR = 0.90, 15000.0
# Report-only GT split (not a bar): production's verified set is bimodal (<= 0.05 m / 0.6 deg, or >= 4.6 m / ~90 deg:
# D176's aliased edges), so an accepted edge counts as GT-true under 1 m and 10 deg.
GT_TRUE_T, GT_TRUE_R = 1.0, 10.0


def fnum(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return math.nan


def med(v: list[float]) -> float:
    v = [x for x in v if not math.isnan(x)]
    return statistics.median(v) if v else math.nan


def put(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # pairs.csv -> res[(scene, pair)][(desc, k, geo, seed_offset)] = row
    res: dict = defaultdict(dict)
    prod: dict = {}
    for r in csv.DictReader(open(a.raw / "pairs.csv")):
        key = (int(r["scene"]), (r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]))
        res[key][(r["desc"], r["k"], r["geo"], int(r["seed_offset"]))] = r
        prod[key] = r["prod_reason"]
    scenes = sorted({s for s, _ in res})
    ok = lambda row: row["reason"] == "verified"
    gt_true = lambda row: ok(row) and fnum(row["gt_err_t_m"]) < GT_TRUE_T and fnum(row["gt_err_r_deg"]) < GT_TRUE_R

    # bytes
    by: dict = defaultdict(list)
    for r in csv.DictReader(open(a.raw / "bytes_kf.csv")):
        by[(r["desc"], r["k"], r["geo"])].append(r)
    mean = lambda s, f: statistics.fmean(float(x[f]) for x in by[s])
    r_wire = mean(R, "wire_bytes")
    put(a.out / "bytes_summary.csv",
        ["desc", "k", "geo", "keyframes", "feature_bytes", "payload_mean", "transmitted_mean", "wire_mean",
         "wire_kB", "reduction_vs_R_kB", "ratio_vs_R", "zstd_ratio", "predicted_today_wire_mean"],
        [[*s, len(v), v[0]["feature_bytes"], round(mean(s, "payload_bytes"), 1), round(mean(s, "transmitted_bytes"), 1),
          round(mean(s, "wire_bytes"), 1), round(mean(s, "wire_bytes") / 1000, 2),
          round((r_wire - mean(s, "wire_bytes")) / 1000, 2), round(mean(s, "wire_bytes") / r_wire, 4),
          round(mean(s, "transmitted_bytes") / mean(s, "payload_bytes"), 4), round(mean(s, "predicted_today_wire"), 1)]
         for s, v in sorted(by.items())])

    # precondition and the seed envelope
    seeds = sorted({k[3] for v in res.values() for k in v if k[:3] == R})
    pre, env, stable = [], [], {}
    for sc in scenes:
        keys = [k for k in res if k[0] == sc]
        same = sum((res[k][(*R, 0)]["reason"] == prod[k]) for k in keys)
        same_acc = sum(ok(res[k][(*R, 0)]) == (prod[k] == "verified") for k in keys)
        pre.append([sc, len(keys), same, same_acc, "PASS" if same == len(keys) else "FAIL (reported; flips stay vs in-run R)"])
        for k in keys:
            stable[k] = len({ok(res[k][(*R, s)]) for s in seeds}) == 1
        for s in seeds:
            acc = [k for k in keys if ok(res[k][(*R, s)])]
            env.append([sc, s, sum(stable[k] for k in keys), len(acc), med([fnum(res[k][(*R, s)]["pose_err_deg"]) for k in acc])])
    put(a.out / "precondition.csv", ["scene", "pairs", "reason_equal_prod", "accept_equal_prod", "verdict"], pre)
    put(a.out / "seed_envelope.csv", ["scene", "seed_offset", "seed_stable_pairs", "r_accepted", "r_median_pose_err_deg"], env)
    env_max = {sc: max(e[4] for e in env if e[0] == sc) for sc in scenes}

    def judge(s: tuple) -> tuple[list[list], str]:
        rows, verdicts = [], []
        reduction = r_wire - mean(s, "wire_bytes")
        for sc in scenes:
            keys = [k for k in res if k[0] == sc]
            ra = rr = un = rc = 0
            ratios, pose = [], []
            r_true = sum(gt_true(res[k][(*R, 0)]) for k in keys)
            s_true = sum(gt_true(res[k][(*s, 0)]) for k in keys)
            s_false = sum(ok(res[k][(*s, 0)]) and not gt_true(res[k][(*s, 0)]) for k in keys)
            flips_of_gt_true = sum(ok(res[k][(*R, 0)]) != ok(res[k][(*s, 0)]) and gt_true(res[k][(*R, 0)]) for k in keys)
            for k in keys:
                x, ref = res[k][(*s, 0)], res[k][(*R, 0)]
                if ok(x) != ok(ref):
                    if not stable[k]:
                        un += 1
                    elif ok(x):
                        ra += 1
                    else:
                        rr += 1
                elif not ok(x) and x["reason"] != ref["reason"]:
                    rc += 1
                if ok(ref):
                    ratios.append(fnum(x["pnp_inliers"]) / max(fnum(ref["pnp_inliers"]), 1.0))
                if ok(ref) and ok(x):
                    pose.append(fnum(x["pose_err_deg"]))
            mr, mp = med(ratios), med(pose)
            b1, b2, b3 = ra == 0 and rr == 0, mr >= INLIER_BAR, reduction >= BYTES_BAR
            v = "adopted" if b1 and b2 and b3 else (
                "M decides" if b2 and b3 and ra == 0 and rr <= 1 else "not adopted")
            verdicts.append(v)
            rows.append([*s, sc, ra, rr, un, rc, round(mr, 4), round(mp, 4), round(env_max[sc], 4),
                         int(mp > env_max[sc]), r_true, s_true, s_false, flips_of_gt_true,
                         round(mean(s, "wire_bytes") / 1000, 2), round(reduction / 1000, 2), v])
        overall = ("adopted" if all(v == "adopted" for v in verdicts) else
                   "M decides" if all(v in ("adopted", "M decides") for v in verdicts) else "not adopted")
        return rows, overall

    hdr = ["desc", "k", "geo", "scene", "codec_flips_reject_to_accept", "codec_flips_accept_to_reject",
           "flips_on_seed_unstable_pairs", "reject_reason_changes", "median_pnp_inlier_ratio_vs_R",
           "median_pose_err_deg", "r_seed_envelope_max_median_pose_err_deg", "pose_flag_to_M", "r_gt_true_accepts",
           "s_gt_true_accepts", "s_gt_false_accepts", "flips_of_r_gt_true_edges", "wire_kB",
           "reduction_vs_R_kB", "verdict_scene"]
    # fp32 control: if it fails bar 1 or 2, int8 is at the resolvable floor - nothing adopted without M
    ctrl_rows, _ = judge(("fp32", "1024", "xyz"))
    control_ok = all(r[hdr.index("codec_flips_reject_to_accept")] == 0 and r[hdr.index("codec_flips_accept_to_reject")] == 0
                     and r[hdr.index("median_pnp_inlier_ratio_vs_R")] >= INLIER_BAR for r in ctrl_rows)
    out, overall = [], {}
    for s in DESC_ROWS + GEO_ROWS:
        if (*s, 0) not in next(iter(res.values())):
            continue
        rows, overall[s] = judge(s)
        if overall[s] == "adopted" and not control_ok:
            overall[s] = "M decides (fp32 control failed)"
        out += [r + [overall[s]] for r in rows]
    put(a.out / "coding_verdicts.csv", hdr + ["verdict"], out)


    # combination: smallest-wire adopted descriptor + smallest-wire adopted geometry (ties: as listed)
    pick = lambda rows: min((s for s in rows if overall.get(s) == "adopted" and s[0] != "fp32"),
                            key=lambda s: mean(s, "wire_bytes"), default=None)
    d, g = pick(DESC_ROWS), pick(GEO_ROWS)
    combo = (d[0] if d else "int8", "1024", g[2] if g else "xyz")
    crow = []
    if combo != R and (*combo, 0) in next(iter(res.values())):
        rows, verdict = judge(combo)
        # predicted vs measured (I6, <= 15 %): the payload of a layout is exact at the codec; the zstd ratio is
        # calibrated on scene 1's keyframes and predicts scene 0's (disjoint, D178); today's predictor (layout-blind,
        # uncalibrated) alongside
        ks = {sc: [x for x in by[combo] if int(x["scene"]) == sc] for sc in scenes}
        ratio = statistics.fmean(float(x["transmitted_bytes"]) / float(x["payload_bytes"]) for x in ks[1])
        frag = lambda framed: math.ceil(framed / 1466)
        errs, errs_today = [], []
        for x in ks[0]:
            framed = 6 + math.ceil(ratio * float(x["payload_bytes"]))   # kFrameHeaderBytes + transmitted
            pred = framed + 34 * frag(framed)
            errs.append(abs(pred - float(x["wire_bytes"])) / float(x["wire_bytes"]))
            errs_today.append(abs(float(x["predicted_today_wire"]) - float(x["wire_bytes"])) / float(x["wire_bytes"]))
        crow = [[*combo, verdict, round(mean(combo, "wire_bytes") / 1000, 2), round(ratio, 4),
                 round(max(errs) * 100, 2), round(statistics.fmean(errs_today) * 100, 2), int(max(errs) <= 0.15)]]
        out_rows = [r + [verdict] for r in rows]
        put(a.out / "combination_bars.csv", hdr + ["verdict"], out_rows)
    put(a.out / "combination.csv", ["desc", "k", "geo", "verdict", "wire_kB", "zstd_ratio_cal_s1",
                                    "max_pred_err_pct_s0_new_formula", "mean_pred_err_pct_s0_today_predictor",
                                    "i6_gate_ok"], crow)
    # POST-HOC, report-only (not part of D233's rule; written after the fp32 control failed): every setting's verdicts
    # against the fp32 control instead of int8, with the GT class of each differing pair.
    ph = []
    for s in [("fp32", "1024", "xyz"), R] + DESC_ROWS[1:] + GEO_ROWS:
        for sc in scenes:
            keys = [k for k in res if k[0] == sc]
            for ref_name, ref in (("int8_R", R), ("fp32", ("fp32", "1024", "xyz"))):
                if s == ref:
                    continue
                diff = [k for k in keys if ok(res[k][(*s, 0)]) != ok(res[k][(*ref, 0)])]
                ph.append([*s, sc, ref_name, len(diff),
                           " ".join(f"{k[1][0]}:{k[1][1]}-{k[1][2]}:{k[1][3]}:{'acc' if ok(res[k][(*s, 0)]) else 'rej'}:"
                                    f"{'GTtrue' if gt_true(res[k][(*s, 0)]) or gt_true(res[k][(*ref, 0)]) else 'GTfalse'}"
                                    for k in diff)])
    put(a.out / "posthoc_vs_fp32.csv", ["desc", "k", "geo", "scene", "reference", "verdict_differences",
                                        "pairs (agent:kf-agent:kf:this setting's verdict:GT class)"], ph)
    print(f"R wire {r_wire / 1000:.2f} kB/KF; control fp32 {'ok' if control_ok else 'FAILS: nothing adopted without M'}")
    for s, v in overall.items():
        print(f"  {'/'.join(s)}: {v}")
    print(f"combination {'/'.join(combo)}: {crow[0][3] if crow else 'none (no coding adopted)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
