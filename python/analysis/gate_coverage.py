"""4.3 / 4.6 planner gate (next-best-view.md §5, amended 2026-10-23 before any gate run; D114, D116): coverage vs mission time.

Every row is scored from the SAME Phase 4 map: the run's shadow-map snapshots `snapshots/p4slab_<ms>.png` (at
7916ee7), 4.1's metric (GT-free cells the map observed free / GT-free cells; volumetric_eval.py). Primary x: mission
time T = 20, 40, ..., 200 s (10 fractions of the 200 s mission, all on 5 s lockstep barriers). Secondary: coverage
against summed metres flown (GT body path length, the same for every row) at the same barriers, actual points only.

    python3 -I python/analysis/gate_coverage.py --runs DIR [DIR ...] --scenes SCENES_ROOT --out OUT

Each DIR is one run directory (or an extracted live_aws batch tarball) holding live_config.yaml / planner_config.yaml,
trace.jsonl and snapshots/. Its row is read from planner_config.yaml (planner.policy) or "nearest_frontier_1.8" when
the run has no planner config (the PlaceholderPlanner); its scene and seed from the run's manifest.yaml.
Writes OUT/gate_rows.jsonl (per run, per T) and OUT/gate_summary.yaml (per scene, policy: mean / min / max over seeds
at every T, area under the curve, and the pre-registered claim "Ariadne >= every baseline at every T", per scene).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from volumetric_eval import FREE, read_png_u8  # noqa: E402  (the 4.1 reader and slab codes, one implementation)

T_GRID_S = [20 * k for k in range(1, 11)]   # pre-registered (next-best-view.md §5 amendment)


def run_identity(run: Path) -> dict:
    man = yaml.safe_load((run / "manifest.yaml").read_text()) if (run / "manifest.yaml").exists() else {}
    # The row: the trace's own planner record (A1Planner writes name a1_phase4 + its policy); anything else is the
    # 1.8 placeholder (the PlaceholderPlanner).
    policy = "nearest_frontier_1.8"
    for line in (run / "trace.jsonl").read_text().splitlines():
        p = json.loads(line).get("planner") or {}
        if p.get("name") == "a1_phase4":
            policy = str(p.get("policy"))
            break
    # The scene is the --scene argument of the recorded command (build/live/scenes/<preset>_<seed>); the manifest's
    # `seed` field is the RUN seed, not the scene's.
    m = re.search(r"--scene\s+\S*/([A-Za-z_]+?)_(\d+)(\s|$)", str(man.get("command", "")))
    if not m:
        raise SystemExit(f"{run}: cannot read the scene from manifest.yaml's command")
    return {"policy": policy, "scene": m.group(1), "seed": int(m.group(2)), "run_seed": man.get("seed")}


def score_run(run: Path, scenes: Path, t_grid: list[int]) -> list[dict]:
    ident = run_identity(run)
    gt = np.load(scenes / f"{ident['scene']}_{ident['seed']}" / "scene_gt.npz")
    gtfree = gt["free"].astype(bool)
    # Metres flown, summed over agents, to each T: the GT body trajectories (raw/*/raw/groundtruth_body.tum, sim
    # seconds), the same quantity for every row (A1Planner's flown_m is this GT path length too).
    tracks = []
    for f in sorted(run.glob("raw/*/raw/groundtruth_body.tum")):
        a = np.loadtxt(f, comments="#", usecols=(0, 1, 2, 3))
        tracks.append(a[np.argsort(a[:, 0])])
    def flown_to(T: float) -> float:
        tot = 0.0
        for a in tracks:
            p = a[a[:, 0] <= T + 1e-9][:, 1:4]
            if len(p) > 1:
                tot += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        return tot
    rows = []
    for T in t_grid:
        snap = run / "snapshots" / f"p4slab_{T * 1000}.png"
        if not snap.exists():
            raise SystemExit(f"{run}: missing {snap.name} (the gate needs every barrier T = 20..200 s)")
        slab = read_png_u8(snap)
        if slab.shape != gtfree.shape:
            raise SystemExit(f"{run}: slab {slab.shape} vs GT {gtfree.shape}")
        cov = float(((slab == FREE) & gtfree).sum() / gtfree.sum())
        rows.append({**ident, "run": run.name, "t_s": T, "coverage": cov, "flown_m": flown_to(T) if tracks else None})
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for scene in sorted({r["scene"] for r in rows}):
        s = out.setdefault(scene, {"policies": {}})
        for pol in sorted({r["policy"] for r in rows if r["scene"] == scene}):
            rr = [r for r in rows if r["scene"] == scene and r["policy"] == pol]
            seeds = sorted({r["seed"] for r in rr})
            curve = []
            for T in T_GRID_S:
                v = [r["coverage"] for r in rr if r["t_s"] == T]
                curve.append({"t_s": T, "mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)})
            span = T_GRID_S[-1] - T_GRID_S[0]
            auc = float(np.trapz([c["mean"] for c in curve], T_GRID_S) / span) if span > 0 else curve[0]["mean"]
            s["policies"][pol] = {"seeds": seeds, "curve": curve, "auc_mean": auc}
        pols = s["policies"]
        if "ariadne" in pols:
            base = [p for p in pols if p != "ariadne"]
            losses = [{"t_s": c["t_s"], "vs": b, "ariadne": c["mean"], "baseline": pols[b]["curve"][i]["mean"]}
                      for b in base for i, c in enumerate(pols["ariadne"]["curve"]) if c["mean"] < pols[b]["curve"][i]["mean"]]
            s["claim_ariadne_ge_every_baseline_every_T"] = not losses
            s["losses"] = losses   # reported, never hidden (next-best-view.md §5)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--scenes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--t-max", type=int, default=200, help="smoke tests on short runs only; the gate is 200")
    a = ap.parse_args(argv)
    global T_GRID_S
    T_GRID_S = [t for t in T_GRID_S if t <= a.t_max]
    rows = [r for run in a.runs for r in score_run(run, a.scenes, T_GRID_S)]
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "gate_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summ = summarise(rows)
    (a.out / "gate_summary.yaml").write_text("# gate_coverage.py (next-best-view.md §5 amended, D114/D116). Generated.\n"
                                             + yaml.safe_dump(summ, sort_keys=False))
    for scene, s in summ.items():
        print(scene, {p: round(v["auc_mean"], 4) for p, v in s["policies"].items()},
              "claim:", s.get("claim_ariadne_ge_every_baseline_every_T"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
