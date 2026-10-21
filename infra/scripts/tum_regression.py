#!/usr/bin/env python3
"""Check TUM RGB-D runs against configs/benchmark/tum_regression.yaml (2.6, D90).

    infra/scripts/tum_regression.py RUNS_DIR      # RUNS_DIR/<rgbd_dataset_SEQ-*>/results.yaml, newest per sequence

Per sequence: ATE (ate_se3.rmse_m) against the pin (regression: ATE > pin*(1+rel_tol)+abs_tol) and against the
published-class target (reported, not a failure). Exit 1 on any regression or missing run; 0 otherwise.
The runs themselves: ariadne_tum_vo with the config's perception/slam configs, seed and --intrinsics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = yaml.safe_load((ROOT / "configs/benchmark/tum_regression.yaml").read_text())
    tol = cfg["tolerance"]
    runs = Path(sys.argv[1])
    bad = 0
    print(f"{'sequence':16} {'ATE m':>8} {'pin m':>8} {'limit m':>8} {'target m':>9}  verdict")
    for seq, s in cfg["sequences"].items():
        found = sorted(runs.glob(f"rgbd_dataset_{seq}-*/results.yaml"))
        if not found:
            print(f"{seq:16} {'-':>8}  MISSING run")
            bad += 1
            continue
        ate = float(yaml.safe_load(found[-1].read_text())["ate_se3"]["rmse_m"])
        pin = s.get("pin_m")
        limit = None if pin is None else pin * (1 + tol["rel_tol"]) + tol["abs_tol_m"]
        regressed = limit is not None and ate > limit
        bad += regressed
        verdict = ("REGRESSION" if regressed else "ok" if limit is not None else "unpinned") + \
                  ("; in class" if ate <= s["target_m"] else "; NOT in class")
        print(f"{seq:16} {ate:8.4f} {pin if pin is not None else '-':>8} "
              f"{f'{limit:.4f}' if limit is not None else '-':>8} {s['target_m']:9.3f}  {verdict}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
