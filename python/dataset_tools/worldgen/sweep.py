"""Seed sweeps: the evidence behind the preset length bands (design §9 step 2).

census(preset, seeds)   for each seed, the first attempt with a topologically valid
                        layout: the rho_responder geodesic from the primary entrance to
                        EVERY valid target room (band ignored). The distribution of
                        achievable lengths, from which bands are chosen.
acceptance(preset, seeds)
                        the real generate(): attempts used, accepted lengths, and the
                        rejection histogram, or the failure reason.
Both are embarrassingly parallel over seeds and deterministic per seed.
"""
from __future__ import annotations

import dataclasses
import math
import traceback
from collections import Counter
from multiprocessing import Pool

import numpy as np

from .layout import Reject, S_TARGET, build_base, complete_for_target
from .params import Preset
from .raster import geodesic, traversable
from .scene import GenerationFailed, _raster, _room_center_cell, generate


def _census_one(args):
    preset, seed = args
    try:
        return _census_one_inner(preset, seed)
    except Exception:
        return {"seed": seed, "attempt": None, "lengths_m": [], "error": traceback.format_exc()}


def _census_one_inner(preset, seed):
    g = preset.structure.grid_m
    rho = preset.clearance.rho_responder_m
    for attempt in range(preset.search.max_attempts):
        try:
            base = build_base(preset, seed, attempt)
        except Reject:
            continue
        entrance_rooms = {e.room for e in base.entrances}
        lengths = []
        for room in base.stream(S_TARGET).shuffled(sorted(c for c in base.rooms if c not in entrance_rooms)):
            try:
                L = complete_for_target(base, room)
            except Reject:
                continue
            grid, _, _, e = _raster(L, cap_m=rho + 2 * g)
            trav = traversable(e, rho)
            start = grid.index(*L.entrances[0].start)
            d = geodesic(trav, start, g)[0][_room_center_cell(grid, L.rooms[room])]
            lengths.append(float(d))
        if lengths:
            return {"seed": seed, "attempt": attempt, "lengths_m": lengths}
    return {"seed": seed, "attempt": None, "lengths_m": []}


def _accept_one(args):
    preset, seed = args
    try:
        s = generate(preset, seed)
    except GenerationFailed as f:
        return {"seed": seed, "ok": False, "reasons": dict(f.reasons)}
    except Exception:           # a bug, not a rejection: record it, never kill the pool
        return {"seed": seed, "ok": False, "reasons": {}, "error": traceback.format_exc()}
    return {"seed": seed, "ok": True, "attempt": s.attempt, "scene_id": s.scene_id,
            "lengths": {str(k): v for k, v in s.lengths.items()},
            "durations_s": [sc.duration for sc in s.scripts],
            "start_delays_s": list(s.start_delays_s),
            "min_separation_m": s.min_separation_m if np.isfinite(s.min_separation_m) else None,
            "inter_pairs": s.overlap.get("inter_pairs"), "inter_rooms": len(s.overlap.get("inter_rooms", [])),
            "props": len(s.props), "reasons": dict(s.rejections)}


def _map(fn, preset, seeds, workers):
    jobs = [(preset, s) for s in seeds]
    if workers <= 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(fn, jobs, chunksize=4)


def census(preset: Preset, seeds, workers: int = 1) -> dict:
    # The band is irrelevant to a census; widen it so nothing is filtered upstream.
    wide = dataclasses.replace(preset, target=dataclasses.replace(preset.target, path_length_m=(0.0, 1e9)))
    rows = _map(_census_one, wide, seeds, workers)
    all_len = np.array([x for r in rows for x in r["lengths_m"]])
    return {"preset": preset.name, "preset_digest": preset.digest(), "seeds": len(rows),
            "errors": sum(1 for r in rows if "error" in r),
            "seeds_without_valid_layout": sum(1 for r in rows if r["attempt"] is None),
            "summary": summarize(all_len), "rows": rows}


def acceptance(preset: Preset, seeds, workers: int = 1) -> dict:
    rows = _map(_accept_one, preset, seeds, workers)
    ok = [r for r in rows if r["ok"]]
    reasons = Counter()
    for r in rows:
        reasons.update(r["reasons"])
    attempts = np.array([r["attempt"] for r in ok])
    return {"preset": preset.name, "preset_digest": preset.digest(), "seeds": len(rows),
            "errors": sum(1 for r in rows if "error" in r),
            "accepted": len(ok), "acceptance_rate": len(ok) / max(len(rows), 1),
            "attempts": summarize(attempts + 1.0) if len(ok) else None,
            "length_agent0_m": summarize(np.array([r["lengths"]["0"]["responder_m"] for r in ok])),
            "length_agent1_m": summarize(np.array([r["lengths"]["1"]["responder_m"] for r in ok
                                                   if "1" in r["lengths"]])),
            "mission_duration_s": summarize(np.array([max(r["durations_s"]) for r in ok])) if ok else None,
            "start_delay_s": summarize(np.array([max(r["start_delays_s"]) for r in ok])) if ok else None,
            "min_separation_m": summarize(np.array([r["min_separation_m"] for r in ok
                                                    if r["min_separation_m"] is not None])) if ok else None,
            "inter_pairs": summarize(np.array([r["inter_pairs"] for r in ok])) if ok else None,
            "inter_rooms": summarize(np.array([r["inter_rooms"] for r in ok])) if ok else None,
            "rejection_reasons": dict(reasons.most_common()),
            "failures": [r for r in rows if not r["ok"]], "rows": rows}


def summarize(x: np.ndarray, bins: int = 12) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    lo, hi = math.floor(q[0]), math.ceil(q[-1])
    edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    counts, _ = np.histogram(x, edges)
    return {"n": int(x.size), "mean": float(x.mean()),
            "p0_p5_p25_p50_p75_p95_p100": [round(float(v), 2) for v in q],
            "hist_edges": [round(float(e), 2) for e in edges], "hist_counts": counts.tolist()}
