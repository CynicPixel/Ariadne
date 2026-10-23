# NBV_4.3 §5 secondary view: coverage vs summed metres flown, at the barrier points. Per seed, linear interpolation
# between barrier points at fixed metres, only within the metres the run actually reached (no extrapolation).
import json, sys, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
runs = collections.defaultdict(list)
for r in rows:
    runs[(r["scene"], r["policy"], r["seed"])].append((sum(r["flown_m"]) if isinstance(r["flown_m"], list) else r["flown_m"], r["coverage"]))
for sc in sorted({k[0] for k in runs}):
    grid = [5, 10, 15, 20, 25, 30, 40]
    print(sc, "metres:", grid)
    for pol in sorted({k[1] for k in runs if k[0] == sc}):
        out = []
        for m in grid:
            vals = []
            for (s, p, seed), pts in runs.items():
                if s != sc or p != pol: continue
                pts = sorted([(0.0, 0.0)] + pts)
                x, y = zip(*pts)
                if m <= max(x): vals.append(float(np.interp(m, x, y)))
            out.append(f"{np.mean(vals):.3f}/{len(vals)}" if vals else "  -  ")
        print(f"  {pol:17s}", " ".join(out))
