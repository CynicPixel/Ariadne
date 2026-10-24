# Gate diagnostic (NBV_4.3 §14): per scene/policy/seed, the median path_m of the chosen candidate over decisions at
# t <= T and the summed metres flown at T, from each run's trace.jsonl. Usage: early_paths.py EXTRACT_DIR T
import json, sys, glob, statistics as st, re, collections
R = collections.defaultdict(list)
for run in sorted(glob.glob(sys.argv[1] + "/*")):
    m = re.search(r"_(ci_tiny|maze_short)_(\d)_(?:mission_v2_(\w+)|-)$", run)
    if not m: continue
    sc, seed, pol = m.group(1), m.group(2), m.group(3) or "nearest_frontier_1.8"   # "-": the 1.8 placeholder
    t = glob.glob(run + "/**/trace.jsonl", recursive=True)[0]
    paths, stalls, fl = [], 0, None
    prev = {}
    for l in open(t):
        d = json.loads(l)
        if d["t_s"] > float(sys.argv[2]): break
        ch = d.get("chosen")
        sel = next((c for c in d.get("candidates", []) if c.get("index") == ch), None)
        if sel and sel.get("path_m") is not None: paths.append(sel["path_m"])
        fl = d.get("planner", {}).get("flown_m", [float("nan")])   # the 1.8 row has no planner block
    R[(sc, pol)].append((st.median(paths) if paths else float("nan"), sum(fl)))
for (sc, pol), v in sorted(R.items()):
    print(f"{sc:10s} {pol:17s} median chosen path_m (t<=T) per seed {[round(a,2) for a,_ in v]}  summed flown_m at T {[round(b,1) for _,b in v]}")
