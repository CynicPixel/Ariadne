# Where does "still" time come from? Per 5 s barrier interval [5k, 5k+5) and agent, from GT at 20 Hz:
#   lead  = still time before the first motion in the interval (command latency / planning),
#   tail  = still time after the last motion in the interval (arrived early, waiting for the barrier),
#   whole = intervals with no motion at all (hold or a no-op command).
# From the trace: decisions with result != "go" or empty waypoints. Also splits turn_back pairs (turn, move, opposite
# turn): "in-command" if the second turn starts > 0.5 s after a barrier, "at barrier" if within 0.5 s.
# Usage: still_decomp.py EXTRACT_DIR T
import sys, glob, re, math, json, collections
T = float(sys.argv[2]); B = 5.0
def load(p):
    t, x, y, h = [], [], [], []
    for i, l in enumerate(open(p)):
        if l.startswith('#') or i % 10: continue
        a = list(map(float, l.split())); qx, qy, qz, qw = a[4:8]
        t.append(a[0]); x.append(a[1]); y.append(a[2]); h.append(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    return t, x, y, h
S = collections.defaultdict(lambda: collections.Counter())
for run in sorted(glob.glob(sys.argv[1] + '/*')):
    m = re.search(r'_(ci_tiny|maze_short)_(\d)_(?:mission_v2_(\w+)|-)$', run)
    if not m: continue
    key = (m.group(1), m.group(3) or 'nearest_frontier_1.8'); c = S[key]
    tr = glob.glob(run + '/**/trace.jsonl', recursive=True)[0]
    for l in open(tr):
        d = json.loads(l)
        if d['t_s'] > T: break
        c['decisions'] += 1
        if d.get('result') != 'go' or not (d.get('command') or {}).get('waypoints_xy_m'): c['no_go'] += 1
    for g in glob.glob(run + '/**/groundtruth_body.tum', recursive=True):
        t, x, y, h = load(g)
        act = []   # (t, active, kind, sign)
        for i in range(1, len(t)):
            if t[i] > T: break
            dt = t[i]-t[i-1]; d = math.hypot(x[i]-x[i-1], y[i]-y[i-1]); dh = (h[i]-h[i-1]+math.pi) % (2*math.pi) - math.pi
            k = 'm' if d/dt > 0.03 else ('t' if abs(dh)/dt > 0.03 else 's')
            act.append((t[i], dt, k, (1 if dh > 0 else -1) if k == 't' else 0, dh))
        nint = int(T // B)
        for j in range(nint):
            seg = [a for a in act if B*j <= a[0] < B*(j+1)]
            if not seg: continue
            mv = [i for i, a in enumerate(seg) if a[2] != 's']
            c['intervals'] += 1
            if not mv: c['whole_s'] += sum(a[1] for a in seg); c['idle_intervals'] += 1; continue
            c['lead_s'] += sum(a[1] for a in seg[:mv[0]]); c['tail_s'] += sum(a[1] for a in seg[mv[-1]+1:])
            c['mid_s'] += sum(a[1] for a in seg[mv[0]:mv[-1]+1] if a[2] == 's')
        # turn-move-opposite-turn pairs, classified by the second turn's start
        segs = []
        for a in act:
            if a[2] == 's': continue
            if segs and segs[-1][0] == a[2] and segs[-1][1] == a[3]: segs[-1][3] += a[4]
            else: segs.append([a[2], a[3], a[0] - a[1], a[4]])
        for j in range(2, len(segs)):
            p, mvv, q = segs[j-2], segs[j-1], segs[j]
            if p[0] == 't' and mvv[0] == 'm' and q[0] == 't' and p[1] != q[1] and abs(p[3]) > 0.35 and abs(q[3]) > 0.35:
                ph = q[2] % B
                c['pair_at_barrier' if ph < 0.5 or ph > B - 0.15 else 'pair_in_command'] += 1
for k, c in sorted(S.items()):
    tot = c['lead_s'] + c['tail_s'] + c['mid_s'] + c['whole_s']; n = max(c['intervals'], 1)
    print(f"{k[0]:10s} {k[1]:21s} T<={T:.0f}s  still per interval: lead {c['lead_s']/n:.2f}s tail {c['tail_s']/n:.2f}s "
          f"mid {c['mid_s']/n:.2f}s whole {c['whole_s']/n:.2f}s (idle intervals {c['idle_intervals']}/{c['intervals']}) | "
          f"no-go decisions {c['no_go']}/{c['decisions']} | turn-back pairs in-command {c['pair_in_command']} at-barrier {c['pair_at_barrier']}")
