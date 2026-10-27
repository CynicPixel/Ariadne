# In-place turn episodes from GT (200 Hz -> 20 Hz): maximal runs with planar speed <= 0.03 m/s and |yaw rate| > 0.03,
# split where the turn direction flips. Reports per policy: episodes, share > 180 deg, max, and direction reversals
# (a turn followed within 1 s of stillness by a turn the other way). Usage: turn_episodes.py EXTRACT_DIR
import sys, glob, re, math, collections
def load(p):
    t, x, y, h = [], [], [], []
    for i, l in enumerate(open(p)):
        if l.startswith('#') or i % 10: continue
        a = list(map(float, l.split())); qx, qy, qz, qw = a[4:8]
        t.append(a[0]); x.append(a[1]); y.append(a[2]); h.append(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    return t, x, y, h
S = collections.defaultdict(lambda: [0, 0, 0.0, 0, []])
for run in sorted(glob.glob(sys.argv[1] + '/*')):
    m = re.search(r'_(ci_tiny|maze_short)_(\d)_(?:mission_v2_(\w+)|-)$', run)
    if not m: continue
    key = (m.group(1), m.group(3) or 'nearest_frontier_1.8')
    for g in glob.glob(run + '/**/groundtruth_body.tum', recursive=True):
        t, x, y, h = load(g)
        eps = []; cur = None
        for i in range(1, len(t)):
            dt = t[i]-t[i-1]; d = math.hypot(x[i]-x[i-1], y[i]-y[i-1]); dh = (h[i]-h[i-1]+math.pi) % (2*math.pi) - math.pi
            turning = d/dt <= 0.03 and abs(dh)/dt > 0.03
            if turning:
                s = 1 if dh > 0 else -1
                if cur and cur[0] == s and t[i-1] - cur[3] < 1e-6: cur[1] += dh; cur[3] = t[i]
                else:
                    if cur: eps.append(tuple(cur))
                    cur = [s, dh, t[i-1], t[i]]
            elif cur and t[i] - cur[3] > 1.0:
                eps.append(tuple(cur)); cur = None
        if cur: eps.append(tuple(cur))
        st = S[key]
        for j, e in enumerate(eps):
            a = abs(math.degrees(e[1])); st[0] += 1; st[1] += a > 180; st[2] = max(st[2], a); st[4].append(a)
            if j and eps[j-1][0] != e[0] and e[2] - eps[j-1][3] <= 1.0: st[3] += 1
for k, (n, big, mx, rev, angs) in sorted(S.items()):
    angs.sort()
    print(f'{k[0]:10s} {k[1]:21s} turns {n:4d}  >180deg {big:3d}  max {mx:6.1f}  median {angs[len(angs)//2]:5.1f}  reversals<=1s {rev:4d} ({rev/max(n,1):.0%})')
