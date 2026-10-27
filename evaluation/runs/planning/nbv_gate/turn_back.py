# Turn -> move -> opposite turn (the face-the-waypoint turn undone by the final-yaw turn), from GT at 20 Hz.
# Segments: maximal runs of 'turn' (speed <= 0.03, |yaw rate| > 0.03, one direction) or 'move' (speed > 0.03);
# stillness between segments is skipped. Per policy, over t <= T: share of all turning time spent in such pairs
# (both turns), their count, and the median net heading change of a pair vs the rotation spent.
# Usage: turn_back.py EXTRACT_DIR T
import sys, glob, re, math, collections
def load(p):
    t, x, y, h = [], [], [], []
    for i, l in enumerate(open(p)):
        if l.startswith('#') or i % 10: continue
        a = list(map(float, l.split())); qx, qy, qz, qw = a[4:8]
        t.append(a[0]); x.append(a[1]); y.append(a[2]); h.append(math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz)))
    return t, x, y, h
T = float(sys.argv[2])
S = collections.defaultdict(lambda: [0.0, 0.0, 0, [], [], 0.0])
for run in sorted(glob.glob(sys.argv[1] + '/*')):
    m = re.search(r'_(ci_tiny|maze_short)_(\d)_(?:mission_v2_(\w+)|-)$', run)
    if not m: continue
    key = (m.group(1), m.group(3) or 'nearest_frontier_1.8')
    for g in glob.glob(run + '/**/groundtruth_body.tum', recursive=True):
        t, x, y, h = load(g)
        segs = []
        for i in range(1, len(t)):
            if t[i] > T: break
            dt = t[i]-t[i-1]; d = math.hypot(x[i]-x[i-1], y[i]-y[i-1]); dh = (h[i]-h[i-1]+math.pi) % (2*math.pi) - math.pi
            if d/dt > 0.03: k, s = 'm', 0
            elif abs(dh)/dt > 0.03: k, s = 't', (1 if dh > 0 else -1)
            else: continue
            if segs and segs[-1][0] == k and segs[-1][1] == s: segs[-1][2] += dh; segs[-1][3] += dt; segs[-1][4] += d
            else: segs.append([k, s, dh, dt, d])
        st = S[key]
        st[0] += sum(sg[3] for sg in segs if sg[0] == 't')
        for j in range(2, len(segs)):
            a, mv, b = segs[j-2], segs[j-1], segs[j]
            if a[0] == 't' and mv[0] == 'm' and b[0] == 't' and a[1] != b[1] and abs(a[2]) > 0.35 and abs(b[2]) > 0.35:
                st[1] += a[3] + b[3]; st[2] += 1
                st[3].append(math.degrees(abs(a[2]) + abs(b[2]))); st[4].append(math.degrees(abs(a[2] + b[2]))); st[5] += mv[4]
for k, (tt, pt, n, spent, net, md) in sorted(S.items()):
    spent.sort(); net.sort()
    med = lambda v: v[len(v)//2] if v else float('nan')
    print(f'{k[0]:10s} {k[1]:21s} T<={T:.0f}s  turn-move-backturn pairs {n:3d}  = {pt/max(tt,1e-9):.0%} of turning time  '
          f'median rotation spent {med(spent):5.1f} deg for net {med(net):5.1f} deg  mean move between {md/max(n,1):.2f} m')
