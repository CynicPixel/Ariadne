# Where does each policy's time go? From each run's GT body trajectories (200 Hz), resampled at 10 Hz:
# move (planar speed > 0.03 m/s), turn in place (speed <= 0.03 and |yaw rate| > 0.03 rad/s), still (neither).
# Per scene/policy: mean over 3 seeds x 2 agents of the fractions over t <= T, plus yaw travelled per metre.
# Usage: time_budget.py EXTRACT_DIR T [T ...]
import sys, glob, re, math, collections
V, W = 0.03, 0.03
def load(p):
    t, x, y, h = [], [], [], []
    for i, l in enumerate(open(p)):
        if l.startswith('#') or i % 20: continue
        a = list(map(float, l.split()))
        qx, qy, qz, qw = a[4:8]
        t.append(a[0]); x.append(a[1]); y.append(a[2])
        h.append(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return t, x, y, h
R = collections.defaultdict(list)
Ts = [float(v) for v in sys.argv[2:]]
for run in sorted(glob.glob(sys.argv[1] + '/*')):
    m = re.search(r'_(ci_tiny|maze_short)_(\d)_(?:mission_v2_(\w+)|-)$', run)
    if not m: continue
    pol = m.group(3) or 'nearest_frontier_1.8'
    for g in sorted(glob.glob(run + '/**/groundtruth_body.tum', recursive=True)):
        t, x, y, h = load(g)
        for T in Ts:
            mv = tu = st = 0.0; yaw = dist = 0.0
            for i in range(1, len(t)):
                if t[i] > T: break
                dt = t[i] - t[i - 1]
                d = math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
                dh = abs((h[i] - h[i - 1] + math.pi) % (2 * math.pi) - math.pi)
                yaw += dh; dist += d
                if d / dt > V: mv += dt
                elif dh / dt > W: tu += dt
                else: st += dt
            tot = mv + tu + st
            R[(m.group(1), pol, T)].append((mv / tot, tu / tot, st / tot, yaw / max(dist, 1e-9), dist))
for k in sorted(R):
    v = R[k]; n = len(v)
    f = [sum(r[j] for r in v) / n for j in range(5)]
    print(f'{k[0]:10s} {k[1]:21s} T<={k[2]:5.0f}s  move {f[0]:.2f}  turn {f[1]:.2f}  still {f[2]:.2f}  yaw/m {f[3]:.2f} rad  m/agent {f[4]:.1f}  (n={n})')
