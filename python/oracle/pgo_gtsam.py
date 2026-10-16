"""GTSAM oracle for Ariadne's pose-graph solver (arch §11.6 / §37.5; design POSE_GRAPH_2.md §5). TEST ONLY.

  python -m oracle.pgo_gtsam --out DIR [--random 100] [--g2o NAME=PATH ...]

For every case writes DIR/<case>.graph (Ariadne's plain format, below) holding the graph, the SHARED initial
estimate (dead reckoning along the odometry chain for random graphs, the file's vertices for g2o graphs; both
solvers start from exactly this point) and GTSAM's LM optimum plus
its final error. The C++ test (tests/unit/test_pgo_oracle.cpp) re-solves each case from the same initial estimate
and compares.

Conventions, converted HERE and only here:
  * Ariadne's residual r = Log(Z^-1 Xi^-1 Xj), tangent [rho; phi] (translation first); GTSAM's BetweenFactorPose3
    error is localCoordinates(Z, Xi^-1 Xj) with tangent [omega; v]. The script ASSERTS they are the same vector
    (permuted) on random poses before solving anything: if this wheel used a non-Expmap Pose3 chart the two
    solvers would minimise different costs and "agreement" would be meaningless.
  * Information matrices are written in Ariadne's order and permuted for GTSAM: I_g = P I P^T.
  * Gauge: the anchor is held fixed (GTSAM: NonlinearEqualityPose3; Ariadne: not a variable).
  * GTSAM's graph.error() is 1/2 sum r^T I r, the same scale as Ariadne's energy.

File format (text, one record per line):
  ariadne-pgo 1
  ANCHOR id
  NODE id  tx ty tz qw qx qy qz              initial estimate (T_W_C)
  EDGE i j tx ty tz qw qx qy qz  I(21)        measurement Z_ij, upper triangle of I, row-major, [rho; phi]
  OPT  id  tx ty tz qw qx qy qz              GTSAM's optimum
  INIT_ERROR e / FINAL_ERROR e / ITERATIONS k
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gtsam
import numpy as np

P = np.zeros((6, 6))   # Ariadne [rho; phi] -> GTSAM [omega; v]
P[0:3, 3:6] = np.eye(3)
P[3:6, 0:3] = np.eye(3)


def pose(t, q_wxyz) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3.Quaternion(*q_wxyz), gtsam.Point3(*t))


def pose_fields(T: gtsam.Pose3) -> list[float]:
    q = T.rotation().toQuaternion()
    return [*T.translation(), q.w(), q.x(), q.y(), q.z()]


def random_pose(rng, rot, trans) -> gtsam.Pose3:
    return gtsam.Pose3.Expmap(np.concatenate([rng.normal(0, rot, 3), rng.normal(0, trans, 3)]))


def check_conventions(rng) -> None:
    """BetweenFactor error == permuted Ariadne residual Log(Z^-1 Xi^-1 Xj), on random poses."""
    for _ in range(20):
        Xi, Xj, Z = random_pose(rng, 1, 3), random_pose(rng, 1, 3), random_pose(rng, 1, 3)
        f = gtsam.BetweenFactorPose3(0, 1, Z, gtsam.noiseModel.Unit.Create(6))
        v = gtsam.Values()
        v.insert(0, Xi)
        v.insert(1, Xj)
        err = f.unwhitenedError(v)
        logmap = gtsam.Pose3.Logmap(Z.between(Xi.between(Xj)))
        if not np.allclose(err, logmap, atol=1e-9):
            raise SystemExit(f"GTSAM Pose3 chart is not Expmap/Logmap: {err} vs {logmap}")


def correlated_cov(rng, rot_sigma, trans_sigma):
    """A full, correlated, anisotropic 6x6 covariance in Ariadne's [rho; phi] order: random rotation of the axes,
    per-axis scales spanning ~33x (information ~1000x: the corridor case, one direction far weaker than the others),
    cross terms between translation and rotation. A permutation or ordering bug cannot hide in this, unlike a diagonal.
    The anisotropy lives BELOW the stated base sigma (scales 0.03-1x), so dead reckoning stays near the basin: an
    oracle compares solvers from one start, it does not test initialisation (that is 2.5's chordal step)."""
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    scales = np.array([trans_sigma] * 3 + [rot_sigma] * 3) * np.exp(rng.uniform(np.log(0.03), np.log(1.0), 6))
    return Q @ np.diag(scales**2) @ Q.T


def random_graph(rng, n, loops, rot_sigma, trans_sigma, correlated=False):
    truth = [random_pose(rng, 1.0, 3.0)]
    for _ in range(n - 1):
        truth.append(truth[-1].compose(random_pose(rng, 0.3, 1.0)))
    edges = []

    def add(a, b):
        Z = truth[a].between(truth[b])
        if correlated:
            cov = correlated_cov(rng, rot_sigma, trans_sigma)          # Ariadne order
            noise_ar = rng.multivariate_normal(np.zeros(6), cov)
            info = np.linalg.inv(cov)
        else:
            noise_ar = np.concatenate([rng.normal(0, trans_sigma, 3), rng.normal(0, rot_sigma, 3)])
            info = np.diag([1 / trans_sigma**2] * 3 + [1 / rot_sigma**2] * 3)
        noise = P @ noise_ar                                            # -> GTSAM [omega; v] for Expmap
        edges.append((a, b, Z.compose(gtsam.Pose3.Expmap(noise)), 0.5 * (info + info.T)))

    for i in range(n - 1):
        add(i, i + 1)
    for _ in range(loops):
        a, b = rng.integers(0, n, 2)
        if a != b:
            add(int(a), int(b))
    return n, edges


def read_g2o(path: Path):
    """3-D g2o (VERTEX_SE3:QUAT / EDGE_SE3:QUAT). The 21 information entries are taken as written, in [t; r] order,
    as the information of Ariadne's [rho; phi] residual: BOTH solvers then minimise the same cost."""
    ids, edges, init = {}, [], gtsam.Values()
    for line in path.read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "VERTEX_SE3:QUAT":
            k = ids.setdefault(int(f[1]), len(ids))
            x, y, z, qx, qy, qz, qw = map(float, f[2:9])
            init.insert(k, pose((x, y, z), (qw, qx, qy, qz)))
        elif f[0] == "EDGE_SE3:QUAT":
            a, b = int(f[1]), int(f[2])
            x, y, z, qx, qy, qz, qw = map(float, f[3:10])
            u = list(map(float, f[10:31]))
            I = np.zeros((6, 6))
            k = 0
            for r in range(6):
                for c in range(r, 6):
                    I[r, c] = I[c, r] = u[k]
                    k += 1
            ids.setdefault(a, len(ids))
            ids.setdefault(b, len(ids))
            edges.append((ids[a], ids[b], pose((x, y, z), (qw, qx, qy, qz)), I))
    if init.size() != len(ids):
        raise SystemExit(f"{path}: an edge references a vertex that has no VERTEX_SE3:QUAT line")
    return len(ids), edges, init


def dead_reckoning(n, edges) -> gtsam.Values:
    """Node 0 at the identity, node i+1 = node i * Z(i, i+1) from the chain edges (the first i -> i+1 edge)."""
    chain = {}
    for a, b, Z, _ in edges:
        if b == a + 1 and a not in chain:
            chain[a] = Z
    v = gtsam.Values()
    T = gtsam.Pose3()
    v.insert(0, T)
    for i in range(n - 1):
        T = T.compose(chain[i])
        v.insert(i + 1, T)
    return v


def solve(n, edges, init):
    graph = gtsam.NonlinearFactorGraph()
    for a, b, Z, I in edges:
        graph.add(gtsam.BetweenFactorPose3(a, b, Z, gtsam.noiseModel.Gaussian.Information(P @ I @ P.T)))
    graph.add(gtsam.NonlinearEqualityPose3(0, init.atPose3(0)))   # the gauge: node 0 held where it starts
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-14)
    params.setAbsoluteErrorTol(1e-14)
    params.setMaxIterations(1000)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, init, params)
    result = opt.optimize()
    return init, result, graph.error(init), graph.error(result), opt.iterations()


def write_case(path: Path, n, edges, init, result, e0, e1, iters) -> None:
    out = ["ariadne-pgo 1", "ANCHOR 0"]
    for i in range(n):
        out.append("NODE %d " % i + " ".join("%.17g" % v for v in pose_fields(init.atPose3(i))))
    for a, b, Z, I in edges:
        up = [I[r, c] for r in range(6) for c in range(r, 6)]
        out.append("EDGE %d %d " % (a, b) + " ".join("%.17g" % v for v in pose_fields(Z) + up))
    for i in range(n):
        out.append("OPT %d " % i + " ".join("%.17g" % v for v in pose_fields(result.atPose3(i))))
    out += ["INIT_ERROR %.17g" % e0, "FINAL_ERROR %.17g" % e1, "ITERATIONS %d" % iters]
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--g2o", action="append", default=[], help="NAME=PATH")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    check_conventions(rng)
    for k in range(a.random):
        n = int(rng.integers(10, 201))
        loops = int(rng.integers(0, n // 2 + 1))
        rot, trans = float(rng.uniform(0.005, 0.1)), float(rng.uniform(0.005, 0.2))
        nn, edges = random_graph(rng, n, loops, rot, trans, correlated=(k % 2 == 1))   # odd cases: full information
        res = solve(nn, edges, dead_reckoning(nn, edges))
        write_case(out / f"random_{k:03d}.graph", nn, edges, *res)
    for spec in a.g2o:
        name, path = spec.split("=", 1)
        nn, edges, init = read_g2o(Path(path))
        res = solve(nn, edges, init)
        write_case(out / f"{name}.graph", nn, edges, *res)
        print(f"{name}: {nn} nodes, {len(edges)} edges, error {res[2]:.6g} -> {res[3]:.6g} in {res[4]} its")
    print(f"wrote {a.random} random cases + {len(a.g2o)} g2o cases to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
