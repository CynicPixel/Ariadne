"""Hilti 2023 control-point scoring for 5.6 (D141): sparse 3-DoF position error at the ground control points (GCPs).

  python -m analysis.hilti_score ESTIMATE_TUM GT_TXT [--gcp GCP_CSV --calib CALIB_DIR] [--out results.yaml]

ESTIMATE_TUM: the run's estimate_body.tum (body = the IMU, the frame Hilti evaluates in; RA-L 2024 §V-B).
GT_TXT: ground_truth/site2_robot_3.txt, TUM rows "t x y z 0 0 0 1": each GCP's surveyed position at the time the
robot observed it (the quaternion is a placeholder; the header line misnames the columns).
With --gcp (to_asl's gcp.csv: the GCP as the robot's LiDAR measured it, in that message's frame_id) the predicted
GCP is T_W_I(t) T_I_F p_F, the paper's construction; without it the IMU position itself is compared (labelled so).

Alignment: SE(3) Umeyama (no scale; stereo is metric) of the predicted points onto the GT points, which uses 6 of the
3N numbers it is then scored on -- optimistic for N = 5. So two numbers are reported:
  fit : align on all reached points, score each;
  loo : leave-one-out, each point scored under the alignment fitted to the others (needs >= 4 reached).
Score per point (Hilti 2023 README, "How are trajectory scores calculated?"): <0.5 cm 20, <1 cm 10, <3 cm 6, <6 cm 5,
<10 cm 3, <40 cm 1, else 0; the sequence score is 100 * sum / (20 N) over ALL N GCPs, so a point the estimate never
reached (outside its time span, or in a gap > --max-gap-s) scores 0 and is listed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.trajectory import load_tum, umeyama_se3

THRESH_POINTS = [(0.005, 20), (0.01, 10), (0.03, 6), (0.06, 5), (0.10, 3), (0.40, 1)]
IMU_FRAMES = {"imu", "imu0", "xsens_imu_link"}


def points_for(err_m: float) -> int:
    for th, pts in THRESH_POINTS:
        if err_m < th:
            return pts
    return 0


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-12:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def _expm(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def interpolate(t_ns, T, t_q, max_gap_s):
    """Pose at t_q: linear in translation, geodesic in rotation, between the bracketing samples; None if t_q is outside
    the estimate or the bracket is wider than max_gap_s (a reset or a lost stretch is not interpolated across)."""
    i = int(np.searchsorted(t_ns, t_q))
    if i == 0 or i == len(t_ns):
        return None
    t0, t1 = t_ns[i - 1], t_ns[i]
    if (t1 - t0) * 1e-9 > max_gap_s:
        return None
    s = (t_q - t0) / (t1 - t0)
    out = np.eye(4)
    out[:3, 3] = (1 - s) * T[i - 1][:3, 3] + s * T[i][:3, 3]
    out[:3, :3] = T[i - 1][:3, :3] @ _expm(s * _rotvec(T[i - 1][:3, :3].T @ T[i][:3, :3]))
    return out


def load_gt(path):
    rows = [[float(v) for v in l.split()] for l in Path(path).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    if any(len(r) != 8 for r in rows):
        raise SystemExit(f"{path}: expected TUM rows (t x y z qx qy qz qw)")
    return np.array([round(r[0] * 1e9) for r in rows], np.int64), np.array([r[1:4] for r in rows])


def load_gcp(path, calib_dir, gt_t, tol_s=0.05):
    """Per GT row: (frame_id, T_I_F @ p_F) from the robot's own GCP measurement nearest in time."""
    rows = [l.split(",") for l in Path(path).read_text().splitlines() if l and not l.startswith("#")]

    def T_I_L():   # LiDAR -> IMU from Kalibr's T_lidar_imu, read only when a GCP is in the LiDAR frame
        if calib_dir is None:
            raise SystemExit("GCPs in the LiDAR frame need --calib (calibration_lidar.yaml)")
        lidar = yaml.safe_load((Path(calib_dir) / "calibration_lidar.yaml").read_text())
        return np.linalg.inv(np.array(next(iter(lidar.values()))["T_here_imu0"], float))

    out = []
    for t in gt_t:
        k = min(range(len(rows)), key=lambda j: abs(int(rows[j][0]) - t))
        if abs(int(rows[k][0]) - t) * 1e-9 > tol_s:
            raise SystemExit(f"no GCP measurement within {tol_s}s of GT stamp {t}")
        fid, p = rows[k][1], np.array([float(v) for v in rows[k][3:6]])
        if fid in IMU_FRAMES:
            out.append((fid, p))
        elif fid.startswith("rslidar"):
            out.append((fid, (T_I_L() @ np.append(p, 1.0))[:3]))
        else:
            raise SystemExit(f"GCP frame '{fid}' has no known transform to the IMU")
    return out


def score(est_tum, gt_txt, gcp_csv=None, calib_dir=None, max_gap_s=0.5):
    t_est, T_est = load_tum(est_tum)
    gt_t, gt_p = load_gt(gt_txt)
    offs = load_gcp(gcp_csv, calib_dir, gt_t) if gcp_csv else [("imu_origin", np.zeros(3))] * len(gt_t)
    pred, reached = [], []
    for k, t in enumerate(gt_t):
        T = interpolate(t_est, T_est, t, max_gap_s)
        if T is not None:
            pred.append((T @ np.append(offs[k][1], 1.0))[:3])
            reached.append(k)
    n = len(gt_t)
    res = {"n_gcp": n, "reached": reached, "unreached": [k for k in range(n) if k not in reached],
           "predicted_point": "T_W_I(t) T_I_F p_F (robot-measured GCP)" if gcp_csv else "IMU position (no gcp.csv)",
           "gcp_frames": sorted({o[0] for o in offs})}
    P, G = np.array(pred), gt_p[reached]
    for mode in ("fit", "loo"):
        errs = {}
        if mode == "fit" and len(reached) >= 3:
            A = umeyama_se3(P, G)
            for j, k in enumerate(reached):
                errs[k] = float(np.linalg.norm(A[:3, :3] @ P[j] + A[:3, 3] - G[j]))
        if mode == "loo" and len(reached) >= 4:
            for j, k in enumerate(reached):
                m = [i for i in range(len(reached)) if i != j]
                A = umeyama_se3(P[m], G[m])
                errs[k] = float(np.linalg.norm(A[:3, :3] @ P[j] + A[:3, 3] - G[j]))
        if not errs:
            res[mode] = None
            continue
        pts = {k: points_for(e) for k, e in errs.items()}
        res[mode] = {"error_m": {int(k): round(e, 4) for k, e in errs.items()},
                     "points": {int(k): p for k, p in pts.items()},
                     "score_0_100": round(100.0 * sum(pts.values()) / (20 * n), 2),
                     "rms_m": round(float(np.sqrt(np.mean(np.square(list(errs.values()))))), 4)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.hilti_score")
    ap.add_argument("estimate"); ap.add_argument("gt")
    ap.add_argument("--gcp"); ap.add_argument("--calib"); ap.add_argument("--max-gap-s", type=float, default=0.5)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    r = score(a.estimate, a.gt, a.gcp, a.calib, a.max_gap_s)
    text = yaml.safe_dump(r, sort_keys=False)
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    sys.exit(main())
