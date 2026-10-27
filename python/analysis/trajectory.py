"""Trajectory metrics for the evaluation harness (design evaluation-harness.md), matching the C++
`runtime/metrics/trajectory_metrics.hpp` definitions (D31):

- association: estimate and GT samples with the SAME timestamp (integer ns; the sim's
  images sit on the 5 ms GT grid, so association is exact);
- ATE: one SE(3) Umeyama alignment of the estimate positions onto GT positions
  (Umeyama, PAMI 1991; no scale: stereo is metric); translation = position residuals,
  rotation = angle of R_gt^T R_align R_est per pose;
- RPE: pose pairs (i, i + delta) for EVERY i (evo `all_pairs=True`, delta in frames;
  delta 20 = 1 s at 20 Hz); error E = (G_i^-1 G_j)^-1 (P_i^-1 P_j); translation |t_E|,
  rotation angle(R_E).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(t_ns int64 (N,), poses (N, 4, 4)) from a TUM file (t x y z qx qy qz qw)."""
    rows = np.array([[float(v) for v in l.split()] for l in Path(path).read_text().splitlines()
                     if l.strip() and not l.startswith("#")])
    t = np.round(rows[:, 0] * 1e9).astype(np.int64)
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, 3] = rows[:, 1:4]
    T[:, :3, :3] = quat_xyzw_to_R(rows[:, 4:8])
    return t, T


def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def associate(t_est, T_est, t_gt, T_gt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact-timestamp association; returns (t, est poses, gt poses) in estimate order."""
    idx = {int(t): i for i, t in enumerate(t_gt)}
    keep = [(i, idx[int(t)]) for i, t in enumerate(t_est) if int(t) in idx]
    if not keep:
        raise ValueError("no estimate/GT timestamps coincide")
    ie, ig = np.array(keep).T
    return t_est[ie], T_est[ie], T_gt[ig]


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 T minimizing sum |dst - (R src + t)|^2 (Umeyama 1991, scale fixed to 1)."""
    ms, md = src.mean(0), dst.mean(0)
    C = (dst - md).T @ (src - ms) / len(src)
    U, _, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, md - R @ ms
    return T


def rot_angle(R: np.ndarray) -> np.ndarray:
    """Rotation angle, numerically stable at small angles: atan2(|vee(R - R^T)| / 2, (tr R - 1) / 2).
    (arccos((tr - 1) / 2) loses ~1e-8 rad near 0, a 1e-6 relative error on mrad-level ATE.)"""
    tr = np.trace(R, axis1=-2, axis2=-1)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return np.arctan2(np.linalg.norm(v, axis=-1) / 2, (tr - 1) / 2)


def ate(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    A = umeyama_se3(T_est[:, :3, 3], T_gt[:, :3, 3])
    p = T_est[:, :3, 3] @ A[:3, :3].T + A[:3, 3]
    e = np.linalg.norm(p - T_gt[:, :3, 3], axis=1)
    Ra = A[:3, :3] @ T_est[:, :3, :3]
    r = rot_angle(np.transpose(T_gt[:, :3, :3], (0, 2, 1)) @ Ra)
    return {"n": int(len(e)), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "max_m": float(e.max()),
            "rotation_rmse_rad": float(np.sqrt(np.mean(r ** 2))), "errors_m": e, "alignment": A}


def rpe_errors(T_est: np.ndarray, T_gt: np.ndarray, delta: int) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv
    i = np.arange(len(T_est) - delta)
    E = inv(inv(T_gt[i]) @ T_gt[i + delta]) @ (inv(T_est[i]) @ T_est[i + delta])
    return np.linalg.norm(E[:, :3, 3], axis=1), rot_angle(E[:, :3, :3])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))
