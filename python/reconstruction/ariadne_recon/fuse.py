"""Fused stereo point cloud (arch §18.1 "local point cloud"; design RECON_1.7).

Every keyframe's depth is back-projected through its rectified pinhole K,
moved into agent 0's odometry frame (agent 1 through the 1.6 merge T_O0_O1),
and voxelized: one point per voxel, the FIRST in bundle order (deterministic, and
no averaging across views whose poses disagree). Pixels without depth, beyond
`max_depth_m`, or with sigma_Z above `max_sigma_m` are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spool import Spool, SpoolError


@dataclass(frozen=True)
class FuseParams:
    voxel_m: float
    max_depth_m: float
    max_sigma_m: float
    max_points: int


@dataclass(frozen=True)
class Cloud:
    xyz: np.ndarray      # (N, 3) float64, agent 0 odometry frame
    rgb: np.ndarray      # (N, 3) uint8
    source: np.ndarray   # (N,) int64 bundle index each point came from
    stats: dict


def backproject(Z: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame points for every pixel with finite Z; returns (points (M,3), flat pixel indices (M,))."""
    h, w = Z.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(Z).ravel()
    z = Z.ravel()[ok].astype(np.float64)
    x = (u.ravel()[ok] - K[0, 2]) / K[0, 0] * z
    y = (v.ravel()[ok] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1), np.flatnonzero(ok)


def _first_per_voxel(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of the first point (in input order) of each occupied voxel, in input order."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def fuse(spool: Spool, p: FuseParams) -> Cloud:
    if not (p.voxel_m > 0 and p.max_depth_m > 0 and p.max_sigma_m > 0 and p.max_points > 0):
        raise SpoolError("fuse: parameters must be positive")
    parts_xyz, parts_rgb, parts_src = [], [], []
    n_pixels = n_depth = n_kept = 0
    for i, b in enumerate(spool.bundles):
        if b.role != "keyframe":
            continue
        Z, S = b.depth()
        Zf = np.where((Z <= p.max_depth_m) & (S <= p.max_sigma_m), Z, np.nan)   # NaN compares False: stays NaN
        n_pixels += Z.size
        n_depth += int(np.isfinite(Z).sum())
        pc, flat = backproject(Zf, b.K)
        n_kept += len(pc)
        if not len(pc):
            continue
        T = spool.T_O0_C(b)
        xyz = pc @ T[:3, :3].T + T[:3, 3]
        rgb = b.rgb().reshape(-1, 3)[flat]
        keep = _first_per_voxel(xyz, p.voxel_m)          # per keyframe first: bounds memory
        parts_xyz.append(xyz[keep])
        parts_rgb.append(rgb[keep])
        parts_src.append(np.full(len(keep), i, dtype=np.int64))
    if not parts_xyz:
        raise SpoolError("fuse: no depth survived the gates")
    xyz, rgb, src = np.concatenate(parts_xyz), np.concatenate(parts_rgb), np.concatenate(parts_src)
    keep = _first_per_voxel(xyz, p.voxel_m)              # then globally, still first-in-bundle-order
    xyz, rgb, src = xyz[keep], rgb[keep], src[keep]
    n_voxels = len(xyz)
    if n_voxels > p.max_points:                          # deterministic stride subsample
        sel = (np.arange(p.max_points, dtype=np.int64) * n_voxels) // p.max_points
        xyz, rgb, src = xyz[sel], rgb[sel], src[sel]
    stats = {"keyframes": len(spool.keyframes()), "pixels": n_pixels, "pixels_with_depth": n_depth,
             "pixels_after_gates": n_kept, "voxels": n_voxels, "points": len(xyz)}
    return Cloud(xyz=xyz, rgb=rgb, source=src, stats=stats)
