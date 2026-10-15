"""Gaussian initialization from a point cloud.

Transcribed from gsplat v1.4.0 `examples/simple_trainer.py::create_splats_with_optimizers`
and `examples/utils.py::{knn, rgb_to_sh}` (tag v1.4.0, fetched 2026-10-15), not
from memory:

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)     # 3 nearest, self excluded
    scales    = log(sqrt(dist2_avg) * init_scale).repeat(3)    # isotropic, log space
    quats     = torch.rand((N, 4))                            # uniform [0, 1), unnormalized
    opacities = logit(full(N, init_opacity))                   # init_opacity 0.1
    sh0       = rgb_to_sh(rgb) = (rgb - 0.5) / 0.28209479177387814;  shN = 0 (sh_degree 3)

Only the random quaternions differ in mechanism: numpy's seeded generator instead
of torch's global RNG, so the CPU stage is reproducible byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814
SH_DEGREE = 3


@dataclass(frozen=True)
class Gaussians:
    means: np.ndarray      # (N, 3) float32
    quats: np.ndarray      # (N, 4) float32, wxyz as gsplat reads them, unnormalized
    scales: np.ndarray     # (N, 3) float32, log
    opacities: np.ndarray  # (N,)   float32, logit
    sh0: np.ndarray        # (N, 1, 3) float32
    shN: np.ndarray        # (N, (SH_DEGREE+1)^2 - 1, 3) float32

    def __len__(self) -> int:
        return len(self.means)


def init_from_points(xyz: np.ndarray, rgb_u8: np.ndarray, init_opacity: float, seed: int,
                     init_scale: float = 1.0) -> Gaussians:
    if len(xyz) < 4:
        raise ValueError("need at least 4 points (3 neighbours each)")
    if not 0 < init_opacity < 1:
        raise ValueError("init_opacity must be in (0, 1)")
    d, _ = cKDTree(xyz).query(xyz, k=4)
    dist2_avg = (d[:, 1:] ** 2).mean(axis=1)
    if np.any(dist2_avg <= 0):
        raise ValueError("coincident points: a zero neighbour distance has no log scale")
    scales = np.repeat(np.log(np.sqrt(dist2_avg) * init_scale)[:, None], 3, axis=1)
    n = len(xyz)
    quats = np.random.default_rng(seed).random((n, 4))
    opac = np.full(n, np.log(init_opacity / (1 - init_opacity)))
    sh0 = ((rgb_u8.astype(np.float64) / 255.0 - 0.5) / SH_C0)[:, None, :]
    shN = np.zeros((n, (SH_DEGREE + 1) ** 2 - 1, 3))
    f = np.float32
    return Gaussians(xyz.astype(f), quats.astype(f), scales.astype(f), opac.astype(f), sh0.astype(f), shN.astype(f))
