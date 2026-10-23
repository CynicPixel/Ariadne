"""Depth seeding of new Gaussians at a keyframe (RECON_6 §2, D190/D193): the live path's only growth.

  candidates  pixels with valid depth (finite Z, sigma_Z <= max_sigma, Z <= max_depth -- RECON_1.7's fusion gates)
              and a defined training image, where the submap's current render is not yet opaque (alpha < 0.5:
              the Gaussian-SLAM alpha mask; Yugay et al. arXiv 2312.10070)
  sample      a fraction of them, seeded (RTG-SLAM samples 5 % of new pixels; Peng et al. arXiv 2404.19706) --
              [ASSUMPTION] that RTG's RGB-D fraction transfers to half-res stereo depth
  radius      dropped if an existing Gaussian of the submap lies within rho (Gaussian-SLAM's rho = 0.01 m), and
              deduplicated among themselves on the same rho grid
  init        mean = back-projected point in the anchor frame; isotropic scale = the pixel footprint Z / f (the
              back-projection of one spool pixel); identity rotation; opacity `init_opacity`; colour = the
              training image at that pixel

Pure numpy (scipy KD-tree), deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class SeedParams:
    alpha_threshold: float = 0.5    # Gaussian-SLAM
    sample_fraction: float = 0.05   # RTG-SLAM
    rho_m: float = 0.01             # Gaussian-SLAM
    max_depth_m: float = 6.0        # configs/recon/recon_v1.yaml service.max_depth_m
    max_sigma_m: float = 0.05       # configs/recon/recon_v1.yaml service.max_sigma_m


@dataclass(frozen=True)
class Seeds:
    means: np.ndarray    # (M, 3) float32, anchor frame
    scales: np.ndarray   # (M,)   float32, metres (isotropic)
    colors: np.ndarray   # (M, C) float32, the training image's value(s) at the pixel
    pixels: np.ndarray   # (M, 2) int32 (row, col), provenance


def seed(Z: np.ndarray, sigma: np.ndarray, image: np.ndarray, K: np.ndarray, T_anchor_C: np.ndarray,
         alpha: np.ndarray | None, existing_means: np.ndarray | None, p: SeedParams, rng: np.random.Generator) -> Seeds:
    """`image` is HxW (albedo proxy, NaN = undefined) or HxWx3; `alpha` the current render's opacity (None = empty
    submap); `existing_means` the submap's Gaussians in the anchor frame."""
    h, w = Z.shape
    img = image.reshape(h, w, -1).astype(np.float32)
    ok = np.isfinite(Z) & np.isfinite(sigma) & (sigma <= p.max_sigma_m) & (Z > 0) & (Z <= p.max_depth_m)
    ok &= np.all(np.isfinite(img), axis=2)
    if alpha is not None:
        ok &= alpha < p.alpha_threshold
    rows, cols = np.nonzero(ok)
    n = len(rows)
    if n == 0:
        return _empty(img.shape[2])
    take = rng.choice(n, size=max(1, int(round(p.sample_fraction * n))), replace=False)
    take.sort()
    r, c = rows[take], cols[take]
    z = Z[r, c].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X_C = np.stack([(c - cx) / fx * z, (r - cy) / fy * z, z], axis=1)
    X_A = X_C @ T_anchor_C[:3, :3].T + T_anchor_C[:3, 3]
    keep = np.ones(len(X_A), bool)
    if existing_means is not None and len(existing_means):
        d, _ = cKDTree(existing_means).query(X_A, k=1)
        keep &= d > p.rho_m
    # Deduplicate the new points among themselves: first one per rho cell, in pixel order.
    cell = np.floor(X_A / p.rho_m).astype(np.int64)
    _, first = np.unique(cell, axis=0, return_index=True)
    uniq = np.zeros(len(X_A), bool)
    uniq[first] = True
    keep &= uniq
    return Seeds(means=X_A[keep].astype(np.float32), scales=(z[keep] / fx).astype(np.float32),
                 colors=img[r[keep], c[keep]], pixels=np.stack([r[keep], c[keep]], 1).astype(np.int32))


def _empty(ch: int) -> Seeds:
    return Seeds(np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros((0, ch), np.float32),
                 np.zeros((0, 2), np.int32))
