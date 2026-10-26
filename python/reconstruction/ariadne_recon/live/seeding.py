"""Depth seeding of new Gaussians at a keyframe (reconstruction.md §2, D190/D193/D198): the live path's only growth.

Gaussian-SLAM's recipe (Yugay et al., arXiv 2312.10070, §3.2 and the implementation details; the TUM-RGBD values,
the real-sensor setting closest to stereo depth), transcribed rather than mixed with other systems (D198):

  first keyframe of a submap   M_u points sampled uniformly from the keyframe's valid depth, plus M_c points from
                               high colour-gradient regions. TUM: M_u = 100,000 and M_c = 50,000 of 640x480 =
                               307,200 pixels, used here as FRACTIONS of the valid pixels (0.326 / 0.163)
                               [ASSUMPTION: the fraction, not the count, transfers to our spool resolution]
  later keyframes              M_k points sampled uniformly where the current render's alpha < alpha_n = 0.6.
                               M_k's value is in the paper's supplement, not its main text [UNVERIFIED]; the uniform
                               fraction of the first keyframe is used
  radius                       a candidate with an existing Gaussian of the submap within rho = 0.01 m is dropped
  init                         scale = the nearest-neighbour distance within the active submap (new and existing);
                               opacity 0.5 (the trainer's init_opacity); colour = the training image at the pixel

"High colour-gradient regions" is not defined further in the main text; here: pixels whose Sobel gradient magnitude of
the training image is above that image's median over valid pixels, sampled uniformly [ASSUMPTION].

Valid depth also passes reconstruction.md#first-reconstruction-loop's fusion gates (sigma_Z <= max_sigma, Z <= max_depth). Pure numpy (scipy KD-tree),
deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import sobel
from scipy.spatial import cKDTree

TUM_PIXELS = 640 * 480


@dataclass(frozen=True)
class SeedParams:
    alpha_threshold: float = 0.6                     # Gaussian-SLAM alpha_n
    uniform_fraction: float = 100_000 / TUM_PIXELS   # Gaussian-SLAM M_u (TUM)
    gradient_fraction: float = 50_000 / TUM_PIXELS   # Gaussian-SLAM M_c
    rho_m: float = 0.01                              # Gaussian-SLAM
    max_depth_m: float = 6.0                         # configs/recon/recon_v1.yaml service.max_depth_m
    max_sigma_m: float = 0.05                        # configs/recon/recon_v1.yaml service.max_sigma_m


@dataclass(frozen=True)
class Seeds:
    means: np.ndarray    # (M, 3) float32, anchor frame
    scales: np.ndarray   # (M,)   float32, metres (isotropic)
    colors: np.ndarray   # (M, C) float32, the training image's value(s) at the pixel
    pixels: np.ndarray   # (M, 2) int32 (row, col), provenance


def seed(Z: np.ndarray, sigma: np.ndarray, image: np.ndarray, K: np.ndarray, T_anchor_C: np.ndarray,
         alpha: np.ndarray | None, existing_means: np.ndarray | None, p: SeedParams, rng: np.random.Generator,
         first: bool | None = None) -> Seeds:
    """`image` HxW (albedo proxy, NaN = undefined) or HxWx3; `alpha` the current render's opacity; `existing_means`
    the submap's Gaussians in the anchor frame. `first` (default: the submap has no Gaussians yet) selects the
    first-keyframe sampling."""
    h, w = Z.shape
    img = image.reshape(h, w, -1).astype(np.float32)
    if first is None:
        first = existing_means is None or not len(existing_means)
    ok = np.isfinite(Z) & np.isfinite(sigma) & (sigma <= p.max_sigma_m) & (Z > 0) & (Z <= p.max_depth_m)
    ok &= np.all(np.isfinite(img), axis=2)
    if not first and alpha is not None:
        ok &= alpha < p.alpha_threshold
    flat = np.flatnonzero(ok)
    if len(flat) == 0:
        return _empty(img.shape[2])
    pick = [rng.choice(flat, size=max(1, int(round(p.uniform_fraction * len(flat)))), replace=False)]
    if first and p.gradient_fraction > 0:
        g = np.nan_to_num(img).mean(axis=2)
        mag = np.hypot(sobel(g, axis=0), sobel(g, axis=1)).ravel()
        hi = flat[mag[flat] > np.median(mag[flat])]
        if len(hi):
            pick.append(rng.choice(hi, size=min(len(hi), max(1, int(round(p.gradient_fraction * len(flat))))),
                                   replace=False))
    idx = np.unique(np.concatenate(pick))          # sorted, no pixel twice
    r, c = np.divmod(idx, w)
    z = Z[r, c].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X_C = np.stack([(c - cx) / fx * z, (r - cy) / fy * z, z], axis=1)
    X_A = X_C @ T_anchor_C[:3, :3].T + T_anchor_C[:3, 3]
    keep = np.ones(len(X_A), bool)
    if existing_means is not None and len(existing_means):
        d, _ = cKDTree(existing_means).query(X_A, k=1)
        keep &= d > p.rho_m
    X = X_A[keep]
    allp = X if existing_means is None or not len(existing_means) else np.concatenate([existing_means, X])
    if len(allp) >= 2:
        d, _ = cKDTree(allp).query(X, k=2)                 # column 0 is the point itself
        scale = np.maximum(d[:, 1], 1e-4)
    else:
        scale = z[keep] / fx                               # a lone point: the pixel footprint
    return Seeds(means=X.astype(np.float32), scales=scale.astype(np.float32),
                 colors=img[r[keep], c[keep]], pixels=np.stack([r[keep], c[keep]], 1).astype(np.int32))


def _empty(ch: int) -> Seeds:
    return Seeds(np.zeros((0, 3), np.float32), np.zeros(0, np.float32), np.zeros((0, ch), np.float32),
                 np.zeros((0, 2), np.int32))
