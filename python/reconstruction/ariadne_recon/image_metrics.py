"""Held-out image metrics computed locally from the returned renders (§30.2:
heldout_psnr_db, heldout_ssim; LPIPS needs torch + network weights and is
computed remotely, D44).

SSIM is the standard Wang et al. 2004 form: 11x11 Gaussian window, sigma 1.5,
K1 0.01, K2 0.03, data range 1, per channel on [0, 1] images, mean over valid
(fully covered) window positions, then over channels.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def _gauss_filter_valid(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    r = len(w) // 2
    y = correlate1d(correlate1d(x, w, axis=0, mode="constant"), w, axis=1, mode="constant")
    return y[r:-r, r:-r]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("ssim: HxWxC images of equal shape")
    t = np.arange(11) - 5
    w = np.exp(-(t ** 2) / (2 * 1.5 ** 2))
    w /= w.sum()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        mx, my = _gauss_filter_valid(x, w), _gauss_filter_valid(y, w)
        sxx = _gauss_filter_valid(x * x, w) - mx * mx
        syy = _gauss_filter_valid(y * y, w) - my * my
        sxy = _gauss_filter_valid(x * y, w) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(s.mean())
    return float(np.mean(vals))
