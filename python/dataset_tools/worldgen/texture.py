"""Procedural surface albedo textures (design §3), deterministic per (seed, surface_id).

Layers, composited in luminance then tinted by the surface style:
  base     1/f value noise, 7 octaves (periods 256..4 px): concrete mottling
  pits     hard-edged discs, radii 2-14 px (aggregate, pores): blob corners
  stains   thresholded low-frequency noise: irregular hard boundaries (20-80 px)
  cracks   random-walk polylines, 1-2 px: thin line features and junctions
  marks    sparse sharp-cornered paint rectangles: strong corners + place identity
The richness knob tau in [0,1] scales the contrast/density of every layer except the
base (tau=0 leaves only the smooth concrete of §5.5; the §4g render showed that is
too soft for FAST). Nothing here is periodic, so SGBM sees no repeated structure.

Determinism: every random value is a Philox4x32-10 word addressed by
(seed; surface_id, layer, index...), evaluated with numpy integer arithmetic; the
float math is + - * / and comparisons only (IEEE exact per element; no reductions,
no transcendentals), so pixels are identical on every platform and numpy version.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import philox4x32_10_np, unit_from_u32

# Layer ids in the Philox counter; never renumber.
L_BASE, L_PITS, L_STAINS, L_CRACKS, L_MARKS, L_STYLE = 1, 2, 3, 4, 5, 6


@dataclass(frozen=True)
class Style:
    """Per-room (or per-corridor-zone) appearance: shared hue, unique noise per surface."""
    tint: tuple[float, float, float]      # multiplies luminance, each in (0, 1.2]
    base_level: float                     # mean luminance of the concrete, 0..1
    base_contrast: float                  # amplitude of the base noise


def _words(seed: int, surface: int, layer: int, a, b=0, c=0):
    """Four uint32 arrays of random words for counters (a, b) at (surface, layer, c)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c0 = (a & 0xFFFFFFFF).astype(np.uint32)
    c1 = (b & 0xFFFFFFFF).astype(np.uint32)
    c2 = np.uint32(surface & 0xFFFFFFFF)
    c3 = np.uint32(((layer & 0xFF) << 24) | (c & 0xFFFFFF))
    return philox4x32_10_np(c0, c1, c2, c3, seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)


def _unit(seed, surface, layer, a, b=0, c=0, word=0):
    return unit_from_u32(_words(seed, surface, layer, a, b, c)[word])


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def value_noise(seed, surface, layer, octave, h, w, period_px) -> np.ndarray:
    """Value noise in [0,1): random values on a lattice of `period_px`, smoothstep-
    interpolated. Lattice coordinates are absolute pixel/period, so the field is a
    function of position only (tiles of one surface would join seamlessly)."""
    ys = np.arange(h, dtype=np.float64) / period_px
    xs = np.arange(w, dtype=np.float64) / period_px
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = _smooth(ys - y0)[:, None]
    fx = _smooth(xs - x0)[None, :]
    gy = np.arange(y0.min(), y0.max() + 2)
    gx = np.arange(x0.min(), x0.max() + 2)
    lat = _unit(seed, surface, layer, gx[None, :], gy[:, None], octave)   # [gy, gx]
    iy = (y0 - gy[0])[:, None]
    ix = (x0 - gx[0])[None, :]
    v00, v01 = lat[iy, ix], lat[iy, ix + 1]
    v10, v11 = lat[iy + 1, ix], lat[iy + 1, ix + 1]
    top = v00 + (v01 - v00) * fx
    bot = v10 + (v11 - v10) * fx
    return top + (bot - top) * fy


def fbm(seed, surface, layer, h, w, periods=(256, 128, 64, 32, 16, 8, 4)) -> np.ndarray:
    """1/f sum: amplitude proportional to period, normalized to [0, 1]."""
    acc = np.zeros((h, w))
    total = 0.0
    for o, p in enumerate(periods):
        amp = p / periods[0]
        acc += amp * value_noise(seed, surface, layer, o, h, w, p)
        total += amp
    return acc / total


def _discs(img, seed, surface, layer, count, rmin, rmax, lo, hi):
    """Stamp `count` hard-edged discs with luminance offsets in [lo, hi]."""
    if count <= 0:
        return
    h, w = img.shape
    idx = np.arange(count)
    u0, u1, u2, u3 = (unit_from_u32(x) for x in _words(seed, surface, layer, idx))
    cy, cx = u0 * h, u1 * w
    r = rmin + (rmax - rmin) * u2 * u2          # small discs are more common
    val = lo + (hi - lo) * u3
    for k in range(count):
        ri = int(np.ceil(r[k]))
        y0, y1 = max(int(cy[k]) - ri, 0), min(int(cy[k]) + ri + 1, h)
        x0, x1 = max(int(cx[k]) - ri, 0), min(int(cx[k]) + ri + 1, w)
        if y0 >= y1 or x0 >= x1:
            continue
        yy = np.arange(y0, y1)[:, None] + 0.5 - cy[k]
        xx = np.arange(x0, x1)[None, :] + 0.5 - cx[k]
        m = yy * yy + xx * xx <= r[k] * r[k]
        img[y0:y1, x0:x1][m] += val[k]


def _cracks(img, seed, surface, count, tau):
    h, w = img.shape
    for k in range(count):
        n_steps = 40 + int(160 * _unit(seed, surface, L_CRACKS, k, 0, 0))
        y = _unit(seed, surface, L_CRACKS, k, 1, 0) * h
        x = _unit(seed, surface, L_CRACKS, k, 2, 0) * w
        # Direction as a unit vector updated by small random turns (no trig).
        dy = _unit(seed, surface, L_CRACKS, k, 3, 0) * 2.0 - 1.0
        dx = _unit(seed, surface, L_CRACKS, k, 4, 0) * 2.0 - 1.0
        steps = np.arange(n_steps)
        t0, t1, t2, _ = (unit_from_u32(x_) for x_ in _words(seed, surface, L_CRACKS, steps, k + 1, 1))
        depth = 0.35 * tau
        for i in range(n_steps):
            dy += (t0[i] - 0.5) * 0.9
            dx += (t1[i] - 0.5) * 0.9
            n = np.sqrt(dy * dy + dx * dx) + 1e-12
            dy, dx = dy / n, dx / n
            y += 2.0 * dy
            x += 2.0 * dx
            iy, ix = int(y), int(x)
            if not (0 <= iy < h and 0 <= ix < w):
                break
            img[iy, ix] -= depth
            if t2[i] < 0.5 and ix + 1 < w:                  # 1-2 px wide
                img[iy, ix + 1] -= depth * 0.7


def _marks(img, seed, surface, count, tau):
    h, w = img.shape
    idx = np.arange(count)
    u = [unit_from_u32(x) for x in _words(seed, surface, L_MARKS, idx)]
    v = [unit_from_u32(x) for x in _words(seed, surface, L_MARKS, idx, 1)]
    for k in range(count):
        mh = int(12 + u[0][k] * 60)
        mw = int(12 + u[1][k] * 90)
        y0 = int(u[2][k] * max(h - mh, 1))
        x0 = int(u[3][k] * max(w - mw, 1))
        level = (v[0][k] - 0.5) * 1.2 * tau                   # dark or light paint
        img[y0:y0 + mh, x0:x0 + mw] += level
        if v[1][k] < 0.4 and mh > 20 and mw > 20:           # hollow frame variant
            img[y0 + 4:y0 + mh - 4, x0 + 4:x0 + mw - 4] -= level


def albedo(seed: int, surface: int, h: int, w: int, style: Style, tau: float) -> np.ndarray:
    """uint8 HxWx3 albedo for one surface of h x w pixels."""
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    lum = style.base_level + style.base_contrast * (fbm(seed, surface, L_BASE, h, w) - 0.5)
    area = h * w / 65536.0                                     # m^2 at 256 px/m
    stains = value_noise(seed, surface, L_STAINS, 0, h, w, 96)
    lum = lum + np.where(stains > 0.72, -0.18 * tau, 0.0) + np.where(stains < 0.18, 0.10 * tau, 0.0)
    _discs(lum, seed, surface, L_PITS, int(tau * 90 * area), 2.0, 14.0, -0.35 * tau, 0.25 * tau)
    _cracks(lum, seed, surface, int(tau * 1.5 * area + 0.5), tau)
    _marks(lum, seed, surface, int(tau * 0.8 * area + 0.5), tau)
    lum = np.clip(lum, 0.02, 0.98)
    rgb = lum[..., None] * np.asarray(style.tint, dtype=np.float64)[None, None, :]
    return np.clip(np.floor(rgb * 255.0 + 0.5), 0, 255).astype(np.uint8)
