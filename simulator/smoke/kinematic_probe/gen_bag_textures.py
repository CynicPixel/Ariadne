#!/usr/bin/env python3
"""Regenerate the tex_*.png albedo maps that bagsize.sdf references (SMOKE §4g).

Seven-octave value noise (amplitude 0.6^octave), 1024x1024, tinted grey. This is
the texture whose soft, cloud-like render motivated the corner-bearing layers in
docs/design/simulation.md §3. Throwaway; superseded by worldgen.
"""
import struct
import zlib

import numpy as np


def png(path, img):
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


rng = np.random.default_rng(1)


def fbm(h, w):
    out = np.zeros((h, w))
    for o in range(7):
        n = 2 ** (o + 2)
        g = rng.random((n + 1, n + 1))
        ys, xs = np.linspace(0, n, h, endpoint=False), np.linspace(0, n, w, endpoint=False)
        y0, x0 = ys.astype(int), xs.astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a, b = g[y0][:, x0], g[y0][:, x0 + 1]
        c, d = g[y0 + 1][:, x0], g[y0 + 1][:, x0 + 1]
        out += (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy) * 0.6 ** o
    return (out - out.min()) / (out.max() - out.min())


for name in ("wn", "ws", "we", "ww", "floor"):
    v = fbm(1024, 1024)
    img = np.stack([v * 200 + 30, v * 190 + 35, v * 180 + 40], -1).clip(0, 255).astype(np.uint8)
    png(f"tex_{name}.png", img)
