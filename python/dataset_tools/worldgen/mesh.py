"""Textured quads as Wavefront OBJ files.

Why quads and not <box>: gz maps the whole albedo image onto every face of a box,
whatever the face's aspect, which visibly stretches textures (SMOKE §4g). A quad
with explicit UVs has exactly one texture mapping, which we control.

Quad convention (what the OBJ UV probe verified; see SMOKE §4h):
  origin  p0     the corner that is BOTTOM-LEFT as seen from the FRONT
  eu             edge along the texture's u axis (left -> right seen from the front)
  ev             edge along the texture's v axis (bottom -> top)
  front normal   n = normalize(eu x ev); triangles are counter-clockwise seen from
                 the front, so back-face culling hides the back.
  UV             (0,0) at p0, (1,0) at p0+eu, (1,1) at p0+eu+ev, (0,1) at p0+ev.

Formatting is fixed (%.5f metres = 10 um) so a world's files are byte-stable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quad:
    p0: tuple[float, float, float]
    eu: tuple[float, float, float]
    ev: tuple[float, float, float]

    def normal(self) -> np.ndarray:
        n = np.cross(np.asarray(self.eu, float), np.asarray(self.ev, float))
        norm = np.linalg.norm(n)
        if norm <= 0:
            raise ValueError("degenerate quad")
        return n / norm

    def size(self) -> tuple[float, float]:
        return float(np.linalg.norm(self.eu)), float(np.linalg.norm(self.ev))

    def corners(self) -> np.ndarray:
        p0, eu, ev = (np.asarray(v, float) for v in (self.p0, self.eu, self.ev))
        return np.array([p0, p0 + eu, p0 + eu + ev, p0 + ev])


def _f(x: float) -> str:
    s = f"{x:.5f}"
    return "0.00000" if s == "-0.00000" else s   # no signed zeros in the files


def obj_text(quad: Quad, name: str = "quad") -> str:
    c = quad.corners()
    n = quad.normal()
    if abs(float(np.dot(quad.eu, quad.ev))) > 1e-9 * max(quad.size()) ** 2:
        raise ValueError("quad edges must be orthogonal (texture would shear)")
    lines = [f"o {name}"]
    lines += [f"v {_f(p[0])} {_f(p[1])} {_f(p[2])}" for p in c]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines.append(f"vn {_f(n[0])} {_f(n[1])} {_f(n[2])}")
    # Two CCW triangles (seen from the front): 1-2-3 and 1-3-4.
    lines += ["f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1"]
    return "\n".join(lines) + "\n"


def write_obj(path, quad: Quad, name: str = "quad") -> None:
    with open(path, "w", newline="\n") as f:
        f.write(obj_text(quad, name))
