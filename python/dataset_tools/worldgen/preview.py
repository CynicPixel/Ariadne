"""Top-down preview PNG of a scene: one pixel per grid cell, north up.

Legend:
  near-black   solid (occupied)
  light grey   free, responder-traversable (ESDF >= rho_responder)
  mid grey     free but only agent-traversable (ESDF in [rho_agent, rho_responder))
  dark grey    free but within rho_agent of a wall
  red tint     narrow (agent-only) passage rectangles
  blue tint    rooms;  yellow: target room
  green/cyan   agent 0 / agent 1 scripted trajectory (every 0.05 s), start dots 3x3
  magenta      agent 0's responder geodesic to the target (the GT egress route)
"""
from __future__ import annotations

import numpy as np

from .layout import passage_rect
from .png import write_png
from .raster import extract_path, geodesic


def render(scene) -> np.ndarray:
    g = scene.grid
    img = np.zeros((g.height, g.width, 3), np.uint8)
    img[:] = (20, 20, 24)
    img[scene.free] = (70, 70, 70)
    img[scene.trav_agent] = (140, 140, 140)
    img[scene.trav_responder] = (215, 215, 215)

    def tint(rect, rgb, alpha=0.35):
        r0, c0 = g.index(rect.x0, rect.y0)
        r1, c1 = g.index(rect.x1, rect.y1)
        sub = img[r0:r1, c0:c1].astype(float)
        m = scene.free[r0:r1, c0:c1]
        sub[m] = (1 - alpha) * sub[m] + alpha * np.array(rgb, float)
        img[r0:r1, c0:c1] = sub.astype(np.uint8)

    L = scene.layout
    for c, rect in sorted(L.rooms.items()):
        tint(rect, (250, 220, 60) if c == scene.target_room else (80, 120, 255))
    for p in L.passages.values():
        if p.narrow:
            tint(passage_rect(L, p), (255, 40, 40), 0.55)

    colors = [(40, 200, 60), (40, 210, 230)]
    dist, parent = geodesic(scene.trav_responder, scene.start_cells[0], g.g)
    for (r, c) in extract_path(parent, g.width, scene.target_cell):
        img[r, c] = (230, 60, 230)
    for k, sc in enumerate(scene.scripts):
        t = 0.0
        while t <= sc.duration:
            (x, y, _, _, _), _, _ = sc.pose(t)
            r, c = g.index(int(np.floor(x / g.g)), int(np.floor(y / g.g)))
            img[r, c] = colors[k % 2]
            t += 0.05
    for k, s in enumerate(scene.start_cells):
        img[max(s[0] - 1, 0):s[0] + 2, max(s[1] - 1, 0):s[1] + 2] = colors[k % 2]
    t = scene.target_cell
    img[t[0] - 2:t[0] + 3, t[1] - 2:t[1] + 3] = (250, 200, 0)
    return img[::-1]          # row 0 is south; images put row 0 at the top


def write_preview(scene, path) -> None:
    write_png(path, render(scene))
