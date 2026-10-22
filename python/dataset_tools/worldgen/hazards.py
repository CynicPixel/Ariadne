"""Structural hazards above a standing responder (design RISK_ZONES_4.2, D112). All in integer grid units, except
heights (m).

Two kinds, from the US&R collapse patterns a rescuer is trained to fear (design §1):
  cantilever  a slab held by one room wall, projecting into the room over the floor
  hung        a debris block suspended over walkable floor (its cable is sub-voxel and not modelled)
Every hazard's underside is >= 1.9 m (params.validate), above a 1.8 m standing body, so GT traversability is
unchanged: a hazard threatens a route without blocking it. GT `risk_zone` = footprints dilated by zone_margin_m.

Placement draws from its own Philox stream (S_HAZARDS), so no other stream's draws move, and a preset without
hazards places none (no draw at all).
Keep-outs: a 1 m square at the room-rect centre and at the lattice-cell centre (the target point and the ceiling
light), and other hazards plus gap_m. Rooms only: never a passage or a vestibule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import Layout, Rect

S_HAZARDS = 40
# Above head height the centre keep-out only protects the target point and the ceiling light: a 1 m square, not the
# props' 2.6 m one (which leaves less than a cantilever's minimum projection in a 4 m room).
KEEPOUT_CENTER_M = 0.5
TRIES_PER_HAZARD = 20


@dataclass(frozen=True)
class Hazard:
    kind: str           # "cantilever" | "hung"
    room: tuple
    index: int
    rect: Rect          # footprint, grid units
    z0: float           # underside, m
    z1: float           # top, m
    side: str = ""      # cantilever: the wall it is held by (W/E/S/N)


def _overlaps(a: Rect, b: Rect, gap: int = 0) -> bool:
    return a.x0 < b.x1 + gap and b.x0 < a.x1 + gap and a.y0 < b.y1 + gap and b.y0 < a.y1 + gap


def _units(m: float, g: float) -> int:
    return max(1, int(round(m / g)))


def place_hazards(L: Layout) -> list[Hazard]:
    P = L.preset
    hz = P.hazards
    if hz.cantilevers[1] == 0 and hz.hung[1] == 0:
        return []
    g = P.structure.grid_m
    st = L.stream(S_HAZARDS)
    rooms = sorted(L.rooms)
    keep_center = _units(KEEPOUT_CENTER_M, g)
    keeps = []
    for c in rooms:
        cx2, cy2 = L.rooms[c].center2()
        ccx, ccy = L.center(c)
        keeps += [Rect(cx2 // 2 - keep_center, cy2 // 2 - keep_center, cx2 // 2 + keep_center, cy2 // 2 + keep_center),
                  Rect(ccx - keep_center, ccy - keep_center, ccx + keep_center, ccy + keep_center)]
    gap = _units(hz.gap_m, g)
    placed: list[Hazard] = []

    def ok(rect: Rect, room: Rect) -> bool:
        inside = room.x0 <= rect.x0 and rect.x1 <= room.x1 and room.y0 <= rect.y0 and rect.y1 <= room.y1
        return inside and not any(_overlaps(rect, k) for k in keeps) and not any(_overlaps(rect, h.rect, gap) for h in placed)

    n_cant = st.integer(*hz.cantilevers)
    n_hung = st.integer(*hz.hung)
    for k in range(n_cant):
        for _ in range(TRIES_PER_HAZARD):
            c = st.choice(rooms)
            room = L.rooms[c]
            proj = _units(st.uniform(*hz.cant_projection_m), g)
            width = _units(st.uniform(*hz.cant_width_m), g)
            thick = round(st.uniform(*hz.cant_thickness_m), 3)
            z0 = round(st.uniform(*hz.cant_underside_m), 3)
            side = st.choice(["W", "E", "S", "N"])
            if side in ("W", "E"):
                span = room.y1 - room.y0 - width
                if span < 0 or proj > room.x1 - room.x0:
                    continue
                y0 = room.y0 + st.integer(0, span)
                x0 = room.x0 if side == "W" else room.x1 - proj
                rect = Rect(x0, y0, x0 + proj, y0 + width)
            else:
                span = room.x1 - room.x0 - width
                if span < 0 or proj > room.y1 - room.y0:
                    continue
                x0 = room.x0 + st.integer(0, span)
                y0 = room.y0 if side == "S" else room.y1 - proj
                rect = Rect(x0, y0, x0 + width, y0 + proj)
            if ok(rect, room):
                placed.append(Hazard("cantilever", c, k, rect, z0, round(z0 + thick, 3), side))
                break
    for k in range(n_hung):
        for _ in range(TRIES_PER_HAZARD):
            c = st.choice(rooms)
            room = L.rooms[c]
            a = _units(st.uniform(*hz.hung_side_m), g)
            b = _units(st.uniform(*hz.hung_side_m), g)
            hgt = round(st.uniform(*hz.hung_height_m), 3)
            z0 = round(st.uniform(*hz.hung_underside_m), 3)
            if room.x1 - room.x0 < a or room.y1 - room.y0 < b:
                continue
            x0 = room.x0 + st.integer(0, room.x1 - room.x0 - a)
            y0 = room.y0 + st.integer(0, room.y1 - room.y0 - b)
            rect = Rect(x0, y0, x0 + a, y0 + b)
            if ok(rect, room):
                placed.append(Hazard("hung", c, n_cant + k, rect, z0, round(z0 + hgt, 3)))
                break
    return placed


def risk_layers(hazards: list[Hazard], grid, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
    """(hazard_footprint, risk_zone) on the GT raster: footprints, and footprints dilated by margin_m (Euclidean,
    cell centre to the footprint rectangle's nearest point)."""
    from .raster import rasterize
    foot = rasterize(grid, [h.rect for h in hazards])
    H, W = foot.shape
    zone = foot.copy()
    g = grid.g
    rr, cc = np.mgrid[0:H, 0:W]
    for h in hazards:
        r0, c0 = grid.index(h.rect.x0, h.rect.y0)
        r1, c1 = grid.index(h.rect.x1, h.rect.y1)   # exclusive corner in cells
        dy = np.maximum(np.maximum(r0 - (rr + 0.5), (rr + 0.5) - r1), 0.0)
        dx = np.maximum(np.maximum(c0 - (cc + 0.5), (cc + 0.5) - c1), 0.0)
        zone |= np.hypot(dx, dy) * g <= margin_m + 1e-9
    return foot, zone
