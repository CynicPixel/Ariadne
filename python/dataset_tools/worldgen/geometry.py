"""Renderable surfaces from the structural free mask (+ props), as mesh.Quad lists.

Walls: every edge between a free cell and a solid cell is a wall surface facing into
the free cell. Maximal straight runs of such edges become one quad each (split into
pieces of at most `max_len_m` so a texture side stays <= 2048 px at 256 px/m).
For a face with horizontal unit normal n: eu = z x n (left->right seen from the
front), ev = z, so eu x ev = n: the convention verified in SMOKE §4h.

Floors/ceilings: a greedy maximal-rectangle decomposition of the free mask
(row-major, extend east then north), each rectangle capped at `max_len_m` per side.

All coordinates stay integer grid units until the final conversion to metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Quad
from .raster import Grid

# surface id = KIND << 24 | index  (texture Philox counter word; never renumber)
K_WALL, K_FLOOR, K_CEIL, K_PROP, K_BOARD = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class Surface:
    sid: int
    kind: int
    quad: Quad
    size_px: tuple[int, int]      # texture (height, width) in pixels
    probe_xy_units: tuple[float, float]   # a point just in front of the face (zone lookup)


@dataclass(frozen=True)
class WallRun:
    normal: str                   # direction the face looks: N/S/E/W
    fixed: int                    # the wall line: y (units) for N/S faces, x for E/W
    a: int                        # run start along the wall (units, inclusive edge)
    b: int                        # run end (units, exclusive edge)


def wall_runs(free: np.ndarray, grid: Grid) -> list[WallRun]:
    """Maximal runs of free/solid edges, deterministic order (normal, fixed, a)."""
    H, W = free.shape
    solid = ~free
    runs = []
    # Face looking north: free cell (r, c) with solid (r-1, c) -> wall line y = oy + r.
    for normal, mask, along_axis in (
        ("N", free[1:, :] & solid[:-1, :], "x"),     # rows 1..H-1: wall below
        ("S", free[:-1, :] & solid[1:, :], "x"),     # wall above
        ("E", free[:, 1:] & solid[:, :-1], "y"),     # wall to the west
        ("W", free[:, :-1] & solid[:, 1:], "y"),     # wall to the east
    ):
        if along_axis == "x":
            for r in range(mask.shape[0]):
                cols = np.flatnonzero(mask[r])
                for s, e in _runs(cols):
                    # mask row r compares grid rows r and r+1: the wall line is y = oy + r + 1
                    runs.append(WallRun(normal, grid.oy + r + 1, grid.ox + s, grid.ox + e + 1))
        else:
            for c in range(mask.shape[1]):
                rows = np.flatnonzero(mask[:, c])
                for s, e in _runs(rows):
                    runs.append(WallRun(normal, grid.ox + c + 1, grid.oy + s, grid.oy + e + 1))
    return runs


def _runs(idx: np.ndarray):
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def _split(a: int, b: int, max_units: int):
    n = -(-(b - a) // max_units)
    edges = [a + (b - a) * k // n for k in range(n + 1)]
    return list(zip(edges[:-1], edges[1:]))


def wall_surfaces(runs, g: float, height_m: float, px_per_m: int, max_len_m: float = 8.0):
    out = []
    max_units = int(round(max_len_m / g))
    idx = 0
    for run in runs:
        for a, b in _split(run.a, run.b, max_units):
            L = (b - a) * g
            f = run.fixed * g
            if run.normal == "N":       # n = +y, eu = z x n = -x: p0 at the +x end
                q = Quad((b * g, f, 0.0), (-L, 0.0, 0.0), (0.0, 0.0, height_m))
                probe = ((a + b) / 2, run.fixed + 0.5)
            elif run.normal == "S":     # n = -y, eu = +x
                q = Quad((a * g, f, 0.0), (L, 0.0, 0.0), (0.0, 0.0, height_m))
                probe = ((a + b) / 2, run.fixed - 0.5)
            elif run.normal == "E":     # n = +x, eu = +y
                q = Quad((f, a * g, 0.0), (0.0, L, 0.0), (0.0, 0.0, height_m))
                probe = (run.fixed + 0.5, (a + b) / 2)
            else:                       # "W": n = -x, eu = -y: p0 at the +y end
                q = Quad((f, b * g, 0.0), (0.0, -L, 0.0), (0.0, 0.0, height_m))
                probe = (run.fixed - 0.5, (a + b) / 2)
            size = (max(int(round(height_m * px_per_m)), 8), max(int(round(L * px_per_m)), 8))
            out.append(Surface(K_WALL << 24 | idx, K_WALL, q, size, probe))
            idx += 1
    return out


def free_rectangles(free: np.ndarray, max_side: int) -> list[tuple[int, int, int, int]]:
    """Greedy disjoint cover of `free` by rectangles (r0, c0, r1, c1), half-open."""
    H, W = free.shape
    used = np.zeros_like(free)
    out = []
    for r in range(H):
        c = 0
        while c < W:
            if not free[r, c] or used[r, c]:
                c += 1
                continue
            c1 = c
            while c1 < W and c1 - c < max_side and free[r, c1] and not used[r, c1]:
                c1 += 1
            r1 = r + 1
            while (r1 < H and r1 - r < max_side and free[r1, c:c1].all()
                   and not used[r1, c:c1].any()):
                r1 += 1
            used[r:r1, c:c1] = True
            out.append((r, c, r1, c1))
            c = c1
    return out


def floor_ceiling_surfaces(free, grid: Grid, height_m: float, px_per_m: int, max_len_m: float = 8.0):
    g = grid.g
    out = []
    rects = free_rectangles(free, int(round(max_len_m / g)))
    for i, (r0, c0, r1, c1) in enumerate(rects):
        x0, y0 = (grid.ox + c0) * g, (grid.oy + r0) * g
        lx, ly = (c1 - c0) * g, (r1 - r0) * g
        size = (max(int(round(ly * px_per_m)), 8), max(int(round(lx * px_per_m)), 8))
        probe = (grid.ox + (c0 + c1) / 2, grid.oy + (r0 + r1) / 2)
        out.append(Surface(K_FLOOR << 24 | i, K_FLOOR,
                           Quad((x0, y0, 0.0), (lx, 0.0, 0.0), (0.0, ly, 0.0)), size, probe))
        out.append(Surface(K_CEIL << 24 | i, K_CEIL,
                           Quad((x0, y0 + ly, height_m), (lx, 0.0, 0.0), (0.0, -ly, 0.0)), size, probe))
    return out


def prop_surfaces(props, g: float, px_per_m: int):
    """Five outward faces per box prop (no bottom)."""
    out = []
    for pi, p in enumerate(props):
        x0, y0, x1, y1 = p.rect.x0 * g, p.rect.y0 * g, p.rect.x1 * g, p.rect.y1 * g
        h = p.height_m
        lx, ly = x1 - x0, y1 - y0
        faces = [
            Quad((x0, y0, 0.0), (lx, 0.0, 0.0), (0.0, 0.0, h)),     # south face, n = -y
            Quad((x1, y1, 0.0), (-lx, 0.0, 0.0), (0.0, 0.0, h)),    # north face, n = +y
            Quad((x1, y0, 0.0), (0.0, ly, 0.0), (0.0, 0.0, h)),     # east face,  n = +x
            Quad((x0, y1, 0.0), (0.0, -ly, 0.0), (0.0, 0.0, h)),    # west face,  n = -x
            Quad((x0, y0, h), (lx, 0.0, 0.0), (0.0, ly, 0.0)),      # top,        n = +z
        ]
        cx, cy = (p.rect.x0 + p.rect.x1) / 2, (p.rect.y0 + p.rect.y1) / 2
        for fi, q in enumerate(faces):
            w_m, h_m = q.size()
            size = (max(int(round(h_m * px_per_m)), 8), max(int(round(w_m * px_per_m)), 8))
            out.append(Surface(K_PROP << 24 | (pi << 4 | fi), K_PROP, q, size, (cx, cy)))
    return out


BOARD_OFFSET_M = 0.002    # proud of the wall: no z-fighting with the wall quad behind it


def board_quad(bp, g: float, width_m: float) -> Quad:
    """The board quad (exact physical width, centred in its reserved wall interval),
    facing into the room; image row 0 maps to its top edge (SMOKE §4h convention)."""
    c = (bp.a + bp.b) / 2 * g
    f = bp.fixed * g
    h = bp.z1 - bp.z0
    hw = width_m / 2
    if bp.wall == "E":      # on the east wall, looking west (n = -x): eu = z x n = -y
        return Quad((f - BOARD_OFFSET_M, c + hw, bp.z0), (0.0, -width_m, 0.0), (0.0, 0.0, h))
    if bp.wall == "W":      # n = +x: eu = +y
        return Quad((f + BOARD_OFFSET_M, c - hw, bp.z0), (0.0, width_m, 0.0), (0.0, 0.0, h))
    if bp.wall == "N":      # n = -y: eu = +x
        return Quad((c - hw, f - BOARD_OFFSET_M, bp.z0), (width_m, 0.0, 0.0), (0.0, 0.0, h))
    return Quad((c + hw, f + BOARD_OFFSET_M, bp.z0), (-width_m, 0.0, 0.0), (0.0, 0.0, h))   # "S": n = +y


def board_surfaces(boards, g: float, width_m: float, image_hw: tuple[int, int]):
    out = []
    for bp in boards:
        q = board_quad(bp, g, width_m)
        probe = ((bp.keepout.x0 + bp.keepout.x1) / 2, (bp.keepout.y0 + bp.keepout.y1) / 2)
        out.append(Surface(K_BOARD << 24 | bp.agent, K_BOARD, q, image_hw, probe))
    return out
