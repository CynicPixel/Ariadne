"""2D ground truth on the grid: occupancy, exact ESDF, traversability, geodesics.

Grid convention: array index [row, col] covers the square [ox+col, ox+col+1) x
[oy+row, oy+row+1) in grid units; row grows with +y (north). Cell centres are at
+0.5. Every wall surface lies on a cell boundary (layout rectangles are integer).

ESDF: for a free cell, the exact Euclidean distance (metres) from its centre to the
union of occupied squares; for an occupied cell, minus the distance from its centre
to the union of free squares. Exact because the point-to-axis-aligned-square
distance separates: d^2 = max(|dx|-1/2, 0)^2 + max(|dy|-1/2, 0)^2 (in cells), so
the classic two-pass (column, then row) lower-envelope decomposition applies with
these edge offsets. Values are capped at `cap_m` (clearance beyond it never matters
for the corridors/rooms we build; the cap is recorded).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .layout import Rect


@dataclass(frozen=True)
class Grid:
    ox: int          # grid units of column 0's west edge
    oy: int          # grid units of row 0's south edge
    width: int       # columns
    height: int      # rows
    g: float         # metres per cell

    def index(self, x_units: int, y_units: int) -> tuple[int, int]:
        """(row, col) of the cell whose square contains the unit point (x, y) as its SW corner."""
        return y_units - self.oy, x_units - self.ox

    def cell_center_m(self, row, col):
        return (self.ox + col + 0.5) * self.g, (self.oy + row + 0.5) * self.g


def grid_for(rects: list[Rect], pad: int, g: float) -> Grid:
    x0 = min(r.x0 for r in rects) - pad
    y0 = min(r.y0 for r in rects) - pad
    x1 = max(r.x1 for r in rects) + pad
    y1 = max(r.y1 for r in rects) + pad
    return Grid(x0, y0, x1 - x0, y1 - y0, g)


def rasterize(grid: Grid, rects: list[Rect]) -> np.ndarray:
    """Boolean FREE mask: True inside any rectangle (half-open, exact)."""
    free = np.zeros((grid.height, grid.width), bool)
    for r in rects:
        r0, c0 = grid.index(r.x0, r.y0)
        r1, c1 = grid.index(r.x1, r.y1)
        if r0 < 0 or c0 < 0 or r1 > grid.height or c1 > grid.width:
            raise ValueError("rectangle outside the grid")
        free[r0:r1, c0:c1] = True
    return free


def _edt_to(target: np.ndarray, cap_cells: int) -> np.ndarray:
    """Exact distance (cells) from every cell centre to the union of `target` squares,
    capped at cap_cells. Cells inside target get 0."""
    H, W = target.shape
    big = H + W + cap_cells + 10
    idx = np.arange(H, dtype=np.int64)[:, None]
    last = np.where(target, idx, -big)
    last = np.maximum.accumulate(last, axis=0)
    nxt = np.where(target, idx, 2 * big)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    dy = np.minimum(idx - last, nxt - idx).astype(np.float64)       # centre-to-centre rows
    dy_edge = np.where(dy == 0, 0.0, np.maximum(dy - 0.5, 0.0))
    dy_edge = np.minimum(dy_edge, cap_cells + 1.0)
    best = np.full((H, W), np.inf)
    kmax = min(cap_cells + 1, W - 1)       # shifts beyond the grid would wrap in slicing
    for k in range(-kmax, kmax + 1):
        dx_edge = 0.0 if k == 0 else abs(k) - 0.5
        shifted = np.full((H, W), np.inf)
        if k >= 0:
            shifted[:, :W - k] = dy_edge[:, k:]
        else:
            shifted[:, -k:] = dy_edge[:, :W + k]
        # A column at offset k with nothing in it contributes inf; k=0 with dy_edge=0 is inside.
        cand = dx_edge * dx_edge + shifted * shifted
        np.minimum(best, cand, out=best)
    d = np.sqrt(best)
    return np.minimum(d, float(cap_cells))


def esdf(free: np.ndarray, g: float, cap_m: float = 4.0) -> np.ndarray:
    cap = int(math.ceil(cap_m / g))
    d_free = _edt_to(~free, cap)        # free cells: distance to occupied
    d_occ = _edt_to(free, cap)          # occupied cells: distance to free
    return np.where(free, d_free, -d_occ) * g


TRAV_EPS_M = 1e-9   # esdf is (cells * g) in floating point; equality must not hinge on one ulp


def traversable(esdf_m: np.ndarray, rho_m: float) -> np.ndarray:
    return esdf_m >= rho_m - TRAV_EPS_M


def octile_m(a: tuple[int, int], b: tuple[int, int], g: float) -> float:
    """Shortest possible 8-connected path length between two cells (a lower bound
    on the geodesic, whatever the obstacles)."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return g * (max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc))


_STEPS = ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
          (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)))


def geodesic(trav: np.ndarray, start: tuple[int, int], g: float, max_m: float = math.inf):
    """8-connected Dijkstra over `trav` from `start` (row, col).

    Diagonal steps require both orthogonal neighbours traversable (no squeezing
    between two corners). Settling stops once distances exceed `max_m` (cells
    beyond it stay inf: "farther than max_m", exact below it). Returns
    (distance_m array, inf where unreachable/beyond; parent flat-index array,
    -1 at the start/unreached).
    """
    H, W = trav.shape
    if not trav[start]:
        raise ValueError("geodesic start is not traversable")
    dist = np.full(H * W, np.inf)
    parent = np.full(H * W, -1, dtype=np.int64)
    flat_trav = trav.ravel()
    s = start[0] * W + start[1]
    dist[s] = 0.0
    limit = max_m / g
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        if d > limit:
            break
        r, c = divmod(u, W)
        for dr, dc, w in _STEPS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = rr * W + cc
            if not flat_trav[v]:
                continue
            if dr and dc and not (flat_trav[r * W + cc] and flat_trav[rr * W + c]):
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    dist[dist > limit] = np.inf        # tentative values past the limit are not settled
    return (dist * g).reshape(H, W), parent


def extract_path(parent: np.ndarray, width: int, goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells from the Dijkstra start to `goal` (inclusive), following parent pointers."""
    u = goal[0] * width + goal[1]
    path = [goal]
    while parent[u] != -1:
        u = int(parent[u])
        path.append(divmod(u, width))
    return path[::-1]
