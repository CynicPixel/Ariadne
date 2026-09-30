import math

import numpy as np
import pytest

from dataset_tools.worldgen import raster
from dataset_tools.worldgen.layout import Rect


def brute_esdf(free: np.ndarray, g: float) -> np.ndarray:
    """Exact signed distance from each cell centre to the other class's squares."""
    H, W = free.shape
    out = np.zeros((H, W))
    rr, cc = np.mgrid[0:H, 0:W]
    for r in range(H):
        for c in range(W):
            other = free != free[r, c]
            if not other.any():
                out[r, c] = np.inf
                continue
            dx = np.maximum(np.abs(cc[other] - c) - 0.5, 0.0)
            dy = np.maximum(np.abs(rr[other] - r) - 0.5, 0.0)
            d = np.sqrt(dx * dx + dy * dy).min() * g
            out[r, c] = d if free[r, c] else -d
    return out


@pytest.mark.parametrize("seed", range(6))
def test_esdf_is_exact_against_brute_force(seed):
    rng = np.random.default_rng(seed)       # test data only; not a worldgen code path
    free = rng.random((23, 31)) > 0.35
    free[0, :] = False                      # both classes present
    free[5, 5] = True
    got = raster.esdf(free, 0.05, cap_m=10.0)
    want = brute_esdf(free, 0.05)
    assert np.allclose(got, want, atol=1e-12)


def test_esdf_cap_is_applied():
    free = np.ones((5, 400), bool)
    free[:, 0] = False
    e = raster.esdf(free, 0.05, cap_m=1.0)
    assert e.max() == pytest.approx(1.0)
    assert e[2, 5] == pytest.approx(4.5 * 0.05)   # 4.5 cells from the wall surface


def test_corridor_clearance_matches_the_width_rule():
    """Max centre clearance of an n-cell corridor is (ceil(n/2) - 1/2) cells: n/2 - 1/2
    for even n (no centre cell), n/2 for odd n. params.validate relies on:
    w >= 2*rho + g always passes rho, and w <= 2*rho always blocks it."""
    g = 0.05
    for n in (15, 16, 17, 20, 28):
        free = np.zeros((n + 4, 10), bool)
        free[2:2 + n, :] = True
        e = raster.esdf(free, g, cap_m=2.0)
        assert e[:, 5].max() == pytest.approx((math.ceil(n / 2) - 0.5) * g)
    for rho in (0.20, 0.40):
        for n in range(4, 40):
            free = np.zeros((n + 4, 6), bool)
            free[2:2 + n, :] = True
            passes = raster.esdf(free, g, cap_m=2.0)[:, 3].max() >= rho - 1e-12
            if n * g >= 2 * rho + g - 1e-9:
                assert passes, (rho, n)
            if n * g <= 2 * rho + 1e-9:
                assert not passes, (rho, n)


def test_rasterize_is_half_open_and_bounds_checked():
    grid = raster.Grid(ox=-2, oy=-3, width=10, height=10, g=0.05)
    free = raster.rasterize(grid, [Rect(0, 0, 3, 2)])
    assert free.sum() == 6
    assert free[grid.index(0, 0)] and free[grid.index(2, 1)] and not free[grid.index(3, 1)]
    with pytest.raises(ValueError):
        raster.rasterize(grid, [Rect(0, 0, 20, 2)])


def test_geodesic_straight_and_diagonal():
    trav = np.ones((50, 50), bool)
    d, _ = raster.geodesic(trav, (0, 0), 0.1)
    assert d[0, 30] == pytest.approx(3.0)
    assert d[20, 20] == pytest.approx(20 * math.sqrt(2) * 0.1)
    assert d[10, 30] == pytest.approx(raster.octile_m((0, 0), (10, 30), 0.1))


def test_geodesic_does_not_squeeze_between_corners():
    trav = np.ones((3, 3), bool)
    trav[0, 1] = False
    trav[1, 0] = False
    d, _ = raster.geodesic(trav, (0, 0), 1.0)
    assert np.isinf(d[1, 1])                 # the only way was a corner-cutting diagonal


def test_geodesic_limit_is_exact_below_and_inf_above():
    rng = np.random.default_rng(3)
    trav = rng.random((60, 60)) > 0.25
    trav[0, 0] = True
    full, _ = raster.geodesic(trav, (0, 0), 0.05)
    lim, _ = raster.geodesic(trav, (0, 0), 0.05, max_m=1.2)
    below = full <= 1.2
    assert np.array_equal(lim[below], full[below])
    assert np.all(np.isinf(lim[~below]))


def test_extract_path_is_connected_and_ends_at_goal():
    trav = np.ones((20, 20), bool)
    trav[5:15, 10] = False
    d, parent = raster.geodesic(trav, (10, 2), 1.0)
    path = raster.extract_path(parent, 20, (10, 18))
    assert path[0] == (10, 2) and path[-1] == (10, 18)
    steps = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(path, path[1:])]
    assert steps and max(steps) == 1
    length = sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
    assert length == pytest.approx(d[10, 18])
