from collections import Counter, deque

import pytest

from dataset_tools.worldgen import layout as lay
from dataset_tools.worldgen.params import parse_preset
from dataset_tools.worldgen.rng import PhiloxStream

PRESET = parse_preset({"name": "test"})


def connected(adj, s, t, banned=None):
    seen, q = {s}, deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if banned and lay.Edge((u, v)) == banned:
                continue
            if v not in seen:
                seen.add(v)
                q.append(v)
    return t in seen


def two_edge_connected_bruteforce(cells, edges, s, t):
    """Menger (k=2): two edge-disjoint paths iff connected and no single edge separates."""
    adj = lay.adjacency(cells, edges)
    if not connected(adj, s, t):
        return False
    return all(connected(adj, s, t, banned=e) for e in edges)


def test_wilson_gives_spanning_trees():
    for seed in range(40):
        edges = lay.wilson_ust(4, 5, PhiloxStream(seed, 0, 1))
        cells = set(lay.lattice_cells(4, 5))
        assert len(edges) == len(cells) - 1
        adj = lay.adjacency(cells, edges)
        assert all(connected(adj, (0, 0), c) for c in cells)


def test_wilson_is_uniform_on_the_four_cycle():
    # The 2x2 lattice is a 4-cycle: exactly 4 spanning trees, each with prob. 1/4.
    counts = Counter(frozenset(lay.wilson_ust(2, 2, PhiloxStream(7, t, 1))) for t in range(8000))
    assert len(counts) == 4
    for n in counts.values():
        assert abs(n - 2000) < 150            # ~3.9 sigma of binomial(8000, 1/4)


def test_edge_disjoint_paths_matches_menger_on_random_graphs():
    for seed in range(200):
        st = PhiloxStream(seed, 1, 2)
        cells = set(lay.lattice_cells(3, 3))
        all_edges = sorted({lay.Edge((c, n)) for c in cells for n in lay.lattice_neighbors(c, 3, 3)},
                           key=lay._edge_key)
        edges = {e for e in all_edges if st.bernoulli(0.6)}
        adj = lay.adjacency(cells, edges)
        s, t = (0, 0), (2, 2)
        k = lay.edge_disjoint_paths(adj, s, t, need=2)
        assert (k >= 2) == two_edge_connected_bruteforce(cells, edges, s, t)
        assert (k >= 1) == connected(adj, s, t)


def test_edge_disjoint_needs_residual_arcs():
    # Graph where the shortest first path s-a-b-t blocks a greedy second path;
    # with residual arcs the answer is 2 (s-a-t and s-b-t... via a-b reversal).
    adj = {"s": ["a", "c"], "a": ["s", "b", "d"], "b": ["a", "t", "c"], "c": ["s", "b"],
           "d": ["a", "t"], "t": ["b", "d"]}
    assert lay.edge_disjoint_paths(adj, "s", "t") == 2


@pytest.mark.parametrize("seed", range(25))
def test_completed_layout_invariants(seed):
    P = PRESET
    for attempt in range(6):
        try:
            base = lay.build_base(P, seed, attempt)
        except lay.Reject:
            continue
        targets = [c for c in base.rooms if c not in {e.room for e in base.entrances}]
        for t in targets:
            try:
                L = lay.complete_for_target(base, t)
            except lay.Reject:
                continue
            check_layout(L, t)
        return
    pytest.skip("no base layout in 6 attempts (covered by the sweep)")


def check_layout(L, target):
    P = L.preset
    S = L.S
    assert set(L.rooms) <= L.cells
    for c, r in L.rooms.items():
        cx, cy = L.center(c)
        assert r.x0 < cx < r.x1 and r.y0 < cy < r.y1               # corridor mouths open into it
        assert c[0] * S < r.x0 and r.x1 < (c[0] + 1) * S            # stays inside its cell
        assert c[1] * S < r.y0 and r.y1 < (c[1] + 1) * S
    adj = lay.adjacency(L.cells, L.edges)
    some = next(iter(L.cells))
    assert all(connected(adj, some, c) for c in L.cells)
    for c in L.stub_cells:
        assert len(adj[c]) == 1 and c not in L.rooms
    for c in L.cells - L.stub_cells:
        assert len(adj[c]) >= 2 or c in L.rooms                     # culled: no other dead ends
    narrow = {e for e, p in L.passages.items() if p.narrow}
    assert len(narrow) == P.corridors.narrow_count
    resp = L.edges - narrow
    for ent in L.entrances:
        assert two_edge_connected_bruteforce(L.cells, resp, ent.room, target)
    for e in narrow:
        a, b = lay._edge_key(e)
        assert connected(lay.adjacency(L.cells, L.edges - {e}), a, b)   # not a bridge
    g = P.structure.grid_m
    for p in L.passages.values():
        lo, hi = P.corridors.narrow_width_m if p.narrow else P.corridors.width_m
        assert lo - 1e-9 <= p.width * g <= hi + 1e-9
    assert len({e.room for e in L.entrances}) == len(L.entrances)


def _first_base(seed):
    for attempt in range(20):
        try:
            return lay.build_base(PRESET, seed, attempt)
        except lay.Reject:
            continue
    raise AssertionError("no base layout in 20 attempts")


def test_build_is_deterministic():
    a, b = _first_base(11), _first_base(11)
    assert a.attempt == b.attempt
    assert a.tree_edges == b.tree_edges and a.rooms == b.rooms and a.entrances == b.entrances
    c = _first_base(12)
    assert (a.tree_edges, a.rooms) != (c.tree_edges, c.rooms)
