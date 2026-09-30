"""Stage 1 of arch §5.5: a seeded room-maze floorplan, in integer grid units.

Pipeline for one attempt (every random choice drawn from its own Philox site):
  build_base:           1. uniform spanning tree (Wilson); 2. rooms; 3. entrances.
  complete_for_target:  4. cycle cover so every entrance has two edge-disjoint routes
                        to the target; 5. extra braids; 6. cull/stubs; 7. narrow
                        passages off the route cycle; 8. widths.
scene.py tries target rooms in seeded order and checks the geodesic length band on
the raster.

Units: every coordinate here is an INTEGER number of grid cells (grid_m each), so
rasterization is exact and the layout is identical on every machine. Rectangles are
half-open [x0, x1) x [y0, y1). World metres = units * grid_m, origin at the lattice's
south-west corner, x east, y north.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field

from .params import Preset
from .rng import PhiloxStream, site_of

Cell = tuple[int, int]
Edge = frozenset  # frozenset({Cell, Cell})

# Stage ids for Philox sites. Never renumber: that would change every world.
S_TREE, S_ROOMS, S_ROOM_GEOM, S_BRAID, S_CULL, S_ENTRANCE, S_NARROW, S_WIDTH = 1, 2, 3, 4, 5, 6, 7, 8
S_TARGET, S_HEIGHT, S_COVER = 9, 10, 11

SIDES = {"W": (-1, 0), "E": (1, 0), "S": (0, -1), "N": (0, 1)}


class Reject(Exception):
    """This attempt cannot satisfy the preset; the caller moves to the next attempt."""


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self):
        if not (self.x0 < self.x1 and self.y0 < self.y1):
            raise ValueError(f"empty rect {self}")

    def center2(self) -> tuple[int, int]:
        """Twice the centre (keeps half-cells exact)."""
        return self.x0 + self.x1, self.y0 + self.y1


@dataclass(frozen=True)
class Passage:
    a: Cell
    b: Cell
    width: int          # grid units
    narrow: bool


@dataclass(frozen=True)
class Entrance:
    room: Cell
    side: str           # exterior side of the room's lattice cell: W/E/S/N
    width: int
    vestibule: Rect     # the corridor from the room out past the lattice border
    start: tuple[int, int]      # agent start, grid units (vestibule's closed end + margin)
    start_yaw_deg: int          # facing into the room: 0 = +x (east), 90 = +y (north)


@dataclass
class Layout:
    preset: Preset
    seed: int
    attempt: int
    S: int                                  # lattice cell size in grid units
    cells: set = field(default_factory=set)         # kept lattice cells
    kind: dict = field(default_factory=dict)        # cell -> "room" | "junction"
    rooms: dict = field(default_factory=dict)       # cell -> Rect
    edges: set = field(default_factory=set)         # kept lattice edges (Edge)
    tree_edges: set = field(default_factory=set)
    braid_edges: set = field(default_factory=set)
    stub_cells: set = field(default_factory=set)
    passages: dict = field(default_factory=dict)    # Edge -> Passage
    entrances: list = field(default_factory=list)   # [Entrance], index = agent id
    target: Cell | None = None

    def stream(self, *site) -> PhiloxStream:
        return PhiloxStream(self.seed, self.attempt, site_of(*site))

    def center(self, c: Cell) -> tuple[int, int]:
        return c[0] * self.S + self.S // 2, c[1] * self.S + self.S // 2

    def neighbors(self, c: Cell, edges=None) -> list[Cell]:
        es = self.edges if edges is None else edges
        out = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if Edge((c, n)) in es:
                out.append(n)
        return out

    def free_rects(self) -> list[tuple[str, object, Rect]]:
        """Every free-space rectangle, tagged (kind, owner, rect). Deterministic order."""
        out = [("room", c, self.rooms[c]) for c in sorted(self.rooms)]
        for e in sorted(self.passages, key=_edge_key):
            p = self.passages[e]
            out.append(("passage", _edge_key(e), passage_rect(self, p)))
        for k, ent in enumerate(self.entrances):
            out.append(("vestibule", k, ent.vestibule))
        return out


def _edge_key(e: Edge) -> tuple[Cell, Cell]:
    a, b = sorted(e)
    return a, b


def lattice_cells(nx: int, ny: int) -> list[Cell]:
    return [(i, j) for j in range(ny) for i in range(nx)]


def lattice_neighbors(c: Cell, nx: int, ny: int) -> list[Cell]:
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # fixed order: draws depend on it
        n = (c[0] + dx, c[1] + dy)
        if 0 <= n[0] < nx and 0 <= n[1] < ny:
            out.append(n)
    return out


def wilson_ust(nx: int, ny: int, st: PhiloxStream) -> set:
    """Uniform spanning tree by Wilson's loop-erased random walks.

    Loop erasure is implicit: `nxt[u]` is overwritten each time the walk leaves u,
    so following nxt from the start yields the walk with every loop removed.
    """
    order = st.shuffled(lattice_cells(nx, ny))
    in_tree = {order[0]}
    edges = set()
    for start in order[1:]:
        if start in in_tree:
            continue
        nxt = {}
        u = start
        while u not in in_tree:
            v = st.choice(lattice_neighbors(u, nx, ny))
            nxt[u] = v
            u = v
        u = start
        while u not in in_tree:
            in_tree.add(u)
            edges.add(Edge((u, nxt[u])))
            u = nxt[u]
    return edges


def edge_disjoint_paths(adj: dict, s: Cell, t: Cell, need: int = 2) -> int:
    """Number of edge-disjoint s-t paths, up to `need` (unit-capacity max-flow).

    Each undirected edge is two opposite arcs of capacity 1; augmenting along
    residual arcs (including reverse flow) makes the count exact, which greedy
    path deletion is not.
    """
    if s == t:
        return need
    flow = {}   # (u, v) -> flow on arc u->v in {-1, 0, 1}
    count = 0
    while count < need:
        parent = {s: None}
        q = deque([s])
        while q and t not in parent:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent and flow.get((u, v), 0) < 1:
                    parent[v] = u
                    q.append(v)
        if t not in parent:
            break
        v = t
        while parent[v] is not None:
            u = parent[v]
            flow[(u, v)] = flow.get((u, v), 0) + 1
            flow[(v, u)] = flow.get((v, u), 0) - 1
            v = u
        count += 1
    return count


def adjacency(cells, edges) -> dict:
    adj = {c: [] for c in sorted(cells)}
    for e in sorted(edges, key=_edge_key):
        a, b = _edge_key(e)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def hop_distances(adj: dict, s: Cell) -> dict:
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _ceil_units(m: float, g: float) -> int:
    """Smallest whole number of grid cells covering m metres (robust to 1.0/0.05 = 19.999...)."""
    return int(-(-round(m / g, 9) // 1))


def _grid_int_range(st: PhiloxStream, lo_m: float, hi_m: float, g: float) -> int:
    """Uniform integer number of grid units whose length lies in [lo_m, hi_m]."""
    lo = int(-(-round(lo_m / g, 9) // 1))       # ceil, robust to 1.0/0.05 = 19.999...
    hi = int(round(hi_m / g, 9) // 1)           # floor
    if lo > hi:
        raise ValueError(f"no grid multiple of {g} in [{lo_m}, {hi_m}]")
    return st.integer(lo, hi)


def passage_rect(L: Layout, p: Passage) -> Rect:
    """Axis-aligned corridor between two lattice-cell centres, `width` units wide.

    It overlaps the rooms/junctions it joins; the union of rectangles is the free
    space, so the overlap is harmless and closes every junction.
    """
    (ax, ay), (bx, by) = sorted((L.center(p.a), L.center(p.b)))
    lo = -(p.width // 2)
    hi = lo + p.width
    if ay == by:
        return Rect(ax + lo, ay + lo, bx + hi, ay + hi)
    return Rect(ax + lo, ay + lo, ax + hi, by + hi)


def tree_path_edges(tree_adj: dict, s: Cell, t: Cell) -> list:
    """Edges of the unique s-t path in a tree (BFS parents)."""
    parent, q = {s: None}, deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in tree_adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if t not in parent:
        raise AssertionError("tree is not spanning")
    out, v = [], t
    while parent[v] is not None:
        out.append(Edge((v, parent[v])))
        v = parent[v]
    return out


def build_base(preset: Preset, seed: int, attempt: int) -> Layout:
    """Target-independent stages: spanning tree, rooms, entrances. Raises Reject."""
    P = preset
    g = P.structure.grid_m
    S = round(P.lattice.cell_m / g)
    nx, ny = P.lattice.nx, P.lattice.ny
    L = Layout(preset=P, seed=seed, attempt=attempt, S=S)

    # 1. spanning tree over the whole lattice
    L.tree_edges = wilson_ust(nx, ny, L.stream(S_TREE))
    all_cells = set(lattice_cells(nx, ny))
    L.cells = set(all_cells)

    # 2. rooms: a rectangle per room cell that contains the cell centre with a margin
    #    (corridors run through cell centres, so every corridor mouth opens into the room)
    st = L.stream(S_ROOMS)
    n_rooms = st.integer(*P.rooms.count)
    room_cells = sorted(st.shuffled(sorted(all_cells))[:n_rooms])
    margin = _ceil_units(P.corridors.width_m[1] / 2 + P.rooms.center_margin_m, g)
    border = _ceil_units(0.5, g)
    for c in room_cells:
        rs = L.stream(S_ROOM_GEOM, c[0], c[1])
        wx = _grid_int_range(rs, *P.rooms.size_m, g)
        wy = _grid_int_range(rs, *P.rooms.size_m, g)
        cx, cy = L.center(c)
        x_lo = max(c[0] * S + border, cx + margin - wx)
        x_hi = min(c[0] * S + S - border - wx, cx - margin)
        y_lo = max(c[1] * S + border, cy + margin - wy)
        y_hi = min(c[1] * S + S - border - wy, cy - margin)
        if x_lo > x_hi or y_lo > y_hi:
            raise AssertionError("room placement range empty; preset validation should prevent this")
        x0, y0 = rs.integer(x_lo, x_hi), rs.integer(y_lo, y_hi)
        L.rooms[c] = Rect(x0, y0, x0 + wx, y0 + wy)
        L.kind[c] = "room"

    # 3. entrances: rooms on the lattice border; the second maximizes tree hop distance
    #    from the first (so the two agents start far apart).
    st = L.stream(S_ENTRANCE)
    def exterior_sides(c):
        return [sd for sd, (dx, dy) in SIDES.items()
                if not (0 <= c[0] + dx < nx and 0 <= c[1] + dy < ny)]
    boundary_rooms = [c for c in sorted(L.rooms) if exterior_sides(c)]
    if len(boundary_rooms) < P.entrances.count:
        raise Reject("not enough boundary rooms for the entrances")
    first = st.choice(boundary_rooms)
    chosen = [first]
    if P.entrances.count == 2:
        hops = hop_distances(adjacency(all_cells, L.tree_edges), first)
        others = [c for c in boundary_rooms if c != first]
        best = max(hops[c] for c in others)
        chosen.append(st.choice([c for c in others if hops[c] == best]))
    for c in chosen:
        side = st.choice(exterior_sides(c))
        width = _grid_int_range(st, *P.corridors.width_m, g)
        L.entrances.append(_make_entrance(L, c, side, width))
    return L


def complete_for_target(base: Layout, target: Cell) -> Layout:
    """Target-dependent stages on a copy of `base`. Raises Reject.

    4. cycle cover: add non-tree edges until every tree edge on every entrance->target
       path lies on a cycle. Theorem: in tree + extra edges, s and t are 2-edge-connected
       iff no bridge separates them, and the bridges are exactly the tree edges not on
       any added edge's fundamental cycle. Greedy set cover (max newly covered edges,
       seeded tie-break), at most `cover_edges_max` edges.
    5. `extra_edges` more random braids (maze loops; room for narrow passages).
    6. cull non-room dead ends; re-attach `dead_end_stubs` of them.
    7. narrow passages: non-bridge edges whose removal keeps every entrance
       2-edge-connected to the target (so the scripted loop route exists at
       rho_responder and the narrow passage is a genuine agent-only shortcut).
    8. widths.
    The 2-edge-connectivity is re-verified by max-flow; a failure is a bug, not a Reject.
    """
    P = base.preset
    g = P.structure.grid_m
    nx, ny = P.lattice.nx, P.lattice.ny
    L = copy.deepcopy(base)
    L.target = target
    all_cells = set(lattice_cells(nx, ny))
    tree_adj = adjacency(all_cells, L.tree_edges)
    tkey = (target[0], target[1])

    required = set()
    if P.topology.route_cycle:
        for ent in L.entrances:
            required.update(tree_path_edges(tree_adj, ent.room, target))
    candidates = sorted({Edge((c, n)) for c in all_cells for n in lattice_neighbors(c, nx, ny)}
                        - L.tree_edges, key=_edge_key)
    fcycle = {e: set(tree_path_edges(tree_adj, *_edge_key(e))) for e in candidates}
    st = L.stream(S_COVER, *tkey)
    cover = []
    uncovered = set(required)
    while uncovered:
        if len(cover) == P.topology.cover_edges_max:
            raise Reject("route cycle needs more than cover_edges_max braids")
        gain = {e: len(fcycle[e] & uncovered) for e in candidates if e not in cover}
        best = max(gain.values(), default=0)
        if best == 0:
            raise Reject("route cycle cannot be closed")
        e = st.choice([e for e in candidates if gain.get(e, -1) == best])
        cover.append(e)
        uncovered -= fcycle[e]
    rest = [e for e in candidates if e not in cover]
    extra = L.stream(S_BRAID, *tkey).shuffled(rest)[:P.topology.extra_edges]
    if len(extra) < P.topology.extra_edges:
        raise Reject("not enough non-tree edges to braid")
    L.braid_edges = set(cover) | set(extra)
    edges = set(L.tree_edges) | L.braid_edges

    cells = set(all_cells)
    removed_edge = {}
    while True:
        adj = adjacency(cells, edges)
        leaves = [c for c in sorted(cells) if len(adj[c]) == 1 and c not in L.rooms]
        if not leaves:
            break
        for c in leaves:
            e = Edge((c, adj[c][0]))
            edges.discard(e)
            cells.discard(c)
            removed_edge[c] = e
    adj = adjacency(cells, edges)
    if any(not adj[c] for c in cells):
        raise AssertionError("culling left an isolated cell")
    stub_candidates = sorted(c for c, e in removed_edge.items() if (set(e) - {c}) <= cells)
    for c in L.stream(S_CULL, *tkey).shuffled(stub_candidates)[:P.topology.dead_end_stubs]:
        cells.add(c)
        edges.add(removed_edge[c])
        L.stub_cells.add(c)
    L.cells, L.edges = cells, edges
    L.kind = {c: ("room" if c in L.rooms else "junction") for c in cells}

    need = 2 if P.topology.route_cycle else 1

    def routes_ok(es):
        a = adjacency(cells, es)
        return all(edge_disjoint_paths(a, ent.room, target, need=need) >= need for ent in L.entrances)

    if not routes_ok(edges):
        raise AssertionError("entrances not connected to the target as the preset requires")
    # A narrow passage is an agent-only ALTERNATIVE connection: it must not be a bridge
    # (its ends stay connected without it; else it is a dead end or the only way in),
    # and removing it must keep every entrance's two-route loop to the target.
    narrow = set()
    for e in L.stream(S_NARROW, *tkey).shuffled(sorted(edges, key=_edge_key)):
        if len(narrow) == P.corridors.narrow_count:
            break
        rest = edges - narrow - {e}
        a, b = _edge_key(e)
        if hop_distances(adjacency(cells, rest), a).get(b) is None:
            continue
        if routes_ok(rest):
            narrow.add(e)
    if len(narrow) < P.corridors.narrow_count:
        raise Reject("no room for a narrow passage off the route cycle")

    for e in sorted(edges, key=_edge_key):
        a, b = _edge_key(e)
        ws = L.stream(S_WIDTH, a[0], a[1], b[0], b[1])
        rng_m = P.corridors.narrow_width_m if e in narrow else P.corridors.width_m
        L.passages[e] = Passage(a, b, _grid_int_range(ws, *rng_m, g), e in narrow)
    return L


def _make_entrance(L: Layout, c: Cell, side: str, width: int) -> Entrance:
    """Vestibule from the room centre out through the exterior side, `vestibule_m` past
    the lattice border, closed at its far end. The agent starts 0.6 m from that end,
    on the axis, facing the room."""
    P = L.preset
    g = P.structure.grid_m
    V = round(P.entrances.vestibule_m / g)
    back = int(-(-round(0.6 / g, 9) // 1))
    cx, cy = L.center(c)
    lo = -(width // 2)
    hi = lo + width
    nx, ny = P.lattice.nx * L.S, P.lattice.ny * L.S
    if side == "W":
        rect, start, yaw = Rect(-V, cy + lo, cx + hi, cy + hi), (-V + back, cy), 0
    elif side == "E":
        rect, start, yaw = Rect(cx + lo, cy + lo, nx + V, cy + hi), (nx + V - back, cy), 180
    elif side == "S":
        rect, start, yaw = Rect(cx + lo, -V, cx + hi, cy + hi), (cx, -V + back), 90
    else:
        rect, start, yaw = Rect(cx + lo, cy + lo, cx + hi, ny + V), (cx, ny + V - back), 270
    return Entrance(room=c, side=side, width=width, vestibule=rect, start=start, start_yaw_deg=yaw)
