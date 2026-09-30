"""Props, appearance zones and lights (design §3). All in integer grid units.

Props are boxes standing against a room wall. They are obstacles (they enter the GT
occupancy) and 3D structure for stereo/PnP, but they are NOT part of the structural
free mask that walls are extracted from, so a prop never grows full-height walls.

Keep-outs, so props cannot break what the scene promises:
  - squares around the room-rect centre (the GT target point) and the lattice-cell
    centre (the survey spot; params.validate keeps every survey offset inside it),
  - every corridor mouth plus a margin (passages, vestibules),
  - other props plus a gap.
scene.py re-verifies routes on the raster after furnishing.
"""
from __future__ import annotations

from dataclasses import dataclass

from .layout import Layout, Rect
from .texture import Style

S_PROPS, S_STYLE = 20, 21

OPPOSITE = {"W": "E", "E": "W", "S": "N", "N": "S"}


@dataclass(frozen=True)
class BoardPlacement:
    """A ChArUco board on an entrance room's wall, facing the entering agent.

    wall: the room side it hangs on (N/S/E/W); fixed: that wall line (grid units);
    a, b: its extent along the wall (grid units, half-open); z0, z1: bottom/top (m).
    The board image (with its white margin) fills [a, b] x [z0, z1] exactly."""
    agent: int
    room: tuple
    wall: str
    fixed: int
    a: int
    b: int
    z0: float
    z1: float
    keepout: Rect           # free volume in front of it that props may not enter


@dataclass(frozen=True)
class Prop:
    room: tuple
    index: int
    rect: Rect          # footprint, grid units
    height_m: float


def _overlaps(a: Rect, b: Rect, gap: int = 0) -> bool:
    return not (a.x1 + gap <= b.x0 or b.x1 + gap <= a.x0 or a.y1 + gap <= b.y0 or b.y1 + gap <= a.y0)


def _units(m: float, g: float) -> int:
    return int(round(m / g))


def place_boards(L: Layout, board_w_m: float, board_h_m: float, center_z_m: float) -> list[BoardPlacement]:
    """One board per entrance, on the room wall opposite the vestibule, clear of every
    corridor mouth (+0.3 m), as close to the corridor axis as possible. Raises
    Reject if a wall has no room for it."""
    from .layout import Reject
    P = L.preset
    g = P.structure.grid_m
    Wu = int(-(-round(board_w_m / g, 9) // 1))
    margin = _units(0.3, g)
    depth = _units(P.routes.calib_keepout_depth_m, g)
    mouths = [r for kind, _, r in L.free_rects() if kind in ("passage", "vestibule")]
    out = []
    for k, ent in enumerate(L.entrances):
        room = L.rooms[ent.room]
        wall = OPPOSITE[ent.side]
        cx, cy = L.center(ent.room)
        if wall in ("E", "W"):
            fixed = room.x1 if wall == "E" else room.x0
            lo, hi, axis = room.y0, room.y1, cy
            # a corridor opens through this wall iff its rectangle strictly spans the line
            blocked = [(m.y0, m.y1) for m in mouths if m.x0 < fixed < m.x1]
        else:
            fixed = room.y1 if wall == "N" else room.y0
            lo, hi, axis = room.x0, room.x1, cx
            blocked = [(m.x0, m.x1) for m in mouths if m.y0 < fixed < m.y1]
        blocked = [(max(a - margin, lo), min(b + margin, hi)) for a, b in blocked if b > lo and a < hi]
        best = None
        for a in range(lo + margin, hi - margin - Wu + 1):
            if any(a < bb and a + Wu > ba for ba, bb in blocked):
                continue
            key = (abs(2 * a + Wu - 2 * axis), a)
            if best is None or key < best[0]:
                best = (key, a)
        if best is None:
            raise Reject("no wall space for the calibration board")
        a = best[1]
        b = a + Wu
        lat = _units(0.3, g)
        if wall == "E":
            ko = Rect(fixed - depth, a - lat, fixed, b + lat)
        elif wall == "W":
            ko = Rect(fixed, a - lat, fixed + depth, b + lat)
        elif wall == "N":
            ko = Rect(a - lat, fixed - depth, b + lat, fixed)
        else:
            ko = Rect(a - lat, fixed, b + lat, fixed + depth)
        out.append(BoardPlacement(k, ent.room, wall, fixed, a, b,
                                  center_z_m - board_h_m / 2, center_z_m + board_h_m / 2, ko))
    return out


def place_props(L: Layout, boards=()) -> list[Prop]:
    P = L.preset
    g = P.structure.grid_m
    cfg = P.props
    mouths = [r for kind, _, r in L.free_rects() if kind in ("passage", "vestibule")]
    keep_mouth = _units(cfg.keepout_mouth_m, g)
    keep_center = _units(cfg.keepout_center_m, g)
    gap = _units(0.3, g)
    props = []
    for c in sorted(L.rooms):
        room = L.rooms[c]
        st = L.stream(S_PROPS, c[0], c[1])
        n = st.integer(*cfg.per_room)
        cx2, cy2 = room.center2()
        ccx, ccy = L.center(c)
        # Two keep-out squares: the room-rect centre (the GT target point) and the
        # lattice-cell centre (where corridor axes meet and surveys happen).
        keeps = [Rect(cx2 // 2 - keep_center, cy2 // 2 - keep_center,
                      cx2 // 2 + keep_center, cy2 // 2 + keep_center),
                 Rect(ccx - keep_center, ccy - keep_center, ccx + keep_center, ccy + keep_center)]
        placed = []
        for k in range(n):
            along = _units(st.uniform(*cfg.size_along_m), g)
            depth = _units(st.uniform(*cfg.depth_m), g)
            height = round(st.uniform(*cfg.height_m), 3)
            side = st.choice(["W", "E", "S", "N"])
            if side in ("W", "E"):
                span = room.y1 - room.y0 - along
                if span < 0:
                    continue
                y0 = room.y0 + st.integer(0, span)
                x0 = room.x0 if side == "W" else room.x1 - depth
                rect = Rect(x0, y0, x0 + depth, y0 + along)
            else:
                span = room.x1 - room.x0 - along
                if span < 0:
                    continue
                x0 = room.x0 + st.integer(0, span)
                y0 = room.y0 if side == "S" else room.y1 - depth
                rect = Rect(x0, y0, x0 + along, y0 + depth)
            if any(_overlaps(rect, kr) for kr in keeps):
                continue
            if any(_overlaps(rect, bp.keepout) for bp in boards):
                continue
            if any(_overlaps(rect, m, keep_mouth) for m in mouths if _overlaps(m, room)):
                continue
            if any(_overlaps(rect, p.rect, gap) for p in placed):
                continue
            placed.append(Prop(c, k, rect, height))
        props.extend(placed)
    return props


def zone_styles(L: Layout) -> dict:
    """One Style per room, per passage and per vestibule: rooms differ in hue and
    brightness (place distinctiveness), corridors are a cooler grey family."""
    out = {}
    for kind, owner, _ in L.free_rects():
        key = (kind, owner)
        st = L.stream(S_STYLE, hash_key(key))
        if kind == "room":
            tint = (0.80 + 0.40 * st.uniform(), 0.80 + 0.35 * st.uniform(), 0.75 + 0.35 * st.uniform())
            level = 0.45 + 0.20 * st.uniform()
        else:
            v = 0.85 + 0.20 * st.uniform()
            tint = (v * 0.95, v * 0.97, v)
            level = 0.40 + 0.15 * st.uniform()
        out[key] = Style(tuple(min(t, 1.2) for t in tint), level, 0.40)
    return out


def hash_key(key) -> int:
    """Stable small int for a zone key (kind, owner): no Python hash() (salted per run)."""
    kind, owner = key
    kid = {"room": 1, "passage": 2, "vestibule": 3}[kind]
    if kind == "room":
        return kid << 40 | owner[0] << 20 | owner[1]
    if kind == "passage":
        (a, b) = owner
        return kid << 40 | a[0] << 30 | a[1] << 20 | b[0] << 10 | b[1]
    return kid << 40 | owner


@dataclass(frozen=True)
class PointLight:
    name: str
    x_m: float
    y_m: float
    z_m: float


def lights(L: Layout, wall_height_m: float) -> list[PointLight]:
    """A ceiling light over every room centre and every junction; none in stubs."""
    P = L.preset
    g = P.structure.grid_m
    out = []
    if not P.lighting.zone_lights:
        return out
    z = wall_height_m - 0.3
    for c in sorted(L.rooms):
        cx2, cy2 = L.rooms[c].center2()
        out.append(PointLight(f"room_{c[0]}_{c[1]}", cx2 * g / 2, cy2 * g / 2, z))
    for c in sorted(L.cells):
        if L.kind[c] == "junction" and c not in L.stub_cells:
            x, y = L.center(c)
            out.append(PointLight(f"junction_{c[0]}_{c[1]}", x * g, y * g, z))
    return out
