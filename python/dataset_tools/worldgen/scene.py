"""The attempt loop: layout -> raster -> target choice, with deterministic rejection.

A scene is accepted at the first attempt a in 0..max_attempts-1 whose layout
satisfies every constraint. All randomness for attempt a is keyed (seed, a, site),
so the accepted world is a pure function of (GENERATOR_VERSION, seed, preset), and
the attempt index is recorded anyway.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import GENERATOR_VERSION
from .assets import load_charuco
from .furnish import place_boards, place_props
from .hazards import place_hazards
from .routes import gate, opportunities, schedule
from .layout import Layout, Reject, S_HEIGHT, S_TARGET, build_base, complete_for_target
from .params import Preset
from .raster import Grid, esdf, geodesic, grid_for, octile_m, rasterize, traversable

ESDF_CAP_M = 4.0


class GenerationFailed(RuntimeError):
    def __init__(self, preset: Preset, seed: int, reasons: Counter):
        self.reasons = reasons
        top = ", ".join(f"{r} x{n}" for r, n in reasons.most_common())
        super().__init__(f"preset '{preset.name}' seed {seed}: no attempt of "
                         f"{preset.search.max_attempts} accepted ({top})")


@dataclass
class Scene:
    preset: Preset
    seed: int
    attempt: int
    layout: Layout
    grid: Grid
    free: np.ndarray                    # navigable free space: structure minus props (the GT)
    esdf_m: np.ndarray
    trav_agent: np.ndarray
    trav_responder: np.ndarray
    target_room: tuple
    target_cell: tuple                  # (row, col) on the grid
    start_cells: list                   # per agent, (row, col)
    lengths: dict = field(default_factory=dict)   # per agent: {"responder_m", "agent_m"}
    rejections: Counter = field(default_factory=Counter)
    structure_free: np.ndarray | None = None   # rooms/passages/vestibules only (walls come from it)
    props: list = field(default_factory=list)
    boards: list = field(default_factory=list)   # furnish.BoardPlacement per entrance
    hazards: list = field(default_factory=list)  # hazards.Hazard (D112); empty unless the preset asks
    wall_height_m: float = 0.0
    scripts: list = field(default_factory=list)   # routes.AgentScript per agent
    start_delays_s: list = field(default_factory=list)   # per-agent extra initial hold
    min_separation_m: float = float("inf")
    overlap: dict = field(default_factory=dict)   # routes.opportunities

    @property
    def scene_id(self) -> str:
        blob = json.dumps({"generator_version": GENERATOR_VERSION, "seed": self.seed,
                           "preset": self.preset.digest(), "attempt": self.attempt},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _room_center_cell(grid: Grid, rect) -> tuple[int, int]:
    cx2, cy2 = rect.center2()
    return grid.index(cx2 // 2, cy2 // 2)


def _raster(L: Layout, cap_m: float, props=()):
    """Grid, structural free mask, navigable free mask (minus prop footprints), ESDF."""
    g = L.preset.structure.grid_m
    rects = [r for _, _, r in L.free_rects()]
    grid = grid_for(rects, pad=round(1.0 / g), g=g)
    structure = rasterize(grid, rects)
    free = structure.copy()
    if props:
        free &= ~rasterize(grid, [p.rect for p in props])
    return grid, structure, free, esdf(free, g, cap_m)


def _passage_midpoints(L: Layout, grid: Grid) -> list[tuple[int, int]]:
    out = []
    for e in sorted(L.passages, key=lambda e_: sorted(e_)):
        p = L.passages[e]
        if p.narrow:
            continue
        (ax, ay), (bx, by) = L.center(p.a), L.center(p.b)
        out.append(grid.index((ax + bx) // 2, (ay + by) // 2))
    return out


def _wall_height(L: Layout) -> float:
    """Per-scene wall height, a whole number of grid cells in the preset range."""
    g = L.preset.structure.grid_m
    lo, hi = L.preset.structure.wall_height_m
    st = L.stream(S_HEIGHT)
    return st.integer(int(-(-round(lo / g, 9) // 1)), int(round(hi / g, 9) // 1)) * g


def evaluate(base: Layout) -> Scene:
    """Try target rooms (seeded order) on a base layout. Raises Reject.

    Cheap tests first: topology (complete_for_target), then an octile lower bound on
    the length, then the raster with an ESDF capped just above rho_responder (enough
    to decide traversability), then Dijkstra from the primary start stopped at the
    band's upper edge. The full ESDF is computed only for the accepted layout.
    """
    P = base.preset
    g = P.structure.grid_m
    lo, hi = P.target.path_length_m
    rho_r, rho_a = P.clearance.rho_responder_m, P.clearance.rho_agent_m
    excluded = ({base.entrances[0].room} if P.target.allow_entrance_room
                else {ent.room for ent in base.entrances})
    candidates = [c for c in sorted(base.rooms) if c not in excluded]
    reasons = Counter()
    for room in base.stream(S_TARGET).shuffled(candidates):
        try:
            L = complete_for_target(base, room)
        except Reject as r:
            reasons[str(r)] += 1
            continue
        grid, _, free, e = _raster(L, cap_m=rho_r + 2 * g)
        tc = _room_center_cell(grid, L.rooms[room])
        starts = [grid.index(*ent.start) for ent in L.entrances]
        if octile_m(starts[0], tc, g) > hi:
            reasons["target beyond the band (octile bound)"] += 1
            continue
        trav_r = traversable(e, rho_r)
        if not trav_r[tc] or not all(trav_r[s_] for s_ in starts):
            raise AssertionError("room centre or entrance start below rho_responder clearance")
        d0 = geodesic(trav_r, starts[0], g, max_m=hi)[0][tc]
        if not lo <= d0 <= hi:
            reasons["target outside the length band"] += 1
            continue
        # Furnish (calibration boards first: props keep out of the space in front of
        # them), then re-verify everything the scene promises on the final raster.
        charuco = load_charuco()
        try:
            boards = place_boards(L, *charuco.size_m, P.motion.altitude_m)
        except Reject as r:
            reasons[str(r)] += 1
            continue
        props = place_props(L, boards)
        grid, structure, free, e = _raster(L, cap_m=ESDF_CAP_M, props=props)
        trav_r, trav_a = traversable(e, rho_r), traversable(e, rho_a)
        if not trav_r[tc] or not all(trav_r[s_] for s_ in starts):
            reasons["props blocked the target or a start"] += 1
            continue
        d_start = [geodesic(trav_r, s_, g)[0] for s_ in starts]
        if not lo <= d_start[0][tc] <= hi:
            reasons["props moved the length out of the band"] += 1
            continue
        if not all(np.isfinite(d_start[0][m]) for m in _passage_midpoints(L, grid)):
            reasons["props blocked a responder passage"] += 1
            continue
        scene = Scene(preset=P, seed=L.seed, attempt=L.attempt, layout=L, grid=grid, free=free,
                      esdf_m=e, trav_agent=trav_a, trav_responder=trav_r, target_room=room,
                      target_cell=tc, start_cells=starts, structure_free=structure, props=props,
                      boards=boards, wall_height_m=_wall_height(L), hazards=place_hazards(L))
        for k, s_ in enumerate(starts):
            dr = d_start[k][tc]
            da = geodesic(trav_a, s_, g)[0][tc]
            if not np.isfinite(dr):
                raise AssertionError("2-edge-connected target unreachable on the raster")
            scene.lengths[k] = {"responder_m": float(dr), "agent_m": float(da)}
        # Scripted trajectories and the closure-opportunity gate (design §4.3).
        try:
            scene.scripts, scene.start_delays_s, scene.min_separation_m = schedule(scene)
        except ValueError as err:          # RouteInfeasible, or no separating start delay
            reasons[str(err)] += 1
            continue
        scene.overlap = opportunities(scene, scene.scripts)
        why = gate(scene, scene.overlap)
        if why is not None:
            reasons[why] += 1
            continue
        return scene
    if not candidates:
        raise Reject("no candidate target room")
    raise Reject(reasons.most_common(1)[0][0])


def generate(preset: Preset, seed: int) -> Scene:
    reasons = Counter()
    for attempt in range(preset.search.max_attempts):
        try:
            scene = evaluate(build_base(preset, seed, attempt))
        except Reject as r:
            reasons[str(r)] += 1
            continue
        scene.rejections = reasons
        return scene
    raise GenerationFailed(preset, seed, reasons)
