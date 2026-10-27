"""Worldgen presets: strict YAML -> frozen dataclasses, validated, hashed.

A scene is identified by (GENERATOR_VERSION, seed, preset hash, attempt). The hash
is over the *parsed* values in canonical JSON, so comments and key order in the
YAML never change a world's identity, while any value change always does.

Strictness mirrors the C++ StrictConfig: an unknown or misspelled key is an error,
never a silent default.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import yaml


class PresetError(ValueError):
    pass


Range = tuple[float, float]


@dataclass(frozen=True)
class Lattice:
    nx: int = 4
    ny: int = 4
    cell_m: float = 6.0


@dataclass(frozen=True)
class Rooms:
    count: tuple[int, int] = (3, 4)          # inclusive range; one room per lattice cell
    size_m: Range = (3.5, 5.0)               # side length; <= cell_m - 1 m (0.5 m border each side)
    center_margin_m: float = 0.3             # room must contain the cell centre +- (w_max/2 + this)


@dataclass(frozen=True)
class Corridors:
    width_m: Range = (1.0, 1.4)              # responder-passable passages
    narrow_count: int = 1                    # agent-only passages (Invariant 4)
    narrow_width_m: Range = (0.55, 0.70)


@dataclass(frozen=True)
class Topology:
    route_cycle: int = 1                     # 1: every entrance gets two edge-disjoint routes to the target
    cover_edges_max: int = 2                 # braids allowed to close the entrance->target route cycle
    extra_edges: int = 1                     # further random braids (maze loops, narrow-passage cycles)
    dead_end_stubs: int = 1                  # culled corridor cells re-attached as stubs


@dataclass(frozen=True)
class Entrances:
    count: int = 2                           # one per agent; each in a distinct boundary room
    vestibule_m: float = 2.5                 # outward corridor stub where the agent starts


@dataclass(frozen=True)
class Target:
    path_length_m: Range = (25.0, 40.0)      # primary entrance -> target, geodesic at rho_responder
    allow_entrance_room: int = 0             # 1: the target may be the SECONDARY entrance's room


@dataclass(frozen=True)
class Clearance:
    rho_agent_m: float = 0.20                # arch §20.5: 0.15-0.25
    rho_responder_m: float = 0.40            # arch §20.5 default


@dataclass(frozen=True)
class Structure:
    grid_m: float = 0.05                     # GT raster resolution; all geometry snaps to it
    wall_height_m: Range = (2.6, 3.0)


@dataclass(frozen=True)
class Props:
    per_room: tuple[int, int] = (2, 5)       # attempted per room; keep-outs may drop some
    size_along_m: Range = (0.4, 1.2)         # along the wall
    depth_m: Range = (0.4, 0.8)              # out from the wall
    height_m: Range = (0.4, 1.8)
    keepout_center_m: float = 1.3            # half-side of the free square at the room centre
    keepout_mouth_m: float = 0.6             # clearance kept around every corridor mouth


@dataclass(frozen=True)
class Appearance:
    tau: float = 1.0                         # texture richness (design §3); 0 = smooth concrete
    repetition: float = 0.0                  # I7 aliasing dial; only 0 is implemented in Phase 1
    wall_px_per_m: int = 256
    floor_px_per_m: int = 128


@dataclass(frozen=True)
class Lighting:
    ambient: float = 0.35                    # world <scene><ambient> grey level
    zone_lights: int = 1                     # 1 = a point light over every room and junction
    light_diffuse: float = 0.8
    light_range_m: float = 12.0
    shadows: int = 0                         # world <scene><shadows> (D131: 1 for lamp-lit dark worlds)


@dataclass(frozen=True)
class Motion:
    lk_budget_px: float = 30.0               # per-frame image motion the tracker must absorb (§4.2)
    focal_px: float = 640.0                  # rig focal length (HFOV 90 deg at 1280 px)
    frame_dt_s: float = 0.05                 # 20 Hz
    v_cap_mps: float = 0.6
    a_max_mps2: float = 0.4
    omega_max_radps: float = 0.44            # 25 deg/s; also capped by the image-edge flow budget
    alpha_max_radps2: float = 0.6
    altitude_m: float = 1.0
    altitude_jitter_m: float = 0.2           # per move, IMU excitation
    hold_start_s: float = 2.0                # stationary start (IMU initialisation)
    hold_end_s: float = 1.0


@dataclass(frozen=True)
class Routes:
    template: str = "cross"                  # "cross" | "loop" | "return" (routes.py)
    surveys: str = "all"                     # "all": every entrance room + target; "target": target only
    keyframe_step_m: float = 0.5
    keyframe_step_deg: float = 15.0
    overlap_radius_m: float = 1.5
    overlap_yaw_deg: float = 30.0
    revisit_min_dt_s: float = 30.0
    survey_offset_m: float = 0.5             # per-agent survey spot offset from the cell centre
    calibration: int = 0                     # 1: agent 0 flies the ChArUco calibration dance
    calib_standoffs_m: Range = (1.3, 1.9)    # view distances from the board
    calib_lateral_m: float = 0.5             # side offsets of the near views
    calib_tilt_yaw_deg: float = 15.0         # orientation diversity per view position
    calib_tilt_pitch_deg: float = 12.0
    calib_pass_speeds_mps: Range = (0.2, 0.5)   # constant-speed passes (render/pose timing check)
    calib_pass_halfspan_m: float = 1.0
    calib_keepout_depth_m: float = 2.3       # prop-free depth in front of each board
    min_agent_separation_m: float = 1.0      # agents never closer (3D), else agent 1 starts later
    start_delay_step_s: float = 5.0
    max_start_delay_s: float = 60.0
    min_inter_pairs: int = 3
    min_inter_rooms: int = 2
    min_intra_pairs: int = 0


@dataclass(frozen=True)
class Hazards:
    """Structural hazards above a standing responder (design traversability-risk.md#risk-zones, D112): they threaten a route without
    blocking it, so GT traversability is unchanged. Opt-in: at the defaults (no hazards) this section is left out of
    Preset.to_canonical(), so every existing preset digest and scene id is unchanged."""
    cantilevers: tuple[int, int] = (0, 0)       # per scene
    cant_projection_m: Range = (0.8, 1.6)       # out from the wall
    cant_width_m: Range = (1.0, 2.0)            # along the wall
    cant_thickness_m: Range = (0.15, 0.25)
    cant_underside_m: Range = (1.95, 2.25)
    hung: tuple[int, int] = (0, 0)              # per scene
    hung_side_m: Range = (0.4, 0.8)
    hung_height_m: Range = (0.2, 0.35)
    hung_underside_m: Range = (1.95, 2.15)
    gap_m: float = 0.3                          # between hazards
    zone_margin_m: float = 0.5                  # GT risk_zone = footprint dilated by this [ASSUMPTION: debris fall]


@dataclass(frozen=True)
class Search:
    max_attempts: int = 200


@dataclass(frozen=True)
class Preset:
    name: str
    lattice: Lattice = field(default_factory=Lattice)
    rooms: Rooms = field(default_factory=Rooms)
    corridors: Corridors = field(default_factory=Corridors)
    topology: Topology = field(default_factory=Topology)
    entrances: Entrances = field(default_factory=Entrances)
    target: Target = field(default_factory=Target)
    clearance: Clearance = field(default_factory=Clearance)
    structure: Structure = field(default_factory=Structure)
    props: Props = field(default_factory=Props)
    appearance: Appearance = field(default_factory=Appearance)
    lighting: Lighting = field(default_factory=Lighting)
    motion: Motion = field(default_factory=Motion)
    routes: Routes = field(default_factory=Routes)
    search: Search = field(default_factory=Search)
    hazards: Hazards = field(default_factory=Hazards)

    def to_canonical(self) -> dict:
        d = dataclasses.asdict(self)
        # Fields added after presets were pinned are dropped at their default, so every older preset keeps its
        # digest (and its scene ids). D112: hazards; D131: lighting.shadows.
        if self.hazards == Hazards():
            del d["hazards"]
        if d["lighting"]["shadows"] == 0:
            del d["lighting"]["shadows"]
        return d

    def digest(self) -> str:
        blob = json.dumps(self.to_canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


_SECTION_TYPES = {"lattice": Lattice, "rooms": Rooms, "corridors": Corridors, "topology": Topology,
                  "entrances": Entrances, "target": Target, "clearance": Clearance,
                  "structure": Structure, "props": Props, "appearance": Appearance,
                  "lighting": Lighting, "motion": Motion, "routes": Routes, "search": Search,
                  "hazards": Hazards}


def _coerce(cls, name, value, default):
    """Coerce one YAML value to the type of the dataclass default it replaces."""
    where = f"{cls.__name__.lower()}.{name}"
    if isinstance(default, tuple):
        if not isinstance(value, list) or len(value) != len(default):
            raise PresetError(f"{where}: expected a list of {len(default)}")
        return tuple(_coerce(cls, name, v, d) for v, d in zip(value, default))
    if isinstance(default, str):
        if not isinstance(value, str):
            raise PresetError(f"{where}: expected a string")
        return value
    if isinstance(default, bool) or isinstance(value, bool):
        raise PresetError(f"{where}: booleans are not used in presets")
    if isinstance(default, int):
        if not isinstance(value, int):
            raise PresetError(f"{where}: expected an integer")
        return value
    if isinstance(default, float):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PresetError(f"{where}: expected a finite number")
        return float(value)
    raise PresetError(f"{where}: unsupported type")


def _section(cls, raw):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise PresetError(f"{cls.__name__.lower()}: expected a mapping")
    known = {f.name: f.default for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise PresetError(f"{cls.__name__.lower()}: unknown key(s) {sorted(unknown)}")
    return cls(**{k: _coerce(cls, k, v, known[k]) for k, v in raw.items()})


def parse_preset(doc: dict) -> Preset:
    if not isinstance(doc, dict) or "name" not in doc or not isinstance(doc["name"], str):
        raise PresetError("preset needs a string 'name'")
    unknown = set(doc) - set(_SECTION_TYPES) - {"name"}
    if unknown:
        raise PresetError(f"unknown section(s) {sorted(unknown)}")
    p = Preset(name=doc["name"], **{k: _section(t, doc.get(k)) for k, t in _SECTION_TYPES.items()})
    validate(p)
    return p


def load_preset(path) -> Preset:
    with open(path) as f:
        return parse_preset(yaml.safe_load(f))


def _range(where, r, lo=None, hi=None):
    if not r[0] <= r[1]:
        raise PresetError(f"{where}: range [{r[0]}, {r[1]}] is empty")
    if lo is not None and r[0] < lo:
        raise PresetError(f"{where}: lower bound {r[0]} < {lo}")
    if hi is not None and r[1] > hi:
        raise PresetError(f"{where}: upper bound {r[1]} > {hi}")


def validate(p: Preset) -> None:
    g = p.structure.grid_m
    hz = p.hazards
    if hz != Hazards() and (hz.cantilevers[1] > 0 or hz.hung[1] > 0):
        _validate_hazards(p, hz, g)
    _validate_rest(p, g)


def _validate_hazards(p: Preset, hz: Hazards, g: float) -> None:
    for name in ("cantilevers", "hung"):
        r = getattr(hz, name)
        if not 0 <= r[0] <= r[1] <= 8:
            raise PresetError(f"hazards.{name}: need 0 <= lo <= hi <= 8")
    for name in ("cant_projection_m", "cant_width_m", "cant_thickness_m", "hung_side_m", "hung_height_m"):
        _range(f"hazards.{name}", getattr(hz, name), lo=g)
    # Above a standing responder (1.8 m, design traversability-risk.md) and under the lowest ceiling.
    top = p.structure.wall_height_m[0] - 0.1
    _range("hazards.cant_underside_m", hz.cant_underside_m, lo=1.9, hi=top - hz.cant_thickness_m[1])
    _range("hazards.hung_underside_m", hz.hung_underside_m, lo=1.9, hi=top - hz.hung_height_m[1])
    if not (hz.gap_m >= 0 and hz.zone_margin_m >= 0):
        raise PresetError("hazards: gap_m and zone_margin_m must be >= 0")


def _validate_rest(p: Preset, g: float) -> None:
    if not 0.01 <= g <= 0.1:
        raise PresetError("structure.grid_m must be in [0.01, 0.1]")
    L = p.lattice
    if L.nx < 1 or L.ny < 1 or not 2 <= L.nx * L.ny <= 64:
        raise PresetError("lattice: nx, ny >= 1 and 2 <= nx*ny <= 64")
    for name, v in (("lattice.cell_m", L.cell_m), ("entrances.vestibule_m", p.entrances.vestibule_m)):
        if abs(v / g - round(v / g)) > 1e-9:
            raise PresetError(f"{name} must be a multiple of structure.grid_m")
    c = p.clearance
    if not 0 < c.rho_agent_m < c.rho_responder_m:
        raise PresetError("clearance: need 0 < rho_agent < rho_responder")
    # A corridor of width w on a g-grid has max clearance w/2 - g/2 (cell centres),
    # so passability at rho needs w >= 2*rho + g.
    w = p.corridors
    _range("corridors.width_m", w.width_m, lo=2 * c.rho_responder_m + g)
    if w.narrow_count < 0:
        raise PresetError("corridors.narrow_count must be >= 0")
    if w.narrow_count:
        # Agent passes (w >= 2*rho_agent + g); responder is blocked (w < 2*rho_responder + g,
        # i.e. w <= 2*rho_responder on the grid).
        _range("corridors.narrow_width_m", w.narrow_width_m, lo=2 * c.rho_agent_m + g,
               hi=2 * c.rho_responder_m)
    r = p.rooms
    if not 2 <= r.count[0] <= r.count[1] <= L.nx * L.ny:
        raise PresetError("rooms.count must satisfy 2 <= lo <= hi <= nx*ny")
    if p.target.allow_entrance_room not in (0, 1):
        raise PresetError("target.allow_entrance_room must be 0 or 1")
    need_rooms = p.entrances.count + (0 if p.target.allow_entrance_room else 1)
    if r.count[0] < need_rooms:
        raise PresetError("rooms.count lower bound leaves no target room besides the entrance rooms")
    _range("rooms.size_m", r.size_m)
    need = 2 * (w.width_m[1] / 2 + r.center_margin_m)
    if r.size_m[0] < need:
        raise PresetError(f"rooms.size_m lower bound must be >= {need:.2f} to contain corridor mouths")
    if r.size_m[1] > L.cell_m - 2 * 0.5:
        raise PresetError("rooms.size_m upper bound must leave >= 0.5 m of wall to the cell border")
    if not 1 <= p.entrances.count <= 3:
        raise PresetError("entrances.count must be 1..3")   # P8 D212: 3 agents; layout picks greedy-maximin rooms
    if p.entrances.count > 2 and p.routes.template == "cross":
        raise PresetError("routes.template 'cross' pairs two agents; use 'return' or 'loop' with 3 entrances")
    _range("target.path_length_m", p.target.path_length_m, lo=0.0)
    _range("structure.wall_height_m", p.structure.wall_height_m, lo=2.0)
    t = p.topology
    if t.route_cycle not in (0, 1):
        raise PresetError("topology.route_cycle must be 0 or 1")
    if t.cover_edges_max < 1 or t.extra_edges < 0 or t.dead_end_stubs < 0:
        raise PresetError("topology: cover_edges_max >= 1, other counts >= 0")
    if not t.route_cycle and (w.narrow_count or p.routes.template != "return"):
        raise PresetError("without topology.route_cycle only the 'return' template and no narrow "
                          "passages are possible")
    if p.search.max_attempts < 1:
        raise PresetError("search.max_attempts must be >= 1")
    pr = p.props
    if not 0 <= pr.per_room[0] <= pr.per_room[1]:
        raise PresetError("props.per_room must satisfy 0 <= lo <= hi")
    _range("props.size_along_m", pr.size_along_m, lo=g)
    _range("props.depth_m", pr.depth_m, lo=g)
    _range("props.height_m", pr.height_m, lo=0.1, hi=p.structure.wall_height_m[0] - 0.2)
    if pr.keepout_center_m < c.rho_responder_m + 0.5:
        raise PresetError("props.keepout_center_m must leave the survey spot responder-clear")
    a = p.appearance
    if not 0.0 <= a.tau <= 1.0:
        raise PresetError("appearance.tau must be in [0, 1]")
    if a.repetition != 0.0:
        raise PresetError("appearance.repetition > 0 is the Phase 3.5 aliasing lab; not implemented yet")
    if not (32 <= a.floor_px_per_m <= a.wall_px_per_m <= 512):
        raise PresetError("appearance: 32 <= floor_px_per_m <= wall_px_per_m <= 512")
    mo = p.motion
    for name in ("lk_budget_px", "focal_px", "frame_dt_s", "v_cap_mps", "a_max_mps2",
                 "omega_max_radps", "alpha_max_radps2"):
        if getattr(mo, name) <= 0:
            raise PresetError(f"motion.{name} must be > 0")
    if not (mo.altitude_jitter_m >= 0 and mo.altitude_m - mo.altitude_jitter_m >= 0.3
            and mo.altitude_m + mo.altitude_jitter_m <= p.structure.wall_height_m[0] - 0.3):
        raise PresetError("motion: altitude +- jitter must stay 0.3 m clear of floor and ceiling")
    ro = p.routes
    if ro.template not in ("cross", "loop", "return"):
        raise PresetError("routes.template must be 'cross', 'loop' or 'return'")
    if ro.surveys not in ("all", "target"):
        raise PresetError("routes.surveys must be 'all' or 'target'")
    if ro.template == "cross" and p.entrances.count != 2:
        raise PresetError("routes.template 'cross' needs two entrances")
    if min(ro.keyframe_step_m, ro.keyframe_step_deg, ro.overlap_radius_m, ro.overlap_yaw_deg) <= 0:
        raise PresetError("routes: steps and overlap tolerances must be > 0")
    # Offsets are drawn in a square, so the farthest survey spot is sqrt(2)*offset from the
    # cell centre; it must stay responder-clear of every prop (props keep out of that square).
    if not (ro.survey_offset_m >= 0
            and math.sqrt(2) * ro.survey_offset_m + c.rho_responder_m <= p.props.keepout_center_m):
        raise PresetError("routes.survey_offset_m: sqrt(2)*offset + rho_responder must be <= "
                          "props.keepout_center_m")
    if ro.calibration not in (0, 1):
        raise PresetError("routes.calibration must be 0 or 1")
    if ro.calibration:
        _range("routes.calib_standoffs_m", ro.calib_standoffs_m, lo=0.6, hi=ro.calib_keepout_depth_m - 0.3)
        _range("routes.calib_pass_speeds_mps", ro.calib_pass_speeds_mps, lo=0.05, hi=mo.v_cap_mps)
        if not 0.0 < ro.calib_pass_halfspan_m <= 2.0 or ro.calib_lateral_m < 0:
            raise PresetError("routes: calib_pass_halfspan_m in (0, 2], calib_lateral_m >= 0")
    if ro.min_agent_separation_m < 2 * c.rho_agent_m or ro.start_delay_step_s <= 0 \
            or ro.max_start_delay_s < 0:
        raise PresetError("routes: min_agent_separation_m >= 2*rho_agent, delay step > 0, max delay >= 0")
    if min(ro.min_inter_pairs, ro.min_inter_rooms, ro.min_intra_pairs) < 0:
        raise PresetError("routes: gate thresholds must be >= 0")
    li = p.lighting
    if not 0.0 <= li.ambient <= 1.0 or li.zone_lights not in (0, 1) or li.light_diffuse < 0 or li.shadows not in (0, 1):
        raise PresetError("lighting: ambient in [0,1], zone_lights 0/1, light_diffuse >= 0, shadows 0/1")
