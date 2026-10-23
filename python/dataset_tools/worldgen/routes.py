"""Scripted agent trajectories and the closure-opportunity gate (design §4.2-§4.3).

Route templates on the responder cell graph (narrow passages excluded: scripted
Phase 1 routes never traverse them):
  cross   agent k: entrance_k -> target (survey) -> the OTHER entrance room (survey)
          -> out through the other entrance to its vestibule start (vacated at t=0).
          Both agents survey the target and each other's start room: >= 2 shared rooms
          with 360-degree views, whatever direction each arrived from.
  loop    agent k: entrance_k -> target (survey) -> entrance_k by an edge-disjoint arc
          (2-edge-connectivity guarantees it) -> out to its own vestibule: intra-agent
          loops, for Phase 2.
  return  agent k: entrance_k -> target (survey) -> back the same way -> own vestibule.
          For maps without a route cycle (ci_tiny).
Every room on a route is entered at its lattice-cell centre (corridor axes meet there).

Motion: hold, then per waypoint a turn-in-place to the next heading and a straight
S-curve move (motion.py). Speed per move is limited by the tracker budget:
v <= lk_budget_px * clearance / (focal_px * frame_dt), clearance = min over the move of
the navigable ESDF and the floor/ceiling distance; yaw rate by the image-edge flow
2 * focal_px * omega * frame_dt <= lk_budget_px. Every sample of every move is asserted
at ESDF >= rho_agent (the collision envelope, checked offline here and again by the
mover). Altitude varies per move for IMU excitation.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np

from .layout import Layout, adjacency
from .assets import load_charuco
from .geometry import board_quad
from .motion import Profile, plan_profile, wrap_pi

S_ALTITUDE, S_SURVEY = 30, 31
GRAVITY = 9.80665


class RouteInfeasible(ValueError):
    """The scripted route cannot be flown in this scene (a rejection, not a bug)."""


@dataclass(frozen=True)
class Segment:
    """One timed primitive. Poses are (x, y, z, yaw, pitch), roll = 0 always.

    Orientation R = Rz(yaw) * Ry(pitch): pitch is right-handed about body +y (left),
    so positive pitch is nose DOWN. A "turn" moves (yaw, pitch) along a straight line
    in angle space; for this Euler pair the body rate is
        omega_B = (-yaw' sin(pitch), pitch', yaw' cos(pitch)),  |omega_B| = |(yaw', pitch')|,
    so the S-curve's rate limit on the angle-space path is exactly the angular-speed
    limit. A "move" translates on a straight 3D line at fixed orientation. "hold" rests.
    """
    kind: str                     # "hold" | "turn" | "move"
    t0: float
    profile: Profile
    duration: float
    start: tuple                  # (x, y, z, yaw, pitch); turns store UNWRAPPED end angles
    end: tuple
    tag: str = ""

    def pose(self, t: float):
        """(x,y,z,yaw,pitch), their rates, and their second derivatives at absolute time t."""
        tl = min(max(t - self.t0, 0.0), self.duration)
        if self.kind == "hold":
            return self.start, (0.0,) * 5, (0.0,) * 5
        u, du, ddu = self.profile.eval(tl)
        D = self.profile.D
        k, kd, kdd = u / D, du / D, ddu / D
        d = [e - s_ for e, s_ in zip(self.end, self.start)]
        if self.kind == "turn":
            x, y, z, h0, p0 = self.start
            return ((x, y, z, h0 + k * d[3], p0 + k * d[4]),
                    (0.0, 0.0, 0.0, kd * d[3], kd * d[4]), (0.0, 0.0, 0.0, kdd * d[3], kdd * d[4]))
        x0, y0, z0, h0, p0 = self.start
        return ((x0 + k * d[0], y0 + k * d[1], z0 + k * d[2], h0, p0),
                (kd * d[0], kd * d[1], kd * d[2], 0.0, 0.0), (kdd * d[0], kdd * d[1], kdd * d[2], 0.0, 0.0))


@dataclass
class AgentScript:
    agent: int
    segments: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        s = self.segments[-1]
        return s.t0 + s.duration

    def pose(self, t: float):
        """The first segment whose end is >= t (bisect on segment end times; the same
        choice the linear scan made, so output is unchanged)."""
        ends = self._ends()
        i = bisect.bisect_left(ends, t)
        return self.segments[min(i, len(self.segments) - 1)].pose(t)

    def _ends(self):
        if getattr(self, "_ends_cache", None) is None or len(self._ends_cache) != len(self.segments):
            self._ends_cache = [s.t0 + s.duration for s in self.segments]
        return self._ends_cache

    def to_json(self) -> dict:
        return {"agent": self.agent, "duration_s": self.duration, "gravity_mps2": GRAVITY,
                "segments": [{"kind": s.kind, "t0": s.t0, "duration": s.duration, "tag": s.tag,
                              "start": list(s.start), "end": list(s.end),
                              "D": s.profile.D, "Ta": s.profile.Ta, "vc": s.profile.vc}
                             for s in self.segments]}


def imu(script: AgentScript, t: float):
    """Ideal IMU in the body frame (x forward, y left, z up): gyro, specific force.
    omega_B as in Segment; f_B = Ry(pitch)^T Rz(yaw)^T (a_W + g e_z)."""
    (_, _, _, h, p), (_, _, _, hd, pd), (ax, ay, az, _, _) = script.pose(t)
    ch, sh, cp, sp = math.cos(h), math.sin(h), math.cos(p), math.sin(p)
    gyro = (-hd * sp, pd, hd * cp)
    wx, wy, wz = ch * ax + sh * ay, -sh * ax + ch * ay, az + GRAVITY      # Rz^T (a + g)
    return gyro, (cp * wx - sp * wz, wy, sp * wx + cp * wz)             # Ry^T


# --- route planning ---------------------------------------------------------------

def disjoint_pair(adj: dict, L: Layout, s, t):
    """The two edge-disjoint s->t paths of least total length (Suurballe's problem as a
    min-cost flow of 2 units): successive shortest augmenting paths (Bellman-Ford, since
    residual arcs carry negative cost) on arcs u->v and v->u of every undirected edge,
    then opposite flows on one edge cancel and the flow decomposes into two paths.
    Returns (shorter, longer) as cell lists; raises ValueError if not 2-edge-connected."""
    cost = {}
    for u in adj:
        for v in adj[u]:
            cost[(u, v)] = math.dist(L.center(u), L.center(v))
    flow = {a: 0 for a in cost}
    nodes = sorted(adj)
    for _ in range(2):
        dist = {n: math.inf for n in nodes}
        prev = {}
        dist[s] = 0.0
        for _ in range(len(nodes)):
            changed = False
            for (u, v), c in sorted(cost.items()):
                if dist[u] == math.inf:
                    continue
                if flow[(u, v)] == 0 and flow[(v, u)] == 0 and dist[u] + c < dist[v] - 1e-12:
                    dist[v], prev[v], changed = dist[u] + c, (u, v, +1), True
                elif flow[(v, u)] == 1 and dist[u] - c < dist[v] - 1e-12:
                    dist[v], prev[v], changed = dist[u] - c, (u, v, -1), True   # undo v->u
            if not changed:
                break
        if dist[t] == math.inf:
            raise ValueError("endpoints are not 2-edge-connected")
        v = t
        while v != s:
            u, _, sign = prev[v]
            if sign > 0:
                flow[(u, v)] = 1
            else:
                flow[(v, u)] = 0
            v = u
    paths = []
    used = {a for a, f in flow.items() if f == 1}
    for _ in range(2):
        path, u = [s], s
        while u != t:
            nxt = sorted(b for (a, b) in used if a == u)[0]
            used.discard((u, nxt))
            path.append(nxt)
            u = nxt
        paths.append(path)
    length = lambda P: sum(math.dist(L.center(a), L.center(b)) for a, b in zip(P, P[1:]))
    paths.sort(key=lambda P: (length(P), P))
    return paths[0], paths[1]


def shortest_path(adj: dict, L: Layout, s, t) -> list:
    """Least-length cell path (Dijkstra, deterministic tie-break by cell order)."""
    import heapq
    dist, prev, heap = {s: 0.0}, {s: None}, [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == t:
            break
        if d > dist[u]:
            continue
        for v in adj[u]:
            nd = d + math.dist(L.center(u), L.center(v))
            if nd < dist.get(v, math.inf) - 1e-12:
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if t not in prev:
        raise ValueError("no route")
    out, v = [], t
    while v is not None:
        out.append(v)
        v = prev[v]
    return out[::-1]


def room_sequence(scene, k: int) -> list[tuple[list, str]]:
    """Legs as lists of lattice cells, each ending at a survey room.

    For each entrance j, (A_j, B_j) is the least-length edge-disjoint pair of routes
    from entrance j to the target. Agent j goes out on A_j.
      cross: agent k returns along reversed B_{other}, the route edge-disjoint from the
             other agent's outbound A_{other}: no head-on meeting in any corridor.
      loop:  agent k returns along reversed B_k.
    """
    L = scene.layout
    resp = {e for e in L.edges if not L.passages[e].narrow}
    adj = adjacency(L.cells, resp)
    T = scene.target_room
    template = scene.preset.routes.template
    if template == "return":
        out = shortest_path(adj, L, L.entrances[k].room, T)
        return [(out, "leg:0"), (out[::-1], "leg:1")]
    pairs = [disjoint_pair(adj, L, ent.room, T) if ent.room != T else ([T], [T])
             for ent in L.entrances]
    A_k, B_k = pairs[k]
    if scene.preset.routes.template == "cross":
        other = (k + 1) % len(L.entrances)
        back = pairs[other][1][::-1]
    else:
        back = B_k[::-1]
    return [(A_k, "leg:0"), (back, "leg:1")]


def _clear_point(scene, x: float, y: float, margin: float = 0.1) -> bool:
    """ESDF at (x, y) >= rho_agent + margin (inside the grid)."""
    g = scene.grid
    r, c = int(math.floor(y / g.g)) - g.oy, int(math.floor(x / g.g)) - g.ox
    if not (0 <= r < g.height and 0 <= c < g.width):
        return False
    return float(scene.esdf_m[r, c]) >= scene.preset.clearance.rho_agent_m + margin


def _clearance_along(scene, p, q, z) -> float:
    """Min over samples every grid cell of the navigable ESDF and vertical clearance;
    asserts the collision envelope (ESDF >= rho_agent)."""
    g = scene.grid
    n = max(int(math.ceil(math.dist(p, q) / g.g)), 1)
    rho_a = scene.preset.clearance.rho_agent_m
    cmin = math.inf
    for i in range(n + 1):
        x = p[0] + (q[0] - p[0]) * i / n
        y = p[1] + (q[1] - p[1]) * i / n
        r = int(math.floor(y / g.g)) - g.oy
        c = int(math.floor(x / g.g)) - g.ox
        e = float(scene.esdf_m[r, c])
        if e < rho_a - 1e-9:
            raise RouteInfeasible(f"route leaves the collision envelope at ({x:.2f}, {y:.2f})")
        cmin = min(cmin, e)
    vert = min(z[0], z[1], scene.wall_height_m - max(z))
    return min(cmin, vert)


def separation(scripts, dt: float = 0.05) -> float:
    """Minimum 3D distance between any two agents over the mission (sampled every dt; an
    agent stays at its final pose after its script ends)."""
    T = max(s.duration for s in scripts)
    best = math.inf
    t = 0.0
    while t <= T + 1e-9:
        ps = [s.pose(t)[0][:3] for s in scripts]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                best = min(best, math.dist(ps[i], ps[j]))
        t += dt
    return best


def schedule(scene) -> tuple[list, list, float]:
    """Scripts with the smallest start delay (multiples of start_delay_step_s up to
    max_start_delay_s), tried first on agent 1 then on agent 0, that keeps the agents at
    least min_agent_separation_m apart. Returns (scripts, [delay_0, delay_1],
    min_separation_m); raises ValueError if no delay works."""
    R = scene.preset.routes
    if len(scene.layout.entrances) == 1:
        return [build_script(scene, 0)], [0.0], math.inf
    steps = int(math.floor(R.max_start_delay_s / R.start_delay_step_s + 1e-9))
    delays = [(0.0, 0.0)] + [(0.0, i * R.start_delay_step_s) for i in range(1, steps + 1)] \
        + [(i * R.start_delay_step_s, 0.0) for i in range(1, steps + 1)]
    cache = {}

    def script(k, d):
        if (k, d) not in cache:
            cache[(k, d)] = build_script(scene, k, start_delay_s=d)
        return cache[(k, d)]

    if len(scene.layout.entrances) == 2:   # the original pair search, byte-identical scenes for every old preset
        for d0, d1 in delays:
            pair = [script(0, d0), script(1, d1)]
            sep = separation(pair)
            if sep >= R.min_agent_separation_m:
                return pair, [d0, d1], sep
        raise ValueError("agents come closer than min_agent_separation_m at every start delay")
    # N >= 3 (P8 D212): greedy - agent 0 starts at 0; each later agent takes the smallest delay that keeps the
    # whole fleet min_agent_separation_m apart from the scripts already fixed.
    out = [script(0, 0.0)]
    chosen = [0.0]
    for k in range(1, len(scene.layout.entrances)):
        for i in range(0, steps + 1):
            d = i * R.start_delay_step_s
            cand = out + [script(k, d)]
            if separation(cand) >= R.min_agent_separation_m:
                out, chosen = cand, chosen + [d]
                break
        else:
            raise ValueError(f"agent {k}: closer than min_agent_separation_m at every start delay")
    return out, chosen, separation(out)


def build_script(scene, k: int, start_delay_s: float = 0.0) -> AgentScript:
    P = scene.preset
    M = P.motion
    L = scene.layout
    g = scene.grid.g
    v_lk = M.lk_budget_px / (M.focal_px * M.frame_dt_s)          # (m/s) per metre of clearance
    w_max = min(M.omega_max_radps, M.lk_budget_px / (2.0 * M.focal_px * M.frame_dt_s))
    st = L.stream(S_ALTITUDE, k)
    ent = L.entrances[k]
    x, y = ent.start[0] * g, ent.start[1] * g
    z = M.altitude_m
    h = math.radians(ent.start_yaw_deg)
    pt = 0.0                                   # pitch
    segs = []
    t = 0.0

    def add(kind, prof, dur, start, end, tag=""):
        nonlocal t
        segs.append(Segment(kind, t, prof, dur, start, end, tag))
        t += dur

    add("hold", Profile(0.0, 0.0, 0.0), M.hold_start_s + start_delay_s, (x, y, z, h, pt),
        (x, y, z, h, pt), "hold:start")

    def look(target_yaw, target_pitch, tag=""):
        """Turn in place to (yaw, pitch) along the shorter yaw direction."""
        nonlocal h, pt
        dh = wrap_pi(target_yaw - h)
        dp = target_pitch - pt
        D = math.hypot(dh, dp)
        if D < 1e-9:
            return
        prof = plan_profile(D, w_max, M.alpha_max_radps2)
        add("turn", prof, prof.T, (x, y, z, h, pt), (x, y, z, h + dh, pt + dp), tag)
        h, pt = wrap_pi(h + dh), target_pitch

    def turn_to(target_yaw, tag=""):
        look(target_yaw, pt, tag)

    def spin(tag):
        nonlocal h
        prof = plan_profile(2.0 * math.pi, w_max, M.alpha_max_radps2)
        add("turn", prof, prof.T, (x, y, z, h, pt), (x, y, z, h + 2.0 * math.pi, pt), tag)
        h = wrap_pi(h + 2.0 * math.pi)

    def survey(cell):
        """Step to a seeded per-agent offset spot, spin 360 degrees, step back to the
        cell centre (corridor axes meet there). Offsets make the two agents' views of
        a shared room differ by a real baseline, not only by altitude."""
        tag = room_tag(cell)
        cx_m, cy_m = x, y
        r = P.routes.survey_offset_m
        if r > 0:
            so = L.stream(S_SURVEY, k, cell[0], cell[1])
            # uniform in the square [-r, r]^2 (no trig in the determinism path)
            ox, oy = so.uniform(-r, r), so.uniform(-r, r)
            move_to(cx_m + ox, cy_m + oy, tag, face=False)
        spin(tag)
        if r > 0:
            move_to(cx_m, cy_m, tag, face=False)

    def move_to(nx, ny, tag="", face=True, v_limit=math.inf, nz=None):
        """Straight move; face=True turns to the travel direction first. The body is
        holonomic, so a short survey step may keep its heading (face=False). nz fixes
        the end altitude (default: seeded jitter). v_limit caps the cruise speed."""
        nonlocal x, y, z
        D2 = math.dist((x, y), (nx, ny))
        if D2 < 1e-9:
            return
        if face:
            turn_to(math.atan2(ny - y, nx - x))
        if nz is None:
            lo, hi = M.altitude_m - M.altitude_jitter_m, M.altitude_m + M.altitude_jitter_m
            nz = round(st.uniform(lo, hi), 4)
        c = _clearance_along(scene, (x, y), (nx, ny), (z, nz))
        v = min(M.v_cap_mps, v_lk * c, v_limit)
        D = math.sqrt(D2 * D2 + (nz - z) ** 2)
        prof = plan_profile(D, v, M.a_max_mps2)
        add("move", prof, prof.T, (x, y, z, h, pt), (nx, ny, nz, h, pt), tag)
        x, y, z = nx, ny, nz

    def room_tag(c):
        return f"survey:room_{c[0]}_{c[1]}"

    # Into the entrance room, survey, then the legs.
    surveyed = set()

    def maybe_survey(cell):
        if cell in surveyed:
            return                      # never survey one room twice in a row (target == entrance)
        if P.routes.surveys == "all" or cell == scene.target_room:
            survey(cell)
            surveyed.add(cell)

    def hold(dur, tag):
        add("hold", Profile(0.0, 0.0, 0.0), dur, (x, y, z, h, pt), (x, y, z, h, pt), tag)

    def calibration_dance():
        """ChArUco views with orientation diversity, then constant-speed passes (design §6)."""
        R = P.routes
        bp = scene.boards[k]
        q = board_quad(bp, g, load_charuco().size_m[0])
        n = q.normal()                                   # into the room
        lat = np.asarray(q.eu, float) / np.linalg.norm(q.eu)
        bc = np.asarray(q.p0) + 0.5 * np.asarray(q.eu) + 0.5 * np.asarray(q.ev)
        face_yaw = math.atan2(-n[1], -n[0])              # looking at the board
        z_view = M.altitude_m

        def at(d, t):
            return float(bc[0] + n[0] * d + lat[0] * t), float(bc[1] + n[1] * d + lat[1] * t)

        # Near-left, near-right, far-centre; at each, the aimed view plus two DIAGONAL
        # tilts (yaw and pitch together): out-of-plane board orientations in both axes,
        # which Zhang calibration needs (views rotated about one axis only are degenerate).
        views = [(R.calib_standoffs_m[0], -R.calib_lateral_m), (R.calib_standoffs_m[0], R.calib_lateral_m),
                 (R.calib_standoffs_m[1], 0.0)]
        ty, tp = math.radians(R.calib_tilt_yaw_deg), math.radians(R.calib_tilt_pitch_deg)
        for d, t in views:
            px, py = at(d, t)
            if not _clear_point(scene, px, py):
                raise RouteInfeasible("calibration view position not clear")
            move_to(px, py, "calib:move", nz=z_view)
            yaw0 = math.atan2(bc[1] - py, bc[0] - px)
            pitch0 = math.atan2(z - bc[2], math.hypot(bc[0] - px, bc[1] - py))   # + = nose down
            look(yaw0, pitch0, "calib:look")
            hold(0.5, "calib:view")
            for dy, dp in ((ty, tp), (-ty, -tp)):
                look(yaw0 + dy, pitch0 + dp, "calib:look")
                hold(0.3, "calib:view")
            look(yaw0, 0.0, "calib:look")
        # Constant-speed passes parallel to the board, facing it, clamped to the clear span.
        d = 0.5 * (R.calib_standoffs_m[0] + R.calib_standoffs_m[1])
        span = []
        for sign in (-1.0, 1.0):
            t = 0.0
            while t + 0.05 <= R.calib_pass_halfspan_m and _clear_point(scene, *at(d, sign * (t + 0.05))):
                t += 0.05
            span.append(sign * t)
        if span[1] - span[0] < 1.2:
            raise RouteInfeasible("calibration passes: less than 1.2 m of clear span")
        for i, v in enumerate(R.calib_pass_speeds_mps):
            a_, b_ = (span[0], span[1]) if i % 2 == 0 else (span[1], span[0])
            move_to(*at(d, a_), "calib:move", nz=z_view)
            look(face_yaw, 0.0, "calib:look")
            move_to(*at(d, b_), f"calib:cruise:{v:.2f}", face=False, v_limit=v, nz=z_view)

    cx, cy = L.center(ent.room)
    move_to(cx * g, cy * g, "enter")
    if P.routes.calibration and k == 0:       # one rig model: agent 0's dance validates both
        calibration_dance()
        move_to(cx * g, cy * g, "calib:return")
    maybe_survey(ent.room)
    for cells, tag in room_sequence(scene, k):
        pts = [(L.center(c)[0] * g, L.center(c)[1] * g) for c in cells]
        if len(pts) > 1:
            for p in _merge_collinear(pts)[1:]:
                move_to(*p, tag)
        maybe_survey(cells[-1])
    # Leave the structure and park at a vestibule start: cross exits through the OTHER
    # entrance (vacated by its agent at the beginning), return/loop through its own.
    exit_ent = L.entrances[(k + 1) % len(L.entrances)] if P.routes.template == "cross" else ent
    move_to(exit_ent.start[0] * g, exit_ent.start[1] * g, "exit")
    add("hold", Profile(0.0, 0.0, 0.0), M.hold_end_s, (x, y, z, h, pt), (x, y, z, h, pt), "hold:end")
    return AgentScript(k, segs)


def _merge_collinear(pts):
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-9:
            out.append(b)
    out.append(pts[-1])
    return out


# --- closure opportunities ---------------------------------------------------------

def keyframes(script: AgentScript, step_m: float, step_deg: float, dt: float = 0.05):
    """Poses sampled like a keyframe policy: a new one after step_m or step_deg."""
    out = []
    t, T = 0.0, script.duration
    last = None
    while t <= T + 1e-9:
        (x, y, z, h, _), _, _ = script.pose(t)
        if last is None or math.dist(last[1:3], (x, y)) >= step_m or \
                abs(wrap_pi(h - last[3])) >= math.radians(step_deg):
            last = (t, x, y, h)
            out.append(last)
        t += dt
    return out


def line_of_sight(free: np.ndarray, grid, p, q) -> bool:
    """All grid cells along the segment p-q (metres) are free (sampled at 1/4 cell)."""
    n = max(int(math.ceil(math.dist(p, q) / grid.g * 4)), 1)
    for i in range(n + 1):
        x = p[0] + (q[0] - p[0]) * i / n
        y = p[1] + (q[1] - p[1]) * i / n
        r = int(math.floor(y / grid.g)) - grid.oy
        c = int(math.floor(x / grid.g)) - grid.ox
        if not free[r, c]:
            return False
    return True


def zone_at(L: Layout, x_units: float, y_units: float):
    ranked = sorted(L.free_rects(), key=lambda t: {"room": 0, "vestibule": 1, "passage": 2}[t[0]])
    for kind, owner, r in ranked:
        if r.x0 <= x_units <= r.x1 and r.y0 <= y_units <= r.y1:
            return kind, owner
    return None


def opportunities(scene, scripts) -> dict:
    R = scene.preset.routes
    kfs = [keyframes(s, R.keyframe_step_m, R.keyframe_step_deg) for s in scripts]
    g = scene.grid.g
    yaw_tol = math.radians(R.overlap_yaw_deg)

    def pairs(A, B, min_dt):
        out = []
        for ta, xa, ya, ha in A:
            for tb, xb, yb, hb in B:
                if abs(ta - tb) < min_dt:
                    continue
                if math.dist((xa, ya), (xb, yb)) > R.overlap_radius_m:
                    continue
                if abs(wrap_pi(ha - hb)) > yaw_tol:
                    continue
                if not line_of_sight(scene.free, scene.grid, (xa, ya), (xb, yb)):
                    continue
                out.append(zone_at(scene.layout, (xa + xb) / 2 / g, (ya + yb) / 2 / g))
        return out

    inter = pairs(kfs[0], kfs[1], 0.0) if len(kfs) > 1 else []
    intra = [pairs(k, k, R.revisit_min_dt_s) for k in kfs]
    rooms = sorted({z[1] for z in inter if z and z[0] == "room"})
    return {"keyframes": [len(k) for k in kfs], "inter_pairs": len(inter),
            "inter_rooms": [list(r) for r in rooms],
            "inter_zones": len({z for z in inter if z}),
            "intra_pairs": [len(p) // 2 for p in intra]}   # symmetric pairs counted once


def gate(scene, opp: dict) -> str | None:
    """None if the scene passes the closure-opportunity gate, else the reason."""
    R = scene.preset.routes
    if opp["inter_pairs"] < R.min_inter_pairs:
        return "too few inter-agent closure opportunities"
    if len(opp["inter_rooms"]) < R.min_inter_rooms:
        return "inter-agent opportunities in too few rooms"
    if any(n < R.min_intra_pairs for n in opp["intra_pairs"]):
        return "too few intra-agent revisits"
    return None
