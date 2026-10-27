"""Beats 1-2 (≈ 40 s): the seeded maze rises in perspective, a red beacon marks the casualty, the camera glides to a top
view as three robots appear at three doors, radio shading, the robots set off; then the title over the dimmed maze.

Data: maze_demo_0 scene.json + scene_gt.npz (walls = the free/occupied boundary of the scene grid, starts, target).
Illustrative: the radio shading (line of sight ray-cast on that grid; the live RF model, D102, attenuates through
walls) and the robots' opening moves (scripted along the scene's ground-truth person-width routes). No numbers.

  python/video/render.sh scenes/s01_worldgen.py WorldGen
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DEGREES, DOWN, RIGHT, UP, AnimationGroup, Circle, Create, DashedVMobject, Dot, FadeIn, FadeOut,
                   GrowFromCenter, LaggedStart, Line, MoveAlongPath, Polygon, Text, ThreeDScene, TracedPath,
                   Succession, Transform, VGroup, Wait, linear, smooth)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, WALL, CaptionedScene, tag  # noqa: E402

RADIO_RANGE_M = 9.0  # illustrative LOS radius for the shading only
WALL_H = 2.95  # scene.json wall_height_m
LIFT = UP * 0.55  # the top view sits above the subtitle band
FRONT = dict(phi=64 * DEGREES, theta=-180 * DEGREES, zoom=1.75)  # angled perspective from the front (south)
SIDE = dict(phi=64 * DEGREES, theta=-90 * DEGREES, zoom=1.75)  # the same perspective after a 90° anticlockwise turn
MOVE_M = 1.6  # how far each robot advances down its starting hall on the closing question


def los_polygon(sc, x0: float, y0: float, n: int = 240, rmax: float = RADIO_RANGE_M) -> np.ndarray:
    """Points where n rays from (x0, y0) first leave free space on the scene grid (or reach rmax)."""
    ox, oy = sc.origin_xy_m
    rows, cols = sc.free.shape
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        d = np.arange(0, rmax, sc.cell_m * 0.5)
        xs, ys = x0 + d * np.cos(th), y0 + d * np.sin(th)
        c = ((xs - ox) / sc.cell_m).astype(int)
        r = ((ys - oy) / sc.cell_m).astype(int)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        blocked = ~inside | ~sc.free[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)]
        k = int(np.argmax(blocked)) if blocked.any() else len(d) - 1
        pts.append((xs[k], ys[k]))
    return np.array(pts)


def head_m(xy: np.ndarray, metres: float) -> np.ndarray:
    """The first `metres` of a polyline."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return xy[d <= metres]


class WorldGen(CaptionedScene, ThreeDScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc, center=(0, 0, 0))  # world built about the origin; the camera orbits it
        w = np.array(sc.walls)
        x0, x1 = w[:, [0, 2]].min() - 1.0, w[:, [0, 2]].max() + 1.0
        y0, y1 = w[:, [1, 3]].min() - 1.0, w[:, [1, 3]].max() + 1.0

        def quad(s, h):
            a, b, c, d = s
            return Polygon(f.p(a, b, 0), f.p(c, d, 0), f.p(c, d, h), f.p(a, b, h), stroke_color=WALL,
                           stroke_width=1.6, fill_color=WALL, fill_opacity=0.10)

        order = sorted(range(len(sc.walls)), key=lambda i: w[i, [0, 2]].mean() + w[i, [1, 3]].mean())
        flat = [quad(sc.walls[i], 0.02) for i in order]
        full = [quad(sc.walls[i], WALL_H) for i in order]
        walls = VGroup(*flat)
        dots = VGroup(*[Dot(f.p(x, y), radius=0.012, color=DIM)
                        for x in np.arange(np.ceil(x0), x1, 1.0) for y in np.arange(np.ceil(y0), y1, 1.0)])
        seed = Text(f"maze generated procedurally (seed {sc.seed})", font=FONT, font_size=18, color=DIM)
        seed.to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        self.add_fixed_in_frame_mobjects(seed)

        # 1. the floor plan from the front in angled perspective; it turns 90° anticlockwise to the side as the walls rise
        cam = self.camera

        def camera_to(run_time, phi, theta, zoom):
            return [tr.animate(run_time=run_time, rate_func=smooth).set_value(v)
                    for tr, v in ((cam.phi_tracker, phi), (cam.theta_tracker, theta), (cam.zoom_tracker, zoom))]

        self.set_camera_orientation(**FRONT)
        self.add(dots, walls)
        self.wait(0.6)
        self.say("Imagine a building after an earthquake.",
                 run_with=[FadeIn(seed), *camera_to(4.2, **SIDE),
                           LaggedStart(*[Transform(a, b) for a, b in zip(flat, full)], lag_ratio=0.035, run_time=4.2)])

        # 2. straight from the turn into a smooth glide to the top view; the beacon rises on the way down, the 3D walls
        # cross-fade to the flat 2D map in the glide's last 30 %, and the beacon becomes a flat 2D target on landing.
        # The lift above the subtitle band is a camera frame-centre move, so nothing in the world has to move.
        tx, ty = sc.target_xy_m
        ring = Circle(radius=0.13, color=TARGET, stroke_width=3).move_to(f.p(tx, ty))
        beam = Line(f.p(tx, ty, 0), f.p(tx, ty, WALL_H * 1.6), color=TARGET, stroke_width=4)
        halo = Line(f.p(tx, ty, 0), f.p(tx, ty, WALL_H * 1.6), color=TARGET, stroke_width=16, stroke_opacity=0.18)
        pulse = Circle(radius=0.13, color=TARGET, stroke_width=2).move_to(f.p(tx, ty))
        target2d = Dot(f.p(tx, ty), radius=0.04, color=TARGET)
        flat_map = maze_walls(sc, f, width=3)
        late = lambda t: smooth(min(1.0, max(0.0, (t - 0.7) / 0.3)))  # noqa: E731
        glide = 3.8
        self.say("The lights are out, and somewhere inside, someone may need help.",
                 run_with=[FadeIn(ring, scale=1.6, run_time=0.8), GrowFromCenter(halo, run_time=1.2),
                           GrowFromCenter(beam, run_time=1.2),
                           pulse.animate(run_time=2.2, rate_func=linear).scale(5).set_stroke(opacity=0),
                           *camera_to(glide, phi=0, theta=-90 * DEGREES, zoom=1.0),
                           cam._frame_center.animate(run_time=glide, rate_func=smooth).move_to(-LIFT),
                           FadeOut(walls, run_time=glide, rate_func=late), FadeIn(flat_map, run_time=glide, rate_func=late),
                           *[Succession(Wait(0.7 * glide), FadeOut(m, run_time=0.3 * glide)) for m in (beam, halo)],
                           FadeIn(target2d, run_time=glide, rate_func=late)])
        self.remove(pulse)
        beam.set_opacity(0)
        halo.set_opacity(0)

        g = f  # top view: same frame (the camera carries the lift)
        bots = VGroup(*[agent_marker(e["agent"], g, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]))
                        for e in sc.starts])
        self.say("Before we send people in, we send three small robots.",
                 run_with=[LaggedStart(*[GrowFromCenter(b) for b in bots], lag_ratio=0.35, run_time=1.6)])

        radios = VGroup()
        for e in sc.starts:
            x, y = e["start_xy_m"]
            col = AGENT[e["agent"]]
            poly = Polygon(*[g.p(px, py) for px, py in los_polygon(sc, x, y)], stroke_width=0,
                           fill_color=col, fill_opacity=0.16)
            rr = DashedVMobject(Circle(radius=g.length(RADIO_RANGE_M), color=col, stroke_width=1,
                                       stroke_opacity=0.35).move_to(g.p(x, y)), num_dashes=60)
            radios.add(VGroup(poly, rr))
        chip = tag("illustrative").next_to(seed, DOWN, 0.15).align_to(seed, RIGHT)
        self.add_fixed_in_frame_mobjects(chip)
        self.say("They have no map, no GPS, and their radios barely reach through the walls.",
                 run_with=[AnimationGroup(*[FadeIn(r) for r in radios], lag_ratio=0.2, run_time=1.2), FadeIn(chip),
                           VGroup(*[r[1] for r in radios]).animate(run_time=1.0, rate_func=lambda t: max(0, t * 2 - 1))
                           .set_stroke(opacity=0)])

        # 3. they set off (scripted: the first metres of each robot's person-width route)
        moves, trails = [], VGroup()
        for e, b in zip(sc.starts, bots):
            route = sc.gt_routes_responder[e["agent"]]
            path = polyline(g, head_m(route, MOVE_M), AGENT[e["agent"]])
            moves.append(MoveAlongPath(b, path, rate_func=linear, run_time=4.5))
            trails.add(TracedPath(b.get_center, stroke_color=AGENT[e["agent"]], stroke_width=3, stroke_opacity=0.8))
        self.add(trails)
        hint = DashedVMobject(polyline(g, sc.gt_routes_responder[0], THREAD, width=4, opacity=0.8), num_dashes=70)
        self.say("Can they put what each one sees together, and find a safe way in and out?",
                 run_with=[*moves, Create(hint, run_time=2.5), VGroup(*[r[0] for r in radios]).animate(run_time=4.5).set_opacity(0.05)])
        self.clear_caption()

        # beat 2: the title over the dimmed maze
        title = Text("Ariadne", font=FONT, font_size=84, color=TEXT, weight="BOLD").shift(UP * 0.5)
        sub = Text("collaborative mapping and egress for a team of robots", font=FONT, font_size=26,
                   color=DIM).next_to(title, DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(title, sub)
        self.remove(title, sub)
        self.play(flat_map.animate.set_stroke(opacity=0.15), VGroup(dots, ring, target2d, bots).animate.set_opacity(0.15),
                  VGroup(trails, hint).animate.set_stroke(opacity=0.15), FadeOut(radios), FadeOut(chip), FadeOut(seed), run_time=0.8)
        self.say("This is Ariadne, named for the thread that led a hero out of the labyrinth.",
                 run_with=[FadeIn(title, shift=UP * 0.2, run_time=1.0), FadeIn(sub, run_time=1.2)])
        # the caption and the whole splash leave together, quickly
        self.play(FadeOut(self._caption), *[FadeOut(m) for m in (dots, flat_map, ring, target2d, bots, trails, hint,
                                                                 title, sub)], run_time=0.4)
        self._caption = None
