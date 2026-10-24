"""Beats 1-2 (≈ 36 s): the seeded WorldGen maze draws itself; the casualty; three robots at three doors; radio
shading; the question; then the title over the dimmed maze.

Data: maze_demo_0 scene.json + scene_gt.npz (walls = the free/occupied boundary of the scene grid, starts, target,
the GT responder route from entrance 0 used as the faint "is there a way?" hint). The radio shading is the
line-of-sight region from each start, ray-cast on that grid: illustrative (the live RF model, D102, attenuates
through walls rather than stopping at them, which the faint full-range ring stands for). No numbers.

  python/video/render.sh scenes/s01_worldgen.py WorldGen
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (AnimationGroup, Circle, Create, DashedVMobject, Dot, FadeIn, FadeOut, GrowFromCenter,
                   LaggedStart, Polygon, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import (AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag,  # noqa: E402
                                 UP, DOWN, RIGHT)

RADIO_RANGE_M = 9.0  # illustrative LOS radius for the shading only


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


class WorldGen(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc)
        w = np.array(sc.walls)
        x0, x1 = w[:, [0, 2]].min() - 1.0, w[:, [0, 2]].max() + 1.0
        y0, y1 = w[:, [1, 3]].min() - 1.0, w[:, [1, 3]].max() + 1.0

        seed = Text(f"procedurally generated · seed {sc.seed}", font=FONT, font_size=18, color=DIM)
        seed.to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        dots = VGroup(*[Dot(f.p(x, y), radius=0.012, color=DIM)
                        for x in np.arange(np.ceil(x0), x1, 1.0) for y in np.arange(np.ceil(y0), y1, 1.0)])
        # walls appear in a diagonal sweep: the generator "laying" the maze from the SW corner
        order = sorted(range(len(sc.walls)), key=lambda i: w[i, [0, 2]].mean() + w[i, [1, 3]].mean())
        walls = maze_walls(sc, f)
        swept = [walls[i] for i in order]

        self.play(FadeIn(dots, run_time=0.8), FadeIn(seed))
        self.say("Imagine a building after an earthquake.",
                 run_with=[LaggedStart(*[Create(m) for m in swept], lag_ratio=0.04, run_time=3.4)])

        tx, ty = sc.target_xy_m
        target = VGroup(Circle(radius=0.13, color=TARGET, stroke_width=3).move_to(f.p(tx, ty)),
                        Dot(f.p(tx, ty), radius=0.04, color=TARGET))
        pulse = Circle(radius=0.13, color=TARGET, stroke_width=2).move_to(f.p(tx, ty))
        self.say("The lights are out, and somewhere inside, someone may need help.",
                 run_with=[FadeIn(target, scale=1.6, run_time=1.0),
                           pulse.animate(run_time=2.0, rate_func=lambda t: t).scale(4).set_stroke(opacity=0)])

        agents = VGroup(*[agent_marker(e["agent"], f, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]))
                          for e in sc.starts])
        self.say("Before we send people in, we send three small robots.",
                 run_with=[LaggedStart(*[GrowFromCenter(a) for a in agents], lag_ratio=0.35, run_time=1.6)])

        radios = VGroup()
        for e in sc.starts:
            x, y = e["start_xy_m"]
            col = AGENT[e["agent"]]
            poly = Polygon(*[f.p(px, py) for px, py in los_polygon(sc, x, y)], stroke_width=0,
                           fill_color=col, fill_opacity=0.16)
            ring = DashedVMobject(Circle(radius=f.length(RADIO_RANGE_M), color=col, stroke_width=1,
                                         stroke_opacity=0.35).move_to(f.p(x, y)), num_dashes=60)
            radios.add(VGroup(poly, ring))
        chip = tag("illustrative").to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        self.say("None of them has a map. GPS can't reach them in here, "
                 "and their radios barely make it through the walls.",
                 run_with=[AnimationGroup(*[FadeIn(r) for r in radios], lag_ratio=0.25, run_time=2.0),
                           FadeOut(seed), FadeIn(chip)])
        self.say("Each one will only ever see a small piece of the building.",
                 run_with=[VGroup(*[r[1] for r in radios]).animate(run_time=1.0).set_opacity(0)])

        hint = DashedVMobject(polyline(f, sc.gt_routes_responder[0], THREAD, width=4, opacity=0.8), num_dashes=70)
        self.say("So the question is: can they put those pieces together, and find a safe way in, and out?",
                 run_with=[Create(hint, run_time=3.0)])
        self.clear_caption()

        # beat 2: the title over the dimmed maze
        scene = VGroup(dots, walls, target, agents, hint)
        title = Text("Ariadne", font=FONT, font_size=84, color=TEXT, weight="BOLD").shift(UP * 0.5)
        sub = Text("collaborative mapping and egress for a team of robots", font=FONT, font_size=26,
                   color=DIM).next_to(title, DOWN, buff=0.3)
        self.play(scene.animate.set_opacity(0.15), FadeOut(radios), FadeOut(chip), run_time=0.8)
        self.say("This is Ariadne, named for the thread that once led a hero out of a labyrinth.",
                 run_with=[FadeIn(title, shift=UP * 0.2, run_time=1.2), FadeIn(sub, run_time=1.6)])
        self.clear_caption()
        self.play(FadeOut(VGroup(scene, title, sub)), run_time=0.8)
