"""Beat 1 (0-12 s): the seeded WorldGen maze draws itself; three agents at three entrances; the target; radio LOS.

Data: maze_demo_0 scene.json + scene_gt.npz (walls = the free/occupied boundary of the scene grid, starts, target).
The radio shading is the line-of-sight region from each start, ray-cast on that grid: illustrative. The live RF model
(D102) attenuates through walls rather than stopping at them, which the faint full-range ring stands for. No numbers.

  manim -ql scenes/s01_worldgen.py WorldGen
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (AnimationGroup, Circle, Create, DashedVMobject, Dot, FadeIn, GrowFromCenter, LaggedStart, Line,
                   Polygon, Text, Triangle, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.style import (AGENT, DATA, DIM, FONT, TARGET, WALL, CaptionedScene, MapFrame, tag,  # noqa: E402
                                 UP, RIGHT)

RADIO_RANGE_M = 9.0  # illustrative LOS radius for the shading only


def los_polygon(sc, x0: float, y0: float, n: int = 240, rmax: float = RADIO_RANGE_M) -> np.ndarray:
    """Points where n rays from (x0, y0) first leave free space on the scene grid (or reach rmax)."""
    ox, oy = sc.origin_xy_m
    rows, cols = sc.free.shape
    step = sc.cell_m * 0.5
    pts = []
    for th in np.linspace(0, 2 * np.pi, n, endpoint=False):
        d = np.arange(0, rmax, step)
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
        w = np.array(sc.walls)
        x0, x1 = w[:, [0, 2]].min() - 1.0, w[:, [0, 2]].max() + 1.0  # wall bbox + 1 m: the maze fills the frame
        y0, y1 = w[:, [1, 3]].min() - 1.0, w[:, [1, 3]].max() + 1.0
        f = MapFrame(x0, x1, y0, y1, height=6.6, center=UP * 0.3)

        seed = Text(f"worldgen · seed {sc.seed}", font=FONT, font_size=18, color=DIM)
        seed.to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        dots = VGroup(*[Dot(f.p(x, y), radius=0.012, color=DIM)
                        for x in np.arange(np.ceil(x0), x1, 1.0) for y in np.arange(np.ceil(y0), y1, 1.0)])
        # walls appear in a diagonal sweep: the generator "laying" the maze from the SW corner
        walls = sorted(sc.walls, key=lambda s: (s[0] + s[2]) / 2 + (s[1] + s[3]) / 2)
        wall_m = VGroup(*[Line(f.p(x0, y0), f.p(x1, y1), color=WALL, stroke_width=3) for x0, y0, x1, y1 in walls])

        self.play(FadeIn(dots, run_time=0.8), FadeIn(seed))
        self.say("A collapsed building. No map. No GPS.", dur=3.2,
                 run_with=[LaggedStart(*[Create(w) for w in wall_m], lag_ratio=0.04, run_time=3.2)])

        agents = VGroup()
        for e in sc.starts:
            x, y = e["start_xy_m"]
            tri = Triangle(color=AGENT[e["agent"]], fill_opacity=1, stroke_width=0).scale(0.11)
            tri.rotate(np.deg2rad(e["start_yaw_deg"]) - np.pi / 2).move_to(f.p(x, y))
            agents.add(tri)
        tx, ty = sc.target_xy_m
        target = VGroup(Circle(radius=0.13, color=TARGET, stroke_width=3).move_to(f.p(tx, ty)),
                        Dot(f.p(tx, ty), radius=0.04, color=TARGET))
        tlabel = Text("possible casualty", font=FONT, font_size=16, color=TARGET).next_to(target, RIGHT, buff=0.12)
        self.say("Three agents enter from three sides.", dur=2.2,
                 run_with=[LaggedStart(*[GrowFromCenter(a) for a in agents], lag_ratio=0.3)])
        self.play(FadeIn(target, scale=1.6), FadeIn(tlabel), run_time=0.8)

        radios = VGroup()
        for e in sc.starts:
            x, y = e["start_xy_m"]
            col = AGENT[e["agent"]]
            poly = Polygon(*[f.p(px, py) for px, py in los_polygon(sc, x, y)], stroke_width=0,
                           fill_color=col, fill_opacity=0.16)
            ring = DashedVMobject(Circle(radius=f.length(RADIO_RANGE_M), color=col, stroke_width=1,
                                         stroke_opacity=0.35).move_to(f.p(x, y)), num_dashes=60)
            radios.add(VGroup(poly, ring))
        chip = tag("illustrative").to_corner(UP + RIGHT * 0).to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        self.say("Radio fades through every wall.", dur=2.4,
                 run_with=[AnimationGroup(*[FadeIn(r) for r in radios], lag_ratio=0.2), seed.animate.set_opacity(0),
                           FadeIn(chip)])
        self.say("One mission clock. One shared byte budget.", dur=3.0)
        self.clear_caption()
