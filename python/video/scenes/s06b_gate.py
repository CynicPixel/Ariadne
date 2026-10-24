"""Beat 6b: false-loop rejection. Why one aliased loop can drag a merged map, how the known-entry gate stops it,
and the measured result (D178).

Part A is illustrative (tagged): the real maze_demo_0 walls and GT routes stand in for two agents' maps; the
alias, the drag and the uncertainty ellipse are schematic. Part B draws only evidence.d178() numbers
(evaluation/runs/collab_d178/verdict/: cells.csv, true_edges.csv): merged ATE per sweep cell, gated vs ungated.

  python/video/render.sh scenes/s06b_gate.py FalseLoop
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, Cross, DashedLine, Dot, Ellipse, FadeIn, FadeOut,
                   GrowFromCenter, Line, Square, Text, Transform, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import d178  # noqa: E402
from ariadne_video.mobjects import maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, CaptionedScene, tag  # noqa: E402

GOOD = "#56d364"


def point_along(xy: np.ndarray, frac: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return np.array([np.interp(frac * d[-1], d, xy[:, 0]), np.interp(frac * d[-1], d, xy[:, 1])])


def head(xy: np.ndarray, frac: float) -> np.ndarray:
    """The first `frac` of a polyline by arc length: each agent's map so far, before the routes meet."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return xy[d <= frac * d[-1]]


class FalseLoop(CaptionedScene):
    def construct(self):
        self.part_a()
        self.part_b()

    def part_a(self):
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc)
        walls = maze_walls(sc, f, opacity=0.55)
        r0, r1 = head(sc.gt_routes_responder[0], 0.5), head(sc.gt_routes_responder[1], 0.5)
        p0 = polyline(f, r0, AGENT[0], width=4)
        p1 = polyline(f, r1, AGENT[1], width=4)
        chip = tag("illustrative").to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        self.play(FadeIn(walls), FadeIn(chip), run_time=0.8)
        self.say("Two agents, two local maps.", dur=2.0, run_with=[Create(p0, run_time=1.6), Create(p1, run_time=1.6)])

        a = point_along(r0, 0.7)  # where agent 0 really is
        b = point_along(r1, 0.7)  # where agent 1 really is: a corridor that looks the same
        look = VGroup(*[Square(0.32, color=c, stroke_width=3).move_to(f.p(*q)) for q, c in ((a, AGENT[0]), (b, AGENT[1]))])
        alias = DashedLine(f.p(*b), f.p(*a), color=TARGET, stroke_width=4)
        self.say("Repeated structure: two different places look identical.", dur=2.6, run_with=[GrowFromCenter(look)])
        self.say("One false loop closure, and a whole map is dragged away.", dur=3.0,
                 run_with=[Create(alias, run_time=0.8),
                           VGroup(p1, look[1]).animate(run_time=2.2).shift(f.p(*a) - f.p(*b))])

        # the gate: agent 1's pose is known at its entrance; odometry uncertainty grows along the way
        self.play(VGroup(p1, look[1]).animate.shift(f.p(*b) - f.p(*a)), FadeOut(alias), run_time=0.9)
        ell = Ellipse(width=f.length(2.4), height=f.length(1.6), color=AGENT[1], stroke_width=3,
                      fill_opacity=0.12).move_to(f.p(*b))
        start = r1[0]
        entry = Dot(f.p(*start), color=AGENT[1], radius=0.07)
        alias2 = DashedLine(f.p(*b), f.p(*a), color=TARGET, stroke_width=4)
        self.say("But Ariadne knows where each agent came in, and how uncertain it is since then.", dur=3.2,
                 run_with=[FadeIn(entry, scale=2), GrowFromCenter(ell)])
        x = Cross(scale_factor=0.18, stroke_color=TARGET).move_to(alias2.get_center())
        self.say("A loop that lands outside that uncertainty is rejected.", dur=2.6,
                 run_with=[Create(alias2, run_time=0.6)])
        self.play(GrowFromCenter(x), alias2.animate.set_opacity(0.25), run_time=0.6)
        self.wait(0.4)
        self.clear_caption()
        self.play(FadeOut(VGroup(walls, p0, p1, look, ell, entry, alias2, x, chip)), run_time=0.7)

    def part_b(self):
        g = d178()
        cells = [c for c in g["cells"] if c["scene"] == 0]
        edges = g["true_edges"]
        tested = sum(e["tested"] for e in edges.values())
        rejected = sum(e["rejected_last"] for e in edges.values())

        # log axis 1 cm .. 10 m
        L, Rr, y_top = -4.2, 5.4, 2.0
        lx = lambda m: L + (np.log10(m) + 2) / 3 * (Rr - L)  # noqa: E731  (0.01 m -> L, 10 m -> Rr)
        axis = Line([L, -1.9, 0], [Rr, -1.9, 0], color=DIM)
        ticks = VGroup()
        for m, s in ((0.01, "1 cm"), (0.1, "10 cm"), (1, "1 m"), (10, "10 m")):
            ticks.add(Line([lx(m), -1.95, 0], [lx(m), -1.85, 0], color=DIM),
                      Text(s, font=FONT, font_size=16, color=DIM).move_to([lx(m), -2.2, 0]))
        title = Text("Merged-map error, replay sweep (known entry poses)", font=FONT, font_size=26,
                     color=TEXT).to_edge(UP, buff=0.3)
        chip = tag("measured", "Ariadne, D178").to_edge(UP, buff=0.85).to_edge(RIGHT, buff=0.4)
        names = {"camp_faults": "campaign · faults", "camp_ideal": "campaign · ideal link",
                 "camp_partition": "campaign · partition", "greedy_p25": "greedy · 25 % budget",
                 "greedy_p50": "greedy · 50 % budget", "spectral_p25": "spectral · 25 % budget",
                 "spectral_p50": "spectral · 50 % budget"}
        rows, bad, good, arrows = VGroup(), VGroup(), VGroup(), VGroup()
        for i, c in enumerate(cells):
            y = y_top - i * 0.52
            rows.add(Text(names.get(c["cell"], c["cell"]), font=FONT, font_size=17, color=DIM).move_to([L - 1.6, y, 0]))
            bad.add(Dot([lx(c["base_m"]), y, 0], color=TARGET, radius=0.07))
            good.add(Dot([lx(c["gated_m"]), y, 0], color=GOOD, radius=0.07))
            arrows.add(Arrow([lx(c["base_m"]), y, 0], [lx(c["gated_m"]), y, 0], buff=0.1, stroke_width=2,
                             color=DIM, max_tip_length_to_length_ratio=0.03))
        worst_base = min(c["base_m"] for c in cells)
        worst_gated = max(c["gated_m"] for c in cells if c["budgeted"])
        lab_bad = Text("no gate", font=FONT, font_size=18, color=TARGET).next_to(bad, UP, 0.15)
        lab_good = Text("entry gate", font=FONT, font_size=18, color=GOOD).next_to(good, UP, 0.15)

        self.play(FadeIn(title), FadeIn(chip), Create(axis), FadeIn(ticks), FadeIn(rows), run_time=1.0)
        self.say(f"Without the gate, aliased loops pulled the merged map {worst_base:.2f} m off.", dur=3.0,
                 run_with=[FadeIn(bad, lag_ratio=0.1), FadeIn(lab_bad)])
        self.say(f"With it: {worst_gated * 100:.1f} cm or better in every budgeted run.", dur=3.0,
                 run_with=[*[Create(a) for a in arrows], Transform(bad.copy(), good), FadeIn(lab_good)])
        self.say(f"And of {tested} genuine loop closures tested, {rejected} were thrown away.", dur=3.2)
        self.clear_caption()
