"""Beat 9: the Ariadne Thread. Clearance -> person-width lattice -> D* Lite from every exit -> thread -> obstruction ->
repair of only the changed cells -> measured results.

Mechanism (tagged illustrative): maze_demo_0's simulator ground truth (scene_gt.npz: esdf_m, trav_agent,
trav_responder, GT routes). The search is computed here on that GT lattice (8-connected, corner rule; it reproduces
scene.json lengths_m exactly) and run backwards from every exit to the casualty, as D113's D* Lite does. It shows what
the planner does; it is not a run of it. Results card: RESULTS_INDEX §8 via evidence.egress() / evidence.mapping()
(replay spool ci_tiny) and D113's unit test.

  python/video/render.sh scenes/s09_thread.py Thread
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, ImageMobject, Square, Text, ValueTracker, VGroup,
                   always_redraw)
from manim.constants import RESAMPLING_ALGORITHMS
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import egress, mapping  # noqa: E402
from ariadne_video.mobjects import CodePanel, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]

DSTAR = [
    "for each exit e:  rhs(e) = 0",
    "ComputeShortestPath()",
    "loop:",
    "  thread = walk g downhill from casualty",
    "  wait for map changes",
    "  for each changed cell u:",
    "    UpdateVertex(u)",
    "  ComputeShortestPath()",
]


def dijkstra(trav: np.ndarray, srcs) -> np.ndarray:
    """Cell-unit distances from the sources over trav, 8-connected; a diagonal needs both side cells free."""
    R, C = trav.shape
    d = np.full((R, C), np.inf)
    h = []
    for s in srcs:
        d[s] = 0.0
        h.append((0.0, *s))
    heapq.heapify(h)
    while h:
        dd, r, c = heapq.heappop(h)
        if dd > d[r, c]:
            continue
        for dr, dc, w in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and trav[rr, cc] and (dr == 0 or dc == 0 or (trav[r + dr, c] and trav[r, c + dc])):
                if dd + w < d[rr, cc]:
                    d[rr, cc] = dd + w
                    heapq.heappush(h, (dd + w, rr, cc))
    return d


def descend(d: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    """Follow the distance field downhill from start to its nearest source: the route."""
    path, (r, c) = [start], start
    while d[r, c] > 0:
        r, c = min(((r + dr, c + dc) for dr, dc, _ in NB), key=lambda q: d[q])
        path.append((r, c))
    return path


class Thread(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        z = np.load(DATA / "maze_demo_0" / "scene_gt.npz")
        f = maze_frame(sc, height=5.7, center=LEFT * 2.9 + UP * 0.55)
        rows, cols = sc.free.shape
        ox, oy = sc.origin_xy_m
        cell_xy = lambda rc: np.array(sc.cell_to_xy(np.asarray(rc)))  # noqa: E731

        def grid_image(rgba: np.ndarray) -> ImageMobject:
            im = ImageMobject(np.flipud(rgba))  # grid row 0 is SOUTH; image row 0 is the top
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            im.stretch_to_fit_height(f.length(rows * sc.cell_m)).stretch_to_fit_width(f.length(cols * sc.cell_m))
            return im.move_to(f.p(ox + cols * sc.cell_m / 2, oy + rows * sc.cell_m / 2))

        def paint(mask: np.ndarray, hex_color: str, alpha) -> np.ndarray:
            rgba = np.zeros((rows, cols, 4), np.uint8)
            rgba[..., :3] = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
            rgba[..., 3] = (np.where(mask, alpha, 0) * 255).astype(np.uint8)
            return rgba

        walls = maze_walls(sc, f)
        chip = tag("illustrative", "on simulator ground truth").to_edge(UP, buff=0.25).to_edge(LEFT, buff=0.4)
        self.add(walls)
        self.play(FadeIn(chip), run_time=0.5)

        # 1. person width vs robot width: the one gap a robot fits through and a person does not
        ta, tr, esdf = z["trav_agent"], z["trav_responder"], z["esdf_m"]
        yy, xx = np.ogrid[-6:7, -6:7]
        gap = ta & ~tr & ~binary_dilation(tr, structure=yy ** 2 + xx ** 2 <= 36)  # agent-only, > 0.3 m from person space
        resp = grid_image(paint(tr, "#58a6ff", 0.30))
        gap_im = grid_image(paint(gap, TARGET, 0.9))
        robot_route = polyline(f, cell_xy(z["route_agent_start1"]), TARGET, width=3)
        person_route = polyline(f, cell_xy(z["route_responder_start1"]), THREAD, width=4)
        legend = VGroup(Text("robot fits", font=FONT, font_size=18, color=TARGET),
                        Text("a person fits", font=FONT, font_size=18, color="#58a6ff")).arrange(DOWN, aligned_edge=LEFT)
        legend.next_to(walls, RIGHT, buff=0.3).align_to(walls, UP)
        self.say("Now for the reason we're here: a way out.", run_with=[FadeIn(resp, run_time=1.2), FadeIn(legend)])
        self.say("A robot can squeeze through gaps a person can't, so Ariadne only plans routes wide enough for a person.",
                 run_with=[FadeIn(gap_im, run_time=0.8), Create(robot_route, run_time=1.6),
                           Create(person_route, run_time=2.6)])
        self.play(FadeOut(robot_route), FadeOut(person_route), FadeOut(gap_im), FadeOut(legend), run_time=0.5)

        # 2. D* Lite, backwards from every exit, until it reaches the casualty
        code = CodePanel("D* Lite  (Koenig & Likhachev)", DSTAR, font_size=17).to_edge(RIGHT, buff=0.35).shift(UP * 0.55)
        target = tuple(int(v) for v in z["target_cell"])
        exits = [tuple(int(v) for v in s) for s in z["start_cells"]]
        d = dijkstra(tr, exits)
        tau = ValueTracker(0.0)
        wave = always_redraw(lambda: grid_image(paint(d <= tau.get_value(), THREAD, 0.30)))
        exit_marks = VGroup(*[Square(0.2, color=TEXT, stroke_width=2).move_to(f.p(*cell_xy([e])[0])) for e in exits])
        tx, ty = sc.target_xy_m
        casualty = Square(0.22, color=TARGET, stroke_width=3).move_to(f.p(tx, ty))
        self.play(FadeIn(code), FadeOut(resp), FadeIn(exit_marks), FadeIn(casualty), code.focus(0), run_time=0.8)
        self.add(wave)
        self.say("It spreads out from every exit at once until it reaches the casualty, and lays down the thread.",
                 run_with=[code.focus(1), tau.animate(run_time=3.4).set_value(d[target])])
        route = descend(d, target)
        thread = polyline(f, cell_xy(route), THREAD, width=7)
        self.play(Create(thread, run_time=1.4), code.focus(3))
        self.remove(wave)

        # 3. an obstruction closes the narrowest stretch of the thread; only the changed cells are searched again
        n = len(route)
        k = min(route[n // 5: 4 * n // 5], key=lambda rc: esdf[rc])  # the tightest corridor cell on the route
        rr, cc = np.ogrid[:rows, :cols]
        block = (rr - k[0]) ** 2 + (cc - k[1]) ** 2 <= ((float(esdf[k]) + 0.15) / sc.cell_m) ** 2
        d2 = dijkstra(tr & ~block, exits)
        changed = ~np.isclose(d, d2) & (np.isfinite(d) | np.isfinite(d2))
        rubble = grid_image(paint(block & sc.free, TARGET, 0.95))
        tau2 = ValueTracker(0.0)
        repair = always_redraw(lambda: grid_image(paint(changed & (d2 <= tau2.get_value()), TARGET, 0.4)))
        self.say("If something collapses across that path, it doesn't start over.",
                 run_with=[FadeIn(rubble, scale=1.5), thread.animate.set_stroke(opacity=0.25), code.focus(4)])
        self.add(repair)
        hi = float(np.max(np.where(np.isfinite(d2) & changed, d2, 0)))
        self.say("It repairs just the part of the plan that changed.",
                 run_with=[code.focus(6), tau2.animate(run_time=2.2).set_value(hi)])
        thread2 = polyline(f, cell_xy(descend(d2, target)), THREAD, width=7)
        self.play(Create(thread2, run_time=1.4), FadeOut(thread), code.focus(7))
        self.remove(repair)
        self.clear_caption()
        self.wait(0.6)
        self.play(*[FadeOut(m) for m in (walls, rubble, thread2, exit_marks, casualty, chip, code)], run_time=0.6)

        # 4. measured: replay spool (ci_tiny) and the unit test
        e, m = egress(), mapping()
        rows_t = [
            (f"{e['valid']} of {e['found']}", "egress routes valid for a 0.40 m responder"),
            (f"{e['length_ratio'] * 100:.1f} %", "final route length vs the true shortest path"),
            (f"{m['trav_precision']:.3f}", "person-width traversability precision"),
            (f"{m['occupied_precision']:.3f}", f"wall-map precision (simple baseline {m['occupied_precision_placeholder']:.3f})"),
            ("7.6×", "less search than replanning from scratch (unit test)"),
        ]
        vals = VGroup(*[Text(a, font=FONT, font_size=30, color=THREAD, weight="BOLD") for a, _ in rows_t])
        vals.arrange(DOWN, aligned_edge=RIGHT, buff=0.32)
        labels = VGroup(*[Text(b, font=FONT, font_size=21, color=TEXT).next_to(v, RIGHT, buff=0.4)
                          for v, (_, b) in zip(vals, rows_t)])
        for l in labels:
            l.align_to(labels[0], LEFT)
        table = VGroup(vals, labels)
        title = Text("The thread, checked against ground truth", font=FONT, font_size=30, color=TEXT)
        VGroup(title, table).arrange(DOWN, buff=0.5).shift(UP * 0.4)
        chips = VGroup(tag("measured", "replay, D110–D113"), tag("measured", "unit test, D113")).arrange(RIGHT, buff=0.2)
        chips.next_to(table, DOWN, buff=0.35)
        self.play(FadeIn(title), FadeIn(table, lag_ratio=0.15, run_time=2.0), FadeIn(chips))
        self.wait(3.0)
        self.play(FadeOut(VGroup(title, table, chips)), run_time=0.5)
