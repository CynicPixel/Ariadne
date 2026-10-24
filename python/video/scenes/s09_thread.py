"""Beat 9: the Ariadne Thread. Clearance -> responder-width lattice -> search -> route -> obstruction -> repair.

Mechanism (tagged): maze_demo_0's simulator ground truth (scene_gt.npz: esdf_m, trav_agent, trav_responder,
GT routes). The search wavefront and the obstruction repair are computed here on that GT lattice (8-connected,
corner rule; reproduces scene.json lengths_m exactly) and are illustrative: they show what the D* Lite planner
does, not a run of it. Measured card: evidence.egress() (egress_4.5 replay) and D113's unit-test equivalence.

  python/video/render.sh scenes/s09_thread.py Thread
"""
from __future__ import annotations

import heapq
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, RIGHT, UP, Create, FadeIn, FadeOut, ImageMobject, Square, Text, ValueTracker, VGroup,
                   always_redraw)
from manim.constants import RESAMPLING_ALGORITHMS  # nearest-neighbour cells

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import egress  # noqa: E402
from ariadne_video.mobjects import maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import (AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene,  # noqa: E402
                                 tag)

NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]


def dijkstra(trav: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    """Cell-unit distances from src over trav, 8-connected; a diagonal needs both side cells free (corner rule)."""
    R, C = trav.shape
    d = np.full((R, C), np.inf)
    d[src] = 0.0
    h = [(0.0, *src)]
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
    """Follow the distance field downhill from start to its source: the route."""
    path, (r, c) = [start], start
    while d[r, c] > 0:
        r, c = min(((r + dr, c + dc) for dr, dc, _ in NB), key=lambda q: d[q])
        path.append((r, c))
    return path


class Thread(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        z = np.load(DATA / "maze_demo_0" / "scene_gt.npz")
        f = maze_frame(sc, height=6.4, center=UP * 0.4)
        rows, cols = sc.free.shape
        ox, oy = sc.origin_xy_m

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
        chip = tag("illustrative", "on simulator ground truth").to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.4)
        self.add(walls)
        self.play(FadeIn(chip), run_time=0.5)

        # 1. clearance: distance to the nearest wall (the ESDF)
        esdf = z["esdf_m"]
        clear = grid_image(paint(esdf > 0, "#39c5bb", np.clip(esdf / 1.5, 0, 1) * 0.75))
        self.say("From the shared map: how far is every point from the nearest wall?", dur=2.8,
                 run_with=[FadeIn(clear, run_time=1.2)])

        # 2. agent width vs responder width
        ta, tr = z["trav_agent"], z["trav_responder"]
        resp = grid_image(paint(tr, "#58a6ff", 0.35))
        only = grid_image(paint(ta & ~tr, TARGET, 0.75))
        self.say("A robot fits through gaps a responder cannot.", dur=2.6,
                 run_with=[FadeOut(clear), FadeIn(resp), FadeIn(only)])
        cell_xy = lambda rc: np.array(sc.cell_to_xy(np.asarray(rc)))  # noqa: E731
        agent_route = polyline(f, cell_xy(z["route_agent_start1"]), TARGET, width=3)
        resp_route = polyline(f, cell_xy(z["route_responder_start1"]), THREAD, width=4)
        self.say(f"So Ariadne plans for a person: {sc.rho_responder_m:.2f} m clearance, never the robot's.", dur=3.2,
                 run_with=[Create(agent_route, run_time=1.4), Create(resp_route, run_time=2.0)])
        self.play(FadeOut(agent_route), FadeOut(resp_route), FadeOut(only), run_time=0.5)

        # 3. search from the target, outward to every exit; the cheapest exit wins
        target = tuple(int(v) for v in z["target_cell"])
        exits = [tuple(int(v) for v in s) for s in z["start_cells"]]
        d = dijkstra(tr, target)
        best = min(exits, key=lambda e: d[e])
        tau = ValueTracker(0.0)
        wave = always_redraw(lambda: grid_image(paint(d <= tau.get_value(), THREAD, 0.28)))
        exit_marks = VGroup(*[Square(0.2, color=TEXT, stroke_width=2).move_to(f.p(*cell_xy([e])[0])) for e in exits])
        self.add(wave)
        self.say("Search from the casualty outward, until the first exit is reached.", dur=3.0,
                 run_with=[FadeIn(exit_marks), tau.animate(run_time=2.8).set_value(d[best])])
        route = descend(d, best)
        thread = polyline(f, cell_xy(route), THREAD, width=7)
        self.say("The Ariadne Thread: a route a person can walk.", dur=2.4,
                 run_with=[Create(thread, run_time=1.6), FadeOut(resp)])
        self.remove(wave)

        # 4. an obstruction lands on the thread: only the affected part of the search is repaired
        mid = route[len(route) // 2]
        rr, cc = np.ogrid[:rows, :cols]
        block = (rr - mid[0]) ** 2 + (cc - mid[1]) ** 2 <= (0.6 / sc.cell_m) ** 2
        d2 = dijkstra(tr & ~block, target)
        changed = ~np.isclose(d, d2) & (np.isfinite(d) | np.isfinite(d2))
        rubble = grid_image(paint(block & sc.free, TARGET, 0.95))
        tau2 = ValueTracker(0.0)
        repair = always_redraw(lambda: grid_image(paint(changed & (d2 <= tau2.get_value()), TARGET, 0.35)))
        best2 = min(exits, key=lambda e: d2[e])
        self.say("Something collapses across the route.", dur=1.8,
                 run_with=[FadeIn(rubble, scale=1.4), thread.animate.set_stroke(opacity=0.25)])
        self.add(repair)
        self.say("Only the cells it changed are searched again.", dur=2.8,
                 run_with=[tau2.animate(run_time=2.4).set_value(np.nanmax(np.where(np.isfinite(d2) & changed, d2, 0)))])
        thread2 = polyline(f, cell_xy(descend(d2, best2)), THREAD, width=7)
        self.say("A new thread, before anyone walks into the old one.", dur=2.4,
                 run_with=[Create(thread2, run_time=1.6), FadeOut(thread)])
        self.remove(repair)
        self.clear_caption()
        self.play(*[FadeOut(m) for m in (walls, rubble, thread2, exit_marks, chip)], run_time=0.6)

        # 5. measured
        e = egress()
        card = VGroup(
            Text(f"{e['valid']} of {e['found']} routes valid against ground truth", font=FONT, font_size=30, color=TEXT),
            Text("incremental repair = full replanning on 180 random map changes,", font=FONT, font_size=24, color=TEXT),
            Text("with 7.6× fewer cells searched", font=FONT, font_size=24, color=TEXT),
        ).arrange(DOWN, buff=0.25)
        chips = VGroup(tag("measured", "Ariadne replay, D113"), tag("measured", "unit test, D113")).arrange(RIGHT, buff=0.2)
        chips.next_to(card, DOWN, buff=0.4)
        self.say("Checked against ground truth, every time.", dur=3.2, run_with=[FadeIn(card), FadeIn(chips)])
        self.clear_caption()
