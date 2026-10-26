"""Beat 9 (≈ 16 s): the Ariadne thread. The project's D* Lite (cpp/planning/route/dstar_lite.cpp, unchanged, driven by
tools/dstar_trace.cpp) on maze_demo_0's person-width lattice (scene_gt.npz trav_responder, simulator ground truth):
the cells the first search settled appear in order of their cost-to-exit g, a wave from every exit at once; the thread
is the route read from the casualty; rubble closes the tightest stretch of it; Update() repairs, and the cells whose g
changed light up in order of their new g; the new thread runs to a different exit. Counts are the C++'s own
(expansions: first search, repair, a fresh search on the blocked grid). Measured rows: docs/results/index.md §8 (replay
spool ci_tiny via evidence.egress(); D113's unit test).

  python3 python/video/tools/dstar_replay.py   # regenerates data/dstar/
  python/video/render.sh scenes/s09_thread.py Thread
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, ImageMobject, Square, Text, ValueTracker, VGroup,
                   always_redraw, linear)
from manim.constants import RESAMPLING_ALGORITHMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import egress  # noqa: E402
from ariadne_video.mobjects import maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.slides import chip_top  # noqa: E402
from ariadne_video.style import DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

CHIP_BOTTOM, CAP2 = 3.4, -2.93
MH, GAPX, RW = 5.2, 0.7, 5.0  # maze height; maze-to-column gap; right column width
PERSON = "#58a6ff"


class Thread(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        tr = json.loads((DATA / "dstar" / "trace.json").read_text())
        fz = np.load(DATA / "dstar" / "fields.npz")
        g0, g1, block, trav = fz["g0"], fz["g1"], fz["block"], fz["trav"]
        assert tr["exit0"] != tr["exit1"] and abs(tr["cost1"] - tr["fresh_cost"]) < 1e-6
        rows, cols = trav.shape
        ox, oy = sc.origin_xy_m
        cm = tr["cell_m"]

        # ---- layout: (maze | counts) centred in both directions -----------------------------------------------------
        mid = (CHIP_BOTTOM + CAP2) / 2
        f = maze_frame(sc, height=MH, center=(0, mid, 0))
        w = np.array(sc.walls)
        mw = f.length(np.ptp(w[:, [0, 2]]) + 2.0)
        left = -(mw + GAPX + RW) / 2
        f = maze_frame(sc, height=MH, center=(left + mw / 2, mid, 0))
        RXL = left + mw + GAPX  # the right column's left edge

        def grid_image(rgba: np.ndarray) -> ImageMobject:
            im = ImageMobject(np.flipud(rgba))  # grid row 0 is SOUTH; image row 0 is the top
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            im.stretch_to_fit_height(f.length(rows * cm)).stretch_to_fit_width(f.length(cols * cm))
            return im.move_to(f.p(ox + cols * cm / 2, oy + rows * cm / 2))

        def paint(mask: np.ndarray, hex_color: str, alpha: float) -> np.ndarray:
            rgba = np.zeros((rows, cols, 4), np.uint8)
            rgba[..., :3] = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
            rgba[..., 3] = (np.where(mask, alpha, 0) * 255).astype(np.uint8)
            return rgba

        cell_xy = lambda p: np.array(sc.cell_to_xy(np.asarray(p)))  # noqa: E731
        walls = maze_walls(sc, f, width=2.5)
        lattice = grid_image(paint(trav, PERSON, 0.14))
        exits = VGroup(*[Square(0.2, color=TEXT, stroke_width=2).move_to(f.p(*cell_xy([g])[0])) for g in tr["goals"]])
        for i, m in enumerate(list(exits)):  # number each exit, just outside the maze
            out = m.get_center() - f.p(*sc.target_xy_m)
            exits.add(Text(str(i + 1), font=FONT, font_size=13, color=TEXT).move_to(m).shift(
                out / np.linalg.norm(out) * 0.26))
        casualty = Square(0.22, color=TARGET, stroke_width=3).move_to(f.p(*cell_xy([tr["start"]])[0]))
        chip = chip_top("illustrative", "the project's D* Lite, run on the generated maze's person-width map")

        # ---- right column: what the search did, in its own counts ----------------------------------------------------
        def line(text: str, color=TEXT, size=16) -> Text:
            return Text(text, font=FONT, font_size=size, color=color)

        head = line("D* Lite · Koenig & Likhachev 2002", TEXT, 18)
        sub = line("searched backwards from every exit,\nover person-width cells only", DIM, 13)
        trav_n = int(trav.sum())
        c1 = line(f"first search: {tr['expanded_first']:,} of {trav_n:,} cells", THREAD)
        c2 = line(f"route {tr['cost0'] * cm:.1f} m to exit {tr['exit0'] + 1}", THREAD)
        c3 = line(f"repair: {tr['expanded_repair']:,} cells · fresh: {tr['expanded_fresh']:,}", TARGET)
        c4 = line(f"new route {tr['cost1'] * cm:.1f} m to exit {tr['exit1'] + 1}, as a fresh search", TARGET)
        info = VGroup(*[VGroup(a, b).arrange(DOWN, aligned_edge=LEFT, buff=0.1) for a, b in ((head, sub), (c1, c2),
                                                                                             (c3, c4))])
        info.arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        tag_this = tag("derived", "this run").scale(0.75)
        e = egress()
        r1 = VGroup(line(f"replay: {e['valid']} of {e['found']} routes passable for a 0.40 m responder;\n"
                         f"the final one is the true shortest path (ratio {e['length_ratio']:.7f})", TEXT, 13),
                    tag("measured", "replay").scale(0.75)).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        r2 = VGroup(line("same routes as a fresh search on 180 random map\nchanges, with 7.6× fewer expansions", TEXT, 13),
                    tag("measured", "unit test").scale(0.75)).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        res = VGroup(r1, r2).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        col = VGroup(info, tag_this, res).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        col.move_to([RXL + col.width / 2, mid, 0])
        assert col.width <= RW and col.height <= MH, (col.width, col.height)

        # ---- 1 · the wave from every exit, then the thread ----------------------------------------------------------
        tau = ValueTracker(0.0)
        wave = always_redraw(lambda: grid_image(paint(np.isfinite(g0) & (g0 <= tau.get_value()), THREAD, 0.55)))
        thread0 = polyline(f, cell_xy(tr["path0"]), THREAD, width=6)
        self.play(FadeIn(chip), FadeIn(walls), FadeIn(lattice), FadeIn(exits), FadeIn(casualty), FadeIn(head),
                  FadeIn(sub), run_time=0.6)
        self.add(wave)
        self.say_steps("Ariadne searches back from every exit, through person-wide gaps only, to the "
                       "casualty;", [
                           [tau.animate(run_time=3.6, rate_func=linear).set_value(tr["cost0"] + 1), FadeIn(c1)],
                           [Create(thread0, run_time=1.5), FadeIn(c2)]])

        # ---- 2 · rubble; the repair touches the cells whose cost-to-exit changed --------------------------------------
        rubble = grid_image(paint(block & sc.free, TARGET, 0.95))
        changed = ~np.isclose(g0, g1) & (np.isfinite(g0) | np.isfinite(g1))
        tau2 = ValueTracker(0.0)
        repair = always_redraw(lambda: grid_image(paint(changed & np.isfinite(g1) & (g1 <= tau2.get_value()),
                                                        TARGET, 0.35)))
        thread1 = polyline(f, cell_xy(tr["path1"]), THREAD, width=6)
        hi = float(np.max(np.where(changed & np.isfinite(g1), g1, 0)))

        def start_repair() -> list:
            self.add(repair)
            return [tau2.animate(run_time=2.4, rate_func=linear).set_value(hi), FadeIn(c3)]

        self.say_steps("when rubble blocks the way, it repairs the plan instead of starting over.", [
            [FadeIn(rubble, scale=1.6, run_time=0.6), thread0.animate(run_time=0.6).set_stroke(opacity=0.25)],
            lambda: [FadeOut(wave, run_time=0.4)],
            lambda: start_repair(),
            [Create(thread1, run_time=1.4), FadeOut(thread0, run_time=1.0), FadeIn(c4)]])
        self.play(FadeIn(tag_this), FadeIn(res, shift=UP * 0.1), run_time=0.6)
        self.wait(2.4)
        self.clear_caption()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)
