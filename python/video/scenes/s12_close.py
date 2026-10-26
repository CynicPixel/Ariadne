"""Beat 12: why it matters, and the team. Back to the maze: the thread lit from the casualty to the nearest exit at
person width (computed on maze_demo_0's ground-truth lattice, as in beat 9: illustrative), three robots at their
doors; then the team card (wording to confirm before the final render).

  python/video/render.sh scenes/s12_close.py Close
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, UP, Circle, Create, Dot, FadeIn, FadeOut, LaggedStart, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene  # noqa: E402
from s09_thread import descend, dijkstra  # noqa: E402

TEAM = [("Suvojit Das", "design, engineering, evaluation")]  # [ASSUMPTION] wording pending review


class Close(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        z = np.load(DATA / "maze_demo_0" / "scene_gt.npz")
        f = maze_frame(sc)
        walls = maze_walls(sc, f, opacity=0.8)
        bots = VGroup(*[agent_marker(e["agent"], f, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"])) for e in sc.starts])
        tx, ty = sc.target_xy_m
        casualty = VGroup(Circle(radius=0.13, color=TARGET, stroke_width=3).move_to(f.p(tx, ty)),
                          Dot(f.p(tx, ty), radius=0.04, color=TARGET))
        exits = [tuple(int(v) for v in s) for s in z["start_cells"]]
        d = dijkstra(z["trav_responder"], exits)
        route = descend(d, tuple(int(v) for v in z["target_cell"]))
        thread = polyline(f, np.array(sc.cell_to_xy(np.array(route))), THREAD, width=7)
        glow = polyline(f, np.array(sc.cell_to_xy(np.array(route))), THREAD, width=18, opacity=0.18)

        self.play(FadeIn(walls), FadeIn(casualty), LaggedStart(*[FadeIn(b) for b in bots], lag_ratio=0.2), run_time=1.0)
        self.say("In the end, Ariadne isn't about a pretty 3D model.", run_with=[Create(glow, run_time=2.0),
                                                                                  Create(thread, run_time=2.0)])
        self.say("It's about whether a small team of robots, each seeing only part of the picture, can build one map "
                 "they can trust, and hand a responder a way in, and a way out.")
        self.clear_caption()
        # dim: strokes only for the open polylines (set_opacity would also fill them)
        self.play(VGroup(walls, thread, glow).animate.set_stroke(opacity=0.12), VGroup(bots, casualty).animate.set_opacity(0.12),
                  run_time=0.8)

        title = Text("Ariadne", font=FONT, font_size=72, color=TEXT, weight="BOLD")
        people = VGroup(*[VGroup(Text(n, font=FONT, font_size=30, color=THREAD),
                                 Text(r, font=FONT, font_size=20, color=DIM)).arrange(DOWN, buff=0.08)
                          for n, r in TEAM]).arrange(DOWN, buff=0.3)
        foot = Text("OpenCV AI Competition 2026", font=FONT, font_size=20, color=DIM)
        card = VGroup(title, people, foot).arrange(DOWN, buff=0.45)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(people), FadeIn(foot), run_time=0.8)
        self.wait(3.0)
        self.play(FadeOut(card), FadeOut(VGroup(walls, bots, casualty, thread, glow)), run_time=1.0)
