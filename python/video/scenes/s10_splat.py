"""Beat 10: one shared model. The real Gaussian seeds of the live 3-robot run, each robot's submaps flying into the
shared frame, then an orbit to viewpoints no robot stood at.

Real data: the 4 submap chunks of the live splat end-to-end run (B's run build/live/splat_e2e, A2's recon service,
cpu_init: Gaussians seeded from keyframes, BEFORE any GPU training) decoded from the antimatter15 .splat layout, placed
with each submap's T_W_anchor from table.json (checked: every point lands inside the maze grid). Numbers:
RESULTS_INDEX §10 (259,994 Gaussians, 0 dropped keyframes, 92.9 % of agent 1-2 wall seeds within 10 cm). A2's trained
AWS GPU render replaces this view when it lands.

  python/video/render.sh scenes/s10_splat.py SharedModel
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (DEGREES, DOWN, IN, OUT, RIGHT, UP, FadeIn, FadeOut, Group, Polygon, PMobject, Text, ThreeDScene, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import maze_frame  # noqa: E402
from ariadne_video.style import AGENT, DATA, FONT, TEXT, THREAD, WALL, CaptionedScene, tag  # noqa: E402

SPLAT = DATA / "splat_e2e"
DT = np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
KEEP = {0: 7000, 1: 3500, 2: 3500, 3: 2000}  # points drawn per submap (subsampled, seeded)
WALL_H = 2.95


class SharedModel(CaptionedScene, ThreeDScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc, height=6.2, center=(0, 0, 0))
        table = json.loads((SPLAT / "table.json").read_text())
        rng = np.random.default_rng(0)

        walls = VGroup(*[Polygon(f.p(x0, y0, 0), f.p(x1, y1, 0), f.p(x1, y1, WALL_H), f.p(x0, y0, WALL_H),
                                 stroke_color=WALL, stroke_width=1, stroke_opacity=0.5, fill_color=WALL,
                                 fill_opacity=0.04) for x0, y0, x1, y1 in sc.walls])
        clouds = []
        for sm in sorted(table["submaps"], key=lambda s: s["sid"]):
            a = np.fromfile(SPLAT / sm["chunk"], dtype=DT)
            idx = rng.choice(len(a), size=min(KEEP[sm["sid"]], len(a)), replace=False)
            T = np.array(sm["T_W_anchor"]).reshape(4, 4)
            w = a["p"][idx] @ T[:3, :3].T + T[:3, 3]
            pts = np.array([f.p(*q) for q in w])
            rgba = a["c"][idx].astype(float) / 255.0
            cloud = PMobject(stroke_width=2)
            cloud.add_points(pts, rgbas=rgba)
            clouds.append((sm, cloud))

        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.play(FadeIn(walls), run_time=0.8)
        self.say("And as they explore, the robots' pieces come together into something you can walk through.",
                 run_with=[walls.animate(run_time=0.1).set_stroke(opacity=0.5)])
        self.move_camera(phi=62 * DEGREES, theta=-60 * DEGREES, zoom=1.25, run_time=2.4)

        chip = tag("measured", "live run · Gaussian seeds, before training").to_edge(UP, buff=0.15).to_edge(RIGHT, buff=0.3)
        self.add_fixed_in_frame_mobjects(chip)
        anims = [FadeIn(cloud, shift=OUT * 1.2, run_time=1.6) for _, cloud in clouds]
        self.say("Each robot's views become a cloud of tiny, soft, coloured blobs, called Gaussian splats.",
                 run_with=[FadeIn(chip), *anims[:2]])
        self.say("Because their maps agree, the pieces slot into one model,", run_with=anims[2:])
        big = max(clouds, key=lambda c: len(c[1].points))[1]  # robot 1's first submap: the richest view
        world = Group(walls, *[c for _, c in clouds])
        self.move_camera(phi=70 * DEGREES, theta=-25 * DEGREES, zoom=2.2, run_time=2.0,
                         added_anims=[world.animate.shift(-big.get_center())])  # orbit about the cloud, not the maze
        self.begin_ambient_camera_rotation(rate=0.35)
        self.say("which we can view from angles no robot ever stood at.")
        self.wait(2.0)
        self.stop_ambient_camera_rotation()
        self.clear_caption()

        n_gauss = sum(s["gaussians"] for s in table["submaps"])
        stats = VGroup(Text(f"{n_gauss:,} Gaussians · 0 keyframes dropped", font=FONT, font_size=22, color=TEXT),
                       Text("92.9 % of wall seed points (robots 2–3) within 10 cm of ground truth", font=FONT,
                            font_size=19, color=THREAD)).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.35)
        self.add_fixed_in_frame_mobjects(stats)
        self.play(FadeIn(stats), run_time=0.6)
        self.wait(2.5)
        self.play(FadeOut(stats), FadeOut(chip), FadeOut(walls), *[FadeOut(c) for _, c in clouds], run_time=0.6)
