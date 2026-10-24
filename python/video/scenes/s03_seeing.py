"""Beat 3: seeing in the dark. Ride along with one robot: stereo cameras and a lamp, real features on a real frame.

Real data: the final camera frame of robot 1 (agent 0) and its 475 tracked feature points from the recorded run
(B's pack: frames/final_agent0.jpg + overlay_final_agent0.json; true-colour files, read as-is). The rig is illustrative.

  python/video/render.sh scenes/s03_seeing.py Seeing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Circle, FadeIn, FadeOut, Group, ImageMobject, Polygon, Rectangle, Text,
                   VGroup)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

PACK = DATA / "pack_collab60d"
MARK = "#ff2d2d"  # tracked-point markers: bright red (maintainer)


class Seeing(CaptionedScene):
    def construct(self):
        # 1. ride along with robot 1
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc)
        walls = maze_walls(sc, f)
        e0 = sc.starts[0]
        bot = agent_marker(0, f, *e0["start_xy_m"], np.deg2rad(e0["start_yaw_deg"]))
        self.play(FadeIn(walls), FadeIn(bot), run_time=0.6)
        self.say("Let's ride along with one of them.",
                 run_with=[Group(walls, bot).animate(run_time=1.6).scale(3.2, about_point=bot.get_center())
                           .set_opacity(0)])
        self.remove(walls, bot)

        # 2. two eyes and a lamp (illustrative rig)
        body = Rectangle(width=1.6, height=0.7, color=TEXT, stroke_width=2).move_to(LEFT * 4.6 + UP * 0.6)
        eyes = VGroup(*[Circle(0.17, color=AGENT[0], stroke_width=3).move_to(body.get_center() + RIGHT * dx)
                        for dx in (-0.45, 0.45)])
        lamp = Polygon(body.get_top() + UP * 0.05, body.get_top() + UP * 2.4 + LEFT * 1.1,
                       body.get_top() + UP * 2.4 + RIGHT * 1.1, stroke_width=0, fill_color=THREAD, fill_opacity=0.18)
        rig_lbl = Text("stereo cameras + lamp", font=FONT, font_size=18, color=DIM).next_to(body, DOWN, 0.2)
        rig = VGroup(lamp, body, eyes, rig_lbl)
        rig.move_to([-4.6, 0.45, 0])  # vertically centred on the frame image (centre y = 0.45)
        im = Image.open(PACK / "frames" / "final_agent0.jpg").convert("RGB")
        frame = ImageMobject(np.asarray(im)).set_height(4.3).move_to(RIGHT * 1.9 + UP * 0.45)
        ill = tag("illustrative").next_to(rig, UP, 0.15)
        self.say("Each robot looks through two cameras, for stereo depth sensing, and carries its own illumination.",
                 run_with=[FadeIn(rig, run_time=1.2), FadeIn(ill), FadeIn(frame, run_time=1.4)])

        # 3. real features on the real frame
        ov = json.loads((PACK / "frames" / "overlay_final_agent0.json").read_text())
        sx, sy = frame.width / ov["w"], frame.height / ov["h"]
        tl = frame.get_corner(UP + LEFT)
        dots = VGroup(*[Circle(0.035, color=MARK, stroke_width=1.8).move_to(tl + RIGHT * x * sx + DOWN * y * sy)
                        for x, y in ov["pts"]])
        flbl = VGroup(Text(f"{len(ov['pts'])} tracked points", font=FONT, font_size=20, color=MARK),
                      Text("FAST corners + ORB · pyramidal Lucas–Kanade", font=FONT, font_size=16, color=DIM)
                      ).arrange(DOWN, buff=0.08).next_to(frame, DOWN, 0.15)
        real = tag("measured", "recorded run").next_to(frame, UP, 0.12).align_to(frame, RIGHT)
        self.say("In every frame it picks out distinctive points, like corners and scuffs on the wall, and follows them "
                 "as it moves.", run_with=[FadeIn(dots, lag_ratio=0.004, run_time=2.4), FadeIn(flbl), FadeIn(real)])

        self.clear_caption()
        self.play(FadeOut(Group(frame, dots, flbl, real)), FadeOut(rig), FadeOut(ill), run_time=0.5)
