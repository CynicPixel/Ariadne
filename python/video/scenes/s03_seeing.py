"""Beat 3: seeing in the dark. Ride along with one robot: stereo eyes and a lamp, real features on a real frame,
depth from two views, and active exposure recovering from a flash.

Real data: the final camera frame of robot 1 (agent 0) and its 475 tracked feature points from the recorded run
(B's pack: frames/final_agent0.jpg + overlay_final_agent0.json; stored BGR, swapped on read). Active exposure:
evaluation/runs/ph5/d145_rows_20261023T113916Z/replays/r06_failE_expo/exposure_trace.csv (controller on, +2-stop
event at 20 s) and the all-frames ATE of r06 vs r07 (fixed exposure) from their results.yaml: RESULTS_INDEX §2 (D147,
replay, sim ci_tiny_dark, one run per row). The rig, the flash and the triangulation figure are illustrative.

  python/video/render.sh scenes/s03_seeing.py Seeing
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Group, ImageMobject, Line,
                   Polygon, Rectangle, Text, VGroup, VMobject)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import REPO  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

PACK = DATA / "pack_collab60d"
PH5 = REPO / "evaluation/runs/ph5/d145_rows_20261023T113916Z/replays"


def ate_all_frames(row: str) -> float:
    t = (PH5 / row / "results.yaml").read_text()
    return float(re.search(r"ate_se3: \{rmse_m: ([0-9.e-]+)", t).group(1))


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
        im = Image.open(PACK / "frames" / "final_agent0.jpg").convert("RGB")
        im = Image.merge("RGB", im.split()[::-1])  # BGR on disk (see B, frame_dump convention)
        frame = ImageMobject(np.asarray(im)).set_height(4.3).move_to(RIGHT * 1.9 + UP * 0.45)
        ill = tag("illustrative").next_to(rig, UP, 0.15)
        self.say("Each robot looks through two cameras, like a pair of eyes, and carries its own lamp.",
                 run_with=[FadeIn(rig, run_time=1.2), FadeIn(ill), FadeIn(frame, run_time=1.4)])

        # 3. real features on the real frame
        ov = json.loads((PACK / "frames" / "overlay_final_agent0.json").read_text())
        sx, sy = frame.width / ov["w"], frame.height / ov["h"]
        tl = frame.get_corner(UP + LEFT)
        dots = VGroup(*[Circle(0.035, color=THREAD, stroke_width=1.6).move_to(tl + RIGHT * x * sx + DOWN * y * sy)
                        for x, y in ov["pts"]])
        flbl = VGroup(Text(f"{len(ov['pts'])} tracked points", font=FONT, font_size=20, color=THREAD),
                      Text("FAST corners + ORB · pyramidal Lucas–Kanade", font=FONT, font_size=16, color=DIM)
                      ).arrange(DOWN, buff=0.08).next_to(frame, DOWN, 0.15)
        real = tag("measured", "recorded run").next_to(frame, UP, 0.12).align_to(frame, RIGHT)
        self.say("In every frame it picks out distinctive points, like corners and scuffs on the wall, and follows them "
                 "as it moves.", run_with=[FadeIn(dots, lag_ratio=0.004, run_time=2.4), FadeIn(flbl), FadeIn(real)])

        # 4. depth from two views (illustrative)
        self.play(FadeOut(Group(frame, dots, flbl, real)), FadeOut(rig), FadeOut(ill), run_time=0.5)
        cl, cr, P = np.array([-1.4, -1.3, 0]), np.array([0.6, -1.3, 0]), np.array([-0.1, 1.6, 0])
        geo = VGroup(Dot(cl, color=AGENT[0]), Dot(cr, color=AGENT[0]), Dot(P, color=THREAD, radius=0.09),
                     Line(cl, P, color=AGENT[0]), Line(cr, P, color=AGENT[0]),
                     DashedLine(cl, cr, color=DIM), Text("baseline", font=FONT, font_size=16, color=DIM).next_to(
                         (cl + cr) / 2, DOWN, 0.15))
        eq = Text("depth = focal length × baseline ÷ disparity", font=FONT, font_size=22, color=TEXT)
        pipe = Text("stereo rectification → SGBM → triangulation", font=FONT, font_size=18, color=DIM)
        side = VGroup(eq, pipe).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to(RIGHT * 3.4 + UP * 0.3)
        ill2 = tag("illustrative").to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.3)
        self.say("With two eyes, it can judge how far away each point is.",
                 run_with=[Create(geo, run_time=1.6), FadeIn(side), FadeIn(ill2)])
        self.play(FadeOut(VGroup(geo, side)), run_time=0.4)

        # 5. a flash, and active exposure (real trace)
        rows = list(csv.DictReader(open(PH5 / "r06_failE_expo" / "exposure_trace.csv")))
        tr = np.array([[int(r["decided_ns"]) / 1e9, float(r["saturated_frac"]), float(r["exposure_rel"])] for r in rows])
        tr = tr[(tr[:, 0] >= 15) & (tr[:, 0] <= 27)]
        X0, X1, Y0, Y1 = -5.6, 0.6, -1.6, 2.2
        px = lambda t: X0 + (t - 15) / 12 * (X1 - X0)  # noqa: E731
        py = lambda v: Y0 + v / 0.08 * (Y1 - Y0)  # noqa: E731  (saturation 0..8 %)
        axes = VGroup(Line([X0, Y0, 0], [X1, Y0, 0], color=DIM), Line([X0, Y0, 0], [X0, Y1, 0], color=DIM),
                      Text("time (s)", font=FONT, font_size=16, color=DIM).next_to([X1, Y0, 0], DOWN, 0.1),
                      Text("overexposed pixels", font=FONT, font_size=16, color=DIM).next_to([X0, Y1, 0], UP, 0.1))
        for s in (15, 20, 25):
            axes.add(Text(f"{s}", font=FONT, font_size=14, color=DIM).next_to([px(s), Y0, 0], DOWN, 0.08))
        for v in (0.0, 0.04, 0.08):
            axes.add(Text(f"{v * 100:.0f} %", font=FONT, font_size=14, color=DIM).next_to([X0, py(v), 0], LEFT, 0.08))
        sat = VMobject(stroke_color=TARGET, stroke_width=4)
        sat.set_points_as_corners([[px(t), py(v), 0] for t, v, _ in tr])
        event = DashedLine([px(20), Y0, 0], [px(20), Y1, 0], color=THREAD)
        ev_lbl = Text("+2-stop flash", font=FONT, font_size=16, color=THREAD).next_to(event, UP, 0.05)
        t_clear = float(tr[(tr[:, 0] > 20) & (tr[:, 1] == 0)][0, 0])
        clear = Text(f"clear again in {t_clear - 20:.2f} s", font=FONT, font_size=20, color=TEXT).next_to(
            [px(t_clear), py(0.0), 0], UP + RIGHT, 0.15)
        ate_on, ate_off = ate_all_frames("r06_failE_expo") * 1000, ate_all_frames("r07_failE_fixed") * 1000
        res = VGroup(Text("trajectory error over the run", font=FONT, font_size=20, color=TEXT),
                     VGroup(Text(f"{ate_on:.1f} mm", font=FONT, font_size=30, color=THREAD, weight="BOLD"),
                            Text("active exposure", font=FONT, font_size=18, color=TEXT)).arrange(RIGHT, buff=0.25),
                     VGroup(Text(f"{ate_off:.1f} mm", font=FONT, font_size=30, color=DIM, weight="BOLD"),
                            Text("fixed exposure", font=FONT, font_size=18, color=DIM)).arrange(RIGHT, buff=0.25),
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to(RIGHT * 3.9 + UP * 0.5)
        chip = tag("measured", "replay, dark world, D147").next_to(res, DOWN, 0.3).align_to(res, LEFT)
        flash = Rectangle(width=16, height=10, stroke_width=0, fill_color="#ffffff", fill_opacity=0.0)
        self.play(FadeOut(ill2), run_time=0.2)
        self.add(flash)
        self.play(flash.animate(run_time=0.25).set_fill(opacity=0.9))
        self.play(flash.animate(run_time=0.7).set_fill(opacity=0))
        self.say("And if a sudden flash blinds it, it turns its own exposure down, and is seeing clearly again in a "
                 "couple of seconds.",
                 run_with=[FadeIn(axes, run_time=0.6), Create(event, run_time=0.6), FadeIn(ev_lbl),
                           Create(sat, run_time=3.0)])
        self.play(FadeIn(clear), FadeIn(res, lag_ratio=0.2), FadeIn(chip), run_time=1.0)
        self.clear_caption()
        self.wait(2.0)
        self.play(FadeOut(VGroup(axes, sat, event, ev_lbl, clear, res, chip)), run_time=0.5)
        self.remove(flash)
