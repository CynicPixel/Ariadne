"""Beat 11 (≈ 45 s, the film's last beat): under the hood, then the conclusion and the author. The architecture as three lanes (docs/Latex/ariadne_architecture_diagram:
per-robot OpenCV 5 estimation, selective collaboration, closed-loop mission control and outputs) over the AWS band.
A data pulse walks the pipeline; each component shows its headline result as the pulse reaches it.

Every number is the report's (docs/report/main.tex, "What the experiments establish" table, and the AWS section):
measured on its stated benchmark; the spectral row is on synthetic instances (said on screen). No EuRoC mean ATE and
no COOL speed-up are shown (neither is a claim the evidence supports).

The conclusion returns to the maze with the route the project's D* Lite found there (data/dstar, beat 9's run), then
the author card (docs/me/me.png; wording from the author). It closes the film.

  python/video/render.sh scenes/s11_architecture.py Architecture
"""
from __future__ import annotations

import sys
from pathlib import Path

import json

import numpy as np
from PIL import Image
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, Dot, FadeIn, FadeOut, Group, ImageMobject, LaggedStart,
                   RoundedRectangle, Text, VGroup, smooth)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import REPO  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.slides import chip_top  # noqa: E402
from ariadne_video.style import DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene  # noqa: E402

AWS = "#ff9900"
GOOD = "#56d364"
CHIP_BOTTOM, CAP2 = 3.4, -2.93
BW, BH, GX = 2.95, 1.0, 0.38  # box width, height; gap between boxes

# (lane title, colour, [(component, method, result)])  — results: docs/report/main.tex tab:wins and the AWS section
LANES = [
    ("each robot · OpenCV 5 on every frame", "#39c5bb", [
        ("Exposure control", "flat-field · lamp · active exposure", "recovers from 2 stops over in 2.35 s"),
        ("Visual front end", "rectify · pyramidal LK · ORB · SGBM", "bit-identical at 2, 5 and 16 threads"),
        ("Stereo-inertial SLAM", "IMU factors · Schur window BA", "≤ ORB-SLAM3 on 7 of 11 EuRoC flights"),
        ("Pose-graph solver", "chordal init · LM · certificate", "= GTSAM on 100 of 100 graphs"),
    ]),
    ("the team · over a weak radio", "#a371f7", [
        ("Recognize overlap", "EigenPlaces · cosine index", "recall@1 0.626 (random 0.167)"),
        ("Spend bytes on evidence", "spectral λ₂ selection", "0.970 vs greedy 0.936 (synthetic)"),
        ("Verify geometry", "ALIKED + LightGlue · PnP", "pose AUC 0.284 → 0.607"),
        ("Robust shared map", "known-entry gate → PCM → GNC-TLS", "merged map 3.78 m → ≤ 1.5 cm"),
    ]),
    ("decisions and outputs", "#56d364", [
        ("World model", "TSDF → ESDF · person clearance", "occupied precision 0.707 → 0.865"),
        ("Explore + allocate", "next-best view · CBBA", "0.994 of the optimal allocation"),
        ("AI decisions", "Bedrock Nova 2 Lite · MCP · C++ check", "80 of 80 picks validated, live"),
        ("Way out + model", "D* Lite egress · gsplat", "12 of 12 routes passable for a person"),
    ]),
]
AWS_LINE = ("AWS", "EC2 c7i · Graviton4 · GPU · Amazon Bedrock · S3 + ECR · CloudFront console",
            "708 of 708 C++ tests and 54 of 54 integration checks on AWS · ALIKED 530 ms on Graviton4 vs 829 ms on c7i")


def box(name: str, method: str, color: str) -> VGroup:
    frame = RoundedRectangle(corner_radius=0.1, width=BW, height=BH, stroke_color=color, stroke_width=1.5,
                             stroke_opacity=0.6, fill_color="#151a22", fill_opacity=0.95)
    body = VGroup(Text(name, font=FONT, font_size=15, color=TEXT, weight="BOLD"),
                  Text(method, font=FONT, font_size=11, color=DIM)).arrange(DOWN, buff=0.07)
    body.move_to(frame).shift(UP * 0.17)
    return VGroup(frame, body)


class Architecture(CaptionedScene):
    def construct(self):
        lanes, boxes, results, heads = VGroup(), [], [], []
        for i, (title, color, comps) in enumerate(LANES):
            row = VGroup(*[box(n, m, color) for n, m, _ in comps]).arrange(RIGHT, buff=GX)
            if i == 1:  # the pipeline snakes: the team lane runs right to left, under the end of the robot lane
                row = VGroup(*reversed(row)).arrange(RIGHT, buff=GX)
                row = VGroup(*reversed(row))
            head = Text(title, font=FONT, font_size=13, color=color).next_to(row, UP, 0.1)
            head.align_to(row, RIGHT if i == 2 else LEFT)  # clear of the connector that enters the lane
            lanes.add(VGroup(head, row))
            heads.append(head)
            boxes += list(row)
            for b, (_, _, res) in zip(row, comps):
                results.append(Text(res, font=FONT, font_size=11, color=GOOD).move_to(b[0]).shift(DOWN * 0.27))
        lanes.arrange(DOWN, buff=0.22)
        for lane in lanes:  # arrange() centres each lane on its own width; the rows must line up, not the headers
            lane.shift(RIGHT * (lanes[0][1].get_x() - lane[1].get_x()))
        aws_box = RoundedRectangle(corner_radius=0.1, width=lanes.width, height=0.62, stroke_color=AWS,
                                   stroke_width=1.5, fill_color="#151a22", fill_opacity=0.95)
        aws_text = VGroup(Text(f"{AWS_LINE[0]}  ·  {AWS_LINE[1]}", font=FONT, font_size=13, color=AWS),
                          Text(AWS_LINE[2], font=FONT, font_size=11, color=GOOD)).arrange(DOWN, buff=0.06)
        aws = VGroup(aws_box, aws_text.move_to(aws_box))
        block = VGroup(lanes, aws).arrange(DOWN, buff=0.3)
        block.move_to([0, (CHIP_BOTTOM + CAP2) / 2, 0])
        for b, r in zip(boxes, results):  # results were placed before the block moved: follow their boxes
            r.move_to(b[0]).shift(DOWN * 0.27)
        assert block.width < 13.4 and block.height < CHIP_BOTTOM - CAP2 - 0.2, (block.width, block.height)
        assert all(r.width < BW - 0.12 for r in results), [r.text for r in results if r.width >= BW - 0.12]

        # the pipeline's order: robot lane left to right, team lane right to left, decisions lane left to right
        order = boxes  # the team lane's boxes are already listed in flow order (placed right to left)
        res_of = {id(b): r for b, r in zip(boxes, results)}
        links = VGroup()
        for a, b in zip(order, order[1:]):
            if abs(a.get_y() - b.get_y()) < 0.1:
                s, e = (a.get_right(), b.get_left()) if b.get_x() > a.get_x() else (a.get_left(), b.get_right())
            else:
                s, e = a.get_bottom(), b.get_top()
            links.add(Arrow(s, e, buff=0.05, color=DIM, stroke_width=2, max_tip_length_to_length_ratio=0.25))
        chip = chip_top("measured", "every number from the report's evaluation suite")
        pulse = Dot(order[0].get_center(), radius=0.07, color=THREAD).set_z_index(5)

        def light(k: int) -> list:
            """The box the pulse has reached brightens and its result appears."""
            b = order[k]
            return [b[0].animate(run_time=0.45).set_stroke(b[0].get_stroke_color(), width=2.6, opacity=1.0),
                    FadeIn(res_of[id(b)], shift=UP * 0.05, run_time=0.45)]

        def visit(k: int) -> list:
            """Two steps: the pulse travels to box k at the narration's pace, then the box lights."""
            go = lambda: [pulse.animate(run_time=0.8, rate_func=smooth).move_to(order[k].get_center() + UP * 0.5)]  # noqa: E731
            return [go, lambda: light(k)]

        def lane_steps(i: int) -> list:
            steps = [[FadeIn(heads[i]), *[FadeIn(b, shift=UP * 0.08) for b in lanes[i][1]],
                      *[FadeIn(links[k]) for k in range(4 * i - 1, 4 * i + 3) if 0 <= k < len(links)]]]
            for k in range(4 * i, 4 * i + 4):
                steps += visit(k)
            return steps

        pulse.move_to(order[0].get_center() + UP * 0.5)
        self.play(FadeIn(chip), run_time=0.4)
        first = lane_steps(0)
        first[1] = lambda: [FadeIn(pulse, scale=2, run_time=0.6)]  # the pulse starts at the first box
        self.say_steps("Under the hood, each robot runs its own OpenCV 5 vision pipeline, frame by frame;", first)
        self.say_steps("the team shares only what is worth the radio, and verifies every link before merging;",
                       lane_steps(1))
        self.say_steps("and the merged map drives exploration, the AI's choices, and the way out.", lane_steps(2))
        self.say_steps("It all runs on AWS, and every run can be replayed exactly.", [
            [FadeOut(pulse), FadeIn(aws, shift=UP * 0.1, run_time=0.8)]])
        self.wait(2.0)

        # ---- conclusion: back to the maze, with the route the project's D* Lite found there -------------------------
        self.play(*[FadeOut(m) for m in self.mobjects if m is not self._caption], run_time=0.6)
        sc = load_scene(DATA / "maze_demo_0")
        tr = json.loads((DATA / "dstar" / "trace.json").read_text())
        f = maze_frame(sc)
        walls = maze_walls(sc, f, opacity=0.8)
        bots = VGroup(*[agent_marker(e["agent"], f, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]))
                        for e in sc.starts])
        tx, ty = sc.target_xy_m
        casualty = VGroup(Circle(radius=0.13, color=TARGET, stroke_width=3).move_to(f.p(tx, ty)),
                          Dot(f.p(tx, ty), radius=0.04, color=TARGET))
        xy = np.array(sc.cell_to_xy(np.array(tr["path0"])))
        thread = polyline(f, xy, THREAD, width=7)
        glow = polyline(f, xy, THREAD, width=18, opacity=0.18)
        chip2 = chip_top("illustrative", "the route D* Lite found on the generated maze")
        self.play(FadeIn(walls), FadeIn(casualty), FadeIn(chip2), LaggedStart(*[FadeIn(b) for b in bots], lag_ratio=0.2),
                  run_time=0.8)
        self.say("In the end, Ariadne isn't about a pretty 3D model.",
                 run_with=[Create(glow, run_time=2.0), Create(thread, run_time=2.0)])
        self.say("It's about whether a few robots, each seeing only part of the picture, can build one map they "
                 "trust, and hand a responder the way out.")
        self.clear_caption()
        self.play(*[FadeOut(m) for m in (walls, thread, glow, bots, casualty, chip2)], run_time=0.8)

        # ---- the author ---------------------------------------------------------------------------------------------
        img = np.array(Image.open(REPO / "docs/me/me.png").convert("RGBA"))
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        r0 = min(img.shape[:2]) / 2
        img[..., 3] = np.where((yy - img.shape[0] / 2) ** 2 + (xx - img.shape[1] / 2) ** 2 <= r0 ** 2, 255, 0)
        photo = ImageMobject(img)
        photo.height = 2.3
        ring = Circle(radius=1.17, color=THREAD, stroke_width=3).move_to(photo)
        who = VGroup(
            Text("Suvojit Das", font=FONT, font_size=34, color=THREAD, weight="BOLD"),
            Text("GPU Systems Engineer · Qualcomm", font=FONT, font_size=21, color=TEXT),
            Text("B.Tech, Electrical Engineering\nM.Tech, Electrical Engineering · Signal Processing & Machine Learning",
                 font=FONT, font_size=15, color=DIM, line_spacing=0.9),
            Text("github.com/CynicPixel", font="Noto Mono", font_size=16, color=TEXT),
            Text("recreational programmer · coffee addict · dad-joke enthusiast", font=FONT, font_size=15,
                 color=THREAD, slant="ITALIC"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        who[1].shift(UP * 0.05)
        person = Group(Group(photo, ring), who).arrange(RIGHT, buff=0.6)
        title = Text("Ariadne", font=FONT, font_size=60, color=TEXT, weight="BOLD")
        foot = Text("OpenCV AI Competition 2026", font=FONT, font_size=18, color=DIM)
        card = Group(title, person, foot).arrange(DOWN, buff=0.5).move_to([0, 0.15, 0])
        title.align_to(who, LEFT)  # the title, the name and the footer share one left edge
        foot.align_to(who, LEFT)
        assert card.width < 13.4 and card.height < 7.2, (card.width, card.height)
        self.play(FadeIn(title, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(photo, scale=0.9), Create(ring), LaggedStart(*[FadeIn(t, shift=RIGHT * 0.1) for t in who],
                                                                     lag_ratio=0.2), run_time=1.6)
        self.play(FadeIn(foot), run_time=0.5)
        self.wait(3.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)
