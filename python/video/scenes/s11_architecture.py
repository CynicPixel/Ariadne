"""Beat 11: under the hood. The architecture (OpenCV 5 on each robot, the team layer, AWS) and what it costs to run.

Diagram content: docs/COMPLETION_REGISTER.md HG1/HG2 rows and RESULTS_INDEX (only components that have run). gsplat on
AWS GPU is not drawn until A2's run lands; COOL is not drawn. Costs: RESULTS_INDEX §12 (D178 sweep $0.0875; EuRoC
11 flights $0.0439; pre-merge gate ~$0.035). `ArchitectureStill` renders the same diagram as one frame for the
submission's standalone architecture figure:

  python/video/render.sh scenes/s11_architecture.py Architecture
  python/video/render.sh scenes/s11_architecture.py ArchitectureStill -s -qk    # 4K PNG
"""
from __future__ import annotations

import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, FadeOut, RoundedRectangle, Scene, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.style import AGENT, DIM, FONT, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

AWS = "#ff9900"
CV = "#5c9ded"
GOOD = "#56d364"


def box(title: str, items: list[str], color, width: float) -> VGroup:
    head = Text(title, font=FONT, font_size=19, color=color, weight="BOLD")
    body = VGroup(*[Text(s, font=FONT, font_size=14, color=TEXT) for s in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
    inner = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
    frame = RoundedRectangle(corner_radius=0.12, width=width, height=inner.height + 0.35, stroke_color=color,
                             stroke_width=2, fill_color="#151a22", fill_opacity=0.95)
    return VGroup(frame, inner.move_to(frame).align_to(frame, LEFT).shift(RIGHT * 0.18))


def diagram() -> VGroup:
    robot = VGroup(
        box("Perception · OpenCV 5", ["stereo rectification, SGBM depth", "FAST + ORB, pyramidal Lucas–Kanade",
                                      "cv::dnn: EigenPlaces place fingerprints", "ALIKED + LightGlue verification",
                                      "findEssentialMat, PnP, LevMarq"], CV, 4.3),
        box("Estimation (from scratch)", ["stereo-inertial odometry, IMU preintegration",
                                          "windowed bundle adjustment (Schur + LM)", "pose graph, PCM → GNC-TLS",
                                          "SE-Sync-style certificate"], TEXT, 4.3),
        box("Mapping & planning", ["TSDF → ESDF clearance map", "D* Lite egress at person width",
                                   "CBBA task allocation"], TEXT, 4.3),
    ).arrange(DOWN, buff=0.18)
    robot_lbl = Text("on every robot  ×3", font=FONT, font_size=18, color=AGENT[0]).next_to(robot, UP, 0.12)
    team = VGroup(
        box("Team layer", ["elected broker with epochs", "two-stage exchange, bytes counted",
                           "spectral (λ₂) link selection", "known-entry gate", "FlatBuffers wire"], GOOD, 3.6),
        box("Human in the loop", ["AI chooser + C++ validation", "risk-gated approval",
                                  "JSONL decision trace"], THREAD, 3.6),
    ).arrange(DOWN, buff=0.18)
    aws = VGroup(
        box("Amazon EC2 c7i (spot)", ["pipeline replays, sweeps,", "pre-merge gates"], AWS, 3.6),
        box("Amazon EC2 g4dn (GPU)", ["Gazebo rendering (EGL)"], AWS, 3.6),
        box("Amazon Bedrock", ["Nova 2 Lite, tools via MCP"], AWS, 3.6),
        box("S3 · ECR · Terraform", ["recordings, images, infra"], AWS, 3.6),
    ).arrange(DOWN, buff=0.15)
    aws_lbl = Text("AWS", font=FONT, font_size=18, color=AWS, weight="BOLD").next_to(aws, UP, 0.12)
    cols = VGroup(VGroup(robot_lbl, robot), team, VGroup(aws_lbl, aws)).arrange(RIGHT, buff=0.45, aligned_edge=UP)
    links = VGroup(Arrow(robot[1].get_right(), team[0].get_left(), buff=0.06, color=DIM, stroke_width=3),
                   Arrow(team[1].get_right(), aws[2].get_left(), buff=0.06, color=DIM, stroke_width=3),
                   Arrow(team[0].get_right(), aws[0].get_left(), buff=0.06, color=DIM, stroke_width=3))
    return VGroup(cols, links).scale_to_fit_width(13.4).to_edge(UP, buff=0.35)


class Architecture(CaptionedScene):
    def construct(self):
        d = diagram()
        cols, links = d
        self.say("Under the hood, the vision is built on OpenCV 5, and the heavy lifting runs on AWS: simulation, "
                 "experiments and the AI assistant.",
                 run_with=[FadeIn(cols[0], shift=RIGHT * 0.2, run_time=1.4), FadeIn(cols[1], run_time=1.8),
                           FadeIn(links, run_time=2.0), FadeIn(cols[2], shift=LEFT * 0.2, run_time=2.4)])
        self.play(FadeOut(d), run_time=0.5)

        costs = VGroup(
            Text("What a run costs on AWS", font=FONT, font_size=28, color=TEXT),
            *[VGroup(Text(a, font=FONT, font_size=32, color=AWS, weight="BOLD"),
                     Text(b, font=FONT, font_size=21, color=TEXT)).arrange(RIGHT, buff=0.35)
              for a, b in (("$0.0875", "full collaboration sweep: 2 buildings × 3 trials"),
                           ("$0.0439", "all 11 EuRoC flights, full system"),
                           ("~$0.035", "complete pre-merge test gate"))],
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(UP * 0.5)
        chip = tag("measured", "spot instances, D178 / D87 / pre-merge").next_to(costs, DOWN, 0.35).align_to(costs, LEFT)
        self.say("A complete collaboration experiment, across two buildings and three trials, costs about nine cents.",
                 run_with=[FadeIn(costs, lag_ratio=0.2, run_time=1.6), FadeIn(chip)])
        replay = Text("every run replays byte for byte", font=FONT, font_size=24, color=GOOD).next_to(chip, DOWN, 0.4)
        self.say("And every run can be replayed, byte for byte.", run_with=[FadeIn(replay)])
        self.clear_caption()
        self.play(FadeOut(VGroup(costs, chip, replay)), run_time=0.5)


class ArchitectureStill(Scene):
    """The diagram alone, one frame: the submission's standalone architecture figure."""

    def construct(self):
        d = diagram().move_to((0, 0, 0))
        title = Text("Ariadne architecture", font=FONT, font_size=26, color=TEXT).to_edge(UP, buff=0.15)
        self.add(d.scale(0.95).next_to(title, DOWN, 0.2), title)
