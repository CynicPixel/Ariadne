"""Beat 3: seeing in the dark. Ride along with one robot: stereo cameras and a lamp, then its recorded camera frames
playing back with every tracked feature in its own colour.

Frames: the collab60d pack, agent 0, every 20th recorded frame f120-f360 (1 s apart; true-colour files, read as-is).
Tracks: an OFFLINE replay of the front end's per-frame loop with OpenCV 5 (tools/track_replay.py -> data/tracks/):
illustrative, not the live run's own tracks (the live tracker runs at 20 Hz and keeps no per-frame dump). The rig is
illustrative.

  python/video/render.sh scenes/s03_seeing.py Seeing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Circle, FadeIn, FadeOut, Group, Polygon, Rectangle, Text, VGroup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TEXT, THREAD, CaptionedScene, tag  # noqa: E402
from ariadne_video.tracks import TrackPlayer, image, load_tracks  # noqa: E402

PER_FRAME_S = 0.8  # playback: one recorded frame (1 s of the run) per 0.8 s of film


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
        tr, arrs = load_tracks()
        frame = image(arrs[0], 7.6).move_to(RIGHT * 1.9 + UP * 0.45)
        ill = tag("illustrative").next_to(rig, UP, 0.15)
        self.say("Each robot looks through two cameras, for stereo depth sensing, and carries its own illumination.",
                 run_with=[FadeIn(rig, run_time=1.2), FadeIn(ill), FadeIn(frame, run_time=1.4)])

        # 3. its recorded frames playing back, every tracked feature in its own colour
        self.play(FadeOut(rig), FadeOut(ill), frame.animate.set_width(8.6).move_to(UP * 0.5), run_time=0.8)
        player = TrackPlayer(tr, arrs, frame, PER_FRAME_S)
        chip = tag("illustrative", "recorded frames · tracks replayed offline").next_to(frame, UP, 0.12).align_to(
            frame, RIGHT)
        legend = VGroup(Text("detection: FAST corners + ORB descriptors", font=FONT, font_size=18, color=TEXT),
                        Text("each colour is one feature, followed from frame to frame", font=FONT, font_size=16,
                             color=DIM)).arrange(DOWN, buff=0.08).next_to(frame, DOWN, 0.15)
        player.start()
        self.say("In every frame it detects FAST corners, described with ORB, and follows them as it moves.", run_with=[FadeIn(player.overlay, run_time=0.5), FadeIn(chip), FadeIn(legend)])
        self.wait(2.5)
        self.clear_caption()
        player.stop()
        self.play(FadeOut(Group(frame, player.overlay, chip, legend)), run_time=0.6)
