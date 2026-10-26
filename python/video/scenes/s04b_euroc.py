"""Beat 4b: the from-scratch stereo-VIO + loops on EuRoC's 11 flights against ORB-SLAM3's published numbers.

Data: evidence.euroc() reads evaluation/runs/vio/euroc_full_aws/<run 2>/runs/*/results.yaml (D87);
ORB-SLAM3 from its paper (tagged published). The cut shows the wins: the 7 flights at or better than
ORB-SLAM3 (at its published 3-dp precision) light up; the other 4 stay visible, dimmed, so "7 of 11" stays honest. No
mean line (docs/results/index.md §3 and do-not-claim 2: never parity). V1_03 is drawn broken at the axis top with its value.

  python/video/render.sh scenes/s04b_euroc.py EuRoC
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, FadeIn, GrowFromEdge, Line, Rectangle,
                   Succession, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.evidence import ORB_SLAM3_CITE, euroc  # noqa: E402
from ariadne_video.style import DIM, FONT, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

YMAX_CM = 10.0
ORB_COLOR = "#7d8fa8"


class EuRoC(CaptionedScene):
    def construct(self):
        rows = euroc()
        W, H = 10.6, 4.0
        x0, y0 = -W / 2 + 0.2, -1.95  # chart sits above the subtitle band
        sy = H / YMAX_CM
        axis = VGroup(Line([x0, y0, 0], [x0 + W, y0, 0], color=DIM), Line([x0, y0, 0], [x0, y0 + H, 0], color=DIM))
        ticks = VGroup()
        for cm in (0, 2.5, 5, 7.5, 10):
            y = y0 + cm * sy
            ticks.add(Line([x0 - 0.08, y, 0], [x0, y, 0], color=DIM),
                      Text(f"{cm:g}", font=FONT, font_size=16, color=DIM).next_to([x0 - 0.08, y, 0], LEFT, 0.08))
        ylab = Text("trajectory error (cm) · lower is better", font=FONT, font_size=17, color=DIM)
        ylab.next_to(axis, UP, 0.12).align_to(axis, LEFT)
        title = Text("EuRoC MAV · 11 real drone flights", font=FONT, font_size=30, color=TEXT).to_edge(UP, buff=0.3)

        slot = W / len(rows)
        bw = slot * 0.34
        wins = [round(r["ate_m"], 3) <= r["orb_m"] for r in rows]  # at or better at ORB-SLAM3's published precision
        ours, orb, labels, notes = VGroup(), VGroup(), VGroup(), VGroup()
        note_of = {}
        for i, r in enumerate(rows):
            cx = x0 + slot * (i + 0.5)
            for val_m, grp, col, dx in ((r["ate_m"], ours, THREAD, -bw / 2 - 0.02), (r["orb_m"], orb, ORB_COLOR, bw / 2 + 0.02)):
                cm = val_m * 100
                h = min(cm, YMAX_CM) * sy
                bar = Rectangle(width=bw, height=max(h, 1e-3), stroke_width=0, fill_color=col, fill_opacity=0.9)
                bar.move_to([cx + dx, y0 + h / 2, 0])
                grp.add(bar)
                if cm > YMAX_CM:  # broken bar: the true value is printed, never hidden
                    notes.add(VGroup(Text(f"{cm:.1f}", font=FONT, font_size=16, color=col).next_to(bar, UP, 0.06),
                              Line(bar.get_corner(UP + LEFT) + DOWN * 0.12, bar.get_corner(UP + RIGHT) + DOWN * 0.05,
                                   color="#0e1116", stroke_width=5)))
                    note_of[i] = notes[-1]
            labels.add(Text(r["short"], font=FONT, font_size=15, color=DIM).move_to([cx, y0 - 0.22, 0]))

        legend = VGroup(
            VGroup(Rectangle(width=0.25, height=0.18, fill_color=THREAD, fill_opacity=0.9, stroke_width=0),
                   Text("Ariadne (stereo-VIO + loops)", font=FONT, font_size=18, color=TEXT)).arrange(RIGHT, buff=0.12),
            VGroup(Rectangle(width=0.25, height=0.18, fill_color=ORB_COLOR, fill_opacity=0.9, stroke_width=0),
                   Text("ORB-SLAM3", font=FONT, font_size=18, color=TEXT)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.5).next_to(title, DOWN, 0.2)
        chips = VGroup(tag("measured", "Ariadne, D87"), tag("published", ORB_SLAM3_CITE)).arrange(DOWN, buff=0.1,
                                                                                                aligned_edge=RIGHT)
        chips.to_edge(RIGHT, buff=0.25).set_y(-2.75)  # bottom right, clear of the centred subtitles

        self.play(FadeIn(title), Create(axis), FadeIn(ticks), FadeIn(ylab), FadeIn(labels), FadeIn(legend),
                  FadeIn(chips), run_time=1.0)
        self.say("We built this from scratch, and before trusting it here, we tested it on real drone flights.",
                 run_with=[Succession(AnimationGroup(*[GrowFromEdge(b, DOWN) for b in orb], run_time=1.4),
                                      AnimationGroup(*[GrowFromEdge(b, DOWN) for b in ours], FadeIn(notes),
                                                     run_time=1.6))])

        n_ok = sum(wins)
        dim = [m for i, w in enumerate(wins) if not w for m in (ours[i], orb[i], labels[i])]
        dim += [n[0] for i, n in note_of.items() if not wins[i]]  # a dimmed flight's printed value dims with it
        lit = [labels[i] for i, w in enumerate(wins) if w]
        count = Text(f"at or better on {n_ok} of {len(rows)} flights", font=FONT, font_size=24, color=THREAD)
        count.move_to([0, y0 + H - 0.25, 0]).align_to([x0 + 0.25, 0, 0], LEFT)
        self.say("On seven of eleven, it matched or beat ORB-SLAM3, one of the field's best-known systems.",
                 run_with=[*[m.animate(run_time=1.0).set_opacity(0.22) for m in dim],
                           *[l.animate(run_time=1.0).set_color(THREAD) for l in lit],
                           FadeIn(count, shift=UP * 0.1, run_time=1.0)])
        self.clear_caption()
        self.wait(1.2)
