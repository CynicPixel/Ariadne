"""Beat 4b: the from-scratch stereo-VIO + loops on EuRoC's 11 flights against ORB-SLAM3's published numbers.

Data: evidence.euroc() reads evaluation/runs/2.6/euroc_full_aws/<run 2>/runs/*/results.yaml (D87);
ORB-SLAM3 from its paper (tagged published). The worst flight (V1_03) is drawn, broken at the axis top, with its value.

  python/video/render.sh scenes/s04b_euroc.py EuRoC
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, DashedLine, FadeIn, GrowFromEdge, Line, Rectangle, Text,
                   VGroup, Write)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.evidence import ORB_SLAM3_CITE, euroc  # noqa: E402
from ariadne_video.style import DIM, FONT, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

YMAX_CM = 10.0
ORB_COLOR = "#7d8fa8"


class EuRoC(CaptionedScene):
    def construct(self):
        rows = euroc()
        W, H = 10.0, 4.6
        x0, y0 = -W / 2 - 0.5, -2.3  # room on the right for the mean labels
        sy = H / YMAX_CM
        axis = VGroup(Line([x0, y0, 0], [x0 + W, y0, 0], color=DIM), Line([x0, y0, 0], [x0, y0 + H, 0], color=DIM))
        ticks = VGroup()
        for cm in (0, 2.5, 5, 7.5, 10):
            y = y0 + cm * sy
            ticks.add(Line([x0 - 0.08, y, 0], [x0, y, 0], color=DIM),
                      Text(f"{cm:g}", font=FONT, font_size=16, color=DIM).next_to([x0 - 0.08, y, 0], LEFT, 0.08))
        ylab = Text("ATE (cm)", font=FONT, font_size=18, color=DIM).next_to(axis, UP, 0.1).align_to(axis, LEFT)
        title = Text("EuRoC MAV · 11 real flights", font=FONT, font_size=30, color=TEXT).to_edge(UP, buff=0.3)

        slot = W / len(rows)
        bw = slot * 0.34
        ours, orb, labels, notes = VGroup(), VGroup(), VGroup(), VGroup()
        for i, r in enumerate(rows):
            cx = x0 + slot * (i + 0.5)
            for val_m, grp, col, dx in ((r["ate_m"], ours, THREAD, -bw / 2 - 0.02), (r["orb_m"], orb, ORB_COLOR, bw / 2 + 0.02)):
                cm = val_m * 100
                h = min(cm, YMAX_CM) * sy
                bar = Rectangle(width=bw, height=max(h, 1e-3), stroke_width=0, fill_color=col, fill_opacity=0.9)
                bar.move_to([cx + dx, y0 + h / 2, 0])
                grp.add(bar)
                if cm > YMAX_CM:  # broken bar: the true value is printed, never hidden
                    notes.add(Text(f"{cm:.1f}", font=FONT, font_size=16, color=col).next_to(bar, UP, 0.06),
                              Line(bar.get_corner(UP + LEFT) + DOWN * 0.12, bar.get_corner(UP + RIGHT) + DOWN * 0.05,
                                   color="#0e1116", stroke_width=5))
            labels.add(Text(r["short"], font=FONT, font_size=15, color=DIM).move_to([cx, y0 - 0.22, 0]))

        legend = VGroup(
            VGroup(Rectangle(width=0.25, height=0.18, fill_color=THREAD, fill_opacity=0.9, stroke_width=0),
                   Text("Ariadne (stereo-VIO + loops)", font=FONT, font_size=18, color=TEXT)).arrange(RIGHT, buff=0.12),
            VGroup(Rectangle(width=0.25, height=0.18, fill_color=ORB_COLOR, fill_opacity=0.9, stroke_width=0),
                   Text("ORB-SLAM3", font=FONT, font_size=18, color=TEXT)).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.5).next_to(title, DOWN, 0.2)
        chips = VGroup(tag("measured", "Ariadne, D87"), tag("published", ORB_SLAM3_CITE)).arrange(RIGHT, buff=0.2)
        chips.to_corner(DOWN + RIGHT, buff=0.25).shift(UP * 0.55)

        self.play(FadeIn(title), Create(axis), FadeIn(ticks), FadeIn(ylab), FadeIn(labels), run_time=1.0)
        self.say("Tested first on public data: EuRoC, 11 real drone flights.", dur=3.2,
                 run_with=[FadeIn(legend), FadeIn(chips),
                           *[GrowFromEdge(b, DOWN, run_time=1.6) for b in orb]])
        self.say("Built from scratch on OpenCV 5, then measured against ORB-SLAM3.", dur=2.8,
                 run_with=[*[GrowFromEdge(b, DOWN, run_time=1.8) for b in ours], FadeIn(notes, run_time=1.8)])

        mean_ours = np.mean([r["ate_m"] for r in rows]) * 100
        mean_orb = np.mean([r["orb_m"] for r in rows]) * 100
        lines = VGroup()
        for m, col, txt in ((mean_ours, THREAD, f"mean {mean_ours:.1f} cm"), (mean_orb, ORB_COLOR, f"mean {mean_orb:.1f} cm")):
            y = y0 + m * sy
            ln = DashedLine([x0, y, 0], [x0 + W, y, 0], color=col, stroke_width=2)
            lines.add(ln, Text(txt, font=FONT, font_size=16, color=col).next_to(ln, RIGHT, 0.08).shift(UP * 0.1))
        n_ok = sum(round(r["ate_m"], 3) <= r["orb_m"] for r in rows)
        self.say(f"At or better on {n_ok} of {len(rows)} flights. Mean {mean_ours:.1f} cm vs {mean_orb:.1f} cm.",
                 dur=3.6, run_with=[*[Create(l) for l in lines]])
        self.say("Not yet ORB-SLAM3 on average. The hardest flights are where we still lose.", dur=3.0)
        self.clear_caption()
