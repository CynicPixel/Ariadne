"""Beat 4a: knowing where you are. A windowed bundle adjustment relaxing reprojection "springs", with a pseudo-code
panel stepping through the solver (IMU preintegration, normal equations, Schur complement, LM step).

The geometry is illustrative (a 2D toy: camera poses along an arc, landmarks, bearing residuals that shrink as the
pose estimate converges). The one number on screen, "solver matches GTSAM on 100 of 100 synthetic graphs", is
RESULTS_INDEX §3 (D52, synthetic graphs).

  python/video/render.sh scenes/s04a_ba.py BundleAdjust
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Dot, FadeIn, Line, Polygon, Rectangle, Succession, Text,
                   ValueTracker, VGroup, always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.mobjects import CodePanel  # noqa: E402
from ariadne_video.style import AGENT, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

CODE = [
    "windowed bundle adjustment",
    "1  preintegrate IMU between keyframes  (Forster)",
    "2  linearise: residuals r, Jacobian J",
    "3  H = JᵀJ + λI,   b = −Jᵀr",
    "4  Schur complement: eliminate the points",
    "5  solve the poses, back-substitute points",
    "6  cost fell? accept, λ↓  :  reject, λ↑",
    "7  repeat until it stops improving",
]


class BundleAdjust(CaptionedScene):
    def construct(self):
        rng = np.random.default_rng(4)
        # toy world (2D, metres -> frame units via a fixed offset/scale), left half of the frame
        n_cam, n_pts = 6, 14
        th = np.linspace(-0.5, 0.5, n_cam)
        cam_true = np.stack([np.sin(th) * 4.0 - 4.2, -np.cos(th) * 1.0 - 0.8, th * 0.6], axis=1)  # x, y, heading
        pts = np.stack([rng.uniform(-6.6, -1.4, n_pts), rng.uniform(0.9, 2.8, n_pts)], axis=1)
        cam_bad = cam_true + np.c_[rng.normal(0, 0.18, (n_cam, 2)), rng.normal(0, 0.12, n_cam)]
        cam_bad[0] = cam_true[0]  # the oldest pose is the window's anchor
        obs = [(i, j) for i in range(n_cam) for j in range(n_pts) if rng.random() < 0.45]
        a = ValueTracker(0.0)  # 0 = initial guess, 1 = converged

        def pose(i):
            return cam_bad[i] + (cam_true[i] - cam_bad[i]) * a.get_value()

        def frustum(i):
            x, y, h = pose(i)
            d = np.array([np.sin(h), np.cos(h)])
            n = np.array([d[1], -d[0]])
            c = np.array([x, y])
            tri = Polygon(*[np.r_[q, 0] for q in (c, c + 0.45 * d + 0.22 * n, c + 0.45 * d - 0.22 * n)],
                          color=AGENT[0], stroke_width=2, fill_opacity=0.25)
            return tri

        def springs():
            g = VGroup()
            for i, j in obs:
                xt, yt, ht = cam_true[i]
                bearing = np.arctan2(pts[j, 0] - xt, pts[j, 1] - yt) - ht  # what camera i measured
                x, y, h = pose(i)
                rng_m = np.hypot(pts[j, 0] - x, pts[j, 1] - y)
                pred = np.array([x + rng_m * np.sin(h + bearing), y + rng_m * np.cos(h + bearing)])
                g.add(Line([x, y, 0], np.r_[pred, 0], color=DIM, stroke_width=1, stroke_opacity=0.35))
                if np.hypot(*(pred - pts[j])) > 0.02:
                    g.add(Line(np.r_[pred, 0], np.r_[pts[j], 0], color=TARGET, stroke_width=3))
            return g

        cams = always_redraw(lambda: VGroup(*[frustum(i) for i in range(n_cam)]))
        path = always_redraw(lambda: VGroup(*[
            Line(np.r_[pose(i)[:2], 0], np.r_[pose(i + 1)[:2], 0], color=AGENT[0], stroke_width=2, stroke_opacity=0.6)
            for i in range(n_cam - 1)]))
        landmarks = VGroup(*[Dot(np.r_[p, 0], radius=0.05, color=THREAD) for p in pts])
        res = always_redraw(springs)
        chip = tag("illustrative").to_edge(UP, buff=0.25).to_edge(LEFT, buff=0.4)

        panel = CodePanel(CODE[0], CODE[1:], font_size=16).to_edge(RIGHT, buff=0.3).shift(UP * 0.6)

        def at(k):  # k counts CODE lines; line 0 is the panel title
            return panel.focus(k - 1, run_time=0.35)

        self.add(chip)
        self.say("Next, the robot has to work out where it is, using only what it sees and what its motion sensor "
                 "feels.", run_with=[FadeIn(cams), FadeIn(path), FadeIn(landmarks, lag_ratio=0.1),
                                     FadeIn(panel), at(1)])
        self.say("Each new view adds a little tension: points that should line up don't quite.",
                 run_with=[FadeIn(res, run_time=1.2), Succession(at(2), at(3))])

        steps = []
        for target in (0.55, 0.82, 0.95, 1.0):
            steps += [at(4), at(5), AnimationGroup(at(6), a.animate(run_time=0.9).set_value(target))]
        self.say("So, again and again, it nudges its recent path until everything agrees as well as it possibly can.",
                 run_with=[Succession(*steps, at(7))])

        claim = VGroup(Text("solver matches GTSAM on 100 of 100 synthetic graphs", font=FONT, font_size=20,
                            color=TEXT), tag("measured", "D52")).arrange(RIGHT, buff=0.25)
        claim.move_to(DOWN * 2.6)  # own row below the illustration; the subtitle is cleared first
        self.clear_caption()
        self.play(FadeIn(claim, shift=UP * 0.1), run_time=0.6)
        self.wait(1.6)
