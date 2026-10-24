"""Beat 6a: trust, but verify. A real false match, the verification funnel, two-view geometry (ALIKED + LightGlue vs
ORB), PCM's consistency clique, and GNC-TLS letting the false matches go.

Real data (recorded run collab60d, gate OFF, B's pack): the aliased keyframe pair agent 0 @ 44.1 s / agent 1 @ 45.1 s
(frames f880 / f900, true-colour PNGs), its verify counts and its GT error from collab/edges.csv; the funnel = the verify
reasons over all candidate edges of that run. Results: RESULTS_INDEX §1 (D63) and §5 (D75). The two-view figure, the
PCM graph and the GNC line fit are illustrative (the GNC weights are computed here with the TLS rule on toy data).

  python/video/render.sh scenes/s06a_verify.py Verify
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Group, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, ImageMobject,
                   Line, Polygon, Rectangle, Text, Transform, ValueTracker, VGroup, always_redraw)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.mobjects import CodePanel  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

PACK = DATA / "pack_collab60d"  # copy of B's build/demo_pack/collab60d (exporter python/analysis/manim_pack.py @300f4c1)
GOOD = "#56d364"

GNC = [
    "w = 1 for every loop;   μ = small",
    "repeat:",
    "  x = argmin Σ w·r(x)²     # weighted fit",
    "  w = TLS weights(r, μ)     # trust inliers",
    "  μ = 1.4 · μ               # less convex",
    "until the weights settle",
]


def frame_image(agent: int, frame: int, height: float) -> ImageMobject:
    im = Image.open(PACK / "frames" / f"agent{agent}" / f"agent{agent}_f{frame}.png").convert("RGB")
    return ImageMobject(np.asarray(im)).set_height(height)


def tls_weights(r2: np.ndarray, mu: float, c2: float) -> np.ndarray:
    """GNC-TLS weight update (Yang et al., RA-L 2020): 1 inside, 0 outside, a linear ramp in between."""
    lo, hi = mu / (mu + 1) * c2, (mu + 1) / mu * c2
    w = np.sqrt(c2 * mu * (mu + 1) / np.maximum(r2, 1e-12)) - mu
    return np.where(r2 <= lo, 1.0, np.where(r2 >= hi, 0.0, np.clip(w, 0, 1)))


class Verify(CaptionedScene):
    def construct(self):
        edges = [r for r in csv.DictReader(open(PACK / "collab" / "edges.csv"))]
        pair = next(r for r in edges if (r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]) == ("0", "96", "1", "52"))

        # 1. a real false match
        a = frame_image(0, 880, 3.4).move_to(LEFT * 3.55 + UP * 0.95)
        b = frame_image(1, 900, 3.4).move_to(RIGHT * 3.55 + UP * 0.95)
        la = Text(f"robot 1 · t = {float(pair['t_s_a']):.0f} s", font=FONT, font_size=18, color=AGENT[0]).next_to(a, UP, 0.12)
        lb = Text(f"robot 2 · t = {float(pair['t_s_b']):.0f} s", font=FONT, font_size=18, color=AGENT[1]).next_to(b, UP, 0.12)
        rec = tag("measured", "recorded run, gate off").to_edge(UP, buff=0.2).to_edge(RIGHT, buff=0.3)
        self.say("Matching places is risky.", run_with=[FadeIn(a), FadeIn(b), FadeIn(la), FadeIn(lb), FadeIn(rec)])
        same = Text("=?", font=FONT, font_size=44, color=THREAD, weight="BOLD").move_to(UP * 0.95)
        verdict = VGroup(
            Text(f"{int(pair['matches'])} feature matches · passes the geometry check", font=FONT, font_size=22, color=TEXT),
            Text(f"truth: {float(pair['gt_err_t_m']):.1f} m apart", font=FONT, font_size=26, color=TARGET, weight="BOLD"),
        ).arrange(DOWN, buff=0.15).next_to(Group(a, b), DOWN, buff=0.3)
        self.say("Many corridors here look identical, and a robot can be fooled into thinking, “I've been here before.”",
                 run_with=[FadeIn(same, scale=1.3), FadeIn(verdict[0], run_time=1.0)])
        self.play(FadeIn(verdict[1], scale=1.2), same.animate.set_color(TARGET), run_time=0.8)
        self.wait(0.6)
        self.play(FadeOut(Group(a, b, la, lb, same, verdict)), run_time=0.6)

        # 2. the gauntlet: what the geometry checks stopped in that run, and what still got through
        reasons: dict[str, int] = {}
        for r in edges:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        total = sum(reasons.values())
        passed = reasons.get("verified", 0)
        stages = [("candidate matches", total, TEXT), ("two-view geometry (E, PnP, degeneracy)", total - passed, DIM),
                  ("got through", passed, TARGET)]
        bars = VGroup()
        for i, (name, n, col) in enumerate(stages):
            w = 5.5 * n / total
            bar = Rectangle(width=max(w, 0.06), height=0.42, stroke_width=0, fill_color=col, fill_opacity=0.85)
            lbl = Text(f"{n}  {name if i != 1 else 'stopped by ' + name}", font=FONT, font_size=19, color=TEXT)
            bars.add(VGroup(bar, lbl.next_to(bar, RIGHT, 0.2)))
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(UP * 0.9)
        note = Text(f"and all {passed} that got through were false", font=FONT, font_size=24,
                    color=TARGET).next_to(bars, DOWN, buff=0.45).align_to(bars, LEFT)
        self.say("So every match runs a gauntlet.",
                 run_with=[*[FadeIn(b_, shift=RIGHT * 0.3) for b_ in bars]])
        self.play(FadeIn(note), run_time=0.6)
        self.wait(0.8)
        self.play(FadeOut(VGroup(bars, note)), run_time=0.5)

        # 3. two-view geometry + the registration result
        c1, c2, X = np.array([-3.2, -0.6, 0]), np.array([0.2, -0.9, 0]), np.array([-1.4, 2.0, 0])
        cam = lambda c, col: Polygon(c, c + [-0.35, 0.5, 0], c + [0.35, 0.5, 0], color=col, stroke_width=3)  # noqa: E731
        geo = VGroup(cam(c1, AGENT[0]), cam(c2, AGENT[1]), Dot(X, color=THREAD, radius=0.08),
                     Line(c1, X, color=AGENT[0]), Line(c2, X, color=AGENT[1]), DashedLine(c1, c2, color=DIM),
                     Polygon(c1, c2, X, stroke_width=0, fill_color=THREAD, fill_opacity=0.12))
        geo_lbl = Text("epipolar constraint: both rays must meet", font=FONT, font_size=18, color=DIM).next_to(geo, DOWN, 0.45)
        tbl = VGroup(
            Text("Two-view registration on EuRoC loops", font=FONT, font_size=22, color=TEXT),
            VGroup(Text("ALIKED + LightGlue", font=FONT, font_size=22, color=THREAD),
                   Text("0.607", font=FONT, font_size=30, color=THREAD, weight="BOLD")).arrange(RIGHT, buff=0.4),
            VGroup(Text("ORB", font=FONT, font_size=22, color=DIM),
                   Text("0.284", font=FONT, font_size=30, color=DIM, weight="BOLD")).arrange(RIGHT, buff=0.4),
            Text("pose accuracy (AUC@10°) · better on 11 of 11 flights", font=FONT, font_size=17, color=TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(RIGHT * 3.6 + UP * 0.9)
        tchip = tag("measured", "D63").next_to(tbl, DOWN, buff=0.25).align_to(tbl, LEFT)
        ichip = tag("illustrative").to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.3)
        self.play(FadeOut(rec), FadeIn(ichip), run_time=0.3)
        self.say("Does the geometry between the two views really add up?",
                 run_with=[Create(geo, run_time=1.6), FadeIn(geo_lbl)])
        self.play(FadeIn(tbl, lag_ratio=0.2, run_time=1.2), FadeIn(tchip))
        self.wait(1.6)
        self.play(FadeOut(VGroup(geo, geo_lbl, tbl, tchip)), run_time=0.5)

        # 4. PCM: keep the largest set of loops that agree pairwise
        rng = np.random.default_rng(7)
        true_pts = [np.array([np.cos(t) * 1.6 - 1.2, np.sin(t) * 1.6 + 0.8, 0]) for t in np.linspace(0.3, 5.9, 7)]
        false_pts = [np.array([3.2 + 0.7 * np.cos(t), 0.8 + 0.7 * np.sin(t), 0]) for t in np.linspace(0, 4.2, 3)]
        nodes_t = VGroup(*[Dot(p, radius=0.11, color=GOOD) for p in true_pts])
        nodes_f = VGroup(*[Dot(p, radius=0.11, color=TARGET) for p in false_pts])
        links = VGroup(*[Line(p, q, stroke_width=1.5, color=GOOD, stroke_opacity=0.6)
                         for i, p in enumerate(true_pts) for q in true_pts[i + 1:]],
                       *[Line(p, q, stroke_width=1.5, color=TARGET, stroke_opacity=0.6)
                         for i, p in enumerate(false_pts) for q in false_pts[i + 1:]])
        pcm_lbl = Text("PCM: loops that agree pairwise are linked;\nkeep the largest fully linked group",
                       font=FONT, font_size=18, color=DIM).to_edge(UP, buff=0.75)
        self.say("Do the matches agree with one another?",
                 run_with=[FadeIn(nodes_t), FadeIn(nodes_f), Create(links, run_time=1.4), FadeIn(pcm_lbl)])
        hull = Circle(radius=2.05, color=GOOD, stroke_width=3).move_to([-1.2, 0.8, 0])
        self.play(Create(hull), VGroup(nodes_f, links[-3:]).animate.set_opacity(0.25), run_time=0.9)
        self.play(FadeOut(VGroup(nodes_t, nodes_f, links, hull, pcm_lbl)), run_time=0.5)

        # 5. GNC-TLS: robust fit where the weights of the outliers anneal to zero
        xs = np.linspace(-3, 3, 16)
        ys = 0.45 * xs + 0.6 + rng.normal(0, 0.12, xs.size)
        out = np.array([2, 5, 11, 13])
        ys[out] += np.array([2.1, -1.9, 2.4, -2.2])
        origin = LEFT * 3.2 + UP * 0.6
        P = lambda x, y: origin + np.array([x * 0.75, y * 0.75, 0])  # noqa: E731
        c2 = 0.35 ** 2
        state = {"w": np.ones(xs.size), "fit": None}

        def wls(w):
            A = np.stack([xs, np.ones_like(xs)], 1) * np.sqrt(w)[:, None]
            return np.linalg.lstsq(A, ys * np.sqrt(w), rcond=None)[0]

        mu_t = ValueTracker(0.0)

        def step(k):
            w = np.ones(xs.size)
            mu = 0.05
            for _ in range(k):
                m, c = wls(w)
                w = tls_weights((ys - (m * xs + c)) ** 2, mu, c2)
                mu *= 1.4
            return w, wls(w)

        sched = [step(k) for k in range(0, 15)]
        pts = always_redraw(lambda: VGroup(*[
            Dot(P(x, y), radius=0.07, color=TARGET if i in out else TEXT,
                fill_opacity=0.15 + 0.85 * sched[int(mu_t.get_value())][0][i]) for i, (x, y) in enumerate(zip(xs, ys))]))
        line = always_redraw(lambda: Line(P(-3.3, sched[int(mu_t.get_value())][1][0] * -3.3 + sched[int(mu_t.get_value())][1][1]),
                                          P(3.3, sched[int(mu_t.get_value())][1][0] * 3.3 + sched[int(mu_t.get_value())][1][1]),
                                          color=THREAD, stroke_width=4))
        code = CodePanel("GNC-TLS  (graduated non-convexity)", GNC, font_size=16).to_edge(RIGHT, buff=0.3).shift(UP * 0.6)
        self.play(FadeIn(pts), FadeIn(line), FadeIn(code), code.focus(0), run_time=0.8)
        self.say("Then the remaining evidence pulls gently, tightening step by step until the false matches let go.",
                 run_with=[mu_t.animate(run_time=4.5).set_value(len(sched) - 1), code.focus(4, run_time=0.6)])
        self.play(code.focus(5), run_time=0.4)
        self.clear_caption()

        # the design choice, measured
        self.play(FadeOut(VGroup(pts, line, code, ichip)), run_time=0.5)
        ramp = VGroup(
            Text("Why PCM first, then GNC?", font=FONT, font_size=28, color=TEXT),
            VGroup(Text("90 %", font=FONT, font_size=34, color=THREAD, weight="BOLD"),
                   Text("false loops, and the map is still exact (groups of 10)", font=FONT, font_size=22,
                        color=TEXT)).arrange(RIGHT, buff=0.3),
            VGroup(Text("~10×", font=FONT, font_size=34, color=THREAD, weight="BOLD"),
                   Text("faster than GNC alone", font=FONT, font_size=22, color=TEXT)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).shift(UP * 0.6)
        rchip = tag("measured", "synthetic aliasing ramp, D75").next_to(ramp, DOWN, buff=0.35).align_to(ramp, LEFT)
        self.play(FadeIn(ramp, lag_ratio=0.25, run_time=1.2), FadeIn(rchip))
        self.wait(2.6)
        self.play(FadeOut(VGroup(ramp, rchip)), run_time=0.5)
