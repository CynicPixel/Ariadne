"""Beat 5: sharing a map on a byte budget. Place fingerprints (EigenPlaces), the two-stage exchange, and spectral
(λ₂ / Fiedler) prioritisation of candidate links.

Data and tags:
- the three camera frames are real (collab60d pack, frames/agent*_f600.png; BGR on disk, swapped on read);
- fingerprint bars, packet flights and the byte meter are illustrative (no number on them);
- λ₂ values are computed live from the toy graph drawn here (illustrative);
- measured numbers (RESULTS_INDEX): EuRoC R@1 EigenPlaces 0.626 / mean-bit 0.444 / random 0.167 (§4, D62);
  per-message cost model ±0.4 % (§6, D180); selection objective spectral 0.970 / greedy 0.936 of the exact
  optimum on synthetic graphs (§6, D170, unit).

  python/video/render.sh scenes/s05_sharing.py Sharing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, Circle, Create, Cross, DashedLine, FadeIn, FadeOut,
                   ImageMobject, LaggedStart, Line, MoveAlongPath, Rectangle, RoundedRectangle, Succession, Text,
                   Transform, VGroup, interpolate_color, ManimColor)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.mobjects import CodePanel  # noqa: E402
from ariadne_video.style import AGENT, BG, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

# RESULTS_INDEX §4 (D62, EuRoC, 297 queries) and §6 (D170 unit, D180)
RECALL = [("EigenPlaces", 0.626), ("mean-bit baseline", 0.444), ("random", 0.167)]
SPECTRAL, GREEDY = 0.970, 0.936


def frame(agent: int, height: float = 1.8) -> ImageMobject:
    im = np.array(Image.open(DATA / "frames" / f"agent{agent}_f600.png").resize((640, 360)))[..., ::-1]  # BGR -> RGB
    return ImageMobject(im).set_height(height)


def fingerprint(vals: np.ndarray, color: str, width: float = 3.0, height: float = 0.5) -> ImageMobject:
    """256 bars as one image: column i filled from the bottom to vals[i] (0..1)."""
    rows = 40
    rgba = np.zeros((rows, len(vals), 4), np.uint8)
    rgb = [int(color[k:k + 2], 16) for k in (1, 3, 5)]
    for i, v in enumerate(vals):
        h = max(1, int(round(v * rows)))
        rgba[rows - h:, i, :3] = rgb
        rgba[rows - h:, i, 3] = 230
    m = ImageMobject(rgba)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.stretch_to_fit_width(width).stretch_to_fit_height(height)


def lam2(n: int, edges: list[tuple[int, int]]) -> tuple[float, np.ndarray]:
    L = np.zeros((n, n))
    for a, b in edges:
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    w, v = np.linalg.eigh(L)
    return float(w[1]), v[:, 1]


class Sharing(CaptionedScene):
    def construct(self):
        self.panel_fingerprints()
        self.panel_exchange()
        self.panel_spectral()

    # A: three robots' views, each turned into a 256-number fingerprint; recall table
    def panel_fingerprints(self):
        rng = np.random.default_rng(5)
        base = rng.random(256) ** 1.5
        vals = [np.clip(base + rng.normal(0, 0.08, 256), 0.02, 1), np.clip(base + rng.normal(0, 0.08, 256), 0.02, 1),
                rng.random(256) ** 1.5]  # robots 1 and 2 saw the same place; robot 3 did not (illustrative)
        thumbs = []
        for a in range(3):
            img = frame(a, 1.6)
            border = Rectangle(width=img.width, height=img.height, stroke_color=AGENT[a], stroke_width=4)
            thumbs.append((img, border))
        xs = [-4.4, 0.0, 4.4]
        for (img, border), x in zip(thumbs, xs):
            img.move_to([x, 1.75, 0])
            border.move_to(img)
        self.thumbs = thumbs
        chip_real = Text("recorded run · camera frames", font=FONT, font_size=16, color=DIM).to_edge(UP, buff=0.22)
        chip_real.to_edge(LEFT, buff=0.3)
        self.say("But one robot's map is only one piece of the puzzle.",
                 run_with=[LaggedStart(*[AnimationGroup(FadeIn(i), Create(b)) for i, b in thumbs], lag_ratio=0.3,
                                       run_time=1.6), FadeIn(chip_real)])

        fps, arrows, names = [], VGroup(), VGroup()
        for a, ((img, _), x) in enumerate(zip(thumbs, xs)):
            fp = fingerprint(vals[a], AGENT[a], width=3.0, height=0.45).move_to([x, -0.3, 0])
            arrows.add(Arrow(img.get_bottom(), fp.get_top() + UP * 0.05, buff=0.1, stroke_width=3, color=DIM,
                             max_tip_length_to_length_ratio=0.15))
            names.add(Text("EigenPlaces", font=FONT, font_size=16, color=DIM).next_to(arrows[-1], RIGHT, 0.08))
            fps.append(fp)
        fp_lab = Text("a 256-number fingerprint of the place", font=FONT, font_size=18, color=TEXT).move_to(
            [0, -0.95, 0])
        chip_ill = tag("illustrative").next_to(chip_real, RIGHT, 0.2)
        self.say("To join them, the robots need to notice when they've seen the same place.",
                 run_with=[LaggedStart(*[AnimationGroup(Create(ar), FadeIn(n), FadeIn(fp, shift=DOWN * 0.1))
                                         for ar, n, fp in zip(arrows, names, fps)], lag_ratio=0.25, run_time=2.0),
                           FadeIn(fp_lab), FadeIn(chip_ill)])
        self.clear_caption()

        # how well do the fingerprints find the same place? (EuRoC)
        head = Text("finding the same place again · EuRoC, recall@1", font=FONT, font_size=20, color=TEXT)
        rows = VGroup()
        for k, (name, v) in enumerate(RECALL):
            col = THREAD if k == 0 else DIM
            bar = Rectangle(width=4.2 * v, height=0.26, stroke_width=0, fill_color=col, fill_opacity=0.9)
            rows.add(VGroup(Text(name, font=FONT, font_size=18, color=TEXT if k == 0 else DIM), bar,
                            Text(f"{v:.3f}", font=FONT, font_size=18, color=col)))
        for k, r in enumerate(rows):
            y = -1.95 - 0.38 * k
            r[0].move_to([-1.55, y, 0], aligned_edge=RIGHT)
            r[1].move_to([-1.35, y, 0], aligned_edge=LEFT)
            r[2].next_to(r[1], RIGHT, 0.15)
        head.move_to([0, -1.5, 0])
        chip_tbl = tag("measured", "D62, EuRoC").next_to(rows, RIGHT, 0.5).align_to(rows, DOWN)
        self.play(FadeOut(fp_lab), FadeIn(head), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows],
                                                             lag_ratio=0.3), FadeIn(chip_tbl), run_time=1.6)
        self.wait(2.4)
        self.fps, self.vals = fps, vals
        self.play(FadeOut(VGroup(head, rows, chip_tbl, arrows, names, chip_ill, chip_real)),
                  *[FadeOut(fp) for fp in fps], run_time=0.6)

    # B: the two-stage exchange on a byte budget
    def panel_exchange(self):
        nodes = [(-4.6, -0.2), (0.0, 1.9), (4.4, -0.2)]
        moves = []
        for (img, border), p in zip(self.thumbs, nodes):
            moves += [img.animate.scale(0.62).move_to([*p, 0]), border.animate.scale(0.62).move_to([*p, 0])]
        self.play(*moves, run_time=0.9)
        meter_frame = RoundedRectangle(corner_radius=0.06, width=0.42, height=3.2, stroke_color=DIM, stroke_width=1.5)
        meter_frame.move_to([6.45, 0.35, 0])
        fill = Rectangle(width=0.3, height=3.05, stroke_width=0, fill_color=AGENT[2], fill_opacity=0.8)
        fill.move_to(meter_frame).align_to(meter_frame, DOWN).shift(UP * 0.075)
        meter_lab = Text("byte\nbudget", font=FONT, font_size=16, color=DIM, line_spacing=0.8).next_to(
            meter_frame, DOWN, 0.12)
        chip = tag("illustrative").to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.3)

        def drain(frac, rt=0.6):
            h = 3.05 * frac
            return fill.animate(run_time=rt).stretch_to_fit_height(max(h, 0.01)).align_to(meter_frame, DOWN).shift(
                UP * 0.075)

        # a whole picture is too expensive
        pic = ImageMobject(np.array(Image.open(DATA / "frames" / "agent0_f600.png").resize((320, 180)))[..., ::-1])
        pic.set_height(0.9).move_to([*nodes[0], 0])
        x = Cross(scale_factor=0.45, stroke_color=TARGET, stroke_width=6).move_to([-2.3, 0.85, 0])
        self.say("Sending pictures over a weak radio is far too expensive,",
                 run_with=[FadeIn(meter_frame), FadeIn(fill), FadeIn(meter_lab), FadeIn(chip),
                           Succession(FadeIn(pic, run_time=0.3), pic.animate(run_time=1.0).move_to([-2.3, 0.85, 0]),
                                      AnimationGroup(FadeIn(x, scale=1.5), pic.animate.set_opacity(0.3),
                                                     run_time=0.4))])
        self.play(FadeOut(pic), FadeOut(x), run_time=0.3)

        def packet(i, j, color, w=0.32, h=0.12, rt=0.9):
            p = Rectangle(width=w, height=h, stroke_width=0, fill_color=color, fill_opacity=0.95)
            path = Line([*nodes[i], 0], [*nodes[j], 0])
            p.move_to(path.get_start())
            return p, Succession(FadeIn(p, run_time=0.1), MoveAlongPath(p, path, run_time=rt), FadeOut(p, run_time=0.1))

        s1 = Text("1 · a tiny fingerprint", font=FONT, font_size=18, color=TEXT).move_to([0, -1.55, 0])
        flights = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    _, anim = packet(i, j, AGENT[i])
                    flights.append(anim)
        self.say("so they talk in two steps:", run_with=[FadeIn(s1)])
        self.say("first a tiny fingerprint of each place,",
                 run_with=[LaggedStart(*flights, lag_ratio=0.12, run_time=2.2), drain(0.9, 2.2)])
        s2 = Text("2 · the details, only for look-alikes", font=FONT, font_size=18, color=TEXT).next_to(s1, DOWN, 0.15)
        match = Line([*nodes[0], 0], [*nodes[1], 0], color=THREAD, stroke_width=3)
        _, req = packet(0, 1, AGENT[0], 0.3, 0.3, 0.8)
        _, resp = packet(1, 0, AGENT[1], 0.9, 0.42, 1.1)
        self.say("and only when two fingerprints look alike, the details.",
                 run_with=[FadeIn(s2), Succession(Create(match, run_time=0.5), req, AnimationGroup(resp, drain(0.55, 1.1)))])
        cost = VGroup(Text("per-message cost model accurate to ±0.4 %", font=FONT, font_size=20, color=TEXT),
                      tag("measured", "D180")).arrange(RIGHT, buff=0.25).move_to([0, -2.45, 0])
        self.say("Even then, every byte counts.", run_with=[FadeIn(cost, shift=UP * 0.1)])
        self.clear_caption()
        self.play(FadeOut(VGroup(meter_frame, fill, meter_lab, chip, s1, s2, match, cost)),
                  *[FadeOut(m) for pair in self.thumbs for m in pair], run_time=0.6)

    # C: spectral prioritisation on a toy graph, λ₂ computed live
    def panel_spectral(self):
        ring = lambda o: [(o + i, o + (i + 1) % 6) for i in range(6)]  # noqa: E731
        edges = ring(0) + ring(6) + [(0, 3), (6, 9), (2, 8)]
        redundant, bridge = (1, 4), (1, 8)  # a chord inside a cluster vs a short link across the gap
        centres = [np.array([-4.6, 0.25]), np.array([-1.5, 0.25])]
        pos = {}
        for c in range(2):
            for k in range(6):
                ang = np.pi / 2 + k * np.pi / 3 + (np.pi if c == 0 else 0)
                pos[c * 6 + k] = centres[c] + 1.0 * np.array([np.cos(ang), np.sin(ang)])
        P = lambda n: [*pos[n], 0]  # noqa: E731
        lam, fied = lam2(12, edges)
        lo, hi = fied.min(), fied.max()
        node_m = VGroup(*[Circle(radius=0.13, stroke_width=0, fill_opacity=1, fill_color=interpolate_color(
            ManimColor(AGENT[0]), ManimColor(AGENT[1]), (fied[n] - lo) / (hi - lo))).move_to(P(n)) for n in range(12)])
        edge_m = VGroup(*[Line(P(a), P(b), color=DIM, stroke_width=3) for a, b in edges])
        chip = tag("illustrative").to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.3)

        def readout(v, delta=None):
            t = Text(f"λ₂ = {v:.3f}", font=FONT, font_size=26, color=TEXT)
            sub = Text("how firmly the map holds together", font=FONT, font_size=16, color=DIM)
            g = VGroup(t, sub).arrange(DOWN, buff=0.08)
            if delta is not None:
                g.add(Text(f"{delta:+.3f}", font=FONT, font_size=22, color=THREAD if delta > 0.05 else DIM).next_to(
                    t, RIGHT, 0.25))
            return g.move_to([-3.05, 2.35, 0])

        r0 = readout(lam)
        code = CodePanel("spectral link selection", [
            "v ← Fiedler vector of the map graph",
            "for each candidate link (a, b):",
            "    score ← κ·‖Vᵀ(e_a − e_b)‖² / bytes",
            "take the best; update; repeat while",
            "    the byte budget lasts",
            "1-swap: trade a pick if λ₂ rises",
        ], font_size=16).to_edge(RIGHT, buff=0.3).shift(UP * 0.45)
        self.say("So Ariadne asks which link would hold the team's map together best.",
                 run_with=[FadeIn(chip), Create(edge_m, run_time=1.2), FadeIn(node_m, lag_ratio=0.05, run_time=1.2),
                           FadeIn(r0), FadeIn(code), code.focus(0)])

        red = DashedLine(P(redundant[0]), P(redundant[1]), color=THREAD, stroke_width=4)
        lam_r, _ = lam2(12, edges + [redundant])
        r1 = readout(lam_r, lam_r - lam)
        br = DashedLine(P(bridge[0]), P(bridge[1]), color=THREAD, stroke_width=4)
        lam_b, _ = lam2(12, edges + [bridge])
        r2 = readout(lam_b, lam_b - lam)
        self.say("A bridge between two loosely joined pieces is worth far more",
                 run_with=[Succession(AnimationGroup(Create(red), code.focus(1), run_time=0.8),
                                      AnimationGroup(Transform(r0, r1), code.focus(2), run_time=0.8),
                                      AnimationGroup(red.animate.set_opacity(0.15), run_time=0.5),
                                      AnimationGroup(Create(br), run_time=0.8),
                                      AnimationGroup(Transform(r0, r2),
                                                     Transform(br, Line(P(bridge[0]), P(bridge[1]), color=THREAD,
                                                                        stroke_width=5)), code.focus(3),
                                                     run_time=0.9))])
        self.say("than one more link where the map is already solid.")
        self.clear_caption()
        self.play(code.focus(5), run_time=0.5)
        tbl = VGroup(
            Text("selection objective, synthetic graphs: share of the exact optimum", font=FONT, font_size=18,
                 color=TEXT),
            VGroup(Text("spectral", font=FONT, font_size=20, color=THREAD), Text(f"{SPECTRAL:.3f}", font=FONT,
                                                                                font_size=20, color=THREAD),
                   Text("greedy", font=FONT, font_size=20, color=DIM), Text(f"{GREEDY:.3f}", font=FONT, font_size=20,
                                                                           color=DIM)).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, buff=0.15)
        tbl.add(tag("measured", "D170, unit test").next_to(tbl, RIGHT, 0.3))
        tbl.move_to([0, -2.45, 0])
        self.play(FadeIn(tbl, shift=UP * 0.1), run_time=0.7)
        self.wait(2.6)
